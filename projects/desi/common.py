if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt

import os
import tempfile

import numpy as np

import fitsio

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_erosion

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import Tan, Sip, anwcs_t
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.miscutils import polygons_intersect, estimate_mode, clip_polygon
from astrometry.sdss.fields import read_photoobjs_in_wcs
from astrometry.sdss import DR9, band_index, AsTransWrapper
from astrometry.util.resample import resample_with_wcs,OverlapError

from tractor.basics import ConstantSky, NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.engine import get_class_from_name, Image
from tractor.psfex import PsfEx
from tractor.sdss import get_tractor_sources_dr9
from tractor.ellipses import *

# search order: $TMPDIR, $TEMP, $TMP, then /tmp, /var/tmp, /usr/tmp
tempdir = tempfile.gettempdir()
decals_dir = os.environ.get('DECALS_DIR')
if decals_dir is None:
    print 'Warning: you should set the $DECALS_DIR environment variable.'
    print 'On NERSC, you can do:'
    print '  module use /project/projectdirs/cosmo/work/decam/versions/modules'
    print '  module load decals'
    print
    
calibdir = os.path.join(decals_dir, 'calib', 'decam')
sedir    = os.path.join(decals_dir, 'calib', 'se-config')
an_config= os.path.join(decals_dir, 'calib', 'an-config', 'cfg')

class SFDMap(object):
    extinctions = {
        'SDSS u': 4.239,
        'DES g': 3.237,
        'DES r': 2.176,
        'DES i': 1.595,
        'DES z': 1.217,
        'DES Y': 1.058,
        }

    def __init__(self, ngp_filename=None, sgp_filename=None):
        dustdir = os.environ.get('DUST_DIR', None)
        if dustdir is not None:
            dustdir = os.path.join(dustdir, 'maps')
        else:
            dustdir = '.'
            print 'Warning: $DUST_DIR not set; looking for SFD maps in current directory.'
        if ngp_filename is None:
            ngp_filename = os.path.join(dustdir, 'SFD_dust_4096_ngp.fits')
        if sgp_filename is None:
            sgp_filename = os.path.join(dustdir, 'SFD_dust_4096_sgp.fits')
        if not os.path.exists(ngp_filename):
            raise RuntimeError('Error: SFD map does not exist: %s' % ngp_filename)
        if not os.path.exists(sgp_filename):
            raise RuntimeError('Error: SFD map does not exist: %s' % sgp_filename)
        self.north = fitsio.read(ngp_filename)
        self.south = fitsio.read(sgp_filename)
        self.northwcs = anwcs_t(ngp_filename, 0)
        self.southwcs = anwcs_t(sgp_filename, 0)

    @staticmethod
    def bilinear_interp_nonzero(image, x, y):
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        # Bilinear interpolate, but not outside the bounds (where ebv=0)
        fx = x - x0
        ebvA = image[y0,x0]
        ebvB = image[y0,x0+1]
        ebv1 = ebvA * fx + ebvB * (1.-fx)
        ebv1[ebvA == 0] = ebvB[ebvA == 0]
        ebv1[ebvB == 0] = ebvA[ebvB == 0]

        ebvA = image[y0+1,x0]
        ebvB = image[y0+1,x0+1]
        ebv2 = ebvA * fx + ebvB * (1.-fx)
        ebv2[ebvA == 0] = ebvB[ebvA == 0]
        ebv2[ebvB == 0] = ebvA[ebvB == 0]

        fy = y - y0
        ebv = ebv1 * fy + ebv2 * (1.-fy)
        ebv[ebv1 == 0] = ebv2[ebv1 == 0]
        ebv[ebv2 == 0] = ebv1[ebv2 == 0]
        return ebv
        
    def extinction(self, filts, ra, dec):
        l,b = radectolb(ra, dec)
        ebv = np.zeros_like(l)
        N = (b >= 0)
        for wcs,image,cut in [(self.northwcs, self.north, N),
                              (self.southwcs, self.south, np.logical_not(N))]:
            # Our WCS routines are mis-named... the SFD WCSes convert 
            #   X,Y <-> L,B.
            ok,x,y = wcs.radec2pixelxy(l[cut], b[cut])
            assert(np.all(ok == 0))
            H,W = image.shape
            assert(np.all(x >= 1.))
            assert(np.all(x <= W))
            assert(np.all(y >= 1.))
            assert(np.all(y <= H))
            ebv[cut] = SFDMap.bilinear_interp_nonzero(image, x-1., y-1.)

        factors = np.array([SFDMap.extinctions[f] for f in filts])

        #a,b = np.broadcast_arrays(factors, ebv)
        #return a*b
        return factors[np.newaxis,:] * ebv[:,np.newaxis]

def segment_and_group_sources(image, T):
    '''
    *image*: binary image that defines "blobs"
    *T*: source table; only ".itx" and ".ity" elements are used (x,y integer pix pos)
      - ".blob" field is added.

    Returns: (blobs, blobsrcs, blobslices)

    *blobs*: image, values -1 = no blob, integer blob indices
    *blobsrcs*: list of np arrays of integers, elements in T within each blob
    *blobslices*: list of slice objects for blob bounding-boxes.
    
    '''

    emptyblob = 0

    blobs,nblobs = label(image)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]

    # Find sets of sources within blobs
    blobsrcs = []
    keepslices = []
    blobmap = {}
    for blob in range(1, nblobs+1):
        Isrcs = np.flatnonzero(T.blob == blob)
        if len(Isrcs) == 0:
            continue
        blobmap[blob] = len(blobsrcs)
        blobsrcs.append(Isrcs)
        bslc = blobslices[blob-1]
        keepslices.append(bslc)

    blobslices = keepslices

    # Find sources that do not belong to a blob and add them as
    # singleton "blobs"; otherwise they don't get optimized.
    # for sources outside the image bounds, what should we do?
    inblobs = np.zeros(len(T), bool)
    for Isrcs in blobsrcs:
        inblobs[Isrcs] = True
    noblobs = np.flatnonzero(inblobs == 0)
    del inblobs
    H,W = image.shape
    # Add new fake blobs!
    for ib,i in enumerate(noblobs):
        S = 3
        bslc = (slice(np.clip(T.ity[i] - S, 0, H-1), np.clip(T.ity[i] + S+1, 0, H)),
                slice(np.clip(T.itx[i] - S, 0, W-1), np.clip(T.itx[i] + S+1, 0, W)))
        print 'Slice:', bslc
        # Set synthetic blob number
        blob = nblobs+1 + ib
        blobs[bslc][blobs[bslc] == emptyblob] = blob
        blobmap[blob] = len(blobsrcs)
        blobslices.append(bslc)
        blobsrcs.append(np.array([i]))
    print 'Added', len(noblobs), 'new fake singleton blobs'

    # Remap the "blobs" image so that empty regions are = -1 and the blob values
    # correspond to their indices in the "blobsrcs" list.
    bm = np.zeros(max(blobmap.keys())+1, int)
    for k,v in blobmap.items():
        bm[k] = v
    bm[0] = -1
    blobs = bm[blobs]

    for j,Isrcs in enumerate(blobsrcs):
        for i in Isrcs:
            assert(blobs[T.ity[i], T.itx[i]] == j)
    T.blob = blobs[T.ity, T.itx]
    assert(len(blobsrcs) == len(blobslices))

    return blobs, blobsrcs, blobslices

def get_sdss_sources(bands, targetwcs, photoobjdir=None, local=True):
    # FIXME?
    margin = 0.

    sdss = DR9(basedir=photoobjdir)
    if local:
        local = (local and ('BOSS_PHOTOOBJ' in os.environ)
                 and ('PHOTO_RESOLVE' in os.environ))
    if local:
        sdss.useLocalTree()

    cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr',
            'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr',
            'phi_exp_deg',
            'resolve_status', 'nchild', 'flags', 'objc_flags',
            'run','camcol','field','id',
            'psfflux', 'psfflux_ivar',
            'cmodelflux', 'cmodelflux_ivar',
            'modelflux', 'modelflux_ivar',
            'devflux', 'expflux', 'extinction']

    objs = read_photoobjs_in_wcs(targetwcs, margin, sdss=sdss, cols=cols)
    print 'Got', len(objs), 'photoObjs'

    # It can be string-valued
    objs.objid = np.array([int(x) if len(x) else 0 for x in objs.objid])

    # Treat as pointsource...
    sband = 'r'
    bandnum = 'ugriz'.index(sband)
    objs.treated_as_pointsource = treat_as_pointsource(objs, bandnum)

    print 'Bands', bands, '->', list(bands)

    srcs = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss,
        bands=list(bands),
        nanomaggies=True, fixedComposites=True,
        useObjcType=True,
        ellipse=EllipseESoft.fromRAbPhi)
    print 'Got', len(srcs), 'Tractor sources'

    # record coordinates in target brick image
    ok,objs.tx,objs.ty = targetwcs.radec2pixelxy(objs.ra, objs.dec)
    objs.tx -= 1
    objs.ty -= 1
    W,H = targetwcs.get_width(), targetwcs.get_height()
    objs.itx = np.clip(np.round(objs.tx), 0, W-1).astype(int)
    objs.ity = np.clip(np.round(objs.ty), 0, H-1).astype(int)

    cat = Catalog(*srcs)
    return cat, objs

def treat_as_pointsource(T, bandnum, setObjcType=True):
    b = bandnum
    gal = (T.objc_type == 3)
    dev = gal * (T.fracdev[:,b] >= 0.5)
    exp = gal * (T.fracdev[:,b] <  0.5)
    stars = (T.objc_type == 6)
    print sum(dev), 'deV,', sum(exp), 'exp, and', sum(stars), 'stars'
    print 'Total', len(T), 'sources'

    thetasn = np.zeros(len(T))
    T.theta_deverr[dev,b] = np.maximum(1e-6, T.theta_deverr[dev,b])
    T.theta_experr[exp,b] = np.maximum(1e-5, T.theta_experr[exp,b])
    # theta_experr nonzero: 1.28507e-05
    # theta_deverr nonzero: 1.92913e-06
    thetasn[dev] = T.theta_dev[dev,b] / T.theta_deverr[dev,b]
    thetasn[exp] = T.theta_exp[exp,b] / T.theta_experr[exp,b]

    # aberrzero = np.zeros(len(T), bool)
    # aberrzero[dev] = (T.ab_deverr[dev,b] == 0.)
    # aberrzero[exp] = (T.ab_experr[exp,b] == 0.)

    maxtheta = np.zeros(len(T), bool)
    maxtheta[dev] = (T.theta_dev[dev,b] >= 29.5)
    maxtheta[exp] = (T.theta_exp[exp,b] >= 59.0)

    # theta S/N > modelflux for dev, 10*modelflux for exp
    bigthetasn = (thetasn > (T.modelflux[:,b] * (1.*dev + 10.*exp)))

    print sum(gal * (thetasn < 3.)), 'have low S/N in theta'
    print sum(gal * (T.modelflux[:,b] > 1e4)), 'have big flux'
    #print sum(aberrzero), 'have zero a/b error'
    print sum(maxtheta), 'have the maximum theta'
    print sum(bigthetasn), 'have large theta S/N vs modelflux'
    
    badgals = gal * reduce(np.logical_or,
                           [thetasn < 3.,
                            T.modelflux[:,b] > 1e4,
                            #aberrzero,
                            maxtheta,
                            bigthetasn,
                            ])
    print 'Found', sum(badgals), 'bad galaxies'
    if setObjcType:
        T.objc_type[badgals] = 6
    return badgals


def _detmap((tim, targetwcs, H, W)):
    R = tim_get_resamp(tim, targetwcs)
    if R is None:
        return None,None,None,None
    ie = tim.getInvvar()
    psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
    detim = tim.getImage().copy()
    detim[ie == 0] = 0.
    detim = gaussian_filter(detim, tim.psf_sigma) / psfnorm**2
    detsig1 = tim.sig1 / psfnorm
    subh,subw = tim.shape
    detiv = np.zeros((subh,subw), np.float32) + (1. / detsig1**2)
    detiv[ie == 0] = 0.
    (Yo,Xo,Yi,Xi) = R
    return Yo, Xo, detim[Yi,Xi], detiv[Yi,Xi]

def tim_get_resamp(tim, targetwcs):
    if hasattr(tim, 'resamp'):
        return tim.resamp
    try:
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, tim.subwcs, [], 2)
    except OverlapError:
        print 'No overlap'
        return None
    if len(Yo) == 0:
        return None
    resamp = [x.astype(np.int16) for x in (Yo,Xo,Yi,Xi)]
    return resamp

def detection_maps(tims, targetwcs, bands, mp):
    # Render the detection maps
    H,W = targetwcs.shape
    detmaps = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    detivs  = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    for tim, (Yo,Xo,incmap,inciv) in zip(
        tims, mp.map(_detmap, [(tim, targetwcs, H, W) for tim in tims])):
        if Yo is None:
            continue
        detmaps[tim.band][Yo,Xo] += incmap*inciv
        detivs [tim.band][Yo,Xo] += inciv

    for band in bands:
        detmaps[band] /= np.maximum(1e-16, detivs[band])

    # back into lists, not dicts
    detmaps = [detmaps[b] for b in bands]
    detivs  = [detivs [b] for b in bands]
        
    return detmaps, detivs

def sed_matched_detection(sedname, sed, detmaps, detivs, bands,
                          xomit, yomit,
                          nsigma=5.,
                          saddle=2.,
                          ps=None):
                          
    '''
    detmaps: list of detmaps, same order as "bands"
    detivs :    ditto
    
    '''
    H,W = detmaps[0].shape
    sedmap = np.zeros((H,W), np.float32)
    sediv  = np.zeros((H,W), np.float32)
    for iband,band in enumerate(bands):
        if sed[iband] == 0:
            continue
        # We convert the detmap to canonical band via
        #   detmap * w
        # And the corresponding change to sig1 is
        #   sig1 * w
        # So the invvar-weighted sum is
        #    (detmap * w) / (sig1**2 * w**2)
        #  = detmap / (sig1**2 * w)
        sedmap += detmaps[iband] * detivs[iband] / sed[iband]
        sediv  += detivs [iband] / sed[iband]**2
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)

    peaks = (sedsn > nsigma)

    if ps is not None:
        pkbounds = binary_dilation(peaks) - peaks
        plt.clf()
        plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest', origin='lower',
                   cmap='hot')
        rgba = np.zeros((H,W,4), np.uint8)
        rgba[:,:,1] = pkbounds
        rgba[:,:,3] = pkbounds
        plt.imshow(rgba, interpolation='nearest', origin='lower')
        plt.title('SED %s: S/N & peaks' % sedname)
        ps.savefig()

    # zero out the edges -- larger margin here?
    peaks[0 ,:] = 0
    peaks[:, 0] = 0
    peaks[-1,:] = 0
    peaks[:,-1] = 0
    # find pixels that are larger than their 8 neighbors
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,2:  ])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,2:  ])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,2:  ])

    # Ok, now for each peak (in decreasing order of peak height):
    #  -find the image area that is "saddle" sigma lower than the peak.
    #  -blank out any lower peaks in that region?
    #  -omit this peak if one of the "omit" points is within the saddle
    # ... or create saddle images for each of the 'omit' points first?

    omitmap = np.zeros(peaks.shape, bool)

    def saddle_level(Y):
        # Require a saddle of (the larger of) "saddle" sigma, or 10% of the peak height
        drop = max(saddle, Y * 0.1)
        return Y - drop

    for x,y in zip(xomit, yomit):
        if sedsn[y,x] > nsigma:
            level = saddle_level(sedsn[y,x])
            blobs,nblobs = label(sedsn > level)
            omitmap |= (blobs == blobs[y,x])
        elif sediv[y,x] == 0:
            # Nil pixel... possibly saturated.  Mask the whole region.
            badpix = (sediv == 0)
            blobs,nblobs = label(badpix)
            #print 'Blobs:', blobs.shape, blobs.dtype
            badpix = (blobs == blobs[y,x])
            badpix = binary_dilation(badpix, iterations=5)
            omitmap |= badpix
        else:
            # omit a 3x3 box around the peak
            omitmap[np.clip(y-1,0,H-1): np.clip(y+2,0,H),
                    np.clip(x-1,0,W-1): np.clip(x+2,0,W)] = True

    # find peaks, sort by flux
    py,px = np.nonzero(peaks)
    I = np.argsort(-sedsn[py,px])
    py = py[I]
    px = px[I]
    #keep = np.ones(len(py), bool)
    # drop peaks inside the 'omit map'
    keep = np.logical_not(omitmap[py, px])


    if ps is not None:
        crossa = dict(ms=10, mew=1.5)
        green = (0,1,0)
        plt.clf()
        plt.imshow(omitmap, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('Omit map for SED %s' % sedname)
        ps.savefig()
        
        plt.clf()
        plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest', origin='lower',
                   cmap='gray')
        ax = plt.axis()
        plt.plot(px[keep], py[keep], '+', color=green, **crossa)
        drop = np.logical_not(keep)
        plt.plot(px[drop], py[drop], 'r+', **crossa)
        plt.axis(ax)
        plt.title('SED %s: keep (green) peaks' % sedname)
        ps.savefig()
    
    for i,(x,y) in enumerate(zip(px, py)):
        if not keep[i]:
            continue

        level = saddle_level(sedsn[y,x])
        blobs,nblobs = label(sedsn > level)
        saddleblob = (blobs == blobs[y,x])
        # ???
        # this source's blob touches the omit map
        if np.any(saddleblob & omitmap):
            keep[i] = False
            continue
        
        omitmap |= saddleblob

    if ps is not None:
        plt.clf()
        plt.imshow(omitmap, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('Final omit map for SED %s' % sedname)
        ps.savefig()
        
        plt.clf()
        plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest', origin='lower',
                   cmap='gray')
        ax = plt.axis()
        plt.plot(px[keep], py[keep], '+', color=green, **crossa)
        drop = np.logical_not(keep)
        plt.plot(px[drop], py[drop], 'r+', **crossa)
        plt.axis(ax)
        plt.title('SED %s: Final keep (green) peaks' % sedname)
        ps.savefig()

    py = py[keep]
    px = px[keep]
    
    return sedsn, px, py

    
    # blobs,nblobs = label(peaks)
    # print 'N detected blobs:', nblobs
    # blobslices = find_objects(blobs)

    # Now... peaks...
    
    # # Un-set existing blobs
    # for x,y in zip(xomit, yomit):
    #     # blob number
    #     bb = blobs[y,x]
    #     if bb == 0:
    #         continue
    #     # un-set 'peaks' within this blob
    #     slc = blobslices[bb-1]
    #     peaks[slc][blobs[slc] == bb] = 0

    # zero out the edges -- larger margin here?
    # peaks[0 ,:] = 0
    # peaks[:, 0] = 0
    # peaks[-1,:] = 0
    # peaks[:,-1] = 0

    

        

def get_rgb(imgs, bands, mnmx=None, arcsinh=None):
    '''
    Given a list of images in the given bands, returns a scaled RGB
    image.
    '''
    bands = ''.join(bands)
    if bands == 'grz':
        scales = dict(g = (2, 0.0066),
                      r = (1, 0.01),
                      z = (0, 0.025),
                      )
    elif bands == 'urz':
        scales = dict(u = (2, 0.0066),
                      r = (1, 0.01),
                      z = (0, 0.025),
                      )
    elif bands == 'gri':
        # scales = dict(g = (2, 0.004),
        #               r = (1, 0.0066),
        #               i = (0, 0.01),
        #               )
        scales = dict(g = (2, 0.002),
                      r = (1, 0.004),
                      i = (0, 0.005),
                      )
    else:
        assert(False)
        
    h,w = imgs[0].shape
    rgb = np.zeros((h,w,3), np.float32)
    # Convert to ~ sigmas
    for im,band in zip(imgs, bands):
        plane,scale = scales[band]
        rgb[:,:,plane] = (im / scale).astype(np.float32)
        #print 'rgb: plane', plane, 'range', rgb[:,:,plane].min(), rgb[:,:,plane].max()

    if mnmx is None:
        mn,mx = -3, 10
    else:
        mn,mx = mnmx

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        rgb = nlmap(rgb)
        mn = nlmap(mn)
        mx = nlmap(mx)

    rgb = (rgb - mn) / (mx - mn)
    return np.clip(rgb, 0., 1.)
    

def switch_to_soft_ellipses(cat):
    from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy
    from tractor.ellipses import EllipseESoft
    for src in cat:
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseESoft.fromEllipseE(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeDev = EllipseESoft.fromEllipseE(src.shapeDev)
            src.shapeExp = EllipseESoft.fromEllipseE(src.shapeExp)

def brick_catalog_for_radec_box(ralo, rahi, declo, dechi,
                                decals, catpattern, bricks=None):
    '''
    Merges multiple Tractor brick catalogs to cover an RA,Dec
    bounding-box.

    No cleverness with RA wrap-around; assumes ralo < rahi.

    decals: Decals object
    
    bricks: table of bricks, eg from Decals.get_bricks()

    catpattern: filename pattern of catalog files to read,
        eg "pipebrick-cats/tractor-phot-%06i.its"
    
    '''
    assert(ralo < rahi)
    assert(declo < dechi)

    if bricks is None:
        bricks = decals.get_bricks_readonly()
    I = decals.bricks_touching_radec_box(bricks, ralo, rahi, declo, dechi)
    print len(I), 'bricks touch RA,Dec box'
    TT = []
    hdr = None
    for i in I:
        brick = bricks[i]
        fn = catpattern % brick.brickid
        print 'Catalog', fn
        if not os.path.exists(fn):
            print 'Warning: catalog does not exist:', fn
            continue
        T = fits_table(fn, header=True)
        if T is None or len(T) == 0:
            print 'Warning: empty catalog', fn
            continue
        T.cut((T.ra  >= ralo ) * (T.ra  <= rahi) *
              (T.dec >= declo) * (T.dec <= dechi))
        TT.append(T)
    if len(TT) == 0:
        return None
    T = merge_tables(TT)
    # arbitrarily keep the first header
    T._header = TT[0]._header
    return T
    
def ccd_map_image(valmap, empty=0.):
    '''
    valmap: { 'N7' : 1., 'N8' : 17.8 }

    Returns: a numpy image (shape (12,14)) with values mapped to their CCD locations.
    '''
    img = np.empty((12,14))
    img[:,:] = empty
    for k,v in valmap.items():
        x0,x1,y0,y1 = ccd_map_extent(k)
        #img[y0+6:y1+6, x0+7:x1+7] = v
        img[y0:y1, x0:x1] = v
    return img

def ccd_map_center(extname):
    x0,x1,y0,y1 = ccd_map_extent(extname)
    return (x0+x1)/2., (y0+y1)/2.

def ccd_map_extent(extname, inset=0.):
    assert(extname.startswith('N') or extname.startswith('S'))
    num = int(extname[1:])
    assert(num >= 1 and num <= 31)
    if num <= 7:
        x0 = 7 - 2*num
        y0 = 0
    elif num <= 13:
        x0 = 6 - (num - 7)*2
        y0 = 1
    elif num <= 19:
        x0 = 6 - (num - 13)*2
        y0 = 2
    elif num <= 24:
        x0 = 5 - (num - 19)*2
        y0 = 3
    elif num <= 28:
        x0 = 4 - (num - 24)*2
        y0 = 4
    else:
        x0 = 3 - (num - 28)*2
        y0 = 5
    if extname.startswith('N'):
        (x0,x1,y0,y1) = (x0, x0+2, -y0-1, -y0)
    else:
        (x0,x1,y0,y1) = (x0, x0+2, y0, y0+1)

    # Shift from being (0,0)-centered to being aligned with the ccd_map_image() image.
    x0 += 7
    x1 += 7
    y0 += 6
    y1 += 6
    
    if inset == 0.:
        return (x0,x1,y0,y1)
    return (x0+inset, x1-inset, y0+inset, y1-inset)

def wcs_for_brick(b, W=3600, H=3600, pixscale=0.262):
    '''
    b: row from decals-bricks.fits file
    W,H: size in pixels
    pixscale: pixel scale in arcsec/pixel.

    Returns: Tan wcs object
    '''
    pixscale = pixscale / 3600.
    return Tan(b.ra, b.dec, W/2.+0.5, H/2.+0.5,
               -pixscale, 0., 0., pixscale,
               float(W), float(H))

def ccds_touching_wcs(targetwcs, T, ccdrad=0.17, polygons=True):
    '''
    targetwcs: wcs object describing region of interest
    T: fits_table object of CCDs

    ccdrad: radius of CCDs, in degrees.  Default 0.17 is for DECam.
    #If None, computed from T.

    Returns: index array I of CCDs within range.
    '''
    trad = targetwcs.radius()
    if ccdrad is None:
        ccdrad = max(np.sqrt(np.abs(T.cd1_1 * T.cd2_2 - T.cd1_2 * T.cd2_1)) *
                     np.hypot(T.width, T.height) / 2.)

    rad = trad + ccdrad
    #r,d = targetwcs.crval
    r,d = targetwcs.radec_center()
    #print len(T), 'ccds'
    #print 'trad', trad, 'ccdrad', ccdrad
    I = np.nonzero(np.abs(T.dec - d) < rad)
    #print 'Cut to', len(I), 'on Dec'
    I = I[degrees_between(T.ra[I], T.dec[I], r, d) < rad]
    #print 'Cut to', len(I), 'on RA,Dec'

    if not polygons:
        return I
    # now check actual polygon intersection
    tw,th = targetwcs.imagew, targetwcs.imageh
    targetpoly = [(0.5,0.5),(tw+0.5,0.5),(tw+0.5,th+0.5),(0.5,th+0.5)]
    cd = targetwcs.get_cd()
    tdet = cd[0]*cd[3] - cd[1]*cd[2]
    #print 'tdet', tdet
    if tdet > 0:
        targetpoly = list(reversed(targetpoly))
    targetpoly = np.array(targetpoly)

    keep = []
    for i in I:
        W,H = T.width[i],T.height[i]
        wcs = Tan(*[float(x) for x in
                    [T.crval1[i], T.crval2[i], T.crpix1[i], T.crpix2[i], T.cd1_1[i],
                     T.cd1_2[i], T.cd2_1[i], T.cd2_2[i], W, H]])
        cd = wcs.get_cd()
        wdet = cd[0]*cd[3] - cd[1]*cd[2]
        #print 'wdet', wdet
        poly = []
        for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5)]:
            rr,dd = wcs.pixelxy2radec(x,y)
            ok,xx,yy = targetwcs.radec2pixelxy(rr,dd)
            poly.append((xx,yy))
        if wdet > 0:
            poly = list(reversed(poly))
        poly = np.array(poly)
        if polygons_intersect(targetpoly, poly):
            keep.append(i)
    I = np.array(keep)
    #print 'Cut to', len(I), 'on polygons'
    return I

def create_temp(**kwargs):
    f,fn = tempfile.mkstemp(dir=tempdir, **kwargs)
    os.close(f)
    os.unlink(fn)
    return fn

def sed_matched_filters(bands):
    # List the SED-matched filters to run
    # single-band filters
    SEDs = []
    for i,band in enumerate(bands):
        sed = np.zeros(len(bands))
        sed[i] = 1.
        SEDs.append((band, sed))
    assert(bands == 'grz')
    SEDs.append(('Flat', (1.,1.,1.)))
    SEDs.append(('Red', (2.5, 1.0, 0.4)))
    return SEDs

def run_sed_matched_filters(SEDs, bands, detmaps, detivs, omit_xy,
                            targetwcs,
                            plots=False, ps=None):
    if omit_xy is not None:
        xx,yy = omit_xy
        n0 = len(xx)
    else:
        xx,yy = [],[]
        n0 = 0

    H,W = detmaps[0].shape
    hot = np.zeros((H,W), np.float32)
    for sedname,sed in SEDs:
        print 'SED', sedname
        if plots:
            pps = ps
        else:
            pps = None
        sedsn,px,py = sed_matched_detection(
            sedname, sed, detmaps, detivs, bands, xx, yy, ps=pps)
        print len(px), 'new peaks'
        hot = np.maximum(hot, sedsn)
        xx = np.append(xx, px)
        yy = np.append(yy, py)

    # New peaks:
    peakx = xx[n0:]
    peaky = yy[n0:]

    # Add sources for the new peaks we found

    # make their initial fluxes ~ 5-sigma
    # fluxes = dict([(b,[]) for b in bands])
    # for tim in tims:
    #     psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
    #     fluxes[tim.band].append(5. * tim.sig1 / psfnorm)
    # fluxes = dict([(b, np.mean(fluxes[b])) for b in bands])

    pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
    print 'Adding', len(pr), 'new sources'
    # Also create FITS table for new sources
    Tnew = fits_table()
    Tnew.ra  = pr
    Tnew.dec = pd
    Tnew.tx = peakx
    Tnew.ty = peaky
    Tnew.itx = np.clip(np.round(Tnew.tx), 0, W-1).astype(int)
    Tnew.ity = np.clip(np.round(Tnew.ty), 0, H-1).astype(int)
    newcat = []
    for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
        fluxes = dict([(band, detmap[Tnew.ity[i], Tnew.itx[i]])
                       for band,detmap in zip(bands,detmaps)])
        newcat.append(PointSource(RaDecPos(r,d),
                                  NanoMaggies(order=bands, **fluxes)))
    # print 'Existing source table:'
    # T.about()
    # print 'New source table:'
    # Tnew.about()
    # T = merge_tables([T, Tnew], columns='fillzero')
    # return peakx,peaky,

    return Tnew, newcat, hot

class Decals(object):
    def __init__(self):
        self.decals_dir = decals_dir
        self.ZP = None
        self.bricks = None

        # Create and cache a kd-tree for bricks_touching_radec_box ?
        self.cache_tree = False
        self.bricktree = None
        ### HACK! Hard-coded brick edge size, in degrees!
        self.bricksize = 0.25
        
    def get_bricks(self):
        return fits_table(os.path.join(self.decals_dir, 'decals-bricks.fits'))

    ### HACK...
    def get_bricks_readonly(self):
        if self.bricks is None:
            self.bricks = self.get_bricks()
            # Assert that bricks are the sizes we think they are.
            assert(np.all(np.abs((self.bricks.dec2 - self.bricks.dec1) -
                                 self.bricksize) < 1e-8))
        return self.bricks

    def get_brick(self, brickid):
        B = self.get_bricks_readonly()
        I = np.nonzero(B.brickid == brickid)
        if len(I) == 0:
            return None
        return B[I[0]]

    def get_brick_by_name(self, brickname):
        B = self.get_bricks_readonly()
        I = np.nonzero(np.array([n == brickname for n in B.brickname]))
        if len(I) == 0:
            return None
        return B[I[0]]

    def bricks_touching_radec_box(self, bricks,
                                  ralo, rahi, declo, dechi):
        '''
        Returns an index vector of the bricks that touch the given RA,Dec box.
        '''
        if bricks is None:
            bricks = self.get_bricks_readonly()
        if self.cache_tree and bricks == self.bricks:
            from astrometry.libkd.spherematch import tree_build_radec, tree_search_radec
            # Use kdtree
            if self.bricktree is None:
                self.bricktree = tree_build_radec(bricks.ra, bricks.dec)
            # brick size
            radius = np.sqrt(2.)/2. * self.bricksize
            # + RA,Dec box size
            radius = radius + degrees_between(ralo, declo, rahi, dechi) / 2.
            dec = (dechi + declo) / 2.
            c = (np.cos(np.deg2rad(rahi)) + np.cos(np.deg2rad(ralo))) / 2.
            s = (np.sin(np.deg2rad(rahi)) + np.sin(np.deg2rad(ralo))) / 2.
            ra  = np.rad2deg(np.arctan2(s, c))
            J = tree_search_radec(self.bricktree, ra, dec, radius)
            I = J[np.nonzero((bricks.ra1[J]  <= rahi ) * (bricks.ra2[J]  >= ralo) *
                             (bricks.dec1[J] <= dechi) * (bricks.dec2[J] >= declo))]
            return I
            
        I = np.nonzero((bricks.ra1  <= rahi ) * (bricks.ra2  >= ralo) *
                       (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
        return I
    
    def get_ccds(self):
        T = fits_table(os.path.join(self.decals_dir, 'decals-ccds.fits'))
        T.extname = np.array([s.strip() for s in T.extname])
        return T

    def ccds_touching_wcs(self, wcs):
        T = self.get_ccds()
        I = ccds_touching_wcs(wcs, T)
        #print len(I), 'CCDs nearby'
        if len(I) == 0:
            return None
        T.cut(I)
        return T

    def tims_touching_wcs(self, targetwcs, mp, mock_psf=False, bands=None):
        '''
        mp: multiprocessing object
        '''
        # Read images
        C = self.ccds_touching_wcs(targetwcs)
        # Sort by band
        if bands is not None:
            C.cut(np.hstack([np.nonzero(C.filter == band) for band in bands]))
        ims = []
        for t in C:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(t)
            ims.append(im)
        # Read images, clip to ROI
        W,H = targetwcs.get_width(), targetwcs.get_height()
        targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        args = [(im, self, targetrd, mock_psf) for im in ims]
        tims = mp.map(read_one_tim, args)
        return tims
    
    def find_ccds(self, expnum=None, extname=None):
        T = self.get_ccds()
        if expnum is not None:
            T.cut(T.expnum == expnum)
        if extname is not None:
            T.cut(T.extname == extname)
        return T
    
    def get_zeropoint_for(self, im):
        if self.ZP is None:
            zpfn = os.path.join(self.decals_dir, 'calib', 'decam', 'photom', 'zeropoints.fits')
            #print 'Reading zeropoints:', zpfn
            self.ZP = fits_table(zpfn)

            if 'ccdname' in self.ZP.get_columns():
                # 'N4 ' -> 'N4'
                self.ZP.ccdname = np.array([s.strip() for s in self.ZP.ccdname])

            #self.ZP.about()

        I = np.nonzero(self.ZP.expnum == im.expnum)
        #print 'Got', len(I), 'matching expnum', im.expnum
        if len(I) > 1:
            #I = np.nonzero((self.ZP.expnum == im.expnum) * (self.ZP.extname == im.extname))
            I = np.nonzero((self.ZP.expnum == im.expnum) * (self.ZP.ccdname == im.extname))
            #print 'Got', len(I), 'matching expnum', im.expnum, 'and extname', im.extname

        # No updated zeropoint -- use header MAGZERO from primary HDU.
        elif len(I) == 0:
            print 'WARNING: using header zeropoints for', im
            hdr = im.read_image_primary_header()

            # DES Year1 Stripe82 images:
            magzero = hdr['MAGZERO']
            #exptime = hdr['EXPTIME']
            #magzero += 2.5 * np.log10(exptime)
            return magzero

        assert(len(I) == 1)
        I = I[0]

        # Arjun says use CCDZPT
        magzp = self.ZP.ccdzpt[I]

        # magzp = self.ZP.zpt[I]
        # print 'Raw magzp', magzp
        # if magzp == 0:
        #     print 'Magzp = 0; using ccdzpt'
        #     magzp = self.ZP.ccdzpt[I]
        #     print 'Got', magzp
        exptime = self.ZP.exptime[I]
        magzp += 2.5 * np.log10(exptime)
        #print 'magzp', magzp
        return magzp

def exposure_metadata(filenames, hdus=None, trim=None):
    nan = np.nan
    primkeys = [('FILTER',''),
                ('RA', nan),
                ('DEC', nan),
                ('AIRMASS', nan),
                ('DATE-OBS', ''),
                ('G-SEEING', nan),
                ('EXPTIME', nan),
                ('EXPNUM', 0),
                ('MJD-OBS', 0),
                ('PROPID', ''),
                ('GUIDER', ''),
                ('OBJECT', ''),
                ]
    hdrkeys = [('AVSKY', nan),
               ('ARAWGAIN', nan),
               ('FWHM', nan),
               #('ZNAXIS1',0),
               #('ZNAXIS2',0),
               ('CRPIX1',nan),
               ('CRPIX2',nan),
               ('CRVAL1',nan),
               ('CRVAL2',nan),
               ('CD1_1',nan),
               ('CD1_2',nan),
               ('CD2_1',nan),
               ('CD2_2',nan),
               ('EXTNAME',''),
               ('CCDNUM',''),
               ]

    otherkeys = [('CPIMAGE',''), ('CPIMAGE_HDU',0), ('CALNAME',''), #('CPDATE',0),
                 ('HEIGHT',0),('WIDTH',0),
                 ]

    allkeys = primkeys + hdrkeys + otherkeys

    vals = dict([(k,[]) for k,d in allkeys])

    for i,fn in enumerate(filenames):
        print 'Reading', (i+1), 'of', len(filenames), ':', fn
        F = fitsio.FITS(fn)
        #print F
        #print len(F)
        primhdr = F[0].read_header()
        #print primhdr

        expstr = '%08i' % primhdr.get('EXPNUM')

        # # Parse date with format: 2014-08-09T04:20:50.812543
        # date = datetime.datetime.strptime(primhdr.get('DATE-OBS'),
        #                                   '%Y-%m-%dT%H:%M:%S.%f')
        # # Subract 12 hours to get the date used by the CP to label the night;
        # # CP20140818 includes observations with date 2014-08-18 evening and
        # # 2014-08-19 early AM.
        # cpdate = date - datetime.timedelta(0.5)
        # #cpdatestr = '%04i%02i%02i' % (cpdate.year, cpdate.month, cpdate.day)
        # #print 'Date', date, '-> CP', cpdatestr
        # cpdateval = cpdate.year * 10000 + cpdate.month * 100 + cpdate.day
        # print 'Date', date, '-> CP', cpdateval

        cpfn = fn
        if trim is not None:
            cpfn = cpfn.replace(trim, '')
        print 'CP fn', cpfn

        if hdus is not None:
            hdulist = hdus
        else:
            hdulist = range(1, len(F))

        for hdu in hdulist:
            hdr = F[hdu].read_header()

            info = F[hdu].get_info()
            #'extname': 'S1', 'dims': [4146L, 2160L]
            H,W = info['dims']

            for k,d in primkeys:
                vals[k].append(primhdr.get(k, d))
            for k,d in hdrkeys:
                vals[k].append(hdr.get(k, d))

            vals['CPIMAGE'].append(cpfn)
            vals['CPIMAGE_HDU'].append(hdu)
            vals['WIDTH'].append(int(W))
            vals['HEIGHT'].append(int(H))
            #vals['CPDATE'].append(cpdateval)

            calname = '%s/%s/decam-%s-%s' % (expstr[:5], expstr, expstr, hdr.get('EXTNAME'))
            vals['CALNAME'].append(calname)

    T = fits_table()
    for k,d in allkeys:
        T.set(k.lower().replace('-','_'), np.array(vals[k]))
    #T.about()

    T.filter = np.array([s.split()[0] for s in T.filter])
    T.ra_bore  = np.array([hmsstring2ra (s) for s in T.ra ])
    T.dec_bore = np.array([dmsstring2dec(s) for s in T.dec])

    T.ra  = np.zeros(len(T))
    T.dec = np.zeros(len(T))
    for i in range(len(T)):
        W,H = T.width[i], T.height[i]

        wcs = Tan(T.crval1[i], T.crval2[i], T.crpix1[i], T.crpix2[i],
                  T.cd1_1[i], T.cd1_2[i], T.cd2_1[i], T.cd2_2[i], float(W), float(H))
        
        xc,yc = W/2.+0.5, H/2.+0.5
        rc,dc = wcs.pixelxy2radec(xc,yc)
        T.ra [i] = rc
        T.dec[i] = dc

    return T

class DecamImage(object):
    def __init__(self, t):
        imgfn, hdu, band, expnum, extname, calname, exptime = (
            t.cpimage.strip(), t.cpimage_hdu, t.filter.strip(), t.expnum,
            t.extname.strip(), t.calname.strip(), t.exptime)

        if os.path.exists(imgfn):
            self.imgfn = imgfn
        else:
            self.imgfn = os.path.join(decals_dir, 'images', 'decam', imgfn)
        self.hdu   = hdu
        self.expnum = expnum
        self.extname = extname
        self.band  = band
        self.exptime = exptime

        # EDR filenames: .imag.fits, .ivar.fits, .mask.fits.gz
        if '.imag.fits' in self.imgfn:
            self.dqfn = self.imgfn.replace('.imag.fits', '.mask.fits.gz')
            self.wtfn = self.imgfn.replace('.imag.fits', '.ivar.fits')
        else:
            self.dqfn = self.imgfn.replace('_ooi_', '_ood_')
            self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            print attr, '->', fn
            if os.path.exists(fn):
                print 'Exists.'
                continue
            if fn.endswith('.fz'):
                fun = fn[:-3]
                if os.path.exists(fun):
                    print 'Using      ', fun
                    print 'rather than', fn
                    setattr(self, attr, fun)
            fn = getattr(self, attr)
            print attr, fn
            print '  exists? ', os.path.exists(fn)

        ibase = os.path.basename(imgfn)
        ibase = ibase.replace('.fits.fz', '')
        ibase = ibase.replace('.fits', '')
        idirname = os.path.basename(os.path.dirname(imgfn))
        #self.name = dirname + '/' + base + ' + %02i' % hdu
        #print 'dir,base', idirname, ibase
        #print 'calibdir', calibdir

        self.calname = calname
        self.name = '%08i-%s' % (expnum, extname)
        #print 'Calname', calname
        
        extnm = '.ext%02i' % hdu
        self.wcsfn = os.path.join(calibdir, 'astrom', calname + '.wcs.fits')
        self.corrfn = self.wcsfn.replace('.wcs.fits', '.corr.fits')
        self.sdssfn = self.wcsfn.replace('.wcs.fits', '.sdss.fits')
        self.sefn = os.path.join(calibdir, 'sextractor', calname + '.fits')
        self.se2fn = os.path.join(calibdir, 'sextractor2', calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', calname + '.fits')
        self.psffitfn = os.path.join(calibdir, 'psfexfit', calname + '.fits')
        self.psffitellfn = os.path.join(calibdir, 'psfexfit', calname + '-ell.fits')
        self.skyfn = os.path.join(calibdir, 'sky', calname + '.fits')
        self.morphfn = os.path.join(calibdir, 'morph', calname + '.fits')

    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

    def get_tractor_image(self, decals, slc=None, radecpoly=None, mock_psf=False):
        '''
        slc: y,x slices
        '''
        band = self.band
        imh,imw = self.get_image_shape()
        wcs = self.read_wcs()
        x0,y0 = 0,0
        if slc is None and radecpoly is not None:
            imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
            ok,tx,ty = wcs.radec2pixelxy(radecpoly[:-1,0], radecpoly[:-1,1])
            tpoly = zip(tx,ty)
            clip = clip_polygon(imgpoly, tpoly)
            clip = np.array(clip)
            if len(clip) == 0:
                return None
            x0,y0 = np.floor(clip.min(axis=0)).astype(int)
            x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
            slc = slice(y0,y1+1), slice(x0,x1+1)

            if y1 - y0 < 5 or x1 - x0 < 5:
                print 'Skipping tiny subimage'
                return None
        if slc is not None:
            sy,sx = slc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop
        
        print 'Reading image from', self.imgfn, 'HDU', self.hdu
        img,imghdr = self.read_image(header=True, slice=slc)
        print 'Reading invvar from', self.wtfn, 'HDU', self.hdu
        invvar = self.read_invvar(slice=slc, clip=True)

        print 'Invvar range:', invvar.min(), invvar.max()
        if np.all(invvar == 0.):
            print 'Skipping zero-invvar image'
            return None
        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(not(np.all(invvar == 0.)))

        # header 'FWHM' is in pixels
        psf_fwhm = imghdr['FWHM']
        psf_sigma = psf_fwhm / 2.35
        primhdr = self.read_image_primary_header()

        magzp = decals.get_zeropoint_for(self)
        print 'magzp', magzp
        zpscale = NanoMaggies.zeropointToScale(magzp)
        print 'zpscale', zpscale

        sky = self.read_sky_model()
        midsky = sky.getConstant()
        img -= midsky
        sky.subtract(midsky)

        # Scale images to Nanomaggies
        img /= zpscale
        invvar *= zpscale**2
        orig_zpscale = zpscale
        zpscale = 1.
        assert(np.sum(invvar > 0) > 0)
        sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(np.isfinite(sig1))

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        if mock_psf:
            from tractor.basics import NCircularGaussianPSF
            psfex = None
            psf = NCircularGaussianPSF([1.5], [1.0])
            print 'WARNING: using mock PSF:', psf
        else:
            # read fit PsfEx model -- with ellipse representation
            psfex = PsfEx.fromFits(self.psffitellfn)
            print 'Read', psfex
            psf = psfex

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=sky, name=self.name + ' ' + band)
        assert(np.all(np.isfinite(tim.getInvError())))
        tim.zr = [-3. * sig1, 10. * sig1]
        tim.midsky = midsky
        tim.sig1 = sig1
        tim.band = band
        tim.psf_fwhm = psf_fwhm
        tim.psf_sigma = psf_sigma
        tim.sip_wcs = wcs
        tim.x0,tim.y0 = int(x0),int(y0)
        tim.psfex = psfex
        tim.imobj = self
        tim.primhdr = primhdr
        tim.hdr = imghdr
        mn,mx = tim.zr
        subh,subw = tim.shape
        tim.subwcs = tim.sip_wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        return tim
    
    def makedirs(self):
        for dirnm in [os.path.dirname(fn) for fn in
                      [self.wcsfn, self.corrfn, self.sdssfn, self.sefn, self.psffn, self.morphfn,
                       self.se2fn, self.psffitfn, self.skyfn]]:
            if not os.path.exists(dirnm):
                try:
                    os.makedirs(dirnm)
                except:
                    pass

    def _read_fits(self, fn, hdu, slice=None, header=None, **kwargs):
        if slice is not None:
            f = fitsio.FITS(fn)[hdu]
            img = f[slice]
            rtn = img
            if header:
                hdr = f.read_header()
                return (img,hdr)
            return img
        return fitsio.read(fn, ext=hdu, header=header, **kwargs)

    def read_image(self, **kwargs):
        return self._read_fits(self.imgfn, self.hdu, **kwargs)

    def get_image_info(self):
        return fitsio.FITS(self.imgfn)[self.hdu].get_info()

    def get_image_shape(self):
        ''' Returns image H,W '''
        return self.get_image_info()['dims']
    
    def read_image_primary_header(self, **kwargs):
        return fitsio.read_header(self.imgfn)

    def read_image_header(self, **kwargs):
        return fitsio.read_header(self.imgfn, ext=self.hdu)

    def read_dq(self, **kwargs):
        return self._read_fits(self.dqfn, self.hdu, **kwargs)
    #return fitsio.FITS(self.dqfn)[self.hdu].read()

    def read_invvar(self, clip=False, **kwargs):
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        if clip:
            sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
            # Clamp near-zero (incl negative!) invvars to zero
            thresh = 0.2 * (1./sig1**2)
            invvar[invvar < thresh] = 0
        return invvar
    #return fitsio.FITS(self.wtfn)[self.hdu].read()

    def read_wcs(self):
        return Sip(self.wcsfn)

    def read_sdss(self):
        S = fits_table(self.sdssfn)
        # ugh!
        if S.objc_type.min() > 128:
            S.objc_type -= 128
        return S

    def read_sky_model(self):
        hdr = fitsio.read_header(self.skyfn)
        skyclass = hdr['SKY']
        clazz = get_class_from_name(skyclass)
        fromfits = getattr(clazz, 'fromFitsHeader')
        skyobj = fromfits(hdr, prefix='SKY_')
        return skyobj

    def run_calibs(self, ra, dec, pixscale, mock_psf,
                   W=2048, H=4096, se=True,
                   astrom=True, psfex=True, sky=True,
                   morph=False, se2=False, psfexfit=True,
                   funpack=True, fcopy=False, use_mask=True,
                   just_check=False):
        '''
        pixscale: in arcsec/pixel

        just_check: if True, returns True if calibs need to be run.
        '''
        print 'run_calibs:', str(self), 'near RA,Dec', ra,dec, 'with pixscale', pixscale, 'arcsec/pix'

        for fn in [self.wcsfn, self.sefn, self.psffn, self.psffitfn, self.skyfn]:
            print 'exists?', os.path.exists(fn), fn
        self.makedirs()

        if mock_psf:
            psfex = False
            psfexfit = False
    
        run_funpack = False
        run_se = False
        run_se2 = False
        run_astrom = False
        run_psfex = False
        run_psfexfit = False
        run_morph = False
        run_sky = False
    
        if se and not all([os.path.exists(fn) for fn in [self.sefn]]):
            run_se = True
            run_funpack = True
        if se2 and not all([os.path.exists(fn) for fn in [self.se2fn]]):
            run_se2 = True
            run_funpack = True
        #if not all([os.path.exists(fn) for fn in [self.wcsfn,self.corrfn,self.sdssfn]]):
        if astrom and not os.path.exists(self.wcsfn):
            run_astrom = True
        if psfex and not os.path.exists(self.psffn):
            run_psfex = True
        if psfexfit and not (os.path.exists(self.psffitfn) and os.path.exists(self.psffitellfn)):
            run_psfexfit = True
        if morph and not os.path.exists(self.morphfn):
            run_morph = True
            run_funpack = True
        if sky and not os.path.exists(self.skyfn):
            run_sky = True

        if just_check:
            return (run_se or run_se2 or run_astrom or run_psfex or run_psfexfit
                    or run_morph or run_sky)

        if run_funpack and (funpack or fcopy):
            tmpimgfn  = create_temp(suffix='.fits')
            tmpmaskfn = create_temp(suffix='.fits')
    
            if funpack:
                cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpimgfn, self.imgfn)
            else:
                cmd = 'imcopy %s"+%i" %s' % (self.imgfn, self.hdu, tmpimgfn)
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
            if use_mask:
                cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpmaskfn, self.dqfn)
                print cmd
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)
    
        if run_astrom or run_morph or run_se or run_se2:
            # grab header values...
            primhdr = self.read_image_primary_header()
            hdr     = self.read_image_header()
    
            magzp  = primhdr['MAGZERO']
            fwhm = hdr['FWHM']
            seeing = pixscale * fwhm
            print 'FWHM', fwhm, 'pix'
            print 'pixscale', pixscale, 'arcsec/pix'
            print 'Seeing', seeing, 'arcsec'
    
        if run_se:
            maskstr = ''
            if use_mask:
                maskstr = '-FLAG_IMAGE ' + tmpmaskfn
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'DECaLS-v2.sex'),
                maskstr, '-SEEING_FWHM %f' % seeing,
                '-PIXEL_SCALE 0',
                #'-PIXEL_SCALE %f' % (pixscale),
                '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', self.sefn,
                tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_se2:
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'DECaLS-v2-2.sex'),
                '-FLAG_IMAGE', tmpmaskfn, '-SEEING_FWHM %f' % seeing,
                '-PIXEL_SCALE %f' % (pixscale),
                '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', self.se2fn,
                tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_astrom:
            cmd = ' '.join([
                'solve-field --config', an_config, '-D . --temp-dir', tempdir,
                '--ra %f --dec %f' % (ra,dec), '--radius 1',
                '-L %f -H %f -u app' % (0.9 * pixscale, 1.1 * pixscale),
                '--continue --no-plots --no-remove-lines --uniformize 0',
                '--no-fits2fits',
                '-X x_image -Y y_image -s flux_auto --extension 2',
                '--width %i --height %i' % (W,H),
                '--crpix-center',
                '-N none -U none -S none -M none',
                #'--rdls', self.sdssfn,
                #'--corr', self.corrfn,
                '--rdls none --corr none',
                '--wcs', self.wcsfn, 
                '--temp-axy', '--tag-all', self.sefn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)

            if not os.path.exists(self.wcsfn):
                # Run a second phase...
                an_config_2 = os.path.join(decals_dir, 'calib', 'an-config', 'cfg2')
                cmd = ' '.join([
                    'solve-field --config', an_config_2, '-D . --temp-dir', tempdir,
                    '--ra %f --dec %f' % (ra,dec), '--radius 1',
                    '-L %f -H %f -u app' % (0.9 * pixscale, 1.1 * pixscale),
                    '--continue --no-plots --uniformize 0',
                    '--no-fits2fits',
                    '-X x_image -Y y_image -s flux_auto --extension 2',
                    '--width %i --height %i' % (W,H),
                    '--crpix-center',
                    '-N none -U none -S none -M none',
                    '--rdls none --corr none',
                    '--wcs', self.wcsfn, 
                    '--temp-axy', '--tag-all', self.sefn])
                    #--no-remove-lines 
                print cmd
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)

        if run_psfex:
            cmd = ('psfex -c %s -PSF_DIR %s %s' %
                   (os.path.join(sedir, 'DECaLS-v2.psfex'),
                    os.path.dirname(self.psffn), self.sefn))
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_psfexfit:
            print 'Fit PSF...'
    
            from tractor.basics import GaussianMixtureEllipsePSF, GaussianMixturePSF
            from tractor.psfex import PsfEx
    
            iminfo = self.get_image_info()
            #print 'img:', iminfo
            H,W = iminfo['dims']
            psfex = PsfEx(self.psffn, W, H, ny=13, nx=7,
                          psfClass=GaussianMixtureEllipsePSF)
            psfex.savesplinedata = True
            print 'Fitting MoG model to PsfEx'
            psfex._fitParamGrid(damp=1)
            pp,XX,YY = psfex.splinedata

            psfex.toFits(self.psffitellfn, merge=True)
            print 'Wrote', self.psffitellfn
    
            # Convert to GaussianMixturePSF
            ppvar = np.zeros_like(pp)
            for iy in range(psfex.ny):
                for ix in range(psfex.nx):
                    psf = GaussianMixtureEllipsePSF(*pp[iy, ix, :])
                    mog = psf.toMog()
                    ppvar[iy,ix,:] = mog.getParams()
            psfexvar = PsfEx(self.psffn, W, H, ny=psfex.ny, nx=psfex.nx,
                             psfClass=GaussianMixturePSF)
            psfexvar.splinedata = (ppvar, XX, YY)
            psfexvar.toFits(self.psffitfn, merge=True)
            print 'Wrote', self.psffitfn
            
        if run_morph:
            cmd = ' '.join(['sex -c', os.path.join(sedir, 'CS82_MF.sex'),
                            '-FLAG_IMAGE', tmpmaskfn,
                            '-SEEING_FWHM %f' % seeing,
                            '-MAG_ZEROPOINT %f' % magzp,
                            '-PSF_NAME', self.psffn,
                            '-CATALOG_NAME', self.morphfn,
                            tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)

        if run_sky:
            img = self.read_image()
            wt = self.read_invvar()
            img = img[wt > 0]
            try:
                skyval = estimate_mode(img, raiseOnWarn=True)
            except:
                skyval = np.median(img)
            sky = ConstantSky(skyval)
            tt = type(sky)
            sky_type = '%s.%s' % (tt.__module__, tt.__name__)
            hdr = fitsio.FITSHDR()
            hdr.add_record(dict(name='SKY', value=sky_type, comment='Sky class'))
            sky.toFitsHeader(hdr, prefix='SKY_')
            fits = fitsio.FITS(self.skyfn, 'rw', clobber=True)
            fits.write(None, header=hdr)
            
def run_calibs(X):
    im = X[0]
    kwargs = X[1]
    args = X[2:]
    print 'run_calibs:', X
    print 'im', im
    print 'args', args
    print 'kwargs', kwargs
    return im.run_calibs(*args, **kwargs)


def read_one_tim((im, decals, targetrd, mock_psf)):
    print 'Reading expnum', im.expnum, 'name', im.extname, 'band', im.band, 'exptime', im.exptime
    tim = im.get_tractor_image(decals, radecpoly=targetrd, mock_psf=mock_psf)
    return tim

