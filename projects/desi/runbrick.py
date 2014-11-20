# Cython
#import pyximport; pyximport.install(pyimport=True)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import tempfile
import os
import time
import datetime

import fitsio

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_erosion

from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.miscutils import clip_polygon
from astrometry.util.resample import resample_with_wcs,OverlapError
from astrometry.libkd.spherematch import match_radec
from astrometry.util.ttime import Time, MemMeas
from astrometry.sdss.fields import read_photoobjs_in_wcs, radec_to_sdss_rcf
from astrometry.sdss import DR9, band_index, AsTransWrapper
from astrometry.util.run_command import *

from tractor import *
from tractor.galaxy import *
from tractor.source_extractor import *
from tractor.sdss import get_tractor_sources_dr9

from common import *

## GLOBALS!  Oh my!
mp = None
nocache = True
photoobjdir = 'photoObjs-new'
useCeres = True

def runbrick_global_init():
    if nocache:
        disable_galaxy_cache()

    from tractor.ceres import ceres_opt
    

def create_tractor(tims, srcs):
    t = Tractor(tims, src)
    if nocache:
        t.disable_cache()
    return t

# didn't I write mp to avoid this foolishness in the first place?
def _map(f, args):
    if mp is not None:
        return mp.map(f, args, chunksize=1)
    else:
        return map(f, args)

class iterwrapper(object):
    def __init__(self, y, n):
        self.n = n
        self.y = y
    def __str__(self):
        return 'iterwrapper: n=%i; ' % self.n + str(self.y)
    def __iter__(self):
        return self
    def next(self):
        try:
            return self.y.next()
        except:
            import traceback
            print str(self), 'next()'
            traceback.print_exc()
            raise
            
    def __len__(self):
        return self.n

def _print_struc(X):
    if X is None:
        print 'None',
    elif type(X) in (list,tuple):
        islist = (type(X) is list)
        if islist:
            print '[',
        else:
            print '(',
        for x in X:
            _print_struc(x)
            print ',',
        if islist:
            print ']',
        else:
            print ')',
    else:
        print type(X),

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
    

def set_globals():
    global imchi
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        hspace=0.2, wspace=0.05)
    imchi = dict(cmap='RdBu', vmin=-5, vmax=5)

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

def get_sdss_sources(bands, targetwcs, local=True):
    # FIXME?
    margin = 0.

    sdss = DR9(basedir=photoobjdir)
    if local:
        sdss.useLocalTree()

    cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
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

    srcs = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss,
        bands=bands,
        nanomaggies=True, fixedComposites=True,
        useObjcType=True,
        ellipse=EllipseESoft.fromRAbPhi)
    print 'Got', len(srcs), 'Tractor sources'

    cat = Catalog(*srcs)
    return cat, objs

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

def _bounce_tim_get_resamp((tim, targetwcs)):
    return tim_get_resamp(tim, targetwcs)

def tims_compute_resamp(tims, targetwcs):
    #for tim in tims:
    #    r = tim_get_resamp(tim, targetwcs)
    #    tim.resamp = r
    R = _map(_bounce_tim_get_resamp, [(tim,targetwcs) for tim in tims])
    for tim,r in zip(tims, R):
        tim.resamp = r

def compute_coadds(tims, bands, W, H, targetwcs):
    coimgs = []
    cons = []
    for ib,band in enumerate(bands):
        coimg = np.zeros((H,W), np.float32)
        con   = np.zeros((H,W), np.uint8)
        for tim in tims:
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            nn = (tim.getInvError()[Yi,Xi] > 0)
            coimg[Yo,Xo] += tim.getImage ()[Yi,Xi] * nn
            con  [Yo,Xo] += nn
        coimg /= np.maximum(con,1)
        coimgs.append(coimg)
        cons  .append(con)
    return coimgs,cons

def _read_tim((im, decals, targetrd, mock_psf)):
    print 'Reading expnum', im.expnum, 'name', im.extname, 'band', im.band, 'exptime', im.exptime
    tim = im.get_tractor_image(decals, radecpoly=targetrd, mock_psf=mock_psf)
    return tim

def stage_tims(W=3600, H=3600, brickid=None, ps=None, plots=False,
               target_extent=None, pipe=False, program_name='runbrick.py',
               mock_psf=False, **kwargs):
    t0 = tlast = Time()

    rtn,version,err = run_command('git describe')
    if rtn:
        raise RuntimeError('Failed to get version string (git describe):' + ver + err)
    version = version.strip()
    print 'Version:', version

    decals = Decals()

    decalsv = decals.decals_dir
    hdr = fitsio.FITSHDR()
    hdr.add_record(dict(name='TRACTORV', value=version,
                        comment='Tractor git version'))
    hdr.add_record(dict(name='DECALSV', value=decalsv,
                        comment='DECaLS version'))
    hdr.add_record(dict(name='DECALSDT', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    version_header = hdr

    B = decals.get_bricks()

    print 'Bricks:'
    B.about()
    print 'Brickid:', brickid
    print 'Brickids:', B.brickid
    ii = np.flatnonzero(B.brickid == brickid)
    print len(ii), 'brick ids match'
    ii = ii[0]
    brick = B[ii]
    print 'Chosen brick:'
    brick.about()

    bands = ['g','r','z']
    catband = 'r'

    targetwcs = wcs_for_brick(brick, W=W, H=H)

    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    pixscale = targetwcs.pixel_scale()
    print 'pixscale', pixscale

    T = decals.get_ccds()
    I = ccds_touching_wcs(targetwcs, T)
    print len(I), 'CCDs nearby'
    if len(I) == 0:
        return None
    T.cut(I)

    ims = []
    for band in bands:
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        for t in TT:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(t)
            ims.append(im)

    # Check that the CCDs_touching cuts are correct.
    #check_touching(decals, targetwcs, bands, brick, pixscale, ps)
    print 'Finding images touching brick:', Time()-tlast
    tlast = Time()

    # Check that the zeropoints exist
    for im in ims:
        decals.get_zeropoint_for(im)

    args = [(im, dict(), brick.ra, brick.dec, pixscale, mock_psf) for im in ims]
    _map(run_calibs, args)
    print 'Calibrations:', Time()-tlast
    tlast = Time()

    # Read images, clip to ROI
    ttim = Time()
    args = [(im, decals, targetrd, mock_psf) for im in ims]
    tims = _map(_read_tim, args)
    tims = [tim for tim in tims if tim is not None]

    print 'Read images:', Time()-tlast
    tlast = Time()

    if not pipe:
        # save resampling params
        tims_compute_resamp(tims, targetwcs)
        print 'Computing resampling:', Time()-tlast
        tlast = Time()
        # Produce per-band coadds, for plots
        coimgs,cons = compute_coadds(tims, bands, W, H, targetwcs)
        print 'Coadds:', Time()-tlast
        tlast = Time()

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'target_extent']
    if not pipe:
        keys.extend(['coimgs', 'cons'])
    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
    return rtn


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

def stage_srcs(coimgs=None, cons=None,
               targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None, brickid=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               pipe=False,
               **kwargs):

    print 'Rendering detection maps...'
    tlast = Time()
    # Render the detection maps
    detmaps = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    detivs  = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    for tim, (Yo,Xo,incmap,inciv) in zip(
        tims, _map(_detmap, [(tim, targetwcs, H, W) for tim in tims])):
        if Yo is None:
            continue
        detmaps[tim.band][Yo,Xo] += incmap*inciv
        detivs [tim.band][Yo,Xo] += inciv
    print 'Detmaps:', Time()-tlast
    tlast = Time()

    # -find significant peaks in the per-band detection maps and SED-matched (hot)
    # -segment into blobs
    # -blank out blobs containing a catalog source
    # -create sources for any remaining peaks

    hot = np.zeros((H,W), np.float32)

    # Single-band SED
    for band in bands:
        detmap = detmaps[band] / np.maximum(1e-16, detivs[band])
        detsn = detmap * np.sqrt(detivs[band])
        hot = np.maximum(hot, detsn)
        detmaps[band] = detmap

        if plots:
            plt.clf()
            dimshow(np.round(detsn), vmin=0, vmax=10, cmap='hot')
            plt.title('Single-band detection filter S/N: %s' % band)
            plt.colorbar()
            ps.savefig()


    for sedname,sed in [('Flat', (1.,1.,1.)), ('Red', (2.5, 1.0, 0.4))]:
        sedmap = np.zeros((H,W), np.float32)
        sediv  = np.zeros((H,W), np.float32)
        for iband,band in enumerate(bands):
            # We convert the detmap to canonical band via
            #   detmap * w
            # And the corresponding change to sig1 is
            #   sig1 * w
            # So the invvar-weighted sum is
            #    (detmap * w) / (sig1**2 * w**2)
            #  = detmap / (sig1**2 * w)
            sedmap += detmaps[band] * detivs[band] / sed[iband]
            sediv  += detivs [band] / sed[iband]**2
        sedmap /= np.maximum(1e-16, sediv)
        sedsn   = sedmap * np.sqrt(sediv)
        hot = np.maximum(hot, sedsn)

        if plots:
            plt.clf()
            dimshow(np.round(sedsn), vmin=0, vmax=10, cmap='hot')
            plt.title('SED-matched detection filter S/N: %s' % sedname)
            plt.colorbar()
            ps.savefig()

    del sedmap
    del sediv
    del sedsn
    if pipe:
        del detmaps
        del detivs

    # Read SDSS sources
    #check_photometric_calib(ims, cat, ps)
    cat,T = get_sdss_sources(bands, targetwcs)
    # record coordinates in target brick image
    ok,T.tx,T.ty = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.tx -= 1
    T.ty -= 1
    T.itx = np.clip(np.round(T.tx).astype(int), 0, W-1)
    T.ity = np.clip(np.round(T.ty).astype(int), 0, H-1)
    print 'SDSS sources:', Time()-tlast
    tlast = Time()

    peaks = (hot > 4)

    if plots:
        crossa = dict(ms=10, mew=1.5)
        plt.clf()
        dimshow(peaks)
        ax = plt.axis()
        plt.plot(T.itx, T.ity, 'r+', **crossa)
        plt.axis(ax)
        plt.title('Detection blobs + SDSS sources')
        ps.savefig()

    blobs,nblobs = label(peaks)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    # Un-set catalog blobs
    for x,y in zip(T.itx, T.ity):
        # blob number
        bb = blobs[y,x]
        if bb == 0:
            continue
        # un-set 'peaks' within this blob
        slc = blobslices[bb-1]
        peaks[slc][blobs[slc] == bb] = 0

    if plots:
        plt.clf()
        dimshow(peaks)
        ax = plt.axis()
        plt.plot(T.itx, T.ity, 'r+', **crossa)
        plt.axis(ax)
        plt.title('Detection blobs minus catalog sources')
        ps.savefig()

    # Now, after having removed catalog sources, crank up the detection threshold
    peaks &= (hot > 5)
        
    # zero out the edges(?)
    peaks[0 ,:] = 0
    peaks[:, 0] = 0
    peaks[-1,:] = 0
    peaks[:,-1] = 0
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[1:-1,2:  ])

    # These are our peaks
    pki = np.flatnonzero(peaks)
    peaky,peakx = np.unravel_index(pki, peaks.shape)
    print len(peaky), 'new peaks'

    print 'Peaks:', Time()-tlast
    tlast = Time()

    if plots:
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ax = plt.axis()
        plt.plot(T.tx, T.ty, 'r+', **crossa)
        plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
        plt.axis(ax)
        plt.title('Catalog + SED-matched detections')
        ps.savefig()
    
    # Add sources for the new peaks we found
    # make their initial fluxes ~ 5-sigma
    fluxes = dict([(b,[]) for b in bands])
    for tim in tims:
        psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
        fluxes[tim.band].append(5. * tim.sig1 / psfnorm)
    fluxes = dict([(b, np.mean(fluxes[b])) for b in bands])
    pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
    print 'Adding', len(pr), 'new sources'
    # Also create FITS table for new sources
    Tnew = fits_table()
    Tnew.ra  = pr
    Tnew.dec = pd
    Tnew.tx = peakx
    Tnew.ty = peaky
    Tnew.itx = np.clip(np.round(Tnew.tx).astype(int), 0, W-1)
    Tnew.ity = np.clip(np.round(Tnew.ty).astype(int), 0, H-1)
    for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
        cat.append(PointSource(RaDecPos(r,d),
                               NanoMaggies(order=bands, **fluxes)))
    # print 'Existing source table:'
    # T.about()
    # print 'New source table:'
    # Tnew.about()

    T = merge_tables([T, Tnew], columns='fillzero')


    # Segment, and record which sources fall into each blob
    # Grow the 'hot' pixels by dilating by a few pixels
    rr = 2.0
    RR = int(np.ceil(rr))
    S = 2*RR+1
    struc = (((np.arange(S)-RR)**2)[:,np.newaxis] +
             ((np.arange(S)-RR)**2)[np.newaxis,:]) <= rr**2
    hot2 = (hot > 5)
    hot2 = binary_dilation(hot2, structure=struc)

    blobs,nblobs = label(hot2)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]
    if pipe:
        # this is a crazy max-over-SEDs S/N map... but we only need it for ranking
        fluximg = hot
    else:
        fluximg = coimgs[1]
    blobsrcs = []
    blobflux = []
    keepblobs = []
    keepslices = []
    for blob in range(1, nblobs+1):
        Isrcs = np.flatnonzero(T.blob == blob)
        if len(Isrcs) == 0:
            continue
        blobsrcs.append(Isrcs)
        bslc = blobslices[blob-1]
        blobflux.append(np.sum(fluximg[bslc][blobs[bslc] == blob]))
        keepblobs.append(blob)
        keepslices.append(bslc)

    # bloblist = [1, 7, 8, 13, ...] -- blob NUMBERs (not indices)
    #  that actually contain sources
    blobslices = keepslices
    bloblist = keepblobs
    nblobs = len(bloblist)
    assert(nblobs == len(blobsrcs))
    assert(nblobs == len(blobflux))
    assert(nblobs == len(blobslices))
    print 'Keeping', nblobs, 'blobs, with', len(T), 'sources'
    print 'Blobs:', Time()-tlast
    tlast = Time()

    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')
    
    keys = ['T', 
            'bloblist','blobsrcs','blobflux','blobslices', 'blobs',
            'tractor', 'cat', 'ps']
    if not pipe:
        keys.extend(['detmaps', 'detivs'])
    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
    return rtn

def set_source_radii(bands, orig_wcsxy0, tims, cat, minsigma, minradius=3):
    # FIXME -- set source radii crudely, based on the maximum of the
    # PSF profiles in all images (!) -- should have a source x image
    # structure -- *and* based on SDSS fluxes.
    profiles = []
    R = 100
    minsig1s = dict([(band,1e100) for band in bands])
    for (ox0,oy0),tim in zip(orig_wcsxy0, tims):
        minsig1s[tim.band] = min(minsig1s[tim.band], tim.sig1)
        th,tw = tim.shape
        print 'PSF', tim.psf
        mog = tim.psf.getMixtureOfGaussians(px=ox0+(tw/2), py=oy0+(th/2))
        profiles.extend([
            mog.evaluate_grid(0, R, 0, 1, 0., 0.).patch.ravel(),
            mog.evaluate_grid(-(R-1), 1, 0, 1, 0., 0.).patch.ravel()[-1::-1],
            mog.evaluate_grid(0, 1, 0, R, 0., 0.).patch.ravel(),
            mog.evaluate_grid(0, 1, -(R-1), 1, 0., 0.).patch.ravel()[-1::-1]])
    profiles = np.array(profiles)
    print 'profiles', profiles.dtype, profiles.shape

    minradius = 3
    pro = np.max(profiles, axis=0)
    for src in cat:
        if not isinstance(src, PointSource):
            continue
        nsigmas = 0.
        bright = src.getBrightness()
        for band in bands:
            nsigmas = max(nsigmas, bright.getFlux(band) / minsig1s[band])
        if nsigmas <= 0:
            continue
        ii = np.flatnonzero(pro > (minsigma / nsigmas))
        if len(ii) == 0:
            continue
        src.fixedRadius = max(minradius, 1 + ii[-1])
        #print 'Nsigma', nsigmas, 'radius', src.fixedRadius

def stage_fitblobs(T=None, coimgs=None, cons=None,
                   detmaps=None, detivs=None,
                   bloblist=None, blobsrcs=None, blobflux=None,
                   blobslices=None, blobs=None,
                   tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
                   W=None,H=None, brickid=None,
                   bands=None, ps=None, tims=None,
                   plots=False, plots2=False,
                   **kwargs):
    for tim in tims:
        assert(np.all(np.isfinite(tim.getInvError())))

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    for tim in tims:
        from tractor.psfex import CachingPsfEx
        tim.psfex.radius = 20
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)

    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1

    set_source_radii(bands, orig_wcsxy0, tims, cat, minsigma)

    ### FIXME
    ## # Find sources that do not belong to a blob and add them as
    ## # singleton "blobs"; otherwise they don't get optimized.
    ## # for sources outside the image bounds, what should we do?

    # Fit blobs in order of flux
    tfitall = Time()
    iter = _blob_iter(bloblist, blobflux, blobslices, blobsrcs, blobs, targetwcs, tims,
                      orig_wcsxy0, cat, bands)
    # to allow debugpool to only queue tasks one at a time
    iter = iterwrapper(iter, len(bloblist))
    R = _map(_bounce_one_blob, iter)

    srcivs = [[] for src in cat]
    for Isrcs,fitsrcs,srcinvvars in R:
        for isrc,fitsrc,srciv in zip(Isrcs, fitsrcs, srcvars):
            src = cat[isrc]
            if isinstance(src, (DevGalaxy, ExpGalaxy)):
                src.shape = fitsrc.shape
            elif isinstance(src, FixedCompositeGalaxy):
                src.shapeExp = fitsrc.shapeExp
                src.shapeDev = fitsrc.shapeDev
            src.setParams(fitsrc.getParams())
            srcivs[isrc].extend(srciv)

    cat.thawAllRecursive()
    for i,src in enumerate(cat):
        print 'Source', i, src
        print 'variances:', srcivs[i]
        print len(srcivs[i]), 'vs', src.numberOfParams()
        if len(srcivs[i]) != src.numberOfParams():
            # This can happen for sources outside the brick bounds: they never get optimized?
            print 'Warning: zeroing variances for source', src
            srcivs[i] = [0]*src.numberOfParams()
            if isinstance(src, (DevGalaxy, ExpGalaxy)):
                src.shape = EllipseE.fromEllipseESoft(src.shape)
            elif isinstance(src, FixedCompositeGalaxy):
                src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
                src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)
        assert(len(srcivs[i]) == src.numberOfParams())
    invvars = np.hstack(srcivs)
    assert(len(invvars) == cat.numberOfParams())

    print 'Fitting sources took:', Time()-tfitall
    print 'Logprob:', tractor.getLogProb()
    
    rtn = dict()
    for k in ['tractor', 'tims', 'ps', 'invvars']:
        rtn[k] = locals()[k]
    return rtn
                          
def _blob_iter(bloblist, blobflux, blobslices, blobsrcs, blobs,
               targetwcs, tims, orig_wcsxy0, cat, bands):
    for blobnumber,iblob in enumerate(np.argsort(-np.array(blobflux))):

        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        if len(Isrcs) == 0:
            # This shouldn't happen any more... and may break iterwrapper dealie
            print 'Blob has no sources!'
            continue

        tblob = Time()
        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        print
        print 'Blob', blobnumber, 'of', len(blobflux), ':',
        print len(Isrcs), 'sources, size', blobw, 'x', blobh
        print

        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])

        subtimargs = []
        for itim,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            if sx1 < 0 or sy1 < 0 or sx1 > w or sy1 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subie  = tim.getInvError()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[itim]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            subtimargs.append((subimg, subie, subwcs, tim.subwcs, tim.getPhotoCal(),
                               tim.getSky(), tim.getPsf(), tim.name, sx0, sx1, sy0, sy1,
                               ox0, oy0, tim.band, tim.sig1, tim.modelMinval))

        blobmask = (blobs[bslc] == bloblist[iblob])
        #print 'Blob mask:', np.sum(blobmask), 'pixels'

        yield (Isrcs, targetwcs, bx0, by0, blobw, blobh, blobmask, subtimargs,
               [cat[i] for i in Isrcs], bands)

def _bounce_one_blob(X):
    try:
        return _one_blob(X)
    except:
        import traceback
        print 'Exception in _one_blob:'
        print 'args:', X
        traceback.print_exc()
        raise

def _one_blob((Isrcs, targetwcs, bx0, by0, blobw, blobh, blobmask, subtimargs,
               srcs, bands)):

    plots = False

    tlast = Time()
    alphas = [0.1, 0.3, 1.0]

    bigblob = (blobw * blobh) > 100*100

    subtims = []
    for (subimg, subie, twcs, subwcs, pcal,
         sky, psf, name, sx0, sx1, sy0, sy1, ox0, oy0,
         band,sig1,modelMinval) in subtimargs:

        # Mask out inverr for pixels that are not within the blob.
        subtarget = targetwcs.get_subimage(bx0, by0, blobw, blobh)
        subsubwcs = subwcs.get_subimage(int(sx0), int(sy0), int(sx1-sx0), int(sy1-sy0))
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(subsubwcs, subtarget, [], 2)
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue
        subie2 = np.zeros_like(subie)
        I = np.flatnonzero(blobmask[Yi,Xi])
        subie2[Yo[I],Xo[I]] = subie[Yo[I],Xo[I]]
        subie = subie2

        # If the subimage (blob) is small enough, instantiate a
        # constant PSF model in the center.
        if sy1-sy0 < 100 and sx1-sx0 < 100:
            subpsf = psf.mogAt(ox0 + (sx0+sx1)/2., oy0 + (sy0+sy1)/2.)
        else:
            # Otherwise, instantiate a (shifted) spatially-varying
            # PsfEx model.
            subpsf = ShiftedPsf(psf, ox0+sx0, oy0+sy0)

        subtim = Image(data=subimg, inverr=subie, wcs=twcs,
                       psf=subpsf, photocal=pcal, sky=sky, name=name)
        subtim.band = band
        subtim.sig1 = sig1
        subtim.modelMinval = modelMinval
        subtims.append(subtim)

        if plots:
            (Yo,Xo,Yi,Xi) = tim.resamp
            I = np.flatnonzero((Yi >= sy0) * (Yi < sy1) * (Xi >= sx0) * (Xi < sx1) *
                               (Yo >=  by0) * (Yo <  by1) * (Xo >=  bx0) * (Xo <  bx1))
            Yo = Yo[I] - by0
            Xo = Xo[I] - bx0
            Yi = Yi[I] - sy0
            Xi = Xi[I] - sx0
            subtim.resamp = (Yo, Xo, Yi, Xi)

            
    subcat = Catalog(*srcs)
    subtr = Tractor(subtims, subcat)
    subtr.freezeParam('images')

    # Try fitting fluxes first?
    subcat.thawAllRecursive()
    for src in srcs:
        src.freezeAllBut('brightness')
    for b in bands:
        tband = Time()
        for src in srcs:
            src.getBrightness().freezeAllBut(b)
        btims = []
        for tim in subtims:
            if tim.band != b:
                continue
            btims.append(tim)
        btr = Tractor(btims, subcat)
        btr.freezeParam('images')
        print 'Optimizing band', b, ':', btr
        print Time()-tband
        if useCeres:
            btr.optimize_forced_photometry(alphas=alphas, shared_params=False,
                                           use_ceres=True, BW=8, BH=8, wantims=False)
        else:
            try:
                btr.optimize_forced_photometry(alphas=alphas, shared_params=False,
                                               wantims=False)
            except:
                import traceback
                print 'Warning: Optimize_forced_photometry failed:'
                traceback.print_exc()
                # carry on
            
        print 'Band', b, 'took', Time()-tband
    subcat.thawAllRecursive()

    if plots:
        plotmods = []
        plotmodnames = []
        plotmods.append(subtr.getModelImages())
        plotmodnames.append('Initial')
        
    # Optimize individual sources in order of flux
    fluxes = []
    for src in subcat:
        # HACK -- here we just *sum* the nanomaggies in each band.  Bogus!
        br = src.getBrightness()
        flux = sum([br.getFlux(band) for band in bands])
        fluxes.append(flux)
    Ibright = np.argsort(-np.array(fluxes))

    if len(Ibright) > 1:
        # -Remember the original subtim images
        # -Compute initial models for each source (in each tim)
        # -Subtract initial models from images
        # -During fitting, for each source:
        #   -add back in the source's initial model (to each tim)
        #   -fit, with Catalog([src])
        #   -subtract final model (from each tim)
        # -Replace original subtim images
        #
        # --Might want to omit newly-added detection-filter sources, since their
        # fluxes are bogus.

        # Remember original tim images
        orig_timages = [tim.getImage() for tim in subtims]
        for tim,img in zip(subtims,orig_timages):
            tim.data = img.copy()
        initial_models = []

        # Create initial models for each tim x each source
        tt = Time()
        for tim in subtims:
            mods = []
            for src in subcat:
                mod = src.getModelPatch(tim)
                mods.append(mod)
                if mod is not None:
                    if mod.patch is not None:
                        if not np.all(np.isfinite(mod.patch)):
                            print 'Non-finite mod patch'
                            print 'source:', src
                            print 'tim:', tim
                            print 'PSF:', tim.getPsf()
                        assert(np.all(np.isfinite(mod.patch)))
                        mod.addTo(tim.getImage(), scale=-1)
            initial_models.append(mods)
        print 'Subtracting initial models:', Time()-tt

        # For sources, in decreasing order of brightness
        for numi,i in enumerate(Ibright):
            tsrc = Time()
            print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
            src = subcat[i]
            print src

            # Add this source's initial model back in.
            for tim,mods in zip(subtims, initial_models):
                mod = mods[i]
                if mod is not None:
                    mod.addTo(tim.getImage())

            if bigblob: # or True:
                # Create super-local sub-sub-tims around this source
                srctims = []
                for tim in subtims:
                    sz = 50
                    h,w = tim.shape
                    x,y = tim.getWcs().positionToPixel(src.getPosition())
                    if x < -sz or y < -sz or x > w+sz or y > h+sz:
                        continue
                    x,y = int(np.round(x)), int(np.round(y))
                    x0,y0 = max(x - sz, 0), max(y - sz, 0)
                    x1,y1 = min(x + sz, w), min(y + sz, h)
                    slc = slice(y0,y1), slice(x0, x1)
                    wcs = tim.getWcs().copy()
                    wx0,wy0 = wcs.getX0Y0()
                    wcs.setX0Y0(wx0 + x0, wy0 + y0)
                    srctim = Image(data=tim.getImage ()[slc],
                                   inverr=tim.getInvError()[slc],
                                   wcs=wcs, psf=ShiftedPsf(tim.getPsf(), x0, y0),
                                   photocal=tim.getPhotoCal(),
                                   sky=tim.getSky(), name=tim.name)
                    srctim.band = tim.band
                    srctim.sig1 = tim.sig1
                    srctim.modelMinval = tim.modelMinval
                    srctims.append(srctim)
                    print 'Big blob: srctim', srctim.shape, 'vs sub', tim.shape
            else:
                srctims = subtims

            srctractor = Tractor(srctims, [src])
            srctractor.freezeParams('images')
            
            # # Try fitting flux first?
            # src.freezeAllBut('brightness')
            # for b in bands:
            #     tband = Time()
            #     src.getBrightness().freezeAllBut(b)
            #     print 'Optimizing band', b, ':', srctractor
            #     srctractor.printThawedParams()
            #     srctractor.optimize_forced_photometry(alphas=alphas, shared_params=False,
            #                                           use_ceres=True, BW=8, BH=8,
            #                                           wantims=False)
            #     print 'Band', b, 'took', Time()-tband
            # src.getBrightness().thawAllParams()
            # src.thawAllParams()

            print 'Optimizing:', srctractor
            srctractor.printThawedParams()
            print 'Tim shapes:', [tim.shape for tim in srctims]

            if plots:
                spmods,spnames = [],[]
                spallmods,spallnames = [],[]
            if plots and numi == 0:
                spmods.append(srctractor.getModelImages())
                spnames.append('Initial')
                spallmods.append(subtr.getModelImages())
                spallnames.append('Initial (all)')

            max_cpu_per_source = 60.

            cpu0 = time.clock()
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                print 'dlnp:', dlnp, 'src', src

                if time.clock()-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    break

                if dlnp < 0.1:
                    break

            if plots:
                spmods.append(srctractor.getModelImages())
                spnames.append('Fit')
                spallmods.append(subtr.getModelImages())
                spallnames.append('Fit (all)')

            if plots:
                plt.figure(1, figsize=(8,6))
                plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                                    hspace=0.1, wspace=0.05)
                plt.figure(2, figsize=(3,3))
                plt.subplots_adjust(left=0.005, right=0.995, top=0.995,bottom=0.005)
                #_plot_mods(subtims, spmods, spnames, bands, None, None, bslc, blobw, blobh, ps,
                #           chi_plots=plots2)
                tempims = [tim.getImage() for tim in subtims]
                for tim,orig in zip(subtims, orig_timages):
                    tim.data = orig
                _plot_mods(subtims, spallmods, spallnames, bands, None, None, bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False)
                for tim,im in zip(subtims, tempims):
                    tim.data = im

            for tim in subtims:
                mod = src.getModelPatch(tim)
                if mod is not None:
                    mod.addTo(tim.getImage(), scale=-1)

            print 'Fitting source took', Time()-tsrc
            print src

            gc = get_galaxy_cache()
            print 'Galaxy cache:', gc
            if gc is not None:
                gc.printStats()
                print 'GC total size:', gc.totalSize()
                gc.clear()
                print 'After clearing cache:', Time()-tsrc

        for tim,img in zip(subtims, orig_timages):
            tim.data = img
        del orig_timages
        del initial_models
        
    else:
        # Fit sources one at a time, but don't subtract other models
        subcat.freezeAllParams()
        for numi,i in enumerate(Ibright):
            tsrc = Time()
            print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
            print subcat[i]
            subcat.freezeAllBut(i)
            print 'Optimizing:', subtr
            subtr.printThawedParams()
            for step in range(10):
                dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                print 'dlnp:', dlnp
                if dlnp < 0.1:
                    break
            print 'Fitting source took', Time()-tsrc
            print subcat[i]

    if plots:
        plotmods.append(subtr.getModelImages())
        plotmodnames.append('Per Source')

    if len(srcs) > 1 and len(srcs) <= 10:
        tfit = Time()
        # Optimize all at once?
        subcat.thawAllParams()
        print 'Optimizing:', subtr
        subtr.printThawedParams()
        for step in range(20):
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                          alphas=alphas)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        print 'Simultaneous fit took:', Time()-tfit

        if plots:
            plotmods.append(subtr.getModelImages())
            plotmodnames.append('All Sources')


    if plots:
        _plot_mods(subtims, plotmods, plotmodnames, bands, coimgs, cons, bslc, blobw, blobh, ps)
        if blobnumber >= 10:
            plots = False

    # FIXME -- for large blobs, fit strata of sources simultaneously?

    print 'Blob finished fitting:', Time()-tlast
    tlast = Time()

    # Variances
    srcinvvars = [[] for src in srcs]
    subcat.thawAllRecursive()
    subcat.freezeAllParams()
    for isub in range(len(srcs)):
        print 'Variances for source', isub
        subcat.thawParam(isub)
        src = subcat[isub]
        print 'Source', src
        print 'Params:', src.getParamNames()
        
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseE.fromEllipseESoft(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
            src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)

        print 'Converted ellipse:', src

        allderivs = subtr.getDerivs()
        for iparam,derivs in enumerate(allderivs):
            dchisq = 0
            for deriv,tim in derivs:
                h,w = tim.shape
                deriv.clipTo(w,h)
                ie = tim.getInvError()
                slc = deriv.getSlice(ie)
                chi = deriv.patch * ie[slc]
                dchisq += (chi**2).sum()
            srcinvvars[isub].append(dchisq)
        assert(len(srcinvvars[isub]) == subcat[isub].numberOfParams())
        subcat.freezeParam(isub)
    print 'Blob variances:', Time()-tlast

    return Isrcs, srcs, srcinvvars


'''
Re-fit sources one at a time.
'''
def stage3(T=None, sedsn=None, coimgs=None, cons=None,
           detmaps=None, detivs=None,
           blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
           tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
           W=None,H=None, brickid=None,
           bands=None, ps=None, tims=None,
           plots=False, plots2=False,
           **kwargs):
    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]

    for tim in tims:
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = tim.psfex

    cat.thawAllRecursive()
    # rough: sum flux in all bands
    bright = []
    for src in cat:
        br = src.getBrightness()
        bright.append(sum([br.getFlux(band) for band in bands]))
        src.minRadius = 3
    I = np.argsort(-np.array(bright))

    # Remember original tim images
    orig_timages = [tim.getImage().copy() for tim in tims]
    initial_models = []
    # Create initial models for each tim x each source
    tt = Time()
    for tim in tims:
        mods = []
        for src in cat:
            mod = src.getModelPatch(tim)
            mods.append(mod)
            if mod is not None:
                mod.addTo(tim.getImage(), scale=-1)
        initial_models.append(mods)
    print 'Subtracting initial models:', Time()-tt

    for i in I:
        src = cat[i]
        print 'Fitting source', i
        print src

        srctractor = Tractor(tims, [src])
        srctractor.freezeParams('images')

        # Add this source's initial model back in.
        for tim,mods in zip(tims, initial_models):
            mod = mods[i]
            if mod is not None:
                mod.addTo(tim.getImage())

        print 'Optimizing:', srctractor
        srctractor.printThawedParams()

        alphas = [0.1, 0.3, 1.0]
        for step in range(50):
            dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                               alphas=alphas)
            print 'dlnp:', dlnp, 'src', src
            print 'Update:', X
            print 'src params:', src.getParams()

            if plots:
                comods = []
                for iband,band in enumerate(bands):
                    comod = np.zeros((H,W))
                    con = np.zeros((H,W))
                    for itim,tim in enumerate(tims):
                        if tim.band != band:
                            continue
                        (Yo,Xo,Yi,Xi) = tim.resamp
                        mod = srctractor.getModelImage(tim)
                        comod[Yo,Xo] += mod[Yi,Xi]
                        con[Yo,Xo] += 1
                    comod /= np.maximum(con, 1)
                    comods.append(comod)
                plt.clf()
                dimshow(get_rgb(comods, bands), ticks=False)
                ps.savefig()
    
            if dlnp < 0.1:
                break

        for tim in tims:
            mod = src.getModelPatch(tim)
            if mod is not None:
                mod.addTo(tim.getImage(), scale=-1)

    for tim,img in zip(tims, orig_timages):
        tim.data = img

    # for i in I:
    #     cat.freezeAllBut(i)
    #     print 'Fitting source', i
    #     src = cat[i]
    #     print src
    # 
    #     tractor.printThawedParams()
    # 
    #     alphas = [0.1, 0.3, 1.0]
    #     for step in range(50):
    #         dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False,
    #                                         alphas=alphas)
    #         print 'dlnp:', dlnp, 'src', src
    #         print 'Update:', X
    #         print 'src params:', src.getParams()
    # 
    #         if plots:
    #             comods = []
    #             for iband,band in enumerate(bands):
    #                 comod = np.zeros((H,W))
    #                 con = np.zeros((H,W))
    #                 for itim,tim in enumerate(tims):
    #                     if tim.band != band:
    #                         continue
    #                     (Yo,Xo,Yi,Xi) = tim.resamp
    #                     mod = tractor.getModelImage(tim)
    #                     comod[Yo,Xo] += mod[Yi,Xi]
    #                     con[Yo,Xo] += 1
    #                 comod /= np.maximum(con, 1)
    #                 comods.append(comod)
    #             plt.clf()
    #             dimshow(get_rgb(comods, bands), ticks=False)
    #             ps.savefig()
    # 
    #         if dlnp < 0.1:
    #             break


def stage4(T=None,
           blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
           tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
           W=None,H=None,
           bands=None, ps=None, tims=None,
           plots=False, plots2=False,
           **kwargs):
    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]






def _plot_mods(tims, mods, titles, bands, coimgs, cons, bslc, blobw, blobh, ps,
               chi_plots=True, rgb_plots=False, main_plot=True):
    subims = [[] for m in mods]
    chis = dict([(b,[]) for b in bands])
    
    make_coimgs = (coimgs is None)
    if make_coimgs:
        coimgs = [np.zeros((blobh,blobw)) for b in bands]
        cons   = [np.zeros((blobh,blobw)) for b in bands]

    for iband,band in enumerate(bands):
        comods = [np.zeros((blobh,blobw)) for m in mods]
        cochis = [np.zeros((blobh,blobw)) for m in mods]
        comodn = np.zeros((blobh,blobw))

        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            rechi = np.zeros((blobh,blobw))
            chilist = []
            comodn[Yo,Xo] += 1
            for imod,mod in enumerate(mods):
                chi = ((tim.getImage()[Yi,Xi] - mod[itim][Yi,Xi]) *
                       tim.getInvError()[Yi,Xi])
                rechi[Yo,Xo] = chi
                chilist.append((rechi.copy(), itim))
                cochis[imod][Yo,Xo] += chi
                comods[imod][Yo,Xo] += mod[itim][Yi,Xi]
            chis[band].append(chilist)
            mn,mx = -10.*tim.sig1, 30.*tim.sig1

            if make_coimgs:
                nn = (tim.getInvError()[Yi,Xi] > 0)
                coimgs[iband][Yo,Xo] += tim.getImage()[Yi,Xi] * nn
                cons  [iband][Yo,Xo] += nn
                
        if make_coimgs:
            coimgs[iband] /= np.maximum(cons[iband], 1)
            coimg  = coimgs[iband]
            coimgn = cons  [iband]
        else:
            coimg = coimgs[iband][bslc]
            coimgn = cons[iband][bslc]
            
        for comod in comods:
            comod /= np.maximum(comodn, 1)
        ima = dict(vmin=mn, vmax=mx, ticks=False)
        for subim,comod,cochi in zip(subims, comods, cochis):
            subim.append((coimg, coimgn, comod, ima, cochi))

    # Plot per-band image, model, and chi coadds, and RGB images
    rgba = dict(ticks=False)
    rgbs = []
    plt.figure(1)
    for i,subim in enumerate(subims):
        plt.clf()
        rows,cols = 3,5
        imgs = []
        themods = []
        resids = []
        for j,(img,imgn,mod,ima,chi) in enumerate(subim):
            imgs.append(img)
            themods.append(mod)
            resid = img - mod
            resid[imgn == 0] = np.nan
            resids.append(resid)

            if main_plot:
                plt.subplot(rows,cols,1 + j + 0)
                dimshow(img, **ima)
                plt.subplot(rows,cols,1 + j + cols)
                dimshow(mod, **ima)
                plt.subplot(rows,cols,1 + j + cols*2)
                # dimshow(-chi, **imchi)
                # dimshow(imgn, vmin=0, vmax=3)
                dimshow(resid, nancolor='r')
        rgb = get_rgb(imgs, bands)
        if i == 0:
            rgbs.append(rgb)
        if main_plot:
            plt.subplot(rows,cols, 4)
            dimshow(rgb, **rgba)
        rgb = get_rgb(themods, bands)
        rgbs.append(rgb)
        if main_plot:
            plt.subplot(rows,cols, cols+4)
            dimshow(rgb, **rgba)
            plt.subplot(rows,cols, cols*2+4)
            dimshow(get_rgb(resids, bands, mnmx=(-10,10)), **rgba)

            mnmx = -5,300
            kwa = dict(mnmx=mnmx, arcsinh=1)
            plt.subplot(rows,cols, 5)
            dimshow(get_rgb(imgs, bands, **kwa), **rgba)
            plt.subplot(rows,cols, cols+5)
            dimshow(get_rgb(themods, bands, **kwa), **rgba)
            plt.subplot(rows,cols, cols*2+5)
            mnmx = -100,100
            kwa = dict(mnmx=mnmx, arcsinh=1)
            dimshow(get_rgb(resids, bands, **kwa), **rgba)
            plt.suptitle(titles[i])
            ps.savefig()

    if rgb_plots:
        # RGB image and model
        plt.figure(2)
        for rgb in rgbs:
            plt.clf()
            dimshow(rgb, **rgba)
            ps.savefig()

    if not chi_plots:
        return

    plt.figure(1)
    # Plot per-image chis: in a grid with band along the rows and images along the cols
    cols = max(len(v) for v in chis.values())
    rows = len(bands)
    for imod in range(len(mods)):
        plt.clf()
        for row,band in enumerate(bands):
            sp0 = 1 + cols*row
            # chis[band] = [ (one for each tim:) [ (one for each mod:) (chi,itim), (chi,itim) ], ...]
            for col,chilist in enumerate(chis[band]):
                chi,itim = chilist[imod]
                plt.subplot(rows, cols, sp0 + col)
                dimshow(-chi, **imchi)
                plt.xticks([]); plt.yticks([])
                plt.title(tims[itim].name)
        #plt.suptitle(titles[imod])
        ps.savefig()




'''
PSF plots
'''
def stage_psfplots(
    T=None, sedsn=None, coimgs=None, cons=None,
    detmaps=None, detivs=None,
    blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
    tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None, brickid=None,
    bands=None, ps=None, tims=None,
    plots=False,
    **kwargs):

    tim = tims[0]
    tim.psfex.fitSavedData(*tim.psfex.splinedata)
    spl = tim.psfex.splines[0]
    print 'Spline:', spl
    knots = spl.get_knots()
    print 'knots:', knots
    tx,ty = knots
    k = 3
    print 'interior knots x:', tx[k+1:-k-1]
    print 'additional knots x:', tx[:k+1], 'and', tx[-k-1:]
    print 'interior knots y:', ty[k+1:-k-1]
    print 'additional knots y:', ty[:k+1], 'and', ty[-k-1:]

    for itim,tim in enumerate(tims):
        psfex = tim.psfex
        psfex.fitSavedData(*psfex.splinedata)
        if plots:
            print
            print 'Tim', tim
            print
            pp,xx,yy = psfex.splinedata
            ny,nx,nparams = pp.shape
            assert(len(xx) == nx)
            assert(len(yy) == ny)
            psfnil = psfex.psfclass(*np.zeros(nparams))
            names = psfnil.getParamNames()
            xa = np.linspace(xx[0], xx[-1],  50)
            ya = np.linspace(yy[0], yy[-1], 100)
            #xa,ya = np.meshgrid(xa,ya)
            #xa = xa.ravel()
            #ya = ya.ravel()
            print 'xa', xa
            print 'ya', ya
            for i in range(nparams):
                plt.clf()
                plt.subplot(1,2,1)
                dimshow(pp[:,:,i])
                plt.title('grid fit')
                plt.colorbar()
                plt.subplot(1,2,2)
                sp = psfex.splines[i](xa, ya)
                sp = sp.T
                print 'spline shape', sp.shape
                assert(sp.shape == (len(ya),len(xa)))
                dimshow(sp, extent=[xx[0],xx[-1],yy[0],yy[-1]])
                plt.title('spline')
                plt.colorbar()
                plt.suptitle('tim %s: PSF param %s' % (tim.name, names[i]))
                ps.savefig()

def stage_initplots(
    coimgs=None, cons=None, bands=None, ps=None,
    targetwcs=None,
    blobs=None,
    T=None, cat=None, tims=None, tractor=None, **kwargs):
    # RGB image
    # plt.clf()
    # dimshow(get_rgb(coimgs, bands))
    # ps.savefig()

    print 'T:'
    T.about()

    # cluster zoom-in
    #x0,x1, y0,y1 = 1700,2700, 200,1200
    #x0,x1, y0,y1 = 1900,2500, 400,1000
    #x0,x1, y0,y1 = 1900,2400, 450,950
    x0,x1, y0,y1 = 0,500, 0,500

    clco = [co[y0:y1, x0:x1] for co in coimgs]
    clW, clH = x1-x0, y1-y0
    clwcs = targetwcs.get_subimage(x0, y0, clW, clH)

    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    ps.suffixes = ['png','pdf']

    # cluster zoom-in
    rgb = get_rgb(clco, bands)
    plt.clf()
    dimshow(rgb, ticks=False)
    ps.savefig()

    # blobs
    #b0 = blobs
    #b1 = binary_dilation(blobs, np.ones((3,3)))
    #bout = np.logical_and(b1, np.logical_not(b0))
    b0 = blobs
    b1 = binary_erosion(b0, np.ones((3,3)))
    bout = np.logical_and(b0, np.logical_not(b1))
    # set green
    rgb[:,:,0][bout] = 0.
    rgb[:,:,1][bout] = 1.
    rgb[:,:,2][bout] = 0.
    plt.clf()
    dimshow(rgb, ticks=False)
    ps.savefig()

    # Initial model (SDSS only)
    try:
        # convert from string to int
        T.objid = np.array([int(x) if len(x) else 0 for x in T.objid])
    except:
        pass
    scat = Catalog(*[cat[i] for i in np.flatnonzero(T.objid)])
    sedcat = Catalog(*[cat[i] for i in np.flatnonzero(T.objid == 0)])

    print len(cat), 'total sources'
    print len(scat), 'SDSS sources'
    print len(sedcat), 'SED-matched sources'
    tr = Tractor(tractor.images, scat)

    comods = []
    comods2 = []
    for iband,band in enumerate(bands):
        comod = np.zeros((clH,clW))
        comod2 = np.zeros((clH,clW))
        con = np.zeros((clH,clW))
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            mod = tr.getModelImage(tim)
            Yo -= y0
            Xo -= x0
            K = np.nonzero((Yo >= 0) * (Yo < clH) * (Xo >= 0) * (Xo < clW))
            Xo,Yo,Xi,Yi = Xo[K],Yo[K],Xi[K],Yi[K]
            comod[Yo,Xo] += mod[Yi,Xi]
            ie = tim.getInvError()
            noise = np.random.normal(size=ie.shape) / ie
            noise[ie == 0] = 0.
            comod2[Yo,Xo] += mod[Yi,Xi] + noise[Yi,Xi]
            con[Yo,Xo] += 1
        comod /= np.maximum(con, 1)
        comods.append(comod)
        comod2 /= np.maximum(con, 1)
        comods2.append(comod2)
    
    plt.clf()
    dimshow(get_rgb(comods2, bands), ticks=False)
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(comods, bands), ticks=False)
    ps.savefig()

    # Overplot SDSS sources
    ax = plt.axis()
    for src in scat:
        rd = src.getPosition()
        ok,x,y = clwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', ms=10, mew=1.5)
        else:
            plt.plot(x-1, y-1, 'o', mec='r', mfc='none', ms=10, mew=1.5)
    plt.axis(ax)
    ps.savefig()

    # Add SED-matched detections
    for src in sedcat:
        rd = src.getPosition()
        ok,x,y = clwcs.radec2pixelxy(rd.ra, rd.dec)
        plt.plot(x-1, y-1, 'o', mec='c', mfc='none', ms=10, mew=1.5)
    plt.axis(ax)
    ps.savefig()

    # Mark SED-matched detections on image
    plt.clf()
    dimshow(get_rgb(clco, bands), ticks=False)
    ax = plt.axis()
    for src in sedcat:
        rd = src.getPosition()
        ok,x,y = clwcs.radec2pixelxy(rd.ra, rd.dec)
        #plt.plot(x-1, y-1, 'o', mec='c', mfc='none', ms=10, mew=1.5)
        x,y = x-1, y-1
        hi,lo = 20,7
        # plt.plot([x-lo,x-hi],[y,y], 'c-')
        # plt.plot([x+lo,x+hi],[y,y], 'c-')
        # plt.plot([x,x],[y+lo,y+hi], 'c-')
        # plt.plot([x,x],[y-lo,y-hi], 'c-')
        plt.annotate('', (x,y+lo), xytext=(x,y+hi),
                     arrowprops=dict(color='c', width=1, frac=0.3, headwidth=5))
    plt.axis(ax)
    ps.savefig()

    # plt.clf()
    # dimshow(get_rgb([gaussian_filter(x,1) for x in clco], bands), ticks=False)
    # ps.savefig()

    # Resid
    # plt.clf()
    # dimshow(get_rgb([im-mo for im,mo in zip(clco,comods)], bands), ticks=False)
    # ps.savefig()

    # find SDSS fields within that WCS
    #sdss = DR9(basedir=photoobjdir)
    #sdss.useLocalTree()
    sdss = DR9(basedir='tmp')
    sdss.saveUnzippedFiles('tmp')

    #wfn = sdss.filenames.get('window_flist', None)
    wfn = os.path.join(os.environ['PHOTO_RESOLVE'], 'window_flist.fits')
    
    clra,cldec = clwcs.radec_center()
    clrad = clwcs.radius()
    clrad = clrad + np.hypot(10.,14.)/2./60.
    print 'Searching for run,camcol,fields with radius', clrad, 'deg'
    RCF = radec_to_sdss_rcf(clra, cldec, radius=clrad*60., tablefn=wfn)
    print 'Found %i fields possibly in range' % len(RCF)

    sdsscoimgs = [np.zeros((clH,clW),np.float32) for band in bands]
    sdsscons   = [np.zeros((clH,clW),np.float32) for band in bands]
    for run,camcol,field,r,d in RCF:
        for iband,band in enumerate(bands):
            bandnum = band_index(band)
            sdss.retrieve('frame', run, camcol, field, band)
            frame = sdss.readFrame(run, camcol, field, bandnum)
            print 'Got frame', frame
            h,w = frame.getImageShape()
            simg = frame.getImage()
            wcs = AsTransWrapper(frame.astrans, w, h, 0.5, 0.5)
            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(clwcs, wcs, [], 3)
            except OverlapError:
                continue
            sdsscoimgs[iband][Yo,Xo] += simg[Yi,Xi]
            sdsscons  [iband][Yo,Xo] += 1
    for co,n in zip(sdsscoimgs, sdsscons):
        co /= np.maximum(1e-6, n)

    plt.clf()
    dimshow(get_rgb(sdsscoimgs, bands), ticks=False)
    #plt.title('SDSS')
    ps.savefig()


    # plt.clf()
    # plt.loglog(clco[0].ravel(), sdsscoimgs[0].ravel(), 'm.', alpha=0.1)
    # plt.loglog(clco[1].ravel(), sdsscoimgs[1].ravel(), 'r.', alpha=0.1)
    # plt.loglog(clco[2].ravel(), sdsscoimgs[2].ravel(), 'g.', alpha=0.1)
    # ps.savefig()
    # 
    # for i,c in enumerate(['m','r','g']):
    #     plt.clf()
    #     plt.loglog(sdsscoimgs[i].ravel(), clco[i].ravel()/sdsscoimgs[i].ravel(), '.', color=c, alpha=0.1)
    #     plt.ylim(0.1, 10.0)
    #     ps.savefig()

    #wa = dict(clobber=True, header=hdr)
    #fitsio.write('image-coadd-%06i-%s.fits' % (brickid, band), comod, **wa)


def _get_mod((tim, srcs)):
    tractor = Tractor([tim], srcs)
    return tractor.getModelImage(0)
    
'''
Plots; single-image image,invvar,model FITS files
'''
def stage_fitplots(
    T=None, coimgs=None, cons=None,
    cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None, brickid=None,
    plots=False, plots2=False, tims=None, tractor=None,
    pipe=None,
    outdir=None,
    **kwargs):

    for tim in tims:
        print 'Tim', tim, 'PSF', tim.getPsf()
        
    writeModels = False

    if pipe:
        t0 = Time()
        # Produce per-band coadds, for plots
        coimgs,cons = compute_coadds(tims, bands, W, H, targetwcs)
        print 'Coadds:', Time()-t0

    plt.figure(figsize=(10,10.5))
    #plt.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)
    plt.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.95)

    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    plt.title('Image')
    ps.savefig()

    ax = plt.axis()
    cat = tractor.getCatalog()
    for i,src in enumerate(cat):
        rd = src.getPosition()
        ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, '+', color=cc, ms=10, mew=1.5)
        else:
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', ms=10, mew=1.5)
        # plt.text(x, y, '%i' % i, color=cc, ha='center', va='bottom')
    plt.axis(ax)
    ps.savefig()

    mnmx = -5,300
    arcsinha = dict(mnmx=mnmx, arcsinh=1)

    # After plot
    rgbmod = []
    rgbmod2 = []
    rgbresids = []
    rgbchisqs = []

    chibins = np.linspace(-10., 10., 200)
    chihist = [np.zeros(len(chibins)-1, int) for band in bands]

    wcsW = targetwcs.get_width()
    wcsH = targetwcs.get_height()
    print 'Target WCS shape', wcsW,wcsH

    t0 = Time()
    mods = _map(_get_mod, [(tim, cat) for tim in tims])
    print 'Getting model images:', Time()-t0

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    for iband,band in enumerate(bands):
        coimg = coimgs[iband]
        comod  = np.zeros((wcsH,wcsW), np.float32)
        comod2 = np.zeros((wcsH,wcsW), np.float32)
        cochi2 = np.zeros((wcsH,wcsW), np.float32)
        for itim, (tim,mod) in enumerate(zip(tims, mods)):
            if tim.band != band:
                continue

            #mod = tractor.getModelImage(tim)

            if plots2:
                plt.clf()
                dimshow(tim.getImage(), **tim.ima)
                plt.title(tim.name)
                ps.savefig()
                plt.clf()
                dimshow(mod, **tim.ima)
                plt.title(tim.name)
                ps.savefig()
                plt.clf()
                dimshow((tim.getImage() - mod) * tim.getInvError(), **imchi)
                plt.title(tim.name)
                ps.savefig()

            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            comod[Yo,Xo] += mod[Yi,Xi]
            ie = tim.getInvError()
            noise = np.random.normal(size=ie.shape) / ie
            noise[ie == 0] = 0.
            comod2[Yo,Xo] += mod[Yi,Xi] + noise[Yi,Xi]
            chi = ((tim.getImage()[Yi,Xi] - mod[Yi,Xi]) * tim.getInvError()[Yi,Xi])
            cochi2[Yo,Xo] += chi**2
            chi = chi[chi != 0.]
            hh,xe = np.histogram(np.clip(chi, -10, 10).ravel(), bins=chibins)
            chihist[iband] += hh

            if not writeModels:
                continue

            im = tim.imobj
            fn = 'image-b%06i-%s-%s.fits' % (brickid, band, im.name)

            wcsfn = create_temp()
            wcs = tim.getWcs().wcs
            x0,y0 = orig_wcsxy0[itim]
            h,w = tim.shape
            subwcs = wcs.get_subimage(int(x0), int(y0), w, h)
            subwcs.write_to(wcsfn)

            primhdr = fitsio.FITSHDR()
            primhdr.add_record(dict(name='X0', value=x0, comment='Pixel origin of subimage'))
            primhdr.add_record(dict(name='Y0', value=y0, comment='Pixel origin of subimage'))
            xfn = im.wcsfn.replace(decals_dir+'/', '')
            primhdr.add_record(dict(name='WCS_FILE', value=xfn))
            xfn = im.psffn.replace(decals_dir+'/', '')
            primhdr.add_record(dict(name='PSF_FILE', value=xfn))
            primhdr.add_record(dict(name='INHERIT', value=True))

            imhdr = fitsio.read_header(wcsfn)
            imhdr.add_record(dict(name='EXTTYPE', value='IMAGE', comment='This HDU contains image data'))
            ivhdr = fitsio.read_header(wcsfn)
            ivhdr.add_record(dict(name='EXTTYPE', value='INVVAR', comment='This HDU contains an inverse-variance map'))
            fits = fitsio.FITS(fn, 'rw', clobber=True)
            tim.toFits(fits, primheader=primhdr, imageheader=imhdr, invvarheader=ivhdr)

            imhdr.add_record(dict(name='EXTTYPE', value='MODEL', comment='This HDU contains a Tractor model image'))
            fits.write(mod, header=imhdr)
            print 'Wrote image and model to', fn
            
        comod  /= np.maximum(cons[iband], 1)
        comod2 /= np.maximum(cons[iband], 1)

        rgbmod.append(comod)
        rgbmod2.append(comod2)
        resid = coimg - comod
        resid[cons[iband] == 0] = np.nan
        rgbresids.append(resid)
        rgbchisqs.append(cochi2)

        # Plug the WCS header cards into these images
        wcsfn = create_temp()
        targetwcs.write_to(wcsfn)
        hdr = fitsio.read_header(wcsfn)
        os.remove(wcsfn)

        if outdir is None:
            outdir = '.'
        wa = dict(clobber=True, header=hdr)
        for name,img in [('image', coimg), ('model', comod), ('resid', resid), ('chi2', cochi2)]:
            fn = os.path.join(outdir, '%s-coadd-%06i-%s.fits' % (name, brickid, band))
            fitsio.write(fn, img,  **wa)
            print 'Wrote', fn

    del cons

    plt.clf()
    dimshow(get_rgb(rgbmod, bands))
    plt.title('Model')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbmod2, bands))
    plt.title('Model + Noise')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbresids, bands))
    plt.title('Residuals')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbresids, bands, mnmx=(-30,30)))
    plt.title('Residuals (2)')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(coimgs, bands, **arcsinha))
    plt.title('Image (stretched)')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbmod2, bands, **arcsinha))
    plt.title('Model + Noise (stretched)')
    ps.savefig()

    del coimgs
    del rgbresids
    del rgbmod
    del rgbmod2

    plt.clf()
    g,r,z = rgbchisqs
    im = np.log10(np.dstack((z,r,g)))
    mn,mx = 0, im.max()
    dimshow(np.clip((im - mn) / (mx - mn), 0., 1.))
    plt.title('Chi-squared')
    ps.savefig()

    plt.clf()
    xx = np.repeat(chibins, 2)[1:-1]
    for y,cc in zip(chihist, 'grm'):
        plt.plot(xx, np.repeat(np.maximum(0.1, y),2), '-', color=cc)
    plt.xlabel('Chi')
    plt.yticks([])
    plt.axvline(0., color='k', alpha=0.25)
    ps.savefig()

    plt.yscale('log')
    mx = np.max([max(y) for y in chihist])
    plt.ylim(1, mx * 1.05)
    ps.savefig()

    return dict(tims=tims)


'''
Write catalog output
'''
def stage_writecat(
    version_header=None,
    T=None, coimgs=None, cons=None,
    cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False, tractor=None,
    brickid=None,
    invvars=None,
    catalogfn=None,
    **kwargs):
    from desi_common import prepare_fits_catalog
    fs = None
    TT = T.copy()
    for k in ['itx','ity','index']:
        TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob']:
            TT.rename(col, 'sdss_%s' % col)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.objid   = np.arange(len(TT)).astype(np.int32)

    cat.thawAllRecursive()
    hdr = version_header
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs)

    # Convert from variances to invvars
    for k in ['ra_var', 'dec_var', 'shapeExp_var', 'shapeDev_var', 'fracDev_var']:
        X = T2.get(k)
        T2.delete_column(k)
        T2.set(k.replace('_var', '_invvar'), (1./X).astype(np.float32))

    # Create DECAM_FLUX columns: [ugrizY]
    bandindex = dict(g=1, r=2, z=4)

    T2.decam_flux      = np.zeros((len(T2), 6), np.float32)
    T2.decam_flux_isig = np.zeros((len(T2), 6), np.float32)
    for band in bands:
        T2.decam_flux[:, bandindex[band]] = T2.get('decam_%s_nanomaggies' % band)
        T2.decam_flux_isig[:, bandindex[band]] = np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band))

    # Unpack shape columns
    T2.shapeExp_r = T2.shapeExp[:,0]
    T2.shapeExp_e1 = T2.shapeExp[:,1]
    T2.shapeExp_e2 = T2.shapeExp[:,2]
    T2.shapeDev_r = T2.shapeExp[:,0]
    T2.shapeDev_e1 = T2.shapeExp[:,1]
    T2.shapeDev_e2 = T2.shapeExp[:,2]
    T2.shapeExp_r_invvar  = T2.shapeExp_invvar[:,0]
    T2.shapeExp_e1_invvar = T2.shapeExp_invvar[:,1]
    T2.shapeExp_e2_invvar = T2.shapeExp_invvar[:,2]
    T2.shapeDev_r_invvar  = T2.shapeExp_invvar[:,0]
    T2.shapeDev_e1_invvar = T2.shapeExp_invvar[:,1]
    T2.shapeDev_e2_invvar = T2.shapeExp_invvar[:,2]

    if catalogfn is not None:
        fn = catalogfn
    else:
        fn = 'tractor-phot-b%i.fits' % brickid
    T2.writeto(fn, header=hdr)
    print 'Wrote', fn

    return dict(variances=variances, cat=cat)

if __name__ == '__main__':
    from astrometry.util.stages import *
    import optparse
    import logging

    ep = '''
eg, to run a small field containing a cluster:
\n
python -u projects/desi/runbrick.py --plots --brick 371589 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle
\n
'''
    parser = optparse.OptionParser(epilog=ep)
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[],
                      help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_option('-s', '--stage', dest='stage', default=[], action='append',
                      help="Run up to the given stage(s)")
    parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')
    parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
                      help='Make more verbose')

    parser.add_option('-b', '--brick', type=int, help='Brick ID to run: default %default',
                      default=377306)

    parser.add_option('--threads', type=int, help='Run multi-threaded')
    parser.add_option('-p', '--plots', dest='plots', action='store_true',
                      help='Per-blob plots?')
    parser.add_option('--plots2', action='store_true',
                      help='More plots?')

    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='pickles/runbrick-%(brick)06i-%%(stage)s.pickle')

    plot_base_default = 'brick-%(brick)06i'
    parser.add_option('--plot-base', help='Base filename for plots, default %s' % plot_base_default)
    parser.add_option('--plot-number', type=int, default=0, help='Set PlotSequence starting number')

    parser.add_option('-W', type=int, default=3600, help='Target image width (default %default)')
    parser.add_option('-H', type=int, default=3600, help='Target image height (default %default)')

    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 3600 0 3600")')

    opt,args = parser.parse_args()

    Time.add_measurement(MemMeas)

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    set_globals()
    
    #stagefunc = CallGlobal('stage%i', globals())
    stagefunc = CallGlobal('stage_%s', globals())

    if len(opt.stage) == 0:
        opt.stage.append('writecat')
    opt.force.extend(opt.stage)

    if opt.plot_base is None:
        opt.plot_base = plot_base_default
    ps = PlotSequence(opt.plot_base % dict(brick=opt.brick))
    initargs = dict(ps=ps)

    kwargs = {}
    if opt.plot_number:
        ps.skipto(opt.plot_number)
        kwargs.update(ps=ps)
    if opt.threads and opt.threads > 1:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)
        
    opt.picklepat = opt.picklepat % dict(brick=opt.brick)

    prereqs = {'tims':None,
               'srcs':'tims',
               'fitblobs':'srcs',

               'fitplots': 'fitblobs',
               'writecat': 'fitblobs',

               'psfplots': 'tims',
               'initplots': 'tims',
               }

    initargs.update(W=opt.W, H=opt.H, brickid=opt.brick, target_extent=opt.zoom)

    for stage in opt.stage:
        runstage(stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
                 prereqs=prereqs, plots=opt.plots, plots2=opt.plots2,
                 initial_args=initargs, **kwargs)
    
