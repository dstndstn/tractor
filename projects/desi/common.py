if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt

import os
import tempfile

from distutils.version import StrictVersion

import numpy as np

import fitsio

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_erosion, binary_fill_holes

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import dimshow
from astrometry.util.util import Tan, Sip, anwcs_t
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.miscutils import polygons_intersect, estimate_mode, clip_polygon, clip_wcs
from astrometry.sdss.fields import read_photoobjs_in_wcs
from astrometry.sdss import DR9, band_index, AsTransWrapper
from astrometry.util.resample import resample_with_wcs,OverlapError

from tractor.basics import ConstantSky, NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.engine import Image
from tractor.utils import get_class_from_name
from tractor.psfex import PsfEx
from tractor.sdss import get_tractor_sources_dr9
from tractor.ellipses import *
from tractor.sfd import *

# search order: $TMPDIR, $TEMP, $TMP, then /tmp, /var/tmp, /usr/tmp
tempdir = tempfile.gettempdir()

# From: http://www.noao.edu/noao/staff/fvaldes/CPDocPrelim/PL201_3.html
# 1   -- detector bad pixel           InstCal
# 1   -- detector bad pixel/no data   Resampled
# 1   -- No data                      Stacked
# 2   -- saturated                    InstCal/Resampled
# 4   -- interpolated                 InstCal/Resampled
# 16  -- single exposure cosmic ray   InstCal/Resampled
# 64  -- bleed trail                  InstCal/Resampled
# 128 -- multi-exposure transient     InstCal/Resampled 
CP_DQ_BITS = dict(badpix=1, satur=2, interp=4, cr=16, bleed=64,
                  trans=128,
                  edge = 256,
                  edge2 = 512) # in z-band images?

class BrickDuck(object):
    pass

def get_version_header(program_name, decals_dir):
    from astrometry.util.run_command import run_command
    if program_name is None:
        import sys
        program_name = sys.argv[0]

    rtn,version,err = run_command('git describe')
    if rtn:
        raise RuntimeError('Failed to get version string (git describe):' + ver + err)
    version = version.strip()
    print 'Version:', version

    hdr = fitsio.FITSHDR()

    for s in [
        'Data product of the DECam Legacy Survey (DECaLS)',
        'Full documentation at http://legacysurvey.org',
        ]:
        hdr.add_record(dict(name='COMMENT', value=s, comment=s))
    hdr.add_record(dict(name='TRACTORV', value=version,
                        comment='Tractor git version'))
    hdr.add_record(dict(name='DECALSV', value=decals_dir,
                        comment='DECaLS version'))
    hdr.add_record(dict(name='DECALSDR', value='DR1',
                        comment='DECaLS release name'))
    hdr.add_record(dict(name='DECALSDT', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    hdr.add_record(dict(name='SURVEY', value='DECaLS',
                        comment='DECam Legacy Survey'))

    import socket
    hdr.add_record(dict(name='HOSTNAME', value=socket.gethostname(),
                        comment='Machine where runbrick.py was run'))
    hdr.add_record(dict(name='HOSTFQDN', value=socket.getfqdn(),
                        comment='Machine where runbrick.py was run'))
    hdr.add_record(dict(name='NERSC', value=os.environ.get('NERSC_HOST', 'none'),
                        comment='NERSC machine where runbrick.py was run'))
    return hdr


class MyFITSHDR(fitsio.FITSHDR):
    ''' This is copied straight from fitsio, simply removing "BUNIT"
    from the list of headers to remove.
    '''
    def clean(self):
        """
        Remove reserved keywords from the header.
        
        These are keywords that the fits writer must write in order
        to maintain consistency between header and data.
        """

        rmnames = ['SIMPLE','EXTEND','XTENSION','BITPIX','PCOUNT','GCOUNT',
                   'THEAP',
                   'EXTNAME',
                   #'BUNIT',
                   'BSCALE','BZERO','BLANK',
                   'ZQUANTIZ','ZDITHER0','ZIMAGE','ZCMPTYPE',
                   'ZSIMPLE','ZTENSION','ZPCOUNT','ZGCOUNT',
                   'ZBITPIX','ZEXTEND',
                   #'FZTILELN','FZALGOR',
                   'CHECKSUM','DATASUM']
        self.delete(rmnames)

        r = self._record_map.get('NAXIS',None)
        if r is not None:
            naxis = int(r['value'])
            self.delete('NAXIS')

            rmnames = ['NAXIS%d' % i for i in xrange(1,naxis+1)]
            self.delete(rmnames)

        r = self._record_map.get('ZNAXIS',None)
        self.delete('ZNAXIS')
        if r is not None:

            znaxis = int(r['value'])

            rmnames = ['ZNAXIS%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZTILE%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZNAME%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)
            rmnames = ['ZVAL%d' % i for i in xrange(1,znaxis+1)]
            self.delete(rmnames)

        
        r = self._record_map.get('TFIELDS',None)
        if r is not None:
            tfields = int(r['value'])
            self.delete('TFIELDS')

            if tfields > 0:

                nbase = ['TFORM','TTYPE','TDIM','TUNIT','TSCAL','TZERO',
                         'TNULL','TDISP','TDMIN','TDMAX','TDESC','TROTA',
                         'TRPIX','TRVAL','TDELT','TCUNI',
                         #'FZALG'
                        ]
                for i in xrange(1,tfields+1):
                    names=['%s%d' % (n,i) for n in nbase]
                    self.delete(names)

def bin_image(data, invvar, S):
    # rebin image data
    H,W = data.shape
    sH,sW = (H+S-1)/S, (W+S-1)/S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    newiv = np.zeros((sH,sW), dtype=invvar.dtype)
    for i in range(S):
        for j in range(S):
            iv = invvar[i::S, j::S]
            subh,subw = iv.shape
            newdata[:subh,:subw] += data[i::S, j::S] * iv
            newiv  [:subh,:subw] += iv
    newdata /= (newiv + (newiv == 0)*1.)
    newdata[newiv == 0] = 0.
    return newdata,newiv

def segment_and_group_sources(image, T, name=None, ps=None, plots=False):
    '''
    *image*: binary image that defines "blobs"
    *T*: source table; only ".itx" and ".ity" elements are used (x,y integer pix pos)
      - ".blob" field is added.

    *name*: for debugging only

    Returns: (blobs, blobsrcs, blobslices)

    *blobs*: image, values -1 = no blob, integer blob indices
    *blobsrcs*: list of np arrays of integers, elements in T within each blob
    *blobslices*: list of slice objects for blob bounding-boxes.
    
    '''

    emptyblob = 0

    image = binary_fill_holes(image)

    blobs,nblobs = label(image)
    print 'N detected blobs:', nblobs
    H,W = image.shape
    del image

    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]

    if plots:
        plt.clf()
        dimshow(blobs > 0, vmin=0, vmax=1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % (i+1), ha='center', va='bottom', color='r')
        plt.plot(T.itx, T.ity, 'rx')
        for i,t in enumerate(T):
            plt.text(t.itx, t.ity, 'src %i' % i, color='red', ha='left', va='center')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

    # Find sets of sources within blobs
    blobsrcs = []
    keepslices = []
    blobmap = {}
    dropslices = {}
    for blob in range(1, nblobs+1):
        Isrcs = np.flatnonzero(T.blob == blob)
        if len(Isrcs) == 0:
            #print 'Blob', blob, 'has no sources'
            blobmap[blob] = -1
            dropslices[blob] = blobslices[blob-1]
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
    noblobs = np.flatnonzero(np.logical_not(inblobs))
    del inblobs
    # Add new fake blobs!
    for ib,i in enumerate(noblobs):
        #S = 3
        S = 5
        bslc = (slice(np.clip(T.ity[i] - S, 0, H-1), np.clip(T.ity[i] + S+1, 0, H)),
                slice(np.clip(T.itx[i] - S, 0, W-1), np.clip(T.itx[i] + S+1, 0, W)))

        # Does this new blob overlap existing blob(s)?
        oblobs = np.unique(blobs[bslc])
        oblobs = oblobs[oblobs != emptyblob]

        ##print 'Adding new blob for source at', (T.itx[i], T.ity[i])
        #print 'This blob overlaps existing blobs:', oblobs
        if len(oblobs) > 1:
            print 'WARNING: not merging overlapping blobs like maybe we should'
        if len(oblobs):
            blob = oblobs[0]
            #print 'Adding source to existing blob', blob
            blobs[bslc][blobs[bslc] == emptyblob] = blob
            blobindex = blobmap[blob]
            #print 'blob index', blobindex
            if blobindex == -1:
                # the overlapping blob was going to be dropped -- restore it.
                blobindex = len(blobsrcs)
                blobmap[blob] = blobindex
                blobslices.append(dropslices[blob])
                blobsrcs.append(np.array([], np.int64))
            # Expand the existing blob slice to encompass this new source
            oldslc = blobslices[blobindex]
            #print 'Old slice:', oldslc
            #print 'New slice:', bslc
            sy,sx = oldslc
            oy0,oy1, ox0,ox1 = sy.start,sy.stop, sx.start,sx.stop
            sy,sx = bslc
            ny0,ny1, nx0,nx1 = sy.start,sy.stop, sx.start,sx.stop
            #print 'Old y', oy0,oy1, 'x', ox0,ox1
            #print 'New y', ny0,ny1, 'x', nx0,nx1
            newslc = slice(min(oy0,ny0), max(oy1,ny1)), slice(min(ox0,nx0), max(ox1,nx1))
            #print 'Updated slice:', newslc
            blobslices[blobindex] = newslc
            # Add this source to the list of source indices for the existing blob.
            blobsrcs[blobindex] = np.append(blobsrcs[blobindex], np.array([i]))

        else:
            # Set synthetic blob number
            blob = nblobs+1 + ib
            blobs[bslc][blobs[bslc] == emptyblob] = blob
            blobmap[blob] = len(blobsrcs)
            blobslices.append(bslc)
            blobsrcs.append(np.array([i]))
    #print 'Added', len(noblobs), 'new fake singleton blobs'

    # Remap the "blobs" image so that empty regions are = -1 and the blob values
    # correspond to their indices in the "blobsrcs" list.
    if len(blobmap):
        maxblob = max(blobmap.keys())
    else:
        maxblob = 0
    maxblob = max(maxblob, blobs.max())
    bm = np.zeros(maxblob + 1, int)
    for k,v in blobmap.items():
        bm[k] = v
    bm[0] = -1

    # DEBUG
    if plots:
        fitsio.write('blobs-before-%s.fits' % name, blobs, clobber=True)

    # Remap blob numbers
    blobs = bm[blobs]

    if plots:
        fitsio.write('blobs-after-%s.fits' % name, blobs, clobber=True)

    if plots:
        plt.clf()
        dimshow(blobs > -1, vmin=0, vmax=1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % (i+1), ha='center', va='bottom', color='r')
        plt.plot(T.itx, T.ity, 'rx')
        for i,t in enumerate(T):
            plt.text(t.itx, t.ity, 'src %i' % i, color='red', ha='left', va='center')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

    for j,Isrcs in enumerate(blobsrcs):
        for i in Isrcs:
            #assert(blobs[T.ity[i], T.itx[i]] == j)
            if (blobs[T.ity[i], T.itx[i]] != j):
                print '---------------------------!!!--------------------------'
                print 'Blob', j, 'sources', Isrcs
                print 'Source', i, 'coords x,y', T.itx[i], T.ity[i]
                print 'Expected blob value', j, 'but got', blobs[T.ity[i], T.itx[i]]

    T.blob = blobs[T.ity, T.itx]
    assert(len(blobsrcs) == len(blobslices))

    return blobs, blobsrcs, blobslices

def get_sdss_sources(bands, targetwcs, photoobjdir=None, local=True,
                     extracols=[], ellipse=None):
    # FIXME?
    margin = 0.

    if ellipse is None:
        ellipse = EllipseESoft.fromRAbPhi

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
            'devflux', 'expflux', 'extinction'] + extracols

    # If we have a window_flist file cut to primary objects, use that.
    # This file comes from svn+ssh://astrometry.net/svn/trunk/projects/wise-sdss-phot
    # cut-window-flist.py, and used resolve/2013-07-29 (pre-DR13) as input.
    wfn = 'window_flist-cut.fits'
    if not os.path.exists(wfn):
        # default to the usual window_flist.fits file.
        wfn = None

    objs = read_photoobjs_in_wcs(targetwcs, margin, sdss=sdss, cols=cols, wfn=wfn)
    if objs is None:
        print 'No photoObjs in wcs'
        return None,None
    print 'Got', len(objs), 'photoObjs'

    # It can be string-valued
    objs.objid = np.array([int(x) if len(x) else 0 for x in objs.objid])

    # Treat as pointsource...
    #sband = 'r'
    #bandnum = 'ugriz'.index(sband)
    #objs.treated_as_pointsource = treat_as_pointsource(objs, bandnum)

    print 'Bands', bands, '->', list(bands)

    srcs = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss, bands=list(bands),
        nanomaggies=True, fixedComposites=True, useObjcType=True,
        ellipse=ellipse)
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
                          cutonaper=True,
                          ps=None):
    '''
    detmaps: list of detmaps, same order as "bands"
    detivs :    ditto
    
    '''
    from astrometry.util.ttime import Time
    t0 = Time()
    H,W = detmaps[0].shape

    allzero = True
    for iband,band in enumerate(bands):
        if sed[iband] == 0:
            continue
        if np.all(detivs[iband] == 0):
            continue
        allzero = False
        break
    if allzero:
        print 'SED', sedname, 'has all zero weight'
        return None,None,None,None,None

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
    del sedmap

    peaks = (sedsn > nsigma)
    print 'SED sn:', Time()-t0
    t0 = Time()

    def saddle_level(Y):
        # Require a saddle of (the larger of) "saddle" sigma, or 10% of the peak height
        drop = max(saddle, Y * 0.2)
        return Y - drop

    lowest_saddle = nsigma - saddle

    # zero out the edges -- larger margin here?
    peaks[0 ,:] = 0
    peaks[:, 0] = 0
    peaks[-1,:] = 0
    peaks[:,-1] = 0

    # Label the N-sigma blobs at this point... we'll use this to build
    # "sedhot", which in turn is used to define the blobs that we will
    # optimize simultaneously.  This also determines which pixels go
    # into the fitting!
    dilate = 8
    hotblobs,nhot = label(binary_fill_holes(
            binary_dilation(peaks, iterations=dilate))) #, structure=np.ones((3,3)))))

    # find pixels that are larger than their 8 neighbors
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,2:  ])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,2:  ])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,2:  ])
    print 'Peaks:', Time()-t0
    t0 = Time()

    if ps is not None:
        crossa = dict(ms=10, mew=1.5)
        green = (0,1,0)

        def plot_boundary_map(X):
            bounds = binary_dilation(X) - X
            H,W = X.shape
            rgba = np.zeros((H,W,4), np.uint8)
            rgba[:,:,1] = bounds*255
            rgba[:,:,3] = bounds*255
            plt.imshow(rgba, interpolation='nearest', origin='lower')

        plt.clf()
        plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest', origin='lower',
                   cmap='hot')
        above = (sedsn > nsigma)
        plot_boundary_map(above)
        ax = plt.axis()
        y,x = np.nonzero(peaks)
        plt.plot(x, y, 'r+')
        plt.axis(ax)
        plt.title('SED %s: S/N & peaks' % sedname)
        ps.savefig()

        # plt.clf()
        # plt.imshow(sedsn, vmin=-2, vmax=10, interpolation='nearest', origin='lower',
        #            cmap='hot')
        # plot_boundary_map(sedsn > lowest_saddle)
        # plt.title('SED %s: S/N & lowest saddle point bounds' % sedname)
        # ps.savefig()

    # For each new source, compute the saddle value, segment at that
    # level, and drop the source if it is in the same blob as a
    # previously-detected source.  We dilate the blobs a bit too, to
    # catch slight differences in centroid vs SDSS sources.
    dilate = 2

    # For efficiency, segment at the minimum saddle level to compute
    # slices; the operations described above need only happen within
    # the slice.
    saddlemap = (sedsn > lowest_saddle)
    saddlemap = binary_dilation(saddlemap, iterations=dilate)
    allblobs,nblobs = label(saddlemap)
    allslices = find_objects(allblobs)
    ally0 = [sy.start for sy,sx in allslices]
    allx0 = [sx.start for sy,sx in allslices]

    # brightest peaks first
    py,px = np.nonzero(peaks)
    I = np.argsort(-sedsn[py,px])
    py = py[I]
    px = px[I]

    keep = np.zeros(len(px), bool)

    peakval = []
    aper = []
    apin = 10
    apout = 20
    
    # For each peak, determine whether it is isolated enough --
    # separated by a low enough saddle from other sources.  Need only
    # search within its "allblob", which is defined by the lowest
    # saddle.
    for i,(x,y) in enumerate(zip(px, py)):
        level = saddle_level(sedsn[y,x])
        ablob = allblobs[y,x]
        index = ablob - 1
        slc = allslices[index]

        saddlemap = (sedsn[slc] > level) * (allblobs[slc] == ablob)
        saddlemap = binary_fill_holes(saddlemap)
        saddlemap = binary_dilation(saddlemap, iterations=dilate)
        blobs,nblobs = label(saddlemap)
        x0,y0 = allx0[index], ally0[index]
        thisblob = blobs[y-y0, x-x0]

        # previously found sources:
        ox = np.append(xomit, px[:i][keep[:i]]) - x0
        oy = np.append(yomit, py[:i][keep[:i]]) - y0
        h,w = blobs.shape
        cut = False
        if len(ox):
            ox = ox.astype(int)
            oy = oy.astype(int)
            cut = any((ox >= 0) * (ox < w) * (oy >= 0) * (oy < h) *
                      blobs[np.clip(oy,0,h-1), np.clip(ox,0,w-1)] == thisblob)

        if False and (not cut) and ps is not None:
            plt.clf()
            plt.subplot(1,2,1)
            dimshow(sedsn, vmin=-2, vmax=10, cmap='hot')
            plot_boundary_map((sedsn > nsigma))
            ax = plt.axis()
            plt.plot(x, y, 'm+', ms=12, mew=2)
            plt.axis(ax)

            plt.subplot(1,2,2)
            y1,x1 = [s.stop for s in slc]
            ext = [x0,x1,y0,y1]
            dimshow(saddlemap, extent=ext)
            #plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'c-')
            #ax = plt.axis()
            #plt.plot(ox+x0, oy+y0, 'rx')
            plt.plot(xomit, yomit, 'rx', ms=8, mew=2)
            plt.plot(px[:i][keep[:i]], py[:i][keep[:i]], '+', color=green, ms=8, mew=2)
            plt.plot(x, y, 'mo', mec='m', mfc='none', ms=12, mew=2)
            plt.axis(ax)
            if cut:
                plt.suptitle('Cut')
            else:
                plt.suptitle('Keep')
            ps.savefig()

        if cut:
            # in same blob as previously found source
            continue

        # Measure in aperture...
        ap   =  sedsn[max(0, y-apout):min(H,y+apout+1),
                      max(0, x-apout):min(W,x+apout+1)]
        apiv = (sediv[max(0, y-apout):min(H,y+apout+1),
                      max(0, x-apout):min(W,x+apout+1)] > 0)
        aph,apw = ap.shape
        apx0, apy0 = max(0, x - apout), max(0, y - apout)
        R2 = ((np.arange(aph)+apy0 - y)[:,np.newaxis]**2 + 
              (np.arange(apw)+apx0 - x)[np.newaxis,:]**2)
        ap = ap[apiv * (R2 >= apin**2) * (R2 <= apout**2)]
        if len(ap):
            # 16th percentile ~ -1 sigma point.
            m = np.percentile(ap, 16.)
        else:
            # fake
            m = -1.
        if cutonaper:
            if sedsn[y,x] - m < nsigma:
                continue

        aper.append(m)
        peakval.append(sedsn[y,x])
        keep[i] = True

        if False and ps is not None:
            plt.clf()
            plt.subplot(1,2,1)
            dimshow(ap, vmin=-2, vmax=10, cmap='hot', extent=[apx0,apx0+apw,apy0,apy0+aph])
            plt.subplot(1,2,2)
            dimshow(ap * ((R2 >= apin**2) * (R2 <= apout**2)), vmin=-2, vmax=10, cmap='hot', extent=[apx0,apx0+apw,apy0,apy0+aph])
            plt.suptitle('peak %.1f vs ap %.1f' % (sedsn[y,x], m))
            ps.savefig()

    print 'New sources:', Time()-t0
    t0 = Time()

    if ps is not None:
        pxdrop = px[np.logical_not(keep)]
        pydrop = py[np.logical_not(keep)]

    py = py[keep]
    px = px[keep]

    # Which of the hotblobs yielded sources?  Those are the ones to keep.
    hbmap = np.zeros(nhot+1, bool)
    hbmap[hotblobs[py,px]] = True
    if len(xomit):
        hbmap[hotblobs[yomit,xomit]] = True
    # in case a source is (somehow) not in a hotblob?
    hbmap[0] = False
    hotblobs = hbmap[hotblobs]
    #print 'Hotblobs:', hotblobs.shape, hotblobs.dtype

    if ps is not None:
        plt.clf()
        dimshow(hotblobs, vmin=0, vmax=1, cmap='hot')
        ax = plt.axis()
        p1 = plt.plot(px, py, 'g+', ms=8, mew=2)
        p2 = plt.plot(pxdrop, pydrop, 'm+', ms=8, mew=2)
        p3 = plt.plot(xomit, yomit, 'r+', ms=8, mew=2)
        plt.axis(ax)
        plt.title('SED %s: hot blobs' % sedname)
        plt.figlegend((p3[0],p1[0],p2[0]), ('Existing', 'Keep', 'Drop'), 'upper left')
        ps.savefig()

    return hotblobs, px, py, aper, peakval

def get_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None):
    '''
    Given a list of images in the given bands, returns a scaled RGB
    image.

    *imgs*  a list of numpy arrays, all the same size, in nanomaggies
    *bands* a list of strings, eg, ['g','r','z']
    *mnmx*  = (min,max), values that will become black/white *after* scaling.
           Default is (-3,10)
    *arcsinh* use nonlinear scaling as in SDSS
    *scales*

    Returns a (H,W,3) numpy array with values between 0 and 1.
    '''
    bands = ''.join(bands)

    grzscales = dict(g = (2, 0.0066),
                      r = (1, 0.01),
                      z = (0, 0.025),
                      )

    if scales is None:
        if bands == 'grz':
            scales = grzscales
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
            scales = grzscales
        
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

def bricks_touching_wcs(targetwcs, decals=None, B=None, margin=20):
    # margin: How far outside the image to keep objects
    # FIXME -- should be adaptive to object size!

    from astrometry.libkd.spherematch import *
    if B is None:
        assert(decals is not None)
        B = decals.get_bricks_readonly()

    ra,dec = targetwcs.radec_center()
    radius = targetwcs.radius()
        
    # MAGIC 0.4 degree search radius =
    # DECam hypot(1024,2048)*0.27/3600 + Brick hypot(0.25, 0.25) ~= 0.35 + margin
    I,J,d = match_radec(B.ra, B.dec, ra, dec,
                        radius + np.hypot(0.25,0.25)/2. + 0.05)
    print len(I), 'bricks nearby'
    keep = []
    for i in I:
        b = B[i]
        brickwcs = wcs_for_brick(b)
        clip = clip_wcs(targetwcs, brickwcs)
        if len(clip) == 0:
            print 'No overlap with brick', b.brickname
            continue
        keep.append(i)
    return B[np.array(keep)]

        
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
    #print 'Target WCS radius:', trad
    #print 'CCD radius:', ccdrad
    r,d = targetwcs.radec_center()
    I, = np.nonzero(np.abs(T.dec - d) < rad)
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

    if len(bands) == 1:
        return [(bands[0], (1.,))]

    # single-band filters
    SEDs = []
    for i,band in enumerate(bands):
        sed = np.zeros(len(bands))
        sed[i] = 1.
        SEDs.append((band, sed))

    if len(bands) > 1:
        flat = dict(g=1., r=1., z=1.)
        SEDs.append(('Flat', [flat[b] for b in bands]))
        red = dict(g=2.5, r=1., z=0.4)
        SEDs.append(('Red', [red[b] for b in bands]))

    return SEDs

def run_sed_matched_filters(SEDs, bands, detmaps, detivs, omit_xy,
                            targetwcs, nsigma=5,
                            plots=False, ps=None, mp=None):
    from astrometry.util.ttime import Time

    if omit_xy is not None:
        xx,yy = omit_xy
        n0 = len(xx)
    else:
        xx,yy = [],[]
        n0 = 0

    H,W = detmaps[0].shape
    hot = np.zeros((H,W), bool)

    peaksn = []
    apsn = []

    for sedname,sed in SEDs:
        print 'SED', sedname
        if plots:
            pps = ps
        else:
            pps = None
        t0 = Time()
        sedhot,px,py,peakval,apval = sed_matched_detection(
            sedname, sed, detmaps, detivs, bands, xx, yy, nsigma=nsigma, ps=pps)
        print 'SED took', Time()-t0
        if sedhot is None:
            continue
        print len(px), 'new peaks'
        hot |= sedhot
        # With an empty xx, np.append turns it into a double!
        xx = np.append(xx, px).astype(int)
        yy = np.append(yy, py).astype(int)

        peaksn.extend(peakval)
        apsn.extend(apval)

    # New peaks:
    peakx = xx[n0:]
    peaky = yy[n0:]

    # Add sources for the new peaks we found
    pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
    print 'Adding', len(pr), 'new sources'
    # Also create FITS table for new sources
    Tnew = fits_table()
    Tnew.ra  = pr
    Tnew.dec = pd
    Tnew.tx = peakx
    Tnew.ty = peaky
    assert(len(peaksn) == len(Tnew))
    assert(len(apsn) == len(Tnew))
    Tnew.peaksn = np.array(peaksn)
    Tnew.apsn = np.array(apsn)

    Tnew.itx = np.clip(np.round(Tnew.tx), 0, W-1).astype(int)
    Tnew.ity = np.clip(np.round(Tnew.ty), 0, H-1).astype(int)
    newcat = []
    for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
        fluxes = dict([(band, detmap[Tnew.ity[i], Tnew.itx[i]])
                       for band,detmap in zip(bands,detmaps)])
        newcat.append(PointSource(RaDecPos(r,d),
                                  NanoMaggies(order=bands, **fluxes)))

    return Tnew, newcat, hot

class Decals(object):
    def __init__(self, decals_dir=None):
        if decals_dir is None:
            decals_dir = os.environ.get('DECALS_DIR')
            if decals_dir is None:
                print 'Warning: you should set the $DECALS_DIR environment variable.'
                print 'On NERSC, you can do:'
                print '  module use /project/projectdirs/cosmo/work/decam/versions/modules'
                print '  module load decals'
                print 'Using the current directory as DECALS_DIR, but this is likely to fail.'
                print
                decals_dir = os.getcwd()
                
        self.decals_dir = decals_dir

        self.ZP = None
        self.bricks = None

        # Create and cache a kd-tree for bricks_touching_radec_box ?
        self.cache_tree = False
        self.bricktree = None
        ### HACK! Hard-coded brick edge size, in degrees!
        self.bricksize = 0.25

    def get_calib_dir(self):
        return os.path.join(self.decals_dir, 'calib', 'decam')

    def get_image_dir(self):
        return os.path.join(self.decals_dir, 'images')

    def get_decals_dir(self):
        return self.decals_dir

    def get_se_dir(self):
        return os.path.join(self.decals_dir, 'calib', 'se-config')

    def get_bricks(self):
        return fits_table(os.path.join(self.decals_dir, 'decals-bricks.fits'))

    ### HACK...
    def get_bricks_readonly(self):
        if self.bricks is None:
            self.bricks = self.get_bricks()
            # Assert that bricks are the sizes we think they are.
            # ... except for the two poles, which are half-sized
            assert(np.all(np.abs((self.bricks.dec2 - self.bricks.dec1)[1:-1] -
                                 self.bricksize) < 1e-8))
        return self.bricks

    def get_brick(self, brickid):
        B = self.get_bricks_readonly()
        I, = np.nonzero(B.brickid == brickid)
        if len(I) == 0:
            return None
        return B[I[0]]

    def get_brick_by_name(self, brickname):
        B = self.get_bricks_readonly()
        I, = np.nonzero(np.array([n == brickname for n in B.brickname]))
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
                             (bricks.dec1[J] <= dechi) * (bricks.dec2[J] >= declo))[0]]
            return I

        if rahi < ralo:
            # Wrap-around
            print 'In Dec slice:', len(np.flatnonzero((bricks.dec1 <= dechi) *
                                                      (bricks.dec2 >= declo)))
            print 'Above RAlo=', ralo, ':', len(np.flatnonzero(bricks.ra2 >= ralo))
            print 'Below RAhi=', rahi, ':', len(np.flatnonzero(bricks.ra1 <= rahi))
            print 'In RA slice:', len(np.nonzero(np.logical_or(bricks.ra2 >= ralo,
                                                               bricks.ra1 <= rahi)))
                    
            I, = np.nonzero(np.logical_or(bricks.ra2 >= ralo, bricks.ra1 <= rahi) *
                            (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
            print 'In RA&Dec slice', len(I)
        else:
            I, = np.nonzero((bricks.ra1  <= rahi ) * (bricks.ra2  >= ralo) *
                            (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
        return I
    
    def get_ccds(self):
        fn = os.path.join(self.decals_dir, 'decals-ccds.fits')
        print 'Reading CCDs from', fn
        T = fits_table(fn)
        print 'Got', len(T), 'CCDs'
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

    def tims_touching_wcs(self, targetwcs, mp, mock_psf=False,
                          const2psf=True, bands=None):
        '''
        mp: multiprocessing object
        '''
        # Read images
        C = self.ccds_touching_wcs(targetwcs)
        # Sort by band
        if bands is not None:
            C.cut(np.hstack([np.nonzero(C.filter == band)[0]
                             for band in bands]))
        ims = []
        for t in C:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(self, t)
            ims.append(im)
        # Read images, clip to ROI
        W,H = targetwcs.get_width(), targetwcs.get_height()
        targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        args = [(im, targetrd, mock_psf, const2psf) for im in ims]
        tims = mp.map(read_one_tim, args)
        return tims
    
    def find_ccds(self, expnum=None, extname=None):
        T = self.get_ccds()
        if expnum is not None:
            T.cut(T.expnum == expnum)
        if extname is not None:
            T.cut(T.extname == extname)
        return T

    def photometric_ccds(self, CCD):
        '''
        Returns an index array for the members of the table "CCD" that are photometric.
        '''
        
        # Using Arjun's zeropoints and recipe: email 2015-02-25
        '''
        I would suggest using the following algorithm to avoid bad data (in IDL-ese):

        iznp = where(abs(zpz.zpt-zpz.ccdzpt) ge 0.05 $
          or (zpz.ccdnmatch ge 20 and zpz.ccdphrms gt 0.2) $
          or (zpz.zpt le z0-0.3 and zpz.zpt ge z0+1.1) $
          or zpz.ccdnmatch lt 20 $
          ; or strcompress(zpz.ccdname,/remove_all) eq 'S7' $
          ,nznp)

        where zpz is a structure containing all the z-band
        zeropoints. Similarly for the g- and r-band zero points.

        Here the nominal zero points for the three bands are roughly
        g0 = 26.61 -2.5*alog10(4.)
        r0 = 26.818 -2.5*alog10(4.)
        z0 = 26.484 -2.5*alog10(4.)

        Using all the listed above criteria (with the S7 selection
        commented out) results in red points on the attached plot. The
        red points are in pretty good agreement with the log notes
        about poor sky conditions. Setting CCDNMATCH < 20 also catches
        places where the overlap with PS1 was not good. You will also
        want to throw out the bad CCD (CCDNAME = S7)
        '''

        ZP = self._get_zeropoints_table()
        zp_rowmap = dict([((expnum,extname),i) for i,(expnum,extname) in enumerate(
            zip(ZP.expnum, ZP.ccdname))])
        I = np.array([zp_rowmap[(expnum,extname)] for expnum,extname in
                      zip(CCD.expnum, CCD.extname)])
        ZP = ZP[I]
        assert(len(ZP) == len(CCD))
        
        z0 = dict(g = 26.610 - 2.5*np.log10(4.),
                  r = 26.818 - 2.5*np.log10(4.),
                  z = 26.464 - 2.5*np.log10(4.),)
        z0 = np.array([z0[f[0]] for f in ZP.filter])

        good = np.ones(len(CCD), bool)
        n0 = sum(good)
        for name,crit in [
            ('zpt - ccdzpt', (np.abs(ZP.zpt - ZP.ccdzpt) >= 0.05)),
            ('ccdnmatch >= 20 and ccdphrms >= 0.2',
             ((ZP.ccdnmatch >= 20) * (ZP.ccdphrms >= 0.2))),
            ('zpt lower than nominal', (ZP.zpt < (z0 - 0.3))),
            ('zpt higher than nominal', (ZP.zpt > (z0 + 1.1))),
            ('ccdnmatch < 20', (ZP.ccdnmatch < 20)),
            ('S7', np.array([s == 'S7' for s in ZP.ccdname])),
            ]:
            good[crit] = False
            n = sum(good)
            print 'Flagged', n0-n, 'more non-photometric using criterion:', name
            n0 = n

        return np.flatnonzero(good)

    def _get_zeropoints_table(self):
        if self.ZP is not None:
            return self.ZP
        zpfn = os.path.join(self.decals_dir, 'calib', 'decam', 'photom', 'zeropoints.fits')
        #print 'Reading zeropoints:', zpfn
        self.ZP = fits_table(zpfn)

        if 'ccdname' in self.ZP.get_columns():
            # 'N4 ' -> 'N4'
            self.ZP.ccdname = np.array([s.strip() for s in self.ZP.ccdname])

        # it's a string in some versions...
        self.ZP.expnum = np.array([int(t) for t in self.ZP.expnum])

        return self.ZP

    def get_zeropoint_row_for(self, im):
        ZP = self._get_zeropoints_table()
        I, = np.nonzero(ZP.expnum == im.expnum)
        if len(I) > 1:
            I, = np.nonzero((ZP.expnum == im.expnum) *
                            (ZP.ccdname == im.extname))
        if len(I) == 0:
            return None
        assert(len(I) == 1)
        return ZP[I[0]]
            
    def get_zeropoint_for(self, im):
        zp = self.get_zeropoint_row_for(im)
        # No updated zeropoint -- use header MAGZERO from primary HDU.
        if zp is None:
            print 'WARNING: using header zeropoints for', im
            hdr = im.read_image_primary_header()
            # DES Year1 Stripe82 images:
            magzero = hdr['MAGZERO']
            return magzero

        magzp = zp.ccdzpt
        magzp += 2.5 * np.log10(zp.exptime)
        return magzp

    def get_astrometric_zeropoint_for(self, im):
        zp = self.get_zeropoint_row_for(im)
        if zp is None:
            print 'WARNING: no astrometric zeropoints found for', im
            return 0.,0.
        dra, ddec = zp.ccdraoff, zp.ccddecoff
        return dra / 3600., ddec / 3600.
        #dec = zp.ccddec
        #return dra / np.cos(np.deg2rad(dec)), ddec

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
    def __init__(self, decals, t):
        self.decals = decals

        imgfn, hdu, band, expnum, extname, calname, exptime = (
            t.cpimage.strip(), t.cpimage_hdu, t.filter.strip(), t.expnum,
            t.extname.strip(), t.calname.strip(), t.exptime)

        if os.path.exists(imgfn):
            self.imgfn = imgfn
        else:
            self.imgfn = os.path.join(self.decals.get_image_dir(), imgfn)
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        self.hdu   = hdu
        self.expnum = expnum
        self.extname = extname
        self.band  = band
        self.exptime = exptime

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

        self.calname = calname
        self.name = '%08i-%s' % (expnum, extname)

        calibdir = self.decals.get_calib_dir()
        self.pvwcsfn = os.path.join(calibdir, 'astrom-pv', calname + '.wcs.fits')
        self.sefn = os.path.join(calibdir, 'sextractor', calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', calname + '.fits')
        self.skyfn = os.path.join(calibdir, 'sky', calname + '.fits')

    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

    def get_tractor_image(self, slc=None, radecpoly=None,
                          mock_psf=False, const2psf=False,
                          nanomaggies=True, subsky=True, tiny=5):
        '''
        slc: y,x slices
        '''
        band = self.band
        imh,imw = self.get_image_shape()

        wcs = self.read_pv_wcs()
        x0,y0 = 0,0
        if slc is None and radecpoly is not None:
            imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
            ok,tx,ty = wcs.radec2pixelxy(radecpoly[:-1,0], radecpoly[:-1,1])
            tpoly = zip(tx,ty)
            #print 'Target RA,Dec polygon in CCD pixels:', tpoly
            #print ' --> as ints:', np.array(tpoly).astype(int)
            #print 'Image polygon:', imgpoly
            clip = clip_polygon(imgpoly, tpoly)
            clip = np.array(clip)
            #print 'Clipped:', clip
            if len(clip) == 0:
                return None
            x0,y0 = np.floor(clip.min(axis=0)).astype(int)
            x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
            slc = slice(y0,y1+1), slice(x0,x1+1)
            if y1 - y0 < tiny or x1 - x0 < tiny:
                print 'Skipping tiny subimage'
                return None
        if slc is not None:
            sy,sx = slc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop

            # Handle 'glowing' edges in DES r-band images
            # aww yeah
            if band == 'r' and (('DES' in self.imgfn) or ('COSMOS' in self.imgfn)):
                # Northern chips: drop 100 pix off the bottom
                if 'N' in self.extname:
                    if y0 < 100:
                        print 'Clipping bottom part of northern DES r-band chip'
                        y0 = 100
                else:
                    # Southern chips: drop 100 pix off the top
                    if y1 > (imh-100):
                        print 'Clipping top part of southern DES r-band chip'
                        y1 = imh-100
                if y0 >= y1:
                    return None
            slc = slice(y0,y1), slice(x0,x1)

        print 'Reading image slice:', slc
        img,imghdr = self.read_image(header=True, slice=slc)
        invvar = self.read_invvar(slice=slc)
        dq = self.read_dq(slice=slc)

        e = imghdr['EXTNAME']
        print 'EXTNAME from image header:', e
        print 'My EXTNAME:', self.extname
        assert(e.strip() == self.extname.strip())

        uq = np.unique(dq)
        bits = reduce(np.bitwise_or, uq)
        print 'DQ bits set: 0x%x' % bits

        invvar[dq != 0] = 0.

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

        magzp = self.decals.get_zeropoint_for(self)
        print 'magzp', magzp
        orig_zpscale = zpscale = NanoMaggies.zeropointToScale(magzp)

        sky = self.read_sky_model()
        midsky = sky.getConstant()

        if subsky:
            img -= midsky
            sky.subtract(midsky)

        if nanomaggies:
            # Scale images to Nanomaggies
            print 'zpscale', zpscale
            img /= zpscale
            invvar *= zpscale**2
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
            psf = NCircularGaussianPSF([psf_sigma], [1.0])
            print 'WARNING: using mock PSF:', psf
        elif const2psf:
            # 2-component constant MoG.
            print 'Reading PsfEx model from', self.psffn
            from tractor.basics import GaussianMixtureEllipsePSF
            psfex = PsfEx(self.psffn, imw, imh, ny=13, nx=7,
                          psfClass=GaussianMixtureEllipsePSF, K=2)
            # FIXME -- could instantiate in the center of the ROI...
            psfim = psfex.instantiateAt(imw/2, imh/2)
            # trim a little
            psfim = psfim[5:-5, 5:-5]
            psf = GaussianMixtureEllipsePSF.fromStamp(psfim, N=2)
        else:
            assert(False)

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=sky, name=self.name + ' ' + band)
        assert(np.all(np.isfinite(tim.getInvError())))
        tim.zr = [-3. * sig1, 10. * sig1]
        tim.zpscale = orig_zpscale
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
        tim.dq = dq
        tim.dq_bits = CP_DQ_BITS
        tim.saturation = imghdr['SATURATE']
        tim.satval = tim.saturation
        if subsky:
            tim.satval -= midsky
        if nanomaggies:
            tim.satval /= zpscale

        mn,mx = tim.zr
        subh,subw = tim.shape
        tim.subwcs = tim.sip_wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        return tim
    
    def makedirs(self):
        for dirnm in [os.path.dirname(fn) for fn in
                      [self.sefn, self.psffn, self.skyfn, self.pvwcsfn]]:
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
        print 'Reading image from', self.imgfn, 'hdu', self.hdu
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

    def read_dq(self, header=False, **kwargs):
        print 'Reading data quality from', self.dqfn, 'hdu', self.hdu
        dq,hdr = self._read_fits(self.dqfn, self.hdu, header=True, **kwargs)
        primhdr = fitsio.read_header(self.dqfn)
        plver = primhdr['PLVER'].strip()
        plver = plver.replace('V','')
        if StrictVersion(plver) >= StrictVersion('3.5.0'):
            # Integer codes, not bit masks.
            dqbits = np.zeros(dq.shape, np.int16)
            '''
            1 = bad
            2 = no value (for remapped and stacked data)
            3 = saturated
            4 = bleed mask
            5 = cosmic ray
            6 = low weight
            7 = diff detect (multi-exposure difference detection from median)
            8 = long streak (e.g. satellite trail)
            '''
            dqbits[dq == 1] |= CP_DQ_BITS['badpix']
            dqbits[dq == 2] |= CP_DQ_BITS['badpix']
            dqbits[dq == 3] |= CP_DQ_BITS['satur']
            dqbits[dq == 4] |= CP_DQ_BITS['bleed']
            dqbits[dq == 5] |= CP_DQ_BITS['cr']
            dqbits[dq == 6] |= CP_DQ_BITS['badpix']
            dqbits[dq == 7] |= CP_DQ_BITS['trans']
            dqbits[dq == 8] |= CP_DQ_BITS['trans']

        else:
            dq = dq.astype(np.int16)

        if header:
            return dq,hdr
        else:
            return dq

    def read_invvar(self, clip=True, **kwargs):
        print 'Reading inverse-variance from', self.wtfn, 'hdu', self.hdu
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        if clip:
            # Clamp near-zero (incl negative!) invvars to zero
            med = np.median(invvar[invvar > 0])
            thresh = 0.2 * med
            invvar[invvar < thresh] = 0
        return invvar

    def read_pv_wcs(self):
        print 'Reading WCS from', self.pvwcsfn
        wcs = Sip(self.pvwcsfn)
        dra,ddec = self.decals.get_astrometric_zeropoint_for(self)
        r,d = wcs.get_crval()
        print 'Applying astrometric zeropoint:', (dra,ddec)
        wcs.set_crval((r + dra, d + ddec))
        return wcs
    
    def read_sdss(self):
        S = fits_table(self.sdssfn)
        # ugh!
        if S.objc_type.min() > 128:
            S.objc_type -= 128
        return S

    def read_sky_model(self):
        print 'Reading sky model from', self.skyfn
        hdr = fitsio.read_header(self.skyfn)
        skyclass = hdr['SKY']
        clazz = get_class_from_name(skyclass)
        fromfits = getattr(clazz, 'fromFitsHeader')
        skyobj = fromfits(hdr, prefix='SKY_')
        return skyobj

    def run_calibs(self, ra, dec, pixscale, mock_psf, W=2048, H=4096,
                   pvastrom=True, psfex=True, sky=True,
                   se=False,
                   funpack=False, fcopy=False, use_mask=True,
                   force=False, just_check=False):
        '''
        pixscale: in arcsec/pixel

        just_check: if True, returns True if calibs need to be run.
        '''
        #print 'run_calibs:', str(self), 'near RA,Dec', ra,dec, 'with pixscale', pixscale, 'arcsec/pix'

        for fn in [self.pvwcsfn, self.sefn, self.psffn, self.skyfn]:
            print 'exists?', os.path.exists(fn), fn
        self.makedirs()

        if mock_psf:
            psfex = False

        if psfex and os.path.exists(self.psffn) and (not force):
            # Sometimes SourceExtractor gets interrupted or something and
            # writes out 0 detections.  Then PsfEx fails but in a way that
            # an output file is still written.  Try to detect & fix this
            # case.
            # Check the PsfEx output file for POLNAME1
            hdr = fitsio.read_header(self.psffn, ext=1)
            if hdr.get('POLNAME1', None) is None:
                print 'Did not find POLNAME1 in PsfEx header', self.psffn, '-- deleting'
                os.unlink(self.psffn)
            else:
                psfex = False
        if psfex:
            se = True
            
        if se and os.path.exists(self.sefn) and (not force):
            # Check SourceExtractor catalog for size = 0
            fn = self.sefn
            T = fits_table(fn, hdu=2)
            print 'Read', len(T), 'sources from SE catalog', fn
            if T is None or len(T) == 0:
                print 'SourceExtractor catalog', fn, 'has no sources -- deleting'
                try:
                    os.unlink(fn)
                except:
                    pass
            if os.path.exists(self.sefn):
                se = False
        if se:
            funpack = True

        if pvastrom and os.path.exists(self.pvwcsfn) and (not force):
            fn = self.pvwcsfn
            if os.path.exists(fn):
                try:
                    wcs = Sip(fn)
                except:
                    print 'Failed to read PV-SIP file', fn, '-- deleting'
                    os.unlink(fn)
            if os.path.exists(fn):
                pvastrom = False

        if sky and os.path.exists(self.skyfn) and (not force):
            fn = self.skyfn
            if os.path.exists(fn):
                try:
                    hdr = fitsio.read_header(fn)
                except:
                    print 'Failed to read sky file', fn, '-- deleting'
                    os.unlink(fn)
            if os.path.exists(fn):
                sky = False

        if just_check:
            return (se or psfex or sky or pvastrom)

        tmpimgfn = None
        tmpmaskfn = None

        # Unpacked image file
        funimgfn = self.imgfn
        funmaskfn = self.dqfn
        
        if funpack:
            # For FITS files that are not actually fpack'ed, funpack -E
            # fails.  Check whether actually fpacked.
            hdr = fitsio.read_header(self.imgfn, ext=self.hdu)
            if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
                print 'Image', self.imgfn, 'HDU', self.hdu, 'is not actually fpacked; not funpacking, just imcopying.'
                fcopy = True

            tmpimgfn  = create_temp(suffix='.fits')
            tmpmaskfn = create_temp(suffix='.fits')
    
            if fcopy:
                cmd = 'imcopy %s"+%i" %s' % (self.imgfn, self.hdu, tmpimgfn)
            else:
                cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpimgfn, self.imgfn)
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
            funimgfn = tmpimgfn
            
            if use_mask:
                if fcopy:
                    cmd = 'imcopy %s"+%i" %s' % (self.dqfn, self.hdu, tmpmaskfn)
                else:
                    cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpmaskfn, self.dqfn)
                print cmd
                if os.system(cmd):
                    #raise RuntimeError('Command failed: ' + cmd)
                    print 'Command failed: ' + cmd
                    M,hdr = fitsio.read(self.dqfn, ext=self.hdu, header=True)
                    print 'Read', M.dtype, M.shape
                    fitsio.write(tmpmaskfn, M, header=hdr, clobber=True)
                    print 'Wrote', tmpmaskfn, 'with fitsio'
                    
                funmaskfn = tmpmaskfn
    
        if se:
            # grab header values...
            primhdr = self.read_image_primary_header()
            hdr     = self.read_image_header()
    
            magzp  = primhdr['MAGZERO']
            fwhm = hdr['FWHM']
            seeing = pixscale * fwhm
            print 'FWHM', fwhm, 'pix'
            print 'pixscale', pixscale, 'arcsec/pix'
            print 'Seeing', seeing, 'arcsec'
    
        if se:
            maskstr = ''
            if use_mask:
                maskstr = '-FLAG_IMAGE ' + funmaskfn
            sedir = self.decals.get_se_dir()
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'DECaLS-v2.sex'),
                maskstr, '-SEEING_FWHM %f' % seeing,
                '-PARAMETERS_NAME', os.path.join(sedir, 'DECaLS-v2.param'),
                '-FILTER_NAME', os.path.join(sedir, 'gauss_5.0_9x9.conv'),
                '-STARNNW_NAME', os.path.join(sedir, 'default.nnw'),
                '-PIXEL_SCALE 0',
                # SE has a *bizarre* notion of "sigma"
                '-DETECT_THRESH 1.0',
                '-ANALYSIS_THRESH 1.0',
                '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', self.sefn,
                funimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)

        if pvastrom:
            # DECam images appear to have PV coefficients up to PVx_10,
            # which are up to cubic terms in xi,eta,r.  Overshoot what we
            # need in SIP terms.
            tmpwcsfn  = create_temp(suffix='.wcs')
            cmd = ('wcs-pv2sip -S -o 6 -e %i %s %s' %
                   (self.hdu, self.imgfn, tmpwcsfn))
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
            # Read the resulting WCS header and add version info cards to it.
            version_hdr = get_version_header(None, self.decals.get_decals_dir())
            wcshdr = fitsio.read_header(tmpwcsfn)
            os.unlink(tmpwcsfn)
            for r in wcshdr.records():
                version_hdr.add_record(r)
            fitsio.write(self.pvwcsfn, None, header=version_hdr, clobber=True)
            print 'Wrote', self.pvwcsfn

        if psfex:
            sedir = self.decals.get_se_dir()
            # If we wrote *.psf instead of *.fits in a previous run...
            oldfn = self.psffn.replace('.fits', '.psf')
            if os.path.exists(oldfn):
                print 'Moving', oldfn, 'to', self.psffn
                os.rename(oldfn, self.psffn)
            else:
                cmd = ('psfex -c %s -PSF_DIR %s %s' %
                       (os.path.join(sedir, 'DECaLS-v2.psfex'),
                        os.path.dirname(self.psffn), self.sefn))
                print cmd
                rtn = os.system(cmd)
                if rtn:
                    raise RuntimeError('Command failed: ' + cmd + ': return value: %i' % rtn)
    
        if sky:
            print 'Fitting sky for', self
            img = self.read_image()
            wt = self.read_invvar()
            img = img[wt > 0]
            try:
                skyval = estimate_mode(img, raiseOnWarn=True)
                skymeth = 'mode'
            except:
                skyval = np.median(img)
                skymeth = 'median'
            tsky = ConstantSky(skyval)
            tt = type(tsky)
            sky_type = '%s.%s' % (tt.__module__, tt.__name__)
            hdr = get_version_header(None, self.decals.get_decals_dir())
            hdr.add_record(dict(name='SKYMETH', value=skymeth,
                                comment='estimate_mode, or fallback to median?'))
            hdr.add_record(dict(name='SKY', value=sky_type, comment='Sky class'))
            tsky.toFitsHeader(hdr, prefix='SKY_')
            fits = fitsio.FITS(self.skyfn, 'rw', clobber=True)
            fits.write(None, header=hdr)

        if tmpimgfn is not None:
            os.unlink(tmpimgfn)
        if tmpmaskfn is not None:
            os.unlink(tmpmaskfn)

            
def run_calibs(X):
    im = X[0]
    kwargs = X[1]
    args = X[2:]
    print 'run_calibs for image', im
    return im.run_calibs(*args, **kwargs)


def read_one_tim((im, targetrd, mock_psf, const2psf)):
    print 'Reading expnum', im.expnum, 'name', im.extname, 'band', im.band, 'exptime', im.exptime
    tim = im.get_tractor_image(radecpoly=targetrd, mock_psf=mock_psf,
                               const2psf=const2psf)
    return tim

