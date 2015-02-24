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
from astrometry.util.run_command import *
from astrometry.sdss import DR9, band_index, AsTransWrapper

from tractor import *
from tractor.galaxy import *
from tractor.source_extractor import *

from common import *

## GLOBALS!  Oh my!
mp = None
nocache = True
#photoobjdir = 'photoObjs-new'
useCeres = True

def runbrick_global_init():
    if nocache:
        disable_galaxy_cache()
    from tractor.ceres import ceres_opt

# Turn on/off caching for all new Tractor instances.
def create_tractor(tims, srcs):
    import tractor
    t = tractor.Tractor(tims, srcs)
    if nocache:
        t.disable_cache()
    return t
### Woot!
Tractor = create_tractor

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
        except StopIteration:
            raise
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

def set_globals():
    global imchi
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        hspace=0.2, wspace=0.05)
    imchi = dict(cmap='RdBu', vmin=-5, vmax=5)

def _bounce_tim_get_resamp((tim, targetwcs)):
    return tim_get_resamp(tim, targetwcs)

def tims_compute_resamp(tims, targetwcs):
    R = _map(_bounce_tim_get_resamp, [(tim,targetwcs) for tim in tims])
    for tim,r in zip(tims, R):
        tim.resamp = r

def compute_coadds(tims, bands, W, H, targetwcs, get_cow=False, get_n2=False,
                   images=None):
    coimgs = []
    cons = []
    if get_n2:
        cons2 = []
    if get_cow:
        # moo
        cowimgs = []
        wimgs = []
    
    for ib,band in enumerate(bands):
        coimg = np.zeros((H,W), np.float32)
        coimg2 = np.zeros((H,W), np.float32)
        con   = np.zeros((H,W), np.uint8)
        con2  = np.zeros((H,W), np.uint8)
        if get_cow:
            cowimg = np.zeros((H,W), np.float32)
            wimg  = np.zeros((H,W), np.float32)
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            nn = (tim.getInvError()[Yi,Xi] > 0)
            if images is None:
                coimg [Yo,Xo] += tim.getImage()[Yi,Xi] * nn
                coimg2[Yo,Xo] += tim.getImage()[Yi,Xi]
            else:
                coimg [Yo,Xo] += images[itim][Yi,Xi] * nn
                coimg2[Yo,Xo] += images[itim][Yi,Xi]
            con   [Yo,Xo] += nn
            if get_cow:
                cowimg[Yo,Xo] += tim.getInvvar()[Yi,Xi] * tim.getImage()[Yi,Xi]
                wimg  [Yo,Xo] += tim.getInvvar()[Yi,Xi]
            con2  [Yo,Xo] += 1
        coimg /= np.maximum(con,1)
        coimg[con == 0] = coimg2[con == 0] / np.maximum(1, con2[con == 0])
        if get_cow:
            cowimg /= np.maximum(wimg, 1e-16)
            cowimg[wimg == 0] = coimg[wimg == 0]
            cowimgs.append(cowimg)
            wimgs.append(wimg)
        coimgs.append(coimg)
        cons.append(con)
        if get_n2:
            cons2.append(con2)

    rtn = [coimgs,cons]
    if get_cow:
        rtn.extend([cowimgs, wimgs])
    if get_n2:
        rtn.append(cons2)
    return rtn

def stage_tims(W=3600, H=3600, brickid=None, brickname=None, ps=None,
               plots=False,
               target_extent=None, pipe=False, program_name='runbrick.py',
               bands='grz',
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
    hdr.add_record(dict(name='DECALSRE', value='pre-EDR2',
                        comment='DECaLS release name'))
    hdr.add_record(dict(name='DECALSDT', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    hdr.add_record(dict(name='SURVEY', value='DECaLS',
                        comment='DECam Legacy Survey'))
    version_header = hdr

    if brickid is not None:
        brick = decals.get_brick(brickid)
    else:
        brick = decals.get_brick_by_name(brickname)
    print 'Chosen brick:'
    brick.about()
    brickid = brick.brickid
    brickname = brick.brickname
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

    T = decals.ccds_touching_wcs(targetwcs)
    # Sort by band
    II = []
    T.cut(np.hstack([np.flatnonzero(T.filter == band) for band in bands]))
    ims = []
    for t in T:
        print
        print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
        im = DecamImage(t)
        ims.append(im)

    print 'Finding images touching brick:', Time()-tlast
    tlast = Time()

    # Check that the zeropoints exist
    for im in ims:
        decals.get_zeropoint_for(im)

    # Run calibrations
    args = [(im, dict(), brick.ra, brick.dec, pixscale, mock_psf)
            for im in ims]
    _map(run_calibs, args)
    print 'Calibrations:', Time()-tlast
    tlast = Time()

    # Read images, clip to ROI
    ttim = Time()
    args = [(im, decals, targetrd, mock_psf) for im in ims]
    tims = _map(read_one_tim, args)

    # Cut the table of CCDs to match the 'tims' list
    print 'Tims:', tims
    print 'T:', len(T)
    T.about()
    I = np.flatnonzero(np.array([tim is not None for tim in tims]))
    print 'I:', I
    T.cut(I)
    ccds = T
    tims = [tim for tim in tims if tim is not None]
    assert(len(T) == len(tims))

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
            'bands', 'tims', 'ps', 'brickid', 'brickname',
            'target_extent', 'ccds']
    if not pipe:
        keys.extend(['coimgs', 'cons'])
    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
    return rtn

def stage_srcs(coimgs=None, cons=None,
               targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               pipe=False, brickname=None,
               mp=None,
               **kwargs):

    tlast = Time()
    # Read SDSS sources
    cat,T = get_sdss_sources(bands, targetwcs)
    print 'SDSS sources:', Time()-tlast
    tlast = Time()

    print 'Rendering detection maps...'
    tlast = Time()
    detmaps, detivs = detection_maps(tims, targetwcs, bands, mp)
    print 'Detmaps:', Time()-tlast
    tlast = Time()

    # SED-matched detections
    SEDs = sed_matched_filters(bands)
    Tnew,newcat,hot = run_sed_matched_filters(SEDs, bands, detmaps, detivs,
                                              (T.itx,T.ity), targetwcs, plots=plots,
                                              ps=ps)
    Nsdss = len(T)
    T = merge_tables([T,Tnew], columns='fillzero')
    cat.extend(newcat)
    # new peaks
    peakx = T.tx[Nsdss:]
    peaky = T.ty[Nsdss:]
    
    if pipe:
        del detmaps
        del detivs

    print 'Peaks:', Time()-tlast
    tlast = Time()

    if plots:
        crossa = dict(ms=10, mew=1.5)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ax = plt.axis()
        plt.plot(T.tx, T.ty, 'r+', **crossa)
        plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
        plt.axis(ax)
        plt.title('Catalog + SED-matched detections')
        ps.savefig()


    hot = (hot > 5)
    hot = binary_dilation(hot, structure=np.ones((3,3), bool), iterations=2)
    # Segment, and record which sources fall into each blob
    blobs,blobsrcs,blobslices = segment_and_group_sources(hot, T, name=brickname)

    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')
    
    keys = ['T', 
            'blobsrcs', 'blobslices', 'blobs',
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

def stage_fitblobs(T=None, 
                   blobsrcs=None, blobslices=None, blobs=None,
                   tractor=None, cat=None, targetrd=None, pixscale=None,
                   targetwcs=None,
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

    if plots:
        coimgs,cons = compute_coadds(tims, bands, W, H, targetwcs)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i, ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

        for i,Isrcs in enumerate(blobsrcs):
            for isrc in Isrcs:
                src = cat[isrc]
                ra,dec = src.getPosition().ra, src.getPosition().dec
                ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                plt.text(x, y, 'b%i/s%i' % (i,isrc), ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs + Sources')
        ps.savefig()

        plt.clf()
        dimshow(blobs)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i, ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

        plt.clf()
        dimshow(blobs != -1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i, ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()


    tfitall = Time()
    iter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                      orig_wcsxy0, cat, bands, plots, ps)
    # to allow debugpool to only queue tasks one at a time
    iter = iterwrapper(iter, len(blobsrcs))
    R = _map(_bounce_one_blob, iter)
    print 'Fitting sources took:', Time()-tfitall

    return dict(fitblobs_R=R, tims=tims, ps=ps)
    
def stage_fitblobs_finish(
        T=None, blobsrcs=None, blobslices=None, blobs=None,
        tractor=None, cat=None, targetrd=None, pixscale=None,
        targetwcs=None,
        W=None,H=None, brickid=None,
        bands=None, ps=None, tims=None,
        plots=False, plots2=False,
        fitblobs_R=None,
        **kwargs):

    print 'Logprob:', tractor.getLogProb()

    # one_blob can reduce the number and change the types of sources!
    # Reorder the sources here...
    R = fitblobs_R

    # print 'R:'
    # for rr in R:
    #     print
    #     for r in rr:
    #         print r
    #     print

    # Drop now-empty blobs.
    R = [r for r in R if len(r[0])]
    
    II       = np.hstack([r[0] for r in R])
    srcivs   = np.hstack([np.hstack(r[2]) for r in R])
    fracflux = np.vstack([r[3] for r in R])
    rchi2    = np.vstack([r[4] for r in R])
    dchisqs  = np.vstack(np.vstack([r[5] for r in R]))
    
    newcat = []
    for r in R:
        newcat.extend(r[1])
    T.cut(II)

    del R
    del fitblobs_R
    
    assert(len(T) == len(newcat))
    print 'Old catalog:', len(cat)
    print 'New catalog:', len(newcat)
    cat = Catalog(*newcat)
    tractor.catalog = cat
    assert(cat.numberOfParams() == len(srcivs))
    ns,nb = fracflux.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = rchi2.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = dchisqs.shape
    assert(ns == len(cat))
    assert(nb == 5) # none, ptsrc, dev, exp, comp

    T.fracflux = fracflux
    T.rchi2 = rchi2
    T.dchisq = dchisqs.astype(np.float32)
    # Set -0 to 0
    T.dchisq[T.dchisq == 0.] = 0.
    # Make dchisq relative to the first element ("none" model)
    T.dchisq = T.dchisq[:, 1:] - T.dchisq[:, 0][:,np.newaxis]
    
    invvars = srcivs

    print 'New catalog:'
    for src in cat:
        print '  ', src
    
    print 'Logprob:', tractor.getLogProb()
    print 'lnprior:', tractor.getLogPrior()
    print 'lnl:', tractor.getLogLikelihood()

    print 'image priors', tractor.images.getLogPrior()
    print 'catalog priors', tractor.catalog.getLogPrior()
    for src in cat:
        print '  prior', src.getLogPrior(), src
    
    rtn = dict(fitblobs_R = None)
    for k in ['tractor', 'cat', 'invvars', 'T']:
        rtn[k] = locals()[k]
    return rtn
                          
def _blob_iter(blobslices, blobsrcs, blobs,
               targetwcs, tims, orig_wcsxy0, cat, bands, plots, ps):
    for iblob, (bslc,Isrcs) in enumerate(zip(blobslices, blobsrcs)):
        assert(len(Isrcs) > 0)

        tblob = Time()
        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        print
        print 'Blob', iblob+1, 'of', len(blobslices), ':',
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



        # Here we assume the "blobs" array has been remapped...
        blobmask = (blobs[bslc] == iblob)
        #print 'Blob mask:', np.sum(blobmask), 'pixels'

        yield (iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh, blobmask, subtimargs,
               [cat[i] for i in Isrcs], bands, plots, ps)

def _bounce_one_blob(X):
    try:
        return _one_blob(X)
    except:
        import traceback
        print 'Exception in _one_blob:'
        print 'args:', X
        traceback.print_exc()
        raise

def _one_blob((iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh, blobmask, subtimargs,
               srcs, bands, plots, ps)):

    plots2 = False

    tlast = Time()
    alphas = [0.1, 0.3, 1.0]

    bigblob = (blobw * blobh) > 100*100

    subtarget = targetwcs.get_subimage(bx0, by0, blobw, blobh)

    subtims = []
    for (subimg, subie, twcs, subwcs, pcal,
         sky, psf, name, sx0, sx1, sy0, sy1, ox0, oy0,
         band,sig1,modelMinval) in subtimargs:

        # Mask out inverr for pixels that are not within the blob.
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
            try:
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(subtarget, subsubwcs, [], 2)
            except OverlapError:
                print 'No overlap'
                continue
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
        bslc = (slice(by0, by0+blobh), slice(bx0, bx0+blobw))
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

            # DEBUG
            DEBUG = False
            if DEBUG:
                params = []
                params.append((srctractor.getLogProb(), srctractor.getParams()))

            cpu0 = time.clock()
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                print 'dlnp:', dlnp, 'src', src

                if DEBUG:
                    params.append((srctractor.getLogProb(), srctractor.getParams()))

                if time.clock()-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    break

                if dlnp < 0.1:
                    break


            if DEBUG:
                thislnp0 = srctractor.getLogProb()
                p0 = np.array(srctractor.getParams())
                print 'logprob:', p0, '=', thislnp0
    
                print 'p0 type:', p0.dtype
                px = p0 + np.zeros_like(p0)
                srctractor.setParams(px)
                lnpx = srctractor.getLogProb()
                assert(lnpx == thislnp0)
                print 'logprob:', px, '=', lnpx
    
                scales = srctractor.getParameterScales()
                print 'Parameter scales:', scales
                print 'Parameters:'
                srctractor.printThawedParams()
    
                # getParameterScales better not have changed the params!!
                assert(np.all(p0 == np.array(srctractor.getParams())))
                assert(srctractor.getLogProb() == thislnp0)
    
                pfinal = srctractor.getParams()
                pnames = srctractor.getParamNames()
    
                plt.figure(3, figsize=(8,6))
    
                plt.clf()
                for i in range(len(scales)):
                    plt.plot([(p[i] - pfinal[i])*scales[i] for lnp,p in params],
                             [lnp for lnp,p in params], '-', label=pnames[i])
                plt.ylabel('lnp')
                plt.legend()
                plt.title('scaled')
                ps.savefig()
    
                for i in range(len(scales)):
                    plt.clf()
                    #plt.subplot(2,1,1)
                    plt.plot([p[i] for lnp,p in params], '-')
                    plt.xlabel('step')
                    plt.title(pnames[i])
                    ps.savefig()
    
                    plt.clf()
                    plt.plot([p[i] for lnp,p in params],
                             [lnp for lnp,p in params], 'b.-')
    
                    # We also want to know about d(lnp)/d(param)
                    # and d(lnp)/d(X)
                    step = 1.1
                    steps = 1.1 ** np.arange(-20, 21)
                    s2 = np.linspace(0, steps[0], 10)[1:-1]
                    steps = reduce(np.append, [-steps[::-1], -s2[::-1], 0, s2, steps])
                    print 'Steps:', steps
    
                    plt.plot(p0[i], thislnp0, 'bx', ms=20)
    
                    print 'Stepping in param', pnames[i], '...'
                    pp = p0.copy()
                    lnps,parms = [],[]
                    for s in steps:
                        parm = p0[i] + s / scales[i]
                        pp[i] = parm
                        srctractor.setParams(pp)
                        lnp = srctractor.getLogProb()
                        parms.append(parm)
                        lnps.append(lnp)
                        print 'logprob:', pp, '=', lnp
                        
                    plt.plot(parms, lnps, 'k.-')
                    j = np.argmin(np.abs(steps - 1.))
                    plt.plot(parms[j], lnps[j], 'ko')
    
                    print 'Stepping in X...'
                    lnps,parms = [],[]
                    for s in steps:
                        pp = p0 + s * X
                        srctractor.setParams(pp)
                        lnp = srctractor.getLogProb()
                        parms.append(pp[i])
                        lnps.append(lnp)
                        print 'logprob:', pp, '=', lnp
    
    
                    ##
                    s3 = s2[:2]
                    ministeps = reduce(np.append, [-s3[::-1], 0, s3])
                    print 'mini steps:', ministeps
                    for s in ministeps:
                        pp = p0 + s * X
                        srctractor.setParams(pp)
                        lnp = srctractor.getLogProb()
                        print 'logprob:', pp, '=', lnp
    
                    rows = len(ministeps)
                    cols = len(srctractor.images)
    
                    plt.figure(4, figsize=(8,6))
                    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.01,
                                        right=0.99, bottom=0.01, top=0.99)
                    plt.clf()
                    k = 1
                    mods = []
                    for s in ministeps:
                        pp = p0 + s * X
                        srctractor.setParams(pp)
                        print 'ministep', s
                        print 'log prior', srctractor.getLogPrior()
                        print 'log likelihood', srctractor.getLogLikelihood()
                        mods.append(srctractor.getModelImages())
                        chis = srctractor.getChiImages()
                        # for chi in chis:
                        #     plt.subplot(rows, cols, k)
                        #     k += 1
                        #     dimshow(chi, ticks=False, vmin=-10, vmax=10, cmap='jet')
                        print 'chisqs:', [(chi**2).sum() for chi in chis]
                        print 'sum:', sum([(chi**2).sum() for chi in chis])
    
                    mod0 = mods[len(ministeps)/2]
                    for modlist in mods:
                        for mi,mod in enumerate(modlist):
                            plt.subplot(rows, cols, k)
                            k += 1
                            m0 = mod0[mi]
                            rng = m0.max() - m0.min()
                            dimshow(mod - mod0[mi], vmin=-0.01*rng, vmax=0.01*rng,
                                    ticks=False, cmap='gray')
                    ps.savefig()
                    plt.figure(3)
                    
                    plt.plot(parms, lnps, 'r.-')
    
                    print 'Stepping in X by alphas...'
                    lnps = []
                    for cc,ss in [('m',0.1), ('m',0.3), ('r',1)]:
                        pp = p0 + ss*X
                        srctractor.setParams(pp)
                        lnp = srctractor.getLogProb()
                        print 'logprob:', pp, '=', lnp
    
                        plt.plot(p0[i] + ss * X[i], lnp, 'o', color=cc)
                        lnps.append(lnp)
    
                    px = p0[i] + X[i]
                    pmid = (px + p0[i]) / 2.
                    dp = np.abs((px - pmid) * 2.)
                    hi,lo = max(max(lnps), thislnp0), min(min(lnps), thislnp0)
                    lnpmid = (hi + lo) / 2.
                    dlnp = np.abs((hi - lo) * 2.)
    
                    plt.ylabel('lnp')
                    plt.title(pnames[i])
                    ps.savefig()
    
                    plt.axis([pmid - dp, pmid + dp, lnpmid-dlnp, lnpmid+dlnp])
                    ps.savefig()
    
                srctractor.setParams(p0)
                ### DEBUG
            


            if plots:
                spmods.append(srctractor.getModelImages())
                spnames.append('Fit')
                spallmods.append(subtr.getModelImages())
                spallnames.append('Fit (all)')

            if plots:
                plt.figure(1, figsize=(8,6))
                plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                                    hspace=0.1, wspace=0.05)
                #plt.figure(2, figsize=(3,3))
                #plt.subplots_adjust(left=0.005, right=0.995, top=0.995,bottom=0.005)
                #_plot_mods(subtims, spmods, spnames, bands, None, None, bslc, blobw, blobh, ps,
                #           chi_plots=plots2)
                plt.figure(2, figsize=(3,3.5))
                plt.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.005)
                plt.suptitle('Blob %i' % iblob)
                tempims = [tim.getImage() for tim in subtims]
                for tim,orig in zip(subtims, orig_timages):
                    tim.data = orig
                _plot_mods(subtims, spallmods, spallnames, bands, None, None, bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format='Blob %i, src %i: %%s' % (iblob, i))
                for tim,im in zip(subtims, tempims):
                    tim.data = im

            # Re-remove the final fit model for this source.
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
        _plot_mods(subtims, plotmods, plotmodnames, bands, None, None, bslc, blobw, blobh, ps)

    # FIXME -- for large blobs, fit strata of sources simultaneously?

    print 'Blob finished fitting:', Time()-tlast
    tlast = Time()


    # Next, model selections: point source vs dev/exp vs composite.

    # FIXME -- render initial models and find significant flux overlap
    # (product)??  (Could use the same logic above!)  This would give
    # families of sources to fit simultaneously.  (The
    # not-friends-of-friends version of blobs!)

    src_lnps = []


    # FIXME -- do we need to do the whole "compute & subtract
    # initial models" thing here?  Probably...

    # -Remember the original subtim images
    # -Compute initial models for each source (in each tim)
    # -Subtract initial models from images
    # -During fitting, for each source:
    #   -add back in the source's initial model (to each tim)
    #   -fit, with Catalog([src])
    #   -subtract final model (from each tim)
    # -Replace original subtim images
    
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
        src = subcat[i]
        print
        print 'Model selection for source', src

        # if plots:
        #     plotmods = []
        #     plotmodnames = []

        # Add this source's initial model back in.
        for tim,mods in zip(subtims, initial_models):
            mod = mods[i]
            if mod is not None:
                mod.addTo(tim.getImage())

        lnp0 = subtr.getLogProb()
        print 'lnp0:', lnp0


        subcat[i] = None
        #print 'Catalog:', [s for s in subcat]
        lnp_null = subtr.getLogProb()
        print 'Removing the source: dlnp', lnp_null - lnp0

        lnps = dict(ptsrc=None, dev=None, exp=None, comp=None,
                    none=lnp_null)

        if isinstance(src, PointSource):
            # logr, ee1, ee2
            shape = EllipseESoft(-1., 0., 0.)
            dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
            exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
            comp = None
            ptsrc = src.copy()
            lnps.update(ptsrc=lnp0)
            trymodels = [('dev', dev), ('exp', exp), ('comp', comp)]
            oldmodel = 'ptsrc'
            
        elif isinstance(src, DevGalaxy):
            dev = src.copy()
            exp = ExpGalaxy(src.getPosition(), src.getBrightness(), src.getShape()).copy()
            comp = None
            ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
            lnps.update(dev=lnp0)
            trymodels = [('ptsrc', ptsrc), ('exp', exp), ('comp', comp)]
            oldmodel = 'dev'

        elif isinstance(src, ExpGalaxy):
            exp = src.copy()
            dev = DevGalaxy(src.getPosition(), src.getBrightness(), src.getShape()).copy()
            comp = None
            ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
            lnps.update(exp=lnp0)
            trymodels = [('ptsrc', ptsrc), ('dev', dev), ('comp', comp)]
            oldmodel = 'exp'
            
        elif isinstance(src, FixedCompositeGalaxy):
            frac = src.fracDev.getValue()
            if frac > 0:
                shape = src.shapeDev
            else:
                shape = src.shapeExp
            dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
            if frac < 1:
                shape = src.shapeExp
            else:
                shape = src.shapeDev
            exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
            comp = src.copy()
            ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
            lnps.update(comp=lnp0)
            trymodels = [('ptsrc', ptsrc), ('dev', dev), ('exp', exp)]
            oldmodel = 'comp'

        for name,newsrc in trymodels:
            print 'Trying model:', name
            if name == 'comp' and newsrc is None:
                newsrc = comp = FixedCompositeGalaxy(src.getPosition(), src.getBrightness(),
                                                     0.5, exp.getShape(), dev.getShape()).copy()
            print 'New source:', newsrc
            subcat[i] = newsrc
            lnp = subtr.getLogProb()
            print 'Initial log-prob:', lnp
            print 'vs original src:', lnp - lnp0

            subcat.freezeAllBut(i)

            max_cpu_per_source = 60.

            cpu0 = time.clock()
            p0 = newsrc.getParams()
            for step in range(50):
                dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                print '  dlnp:', dlnp, 'new src', newsrc
                if time.clock()-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    break
                if dlnp < 0.1:
                    break

            print 'New source (after optimization):', newsrc
            lnp = subtr.getLogProb()
            print 'Optimized log-prob:', lnp
            print 'vs original src:', lnp - lnp0

            # Try Ceres...
            # newsrc.setParams(p0)
            # print 'Ceres opt...'
            # R = subtr._ceres_opt(max_iterations=50)
            # print 'Result:', R
            # print 'New source (after optimization):', newsrc
            # lnp = subtr.getLogProb()
            # print 'Optimized log-prob:', lnp
            # print 'vs original src:', lnp - lnp0
            
            lnps[name] = lnp

            # if plots:
            #     plotmods.append(subtr.getModelImages())
            #     plotmodnames.append('try ' + name)

        print 'Log-probs:', lnps

        # if plots:
        #    _plot_mods(subtims, plotmods, plotmodnames, bands, None, None, bslc, blobw, blobh, ps)

        
        nbands = len(bands)
        nparams = dict(none=0, ptsrc=2, exp=5, dev=5, comp=9)

        plnps = dict([(k, (lnps[k]-lnp0) - 0.5 * nparams[k])
                      for k in nparams.keys()])

        #print 'Relative penalized log-probs:'
        #for k in keys:
        #    print '  ', k, ':', plnps[k]

        if plots:
            plt.clf()
            rows,cols = 2, 5

            #mods = OrderedDict(none=None, ptsrc=ptsrc, dev=dev, exp=exp, comp=comp)
            mods = OrderedDict([('none',None), ('ptsrc',ptsrc), ('dev',dev),
                                ('exp',exp), ('comp',comp)])
            for imod,modname in enumerate(mods.keys()):
                subcat[i] = mods[modname]

                plt.subplot(rows, cols, imod+1)
                if modname != 'none':
                    modimgs = subtr.getModelImages()
                    comods,nil = compute_coadds(subtims, bands, blobw, blobh, subtarget,
                                                images=modimgs)
                    dimshow(get_rgb(comods, bands))
                    plt.title(modname)

                    chisqs = [((tim.getImage() - mod) * tim.getInvError())**2
                              for tim,mod in zip(subtims, modimgs)]
                else:

                    coimgs, cons = compute_coadds(subtims, bands, blobw, blobh, subtarget)
                    dimshow(get_rgb(coimgs, bands))
                    ax = plt.axis()
                    ok,x,y = subtarget.radec2pixelxy(src.getPosition().ra, src.getPosition().dec)
                    plt.plot(x-1, y-1, 'r+')
                    plt.axis(ax)
                    plt.title('Image')

                    chisqs = [((tim.getImage()) * tim.getInvError())**2
                              for tim in subtims]
                cochisqs,nil = compute_coadds(subtims, bands, blobw, blobh, subtarget,
                                             images=chisqs)
                cochisq = reduce(np.add, cochisqs)

                plt.subplot(rows, cols, imod+1+cols)
                dimshow(cochisq, vmin=0, vmax=25)
                plt.title('dlnp %.0f' % plnps[modname])

            plt.suptitle('Blob %i, source %i: model selection' % (iblob, i))
            ps.savefig()

        keepmod = 'none'
        keepsrc = None

        # We decide separately whether to include the source in the
        # catalog and what type to give it.

        # This is our "detection threshold": 5-sigma in
        # *penalized* units; ie, ~5.2-sigma for point sources
        dlnp = 0.5 * 5.**2
        # Take the best of ptsrc, dev, exp, comp
        diff = max([plnps[name] - plnps[keepmod]
                    for name in ['ptsrc', 'dev', 'exp', 'comp']])
        if diff > dlnp:
            # We're going to keep this source!
            # It starts out as a point source.
            # This has the weird outcome that a source can be accepted
            # into the catalog on the basis of its "deV" fit, but appear
            # as a point source because the deV fit is not *convincingly*
            # better than the ptsrc fit.
            keepsrc = ptsrc
            keepmod = 'ptsrc'

            # This is our "upgrade" threshold: how much better a galaxy
            # fit has to be versus ptsrc, and comp versus galaxy.
            dlnp = 0.5 * 3.**2
            
            expdiff = plnps['exp'] - plnps[keepmod]
            devdiff = plnps['dev'] - plnps[keepmod]
            if expdiff > dlnp or devdiff > dlnp:
                if expdiff > devdiff:
                    print 'Upgrading from ptsrc to exp: diff', expdiff
                    keepsrc = exp
                    keepmod = 'exp'
                else:
                    print 'Upgrading from ptsrc to dev: diff', devdiff
                    keepsrc = dev
                    keepmod = 'dev'

                diff = plnps['comp'] - plnps[keepmod]
                if diff > dlnp:
                    print 'Upgrading for dev/exp to composite: diff', diff
                    keepsrc = comp
                    keepmod = 'comp'

        # Actually, penalized delta chi-squareds!
        src_lnps.append([-2. * (plnps[k] - plnps[keepmod])
                         for k in ['none', 'ptsrc', 'dev', 'exp', 'comp']])
                    
        if keepmod != oldmodel:
            print 'Switching source!'
            print 'Old:', src
            print 'New:', keepsrc
        else:
            print 'Not switching source'
            print 'Old:', src

        subcat[i] = keepsrc

        src = keepsrc
        if src is not None:
            # Re-remove the final fit model for this source.
            for tim in subtims:
                mod = src.getModelPatch(tim)
                if mod is not None:
                    mod.addTo(tim.getImage(), scale=-1)

    for tim,img in zip(subtims, orig_timages):
        tim.data = img
    del orig_timages
    del initial_models


    srcs = subcat
    keepI = [i for i,s in zip(Isrcs, srcs) if s is not None]
    keepsrcs = [s for s in srcs if s is not None]
    keeplnps = [x for x,s in zip(src_lnps,srcs) if s is not None]
    Isrcs = keepI
    srcs = keepsrcs
    src_lnps = keeplnps
    subcat = Catalog(*srcs)
    subtr.catalog = subcat
    
    # Variances
    srcinvvars = [[] for src in srcs]
    subcat.thawAllRecursive()
    subcat.freezeAllParams()
    for isub in range(len(srcs)):
        print 'Variances for source', isub
        subcat.thawParam(isub)
        src = subcat[isub]
        print 'Source', src
        if src is None:
            subcat.freezeParam(isub)
            continue
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
    
    # rchi2 quality-of-fit metric
    rchi2_num    = np.zeros((len(srcs),len(bands)), np.float32)
    rchi2_den    = np.zeros((len(srcs),len(bands)), np.float32)
    # fracflux degree-of-blending metric
    fracflux_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracflux_den = np.zeros((len(srcs),len(bands)), np.float32)

    for iband,band in enumerate(bands):
        for tim in subtims:
            if tim.band != band:
                continue
            mod = np.zeros(tim.getModelShape(), subtr.modtype)
            srcmods = [None for src in srcs]
            counts = np.zeros(len(srcs))
            pcal = tim.getPhotoCal()
            
            for isrc,src in enumerate(srcs):
                patch = subtr.getModelPatch(tim, src, minsb=tim.modelMinval)
                if patch is None or patch.patch is None:
                    continue
                counts[isrc] = np.sum([np.abs(pcal.brightnessToCounts(b))
                                              for b in src.getBrightnesses()])
                if counts[isrc] == 0:
                    continue
                H,W = mod.shape
                patch.clipTo(W,H)
                srcmods[isrc] = patch
                patch.addTo(mod)
                
            for isrc,patch in enumerate(srcmods):
                if patch is None:
                    continue
                slc = patch.getSlice(mod)
                # (mod - patch) is flux from others
                # (mod - patch) / counts is normalized flux from others
                # patch/counts is unit profile
                fracflux_num[isrc,iband] += np.sum((mod[slc] - patch.patch) * np.abs(patch.patch)) / counts[isrc]**2
                fracflux_den[isrc,iband] += np.sum(np.abs(patch.patch)) / np.abs(counts[isrc])

            tim.getSky().addTo(mod)
            chisq = ((tim.getImage() - mod) * tim.getInvError())**2
            
            for isrc,patch in enumerate(srcmods):
                if patch is None:
                    continue
                slc = patch.getSlice(mod)
                #rchi2_num[isrc,iband] += np.sum(np.abs(chisq[slc] * patch.patch)) / np.abs(counts[isrc])
                #rchi2_den[isrc,iband] += np.sum(np.abs(patch.patch) / counts[isrc]
                # We compute numerator and denom separately to handle edge objects, where
                # sum(patch.patch) < counts.  Also, to normalize by the number of images.
                # (Being on the edge of an image is like being in half an image.)
                rchi2_num[isrc,iband] += np.sum(chisq[slc] * patch.patch) / counts[isrc]
                # If the source is not near an image edge, sum(patch.patch) == counts[isrc].
                rchi2_den[isrc,iband] += np.sum(patch.patch) / counts[isrc]

    fracflux = fracflux_num / fracflux_den
    rchi2    = rchi2_num    / rchi2_den
    
    return Isrcs, srcs, srcinvvars, fracflux, rchi2, src_lnps
    



def _plot_mods(tims, mods, titles, bands, coimgs, cons, bslc, blobw, blobh, ps,
               chi_plots=True, rgb_plots=False, main_plot=True,
               rgb_format='%s'):
    subims = [[] for m in mods]
    chis = dict([(b,[]) for b in bands])
    
    make_coimgs = (coimgs is None)
    if make_coimgs:
        print '_plot_mods: blob shape', (blobh, blobw)
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
            # we'll use 'sig1' of the last tim in the list below...
            mn,mx = -10.*tim.sig1, 30.*tim.sig1
            sig1 = tim.sig1
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
        resida = dict(vmin=-5.*sig1, vmax=5.*sig1, ticks=False)
        for subim,comod,cochi in zip(subims, comods, cochis):
            subim.append((coimg, coimgn, comod, ima, cochi, resida))

    # Plot per-band image, model, and chi coadds, and RGB images
    rgba = dict(ticks=False)
    rgbs = []
    rgbnames = []
    plt.figure(1)
    for i,subim in enumerate(subims):
        plt.clf()
        rows,cols = 3,5
        imgs = []
        themods = []
        resids = []
        for j,(img,imgn,mod,ima,chi,resida) in enumerate(subim):
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
                dimshow(resid, nancolor='r', **resida)
        rgb = get_rgb(imgs, bands)
        if i == 0:
            rgbs.append(rgb)
            rgbnames.append(rgb_format % 'Image')
        if main_plot:
            plt.subplot(rows,cols, 4)
            dimshow(rgb, **rgba)
        rgb = get_rgb(themods, bands)
        rgbs.append(rgb)
        rgbnames.append(rgb_format % titles[i])
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
        for rgb,tt in zip(rgbs, rgbnames):
            plt.clf()
            dimshow(rgb, **rgba)
            plt.title(tt)
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
            K, = np.nonzero((Yo >= 0) * (Yo < clH) * (Xo >= 0) * (Xo < clW))
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


def stage_coadds(bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 outdir=None, T=None, cat=None, **kwargs):

    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    if not os.path.exists(basedir):
        try:
            os.makedirs(basedir)
        except:
            pass
        
    fn = os.path.join(basedir, 'decals-%s-ccds.fits' % brickname)
    ccds.writeto(fn)
    print 'Wrote', fn
    
    t0 = Time()
    mods = _map(_get_mod, [(tim, cat) for tim in tims])
    print 'Getting model images:', Time()-t0

    W = targetwcs.get_width()
    H = targetwcs.get_height()

    # Look up number of images overlapping each source's position.
    assert(len(T) == len(cat))
    nobs = np.zeros((len(T), len(bands)), np.uint8)
    rr = np.array([s.getPosition().ra  for s in cat])
    dd = np.array([s.getPosition().dec for s in cat])
    ok,ix,iy = targetwcs.radec2pixelxy(rr, dd)
    ix = np.clip(np.round(ix - 1), 0, W-1).astype(int)
    iy = np.clip(np.round(iy - 1), 0, H-1).astype(int)

    coimgs = []
    comods = []
    
    for iband,band in enumerate(bands):

        cow    = np.zeros((H,W), np.float32)
        cowimg = np.zeros((H,W), np.float32)
        cowmod = np.zeros((H,W), np.float32)
        coimg  = np.zeros((H,W), np.float32)
        comod  = np.zeros((H,W), np.float32)
        cochi2 = np.zeros((H,W), np.float32)
        con     = np.zeros((H,W), np.uint8)
        congood = np.zeros((H,W), np.uint8)
        detiv   = np.zeros((H,W), np.float32)

        for itim, (tim,mod) in enumerate(zip(tims, mods)):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R

            # number of good exposures
            good = (tim.getInvError()[Yi,Xi] > 0)
            congood[Yo,Xo] += good

            iv = tim.getInvvar()[Yi,Xi]
            im = tim.getImage()[Yi,Xi]

            # invvar-weighted image & model
            cowimg[Yo,Xo] += iv * im
            cowmod[Yo,Xo] += iv * mod[Yi,Xi]
            cow   [Yo,Xo] += iv

            # chi-squared
            cochi2[Yo,Xo] += iv * (im - mod[Yi,Xi])**2
            
            # straight-up image & model
            coimg[Yo,Xo] += im
            comod[Yo,Xo] += mod[Yi,Xi]
            con  [Yo,Xo] += 1

            # point-source depth
            psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
            detsig1 = tim.sig1 / psfnorm
            detiv[Yo,Xo] += good * (1. / detsig1**2)


        nobs[:, iband] = con[iy,ix]
            
        cowimg /= np.maximum(cow, 1e-16)
        cowmod /= np.maximum(cow, 1e-16)

        cowimg[cow == 0] = (coimg[cow == 0] / np.maximum(1, con[cow == 0]))
        cowmod[cow == 0] = (comod[cow == 0] / np.maximum(1, con[cow == 0]))

        coimgs.append(cowimg)
        comods.append(cowmod)

        del coimg
        del comod

        # Plug the WCS header cards into these images
        # copy version_header before modifying...
        hdr = fitsio.FITSHDR()
        for r in version_header.records():
            hdr.add_record(r)
        hdr.add_record(dict(name='BRICKNAM', value=brickname,
                            comment='DECaLS brick name'))
        keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
                'INSTRUME']
        vals = set()
        for tim in tims:
            if tim.band != band:
                continue
            v = []
            for key in keys:
                v.append(tim.primhdr.get(key,''))
            vals.add(tuple(v))
        for i,v in enumerate(vals):
            for ik,key in enumerate(keys):
                if i == 0:
                    kk = key
                else:
                    kk = key[:7] + '%i'%i
                hdr.add_record(dict(name=kk, value=v[ik]))
            
        hdr.add_record(dict(name='FILTER', value=band))

        targetwcs.add_to_header(hdr)
        hdr.delete('IMAGEW')
        hdr.delete('IMAGEH')
        
        for name,img in [('image',  cowimg),
                         ('invvar', cow),
                         ('model',  cowmod),
                         ('chi2',   cochi2),
                         ('depth',  detiv),
                         ('nexp',   congood),
                         ]:

            hdr.add_record(dict(name='IMTYPE', value=name,
                                comment='DECaLS image type'))
            fn = os.path.join(basedir,
                              'decals-%s-%s-%s.fits' % (brickname, name, band))
            fitsio.write(fn, img, clobber=True, header=hdr)
            print 'Wrote', fn

    tmpfn = create_temp(suffix='.png')
    for name,ims in [('image',coimgs), ('model',comods)]:
        plt.imsave(tmpfn, get_rgb(ims, bands), origin='lower')
        cmd = ('pngtopnm %s | pnmtojpeg -quality 90 > %s' %
               (tmpfn, os.path.join(basedir, 'decals-%s-%s.jpg' %
                                    (brickname, name))))
        os.system(cmd)
        os.unlink(tmpfn)

    T.nobs = nobs
    return dict(T = T)
    

'''
Write catalog output
'''
def stage_writecat(
    version_header=None,
    T=None,
    cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False, tractor=None,
    brickname=None,
    brickid=None,
    invvars=None,
    catalogfn=None,
    outdir=None,
    **kwargs):

    print 'Source types:'
    for src in cat:
        print '  ', type(src)
    
    from desi_common import prepare_fits_catalog
    fs = None
    TT = T.copy()
    for k in ['itx','ity','index']:
        TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob',
                       'fracflux','rchi2','dchisq','nobs']:
            TT.rename(col, 'sdss_%s' % col)
    TT.tx = (TT.tx + 1.).astype(np.float32)
    TT.ty = (TT.ty + 1.).astype(np.float32)
    TT.blob = TT.blob.astype(np.int32)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.brickname = np.array([brickname] * len(TT))
    TT.objid   = np.arange(len(TT)).astype(np.int32)
    
    allbands = 'ugrizY'

    TT.decam_rchi2    = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracflux = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_nobs     = np.zeros((len(TT), len(allbands)), np.uint8)
    for iband,band in enumerate(bands):
        i = allbands.index(band)
        TT.decam_rchi2[:,i] = TT.rchi2[:,iband]
        TT.decam_fracflux[:,i] = TT.fracflux[:,iband]
        TT.decam_nobs[:,i] = TT.nobs[:,iband]

    TT.delete_column('rchi2')
    TT.delete_column('fracflux')
    TT.delete_column('nobs')

    cat.thawAllRecursive()
    hdr = version_header
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs,
                                  allbands=allbands)

    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = bx.astype(np.float32)
    T2.by = by.astype(np.float32)

    T2.ra_ivar  = T2.ra_ivar .astype(np.float32)
    T2.dec_ivar = T2.dec_ivar.astype(np.float32)
    
    # Unpack shape columns
    T2.shapeExp_r  = T2.shapeExp[:,0]
    T2.shapeExp_e1 = T2.shapeExp[:,1]
    T2.shapeExp_e2 = T2.shapeExp[:,2]
    T2.shapeDev_r  = T2.shapeDev[:,0]
    T2.shapeDev_e1 = T2.shapeDev[:,1]
    T2.shapeDev_e2 = T2.shapeDev[:,2]
    T2.shapeExp_r_ivar  = T2.shapeExp_ivar[:,0]
    T2.shapeExp_e1_ivar = T2.shapeExp_ivar[:,1]
    T2.shapeExp_e2_ivar = T2.shapeExp_ivar[:,2]
    T2.shapeDev_r_ivar  = T2.shapeDev_ivar[:,0]
    T2.shapeDev_e1_ivar = T2.shapeDev_ivar[:,1]
    T2.shapeDev_e2_ivar = T2.shapeDev_ivar[:,2]

    if catalogfn is not None:
        fn = catalogfn
    else:
        if outdir is None:
            outdir = '.'
        outdir = os.path.join(outdir, 'tractor', brickname[:3])
        fn = os.path.join(outdir, 'tractor-%s.fits' % brickname)
    dirnm = os.path.dirname(fn)
    if not os.path.exists(dirnm):
        try:
            os.makedirs(dirnm)
        except:
            pass
        
    T2.writeto(fn, header=hdr)
    print 'Wrote', fn

    print 'Reading SFD maps...'

    sfd = SFDMap()
    system = dict(u='SDSS', g='DES', r='DES', i='DES', z='DES', Y='DES')
    filts = ['%s %s' % (system[f], f) for f in allbands]
    T2.decam_extinction = sfd.extinction(filts, T2.ra, T2.dec).astype(np.float32)
    
    T2.writeto(fn, header=hdr)
    print 'Wrote', fn, 'with extinction'


def main():
    from astrometry.util.stages import *
    import optparse
    import logging
    from astrometry.util.multiproc import multiproc

    ep = '''
eg, to run a small field containing a cluster:
\n
python -u projects/desi/runbrick.py --plots --brick 371589 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle
\n
'''
    parser = optparse.OptionParser(epilog=ep)
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[],
                      help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_option('-F', '--force-all', dest='forceall', action='store_true',
                      help='Force all stages to run')
    parser.add_option('-s', '--stage', dest='stage', default=[], action='append',
                      help="Run up to the given stage(s)")
    parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')
    parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
                      help='Make more verbose')

    parser.add_option('-b', '--brick', help='Brick ID or name to run: default %default',
                      default='377306')

    parser.add_option('-d', '--outdir', help='Set output base directory')
    
    parser.add_option('--threads', type=int, help='Run multi-threaded')
    parser.add_option('-p', '--plots', dest='plots', action='store_true',
                      help='Per-blob plots?')
    parser.add_option('--plots2', action='store_true',
                      help='More plots?')

    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')

    plot_base_default = 'brick-%(brick)s'
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
    
    stagefunc = CallGlobalTime('stage_%s', globals())

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
        mp = multiproc(opt.threads)
    else:
        mp = multiproc()
    # ??
    kwargs.update(mp=mp)

    if opt.outdir:
        kwargs.update(outdir=opt.outdir)

    if opt.forceall:
        kwargs.update(forceall=True)
        
    opt.picklepat = opt.picklepat % dict(brick=opt.brick)

    prereqs = {
        'tims':None,
        'srcs':'tims',
        'fitblobs':'srcs',
        'fitblobs_finish':'fitblobs',
        'coadds': 'fitblobs_finish',
        'writecat': 'coadds',

        'fitplots': 'fitblobs_finish',
        'psfplots': 'tims',
        'initplots': 'tims',
        }

    initargs.update(W=opt.W, H=opt.H, target_extent=opt.zoom)
    try:
        brickid = int(opt.brick, 10)
        initargs.update(brickid=brickid)
    except:
        initargs.update(brickname = opt.brick)

    t0 = Time()

    for stage in opt.stage:
        runstage(stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
                 prereqs=prereqs, plots=opt.plots, plots2=opt.plots2,
                 initial_args=initargs, **kwargs)

    print 'All done:', Time()-t0


if __name__ == '__main__':
    main()
