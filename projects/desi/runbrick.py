# Cython
#import pyximport; pyximport.install(pyimport=True)

# python -u projects/desi/runbrick.py -b 2437p082 --zoom 2575 2675 400 500 -P "pickles/zoom2-%(brick)s-%%(stage)s.pickle" > log 2>&1 &

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
unwise_dir = 'unwise-coadds'

# RGB image args used in the tile viewer:
rgbkwargs = dict(mnmx=(-1,100.), arcsinh=1.)
rgbkwargs_resid = dict(mnmx=(-5,5))


def runbrick_global_init():
    if nocache:
        disable_galaxy_cache()
    if useCeres:
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

def try_makedirs(dirs):
    if not os.path.exists(dirs):
        # there can be a race
        try:
            os.makedirs(dirs)
        except:
            import traceback
            traceback.print_exc()
            pass

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
               bands='grz', pvwcs=False,
               mock_psf=False, **kwargs):
    t0 = tlast = Time()

    # early fail for mysterious "ImportError: c.so.6: cannot open shared object file: No such file or directory"
    from tractor.mix import c_gauss_2d_grid

    rtn,version,err = run_command('git describe')
    if rtn:
        raise RuntimeError('Failed to get version string (git describe):' + ver + err)
    version = version.strip()
    print 'Version:', version

    decals = Decals()

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
    if T is None:
        print 'No CCDs touching target WCS'
        sys.exit(0)

    print len(T), 'CCDs touching target WCS'

    # Sort by band
    T.cut(np.hstack([np.flatnonzero(T.filter == band) for band in bands]))

    print 'Cutting out non-photometric CCDs...'
    #I = decals.photometric_ccds(T)
    I = np.flatnonzero(T.dr1 == 1)
    print len(I), 'of', len(T), 'CCDs are photometric'
    T.cut(I)

    ims = []
    for t in T:
        print
        print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
        im = DecamImage(t)
        ims.append(im)

    print 'Finding images touching brick:', Time()-tlast
    tlast = Time()

    decalsv = decals.decals_dir
    hdr = fitsio.FITSHDR()

    for s in [
        'Data product of the DECam Legacy Survey (DECaLS)',
        'Full documentation at http://legacysurvey.org',
        ]:
        hdr.add_record(dict(name='COMMENT', value=s, comment=s))
    hdr.add_record(dict(name='TRACTORV', value=version,
                        comment='Tractor git version'))
    hdr.add_record(dict(name='DECALSV', value=decalsv,
                        comment='DECaLS version'))
    hdr.add_record(dict(name='DECALSDR', value='DR1',
                        comment='DECaLS release name'))
    hdr.add_record(dict(name='DECALSDT', value=datetime.datetime.now().isoformat(),
                        comment='%s run time' % program_name))
    hdr.add_record(dict(name='SURVEY', value='DECaLS',
                        comment='DECam Legacy Survey'))

    hdr.add_record(dict(name='BRICKNAM', value=brickname, comment='DECaLS brick RRRr[pm]DDd'))
    hdr.add_record(dict(name='BRICKID' , value=brickid,   comment='DECaLS brick id'))
    hdr.add_record(dict(name='RAMIN'   , value=brick.ra1, comment='Brick RA min'))
    hdr.add_record(dict(name='RAMAX'   , value=brick.ra2, comment='Brick RA max'))
    hdr.add_record(dict(name='DECMIN'  , value=brick.ra1, comment='Brick Dec min'))
    hdr.add_record(dict(name='DECMAX'  , value=brick.ra2, comment='Brick Dec max'))

    import socket
    hdr.add_record(dict(name='HOSTNAME', value=socket.gethostname(),
                        comment='Machine where runbrick.py was run'))
    hdr.add_record(dict(name='HOSTFQDN', value=socket.getfqdn(),
                        comment='Machine where runbrick.py was run'))
    hdr.add_record(dict(name='NERSC', value=os.environ.get('NERSC_HOST', 'none'),
                        comment='NERSC machine where runbrick.py was run'))

    version_header = hdr

    # Run calibrations
    kwa = dict()
    if pvwcs:
        kwa.update(pvastrom=True, astrom=False)

    args = [(im, kwa, brick.ra, brick.dec, pixscale, mock_psf)
            for im in ims]
    _map(run_calibs, args)
    print 'Calibrations:', Time()-tlast
    tlast = Time()

    # Read images, clip to ROI
    ttim = Time()
    args = [(im, decals, targetrd, mock_psf, pvwcs) for im in ims]
    tims = _map(read_one_tim, args)

    # Cut the table of CCDs to match the 'tims' list
    #print 'Tims:', tims
    #print 'T:', len(T)
    #T.about()
    I = np.flatnonzero(np.array([tim is not None for tim in tims]))
    #print 'I:', I
    T.cut(I)
    ccds = T
    tims = [tim for tim in tims if tim is not None]
    assert(len(T) == len(tims))

    print 'Read', len(T), 'images:', Time()-tlast
    tlast = Time()

    if len(tims) == 0:
        print 'No photometric CCDs overlap.  Quitting.'
        sys.exit(0)

    if not pipe:
        # save resampling params
        tims_compute_resamp(tims, targetwcs)
        print 'Computing resampling:', Time()-tlast
        tlast = Time()
        # Produce per-band coadds, for plots
        coimgs,cons = compute_coadds(tims, bands, W, H, targetwcs)
        print 'Coadds:', Time()-tlast
        tlast = Time()

    # Cut "bands" down to just the bands for which we have images.
    allbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in allbands]
    print 'Cut bands to', bands

    for band in 'grz':
        hasit = band in bands
        hdr.add_record(dict(name='BRICK_%s' % band, value=hasit,
                            comment='Does band %s touch this brick?' % band))
    hdr.add_record(dict(name='BRICKBND', value=''.join(bands),
                        comment='Bands touching this brick'))

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick',
            'target_extent', 'ccds', 'bands']
    if not pipe:
        keys.extend(['coimgs', 'cons'])
    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
    return rtn


def stage_image_coadds(targetwcs=None, bands=None, tims=None, outdir=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None,
                       **kwargs):
    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    try_makedirs(basedir)

    W = targetwcs.get_width()
    H = targetwcs.get_height()

    coimgs = []
    for iband,band in enumerate(bands):

        cowimg = np.zeros((H,W), np.float32)
        cow    = np.zeros((H,W), np.float32)
        cosatw = np.zeros((H,W), np.float32)
        cosatim= np.zeros((H,W), np.float32)

        sig1 = 1.
        tinyw = 1e-30
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            iv = tim.getInvvar()[Yi,Xi]
            im = tim.getImage ()[Yi,Xi]
            # invvar-weighted image
            cowimg[Yo,Xo] += iv * im
            cow   [Yo,Xo] += iv

            # Saturated (but not otherwise bad) pixels
            sat = ((tim.dq_bits['satur'] & tim.dq) > 0)
            bad = ((sum(tim.dq_bits[b] for b in ['badpix', 'cr', 'trans', 'edge', 'edge2']) & tim.dq) > 0)
            #print 'sat:', sat.sum(), 'pixels'
            #print 'bad:', bad.sum(), 'pixels'
            sat &= np.logical_not(bad)
            #print 'sat & ~bad:', sat.sum(), 'pixels'
            # Pixels near saturated pix
            sat = binary_dilation(sat, iterations=10)
            cosatw [Yo,Xo] += tinyw * sat[Yi,Xi]
            cosatim[Yo,Xo] += tinyw * sat[Yi,Xi] * im #tim.satval

            sig1 = tim.sig1
            if plots:
                plt.clf()
                plt.subplot(1,3,1)
                thisco = np.zeros((H,W), np.float32)
                thisco[Yo,Xo] = im
                dimshow(thisco, vmin=-2.*tim.sig1, vmax=5.*tim.sig1)
                plt.title('co: %s' % tim.name)

                plt.subplot(1,3,2)
                thisco = np.zeros((H,W), np.float32)
                thisco[Yo,Xo] = iv
                dimshow(thisco, vmin=0., vmax=(1.2/tim.sig1**2))
                plt.title('iv: %s' % tim.name)

                plt.subplot(1,3,3)
                thisco = np.zeros((H,W), np.float32)
                thisco[Yo,Xo] = tinyw * sat[Yi,Xi] * im
                dimshow(thisco, vmin=-2.*tim.sig1, vmax=5.*tim.sig1)
                plt.title('sat: %s' % tim.name)

                ps.savefig()

                if False:
                    thisco[Yo,Xo] = im
                    vals = tim.dq_bits.values()
                    vals.sort()
                    thisbit = np.zeros((H,W), bool)
                    for i,v in enumerate(vals):
                        #plt.subplot(3,3,i+1)
                        thisbit[:,:] = False
                        thisbit[Yo,Xo] = (tim.dq[Yi,Xi] & v) > 0
                        print 'bit', v, 'has', thisbit.sum(), 'pixels set'
                        if sum(thisbit) > 0:
                            plt.clf()
                            dimshow(thisbit * thisco, vmin=-5.*tim.sig1, vmax=5.*tim.sig1)
                            plt.title('bit 0x%x' % v)
                            ps.savefig()

        if plots:
            plt.clf()
            plt.subplot(1,2,1)
            dimshow(cowimg / np.maximum(cow, tinyw), vmin=-2.*sig1, vmax=5.*sig1)
            plt.title('cowimg')
            plt.subplot(1,2,2)
            dimshow(cow)
            plt.title('cow')
            ps.savefig()

            plt.clf()
            plt.subplot(1,2,1)
            dimshow(cosatim / np.maximum(cosatw, tinyw), vmin=-2.*sig1, vmax=5.*sig1)
            plt.title('cosatim')
            plt.subplot(1,2,2)
            dimshow(cosatw)
            plt.title('cosatw')
            ps.savefig()

        if plots:
            plt.clf()
            dimshow(cowimg, vmin=-2.*sig1, vmax=5.*sig1)
            plt.title('cowimg (pre-patched)')
            ps.savefig()

        copretty = (cowimg + cosatim) / np.maximum(tinyw, cow + cosatw)
        coimgs.append(copretty)
        del cosatw
        del cosatim

        cowimg /= np.maximum(cow, tinyw)

        if plots:
            plt.clf()
            dimshow(copretty, vmin=-2.*sig1, vmax=5.*sig1)
            plt.title('cowimg (patched)')
            ps.savefig()

        # Plug the WCS header cards into these images
        # copy version_header before modifying...
        hdr = MyFITSHDR()
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

        name,img = ('image',  cowimg)
        hdr.add_record(dict(name='IMTYPE', value=name,
                            comment='DECaLS image type'))
        hdr.add_record(dict(name='MAGZERO', value=22.5,
                            comment='Magnitude zeropoint'))
        hdr.add_record(dict(name='BUNIT', value='nanomaggie',
                            comment='AB mag = 22.5 - 2.5*log10(nanomaggie)'))

        fn = os.path.join(basedir, 'decals-%s-%s-%s.fits' % (brickname, name, band))
        fitsio.write(fn, img, clobber=True, header=hdr)
        print 'Wrote', fn

    tmpfn = create_temp(suffix='.png')
    for name,ims,rgbkw in [('image',coimgs,rgbkwargs)]:
        plt.imsave(tmpfn, get_rgb(ims, bands, **rgbkw), origin='lower')
        jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name))
        cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn)
        os.system(cmd)
        os.unlink(tmpfn)
        print 'Wrote', jpegfn

    return None

def stage_srcs(coimgs=None, cons=None,
               targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               pipe=False, brickname=None,
               mp=None, outdir=None, nsigma=5,
               **kwargs):

    tlast = Time()
    # Read SDSS sources
    cols = ['parent', 'tai', 'mjd', 'psf_fwhm', 'objc_flags2', 'flags2',
            'devflux_ivar', 'expflux_ivar', 'calib_status', 'raerr', 'decerr']
    cat,T = get_sdss_sources(bands, targetwcs, extracols=cols)

    if T is not None:
        # SDSS RAERR, DECERR are in arcsec.  Convert to deg.
        err = T.raerr / 3600.
        T.ra_ivar  = 1./err**2
        err = T.decerr / 3600.
        T.dec_ivar  = 1./err**2
        T.delete_column('raerr')
        T.delete_column('decerr')
        sdss_xy = T.itx, T.ity
    else:
        sdss_xy = None

    print 'SDSS sources:', Time()-tlast
    tlast = Time()

    print 'Rendering detection maps...'
    tlast = Time()
    detmaps, detivs = detection_maps(tims, targetwcs, bands, mp)
    print 'Detmaps:', Time()-tlast
    tlast = Time()

    # Median-smooth detection maps?
    #if False:
    for i,(detmap,detiv) in enumerate(zip(detmaps,detivs)):
        #from astrometry.util.util import median_smooth
        #smoo = np.zeros_like(detmap)
        #median_smooth(detmap, detiv>0, 100, smoo)
        from scipy.ndimage.filters import median_filter
        #tmed = Time()
        #smoo = median_filter(detmap, (50,50))
        #print 'Median filter 50:', Time()-tmed

        # Bin down before median-filtering, for speed.
        binning = 4
        binned,nil = bin_image(detmap, detiv, binning)
        tmed = Time()
        smoo = median_filter(binned, (50,50))
        print 'Median filter:', Time()-tmed

        if plots:
            sig1 = 1./np.sqrt(np.median(detiv[detiv > 0]))
            kwa = dict(vmin=-2.*sig1, vmax=10.*sig1)
            kwa2 = dict(vmin=-2.*sig1, vmax=50.*sig1)

            subbed = detmap.copy()
            S = binning
            for i in range(S):
                for j in range(S):
                    subbed[i::S, j::S] -= smoo

            plt.clf()
            plt.subplot(2,3,1)
            dimshow(detmap, **kwa)
            plt.subplot(2,3,2)
            dimshow(smoo, **kwa)
            plt.subplot(2,3,3)
            dimshow(subbed, **kwa)
            plt.subplot(2,3,4)
            dimshow(detmap, **kwa2)
            plt.subplot(2,3,5)
            dimshow(smoo, **kwa2)
            plt.subplot(2,3,6)
            dimshow(subbed, **kwa2)
            plt.suptitle('Median filter of detection map: %s band' % bands[i])
            ps.savefig()

        # Subtract binned median image.
        S = binning
        for i in range(S):
            for j in range(S):
                detmap[i::S, j::S] -= smoo

    # SED-matched detections
    print 'Running source detection at', nsigma, 'sigma'
    SEDs = sed_matched_filters(bands)
    Tnew,newcat,hot = run_sed_matched_filters(SEDs, bands, detmaps, detivs,
                                              sdss_xy, targetwcs, nsigma=nsigma,
                                              plots=plots, ps=ps, mp=mp)
                                              

    peaksn = Tnew.peaksn
    apsn = Tnew.apsn
    Tnew.delete_column('peaksn')
    Tnew.delete_column('apsn')

    if T is None:
        Nsdss = 0
        T = Tnew
        cat = Catalog(*newcat)
    else:
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
        if False and not plots:
            plt.figure(figsize=(18,18))
            plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                                hspace=0.2, wspace=0.05)
            if outdir is None:
                outdir = '.'
            outdir = os.path.join(outdir, 'metrics', brickname[:3])
            try_makedirs(outdir)
            fn = os.path.join(outdir, 'sources-%s' % brickname)
            ps = PlotSequence(fn)

        crossa = dict(ms=10, mew=1.5)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Catalog + SED-matched detections')
        ps.savefig()

        ax = plt.axis()
        p1 = plt.plot(T.tx, T.ty, 'r+', **crossa)
        p2 = plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
        plt.axis(ax)
        plt.title('Catalog + SED-matched detections')
        plt.figlegend((p1[0], p2[0]), ('SDSS', 'New'), 'upper left')
        ps.savefig()
        # for x,y in zip(peakx,peaky):
        #     plt.text(x+5, y, '%.1f' % (hot[np.clip(int(y),0,H-1),
        #                                  np.clip(int(x),0,W-1)]), color='r',
        #              ha='left', va='center')
        # ps.savefig()
 
    # Segment, and record which sources fall into each blob
    blobs,blobsrcs,blobslices = segment_and_group_sources(hot, T, name=brickname,
                                                          ps=ps, plots=plots)
    del hot

    for i,Isrcs in enumerate(blobsrcs):
        #print 'Isrcs dtype', Isrcs.dtype
        if not (Isrcs.dtype in [int, np.int64]):
            print 'Isrcs dtype', Isrcs.dtype
            print 'i:', i
            print 'Isrcs:', Isrcs
            print 'blobslice:', blobslices[i]

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
                   nblobs=None, blob0=None,
                   **kwargs):
    print 'Multiproc:', mp
    print 'Blob0:', blob0
    print 'Nblobs:', nblobs

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


    T.orig_ra  = T.ra.copy()
    T.orig_dec = T.dec.copy()

    tfitall = Time()

    if blob0 is not None or (nblobs is not None and nblobs < len(blobslices)):
        if blob0 is None:
            blob0 = 0
        print 'Unique blob values:', np.unique(blobs)
        if blob0 > 0:
            blobs[blobs < blob0] = -1
            blobs[blobs >= 0] -= blob0
            blobslices = blobslices[blob0:]
            blobsrcs = blobsrcs[blob0:]
        print 'Unique blob values:', np.unique(blobs)
        if nblobs is not None:
            blobs[blobs >= nblobs] = -1
            blobslices = blobslices[:nblobs]
            blobsrcs = blobsrcs[:nblobs]
        iter = _blob_iter(blobslices, blobsrcs, blobs,
                          targetwcs, tims, orig_wcsxy0, cat, bands, plots, ps)
        iter = iterwrapper(iter, len(blobslices))
    else:
        iter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                          orig_wcsxy0, cat, bands, plots, ps)
        # to allow debugpool to only queue tasks one at a time
        iter = iterwrapper(iter, len(blobsrcs))
    R = _map(_bounce_one_blob, iter)
    print 'Fitting sources took:', Time()-tfitall

    return dict(fitblobs_R=R, tims=tims, ps=ps, blobs=blobs, blobslices=blobslices,
                blobsrcs=blobsrcs)
    
def stage_fitblobs_finish(
    brickname=None,
        T=None, blobsrcs=None, blobslices=None, blobs=None,
        tractor=None, cat=None, targetrd=None, pixscale=None,
        targetwcs=None,
        W=None,H=None, brickid=None,
        bands=None, ps=None, tims=None,
        plots=False, plots2=False,
        fitblobs_R=None,
        outdir=None,
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

    assert(len(R) == len(blobsrcs))
    
    # DEBUGging / metrics for us
    all_models = [r[8] for r in R]
    performance = [r[9] for r in R]

    allmods = [None]*len(T)
    allperfs = [None]*len(T)
    for Isrcs,mods,perf in zip(blobsrcs,all_models,performance):
        for i,mod,per in zip(Isrcs,mods,perf):
            allmods[i] = mod
            allperfs[i] = per
    del all_models
    del performance

    from desi_common import prepare_fits_catalog
    from astrometry.util.file import pickle_to_file
    
    hdr = fitsio.FITSHDR()
    TT = T.copy()
    for srctype in ['ptsrc', 'dev','exp','comp']:
        xcat = Catalog(*[m[srctype] for m in allmods])
        xcat.thawAllRecursive()
        allbands = 'ugrizY'
        TT,hdr = prepare_fits_catalog(xcat, None, TT, hdr, bands, None,
                                      allbands=allbands, prefix=srctype+'_',
                                      save_invvars=False)
        TT.set('%s_flags' % srctype, np.array([m['flags'][srctype] for m in allmods]))
    TT.delete_column('ptsrc_shapeExp')
    TT.delete_column('ptsrc_shapeDev')
    TT.delete_column('ptsrc_fracDev')
    TT.delete_column('ptsrc_type')
    TT.delete_column('dev_shapeExp')
    TT.delete_column('dev_fracDev')
    TT.delete_column('dev_type')
    TT.delete_column('exp_shapeDev')
    TT.delete_column('exp_fracDev')
    TT.delete_column('exp_type')
    TT.delete_column('comp_type')

    TT.keepmod = np.array([m['keep'] for m in allmods])
    TT.dchisq = np.array([m['dchisqs'] for m in allmods])
    if outdir is None:
        outdir = '.'
    outdir = os.path.join(outdir, 'metrics', brickname[:3])
    try_makedirs(outdir)
    fn = os.path.join(outdir, 'all-models-%s.fits' % brickname)
    TT.writeto(fn, header=hdr)
    del TT
    print 'Wrote', fn

    fn = os.path.join(outdir, 'performance-%s.pickle' % brickname)
    pickle_to_file(allperfs, fn)
    print 'Wrote', fn

    # Drop now-empty blobs.
    R = [r for r in R if len(r[0])]
    
    II       = np.hstack([r[0] for r in R])
    srcivs   = np.hstack([np.hstack(r[2]) for r in R])
    fracflux = np.vstack([r[3] for r in R])
    rchi2    = np.vstack([r[4] for r in R])
    dchisqs  = np.vstack(np.vstack([r[5] for r in R]))
    fracmasked = np.hstack([r[6] for r in R])
    flags = np.hstack([r[7] for r in R])
    fracin = np.vstack([r[10] for r in R])
    
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
    ns,nb = fracin.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = rchi2.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = dchisqs.shape
    assert(ns == len(cat))
    assert(nb == 5) # none, ptsrc, dev, exp, comp
    assert(len(flags) == len(cat))

    T.decam_flags = flags
    T.fracflux = fracflux
    T.fracin = fracin
    T.fracmasked = fracmasked
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

        print 'Blob', iblob+1, 'of', len(blobslices), ':',
        print len(Isrcs), 'sources, size', blobw, 'x', blobh, 'center', (bx0+bx1)/2, (by0+by1)/2

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

        yield (iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh, blobmask, subtimargs,
               [cat[i] for i in Isrcs], bands, plots, ps)

def _bounce_one_blob(X):
    try:
        return _one_blob(X)
    except:
        import traceback
        print 'Exception in _one_blob:'
        #print 'args:', X
        traceback.print_exc()
        raise

def _debug_plots(srctractor, ps):
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

def _plot_derivs(subtims, newsrc, srctractor, ps):
    plt.clf()
    rows = len(subtims)
    cols = 1 + newsrc.numberOfParams()
    for it,tim in enumerate(subtims):
        derivs = srctractor._getSourceDerivatives(newsrc, tim)
        c0 = 1 + cols*it
        mod = srctractor.getModelPatchNoCache(tim, src)
        if mod is not None and mod.patch is not None:
            plt.subplot(rows, cols, c0)
            dimshow(mod.patch, extent=mod.getExtent())
        c0 += 1
        for ip,deriv in enumerate(derivs):
            if deriv is None:
                continue
            plt.subplot(rows, cols, c0+ip)
            mx = np.max(np.abs(deriv.patch))
            dimshow(deriv.patch, extent=deriv.getExtent(), vmin=-mx, vmax=mx)
    plt.title('Derivatives for ' + name)
    ps.savefig()
    plt.clf()
    modimgs = srctractor.getModelImages()
    comods,nil = compute_coadds(subtims, bands, blobw, blobh, subtarget,
                                images=modimgs)
    dimshow(get_rgb(comods, bands))
    plt.title('Initial ' + name)
    ps.savefig()
            

def _clip_model_to_blob(mod, sh, ie):
    '''
    mod: Patch
    sh: tim shape
    ie: tim invError
    Returns: new Patch
    '''
    mslc,islc = mod.getSlices(sh)
    sy,sx = mslc
    mod = Patch(mod.x0 + sx.start, mod.y0 + sy.start, mod.patch[mslc] * (ie[islc]>0))
    return mod

FLAG_CPU_A   = 1
FLAG_STEPS_A = 2
FLAG_CPU_B   = 4
FLAG_STEPS_B = 8
FLAG_TRIED_C = 0x10
FLAG_CPU_C   = 0x20

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
        # print 'Optimizing band', b, ':', btr
        # print Time()-tband
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
            
        # print 'Band', b, 'took', Time()-tband
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
        #tt = Time()
        for tim in subtims:
            mods = []
            sh = tim.shape
            ie = tim.getInvError()
            for src in subcat:
                mod = src.getModelPatch(tim)
                if mod is not None and mod.patch is not None:
                    if not np.all(np.isfinite(mod.patch)):
                        print 'Non-finite mod patch'
                        print 'source:', src
                        print 'tim:', tim
                        print 'PSF:', tim.getPsf()
                    assert(np.all(np.isfinite(mod.patch)))
                    mod = _clip_model_to_blob(mod, sh, ie)
                    mod.addTo(tim.getImage(), scale=-1)
                mods.append(mod)

            initial_models.append(mods)
        #print 'Subtracting initial models:', Time()-tt

        # For sources, in decreasing order of brightness
        for numi,i in enumerate(Ibright):
            #tsrc = Time()
            #print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
            src = subcat[i]

            # Add this source's initial model back in.
            for tim,mods in zip(subtims, initial_models):
                mod = mods[i]
                if mod is not None:
                    mod.addTo(tim.getImage())

            #if bigblob: # or True:
            if False:
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

            modelMasks = []
            for imods in initial_models:
                d = dict()
                modelMasks.append(d)
                mod = imods[i]
                if mod is not None:
                    d[src] = Patch(mod.x0, mod.y0, mod.patch != 0)

            srctractor = Tractor(srctims, [src])
            srctractor.freezeParams('images')
            srctractor.setModelMasks(modelMasks)

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
                #print 'dlnp:', dlnp, 'src', src

                if DEBUG:
                    params.append((srctractor.getLogProb(), srctractor.getParams()))

                if time.clock()-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    break

                if dlnp < 0.1:
                    break

            if DEBUG:
                _debug_plots(srctractor, ps)

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

                _plot_mods(srctractor.getImages(), spmods, spnames, bands, None, None, bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format='spmods Blob %i, src %i: %%s' % (iblob, i))
                _plot_mods(subtims, spallmods, spallnames, bands, None, None, bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format='spallmods Blob %i, src %i: %%s' % (iblob, i))

                for tim,orig in zip(subtims, orig_timages):
                    tim.data = orig
                _plot_mods(subtims, spallmods, spallnames, bands, None, None, bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format='Blob %i, src %i: %%s' % (iblob, i))
                for tim,im in zip(subtims, tempims):
                    tim.data = im

            # Re-remove the final fit model for this source (pull from cache)
            for tim in subtims:
                mod = srctractor.getModelPatch(tim, src)
                if mod is not None:
                    mod.addTo(tim.getImage(), scale=-1)

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            #print 'Fitting source took', Time()-tsrc
            #print src

        for tim,img in zip(subtims, orig_timages):
            tim.data = img
        del orig_timages
        del initial_models
        
    else:
        # Single source (though this is coded to handle multiple sources)
        # Fit sources one at a time, but don't subtract other models
        subcat.freezeAllParams()

        modelMasks = []
        for tim in subtims:
            d = dict()
            modelMasks.append(d)
            for src in subcat:
                mod = src.getModelPatch(tim)
                if mod is not None:
                    mod = _clip_model_to_blob(mod, tim.shape, tim.getInvError())
                    d[src] = Patch(mod.x0, mod.y0, mod.patch != 0)
        subtr.setModelMasks(modelMasks)
        enable_galaxy_cache()

        for numi,i in enumerate(Ibright):
            #tsrc = Time()
            #print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
            subcat.freezeAllBut(i)

            max_cpu_per_source = 60.
            cpu0 = time.clock()
            for step in range(50):
                dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                # print 'dlnp:', dlnp
                if time.clock()-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    break
                if dlnp < 0.1:
                    break
            #print 'Fitting source took', Time()-tsrc
            # print subcat[i]

        subtr.setModelMasks(None)
        disable_galaxy_cache()

    if plots:
        plotmods.append(subtr.getModelImages())
        plotmodnames.append('Per Source')

    if len(srcs) > 1 and len(srcs) <= 10:
        #tfit = Time()
        # Optimize all at once?
        subcat.thawAllParams()
        #print 'Optimizing:', subtr
        # subtr.printThawedParams()
        for step in range(20):
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                          alphas=alphas)
            # print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        #print 'Simultaneous fit took:', Time()-tfit

        if plots:
            plotmods.append(subtr.getModelImages())
            plotmodnames.append('All Sources')


    if plots:
        _plot_mods(subtims, plotmods, plotmodnames, bands, None, None, bslc, blobw, blobh, ps)

    # FIXME -- for large blobs, fit strata of sources simultaneously?

    #print 'Blob finished fitting:', Time()-tlast
    #tlast = Time()

    # Next, model selections: point source vs dev/exp vs composite.

    # FIXME -- render initial models and find significant flux overlap
    # (product)??  (Could use the same logic above!)  This would give
    # families of sources to fit simultaneously.  (The
    # not-friends-of-friends version of blobs!)

    delta_chisqs = []

    # We repeat the "compute & subtract initial models" logic from above.
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
    # tt = Time()
    for tim in subtims:
        mods = []
        sh = tim.shape
        ie = tim.getInvError()
        img = tim.getImage()
        for src in subcat:
            mod = src.getModelPatch(tim)
            if mod is not None and mod.patch is not None:
                if not np.all(np.isfinite(mod.patch)):
                    print 'Non-finite mod patch'
                    print 'source:', src
                    print 'tim:', tim
                    print 'PSF:', tim.getPsf()
                assert(np.all(np.isfinite(mod.patch)))

                # Blank out pixels that are outside the blob ROI.
                mod = _clip_model_to_blob(mod, sh, ie)
                mod.addTo(img, scale=-1)
            mods.append(mod)

        initial_models.append(mods)
    # print 'Subtracting initial models:', Time()-tt

    all_models = [{} for i in range(len(Isrcs))]
    performance = [[] for i in range(len(Isrcs))]
    flags = np.zeros(len(Isrcs), np.uint16)

    # For sources, in decreasing order of brightness
    for numi,i in enumerate(Ibright):
        
        src = subcat[i]
        #print 'Model selection for source %i of %i in blob' % (numi, len(Ibright))

        # if plots:
        #     plotmods = []
        #     plotmodnames = []

        # Add this source's initial model back in.
        for tim,mods in zip(subtims, initial_models):
            mod = mods[i]
            if mod is not None:
                mod.addTo(tim.getImage())

        modelMasks = []
        for imods in initial_models:
            d = dict()
            modelMasks.append(d)
            mod = imods[i]
            if mod is not None:
                d[src] = Patch(mod.x0, mod.y0, mod.patch != 0)

        srctractor = Tractor(subtims, [src])
        srctractor.freezeParams('images')
        srctractor.setModelMasks(modelMasks)
        enable_galaxy_cache()

        lnp0 = srctractor.getLogProb()
        # print 'lnp0:', lnp0

        srccat = srctractor.getCatalog()
        srccat[0] = None
        
        lnp_null = srctractor.getLogProb()
        # print 'Removing the source: dlnp', lnp_null - lnp0

        lnps = dict(ptsrc=None, dev=None, exp=None, comp=None,
                    none=lnp_null)

        if isinstance(src, PointSource):
            # logr, ee1, ee2
            shape = EllipseESoft(-1., 0., 0.)
            dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
            exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
            comp = None
            ptsrc = src.copy()
            trymodels = [('ptsrc', ptsrc), ('dev', dev), ('exp', exp), ('comp', comp)]
            oldmodel = 'ptsrc'
            
        elif isinstance(src, DevGalaxy):
            dev = src.copy()
            exp = ExpGalaxy(src.getPosition(), src.getBrightness(), src.getShape()).copy()
            comp = None
            ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
            trymodels = [('ptsrc', ptsrc), ('dev', dev), ('exp', exp), ('comp', comp)]
            oldmodel = 'dev'

        elif isinstance(src, ExpGalaxy):
            exp = src.copy()
            dev = DevGalaxy(src.getPosition(), src.getBrightness(), src.getShape()).copy()
            comp = None
            ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
            trymodels = [('ptsrc', ptsrc), ('dev', dev), ('exp', exp), ('comp', comp)]
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
            trymodels = [('ptsrc', ptsrc), ('dev', dev), ('exp', exp), ('comp', comp)]
            oldmodel = 'comp'

        allflags = {}
        for name,newsrc in trymodels:
            #print 'Trying model:', name
            if name == 'comp' and newsrc is None:
                newsrc = comp = FixedCompositeGalaxy(src.getPosition(), src.getBrightness(),
                                                     0.5, exp.getShape(), dev.getShape()).copy()
            #print 'New source:', newsrc
            srccat[0] = newsrc

            # Use the same initial modelMasks as the original source; we'll do a second
            # round below.  Need to create newsrc->mask mappings though:
            mm = []
            for mim in modelMasks:
                d = dict()
                mm.append(d)
                try:
                    d[newsrc] = mim[src]
                except KeyError:
                    pass
            srctractor.setModelMasks(mm)
            enable_galaxy_cache()

            #lnp = srctractor.getLogProb()
            #print 'Initial log-prob:', lnp
            #print 'vs original src: ', lnp - lnp0

            if plots and False:
                # Grid of derivatives.
                _plot_derivs(subtims, newsrc, srctractor, ps)

            max_cpu_per_source = 60.

            thisflags = 0

            cpu0 = time.clock()
            p0 = newsrc.getParams()
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                #print '  dlnp:', dlnp, 'new src', newsrc
                cpu = time.clock()
                performance[i].append((name,'A',step,dlnp,alpha,cpu-cpu0))
                if cpu-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    thisflags |= FLAG_CPU_A
                    break
                if dlnp < 0.1:
                    break
            else:
                thisflags |= FLAG_STEPS_A

            # print 'New source (after first round optimization):', newsrc
            # lnp = srctractor.getLogProb()
            # print 'Optimized log-prob:', lnp

            if plots and False:
                plt.clf()
                modimgs = srctractor.getModelImages()
                comods,nil = compute_coadds(subtims, bands, blobw, blobh, subtarget,
                                            images=modimgs)
                dimshow(get_rgb(comods, bands))
                plt.title('First-round opt ' + name)
                ps.savefig()

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            # Recompute modelMasks
            mm = []
            for tim in subtims:
                d = dict()
                mm.append(d)
                mod = src.getModelPatch(tim)
                if mod is None:
                    continue
                mod = _clip_model_to_blob(mod, tim.shape, tim.getInvError())
                d[newsrc] = Patch(mod.x0, mod.y0, mod.patch != 0)
            srctractor.setModelMasks(mm)
            enable_galaxy_cache()

            # Run another round of opt.
            cpu0 = time.clock()
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                #print '  dlnp:', dlnp, 'new src', newsrc
                cpu = time.clock()
                performance[i].append((name,'B',step,dlnp,alpha,cpu-cpu0))
                if cpu-cpu0 > max_cpu_per_source:
                    print 'Warning: Exceeded maximum CPU time for source'
                    thisflags |= FLAG_CPU_B
                    break
                if dlnp < 0.1:
                    break
            else:
                thisflags |= FLAG_STEPS_B

            # print 'New source (after optimization):', newsrc
            # print 'Optimized log-prob:', lnp
            # print 'vs original src:   ', lnp - lnp0

            if plots and False:
                plt.clf()
                modimgs = srctractor.getModelImages()
                comods,nil = compute_coadds(subtims, bands, blobw, blobh, subtarget,
                                            images=modimgs)
                dimshow(get_rgb(comods, bands))
                plt.title('Second-round opt ' + name)
                ps.savefig()

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            lnp = srctractor.getLogProb()
            lnps[name] = lnp
            all_models[i][name] = newsrc.copy()
            allflags[name] = thisflags
            
        # if plots:
        #    _plot_mods(subtims, plotmods, plotmodnames, bands, None, None, bslc, blobw, blobh, ps)
        
        nbands = len(bands)
        nparams = dict(none=0, ptsrc=2, exp=5, dev=5, comp=9)

        plnps = dict([(k, (lnps[k]-lnp0) - 0.5 * nparams[k])
                      for k in nparams.keys()])

        if plots:
            plt.clf()
            rows,cols = 2, 5
            mods = OrderedDict([('none',None), ('ptsrc',ptsrc), ('dev',dev),
                                ('exp',exp), ('comp',comp)])
            for imod,modname in enumerate(mods.keys()):
                srccat[0] = mods[modname]

                print 'Plotting model for blob', iblob, 'source', i, ':', modname
                print srccat[0]

                print 'cat:', srctractor.getCatalog()
                
                plt.subplot(rows, cols, imod+1)
                if modname != 'none':
                    modimgs = srctractor.getModelImages()
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
            plt.suptitle('Blob %i, source %i: was: %s' %
                         (iblob, i, str(src)))
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
                    #print 'Upgrading from ptsrc to exp: diff', expdiff
                    keepsrc = exp
                    keepmod = 'exp'
                else:
                    #print 'Upgrading from ptsrc to dev: diff', devdiff
                    keepsrc = dev
                    keepmod = 'dev'

                diff = plnps['comp'] - plnps[keepmod]
                if diff > dlnp:
                    #print 'Upgrading for dev/exp to composite: diff', diff
                    keepsrc = comp
                    keepmod = 'comp'

        # Actually, penalized delta chi-squareds!
        delta_chisqs.append([-2. * (plnps[k] - plnps[keepmod])
                         for k in ['none', 'ptsrc', 'dev', 'exp', 'comp']])
                    
        # if keepmod != oldmodel:
        #     print 'Switching source!'
        #     print 'Old:', src
        #     print 'New:', keepsrc
        # else:
        #     print 'Not switching source'
        #     print 'Old:', src

        subcat[i] = keepsrc
        flags[i] = allflags.get(keepmod, 0)
        all_models[i]['keep'] = keepmod
        all_models[i]['dchisqs'] = delta_chisqs[-1]
        all_models[i]['flags'] = allflags

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

    #print 'Blob finished model selection:', Time()-tlast
    #tlast = Time()

    if plots:
        plotmods, plotmodnames = [],[]
        plotmods.append(subtr.getModelImages())
        plotmodnames.append('All model selection')
        _plot_mods(subtims, plotmods, plotmodnames, bands, None, None, bslc, blobw, blobh, ps)

    srcs = subcat
    keepI = [i for i,s in zip(Isrcs, srcs) if s is not None]
    keepsrcs = [s for s in srcs if s is not None]
    keepdeltas = [x for x,s in zip(delta_chisqs,srcs) if s is not None]
    flags = np.array([f for f,s in zip(flags, srcs) if s is not None])
    Isrcs = keepI
    srcs = keepsrcs
    delta_chisqs = keepdeltas
    subcat = Catalog(*srcs)
    subtr.catalog = subcat

    ### Simultaneous re-opt.
    if False and len(subcat) > 1 and len(subcat) <= 10:
        #tfit = Time()
        # Optimize all at once?
        subcat.thawAllParams()
        #print 'Optimizing:', subtr
        #subtr.printThawedParams()

        flags |= FLAG_TRIED_C
        max_cpu = 300.
        cpu0 = time.clock()
        for step in range(50):
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                          alphas=alphas)
            #print 'dlnp:', dlnp
            cpu = time.clock()
            performance[0].append(('All','J',step,dlnp,alpha,cpu-cpu0))
            if cpu-cpu0 > max_cpu:
                print 'Warning: Exceeded maximum CPU time for source'
                flags |= FLAG_CPU_C
                break
            if dlnp < 0.1:
                break
        #print 'Simultaneous fit took:', Time()-tfit

    #print 'Blob finished re-opt:', Time()-tlast
    #tlast = Time()

    # Variances
    srcinvvars = [[] for src in srcs]
    subcat.thawAllRecursive()
    subcat.freezeAllParams()
    for isub in range(len(srcs)):
        #print 'Variances for source', isub
        subcat.thawParam(isub)
        src = subcat[isub]
        #print 'Source', src
        if src is None:
            subcat.freezeParam(isub)
            continue
        #print 'Params:', src.getParamNames()
        
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseE.fromEllipseESoft(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
            src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)

        #print 'Converted ellipse:', src

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
    #print 'Blob variances:', Time()-tlast
    #tlast = Time()

    # Check for sources with zero inverse-variance -- I think these
    # can be generated during the "Simultaneous re-opt" stage above --
    # sources can get scattered outside the blob.
    keep = []
    for i,(src,ivar) in enumerate(zip(srcs, srcinvvars)):
        #print 'ivar', ivar
        # Arbitrarily look at the first element (RA)?
        if ivar[0] == 0.:
            continue
        keep.append(i)
    if len(keep) < len(srcs):
        print 'Keeping', len(keep), 'of', len(srcs), 'sources with non-zero ivar'
        Isrcs        = [Isrcs[i]        for i in keep]
        srcs         = [srcs[i]         for i in keep]
        srcinvvars   = [srcinvvars[i]   for i in keep]
        delta_chisqs = [delta_chisqs[i] for i in keep]
        flags        = [flags[i]        for i in keep]
        subcat = Catalog(*srcs)
        subtr.catalog = subcat
    
    # rchi2 quality-of-fit metric
    rchi2_num    = np.zeros((len(srcs),len(bands)), np.float32)
    rchi2_den    = np.zeros((len(srcs),len(bands)), np.float32)

    # fracflux degree-of-blending metric
    fracflux_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracflux_den = np.zeros((len(srcs),len(bands)), np.float32)

    # fracin flux-inside-blob metric
    fracin_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracin_den = np.zeros((len(srcs),len(bands)), np.float32)

    # fracmasked: fraction of masked pixels metric
    fracmasked_num = np.zeros(len(srcs), np.float32)
    fracmasked_den = np.zeros(len(srcs), np.float32)

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

                fracmasked_num[isrc] += np.sum((tim.getInvError()[slc] == 0) * np.abs(patch.patch)) / np.abs(counts[isrc])
                fracmasked_den[isrc] += np.sum(np.abs(patch.patch)) / np.abs(counts[isrc])

                fracin_num[isrc,iband] += np.sum(patch.patch)
                fracin_den[isrc,iband] += counts[isrc]

            tim.getSky().addTo(mod)
            chisq = ((tim.getImage() - mod) * tim.getInvError())**2
            
            for isrc,patch in enumerate(srcmods):
                if patch is None:
                    continue
                slc = patch.getSlice(mod)
                # We compute numerator and denom separately to handle edge objects, where
                # sum(patch.patch) < counts.  Also, to normalize by the number of images.
                # (Being on the edge of an image is like being in half an image.)
                rchi2_num[isrc,iband] += np.sum(chisq[slc] * patch.patch) / counts[isrc]
                # If the source is not near an image edge, sum(patch.patch) == counts[isrc].
                rchi2_den[isrc,iband] += np.sum(patch.patch) / counts[isrc]

    fracflux = fracflux_num / fracflux_den
    rchi2    = rchi2_num    / rchi2_den
    fracmasked = fracmasked_num / fracmasked_den
    fracin     = fracin_num     / fracin_den

    #print 'Blob finished metrics:', Time()-tlast
    print 'Blob', iblob, 'finished:', Time()-tlast

    return (Isrcs, srcs, srcinvvars, fracflux, rchi2, delta_chisqs, fracmasked, flags,
            all_models, performance, fracin)


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
        mn,mx = 0,0
        sig1 = 1.
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
                 outdir=None, T=None, cat=None, pixscale=None, **kwargs):

    import photutils

    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    try_makedirs(basedir)
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
    rr = np.array([s.getPosition().ra  for s in cat])
    dd = np.array([s.getPosition().dec for s in cat])
    ok,ix,iy = targetwcs.radec2pixelxy(rr, dd)
    T.oob = reduce(np.logical_or, [ix < 0.5, iy < 0.5, ix > W+0.5, iy > H+0.5])
    ix = np.clip(np.round(ix - 1), 0, W-1).astype(int)
    iy = np.clip(np.round(iy - 1), 0, H-1).astype(int)

    coimgs = []
    comods = []
    coresids = []

    AP = fits_table()

    T.saturated = np.zeros((len(T),len(bands)), bool)
    T.nobs = np.zeros((len(T), len(bands)), np.uint8)
    T.anymask = np.zeros((len(T), len(bands)), np.uint16)
    T.allmask = np.zeros((len(T), len(bands)), np.uint16)

    for iband,band in enumerate(bands):

        cow    = np.zeros((H,W), np.float32)
        cowimg = np.zeros((H,W), np.float32)
        cowmod = np.zeros((H,W), np.float32)

        cosatw = np.zeros((H,W), np.float32)
        cosatim= np.zeros((H,W), np.float32)

        #coimg  = np.zeros((H,W), np.float32)
        comod  = np.zeros((H,W), np.float32)
        cochi2 = np.zeros((H,W), np.float32)
        con     = np.zeros((H,W), np.uint8)
        congood = np.zeros((H,W), np.uint8)
        detiv   = np.zeros((H,W), np.float32)

        anysatur = np.zeros((H,W), bool)

        # These match the type of the "DQ" images.
        ormask   = np.zeros((H,W), np.int32)
        andmask  = np.empty((H,W), np.int32)
        andmask[:,:] = 0x7fffffff

        tinyw = 1e-30

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
            mo = mod[Yi,Xi]

            # invvar-weighted image & model
            cowimg[Yo,Xo] += iv * im
            cowmod[Yo,Xo] += iv * mo
            cow   [Yo,Xo] += iv

            # chi-squared
            cochi2[Yo,Xo] += iv * (im - mo)**2

            # Saturated (but not otherwise bad) pixels
            sat = ((tim.dq_bits['satur'] & tim.dq) > 0)
            bad = ((sum(tim.dq_bits[b] for b in ['badpix', 'cr', 'trans', 'edge', 'edge2']) & tim.dq) > 0)
            sat &= np.logical_not(bad)
            sat = binary_dilation(sat, iterations=10)
            cosatw [Yo,Xo] += tinyw * sat[Yi,Xi]
            cosatim[Yo,Xo] += tinyw * sat[Yi,Xi] * im
            
            # straight-up image & model
            #coimg[Yo,Xo] += im
            comod[Yo,Xo] += mod[Yi,Xi]
            con  [Yo,Xo] += 1

            dq = tim.dq[Yi,Xi]

            # FIXME -- this is subsumed by "ormask".
            anysatur[Yo,Xo] |= ((dq & tim.dq_bits['satur']) != 0)

            ormask [Yo,Xo] |= dq
            andmask[Yo,Xo] &= dq

            # point-source depth
            psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
            detsig1 = tim.sig1 / psfnorm
            detiv[Yo,Xo] += good * (1. / detsig1**2)


        T.saturated[:,iband] = anysatur[iy,ix]
        T.nobs [:,iband] = con[iy,ix]

        andmask_bits = np.sum(CP_DQ_BITS.values())
        T.anymask[:,iband] =  ormask [iy,ix]
        T.allmask[:,iband] = (andmask[iy,ix] & andmask_bits)
        # unless there were no images there...
        T.allmask[con[iy,ix] == 0, iband] = 0

        cowimg  /= np.maximum(cow, tinyw)
        cowmod  /= np.maximum(cow, tinyw)
        cosatim /= np.maximum(cosatw, tinyw)

        coresid = cowimg - cowmod
        coresid[cow == 0] = 0.

        cowimg[cow == 0] = cosatim[cow == 0]
        cowmod[cow == 0] = (comod[cow == 0] / np.maximum(1, con[cow == 0]))

        coimgs.append(cowimg)
        comods.append(cowmod)
        coresids.append(coresid)

        del comod
        #del coimg
        del cosatw
        del cosatim

        # Apertures, radii in ARCSEC.
        apertures_arcsec = np.array([0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.])
        apertures = apertures_arcsec / pixscale
        
        # Aperture photometry.
        invvar = cow
        resid = cowimg - cowmod
        image = cowimg
        with np.errstate(divide='ignore'):
            imsigma = 1.0/np.sqrt(invvar)
            imsigma[invvar == 0] = 0
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
        xy = np.vstack((xx - 1., yy - 1.)).T
        apimg = []
        apimgerr = []
        apres = []
        for rad in apertures:
            aper = photutils.CircularAperture(xy, rad)
            p = photutils.aperture_photometry(image, aper, error=imsigma)
            apimg.append(p.field('aperture_sum'))
            apimgerr.append(p.field('aperture_sum_err'))
            p = photutils.aperture_photometry(resid, aper)
            apres.append(p.field('aperture_sum'))
        ap = np.vstack(apimg).T
        ap[np.logical_not(np.isfinite(ap))] = 0.
        AP.set('apflux_img_%s' % band, ap)
        ap = 1./(np.vstack(apimgerr).T)**2
        ap[np.logical_not(np.isfinite(ap))] = 0.
        AP.set('apflux_img_ivar_%s' % band, ap)
        ap = np.vstack(apres).T
        ap[np.logical_not(np.isfinite(ap))] = 0.
        AP.set('apflux_resid_%s' % band, ap)
            
        # remove aliases
        del invvar
        del resid
        del image
        del apimg
        del apres
        del apimgerr
        del ap

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

        for name,img,gzip in [
            ('image',  cowimg,  False),
            ('invvar', cow,     False),
            ('model',  cowmod,  True),
            ('chi2',   cochi2,  False),
            ('depth',  detiv,   True),
            ('nexp',   congood, True),
            ]:

            # Make a copy, because each image has different values for these headers...
            hdr2 = MyFITSHDR()
            for r in hdr.records():
                hdr2.add_record(r)
            hdr2.add_record(dict(name='IMTYPE', value=name,
                                 comment='DECaLS image type'))
            if name in ['image', 'model']:
                hdr2.add_record(dict(name='MAGZERO', value=22.5,
                                     comment='Magnitude zeropoint'))
                hdr2.add_record(dict(name='BUNIT', value='nanomaggie',
                                     comment='AB mag = 22.5 - 2.5*log10(nanomaggie)'))
            if name in ['invvar', 'depth']:
                hdr2.add_record(dict(name='BUNIT', value='1/nanomaggie^2',
                                     comment='Ivar of AB mag = 22.5-2.5*log10(nanomaggie)'))

            fn = os.path.join(basedir,
                              'decals-%s-%s-%s.fits' % (brickname, name, band))
            if gzip:
                fn += '.gz'
            fitsio.write(fn, img, clobber=True, header=hdr2)
            print 'Wrote', fn

    tmpfn = create_temp(suffix='.png')

    for name,ims,rgbkw in [('image', coimgs, rgbkwargs),
                           ('model', comods, rgbkwargs),
                           ('resid', coresids, rgbkwargs_resid),
                           ]:
        plt.imsave(tmpfn, get_rgb(ims, bands, **rgbkw), origin='lower')
        jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name))
        cmd = ('pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn))
        os.system(cmd)
        os.unlink(tmpfn)
        print 'Wrote', jpegfn

    if ps is not None:
        plt.clf()
        ok,x0,y0 = targetwcs.radec2pixelxy(T.orig_ra, T.orig_dec)
        ok,x1,y1 = targetwcs.radec2pixelxy(T.ra, T.dec)
        dimshow(get_rgb(coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        plt.plot(np.vstack((x0,x1))-1, np.vstack((y0,y1))-1, 'r-')
        plt.plot(x1-1, y1-1, 'r.')
        plt.axis(ax)
        ps.savefig()

    return dict(T = T, AP=AP, apertures_pix=apertures, apertures_arcsec=apertures_arcsec)


def stage_wise_forced(
    cat=None,
    T=None,
    targetwcs=None,
    brickname=None,
    outdir=None,
    **kwargs):

    from wise.forcedphot import unwise_forcedphot, unwise_tiles_touching_wcs

    decals = Decals()
    brick = decals.get_brick_by_name(brickname)
    roiradec = [brick.ra1, brick.ra2, brick.dec1, brick.dec2]
    tiles = unwise_tiles_touching_wcs(targetwcs)
    print 'Cut to', len(tiles), 'unWISE tiles'

    wcat = []
    for src in cat:
        src = src.copy()
        src.setBrightness(NanoMaggies(w=1.))
        wcat.append(src)

    W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec,
                          unwise_dir=unwise_dir, use_ceres=useCeres)
    W.rename('tile', 'unwise_tile')
    return dict(WISE=W)
    

'''
Write catalog output
'''
def stage_writecat(
    version_header=None,
    T=None,
    WISE=None,
    AP=None,
    apertures_arcsec=None,
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

    # print 'Source types:'
    # for src in cat:
    #     print '  ', type(src)
    #print 'T:'
    #T.about()

    from desi_common import prepare_fits_catalog
    fs = None
    TT = T.copy()
    for k in ['itx','ity','index']:
        if k in TT.get_columns():
            TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob',
                       'fracflux','fracmasked','saturated','rchi2','dchisq','nobs',
                       'fracin',
                       'oob', 'anymask', 'allmask',
                       'decam_flags']:
            TT.rename(col, 'sdss_%s' % col)
    TT.tx = TT.tx.astype(np.float32)
    TT.ty = TT.ty.astype(np.float32)

    # Renumber blobs to make them contiguous.
    ublob,iblob = np.unique(TT.blob, return_inverse=True)
    del ublob
    assert(len(iblob) == len(TT))
    TT.blob = iblob.astype(np.int32)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.brickname = np.array([brickname] * len(TT))
    TT.objid   = np.arange(len(TT)).astype(np.int32)
    
    allbands = 'ugrizY'

    TT.decam_rchi2    = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracflux = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracin   = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_nobs     = np.zeros((len(TT), len(allbands)), np.uint8)
    TT.decam_saturated = np.zeros((len(TT), len(allbands)), TT.saturated.dtype)
    TT.decam_anymask = np.zeros((len(TT), len(allbands)), TT.anymask.dtype)
    TT.decam_allmask = np.zeros((len(TT), len(allbands)), TT.allmask.dtype)
    for iband,band in enumerate(bands):
        i = allbands.index(band)
        TT.decam_rchi2[:,i] = TT.rchi2[:,iband]
        TT.decam_fracflux[:,i] = TT.fracflux[:,iband]
        TT.decam_fracin[:,i] = TT.fracin[:,iband]
        TT.decam_nobs[:,i] = TT.nobs[:,iband]
        TT.decam_saturated[:,i] = TT.saturated[:,iband]
        TT.decam_anymask[:,i] = TT.allmask[:,iband]
        TT.decam_allmask[:,i] = TT.anymask[:,iband]

    TT.rename('fracmasked', 'decam_fracmasked')
    TT.rename('oob', 'out_of_bounds')

    TT.delete_column('rchi2')
    TT.delete_column('fracflux')
    TT.delete_column('fracin')
    TT.delete_column('nobs')
    TT.delete_column('saturated')
    TT.delete_column('anymask')
    TT.delete_column('allmask')

    # How many apertures?
    ap = AP.get('apflux_img_%s' % bands[0])
    #print 'Aperture flux shape:', ap.shape
    #print 'T:', len(TT)
    n,A = ap.shape
    
    TT.decam_apflux = np.zeros((len(TT), len(allbands), A), np.float32)
    TT.decam_apflux_ivar = np.zeros((len(TT), len(allbands), A), np.float32)
    TT.decam_apflux_resid = np.zeros((len(TT), len(allbands), A), np.float32)
    for iband,band in enumerate(bands):
        i = allbands.index(band)
        TT.decam_apflux[:,i,:] = AP.get('apflux_img_%s' % band)
        TT.decam_apflux_ivar[:,i,:] = AP.get('apflux_img_ivar_%s' % band)
        TT.decam_apflux_resid[:,i,:] = AP.get('apflux_resid_%s' % band)

    cat.thawAllRecursive()
    hdr = None
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs,
                                  allbands=allbands)

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)

    for i,ap in enumerate(apertures_arcsec):
        primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                comment='Aperture radius, in arcsec'))

    bits = CP_DQ_BITS.values()
    bits.sort()
    bitmap = dict((v,k) for k,v in CP_DQ_BITS.items())
    for i in range(16):
        bit = 1<<i
        if bit in bitmap:
            primhdr.add_record(dict(name='MASKB%i' % i, value=bitmap[bit],
                                    comment='Mask bit 2**%i=%i meaning' % (i, bit)))

    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = (bx - 1.).astype(np.float32)
    T2.by = (by - 1.).astype(np.float32)

    decals = Decals()
    brick = decals.get_brick_by_name(brickname)
    T2.brick_primary = ((T2.ra  >= brick.ra1 ) * (T2.ra  < brick.ra2) *
                        (T2.dec >= brick.dec1) * (T2.dec < brick.dec2))
    
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

    # Rename source types.
    typemap = dict(S='PSF', E='EXP', D='DEV', C='COMP')
    T2.type = np.array([typemap[t] for t in T2.type])

    # Convert WISE fluxes from Vega to AB.
    # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2ab
    vega_to_ab = dict(w1=2.699,
                      w2=3.339,
                      w3=5.174,
                      w4=6.620)

    for band in [1,2,3,4]:
        primhdr.add_record(dict(name='WISEAB%i' % band, value=vega_to_ab['w%i' % band],
                                comment='WISE Vega to AB conv for band %i' % band))

    for band in [1,2,3,4]:
        dm = vega_to_ab['w%i' % band]
        #print 'WISE', band
        #print 'dm', dm
        fluxfactor = 10.** (dm / -2.5)
        #print 'flux factor', fluxfactor
        f = WISE.get('w%i_nanomaggies' % band)
        f *= fluxfactor
        f = WISE.get('w%i_nanomaggies_ivar' % band)
        f *= (1./fluxfactor**2)

    T2.wise_flux = np.vstack([WISE.w1_nanomaggies, WISE.w2_nanomaggies,
                              WISE.w3_nanomaggies, WISE.w4_nanomaggies]).T
    T2.wise_flux_ivar = np.vstack([WISE.w1_nanomaggies_ivar, WISE.w2_nanomaggies_ivar,
                                   WISE.w3_nanomaggies_ivar, WISE.w4_nanomaggies_ivar]).T
    # T2.wise_nobs = np.vstack([WISE.w1_pronexp, WISE.w2_pronexp,
    #                           WISE.w3_pronexp, WISE.w4_pronexp]).T
    T2.wise_nobs = np.vstack([WISE.w1_nexp, WISE.w2_nexp,
                              WISE.w3_nexp, WISE.w4_nexp]).T

    T2.wise_fracflux = np.vstack([WISE.w1_profracflux, WISE.w2_profracflux,
                                  WISE.w3_profracflux, WISE.w4_profracflux]).T
    T2.wise_rchi2 = np.vstack([WISE.w1_prochi2, WISE.w2_prochi2,
                               WISE.w3_prochi2, WISE.w4_prochi2]).T

    if catalogfn is not None:
        fn = catalogfn
    else:
        if outdir is None:
            outdir = '.'
        outdir = os.path.join(outdir, 'tractor', brickname[:3])
        fn = os.path.join(outdir, 'tractor-%s.fits' % brickname)
    dirnm = os.path.dirname(fn)
    try_makedirs(dirnm)
        
    #T2.writeto(fn, header=hdr)
    #print 'Wrote', fn

    print 'Reading SFD maps...'

    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, T2.ra, T2.dec, get_ebv=True)
    ext = ext.astype(np.float32)
    
    decam_extinction = ext[:,:len(allbands)]
    wise_extinction = ext[:,len(allbands):]
    T2.ebv = ebv.astype(np.float32)

    T2.decam_mw_transmission = 10.**(-decam_extinction / 2.5)
    T2.wise_mw_transmission  = 10.**(-wise_extinction  / 2.5)

    # 'tx', 'ty', 
    # 'sdss_treated_as_pointsource', 
    # 'decam_flags',

    cols = [
        'brickid', 'brickname', 'objid', 'brick_primary', 'blob', 'type', 'ra', 'ra_ivar', 'dec', 'dec_ivar',
        'bx', 'by', 'decam_flux', 'decam_flux_ivar', 'decam_apflux',
        'decam_apflux_resid', 'decam_apflux_ivar', 'decam_mw_transmission', 'decam_nobs',
        'decam_rchi2', 'decam_fracflux', 'decam_fracmasked', 'decam_fracin',
        'decam_saturated',
        'out_of_bounds',
        'decam_anymask', 'decam_allmask',
        'wise_flux', 'wise_flux_ivar',
        'wise_mw_transmission', 'wise_nobs', 'wise_fracflux', 'wise_rchi2', 'dchisq',
        'fracdev', 'fracDev_ivar', 'shapeexp_r', 'shapeexp_r_ivar', 'shapeexp_e1',
        'shapeexp_e1_ivar', 'shapeexp_e2', 'shapeexp_e2_ivar', 'shapedev_r',
        'shapedev_r_ivar', 'shapedev_e1', 'shapedev_e1_ivar', 'shapedev_e2',
        'shapedev_e2_ivar', 'ebv', 'sdss_run',
        'sdss_camcol', 'sdss_field', 'sdss_id', 'sdss_objid', 'sdss_parent',
        'sdss_nchild', 'sdss_objc_type', 'sdss_objc_flags', 'sdss_objc_flags2',
        'sdss_flags', 'sdss_flags2', 'sdss_tai', 'sdss_ra', 'sdss_ra_ivar',
        'sdss_dec', 'sdss_dec_ivar', 'sdss_psf_fwhm',
        'sdss_mjd', 'sdss_theta_dev', 'sdss_theta_deverr', 'sdss_ab_dev', 'sdss_ab_deverr',
        'sdss_theta_exp', 'sdss_theta_experr', 'sdss_ab_exp', 'sdss_ab_experr',
        'sdss_fracdev', 'sdss_phi_dev_deg', 'sdss_phi_exp_deg', 'sdss_psfflux',
        'sdss_psfflux_ivar', 'sdss_cmodelflux', 'sdss_cmodelflux_ivar', 'sdss_modelflux',
        'sdss_modelflux_ivar', 'sdss_devflux', 'sdss_devflux_ivar', 'sdss_expflux',
        'sdss_expflux_ivar', 'sdss_extinction', 'sdss_calib_status',
        'sdss_resolve_status',
        ]

    # TUNIT cards.
    deg='deg'
    degiv='1/deg^2'
    units = dict(
        ra=deg, dec=deg,
        ra_ivar=degiv, dec_ivar=degiv,
        decam_flux='nanomaggies', decam_flux_ivar='1/nanomaggies^2',
        decam_apflux='nanomaggies', decam_apflux_ivar='1/nanomaggies^2',
        decam_apflux_resid='nanomaggies',
        wise_flux='nanomaggies', wise_flux_ivar='1/nanomaggies^2',
        shapeexp_r='arcsec', shapeexp_r_ivar='1/arcsec^2',
        shapedev_r='arcsec', shapedev_r_ivar='1/arcsec^2',
        ebv='mag',
        sdss_ra=deg, sdss_ra_ivar=degiv,
        sdss_dec=deg, sdss_dec_ivar=degiv,
        sdss_tai='sec', sdss_psf_fwhm='arcsec', sdss_mjd='days',
        sdss_theta_dev='arcsec', sdss_theta_exp='arcsec',
        sdss_theta_deverr='1/arcsec', sdss_theta_experr='1/arcsec',
        sdss_phi_dev_deg=deg, sdss_phi_exp_deg=deg,
        sdss_psfflux='nanomaggies', sdss_psfflux_ivar='1/nanomaggies^2',
        sdss_cmodelflux='nanomaggies', sdss_cmodelflux_ivar='1/nanomaggies^2',
        sdss_modelflux='nanomaggies', sdss_modelflux_ivar='1/nanomaggies^2',
        sdss_devflux='nanomaggies', sdss_devflux_ivar='1/nanomaggies^2',
        sdss_expflux='nanomaggies', sdss_expflux_ivar='1/nanomaggies^2',
        sdss_extinction='mag')

    for i,col in enumerate(cols):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    # match case to T2.
    cc = T2.get_columns()
    cclower = [c.lower() for c in cc]
    for i,c in enumerate(cols):
        if (not c in cc) and c in cclower:
            j = cclower.index(c)
            cols[i] = cc[j]

    #T2.writeto(fn, header=hdr, primheader=version_header, columns=cols)
    
    # 'primheader' is not written in Astrometry.net 0.53; we write ourselves.

    # fill in any empty columns (eg, SDSS columns in areas outside the
    # footprint)
    fitsB = np.uint8
    fitsI = np.int16
    fitsJ = np.int32
    fitsK = np.int64
    fitsD = float
    fitsE = np.float32
    coltypes = dict(sdss_run=fitsI,
                    sdss_camcol=fitsB,
                    sdss_field=fitsI,
                    sdss_id=fitsI,
                    sdss_objid=fitsK,
                    sdss_parent=fitsI,
                    sdss_nchild=fitsI,
                    sdss_objc_type=fitsJ,
                    sdss_objc_flags=fitsJ,
                    sdss_objc_flags2=fitsJ,
                    sdss_ra=fitsD,
                    sdss_ra_ivar=fitsD,
                    sdss_dec=fitsD,
                    sdss_dec_ivar=fitsD,
                    sdss_mjd=fitsJ,
                    sdss_resolve_status=fitsJ
                    )
    arrtypes = dict(sdss_flags=fitsJ,
                    sdss_flags2=fitsJ,
                    sdss_tai=fitsD,
                    sdss_psf_fwhm=fitsE,
                    sdss_theta_dev=fitsE,
                    sdss_theta_deverr=fitsE,
                    sdss_ab_dev=fitsE,
                    sdss_ab_deverr=fitsE,
                    sdss_theta_exp=fitsE,
                    sdss_theta_experr=fitsE,
                    sdss_ab_exp=fitsE,
                    sdss_ab_experr=fitsE,
                    sdss_fracdev=fitsE,
                    sdss_phi_dev_deg=fitsE,
                    sdss_phi_exp_deg=fitsE,
                    sdss_psfflux=fitsE,
                    sdss_psfflux_ivar=fitsE,
                    sdss_cmodelflux=fitsE,
                    sdss_cmodelflux_ivar=fitsE,
                    sdss_modelflux=fitsE,
                    sdss_modelflux_ivar=fitsE,
                    sdss_devflux=fitsE,
                    sdss_devflux_ivar=fitsE,
                    sdss_expflux=fitsE,
                    sdss_expflux_ivar=fitsE,
                    sdss_extinction=fitsE,
                    sdss_calib_status=fitsJ)
    Tcols = T2.get_columns()
    for c in cols:
        if not c in Tcols:
            print 'Filling in missing column', c
            if c in coltypes:
                print '  type', coltypes[c]
                T2.set(c, np.zeros(len(T2), coltypes[c]))
            if c in arrtypes:
                print '  array type', arrtypes[c]
                T2.set(c, np.zeros((len(T2),5), arrtypes[c]))

    # Blank out all SDSS fields for sources that have moved too much.
    xyz1 = radectoxyz(T2.ra, T2.dec)
    xyz2 = radectoxyz(T2.sdss_ra, T2.sdss_dec)
    d2 = np.sum((xyz2-xyz1)**2, axis=1)
    # 1.5 arcsec
    maxd2 = np.deg2rad(1.5 / 3600.)**2
    blankout = np.flatnonzero((T2.sdss_ra != 0) * (d2 > maxd2))
    print 'Blanking out', len(blankout), 'SDSS no-longer-matches'
    if len(blankout):
        Tcols = T2.get_columns()
        for c in Tcols:
            if c.startswith('sdss'):
                x = T2.get(c)
                x[blankout] = 0

    # If there are no "COMP" sources, it will be 'S3'...
    T2.type = T2.type.astype('S4')

    arrays = [T2.get(c) for c in cols]
    arrays = [np.array(a) if isinstance(a,list) else a
              for a in arrays]
    fitsio.write(fn, None, header=primhdr, clobber=True)
    fitsio.write(fn, arrays, names=cols, header=hdr)


    print 'Wrote', fn


def main():
    from astrometry.util.stages import CallGlobalTime, runstage
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

    parser.add_option('--no-ceres', dest='ceres', default=True, action='store_false',
                      help='Do not use Ceres Solver')

    parser.add_option('--nblobs', type=int, help='Debugging: only fit N blobs')
    parser.add_option('--blob', type=int, help='Debugging: start with blob #')

    parser.add_option('--no-pv', dest='pv', default='True', action='store_false',
                      help='Do not use Community Pipeline WCS with PV distortion terms -- solve using Astrometry.net')

    parser.add_option('--pipe', default=False, action='store_true',
                      help='"pipeline" mode')

    parser.add_option('--check-done', default=False, action='store_true',
                      help='Just check for existence of output files for this brick?')
    parser.add_option('--skip', default=False, action='store_true',
                      help='Quit if the output catalog already exists.')
    parser.add_option('--skip-coadd', default=False, action='store_true',
                      help='Quit if the output coadd jpeg already exists.')

    parser.add_option('--nsigma', type=float, help='Set N sigma source detection thresh')

    print
    print 'runbrick.py starting at', datetime.datetime.now().isoformat()
    print 'Command-line args:', sys.argv
    print

    opt,args = parser.parse_args()

    if opt.check_done or opt.skip or opt.skip_coadd:
        outdir = opt.outdir
        if outdir is None:
            outdir = '.'
        brickname = opt.brick
        if opt.skip_coadd:
            fn = os.path.join(outdir, 'coadd', brickname[:3], brickname, 'decals-%s-image.jpg' % brickname)
        else:
            fn = os.path.join(outdir, 'tractor', brickname[:3], 'tractor-%s.fits' % brickname)
        print 'Checking for', fn
        exists = os.path.exists(fn)
        if opt.skip_coadd and exists:
            return 0
        if exists:
            try:
                T = fits_table(fn)
                print 'Read', len(T), 'sources from', fn
            except:
                print 'Failed to read file', fn
                exists = False

        if opt.skip:
            if exists:
                return 0
        elif opt.check_done:
            if not exists:
                print 'Does not exist:', fn
                return -1
            print 'Found:', fn
            return 0

    Time.add_measurement(MemMeas)

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    global useCeres
    useCeres = opt.ceres

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

    if opt.nsigma:
        kwargs.update(nsigma=opt.nsigma)

    kwargs.update(pipe=opt.pipe)

    global mp
    if opt.threads and opt.threads > 1:
        from utils.debugpool import DebugPool, DebugPoolMeas
        pool = DebugPool(opt.threads, initializer=runbrick_global_init,
                         initargs=[])
        Time.add_measurement(DebugPoolMeas(pool))
        mp = multiproc(None, pool=pool)

        #mp = multiproc(opt.threads, init=runbrick_global_init,
        #               initargs=[])
    else:
        mp = multiproc(init=runbrick_global_init, initargs=[])
    # ??
    kwargs.update(mp=mp)

    if opt.nblobs is not None:
        kwargs.update(nblobs=opt.nblobs)
    if opt.blob is not None:
        kwargs.update(blob0=opt.blob)

    if opt.outdir:
        kwargs.update(outdir=opt.outdir)

    if opt.forceall:
        kwargs.update(forceall=True)
        
    opt.picklepat = opt.picklepat % dict(brick=opt.brick)

    prereqs = {
        'tims':None,

        #'srcs':'tims',

        'image_coadds':'tims',
        'srcs':'image_coadds',

        'fitblobs':'srcs',
        'fitblobs_finish':'fitblobs',
        'coadds': 'fitblobs_finish',
        'wise_forced': 'coadds',
        'writecat': 'wise_forced',

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

    initargs.update(pvwcs=opt.pv)

    t0 = Time()

    for stage in opt.stage:
        runstage(stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
                 prereqs=prereqs, plots=opt.plots, plots2=opt.plots2,
                 initial_args=initargs, **kwargs)

    print 'All done:', Time()-t0
    return 0

def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace

if __name__ == '__main__':
    #sys.settrace(trace)
    sys.exit(main())
