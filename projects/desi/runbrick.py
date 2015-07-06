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
from astrometry.util.ttime import Time, MemMeas, CpuMeas
from astrometry.sdss import DR9, band_index, AsTransWrapper

from tractor import *
from tractor.galaxy import *
from tractor.source_extractor import *
from tractor.utils import _GaussianPriors

from common import *
from runbrick_plots import *
from runbrick_plots import _plot_mods

## GLOBALS!  Oh my!
nocache = True
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

# Prior on (softened) ellipticity: Gaussian with this standard deviation
ellipticityStd = 0.25

ellipsePriors = _GaussianPriors(None)
ellipsePriors.add('ee1', 0., ellipticityStd, param=EllipseESoft(1.,0.,0.))
ellipsePriors.add('ee2', 0., ellipticityStd, param=EllipseESoft(1.,0.,0.))

class EllipseWithPriors(EllipseESoft):
    # EllipseESoft extends EllipseE extends ParamList, has GaussianPriorsMixin.
    # GaussianPriorsMixin sets a "gpriors" member variable to a _GaussianPriors
    def __init__(self, *args, **kwargs):
        super(EllipseWithPriors, self).__init__(*args, **kwargs)
        self.gpriors = ellipsePriors

    @staticmethod
    def fromRAbPhi(r, ba, phi):
        logr, ee1, ee2 = EllipseESoft.rAbPhiToESoft(r, ba, phi)
        return EllipseWithPriors(logr, ee1, ee2)

# Turn on/off caching for all new Tractor instances.
def create_tractor(tims, srcs):
    import tractor
    t = tractor.Tractor(tims, srcs)
    if nocache:
        t.disable_cache()
    return t
### Woot!
Tractor = create_tractor

from utils.debugpool import DebugPoolTimestamp
from astrometry.util.multiproc import multiproc
class MyMultiproc(multiproc):
    def __init__(self, *args, **kwargs):
        super(MyMultiproc, self).__init__(*args, **kwargs)
        self.t0 = Time()
        self.serial = []
        self.parallel = []
    def map(self, *args, **kwargs):
        tstart = Time()
        res = super(MyMultiproc, self).map(*args, **kwargs)
        tend = Time()
        self.serial.append((self.t0, tstart))
        self.parallel.append((tstart, tend))
        self.t0 = tend
        return res

    def report(self, nthreads):
        # Tally the serial time up to now
        tend = Time()
        self.serial.append((self.t0, tend))
        self.t0 = tend

        # Nasty... peek into Time members
        scpu = 0.
        swall = 0.
        print 'Serial:'
        for t0,t1 in self.serial:
            print t1-t0
            for m0,m1 in zip(t0.meas, t1.meas):
                if isinstance(m0, CpuMeas):
                    scpu  += m1.cpu_seconds_since(m0)
                    swall += m1.wall_seconds_since(m0)
                    #print '  total cpu', scpu, 'wall', swall
        pworkercpu = 0.
        pworkerwall = 0.
        pwall = 0.
        pcpu = 0.
        print 'Parallel:'
        for t0,t1 in self.parallel:
            print t1-t0
            for m0,m1 in zip(t0.meas, t1.meas):
                if isinstance(m0, DebugPoolTimestamp):
                    mt0 = m0.t0
                    mt1 = m1.t0
                    pworkercpu  += mt1['worker_cpu' ] - mt0['worker_cpu' ]
                    pworkerwall += mt1['worker_wall'] - mt0['worker_wall']
                elif isinstance(m0, CpuMeas):
                    pwall += m1.wall_seconds_since(m0)
                    pcpu  += m1.cpu_seconds_since(m0)
        print
        print 'Total serial CPU   ', scpu
        print 'Total serial Wall  ', swall
        print 'Total worker CPU   ', pworkercpu
        print 'Total worker Wall  ', pworkerwall
        print 'Total parallel Wall', pwall
        print 'Total parallel CPU ', pcpu
        print
        tcpu = scpu + pworkercpu + pcpu
        twall = swall + pwall
        if nthreads is None:
            nthreads = 1
        print 'Grand total CPU:              %.1f sec' % tcpu
        print 'Grand total Wall:             %.1f sec' % twall
        print 'Grand total CPU utilization:  %.2f cores' % (tcpu / twall)
        print 'Grand total efficiency:       %.1f %%' % (100. * tcpu / (twall * nthreads))
        print

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
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        hspace=0.2, wspace=0.05)

def _bounce_tim_get_resamp((tim, targetwcs)):
    return tim_get_resamp(tim, targetwcs)

def tims_compute_resamp(mp, tims, targetwcs):
    R = mp.map(_bounce_tim_get_resamp, [(tim,targetwcs) for tim in tims])
    for tim,r in zip(tims, R):
        tim.resamp = r

# Pretty much only used for plots; the real deal is _coadds().
def compute_coadds(tims, bands, targetwcs, images=None,
                   get_cow=False, get_n2=False):

    W = targetwcs.get_width()
    H = targetwcs.get_height()

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

def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               ra=None, dec=None,
               plots=False, ps=None, decals_dir=None, 
               target_extent=None, pipe=False, program_name='runbrick.py',
               bands='grz', const2psf=True, mp=None,
               mock_psf=False, **kwargs):
    t0 = tlast = Time()

    # early fail for mysterious "ImportError: c.so.6: cannot open shared object file: No such file or directory"
    from tractor.mix import c_gauss_2d_grid

    decals = Decals(decals_dir)
    if ra is not None:
        # Custom brick; fake 'brick' object
        brick = BrickDuck()
        brick.ra  = ra
        brick.dec = dec
        brickid = brick.brickid = -1
        brick.brickname = brickname
    else:
        brick = decals.get_brick_by_name(brickname)
        print 'Chosen brick:'
        brick.about()
        brickid = brick.brickid
        brickname = brick.brickname
        
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    if ra is not None:
        brick.ra1,nil  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,nil  = targetwcs.pixelxy2radec(1, H/2)
        nil, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        nil, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)
        print 'RA1,RA2', brick.ra1, brick.ra2
        print 'Dec1,Dec2', brick.dec1, brick.dec2


    version_hdr = get_version_header(program_name, decals.decals_dir)
    version_hdr.add_record(dict(name='BRICKNAM', value=brickname, comment='DECaLS brick RRRr[pm]DDd'))
    version_hdr.add_record(dict(name='BRICKID' , value=brickid,   comment='DECaLS brick id'))
    version_hdr.add_record(dict(name='RAMIN'   , value=brick.ra1, comment='Brick RA min'))
    version_hdr.add_record(dict(name='RAMAX'   , value=brick.ra2, comment='Brick RA max'))
    version_hdr.add_record(dict(name='DECMIN'  , value=brick.dec1, comment='Brick Dec min'))
    version_hdr.add_record(dict(name='DECMAX'  , value=brick.dec2, comment='Brick Dec max'))
    version_hdr.add_record(dict(name='BRICKRA' , value=brick.ra,  comment='Brick center'))
    version_hdr.add_record(dict(name='BRICKDEC', value=brick.dec, comment='Brick center'))
    print 'Version header:'
    print version_hdr

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
    if 'dr1' in T.get_columns():
        I = np.flatnonzero(T.dr1 == 1)
        print len(I), 'of', len(T), 'CCDs are photometric'
        T.cut(I)
    else:
        print 'WARNING: no "dr1" column in CCDs table; assuming all CCDs are photometric!'

    ims = []
    for t in T:
        print
        print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
        im = DecamImage(decals, t)
        ims.append(im)

    tnow = Time()
    print '[serial tims] Finding images touching brick:', tnow-tlast
    tlast = tnow

    # Run calibrations
    kwa = dict()
    args = [(im, kwa, brick.ra, brick.dec, pixscale, mock_psf)
            for im in ims]
    mp.map(run_calibs, args)
    tnow = Time()
    print '[parallel tims] Calibrations:', tnow-tlast
    tlast = tnow

    # Read images, clip to ROI
    args = [(im, targetrd, mock_psf, const2psf) for im in ims]
    tims = mp.map(read_one_tim, args)

    # Cut the table of CCDs to match the 'tims' list
    I = np.flatnonzero(np.array([tim is not None for tim in tims]))
    T.cut(I)
    ccds = T
    tims = [tim for tim in tims if tim is not None]
    assert(len(T) == len(tims))

    tnow = Time()
    print '[parallel tims] Read', len(T), 'images:', tnow-tlast
    tlast = tnow

    if len(tims) == 0:
        print 'No photometric CCDs overlap.  Quitting.'
        sys.exit(0)

    # _psf_check_plots(tims)
                    
    if not pipe:
        # save resampling params
        tims_compute_resamp(mp, tims, targetwcs)
        tnow = Time()
        print 'Computing resampling:', tnow-tlast
        tlast = tnow
        # Produce per-band coadds, for plots
        coimgs,cons = compute_coadds(tims, bands, targetwcs)
        tnow = Time()
        print 'Coadds:', tnow-tlast
        tlast = tnow

    # Cut "bands" down to just the bands for which we have images.
    allbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in allbands]
    print 'Cut bands to', bands

    for band in 'grz':
        hasit = band in bands
        version_hdr.add_record(dict(name='BRICK_%s' % band, value=hasit,
                                    comment='Does band %s touch this brick?' % band))
    version_hdr.add_record(dict(name='BRICKBND', value=''.join(bands),
                                comment='Bands touching this brick'))

    version_header = version_hdr

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick',
            'target_extent', 'ccds', 'bands']
    if not pipe:
        keys.extend(['coimgs', 'cons'])
    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
        print 'Pickling value', k, '=', rtn[k]
    return rtn

def _coadds(tims, bands, targetwcs,
            mods=None, xy=None, apertures=None, apxy=None,
            ngood=False, callback=None, callback_args=[],
            plots=False, ps=None):
    class Duck(object):
        pass
    C = Duck()

    W = targetwcs.get_width()
    H = targetwcs.get_height()
    
    # always, for patching?
    unweighted=True

    C.coimgs = []
    if mods:
        C.comods = []
        C.coresids = []

    #if unweighted:
    #    C.coimgs = []

    if apertures is not None:
        unweighted = True
        C.AP = fits_table()

    if xy:
        ix,iy = xy
        C.T = fits_table()
        C.T.nobs    = np.zeros((len(ix), len(bands)), np.uint8)
        C.T.anymask = np.zeros((len(ix), len(bands)), np.int16)
        C.T.allmask = np.zeros((len(ix), len(bands)), np.int16)

    tinyw = 1e-30
    for iband,band in enumerate(bands):

        cow    = np.zeros((H,W), np.float32)
        cowimg = np.zeros((H,W), np.float32)

        kwargs = dict(cowimg=cowimg, cow=cow)

        if mods:
            cowmod = np.zeros((H,W), np.float32)
            cochi2 = np.zeros((H,W), np.float32)
            kwargs.update(cowmod=cowmod, cochi2=cochi2)

        if unweighted:
            coimg  = np.zeros((H,W), np.float32)
            if mods:
                comod  = np.zeros((H,W), np.float32)
            con    = np.zeros((H,W), np.uint8)
            coiv   = np.zeros((H,W), np.float32)
            kwargs.update(coimg=coimg, coiv=coiv)

        # Note that we have 'congood' as well as 'nobs'.
        # 'congood' is used for the 'nexp' *image*
        # 'nobs' is used for the per-source measurement (you want to know the number
        #        of observations within the source footprint, not just the peak pixel
        #        which may be saturated, etc.)

        if ngood:
            congood = np.zeros((H,W), np.uint8)
            kwargs.update(congood=congood)

        if xy:
            # These match the type of the "DQ" images.
            ormask  = np.zeros((H,W), np.int16)
            andmask = np.empty((H,W), np.int16)
            allbits = reduce(np.bitwise_or, CP_DQ_BITS.values())
            andmask[:,:] = allbits
            detiv = np.zeros((H,W), np.float32)
            nobs  = np.zeros((H,W), np.uint8)
            kwargs.update(ormask=ormask, andmask=andmask, detiv=detiv, nobs=nobs)

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

            if unweighted:
                dq = tim.dq[Yi,Xi]
                # include BLEED, SATUR, INTERP pixels if no other pixels exists
                # (do this by eliminating all other CP flags)
                badbits = 0
                for bitname in ['badpix', 'cr', 'trans', 'edge', 'edge2']:
                    badbits |= CP_DQ_BITS[bitname]
                goodpix = ((dq & badbits) == 0)

                coimg[Yo,Xo] += goodpix * im
                con  [Yo,Xo] += goodpix
                coiv [Yo,Xo] += goodpix * 1./tim.sig1**2  # ...ish
                #del goodpix
                del dq

            if xy:
                dq = tim.dq[Yi,Xi]
                ormask [Yo,Xo] |= dq
                andmask[Yo,Xo] &= dq
                del dq
                # point-source depth
                psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
                detsig1 = tim.sig1 / psfnorm
                detiv[Yo,Xo] += (iv > 0) * (1. / detsig1**2)
                # raw exposure count
                nobs[Yo,Xo] += 1

            if ngood:
                congood[Yo,Xo] += (iv > 0)

            if mods:
                mo = mods[itim][Yi,Xi]
                # straight-up
                comod[Yo,Xo] += goodpix * mo
                # invvar-weighted
                cowmod[Yo,Xo] += iv * mo
                # chi-squared
                cochi2[Yo,Xo] += iv * (im - mo)**2
                del mo

            del Yo,Xo,Yi,Xi,im,iv

        cowimg /= np.maximum(cow, tinyw)
        C.coimgs.append(cowimg)
        if mods:
            cowmod  /= np.maximum(cow, tinyw)
            C.comods.append(cowmod)
            coresid = cowimg - cowmod
            coresid[cow == 0] = 0.
            C.coresids.append(coresid)

        if unweighted:
            coimg  /= np.maximum(con, 1)
            del con
            cowimg[cow == 0] = coimg[cow == 0]
            if mods:
                cowmod[cow == 0] = comod[cow == 0]

        if plots:
            plt.clf()
            dimshow(nobs, vmin=0, vmax=max(1,nobs.max()), cmap='jet')
            plt.colorbar()
            plt.title('Number of observations, %s band' % band)
            ps.savefig()

            for k,v in CP_DQ_BITS.items():
                plt.clf()
                dimshow(ormask & v, vmin=0, vmax=v)
                plt.title('OR mask, %s band: %s' % (band, k))
                ps.savefig()

                plt.clf()
                dimshow(andmask & v, vmin=0, vmax=v)
                plt.title('AND mask, %s band: %s' % (band,k))
                ps.savefig()


        if xy:
            C.T.nobs [:,iband] = nobs[iy,ix]
            C.T.anymask[:,iband] =  ormask [iy,ix]
            C.T.allmask[:,iband] =  andmask[iy,ix]
            # unless there were no images there...
            C.T.allmask[nobs[iy,ix] == 0, iband] = 0

        if apertures is not None:
            import photutils

            # Aperture photometry, using the unweighted "coimg" and "coiv" arrays.
            with np.errstate(divide='ignore'):
                imsigma = 1.0/np.sqrt(coiv)
                imsigma[coiv == 0] = 0

            apimg = []
            apimgerr = []
            if mods:
                apres = []
                
            for rad in apertures:
                aper = photutils.CircularAperture(apxy, rad)
                p = photutils.aperture_photometry(coimg, aper, error=imsigma)
                apimg.append(p.field('aperture_sum'))
                apimgerr.append(p.field('aperture_sum_err'))
                if mods:
                    p = photutils.aperture_photometry(coresid, aper)
                    apres.append(p.field('aperture_sum'))
            ap = np.vstack(apimg).T
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_%s' % band, ap)
            ap = 1./(np.vstack(apimgerr).T)**2
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_ivar_%s' % band, ap)
            if mods:
                ap = np.vstack(apres).T
                ap[np.logical_not(np.isfinite(ap))] = 0.
                C.AP.set('apflux_resid_%s' % band, ap)
                del apres
            del apimg,apimgerr,ap

        if callback is not None:
            callback(band, *callback_args, **kwargs)

    return C


def _write_band_images(band,
                       brickname, version_header, tims, targetwcs, basedir,
                       cowimg=None, cow=None, cowmod=None, cochi2=None,
                       detiv=None, congood=None, **kwargs):
    # copy version_header before modifying...
    hdr = fitsio.FITSHDR()
    for r in version_header.records():
        hdr.add_record(r)
    # Grab these keywords from all input files for this band...
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

    # Plug the WCS header cards into these images
    targetwcs.add_to_header(hdr)
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')

    imgs = [
        ('image',  cowimg,  False),
        ]
    if cowmod is not None:
        imgs.extend([
                ('invvar', cow,     False),
                ('model',  cowmod,  True),
                ('chi2',   cochi2,  False),
                ('depth',  detiv,   True),
                ('nexp',   congood, True),
                ])
    for name,img,gzip in imgs:
        # Make a copy, because each image has different values for these headers...
        hdr2 = MyFITSHDR()
        for r in hdr.records():
            hdr2.add_record(r)
        hdr2.add_record(dict(name='IMTYPE', value=name,
                             comment='DECaLS image type'))
        if name in ['image', 'model']:
            hdr2.add_record(dict(name='MAGZERO', value=22.5,
                                 comment='Magnitude zeropoint'))
            hdr2.add_record(dict(name='BUNIT', value='nanomaggy',
                                 comment='AB mag = 22.5 - 2.5*log10(nanomaggy)'))
        if name in ['invvar', 'depth']:
            hdr2.add_record(dict(name='BUNIT', value='1/nanomaggy^2',
                                 comment='Ivar of ABmag=22.5-2.5*log10(nmgy)'))

        fn = os.path.join(basedir,
                          'decals-%s-%s-%s.fits' % (brickname, name, band))
        if gzip:
            fn += '.gz'
        fitsio.write(fn, img, clobber=True, header=hdr2)
        print 'Wrote', fn

def stage_image_coadds(targetwcs=None, bands=None, tims=None, outdir=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None,
                       **kwargs):
    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    try_makedirs(basedir)

    C = _coadds(tims, bands, targetwcs,
                callback=_write_band_images,
                callback_args=(brickname, version_header, tims, targetwcs, basedir))

    tmpfn = create_temp(suffix='.png')
    for name,ims,rgbkw in [('image',C.coimgs,rgbkwargs)]:
        plt.imsave(tmpfn, get_rgb(ims, bands, **rgbkw), origin='lower')
        jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name))
        cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn)
        os.system(cmd)
        os.unlink(tmpfn)
        print 'Wrote', jpegfn

    return None

def _median_smooth_detmap((detmap, detiv, binning)):
    from scipy.ndimage.filters import median_filter
    #from astrometry.util.util import median_smooth
    #smoo = np.zeros_like(detmap)
    #median_smooth(detmap, detiv>0, 100, smoo)
    #smoo = median_filter(detmap, (50,50))
    # Bin down before median-filtering, for speed.
    binned,nil = bin_image(detmap, detiv, binning)
    smoo = median_filter(binned, (50,50))
    return smoo

def stage_srcs(coimgs=None, cons=None,
               targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               pipe=False, brickname=None,
               mp=None, outdir=None, nsigma=5,
               no_sdss=False,
               **kwargs):

    tlast = Time()
    if not no_sdss:
        # Read SDSS sources
        cols = ['parent', 'tai', 'mjd', 'psf_fwhm', 'objc_flags2', 'flags2',
                'devflux_ivar', 'expflux_ivar', 'calib_status', 'raerr',
                'decerr']
        cat,T = get_sdss_sources(bands, targetwcs, extracols=cols,
                                 ellipse=EllipseWithPriors.fromRAbPhi)
        tnow = Time()
        print '[serial srcs] SDSS sources:', tnow-tlast
        tlast = tnow
    else:
        cat = []
        T = None

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

    print 'Rendering detection maps...'
    detmaps, detivs = detection_maps(tims, targetwcs, bands, mp)
    tnow = Time()
    print '[parallel srcs] Detmaps:', tnow-tlast
    tlast = tnow

    # Median-smooth detection maps
    binning = 4
    smoos = mp.map(_median_smooth_detmap,
                   [(m,iv,binning) for m,iv in zip(detmaps, detivs)])
    tnow = Time()
    print '[parallel srcs] Median-filter detmaps:', tnow-tlast
    tlast = tnow

    print 'Bands:', bands
    print 'detmaps:', len(detmaps)

    for i,(detmap,detiv,smoo) in enumerate(zip(detmaps, detivs, smoos)):
        # Subtract binned median image.
        S = binning
        for ii in range(S):
            for jj in range(S):
                sh,sw = detmap[ii::S, jj::S].shape
                detmap[ii::S, jj::S] -= smoo[:sh,:sw]

        if plots:
            sig1 = 1./np.sqrt(np.median(detiv[detiv > 0]))
            kwa = dict(vmin=-2.*sig1, vmax=10.*sig1)
            kwa2 = dict(vmin=-2.*sig1, vmax=50.*sig1)

            subbed = detmap.copy()
            S = binning
            for ii in range(S):
                for jj in range(S):
                    subbed[ii::S, jj::S] -= smoo

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

    tnow = Time()
    print '[serial srcs] Peaks:', tnow-tlast
    tlast = tnow

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

    tnow = Time()
    print '[serial srcs] Blobs:', tnow-tlast
    tlast = tnow
    
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
        #print 'PSF', tim.psf
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
                   W=None,H=None, 
                   bands=None, ps=None, tims=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None,
                   simul_opt=False, mp=None,
                   **kwargs):
    tlast = Time()
    for tim in tims:
        assert(np.all(np.isfinite(tim.getInvError())))

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]

    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1

    set_source_radii(bands, orig_wcsxy0, tims, cat, minsigma)

    if plots:
        coimgs,cons = compute_coadds(tims, bands, targetwcs)
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

    tnow = Time()
    print '[serial fitblobs]:', tnow-tlast
    tlast = tnow

    if blobxy is not None:
        print 'Blobxy', blobxy
        bx,by = blobxy
        # Just set blob0,nblobs and let the code below take care of the rest...
        blob0 = blobs[by,bx]
        nblobs = 1
        print 'Blob:', blob0

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
                          targetwcs, tims, orig_wcsxy0, cat, bands, plots, ps,
                          simul_opt)
        iter = iterwrapper(iter, len(blobslices))
    else:
        iter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                          orig_wcsxy0, cat, bands, plots, ps, simul_opt)
        # to allow debugpool to only queue tasks one at a time
        iter = iterwrapper(iter, len(blobsrcs))
    R = mp.map(_bounce_one_blob, iter)
    print '[parallel fitblobs] Fitting sources took:', Time()-tlast

    return dict(fitblobs_R=R, tims=tims, ps=ps, blobs=blobs,
                blobslices=blobslices, blobsrcs=blobsrcs)
    
def stage_fitblobs_finish(
    brickname=None, version_header=None,
        T=None, blobsrcs=None, blobslices=None, blobs=None,
        tractor=None, cat=None, targetrd=None, pixscale=None,
        targetwcs=None,
        W=None,H=None, 
        bands=None, ps=None, tims=None,
        plots=False, plots2=False,
        fitblobs_R=None,
        outdir=None,
        **kwargs):

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
    
    ##
    if True:
        # Drop blobs that failed.
        good_blobs = np.array([i for i,r in enumerate(R) if r is not None])
        good_blobsrcs = [blobsrcs[i] for i in good_blobs]
        good_R        = [R       [i] for i in good_blobs]
        # DEBUGging / metrics for us
        all_models  = [r[8] for r in good_R]
        performance = [r[9] for r in good_R]
    
        allmods  = [None]*len(T)
        allperfs = [None]*len(T)
        for Isrcs,mods,perf in zip(good_blobsrcs,all_models,performance):
            for i,mod,per in zip(Isrcs,mods,perf):
                allmods[i] = mod
                allperfs[i] = per
        del all_models
        del performance

        from desi_common import prepare_fits_catalog
        from astrometry.util.file import pickle_to_file

        goodI = np.array([i for i,m in enumerate(allmods) if m is not None])
        TT = T[goodI]
        allmods  = [allmods [i] for i in goodI]
        allperfs = [allperfs[i] for i in goodI]
        assert(len(TT) == len(allmods))
        assert(len(TT) == len(allperfs))
        
        hdr = fitsio.FITSHDR()
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
    R = [r for r in R if r is not None and len(r[0])]

    II       = np.hstack([r[0] for r in R])
    srcivs   = np.hstack([np.hstack(r[2]) for r in R])
    fracflux = np.vstack([r[3] for r in R])
    rchi2    = np.vstack([r[4] for r in R])
    dchisqs  = np.vstack(np.vstack([r[5] for r in R]))
    fracmasked = np.vstack([r[6] for r in R])
    flags = np.hstack([r[7] for r in R])
    fracin = np.vstack([r[10] for r in R])
    started_in = np.hstack([r[11] for r in R])
    finished_in = np.hstack([r[12] for r in R])
    
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
    ns,nb = fracmasked.shape
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

    # Renumber blobs to make them contiguous.
    ublob,iblob = np.unique(T.blob, return_inverse=True)
    assert(len(iblob) == len(T))
    # Build map from (old+1) to new blob numbers, for the blob image.
    blobmap = np.empty(blobs.max()+2, int)
    # make sure that dropped blobs -> -1
    blobmap[:] = -1
    # in particular,
    blobmap[0] = -1
    blobmap[T.blob + 1] = iblob
    newblobs = blobmap[blobs+1]
    # write out blob map
    if outdir is None:
        outdir = '.'
    outdir = os.path.join(outdir, 'metrics', brickname[:3])
    try_makedirs(outdir)
    fn = os.path.join(outdir, 'blobs-%s.fits.gz' % brickname)
    fitsio.write(fn, newblobs, header=version_header, clobber=True)
    print 'Wrote', fn
    del newblobs
    del ublob
    T.blob = iblob.astype(np.int32)
    

    T.decam_flags = flags
    T.fracflux = fracflux
    T.fracin = fracin
    T.left_blob = np.logical_and(started_in, np.logical_not(finished_in))
    T.fracmasked = fracmasked
    T.rchi2 = rchi2
    T.dchisq = dchisqs.astype(np.float32)
    # Set -0 to 0
    T.dchisq[T.dchisq == 0.] = 0.
    # Make dchisq relative to the first element ("none" model)
    T.dchisq = T.dchisq[:, 1:] - T.dchisq[:, 0][:,np.newaxis]
    
    invvars = srcivs

    #print 'New catalog:'
    #for src in cat:
    #    print '  ', src
    
    rtn = dict(fitblobs_R = None)
    for k in ['tractor', 'cat', 'invvars', 'T']:
        rtn[k] = locals()[k]
    return rtn
                          
def _blob_iter(blobslices, blobsrcs, blobs,
               targetwcs, tims, orig_wcsxy0, cat, bands, plots, ps, simul_opt):
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

            # We pass the *original*, full-image PSF model; _one_blob applies offsets
            #psf = tim.psfex
            psf = tim.psf

            subtimargs.append((subimg, subie, subwcs, tim.subwcs, tim.getPhotoCal(),
                               tim.getSky(), psf, tim.name, sx0, sx1, sy0, sy1,
                               ox0, oy0, tim.band, tim.sig1, tim.modelMinval))

        # Here we assume the "blobs" array has been remapped...
        blobmask = (blobs[bslc] == iblob)

        yield (iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh, blobmask, subtimargs,
               [cat[i] for i in Isrcs], bands, plots, ps, simul_opt)

def _bounce_one_blob(X):
    try:
        return _one_blob(X)
    except:
        import traceback
        print 'Exception in _one_blob: (iblob = %i)' % (X[0])
        traceback.print_exc()
        print 'CARRYING ON...'
        return None

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
               srcs, bands, plots, ps, simul_opt)):

    print 'Fitting blob', iblob, ':', len(Isrcs), 'sources, size', blobw, 'x', blobh, len(subtimargs), 'images'

    plots2 = False

    #tlast = Time()
    alphas = [0.1, 0.3, 1.0]
    optargs = dict(priors=True, shared_params=False, alphas=alphas)

    bigblob = (blobw * blobh) > 100*100

    subtarget = targetwcs.get_subimage(bx0, by0, blobw, blobh)

    ok,x0,y0 = subtarget.radec2pixelxy(np.array([src.getPosition().ra  for src in srcs]),
                                       np.array([src.getPosition().dec for src in srcs]))
    started_in_blob = blobmask[np.clip(np.round(y0-1).astype(int), 0, blobh-1),
                               np.clip(np.round(x0-1).astype(int), 0, blobw-1)]
    #print 'Sources started in blob: ', started_in_blob

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
        if sy1-sy0 < 400 and sx1-sx0 < 400:
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
        subtim.subwcs = subsubwcs
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
        #tband = Time()
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

        # Create initial models for each tim x each source
        # initial_models is a list-of-lists: initial_models[itim][isrc]
        initial_models = []
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
            print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
            src = subcat[i]

            # Add this source's initial model back in.
            for tim,mods in zip(subtims, initial_models):
                mod = mods[i]
                if mod is not None:
                    mod.addTo(tim.getImage())

            if bigblob:
                # Create super-local sub-sub-tims around this source

                # Make the subimages the same size as the modelMasks.
                srctims = []
                modelMasks = []
                print 'Big blob: trimming:'
                for tim,imods in zip(subtims, initial_models):
                    mod = imods[i]
                    if mod is None:
                        continue
                    # for modelMasks
                    d = dict()
                    d[src] = Patch(0, 0, mod.patch != 0)
                    modelMasks.append(d)

                    mh,mw = mod.shape
                    x0,y0 = mod.x0 , mod.y0
                    x1,y1 = x0 + mw, y0 + mh
                    slc = slice(y0,y1), slice(x0, x1)
                    wcs = tim.getWcs().copy()
                    wx0,wy0 = wcs.getX0Y0()
                    wcs.setX0Y0(wx0 + x0, wy0 + y0)
                    srctim = Image(data=tim.getImage ()[slc],
                                   inverr=tim.getInvError()[slc],
                                   wcs=wcs, psf=ShiftedPsf(tim.getPsf(), x0, y0),
                                   photocal=tim.getPhotoCal(),
                                   sky=tim.getSky(), name=tim.name)
                    #srctim.subwcs = tim.getWcs().wcs.get_subimage(x0, y0, mw, mh)
                    srctim.subwcs = tim.subwcs.get_subimage(x0, y0, mw, mh)
                    srctim.band = tim.band
                    srctim.sig1 = tim.sig1
                    srctim.modelMinval = tim.modelMinval
                    srctim.x0 = x0
                    srctim.y0 = y0
                    srctims.append(srctim)
                    print '  ', tim.shape, 'to', srctim.shape

                if plots:
                    bx1 = bx0 + blobw
                    by1 = by0 + blobh
                    plt.clf()
                    coimgs,cons = compute_coadds(subtims, bands, subtarget)
                    dimshow(get_rgb(coimgs, bands), extent=(bx0,bx1,by0,by1))
                    plt.plot([bx0,bx0,bx1,bx1,bx0],[by0,by1,by1,by0,by0],'r-')
                    for tim in srctims:
                        h,w = tim.shape
                        tx,ty = [0,0,w,w,0], [0,h,h,0,0]
                        rd = [tim.getWcs().pixelToPosition(xi,yi)
                              for xi,yi in zip(tx,ty)]
                        ra  = [p.ra  for p in rd]
                        dec = [p.dec for p in rd]
                        ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'g-')

                        ra,dec = tim.subwcs.pixelxy2radec(tx, ty)
                        ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'm-')
                        
                    for tim in subtims:
                        h,w = tim.shape
                        tx,ty = [0,0,w,w,0], [0,h,h,0,0]
                        rd = [tim.getWcs().pixelToPosition(xi,yi)
                              for xi,yi in zip(tx,ty)]
                        ra  = [p.ra  for p in rd]
                        dec = [p.dec for p in rd]
                        ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'b-')

                        ra,dec = tim.subwcs.pixelxy2radec(tx, ty)
                        ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'c-')
                        
                    ps.savefig()
                        
                    
            else:
                srctims = subtims

                modelMasks = []
                for imods in initial_models:
                    d = dict()
                    modelMasks.append(d)
                    mod = imods[i]
                    if mod is not None:
                        d[src] = Patch(mod.x0, mod.y0, mod.patch != 0)

            #srctractor = BlobTractor(srctims, [src])
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
                dlnp,X,alpha = srctractor.optimize(**optargs)
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

                mp = MyMultiproc()
                tims_compute_resamp(mp, srctractor.getImages(), targetwcs)
                tims_compute_resamp(mp, subtims, targetwcs)

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
            for tim in srctims:
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
                dlnp,X,alpha = subtr.optimize(**optargs)
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
            dlnp,X,alpha = subtr.optimize(**optargs)
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

        #srctractor = BlobTractor(subtims, [src])
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
            shape = EllipseWithPriors(-1., 0., 0.)
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

            #tt0 = Time()
            cpu0 = time.clock()
            p0 = newsrc.getParams()
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(**optargs)
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
            # print 'Blob', iblob, 'src', i
            # print 'Fit', name, ': cpu time', time.clock() - cpu0, 'in', step, 'steps'
            # print 'time:', Time()-tt0
            # print 'src:', newsrc

            if plots and False:
                plt.clf()
                modimgs = srctractor.getModelImages()
                comods,nil = compute_coadds(subtims, bands, subtarget, images=modimgs)
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
                #print 'After first-round fit: model is', mod.shape
                mod = _clip_model_to_blob(mod, tim.shape, tim.getInvError())
                #print 'Clipped to', mod.shape
                d[newsrc] = Patch(mod.x0, mod.y0, mod.patch != 0)
            srctractor.setModelMasks(mm)
            enable_galaxy_cache()

            # Run another round of opt.
            cpu0 = time.clock()
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(**optargs)
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
            #print 'Refit', name, ': cpu time', time.clock() - cpu0
            #print 'src:', newsrc

            if plots and False:
                plt.clf()
                modimgs = srctractor.getModelImages()
                comods,nil = compute_coadds(subtims, bands, subtarget, images=modimgs)
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
                    comods,nil = compute_coadds(subtims, bands, subtarget, images=modimgs)
                    dimshow(get_rgb(comods, bands))
                    plt.title(modname)

                    chisqs = [((tim.getImage() - mod) * tim.getInvError())**2
                              for tim,mod in zip(subtims, modimgs)]
                else:
                    coimgs, cons = compute_coadds(subtims, bands, subtarget)
                    dimshow(get_rgb(coimgs, bands))
                    ax = plt.axis()
                    ok,x,y = subtarget.radec2pixelxy(src.getPosition().ra, src.getPosition().dec)
                    plt.plot(x-1, y-1, 'r+')
                    plt.axis(ax)
                    plt.title('Image')

                    chisqs = [((tim.getImage()) * tim.getInvError())**2
                              for tim in subtims]
                cochisqs,nil = compute_coadds(subtims, bands, subtarget, images=chisqs)
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

            # This is the "fractional" upgrade threshold for ptsrc->dev/exp:
            # 2% of ptsrc vs nothing
            fdlnp = 0.02 * (plnps['ptsrc'] - plnps['none'])

            #print 'dlnp:', dlnp
            #print 'fractional dlnp:', fdlnp
            #print 'n sigma:', np.sqrt(2.*(plnps['ptsrc'] - plnps['none']))
            dlnp = max(dlnp, fdlnp)

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
            else:
                #print 'Keeping ptsrc model:', expdiff, devdiff, '<', devexp_dlnp
                pass
        else:
            #print
            #print 'Dropping source:', src
            #srccat[0] = src
            #srctractor.getLogProb()
            pass

        # Penalized delta chi-squareds
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
    I = np.array([i for i,s in enumerate(srcs) if s is not None])
    keepI = [i for i,s in zip(Isrcs, srcs) if s is not None]
    keepsrcs = [s for s in srcs if s is not None]
    keepdeltas = [x for x,s in zip(delta_chisqs,srcs) if s is not None]
    flags = np.array([f for f,s in zip(flags, srcs) if s is not None])
    Isrcs = keepI
    srcs = keepsrcs
    delta_chisqs = keepdeltas
    subcat = Catalog(*srcs)
    subtr.catalog = subcat
    if len(I):
        started_in_blob = started_in_blob[I]
    else:
        started_in_blob = np.array([], bool)
    assert(len(started_in_blob) == len(srcs))

    ### Simultaneous re-opt.
    if simul_opt and len(subcat) > 1 and len(subcat) <= 10:
        #tfit = Time()
        # Optimize all at once?
        subcat.thawAllParams()
        #print 'Optimizing:', subtr
        #subtr.printThawedParams()

        flags |= FLAG_TRIED_C
        max_cpu = 300.
        cpu0 = time.clock()
        for step in range(50):
            dlnp,X,alpha = subtr.optimize(**optargs)
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
        started_in_blob = [started_in_blob[i] for i in keep]
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
    fracmasked_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracmasked_den = np.zeros((len(srcs),len(bands)), np.float32)

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
                if counts[isrc] == 0:
                    continue
                slc = patch.getSlice(mod)
                # (mod - patch) is flux from others
                # (mod - patch) / counts is normalized flux from others
                # patch/counts is unit profile
                fracflux_num[isrc,iband] += np.sum((mod[slc] - patch.patch) * np.abs(patch.patch)) / counts[isrc]**2
                fracflux_den[isrc,iband] += np.sum(np.abs(patch.patch)) / np.abs(counts[isrc])

                fracmasked_num[isrc,iband] += np.sum((tim.getInvError()[slc] == 0) * np.abs(patch.patch)) / np.abs(counts[isrc])
                fracmasked_den[isrc,iband] += np.sum(np.abs(patch.patch)) / np.abs(counts[isrc])

                fracin_num[isrc,iband] += np.abs(np.sum(patch.patch))
                fracin_den[isrc,iband] += np.abs(counts[isrc])

            tim.getSky().addTo(mod)
            chisq = ((tim.getImage() - mod) * tim.getInvError())**2
            
            for isrc,patch in enumerate(srcmods):
                if patch is None:
                    continue
                if counts[isrc] == 0:
                    continue
                slc = patch.getSlice(mod)
                # We compute numerator and denom separately to handle edge objects, where
                # sum(patch.patch) < counts.  Also, to normalize by the number of images.
                # (Being on the edge of an image is like being in half an image.)
                rchi2_num[isrc,iband] += np.sum(chisq[slc] * patch.patch) / counts[isrc]
                # If the source is not near an image edge, sum(patch.patch) == counts[isrc].
                rchi2_den[isrc,iband] += np.sum(patch.patch) / counts[isrc]

    fracflux   = fracflux_num   / np.maximum(1, fracflux_den)
    rchi2      = rchi2_num      / np.maximum(1, rchi2_den)
    fracmasked = fracmasked_num / np.maximum(1, fracmasked_den)
    # fracin_{num,den} are in flux * nimages units
    tinyflux = 1e-9
    fracin     = fracin_num     / np.maximum(tinyflux, fracin_den)

    ok,x1,y1 = subtarget.radec2pixelxy(np.array([src.getPosition().ra  for src in srcs]),
                                       np.array([src.getPosition().dec for src in srcs]))
    finished_in_blob = blobmask[np.clip(np.round(y1-1).astype(int), 0, blobh-1),
                                np.clip(np.round(x1-1).astype(int), 0, blobw-1)]
    #print 'Sources finished in blob:', finished_in_blob

    assert(len(finished_in_blob) == len(srcs))
    assert(len(finished_in_blob) == len(started_in_blob))

    #print 'Blob finished metrics:', Time()-tlast
    print 'Blob', iblob+1, 'finished' #:', Time()-tlast

    return (Isrcs, srcs, srcinvvars, fracflux, rchi2, delta_chisqs, fracmasked, flags,
            all_models, performance, fracin, started_in_blob, finished_in_blob)


def _get_mod((tim, srcs)):
    tractor = Tractor([tim], srcs)
    return tractor.getModelImage(0)

def stage_coadds(bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 outdir=None, T=None, cat=None, pixscale=None, plots=False,
                 mp=None,
                 **kwargs):
    tlast = Time()

    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    try_makedirs(basedir)
    fn = os.path.join(basedir, 'decals-%s-ccds.fits' % brickname)
    #
    ccds.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
    ccds.ccd_x1 = np.array([tim.x0 + tim.shape[1] for tim in tims]).astype(np.int16)
    ccds.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
    ccds.ccd_y1 = np.array([tim.y0 + tim.shape[0] for tim in tims]).astype(np.int16)
    rd = np.array([[tim.subwcs.pixelxy2radec(1, 1)[-2:],
                    tim.subwcs.pixelxy2radec(1, y1-y0)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, 1)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, y1-y0)[-2:]]
                    for tim,x0,y0,x1,y1 in
                   zip(tims, ccds.ccd_x0+1, ccds.ccd_y0+1, ccds.ccd_x1, ccds.ccd_y1)])
    ok,x,y = targetwcs.radec2pixelxy(rd[:,:,0], rd[:,:,1])
    ccds.brick_x0 = np.floor(np.min(x, axis=1)).astype(np.int16)
    ccds.brick_x1 = np.ceil (np.max(x, axis=1)).astype(np.int16)
    ccds.brick_y0 = np.floor(np.min(y, axis=1)).astype(np.int16)
    ccds.brick_y1 = np.ceil (np.max(y, axis=1)).astype(np.int16)
    ccds.sig1 = np.array([tim.sig1 for tim in tims])
    ccds.writeto(fn)
    print 'Wrote', fn

    #for tim in tims:
    #    print 'Tim', tim.name, 'PSF', tim.getPsf()

    tnow = Time()
    print '[serial coadds]:', tnow-tlast
    tlast = tnow
    mods = mp.map(_get_mod, [(tim, cat) for tim in tims])
    tnow = Time()
    print '[parallel coadds] Getting model images:', tnow-tlast
    tlast = tnow

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

    # Apertures, radii in ARCSEC.
    apertures_arcsec = np.array([0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.])
    apertures = apertures_arcsec / pixscale

    # Aperture photometry locations
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
    ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
    apxy = np.vstack((xx - 1., yy - 1.)).T
    del xx,yy,ok,ra,dec

    C = _coadds(tims, bands, targetwcs, mods=mods, xy=(ix,iy), ngood=True,
                apertures=apertures, apxy=apxy,
                callback=_write_band_images,
                callback_args=(brickname, version_header, tims, targetwcs, basedir),
                plots=plots, ps=ps)

    for c in ['nobs', 'anymask', 'allmask']:
        T.set(c, C.T.get(c))

    tmpfn = create_temp(suffix='.png')
    for name,ims,rgbkw in [('image', C.coimgs, rgbkwargs),
                           ('model', C.comods, rgbkwargs),
                           ('resid', C.coresids, rgbkwargs_resid),
                           ]:
        plt.imsave(tmpfn, get_rgb(ims, bands, **rgbkw), origin='lower')
        jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name))
        cmd = ('pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn))
        os.system(cmd)
        os.unlink(tmpfn)
        print 'Wrote', jpegfn

    if plots:
        plt.clf()
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        ok,x0,y0 = targetwcs.radec2pixelxy(T.orig_ra, T.orig_dec)
        ok,x1,y1 = targetwcs.radec2pixelxy(ra, dec)
        dimshow(get_rgb(C.coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        #plt.plot(np.vstack((x0,x1))-1, np.vstack((y0,y1))-1, 'r-')
        for xx0,yy0,xx1,yy1 in zip(x0,y0,x1,y1):
            plt.plot([xx0-1,xx1-1], [yy0-1,yy1-1], 'r-')
        plt.plot(x1-1, y1-1, 'r.')
        plt.axis(ax)
        ps.savefig()

    tnow = Time()
    print '[serial coadds] Aperture photometry, wrap-up', tnow-tlast

    return dict(T=T, AP=C.AP, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec)


def stage_wise_forced(
    cat=None,
    T=None,
    targetwcs=None,
    brickname=None,
    brick=None,
    outdir=None,
    **kwargs):
    from wise.forcedphot import unwise_forcedphot, unwise_tiles_touching_wcs

    roiradec = [brick.ra1, brick.ra2, brick.dec1, brick.dec2]
    tiles = unwise_tiles_touching_wcs(targetwcs)
    print 'Cut to', len(tiles), 'unWISE tiles'

    wcat = []
    for src in cat:
        src = src.copy()
        src.setBrightness(NanoMaggies(w=1.))
        wcat.append(src)

    try:
        W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec,
                              unwise_dir=unwise_dir, use_ceres=useCeres)
    except:
        import traceback
        print 'unwise_forcedphot failed:'
        traceback.print_exc()

        if useCeres:
            print 'Trying without Ceres...'
            W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec,
                                  unwise_dir=unwise_dir, use_ceres=False)

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
    brick=None,
    invvars=None,
    catalogfn=None,
    outdir=None,
    **kwargs):

    from desi_common import prepare_fits_catalog
    fs = None
    TT = T.copy()
    for k in ['itx','ity','index']:
        if k in TT.get_columns():
            TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob',
                       'fracflux','fracmasked', 'rchi2','dchisq','nobs',
                       'fracin', 'orig_ra', 'orig_dec', 'left_blob',
                       'oob', 'anymask', 'allmask',
                       'decam_flags']:
            TT.rename(col, 'sdss_%s' % col)
    TT.tx = TT.tx.astype(np.float32)
    TT.ty = TT.ty.astype(np.float32)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.brickname = np.array([brickname] * len(TT))
    TT.objid   = np.arange(len(TT)).astype(np.int32)
    
    allbands = 'ugrizY'

    TT.decam_rchi2    = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracflux = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracmasked = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracin   = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_nobs     = np.zeros((len(TT), len(allbands)), np.uint8)
    TT.decam_anymask  = np.zeros((len(TT), len(allbands)), TT.anymask.dtype)
    TT.decam_allmask  = np.zeros((len(TT), len(allbands)), TT.allmask.dtype)
    B = np.array([allbands.index(band) for band in bands])
    TT.decam_rchi2     [:,B] = TT.rchi2
    TT.decam_fracflux  [:,B] = TT.fracflux
    TT.decam_fracmasked[:,B] = TT.fracmasked
    TT.decam_fracin    [:,B] = TT.fracin
    TT.decam_nobs      [:,B] = TT.nobs
    TT.decam_anymask   [:,B] = TT.anymask
    TT.decam_allmask   [:,B] = TT.allmask
    TT.delete_column('rchi2')
    TT.delete_column('fracflux')
    TT.delete_column('fracin')
    TT.delete_column('nobs')
    TT.delete_column('anymask')
    TT.delete_column('allmask')

    TT.rename('oob', 'out_of_bounds')

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

    # mod
    T2.ra += (T2.ra <   0) * 360.
    T2.ra -= (T2.ra > 360) * 360.

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)

    primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                            comment='Band order in array values'))

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

    ok,bx,by = targetwcs.radec2pixelxy(T2.orig_ra, T2.orig_dec)
    T2.bx0 = (bx - 1.).astype(np.float32)
    T2.by0 = (by - 1.).astype(np.float32)

    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = (bx - 1.).astype(np.float32)
    T2.by = (by - 1.).astype(np.float32)

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
    T2.type = np.array([typemap.get(t,t) for t in T2.type])

    # For sources that had DECam flux initialization from SDSS but no
    # overlapping images (hence decam_flux_ivar = 0), zero out the DECam flux.
    T2.decam_flux[T2.decam_flux_ivar == 0] = 0.

    if WISE is not None:
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
    if WISE is not None:
        T2.wise_mw_transmission  = 10.**(-wise_extinction  / 2.5)

    # 'tx', 'ty', 
    # 'sdss_treated_as_pointsource', 
    # 'decam_flags',

    T2.dchisq[T2.dchisq != 0] *= -1.

    cols = [
        'brickid', 'brickname', 'objid', 'brick_primary', 'blob', 'type',
        'ra', 'ra_ivar', 'dec', 'dec_ivar',
        'bx', 'by', 'bx0', 'by0',
        'left_blob', 
        'decam_flux', 'decam_flux_ivar', 'decam_apflux',
        'decam_apflux_resid', 'decam_apflux_ivar', 'decam_mw_transmission', 'decam_nobs',
        'decam_rchi2', 'decam_fracflux', 'decam_fracmasked', 'decam_fracin',
        'out_of_bounds',
        'decam_anymask', 'decam_allmask',
        ]
    if WISE is not None:
        cols.extend([
                'wise_flux', 'wise_flux_ivar',
                'wise_mw_transmission', 'wise_nobs', 'wise_fracflux', 'wise_rchi2',
                ])
    cols.extend([
        'dchisq',
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
        ])

    # TUNIT cards.
    deg='deg'
    degiv='1/deg^2'
    units = dict(
        ra=deg, dec=deg,
        ra_ivar=degiv, dec_ivar=degiv,
        decam_flux='nanomaggy', decam_flux_ivar='1/nanomaggy^2',
        decam_apflux='nanomaggy', decam_apflux_ivar='1/nanomaggy^2',
        decam_apflux_resid='nanomaggy',
        wise_flux='nanomaggy', wise_flux_ivar='1/nanomaggy^2',
        shapeexp_r='arcsec', shapeexp_r_ivar='1/arcsec^2',
        shapedev_r='arcsec', shapedev_r_ivar='1/arcsec^2',
        ebv='mag',
        sdss_ra=deg, sdss_ra_ivar=degiv,
        sdss_dec=deg, sdss_dec_ivar=degiv,
        sdss_tai='sec', sdss_psf_fwhm='arcsec', sdss_mjd='days',
        sdss_theta_dev='arcsec', sdss_theta_exp='arcsec',
        sdss_theta_deverr='1/arcsec', sdss_theta_experr='1/arcsec',
        sdss_phi_dev_deg=deg, sdss_phi_exp_deg=deg,
        sdss_psfflux='nanomaggy', sdss_psfflux_ivar='1/nanomaggy^2',
        sdss_cmodelflux='nanomaggy', sdss_cmodelflux_ivar='1/nanomaggy^2',
        sdss_modelflux='nanomaggy', sdss_modelflux_ivar='1/nanomaggy^2',
        sdss_devflux='nanomaggy', sdss_devflux_ivar='1/nanomaggy^2',
        sdss_expflux='nanomaggy', sdss_expflux_ivar='1/nanomaggy^2',
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

    arrays = [T2.get(c) for c in cols]
    arrays = [np.array(a) if isinstance(a,list) else a
              for a in arrays]
    fitsio.write(fn, None, header=primhdr, clobber=True)
    fitsio.write(fn, arrays, names=cols, header=hdr)


    print 'Wrote', fn


def stage_redo_apphot(targetwcs=None, bands=None, tims=None, outdir=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, pixscale=None,
                       **kwargs):
    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    try_makedirs(basedir)

    # read tractor output catalog
    T = fits_table(os.path.join('dr1d', 'tractor', brickname[:3], 'tractor-%s.fits' % brickname))

    # Apertures, radii in ARCSEC.
    apertures_arcsec = np.array([0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.])
    apertures = apertures_arcsec / pixscale

    # photometer positions from catalog
    ok,xx,yy = targetwcs.radec2pixelxy(T.ra, T.dec)
    apxy = np.vstack((xx - 1., yy - 1.)).T
    del xx,yy

    C = _coadd(tims, bands, targetwcs, apertures=apertures, apxy=apxy)

    T.add_columns_from(C.AP)
    T.about()
    
    plt.clf()
    plt.plot(T.decam_apflux[:,1], T.apflux_img_g, 'g.')
    plt.plot(T.decam_apflux[:,2], T.apflux_img_r, 'r.')
    plt.plot(T.decam_apflux[:,4], T.apflux_img_z, 'm.')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel('Old')
    plt.ylabel('New')
    plt.savefig('redo-ap-%s.png' % brickname)

    fn = 'redo-ap-%s.fits' % brickname
    T.writeto(fn)
    print 'Wrote', fn


def run_brick(brick, radec=None, pixscale=0.262,
              width=3600, height=3600,
              zoom=None,
              nblobs=None, blob=None, blobxy=None,
              pv=True, pipe=True, nsigma=6,
              simulOpt=False,
              wise=True,
              sdssInit=True,
              gaussPsf=False,
              ceres=True,
              outdir=None, decals_dir=None, threads=None,
              plots=False, plots2=False,
              plotbase=None, plotnumber=0,
              picklePattern='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=['writecat'],
              force=[], forceAll=False, writePickles=True):

    from astrometry.util.stages import CallGlobalTime, runstage
    from astrometry.util.multiproc import multiproc

    global useCeres

    initargs = {}
    kwargs = {}

    forceStages = [s for s in stages]
    forceStages.extend(stages)

    if forceAll:
        kwargs.update(forceall=True)

    if radec is not None:
        print 'RA,Dec:', radec
        assert(len(radec) == 2)
        ra,dec = radec
        try:
            ra = float(ra)
        except:
            from astrometry.util.starutil_numpy import hmsstring2ra
            ra = hmsstring2ra(ra)
        try:
            dec = float(dec)
        except:
            from astrometry.util.starutil_numpy import dmsstring2dec
            dec = dmsstring2dec(dec)
        print 'Parsed RA,Dec', ra,dec
        initargs.update(ra=ra, dec=dec)
        if brick is None:
            brick = ('custom-%06i%s%05i' %
                         (int(1000*ra), 'm' if dec < 0 else 'p',
                          int(1000*np.abs(dec))))
    initargs.update(brickname=brick)

    useCeres = ceres

    stagefunc = CallGlobalTime('stage_%s', globals())

    plot_base_default = 'brick-%(brick)s'
    if plotbase is None:
        plotbase = plot_base_default
    ps = PlotSequence(plotbase % dict(brick=brick))
    initargs.update(ps=ps)

    if plotnumber:
        ps.skipto(plotnumber)

    kwargs.update(ps=ps, nsigma=nsigma, mock_psf=gaussPsf,
                  simul_opt=simulOpt, pipe=pipe,
                  no_sdss=not(sdssInit),
                  outdir=outdir, decals_dir=decals_dir,
                  plots=plots, plots2=plots2,
                  force=forceStages, write=writePickles)

    if threads and threads > 1:
        from utils.debugpool import DebugPool, DebugPoolMeas
        pool = DebugPool(threads, initializer=runbrick_global_init,
                         initargs=[])
        Time.add_measurement(DebugPoolMeas(pool, pickleTraffic=False))
        mp = MyMultiproc(None, pool=pool)
    else:
        mp = MyMultiproc(init=runbrick_global_init, initargs=[])
    kwargs.update(mp=mp)

    if nblobs is not None:
        kwargs.update(nblobs=nblobs)
    if blob is not None:
        kwargs.update(blob0=blob)
    if blobxy is not None:
        kwargs.update(blobxy=blobxy)

    picklePattern = picklePattern % dict(brick=brick)

    prereqs = {
        'tims':None,

        #'srcs':'tims',

        'image_coadds':'tims',
        'srcs':'image_coadds',

        'fitblobs':'srcs',
        'fitblobs_finish':'fitblobs',
        'coadds': 'fitblobs_finish',

        # wise_forced: see below

        'fitplots': 'fitblobs_finish',
        'psfplots': 'tims',
        'initplots': 'srcs',

        'redo_apphot': 'tims',
        }

    if wise:
        prereqs.update({
                'wise_forced': 'coadds',
                'writecat': 'wise_forced',
                })
    else:
        prereqs.update({
                'writecat': 'coadds',
                })
        
    initargs.update(W=width, H=height, pixscale=pixscale,
                    target_extent=zoom)

    t0 = Time()

    for stage in stages:
        runstage(stage, picklePattern, stagefunc, prereqs=prereqs,
                 initial_args=initargs, **kwargs)

    print 'All done:', Time()-t0
    mp.report(threads)
    

def main():
    import optparse
    import logging

    ep = '''
eg, to run a small field containing a cluster:
\n
python -u projects/desi/runbrick.py --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle
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

    parser.add_option('-b', '--brick', help='Brick name to run: default %default',
                      default='2440p070')

    parser.add_option('--radec', help='RA,Dec center for a custom location (not a brick)',
                      nargs=2)
    parser.add_option('--pixscale', type=float, default=0.262,
                      help='Pixel scale of the output coadds (arcsec/pixel)')

    parser.add_option('-d', '--outdir', help='Set output base directory')
    parser.add_option('--decals-dir', type=str, default=None,
                      help='Overwrite the $DECALS_DIR environment variable')
    
    parser.add_option('--threads', type=int, help='Run multi-threaded')
    parser.add_option('-p', '--plots', dest='plots', action='store_true',
                      help='Per-blob plots?')
    parser.add_option('--plots2', action='store_true',
                      help='More plots?')

    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')

    parser.add_option('--plot-base', help='Base filename for plots, default brick-BRICK')
    parser.add_option('--plot-number', type=int, default=0, help='Set PlotSequence starting number')

    parser.add_option('-W', '--width', type=int, default=3600, help='Target image width (default %default)')
    parser.add_option('-H', '--height', type=int, default=3600, help='Target image height (default %default)')

    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 3600 0 3600")')

    parser.add_option('--no-ceres', dest='ceres', default=True, action='store_false',
                      help='Do not use Ceres Solver')

    parser.add_option('--nblobs', type=int, help='Debugging: only fit N blobs')
    parser.add_option('--blob', type=int, help='Debugging: start with blob #')
    parser.add_option('--blobxy', type=int, nargs=2, help='Debugging: run the single blob containing pixel <bx> <by>')

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

    parser.add_option('--simul-opt', action='store_true', default=False,
                      help='Do simultaneous optimization after model selection')

    parser.add_option('--no-wise', action='store_true', default=False,
                      help='Skip unWISE forced photometry')

    parser.add_option('--no-sdss', action='store_true', default=False,
                      help='Do not initialize from SDSS')

    parser.add_option('--gpsf', action='store_true', default=False,
                      help='Use a fixed single-Gaussian PSF')
    
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
            fn = os.path.join(outdir, 'coadd', brickname[:3], brickname,
                              'decals-%s-image.jpg' % brickname)
        else:
            fn = os.path.join(outdir, 'tractor', brickname[:3],
                              'tractor-%s.fits' % brickname)
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

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    Time.add_measurement(MemMeas)
    set_globals()

    if len(opt.stage) == 0:
        opt.stage.append('writecat')

    kwa = {}
    if opt.nsigma:
        kwa.update(nsigma=opt.nsigma)
    if opt.no_sdss:
        kwa.update(sdssInit=False)
    if opt.no_wise:
        kwa.update(wise=False)
        
    run_brick(opt.brick, radec=opt.radec, pixscale=opt.pixscale,
              width=opt.width, height=opt.height, zoom=opt.zoom,
              pv=opt.pv,
              threads=opt.threads, ceres=opt.ceres,
              gaussPsf=opt.gpsf, simulOpt=opt.simul_opt,
              nblobs=opt.nblobs, blob=opt.blob, blobxy=opt.blobxy,
              pipe=opt.pipe, outdir=opt.outdir, decals_dir=opt.decals_dir,
              plots=opt.plots, plots2=opt.plots2,
              plotbase=opt.plot_base, plotnumber=opt.plot_number,
              force=opt.force, forceAll=opt.forceall,
              stages=opt.stage, writePickles=opt.write,
              picklePattern=opt.picklepat, **kwa)
    return 0

def trace(frame, event, arg):
    print "%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno)
    return trace

if __name__ == '__main__':
    #sys.settrace(trace)
    sys.exit(main())
