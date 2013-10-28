#! /usr/bin/env python

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
from glob import glob

import fitsio

# qsub -d $(pwd) -N sequels -l "nodes=1:ppn=1" -l "pvmem=4gb" -o sequels.log -t 0-99 ./sequels.py

''' CFHT-LS W3 test area
http://terapix.iap.fr/article.php?id_article=841
wget 'ftp://ftpix.iap.fr/pub/CFHTLS-zphot-T0007/photozCFHTLS-W3_270912.out.gz'
gunzip photozCFHTLS-W3_270912.out.gz 
text2fits.py -n "*********" -f sddjjffffffjfjffffffjfffjffffffffffffffffff -H "id ra dec flag stargal r2 photoz zpdf zpdf_l68 zpdf_u168 chi2_zpdf mod ebv nbfilt zmin zl68 zu68 chi2_best zp_2 chi2_2 mods chis zq chiq modq u g r i z y eu eg er ei ez ey mu mg mr mi mz my" photozCFHTLS-W3_270912.out photozCFHTLS-W3_270912.fits
'''

if __name__ == '__main__':
    arr = os.environ.get('PBS_ARRAYID')
    d = os.environ.get('PBS_O_WORKDIR')
    if arr is not None and d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.ttime import *

from tractor import *
from tractor.sdss import *

from wisecat import wise_catalog_radecbox

import logging
'''
ln -s /clusterfs/riemann/raid007/ebosswork/eboss/photoObj photoObjs-new
ln -s /clusterfs/riemann/raid006/bosswork/boss/resolve/2013-07-29 photoResolve-new
'''

photoobjdir = 'photoObjs-new'
resolvedir = 'photoResolve-new'

if __name__ == '__main__':
    tiledir = 'wise-coadds'

    outdir = '%s-phot'
    tempoutdir = '%s-phot-temp'
    pobjoutdir = '%s-pobj'

    Time.add_measurement(MemMeas)

def get_tile_dir(basedir, coadd_id):
    return os.path.join(basedir, coadd_id[:3], coadd_id)

def get_photoobj_filename(rr, run, camcol, field):
    fn = os.path.join(photoobjdir, rr, '%i'%run, '%i'%camcol,
                      'photoObj-%06i-%i-%04i.fits' % (run, camcol, field))
    return fn

def read_photoobjs(r0, r1, d0, d1, margin, cols=None):
    log = logging.getLogger('sequels.read_photoobjs')

    if cols is None:
        cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type', 'modelflux',
                'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
                'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
                'resolve_status', 'nchild', 'flags', 'objc_flags',
                'run','camcol','field','id'
                ]

    wfn = os.path.join(resolvedir, 'window_flist.fits')

    ra,dec = (r0+r1)/2., (d0+d1)/2.
    rad = degrees_between(ra,dec, r0,d0)
    rad += np.hypot(13., 9.)/60.
    # a little extra margin
    rad += margin

    RCF = radec_to_sdss_rcf(ra, dec, radius=rad*60., tablefn=wfn)
    log.debug('Found', len(RCF), 'fields possibly in range')

    ddec = margin
    dra  = margin / min([np.cos(np.deg2rad(x)) for x in [d0,d1]])

    TT = []
    sdss = DR9()
    for run,camcol,field,r,d in RCF:
        log.debug('RCF', run, camcol, field)
        rr = sdss.get_rerun(run, field=field)
        if rr in [None, '157']:
            log.debug('Rerun 157')
            continue

        fn = get_photoobj_filename(rr, run, camcol, field)

        T = fits_table(fn, columns=cols)
        if T is None:
            log.debug('read 0 from', fn)
            continue
        log.debug('read', len(T), 'from', fn)
        T.cut((T.ra  >= (r0-dra )) * (T.ra  <= (r1+dra)) *
              (T.dec >= (d0-ddec)) * (T.dec <= (d1+ddec)) *
              ((T.resolve_status & 256) > 0))
        log.debug('cut to', len(T), 'in RA,Dec box and PRIMARY.')
        if len(T) == 0:
            continue
        TT.append(T)
    T = merge_tables(TT)
    return T

class BrightPointSource(PointSource):
    '''
    A class to use a pre-computed (constant) model, if available.
    '''
    def __init__(self, *args):
        super(BrightPointSource, self).__init__(*args)
        self.pixmodel = None
    def getUnitFluxModelPatch(self, *args, **kwargs):
        if self.pixmodel is not None:
            return self.pixmodel
        return super(BrightPointSource, self).getUnitFluxModelPatch(*args, **kwargs)

def set_bright_psf_mods(cat, WISE, T, brightcut, band, tile, wcs, sourcerad):
    mag = WISE.get('w%impro' % band)
    I = np.flatnonzero(mag < brightcut)
    if len(I) == 0:
        return
    BW = WISE[I]
    BW.nm = NanoMaggies.magToNanomaggies(mag[I])
    print len(I), 'catalog sources brighter than mag', brightcut
    I,J,d = match_radec(BW.ra, BW.dec, T.ra, T.dec, 4./3600., nearest=True)
    print 'Matched to', len(I), 'catalog sources (nearest)'
    if len(I) == 0:
        return

    fn = 'wise-psf-avg-pix-bright.fits'
    psfimg = fitsio.read(fn, ext=band-1).astype(np.float32)
    psfimg = np.maximum(0, psfimg)
    psfimg /= psfimg.sum()
    print 'PSF image', psfimg.shape
    print 'PSF image range:', psfimg.min(), psfimg.max()
    ph,pw = psfimg.shape
    pcx,pcy = ph/2, pw/2
    assert(ph == pw)
    phalf = ph/2

    ## HACK -- read an L1b frame to get the field rotation...
    thisdir = get_tile_dir(tiledir, tile.coadd_id)
    framesfn = os.path.join(thisdir, 'unwise-%s-w%i-frames.fits' % (tile.coadd_id, band))
    F = fits_table(framesfn)
    print 'intfn', F.intfn[0]
    #fwcs = fits_table(F.intfn[
    wisedir = 'wise-frames'
    scanid,frame = F.scan_id[0], F.frame_num[0]
    scangrp = scanid[-2:]
    fn = os.path.join(wisedir, scangrp, scanid, '%03i' % frame, 
                      '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))
    fwcs = Tan(fn)
    # Keep CD matrix, set CRVAL/CRPIX to star position
    fwcs.set_crpix(pcx+1, pcy+1)
    fwcs.set_imagesize(float(pw), float(ph))

    for i,j in zip(I, J):
        if not isinstance(cat[j], BrightPointSource):
            print 'Bright source matched non-point source', cat[j]
            continue

        fwcs.set_crval(BW.ra[i], BW.dec[i])
        L=3
        Yo,Xo,Yi,Xi,rims = resample_with_wcs(wcs, fwcs, [psfimg], L)
        x0,x1 = int(Xo.min()), int(Xo.max())
        y0,y1 = int(Yo.min()), int(Yo.max())
        mod = np.zeros((1+y1-y0, 1+x1-x0), np.float32)
        mod[Yo-y0, Xo-x0] += rims[0]

        pat = Patch(x0, y0, mod)
        cat[j].pixmodel = pat

        cat[j].fixedRadius = phalf
        sourcerad[j] = max(sourcerad[j], phalf)

def one_tile(tile, opt, savepickle, ps):
    bands = opt.bands
    outfn = opt.output % (tile.coadd_id)
    savewise_outfn = opt.save_wise_output % (tile.coadd_id)

    version = get_svn_version()
    print 'SVN version info:', version

    sband = 'r'
    bandnum = 'ugriz'.index(sband)

    tt0 = Time()
    print
    print 'Coadd tile', tile.coadd_id

    thisdir = get_tile_dir(tiledir, tile.coadd_id)
    fn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits' % (tile.coadd_id, bands[0]))
    print 'Reading', fn
    wcs = Tan(fn)
    r0,r1,d0,d1 = wcs.radec_bounds()
    print 'RA,Dec bounds:', r0,r1,d0,d1
    H,W = wcs.get_height(), wcs.get_width()

    objfn = os.path.join(tempoutdir, 'photoobjs-%s.fits' % tile.coadd_id)
    if os.path.exists(objfn):
        print 'Reading', objfn
        T = fits_table(objfn)
    else:
        print 'Did not find', objfn, '-- reading photoObjs'
        T = read_photoobjs(r0, r1, d0, d1, 1./60.)
        T.writeto(objfn)

    if opt.photoObjsOnly:
        return

    # Cut galaxies based on signal-to-noise of theta (effective
    # radius) measurement.
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

    aberrzero = np.zeros(len(T), bool)
    aberrzero[dev] = (T.ab_deverr[dev,b] == 0.)
    aberrzero[exp] = (T.ab_experr[exp,b] == 0.)

    maxtheta = np.zeros(len(T), bool)
    maxtheta[dev] = (T.theta_dev[dev,b] >= 29.5)
    maxtheta[exp] = (T.theta_exp[exp,b] >= 59.0)

    # theta S/N > modelflux for dev, 10*modelflux for exp
    bigthetasn = (thetasn > (T.modelflux[:,b] * (1.*dev + 10.*exp)))

    print sum(gal * (thetasn < 3.)), 'have low S/N in theta'
    print sum(gal * (T.modelflux[:,b] > 1e4)), 'have big flux'
    print sum(aberrzero), 'have zero a/b error'
    print sum(maxtheta), 'have the maximum theta'
    print sum(bigthetasn), 'have large theta S/N vs modelflux'
    
    badgals = gal * reduce(np.logical_or,
                           [thetasn < 3.,
                            T.modelflux[:,b] > 1e4,
                            aberrzero,
                            maxtheta,
                            bigthetasn,
                            ])
    print 'Found', sum(badgals), 'bad galaxies'
    T.treated_as_pointsource = badgals
    T.objc_type[badgals] = 6

    defaultflux = 1.

    # hack
    T.psfflux    = np.zeros((len(T),5), np.float32) + defaultflux
    T.cmodelflux = T.psfflux
    T.devflux    = T.psfflux
    T.expflux    = T.psfflux

    ok,T.x,T.y = wcs.radec2pixelxy(T.ra, T.dec)
    T.x = (T.x - 1.).astype(np.float32)
    T.y = (T.y - 1.).astype(np.float32)
    margin = 20.
    I = np.flatnonzero((T.x >= -margin) * (T.x < W+margin) *
                       (T.y >= -margin) * (T.y < H+margin))
    T.cut(I)
    print 'N objects:', len(T)

    wanyband = wband = 'w'
    print 'Creating tractor sources...'
    cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                  objs=T, bands=[], nanomaggies=True,
                                  extrabands=[wband],
                                  fixedComposites=True,
                                  useObjcType=True,
                                  classmap={PointSource: BrightPointSource})
    print 'Created', len(T), 'sources'
    assert(len(cat) == len(T))

    pixscale = wcs.pixel_scale()
    # crude intrinsic source radii, in pixels
    sourcerad = np.zeros(len(cat))
    for i in range(len(cat)):
        src = cat[i]
        if isinstance(src, PointSource):
            continue
        elif isinstance(src, HoggGalaxy):
            sourcerad[i] = (src.nre * src.re / pixscale)
        elif isinstance(src, FixedCompositeGalaxy):
            sourcerad[i] = max(src.shapeExp.re * ExpGalaxy.nre,
                               src.shapeDev.re * DevGalaxy.nre) / pixscale
    print 'sourcerad range:', min(sourcerad), max(sourcerad)

    # Find WISE-only catalog sources
    wfn = os.path.join(tempoutdir, 'wise-sources-%s.fits' % (tile.coadd_id))
    print 'looking for', wfn
    if os.path.exists(wfn):
        WISE = fits_table(wfn)
        print 'Read', len(WISE), 'WISE sources nearby'
    else:
        cols = ['ra','dec'] + ['w%impro'%band for band in [1,2,3,4]]
        WISE = wise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
        WISE.writeto(wfn)
        print 'Found', len(WISE), 'WISE sources nearby'

    for band in bands:
        mag = WISE.get('w%impro' % band)
        nm = NanoMaggies.magToNanomaggies(mag)
        WISE.set('w%inm' % band, nm)
        print 'Band', band, 'max WISE catalog flux:', max(nm)
        print '  (min mag:', mag.min(), ')'

    unmatched = np.ones(len(WISE), bool)
    I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 4./3600.)
    unmatched[I] = False
    UW = WISE[unmatched]
    print 'Got', len(UW), 'unmatched WISE sources'

    if opt.savewise:
        fitwiseflux = {}
        for band in bands:
            fitwiseflux[band] = np.zeros(len(UW))

    # Record WISE fluxes for catalog matches.
    # (this provides decent initialization for 'minsb' approx.)
    wiseflux = {}
    for band in bands:
        wiseflux[band] = np.zeros(len(T))
        # X[I] += Y[J] with duplicate I doesn't work.
        #wiseflux[band][J] += WISE.get('w%inm' % band)[I]
        lhs = wiseflux[band]
        rhs = WISE.get('w%inm' % band)[I]
        print 'Band', band, 'max matched WISE flux:', max(rhs)
        for j,f in zip(J, rhs):
            lhs[j] += f

    ok,UW.x,UW.y = wcs.radec2pixelxy(UW.ra, UW.dec)
    UW.x -= 1.
    UW.y -= 1.

    T.coadd_id = np.array([tile.coadd_id] * len(T))
    T.cell = np.zeros(len(T), np.int16)
    T.cell_x0 = np.zeros(len(T), np.int16)
    T.cell_y0 = np.zeros(len(T), np.int16)
    T.cell_x1 = np.zeros(len(T), np.int16)
    T.cell_y1 = np.zeros(len(T), np.int16)

    inbounds = np.flatnonzero((T.x >= -0.5) * (T.x < W-0.5) *
                              (T.y >= -0.5) * (T.y < H-0.5))

    for band in bands:
        tb0 = Time()
        print
        print 'Coadd tile', tile.coadd_id
        print 'Band', band
        wband = 'w%i' % band

        imfn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits'    % (tile.coadd_id, band))
        ivfn = os.path.join(thisdir, 'unwise-%s-w%i-invvar-m.fits' % (tile.coadd_id, band))
        ppfn = os.path.join(thisdir, 'unwise-%s-w%i-std-m.fits'    % (tile.coadd_id, band))
        nifn = os.path.join(thisdir, 'unwise-%s-w%i-n-m.fits'      % (tile.coadd_id, band))

        print 'Reading', imfn
        wcs = Tan(imfn)
        r0,r1,d0,d1 = wcs.radec_bounds()
        print 'RA,Dec bounds:', r0,r1,d0,d1
        ra,dec = wcs.radec_center()
        print 'Center:', ra,dec
        img = fitsio.read(imfn)
        print 'Reading', ivfn
        iv = fitsio.read(ivfn)
        print 'Reading', ppfn
        pp = fitsio.read(ppfn)
        print 'Reading', nifn
        nims = fitsio.read(nifn)

        sig1 = 1./np.sqrt(np.median(iv))
        minsig = getattr(opt, 'minsig%i' % band)
        minsb = sig1 * minsig
        print 'Sigma1:', sig1, 'minsig', minsig, 'minsb', minsb

        # Load the average PSF model (generated by wise_psf.py)
        P = fits_table('wise-psf-avg.fits', hdu=band)
        psf = GaussianMixturePSF(P.amp, P.mean, P.var)

        # Render the PSF profile for figuring out source radii for
        # approximation purposes.
        R = 100
        psf.radius = R
        pat = psf.getPointSourcePatch(0., 0.)
        assert(pat.x0 == pat.y0)
        assert(pat.x0 == -R)
        psfprofile = pat.patch[R, R:]
        #print 'PSF profile:', psfprofile

        # Reset default flux based on min radius
        defaultflux = minsb / psfprofile[opt.minradius]
        print 'Setting default flux', defaultflux

        # Set WISE source radii based on flux
        UW.rad = np.zeros(len(UW), int)
        wnm = UW.get('w%inm' % band)
        for r,pro in enumerate(psfprofile):
            flux = minsb / pro
            UW.rad[wnm > flux] = r
        UW.rad = np.maximum(UW.rad + 1, 3)

        # Set SDSS fluxes based on WISE catalog matches.
        wf = wiseflux[band]
        I = np.flatnonzero(wf > defaultflux)
        wfi = wf[I]
        print 'Initializing', len(I), 'fluxes based on catalog matches'
        for i,flux in zip(I, wf[I]):
            assert(np.isfinite(flux))
            cat[i].getBrightness().setBand(wanyband, flux)

        # Set SDSS radii based on WISE flux
        rad = np.zeros(len(I), int)
        for r,pro in enumerate(psfprofile):
            flux = minsb / pro
            rad[wfi > flux] = r
        srad2 = np.zeros(len(cat), int)
        srad2[I] = rad
        del rad

        # Set radii
        for i in range(len(cat)):
            src = cat[i]
            # set fluxes
            b = src.getBrightness()
            if b.getBand(wanyband) <= defaultflux:
                b.setBand(wanyband, defaultflux)
                
            R = max([opt.minradius, sourcerad[i], srad2[i]])
            # ??  This is used to select which sources are in-range
            sourcerad[i] = R
            if isinstance(src, PointSource):
                src.fixedRadius = R
            elif (isinstance(src, HoggGalaxy) or
                  isinstance(src, FixedCompositeGalaxy)):
                src.halfsize = R

        # Use pixelized PSF models for bright sources?
        bright_mods = ((band == 1) and (opt.bright1 is not None))
        if bright_mods:
            set_bright_psf_mods(cat, WISE, T, opt.bright1, band, tile, wcs, sourcerad)

        # We're going to dice the image up into cells for
        # photometry... remember the whole image and initialize
        # whole-image results.
        fullimg = img
        fullinvvar = iv
        fullIV = np.zeros(len(cat))
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
        fitstats = dict([(k, np.zeros(len(cat))) for k in fskeys])

        twcs = ConstantFitsWcs(wcs)

        #sky = estimate_sky(img, iv)
        #print 'Estimated sky', sky
        sky = 0.
        
        imgoffset = 0.
        if opt.sky and opt.nonneg:
            # the non-negative constraint applies to the sky too!
            # Artificially offset the sky value, AND the image pixels.
            offset = 10. * sig1
            if sky < 0:
                offset += -sky

            imgoffset = offset
            sky += offset
            img += offset
            print 'Offsetting image by', offset
            
        tsky = ConstantSky(sky)

        if opt.errfrac > 0:
            pix = (fullimg - imgoffset)
            nz = (fullinvvar > 0)
            iv2 = np.zeros_like(fullinvvar)
            iv2[nz] = 1./(1./fullinvvar[nz] + (pix[nz] * opt.errfrac)**2)
            print 'Increasing error estimate by', opt.errfrac, 'of image flux'
            fullinvvar = iv2

        # cell positions
        XX = np.round(np.linspace(0, W, opt.blocks+1)).astype(int)
        YY = np.round(np.linspace(0, H, opt.blocks+1)).astype(int)

        if ps:

            tag = '%s W%i' % (tile.coadd_id, band)
            
            plt.clf()
            n,b,p = plt.hist((fullimg - imgoffset).ravel(), bins=100,
                             range=(-10*sig1, 20*sig1), log=True,
                             histtype='step', color='b')
            mx = max(n)
            plt.ylim(0.1, mx)
            plt.xlim(-10*sig1, 20*sig1)
            plt.axvline(sky, color='r')
            plt.title('%s: Pixel histogram' % tag)
            ps.savefig()

            if band == 1 and opt.bright1 is not None:
                #from wise_psf import 
                fn = 'wise-psf-avg-pix-bright.fits'
                psfimg = fitsio.read(fn, ext=band-1).astype(np.float32)
                psfimg = np.maximum(0, psfimg)
                psfimg /= psfimg.sum()
                print 'PSF image', psfimg.shape
                print 'PSF image range:', psfimg.min(), psfimg.max()
                ph,pw = psfimg.shape
                pcx,pcy = ph/2, pw/2
                assert(ph == pw)
                phalf = ph/2

                framesfn = os.path.join(thisdir, 'unwise-%s-w%i-frames.fits' % (tile.coadd_id, band))
                F = fits_table(framesfn)
                print 'intfn', F.intfn[0]
                #fwcs = fits_table(F.intfn[
                wisedir = 'wise-frames'
                scanid,frame = F.scan_id[0], F.frame_num[0]
                scangrp = scanid[-2:]
                fn = os.path.join(wisedir, scangrp, scanid, '%03i' % frame, 
                                  '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))
                fwcs = Tan(fn)
                # Keep CD matrix, set CRVAL/CRPIX to star position
                fwcs.set_crpix(pcx+1, pcy+1)
                fwcs.set_imagesize(float(pw), float(ph))

                mag = WISE.get('w%impro' % band)
                I = np.flatnonzero(mag < opt.bright1)
                nm = NanoMaggies.magToNanomaggies(mag)
                print len(I), 'catalog sources brighter than mag', opt.bright1
                mod = np.zeros_like(fullimg)

                MI,MJ,d = match_radec(WISE.ra[I], WISE.dec[I], T.ra, T.dec, 4./3600.,
                                      nearest=True)
                print 'Matched', len(MI), '(', len(np.unique(MI)), 'unique) of the bright WISE sources'

                for i in I:
                    fwcs.set_crval(WISE.ra[i], WISE.dec[i])
                    L=3
                    Yo,Xo,Yi,Xi,rims = resample_with_wcs(wcs, fwcs, [psfimg], L)
                    mod[Yo,Xo] += rims[0] * nm[i]

                    # ok,x,y = wcs.radec2pixelxy(WISE.ra[i], WISE.dec[i])
                    # x -= 1.
                    # y -= 1.
                    # print 'x,y', x,y
                    # print 'mag', mag[i]
                    # print 'nm', nm[i]
                    # ix = int(np.round(x))
                    # iy = int(np.round(y))
                    # dx = x - ix
                    # dy = y - iy
                    # 
                    # x0 = max(0,   ix - phalf)
                    # x1 = min(W-1, ix + phalf)
                    # y0 = max(0,   iy - phalf)
                    # y1 = min(H-1, iy + phalf)
                    # print 'mod x,y range', x0,x1, y0,y1
                    # xx,yy = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
                    # xx = xx.ravel().astype(np.int32)
                    # yy = yy.ravel().astype(np.int32)
                    # rx = xx - (ix-phalf)
                    # ry = yy - (iy-phalf)
                    # print 'resampled x,y range', rx.min(),rx.max(), ry.min(),ry.max()
                    # rpsf = np.zeros(len(rx), np.float32)
                    # rtn = lanczos3_interpolate(rx, ry,
                    #                            np.zeros(len(rx),np.float32)+dx,
                    #                            np.zeros(len(rx),np.float32)+dy,
                    #                            [rpsf], [psfimg])
                    # mod[yy.ravel(),xx.ravel()] += rpsf * nm[i]

                plt.clf()
                plt.imshow(mod, interpolation='nearest', origin='lower',
                           cmap='gray',
                           vmin=-3*sig1, vmax=10*sig1)
                plt.colorbar()
                plt.title('%s: bright star models' % tag)
                ps.savefig()
    
                plt.clf()
                plt.imshow(fullimg - imgoffset - mod, interpolation='nearest', origin='lower',
                           cmap='gray',
                           vmin=-3*sig1, vmax=10*sig1)
                plt.colorbar()
                plt.title('%s: data - bright star models' % tag)
                ps.savefig()

            # plt.clf()
            # plt.imshow(np.log10(pp), interpolation='nearest', origin='lower', cmap='gray',
            #            vmin=0)
            # plt.title('log Per-pixel std')
            # ps.savefig()

            plt.clf()
            plt.imshow(fullimg - imgoffset, interpolation='nearest', origin='lower',
                       cmap='gray',
                       vmin=-3*sig1, vmax=10*sig1)
            ax = plt.axis()
            plt.colorbar()
            for x in XX:
                plt.plot([x,x], [0,H], 'r-', alpha=0.5)
            for y in YY:
                plt.plot([0,W], [y,y], 'r-', alpha=0.5)
                celli = -1
                for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
                    for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
                        celli += 1
                        xc,yc = (xlo+xhi)/2., (ylo+yhi)/2.
                        plt.text(xc, yc, '%i' % celli, color='r')

                        print 'Cell', celli, 'xc,yc', xc,yc
                        print 'W, H', xhi-xlo, yhi-ylo
                        print 'RA,Dec center', wcs.pixelxy2radec(xc+1, yc+1)

            if bright_mods:
                mag = WISE.get('w%impro' % band)
                I = np.flatnonzero(mag < opt.bright1)
                for i in I:
                    ok,x,y = wcs.radec2pixelxy(WISE.ra[i], WISE.dec[i])
                    plt.text(x-1, y-1, '%.1f' % mag[i], color='g')

            plt.axis(ax)
            plt.title('%s: cells' % tag)


            ps.savefig()
            
            print 'Median # ims:', np.median(nims)
            #ppn = pp / np.sqrt(np.maximum(nims - 1, 1))
            # plt.clf()
            # plt.imshow(((fullimg - imgoffset) / ppn),
            #            interpolation='nearest', origin='lower',
            #            cmap='jet', vmin=-3)
            # plt.title('Img / PPstdn')
            # plt.colorbar()
            # ps.savefig()
            # 
            # plt.clf()
            # loghist((fullimg - imgoffset).ravel(), pp.ravel(), bins=200)
            # plt.xlabel('img pix')
            # plt.ylabel('pp std')
            # ps.savefig()
            # 
            # plo,phi = [np.percentile(pp.ravel(), p) for p in [1,99.9]]
            # print 'pp percentiles:', plo,phi
            # 
            # print 'pp max', pp.max()
            # phi = pp.max()
            # 
            # plt.clf()
            # loghist(np.log10(np.clip((fullimg - imgoffset).ravel(), 0.1*sig1, 1e6*sig1)),
            #         np.log10(np.clip(pp.ravel(), plo, phi)), bins=200)
            # plt.xlabel('log img pix')
            # plt.ylabel('log pp std')
            # ps.savefig()
            # 
            # pnlo,pnhi = [np.percentile(ppn.ravel(), p) for p in [1,99.9]]
            # print 'ppn percentiles:', pnlo,pnhi
            # print 'ppn max', ppn.max()
            # pnhi = ppn.max()
            # 
            # #iv2 = 1./(1./iv + np.maximum(0, ppn - sig1)**2)
            # 
            # plt.clf()
            # loghist(np.log10(np.clip((fullimg - imgoffset).ravel(), 0.1*sig1, 1e6*sig1)),
            #         np.log10(np.clip(ppn.ravel(), pnlo, pnhi)), bins=200)
            # ax = plt.axis()
            # plt.axhline(np.log10(sig1), color='g')
            # x0,x1 = plt.xlim()
            # x = np.linspace(x0, x1, 500)
            # plt.plot(x, x-1, 'b-')
            # plt.plot(x, x-2, 'b-')
            # plt.plot(x, x/2., 'w-')
            # plt.axis(ax)
            # plt.xlabel('log img pix')
            # plt.ylabel('log ppn std')
            # ps.savefig()

            # plt.clf()
            # loghist(np.log10(np.clip((fullimg - imgoffset).ravel(), 0.1*sig1, 1e6*sig1)),
            #         np.log10(np.clip(np.sqrt(1./iv2.ravel()), pnlo, pnhi)), bins=200)
            # ax = plt.axis()
            # plt.axhline(np.log10(sig1), color='g')
            # x0,x1 = plt.xlim()
            # x = np.linspace(x0, x1, 500)
            # plt.plot(x, x-1, 'b-')
            # plt.plot(x, x-2, 'b-')
            # plt.plot(x, x/2., 'w-')
            # plt.axis(ax)
            # plt.xlabel('log img pix')
            # plt.ylabel('log sqrt(iv2)')
            # ps.savefig()
            # 
            # 
            # plt.clf()
            # loghist(np.log10(np.clip((fullimg - imgoffset).ravel(), 0.1*sig1, 1e6*sig1)),
            #         np.log10(np.clip(np.sqrt(1./iv3.ravel()), pnlo, pnhi)), bins=200)
            # ax = plt.axis()
            # plt.axhline(np.log10(sig1), color='g')
            # x0,x1 = plt.xlim()
            # x = np.linspace(x0, x1, 500)
            # plt.plot(x, x-1, 'b-')
            # plt.plot(x, x-2, 'b-')
            # plt.plot(x, x/2., 'w-')
            # plt.axis(ax)
            # plt.xlabel('log img pix')
            # plt.ylabel('log sqrt(iv3)')
            # ps.savefig()
            # 
            # 
            # plt.clf()
            # loghist(np.log10(np.clip((fullimg - imgoffset).ravel(), 0.1*sig1, 1e6*sig1)),
            #         np.log10(np.clip(np.sqrt(1./iv4.ravel()), pnlo, pnhi)), bins=200)
            # ax = plt.axis()
            # plt.axhline(np.log10(sig1), color='g')
            # x0,x1 = plt.xlim()
            # x = np.linspace(x0, x1, 500)
            # plt.plot(x, x-1, 'b-')
            # plt.plot(x, x-2, 'b-')
            # plt.plot(x, x/2., 'w-')
            # plt.axis(ax)
            # plt.xlabel('log img pix')
            # plt.ylabel('log sqrt(iv4)')
            # ps.savefig()


            # xx,yy = [],[]
            # cc = []
            # for src in cat:
            #     c = 'b'
            #     if src.getBrightness().getBand(wanyband) > defaultflux:
            #         c = 'g'
            #     x,y = twcs.positionToPixel(src.getPosition())
            #     xx.append(x)
            #     yy.append(y)
            #     cc.append(c)
            # p1 = plt.scatter(xx, yy, c=cc, marker='+')

            # notI = np.flatnonzero(wf <= defaultflux)
            # plt.plot(T.x[notI], T.y[notI], 'b+')
            # plt.plot(T.x[I], T.y[I], 'g+')
            # plt.axis(ax)
            # ps.savefig()
            # plt.plot(UW.x, UW.y, 'r+')
            # plt.axis(ax)
            # ps.savefig()

        if savepickle:
            mods = []
            cats = []

        celli = -1
        for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
            for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
                celli += 1

                if len(opt.cells) and not celli in opt.cells:
                    print 'Skipping cell', celli
                    continue

                print
                print 'Cell', celli, 'of', (opt.blocks**2), 'for', tile.coadd_id, 'band', wband

                imargin = 12
                # SDSS and WISE source margins beyond the image margins ( + source radii )
                smargin = 1
                wmargin = 1

                # image region: [ix0,ix1)
                ix0 = max(0, xlo - imargin)
                ix1 = min(W, xhi + imargin)
                iy0 = max(0, ylo - imargin)
                iy1 = min(H, yhi + imargin)
                slc = (slice(iy0, iy1), slice(ix0, ix1))
                print 'Image ROI', ix0, ix1, iy0, iy1
                img    = fullimg   [slc]
                invvar = fullinvvar[slc]
                twcs.setX0Y0(ix0, iy0)

                tim = Image(data=img, invvar=invvar, psf=psf, wcs=twcs,
                            sky=tsky, photocal=LinearPhotoCal(1., band=wanyband),
                            name='Coadd %s W%i (%i,%i)' % (tile.coadd_id, band, xi,yi),
                            domask=False)

                # Relevant SDSS sources:
                m = smargin + sourcerad
                I = np.flatnonzero(((T.x+m) >= (ix0-0.5)) * ((T.x-m) < (ix1-0.5)) *
                                   ((T.y+m) >= (iy0-0.5)) * ((T.y-m) < (iy1-0.5)))
                inbox = ((T.x[I] >= (xlo-0.5)) * (T.x[I] < (xhi-0.5)) *
                         (T.y[I] >= (ylo-0.5)) * (T.y[I] < (yhi-0.5)))
                # Inside this cell
                srci = I[inbox]
                # In the margin
                margi = I[np.logical_not(inbox)]

                # sources in the ROI box
                subcat = [cat[i] for i in srci]

                # include *copies* of sources in the margins
                # (that way we automatically don't save the results)
                subcat.extend([cat[i].copy() for i in margi])
                assert(len(subcat) == len(I))

                # add WISE-only sources in the expanded region
                m = wmargin + UW.rad
                J = np.flatnonzero(((UW.x+m) >= (ix0-0.5)) * ((UW.x-m) < (ix1-0.5)) *
                                   ((UW.y+m) >= (iy0-0.5)) * ((UW.y-m) < (iy1-0.5)))

                if opt.savewise:
                    jinbox = ((UW.x[J] >= (xlo-0.5)) * (UW.x[J] < (xhi-0.5)) *
                              (UW.y[J] >= (ylo-0.5)) * (UW.y[J] < (yhi-0.5)))
                    uwcat = []
                wnm = UW.get('w%inm' % band)
                nomag = 0
                for ji,j in enumerate(J):
                    if not np.isfinite(wnm[j]):
                        nomag += 1
                        continue
                    ptsrc = PointSource(RaDecPos(UW.ra[j], UW.dec[j]),
                                              NanoMaggies(**{wanyband: wnm[j]}))
                    ptsrc.radius = UW.rad[j]
                    subcat.append(ptsrc)
                    if opt.savewise:
                        if jinbox[ji]:
                            uwcat.append((j, ptsrc))
                        
                print 'WISE-only:', nomag, 'of', len(J), 'had invalid mags'
                print 'Sources:', len(srci), 'in the box,', len(I)-len(srci), 'in the margins, and', len(J), 'WISE-only'

                # if ps:
                #     plt.clf()
                #     plt.imshow(img - imgoffset, interpolation='nearest', origin='lower',
                #                cmap='gray', vmin=-3*sig1, vmax=10*sig1)
                #     plt.colorbar()
                #     xx,yy = [],[]
                #     for src in subcat:
                #         x,y = twcs.positionToPixel(src.getPosition())
                #         xx.append(x)
                #         yy.append(y)
                #     p1 = plt.plot(xx[:len(srci)], yy[:len(srci)], 'b+')
                #     p2 = plt.plot(xx[len(srci):len(I)], yy[len(srci):len(I)], 'g+')
                #     p3 = plt.plot(xx[len(I):], yy[len(I):], 'r+')
                #     p4 = plt.plot(UW.x[np.logical_not(np.isfinite(wnm[J]))],
                #                   UW.y[np.logical_not(np.isfinite(wnm[J]))],
                #                   'y+')
                #     ps.savefig()
                # 
                #     # plt.clf()
                #     # plt.imshow(invvar, interpolation='nearest', origin='lower')
                #     # plt.colorbar()
                #     # plt.title('invvar')
                #     # ps.savefig()

                print 'Creating a Tractor with image', tim.shape, 'and', len(subcat), 'sources'
                tractor = Tractor([tim], subcat)

                print 'Running forced photometry...'
                t0 = Time()
                tractor.freezeParamsRecursive('*')

                if opt.sky:
                    tractor.thawPathsTo('sky')
                    print 'Initial sky values:'
                    for tim in tractor.getImages():
                        print tim.getSky()

                tractor.thawPathsTo(wanyband)

                wantims = (savepickle or opt.pickle2 or (ps is not None))

                R = tractor.optimize_forced_photometry(
                    minsb=minsb, mindlnp=1., sky=opt.sky, minFlux=None,
                    fitstats=True, variance=True, shared_params=False,
                    use_ceres=opt.ceres, BW=opt.ceresblock, BH=opt.ceresblock,
                    wantims=wantims, nonneg=opt.nonneg, negfluxval=0.1*sig1)
                print 'That took', Time()-t0

                if wantims:
                    ims0 = R.ims0
                    ims1 = R.ims1
                IV,fs = R.IV, R.fitstats

                if opt.sky:
                    print 'Fit sky values:'
                    for tim in tractor.getImages():
                        print tim.getSky()

                if opt.savewise:
                    for (j,src) in uwcat:
                        fitwiseflux[band][j] = src.getBrightness().getBand(wanyband)

                if ps:

                    tag = '%s W%i cell %i/%i' % (tile.coadd_id, band, celli, opt.blocks**2)

                    (dat,mod,ie,chi,roi) = ims1[0]

                    plt.clf()
                    plt.imshow(dat - imgoffset, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-3*sig1, vmax=10*sig1)
                    plt.colorbar()
                    plt.title('%s: data' % tag)
                    ps.savefig()

                    plt.clf()
                    plt.imshow(1./ie, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=0, vmax=10*sig1)
                    plt.colorbar()
                    plt.title('%s: sigma' % tag)
                    ps.savefig()

                    plt.clf()
                    plt.imshow(mod - imgoffset, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-3*sig1, vmax=10*sig1)
                    plt.colorbar()
                    plt.title('%s: model' % tag)
                    ps.savefig()

                    plt.clf()
                    plt.imshow(chi, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-5, vmax=+5)
                    plt.colorbar()
                    plt.title('%s: chi' % tag)
                    ps.savefig()

                    # plt.clf()
                    # plt.imshow(np.round(chi), interpolation='nearest', origin='lower',
                    #            cmap='jet', vmin=-5, vmax=+5)
                    # plt.colorbar()
                    # plt.title('Chi')
                    # ps.savefig()

                    plt.clf()
                    plt.imshow(chi, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-20, vmax=+20)
                    plt.colorbar()
                    plt.title('%s: chi 2' % tag)
                    ps.savefig()

                    plt.clf()
                    n,b,p = plt.hist(chi.ravel(), bins=100,
                                     range=(-10, 10), log=True,
                                     histtype='step', color='b')
                    mx = max(n)
                    plt.ylim(0.1, mx)
                    plt.axvline(0, color='r')
                    plt.title('%s: chi' % tag)
                    ps.savefig()

                    # fn = ps.basefn + '-chi.fits'
                    # fitsio.write(fn, chi, clobber=True)
                    # print 'Wrote', fn

                if savepickle:
                    # FIXME -- imgoffset
                    if ims1 is None:
                        mod = None
                    else:
                        im,mod,ie,chi,roi = ims1[0]
                    mods.append(mod)
                    cats.append((
                        srci, margi, UW.x[J], UW.y[J],
                        T.x[srci], T.y[srci], T.x[margi], T.y[margi],
                        [src.copy() for src in cat],
                        [src.copy() for src in subcat]))

                if opt.pickle2:
                    fn = opt.output % (tile.coadd_id)
                    fn = fn.replace('.fits','-cell%02i.pickle' % celli)
                    pickle_to_file((ims0, ims1, cat, subcat), fn)
                    print 'Pickled', fn
                    

                if len(srci):
                    T.cell[srci] = celli
                    T.cell_x0[srci] = ix0
                    T.cell_x1[srci] = ix1
                    T.cell_y0[srci] = iy0
                    T.cell_y1[srci] = iy1
                    # Save fit stats
                    fullIV[srci] = IV[:len(srci)]
                    for k in fskeys:
                        x = getattr(fs, k)
                        fitstats[k][srci] = np.array(x)

                cpu0 = tb0.meas[0]
                t = Time()
                cpu = t.meas[0]
                dcpu = (cpu.cpu - cpu0.cpu)
                print 'So far:', Time()-tb0, '-> predict CPU time', (dcpu * (opt.blocks**2) / float(celli+1))

        if bright_mods:
            # Reset pixelized models
            for src in cat:
                if isinstance(src, BrightPointSource):
                    src.pixmodel = None

        nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
        nm_ivar = fullIV
        T.set(wband + '_nanomaggies', nm.astype(np.float32))
        T.set(wband + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
        dnm = np.zeros(len(nm_ivar), np.float32)
        okiv = (nm_ivar > 0)
        dnm[okiv] = (1./np.sqrt(nm_ivar[okiv])).astype(np.float32)
        okflux = (nm > 0)
        mag = np.zeros(len(nm), np.float32)
        mag[okflux] = (NanoMaggies.nanomaggiesToMag(nm[okflux])).astype(np.float32)
        dmag = np.zeros(len(nm), np.float32)
        ok = (okiv * okflux)
        dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dnm[ok] / nm[ok])).astype(np.float32)

        mag[np.logical_not(okflux)] = np.nan
        dmag[np.logical_not(ok)] = np.nan
        
        T.set(wband + '_mag', mag)
        T.set(wband + '_mag_err', dmag)
        for k in fskeys:
            T.set(wband + '_' + k, fitstats[k].astype(np.float32))

        if ps:
            I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 4./3600.)

            plt.clf()
            lo,hi = 10,25
            cathi = 18
            loghist(WISE.get('w%impro'%band)[I], T.get(wband+'_mag')[J],
                    range=((lo,cathi),(lo,cathi)), bins=200)
            plt.xlabel('WISE W1 mag')
            plt.ylabel('Tractor W1 mag')
            plt.title('WISE catalog vs Tractor forced photometry')
            plt.axis([cathi,lo,cathi,lo])
            ps.savefig()

        print 'Tile', tile.coadd_id, 'band', wband, 'took', Time()-tb0

    T.cut(inbounds)

    T.delete_column('psfflux')
    T.delete_column('cmodelflux')
    T.delete_column('devflux')
    T.delete_column('expflux')
    T.treated_as_pointsource = T.treated_as_pointsource.astype(np.uint8)

    hdr = fitsio.FITSHDR()
    hdr.add_record(dict(name='SEQ_VER', value=version['Revision'],
                        comment='SVN revision'))
    hdr.add_record(dict(name='SEQ_URL', value=version['URL'], comment='SVN URL'))
    hdr.add_record(dict(name='SEQ_DATE', value=datetime.datetime.now().isoformat(),
                        comment='forced phot run time'))
    hdr.add_record(dict(name='SEQ_NNEG', value=opt.nonneg, comment='non-negative?'))
    hdr.add_record(dict(name='SEQ_SKY', value=opt.sky, comment='fit sky?'))
    for b in bands:
        minsig = getattr(opt, 'minsig%i' % band)
        hdr.add_record(dict(name='SEQ_MNS%i' % band, value=minsig,
                            comment='min surf brightness in sig, band %i' % band))
    hdr.add_record(dict(name='SEQ_BL', value=opt.blocks, comment='image blocks'))
    hdr.add_record(dict(name='SEQ_CERE', value=opt.ceres, comment='use Ceres?'))
    hdr.add_record(dict(name='SEQ_ERRF', value=opt.errfrac, comment='error flux fraction'))
    if opt.ceres:
        hdr.add_record(dict(name='SEQ_CEBL', value=opt.ceresblock,
                        comment='Ceres blocksize'))
    
    T.writeto(outfn, header=hdr)
    print 'Wrote', outfn

    if opt.savewise:
        for band in bands:
            UW.set('fit_flux_w%i' % band, fitwiseflux[band])
        UW.writeto(savewise_outfn)
        print 'Wrote', savewise_outfn

    if savepickle:
        fn = opt.output % (tile.coadd_id)
        fn = fn.replace('.fits','.pickle')
        pickle_to_file((mods, cats, T, sourcerad), fn)
        print 'Pickled', fn

    print 'Tile', tile.coadd_id, 'took', Time()-tt0


def summary(A, opt, ps):
    plt.clf()
    missing = []
    for i in range(len(A)):
        r,d = A.ra[i], A.dec[i]
        dd = 1024 * 2.75 / 3600.
        dr = dd / np.cos(np.deg2rad(d))
        outfn = opt.output % (A.coadd_id[i])
        rr,dd = [r-dr,r-dr,r+dr,r+dr,r-dr], [d-dd,d+dd,d+dd,d-dd,d-dd]
        print 'Looking for', outfn
        if not os.path.exists(outfn):
            missing.append((i,rr,dd,r,d))
        plt.plot(rr, dd, 'k-')
    for i,rr,dd,r,d in missing:
        plt.plot(rr, dd, 'r-')
        plt.text(r, d, '%i' % i, rotation=90, color='b', va='center', ha='center')
    plt.title('missing tiles')
    plt.axis([118, 212, 44,61])
    ps.savefig()

    print 'Missing tiles:', [m[0] for m in missing]

    rdfn = 'rd.fits'
    if not os.path.exists(rdfn):
        fns = glob(os.path.join(tempoutdir, 'photoobjs-*.fits'))
        fns.sort()
        TT = []
        for fn in fns:
            T = fits_table(fn, columns=['ra','dec'])
            print len(T), 'from', fn
            TT.append(T)
        T = merge_tables(TT)
        print 'Total of', len(T)
        T.writeto(rdfn)
    else:
        T = fits_table(rdfn)
        print 'Got', len(T), 'from', rdfn
    
    plt.clf()
    loghist(T.ra, T.dec, 500, range=((118,212),(44,61)))
    plt.xlabel('RA')
    plt.ylabel('Dec')
    ps.savefig()

    ax = plt.axis()
    for i in range(len(A)):
        r,d = A.ra[i], A.dec[i]
        dd = 1024 * 2.75 / 3600.
        dr = dd / np.cos(np.deg2rad(d))
        plt.plot([r-dr,r-dr,r+dr,r+dr,r-dr], [d-dd,d+dd,d+dd,d-dd,d-dd], 'r-')
    plt.axis(ax)
    ps.savefig()

def finish(T, opt, args, ps):
    # Find all *-phot.fits outputs
    # Determine which photoObj files are involved
    # Collate and resolve objs measured in multiple tiles
    # Expand into photoObj-parallel files
    if len(args):
        fns = args
    else:
        fns = glob(os.path.join(outdir, 'phot-????????.fits'))
        fns.sort()
        print 'Found', len(fns), 'photometry output files'
    flats = []
    fieldmap = {}
    for ifn,fn in enumerate(fns):
        print 'Reading', (ifn+1), 'of', len(fns), fn
        cols = ['ra','dec',
                'objid', 'index', 'x','y', 
                'treated_as_pointsource', 'coadd_id']
        for band in opt.bands:
            for k in ['nanomaggies', 'nanomaggies_ivar', 'mag', 'mag_err',
                      'prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
                cols.append('w%i_%s' % (band, k))
        rcfcols = ['run','camcol','field','id',]
        try:
            T = fits_table(fn, columns=cols + rcfcols)
        except:
            print 'Run,camcol,field columns not found; reading photoobjs file to get them.'
            print fn
            T = fits_table(fn, columns=cols)
            print 'Got', len(T), 'from', fn
            pfn = fn.replace('phot-', 'photoobjs-').replace(outdir, tempoutdir)
            print 'Reading', pfn
            P = fits_table(pfn, columns=rcfcols + ['objid'])
            print 'Got', len(P), 'from', pfn
            print 'Applying objid map...'
            objidmap = dict([(oid,i) for i,oid in enumerate(P.objid)])
            I = np.array([objidmap[oid] for oid in T.objid])
            P.cut(I)
            assert(len(P) == len(T))
            assert(np.all(P.objid == T.objid))
            for k in rcfcols:
                T.set(k, P.get(k))
            
        print 'Read', len(T), 'entries'

        if opt.flat is not None:
            flats.append(T)
            continue

        rcf = np.unique(zip(T.run, T.camcol, T.field))
        for run,camcol,field in rcf:
            if not (run,camcol,field) in fieldmap:
                fieldmap[(run,camcol,field)] = []
            Tsub = T[(T.run == run) * (T.camcol == camcol) * (T.field == field)]
            #print len(Tsub), 'in', (run,camcol,field)
            for col in ['run','camcol','field']:
                Tsub.delete_column(col)
            fieldmap[(run,camcol,field)].append(Tsub)

    # WISE coadd tile CRPIX-1 (x,y in the phot-*.fits files are 0-indexed)
    # (and x,y are based on the first-band (W1 usually) WCS)
    cx,cy = 1023.5, 1023.5

    if opt.flat is not None:
        F = merge_tables(flats)
        print 'Total of', len(F), 'measurements'
        r2 = (F.x - cx)**2 + (F.y - cy)**2
        I,J,d = match_radec(F.ra, F.dec, F.ra, F.dec, 1e-6, notself=True)
        print 'Matched', len(I), 'duplicates'
        keep = np.ones(len(F), bool)
        keep[I] = False
        keep[J] = False
        keep[np.where(r2[I] < r2[J], I, J)] = True
        F.cut(keep)
        print 'Cut to', len(F)
        F.delete_column('index')
        F.delete_column('x')
        F.delete_column('y')
        F.writeto(opt.flat)
        return

    pfn = 'photoobj-lengths.pickle'
    if os.path.exists(pfn):
        print 'Reading photoObj lengths from', pfn
        pobjlengths = unpickle_from_file(pfn)
    else:
        pobjlengths = {}

    keys = fieldmap.keys()
    keys.sort()
    #for i,((run,camcol,field),TT) in enumerate(fieldmap.items()):
    for i,(run,camcol,field) in enumerate(keys):
        TT = fieldmap.get((run,camcol,field))
        print
        print (i+1), 'of', len(fieldmap), ': R,C,F', (run,camcol,field)
        print len(TT), 'tiles for', (run,camcol,field)

        # HACK
        rr = '301'

        key = (rr,run,camcol,field)
        N = pobjlengths.get(key, None)
        if N is None:
            pofn = get_photoobj_filename(rr, run,camcol,field)
            F = fitsio.FITS(pofn)
            N = F[1].get_nrows()
            pobjlengths[key] = N
        if i % 1000 == 0:
            pickle_to_file(pobjlengths, pfn)
            print 'Wrote', pfn

        P = fits_table()
        P.has_wise_phot = np.zeros(N, bool)
        if len(TT) > 1:
            # Resolve duplicate measurements (in multiple tiles)
            # based on || (x,y) - center ||^2
            P.R2 = np.empty(N, np.float32)
            P.R2[:] = 1e9
        for T in TT:
            coadd = T.coadd_id[0]
            if len(TT) > 1:
                I = T.id - 1
                R2 = (T.x - cx)**2 + (T.y - cy)**2
                J = (R2 < P.R2[I])
                I = I[J]
                P.R2[I] = R2[J].astype(np.float32)
                #print len(I), 'are closest'
                T.cut(J)
            print '  ', len(T), 'from', coadd
            if len(T) == 0:
                continue
            I = T.id - 1
            P.has_wise_phot[I] = True
            pcols = P.get_columns()
            for col in T.get_columns():
                if col in pcols:
                    pval = P.get(col)
                    print '  ', col, pval.dtype
                    pval[I] = (T.get(col)).astype(pval.dtype)
                else:
                    tval = T.get(col)
                    X = np.zeros(N, tval.dtype)
                    X[I] = tval
                    P.set(col, X)

        P.delete_column('index')
        P.delete_column('id')
        if len(TT) > 1:
            P.delete_column('R2')

        myoutdir = os.path.join(pobjoutdir, rr, '%i'%run, '%i'%camcol)
        if not os.path.exists(myoutdir):
            os.makedirs(myoutdir)
        outfn = os.path.join(myoutdir, 'photoWiseForced-%06i-%i-%04i.fits' % (run, camcol, field))
        P.writeto(outfn)
        print 'Wrote', outfn
    pickle_to_file(pobjlengths, pfn)
    print 'Wrote', pfn

def main():
    import optparse

    global outdir
    global tempoutdir
    global pobjoutdir
    global tiledir

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--minsig1', dest='minsig1', default=0.1, type=float)
    parser.add_option('--minsig2', dest='minsig2', default=0.1, type=float)
    parser.add_option('--minsig3', dest='minsig3', default=0.1, type=float)
    parser.add_option('--minsig4', dest='minsig4', default=0.1, type=float)
    parser.add_option('--blocks', dest='blocks', default=10, type=int,
                      help='NxN number of blocks to cut the image into')
    parser.add_option('-d', dest='outdir', default=None)
    parser.add_option('--pobj', dest='pobjdir', default=None,
                      help='Output directory for photoObj-parallels')
    parser.add_option('-o', dest='output', default=None)
    parser.add_option('-b', dest='bands', action='append', type=int, default=[],
                      help='Add WISE band (default: 1,2)')

    parser.add_option('--tiledir', type=str, help='Set input wise-coadds/ dir')

    parser.add_option('--photoobjs-only', dest='photoObjsOnly',
                      action='store_true', default=False,
                      help='Ensure photoobjs file exists and then quit?')

    parser.add_option('-p', dest='pickle', default=False, action='store_true',
                      help='Save .pickle file for debugging purposes')
    parser.add_option('--pp', dest='pickle2', default=False, action='store_true',
                      help='Save .pickle file for each cell?')
    parser.add_option('--plots', dest='plots', default=False, action='store_true')

    parser.add_option('--plotbase', dest='plotbase', help='Base filename for plots')

    parser.add_option('--finish', dest='finish', default=False, action='store_true')

    #parser.add_option('--extra-dir', dest='extradir', default=None,
    #                  help='With --finish, also read parallel files from this directory')

    parser.add_option('--flat', dest='flat', type='str', default=None,
                      help='Just write a flat-file of (deduplicated) results, not photoObj-parallels')

    parser.add_option('--summary', dest='summary', default=False, action='store_true')

    parser.add_option('--cell', dest='cells', default=[], type=int, action='append',
                      help='Just run certain cells?')

    parser.add_option('--ceres', dest='ceres', action='store_true', default=False,
                      help='Use Ceres Solver?')
    parser.add_option('--ceres-block', '-B', dest='ceresblock', type=int,
                      help='Ceres image block size (default: 50)')
    parser.add_option('--nonneg', dest='nonneg', action='store_true', default=False,
                      help='With ceres, enable non-negative fluxes?')

    parser.add_option('--minrad', dest='minradius', type=int, default=2,
                      help='Minimum radius, in pixels, for evaluating source models; default %default')

    parser.add_option('--sky', dest='sky', action='store_true', default=False,
                      help='Fit sky level also?')

    parser.add_option('--save-wise', dest='savewise', action='store_true', default=False,
                      help='Save WISE catalog source fits also?')
    parser.add_option('--save-wise-out', dest='save_wise_output', default=None)

    parser.add_option('--dataset', dest='dataset', default='sequels',
                      help='Dataset (region of sky) to work on')

    parser.add_option('--errfrac', dest='errfrac', type=float,
                      help='Add this fraction of flux to the error model.')

    parser.add_option('--bright1', dest='bright1', type=float, default=None,
                      help='Subtract WISE model PSF for stars brighter than this in W1')

    parser.add_option('-v', dest='verbose', default=False, action='store_true')

    opt,args = parser.parse_args()

    if opt.tiledir:
        tiledir = opt.tiledir

    if len(opt.bands) == 0:
        opt.bands = [1,2]

    # Allow specifying bands like "123"
    bb = []
    for band in opt.bands:
        for s in str(band):
            bb.append(int(s))
    opt.bands = bb
    print 'Bands', opt.bands

    lvl = logging.INFO
    if opt.verbose:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    # sequels-atlas.fits: written by wise-coadd.py
    fn = '%s-atlas.fits' % opt.dataset
    print 'Reading', fn
    T = fits_table(fn)

    if opt.plotbase is None:
        opt.plotbase = opt.dataset + '-phot'
    ps = PlotSequence(opt.plotbase)

    outdir     = outdir     % opt.dataset
    tempoutdir = tempoutdir % opt.dataset
    pobjoutdir = pobjoutdir % opt.dataset

    if opt.pobjdir is not None:
        pobjoutdir = opt.pobjdir

    if opt.outdir is not None:
        outdir = opt.outdir
    if opt.output is None:
        opt.output = os.path.join(outdir, 'phot-%s.fits')
    if opt.save_wise_output is None:
        opt.save_wise_output = opt.output.replace('phot-', 'phot-wise-')

    if opt.summary:
        summary(T, opt, ps)
        sys.exit(0)

    if opt.finish:
        finish(T, opt, args, ps)
        sys.exit(0)

    tiles = []
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        tiles.append(arr)
    else:
        if len(args) == 0:
            tiles.append(0)
        else:
            for a in args:
                if '-' in a:
                    aa = a.split('-')
                    if len(aa) != 2:
                        print 'With arg containing a dash, expect two parts'
                        print aa
                        sys.exit(-1)
                    start = int(aa[0])
                    end = int(aa[1])
                    for i in range(start, end+1):
                        tiles.append(i)
                else:
                    tiles.append(int(a))

    for i in tiles:
        if opt.plots:
            plot = ps
        else:
            plot = None

        # W3,W4
        if i >= 1000:
            i -= 1000
            opt.bands = [3,4]
            outdir = 'sequels-phot-w34'
            if opt.output is None:
                opt.output = os.path.join(outdir, 'phot-%s.fits')
            print 'Changed bands to', opt.bands, 'and output dir to', outdir
            print 'Output file pattern', opt.output

        one_tile(T[i], opt, opt.pickle, plot)

if __name__ == '__main__':
    main()

