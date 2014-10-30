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

'''
See the file "README" about the output products.
'''

'''
Relevant files/directories are:

DATASET-atlas.fits
  the coadd tiles to process

(--tiledir)
tiledir ("wise-coadds")/xxx/xxxx[pm]xxx/unwise-xxxx[pm]xxx-wW-img-m.fits
                             and {invvar,std,n}-m.fits
  WISE coadd tiles

tempoutdir ("DATASET-phot-temp")/wise-sources-TILE.fits
  WISE catalog sources
  
(-d)
outdir ("DATASET-phot")/phot-TILE.fits
  phot-TILE.fits: WISE forced photometry for photoobjs-TILE.fits objects

'''

if __name__ == '__main__':
    d = os.environ.get('PBS_O_WORKDIR')
    if d is not None:
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
from astrometry.util.ttime import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

from wise.unwise import *
from wise.wisecat import wise_catalog_radecbox
from wise.allwisecat import allwise_catalog_radecbox

import logging

if __name__ == '__main__':
    tiledir = 'unwise-coadds'

    outdir = '%s-phot'
    tempoutdir = '%s-phot-temp'

    Time.add_measurement(MemMeas)

def read_wise_sources(wfn, r0,r1,d0,d1, extracols=[], allwise=False):
    print 'looking for', wfn
    if os.path.exists(wfn):
        WISE = fits_table(wfn)
        print 'Read', len(WISE), 'WISE sources nearby'
    else:
        cols = ['ra','dec'] + ['w%impro'%band for band in [1,2,3,4]]
        cols += extracols
        
        print 'wise_catalog_radecbox:', r0,r1,d0,d1
        if r1 - r0 > 180:
            # assume wrap-around; glue together 0-r0 and r1-360
            Wa = wise_catalog_radecbox(0., r0, d0, d1, cols=cols)
            Wb = wise_catalog_radecbox(r1, 360., d0, d1, cols=cols)
            WISE = merge_tables([Wa, Wb])
        else:
            if allwise:
                WISE = allwise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
            else:
                WISE = wise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
        WISE.writeto(wfn)
        print 'Found', len(WISE), 'WISE sources nearby'
    return WISE

def one_tile(tile, opt, savepickle, ps, tiles, tiledir, tempoutdir, T=None, hdr=None):

    bands = opt.bands
    outfn = opt.output % (tile.coadd_id)
    savewise_outfn = opt.save_wise_output % (tile.coadd_id)

    sband = 'r'
    bandnum = 'ugriz'.index(sband)

    tt0 = Time()
    print
    print 'Coadd tile', tile.coadd_id

    thisdir = get_unwise_tile_dir(tiledir, tile.coadd_id)
    fn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits' % (tile.coadd_id, bands[0]))
    if os.path.exists(fn):
        print 'Reading', fn
        wcs = Tan(fn)
    else:
        print 'File', fn, 'does not exist; faking WCS'
        from unwise_coadd import get_coadd_tile_wcs
        wcs = get_coadd_tile_wcs(tile.ra, tile.dec)

    r0,r1,d0,d1 = wcs.radec_bounds()
    print 'RA,Dec bounds:', r0,r1,d0,d1
    H,W = wcs.get_height(), wcs.get_width()

    if T is None:
        
        T = merge_tables([fits_table(fn, columns=[x.upper() for x in [
            'chi2_psf_r', 'chi2_model_r', 'mag_psf_r',
            #'mag_disk_r',
            'mag_spheroid_r', 'spheroid_reff_world',
            'spheroid_aspect_world', 'spheroid_theta_world',
            #'disk_scale_world', 'disk_aspect_world',
            #'disk_theta_world',
            'alphamodel_j2000', 'deltamodel_j2000']],
                                     column_map=dict(CHI2_PSF_R='chi2_psf',
                                                     CHI2_MODEL_R='chi2_model',
                                                     MAG_PSF_R='mag_psf',
                                                     MAG_SPHEROID_R='mag_spheroid_r',
                                                     ))
                          for fn in
                          ['DES_SNX3cat_000001.fits', 'DES_SNX3cat_000002.fits']]
                          )
        T.mag_disk = np.zeros(len(T), np.float32) + 99.
        print 'Read total of', len(T), 'DES sources'
        ok,T.x,T.y = wcs.radec2pixelxy(T.alphamodel_j2000, T.deltamodel_j2000)
        margin = int(60. * wcs.pixel_scale())
        print 'Margin:', margin, 'pixels'
        T.cut((T.x > -margin) * (T.x < (W+margin)) *
              (T.y > -margin) * (T.y < (H+margin)))
        print 'Cut to', len(T), 'in bounds'
        if opt.photoObjsOnly:
            return
    print len(T), 'objects'
    if len(T) == 0:
        return

    defaultflux = 1.

    # hack
    T.x = (T.x - 1.).astype(np.float32)
    T.y = (T.y - 1.).astype(np.float32)
    margin = 20.
    I = np.flatnonzero((T.x >= -margin) * (T.x < W+margin) *
                       (T.y >= -margin) * (T.y < H+margin))
    T.cut(I)
    print 'Cut to margins: N objects:', len(T)
    if len(T) == 0:
        return

    wanyband = wband = 'w'

    classmap = {}

    print 'Creating tractor sources...'
    cat = get_se_modelfit_cat(T, bands=[wanyband])
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
            sourcerad[i] = (src.nre * src.shape.re / pixscale)
        elif isinstance(src, FixedCompositeGalaxy):
            sourcerad[i] = max(src.shapeExp.re * ExpGalaxy.nre,
                               src.shapeDev.re * DevGalaxy.nre) / pixscale
    print 'sourcerad range:', min(sourcerad), max(sourcerad)

    # Find WISE-only catalog sources
    wfn = os.path.join(tempoutdir, 'wise-sources-%s.fits' % (tile.coadd_id))
    WISE = read_wise_sources(wfn, r0,r1,d0,d1, allwise=True)

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
        if len(I) == 0:
            continue
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

    inbounds = np.flatnonzero((T.x >= -0.5) * (T.x < W-0.5) *
                              (T.y >= -0.5) * (T.y < H-0.5))

    print 'Before looping over bands:', Time()-tt0
   
    for band in bands:
        tb0 = Time()
        print
        print 'Coadd tile', tile.coadd_id
        print 'Band', band
        wband = 'w%i' % band

        imfn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits'    % (tile.coadd_id, band))
        ivfn = os.path.join(thisdir, 'unwise-%s-w%i-invvar-m.fits.gz' % (tile.coadd_id, band))
        ppfn = os.path.join(thisdir, 'unwise-%s-w%i-std-m.fits.gz'    % (tile.coadd_id, band))
        nifn = os.path.join(thisdir, 'unwise-%s-w%i-n-m.fits.gz'      % (tile.coadd_id, band))

        print 'Reading', imfn
        wcs = Tan(imfn)
        r0,r1,d0,d1 = wcs.radec_bounds()
        print 'RA,Dec bounds:', r0,r1,d0,d1
        ra,dec = wcs.radec_center()
        print 'Center:', ra,dec
        img = fitsio.read(imfn)
        print 'Reading', ivfn
        invvar = fitsio.read(ivfn)
        print 'Reading', ppfn
        pp = fitsio.read(ppfn)
        print 'Reading', nifn
        nims = fitsio.read(nifn)
        print 'Median # ims:', np.median(nims)

        good = (nims > 0)
        invvar[np.logical_not(good)] = 0.

        sig1 = 1./np.sqrt(np.median(invvar[good]))
        minsig = getattr(opt, 'minsig%i' % band)
        minsb = sig1 * minsig
        print 'Sigma1:', sig1, 'minsig', minsig, 'minsb', minsb

        # Load the average PSF model (generated by wise_psf.py)
        print 'Reading PSF from', opt.psffn
        P = fits_table(opt.psffn, hdu=band)
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
                src.minradius = opt.minradius
                
            elif (isinstance(src, HoggGalaxy) or
                  isinstance(src, FixedCompositeGalaxy)):
                src.halfsize = R
                
        # We used to dice the image into blocks/cells...
        fullIV = np.zeros(len(cat))
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix', 'pronexp']
        fitstats = dict([(k, np.zeros(len(cat))) for k in fskeys])

        twcs = ConstantFitsWcs(wcs)
        sky = 0.
        tsky = ConstantSky(sky)

        if ps:
            tag = '%s W%i' % (tile.coadd_id, band)
            
            plt.clf()
            n,b,p = plt.hist(img.ravel(), bins=100,
                             range=(-10*sig1, 20*sig1), log=True,
                             histtype='step', color='b')
            mx = max(n)
            plt.ylim(0.1, mx)
            plt.xlim(-10*sig1, 20*sig1)
            plt.axvline(sky, color='r')
            plt.title('%s: Pixel histogram' % tag)
            ps.savefig()

        if savepickle:
            mods = []
            cats = []

        # SDSS and WISE source margins beyond the image margins ( + source radii )
        smargin = 1
        wmargin = 1

        tim = Image(data=img, invvar=invvar, psf=psf, wcs=twcs,
                    sky=tsky, photocal=LinearPhotoCal(1., band=wanyband),
                    name='Coadd %s W%i' % (tile.coadd_id, band))

        # Relevant SDSS sources:
        m = smargin + sourcerad
        I = np.flatnonzero(((T.x+m) >= -0.5) * ((T.x-m) < (W-0.5)) *
                           ((T.y+m) >= -0.5) * ((T.y-m) < (H-0.5)))
        inbox = ((T.x[I] >= -0.5) * (T.x[I] < (W-0.5)) *
                 (T.y[I] >= -0.5) * (T.y[I] < (H-0.5)))
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
        J = np.flatnonzero(((UW.x+m) >= -0.5) * ((UW.x-m) < (W-0.5)) *
                           ((UW.y+m) >= -0.5) * ((UW.y-m) < (H-0.5)))

        if opt.savewise:
            jinbox = ((UW.x[J] >= -0.5) * (UW.x[J] < (W-0.5)) *
                      (UW.y[J] >= -0.5) * (UW.y[J] < (H-0.5)))
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
        print 'Creating a Tractor with image', tim.shape, 'and', len(subcat), 'sources'
        tractor = Tractor([tim], subcat)
        tractor.disable_cache()

        print 'Running forced photometry...'
        t0 = Time()
        tractor.freezeParamsRecursive('*')

        if opt.sky:
            tractor.thawPathsTo('sky')
            print 'Initial sky values:'
            for tim in tractor.getImages():
                print tim.getSky()

        tractor.thawPathsTo(wanyband)

        wantims = (savepickle or (ps is not None) or opt.save_fits)

        R = tractor.optimize_forced_photometry(
            minsb=minsb, mindlnp=1., sky=opt.sky, minFlux=None,
            fitstats=True, fitstat_extras=[('pronexp', [nims])],
            variance=True, shared_params=False,
            use_ceres=opt.ceres, BW=opt.ceresblock, BH=opt.ceresblock,
            wantims=wantims, negfluxval=0.1*sig1)
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

        if opt.save_fits:
            (dat,mod,ie,chi,roi) = ims1[0]

            tag = 'fit-%s-w%i' % (tile.coadd_id, band)
            fitsio.write('%s-data.fits' % tag, dat, clobber=True)
            fitsio.write('%s-mod.fits' % tag,  mod, clobber=True)
            fitsio.write('%s-chi.fits' % tag,  chi, clobber=True)

        if ps:
            tag = '%s W%i' % (tile.coadd_id, band)

            (dat,mod,ie,chi,roi) = ims1[0]

            plt.clf()
            plt.imshow(dat, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=-3*sig1, vmax=10*sig1)
            plt.colorbar()
            plt.title('%s: data' % tag)
            ps.savefig()

            # plt.clf()
            # plt.imshow(1./ie, interpolation='nearest', origin='lower',
            #            cmap='gray', vmin=0, vmax=10*sig1)
            # plt.colorbar()
            # plt.title('%s: sigma' % tag)
            # ps.savefig()

            plt.clf()
            plt.imshow(mod, interpolation='nearest', origin='lower',
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

        if len(srci):
            # Save fit stats
            fullIV[srci] = IV[:len(srci)]
            for k in fskeys:
                x = getattr(fs, k)
                fitstats[k][srci] = np.array(x)

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
            lo,cathi = 10,18
            if band == 3:
                lo,cathi = 8, 13
            elif band == 4:
                #lo,cathi = 4.5, 10.5
                lo,cathi = 4.5, 12
            loghist(WISE.get('w%impro'%band)[I], T.get(wband+'_mag')[J],
                    range=((lo,cathi),(lo,cathi)), bins=200)
            plt.xlabel('WISE W%i mag' % band)
            plt.ylabel('Tractor W%i mag' % band)
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
    T.pointsource = T.pointsource.astype(np.uint8)

    T.writeto(outfn, header=hdr)
    print 'Wrote', outfn

    if savepickle:
        fn = opt.output % (tile.coadd_id)
        fn = fn.replace('.fits','.pickle')
        pickle_to_file((mods, cats, T, sourcerad), fn)
        print 'Pickled', fn

    print 'Tile', tile.coadd_id, 'took', Time()-tt0

def _write_output(T, fn, cols, dropcols, hdr):
    cols = ['has_wise_phot'] + [c for c in cols if not c in ['id']+dropcols]
    T.writeto(fn, columns=cols, header=hdr)

def todo(A, opt, ps):
    need = []
    for i in range(len(A)):
        outfn = opt.output % (A.coadd_id[i])
        #outfn = opt.unsplitoutput % (A.coadd_id[i])
        print 'Looking for', outfn
        if not os.path.exists(outfn):
            need.append(i)
    print ' '.join('%i' %i for i in need)
            
    # Collapse contiguous ranges
    strings = []
    if len(need):
        start = need.pop(0)
        end = start
        while len(need):
            x = need.pop(0)
            if x == end + 1:
                # extend this run
                end = x
            else:
                # run finished; output and start new one.
                if start == end:
                    strings.append('%i' % start)
                else:
                    strings.append('%i-%i' % (start, end))
                start = end = x
        # done; output
        if start == end:
            strings.append('%i' % start)
        else:
            strings.append('%i-%i' % (start, end))
        print ','.join(strings)
    else:
        print 'Done (party now)'
        
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

def _get_output_column_names(bands):
    cols = ['ra','dec', 'objid', 'x','y', 
            'treated_as_pointsource', 'pointsource', 'coadd_id', 'modelflux']
    for band in bands:
        for k in ['nanomaggies', 'nanomaggies_ivar', 'mag', 'mag_err',
                  'prochi2', 'pronpix', 'profracflux', 'proflux', 'npix',
                  'pronexp']:
            cols.append('w%i_%s' % (band, k))
    cols.extend(['run','camcol','field','id'])
    # columns to drop from the photoObj-parallels
    dropcols = ['run', 'camcol', 'field', 'modelflux']
    return cols, dropcols

def main():
    import optparse

    global outdir
    global tempoutdir
    global tiledir

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--minsig1', dest='minsig1', default=0.1, type=float)
    parser.add_option('--minsig2', dest='minsig2', default=0.1, type=float)
    parser.add_option('--minsig3', dest='minsig3', default=0.1, type=float)
    parser.add_option('--minsig4', dest='minsig4', default=0.1, type=float)
    parser.add_option('-d', dest='outdir', default=None,
                      help='Output directory')
    parser.add_option('--tempdir', default=None,
                      help='"Temp"-file output directory')
    parser.add_option('-o', dest='output', default=None, help='Output filename pattern')
    parser.add_option('-b', '--band', dest='bands', action='append', type=int, default=[],
                      help='Add WISE band (default: 1,2)')

    parser.add_option('--tiledir', type=str, help='Set input unWISE coadds dir; default %s' % tiledir)

    parser.add_option('--wise-only', dest='wiseOnly',
                      action='store_true', default=False,
                      help='Ensure WISE file exists and then quit?')

    parser.add_option('-p', dest='pickle', default=False, action='store_true',
                      help='Save .pickle file for debugging purposes')
    parser.add_option('--plots', dest='plots', default=False, action='store_true')

    parser.add_option('--save-fits', dest='save_fits', default=False, action='store_true')

    parser.add_option('--plotbase', dest='plotbase', help='Base filename for plots')

    parser.add_option('--summary', dest='summary', default=False, action='store_true')
    parser.add_option('--todo', dest='todo', default=False, action='store_true')

    parser.add_option('--no-ceres', dest='ceres', action='store_false', default=True,
                       help='Use scipy lsqr rather than Ceres Solver?')

    parser.add_option('--ceres-block', '-B', dest='ceresblock', type=int, default=10,
                      help='Ceres image block size (default: %default)')

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

    parser.add_option('--tile', dest='tile', action='append', default=[],
                      help='Run a single tile')

    parser.add_option('--psf', dest='psffn', default='psf-allwise-con3.fits')

    parser.add_option('-v', dest='verbose', default=False, action='store_true')

    parser.add_option('--no-threads', action='store_true')

    opt,args = parser.parse_args()

    opt.unsplitoutput = None
    
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

    #if len(opt.tile) == 0:
    # sequels-atlas.fits: written by wise-coadd.py
    fn = '%s-atlas.fits' % opt.dataset
    print 'Reading', fn
    T = fits_table(fn)

    try:
        version = get_svn_version()
    except:
        version = dict(Revision=-1, URL='')
    print 'SVN version info:', version

    hdr = fitsio.FITSHDR()
    hdr.add_record(dict(name='SEQ_VER', value=version['Revision'],
                        comment='SVN revision'))
    hdr.add_record(dict(name='SEQ_URL', value=version['URL'], comment='SVN URL'))
    hdr.add_record(dict(name='SEQ_DATE', value=datetime.datetime.now().isoformat(),
                        comment='forced phot run time'))
    hdr.add_record(dict(name='SEQ_SKY', value=opt.sky, comment='fit sky?'))
    for band in opt.bands:
        minsig = getattr(opt, 'minsig%i' % band)
        hdr.add_record(dict(name='SEQ_MNS%i' % band, value=minsig,
                            comment='min surf brightness in sig, band %i' % band))
    hdr.add_record(dict(name='SEQ_CERE', value=opt.ceres, comment='use Ceres?'))
    hdr.add_record(dict(name='SEQ_ERRF', value=opt.errfrac, comment='error flux fraction'))
    if opt.ceres:
        hdr.add_record(dict(name='SEQ_CEBL', value=opt.ceresblock,
                        comment='Ceres blocksize'))
    
    if opt.plotbase is None:
        opt.plotbase = opt.dataset + '-phot'
    ps = PlotSequence(opt.plotbase)

    outdir     = outdir     % opt.dataset
    tempoutdir = tempoutdir % opt.dataset

    if opt.outdir is not None:
        outdir = opt.outdir
    else:
        # default
        opt.outdir = outdir

    if opt.tempdir is not None:
        tempoutdir = opt.tempdir
    else:
        # default
        opt.tempdir = tempoutdir
        
    if opt.output is None:
        opt.output = os.path.join(outdir, 'phot-%s.fits')
    if opt.unsplitoutput is None:
        opt.unsplitoutput = os.path.join(outdir, 'phot-unsplit-%s.fits')
    if opt.save_wise_output is None:
        opt.save_wise_output = opt.output.replace('phot-', 'phot-wise-')

    if opt.summary:
        summary(T, opt, ps)
        sys.exit(0)

    if opt.todo:
        todo(T, opt, ps)
        sys.exit(0)
        
    for dirnm in [outdir, tempoutdir]:
        if not os.path.exists(dirnm):
            try:
                os.makedirs(dirnm)
            except:
                pass

    # don't need this
    disable_galaxy_cache()

    tiles = []
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        tiles.append(arr)

    if len(opt.tile):
        for t in opt.tile:
            I = np.flatnonzero(T.coadd_id == t)
            if len(I) == 0:
                print 'Failed to find tile id', t, 'in dataset', opt.dataset
                return -1
            assert(len(I) == 1)
            tiles.append(I[0])

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

    if len(tiles) == 0:
        tiles.append(0)

    for i in tiles:
        if opt.plots:
            plot = ps
        else:
            plot = None
        print
        print 'Tile index', i, 'coadd', T.coadd_id[i]

        if opt.wiseOnly:
            # Find WISE-only catalog sources
            tile = T[i]
            thisdir = get_unwise_tile_dir(tiledir, tile.coadd_id)
            bands = opt.bands
            fn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits' %
                              (tile.coadd_id, bands[0]))
            print 'Reading', fn
            wcs = Tan(fn)
            r0,r1,d0,d1 = wcs.radec_bounds()
            print 'RA,Dec bounds:', r0,r1,d0,d1
            wfn = os.path.join(tempoutdir, 'wise-sources-%s.fits' % (tile.coadd_id))
            WISE = read_wise_sources(wfn, r0,r1,d0,d1, allwise=True)
            continue
        
        one_tile(T[i], opt, opt.pickle, plot, T, tiledir, tempoutdir, hdr=hdr)

if __name__ == '__main__':
    main()

