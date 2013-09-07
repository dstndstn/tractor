import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *

from tractor import *
from tractor.ttime import *
from tractor.sdss import *

from wisecat import wise_catalog_radecbox

import logging
lvl = logging.INFO
#lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

sweepdir = 'sweeps'
tiledir = 'wise-coadds'
photoobjdir = 'photoObjs'

Time.add_measurement(MemMeas)

def read_sweeps(sweeps, r0,r1,d0,d1):
    margin = 0.
    # Add approx SDSS field size margin
    margin += np.hypot(13., 9.)/60.
    cosd = np.cos(np.deg2rad(sweeps.dec))            
    S = sweeps[(sweeps.ra  > (r0-margin/cosd)) * (sweeps.ra  < (r1+margin/cosd)) *
               (sweeps.dec > (d0-margin))      * (sweeps.dec < (d1+margin))]
    print 'Cut to', len(S), 'datasweeps in this tile'

    stars = []
    gals = []
    for si,sweep in enumerate(S):
        print 'Datasweep', si+1, 'of', len(S)
        fn = 'calibObj-%06i-%i-%s.fits.gz' % (sweep.run, sweep.camcol, 'gal' if sweep.isgal else 'star')
        fn = os.path.join(sweepdir, sweep.rerun, fn)
        print 'Reading', fn, 'rows', sweep.istart, 'to', sweep.iend

        columns = ['ra','dec']
        if sweep.isgal:
            columns += ['id'] #'theta_dev', 'theta_exp', 'id']


        with fitsio.FITS(fn, lower=True) as F:
            T = F[1][columns][sweep.istart : sweep.iend+1]
            #print 'Read table', type(T), T.dtype
            #print dir(T)
            T = fits_table(T)
            #print 'Read table:', T
            #T.about()
            print 'Read', len(T)

        # Cut to RA,Dec box
        T.cut((T.ra > r0) * (T.ra < r1) * (T.dec > d0) * (T.dec < d1))
        print 'Cut to', len(T), 'in RA,Dec box'
        if len(T) == 0:
            continue

        if sweep.isgal:
            # Cross-reference to photoObj files to get the galaxy shape
            fn = 'photoObj-%06i-%i-%04i.fits' % (sweep.run, sweep.camcol, sweep.field)
            fn = os.path.join(photoobjdir, '%i'%sweep.run, '%i'%sweep.camcol, fn)
            print 'Reading photoObj', fn
            cols = ['id', 'theta_dev', 'ab_dev', 'theta_exp',
                    'ab_exp', 'fracdev', 'phi_dev_deg', 'phi_exp_deg']
            # DEBUG
            cols += ['modelflux', 'modelflux_ivar',
                     'devflux', 'devflux_ivar',
                     'expflux', 'expflux_ivar']
            cols += ['ab_experr', 'ab_deverr', 'theta_deverr', 'theta_experr']

            P = fits_table(fn, columns=cols, rows=T.id-1)
            print 'Read', len(P), 'photoObj entries'
            assert(np.all(P.id == T.id))
            P.ra = T.ra
            P.dec = T.dec
            T = P

        if sweep.isgal:
            gals.append(T)
        else:
            stars.append(T)
        
    stars = merge_tables(stars)
    gals =  merge_tables(gals)
    print 'Total of', len(stars), 'stars and', len(gals), 'galaxies in this tile'
    return gals,stars


def main():
    import optparse

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--minsb1', dest='minsb1', default=1e-3, type=float)
    parser.add_option('--minsb2', dest='minsb2', default=1e-3, type=float)
    parser.add_option('--minsb3', dest='minsb3', default=1e-3, type=float)
    parser.add_option('--minsb4', dest='minsb4', default=1e-3, type=float)
    parser.add_option('--blocks', dest='blocks', default=10, type=int,
                      help='NxN number of blocks to cut the image into')
    parser.add_option('-o', dest='output', default='phot-%s.fits')
    opt,args = parser.parse_args()

    '''
    ln -s /clusterfs/riemann/raid006/dr10/boss/sweeps/dr9 sweeps
    ln -s /clusterfs/riemann/raid006/dr10/boss/photoObj/301 photoObjs    
    '''
    dataset = 'sequels'
    fn = '%s-atlas.fits' % dataset
    print 'Reading', fn
    T = fits_table(fn)

    bands = [1,2,3,4]
    
    ps = PlotSequence(dataset)

    # SEQUELS
    R0,R1 = 120.0, 200.0
    D0,D1 =  45.0,  60.0

    sband = 'r'
    bandnum = 'ugriz'.index(sband)
    
    gsweeps = fits_table(os.path.join(sweepdir, 'datasweep-index-gal.fits'))
    ssweeps = fits_table(os.path.join(sweepdir, 'datasweep-index-star.fits'))
    print 'Read', len(gsweeps), 'galaxy sweep entries'
    print 'Read', len(ssweeps), 'star sweep entries'
    gsweeps.cut(gsweeps.nprimary > 0)
    ssweeps.cut(ssweeps.nprimary > 0)
    print 'Cut to', len(gsweeps), 'gal and', len(ssweeps), 'star on NPRIMARY'
    margin = 1
    gsweeps.cut((gsweeps.ra  > (R0-margin)) * (gsweeps.ra  < (R1+margin)) *
                (gsweeps.dec > (D0-margin)) * (gsweeps.dec < (D1+margin)))
    ssweeps.cut((ssweeps.ra  > (R0-margin)) * (ssweeps.ra  < (R1+margin)) *
                (ssweeps.dec > (D0-margin)) * (ssweeps.dec < (D1+margin)))
    print 'Cut to', len(gsweeps), 'gal and', len(ssweeps), 'star on RA,Dec box'
    gsweeps.isgal = np.ones( len(gsweeps), int)
    ssweeps.isgal = np.zeros(len(ssweeps), int)
    sweeps = merge_tables([gsweeps, ssweeps])
    print 'Merged:', len(sweeps)
    
    for tile in T:
        tt0 = Time()
        print
        print 'Coadd tile', tile.coadd_id

        # Our results table
        PHOT = tabledata()

        fn = os.path.join(tiledir, 'coadd-%s-w%i-img.fits' % (tile.coadd_id, bands[0]))
        print 'Reading', fn
        wcs = Tan(fn)
        r0,r1,d0,d1 = wcs.radec_bounds()
        print 'RA,Dec bounds:', r0,r1,d0,d1
        #ra,dec = wcs.radec_center()
        #print 'Center:', ra,dec
        H,W = wcs.get_height(), wcs.get_width()

        ### HACK!
        H,W = 1024,1024


        ssweepfn = 'sweeps-%s-stars.fits' % tile.coadd_id
        gsweepfn = 'sweeps-%s-gals.fits' % tile.coadd_id
        if os.path.exists(ssweepfn) and os.path.exists(gsweepfn):
            stars = fits_table(ssweepfn)
            gals  = fits_table(gsweepfn)
        else:
            gals,stars = read_sweeps(sweeps, r0,r1,d0,d1)
            stars.writeto(ssweepfn)
            gals.writeto(gsweepfn)


        # Cut galaxies based on signal-to-noise of theta (effective radius)
        # measurement.
        b = bandnum
        dev = (gals.fracdev[:,b] >= 0.5)
        exp = (gals.fracdev[:,b] < 0.5)
        tsn = np.zeros(len(gals))
        gals.theta_deverr[dev,b] = np.maximum(1e-6, gals.theta_deverr[dev,b])
        gals.theta_experr[exp,b] = np.maximum(1e-5, gals.theta_experr[exp,b])
        # theta_experr nonzero: 1.28507e-05
        # theta_deverr nonzero: 1.92913e-06
        tsn[dev] = gals.theta_dev[dev,b] / gals.theta_deverr[dev,b]
        tsn[exp] = gals.theta_exp[exp,b] / gals.theta_experr[exp,b]
        # print 'theta_experr range:', gals.theta_experr[exp,b].min(), gals.theta_experr[exp,b].max()
        # print 'theta_deverr range:', gals.theta_deverr[dev,b].min(), gals.theta_deverr[dev,b].max()
        # print 'theta_experr nonzero:', (gals.theta_experr[exp,b][gals.theta_experr[exp,b] > 0]).min()
        # print 'theta_deverr nonzero:', (gals.theta_deverr[dev,b][gals.theta_deverr[dev,b] > 0]).min()
        # I = (gals.theta_experr[exp,b] == 0)
        # print sum(I), 'theta exp errors are zero'
        # print 'thetas are, eg,', gals.theta_exp[exp[np.flatnonzero(I)[:50]],b]
        # print 'modelfluxes are, eg,', gals.modelflux[exp[np.flatnonzero(I)[:50]],b]

        assert(np.all(np.isfinite(gals.theta_dev[dev,b])))
        assert(np.all(np.isfinite(gals.theta_exp[exp,b])))
        assert(np.all(gals.theta_experr[exp,b] > 0))
        assert(np.all(gals.theta_deverr[dev,b] > 0))

        print len(gals), 'galaxies'
        bad = np.logical_or(tsn < 3., gals.modelflux[:,b] > 1e4)
        print 'Found', sum(bad), 'low theta S/N or huge-flux galaxies'

        gstars = fits_table()
        gstars.ra  = gals.ra[bad]
        gstars.dec = gals.dec[bad]
        print 'Adding', len(gstars), 'bad galaxies to "stars"'
        stars = merge_tables([stars, gstars])
        gals.cut(np.logical_not(bad))
        print 'Cut to', len(gals), 'not-bad galaxies'
            
        # hack
        gals.objc_type  = np.zeros(len(gals), int) + 3
        gals.psfflux    = np.ones((len(gals),5))
        gals.cmodelflux = np.ones((len(gals),5))
        gals.devflux    = np.ones((len(gals),5))
        gals.expflux    = np.ones((len(gals),5))
        gals.nchild     = np.zeros(len(gals), int)
        gals.objc_flags = np.zeros(len(gals), int)

        wfn = 'wise-sources-%s.fits' % (tile.coadd_id)
        if os.path.exists(wfn):
            WISE = fits_table(wfn)
            print 'Read', len(WISE), 'WISE sources nearby'
        else:
            cols = ['ra','dec'] + ['w%impro'%band for band in [1,2,3,4]]
            WISE = wise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
            WISE.writeto(wfn)
            print 'Found', len(WISE), 'WISE sources nearby'

        # unmatched = np.ones(len(WISE), bool)
        # I,J,d = match_radec(WISE.ra, WISE.dec, stars.ra, stars.dec, 4./3600.)
        # unmatched[I] = False
        # I,J,d = match_radec(WISE.ra, WISE.dec, gals.ra, gals.dec, 4./3600.)
        # unmatched[I] = False
        # UW = WISE[unmatched]
        # print 'Got', len(UW), 'unmatched WISE sources'
        # #del WISE
        # WISE = WISE[np.logical_not(unmatched)]


        ### HACK -- cut star/gal lists
        ok,gx,gy = wcs.radec2pixelxy(gals.ra, gals.dec)
        gx -= 1.
        gy -= 1.
        margin = 20.
        I = np.flatnonzero((gx >= -margin) * (gx < W+margin) *
                           (gy >= -margin) * (gy < H+margin))
        gals.cut(I)
        ok,gx,gy = wcs.radec2pixelxy(stars.ra, stars.dec)
        gx -= 1.
        gy -= 1.
        margin = 20.
        I = np.flatnonzero((gx >= -margin) * (gx < W+margin) *
                           (gy >= -margin) * (gy < H+margin))
        stars.cut(I)


        wanyband = wband = 'w'
        print 'Creating tractor galaxies...'
        cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                         objs=gals, bands=[], nanomaggies=True,
                                         extrabands=[wband],
                                         fixedComposites=True,
                                         useObjcType=True)
        ngals = len(cat)
        print 'Adding tractor stars...'
        for i in range(len(stars)):
            cat.append(PointSource(RaDecPos(stars.ra[i], stars.dec[i]),
                                   NanoMaggies(**{wband:1.})))

        # print 'Adding WISE stars...'
        # wcat = []
        # for i in range(len(UW)):
        #     wcat.append(PointSource(RaDecPos(UW.ra[i], UW.dec[i]),
        #                             NanoMaggies(**{wband:1.})))

        PHOT.ra  = np.array([src.getPosition().ra  for src in cat])
        PHOT.dec = np.array([src.getPosition().dec for src in cat])

        unmatched = np.ones(len(WISE), bool)
        I,J,d = match_radec(WISE.ra, WISE.dec, PHOT.ra, PHOT.dec, 4./3600., nearest=True)
        unmatched[I] = False
        for band in bands:
            WISE.set('w%inm' % band,
                     NanoMaggies.magToNanomaggies(WISE.get('w%impro' % band)))
        UW = WISE[unmatched]
        print 'Got', len(UW), 'unmatched WISE sources'
        #del WISE
        #WISE = WISE[np.logical_not(unmatched)]
        wiseflux = {}
        for band in bands:
            wiseflux[band] = np.zeros(len(PHOT))
            wiseflux[band][J] = WISE.get('w%inm' % band)[I]

        ### ASSUME the atlas tile WCSes are the same between bands!
        ok,sx,sy = wcs.radec2pixelxy(PHOT.ra, PHOT.dec)
        sx -= 1.
        sy -= 1.
        ok,wx,wy = wcs.radec2pixelxy(UW.ra, UW.dec)
        wx -= 1.
        wy -= 1.

        PHOT.x = sx
        PHOT.y = sy

        pixscale = wcs.pixel_scale()
        # crude source radii, in pixels
        sourcerad = []
        for i in range(ngals):
            src = cat[i]
            if isinstance(src, HoggGalaxy):
                #print '  Hogg:', src
                sourcerad.append(src.nre * src.re / pixscale)
            elif isinstance(src, FixedCompositeGalaxy):
                #print '  Comp:', src
                sourcerad.append(max(src.shapeExp.re * ExpGalaxy.nre,
                                     src.shapeDev.re * DevGalaxy.nre) / pixscale)
            else:
                assert(False)
        sourcerad.extend([0] * (len(cat)-ngals))
        sourcerad = np.array(sourcerad)

        print 'Cat:', len(cat)
        print 'PHOT:', len(PHOT)
        print 'sourcerad:', len(sourcerad), sourcerad.shape, sourcerad.dtype
        print 'sx:', sx.shape, sx.dtype
        print 'sourcerad range:', min(sourcerad), max(sourcerad)

        PHOT.cell = np.zeros(len(PHOT), int)
        PHOT.cell_x0 = np.zeros(len(PHOT), int)
        PHOT.cell_y0 = np.zeros(len(PHOT), int)
        PHOT.cell_x1 = np.zeros(len(PHOT), int)
        PHOT.cell_y1 = np.zeros(len(PHOT), int)

        inbounds = np.flatnonzero((sx >= -0.5) * (sx < W-0.5) *
                                  (sy >= -0.5) * (sy < H-0.5))

        for band in bands:
            tb0 = Time()
            print
            print 'Coadd tile', tile.coadd_id
            print 'Band', band
            wband = 'w%i' % band

            fn = os.path.join(tiledir, 'coadd-%s-w%i-img.fits' % (tile.coadd_id, band))
            print 'Reading', fn
            wcs = Tan(fn)
            r0,r1,d0,d1 = wcs.radec_bounds()
            print 'RA,Dec bounds:', r0,r1,d0,d1
            ra,dec = wcs.radec_center()
            print 'Center:', ra,dec
            img = fitsio.read(fn)
            #H,W = img.shape

            fn = os.path.join(tiledir, 'coadd-%s-w%i-invvar.fits' % (tile.coadd_id, band))
            print 'Reading', fn
            iv = fitsio.read(fn)

            # HACK -- should we *average* the PSF over the whole image, maybe?

            # Load the spatially-varying PSF model
            from wise_psf import WisePSF
            psf = WisePSF(band, savedfn='w%ipsffit.fits' % band)
            # Instantiate a (non-varying) mixture-of-Gaussians PSF
            psf = psf.mogAt(W/2., H/2.)

            # tim = Image(data=img, invvar=iv, psf=psf, wcs=ConstantFitsWcs(wcs),
            #             sky=ConstantSky(0.), photocal=LinearPhotoCal(1., band=wband),
            #             name='Coadd %s W%i' % (tile.coadd_id, band), domask=False)

            fullimg = img
            fullinvvar = iv
            fullIV = np.zeros(len(cat))
            fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
            fitstats = dict([(k, np.zeros(len(cat))) for k in fskeys])

            #ntimes = np.zeros(len(cat), int)

            twcs = ConstantFitsWcs(wcs)

            # cell positions
            XX = np.round(np.linspace(0, W, opt.blocks+1)).astype(int)
            YY = np.round(np.linspace(0, H, opt.blocks+1)).astype(int)

            mods = []
            cats = []

            celli = -1
            for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
                for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
                    celli += 1

                    print
                    print 'Cell', celli, 'of', (opt.blocks**2), 'for', tile.coadd_id, 'band', wband

                    # imargin = 4
                    # smargin = 8
                    imargin = 12
                    #smargin = 16
                    wmargin = 16

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
                                sky=ConstantSky(0.), photocal=LinearPhotoCal(1., band=wanyband),
                                name='Coadd %s W%i (%i,%i)' % (tile.coadd_id, band, xi,yi),
                                domask=False)

                    smargin = imargin + sourcerad

                    I = np.flatnonzero(((sx+smargin) >= (xlo-0.5)) * ((sx-smargin) < (xhi-0.5)) *
                                       ((sy+smargin) >= (ylo-0.5)) * ((sy-smargin) < (yhi-0.5)))
                    #I = np.flatnonzero((sx >= (xlo-0.5-smargin)) * (sx < (xhi-0.5+smargin)) *
                    #                   (sy >= (ylo-0.5-smargin)) * (sy < (yhi-0.5+smargin)))

                    inbox = ((sx[I] >= (xlo-0.5)) * (sx[I] < (xhi-0.5)) *
                             (sy[I] >= (ylo-0.5)) * (sy[I] < (yhi-0.5)))

                    srci = I[inbox]

                    PHOT.cell[I[inbox]] = celli
                    PHOT.cell_x0[I[inbox]] = ix0
                    PHOT.cell_x1[I[inbox]] = ix1
                    PHOT.cell_y0[I[inbox]] = iy0
                    PHOT.cell_y1[I[inbox]] = iy1
                    
                    # sources in the ROI box
                    subcat = [cat[i] for i in srci]

                    # WISE-only sources in the expanded region
                    J = np.flatnonzero((wx >= xlo - wmargin) * (wx < xhi + wmargin) *
                                       (wy >= ylo - wmargin) * (wy < yhi + wmargin))
                    if True:
                        # include *copies* of sources in the margins
                        subcat.extend([cat[i].copy() for i in I[np.logical_not(inbox)]])
                        assert(len(subcat) == len(I))
                        # add WISE-only point sources
                        for i in J:
                            subcat.append(PointSource(RaDecPos(UW.ra[i], UW.ra[i]),
                                                      NanoMaggies(**{wanyband:1.})))
                        
                    if False:
                        for i in I[np.logical_not(inbox)]:
                            src = cat[i].copy()
                            nm = wiseflux[band][i]
                            src.setBrightness(NanoMaggies(**{wanyband:nm}))
                            subcat.append(src)
                        for i in J:
                            nm = UW.get('w%inm' % band)[i]
                            subcat.append(PointSource(RaDecPos(UW.ra[i], UW.ra[i]),
                                                      NanoMaggies(**{wanyband:nm})))

                    print 'Sources:', len(srci), 'in the box,', len(I)-len(srci), 'in the margins, and', len(J), 'WISE-only'

                    print 'Creating a Tractor with image', tim.shape, 'and', len(subcat), 'sources'
                    tractor = Tractor([tim], subcat)

                    print 'Running forced photometry...'
                    t0 = Time()
                    tractor.freezeParamsRecursive('*')
                    # tractor.thawPathsTo('sky')
                    tractor.thawPathsTo(wanyband)

                    ###### Freeze sources outside the ROI?
                    #tractor.catalog.freezeParams(*range(len(srci), len(subcat)))

                    # Reset initial fluxes (note that this is only for the unfrozen ones)
                    #tractor.setParams([1.] * tractor.numberOfParams()) #np.ones(tractor.numberOfParams()))
                    tractor.setParams(np.ones(tractor.numberOfParams()))
    
                    minsb = getattr(opt, 'minsb%i' % band)
                    print 'Minsb:', minsb
                    ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
                        minsb=minsb, mindlnp=1., sky=False, minFlux=None,
                        fitstats=True, variance=True, shared_params=False)
                    print 'That took', Time()-t0

                    im,mod,ie,chi,roi = ims1[0]

                    mods.append(mod)
                    cats.append(([src.getPosition().ra  for src in subcat],
                                 [src.getPosition().dec for src in subcat],
                                 [src.copy() for src in subcat], tim))
                    
                    fullIV[srci] = IV[:len(srci)]

                    for k in fskeys:
                        x = getattr(fs, k)
                        fitstats[k][srci] = np.array(x)

                    cpu0 = tb0.meas[0]
                    t = Time()
                    cpu = t.meas[0]
                    dcpu = (cpu.cpu - cpu0.cpu)
                    print 'So far:', Time()-tb0, '-> predict CPU time', (dcpu * (opt.blocks**2) / float(celli+1))

                    #ntimes[srci] += 1

            # print 'Number of times sources were fit:', np.unique(ntimes)
            # print np.bincount(ntimes)
            # PHOT.set(wband + '_ntimes', ntimes)

            #print 'ntimes == inbounds?', np.sum(np.flatnonzero(ntimes) == inbounds)
            #print 'of', len(inbounds)

            nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
            nm_ivar = fullIV
            PHOT.set(wband + '_nanomaggies', nm)
            PHOT.set(wband + '_nanomaggies_ivar', fullIV)
            dnm = 1./np.sqrt(nm_ivar)
            mag = NanoMaggies.nanomaggiesToMag(nm)
            dmag = np.abs((-2.5 / np.log(10.)) * dnm / nm)
            PHOT.set(wband + '_mag', mag)
            PHOT.set(wband + '_mag_err', dmag)
            for k in fskeys:
                PHOT.set(wband + '_' + k, fitstats[k])
            # pickle_to_file((ims0,ims1,IV,cat), 'phot-%s-%s.pickle' % (tile.coadd_id, wband))

            print 'Tile', tile.coadd_id, 'band', wband, 'took', Time()-tb0

            # HACK
            break

        PHOT.cut(inbounds)

        PHOT.writeto(opt.output % (tile.coadd_id))

        ## HACK
        fn = opt.output % (tile.coadd_id)
        fn = fn.replace('.fits','.pickle')
        pickle_to_file((mods, cats, sx, sy, sourcerad), fn)
        print 'Pickled', fn

        print 'Tile', tile.coadd_id, 'took', Time()-tt0

        # HACK
        break


if __name__ == '__main__':
    main()

