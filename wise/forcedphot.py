from __future__ import print_function
import sys

import numpy as np

from tractor import RaDecPos, NanoMaggies, PointSource, Tractor
from tractor.galaxy import (ExpGalaxy, DevGalaxy, FixedCompositeGalaxy,
                            disable_galaxy_cache)
from tractor.ellipses import EllipseE

from astrometry.util.fits import fits_table
from astrometry.util.ttime import Time

from .unwise import (unwise_tile_wcs, unwise_tiles_touching_wcs,
                     get_unwise_tractor_image)


def unwise_forcedphot(cat, tiles, bands=None, roiradecbox=None,
                      unwise_dir='.',
                      use_ceres=True, ceres_block=8,
                      save_fits=False, get_models=False, ps=None,
                      psf_broadening=None,
                      pixelized_psf=False):
    '''
    Given a list of tractor sources *cat*
    and a list of unWISE tiles *tiles* (a fits_table with RA,Dec,coadd_id)
    runs forced photometry, returning a FITS table the same length as *cat*.
    '''

    if bands is None:
        bands = [1,2,3,4]

    # # Severely limit sizes of models
    for src in cat:
        if isinstance(src, PointSource):
            src.fixedRadius = 20
        else:
            src.halfsize = 20

    wantims = ((ps is not None) or save_fits or get_models)
    wanyband = 'w'
    if get_models:
        models = {}

    fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix',
              'pronexp']

    Nsrcs = len(cat)
    phot = fits_table()
    phot.tile = np.array(['        '] * Nsrcs)

    ra = np.array([src.getPosition().ra for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])

    for band in bands:
        print('Photometering WISE band', band)
        wband = 'w%i' % band

        # The tiles have some overlap, so for each source, keep the
        # fit in the tile whose center is closest to the source.
        tiledists = np.empty(Nsrcs)
        tiledists[:] = 1e100
        flux_invvars = np.zeros(Nsrcs, np.float32)
        fitstats = dict([(k, np.zeros(Nsrcs, np.float32)) for k in fskeys])
        nexp = np.zeros(Nsrcs, np.int16)
        mjd = np.zeros(Nsrcs, np.float64)

        for tile in tiles:
            print('Reading tile', tile.coadd_id)

            tim = get_unwise_tractor_image(unwise_dir, tile.coadd_id, band,
                                           bandname=wanyband, roiradecbox=roiradecbox)
            if tim is None:
                print('Actually, no overlap with tile', tile.coadd_id)
                continue

            if pixelized_psf:
                import unwise_psf
                psfimg = unwise_psf.get_unwise_psf(band, tile.coadd_id)
                print('PSF postage stamp', psfimg.shape, 'sum', psfimg.sum())
                from tractor.psf import PixelizedPSF
                psfimg /= psfimg.sum()
                tim.psf = PixelizedPSF(psfimg)
                print('### HACK ### normalized PSF to 1.0')
                print('Set PSF to', tim.psf)

                if False:
                    ph,pw = psfimg.shape
                    px,py = np.meshgrid(np.arange(ph), np.arange(pw))
                    cx = np.sum(psfimg * px)
                    cy = np.sum(psfimg * py)
                    print('PSF center of mass: %.2f, %.2f' % (cx, cy))
        
                    for sz in range(1, 11):
                        middle = pw//2
                        sub = (slice(middle-sz, middle+sz+1),
                               slice(middle-sz, middle+sz+1))
                        cx = np.sum((psfimg * px)[sub]) / np.sum(psfimg[sub])
                        cy = np.sum((psfimg * py)[sub]) / np.sum(psfimg[sub])
                        print('Size', sz, ': PSF center of mass: %.2f, %.2f' % (cx, cy))
                    
                    import fitsio
                    fitsio.write('psfimg-%s-w%i.fits' % (tile.coadd_id, band), psfimg,
                             clobber=True)
            
            if psf_broadening is not None and not pixelized_psf:
                # psf_broadening is a factor by which the PSF FWHMs
                # should be scaled; the PSF is a little wider
                # post-reactivation.
                psf = tim.getPsf()
                from tractor import GaussianMixturePSF
                if isinstance(psf, GaussianMixturePSF):
                    #
                    print('Broadening PSF: from', psf)
                    p0 = psf.getParams()
                    #print('Params:', p0)
                    pnames = psf.getParamNames()
                    #print('Param names:', pnames)
                    p1 = [p * psf_broadening**2 if 'var' in name else p
                          for (p, name) in zip(p0, pnames)]
                    #print('Broadened:', p1)
                    psf.setParams(p1)
                    print('Broadened PSF:', psf)
                else:
                    print(
                        'WARNING: cannot apply psf_broadening to WISE PSF of type', type(psf))

            print('Read image with shape', tim.shape)

            # Select sources in play.
            wcs = tim.wcs.wcs
            H, W = tim.shape
            ok, x, y = wcs.radec2pixelxy(ra, dec)
            x = (x - 1.).astype(np.float32)
            y = (y - 1.).astype(np.float32)
            margin = 10.
            I = np.flatnonzero((x >= -margin) * (x < W + margin) *
                               (y >= -margin) * (y < H + margin))
            print(len(I), 'within the image + margin')

            inbox = ((x[I] >= -0.5) * (x[I] < (W - 0.5)) *
                     (y[I] >= -0.5) * (y[I] < (H - 0.5)))
            print(sum(inbox), 'strictly within the image')

            # Compute L_inf distance to (full) tile center.
            tilewcs = unwise_tile_wcs(tile.ra, tile.dec)
            cx, cy = tilewcs.crpix
            ok, tx, ty = tilewcs.radec2pixelxy(ra[I], dec[I])
            td = np.maximum(np.abs(tx - cx), np.abs(ty - cy))
            closest = (td < tiledists[I])
            tiledists[I[closest]] = td[closest]

            keep = inbox * closest

            # Source indices (in the full "cat") to keep (the fit values for)
            srci = I[keep]

            if not len(srci):
                print('No sources to be kept; skipping.')
                continue

            phot.tile[srci] = tile.coadd_id
            nexp[srci] = tim.nuims[np.clip(np.round(y[srci]).astype(int), 0, H - 1),
                                   np.clip(np.round(x[srci]).astype(int), 0, W - 1)]

            # Source indices in the margins
            margi = I[np.logical_not(keep)]

            # sources in the box -- at the start of the subcat list.
            subcat = [cat[i] for i in srci]

            # include *copies* of sources in the margins
            # (that way we automatically don't save the results)
            subcat.extend([cat[i].copy() for i in margi])
            assert(len(subcat) == len(I))

            # FIXME -- set source radii, ...?

            minsb = 0.
            fitsky = False

            # Look in image and set radius based on peak height??

            tractor = Tractor([tim], subcat)
            if use_ceres:
                from tractor.ceres_optimizer import CeresOptimizer
                tractor.optimizer = CeresOptimizer(BW=ceres_block,
                                                   BH=ceres_block)
            tractor.freezeParamsRecursive('*')
            tractor.thawPathsTo(wanyband)

            kwa = dict(fitstat_extras=[('pronexp', [tim.nims])])
            t0 = Time()

            R = tractor.optimize_forced_photometry(
                minsb=minsb, mindlnp=1., sky=fitsky, fitstats=True,
                variance=True, shared_params=False,
                wantims=wantims, **kwa)
            print('unWISE forced photometry took', Time() - t0)

            if use_ceres:
                term = R.ceres_status['termination']
                print('Ceres termination status:', term)
                # Running out of memory can cause failure to converge
                # and term status = 2.
                # Fail completely in this case.
                if term != 0:
                    raise RuntimeError(
                        'Ceres terminated with status %i' % term)

            if wantims:
                ims0 = R.ims0
                ims1 = R.ims1
            IV, fs = R.IV, R.fitstats

            if save_fits:
                import fitsio
                (dat, mod, ie, chi, roi) = ims1[0]
                wcshdr = fitsio.FITSHDR()
                tim.wcs.wcs.add_to_header(wcshdr)

                tag = 'fit-%s-w%i' % (tile.coadd_id, band)
                fitsio.write('%s-data.fits' %
                             tag, dat, clobber=True, header=wcshdr)
                fitsio.write('%s-mod.fits' % tag,  mod,
                             clobber=True, header=wcshdr)
                fitsio.write('%s-chi.fits' % tag,  chi,
                             clobber=True, header=wcshdr)

            if get_models:
                (dat, mod, ie, chi, roi) = ims1[0]
                models[(tile.coadd_id, band)] = (mod, tim.roi)

            if ps:
                tag = '%s W%i' % (tile.coadd_id, band)
                (dat, mod, ie, chi, roi) = ims1[0]

                sig1 = tim.sig1
                plt.clf()
                plt.imshow(dat, interpolation='nearest', origin='lower',
                           cmap='gray', vmin=-3 * sig1, vmax=10 * sig1)
                plt.colorbar()
                plt.title('%s: data' % tag)
                ps.savefig()

                plt.clf()
                plt.imshow(mod, interpolation='nearest', origin='lower',
                           cmap='gray', vmin=-3 * sig1, vmax=10 * sig1)
                plt.colorbar()
                plt.title('%s: model' % tag)
                ps.savefig()

                plt.clf()
                plt.imshow(chi, interpolation='nearest', origin='lower',
                           cmap='gray', vmin=-5, vmax=+5)
                plt.colorbar()
                plt.title('%s: chi' % tag)
                ps.savefig()

            # Save results for this tile.
            # the "keep" sources are at the beginning of the "subcat" list
            flux_invvars[srci] = IV[:len(srci)].astype(np.float32)
            if hasattr(tim, 'mjdmin') and hasattr(tim, 'mjdmax'):
                mjd[srci] = (tim.mjdmin + tim.mjdmax) / 2.
            if fs is None:
                continue
            for k in fskeys:
                x = getattr(fs, k)
                # fitstats are returned only for un-frozen sources
                fitstats[k][srci] = np.array(x).astype(np.float32)[:len(srci)]

        # Note, this is *outside* the loop over tiles.
        # The fluxes are saved in the source objects, and will be set based on
        # the 'tiledists' logic above.
        nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
        nm_ivar = flux_invvars

        # Sources out of bounds, eg, never change from their default
        # (1-sigma or whatever) initial fluxes.  Zero them out instead.
        nm[nm_ivar == 0] = 0.

        phot.set(wband + '_nanomaggies', nm.astype(np.float32))
        phot.set(wband + '_nanomaggies_ivar', nm_ivar)
        dnm = np.zeros(len(nm_ivar), np.float32)
        okiv = (nm_ivar > 0)
        dnm[okiv] = (1. / np.sqrt(nm_ivar[okiv])).astype(np.float32)
        okflux = (nm > 0)
        mag = np.zeros(len(nm), np.float32)
        mag[okflux] = (NanoMaggies.nanomaggiesToMag(nm[okflux])
                       ).astype(np.float32)
        dmag = np.zeros(len(nm), np.float32)
        ok = (okiv * okflux)
        dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dnm[ok] / nm[ok])
                    ).astype(np.float32)
        mag[np.logical_not(okflux)] = np.nan
        dmag[np.logical_not(ok)] = np.nan

        phot.set(wband + '_mag', mag)
        phot.set(wband + '_mag_err', dmag)
        for k in fskeys:
            phot.set(wband + '_' + k, fitstats[k])
        phot.set(wband + '_nexp', nexp)

        if not np.all(mjd == 0):
            phot.set(wband + '_mjd', mjd)

    if get_models:
        return phot,models
    return phot


def main():
    import optparse
    from astrometry.util.plotutils import PlotSequence
    from astrometry.util.util import Tan

    parser = optparse.OptionParser(usage='%prog [options] incat.fits out.fits')
    parser.add_option('-r', '--ralo',  dest='ralo',  type=float,
                      help='Minimum RA')
    parser.add_option('-R', '--rahi',  dest='rahi',  type=float,
                      help='Maximum RA')
    parser.add_option('-d', '--declo', dest='declo', type=float,
                      help='Minimum Dec')
    parser.add_option('-D', '--dechi', dest='dechi', type=float,
                      help='Maximum Dec')

    parser.add_option('-b', '--band', dest='bands', action='append', type=int,
                      default=[], help='WISE band to photometer (default: 1,2)')

    parser.add_option('-u', '--unwise', dest='unwise_dir',
                      default='unwise-coadds',
                      help='Directory containing unWISE coadds')

    parser.add_option('--no-ceres', dest='ceres', action='store_false',
                      default=True,
                      help='Use scipy lsqr rather than Ceres Solver?')

    parser.add_option('--ceres-block', '-B', dest='ceresblock', type=int,
                      default=8,
                      help='Ceres image block size (default: %default)')

    parser.add_option('--plots', dest='plots',
                      default=False, action='store_true')
    parser.add_option('--save-fits', dest='save_fits',
                      default=False, action='store_true')

    # parser.add_option('--ellipses', action='store_true',
    #                  help='Assume catalog shapes are ellipse descriptions (not r,ab,phi)')

    # parser.add_option('--ra', help='Center RA')
    # parser.add_option('--dec', help='Center Dec')
    # parser.add_option('--width', help='Degrees width (in RA*cos(Dec))')
    # parser.add_option('--height', help='Degrees height (Dec)')
    opt, args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)

    if len(opt.bands) == 0:
        opt.bands = [1, 2]
    # Allow specifying bands like "123"
    bb = []
    for band in opt.bands:
        for s in str(band):
            bb.append(int(s))
    opt.bands = bb
    print('Bands', opt.bands)

    ps = None
    if opt.plots:
        ps = PlotSequence('unwise')

    infn, outfn = args

    T = fits_table(infn)
    print('Read', len(T), 'sources from', infn)
    if opt.declo is not None:
        T.cut(T.dec >= opt.declo)
    if opt.dechi is not None:
        T.cut(T.dec <= opt.dechi)

    # Let's be a bit smart about RA wrap-around.  Compute the 'center'
    # of the RA points, use the cross product against that to define
    # inequality (clockwise-of).
    r = np.deg2rad(T.ra)
    x = np.mean(np.cos(r))
    y = np.mean(np.sin(r))
    rr = np.hypot(x, y)
    x /= rr
    y /= rr
    midra = np.rad2deg(np.arctan2(y, x))
    midra += 360. * (midra < 0)
    xx = np.cos(r)
    yy = np.sin(r)
    T.cross = x * yy - y * xx
    minra = T.ra[np.argmin(T.cross)]
    maxra = T.ra[np.argmax(T.cross)]

    if opt.ralo is not None:
        r = np.deg2rad(opt.ralo)
        xx = np.cos(r)
        yy = np.sin(r)
        crosscut = x * yy - y * xx
        T.cut(T.cross >= crosscut)
        print('Cut to', len(T), 'with RA >', opt.ralo)

    if opt.rahi is not None:
        r = np.deg2rad(opt.rahi)
        xx = np.cos(r)
        yy = np.sin(r)
        crosscut = x * yy - y * xx
        T.cut(T.cross <= crosscut)
        print('Cut to', len(T), 'with RA <', opt.rahi)
    if opt.declo is None:
        opt.declo = T.dec.min()
    if opt.dechi is None:
        opt.dechi = T.dec.max()
    if opt.ralo is None:
        opt.ralo = T.ra[np.argmin(T.cross)]
    if opt.rahi is None:
        opt.rahi = T.ra[np.argmax(T.cross)]
    T.delete_column('cross')

    print('RA range:', opt.ralo, opt.rahi)
    print('Dec range:', opt.declo, opt.dechi)

    x = np.mean([np.cos(np.deg2rad(r)) for r in (opt.ralo, opt.rahi)])
    y = np.mean([np.sin(np.deg2rad(r)) for r in (opt.ralo, opt.rahi)])
    midra = np.rad2deg(np.arctan2(y, x))
    midra += 360. * (midra < 0)
    middec = (opt.declo + opt.dechi) / 2.

    print('RA,Dec center:', midra, middec)

    pixscale = 2.75 / 3600.
    H = (opt.dechi - opt.declo) / pixscale
    dra = 2. * min(np.abs(midra - opt.ralo), np.abs(midra - opt.rahi))
    W = dra * np.cos(np.deg2rad(middec)) / pixscale

    margin = 5
    W = int(W) + margin * 2
    H = int(H) + margin * 2
    print('W,H', W, H)
    targetwcs = Tan(midra, middec, (W + 1) / 2., (H + 1) / 2.,
                    -pixscale, 0., 0., pixscale, float(W), float(H))
    #print('Target WCS:', targetwcs)

    ra0, dec0 = targetwcs.pixelxy2radec(0.5, 0.5)
    ra1, dec1 = targetwcs.pixelxy2radec(W + 0.5, H + 0.5)
    roiradecbox = [ra0, ra1, dec0, dec1]
    #print('ROI RA,Dec box', roiradecbox)

    tiles = unwise_tiles_touching_wcs(targetwcs)
    print('Cut to', len(tiles), 'unWISE tiles')

    disable_galaxy_cache()

    cols = T.get_columns()
    all_ptsrcs = not('type' in cols)
    if not all_ptsrcs:
        assert('shapeexp' in cols)
        assert('shapedev' in cols)
        assert('fracdev' in cols)

    wanyband = 'w'

    print('Creating Tractor catalog...')
    cat = []
    for i, t in enumerate(T):
        pos = RaDecPos(t.ra, t.dec)
        flux = NanoMaggies(**{wanyband: 1.})
        if all_ptsrcs:
            cat.append(PointSource(pos, flux))
            continue

        tt = t.type.strip()
        if tt in ['PTSRC', 'STAR', 'S']:
            cat.append(PointSource(pos, flux))
        elif tt in ['EXP', 'E']:
            shape = EllipseE(*t.shapeexp)
            cat.append(ExpGalaxy(pos, flux, shape))
        elif tt in ['DEV', 'D']:
            shape = EllipseE(*t.shapedev)
            cat.append(DevGalaxy(pos, flux, shape))
        elif tt in ['COMP', 'C']:
            eshape = EllipseE(*t.shapeexp)
            dshape = EllipseE(*t.shapedev)
            cat.append(FixedCompositeGalaxy(pos, flux, t.fracdev,
                                            eshape, dshape))
        else:
            print('Did not understand row', i, 'of input catalog:')
            t.about()
            assert(False)

    W = unwise_forcedphot(cat, tiles, roiradecbox=roiradecbox,
                          bands=opt.bands, unwise_dir=opt.unwise_dir,
                          use_ceres=opt.ceres, ceres_block=opt.ceresblock,
                          save_fits=opt.save_fits, ps=ps)
    W.writeto(outfn)


if __name__ == '__main__':
    main()
    sys.exit(0)

    # T = fits_table()
    # T.ra = np.arange(0., 10.)
    # T.dec = np.arange(len(T))
    # fn = 'x.fits'
    # outfn = 'out.fits'
    # T.writeto(fn)
    # prog = sys.argv[0]
    # sys.argv = [prog, fn, outfn]
    # main()
    #
    # T = fits_table()
    # T.ra = np.arange(350., 371.) % 360
    # T.dec = np.arange(len(T))
    # T.writeto(fn)
    # sys.argv = [prog, fn, outfn]
    # main()
    #
    # T = fits_table()
    # T.ra = np.arange(350., 371.) % 360
    # T.dec = np.arange(len(T))
    # T.writeto(fn)
    # sys.argv = [prog, fn, outfn, '-r', '355', '-R', 9]
    # main()
