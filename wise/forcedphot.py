import sys

from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

from astrometry.util.fits import *
from astrometry.util.ttime import *

#from wise.unwise import *
from unwise import *

def main():
    import optparse
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

    
    #parser.add_option('--ellipses', action='store_true',
    #                  help='Assume catalog shapes are ellipse descriptions (not r,ab,phi)')
    
    # parser.add_option('--ra', help='Center RA')
    # parser.add_option('--dec', help='Center Dec')
    # parser.add_option('--width', help='Degrees width (in RA*cos(Dec))')
    # parser.add_option('--height', help='Degrees height (Dec)')
    opt,args = parser.parse_args()
    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)

    if len(opt.bands) == 0:
        opt.bands = [1,2]
    # Allow specifying bands like "123"
    bb = []
    for band in opt.bands:
        for s in str(band):
            bb.append(int(s))
    opt.bands = bb
    print 'Bands', opt.bands

    infn,outfn = args

    
    T = fits_table(infn)
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
    midra += 360.*(midra < 0)
    xx = np.cos(r)
    yy = np.sin(r)
    T.cross = x * yy - y * xx
    minra = T.ra[np.argmin(T.cross)]
    maxra = T.ra[np.argmax(T.cross)]
    #print 'Mid RA:', midra
    #print 'min ra:', minra
    #print 'max ra:', maxra

    if opt.ralo is not None:
        r = np.deg2rad(opt.ralo)
        xx = np.cos(r)
        yy = np.sin(r)
        crosscut = x * yy - y * xx
        T.cut(T.cross >= crosscut)
        print 'Cut to', len(T), 'with RA >', opt.ralo

    if opt.rahi is not None:
        r = np.deg2rad(opt.rahi)
        xx = np.cos(r)
        yy = np.sin(r)
        crosscut = x * yy - y * xx
        T.cut(T.cross <= crosscut)
        print 'Cut to', len(T), 'with RA <', opt.rahi
    if opt.declo is None:
        opt.declo = T.dec.min()
    if opt.dechi is None:
        opt.dechi = T.dec.max()
    if opt.ralo is None:
        opt.ralo = T.ra[np.argmin(T.cross)]
    if opt.rahi is None:
        opt.rahi = T.ra[np.argmax(T.cross)]
    T.delete_column('cross')
    
    x = np.mean([np.cos(np.deg2rad(r)) for r in (opt.ralo, opt.rahi)])
    y = np.mean([np.sin(np.deg2rad(r)) for r in (opt.ralo, opt.rahi)])
    midra = np.rad2deg(np.arctan2(y, x))
    midra += 360.*(midra < 0)
    middec = (opt.declo + opt.dechi) / 2.

    pixscale = 2.75/3600.
    H = (opt.dechi - opt.declo) / pixscale
    dra = min(np.abs(midra - opt.ralo), np.abs(midra - opt.rahi))
    W = dra / pixscale / np.cos(np.deg2rad(middec))

    print 'W,H', W,H
    targetwcs = Tan(midra, middec, (int(W)+1)/2., (int(H)+1)/2.,
                    -pixscale, 0., 0., pixscale, float(int(W)), float(int(H)))
    W = unwise_tiles_touching_wcs(targetwcs)
    print 'Cut to', len(W), 'tiles'

    disable_galaxy_cache()

    cols = T.get_columns()
    all_ptsrcs = not('type' in cols)
    if not all_ptsrcs:
        assert('shape_exp' in cols)
        assert('shape_dev' in cols)
        assert('fracdev' in cols)

    wanyband = 'w'
        
    cat = []
    for i,t in enumerate(T):
        pos = RaDecPos(t.ra, t.dec)
        flux = NanoMaggies(**{wanyband: 1.})
        if all_ptsrcs:
            cat.append(PointSource(pos, flux))
            continue

        tt = t.type.strip()
        if tt in ['PTSRC', 'STAR', 'S']:
            cat.append(PointSource(pos, flux))
        elif tt in ['EXP', 'E']:
            shape = EllipseE(*t.shape_exp)
            cat.append(ExpGalaxy(pos, flux, shape))
        elif tt in ['DEV', 'D']:
            shape = EllipseE(*t.shape_dev)
            cat.append(DevGalaxy(pos, flux, shape))
        elif tt in ['COMP', 'C']:
            eshape = EllipseE(*t.shape_exp)
            dshape = EllipseE(*t.shape_dev)
            cat.append(FixedCompositeGalaxy(pos, flux, t.fracdev,
                                            eshape, dshape))
        else:
            print 'Did not understand row', i, 'of input catalog:'
            t.about()
            assert(False)


    
            
    for band in opt.bands:
        print 'Band', band
        wband = 'w%i' % band
        for tile in W:

            flux_invvars = np.zeros(len(cat))
            fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix', 'pronexp']
            fitstats = dict([(k, np.zeros(len(cat))) for k in fskeys])


            tim = get_unwise_tractor_image(opt.unwise_dir, tile.coadd_id, band,
                                           bandname=wanyband)

            # Select sources in play.
            wcs = tim.wcs.wcs
            H,W = tim.shape
            ok,T.x,T.y = wcs.radec2pixelxy(T.ra, T.dec)
            T.x = (T.x - 1.).astype(np.float32)
            T.y = (T.y - 1.).astype(np.float32)
            margin = 20.
            I = np.flatnonzero((T.x >= -margin) * (T.x < W+margin) *
                               (T.y >= -margin) * (T.y < H+margin))

            inbox = ((T.x[I] >= -0.5) * (T.x[I] < (W-0.5)) *
                     (T.y[I] >= -0.5) * (T.y[I] < (H-0.5)))
            
            # Source indices inside the box
            srci = I[inbox]
            # Source indices in the margins
            margi = I[np.logical_not(inbox)]

            # sources in the box
            subcat = [cat[i] for i in srci]

            # include *copies* of sources in the margins
            # (that way we automatically don't save the results)
            subcat.extend([cat[i].copy() for i in margi])
            assert(len(subcat) == len(I))

            #### FIXME -- set source radii, ...?

            minsb = None
            fitsky = False
            
            ## Look in image and set radius based on peak height??


            
            tractor = Tractor([tim], subcat)
            tractor.disable_cache()
            tractor.freezeParamsRecursive('*')
            tractor.thawPathsTo(wanyband)

            
            kwa = dict(fitstat_extras=[('pronexp', [tim.nims])])
            t0 = Time()
            R = tractor.optimize_forced_photometry(
                minsb=minsb, mindlnp=1., sky=fitsky, fitstats=True, 
                variance=True, shared_params=False,
                use_ceres=opt.ceres, BW=opt.ceresblock, BH=opt.ceresblock,
                wantims=wantims, **kwa)
            print 'That took', Time()-t0

            if opt.ceres:
                term = R.ceres_status['termination']
                print 'Ceres termination status:', term
                # Running out of memory can cause failure to converge
                # and term status = 2.
                # Fail completely in this case.
                if term != 0:
                    raise RuntimeError('Ceres terminated with status %i' % term)

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

            if len(srci):
                # Save fit stats
                flux_invvars[srci] = IV[:len(srci)]
                for k in fskeys:
                    x = getattr(fs, k)
                    fitstats[k][srci] = np.array(x)

        # Note, this is *outside* the loop over tiles.
        nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
        nm_ivar = flux_invvars
        T.set(wband + '_nanomaggies', nm.astype(np.float32))
        T.set(wband + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
        dnm = np.zeros(len(nm_ivar), np.float32)
        okiv = (nm_ivar > 0)
        dnm[okiv] = (1./np.sqrt(nm_ivar[okiv])).astype(np.float32)
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
            
        T.set(wband + '_mag', mag)
        T.set(wband + '_mag_err', dmag)
        for k in fskeys:
            T.set(wband + '_' + k, fitstats[k].astype(np.float32))

    T.writeto(outfn)
    
if __name__ == '__main__':
    #main()

    T = fits_table()
    T.ra = np.arange(0., 10.)
    T.dec = np.arange(len(T))
    fn = 'x.fits'
    outfn = 'out.fits'
    T.writeto(fn)
    prog = sys.argv[0]
    sys.argv = [prog, fn, outfn]
    main()

    T = fits_table()
    T.ra = np.arange(350., 371.) % 360
    T.dec = np.arange(len(T))
    T.writeto(fn)
    sys.argv = [prog, fn, outfn]
    main()

    T = fits_table()
    T.ra = np.arange(350., 371.) % 360
    T.dec = np.arange(len(T))
    T.writeto(fn)
    sys.argv = [prog, fn, outfn, '-r', '355', '-R', 9]
    main()


    
