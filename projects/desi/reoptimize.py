import sys
from common import *
from desi_common import *


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <catalog.fits> <output-catalog.fits>')

    #parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    #parser.add_option('--no-ceres', action='store_false', default=True, dest='ceres', help='Do not use Ceres optimiziation engine (use scipy)')

    parser.add_option('--image', action='append', help='Include image filename+HDU in fitting (can be repeated)')
    parser.add_option('--images', help='Use a FITS table of images to include in fitting (like "decals-ccds.fits")')

    parser.add_option('--sources', action='store_true', help='Fit sources?')
    parser.add_option('--sky', action='store_true', help='Fit sky?')
    parser.add_option('--astrom', action='store_true', help='Fit astrometry?')
    parser.add_option('--psf', action='store_true', help='Fit PSF model?')
    parser.add_option('--photom', action='store_true', help='Fit photometric cal?')

    opt,args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)

    catfn = args[0]
    outfn = args[1]

    # zoomslice = None
    # if opt.zoom is not None:
    #     (x0,x1,y0,y1) = opt.zoom
    #     zoomslice = (slice(y0,y1), slice(x0,x1))

    C = fits_table(catfn)
    cat,invvars = read_fits_catalog(C, invvars=True)
    print 'Got cat:', cat

    TT = []
    if opt.images:
        T = fits_table(opt.images)
        print len(T), 'ccds in', opt.images
        TT.append(T)
    for fnhdu in opt.image:
        try:
            i = fnhdu.rindex('+')
        except:
            print 'Warning: expected FILENAME+HDU in', fn
            raise

        fn = fnhdu[:i]
        hdu = int(fnhdu[i+1:])
        print 'Filename', fn, 'HDU', hdu

        T = exposure_metadata([fn], hdus=[hdu])
        TT.append(T)
    T = merge_tables(TT)
    print 'Total of', len(T), 'CCDs'

    decals = Decals()
    tims = []
    for t in T:
        im = DecamImage(t)
        tim = im.get_tractor_image(decals) #, slc=zoomslice)
        print 'Got tim:', tim
        from tractor.psfex import CachingPsfEx
        tim.psfex.radius = 20
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)
        tims.append(tim)

    tr = Tractor(tims, cat)

    tr.thawAllRecursive()

    if not opt.sources:
        tr.freezeParam('catalog')

    if not (opt.sky or opt.astrom or opt.psf or opt.photom):
        tr.freezeParams('images')
    else:
        but = []
        if opt.sky:
            but.append('sky')
        if opt.psf:
            but.append('psf')
        if opt.astrom:
            but.append('wcs')
        if opt.photom:
            but.append('photocal')
            
        for tim in tims:
            tim.freezeAllBut(*but)

    print 'Parameters to fit:'
    tr.printThawedParams()

    # kwa = {}
    # if opt.ceres:
    #     B = 8
    #     kwa.update(use_ceres=True, BW=B, BH=B)
    #R = tr.optimize_forced_photometry(variance=True, **kwa)

    for i in range(50):
        print 'Optimizing...'
        dlnp,x,alpha,variance = tr.optimize(shared_params=False, variance=True)
        if dlnp < 0.1:
            break
        print 'dlnp', dlnp



    if opt.sources:
        hdr = None
        bands = 'grz'
        fs=None

        # FIXME
        invvars = None

        TT = C.copy()
        TT.about()
        
        tr.thawAllRecursive()
        cat = tr.catalog
        T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs)
        T2.writeto(outfn, header=hdr)
        print 'Wrote', outfn

    if opt.sky:
        for tim in tims:
            hdr = fitsio.FITSHDR()

            sky = tim.getSky()
            tt = type(sky)
            sky_type = '%s.%s' % (tt.__module__, tt.__name__)
            prefix = ''
            hdr.add_record(dict(name=prefix + 'SKY', value=sky_type,
                                comment='Sky class'))
            sky.toFitsHeader(hdr,  prefix + 'SKY_')

            print 'Header:', hdr

            fn = tim.imobj.skyfn.replace(decals_dir, outfn)
            print 'Output filename', fn
            try:
                os.makedirs(os.path.dirname(fn))
            except:
                pass
            fitsio.write(fn, None, header=hdr)
            print 'Wrote', fn
            
