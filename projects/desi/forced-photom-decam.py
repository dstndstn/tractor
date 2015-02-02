import sys

from common import *
from desi_common import *


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <decam-image-filename> <decam-HDU> <catalog.fits> <output-catalog.fits>')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    parser.add_option('--no-ceres', action='store_false', default=True, dest='ceres', help='Do not use Ceres optimiziation engine (use scipy)')
    opt,args = parser.parse_args()

    if len(args) != 4:
        parser.print_help()
        sys.exit(-1)

    filename = args[0]
    hdu = int(args[1])
    catfn = args[2]
    outfn = args[3]

    zoomslice = None
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        zoomslice = (slice(y0,y1), slice(x0,x1))

    T = fits_table(catfn)
    cat,invvars = read_fits_catalog(T, invvars=True)
    print 'Got cat:', cat

    T = exposure_metadata([filename], hdus=[hdu])
    print 'Metadata:'
    T.about()

    im = DecamImage(T[0])
    decals = Decals()
    tim = im.get_tractor_image(decals, slc=zoomslice)
    print 'Got tim:', tim

    from tractor.psfex import CachingPsfEx
    tim.psfex.radius = 20
    tim.psfex.fitSavedData(*tim.psfex.splinedata)
    tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)

    print 'Forced photom...'

    tr = Tractor([tim], cat)
    tr.freezeParam('images')
    for src in cat:
        src.freezeAllBut('brightness')
        src.getBrightness().freezeAllBut(tim.band)

    kwa = {}
    if opt.ceres:
        B = 8
        kwa.update(use_ceres=True, BW=B, BH=B)

    R = tr.optimize_forced_photometry(variance=True, **kwa)

    T = fits_table()
    T.flux = np.array([src.getBrightness().getFlux(tim.band) for src in cat]).astype(np.float32)
    T.flux_ivar = R.IV.astype(np.float32)
    T.writeto(outfn)
    print 'Wrote', outfn
