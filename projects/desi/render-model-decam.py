import sys

from common import *
from desi_common import *


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <decam-image-filename> <decam-HDU> <catalog.fits> <output-image.fits>')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
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

    print 'Rendering model image...'
    tr = Tractor([tim], cat)
    mod = tr.getModelImage(0)
    print 'mod range', mod.min(), mod.max()
    fitsio.write(outfn, mod, clobber=True)
    print 'Wrote model to', outfn
    
