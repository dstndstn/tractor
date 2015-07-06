import sys

from common import *
from desi_common import *


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <decam-image-filename> <decam-HDU> <catalog.fits> <output-image.fits>')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    parser.add_option('--nmgy', action='store_true', default=False,
                      help='Render image in nanomaggy units, not counts')
    parser.add_option('--no-sky', action='store_true', default=False,
                      help='Do not add sky model (default is to add sky estimate back in)')
    parser.add_option('--invvar', action='store_true', default=False,
                      help='Write inverse-variance image as second HDU')
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
    # FIXME -- pack ellipse parameters back into arrays
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T
    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev_ivar = np.vstack((T.shapedev_r_ivar, T.shapedev_e1_ivar,
                                 T.shapedev_e2_ivar)).T
    T.shapeexp_ivar = np.vstack((T.shapeexp_r_ivar, T.shapeexp_e1_ivar,
                                 T.shapeexp_e2_ivar)).T
    cat,invvars = read_fits_catalog(T, invvars=True)
    print 'Got cat:', cat

    T = exposure_metadata([filename], hdus=[hdu])
    print 'Metadata:'
    T.about()

    decals = Decals()
    im = DecamImage(decals, T[0])
    tim = im.get_tractor_image(slc=zoomslice, nanomaggies=opt.nmgy,
                               subsky=opt.no_sky, const2psf=True)
    print 'Got tim:', tim

    #if opt.nmgy:
    #    tim.photocal = 
    
    # from tractor.psfex import CachingPsfEx
    # tim.psfex.radius = 20
    # tim.psfex.fitSavedData(*tim.psfex.splinedata)
    # tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)

    print 'Rendering model image...'
    tr = Tractor([tim], cat)
    mod = tr.getModelImage(0)
    print 'mod range', mod.min(), mod.max()

    print 'Creating FITS header...'
    hdr = tim.getFitsHeader()
    tim.getStandardFitsHeader(header=hdr)

    if opt.nmgy:
        scale = tim.zpscale
    else:
        scale = 1.
    hdr.add_record(dict(name='ZPSCALE', value=scale,
                        comment='Scaling to get to image counts'))
    if opt.no_sky:
        sky = tim.midsky
    else:
        sky = 0.
    hdr.add_record(dict(name='SKYLEVEL', value=sky,
                        comment='Sky level to add to model'))

    fits = fitsio.FITS(outfn, 'rw', clobber=True)
    fits.write(None, header=hdr)
    fits.write(mod)
    if opt.invvar:
        fits.write(tim.getInvvar())
    fits.close()

    #fitsio.write(outfn, mod, header=hdr, clobber=True)
    print 'Wrote model to', outfn
    
