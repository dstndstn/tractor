import sys

from common import *
from desi_common import *


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <decam-image-filename> <decam-HDU> <catalog.fits or "DR1"> <output-catalog.fits>')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    parser.add_option('--no-ceres', action='store_false', default=True, dest='ceres', help='Do not use Ceres optimiziation engine (use scipy)')
    parser.add_option('--catalog-path', default='dr1',
                      help='Path to DECaLS DR1 catalogs; default %default, eg, /project/projectdirs/cosmo/data/legacysurvey/dr1')
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

    T = exposure_metadata([filename], hdus=[hdu])
    print 'Metadata:'
    T.about()

    im = DecamImage(T[0])
    decals = Decals()
    tim = im.get_tractor_image(decals, slc=zoomslice, const2psf=True, pvwcs=True)
    print 'Got tim:', tim

    if catfn == 'DR1':

        # How far outside the image to keep objects
        # FIXME -- should be adaptive to object size!
        margin = 20

        from astrometry.libkd.spherematch import *
        B = decals.get_bricks_readonly()
        # MAGIC 0.4 degree search radius =
        # DECam hypot(1024,2048)*0.27/3600 + Brick hypot(0.25, 0.25) ~= 0.35 + margin
        I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, 0.4)
        print len(I), 'bricks nearby'
        bricks = B[I]
        TT = []
        for b in bricks:
            brickwcs = wcs_for_brick(b)
            chipwcs = tim.subwcs

            clip = clip_wcs(chipwcs, brickwcs)
            print 'Clipped chip coordinates:', clip
            if len(clip) == 0:
                continue

            # there is some overlap with this brick... read the catalog.

            fn = os.path.join(opt.catalog_path, 'tractor', b.brickname[:3],
                              'tractor-%s.fits' % b.brickname)
            print 'Reading', fn
            T = fits_table(fn)
            ok,xx,yy = chipwcs.radec2pixelxy(T.ra, T.dec)
            W,H = chipwcs.get_width(), chipwcs.get_height()
            I = np.flatnonzero((xx >= -margin) * (xx <= (W+margin)) *
                               (yy >= -margin) * (yy <= (H+margin)))
            T.cut(I)
            print 'Cut to', len(T), 'sources within image + margin'
            print 'Brick_primary:', np.unique(T.brick_primary)
            T.cut(T.brick_primary)
            print 'Cut to', len(T), 'on brick_primary'
            TT.append(T)
        T = merge_tables(TT)

        T.writeto('cat.fits')

        T.cut((T.out_of_bounds == False) * (T.left_blob == False))

        allbands = 'ugrizY'

        del TT
    else:
        T = fits_table(catfn)

    cat,invvars = read_fits_catalog(T, invvars=True)
    print 'Got cat:', cat

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
