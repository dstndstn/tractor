import os
import sys

import numpy as np
import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import *
from astrometry.util.ttime import *
from tractor import *

from common import *
from desi_common import *
import tractor

# python projects/desi/forced-photom-decam.py decals/images/decam/CP20140810_g_v2/c4d_140816_032035_ooi_g_v2.fits.fz 43 DR1 f.fits

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <decam-image-filename> <decam-HDU> <catalog.fits or "DR1"> <output-catalog.fits>')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 2046 0 4094")')
    parser.add_option('--no-ceres', action='store_false', default=True, dest='ceres', help='Do not use Ceres optimiziation engine (use scipy)')
    parser.add_option('--catalog-path', default='dr1',
                      help='Path to DECaLS DR1 catalogs; default %default, eg, /project/projectdirs/cosmo/data/legacysurvey/dr1')
    parser.add_option('--plots', default=None, help='Create plots; specify a base filename for the plots')
    opt,args = parser.parse_args()

    if len(args) != 4:
        parser.print_help()
        sys.exit(-1)

    Time.add_measurement(MemMeas)
    t0 = Time()
    

    filename = args[0]
    hdu = int(args[1])
    catfn = args[2]
    outfn = args[3]

    if os.path.exists(outfn):
        print 'Ouput file exists:', outfn
        sys.exit(0)

    zoomslice = None
    if opt.zoom is not None:
        (x0,x1,y0,y1) = opt.zoom
        zoomslice = (slice(y0,y1), slice(x0,x1))

    ps = None
    if opt.plots is not None:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(opt.plots)

    T = exposure_metadata([filename], hdus=[hdu])
    print 'Metadata:'
    T.about()

    decals = Decals()
    im = DecamImage(decals, T[0])
    tim = im.get_tractor_image(slc=zoomslice, const2psf=True)
    print 'Got tim:', tim

    if catfn == 'DR1':
        margin = 20
        TT = []
        chipwcs = tim.subwcs
        bricks = bricks_touching_wcs(chipwcs, decals=decals)
        for b in bricks:
            # there is some overlap with this brick... read the catalog.
            fn = os.path.join(opt.catalog_path, 'tractor', b.brickname[:3],
                              'tractor-%s.fits' % b.brickname)
            if not os.path.exists(fn):
                print 'WARNING: catalog', fn, 'does not exist.  Skipping!'
                continue
            print 'Reading', fn
            T = fits_table(fn)
            ok,xx,yy = chipwcs.radec2pixelxy(T.ra, T.dec)
            W,H = chipwcs.get_width(), chipwcs.get_height()
            I = np.flatnonzero((xx >= -margin) * (xx <= (W+margin)) *
                               (yy >= -margin) * (yy <= (H+margin)))
            T.cut(I)
            print 'Cut to', len(T), 'sources within image + margin'
            #print 'Brick_primary:', np.unique(T.brick_primary)
            T.cut(T.brick_primary)
            print 'Cut to', len(T), 'on brick_primary'
            T.cut((T.out_of_bounds == False) * (T.left_blob == False))
            print 'Cut to', len(T), 'on out_of_bounds and left_blob'
            TT.append(T)
        T = merge_tables(TT)
        T._header = TT[0]._header
        del TT
        #T.writeto('cat.fits')

        # Fix up various failure modes:
        # FixedCompositeGalaxy(pos=RaDecPos[240.51147402832561, 10.385488075518923], brightness=NanoMaggies: g=(flux -2.87), r=(flux -5.26), z=(flux -7.65), fracDev=FracDev(0.60177207), shapeExp=re=3.78351e-44, e1=9.30367e-13, e2=1.24392e-16, shapeDev=re=inf, e1=-0, e2=-0)
        # -> convert to EXP
        I = np.flatnonzero(np.array([((t.type == 'COMP') and 
                                      (not np.isfinite(t.shapedev_r)))
                                     for t in T]))
        if len(I):
            print 'Converting', len(I), 'bogus COMP galaxies to EXP'
            for i in I:
                T.type[i] = 'EXP'

        # Same thing with the exp component.
        # -> convert to DEV
        I = np.flatnonzero(np.array([((t.type == 'COMP') and 
                                      (not np.isfinite(t.shapeexp_r)))
                                     for t in T]))
        if len(I):
            print 'Converting', len(I), 'bogus COMP galaxies to DEV'
            for i in I:
                T.type[i] = 'DEV'

    else:
        T = fits_table(catfn)

    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T

    cat = read_fits_catalog(T, ellipseClass=tractor.ellipses.EllipseE)
    #print 'Got cat:', cat

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

    if opt.plots is None:
        kwa.update(wantims=False)

    R = tr.optimize_forced_photometry(variance=True, fitstats=True,
                                      shared_params=False, **kwa)

    if opt.plots:
        (data,mod,ie,chi,roi) = R.ims1[0]

        ima = tim.ima
        imchi = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)
        plt.clf()
        plt.imshow(data, **ima)
        plt.title('Data: %s' % tim.name)
        ps.savefig()

        plt.clf()
        plt.imshow(mod, **ima)
        plt.title('Model: %s' % tim.name)
        ps.savefig()

        plt.clf()
        plt.imshow(chi, **imchi)
        plt.title('Chi: %s' % tim.name)
        ps.savefig()


    F = fits_table()
    F.brickid   = T.brickid
    F.brickname = T.brickname
    F.objid     = T.objid

    F.filter  = np.array([tim.band]               * len(T))
    F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(T))
    F.exptime = np.array([tim.primhdr['EXPTIME']] * len(T))

    ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
    F.x = (x-1).astype(np.float32)
    F.y = (y-1).astype(np.float32)

    F.flux = np.array([src.getBrightness().getFlux(tim.band)
                       for src in cat]).astype(np.float32)
    F.flux_ivar = R.IV.astype(np.float32)

    F.fracflux = R.fitstats.profracflux.astype(np.float32)
    F.rchi2    = R.fitstats.prochi2    .astype(np.float32)

    program_name = sys.argv[0]
    version_hdr = get_version_header(program_name, decals.decals_dir)
    # HACK -- print only two directory names + filename of CPFILE.
    fname = os.path.basename(im.imgfn)
    d = os.path.dirname(im.imgfn)
    d1 = os.path.basename(d)
    d = os.path.dirname(d)
    d2 = os.path.basename(d)
    fname = os.path.join(d2, d1, fname)
    print 'Trimmed filename to', fname
    #version_hdr.add_record(dict(name='CPFILE', value=im.imgfn, comment='DECam comm.pipeline file'))
    version_hdr.add_record(dict(name='CPFILE', value=fname, comment='DECam comm.pipeline file'))
    version_hdr.add_record(dict(name='CPHDU', value=im.hdu, comment='DECam comm.pipeline ext'))
    version_hdr.add_record(dict(name='CAMERA', value='DECam', comment='Dark Energy Camera'))
    version_hdr.add_record(dict(name='EXPNUM', value=im.expnum, comment='DECam exposure num'))
    version_hdr.add_record(dict(name='CCDNAME', value=im.extname, comment='DECam CCD name'))
    version_hdr.add_record(dict(name='FILTER', value=tim.band, comment='Bandpass of this image'))
    version_hdr.add_record(dict(name='EXPOSURE', value='decam-%s-%s' % (im.expnum, im.extname), comment='Name of this image'))
    
    keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME']
    for key in keys:
        if key in tim.primhdr:
            version_hdr.add_record(dict(name=key, value=tim.primhdr[key]))

    hdr = fitsio.FITSHDR()

    units = {'mjd':'sec', 'exptime':'sec', 'flux':'nanomaggy',
             'flux_ivar':'1/nanomaggy^2'}
    columns = F.get_columns()
    for i,col in enumerate(columns):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    outdir = os.path.dirname(outfn)
    if len(outdir):
        trymakedirs(outdir)
    fitsio.write(outfn, None, header=version_hdr, clobber=True)
    F.writeto(outfn, header=hdr, append=True)
    print 'Wrote', outfn

    print 'Finished forced phot:', Time()-t0

