if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np

#from runbrick import *
from common import *
from tractor import *


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-b', '--brick', type=int, help='Brick ID to run: default %default',
                      default=377306)
    parser.add_option('-s', '--sed-matched', action='store_true', default=False,
                      help='Run SED-matched filter?')
    parser.add_option('--bands', default='grz', help='Bands to retrieve')
    parser.add_option('-o', '--output', help='Output filename for catalog',
                      default='initial-cat.fits')
    parser.add_option('--threads', type=int, help='Run multi-threaded')
    parser.add_option('-W', type=int, default=3600, help='Target image width (default %default)')
    parser.add_option('-H', type=int, default=3600, help='Target image height (default %default)')

    if not (('BOSS_PHOTOOBJ' in os.environ) and ('PHOTO_RESOLVE' in os.environ)):
        print '$BOSS_PHOTOOBJ and $PHOTO_RESOLVE not set -- on NERSC, you can do:'
        print '  export BOSS_PHOTOOBJ=/project/projectdirs/cosmo/data/sdss/pre13/eboss/photoObj.v5b'
        print '  export PHOTO_RESOLVE=/project/projectdirs/cosmo/data/sdss/pre13/eboss/resolve/2013-07-29'
        print 'To read SDSS files from the local filesystem rather than downloading them.'
        print
        
    opt,args = parser.parse_args()
    brickid = opt.brick
    bands = opt.bands
    if opt.threads and opt.threads > 1:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)
    else:
        mp = multiproc()

    ps = None
    plots = False

    decals = Decals()
    brick = decals.get_brick(brickid)
    print 'Chosen brick:'
    brick.about()
    targetwcs = wcs_for_brick(brick, W=opt.W, H=opt.H)
    W,H = targetwcs.get_width(), targetwcs.get_height()

    # Read SDSS sources
    cat,T = get_sdss_sources(bands, targetwcs)

    if opt.sed_matched:
        # Read images
        tims = decals.tims_touching_wcs(targetwcs, mp, mock_psf=True, bands=bands)
        print 'Rendering detection maps...'
        detmaps, detivs = detection_maps(tims, targetwcs, bands, mp)

        SEDs = sed_matched_filters(bands)
        Tnew,newcat,nil = run_sed_matched_filters(SEDs, bands, detmaps, detivs,
                                                  (T.itx,T.ity), targetwcs)
        T = merge_tables([T,Tnew], columns='fillzero')
        cat.extend(newcat)


    from desi_common import prepare_fits_catalog
    TT = T.copy()
    for k in ['itx','ity','index']:
        TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob']:
            TT.rename(col, 'sdss_%s' % col)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.objid   = np.arange(len(TT)).astype(np.int32)

    invvars = None
    hdr = None
    fs = None
    
    cat.thawAllRecursive()
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs)
    # Unpack shape columns
    T2.shapeExp_r = T2.shapeExp[:,0]
    T2.shapeExp_e1 = T2.shapeExp[:,1]
    T2.shapeExp_e2 = T2.shapeExp[:,2]
    T2.shapeDev_r = T2.shapeExp[:,0]
    T2.shapeDev_e1 = T2.shapeExp[:,1]
    T2.shapeDev_e2 = T2.shapeExp[:,2]
    T2.shapeExp_r_ivar  = T2.shapeExp_ivar[:,0]
    T2.shapeExp_e1_ivar = T2.shapeExp_ivar[:,1]
    T2.shapeExp_e2_ivar = T2.shapeExp_ivar[:,2]
    T2.shapeDev_r_ivar  = T2.shapeExp_ivar[:,0]
    T2.shapeDev_e1_ivar = T2.shapeExp_ivar[:,1]
    T2.shapeDev_e2_ivar = T2.shapeExp_ivar[:,2]
    
    T2.writeto(opt.output)
    print 'Wrote', opt.output
    
