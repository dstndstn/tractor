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
        C = decals.ccds_touching_wcs(targetwcs)
        # Sort by band
        II = []
        C.cut(np.hstack([np.flatnonzero(C.filter == band) for band in bands]))
        ims = []
        for t in C:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(t)
            ims.append(im)
        # Read images, clip to ROI
        mock_psf = True
        targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                             [(1,1),(W,1),(W,H),(1,H),(1,1)]])
        args = [(im, decals, targetrd, mock_psf) for im in ims]
        tims = mp.map(read_one_tim, args)

        print 'Rendering detection maps...'
        detmaps, detivs = detection_maps(tims, targetwcs, bands, mp)

        # List the SED-matched filters to run
        # single-band filters
        SEDs = []
        for i,band in enumerate(bands):
            sed = np.zeros(len(bands))
            sed[i] = 1.
            SEDs.append((band, sed))
        assert(bands == 'grz')
        SEDs.append(('Flat', (1.,1.,1.)))
        SEDs.append(('Red', (2.5, 1.0, 0.4)))

        # all source positions
        xx = T.itx
        yy = T.ity

        hot = np.zeros((H,W), np.float32)
        for sedname,sed in SEDs:
            print 'SED', sedname
            if plots:
                pps = ps
            else:
                pps = None
            sedsn,px,py = sed_matched_detection(
                sedname, sed, detmaps, detivs, bands, xx, yy, ps=pps)
            print len(px), 'new peaks'
            hot = np.maximum(hot, sedsn)
            xx = np.append(xx, px)
            yy = np.append(yy, py)

        # New peaks:
        peakx = xx[len(T):]
        peaky = yy[len(T):]

        # Add sources for the new peaks we found
        # make their initial fluxes ~ 5-sigma
        fluxes = dict([(b,[]) for b in bands])
        for tim in tims:
            psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
            fluxes[tim.band].append(5. * tim.sig1 / psfnorm)
        fluxes = dict([(b, np.mean(fluxes[b])) for b in bands])
        pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
        print 'Adding', len(pr), 'new sources'
        # Also create FITS table for new sources
        Tnew = fits_table()
        Tnew.ra  = pr
        Tnew.dec = pd
        Tnew.tx = peakx
        Tnew.ty = peaky
        Tnew.itx = np.clip(np.round(Tnew.tx).astype(int), 0, W-1)
        Tnew.ity = np.clip(np.round(Tnew.ty).astype(int), 0, H-1)
        for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
            cat.append(PointSource(RaDecPos(r,d),
                                   NanoMaggies(order=bands, **fluxes)))
    
        print 'Existing source table:'
        T.about()
        print 'New source table:'
        Tnew.about()
    
        T = merge_tables([T, Tnew], columns='fillzero')

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
    
