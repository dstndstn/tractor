import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import fitsio

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.resample import *
from astrometry.util.starutil_numpy import *

'''
imcopy /global/homes/d/dstn/cosmo/data/staging/decam/CP20140815/c4d_140816_001323_ooi_g_v1.fits.fz+27 1.fits

solve-field --config ~/desi-dstn/sdss-astrometry-index/r2/cfg -v -D . --temp-dir tmp --ra 244 --dec 8 --radius 1 --continue --no-plots --sextractor-config /project/projectdirs/desi/imaging/code/cats/CS82.sex 1.fits -X x_image -Y y_image -s flux_auto

funpack -E 27 -O flag.fits /global/homes/d/dstn/cosmo/data/staging/decam/CP20140815/c4d_140816_001323_ood_g_v1.fits.fz

an-fitstopnm -i 1.fits | pnmscale -reduce 4 | pnmtojpeg > 1-4.jpg

solve-field --config /data/INDEXES/sdss-astrometry-index/r2/cfg -v -D . --ra 244 --dec 8 --radius 1 --continue -X x_image -Y y_image -s flux_auto 1.axy --plot-bg 1.jpg --plot-scale 0.25

'''

if __name__ == '__main__':

    B = fits_table('bricks.fits')
    B.index = np.arange(len(B))

    ii = 377305

    ra,dec = B.ra[ii], B.dec[ii]
    W,H = 3600,3600
    pixscale = 0.27 / 3600.
    
    # ra,dec = 351.56, 0.33
    # W,H = 4096,2048
    # pixscale = 0.262 / 3600.

    ps = PlotSequence('decals')

    print (ra, dec, W/2.+0.5, H/2.+0.5, -pixscale, 0., 0., pixscale, float(W), float(H))

    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    float(W), float(H))

    nearest = True

    coims = []

    T = fits_table('ccds.fits')
    sz = 0.25
    T.cut(np.abs(T.dec - dec) < sz)
    T.cut(degrees_between(T.ra, T.dec, ra, dec) < sz)
    print len(T), 'CCDs nearby'
    
    for band in 'grz':
        #fns = glob('data/decam/proc/2013080?/C01/%sband/dec??????.??.p.w.fits' %
        #           band)
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        print 'filenames:', TT.filename

        coimg = np.zeros((H,W), np.float32)
        cowt  = np.zeros((H,W), np.float32)
        coimgm = np.zeros((H,W), np.float32)
        cowtm  = np.zeros((H,W), np.float32)
        
        for fn,hdu in zip(TT.filename, TT.hdu):

            img = fitsio.FITS(fn)[hdu].read()
            #img = fitsio.read(fn)
            #wcsfn = fn.replace('.fits','.cat.wcs')
            wcsfn = fn
            print 'Image', img.shape
            print 'WCS filename', wcsfn
            wcs = Sip(wcsfn)
            print 'WCS', wcs

            dqfn = fn.replace('_ooi_', '_ood_')
            print 'DQ', dqfn
            dq = fitsio.FITS(dqfn)[hdu].read()
            print 'DQ', dq.shape, dq.dtype

            print len(dq.ravel()), 'DQ pixels'
            print 'Unique vals:', np.unique(dq)
            print sum(dq == 0), 'have value 0'
            
            wtfn = fn.replace('_ooi_', '_oow_')
            print 'Weight', wtfn
            wt = fitsio.FITS(wtfn)[hdu].read()
            print 'WT', wt.shape, wt.dtype
            
            if False:
                # Nugent's bad pixel masks
                bpfn = fn.replace('.fits', '.bpm.fits')
                print 'Bad pixel mask', bpfn
                mask = fitsio.read(bpfn)
                mask = (mask == 0)
            
            L = 2
            try:
                if nearest:
                    lims = []
                else:
                    lims = [img]
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, wcs, lims, L)
            except OverlapError:
                print 'No overlap'
                continue

            if nearest:
                coimg[Yo,Xo]  += img[Yi,Xi]
                coimgm[Yo,Xo] += img[Yi,Xi] * wt[Yi,Xi] * (dq[Yi,Xi] == 0)
                #mask[Yi,Xi]
            else:
                coimg[Yo,Xo]  += rim
                coimgm[Yo,Xo] += rim * wt[Yi,Xi] * (dq[Yi,Xi] == 0)
                #mask[Yi,Xi]

            cowt [Yo,Xo] += 1.
            cowtm[Yo,Xo] += wt[Yi,Xi] * (dq[Yi,Xi] == 0)
            #wt * mask[Yi,Xi]
            
        coadd = coimg / np.maximum(1e-16, cowt)
        coaddm = coimgm / np.maximum(1e-16, cowtm)

        plt.figure(figsize=(42,21))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        plt.clf()

        mn,mx = [np.percentile(coadd[cowt > 0], p) for p in [25,99.5]]

        plt.imshow(coadd, interpolation='nearest', origin='lower',
                   cmap='gray', vmin=mn, vmax=mx)
        plt.title('%s band stack' % band)
        ps.savefig()

        plt.clf()
        plt.imshow(coaddm, interpolation='nearest', origin='lower',
                   cmap='gray', vmin=mn, vmax=mx)
        plt.title('%s band stack; masked' % band)
        ps.savefig()

        coims.append((coadd, mn, mx))
        coims.append((coaddm, mn, mx))


    rgb = np.zeros((H,W,3), np.float32)
    for plane,(im,mn,mx) in enumerate([coims[4],coims[2],coims[0]]):
        rgb[:,:,plane] = (im - mn) / (mx - mn)

    plt.clf()
    plt.imshow(np.clip(rgb, 0.,1.), interpolation='nearest', origin='lower')
    plt.title('zrg stack')
    ps.savefig()

    for plane,(im,mn,mx) in enumerate([coims[5],coims[3],coims[1]]):
        rgb[:,:,plane] = (im - mn) / (mx - mn)
    plt.clf()
    plt.imshow(np.clip(rgb, 0.,1.), interpolation='nearest', origin='lower')
    plt.title('zrg stack; masked')
    ps.savefig()
    
