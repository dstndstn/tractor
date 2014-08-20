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

if __name__ == '__main__':
    ra,dec = 351.56, 0.33
    #W,H = 2048,4096
    #W,H = 4096,4096
    W,H = 4096,2048
    pixscale = 0.262 / 3600.

    ps = PlotSequence('decals')
    
    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    W, H)

    #data/decam/proc/20130804/C01/gband/dec095705.01.p.w.fits

    nearest = True

    coims = []
    
    for band in 'grz':
        fns = glob('data/decam/proc/2013080?/C01/%sband/dec??????.??.p.w.fits' %
                   band)
        print 'filenames:', fns

        coimg = np.zeros((H,W), np.float32)
        cowt  = np.zeros((H,W), np.float32)
        
        for fn in fns:
            wt = 1.
            img = fitsio.read(fn)
            print 'Image', img.shape
            wcs = Sip(fn)
            print 'WCS', wcs
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
                coimg[Yo,Xo] += img[Yi,Xi]
            else:
                coimg[Yo,Xo] += rim
            cowt [Yo,Xo] += wt
            
        coadd = coimg / np.maximum(1e-16, wt)

        plt.figure(figsize=(12,6))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
        plt.clf()

        mn,mx = [np.percentile(coadd[cowt > 0], p) for p in [25,99.5]]

        plt.imshow(coadd, interpolation='nearest', origin='lower',
                   cmap='gray', vmin=mn, vmax=mx)
        ps.savefig()

        coims.append((coadd, mn, mx))


    rgb = np.zeros((H,W,3))
    im,mn,mx = coims[2]
    rgb[:,:,0] = (im - mn) / (mx - mn)
    im,mn,mx = coims[1]
    rgb[:,:,1] = (im - mn) / (mx - mn)
    im,mn,mx = coims[0]
    rgb[:,:,2] = (im - mn) / (mx - mn)

    plt.clf()
    plt.imshow(np.clip(rgb, 0.,1.), interpolation='nearest', origin='lower')
    ps.savefig()
    
