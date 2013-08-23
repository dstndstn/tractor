import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.fits import *

ps = PlotSequence('co')

for coadd in ['2195p545']:

    for band in [1]:
        F = fits_table('coadd-%s-w%i-frames.fits' % (coadd,band))

        frame0 = F[0]

        overlaps = np.zeros(len(F))
        for i in range(len(F)):
            ext = F.coextent[i]
            x0,x1,y0,y1 = ext
            poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
            if i == 0:
                poly0 = poly
            clip = clip_polygon(poly, poly0)
            if len(clip) == 0:
                continue
            print 'clip:', clip
            x0,y0 = np.min(clip, axis=0)
            x1,y1 = np.max(clip, axis=0)
            overlaps[i] = (y1-y0)*(x1-x0)
        I = np.argsort(-overlaps)
        for i in I[:5]:
            frame = '%s%03i' % (F.scan_id[i], F.frame_num[i])
            imgfn = '%s-w%i-int-1b.fits' % (frame, band)
            img = fitsio.read(imgfn)

            plo,phi = [np.percentile(img, p) for p in [25,98]]
            ima = dict(interpolation='nearest', origin='lower',
                       vmin=plo, vmax=phi)
            plt.clf()
            plt.imshow(img, **ima)
            plt.title('Image %s W%i' % (frame,band))
            ps.savefig()

        for i in I[:5]:
            frame = '%s%03i' % (F.scan_id[i], F.frame_num[i])
            #maskfn = '%s-w%i-msk-1b.fits.gz' % (frame, band)
            #mask = fitsio.read(maskfn)
            comaskfn = 'coadd-mask-%s-%s-w%i-1b.fits' % (coadd, frame, band)
            comask = fitsio.read(comaskfn)

            #plt.clf()
            #plt.imshow(mask > 0, interpolation='nearest', origin='lower',
            #           vmin=0, vmax=1)
            #plt.axis(ax)
            #plt.title('WISE mask')
            #ps.savefig()

            plt.clf()
            plt.imshow(comask > 0, interpolation='nearest', origin='lower',
                       vmin=0, vmax=1)
            plt.axis(ax)
            plt.title('Coadd mask')
            ps.savefig()


    for frame in ['05579a167']:
        for band in [1]:
            imgfn = '%s-w%i-int-1b.fits' % (frame, band)
            img = fitsio.read(imgfn)
            maskfn = '%s-w%i-msk-1b.fits.gz' % (frame, band)
            mask = fitsio.read(maskfn)
            comaskfn = 'coadd-mask-%s-%s-w%i-1b.fits' % (coadd, frame, band)
            comask = fitsio.read(comaskfn)

            plo,phi = [np.percentile(img, p) for p in [25,98]]
            ima = dict(interpolation='nearest', origin='lower',
                       vmin=plo, vmax=phi)
            ax = [200,700,200,700]
            plt.clf()
            plt.imshow(img, **ima)
            plt.axis(ax)
            plt.title('Image %s W%i' % (frame,band))
            ps.savefig()

            plt.clf()
            plt.imshow(mask > 0, interpolation='nearest', origin='lower',
                       vmin=0, vmax=1)
            plt.axis(ax)
            plt.title('WISE mask')
            ps.savefig()

            plt.clf()
            plt.imshow(comask > 0, interpolation='nearest', origin='lower',
                       vmin=0, vmax=1)
            plt.axis(ax)
            plt.title('Coadd mask')
            ps.savefig()
            

    for band in [1,2]: #,3,4]:
        fni = 'L3a/%s_ab41/%s_ab41-w%i-int-3.fits' % (coadd, coadd, band)
        I = fitsio.read(fni)
        fnj = 'coadd-%s-w%i-img.fits' % (coadd, band)
        J = fitsio.read(fnj)
        fnk = 'coadd-%s-w%i-img-w.fits' % (coadd, band)
        K = fitsio.read(fnk)
        
        plt.clf()
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        imai = dict(interpolation='nearest', origin='lower',
                   vmin=plo, vmax=phi)
        plt.imshow(I, **imai)
        plt.title(fni)
        ps.savefig()

        plt.clf()
        plo,phi = [np.percentile(J, p) for p in [25,99]]
        imaj = dict(interpolation='nearest', origin='lower',
                   vmin=plo, vmax=phi)
        plt.imshow(J, **imaj)
        plt.title(fnj)
        ps.savefig()

        plt.clf()
        plt.imshow(K, **imaj)
        plt.title(fnk)
        ps.savefig()
        
        hi,wi = I.shape
        hj,wj = J.shape

        flo,fhi = 0.45, 0.55

        slcI = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
        slcJ = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)

        plt.clf()
        plt.imshow(I[slcI], **imai)
        plt.title(fni)
        ps.savefig()

        plt.clf()
        plt.imshow(J[slcJ], **imaj)
        plt.title(fnj)
        ps.savefig()

        plt.clf()
        plt.imshow(K[slcJ], **imaj)
        plt.title(fnk)
        ps.savefig()
        
