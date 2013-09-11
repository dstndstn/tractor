import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.fits import *
from astrometry.util.util import Tan

ps = PlotSequence('co')

wisel3 = 'wise-L3'
coadds = 'wise-coadds'

lst = os.listdir(wisel3)
lst.sort()
for band in [1,2,3,4]:
    for l3dir in lst:
        print 'dir', l3dir
        coadd = l3dir.replace('_ab41','')
        l3fn = os.path.join(wisel3, l3dir, '%s-w%i-int-3.fits' % (l3dir, band))
        if not os.path.exists(l3fn):
            print 'Missing', l3fn
            continue
        cofn  = os.path.join(coadds, 'coadd-%s-w%i-img.fits'   % (coadd, band))
        cowfn = os.path.join(coadds, 'coadd-%s-w%i-img-w.fits' % (coadd, band))
        if not os.path.exists(cofn) or not os.path.exists(cowfn):
            print 'Missing', cofn, 'or', cowfn
            continue

        I = fitsio.read(l3fn)
        J = fitsio.read(cofn)
        K = fitsio.read(cowfn)

        hi,wi = I.shape
        hj,wj = J.shape
        flo,fhi = 0.45, 0.55
        slcI = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
        slcJ = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)

        ima = dict(interpolation='nearest', origin='lower', cmap='gray')

        plo,phi = [np.percentile(I, p) for p in [25,99]]
        imai = ima.copy()
        imai.update(vmin=plo, vmax=phi)

        plt.clf()
        plt.imshow(I[slcI], **imai)
        plt.title('WISE team %s' % os.path.basename(l3fn))
        ps.savefig()

        plo,phi = [np.percentile(J, p) for p in [25,99]]
        imaj = ima.copy()
        imaj.update(vmin=plo, vmax=phi)

        plt.clf()
        plt.imshow(J[slcJ], **imaj)
        plt.title('My unweighted %s' % os.path.basename(cofn))
        ps.savefig()

        plt.clf()
        plt.imshow(K[slcJ], **imaj)
        plt.title('My weighted %s' % os.path.basename(cowfn))
        ps.savefig()

                            
sys.exit(0)







for coadd in ['1384p454',
    #'2195p545',
              ]:

    for band in []: #1,2,3,4]: #[1]:
        F = fits_table('wise-coadds/coadd-%s-w%i-frames.fits' % (coadd,band))

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
            #imgfn = '%s-w%i-int-1b.fits' % (frame, band)
            imgfn = F.intfn[i]
            print 'Reading image', imgfn
            img = fitsio.read(imgfn)

            okimg = img.flat[np.flatnonzero(np.isfinite(img))]
            plo,phi = [np.percentile(okimg, p) for p in [25,99]]
            print 'Percentiles', plo, phi
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
            print 'Reading', comaskfn
            comaskfn = 'wise-coadds/masks-coadd-%s-w%i/coadd-mask-%s-%s-w%i-1b.fits' % (coadd, band, coadd, frame, band)
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
            plt.title('Coadd mask')
            ps.savefig()


    for frame in []: #'05579a167']:
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
            

    II = []
    JJ = []
    KK = []
    ppI = []
    ppJ = []
    for band in [1,2]:#,3,4]:
        fni = 'L3a/%s_ab41/%s_ab41-w%i-int-3.fits' % (coadd, coadd, band)
        I = fitsio.read(fni)
        fnj = 'wise-coadds/coadd-%s-w%i-img.fits' % (coadd, band)
        J = fitsio.read(fnj)
        fnk = 'wise-coadds/coadd-%s-w%i-img-w.fits' % (coadd, band)
        K = fitsio.read(fnk)

        wcsJ = Tan(fnj)

        II.append(I)
        JJ.append(J)
        KK.append(K)
        
        plt.clf()
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        pmid = np.percentile(I, 50)
        p95 = np.percentile(I, 95)
        ppI.append((plo,pmid, p95, phi))

        print 'Percentiles', plo,phi
        imai = dict(interpolation='nearest', origin='lower',
                   vmin=plo, vmax=phi)
        plt.imshow(I, **imai)
        plt.title(fni)
        ps.savefig()

        plt.clf()
        plo,phi = [np.percentile(J, p) for p in [25,99]]
        pmid = np.percentile(J, 50)
        p95 = np.percentile(J, 95)
        ppJ.append((plo,pmid,p95,phi))
        print 'Percentiles', plo,phi
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

        x,y = int(wj*(flo+fhi)/2.), int(hj*(flo+fhi)/2.)
        print 'J: x,y =', x,y
        print 'RA,Dec', wcsJ.pixelxy2radec(x,y)

        plt.clf()
        plt.imshow(I[slcI], **imai)
        plt.title(fni)
        ps.savefig()

        plt.clf()
        plt.imshow(J[slcJ], **imaj)
        plt.title(fnj)
        ps.savefig()

        print 'J size', J[slcJ].shape

        plt.clf()
        plt.imshow(K[slcJ], **imaj)
        plt.title(fnk)
        ps.savefig()

    flo,fhi = 0.45, 0.55
    hi,wi = I.shape
    hj,wj = J.shape
    slcI = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
    slcJ = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)

    s = II[0][slcI]
    HI,WI = s.shape
    rgbI = np.zeros((HI, WI, 3))
    p0,px,p1,px = ppI[0]
    rgbI[:,:,0] = (II[0][slcI] - p0) / (p1-p0)
    p0,px,p1,px = ppI[1]
    rgbI[:,:,2] = (II[1][slcI] - p0) / (p1-p0)
    rgbI[:,:,1] = (rgbI[:,:,0] + rgbI[:,:,2])/2.

    plt.clf()
    plt.imshow(np.clip(rgbI, 0., 1.), interpolation='nearest', origin='lower')
    ps.savefig()

    plt.clf()
    plt.imshow(np.sqrt(np.clip(rgbI, 0., 1.)), interpolation='nearest', origin='lower')
    ps.savefig()

    s = JJ[0][slcJ]
    HJ,WJ = s.shape
    rgbJ = np.zeros((HJ, WJ, 3))
    p0,px,p1,px = ppJ[0]
    rgbJ[:,:,0] = (JJ[0][slcJ] - p0) / (p1-p0)
    p0,px,p1,px = ppJ[1]
    rgbJ[:,:,2] = (JJ[1][slcJ] - p0) / (p1-p0)
    rgbJ[:,:,1] = (rgbJ[:,:,0] + rgbJ[:,:,2])/2.

    plt.clf()
    plt.imshow(np.clip(rgbJ, 0., 1.), interpolation='nearest', origin='lower')
    ps.savefig()

    plt.clf()
    plt.imshow(np.sqrt(np.clip(rgbJ, 0., 1.)), interpolation='nearest', origin='lower')
    ps.savefig()

    I = (np.sqrt(np.clip(rgbI, 0., 1.))*255.).astype(np.uint8)
    I2 = np.zeros((3,HI,WI))
    I2[0,:,:] = I[:,:,0]
    I2[1,:,:] = I[:,:,1]
    I2[2,:,:] = I[:,:,2]

    J = (np.sqrt(np.clip(rgbJ, 0., 1.))*255.).astype(np.uint8)
    J2 = np.zeros((3,HJ,WJ))
    J2[0,:,:] = J[:,:,0]
    J2[1,:,:] = J[:,:,1]
    J2[2,:,:] = J[:,:,2]

    fitsio.write('I.fits', I2, clobber=True)
    fitsio.write('J.fits', J2, clobber=True)

    for fn in ['I.fits', 'J.fits']:
        os.system('an-fitstopnm -N 0 -X 255 -i %s -p 0 > r.pgm' % fn)
        os.system('an-fitstopnm -N 0 -X 255 -i %s -p 1 > g.pgm' % fn)
        os.system('an-fitstopnm -N 0 -X 255 -i %s -p 2 > b.pgm' % fn)
        os.system('rgb3toppm r.pgm g.pnm b.pnm | pnmtopng > %s' % ps.getnext())
    
    cmd = 'an-fitstopnm -N 0 -X 255 -i I.fits | pnmtopng > %s' % ps.getnext()
    os.system(cmd)
    cmd = 'an-fitstopnm -N 0 -X 255 -i J.fits | pnmtopng > %s' % ps.getnext()
    os.system(cmd)

    plt.clf()
    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.imshow(I, interpolation='nearest', origin='lower')
    ps.savefig()
    plt.imshow(J, interpolation='nearest', origin='lower')
    ps.savefig()


    # fn = ps.getnext()
    # plt.imsave(fn, (np.sqrt(np.clip(rgbI, 0., 1.))*255.).astype(np.uint8))
    # fn = ps.getnext()
    # plt.imsave(fn, (np.sqrt(np.clip(rgbJ, 0., 1.))*255.).astype(np.uint8))
