import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.fits import *
from astrometry.util.util import Tan, Sip

ps = PlotSequence('co')

wisel3 = 'wise-L3'
coadds = 'wise-coadds'

from wise3 import get_l1b_file

def plot_exposures():

    plt.subplots_adjust(bottom=0.01, top=0.9, left=0., right=1., wspace=0.05, hspace=0.2)
    for coadd_id,band in [('1384p454', 3)]:
        print coadd_id, band
    
        plt.clf()
        plt.subplot(1,2,1)
        fn2 = os.path.join(coadds, 'unwise-%s-w%i-img-w.fits' % (coadd_id, band))
        J = fitsio.read(fn2)
        binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)])
        plo,phi = [np.percentile(binJ, p) for p in [25,99]]
        plt.imshow(binJ, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.subplot(1,2,2)
        #fn3 = os.path.join(coadds, 'unwise-%s-w%i-invvar-w.fits' % (coadd_id, band))
        fn3 = os.path.join(coadds, 'unwise-%s-w%i-ppstd.fits' % (coadd_id, band))
        J = fitsio.read(fn3)
        binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)])
        phi = np.percentile(binJ, 99)
        plt.imshow(binJ, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=0, vmax=phi)
        plt.xticks([]); plt.yticks([])
        ps.savefig()
    
    
        fn = os.path.join(coadds, 'unwise-%s-w%i-frames.fits' % (coadd_id, band))
        T = fits_table(fn)
        print len(T), 'frames'
        T.cut(np.lexsort((T.frame_num, T.scan_id)))
    
        plt.clf()
        n,b,p = plt.hist(np.log10(np.maximum(0.1, T.npixrchi)), bins=100, range=(-1,6),
                         log=True)
        plt.xlabel('log10( N pix with bad rchi )')
        plt.ylabel('Number of images')
        plt.ylim(0.1, np.max(n) + 5)
        ps.savefig()
    
        J = np.argsort(-T.npixrchi)
        print 'Largest npixrchi:'
        for n,s,f in zip(T.npixrchi[J], T.scan_id[J], T.frame_num[J[:20]]):
            print '  n', n, 'scan', s, 'frame', f
    
        i0 = 0
        while i0 <= len(T):
            plt.clf()
            R,C = 4,6
            for i in range(i0, i0+(R*C)):
                if i >= len(T):
                    break
                t = T[i]
                fn = get_l1b_file('wise-frames', t.scan_id, t.frame_num, band)
                print fn
                I = fitsio.read(fn)
                bad = np.flatnonzero(np.logical_not(np.isfinite(I)))
                I.flat[bad] = 0.
                print I.shape
                binI = reduce(np.add, [I[j/4::4, j%4::4] for j in range(16)])
                print binI.shape
                plt.subplot(R,C,i-i0+1)
                plo,phi = [np.percentile(binI, p) for p in [25,99]]
                print 'p', plo,phi
                plt.imshow(binI, interpolation='nearest', origin='lower',
                           vmin=plo, vmax=phi, cmap='gray')
                plt.xticks([]); plt.yticks([])
                plt.title('%s %i' % (t.scan_id, t.frame_num))
            plt.suptitle('%s W%i' % (coadd_id, band))
            ps.savefig()
            i0 += R*C


# T = fits_table('tab.fits')
# T.cut(T.band == 3)
# print len(T), 'in WISE coadd'
# F = fits_table('wise-coadds/unwise-1384p454-w3-frames.fits')
# print len(F), 'in unWISE coadd'
# 
# for s,f in zip(T.scan_id, T.frame_num):
#     I = np.flatnonzero((F.scan_id == s) * (F.frame_num == f))
#     if len(I) == 1:
#         continue
#     print 'scan/frame', s,f, ': not found'
#     #W = fits_table('sequels-frames.fits')
# sys.exit(0)

def pixel_area():
    for wcs in [Sip('wise-frames/2a/03242a/215/03242a215-w1-int-1b.fits'),
                Tan('wise-coadds/unwise-1384p454-w1-img.fits'),
                ]:
        W,H = wcs.get_width(), wcs.get_height()
        print 'W,H', W,H
        #xx,yy = np.meshgrid(np.arange(0, W, 10), np.arange(0, H, 10))
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        rr,dd = wcs.pixelxy2radec(xx, yy)
        rr -= wcs.crval[0]
        rr *= np.cos(np.deg2rad(dd))
        dd -= wcs.crval[1]
        
        # (zero,zero) r,d
        zzr = rr[:-1, :-1]
        zzd = dd[:-1, :-1]
        ozr = rr[:-1, 1:]
        ozd = dd[:-1, 1:]
        zor = rr[1:, :-1]
        zod = dd[1:, :-1]
        oor = rr[1:, 1:]
        ood = dd[1:, 1:]
        
        a = np.hypot(zor - zzr, zod - zzd)
        A = np.hypot(oor - ozr, ood - ozd)
        b = np.hypot(ozr - zzr, ozd - zzd)
        B = np.hypot(oor - zor, ood - zod)
        C = np.hypot(ozr - zor, ozd - zod)
        c = C
        
        s = (a + b + c)/2.
        S = (A + B + C)/2.
        
        area = np.sqrt(s * (s-a) * (s-b) * (s-c)) + np.sqrt(S * (S-A) * (S-B) * (S-C))
        
        plt.clf()
        plt.imshow(area, interpolation='nearest', origin='lower')
        plt.title('Pixel area')
        plt.colorbar()
        ps.savefig()

# plt.clf()
# plt.plot(rr.ravel(), dd.ravel(), 'r.')
# plt.axis('scaled')
# ps.savefig()


def paper_plots(coadd_id, band):
    figsize = (4,4)
    spa = dict(left=0.01, right=0.99, bottom=0.01, top=0.99)

    bigfigsize = (8,6)
    bigspa = dict(left=0.1, right=0.98, bottom=0.1, top=0.98)

    plt.figure(figsize=figsize)
    plt.subplots_adjust(**spa)

    dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')
    dir2 = 'e'
    
    wiseim,wisehdr = read(dir1, '%s_ab41-w%i-int-3.fits' % (coadd_id, band), header=True)
    imw    = read(dir2, 'unwise-%s-w%i-img-w.fits' % (coadd_id, band))
    im     = read(dir2, 'unwise-%s-w%i-img.fits' % (coadd_id, band))

    unc    = read(dir1, '%s_ab41-w%i-unc-3.fits.gz' % (coadd_id, band))
    ivw    = read(dir2, 'unwise-%s-w%i-invvar-w.fits' % (coadd_id, band))
    iv     = read(dir2, 'unwise-%s-w%i-invvar.fits' % (coadd_id, band))

    ppstdw = read(dir2, 'unwise-%s-w%i-ppstd-w.fits' % (coadd_id, band))

    wisen  = read(dir1, '%s_ab41-w%i-cov-3.fits.gz' % (coadd_id, band))
    un     = read(dir2, 'unwise-%s-w%i-n.fits' % (coadd_id, band))
    unw    = read(dir2, 'unwise-%s-w%i-n-w.fits' % (coadd_id, band))

    binwise = reduce(np.add, [wiseim[i/5::5, i%5::5] for i in range(25)]) / 25.
    binim   = reduce(np.add, [im    [i/4::4, i%4::4] for i in range(16)]) / 16.
    binimw  = reduce(np.add, [imw   [i/4::4, i%4::4] for i in range(16)]) / 16.

    ima = dict(interpolation='nearest', origin='lower', cmap='gray')

    def myimshow(img):
        #plo,phi = [np.percentile(img, p) for p in [25,99]]
        plo,phi = [np.percentile(img, p) for p in [25,95]]
        imai = ima.copy()
        imai.update(vmin=plo, vmax=phi)
        plt.clf()
        plt.imshow(img, **imai)
        plt.xticks([]); plt.yticks([])

    if False:
        for img in [binwise, binim, binimw]:
            myimshow(img)
            ps.savefig()
            
        hi,wi = wiseim.shape
        hj,wj = imw.shape
        #flo,fhi = 0.45, 0.55
        flo,fhi = 0.45, 0.50
        slcW = slice(int(hi*flo), int(hi*fhi)+1), slice(int(wi*flo), int(wi*fhi)+1)
        slcU = slice(int(hj*flo), int(hj*fhi)+1), slice(int(wj*flo), int(wj*fhi)+1)
    
        subwise = wiseim[slcW]
        subimw  = imw[slcU]
        subim   = im [slcU]
    
        for img in [subwise, subim, subimw]:
            myimshow(img)
            ps.savefig()
    
        mx = max(wisen.max(), un.max(), unw.max())
        na = ima.copy()
        na.update(vmin=0, vmax=mx, cmap='jet')
        plt.clf()
        plt.imshow(wisen, **na)
        plt.xticks([]); plt.yticks([])
        ps.savefig()
    
        plt.clf()
        plt.imshow(un, **na)
        plt.xticks([]); plt.yticks([])
        ps.savefig()
    
        w,h = figsize
        plt.figure(figsize=(w+1,h))
        plt.subplots_adjust(**spa)
    
        plt.clf()
        plt.imshow(unw, **na)
        plt.xticks([]); plt.yticks([])
    
        parent = plt.gca()
        pb = parent.get_position(original=True).frozen()
        print 'pb', pb
        # new parent box, padding, child box
        frac = 0.15
        pad  = 0.05
        (pbnew, padbox, cbox) = pb.splitx(1.0-(frac+pad), 1.0-frac)
        print 'pbnew', pbnew
        print 'padbox', padbox
        print 'cbox', cbox
        cbox = cbox.anchored('C', cbox)
        parent.set_position(pbnew)
        parent.set_anchor((1.0, 0.5))
        cax = parent.get_figure().add_axes(cbox)
        aspect = 20
        cax.set_aspect(aspect, anchor=((0.0, 0.5)), adjustable='box')
        parent.get_figure().sca(parent)
    
        plt.colorbar(cax=cax)
        ps.savefig()
    

    if True:
        from tractor import GaussianMixturePSF, NanoMaggies
        from unwise_coadd import estimate_sky

        wisemed = np.median(wiseim[::4,::4])
        wisesig = np.median(unc[::4,::4])
        wisesky = estimate_sky(wiseim, wisemed-2.*wisesig, wisemed+1.*wisesig)
        print 'WISE sky estimate:', wisesky

        P = fits_table('wise-psf-avg.fits', hdu=band)
        psf = GaussianMixturePSF(P.amp, P.mean, P.var)
        R = 100
        psf.radius = R
        pat = psf.getPointSourcePatch(0., 0.)
        pat = pat.patch
        pat /= pat.sum()
        psfnorm = np.sqrt(np.sum(pat**2))
        print 'PSF norm (native pixel scale):', psfnorm
        psfnorm2 = psfnorm / 2.
        print 'PSF norm (finer pixel scale):', psfnorm2
        
        sigw = 1./np.sqrt(np.maximum(ivw, 1e-16))

        wise_unc_fudge = 2.

    
    if False:
        # Sky / Error properties
    
        plt.figure(figsize=bigfigsize)
        plt.subplots_adjust(**bigspa)
        wisechi = ((wiseim-wisesky) / unc).ravel()
        #wisechi2 = 2. * ((wiseim-wisesky) / (unc/psfnorm)).ravel()
        #wisechi2 = (2.*psfnorm * (wiseim-wisesky) / unc).ravel()
        #wisechi2 = 0.5 * ((wiseim-wisesky) / unc).ravel()
        wisechi2 = ((wiseim-wisesky) / (wise_unc_fudge * unc)).ravel()

        chiw = (imw / sigw).ravel()
        lo,hi = -6,12
        ha = dict(range=(lo,hi), bins=100, log=True, histtype='step')
        ha1 = dict(range=(lo,hi), bins=100)
        plt.clf()
        h1,e = np.histogram(wisechi, **ha1)
        h2,e = np.histogram(wisechi2, **ha1)
        h3,e = np.histogram(chiw, **ha1)
        nw = h3
        nwise = h2
        ee = e.repeat(2)[1:-1]
        p1 = plt.plot(ee, (h1/1.).repeat(2), color='r', lw=3, alpha=0.5)
        p2 = plt.plot(ee, (h2/1.).repeat(2), color='m', lw=2, alpha=0.75)
        p3 = plt.plot(ee, h3.repeat(2), color='b')
        plt.yscale('log')
        xx = np.linspace(lo, hi, 300)
        plt.plot(xx, max(nw) * np.exp(-(xx**2)/(2.)), 'b-', alpha=0.5)
        plt.plot(xx, max(nwise) * np.exp(-(xx**2)/(2.)), 'm-', alpha=0.5)
        plt.xlabel('Pixel / Uncertainty ($\sigma$)')
        plt.ylabel('Number of pixels')
        plt.legend((p1,p2,p3), ('WISE', 'WISE corr', 'unWISE'))
        #plt.legend((p1,p2), ('WISE', 'WISE corr'))
        yl,yh = plt.ylim()
        plt.ylim(3, yh)
        plt.xlim(lo,hi)
        plt.axvline(0, color='k', alpha=0.1)
        ps.savefig()
    
    if True:
        
        plt.figure(figsize=bigfigsize)
        plt.subplots_adjust(**bigspa)

        zp = wisehdr['MAGZP']
        print 'WISE image zeropoint:', zp
        zpscale = 1. / NanoMaggies.zeropointToScale(zp)
        print 'zpscale', zpscale

        wiseflux = (wiseim - wisesky) * zpscale
        #wiseerr  = (unc * zpscale / (2. * psfnorm)).ravel()
        wiseerr  = unc * zpscale

        wiseflux /= psfnorm
        wiseerr *= wise_unc_fudge / psfnorm

        wiseerr1 = np.median(wiseerr)

        #print 'zpscale for 22.5:', 1./NanoMaggies.zeropointToScale(22.5)
        # print 'median wise flux: ', np.median(wiseflux)
        # print 'median wise error:', wiseerr1

        unflux = imw.ravel()
        unerr = (ppstdw / np.sqrt(unw.astype(np.float32))).ravel()

        # print 'median unwise flux: ', np.median(unflux)
        # print 'median unwise error:', np.median(unerr)
        # print 'median n:', np.median(unw)

        logflo,logfhi = -2, 5.
        logelo,logehi = 0., 3.

        flo,fhi = 10.**logflo, 10.**logfhi
        elo,ehi = 10.**logelo, 10.**logehi

        # wf = wiseflux[::2, ::2].ravel()
        # plt.clf()
        # loghist(np.log10(wf), np.log10(unflux), range=[[np.log10(flo),np.log10(fhi)]]*2,
        #         nbins=200, hot=False, doclf=False,
        #         docolorbar=False, imshowargs=dict(cmap=antigray))
        # plt.xlabel('log WISE flux')
        # plt.ylabel('log unWISE flux')
        # ps.savefig()

        wiseflux = wiseflux.ravel()
        wiseerr  = wiseerr.ravel()

        ha = dict(hot=False, doclf=False, nbins=200,
                  range=((np.log10(flo),np.log10(fhi)), (np.log10(elo),np.log10(ehi))),
                  docolorbar=False, imshowargs=dict(cmap=antigray))

        plt.clf()
        loghist(np.log10(np.clip(wiseflux, flo,fhi)),
                np.log10(np.clip(wiseerr,  elo,ehi)), **ha)
        #plt.xlabel('log WISE flux')
        #plt.ylabel('log WISE error')
        plt.xlabel('WISE flux')
        plt.ylabel('WISE flux uncertainty')
        ax = plt.axis()
        xx = np.linspace(np.log10(flo), np.log10(fhi), 500)
        #yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.1 * 10.**xx)))
        # yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.3 * 10.**xx)))
        # plt.plot(xx, yy, 'b-')
        # yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.03 * 10.**xx)))
        # plt.plot(xx, yy, 'b-')
        # yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.015 * 10.**xx)))
        # plt.plot(xx, yy, 'b-')
        #yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.01 * 10.**xx)))
        yy = np.log10(np.hypot(wiseerr1, np.sqrt(0.02 * 10.**xx)))
        plt.plot(xx, yy, 'r-', lw=2)
        plt.axis(ax)
        logf = np.arange(logflo,logfhi+1)
        plt.xticks(logf, ['$10^{%i}$' % x for x in logf])
        loge = np.arange(logelo,logehi+1)
        plt.yticks(loge, ['$10^{%i}$' % x for x in loge])
        ps.savefig()

        plt.clf()
        loghist(np.log10(np.clip(unflux, flo,fhi)),
                np.log10(np.clip(unerr,  elo,ehi)), **ha)
        #plt.xlabel('log unWISE flux')
        #plt.ylabel('log unWISE error')
        plt.xlabel('unWISE flux')
        plt.ylabel('unWISE sample standard deviation')
        #yy = np.log10(np.hypot(np.hypot(wiseerr1, np.sqrt(0.1 * 10.**xx)), 1e-2*(10.**xx)))
        #yy = np.log10(np.hypot(np.hypot(wiseerr1, np.sqrt(0.02 * 10.**xx)), 2e-2*(10.**xx)))
        yy = np.log10(np.hypot(np.hypot(wiseerr1, np.sqrt(0.02 * 10.**xx)), 3e-2*(10.**xx)))
        ax = plt.axis()
        plt.plot(xx, yy, 'r-', lw=2)
        plt.axis(ax)
        logf = np.arange(logflo,logfhi+1)
        plt.xticks(logf, ['$10^{%i}$' % x for x in logf])
        loge = np.arange(logelo,logehi+1)
        plt.yticks(loge, ['$10^{%i}$' % x for x in loge])
        ps.savefig()


        plt.clf()
        plt.hist(wiseflux / wiseerr, range=(-6,10), log=True, bins=100,
                 histtype='step', color='r')
        plt.hist(unflux / unerr, range=(-6,10), log=True, bins=100,
                 histtype='step', color='b')
        yl,yh = plt.ylim()
        plt.ylim(0.1, yh)
        ps.savefig()
        



# getfn=False,
def read(dirnm, fn, header=False):
    pth = os.path.join(dirnm, fn)
    print pth
    data = fitsio.read(pth, header=header)
    #if getfn:
    #    return data,pth
    return data


T = fits_table('sequels-atlas.fits')

#ps.suffixes = ['png','pdf']

paper_plots(T.coadd_id[0], 1)
#pixel_area()
sys.exit(0)


# Download from IRSA:
# for coadd_id in T.coadd_id:
#     print 'Coadd id', coadd_id
#     cmd = 'wget -r -N -nH -np -nv --cut-dirs=4 -A "*int-3.fits" "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/"' % (coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')
#     print 'Cmd:', cmd
#     os.system(cmd)


#T.cut(np.array([0]))
bands = [1,2,3,4]

plt.figure(figsize=(12,4))
#plt.subplots_adjust(bottom=0.01, top=0.85, left=0., right=1., wspace=0.05)
#plt.subplots_adjust(bottom=0.1, top=0.85, left=0., right=0.9, wspace=0.05)
plt.subplots_adjust(bottom=0.1, top=0.85, left=0.05, right=0.9, wspace=0.15)


for coadd_id in T.coadd_id[:5]:
    dir1 = os.path.join(wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41')

    for band in bands:

        dir2 = 'e'

        try:
            wiseim = read(dir1, '%s_ab41-w%i-int-3.fits' % (coadd_id, band))
            imw    = read(dir2, 'unwise-%s-w%i-img-w.fits' % (coadd_id, band))
            im     = read(dir2, 'unwise-%s-w%i-img.fits' % (coadd_id, band))

            unc    = read(dir1, '%s_ab41-w%i-unc-3.fits.gz' % (coadd_id, band))
            ivw    = read(dir2, 'unwise-%s-w%i-invvar-w.fits' % (coadd_id, band))
            iv     = read(dir2, 'unwise-%s-w%i-invvar.fits' % (coadd_id, band))

            # cmd = ('wget -r -N -nH -np -nv --cut-dirs=5 -P %s "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/%s"' %
            #        (wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41', os.path.basename(ufn1)))
            # print 'Cmd:', cmd
            # os.system(cmd)

        except:
            continue

        I = wiseim
        J = imw
        K = im
        
        L = ivw
        M = iv

        binI = reduce(np.add, [I[i/5::5, i%5::5] for i in range(25)]) / 25.
        binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)]) / 16.
        binK = reduce(np.add, [K[i/4::4, i%4::4] for i in range(16)]) / 16.
        # binI = I[::5,::5]
        # binJ = J[::4,::4]
        # binK = K[::4,::4]
        
        ima = dict(interpolation='nearest', origin='lower', cmap='gray')

        plo,phi = [np.percentile(binI, p) for p in [25,99]]
        imai = ima.copy()
        imai.update(vmin=plo, vmax=phi)
        plo,phi = [np.percentile(binJ, p) for p in [25,99]]
        imaj = ima.copy()
        imaj.update(vmin=plo, vmax=phi)

        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(binI, **imai)
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        plt.title('WISE')
        plt.subplot(1,3,2)
        plt.imshow(binJ, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE weighted')
        plt.subplot(1,3,3)
        plt.imshow(binK, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE')
        plt.colorbar()
        plt.suptitle('%s W%i' % (coadd_id, band))
        ps.savefig()


        # Emphasize the sky levels
        
        plo,phi = [np.percentile(binI, p) for p in [1,70]]
        imai = ima.copy()
        imai.update(vmin=plo, vmax=phi)
        plo,phi = [np.percentile(binJ, p) for p in [1,70]]
        imaj = ima.copy()
        imaj.update(vmin=plo, vmax=phi)

        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(binI, **imai)
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        plt.title('WISE')
        plt.subplot(1,3,2)
        plt.imshow(binJ, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE weighted')
        plt.subplot(1,3,3)
        plt.imshow(binK, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE')
        plt.colorbar()
        plt.suptitle('%s W%i' % (coadd_id, band))
        ps.savefig()




        sig1w = 1./np.sqrt(np.median(ivw))
        sig1 = 1./np.sqrt(np.median(iv))
        unc1 = np.median(unc)
        print 'sig1w:', sig1w
        print 'sig1:', sig1
        print 'unc:', unc1

        med = np.median(wiseim)
        sigw = 1./np.sqrt(ivw)

        plt.clf()
        lo,hi = -8,10
        ha = dict(bins=100, histtype='step', range=(lo,hi), log=True)
        plt.hist((im / sig1).ravel(), color='g', lw=2, **ha)
        n,b,p = plt.hist((imw / sig1w).ravel(), color='b', **ha)
        plt.hist((imw / sigw).ravel(), color='c', **ha)
        plt.hist(((wiseim - med) / unc1).ravel(), color='r', **ha)
        plt.hist(((wiseim - med) / unc).ravel(), color='m', **ha)
        yl,yh = plt.ylim()
        xx = np.linspace(lo, hi, 300)
        plt.plot(xx, max(n) * np.exp(-(xx**2)/(2.)), 'r--')
        plt.ylim(0.1, yh)
        plt.xlim(lo,hi)
        ps.savefig()

        # plt.clf()
        # loghist(im.ravel(), imw.ravel(), range=[[-10*sig1,10*sig1]]*2, bins=200)
        # plt.xlabel('im')
        # plt.ylabel('imw')
        # ps.savefig()


        L = 1./np.sqrt(L)
        M = 1./np.sqrt(M)
        # binL = reduce(np.add, [L[i/4::4, i%4::4] for i in range(16)])
        # binM = reduce(np.add, [M[i/4::4, i%4::4] for i in range(16)])
        # binunc = reduce(np.add, [unc[i/5::5, i%5::5] for i in range(25)])
        binL = L[::4,::4]
        binM = M[::4,::4]
        binunc = unc[::5,::5]


        plt.clf()

        plo,phi = [np.percentile(binunc, p) for p in [25,99]]
        imaj = ima.copy()
        imaj.update(vmin=plo, vmax=phi)

        plt.subplot(1,3,1)
        plt.imshow(binunc, **imaj)
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        plt.title('WISE unc')

        plo,phi = [np.percentile(binL, p) for p in [25,99]]
        imaj = ima.copy()
        imaj.update(vmin=plo, vmax=phi)

        plt.subplot(1,3,2)
        plt.imshow(binL, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE unc (weighted)')

        plt.subplot(1,3,3)
        plt.imshow(binM, **imaj)
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE unc')

        ps.savefig()

        ## J: wim
        ## K: im
        ## L: wiv -> sig
        ## M: iv  -> sig
        chia = J / L
        chib = K / M

        print 'chia:', chia.min(), chia.max()
        print 'chib:', chib.min(), chib.max()

        # plt.clf()
        # plt.subplot(1,3,2)
        # n,b,p = plt.hist(chia.ravel(), bins=100, log=True, range=(-20,20), histtype='step', color='r')
        # plt.ylim(0.1, max(n)*2)
        # plt.subplot(1,3,3)
        # n,b,p = plt.hist(chib.ravel(), bins=100, log=True, range=(-20,20), histtype='step', color='b')
        # plt.ylim(0.1, max(n)*2)
        # ps.savefig()


        fn6 = os.path.join(dir2, 'unwise-%s-w%i-ppstd-w.fits' % (coadd_id, band))
        print fn6
        if not os.path.exists(fn6):
            print '-> does not exist'
            continue
        ppstd = fitsio.read(fn6)


        plt.clf()
        plt.subplot(1,3,1)
        loghist(np.clip(np.log10(I.ravel()), -2,4), np.clip(np.log10(unc.ravel()), -2, 4), doclf=False, docolorbar=False)
        plt.title('WISE int vs unc')
        plt.subplot(1,3,2)
        loghist(np.clip(np.log10(J.ravel()), -1, 5), np.clip(np.log10(L.ravel()), -1, 5), doclf=False, docolorbar=False)
        plt.title('unWISE img vs 1/sqrt(iv)')
        plt.subplot(1,3,3)
        loghist(np.clip(np.log10(J.ravel()), -1, 5), np.clip(np.log10(ppstd.ravel()), -1, 5), doclf=False, docolorbar=False)
        plt.title('unWISE img vs ppstd')
        ps.savefig()



        fn1 = os.path.join(dir1, '%s_ab41-w%i-cov-3.fits.gz' % (coadd_id, band))
        print fn1
        if not os.path.exists(fn1):
            print '-> does not exist'
            cmd = ('wget -r -N -nH -np -nv --cut-dirs=5 -P %s "http://irsa.ipac.caltech.edu/ibe/data/wise/merge/merge_p3am_cdd/%s/%s/%s/%s"' %
                   (wisel3, coadd_id[:2], coadd_id[:4], coadd_id + '_ab41', os.path.basename(fn1)))
            print 'Cmd:', cmd
            os.system(cmd)

        fn2 = os.path.join(dir2, 'unwise-%s-w%i-n-w.fits' % (coadd_id, band))
        print fn2
        if not os.path.exists(fn2):
            print '-> does not exist'
            continue

        fn3 = os.path.join(dir2, 'unwise-%s-w%i-n.fits' % (coadd_id, band))
        print fn3
        if not os.path.exists(fn3):
            print '-> does not exist'
            continue

        I = fitsio.read(fn1)
        J = fitsio.read(fn2)
        K = fitsio.read(fn3)
        # binJ = reduce(np.add, [J[i/4::4, i%4::4] for i in range(16)])
        # binK = reduce(np.add, [K[i/4::4, i%4::4] for i in range(16)])
        binI = I[::5,::5]
        binJ = J[::4,::4]
        binK = K[::4,::4]

        plo,phi = min(binI.min(), binJ.min(), binK.min()), max(binI.max(), binJ.max(),binK.max())
        imaj = ima.copy()
        imaj.update(vmin=plo, vmax=phi, cmap='jet')

        plt.clf()

        plt.subplot(1,3,1)
        plt.imshow(binI, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('WISE cov')

        plt.subplot(1,3,2)
        plt.imshow(binJ, **imaj)
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE n (weighted)')

        plt.subplot(1,3,3)
        plt.imshow(binK, **imaj)
        plt.colorbar()
        plt.xticks([]); plt.yticks([])
        plt.title('unWISE n')

        ps.savefig()

    #break


sys.exit(0)


    

lst = os.listdir(wisel3)
lst.sort()
bands = [1,2,3,4]

# HACK
#lst = ['1917p454_ab41', '1273p575_ab41']
#lst = ['1273p575_ab41']
lst = ['1190p575_ab41']
bands = [1,2]
#bands = [2]


for band in bands:
    for l3dir in lst:
        print 'dir', l3dir
        coadd = l3dir.replace('_ab41','')
        l3fn = os.path.join(wisel3, l3dir, '%s-w%i-int-3.fits' % (l3dir, band))
        if not os.path.exists(l3fn):
            print 'Missing', l3fn
            continue
        cofn  = os.path.join(coadds, 'unwise-%s-w%i-img.fits'   % (coadd, band))
        cowfn = os.path.join(coadds, 'unwise-%s-w%i-img-w.fits' % (coadd, band))
        if not os.path.exists(cofn) or not os.path.exists(cowfn):
            print 'Missing', cofn, 'or', cowfn
            continue

        I = fitsio.read(l3fn)
        J = fitsio.read(cofn)
        K = fitsio.read(cowfn)

        print 'coadd range:', J.min(), J.max()
        print 'w coadd range:', K.min(), K.max()

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
        plt.imshow(I, **imai)
        plt.title('WISE team %s' % os.path.basename(l3fn))
        ps.savefig()

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
        F = fits_table('wise-coadds/unwise-%s-w%i-frames.fits' % (coadd,band))

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
