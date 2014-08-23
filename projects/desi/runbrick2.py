# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.resample import *
import pylab as plt
from tractor import *
from scipy.ndimage.filters import *

import sys

if True:
    X = unpickle_from_file('brick.pickle')
    print X.keys()
    L = locals()
    for k in X.keys():
        L[k] = X[k]

    B = fits_table('bricks.fits')
    # brick index...
    ii = 377305
    brick = B[ii]
    ra,dec = brick.ra, brick.dec
    W,H = 400,400
    pixscale = 0.27 / 3600.
    bands = ['g','r','z']
    catband = 'r'
    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    float(W), float(H))

# <codecell>

    #plt.imshow(X['sedsn'], cmap='hot', interpolation='nearest', origin='lower', vmax=10.)

# <codecell>

    tims = tractor.getImages()
    # save resampling params
    for tim in tims:
        wcs = tim.sip_wcs

        x0,y0 = int(tim.x0),int(tim.y0)
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(x0, y0, subw, subh)
        tim.subwcs = subwcs
        
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
            print 'Resampled', len(Yo), 'pixels'
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue
        tim.resamp = (Yo,Xo,Yi,Xi)

# <codecell>

    cat = tractor.getCatalog()

    
    ps = PlotSequence('brick2', format='%03i')
    #plt.figure(figsize=(8,6));
    plt.figure(figsize=(12,9));
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95,
                        hspace=0.05, wspace=0.05)

    # Clamp near-zero invvars to zero
    for tim in tims:
        iv = tim.getInvvar()
        thresh = 0.2 * (1./tim.sig1**2)
        iv[iv < thresh] = 0
        tim.setInvvar(iv)

    rgbim = np.zeros((H,W,3))
    coimgs = []
    imas = []
    for ib,band in enumerate(bands):
        coimg = np.zeros((H,W))
        con   = np.zeros((H,W))
        for tim in tims:
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            nn = (tim.getInvvar()[Yi,Xi] > 0)
            coimg[Yo,Xo] += tim.getImage ()[Yi,Xi] * nn
            con  [Yo,Xo] += nn
            mn,mx = tim.zr
        coimg /= np.maximum(con,1)
        c = 2-ib
        rgbim[:,:,c] = np.clip((coimg - mn) / (mx - mn), 0., 1.)
        coimgs.append(coimg)
        imas.append(dict(interpolation='nearest', origin='lower', cmap='gray',
                         vmin=mn, vmax=mx))

    detmaps = dict([(b, np.zeros((H,W))) for b in bands])
    detivs  = dict([(b, np.zeros((H,W))) for b in bands])

    for tim in tims:
        # Render the detection map
        wcs = tim.sip_wcs
        x0,y0 = tim.x0,tim.y0
        psf_sigma = tim.psf_sigma
        band = tim.band
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(int(x0), int(y0), subw, subh)
        subiv = tim.getInvvar()
        psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        detim = tim.getImage().copy()
        detim[subiv == 0] = 0.
        detim = gaussian_filter(detim, psf_sigma) / psfnorm**2
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
            print 'Resampled', len(Yo), 'pixels'
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue

        detsig1 = tim.sig1 / psfnorm
        detiv = np.zeros((subh,subw)) + (1. / detsig1**2)
        detiv[subiv == 0] = 0.

        detmaps[band][Yo,Xo] += detiv[Yi,Xi] * detim[Yi,Xi]
        detivs [band][Yo,Xo] += detiv[Yi,Xi]

        # Debug small invvars
        # plt.clf()
        # plt.subplot(2,3,3)
        # plt.hist(subiv.ravel(), 100)
        # plt.subplot(2,3,1)
        # plt.imshow(detim, interpolation='nearest', origin='lower',
        #            vmin=0, vmax=10.*detsig1)
        # plt.colorbar()
        # plt.subplot(2,3,2)
        # plt.imshow(detiv, interpolation='nearest', origin='lower',
        #            vmin=0, vmax=2./(detsig1**2))
        # plt.colorbar()
        # plt.subplot(2,3,4)
        # plt.imshow(detmaps[band] / np.maximum(1e-16,detivs[band]),
        #            interpolation='nearest', origin='lower',
        #            vmin=0, vmax=10.*detsig1)
        # plt.colorbar()
        # plt.subplot(2,3,5)
        # plt.imshow(detivs[band], interpolation='nearest', origin='lower',
        #            vmin=0, vmax=6./(detsig1**2))
        # plt.colorbar()
        # ps.savefig()
        
    sedmap = np.zeros((H,W))
    sediv  = np.zeros((H,W))
    for band in bands:
        detmap = detmaps[band] / np.maximum(1e-16, detivs[band])
        detsn = detmap * np.sqrt(detivs[band])
        #hot |= (detsn > 5.)
        sedmap += detmaps[band]
        sediv  += detivs [band]
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
        
    # for tim in tims:
    #     plt.clf()
    #     plt.subplot(1,2,1)
    #     plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
    #                cmap='gray', vmin=tim.zr[0], vmax=tim.zr[1])
    #     plt.subplot(1,2,2)
    #     plt.imshow(tim.getInvvar(), interpolation='nearest', origin='lower',
    #                cmap='gray')
    #     ps.savefig()
        
    plt.clf()
    plt.imshow(np.round(sedsn), interpolation='nearest', origin='lower',
               vmin=0, vmax=10)
    ps.savefig()

    # find significant peaks in the detection maps
    # FIXME -- also the per-band detmaps
    # blank out ones within a couple of pixels of a catalog source
    # create sources for any remaining peaks

    peaks = (sedsn > 5)
    # zero out the edges(?)
    peaks[0 ,:] = peaks[:, 0] = 0
    peaks[-1,:] = peaks[:,-1] = 0
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,2:  ])

    T.itx = np.round(T.tx).astype(int)
    T.ity = np.round(T.ty).astype(int)

    for x,y in zip(T.itx,T.ity):
        peaks[np.maximum(y - 2, 0) : np.minimum(y + 3, H),
              np.maximum(x - 2, 0) : np.minimum(x + 3, W)] = 0
    pki = np.flatnonzero(peaks)
    peaky,peakx = np.unravel_index(pki, peaks.shape)
    print len(peaky), 'peaks'
    
    plt.clf()
    #plt.imshow(rgbim)
    plt.imshow(coimgs[1], **imas[1])
    ax = plt.axis()
    plt.plot(T.tx, T.ty, 'r+', mew=1.5, ms=8)
    plt.plot(peakx, peaky, 'g+', mew=1.5, ms=8)
    plt.axis(ax)
    ps.savefig()


    blobsrcs = []
    for blob in range(1, nblobs+1):
        blobsrcs.append(np.flatnonzero(T.blob == blob))
    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')

    
# <codecell>

    imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
                vmin=-5, vmax=5)

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    
    for b,I in enumerate(blobsrcs):
        bslc = blobslices[b]
        bsrcs = blobsrcs[b]

        #print 'blob slice:', bslc
        #print 'sources in blob:', I
        if len(I) == 0:
            continue

        cat.freezeAllParams()
        #cat.thawParams(I)
        print 'Fitting:'
        for i in I:
            cat.thawParams(i)
            print cat[i]
            
        #print 'Fitting:'
        #tractor.printThawedParams()

        # before-n-after plots
        mod0 = [tractor.getModelImage(tim) for tim in tims]

        # blob bbox in target coords
        sy,sx = bslc
        y0,y1 = sy.start, sy.stop
        x0,x1 = sx.start, sx.stop

        rr,dd = targetwcs.pixelxy2radec([x0,x0,x1,x1],[y0,y1,y1,y0])
        
        subtims = []
        for i,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            if sx1 < 0 or sy1 < 0 or sx1 > w or sy1 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            
            print 'image subregion', sx0,sx1,sy0,sy1

            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subiv  = tim.getInvvar()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[i]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            # FIXME --
            #subpsf = tim.getPsf().mogAt((ox0+x0+x1)/2., (oy0+y0+y1)/2.)
            subpsf = tim.getPsf()

            subtim = Image(data=subimg, invvar=subiv, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky())
            subtims.append(subtim)
            
        subtr = Tractor(subtims, cat)
        subtr.freezeParam('images')
        print 'Optimizing:', subtr
        subtr.printThawedParams()
        
        for step in range(10):
            #dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        mod1 = [tractor.getModelImage(tim) for tim in tims]

        # rgbim = np.zeros((H,W,3))
        rgbm0 = np.zeros((H,W,3))
        rgbm1 = np.zeros((H,W,3))
        rgbchi0 = np.zeros((H,W,3))
        rgbchi1 = np.zeros((H,W,3))

        subims_b = []
        subims_a = []
        
        for iband,band in enumerate(bands):
            # coimg = np.zeros((H,W))
            coimg = coimgs[iband]
            com0  = np.zeros((H,W))
            com1  = np.zeros((H,W))
            #con   = np.zeros((H,W))
            cochi0 = np.zeros((H,W))
            cochi1 = np.zeros((H,W))
            for tim,m0,m1 in zip(tims, mod0, mod1):
                if tim.band != band:
                    continue
                (Yo,Xo,Yi,Xi) = tim.resamp
                #coimg[Yo,Xo] += tim.getImage ()[Yi,Xi]
                cochi0[Yo,Xo] += (tim.getImage()[Yi,Xi] - m0[Yi,Xi]) * tim.getInvError()[Yi,Xi]
                cochi1[Yo,Xo] += (tim.getImage()[Yi,Xi] - m1[Yi,Xi]) * tim.getInvError()[Yi,Xi]
                com0 [Yo,Xo] += m0[Yi,Xi]
                com1 [Yo,Xo] += m1[Yi,Xi]
                #con  [Yo,Xo] += 1
                mn,mx = tim.zr

            # coimg /= np.maximum(con,1)
            com0  /= np.maximum(con,1)
            com1  /= np.maximum(con,1)

            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            c = 2-iband
            #rgbim[:,:,c] = np.clip((coimg - mn) / (mx - mn), 0., 1.)
            rgbm0[:,:,c] = np.clip((com0  - mn) / (mx - mn), 0., 1.)
            rgbm1[:,:,c] = np.clip((com1  - mn) / (mx - mn), 0., 1.)

            mn,mx = -5,5
            rgbchi0[:,:,c] = np.clip((cochi0 - mn) / (mx - mn), 0, 1)
            rgbchi1[:,:,c] = np.clip((cochi1 - mn) / (mx - mn), 0, 1)

            subims_b.append((coimg[bslc], com0[bslc], ima, cochi0[bslc]))
            subims_a.append((coimg[bslc], com1[bslc], ima, cochi1[bslc]))
            
            # for m,chi,txt in [(com0,cochi0,'Before'),(com1,cochi1,'After')]:
            #     plt.clf()
            #     plt.subplot(2,2,1)
            #     plt.imshow(coimg, **ima)
            #     plt.subplot(2,2,2)
            #     plt.imshow(m, **ima)
            #     ax = plt.axis()
            #     plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
            #     plt.axis(ax)
            #     plt.subplot(2,2,3)
            #     plt.imshow(-chi, **imchi)
            #     plt.suptitle('%s optimization: %s band' % (txt, band))
            #     ps.savefig()


        for subims,rgbm in [(subims_b,rgbm0), (subims_a,rgbm1)]:
            plt.clf()
            for j,(im,m,ima,chi) in enumerate(subims):
                plt.subplot(3,4,1 + j + 0)
                plt.imshow(im, **ima)
                plt.subplot(3,4,1 + j + 4)
                plt.imshow(m, **ima)
                plt.subplot(3,4,1 + j + 8)
                plt.imshow(chi, **imchi)
            imx = dict(interpolation='nearest', origin='lower')
            plt.subplot(3,4,4)
            plt.imshow(np.dstack([rgbim[:,:,c][bslc] for c in [0,1,2]]), **imx)
            plt.subplot(3,4,8)
            plt.imshow(np.dstack([rgbm[:,:,c][bslc] for c in [0,1,2]]), **imx)
            #plt.imshow(rgbm, **imx)
            plt.subplot(3,4,12)
            plt.imshow(rgbim, **imx)
            ax = plt.axis()
            plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
            plt.axis(ax)
            ps.savefig()

        if b >= 4:
            break
        
        # ima = dict(interpolation='nearest', origin='lower')
        # for m,chi,txt in [(rgbm0,rgbchi0,'Before'), (rgbm1,rgbchi1,'After')]:
        #     plt.clf()
        #     for j,im in enumerate([rgbim, m]):
        #         plt.subplot(2,2,1+j)
        #         plt.imshow(im, **ima)
        #         ax = plt.axis()
        #         plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
        #         plt.axis(ax)
        #     plt.subplot(2,2,3)
        #     plt.imshow(-chi, **imchi)
        #     plt.suptitle('%s optimization: %s' % (txt, bands))
        #     ps.savefig()

# <codecell>


