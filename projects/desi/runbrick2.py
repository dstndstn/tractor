# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.resample import *
import pylab as plt
from tractor import *

if True:
    X = unpickle_from_file('brick.pickle')
    X.keys()
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
    plt.figure(figsize=(8,6));
    blobsrcs = []
    for blob in range(1, nblobs+1):
        blobsrcs.append(np.flatnonzero(T.blob == blob))

    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')

# <codecell>

    imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
                vmin=-5, vmax=5)

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
        for tim in tims:
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
            
            
        for step in range(10):
            dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        mod1 = [tractor.getModelImage(tim) for tim in tims]

        rgbim = np.zeros((H,W,3))
        rgbm0 = np.zeros((H,W,3))
        rgbm1 = np.zeros((H,W,3))
        rgbchi0 = np.zeros((H,W,3))
        rgbchi1 = np.zeros((H,W,3))

        for ib,band in enumerate(bands):
            coimg = np.zeros((H,W))
            com0  = np.zeros((H,W))
            com1  = np.zeros((H,W))
            con   = np.zeros((H,W))
            cochi0 = np.zeros((H,W))
            cochi1 = np.zeros((H,W))
            for tim,m0,m1 in zip(tims, mod0, mod1):
                if tim.band != band:
                    continue
                (Yo,Xo,Yi,Xi) = tim.resamp
                coimg[Yo,Xo] += tim.getImage ()[Yi,Xi]
                cochi0[Yo,Xo] += (tim.getImage()[Yi,Xi] - m0[Yi,Xi]) * tim.getInvError()[Yi,Xi]
                cochi1[Yo,Xo] += (tim.getImage()[Yi,Xi] - m1[Yi,Xi]) * tim.getInvError()[Yi,Xi]
                com0 [Yo,Xo] += m0[Yi,Xi]
                com1 [Yo,Xo] += m1[Yi,Xi]
                con  [Yo,Xo] += 1
                mn,mx = tim.zr

            coimg /= np.maximum(con,1)
            com0  /= np.maximum(con,1)
            com1  /= np.maximum(con,1)

            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            c = 2-ib
            rgbim[:,:,c] = np.clip((coimg - mn) / (mx - mn), 0., 1.)
            rgbm0[:,:,c] = np.clip((com0  - mn) / (mx - mn), 0., 1.)
            rgbm1[:,:,c] = np.clip((com1  - mn) / (mx - mn), 0., 1.)

            mn,mx = -5,5
            rgbchi0[:,:,c] = np.clip((cochi0 - mn) / (mx - mn), 0, 1)
            rgbchi1[:,:,c] = np.clip((cochi1 - mn) / (mx - mn), 0, 1)
            
            for m,chi,txt in [(com0,cochi0,'Before'),(com1,cochi1,'After')]:
                plt.clf()
                plt.subplot(2,2,1)
                plt.imshow(coimg, **ima)
                plt.subplot(2,2,2)
                plt.imshow(m, **ima)
                ax = plt.axis()
                plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
                plt.axis(ax)
                plt.subplot(2,2,3)
                plt.imshow(-chi, **imchi)
                plt.suptitle('%s optimization: %s band' % (txt, band))
                ps.savefig()

        ima = dict(interpolation='nearest', origin='lower')
        for m,chi,txt in [(rgbm0,rgbchi0,'Before'), (rgbm1,rgbchi1,'After')]:
            plt.clf()
            for j,im in enumerate([rgbim, m]):
                plt.subplot(2,2,1+j)
                plt.imshow(im, **ima)
                ax = plt.axis()
                plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
                plt.axis(ax)
            plt.subplot(2,2,3)
            plt.imshow(-chi, **imchi)
            plt.suptitle('%s optimization: %s' % (txt, bands))
            ps.savefig()

# <codecell>


