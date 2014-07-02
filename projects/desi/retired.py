### Code that is not being used but that I'm not quite ready
### to part with...


def read_source_extractor_catalog(secatfn, zp):
    T = fits_table(secatfn, hdu=2,
                   column_map={'alpha_j2000':'ra', 'delta_j2000':'dec'},)
    print len(T), 'sources in SExtractor catalog'
    T.mag_psf      += zp
    T.mag_spheroid += zp
    T.mag_disk     += zp

    from projects.cs82.cs82 import get_cs82_sources
    cat,catI = get_cs82_sources(T, bands=['z'])
    T.cut(catI)
    return T, cat



def checkout_psf_models():
    print 'PSF', psf
    print type(psf)
    flux = SEcat.flux_auto
    ima = dict(interpolation='nearest', origin='lower', cmap='gray')

    H,W = tim.shape
    S = 20
    I = np.argsort(-flux)
    while False:
    #for i in I[:20]:
        #x,y = tim.wcs.positionToPixel(cat[i].getPosition())
        x,y = SEcat.x_image[i] - 1, SEcat.y_image[i] - 1
        if x < S or y < S or x+S >= W or y+S >= H:
            continue
        ix,iy = int(np.round(x)), int(np.round(y))
        subim = tim.getImage()[iy-S:iy+S+1, ix-S:ix+S+1]
        ext = [ix-S, ix+S, iy-S, iy+S]
        subiv = tim.getInvvar()[iy-S:iy+S+1, ix-S:ix+S+1]

        print 'Subimage max', subim.max()

        psfimg = psf.instantiateAt(ix, iy, nativeScale=True)

        subim /= subim.sum()
        psfimg /= psfimg.sum()
        mx = max(subim.max(), psfimg.max())
        ima.update(vmin=-0.05*mx, vmax=mx)

        sh,sw = subim.shape
        #subrgb = np.zeros((h,w,3))
        subrgb = plt.cm.gray((subim - ima['vmin']) / (ima['vmax'] - ima['vmin']))
        print 'subrgb', subrgb.shape
        bad = (1,0,0)
        for i in range(3):
            subrgb[:,:,i][subiv == 0] = bad[i]

        plt.clf()
        plt.subplot(2,4,1)
        #plt.imshow(subim, extent=ext, **ima)
        plt.imshow(subrgb, extent=ext, **ima)
        ax = plt.axis()
        plt.plot(x, y, 'o', mfc='none', mec='r', ms=12)
        plt.axis(ax)
        plt.title('Image')
        plt.subplot(2,4,2)
        #plt.imshow(psfimg, **ima)
        #plt.title('Image')

        pixpsf = PixelizedPSF(psfimg)
        patch = pixpsf.getPointSourcePatch(x - (ix-S), y - (iy-S))
        print 'Patch', patch.x0, patch.y0, patch.patch.shape

        psfsub = np.zeros_like(subim)
        patch.addTo(psfsub)
        psfsub /= psfsub.sum()
        print 'Pix sum', patch.patch.sum()
        print 'Pix max', psfsub.max()

        plt.imshow(psfsub, **ima)
        plt.title('PSF pix')

        mog = psf.mogAt(x, y)
        print 'PSF MOG:', mog
        patch = mog.getPointSourcePatch(x, y)
        print 'Patch', patch.x0, patch.y0, patch.patch.shape
        patch.x0 -= (ix - S)
        patch.y0 -= (iy - S)
        psfg = np.zeros_like(subim)
        patch.addTo(psfg)
        psfg /= psfg.sum()

        print 'Gauss sum', patch.patch.sum()
        print 'Gauss max', psfg.max()

        # Re-fit the PSF image as MoG
        # im = np.maximum(psfimg, 0)
        # PS = im.shape[0]
        # xm,ym = -(PS/2), -(PS/2)
        # K = 3
        # w,mu,var = em_init_params(K, None, None, None)
        # em_fit_2d(im, xm, ym, w, mu, var)
        # #print 'Re-fit params:', w, mu, var
        # repsf = GaussianMixturePSF(w, mu, var)
        # print 'Re-fit MOG:', repsf
        # patch = repsf.getPointSourcePatch(x, y)
        # print 'Patch', patch.x0, patch.y0, patch.patch.shape
        # patch.x0 -= (ix - S)
        # patch.y0 -= (iy - S)
        # psfg2 = np.zeros_like(subim)
        # patch.addTo(psfg2)
        # psfg2 /= psfg2.sum()

        plt.subplot(2,4,3)
        plt.imshow(psfg, **ima)
        plt.title('PSF Gaussian')


        plt.subplot(2,4,7)
        plt.imshow(-(subim - psfsub), interpolation='nearest', origin='lower',
                   cmap='RdBu')
        plt.title('Image - PsfPix')

        plt.subplot(2,4,8)
        plt.imshow(-(subim - psfg), interpolation='nearest', origin='lower',
                   cmap='RdBu')
        plt.title('Image - PsfG')
                   
        ima.update(vmin=0, vmax=np.sqrt(mx * 1.05))

        plt.subplot(2,4,5)
        plt.imshow(np.sqrt(subim + 0.05*mx), extent=ext, **ima)
        plt.title('sqrt Image')
        plt.subplot(2,4,6)
        #plt.imshow(np.sqrt(psfimg + 0.05*mx), **ima)
        plt.imshow(np.sqrt(psfsub + 0.05*mx), **ima)
        plt.title('sqrt PSF pix')
        
        ps.savefig()

    if secat:
        H,W = tim.shape
        I = np.argsort(T.mag_psf)
        ims = []
        ratios = []
        for i in I[:20]:
            print
            x,y = T.x_image[i], T.y_image[i]
            ix,iy = int(np.round(x)), int(np.round(y))

            psfim = tim.getPsf().getPointSourcePatch(x-1,y-1)
            ph,pw = psfim.shape
            print 'PSF shape', pw,ph
            S = ph/2
            if ix < S or ix > (W-S) or iy < S or iy > (H-S):
                continue
            #subim = tim.getImage()[iy-S:iy+S+1, ix-S:ix+S+1]
            x0,y0 = psfim.x0, psfim.y0
            subim = tim.getImage()[y0:y0+ph, x0:x0+pw]

            pixim = tim.getPsf().instantiateAt(x-1, y-1)
            print 'pixim', pixim.sum()
            
            mn,mx = [np.percentile(subim, p) for p in [25,100]]
            zp = tim.ozpscale
            print 'subim sum', np.sum(subim)
            print 'flux', T.flux_psf[i]
            print 'zpscale', zp
            flux = T.flux_psf[i] / zp
            print 'flux/zpscale', flux
            print 'psfim sum', psfim.patch.sum()

            dy = y - iy
            dx = x - ix
            
            ims.append((subim, psfim, flux, mn,mx, pixim,dx,dy))
            ratios.append(subim.sum() / flux)

        ratio = np.median(ratios)
        for subim, psfim, flux, mn,mx, pixim,dx,dy in ims:
            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            dd = (mx - mn) * 0.05
            imdiff = dict(interpolation='nearest', origin='lower', cmap='gray',
                          vmin=-dd, vmax=dd)

            mod = psfim.patch * flux * ratio
            pixmod = pixim * flux * ratio

            if dx < 0:
                dx += 1
            if dy < 0:
                dy += 1

            spix = shift(pixmod, (dy,dx))
            ph,pw = spix.shape
            fh,fw = mod.shape
            sx = (fw - pw)/2
            sy = (fh - ph)/2
            xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
            pixx = np.sum(xx * pixmod) / pixmod.sum()
            pixy = np.sum(yy * pixmod) / pixmod.sum()

            shiftpix = np.zeros_like(mod)
            shiftpix[sy:sy+ph, sx:sx+pw] = spix

            xx,yy = np.meshgrid(np.arange(fw), np.arange(fh))
            modx = np.sum(xx * mod) / mod.sum()
            mody = np.sum(yy * mod) / mod.sum()
            shx = np.sum(xx * shiftpix) / shiftpix.sum()
            shy = np.sum(yy * shiftpix) / shiftpix.sum()

            imx = np.sum(xx * subim) / subim.sum()
            imy = np.sum(yy * subim) / subim.sum()
            
            print
            print 'Dx,Dy', dx,dy
            print 'Model    centroid %.5f, %.5f' % (modx,mody)
            print 'Shiftpix centroid %.5f, %.5f' % (shx, shy )
            print 'Image    centroid %.5f, %.5f' % (imx,imy)
            #print 'Pixim    centroid %.5f, %.5f' % (pixx,pixy)

            print 'Image - Model     %.5f, %.5f' % (imx-shx,imy-shy)

            #shiftpix2 = shift(shiftpix, (imy-shy, imx-shx))
            shiftpix2 = shift(shiftpix,
                              (-(pixy-np.round(pixy)), -(pixx-np.round(pixx))))

            plt.clf()
            plt.suptitle('dx,dy %.2f,%.2f' % (dx,dy))

            plt.subplot(3,4,1)
            imshow(subim, **ima)
            plt.title('image')

            plt.subplot(3,4,5)
            imshow(mod, **ima)
            plt.title('G model')

            plt.subplot(3,4,9)
            imshow(subim - mod, **imdiff)
            plt.title('G resid')

            #plt.subplot(3,4,3)
            #imshow(subim, **ima)
            plt.subplot(3,4,3)
            imshow(pixmod, **ima)
            plt.title('Pixelize mod')
            plt.subplot(3,4,7)
            imshow(shiftpix, **ima)
            plt.title('Shifted pix mod')
            plt.subplot(3,4,11)
            imshow(subim - shiftpix, **imdiff)
            plt.title('Shifted pix resid')

            plt.subplot(3,4,10)
            imshow(subim - shiftpix2, **imdiff)
            plt.title('Shifted pix2 resid')

            
            plt.subplot(3,4,12)
            imshow(shiftpix - mod, **imdiff)
            plt.title('Shifted pix - G mod')

            
            #plt.subplot(3,4,11)
            #imshow(subim - pixmod, **imdiff)

            plt.subplot(3,4,4)
            sqimshow(subim, **ima)
            plt.title('Image')
            plt.subplot(3,4,8)
            sqimshow(pixmod, **ima)
            plt.title('Pix mod')

            
            plt.subplot(3,4,2)
            sqimshow(subim, **ima)
            plt.title('Image')
            plt.subplot(3,4,6)
            sqimshow(mod, **ima)
            plt.title('G mod')
            ps.savefig()
