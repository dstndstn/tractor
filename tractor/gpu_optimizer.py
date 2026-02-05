import time

import numpy as np
from cupy_wrapper import cp
#import cupy as cp

from tractor.factored_optimizer import FactoredDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, ConstantSky
from tractor.brightness import LinearPhotoCal

'''
TO-DO:
-- check in oneblob.py - are we passing NormalizedPixelizedPsfEx (VARYING) PSF objects
   in at some point?  They should all get turned into constant PSFs.
-- check PSF sampling != 1.0

'''


class GPUFriendlyOptimizer(FactoredDenseOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_super = 0.
        self.n_super = 0

    def getLinearUpdateDirection(self, tr, priors=True, get_icov=False, **kwargs):
        if not (tr.isParamFrozen('images') and
            (len(tr.catalog) == 1) and
            isinstance(tr.catalog[0], ProfileGalaxy)):
            return super().getLinearUpdateDirection(tr, priors=priors, get_icov=get_icov,
                                                    **kwargs)
        assert(get_icov == False)
        return self.one_galaxy_update(tr, priors=priors, **kwargs)

    # def per_image_updates(self, tr, **kwargs):
    #     if not (tr.isParamFrozen('images') and
    #         (len(tr.catalog) == 1) and
    #         isinstance(tr.catalog[0], ProfileGalaxy)):
    #         t0 = time.time()
    #         R = super().per_image_updates(tr, **kwargs)
    #         dt = time.time() - t0
    #         self.t_super += dt
    #         self.n_super += 1
    #         return R
    #     return self.one_galaxy_update(tr, **kwargs)

    def one_galaxy_update(self, tr, **kwargs):
        #return super().all_image_updates(tr, **kwargs)
        return super().getLinearUpdateDirection(tr, **kwargs)

class GPUOptimizer(GPUFriendlyOptimizer):
    # This is the Vectorized version. (gpumode = 2)
    def one_galaxy_update(self, tr, **kwargs):
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        # predict memory required
        totalpix = sum([np.prod(tim.shape) for tim in tr.images])
        nd = tr.numberOfParams()+2
        kmax = 9
        # Double size of 16 bit (complex 128) array x npix x
        # n derivs x kmax.  5D array in batch_mixture_profiles.py
        est_mem = totalpix * nd * kmax * 16

        if free_mem < est_mem:
            print (f"Warning: Estimated memory {est_mem} is greater than free memory {free_mem}; Running CPU mode instead!")
            R_gpuv = super().one_galaxy_update(tr, **kwargs)
            return R_gpuv

        try:
            t0 = time.time()
            R_gpuv = self.gpu_one_galaxy_update(tr, **kwargs)
            #mempool = cp.get_default_memory_pool()
            #mempool.free_all_blocks()
            dt = time.time() - t0
            return R_gpuv
        except AssertionError:
            import traceback
            print ("AssertionError in VECTORIZED GPU code:")
            traceback.print_exc()
        except cp.cuda.memory.OutOfMemoryError:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            mempool = cp.get_default_memory_pool()
            used_bytes = mempool.used_bytes()
            tot_bytes = mempool.total_bytes()
            print ('Out of Memory for source: ', tr.catalog[0])
            print (f'OOM Device {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=}')
            mempool.free_all_blocks()
        except Exception as ex:
            import traceback
            print('Exception in GPU Vectorized code:')
            traceback.print_exc()
            raise(ex)

        # Fallback to CPU version
        print('Falling back to CPU code...')
        t0 = time.time()
        R = super().one_galaxy_update(tr, **kwargs)
        dt = time.time() - t0
        return R

    def gpu_one_galaxy_update(self, tr, priors=True, get_A=False, **kwargs):
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
        # Assume sky levels are zero.
        assert(all([tim.getSky().getValue() == 0 for tim in tr.images]))

        t0 = time.time()

        Nimages = len(tr.images)
        img_pix = [tim.getGpuImage() for tim in tr.images]
        img_ie  = [tim.getGpuInvError() for tim in tr.images]

        #print('img_pix:', img_pix)
        
        # Assume galaxy
        src = tr.catalog[0]
        assert(isinstance(src, ProfileGalaxy))
        psfs = [tim.getPsf() for tim in tr.images]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))
        #assert(all([isinstance(psf.pix, NormalizedPixelizedPsf) for psf in psfs]))
        # Assume ConstantSky models, grab constant sky levels
        # NOTE - instead of building this list and passing it around in ImageDerivs, etc,
        # we could perhaps just subtract it off img_pix at the start...
        img_sky = [tim.getSky().getConstant() for tim in tr.images]
        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskByIdx(i, src) for i in range(len(tr.images))]
        #masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        if any(m is None for m in masks):
            raise RuntimeError('One or more modelMasks is None in GPU code')

        print('sky', img_sky)
        assert(np.all([s == 0 for s in img_sky]))
        
        assert(all([m is not None for m in masks]))
        assert(src.isParamThawed('pos'))

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        px, py = np.array(pxy, dtype=np.float32).T

        # WCS inv(CD) matrix
        #img_cdi = [tim.getWcs().cdInverseAtPosition(src.getPosition(), src=src)
        #           for tim in tr.images]
        img_cdi = [tim.getWcs().cdInverseAtPixel(x,y) for tim,x,y in zip(tr.images, px, py)]
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]
        src_bands = src.getBrightness().getParamNames()

        img_bands = [src_bands.index(tim.getPhotoCal().band) for tim in tr.images]

        #img_params, cx,cy,pW,pH = self._getBatchImageParams(tr, masks, pxy)
        #def _getBatchImageParams(self, tr, masks, pxy):
        extents = [mm.extent for mm in masks]
        mh = [mm.h for mm in masks]
        mw = [mm.w for mm in masks]

        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        x0, x1, y0, y1 = np.asarray(extents).T
        print('extents:', extents)
        print('pixel positions:', pxy)
        print('gpu_halfsize terms: mm', (x1-x0)/2, (y1-y0)/2, 'source pos',
              1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py, 'PSF size', psfH//2, psfW//2)
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                                1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                                psfH//2, psfW//2]), axis=0)
        print('gpu_halfsize:', gpu_halfsize)

        # PSF: Fourier transforms & Mixtures-of-Gaussians
        P, (cx, cy), (pH, pW), (v, w), psf_mogs = get_vectorized_psfs(psfs, gpu_halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (Nimages,len(w),len(v)))

        amixes = [[('current', src._getShearedProfile(tim,x,y), 0.)] +
                  src.getDerivativeShearedProfiles(tim,x,y)
                  for tim,x,y in zip(tr.images, px, py)]

        #print('amixes', amixes)

        # nimages x nmixes(nderivs) x ncomponents x 3 (a,b,d variance terms)

        Nprofiles = max([len(d) for d in amixes])
        print('Nprofiles:', Nprofiles)
        Kmax = 0
        for amix_img in amixes:
            for _,m,_ in amix_img:
                #print('var shape', m.var.shape)
                K,_,_ = m.var.shape
                Kmax = max(K, Kmax)
        amix_vars = np.zeros((Nimages, Nprofiles, Kmax, 3), np.float32)
        amix_amps = np.zeros((Nimages, Nprofiles, Kmax), np.float32)
        for i,amix_img in enumerate(amixes):
            for j,(_,m,_) in enumerate(amix_img):
                K,_,_ = m.var.shape
                # called "a,b,d" elsewhere in the code
                amix_vars[i, j, :K, 0] = m.var[:, 0, 0]
                amix_vars[i, j, :K, 1] = m.var[:, 0, 1]
                amix_vars[i, j, :K, 2] = m.var[:, 1, 1]
                amix_amps[i, j, :K] = m.amp
        #print('amix_vars:', amix_vars)
        print('px,py', px,py)
        print('cx,cy', cx,cy)
        print('pH,pW', pH,pW)

        # Round source pixel position to nearest integer
        ipx = px.round().astype(np.int32)
        ipy = py.round().astype(np.int32)
        # subpixel portion: shifted via Lanczos interpolation
        mux = px - ipx
        muy = py - ipy
        print('mu x,y', mux, muy)
        assert(np.abs(mux).max() <= 0.5)
        assert(np.abs(muy).max() <= 0.5)

        # ugh, pixel shifts
        # dx = px - cx
        # dy = py - cy
        # mux = dx - x0
        # muy = dy - y0
        # sx = mux.round().astype(np.int32)
        # sy = muy.round().astype(np.int32)
        # # the subpixel portion will be handled with a Lanczos interpolation
        # mux -= sx
        # muy -= sy
        # print('mu x,y', mux, muy)
        # dxi = np.asarray(x0+sx)
        # dyi = np.asarray(y0+sy)
        # assert(np.all(sy <= 0))
        # assert(np.all(sx <= 0))
        # print('sx,sy', sx,sy)

        # Embed pix and ie in images the same size as pW,pH.
        # FIXME -- should be able to cache this; rationalize pixel transfer to GPU.
        padpix = cp.zeros((Nimages, pH,pW), cp.float32)
        padie  = cp.zeros((Nimages, pH,pW), cp.float32)
        for i, (pix,ie) in enumerate(zip(img_pix, img_ie)):
            #print('pix:')
            #p = cp.array(pix[y0[i]:y1[i], x0[i]:x1[i]])
            p = pix[y0[i]:y1[i], x0[i]:x1[i]]
            #print('p:', p)

            subx = x0[i] - (ipx[i] - cx)
            suby = y0[i] - (ipy[i] - cy)

            #padpix[i, -sy[i]:-sy[i]+mh[i], -sx[i]:-sx[i]+mw[i]] = p
            padpix[i, suby:suby+mh[i], subx:subx+mw[i]] = p
            #p = cp.array( ie[y0[i]:y1[i], x0[i]:x1[i]])
            p = ie[y0[i]:y1[i], x0[i]:x1[i]]
            #padie [i, -sy[i]:-sy[i]+mh[i], -sx[i]:-sx[i]+mw[i]] = p
            padie[i, suby:suby+mh[i], subx:subx+mw[i]] = p

        nsigma1 = 3.
        nsigma2 = 4.

        # Nimages x Nprofiles x Nmog
        vv = amix_vars[:,:,:,0] + amix_vars[:,:,:,2]
        IM = ((pW/2)**2 < (nsigma2**2 * vv))
        IF = ((pW/2)**2 > (nsigma1**2 * vv))

        print('IM', IM.shape)
        print(IM)

        print('IF', IF.shape)
        print(IF)

        ramp = np.any((IM*IF))
        print('Ramp?', ramp)

        mogweights = np.ones(vv.shape, dtype=cp.float32)
        fftweights = np.ones(vv.shape, dtype=cp.float32)

        # print('HACK render all')
        # IM[:,:,:] = True
        # IF[:,:,:] = True
        # ramp = False

        if ramp:
            # ramp
            ns = (pW/2) / np.maximum(1e-6, np.sqrt(vv))
            mogweights = np.minimum(1., (nsigma2 - ns) / (nsigma2 - nsigma1))*IM
            fftweights = np.minimum(1., (ns - nsigma1) / (nsigma2 - nsigma1))*IF
            assert(np.all(mogweights[IM] > 0.))
            assert(np.all(mogweights[IM] <= 1.))
            assert(np.all(fftweights[IF] > 0.))
            assert(np.all(fftweights[IF] <= 1.))

        # Between the images and the derivatives, "IM" and "IF" should be largely the same;
        # they mostly just depend on the K.  Set "nmog" and "nfft" to be scalars (0 to K).
        Nmog = IM.sum(axis=2).max(axis=(0,1))
        Nfft = IF.sum(axis=2).max(axis=(0,1))
        print('Nmog', Nmog)
        print('Nfft', Nfft)

        # mog_mix_vars = np.zeros((Nimages, Nprofiles, Nmog, 3), np.float32)
        # mog_mix_amps = np.zeros((Nimages, Nprofiles, Nmog), np.float32)
        # fft_mix_vars = np.zeros((Nimages, Nprofiles, Nfft, 3), np.float32)
        # fft_mix_amps = np.zeros((Nimages, Nprofiles, Nfft), np.float32)

        # -> BatchGalaxyProfiles in batch_mixture_profiles.py
        
        print('mogweights:', mogweights.shape)

        mog_mix_vars = amix_vars[:, :, -Nmog:, :]
        mog_mix_amps = (amix_amps * mogweights)[:, :, -Nmog:]

        fft_mix_vars = amix_vars[:, :, :Nfft:, :]
        fft_mix_amps = (amix_amps * fftweights)[:, :, :Nfft]

        # -> computeUpdateDirectionsVectorized in OLD_gpu_optimizer.py

        # -> computeGalaxyModelsVectorized in OLD_gpu_optimizer.py

        print('fft_mix_amps:', fft_mix_amps.shape)
        # Nimages x Nprofiles x Nfft(K)

        ## FIXME -- chunk large arrays / memory efficient...

        g_fft_mix_vars = cp.array(fft_mix_vars)
        g_fft_mix_amps = cp.array(fft_mix_amps)
        a = g_fft_mix_vars[:, :, :, 0]
        b = 2. * g_fft_mix_vars[:, :, :, 1]
        d = g_fft_mix_vars[:, :, :, 2]
        gv = cp.array(v)
        gw = cp.array(w)
        Nv = len(v)
        Nw = len(w)
        print('v,w', Nv, Nw)
        
        Fsum = cp.zeros((Nimages, Nprofiles, Nw, Nv), cp.float32)
        nu = cp.newaxis
        for k in range(Nfft):
            Fsum += (g_fft_mix_amps[:, :, k, nu, nu] *
                     cp.exp(-2. * cp.pi**2 *
                            (a[:, :, k, nu, nu] *  v[nu, nu, nu, :]**2 +
                             d[:, :, k, nu, nu] *  w[nu, nu, :, nu]**2 +
                             b[:, :, k, nu, nu] * (v[nu, nu, nu, :] *
                                                   w[nu, nu, :, nu]))))
        print('P', P.shape, P.dtype)
        print('Fsum', Fsum.shape, Fsum.dtype)

        G = cp.fft.irfft2(Fsum * P[:, nu, :, :])
        print('G', G.shape, G.dtype)
        print('mu x, y', mux, muy)

        import pylab as plt
        cG = G.get()
        plt.clf()
        k = 1
        for i in range(Nimages):
            for j in range(Nprofiles):
                plt.subplot(Nimages, Nprofiles, k)
                k += 1
                if j == 0:
                    plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
                else:
                    plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
        plt.savefig('gf.png')

        if Nmog > 0:
            ## FIXME -- trim these arrays to just non-zero weighted elements???

            print('mog_mix_amps:', mog_mix_amps.shape)
            print('mog_mix_vars:', mog_mix_vars.shape)

            psf_amps, psf_vars = psf_mogs
            #print('PSF amp')
            #print(psf_amps)
            #print('PSF var')
            #print(psf_vars)
            _,Npsfmog = psf_amps.shape

            assert(G.shape == (Nimages,Nprofiles,pH,pW))
            assert(mog_mix_amps.shape == (Nimages, Nprofiles, Nmog))
            assert(mog_mix_vars.shape == (Nimages, Nprofiles, Nmog, 3))
            assert(psf_amps.shape == (Nimages, Npsfmog))
            assert(psf_vars.shape == (Nimages, Npsfmog, 3))

            # Convolve!
            Ncmog = Nmog * Npsfmog
            conv_amps = (mog_mix_amps[:, :, :, nu]    * psf_amps[:, nu, nu, :]).reshape((Nimages, Nprofiles, Ncmog))
            conv_vars = (mog_mix_vars[:, :, :, nu, :] + psf_vars[:, nu, nu, :, :]).reshape((Nimages, Nprofiles, Ncmog, 3))

            # variance terms: (00, 01, 11) covariance matrix elements
            det = (conv_vars[:,:,:,0] * conv_vars[:,:,:,2] - conv_vars[:,:,:,1]**2)
            iv0 = conv_vars[:,:,:,2] / det
            iv1 = -2. * conv_vars[:,:,:,1] / det
            iv2 = conv_vars[:,:,:,0] / det
            scale = conv_amps / (2. * np.pi * np.sqrt(det))

            # print('dx', dx.shape, dx.dtype)
            # print('dy', dy.shape, dy.dtype)
            # print('dx', dx)
            # print('dy', dy)
            #means[:,:,:,0] -= img_params.dx[:,None,None]
            #means[:,:,:,1] -= img_params.dy[:,None,None]
    
            #print('iv0:', iv0.shape)
            assert(iv0.shape == (Nimages, Nprofiles, Ncmog))
            #print('iv1:', iv1.shape)
            #print('iv2:', iv2.shape)
    
             #print('dxi', dxi)
             #print('dyi', dyi)
    
            iv0 = cp.array(iv0, dtype=cp.float32)
            iv1 = cp.array(iv1, dtype=cp.float32)
            iv2 = cp.array(iv2, dtype=cp.float32)
            scale = cp.array(scale, dtype=cp.float32)

            # We're going to do the sub-pixel shift with Lanczos, so the
            # x position and mean are both integers.
            ## FIXME -- int type???
            #meanx = cp.array(-dx, dtype=cp.float32)
            #meany = cp.array(-dy, dtype=cp.float32)
            t = cp.int32
            xx = cp.arange(0-cx, pW-cx, dtype=t)
            yy = cp.arange(0-cy, pH-cy, dtype=t)
            del t

            # The distsq array is going to be nimages x nderivs x nmog x ny x nx
            distsq = -0.5 * (iv0[:,:,:,nu,nu] *  xx[nu,nu,nu,nu,:]**2 +
                             iv1[:,:,:,nu,nu] * (xx[nu,nu,nu,nu,:] *
                                                 yy[nu,nu,nu,:,nu]) +
                             iv2[:,:,:,nu,nu] *  yy[nu,nu,nu,:,nu]**2)
            del xx, yy, iv0, iv1, iv2
            distsq = cp.exp(distsq)
            assert(distsq.shape == (Nimages, Nprofiles, Ncmog, pH, pW))
            assert(scale.shape == (Nimages, Nprofiles, Ncmog))
            # Sum over the MoG components
            G_mog = cp.sum(distsq * scale[..., nu, nu], axis=2)
            del distsq
            print('G_mog', G_mog)
            assert(G_mog.shape == G.shape)

            import pylab as plt
            cG = G_mog.get()
            plt.clf()
            k = 1
            for i in range(Nimages):
                for j in range(Nprofiles):
                    plt.subplot(Nimages, Nprofiles, k)
                    k += 1
                    if j == 0:
                        plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
                    else:
                        plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
            plt.savefig('gm.png')

            G += G_mog
            del G_mog

        # FIXME -- check that this all remains float32 through the computations
        lanczos_shift_images_inplace_gpu(G, mux, muy)

        import pylab as plt
        cG = G.get()
        plt.clf()
        k = 1
        for i in range(Nimages):
            for j in range(Nprofiles):
                plt.subplot(Nimages, Nprofiles, k)
                k += 1
                if j == 0:
                    plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
                else:
                    plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
        plt.savefig('gf2.png')

        assert(G.shape == (Nimages,Nprofiles,pH,pW))

        # Now we have computed the galaxy profiles for the finite-differences steps.
        # Turn these into derivatives and solve the update matrix.
        # - compute per-image updates and commbine?
        # - build one big matrix?

        print('Galaxy params:')
        src.printThawedParams()

        assert(padpix.shape == (Nimages, pH, pW))
        assert(padie.shape == (Nimages, pH, pW))

        # We have either RaDecPos or GaiaPositions for the pos -- 2 or 0 params.
        Npos = len(src.pos.getParams())
        print('Source position parameters:', Npos)
        assert(Npos in [0, 2])

        print('source has bands:', src_bands)

        # List of the band for each tim
        tim_bands = [tim.getPhotoCal().band for tim in tr.images]
        print('tims have bands:', tim_bands)

        # The unique set of bands in all tims.  This is the number of flux parameters we
        # will fit for in the A matrix, in this order.
        tim_ubands = list(np.unique(tim_bands))
        print('Unique set of tim bands:', tim_ubands)
        # List of the index into tim_ubands for each tim
        # This is what we'll use to build the A matrix
        tim_band_index = [tim_ubands.index(b) for b in tim_bands]

        #print('tims index into source bands:', img_bands)
        Nbands = len(tim_ubands)

        # Nprofiles - 1: profile 0 is the current galaxy shape, the rest are shape/sersic derivs.
        Nshapes = (Nprofiles - 1)
        Nderivs = Nshapes + Npos + Nbands

        # We'll need both directions of the mapping between
        # source parameter index and index in our A matrix
        # due to "tims" having only a subset of the bands in the source's params.
        # List of (source parameter index, A matrix index)
        param_indices = []
        if Npos:
            param_indices.append((0,0))
            param_indices.append((1,1))
        for i,band in enumerate(src_bands):
            if band in tim_ubands:
                param_indices.append((Npos + i, Npos + tim_ubands.index(band)))
        for i in range(Nshapes):
            param_indices.append((Npos + len(src_bands) + i, Npos + Nbands + i))
        print('Parameter indices:', param_indices)

        src_to_fit_param = dict(param_indices)
        fit_to_src_param = dict([(f,s) for s,f in param_indices])

        # We can produce the residual map from the nominal galaxy profile model
        # (scaled by flux) and the image pix.
        assert(all([isinstance(tim.getPhotoCal(), LinearPhotoCal) for tim in tr.images]))
        #fluxes = []
        #for tim in tr.images:
        #fluxes = cp.array([tim.getPhotoCal().getScale() for tim in tr.images])
        fluxes = cp.array([tim.getPhotoCal().brightnessToCounts(src.brightness)
                           for tim in tr.images])

        import pylab as plt
        chi_pix = (padpix - fluxes[:, nu, nu] * G[:, 0, :, :]) * padie
        cChi = chi_pix.get()
        cG = G.get()
        cpix = padpix.get()
        cflux = fluxes.get()
        print('Fluxes:', cflux)
        plt.clf()
        k = 1
        for i in range(Nimages):
            plt.subplot(Nimages, 3, 3*i+1)
            mod = cflux[i, nu, nu] * cG[i, 0, :, :]
            plt.imshow(mod, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.subplot(Nimages, 3, 3*i+2)
            img = cpix[i,:,:]
            plt.imshow(img, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.subplot(Nimages, 3, 3*i+3)
            plt.imshow(cChi[i, :, :], interpolation='nearest', origin='lower')
        plt.savefig('gchi.png')

        Npix_total = Nimages * pH * pW

        Npriors = 0
        if priors:
            priorVals = tr.getLogPriorDerivatives()
            if priorVals is not None:
                rA, cA, vA, pb, mub = priorVals
                Npriors = max(Npriors, max([1+max(r) for r in rA]))

        Nrows = Npix_total + Npriors
        Ncols = Nderivs
        A = cp.zeros((Nrows, Ncols), cp.float32)
        B = cp.zeros(Nrows, cp.float32)

        if Npos:
            # Spatial derivatives:
            dx = cp.empty((Nimages, pH, pW), np.float32)
            dy = cp.empty((Nimages, pH, pW), np.float32)
            # zero out the edge pixels
            dx[:,:, 0] = 0
            dx[:,:,-1] = 0
            dx[:, 0,:] = 0
            dx[:,-1,:] = 0
            dy[:,:, 0] = 0
            dy[:,:,-1] = 0
            dy[:, 0,:] = 0
            dy[:,-1,:] = 0
            # We leave a one-pixel margin in the spatial derivatives
            # in both axes, because we're going to turn these dx,dy
            # derivatives into RA,Dec derivatives, and having one of
            # the terms artificially zero is confusing.
            dx[:, 1:-1, 1:-1] = (G[:, 0, 1:-1, 2:  ] - G[:, 0, 1:-1,  :-2]) / 2.
            dy[:, 1:-1, 1:-1] = (G[:, 0, 2:  , 1:-1] - G[:, 0,  :-2, 1:-1]) / 2.

            cdi = cp.array(img_cdi)
            assert(cdi.shape == (Nimages,2,2))

            A[:Npix_total, 0] = -(fluxes[:, nu, nu] *
                                  (dx * cdi[:, 0, 0][:, nu, nu] +
                                   dy * cdi[:, 1, 0][:, nu, nu]) * padie).flat
            A[:Npix_total, 1] = -(fluxes[:, nu, nu] *
                                  (dx * cdi[:, 0, 1][:, nu, nu] +
                                   dy * cdi[:, 1, 1][:, nu, nu]) * padie).flat

        # Flux derivatives.
        for i in range(Nimages):
            # Image i fills in the column corresponding to its flux
            # and a block of rows corresponding to its pixels.
            col = tim_band_index[i] + Npos
            A[i * pH*pW: (i+1) * pH*pW, col] = (G[i, 0, :, :] * padie[i, :, :]).flat
        # We *could* form the *col* into an array and do this as a
        # one-liner; not sure that would be faster.

        # Shape derivatives.
        # This gets the axis ordering wrong in the reshape -- need a swapaxes or something...
        #A[:Npix_total, Npos + Nbands:] = ((G[:, 1:, :, :] - G[:, 0, :, :][:, nu, :, :]) *
        #   padie[:, nu, :, :]).reshape((-1, Nshapes))
        stepsizes = np.empty((Nimages, Nshapes), np.float32)
        for i_img, amix_img in enumerate(amixes):
            for i_shape,(_,_,step) in enumerate(amix_img[1:]):
                stepsizes[i_img, i_shape] = step
        steps = cp.array(stepsizes)
        del stepsizes
        for i in range(Nshapes):
            A[:Npix_total, Npos + Nbands + i] = (fluxes[:, nu, nu] *
                                                 (G[:, i+1, :, :] - G[:, 0, :, :]) /
                                                 steps[:, i, nu, nu] *
                                                 padie).flat

        B[:Npix_total] = ((padpix - fluxes[:, nu, nu] * G[:, 0, :, :]) * padie).flat
        #chi_pix = (padpix - fluxes[:, nu, nu] * G[:, 0, :, :]) * padie

        if Npriors > 0:
            rA, cA, vA, pb, mub = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                for rij,vij,bij in zip(ri, vi, bi):
                    # Map the source's parameter index to our local fitting parameter index
                    ci = src_to_fit_param[ci]
                    A[Npix_total + rij, ci] = vij
                    B[Npix_total + rij] += bij
            del priorVals, rA, cA, vA, pb, mub

        plt.clf()
        plt.imshow(A.get() != 0, interpolation='nearest', origin='lower',
                   vmin=0, vmax=1, cmap='gray', aspect='auto')
        plt.savefig('a.png')

        plt.clf()
        cB = B.get()
        mx = np.abs(cB).max()
        plt.imshow(cB[:,nu], interpolation='nearest', origin='lower', aspect='auto',
                   cmap='RdBu', vmin=-mx, vmax=mx)
        plt.savefig('b.png')

        # Precondition A: column scales
        colscales = cp.sqrt((A**2).sum(axis=0))
        print('Colscales:', colscales.get())
        A /= colscales[nu, :]

        plt.clf()
        cA = A.get()
        mx = np.abs(cA).max()
        plt.imshow(cA, interpolation='nearest', origin='lower', aspect='auto',
                   cmap='RdBu', vmin=-mx, vmax=mx)
        plt.savefig('a2.png')

        X,_,_,_ = cp.linalg.lstsq(A, B)
        print('(scaled) X:', X.get())
        X /= colscales
        print('X:', X.get())

        # Parameter indices in src of our vector X:
        X = X.get()
        I = np.array([fit_to_src_param[i] for i in range(len(X))])

        print('Source parameter indices:', I)
        sX = np.zeros(src.numberOfParams(), np.float32)
        sX[I] = X

        if get_A:
            sA = np.zeros((Nrows, src.numberOfParams()), np.float32)
            sA[:,I] = A.get()
            return sA, B.get(), sX, colscales.get(), pH,pW

        return sX
        # HACK -- we should override getLinearUpdateDirection
        #return [(X, 1., I)]

def get_vectorized_psfs(psfs, halfsize):
    from tractor.batch_mixture_profiles import BatchMixtureOfGaussians
    #from tractor.batch_psf import BatchPixelizedPSF

    psfmogs = []
    maxK = 0
    for i,psf in enumerate(psfs):
        assert(isinstance(psf, HybridPSF))
        psfmog = psf.getMixtureOfGaussians()
        psfmogs.append(psfmog)
        maxK = max(maxK, psfmog.K)
    N = len(psfs)
    # We're going to assert zero mean here, and flatten the variance
    amps = np.zeros((N, maxK))
    #means = np.zeros((N, maxK, 2))
    varrs = np.zeros((N, maxK, 3))
    for i,psfmog in enumerate(psfmogs):
        amps [i, :psfmog.K] = psfmog.amp
        assert(np.all(psfmog.mean == 0))
        #means[i, :psfmog.K, :] = psfmog.mean
        varrs[i, :psfmog.K, 0] = psfmog.var[:, 0, 0]
        varrs[i, :psfmog.K, 1] = psfmog.var[:, 0, 1]
        varrs[i, :psfmog.K, 2] = psfmog.var[:, 1, 1]
    psf_mogs = amps,varrs

    imsize = psfs[0].img.shape
    for psf in psfs:
        print('PSF:', psf)
        print('sampling:', psf.sampling)
        assert(psf.sampling == 1.)
        print('pixelized size', psf.img.shape)

    sz = 2**int(np.ceil(np.log2(halfsize.max() * 2.)))
    ###pad, cx, cy = self._padInImageBatchGPU(sz, sz)
    W = H = sz
    pad = cp.zeros((N, H, W), cp.float32)
    cx = W//2
    cy = H//2
    for i,psf in enumerate(psfs):
        psfimg = psf.img
        ph,pw = psfimg.shape
        # We assume the center of the PSF image is at:
        pcy,pcx = ph//2, pw//2
        # And it must end up at cx,cy in the padded image.
        if pcx >= cx:
            # Trimming the PSF image
            out_x0 = 0
            in_x0 = pcx - cx
        else:
            # Padding the PSF image
            in_x0 = 0
            out_x0 = cx - pcx
        nx = min(pw, W)

        if pcy >= cy:
            # Trimming the PSF image
            out_y0 = 0
            in_y0 = pcy - cy
        else:
            # Padding the PSF image
            in_y0 = 0
            out_y0 = cy - pcy
        ny = min(ph, H)

        pad[i, out_y0 : out_y0 + ny, out_x0 : out_x0 + nx] = cp.array(
            psfimg[in_y0 : in_y0 + ny, in_x0 : in_x0 + nx])

    P = cp.fft.rfft2(pad)
    #print('P type', P.dtype)
    v = cp.fft.rfftfreq(W)
    w = cp.fft.fftfreq(H)
    #print('v,w', v.dtype, w.dtype)
    # FIXME -- ??
    v = v.astype(cp.float32)
    w = w.astype(cp.float32)

    ## FIXME -- turn this into a function, rather than a class
    #batch_psf = BatchPixelizedPSF(psfs)
    ###sz = self.getFourierTransformSizeBatchGPU(radius)
    ###sz = 2**int(np.ceil(np.log2(radius.max() * 2.)))
    #pad, cx, cy = self._padInImageBatchGPU(sz, sz)
    #P = cp.fft.rfft2(pad)
    #P = P.astype(cp.complex64)
    #nimages, pH, pW = pad.shape
    #v = cp.fft.rfftfreq(pW)
    #w = cp.fft.fftfreq(pH)
    #P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(px, py, halfsize)

    return P, (cx, cy), (H, W), (v, w), psf_mogs

def lanczos_shift_images_inplace_gpu(G, x, y, work=None):
    '''
    Only vectorized in the specific way we need:
    G images:
    (Nimages x Nmodels x H x W)
    x, y:
    each of length (Nimages,)
    work:
    same shape as G; pre-allocated work array.
    '''
    assert(len(G.shape) == 4)
    if work is not None:
        assert(work.shape == G.shape)
    assert(len(x) == len(y))

    Nim, Nmod, H, W = G.shape
    assert(len(x) == Nim)

    # Create Lanczos filter arrays, shape (Nim, 7)
    fx = np.arange(-3, +4)[np.newaxis, :] + x[:, np.newaxis]
    fy = np.arange(-3, +4)[np.newaxis, :] + y[:, np.newaxis]
    fx = lanczos_filter(3, fx)
    fy = lanczos_filter(3, fy)
    #print('lanczos x', fx)
    #print('lanczos y', fy)

    correlate7_2d_inplace_gpu(G, fx, fy, work=work)
    # correlate7f_inout(inimg, inimg_dim1, inimg_dim2,
    #                   outimg, outimg_dim1, outimg_dim2,
    #                   filtx, 7,
    #                   filty, 7,
    #                   work, work_dim1, work_dim2);

    del work

# eg, lanczos_filter(3, -0.3 + np.arange(-3, +4))
def lanczos_filter(order, x):
    x = np.atleast_1d(x)
    nz = np.flatnonzero(np.logical_and(x != 0., np.logical_and(x < order, x > -order)))
    out = np.zeros(x.shape, dtype=np.float32)
    pinz = np.pi * x.flat[nz]
    out.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    out /= np.sum(out, axis=-1)[..., np.newaxis]
    return out

def correlate7_2d_inplace_gpu(G, fx, fy, work=None):
    if work is None:
        work = cp.empty_like(G)
    else:
        ## FIXME - only really need work array to be larger; use a view
        assert(work.shape == G.shape)

    fx = cp.array(fx)
    fy = cp.array(fy)

    assert(len(G.shape) == 4)
    Nim,Nmod,H,W = G.shape
    assert(len(fx.shape) == 2)
    assert(fx.shape == fy.shape)
    Nim2,K = fx.shape
    assert(Nim2 == Nim)
    assert(K == 7)
    
    # Apply X filter

    na = cp.newaxis

    # Special handling - left edge.
    work[:, :, :, 0] = cp.sum(G[:, :, :, :4] * fx[:, na, na, 3:], axis=-1)
    work[:, :, :, 1] = cp.sum(G[:, :, :, :5] * fx[:, na, na, 2:], axis=-1)
    work[:, :, :, 2] = cp.sum(G[:, :, :, :6] * fx[:, na, na, 1:], axis=-1)

    # Special handling - right edge.
    work[:, :, :, -1] = cp.sum(G[:, :, :, -4:] * fx[:, na, na, :4], axis=-1)
    work[:, :, :, -2] = cp.sum(G[:, :, :, -5:] * fx[:, na, na, :5], axis=-1)
    work[:, :, :, -3] = cp.sum(G[:, :, :, -6:] * fx[:, na, na, :6], axis=-1)

    # Middle
    work[:, :, :, 3:-3]  = G[:, :, :,  :-6] * fx[:, na, na, na, 0]
    work[:, :, :, 3:-3] += G[:, :, :, 1:-5] * fx[:, na, na, na, 1]
    work[:, :, :, 3:-3] += G[:, :, :, 2:-4] * fx[:, na, na, na, 2]
    work[:, :, :, 3:-3] += G[:, :, :, 3:-3] * fx[:, na, na, na, 3]
    work[:, :, :, 3:-3] += G[:, :, :, 4:-2] * fx[:, na, na, na, 4]
    work[:, :, :, 3:-3] += G[:, :, :, 5:-1] * fx[:, na, na, na, 5]
    work[:, :, :, 3:-3] += G[:, :, :, 6:  ] * fx[:, na, na, na, 6]

    # Apply Y filter

    # Special handling - bottom edge.
    G[:, :, 0, :] = cp.sum(work[:, :, :4, :] * fy[:, na, 3:, na], axis=-2)
    G[:, :, 1, :] = cp.sum(work[:, :, :5, :] * fy[:, na, 2:, na], axis=-2)
    G[:, :, 2, :] = cp.sum(work[:, :, :6, :] * fy[:, na, 1:, na], axis=-2)

    # Special handling - top edge.
    G[:, :, -1, :] = cp.sum(work[:, :, -4:, :] * fy[:, na, :4, na], axis=-2)
    G[:, :, -2, :] = cp.sum(work[:, :, -5:, :] * fy[:, na, :5, na], axis=-2)
    G[:, :, -3, :] = cp.sum(work[:, :, -6:, :] * fy[:, na, :6, na], axis=-2)

    # Middle
    G[:, :, 3:-3, :]  = work[:, :,  :-6, :] * fy[:, na, na, 0, na]
    G[:, :, 3:-3, :] += work[:, :, 1:-5, :] * fy[:, na, na, 1, na]
    G[:, :, 3:-3, :] += work[:, :, 2:-4, :] * fy[:, na, na, 2, na]
    G[:, :, 3:-3, :] += work[:, :, 3:-3, :] * fy[:, na, na, 3, na]
    G[:, :, 3:-3, :] += work[:, :, 4:-2, :] * fy[:, na, na, 4, na]
    G[:, :, 3:-3, :] += work[:, :, 5:-1, :] * fy[:, na, na, 5, na]
    G[:, :, 3:-3, :] += work[:, :, 6:  , :] * fy[:, na, na, 6, na]




if __name__ == '__main__':
    # test PSF padding logic
    W = 64
    cx = W//2
    pad = np.zeros(W, np.float32)
    bigpsf = np.zeros(2*W, np.float32)
    bigcx = 64
    bigpsf[bigcx] = 1.
    for psfW in [32, 33, 63, 64, 65, 127]:
        pad = np.zeros(W, np.float32)
        print()
        offset = bigcx - psfW//2
        psf = bigpsf[offset : offset + psfW]
        print('PSF size:', psf.shape, 'vs', psfW)
        print('nz pix:', np.flatnonzero(psf))
        pw = psfW
        psfimg = psf
        # We assume the center of the PSF image is at:
        pcx = pw//2
        assert(pcx == np.flatnonzero(psf)[0])
        # And it must end up at cx,cy in the padded image.
        if pcx >= cx:
            # Trimming the PSF image
            out_x0 = 0
            in_x0 = pcx - cx
        else:
            # Padding the PSF image
            in_x0 = 0
            out_x0 = cx - pcx
        n = min(psfW, W)
        pad[out_x0: out_x0 + n] = psfimg[in_x0: in_x0 + n]
        print('Padded size:', pad.shape)
        print('nz pix:', np.flatnonzero(pad))
        print('cx:', cx)
