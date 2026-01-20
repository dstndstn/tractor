import time

import numpy as np
from cupy_wrapper import cp
#import cupy as cp

from tractor.factored_optimizer import FactoredDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, ConstantSky

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

    def all_image_updates(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
            (len(tr.catalog) == 1) and
            isinstance(tr.catalog[0], ProfileGalaxy)):
            t0 = time.time()
            R = super().all_image_updates(tr, **kwargs)
            dt = time.time() - t0
            self.t_super += dt
            self.n_super += 1
            return R
        return self.one_galaxy_updates(tr, **kwargs)

    def one_galaxy_updates(self, tr, **kwargs):
        return super().all_image_updates(tr, **kwargs)

class GPUOptimizer(GPUFriendlyOptimizer):
    # This is the Vectorized version. (gpumode = 2)
    def one_galaxy_updates(self, tr, **kwargs):
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
            R_gpuv = super().one_galaxy_updates(tr, **kwargs)
            return R_gpuv

        try:
            t0 = time.time()
            R_gpuv = self.gpu_one_galaxy_updates(tr, **kwargs)
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
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
        R = super().one_galaxy_updates(tr, **kwargs)
        dt = time.time() - t0
        return R

    def gpu_one_galaxy_updates(self, tr, **kwargs):
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)

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
        bands = src.getBrightness().getParamNames()

        img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]

        #img_params, cx,cy,pW,pH = self._getBatchImageParams(tr, masks, pxy)
        #def _getBatchImageParams(self, tr, masks, pxy):
        extents = [mm.extent for mm in masks]
        mh = [mm.h for mm in masks]
        mw = [mm.w for mm in masks]

        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        x0, x1, y0, y1 = np.asarray(extents).T
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                                1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                                psfH//2, psfW//2]), axis=0)
        print('gpu_halfsize:', gpu_halfsize)

        # PSF: Fourier transforms & Mixtures-of-Gaussians
        P, (cx, cy), (pH, pW), (v, w), psf_mogs = get_vectorized_psfs(psfs, px, py, gpu_halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (Nimages,len(w),len(v)))

        amixes = [[('current', src._getShearedProfile(tim,x,y), 0.)] +
                  src.getDerivativeShearedProfiles(tim,x,y)
                  for tim,x,y in zip(tr.images, px, py)]

        print('amixes', amixes)

        # nimages x nmixes(nderivs) x ncomponents x 3 (a,b,d variance terms)

        Nderivs = max([len(d) for d in amixes])
        print('Nderivs:', Nderivs)
        Kmax = 0
        for amix_img in amixes:
            for _,m,_ in amix_img:
                #print('var shape', m.var.shape)
                K,_,_ = m.var.shape
                Kmax = max(K, Kmax)
        amix_vars = np.zeros((Nimages, Nderivs, Kmax, 3), np.float32)
        amix_amps = np.zeros((Nimages, Nderivs, Kmax), np.float32)
        for i,amix_img in enumerate(amixes):
            for j,(_,m,_) in enumerate(amix_img):
                K,_,_ = m.var.shape
                # called "a,b,d" elsewhere in the code
                amix_vars[i, j, :K, 0] = m.var[:, 0, 0]
                amix_vars[i, j, :K, 1] = m.var[:, 0, 1]
                amix_vars[i, j, :K, 2] = m.var[:, 1, 1]
                amix_amps[i, j, :K] = m.amp
        #print('amix_vars:', amix_vars)

        print('cx,cy', cx,cy)
        print('pH,pW', pH,pW)
        print('px,py', px,py)
        # ugh, pixel shifts
        dx = px - cx
        dy = py - cy
        mux = dx - x0
        muy = dy - y0
        sx = mux.round().astype(np.int32)
        sy = muy.round().astype(np.int32)
        # the subpixel portion will be handled with a Lanczos interpolation
        mux -= sx
        muy -= sy
        dxi = np.asarray(x0+sx)
        dyi = np.asarray(y0+sy)
        assert(np.abs(mux).max() <= 0.5)
        assert(np.abs(muy).max() <= 0.5)
        assert(np.all(sy <= 0))
        assert(np.all(sx <= 0))
        print('sx,sy', sx,sy)
        
        # Embed pix and ie in images the same size as pW,pH.
        # FIXME -- should be able to cache this; rationalize pixel transfer to GPU.
        padpix = cp.zeros((Nimages, pH,pW), cp.float32)
        padie  = cp.zeros((Nimages, pH,pW), cp.float32)
        for i, (pix,ie) in enumerate(zip(img_pix, img_ie)):
            #print('pix:')
            #p = cp.array(pix[y0[i]:y1[i], x0[i]:x1[i]])
            p = pix[y0[i]:y1[i], x0[i]:x1[i]]
            #print('p:', p)
            padpix[i, -sy[i]:-sy[i]+mh[i], -sx[i]:-sx[i]+mw[i]] = p
            #p = cp.array( ie[y0[i]:y1[i], x0[i]:x1[i]])
            p = ie[y0[i]:y1[i], x0[i]:x1[i]]
            padie [i, -sy[i]:-sy[i]+mh[i], -sx[i]:-sx[i]+mw[i]] = p

        nsigma1 = 3.
        nsigma2 = 4.

        # Nimages x Nderivs x Kmax
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

        # mog_mix_vars = np.zeros((Nimages, Nderivs, Nmog, 3), np.float32)
        # mog_mix_amps = np.zeros((Nimages, Nderivs, Nmog), np.float32)
        # fft_mix_vars = np.zeros((Nimages, Nderivs, Nfft, 3), np.float32)
        # fft_mix_amps = np.zeros((Nimages, Nderivs, Nfft), np.float32)

        # -> BatchGalaxyProfiles in batch_mixture_profiles.py
        
        print('mogweights:', mogweights.shape)

        mog_mix_vars = amix_vars[:, :, -Nmog:, :]
        mog_mix_amps = (amix_amps * mogweights)[:, :, -Nmog:]

        fft_mix_vars = amix_vars[:, :, :Nfft:, :]
        fft_mix_amps = (amix_amps * fftweights)[:, :, :Nfft]

        # -> computeUpdateDirectionsVectorized in OLD_gpu_optimizer.py

        # -> computeGalaxyModelsVectorized in OLD_gpu_optimizer.py

        print('fft_mix_amps:', fft_mix_amps.shape)
        # Nimages x Nderivs x Nfft(K)

        ## FIXME -- chunk large arrays / memory efficient...

        g_fft_mix_vars = cp.array(fft_mix_vars)
        g_fft_mix_amps = cp.array(fft_mix_amps)
        a = g_fft_mix_vars[:, :, :, 0]
        b = g_fft_mix_vars[:, :, :, 1]
        d = g_fft_mix_vars[:, :, :, 2]
        gv = cp.array(v)
        gw = cp.array(w)
        Nv = len(v)
        Nw = len(w)
        print('v,w', Nv, Nw)
        
        Fsum = cp.zeros((Nimages, Nderivs, Nw, Nv), cp.float32)
        nu = cp.newaxis
        for k in range(Nfft):
            Fsum += (g_fft_mix_amps[:, :, k, nu, nu] *
                     cp.exp(-2. * cp.pi**2 *
                            (a[:, :, k, nu, nu] * v[nu, nu, nu, :]**2 +
                             d[:, :, k, nu, nu] * w[nu, nu, :, nu]**2 +
                             b[:, :, k, nu, nu] * v[nu, nu, nu, :] * w[nu, nu, :, nu])))

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
            for j in range(Nderivs):
                plt.subplot(Nimages, Nderivs, k)
                k += 1
                if j == 0:
                    plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
                else:
                    plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
        plt.savefig('gf.png')

        print('mog_mix_amps:', mog_mix_amps.shape)
        print('mog_mix_vars:', mog_mix_vars.shape)

        assert(G.shape == (Nimages,Nderivs,pH,pW))
        assert(mog_mix_amps.shape == (Nimages, Nderivs, Nmog))
        assert(mog_mix_vars.shape == (Nimages, Nderivs, Nmog, 3))
        # variance terms: (00, 01, 11) covariance matrix elements

        det = (mog_mix_vars[:,:,:,0] * mog_mix_vars[:,:,:,2] - mog_mix_vars[:,:,:,1]**2)
        iv0 = mog_mix_vars[:,:,:,2] / det
        iv1 = -2. * mog_mix_vars[:,:,:,1] / det
        iv2 = mog_mix_vars[:,:,:,0] / det
        scale = mog_mix_amps / (2. * np.pi * np.sqrt(det))

        print('dx', dx.shape, dx.dtype)
        print('dy', dy.shape, dy.dtype)

        print('dx', dx)
        print('dy', dy)
        #means[:,:,:,0] -= img_params.dx[:,None,None]
        #means[:,:,:,1] -= img_params.dy[:,None,None]

        print('iv0:', iv0.shape)
        assert(iv0.shape == (Nimages, Nderivs, Nmog))
        #print('iv1:', iv1.shape)
        #print('iv2:', iv2.shape)

        print('dxi', dxi)
        print('dyi', dyi)

        meanx = cp.array(-dx, dtype=cp.float32)
        meany = cp.array(-dy, dtype=cp.float32)
        iv0 = cp.array(iv0, dtype=cp.float32)
        iv1 = cp.array(iv1, dtype=cp.float32)
        iv2 = cp.array(iv2, dtype=cp.float32)
        xx = cp.arange(pW, dtype=cp.float32)
        yy = cp.arange(pH, dtype=cp.float32)
        # The distsq array is going to be nimages x nderivs x nmog x ny x nx
        na = cp.newaxis
        distsq = (iv0[:,:,:,na,na] * (xx[na,na,na,na,:] - meanx[:,na,na,na,na])**2 +
                  iv1[:,:,:,na,na] * (xx[na,na,na,na,:] - meanx[:,na,na,na,na]) *
                                     (yy[na,na,na,:,na] - meany[:,na,na,na,na]) +
                  iv2[:,:,:,na,na] * (yy[na,na,na,:,na] - meany[:,na,na,na,na])**2)
        distsq *= -0.5
        distsq = cp.exp(distsq)
        assert(distsq.shape == (Nimages, Nderivs, Nmog, pH, pW))
        assert(scale.shape == (Nimages, Nderivs, Nmog))
        G_mog = cp.sum(distsq * scale[..., na, na], axis=2)
        print('G_mog', G_mog)
        assert(G_mog.shape == G.shape)
        #mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(-0.5*distsq), axis=2).astype(cp.float32)
        #del means, distsq, varcopy

        import pylab as plt
        cG = G_mog.get()
        plt.clf()
        k = 1
        for i in range(Nimages):
            for j in range(Nderivs):
                plt.subplot(Nimages, Nderivs, k)
                k += 1
                if j == 0:
                    plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
                else:
                    plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
        plt.savefig('gm.png')

        # FIXME -- check that this all remains float32 through the computations
        lanczos_shift_images_inplace_gpu(G, mux, muy)

        import pylab as plt
        cG = G.get()
        plt.clf()
        k = 1
        for i in range(Nimages):
            for j in range(Nderivs):
                plt.subplot(Nimages, Nderivs, k)
                k += 1
                if j == 0:
                    plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
                else:
                    plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
        plt.savefig('gf2.png')


        # psf_mog.var + gal_mog.var
        # psf_mag.amp * gal_mog.amp
        # means (use lanczos instead)

        from tractor.batch_galaxy import getShearedProfileGPU, getDerivativeShearedProfilesGPU
        amix_gpu = getShearedProfileGPU(src, tr.images, px, py)
        amixes_gpu = getDerivativeShearedProfilesGPU(src, tr.images, px, py)
        amixes_gpu = [('current', amix_gpu, 0.)] + amixes_gpu

        print('amixes_gpu', amixes_gpu)
        mog = amixes_gpu[0][1]
        print('mog shape', mog)
        print('var', mog.var.get().shape)
        
        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)
        #return img_params, cx,cy, pH,pW

        
        img_derivs = self._getBatchGalaxyProfiles(amixes_gpu, masks, px, py, cx, cy, pW, pH,
                                                  img_counts, img_sky, img_pix, img_ie)
        img_params.addBatchGalaxyProfiles(img_derivs)

        # are we fitting for the position of this source?
        fit_pos = np.asarray([(src.getPosition().numberOfParams() > 0)]*Nimages)
        cdi = cp.asarray(img_cdi)
        img_params.addBatchGalaxyDerivs(cdi, fit_pos)

        #img_derivs.tostr()

        i = 0
        #Call collect_params() to finalize BatchImageParams object with all ImageDerivs
        #img_params.collect_params()
        #nbands = 1 + max(img_bands)
        nbands = len(bands)

        if img_params.ffts is None:
            raise RuntimeError('Warning> img_params.ffts is None!  Running CPU version')

        Xic = self.computeUpdateDirectionsVectorized(img_params, priorVals)

        mpool = cp.get_default_memory_pool()
        del img_params
        del amixes_gpu
        mpool.free_all_blocks()

        return Xic

    def gpu_one_galaxy_updates_2(self, img_params):
        print('gpu_one_galaxy_updates_2')
        raise RuntimeError('not implemented')

def get_vectorized_psfs(psfs, px, py, halfsize):
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
    amps = np.zeros((N, maxK))
    means = np.zeros((N, maxK, 2))
    varrs = np.zeros((N, maxK, 2, 2))
    for i,psfmog in enumerate(psfmogs):
        amps [i, :psfmog.K] = psfmog.amp
        means[i, :psfmog.K, :] = psfmog.mean
        varrs[i, :psfmog.K, :, :] = psfmog.var
    amps  = cp.asarray(amps)
    means = cp.asarray(means)
    varrs = cp.asarray(varrs)
    psf_mogs = BatchMixtureOfGaussians(amps, means, varrs, quick=True)

    imsize = psfs[0].img.shape
    for psf in psfs:
        print('PSF:', psf)
        print('sampling:', psf.sampling)
        assert(psf.sampling == 1.)
        print('pixelized size', psf.img.shape)
        assert(psf.img.shape == imsize)

    sz = 2**int(np.ceil(np.log2(halfsize.max() * 2.)))
    ###pad, cx, cy = self._padInImageBatchGPU(sz, sz)
    W = H = sz
    pad = cp.zeros((N, H, W), cp.float32)
    for i,psf in enumerate(psfs):
        img = psf.img
        ph,pw = img.shape
        if H >= ph:
            y0 = (H - ph) // 2
            cy = y0 + ph // 2
        else:
            y0 = 0
            cut = (ph - H) // 2
            img = img[:, cut:cut + H, :]
            cy = ph // 2 - cut
        if W >= pw:
            x0 = (W - pw) // 2
            cx = x0 + pw // 2
        else:
            x0 = 0
            cut = (pw - W) // 2
            img = img[:, :, cut:cut + W]
            cx = pw // 2 - cut
        sh, sw = img.shape
        img = cp.array(img)
        pad[i, y0:y0 + sh, x0:x0 + sw] = img
    
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
