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

        print('cx,cy', cx,cy)
        print('pH,pW', pH,pW)
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
        dxi = cp.asarray(x0+sx)
        dyi = cp.asarray(y0+sy)
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
            padpix[i, -sy[i]:-sy[i]+mh[i], -sx[i]:-sx[i]+mw[i]] = pix[y0[i]:y1[i], x0[i]:x1[i]]
            padie [i, -sy[i]:-sy[i]+mh[i], -sx[i]:-sx[i]+mw[i]] =  ie[y0[i]:y1[i], x0[i]:x1[i]]

        # nimages x nmixes x ncomponents

        from tractor.batch_galaxy import getShearedProfileGPU, getDerivativeShearedProfilesGPU
        amix_gpu = getShearedProfileGPU(src, tr.images, px, py)
        amixes_gpu = getDerivativeShearedProfilesGPU(src, tr.images, px, py)
        amixes_gpu = [('current', amix_gpu, 0.)] + amixes_gpu

        print('amixes_gpu', amixes_gpu)
        mog = amixes_gpu[0][1]
        print('mog shape', mog)
        print('var', mog.var.get())
        
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
