import time

import numpy as np
from cupy_wrapper import cp
#import cupy as cp

from tractor.factored_optimizer import FactoredDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, ConstantSky

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
        # Assume ConstantSky models, grab constant sky levels
        # NOTE - instead of building this list and passing it around in ImageDerivs, etc,
        # we could perhaps just subtract it off img_pix at the start...
        img_sky = [tim.getSky().getConstant() for tim in tr.images]
        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskByIdx(i, src) for i in range(len(tr.images))]
        #masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        if any(m is None for m in masks):
            raise RuntimeError('One or more modelMasks is None in GPU code')

        assert(all([m is not None for m in masks]))
        assert(src.isParamThawed('pos'))

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        # WCS inv(CD) matrix
        img_cdi = [tim.getWcs().cdInverseAtPosition(src.getPosition(), src=src)
                   for tim in tr.images]
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]
        bands = src.getBrightness().getParamNames()

        img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]

        #img_params, cx,cy,pW,pH = self._getBatchImageParams(tr, masks, pxy)
        #def _getBatchImageParams(self, tr, masks, pxy):
        extents = [mm.extent for mm in masks]

        px, py = np.array(pxy, dtype=np.float32).T
        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        x0, x1, y0, y1 = np.asarray(extents).T
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                                1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                                psfH//2, psfW//2]), axis=0)
        # PSF Fourier transforms
        batch_psf = BatchPixelizedPSF(psfs)
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(px, py, gpu_halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (Nimages,len(w),len(v)))

        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)
        #return img_params, cx,cy, pH,pW

        amix_gpu = getShearedProfileGPU(src, tr.images, px, py)
        amixes_gpu = getDerivativeShearedProfilesGPU(src, tr.images, px, py)
        amixes_gpu = [('current', amix_gpu, 0.)] + amixes_gpu

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

