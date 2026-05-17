import sys
from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, PointSource
from tractor import mixture_profiles as mp
from tractor.psf import lanczos_shift_image
from tractor.sky import ConstantSky
from astrometry.util.miscutils import get_overlapping_region
import numpy as np
import scipy
import scipy.fft
import time
from tractor.batch_psf import BatchPixelizedPSF, lanczos_shift_image_batch_gpu
from tractor.batch_mixture_profiles import ImageDerivs, BatchImageParams, BatchMixtureOfGaussians, BatchGalaxyProfiles
from tractor.batch_galaxy import getShearedProfileGPU, getDerivativeShearedProfilesGPU
from tractor.patch import ModelMask
import cupy as cp
import os, gc
from tractor.miscutils import get_mog_eval_kernel
from tractor.brightness import NanoMaggies

tt = np.zeros(8)
tct = np.zeros(9, np.int32)

#from astrometry.util.plotutils import PlotSequence
#ps = PlotSequence('fourier')

class GPUFriendlyPointSourceFitter:
    def __init__(self, tractor, source, images):
        self.tractor = tractor
        self.source = source
        self.images = images
        self.cpucomp = None
        self.batch_psf = None
        self.tiny_alpha = 1e-8
        self.stepLimited = False
        self.current_bounds = None
        self.ie_batch = None

    def compute_update_vec(self, allderivs):
        """
        Calculates the total Hessian (H) and Gradient (G) for the given derivatives
        across all images. Returns (H, G) as CuPy arrays.
        
        allderivs: list of lists [ (param0) [ (patch, img), ... ], (param1) [...] ]
        """
        num_params = len(allderivs)
        #print (f'{allderivs=}')
        #print (f'GPU {num_params=}')
        if num_params == 0:
            return None, None, 0

        # 1. Pivot: Group patches by Image to minimize GPU overhead
        # Structure: img -> {global_param_idx: PatchObject}
        img_map = {}
        for p_idx, patches in enumerate(allderivs):
            for patch, img in patches:
                if patch is None or patch.patch is None:
                    print (f'{p_idx=} patch is none')
                    continue
                if img not in img_map:
                    img_map[img] = {}
                img_map[img][p_idx] = patch

        # 2. CREATE THE ORDERED LIST OF ACTIVE IMAGES
        # This ensures the i-th slice of our batch matches the i-th image in our loop
        active_tims = list(img_map.keys())

        # 3. Calculate the 'true' global union from patch extents
        # This replaces the logic that relies on the (sometimes too tight) modelMasks
        all_x0, all_x1, all_y0, all_y1 = [], [], [], []
        for img, params in img_map.items():
            for p_idx, patch in params.items():
                px0, px1, py0, py1 = patch.extent
                all_x0.append(px0); all_x1.append(px1)
                all_y0.append(py0); all_y1.append(py1)

        if len(all_x0) == 0:
            return None, None, 0
        ux0, ux1 = min(all_x0), max(all_x1)
        uy0, uy1 = min(all_y0), max(all_y1)
        ux0 = int(ux0)
        uy0 = int(uy0)
        ux1 = int(ux1)
        uy1 = int(uy1)
        #uH, uW = uy1 - uy0, ux1 - ux0
        u_height, u_width = uy1 - uy0, ux1 - ux0
        total_pixels = u_height * u_width
        num_imgs = len(img_map)
        num_params = len(allderivs)

        # 4. Pre-allocate Tensors on GPU
        # J: (N_images, N_params, Total_Pixels)
        # Chi: (N_images, Total_Pixels)
        #J_tensor = cp.zeros((num_imgs, num_params, u_height, u_width), dtype=cp.float64)
        cJ_tensor = np.zeros((num_imgs, num_params, u_height, u_width), dtype=np.float32)
        Chi_tensor = cp.zeros((num_imgs, total_pixels), dtype=cp.float64)

        masks = [self.tractor._getModelMaskFor(tim, self.source) for tim in active_tims]
        ie_batch = self.tractor.getBatchInvErrors(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True, tims=active_tims)
        #Save params
        
        self.current_bounds = (ux0, ux1, uy0, uy1)
        if len(active_tims) == len(self.tractor.images):
            #If only a subset of images are "in frame" we can calculate this later
            self.ie_batch = ie_batch

        chi_batch = self.tractor.getBatchChiImages(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True, batch_psf=self.batch_psf, tims=active_tims, ie_stack=ie_batch)
        Chi_tensor = chi_batch.reshape((num_imgs, total_pixels))
        for i, (img, params) in enumerate(img_map.items()):
            # Fetch and crop to global bounds
            # (Ensure getChiImage/getInvError return CuPy arrays to avoid transfers)
            iex = ie_batch[i]
            for p_idx, patch in params.items():
                px0, px1, py0, py1 = patch.extent
                # Patch is already relative to image coordinates, 
                # but we need to place it relative to the Global Box
                dx0 = px0 - ux0
                dx1 = px1 - ux0
                dy0 = py0 - uy0
                dy1 = py1 - uy0
                
                # View into the specific image/parameter row as a 2D grid for insertion
                cJ_tensor[i, p_idx, dy0:dy1, dx0:dx1] = patch.getPatch()

        # 4. The "Two-Dot" Solution (The Big Win)
        # Reshape J to (N_params, N_images * Total_Pixels)
        # This effectively flattens the image and pixel dimensions into one "measurement" space
        cJ_tensor = cp.asarray(cJ_tensor)
        cJ_tensor *= ie_batch[:, cp.newaxis, :,:]
        cJ_flat = cJ_tensor.transpose(1, 0, 2, 3).reshape(num_params, -1)
        cJ_flat = cp.ascontiguousarray(cJ_flat) # Force BLAS-friendly layout
        Chi_flat = Chi_tensor.ravel()

        # Gradient: G = J_flat @ Chi_flat
        # Result: (N_params,)
        G_full = cJ_flat.dot(Chi_flat)

        # Hessian: H = J_flat @ J_flat^T
        # Result: (N_params, N_params)
        H_full = cJ_flat.dot(cJ_flat.T)

        return H_full, G_full, num_imgs


    def tryUpdates_gpu(self, X, alphas=None, check_step=None, use_less_mem=False):
        (ux0, ux1, uy0, uy1) = self.current_bounds
        if self.ie_batch is None:
            self.ie_batch = self.tractor.getBatchInvErrors(ux0=ux0, ux1=ux1, uy0=uy0, uy1=uy1, use_gpu=True)

        # 1. Prepare Alphas (inject 0.0 at the front)
        if alphas is None:
            alphas = np.append(0.0, np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.]))
        else:
            alphas = np.unique(np.append(0.0, alphas))

        # 2. Get Steps
        steps = self.getParameterSteps(X, alphas, firstAlphaIsZero=True)
        candidate_ps = [s['p'] for s in steps]

        # 3. The Big Batch Evaluation
        # We pass the cached bounds and IE stack to avoid re-calculation
        logprobs = self.tractor.getLogProbBatch(
            candidate_ps,
            bounds=self.current_bounds,
            ie_stack=self.ie_batch,
            use_gpu=True
        )

        # logprobs[0] is our baseline (alpha=0)
        # 3. Use the GPU's alpha=0 as the local baseline
        lps_cpu = cp.asnumpy(logprobs)
        lp_before = lps_cpu[0]

        lp_best = lp_before
        alpha_best = 0.0
        p_best = candidate_ps[0]
        hit_limit_best = False

        # 4. Standard Line Search logic (Identify best alpha)
        for i, step in enumerate(steps[1:], start=1):
            lp = lps_cpu[i]
            
            # Tractor exit conditions: significantly worse or non-finite
            if lp < (lp_best - 1.0) or not np.isfinite(lp):
                break
                
            if lp > lp_best:
                lp_best = lp
                alpha_best = step['alpha']
                p_best = step['p']
                hit_limit_best = step['hit_limit']

        # 5. Finalize State
        self.tractor.setParams(p_best)
        self.last_step_hit_limit = hit_limit_best
        #print ("Pbest", p_best)
        #print ("LOGPROB", lp_best, lp_before)
        #print ("ALPHA", alpha_best, hit_limit_best, self.last_step_hit_limit)
        
        return lp_best - lp_before, alpha_best


    def getParameterSteps(self, X, alphas=None, firstAlphaIsZero=False):
        if alphas is None:
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        p0 = self.tractor.getParams()
        max_step_size = self.tractor.getMaxStep()
        lowers = self.tractor.getLowerBounds()
        uppers = self.tractor.getUpperBounds()

        X_np = cp.asnumpy(X) # Ensure we work on CPU for the logic
        results = []

        for i, alpha in enumerate(alphas):
            do_break = False
            step_limit = False
            hit_limit = False

            # Apply max_step_size
            for i, mx in enumerate(max_step_size):
                if mx is None or abs(X_np[j]) == 0: continue
                step = alpha * abs(X_np[j])
                if step > mx:
                    step_limit = True
                    do_break = True
                    alpha = mx / abs(X_np[j])

            # Apply bounds and compute p
            p = []
            for p_start, s, l, u in zip(p0, X_np, lowers, uppers):
                px = p_start + alpha * s
                if l is not None and px < l:
                    px = l
                    hit_limit = True
                if u is not None and px > u:
                    px = u
                    hit_limit = True
                p.append(px)

            if alpha < self.tiny_alpha and alpha != 0:
                #if i > 0 or not firstAlphaIsZero or alpha != 0:
                break

            results.append({
                'alpha': alpha,
                'p': np.array(p),
                'step_limit': step_limit,
                'hit_limit': hit_limit
            })

            if do_break:
                break
        return results


'''
A mixin class for LsqrOptimizer that does the linear update direction step
by factorizing over images -- it solves the linear problem for each image
independently, and then combines those results (via their covariances) into
the overall result.
'''
class FactoredOptimizer(object):
    def __init__(self, *args, **kwargs):
        self.ps = None
        super().__init__(*args, **kwargs)

    def getSingleImageUpdateDirection(self, tr, max_size=0, **kwargs):
        allderivs = tr.getDerivs()
        r = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, max_size=max_size, **kwargs)
        if r is None:
            return None
        x,A,colscales,B,Ao = r
        #print ("X", x.shape, x)
        #print ("A", A.shape, A.max())
        #print ("B", B.shape, B.max())
        # print('SingeImageUpdateDirection: tr thawed params:')
        # tr.printThawedParams()
        # print('allderivs:', len(allderivs))

        if self.ps is not None:
            mod0 = tr.getModelImage(0)
            tim = tr.getImage(0)
            B = ((tim.getImage() - mod0) * tim.getInvError()).ravel()
            import pylab as plt
            plt.clf()
            ima = dict(interpolation='nearest', origin='lower')
            rr,cc = 3,4
            plt.subplot(rr,cc,1)
            plt.imshow(mod0, **ima)
            plt.title('mod0')
            sh = mod0.shape
            plt.subplot(rr,cc,2)
            mx = max(np.abs(B))
            imx = ima.copy()
            imx.update(vmin=-mx, vmax=+mx)
            plt.imshow(B.reshape(sh), **imx)
            plt.title('B')
            AX = np.dot(A, x)
            plt.subplot(rr,cc,3)
            plt.imshow(AX.reshape(sh), **imx)
            plt.title('A X')
            ah,aw = A.shape
            for i in range(min(aw, 8)):
                plt.subplot(rr,cc,5+i)
                plt.imshow(A[:,i].reshape(sh), **ima)
                if i == 0:
                    plt.title('dx')
                elif i == 1:
                    plt.title('dy')
                elif i == 2:
                    plt.title('dflux')
            self.ps.savefig()

        icov = np.matmul(A.T, A)
        del A
        return x, icov

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        from tractor import Images
        img_opts = []
        imgs = tr.images
        mm = tr.modelMasks

        #Remove 'priors' from kwargs
        orig_priors = kwargs.pop('priors', True)
        max_size = 0
        for i,img in enumerate(imgs):
            tr.images = Images(img)
            if mm is not None:
                tr.modelMasks = [mm[i]]
            #Run with PRIORS = FALSE
            r = self.getSingleImageUpdateDirection(tr, priors=False, max_size=max_size, **kwargs)
            if r is None:
                continue
            x,x_icov = r
            max_size = max(max_size, len(x))
            #print('FO: X', x, 'x_icov', x_icov)
            img_opts.append((x,x_icov))
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def getLinearUpdateDirection(self, tr, **kwargs):

        # We can fit for image-based parameters (eg, sky level) and source-based parameters.
        # If image parameters are being fit, use the base code (eg in lsqr_optimizer.py)
        # to fit those, and prepend them to the results below.
        t = time.time()
        tct[1] += 1
        #tr.images = tr.images[:5]
        #if tct[1] > 2:
        #    import sys
        #    sys.exit(0)
        x_imgs = None
        image_thawed = tr.isParamThawed('images')
        if image_thawed:
            cat_frozen = tr.isParamFrozen('catalog')
            if not cat_frozen:
                tr.freezeParam('catalog')
            x_imgs = super().getLinearUpdateDirection(tr, **kwargs)
            if not cat_frozen:
                tr.thawParam('catalog')
            else:
                return x_imgs

        if image_thawed:
            tr.freezeParam('images')
        #print('getLinearUpdateDirection( kwargs=', kwargs, ')')
        #print ("Running Factored getSingleImageUpdateDirections")
        if len(tr.images) == 0:
            if x_imgs is not None:
                return x_imgs
            return None
        #Store original value of priors
        orig_priors = kwargs['priors']
        img_opts = self.getSingleImageUpdateDirections(tr, **kwargs)
        tt[5] += time.time()-t
        t = time.time()
        if len(img_opts) == 0:
            if x_imgs is not None:
                return x_imgs
            return None
        # ~ inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        if len(img_opts) == 1 and len(img_opts[0]) == 3:
            xicsum, icsum, _ = img_opts[0]
        else:
            for x,ic in img_opts:
                xicsum = xicsum + np.dot(ic, x)
                icsum = icsum + ic
        #C = np.linalg.inv(icsum)
        #x = np.dot(C, xicsum)
        if np.any(np.isnan(xicsum)):
            print (f"WARNING: np.dot failed with NAN {xicsum=}.  Running CPU version instead.")
            x = super().getLinearUpdateDirection(tr, **kwargs) 
            return x 

        #Add the priors if needed.
        if orig_priors:
            priors_ATA, priors_ATB = self.getPriorsHessianAndGradient(tr)
            if priors_ATA.shape == icsum.shape:
                icsum += priors_ATA
                xicsum += priors_ATB
            elif np.all(priors_ATA == 0) and np.all(priors_ATB == 0):
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=} but priors are zero so ignorning.")
            else:
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=}; using CPU mode instead.")
                return super().getLinearUpdateDirection(tr, **kwargs)
        try:
            x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
        except Exception as ex:
            print ("WARNING: lstsq failed "+str(ex)+"; using CPU version")
            return super().getLinearUpdateDirection(tr, **kwargs)
        tt[6] += time.time()-t
        #print ("Final X: ",x, "Source", tr.catalog[0])
        if x_imgs is not None:
            x = np.append(x_imgs, x)
        if image_thawed:
            tr.thawParam('images')
        return x

from tractor.smarter_dense_optimizer import SmarterDenseOptimizer

#class FactoredDenseOptimizer(FactoredOptimizer, ConstrainedDenseOptimizer):
class FactoredDenseOptimizer(FactoredOptimizer, SmarterDenseOptimizer):
    pass


class GPUFriendlyOptimizer(FactoredDenseOptimizer):
    _gpumode = 3

    def setGPUMode(self, gpumode):
        print ("Setting GPUMODE = ",gpumode)
        #0 = run CPU only
        #1 = run GPU only
        #2 = run VECTORIZED only
        #3 = run all
        #10 = run CPU only - Profile only
        #11 = run GPU only - Profile only
        #12 = run VECTORIZED only - Profile only
        #13 = run all - Profile only
        self._gpumode = gpumode

    def printTiming(self):
        print ("Times:",tt,tct)

    def printMemory(self, tag=""):
        import cupy as cp
        import datetime
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        mempool = cp.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        tot_bytes = mempool.total_bytes()
        print (f'{tag=} {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=} at ',datetime.datetime.now())
        if free_mem/1.e+9 < 20:
            import gc
            print ("Freeing memory")
            gc.collect()
            print (f'{tag=} {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=} at ',datetime.datetime.now())
            print ("GPU free all blocks")
            mpool = cp.get_default_memory_pool()
            mpool.free_all_blocks()
            print (f'{tag=} {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=} at ',datetime.datetime.now())
        sys.stdout.flush()

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        #import traceback
        #traceback.print_stack()
        print ("GPU getSingleImageUpdateDirections")
        print ("profile galaxy", isinstance(tr.catalog[0], ProfileGalaxy))
        print ("len", len(tr.catalog), "Nimages", len(tr.images))
        R_gpu = None
        R_cpu = None
        R_gpuv = None
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            if self._gpumode == 2 and isinstance(tr.catalog[0], PointSource):
                tct[5] += 1
                t = time.time()
                # Initialize the GPU Fitter
                print ("Running new GPU point source fitter...")
                # Initialize and CACHE the fitter on self
                self.current_fitter = GPUFriendlyPointSourceFitter(tr, tr.catalog[0], tr.images)
                allderivs = tr.getDerivs()

                # Get the update
                # Solve A^T A x = A^T b
                hess, grad, ni = self.current_fitter.compute_update_vec(allderivs) # Updated to use internal G
                if ni == 0 or np.all(grad == 0):
                    print ("Point source fitter returned EMPTY SET")
                    return []

                tt[0] += time.time()-t
                tct[0] += 1
                return [(grad.get(), hess.get(), 1)]

            print ("Running CPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
            t = time.time()
            #print ("SRC", type(tr.catalog[0]))
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            tt[0] += time.time()-t
            tct[0] += 1
            return R

        if self._gpumode > 0:
            import cupy as cp
            import datetime
            #Get free memory
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            #Estimate memory needed with new helper method based on padded image sizes and nd
            nimages = len(tr.images)
            imsize = tr.images[0].data.size
            nd = tr.numberOfParams()+2
            src = tr.catalog[0]
            masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
            if any(m is None for m in masks):
                print (f"WARNING: One or more modelMasks is None; running CPU version.")
                xout = super().getSingleImageUpdateDirections(tr, **kwargs)
                return xout

            assert(all([m is not None for m in masks]))
            # Pixel positions
            pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
                   for tim in tr.images]
            px, py = np.array(pxy).T
            if np.any(np.isnan(px)) or np.any(np.isnan(py)):
                print ("WARNING: px, py contain NANs.  Running CPU version instead.")
                xout = super().getSingleImageUpdateDirections(tr, **kwargs)
                return xout
            #New helper method predicts memory needed
            est_mem = self.predict_fft_memory(tr, masks, pxy, nd)
            free_mem /= 1.e+9
            # 3.2 factor: NGC 3585 example
            #print (f'Estimated memory {est_mem} GB free memory {free_mem} for {nimages=} {imsize=} {nd=} at ', datetime.datetime.now(), "src = ", tr.catalog[0])

            if free_mem < est_mem:
                try:
                    import datetime
                    print(f"Warning: Estimated memory {est_mem} GB is greater than free memory {free_mem} GB; Running less-memory GPU mode instead!", datetime.datetime.now(), "src = ", tr.catalog[0])
                    #print("Warning: Estimated memory %.1f GB is greater than free memory %.1f GB; Running less-memory GPU mode instead!" % (est_mem / 1e9, free_mem / 1e9))
                    t = time.time()
                    R_gpu = self.gpuSingleImageUpdateDirectionsVectorized_less_mem(tr, **kwargs)
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    tt[4] += time.time()-t
                    return R_gpu
                except Exception as e:
                    print('Fallback to less-memory GPU version failed:', e)
                    import traceback
                    traceback.print_exc()
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()

                print('Running CPU version instead')
                R_cpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                return R_cpu

            try:
                #import datetime
                #print ('Running VECTORIZED GPU code...', datetime.datetime.now())
                t = time.time()
                R_gpuv = self.gpuSingleImageUpdateDirectionsVectorized(tr, **kwargs)
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                #print ("Finished VECTORIZED code")
                tt[2] += time.time()-t
                tct[2] += 1
                #print ("GPU Vectorized time:", time.time()-t)
                if self._gpumode == 2 or self._gpumode == 12 or self._gpumode == 4:
                    return R_gpuv
            except Exception as e:
                #New consolidated except
                import cupy as cp
                # 1. Capture the error string
                err_msg = str(e).lower()
                is_oom = "outofmemory" in err_msg or "out of memory" in err_msg

                # 2. Print immediately to stderr (bypasses buffering)
                import sys
                sys.stderr.write(f"\nCaught Exception in GPU code: {type(e).__name__}\n")
                if is_oom:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    mempool = cp.get_default_memory_pool()
                    used_bytes = mempool.used_bytes()
                    tot_bytes = mempool.total_bytes()
                    sys.stderr.write(f"Confirmed OOM for source: {tr.catalog[0]}\n")
                    print (f'OOM Device {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=}')

                sys.stderr.write(f"\n----Traceback below----\n")
                import traceback
                traceback.print_exc()
                sys.stderr.flush()

                # 3. Emergency Memory Release
                try:
                    import cupy as cp
                    # Force drop image caches if they exist
                    for tim in tr.images:
                        if hasattr(tim, 'data_gpu'): tim.data_gpu = None
                        if hasattr(tim, 'inverr_gpu'): tim.inverr_gpu = None
                    cp.get_default_memory_pool().free_all_blocks()
                except:
                    pass

                # 4. Final Fallback
                print("Running CPU version instead... for source: "+str(tr.catalog[0]))
                sys.stdout.flush()
                t = time.time()
                R_gpuv = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 2 or self._gpumode == 12 or self._gpumode == 4:
                    return R_gpuv

        if self._gpumode == 0:
            try:
                print('Running CPU code for comparison...')
                if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
                    print ("Running CPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
                    tct[0] += 1
                    #print ("TCT0", tct[0])
                t = time.time()
                R_cpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                tt[3] += time.time()-t
                tct[3] += 1
                #print ("TCT3", tct[3])
                #print ("CPU time", time.time()-t)
                if self._gpumode == 0 or self._gpumode == 10:
                    return R_cpu
            except Exception as ex:
                import traceback
                print('Exception in CPU code:')
                traceback.print_exc()

                src = tr.catalog[0]
                print('Source:', src)
                print(repr(src))
                raise(ex)
                #sys.exit(-1)
        return R_gpuv

    def gpuSingleImageUpdateDirectionsVectorized_less_mem(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            p = self.ps
            self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            self.ps = p
            return R
        if len(tr.images) == 0:
            print (f"WARNING: len images == 0, running CPU version")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout
        tims = tr.images
        modelmasks = tr.modelMasks
        try:
            xx = []
            for i,(tim,mm) in enumerate(zip(tims, modelmasks)):
                tr.images = [tim]
                tr.modelMasks = [mm]
                src = tr.catalog[0]
                sx,sy = tim.getWcs().positionToPixel(src.getPosition(), src)
                h,w = tim.shape
                try:
                    x = self.gpuSingleImageUpdateDirectionsVectorized(tr, **kwargs)
                except Exception as e:
                    x = super().getSingleImageUpdateDirections(tr, **kwargs)
                xx.extend(x)
        finally:
            tr.images = tims
            tr.modelMasks = modelmasks
        return xx

    def gpuSingleImageUpdateDirectionsVectorized(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            p = self.ps
            self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            self.ps = p
            return R
        if len(tr.images) == 0: 
            print (f"WARNING: len images == 0, running CPU version")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout 

        #print('Using GpuFriendly vectorized code')
        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        #assert(all([tim.sky.getConstant() == 0 for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
        t1 = time.time()
        Nimages = len(tr.images)

        #img_pix = [tim.data for tim in tr.images]
        img_pix = [tim.getImage(use_gpu=True) for tim in tr.images]
        img_ie  = [tim.getInvError(use_gpu=True) for tim in tr.images]
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
            print (f"WARNING: One or more modelMasks is None; running CPU version.")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout
        assert(all([m is not None for m in masks]))

        assert(src.isParamThawed('pos'))

        # FIXME -- must handle priors (ellipticity)!!

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
        #print('Bands:', bands)
        try:
            img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]
        except ValueError as ex:
            print (f'WARNING: Image bands error - {bands=} - '+str(ex))
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout 
        #print('ibands', img_bands)

        #print ("Calling VECTORIZED version")
        img_params, cx,cy,pW,pH = self._getBatchImageParams(tr, masks, pxy)
        # Dustin FIXME - cx,cy,pW,pH should probably be in img_params, they're about the PSF sizes
        # and centering.
        
        px, py = np.array(pxy).T
        px = px.astype(np.float32)
        py = py.astype(np.float32)
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

        #Comment out priorVals
        priorVals = None

        if img_params.ffts is None:
            print ("Warning> img_params.ffts is None!  Running CPU version")
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            return xout

        Xic = self.computeUpdateDirectionsVectorized(img_params, priorVals)

        mpool = cp.get_default_memory_pool()
        del img_params
        del amixes_gpu
        mpool.free_all_blocks()
        #print (f'{nbands=}')

        if nbands >= 1:
            full_xic = []
            fullN = tr.numberOfParams()
            # number of source position parameters
            # - these would be RA and Dec, except for when we're fitting Gaia stars,
            #   whose positions are assumed correct and not re-fit!
            npos = src.getPosition().numberOfParams()
            assert(npos in [0,2])
            # The GPU-based fitting was done assuming 2 positions being fit
            # (the coefficients are zeroed out, but the resulting arrays "x" below are still sized to
            #  hold this many positional parameters)
            npos_fit = 2
            # fitting was done on a single band
            nbands_fit = 1
            (X, Xicov) = Xic
            X = X.get()
            Xicov = Xicov.get()
            print (f"{fullN=} {npos=} {nbands=}")
            for iband,x,ic in zip(img_bands, X, Xicov):
                #assert(fullN == len(x) + nbands - 1)
                # the "x" here are from fitting on a single image, *assuming* 2 positions are being fit
                assert(fullN == len(x) + nbands - nbands_fit + npos - npos_fit)
                x2 = np.zeros(fullN, np.float32)
                ic2 = np.zeros((fullN,fullN), np.float32)
                # source params are ordered: position, brightness, others
                #npos = 2
                #nothers = len(x)-3
                #print (f'{npos_fit=} {nbands_fit=} {npos=} {nbands=} {fullN=}')
                nothers = len(x) - npos_fit - nbands_fit

                # Where aa is a block of npos elements
                #       cc is a block of nothers elements
                #       b is a single element
                #       z1,z2 are some number of zeros
                #
                # x = [ aa b cc ] ---> x2 = [ aa z1 b z2 cc ]

                # ic = [ A B C ]      ic2 = [ A z B z C ]
                #      [ B D E ]  -->       [ z z z z z ]
                #      [ C E F ]            [ B z D z E ]
                #                           [ z z z z z ]
                #                           [ C z E z F ]
                # (note, sometimes npos=0, so then we have x2[:0] = x[:0], ie a no-op, but that's okay)

                # aa
                x2[:npos] = x[:npos]
                # b
                if npos == 0:
                    x2[npos + iband] = x[npos_fit]
                else:
                    x2[npos + iband] = x[npos]
                # cc
                x2[-nothers:] = x[-nothers:]

                # A
                ic2[:npos,:npos] = ic[:npos,:npos]
                # B
                ic2[npos + iband, :npos] = ic[npos, :npos]
                ic2[:npos, npos + iband] = ic[:npos, npos]
                # C
                ic2[:npos, -nothers:] = ic[:npos,    -nothers:]
                ic2[-nothers:, :npos] = ic[-nothers:, :npos]
                # D
                if npos == 0:
                    ic2[npos + iband, npos + iband] = ic[npos_fit, npos_fit]
                else:
                    ic2[npos + iband, npos + iband] = ic[npos, npos]
                # E
                if npos == 0:
                    ic2[npos + iband, -nothers:] = ic[npos_fit, -nothers:]
                    ic2[-nothers:, npos + iband] = ic[-nothers:, npos_fit]
                else:
                    ic2[npos + iband, -nothers:] = ic[npos, -nothers:]
                    ic2[-nothers:, npos + iband] = ic[-nothers:, npos]
                # F
                ic2[-nothers:,-nothers:] = ic[-nothers:,-nothers:]

                #full_xic.append((x2.get(),ic2.get()))
                full_xic.append((x2, ic2))
            Xic = full_xic
        else:
            (X, Xicov) = Xic
            full_xic = []
            X = X.get()
            Xicov = Xicov.get()
            for iband,x,ic in zip(img_bands, X, Xicov):
                full_xic.append((x, ic))
            Xic = full_xic

        #print (f"Final {Xic=}")
        return Xic

    def computeUpdateDirectionsVectorized(self, img_params, priorVals):
        '''
        This is the function you'll want to GPU-ify!

        "imgs" is a list with one element per image to process.  These can all be performed in parallel.
        A typical length for this would be ~ 10.

        The return value is a list of (vector, matrix) pairs, one per
        image.  The vector,matrix sizes are roughly (6, 6x6), giving
        the parameter update direction and its inverse-covariance.

        Each element in "imgs" is a giant tuple:

        (img_derivs, pix, ie, P, mux, muy, mw, mh, counts, cdi, roi)

        - img_derivs is a list (typical length ~ 4) of tuples,
        describing the parameters that we want to adjust in this
        parameter update.  (Plus two spatial ones that we handle specially.)
        Each tuple has (name, stepsize, mogs, fftmix),
        - name (string) is just the name of the parameter
        - stepsize (float) is the finite-difference step size we took in this parameter;
          it is used to scale the derivative
        - mogs will be None in this setting
        - fftmix is a MixtureOfGaussians objects (see
          mixture_profiles.py); in this setting, the "mean" values are
          zero, so only the "amp" and "var" (variance) fields are
          relevant.  The elements in this list correspond to the shape
          of the galaxy at the current parameters, and after stepping in
          each of a few different galaxy shape parameters.  We evaluate
          the Fourier transform of this weighted sum of Gaussians,
          directly in Fourier space; see mp_fourier.i :
          gaussian_fourier_transform_zero_mean_2 for the implementation.

        - pix is a (float32) matrix of pixels with size = (mh,mw), typically 64x64.

        - ie is a (float32) matrix of inverse-uncertainty ("inverse-error") pixels,
        same size as "pix".  Some of these will be zero, corresponding to zero-weight pixels.

        - P is the Fourier transform of the point-spread function (how the atmosphere blurs
        a point source in this image), of shape typically 64x33, and type numpy.complex64
        (ie, two float32 values for real and imaginary)

        - mux, muy are floating-point values giving the sub-pixel shift to apply to this
        galaxy.  They will be in the range [-0.5, +0.5].

        - mw, mh are the integer size of the images

        - counts is the floating-point brightness of the galaxy model.

        - cdi is a 2x2 matrix giving the transformation between sky coordinates (RA,Dec) and
        pixel coordinates.  We use this to produce the parameter update in sky coordinates,
        but instead of computing the derivatives by finite differences in those parameters,
        we compute them directly from the model image.

        - roi ("region of interest") is a tuple of 4 integers, (x0, y0, w, h), of the
        sub-region within "pix" and "ie" that contain non-zero values.
        I padded "pix" and "ie" out to be the same size as the galaxy
        model (ie, size "w x w", = the size of the ifft(P)) to make
        life easier in GPU land.  The code below shows how you might use
        the ROI information to process fewer pixels, but at the expense of
        shuffling them around a bit.

        '''
        use_roi = False
        Xic = []

        Npriors = 0
        if priorVals is not None:
            rA, cA, vA, pb, mub = priorVals
            Npriors = max(Npriors, max([1+max(r) for r in rA]))

        G = self.computeGalaxyModelsVectorized(img_params)

        Npix = img_params.mh*img_params.mw
        #Npix is a scalar
        Nd = img_params.maxNd
        Nimages = img_params.Nimages
        cdi = img_params.cdi
        A = cp.zeros((img_params.Nimages, Npix + Npriors, Nd+2), cp.float32)
        # A is of shape (Nimages, Npix+Npriors, Nd+2)
        # The first element in img_derivs is the current galaxy model parameters.
        mod0 = G[:,0,:,:]
        # mod0 should be (Nimages, nw, nv)
        assert (mod0.shape == (img_params.Nimages, img_params.mh, img_params.mw))
        ## 02-13-25 - make a mask with only the ROI as ones to mask out background
        #print (img_params.roi.shape)
        roi = img_params.roi
        roimask = cp.zeros((img_params.Nimages, mod0.shape[1], mod0.shape[2]), dtype=bool)
        for i, ri in enumerate(roi):
            roimask[i, ri[1]:ri[1]+ri[3], ri[0]:ri[0]+ri[2]] = True

        # boolean vector, are we fitting for positions?
        fit_pos = img_params.fit_pos
        # Shift this initial model image to get X,Y pixel derivatives
        ###  TAG: ISSUE 1 4/9/25 - comment / uncomment dx and dy to toggle fit_pos 
        dx = cp.zeros_like(mod0)
        # dx is of shape (Nimages, nw, nv)
        # X derivative -- difference between shifted-left and shifted-right arrays
        dx[fit_pos,:,1:-1] = mod0[fit_pos,:, 2:] - mod0[fit_pos,:, :-2]
        #dx[:,:,1:-1] = mod0[:,:, 2:] - mod0[:,:, :-2]
        # Y derivative -- difference between shifted-down and shifted-up arrays
        dy = cp.zeros_like(mod0)
        # dy is of shape (Nimages, nw, nv)
        #dy[:,1:-1, :] = mod0[:,2:, :] - mod0[:,:-2, :]
        dy[fit_pos,1:-1, :] = mod0[fit_pos,2:, :] - mod0[fit_pos,:-2, :]
        # Push through local WCS transformation to get to RA,Dec param derivatives
        assert(cdi.shape == (img_params.Nimages,2,2))

        ##Multiply everything by roimask 02-13-25
        dx *= roimask
        dy *= roimask
        mod0 *= roimask
        #Need to reshape for G
        G *= roimask[:,None,:,:]

        # divide by 2 because we did +- 1 pixel
        # negative because we shifted the *image*, which is opposite
        # from shifting the *model*
        A[:,:Npix, 0] = cp.reshape(-((dx * cdi[:,0, 0][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 0][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2),(img_params.Nimages, -1))
        A[:,:Npix, 1] = cp.reshape(-((dx * cdi[:,0, 1][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 1][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2), (img_params.Nimages, -1))
        del dx,dy
        A[:,:Npix,2] = cp.reshape(mod0,(img_params.Nimages, -1))
        stepsizes = img_params.steps
        A[:,:Npix, 3:] = img_params.counts[:,cp.newaxis, cp.newaxis] / stepsizes[:,cp.newaxis,1:] * cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1))
        A[:,:Npix,:] *= img_params.ie.reshape((img_params.Nimages, Npix))[:,:,cp.newaxis]
        del G

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))
        del mod0, img_params

        # Append priors --do priors depend on which image I am looking at?
        #TODO not sure if this is correct for priors? 

        ###Priors should NOT be added in here - this will be removed

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)

        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        colscales[colscales == 0] = 1.
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        A /= colscales[:,cp.newaxis, :]

        # Solve the least-squares problem!

        ## TAG: indidious hang issue 10/17/25 - pinv can occasionally hang.
        # But we had the issue where if fit_pos is False for all images, and
        # first two columns of A are all zero, the ATA ATB method fails with
        # NaNs.  But we can strip out those rows and zero-pad the output!
        if not np.any(fit_pos):
            print ("WARNING: fit_pos is False for all images, stripping off rows to prevent NaNs before fitting")
            A_T_dot_A = cp.einsum("...ji,...jk", A[:,:,2:], A[:,:,2:])
            A_T_dot_B = cp.einsum("...ji,...j", A[:,:,2:], B)
            X = cp.zeros((Nimages, Nd+2), dtype=cp.float32)
            X[:,2:] = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        else:
            A_T_dot_A = cp.einsum("...ji,...jk", A, A)
            A_T_dot_B = cp.einsum("...ji,...j", A, B)
            X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)

        # Undo pre-scaling
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        X /= colscales
        #print ("NANs", cp.isnan(X).sum())
        X[cp.isnan(X)] = 0
        X[cp.isinf(X)] = 0
        #Have to corectly make Xic a list of tuples
        #for i in range(img_params.Nimages):
        #    Xic.append((X[i], Xicov[i]))
        ##Change return args to keep as cupy arrays
        Xic = (X, Xicov)
        del A, B, X
        return Xic

    def evaluate_mog_gpu(self, conv_mog, xx, yy, img_params, G):
        import cupy as cp
        from tractor.miscutils import get_mog_eval_kernel
        
        N, Nd, K = conv_mog.amp.shape
        H, W = yy.size, xx.size
        
        # Heuristic for A100 40GB:
        # 208 images * 10 derivs * 6 components * 64 * 64 is small,
        # but we chunk anyway for scalability.
        from tractor.miscutils import get_safe_chunk_size
        chunk_size = get_safe_chunk_size(N, H, W, vram_gb=40)
        
        kernel = get_mog_eval_kernel()
        
        # Pre-calculate inversion terms
        det = conv_mog.var[:,:,:,0,0] * conv_mog.var[:,:,:,1,1] - \
              conv_mog.var[:,:,:,0,1] * conv_mog.var[:,:,:,1,0]
        
        # Broadcastable components
        iv0 = (conv_mog.var[:,:,:,1,1] / det).astype(cp.float32)[:,:,:,None,None]
        iv1 = (-(conv_mog.var[:,:,:,0,1] + conv_mog.var[:,:,:,1,0]) / det).astype(cp.float32)[:,:,:,None,None]
        iv2 = (conv_mog.var[:,:,:,0,0] / det).astype(cp.float32)[:,:,:,None,None]
        scale = (conv_mog.amp / (2.*cp.pi*cp.sqrt(det))).astype(cp.float32)[:,:,:,None,None]
        
        mx = (conv_mog.mean[:,:,:,0] - img_params.dx[:,None,None]).astype(cp.float32)[:,:,:,None,None]
        my = (conv_mog.mean[:,:,:,1] - img_params.dy[:,None,None]).astype(cp.float32)[:,:,:,None,None]

        # Grid (1, 1, 1, H, W)
        XX = xx[None, None, None, None, :]
        YY = yy[None, None, None, :, None]

        for i in range(0, N, chunk_size):
            e = min(i + chunk_size, N)
            # kernel computes (chunk, Nd, K, H, W)
            # we sum over K (axis 2) immediately to keep it 4D (chunk, Nd, H, W)
            res = kernel(iv0[i:e], iv1[i:e], iv2[i:e], scale[i:e], mx[i:e], my[i:e], XX.astype(cp.float32), YY.astype(cp.float32))
            G[i:e] += cp.sum(res, axis=2)
            
            del res
            cp.get_default_memory_pool().free_all_blocks()
        return G

    def predict_fft_memory(self, tr, masks, pxy, max_nd):
        # 1. Replicate the gpu_halfsize logic
        # (Simplified version of your _getBatchImageParams logic)
        px, py = np.array(pxy).T
        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T
        x0 = x0.astype(np.int32) 
        x1 = x1.astype(np.int32) 
        y0 = y0.astype(np.int32) 
        y1 = y1.astype(np.int32)

        # This is the "radius" that determines pH/pW
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                                1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py]), axis=0)

        # 2. Calculate the Power-of-Two padding
        radius_max = gpu_halfsize.max()
        try:
            sz = 2**int(np.ceil(np.log2(radius_max * 2.)))
        except Exception as ex:
            print (f'predict_fft_memory> {gpu_halfsize=} {radius_max=} {x0=} {x1=} {y0=} {y1=} {px=} {py=} '+str(ex))
            return 1.e+12

        pH, pW = sz, sz
        n_images = len(tr.images)

        # 3. Calculate Actual Bytes
        # Complex64 = 8 bytes, Float32 = 4 bytes
        # Assume 8 arrays of complex64
        # Assume 3 x arrays needed
        est_mem = n_images * max_nd * pH * pW * 8. * 3 / (1024.**3)
        #print (f'{est_mem=} {n_images=} {max_nd=} {pH=} {pW=}')

        # Total predicted for the FFT block (including a safety factor for IRFFT workspace)
        #total_predicted_gb = (fsum_bytes + p_bytes + g_bytes * 2) / (1024**3)
        #return total_predicted_gb, pH, pW
        return est_mem

    def computeGalaxyModelsVectorized(self, img_params):
        # Note, this does *not* scale by *counts*, it produces unit-flux models
        #assert(img_params.mogs is None)
        assert(img_params.ffts is not None)
        assert(img_params.mux is not None)
        assert(img_params.muy is not None)
        v = img_params.v
        w = img_params.w
        #img_params.ffts is a BatchMixtureOfGaussians (see batch_mixture_profiles.py)
        Fsum = img_params.ffts.getFourierTransform(v, w, zero_mean=True)
        #Fsum shape (Nimages, maxNd, nw, nv)
        #print ("Fsum", Fsum.shape, img_params.Nimages, img_params.maxNd, w.shape, v.shape)
        # P is created in psf.getFourierTransform - this should be done in batch on GPU
        # resulting in P already being a CuPy array
        #P shape (Nimages, nw, nv)
        P = img_params.P[:,cp.newaxis,:,:]
        #print ("FSUM", Fsum.shape, "P", P.shape)
        # Multiply P into Fsum in-place to save one massive array allocation
        Fsum *= P
        del P # Free P immediately
        # Perform IRFFT on the modified Fsum
        G = cp.fft.irfft2(Fsum).astype(cp.float32)
        del Fsum
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #G should be (nimages, maxNd, nw, nv) and mux and muy should be 1d vectors
        #print ("G shape", G.shape, img_params.Nimages,img_params.maxNd,img_params.mh,img_params.mw)
        assert (G.shape == (img_params.Nimages, img_params.maxNd, img_params.mh, img_params.mw))
        
        if img_params.mogs is not None:
            psfmog = img_params.psf_mogs
            #print('Img_params.mogs:', img_params.mogs)
            #print('PSF mogs:', img_params.psf_mogs)
            # assert trivial mixture of Gaussians - single circular Gaussian
            # If we relax this, then convolution becomes much harder!
            assert(psfmog.K == 1)
            assert(np.all(psfmog.mean == 0.))
            assert(np.all(psfmog.amp == 1.))
            assert(np.all(psfmog.var[..., 0, 0,1] == 0))
            assert(np.all(psfmog.var[..., 0, 1,0] == 0))
            assert(np.all(psfmog.var[..., 0, 0,0] == psfmog.var[..., 0, 1,1]))

            # Trivial convolution
            mogs = img_params.mogs
            varcopy = mogs.var.copy()
            # varcopy shape: (13, 4, 1, 2, 2)
            # psfmog.var shape: (13, 1, 2, 2)
            #varcopy[..., 0, 0, 0] += psfmog.var[..., cp.newaxis, 0, 0, 0]
            #varcopy[..., 0, 1, 1] += psfmog.var[..., cp.newaxis, 0, 1, 1]
            varcopy[..., :, 0, 0] += psfmog.var[..., cp.newaxis, cp.newaxis, 0, 0, 0]
            varcopy[..., :, 1, 1] += psfmog.var[..., cp.newaxis, cp.newaxis, 0, 1, 1]

            #print('mogs.amp', mogs.amp.shape)
            #print('mogs.mean', mogs.mean.shape)
            #print('varcopy', varcopy.shape)

            conv_mog = BatchMixtureOfGaussians(mogs.amp, mogs.mean, varcopy, quick=True)
            #print('Convolved MoG:', conv_mog)

            ### Could also assert that the Gaussians are all concentric... means for all K are equal

            #print('var shape', conv_mog.var.shape)
            #print('mogs K', mogs.K)
            #print('maxNmogs', img_params.maxNmogs)
            #print(img_params.Nimages, img_params.maxNd, img_params.maxNmogs)

            # Evaluate MoG on pixel grid, add to G
            use_roi = True
            xx = cp.arange(img_params.mw)
            yy = cp.arange(img_params.mh)

            G = self.evaluate_mog_gpu(conv_mog, xx, yy, img_params, G)

        #Do no use roi since images are padded to be (mh, mw)
        use_roi = False
        return G

    def tryUpdates(self, tractor, X, alphas=None, check_step=None):
        """
        Attempts to find the optimal step size (alpha) along the update direction X
        to maximize the log probability. Leverages GPU for step calculation and
        likelihood evaluation.

        Parameters
        ----------
        tractor : object
            An instance of the Tractor class (or similar) with methods:
            - getLogProb(use_gpu=False, **kwargs): Returns scalar log probability.
              Must have 'use_gpu=True' option that uses getLogLikelihood with GPU.
            - getParams(): Returns current parameters as a CuPy array.
            - setParams(p): Sets parameters. Expects CuPy array, handles GPU->CPU transfer.
            - getLowerBounds(): Returns lower bounds as NumPy array.
            - getUpperBounds(): Returns upper bounds as NumPy array.
            - getMaxStep(): Returns max step sizes as NumPy array.
        X : cupy.ndarray
            The update direction (from getUpdateDirection, expected to be CuPy array).
        alphas : list or numpy.ndarray or None
            List of alpha values to test. If None, default values are used.

        Returns
        -------
        dlogprob : float
            The change in log probability from the initial state to the best found state.
        alphaBest : float
            The optimal alpha (step size) found.
        """

        t = time.time()
        # 1. Fallback if GPU is disabled
        if self._gpumode == 0:
            return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
        import cupy as cp
        import datetime
        #return super().tryUpdates(tractor, X, alphas=alphas)
        #print ("GPU FACTORED TRY UPDATES")

        # 2. Check if we can use the PointSource GPU path
        is_ptsrc = (len(tractor.catalog) == 1 and 
                    isinstance(tractor.catalog[0], PointSource))
        
        # 3. Check if we can use the Galaxy GPU path
        is_galaxy = (len(tractor.catalog) == 1 and 
                     isinstance(tractor.catalog[0], ProfileGalaxy))

        if tractor.isParamFrozen('images'):
            if is_ptsrc:
                # Use the new GPU Point Source Engine
                # Check if we have a cached fitter for this source
                if hasattr(self, 'current_fitter'):
                    # Pass the work to the engine
                    print ("Using new GPU PointSource Engine")

                    p0 = tractor.getParams()
                    #Estimate memory needed with new helper method based on padded image sizes and nd
                    src = tractor.catalog[0]

                    # 1. Get masks and find which images actually overlap
                    all_tims = tractor.images
                    all_masks = tractor.modelMasks

                    # Filter out the "None" masks
                    # Store the original full list of dicts to restore later
                    all_mask_dicts = tractor.modelMasks 
                    
                    valid_tims = []
                    valid_mask_objects = []
                    valid_mask_dicts = []

                    for tim in all_tims:
                        m = tractor._getModelMaskFor(tim, src)
                        if m is not None:
                            valid_tims.append(tim)
                            valid_mask_objects.append(m)
                            # Re-wrap the mask in a dict so tractor can still query it by source
                            valid_mask_dicts.append({src: m})

                    if len(valid_tims) == 0:
                        #print("Source has no overlap with any image; log-likelihood is 0.")
                        # Handle edge case: no overlap at all
                        return 0.0, 0.0 

                    # 2. Update context with the expected data structures
                    tractor.images = valid_tims
                    # This is now a list of dicts, keeping tractor happy
                    tractor.modelMasks = valid_mask_dicts
                    if len(valid_tims) != len(all_tims):
                        print (f'TryUpdates PS: Warning {len(valid_tims)=} {len(all_tims)=} - ONE OR MORE model masks are None.')
                    Nimages = len(valid_tims) # The new, filtered count
                    #Get free memory
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    nd = tractor.numberOfParams()+2

                    try:
                        # Estimate memory using ONLY the images we will actually render
                        # pixel position of the source in each image
                        xy_valid = [tim.getWcs().positionToPixel(src.pos) for tim in valid_tims]
                        px, py = np.array(xy_valid).T

                        est_mem = self.predict_fft_memory(tractor, valid_mask_objects, xy_valid, nd)
                        free_mem /= 1.e+9
                        #print (f'TryUpdates PS: Estimated memory {est_mem} GB free memory {free_mem} for {Nimages=} {nd=} at ', datetime.datetime.now(), "src = ", tractor.catalog[0])
                        use_less_mem = False
                        if free_mem < est_mem:
                            print(f"Warning: TryUpdates PS Estimated memory {est_mem} GB is greater than free memory {free_mem} GB; Running less-mem GPU mode instead!", datetime.datetime.now(), "src = ", tractor.catalog[0])
                            use_less_mem = True

                        self.current_fitter.ie_batch = None
                        best_dlnp, best_alpha = self.current_fitter.tryUpdates_gpu(X, alphas, check_step=check_step, use_less_mem=use_less_mem)
                    except Exception as ex:
                        print ("Exception in TRY UPDATES PS GPU - running CPU version instead: "+str(ex))
                        import sys
                        #sys.stderr.write(f"\n----Traceback below----\n")
                        #import traceback 
                        #traceback.print_exc()
                        #sys.stderr.flush()
                        tractor.images = all_tims
                        tractor.modelMasks = all_masks
                        tractor.setParams(p0)
                        s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
                        return s_d, s_a
                    finally:
                        # CRITICAL: Restore the original image list so Tractor doesn't
                        # "forget" the other images for the rest of the brick
                        tractor.images = all_tims
                        tractor.modelMasks = all_masks

                    # Clean up after the line search is done
                    del self.current_fitter 
                    return best_dlnp, best_alpha
                return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step) 
            
        if not (is_galaxy and tractor.isParamFrozen('images')):
            # 4. Fallback for everything else
            print('Calling superclass tryUpdates')
            return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)

        if not ((len(tractor.catalog) == 1) and
                isinstance(tractor.catalog[0], ProfileGalaxy) and
                tractor.isParamFrozen('images')):
            print('Calling superclass tryUpdates')
            return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
 
        # Dustin's FIXME improvement ideas
        # - pass in / cache initial logprob / model?
        # - pass in / cache BatchPixelizedPSF ??

        #s_d, s_a = super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
        #return s_d, s_a

        #print('number of images:', len(tractor.images))
        Nimages = len(tractor.images)
        src = tractor.catalog[0]
        steps = self.getParameterSteps(tractor, X, alphas)
        #print('number of steps (alphas) to try:', len(steps))
        #print ("Alphas", alphas, "STEPS", steps)
        if len(steps) == 0:
            self.last_step_hit_limit = True
            self.hit_limit = True
            return 0., 0.

        p0 = tractor.getParams()
        # Also add the current parameter values (zero step size) for a baseline comparison.
        # (it would be nice to be able to track this and pass it in rather than recomputing,
        # but with modelmasks etc I think it will be simpler to just recompute it here...)
        # steps: list of (alpha, params, step_lim, param_lim)
        steps.append((0., p0, False, False))

        # Filter out the "None" masks
        # Store the original full list of dicts to restore later
        all_tims = tractor.images
        all_masks = tractor.modelMasks
        
        valid_tims = [] 
        valid_mask_objects = []
        valid_mask_dicts = []

        for tim in all_tims:
            m = tractor._getModelMaskFor(tim, src)
            if m is not None:
                valid_tims.append(tim)
                valid_mask_objects.append(m)
                # Re-wrap the mask in a dict so tractor can still query it by source
                valid_mask_dicts.append({src: m})

        if len(valid_tims) == 0:
            print("Source has no overlap with any image; log-likelihood is 0.")
            # Handle edge case: no overlap at all
            return 0.0, 0.0

        # 2. Update context with the expected data structures
        tractor.images = valid_tims
        # This is now a list of dicts, keeping tractor happy
        tractor.modelMasks = valid_mask_dicts

        if len(valid_tims) != len(all_tims):
            print (f'TryUpdates Gal: Warning {len(valid_tims)=} {len(all_tims)=} - ONE OR MORE model masks are None.')
        Nimages = len(valid_tims)
        nd = tractor.numberOfParams() + 2

        try:
            # Position in valid images only
            xy = [tim.getWcs().positionToPixel(src.pos) for tim in valid_tims]
            px, py = np.array(xy).T
            if np.any(np.isnan(px)):
                xy2 = [tim.getWcs().positionToPixel(src.pos) for tim in all_tims]
                px2, py2 = np.array(xy2).T

            #Get free memory
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            #New helper method predicts memory needed
            est_mem = self.predict_fft_memory(tractor, valid_mask_objects, xy, nd)
            free_mem /= 1.e+9
            #print (f'TryUpdates: Estimated memory {est_mem} GB free memory {free_mem} for {Nimages=} {nd=} at ', datetime.datetime.now(), "src = ", tractor.catalog[0])
            use_less_mem = False
            if free_mem < est_mem:
                print(f"Warning: TryUpdates Estimated memory {est_mem} GB is greater than free memory {free_mem} GB; Running less-mem GPU mode instead!", datetime.datetime.now(), "src = ", tractor.catalog[0])
                use_less_mem = True
            img_sky = [tim.getSky().getConstant() for tim in tractor.images]
            img_pix = [tim.getImage(use_gpu=True) for tim in tractor.images]
            img_ie  = [tim.getInvError(use_gpu=True) for tim in tractor.images]
            profiles = []
            fluxes = []
            logpriors = []
            for _,p,_,_ in steps:
                #print ("P", p)
                tractor.setParams(p)
                # for _getBatchGalaxyProfiles we need tuples of (name, amix, step)...
                profiles.append(('', getShearedProfileGPU(src, tractor.images, px, py), 0.))
                # grab the brightnesses
                fluxes.append([tim.getPhotoCal().brightnessToCounts(src.brightness) for tim in tractor.images])
                # While we're stepping through the parameters, compute log-priors...
                lp = tractor.getLogPrior()
                logpriors.append(lp)

            # Unfortunately, Sersic galaxies can get a different number of Gaussian mixture components
            # as we step the Sersic index parameter, so in the "profiles" above, they can have different
            # sizes; _getBatchGalaxyProfiles requires them to be the same sizes - so check and pad if
            # necessary.
            ng = np.unique([pro.var.shape[1] for _,pro,_ in profiles])
            if len(ng) != 1:
                # pad
                padsize = max(ng)
                for _,pro,_ in profiles:
                    shape = pro.var.shape
                    # needs padding?
                    oldsize = shape[1]
                    if shape[1] == padsize:
                        continue
                    newshape = (shape[0], padsize, shape[2], shape[3])
                    # var: size (nimages, ng, 2, 2) - cupy
                    padded = cp.zeros(newshape)
                    padded[:,:oldsize,:,:] = pro.var
                    pro.var = padded
                    # amp: size (ng,) -- a numpy array (not cupy) for some reason
                    padded = np.zeros(padsize)
                    padded[:oldsize] = pro.amp
                    pro.amp = padded
                    # mean: size (ng,2) -- also a numpy array
                    shape = pro.mean.shape
                    newshape = (padsize, shape[1])
                    padded = np.zeros(newshape)
                    padded[:oldsize,:] = pro.mean
                    pro.mean = padded

            if use_less_mem:
                # Initialize the accumulator for all alpha steps
                # This stays on the GPU so we don't transfer back and forth in the loop
                total_chisq = cp.zeros(len(steps), dtype=cp.float32)

                # Pre-convert fluxes to GPU once (it's a small array)
                # Shape: (nsteps, Nimages)
                fluxes_gpu = cp.asarray(fluxes, dtype=cp.float32)

                tims = tractor.images
                modelmasks = valid_mask_objects 

                try:
                    for i in range(Nimages):
                        # 1. Extract single-image data
                        # We wrap these in lists so your existing Batch methods still work
                        s_masks = [modelmasks[i]]
                        s_xy = [xy[i]]
                        s_sky = [img_sky[i]]
                        s_pix = [img_pix[i]]
                        s_ie = [img_ie[i]]


                        # 1. Create single-image profile data
                        s_profiles = []
                        for name, amix, step in profiles:
                            #print (f'{amix.mean.shape=} {amix.amp.shape=} {amix.var.shape=}')
                            #print (f'{type(amix)=}')
                            s_var  = amix.var[i:i+1]  # (1, Nd, K, D, D)
                            # amix.amp and amix.mean are usually shared (ng, ...) 
                            # but if they were per-image, you'd slice them here too.
                            #s_amix = BatchMixtureOfGaussians(s_amp, s_mean, s_var)
                            s_amix = BatchMixtureOfGaussians(amix.amp, amix.mean, s_var, unbalanced=True)
                            s_profiles.append((name, s_amix, step))

                        tractor.images = [tims[i]]
                        tractor.modelMasks = [valid_mask_dicts[i]]
                        # 2. Setup single-image parameters and profiles
                        # This keeps pH/pW consistent with the single image's needs
                        img_params_i, cx_i, cy_i, pW_i, pH_i = self._getBatchImageParams(tractor, s_masks, s_xy)

                        # Slicing fluxes_gpu for just this image: fluxes_gpu[:, i]
                        gals_i = self._getBatchGalaxyProfiles(s_profiles, s_masks, px[i:i+1], py[i:i+1],
                                                              cx_i, cy_i, pW_i, pH_i,
                                                              fluxes_gpu[0, i:i+1], s_sky, s_pix, s_ie)

                        img_params_i.addBatchGalaxyProfiles(gals_i)

                        # 3. Render and Calculate Chi2 for this image
                        # G_i shape: (1, nsteps, pH_i, pW_i)
                        G_i = self.computeGalaxyModelsVectorized(img_params_i)

                        # Call a modified version of your chi2 function or inline it:
                        # We sum over the image axes (2 and 3) and the single image axis (0)
                        # to get a (nsteps,) result to add to total_chisq.
                        total_chisq += self.calculate_chi2_cupy(G_i, fluxes_gpu[:, [i]],
                                                                gals_i.mmpix, gals_i.mmie,
                                                                cp.asarray(s_sky))

                        # 4. CRITICAL: Clear the deck for the next image
                        del G_i, img_params_i, gals_i, s_pix, s_ie, s_amix, s_var
                        cp.get_default_memory_pool().free_all_blocks()

                finally:
                    # Always restore state even if the loop fails
                    tractor.images = tims
                    tractor.modelMasks = modelmasks
                # Move final result back to CPU for the logprob logic
                chisq = total_chisq
            else:
                img_params, cx,cy,pW,pH = self._getBatchImageParams(tractor, valid_mask_objects, xy)
                gals = self._getBatchGalaxyProfiles(profiles, valid_mask_objects, px, py, cx, cy, pW, pH,
                                                    fluxes[0], img_sky, img_pix, img_ie)
                img_params.addBatchGalaxyProfiles(gals)
                if img_params.ffts is None:
                    print ("Warning> img_params.ffts is None!  Calling superclass tryUpdates (on CPU)")
                    return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
                G = self.computeGalaxyModelsVectorized(img_params)
                assert(G.shape == (Nimages, len(steps), pH, pW))
                mmpix = gals.mmpix
                mmie = gals.mmie
                fluxes = cp.asarray(fluxes, dtype=cp.float32)
                sky = cp.asarray(img_sky, dtype=cp.float32)

                chisq = self.calculate_chi2_cupy(G, fluxes, mmpix, mmie, sky)
                del G, img_params
                mpool = cp.get_default_memory_pool()
                mpool.free_all_blocks()
                #print ("CHISQ", chisq.shape)

            #print ("PRIOR", logpriors)
            logprob = np.array(logpriors)-0.5*chisq.get()
        except Exception as ex:
            print(f"Exception in Galaxy TRY UPDATES GPU: {ex}. Falling back to CPU.")
            tractor.setParams(p0)
            # Restore full image list for CPU fallback
            tractor.images = all_tims
            tractor.modelMasks = all_masks
            return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)

        finally:
            # CRITICAL: Restore the full list
            tractor.images = all_tims
            tractor.modelMasks = all_masks

        #print ("LOGPROB", logprob.shape)

        # CRAIG -- at this point, we have rendered the galaxy models, in G.
        # We should have the (padded) image pixels and inverse-errors in 'mmpix' and 'mmie' on the GPU.
        # (ie, I think they're all the same shape and ready to go...)
        # At this point, we need to compute the log-probs for each step in `steps`.
        # The log-probs are equal to the logpriors plus the log-likelihoods, which are
        # equal to the -0.5 * the sum of chi-squared values: (data - model) * inverr.

        # Once the log-probs for each step have been computed, we want to select the best one,
        # set the tractor params to the corresponding params, and return the
        # (delta-logprob, alpha)
        # of the best one.  delta-logprob will be the log-prob relative to the last one
        # (corresponding to the "step" we added at p0).

        # copy-pasting some additional description from slack:

        # I don’t know exactly what shape G is at this point, but conceptually it is
        #   N_steps x N_images x postage-stamp-size
        # For each step, we want to compute
        #    chisq = sum([((G[i_step, i_image] * fluxes[i_step, i_image] - mm_pix[i_image]) * mm_ie[i_image])**2 for i_image in range(N_images)])
        # that is,
        #    chisq = sum over i_image:
        #                 of chi**2  (pixelwise)
        #                 where chi = (model - data) * inverr
        #                 and data = mm_pix[i_image,...], and inverr = mm_ie[i_image...]
        #                 and model = G * flux[i_step,i_image] + sky[i_image]
        # and then
        #    logprob = logprior[i_step] + -0.5*chisq

        max_idx = np.argmax(logprob)
        #print ("LOGPROB", logprob)
        #print ("MAX", max_idx, logprob[max_idx])
        #print ("Pbest", steps[max_idx][1])
        pbest = steps[max_idx][1]
        hit_limit = steps[max_idx][2]
        tractor.setParams(pbest)
        for i in range(len(steps)):
            if steps[i][2]:
                self.hit_limit = True #set if any hits limit

        tt[6] += time.time()-t
        if max_idx == len(logprob)-1:
            print ("Best is previous")
            return (0., 0.)

        lp_best = logprob[max_idx]
        lp_last = logprob[-1]
        alpha = steps[max_idx][0] #get alpha from steps not array of alphas!
        if lp_best > lp_last:
            # Best we've found so far -- accept this step!
            self.last_step_hit_limit = hit_limit
        return (lp_best-lp_last, alpha)

    def calculate_chi2_cupy(self, G, fluxes, mmpix, mmie, sky):
        """
        Calculates the chi-squared efficiently using CuPy.

        Args:
            G (cupy.ndarray): Shape (Nimages, nsteps, height, width).
            fluxes (cupy.ndarray): Shape (nsteps, Nimages).
            mmpix (cupy.ndarray): Shape (height, width).
            mmie (cupy.ndarray): Shape (height, width).
            sky (cupy.ndarray): Shape (Nimages,).

        Returns:
            cupy.ndarray: chi2 with shape (Nimages, nsteps).
        """
        Nimages, nsteps, height, width = G.shape

        # 1. Reshape sky and fluxes for broadcasting
        # sky needs to broadcast as (Nimages, 1, 1, 1)
        sky_reshaped = sky[:, cp.newaxis, cp.newaxis, cp.newaxis]

        # fluxes needs to broadcast as (1, nsteps, 1, 1) across height/width for G,
        # but the i_image index needs to align with G's first dimension.
        # It's better to think of fluxes as (nsteps, Nimages)
        # and then carefully expand its dimensions for the multiplication.

        # We need fluxes to be (Nimages, nsteps, 1, 1) for element-wise multiplication with G
        # Let's reorder fluxes to (Nimages, nsteps) for easier broadcasting with G
        # If fluxes is (nsteps, Nimages), we should transpose it for this logic
        # Or, we can reshape fluxes to (1, nsteps, 1, 1) and G to (Nimages, nsteps, height, width)
        # Let's adjust for the given fluxes shape (nsteps, Nimages)
        # Model = G * fluxes[i_step, i_image] + sky[i_image]

        # To align dimensions:
        # G: (Nimages, nsteps, height, width)
        # fluxes: (nsteps, Nimages) -> we want to multiply G[i,j,:,:] by fluxes[j,i]
        # sky: (Nimages,) -> we want to add sky[i]

        # The most straightforward way to handle fluxes given its shape (nsteps, Nimages)
        # is to transpose it and then expand dimensions for broadcasting with G.
        # We want fluxes_broadcastable[i_image, i_step, 1, 1]
        # So, fluxes.T will be (Nimages, nsteps).
        fluxes_broadcastable = fluxes.T[:, :, cp.newaxis, cp.newaxis] # Shape (Nimages, nsteps, 1, 1)

        # Calculate the model: G * flux + sky
        # G: (Nimages, nsteps, height, width)
        # fluxes_broadcastable: (Nimages, nsteps, 1, 1) - broadcasts across height and width
        # sky_reshaped: (Nimages, 1, 1, 1) - broadcasts across nsteps, height, and width
        model = G * fluxes_broadcastable + sky_reshaped

        # Calculate (model - data) * inverr
        # mmpix: (Nimages, height, width)
        # mmie: (Nimags, height, width)
        diff = model - mmpix[:,cp.newaxis,:,:]
        chi = diff * mmie[:,cp.newaxis,:,:] # This is (model - data) * inverr (mmie is inverr)

        # Square chi and sum over Nimages, height and width
        # Sum over Nimages (axis 0), height (axis 2), and width (axis 3)
        # The result will be a 1D array of shape (nsteps,)
        chi2 = cp.sum(chi**2, axis=(0, 2, 3))
        #print ("CHI2", cp.sum(chi**2, axis=(2, 3)), chi2.dtype)
        return chi2

    def _getBatchImageParams(self, tr, masks, pxy):
        Nimages = len(tr.images)
        # pxy: [(x,y), ...] length tr.images
        psfs = [tim.getPsf() for tim in tr.images]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        extents = [mm.extent for mm in masks]

        px, py = np.array(pxy).T
        px = px.astype(np.float32)
        py = py.astype(np.float32)
        imgH, imgW = np.array([tim.shape for tim in tr.images]).T
        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        x0, x1, y0, y1 = np.asarray(extents).T
        #print (f'{x0=} {x0.dtype=}')

        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2]), axis=0)

        # PSF Fourier transforms
        batch_psf = BatchPixelizedPSF(psfs)
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(px, py, gpu_halfsize)
        #print ("pH", pH, pW)
        #print ("Cx", cx, cy)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (Nimages,len(w),len(v)))

        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)
        return img_params, cx,cy, pH,pW


    def _getBatchGalaxyProfiles(self, amixes_gpu, masks, px, py, cx, cy, pW, pH,
                                img_counts, img_sky, img_pix, img_ie):
        Nimages = len(img_counts)
        #Nimages = len(img_pix)

        #Not optimal but for now go back into loop
        mx0, mx1, my0, my1, mh, mw = np.array([(mm.x0, mm.x1, mm.y0, mm.y1)+mm.shape for mm in masks]).T
        counts = cp.asarray(img_counts, dtype=cp.float32)

        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T

        # sub-pixel shift we have to do at the end...
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

        # Embed pix and ie in images the same size as pW,pH.
        padpix = cp.zeros((Nimages, pH,pW), cp.float32)
        padie  = cp.zeros((Nimages, pH,pW), cp.float32)
        assert(np.all(sy <= 0) and np.all(sx <= 0))

        x_delta = np.ones(mx0.shape, np.int32)
        y_delta = np.ones(my0.shape, np.int32)
        x_delta[mx0 == 0] = 0
        y_delta[my0 == 0] = 0
        x_delta[sx == 0] = 0
        y_delta[sy == 0] = 0
        for i, pix in enumerate(img_pix):
            #padpix[i, -sy[i]-1:-sy[i]+mh[i], -sx[i]-1:-sx[i]+mw[i]] = pix[my0[i]-1:my1[i], mx0[i]-1:mx1[i]]
            padpix[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = pix[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
            #print ("PIX", pix.max(), pix[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]].max(), padpix[i].max(), pix[my0[i]-y_delta[i], mx0[i]-x_delta[i]], padpix[i, -sy[i]-y_delta[i], -sx[i]-x_delta[i]])
        for i, ie in enumerate(img_ie):
            #padie[i, -sy[i]-1:-sy[i]+mh[i], -sx[i]-1:-sx[i]+mw[i]] = ie[my0[i]-1:my1[i], mx0[i]-1:mx1[i]]
            padie[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = ie[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
            #print ("IE",ie.max(), padie[i].max())
        roi = cp.asarray([-sx, -sy, mw, mh]).T
        mmpix = cp.asarray(padpix)
        mmie = cp.asarray(padie)
        sky = cp.asarray(img_sky, dtype=cp.float32)

        # Split "amix" into terms that we will evaluate using MoG vs FFT.
        # (we'll use that same split for the derivatives also)
        #vv = amix_gpu.var[:,:,0,0] + amix_gpu.var[:,:,1,1]
        _,amix_gpu,_ = amixes_gpu[0]
        nvar =  amix_gpu.var.shape[1]
        for i in range(len(amixes_gpu)):
            assert(amixes_gpu[i][1].var.shape[1] == nvar)

        vv = cp.zeros((Nimages, len(amixes_gpu), amix_gpu.var.shape[1]))
        for i, am in enumerate(amixes_gpu):
            vv[:,i] = am[1].var[:,:,0,0]+am[1].var[:,:,1,1]

        # Ramp between:
        inner_real_nsigma = 3.
        outer_real_nsigma = 4.
        nsigma1 = inner_real_nsigma
        nsigma2 = outer_real_nsigma

        # Terms that will wrap-around significantly if evaluated
        # with FFT...  We want to know: at the outer edge of this
        # patch, how many sigmas out are we?  If small (ie, the
        # edge still has a significant fraction of the flux),
        # render w/ MoG.
        IM = ((pW/2)**2 < (nsigma2**2 * vv))
        IF = ((pW/2)**2 > (nsigma1**2 * vv))
        ramp = np.any(IM*IF)
        #mogweights = cp.ones(Nimages, dtype=np.float32)
        #fftweights = cp.ones(Nimages, dtype=np.float32)
        mogweights = cp.ones(vv.shape, dtype=cp.float32)
        fftweights = cp.ones(vv.shape, dtype=cp.float32)
        if ramp:
            # ramp
            ns = (pW/2) / cp.maximum(1e-6, cp.sqrt(vv))
            #mogweights = cp.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
            #fftweights = cp.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
            mogweights = cp.minimum(1., (nsigma2 - ns) / (nsigma2 - nsigma1))*IM
            fftweights = cp.minimum(1., (ns - nsigma1) / (nsigma2 - nsigma1))*IF
            assert(cp.all(mogweights[IM] > 0.))
            assert(cp.all(mogweights[IM] <= 1.))
            assert(cp.all(fftweights[IF] > 0.))
            assert(cp.all(fftweights[IF] <= 1.))

        K = amix_gpu.var.shape[1]
        D = amix_gpu.var.shape[2]
        mh = pH
        mw = pW

        #print ("amixes_gpu", type(amixes_gpu), type(IF), type(IM), type(K), type(D))
        #print ("mogweights", type(mogweights), type(fftweights), fftweights.shape, type(px), type(py), type(mux), type(muy))
        #print ("MH", type(mh), type(mw), type(counts), type(cdi), type(roi))

        #img_derivs = BatchImageDerivs(amixes_gpu, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, cdi, roi, sky, dxi, dyi, fit_pos)
        #img_params.addBatchImageDerivs(img_derivs)
        return BatchGalaxyProfiles(amixes_gpu, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, roi, sky, dxi, dyi)

class MyAwesomeGpuImplementation(GPUFriendlyOptimizer):
    def computeUpdateDirections(self, imgs):
        return super().computeUpdateDirections(imgs)

    
if __name__ == '__main__':

    import pylab as plt
    from tractor import Image, PixPos, Flux, Tractor, NullWCS, NCircularGaussianPSF, PointSource
    from tractor import ExpGalaxy, PixelizedPSF, HybridPixelizedPSF, GaussianMixturePSF
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor import ModelMask, ConstantFitsWcs, RaDecPos
    from astrometry.util.util import Tan

    n_ims = 2
    sig1s = [3., 10.] * 5
    psf_sigmas = [2., 1.] * 5
    #pixscales = [0.25, 0.25] * 5
    pixscales = [1, 1] * 5
    H,W = 50,50
    cx,cy = 23,27
    #H,W = 100,100
    #cx,cy = 53,47
    # True source...
    true_flux = 1000.
    true_shape = [3., 0.5, 0.3]

    pixscale = 1.0
    #wcs_rot = np.linspace(0, 2.*np.pi, n_ims)
    wcs_rot = np.array([0, np.pi/4.])
    ra,dec = 100., 42.

    ra_off,dec_off = 0.0003 * np.random.normal(size=(2,))

    true_pos = RaDecPos(ra + ra_off, dec + dec_off)
    true_src = ExpGalaxy(true_pos, Flux(true_flux), EllipseE(*true_shape))

    tims = []
    for i in range(n_ims):
        #x = np.arange(W)
        #y = np.arange(H)
        #data = np.exp(-0.5 * ((x[np.newaxis,:] - cx)**2 + (y[:,np.newaxis] - cy)**2) /
        #              psf_sigmas[i]**2)
        #data *= fluxes[i] / (2. * np.pi * psf_sigmas[i]**2)
        data = np.random.normal(size=(H,W)) * sig1s[i]

        pW = pH = 63
        pp = np.arange(pW)
        psf_stamp = np.exp(-0.5 * ((pp[np.newaxis,:] - pW//2)**2 + (pp[:,np.newaxis] - pH//2)**2) /
                           psf_sigmas[i]**2) / (2. * np.pi * psf_sigmas[i]**2)
        gpsf = GaussianMixturePSF(np.array([1.]), np.array([[0.,0.]]),
                                  np.eye(2)[np.newaxis,:,:] * psf_sigmas[i]**2)
        pix = PixelizedPSF(psf_stamp)
        psf = HybridPixelizedPSF(pix, gauss=gpsf)

        c,s = np.cos(wcs_rot[i]), np.sin(wcs_rot[i])
        pixsc = pixscales[i] / 3600.
        tan = Tan(ra, dec, float(cx+i), float(cy), c*pixsc, s*pixsc, -s*pixsc, c*pixsc,
                  float(W), float(H))
        #print('Pixel scale:', tan.pixel_scale())
        wcs = ConstantFitsWcs(tan)

        tims.append(Image(data=data, inverr=np.ones_like(data) / sig1s[i],
                          psf=psf, wcs=wcs))

        tr = Tractor([tims[i]], [true_src])
        true_mod = tr.getModelImage(0)
        data += true_mod

    #src = PointSource(PixPos(W//2, H//2), Flux(100.))
    e = EllipseE(2., 0., 0.)
    src  = ExpGalaxy(RaDecPos(ra, dec), Flux(100.), EllipseESoft.fromEllipseE(e))
    src2 = ExpGalaxy(RaDecPos(ra, dec), Flux(100.), EllipseESoft.fromEllipseE(e))

    opt = ConstrainedDenseOptimizer()
    #opt2 = GPUFriendlyOptimizer()
    opt2 = MyAwesomeGpuImplementation()

    tr = Tractor(tims, [src], optimizer=opt)
    tr.setModelMasks([{src: ModelMask(0, 0, W, H)} for tim in tims])

    tr2 = Tractor(tims, [src2], optimizer=opt2)
    tr2.setModelMasks([{src2: ModelMask(0, 0, W, H)} for tim in tims])

    tr.freezeParam('images')
    tr2.freezeParam('images')

    mods = list(tr.getModelImages())

    fit_kwargs = dict(shared_params=False, priors=False)
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **fit_kwargs)
    up2 = tr2.optimizer.getLinearUpdateDirection(tr2, **fit_kwargs)
    print('Update directions:')
    print('Orig:', up1)
    print('GPU :', up2)

    print('True source:', true_src)
    print('Initial source:', src)

    R = tr.optimize_loop(**fit_kwargs)
    print('Normal fitter: took', R['steps'], 'steps')

    R2 = tr2.optimize_loop(**fit_kwargs)
    print('GPU-friendly fitter: took', R2['steps'], 'steps')

    # for step in range(20):
    #     dlnp1,x1,alpha1 = tr.optimize(**fit_kwargs)
    #     dlnp2,x2,alpha2 = tr2.optimize(**fit_kwargs)
    #     print('Step', step)
    #     print('  dlnp1:', dlnp1)
    #     print('  dlnp2:', dlnp2)
    #     print('  alpha1:', alpha1)
    #     print('  alpha2:', alpha2)
    #     print('  x1:', x1)
    #     print('  x2:', x2)
    #     print('  step1:', x1*alpha1)
    #     print('  step2:', x2*alpha2)
    #     print('  Source:', src)
    #     print('  Source2:', src2)
        
    mods_after = list(tr.getModelImages())
    print('Source:', src)
    
    mods2_after = list(tr2.getModelImages())
    print('Source:', src2)

    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods[i], **ima)
    plt.suptitle('Original optimizer, before')
    plt.savefig('1.png')

    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods_after[i], **ima)
    plt.suptitle('Original optimizer, after')
    plt.savefig('2.png')
    
    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods2_after[i], **ima)
    plt.suptitle('GPU-friendly optimizer, after')
    plt.savefig('3.png')
