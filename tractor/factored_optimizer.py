import sys
from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF
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
import time
import os, gc

tt = np.zeros(8)
tct = np.zeros(9, np.int32)
tt2 = np.zeros(1, np.int32)

image_counter = 0
#from astrometry.util.plotutils import PlotSequence
#ps = PlotSequence('fourier')

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
        if len(img_opts) == 0:
            if x_imgs is not None:
                return x_imgs
            return None
        # ~ inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        for x,ic in img_opts:
            xicsum = xicsum + np.dot(ic, x)
            icsum = icsum + ic
        #C = np.linalg.inv(icsum)
        #x = np.dot(C, xicsum)
        #print (f'{icsum=} {xicsum=}')

        #Add the priors if needed.
        if orig_priors:
            priors_ATA, priors_ATB = self.getPriorsHessianAndGradient(tr)
            #print (f'{priors_ATA=}, {priors_ATB=}')
            # Add the raw priors to the sums
            #icsum += priors_ATA
            #xicsum += priors_ATB
            if priors_ATA.shape == icsum.shape:
                icsum += priors_ATA
                xicsum += priors_ATB
            elif np.all(priors_ATA == 0) and np.all(priors_ATB == 0):
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=} but priors are zero so ignorning.")
            else:
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=}; using CPU mode instead.")
                return super().getLinearUpdateDirection(tr, **kwargs)
        x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
        if x_imgs is not None:
            x = np.append(x_imgs, x)
        #print (f'{icsum=} {xicsum=}')
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

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        #import traceback
        #traceback.print_stack()
        #print ("GPU getSingleImageUpdateDirections")
        #print ("profile galaxy", isinstance(tr.catalog[0], ProfileGalaxy))
        R_gpu = None
        R_cpu = None
        R_gpuv = None
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            if self._gpumode >= 10:
                print ("Skipping non-profile galaxy")
                return []
            #print ("Running CPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
            #p = self.ps
            #self.ps = None
            t = time.time()
            #print ("SRC", type(tr.catalog[0]))
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            #if len(R) != 0:
            #    fname = "bad_nonprofile.pickle"
            #    if not os.access(fname, os.F_OK):
            #        f = open(fname,'wb')
            #        import pickle
            #        pickle.dump(tr, f)
            #        f.close()
            tt[0] += time.time()-t
            tct[0] += 1
            #self.ps = p
            return R

        if self._gpumode == 1 or self._gpumode == 3 or self._gpumode == 11 or self._gpumode == 13:
            try:
                #print('Running GPU code...')
                if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
                    print ("Running GPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
                    tct[0] += 1
                t = time.time()
                R_gpu = self.gpuSingleImageUpdateDirections(tr, **kwargs)
                tt[1] += time.time()-t
                tct[1] += 1
                #print ("GPU time:",time.time()-t)
                if self._gpumode == 1 or self._gpumode == 11:
                    return R_gpu
            except AssertionError:
                import traceback
                print ("AssertionError in GPU code:")
                traceback.print_exc()
                print ("Running CPU version instead...")
                t = time.time()
                R_gpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 1 or self._gpumode == 11:
                    return R_gpu
            except:
                import traceback
                print('Exception in GPU code:')
                traceback.print_exc()
 
                src = tr.catalog[0]
                print('Source:', src)
                print(repr(src))
                print ("Running CPU version instead...")
                t = time.time()
                R_gpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 1 or self._gpumode == 11:
                    return R_gpu

        if self._gpumode == 2 or self._gpumode == 3 or self._gpumode == 12 or self._gpumode == 13:
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            nimages = len(tr.images)
            imsize = tr.images[0].data.size
            nd = tr.numberOfParams()+2
            kmax = 9
            #Double size of 16 bit (complex 128) array x nimage x
            #n derivs x kmax x imsize.  5D array in batch_mixture_profiles.py
            """
            Dustin less_mem version
            est_mem = nimages*imsize*nd*kmax*16 * 3.2
            # 3.2 factor: NGC 3585 example
            
            if free_mem < est_mem:
                try:
                    print("Warning: Estimated memory %.1f GB is greater than free memory %.1f GB; Running less-memory GPU mode instead!" % (est_mem / 1e9, free_mem / 1e9))
                    R_gpu = self.gpuSingleImageUpdateDirectionsVectorized_less_mem(tr, **kwargs)
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
            """

            est_mem = nimages*imsize*nd*kmax*16
            import datetime
            print (f"Running VECTORIZED GPU code...", datetime.datetime.now(), "src = ", tr.catalog[0], f'Estimated mem {est_mem=}')

            #if free_mem < est_mem:
            if False:
                print (f"Warning: Estimated memory {est_mem} is greater than free memory {free_mem}; Running CPU mode instead!")
                print (f"\t{nimages=} {imsize=} {nd=}")
                R_gpuv = super().getSingleImageUpdateDirections(tr, **kwargs)
                return R_gpuv
            #else:
                #print (f"Estimated memory {est_mem} is less than free memory {free_mem}")

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
                if self._gpumode == 2 or self._gpumode == 12:
                    return R_gpuv
            except AssertionError:
                import traceback
                print ("AssertionError in VECTORIZED GPU code:")
                traceback.print_exc()
                print ("Running CPU version instead...")
                t = time.time()
                R_gpuv = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 2 or self._gpumode == 12:
                    return R_gpuv
            except cp.cuda.memory.OutOfMemoryError:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                tot_bytes = mempool.total_bytes()
                print (f"Out of Memory for source "+str(tr.catalog[0]))
                print (f'OOM Device {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=}')
                print ("Running CPU version instead...")
                mempool.free_all_blocks()
                t = time.time()
                R_gpuv = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 2 or self._gpumode == 12:
                    return R_gpuv
            except Exception as ex:
                import traceback
                print('Exception in GPU Vectorized code: ', ex)
                traceback.print_exc()

                src = tr.catalog[0]
                print('Source:', src)
                print(repr(src))
                print ("Running CPU version instead...")
                t = time.time()
                R_gpuv = super().getSingleImageUpdateDirections(tr, **kwargs)
                if self._gpumode == 2 or self._gpumode == 12:
                    return R_gpuv

        if self._gpumode == 0 or self._gpumode == 3 or self._gpumode == 10 or self._gpumode == 13:
            try:
               # print('Running CPU code for comparison...')
                if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
                    print ("Running CPU version, frozen = ", tr.isParamFrozen('images'), "len = ", len(tr.catalog), " profile = ", isinstance(tr.catalog[0], ProfileGalaxy))
                    tct[0] += 1
                t = time.time()
                R_cpu = super().getSingleImageUpdateDirections(tr, **kwargs)
                tt[3] += time.time()-t
                tct[3] += 1
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

        xicacc_cpu = 0.
        icacc_cpu = 0.
        for x,ic in R_cpu:
            #print (f'{ic=} {x=}')
            xicacc_cpu = xicacc_cpu + np.dot(ic, x)
            icacc_cpu = icacc_cpu + ic
        #print (f'{icacc_cpu=} {xicacc_cpu=}')
        x_cpu,_,_,_ = np.linalg.lstsq(icacc_cpu, xicacc_cpu, rcond=None)
            
        xicacc_gpu = 0.
        icacc_gpu = 0.
        for x,ic in R_gpu:
            #print (f'{ic=} {x=}')
            xicacc_gpu = xicacc_gpu + np.dot(ic, x)
            icacc_gpu = icacc_gpu + ic
        #print (f'{icacc_gpu=} {xicacc_gpu=}')
        x_gpu,_,_,_ = np.linalg.lstsq(icacc_gpu, xicacc_gpu, rcond=None)

        xicacc_gpuv = 0.
        icacc_gpuv = 0.
        for x,ic in R_gpuv:
            xicacc_gpuv = xicacc_gpuv + np.dot(ic, x)
            icacc_gpuv = icacc_gpuv + ic
        x_gpuv,_,_,_ = np.linalg.lstsq(icacc_gpuv, xicacc_gpuv, rcond=None)

        print('CPU:', x_cpu)
        print('GPU:', x_gpu)
        print('GPU V:', x_gpuv)
        s = np.sum(x_cpu * x_gpu) / np.sqrt(np.sum(x_cpu**2) * np.sum(x_gpu**2))
        sv = np.sum(x_gpu * x_gpuv) / np.sqrt(np.sum(x_gpu**2) * np.sum(x_gpuv**2))
        print('Similarity CPU/GPU:', s)
        print('Similarity GPU/V:', sv)
        print ("Times:",tt,tct)
        """
        if s < 0.999 or s > 1.0001:
            src = tr.catalog[0]
            print('Source:', src)
            self.outputPickle("full.new.good", tr)
            print ("SIMILARITY BAD  source="+str(tr.catalog[0]))
            for i,((x_cpu,ic_cpu), (x_gpu,ic_gpu)) in enumerate(zip(R_cpu, R_gpu)):
                print('  CPU:', x_cpu)
                print('  GPU:', x_gpu)
                s = np.sum(x_cpu * x_gpu) / np.sqrt(np.sum(x_cpu**2) * np.sum(x_gpu**2))
                print('  similarity:', s)
                if s < 0.999 or s > 1.0001:
                  try:
                    print ("I", i, len(tr.images))
                    tim = tr.images[i]
                    print('Tim:', tim)

                    num = tt2[0]
                    fname = os.getenv('SCRATCH')+'/pickles/bad_x_new'+str(num)+'.pickle'
                    while os.access(fname, os.F_OK):
                        num += 100
                        fname = os.getenv('SCRATCH')+'/pickles/bad_x_new'+str(num)+'.pickle'
                    z = tr.images
                    tr.images = [tim]
                    self.outputPickle("x", tr, num=num)
                    tr.images = z
                    if tt2[0] > 40:
                        #Exit at 10 differences
                        sys.exit(-1)
                  except Exception as ex:
                    print ("WEIRD CRASH", str(ex))
            """
        return R_gpu

    def outputPickle(self, name, tr, num=None):
        if not os.access(os.getenv('SCRATCH')+'/pickles',os.F_OK):
            os.mkdir(os.getenv('SCRATCH')+'/pickles')
        if num is None:
            num = tt2[0]
        fname = os.getenv('SCRATCH')+'/pickles/bad_'+name+'_'+str(num)+'.pickle'
        if os.access(fname, os.F_OK):
            print ("OUTPUT ERROR: "+fname+" already exists")
            return
        print ("OUTPUTTING "+fname)
        print ("N images = ", len(tr.images))
        tt2[0] += 1
        #f = open(fname,'wb')
        #import pickle
        #pickle.dump(tr, f)
        #f.close()
        #print ("SIMILARITY BAD blobid="+str(tr.blobid)+" source="+str(tr.catalog[0]))
        #print ("SIMILARITY BAD  source="+str(tr.catalog[0]))
        return

    def gpuPointSourceUpdateDirections(self, tr, **kwargs):
        #Should call batch_pointsource to get derivs, propagates to batch_psf
        #This should be equivalent of G array and then propagate to A, B arrays as in galaxies.
        allderivs = tr.getDerivs()
        #ComputeUpdateDirections after calculation of G should then be performed
        return []

    def gpuSingleImageUpdateDirections(self, tr, **kwargs):
        
        print('Using GpuFriendly code')
        #if not (tr.isParamFrozen('images') and (len(tr.catalog) == 1) and isinstance(tr.catalog[0], ProfileGalaxy)):
        #    return gpuPointSourceUpdateDirections(tr, **kwargs)

        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        #assert(all([tim.sky.getConstant() == 0 for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
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
        masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        assert(all([m is not None for m in masks]))

        # Assume we *are* fitting for the (RA,Dec) position
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
        #print('Bands:', bands)
        img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]
        #print('ibands', img_bands)

        # (x0,x1,y0,y1) in image coordinates
        extents = [mm.extent for mm in masks]

        inner_real_nsigma = 3.
        outer_real_nsigma = 4.

        nimages = len(masks)
        gpu_px = np.zeros(nimages, dtype=np.float32)
        gpu_py = np.zeros(nimages, dtype=np.float32)
        gpu_halfsize = np.zeros(nimages, dtype=np.float32)
        sourceOut = np.zeros(nimages, dtype=bool)
        i = 0
        for mm,(px,py),(x0,x1,y0,y1),psf,pix,ie,counts,cdi,tim in zip(
                masks, pxy, extents, psfs, img_pix, img_ie, img_counts, img_cdi, tr.images):

            psfH,psfW = psf.shape
            halfsize = max([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2])
            gpu_px[i] = px
            gpu_py[i] = py
            gpu_halfsize[i] = halfsize
            sourceOut[i] = (px < x0 or px > x1 - 1 or py < y0 or py > y1 - 1)
            i += 1
        #print (f'{px=} {x0=} {x1=} {py=} {y0=} {y1=} {sourceOut=}')
        #print (f'{gpu_halfsize}')

        #print (f'{sourceOut=}')
        """
        if np.any(sourceOut):
            print ("running CPU version")
            tct[7] += 1
            sout[0] += sourceOut.sum()
            sout[1] += nimages
            t = time.time()
            xout = super().getSingleImageUpdateDirections(tr, **kwargs)
            tt[7] += time.time()-t
            return xout 
        """

        # PSF Fourier transforms
        #print ("Nimges = ",nimages, gpu_px.shape, gpu_py.shape)
        #for psf in psfs:
        #    print ("PSF type", type(psf), psf)
        #    print ("PIX type", type(psf.pix), psf.pix)
        batch_psf = BatchPixelizedPSF(psfs)
        t1 = time.time()
        #print('Getting Fourier transform of PSF at', gpu_px,gpu_py)
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(gpu_px, gpu_py, gpu_halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (nimages,len(w),len(v)))

        t1 = time.time()
        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)

        #Not optimal but for now go back into loop
        pi = 0
        for mm,(px,py),(x0,x1,y0,y1),psf,pix,ie,counts,sky,cdi,tim in zip(
                masks, pxy, extents, psfs, img_pix, img_ie, img_counts, img_sky, img_cdi, tr.images):
            #Subtract 1 from y0 and x0 CW 1/22/25 -- unknown why but rectifies difference with CPU IE
            y_delta = 1
            x_delta = 1

            # sub-pixel shift we have to do at the end...
            dx = px - cx
            dy = py - cy
            mux = dx - x0
            muy = dy - y0
            sx = int(np.round(mux))
            sy = int(np.round(muy))

            if mm.y0 == 0 or sy == 0:
                y_delta = 0
            if mm.x0 == 0 or sx == 0:
                x_delta = 0
            if mm.y1 > pix.shape[0]:
                print ("Updating Y1")
                mm.h = pix.shape[0]-mm.y0
            if mm.x1 > pix.shape[1]:
                print ("Updating X1")
                mm.w = pix.shape[1]-mm.x0
            #mmpix = pix[mm.y0-1:mm.y1, mm.x0-1:mm.x1]
            #mmie =   ie[mm.y0-1:mm.y1, mm.x0-1:mm.x1]
            mmpix = pix[mm.y0-y_delta:mm.y1, mm.x0-x_delta:mm.x1]
            mmie =   ie[mm.y0-y_delta:mm.y1, mm.x0-x_delta:mm.x1]

            # PSF Fourier transforms
            #P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)
            mh,mw = mm.shape

            # the subpixel portion will be handled with a Lanczos interpolation
            mux -= sx
            muy -= sy
            dxi = x0+sx
            dyi = y0+sy
            assert(np.abs(mux) <= 0.5)
            assert(np.abs(muy) <= 0.5)

            # Embed pix and ie in images the same size as pW,pH.
            padpix = cp.zeros((pH,pW), cp.float32)
            padie  = cp.zeros((pH,pW), cp.float32)
            assert(sy <= 0 and sx <= 0)
            #print (f'{sy=} {y_delta=} {mh=} {sx=} {x_delta=} {mw=}')

            padpix[-sy-y_delta: -sy+mh, -sx-x_delta: -sx+mw] = mmpix
            padie [-sy-y_delta: -sy+mh, -sx-x_delta: -sx+mw] = mmie
            roi = (-sx, -sy, mw, mh)
            mmpix = padpix
            mmie  = padie
            sx = sy = 0
            mh = pH
            mw = pW

            # Compute the mixture-of-Gaussian components for this galaxy model
            # (at its current parameter values)
            amix = src._getShearedProfile(tim, px, py)

            # Get derivatives for each galaxy shape parameter.
            amixes = src.getDerivativeShearedProfiles(tim, px, py)
            amixes = [('current', amix, 0.)] + amixes

            # Split "amix" into terms that we will evaluate using MoG vs FFT.
            # (we'll use that same split for the derivatives also)
            nvar =  amix.var.shape[0]
            for i in range(len(amixes)):
                assert(amixes[i][1].var.shape[0] == nvar)
                #maxnvar = max(maxnvar, amixes[i][1].var.shape[0])
            vv = np.zeros((len(amixes), amix.var.shape[0]))
            #vv = np.zeros((len(amixes), maxnvar))
            for i, am in enumerate(amixes):
                #print ("VV", vv.shape, "AMV", am[1].var.shape)
                vv[i] = am[1].var[:,0,0]+am[1].var[:,1,1]
                #vv[i,:am[1].var[:,0,0].size] = am[1].var[:,0,0]+am[1].var[:,1,1]
            #vv = amix.var[:, 0, 0] + amix.var[:, 1, 1]
            # Ramp between:
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
            #mogweights = 1.
            #fftweights = 1.
            mogweights = np.ones(vv.shape, dtype=np.float32)
            fftweights = np.ones(vv.shape, dtype=np.float32)
            if ramp:
                # ramp
                ns = (pW/2) / np.maximum(1e-6, np.sqrt(vv))
                #mogweights = np.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
                #fftweights = np.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
                mogweights = np.minimum(1., (nsigma2 - ns) / (nsigma2 - nsigma1))*IM
                fftweights = np.minimum(1., (ns - nsigma1) / (nsigma2 - nsigma1))*IF
                assert(np.all(mogweights[IM] > 0.))
                assert(np.all(mogweights[IM] <= 1.))
                assert(np.all(fftweights[IF] > 0.))
                assert(np.all(fftweights[IF] <= 1.))

            K = amix.var.shape[0]
            D = amix.var.shape[1]
            # are we fitting for the position of this source?
            fit_pos = (src.getPosition().numberOfParams() > 0)
            #print ("LEN AMIXES", len(amixes))
            img_derivs = ImageDerivs(amixes, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, cdi, roi, sky, dxi, dyi, fit_pos)
            img_params.add_image_deriv(img_derivs)
            #Commented out print below
            #img_derivs.tostr()

            assert(sx == 0 and sy == 0)
            pi += 1

        #Call collect_params() to finalize BatchImageParams object with all ImageDerivs
        img_params.collect_params()
        #nbands = 1 + max(img_bands)
        nbands = len(bands)

        priorVals = tr.getLogPriorDerivatives()
        if priorVals is not None:
            # Adjust the priors to handle the single-band case that we consider...
            bright = src.getBrightness()
            pnames = bright.getThawedParams()
            assert(len(pnames) > 0)
            bright.freezeAllBut(pnames[0])
            priorVals = tr.getLogPriorDerivatives()
            print('Prior vals with one band:', priorVals)
            bright.thawParams(*pnames)

        Xic = self.computeUpdateDirections(img_params, priorVals)
        #print ("LEN Xic", len(Xic))

        if nbands > 1:
            t1 = time.time()
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
            for iband,(x,ic) in zip(img_bands, Xic):
                #print ("X1", x.shape, x, "IC1", ic)
                #assert(fullN == len(x) + nbands - 1)
                # the "x" here are from fitting on a single image, *assuming* 2 positions are being fit
                assert(fullN == len(x) + nbands - nbands_fit + npos - npos_fit)
                x2 = cp.zeros(fullN, cp.float32)
                ic2 = cp.zeros((fullN,fullN), cp.float32)
                # source params are ordered: position, brightness, others
                #npos = 2
                #nothers = len(x)-3
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
                #print (f'{nothers=} {npos_fit=} {nbands_fit=} {npos=} {nbands=} {iband=}')

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

                # print('x')
                # print(x)
                # print('x2')
                # print(x2)
                # print('ic')
                # print(ic)
                # print('ic2')
                # print(ic2)
                
                full_xic.append((x2.get(),ic2.get()))
                #print ("X2", x2, "IC2", ic2)
            Xic = full_xic
        else:
            print ("SKIPPING nbands branch - fullN=",fullN, "nbands=",nbands,"X",Xic[0][0])

        #
        #print('Calling original version...')
        #sXic = super().getSingleImageUpdateDirections(tr, **kwargs)

        return Xic

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
-
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

        """
        #Comment out priorVals
        priorVals = tr.getLogPriorDerivatives()
        if priorVals is not None:
            # Adjust the priors to handle the single-band case that we consider...
            bright = src.getBrightness()
            pnames = bright.getThawedParams()
            assert(len(pnames) > 0)
            bright.freezeAllBut(pnames[0])
            priorVals = tr.getLogPriorDerivatives()
            print('Prior vals with one band:', priorVals)
            bright.thawParams(*pnames)
        """
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

                # print('x')
                # print(x)
                # print('x2')
                # print(x2)
                # print('ic')
                # print(ic)
                # print('ic2')
                # print(ic2)

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

        #
        #print('Calling original version...')
        #sXic = super().getSingleImageUpdateDirections(tr, **kwargs)
        return Xic

    def computeUpdateDirections(self, img_params, priorVals):
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

        #assert(img_params.mogs is None)
        assert(img_params.ffts is not None)
        assert(img_params.mux is not None)
        assert(img_params.muy is not None)
        v = img_params.v
        w = img_params.w
        #img_params.ffts is a BatchMixtureOfGaussians (see batch_mixture_profiles.py)
        Fsum = img_params.ffts.getFourierTransform(v, w, zero_mean=True)
        #Fsum shape (Nimages, maxNd, nw, nv)
        # P is created in psf.getFourierTransform - this should be done in batch on GPU
        # resulting in P already being a CuPy array
        #P shape (Nimages, nw, nv)
        P = img_params.P[:,cp.newaxis,:,:]
        #print ("FSUM", Fsum.shape, "P", P.shape, P.max())
        #cp.savetxt('gfsum.txt', Fsum.ravel())
        #cp.savetxt('gp.txt', P.ravel())
        G = cp.fft.irfft2(Fsum*P).astype(cp.float32)
        del Fsum, P
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #cp.savetxt('gg.txt', G.ravel())
        #G should be (nimages, maxNd, nw, nv) and mux and muy should be 1d vectors
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
            #print ("VARCOPY", varcopy.shape, psfmog.var.shape)
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
            det = conv_mog.var[:,:,:,0,0] * conv_mog.var[:,:,:,1,1] - conv_mog.var[:,:,:,0,1] * conv_mog.var[:,:,:,1,0]
            iv0 = conv_mog.var[:,:,:,1,1] / det
            iv1 = -(conv_mog.var[:,:,:,0,1] + conv_mog.var[:,:,:,1,0]) / det
            iv2 = conv_mog.var[:,:,:,0,0] / det
            scale = conv_mog.amp / (2.*cp.pi*cp.sqrt(det))
            #print ("IV", iv0.max(), iv1.max(), iv2.max(), iv0.shape, np.where(iv0 == iv0.max()), np.where(iv1 == iv1.max()), np.where(iv2 == iv2.max()))

            #print('conv_mog.mean shape:', conv_mog.mean.shape)
            #print('xx shape:', xx.shape)

            #print('conv_mog.means:', conv_mog.mean)
            #print('img_params.mux,muy:', img_params.mux, img_params.muy)

            # conv_mog.mean is, eg, (13 x 4 x 2 x 2)
            # (nimages x nderivs x nmog x 2), where the 2 is x,y coordinates.
            # BUT, it's really only (nimages x 2), the values for all the derivs and mogs are equal!
            #print (conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:].shape)
            #b1 = np.where(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] != conv_mog.mean)
            #print ("B1", b1)
            #print (conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:][b1])
            #print ("ORIG", conv_mog.mean[b1])

            ###  TAG: ISSUE 2 4/9/25 - uncomment assert to see crash
            #assert(np.all(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] == conv_mog.mean))
            #means = conv_mog.mean[:, 0, 0, :].copy()
            means = conv_mog.mean.copy()
            #print ("MEANS", means.shape, means)
            # now "means" is (nimages x 2)

            # xx, yy are each 64 elements long.

            #print('img_derivs length:', len(img_params.img_derivs))

            #dx = xx - means[:,0]
            #dy = yy - means[:,1]
            #TODO - @Dustin - if I make the following changes, I then need to add another axis to distsq below
            # conv_mog.mean[:,:,:,0].shape == (13,4,1) in one pass through run-one-blob.py and (13,4,2) in another
            #dx = xx - conv_mog.mean[:,:,:,cp.newaxis,0]
            #dy = yy - conv_mog.mean[:,:,:,cp.newaxis,1]
            #if use_roi:

                # rois: (rx0,ry0, rw,rh)
                #means[:,0] += cp.array([d.roi[0] for d in img_params.img_derivs])
                #means[:,1] += cp.array([d.roi[1] for d in img_params.img_derivs])

                # #Loop over img_params.img_derivs and correct mean so that mogs are centered with G
                # for img_i, imderiv in enumerate(img_params.img_derivs):
                #     (rx0,ry0,rw,rh) = imderiv.roi
                #     dx[img_i] -= rx0
                #     dy[img_i] -= ry0
            # distsq = (iv0[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dx[:,:,cp.newaxis,:] +
            #           iv1[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dy[:,:,:,cp.newaxis] +
            #           iv2[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis])

            for i, deriv in enumerate(img_params.img_derivs):
                #means[i,0] -= deriv.dx
                #means[i,1] -= deriv.dy
                means[i,:,:,0] -= deriv.dx
                means[i,:,:,1] -= deriv.dy
            #print ("MEANS2", means.shape, means)
            #print ("ROI", img_params.roi)

            # (13,4,1) = (nimages, nderivs, nmog)
            #print('scale:', scale.shape)
            #print (np.where(scale == scale.max()))
            #print('iv shapes:', iv0.shape, iv1.shape, iv2.shape)

            # (13,4,64,64) = (nimages, nderivs, ny,nx)
            #print('G:', G.shape)

            # The distsq array is going to be nimages x nderivs x nmog x ny=64 x nx=64

            # distsq = (iv0[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dx[:,:,cp.newaxis,:] +
            #           iv1[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dy[:,:,:,cp.newaxis] +
            #           iv2[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis])

            n = cp.newaxis
            #distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2 +
            #          iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]) +
            #          iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)

            iv0 = iv0.astype(cp.float32)
            iv1 = iv1.astype(cp.float32)
            iv2 = iv2.astype(cp.float32)
            xx = xx.astype(cp.float32)
            yy = yy.astype(cp.float32)
            means = means.astype(cp.float32)

            distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:])**2 +
                      iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n]) +
                      iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n])**2)
            distsq *= -0.5
            # t1 = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2)
            # print('t1', t1.shape)
            # t3 = (iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # print('t3', t3.shape)
            # t2 = (iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]))
            # print('t2', t2.shape)
            distsq = cp.exp(distsq)
            scale = scale.astype(cp.float32)
            distsq *= scale[:,:,:,cp.newaxis,cp.newaxis]
            mog_g = cp.sum(distsq, axis=2)
            # Sum over the nmog
            #mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(distsq), axis=2).astype(cp.float32)
            del means, distsq, varcopy
            #print ("MOGG", mog_g.shape, np.where(mog_g == mog_g.max()))
            #cp.savetxt('gmogpatch.txt',mog_g.ravel())
            G += mog_g

        #Do no use roi since images are padded to be (mh, mw)
        #use_roi = False
        t1 = time.time()

        Npix = img_params.mh*img_params.mw
        #Npix is a scalar
        Nd = img_params.maxNd
        cdi = img_params.cdi
        A = cp.zeros((img_params.Nimages, Npix + Npriors, Nd+2), cp.float32)
        # A is of shape (Nimages, Npix+Npriors, Nd+2)
        # The first element in img_derivs is the current galaxy model parameters.
        mod0 = G[:,0,:,:]
        # mod0 should be (Nimages, nw, nv)
        assert (mod0.shape == (img_params.Nimages, img_params.mh, img_params.mw))
        ## 02-13-25 - make a mask with only the ROI as ones to mask out background
        roi = img_params.roi
        #roimask can be different for each image
        roimask = cp.zeros((img_params.Nimages, mod0.shape[1], mod0.shape[2]), dtype=bool)
        #print ("ROI", roi)
        for i, ri in enumerate(roi): 
            roimask[i, ri[1]:ri[1]+ri[3], ri[0]:ri[0]+ri[2]] = True
        ###############

        # boolean vector, are we fitting for positions?
        fit_pos = img_params.fit_pos
        # Shift this initial model image to get X,Y pixel derivatives
        ###  TAG: ISSUE 1 4/9/25 - comment / uncomment dx and dy to toggle fit_pos 
        dx = cp.zeros_like(mod0)
        # dx is of shape (Nimages, nw, nv)
        # X derivative -- difference between shifted-left and shifted-right arrays
        #dx[:,:,1:-1] = mod0[:,:, 2:] - mod0[:,:, :-2]
        dx[fit_pos,:,1:-1] = mod0[fit_pos,:, 2:] - mod0[fit_pos,:, :-2]
        #print ("FITPOS", fit_pos)
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

        #cp.savetxt('roi.txt', roimask.ravel())

        #cp.savetxt('dx.txt', dx.ravel())
        #cp.savetxt('dy.txt', dy.ravel())
        #cp.savetxt('cdi.txt', cdi.ravel())
        #print ("DX", dx.shape, "CDI", cdi.shape, cdi)
        #print ("COUNTS", img_params.counts, "SKY", img_params.sky)

        # divide by 2 because we did +- 1 pixel
        # negative because we shifted the *image*, which is opposite
        # from shifting the *model*
        A[:,:Npix, 0] = cp.reshape(-((dx * cdi[:,0, 0][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 0][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2),(img_params.Nimages, -1))
        A[:,:Npix, 1] = cp.reshape(-((dx * cdi[:,0, 1][:,cp.newaxis, cp.newaxis] + dy * cdi[:,1, 1][:,cp.newaxis, cp.newaxis]) * img_params.counts[:,cp.newaxis, cp.newaxis] / 2), (img_params.Nimages, -1))
        del dx,dy
        A[:,:Npix,2] = cp.reshape(mod0,(img_params.Nimages, -1))
#
        #A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()
        stepsizes = img_params.steps
        #print (stepsizes.shape, img_params.counts.shape)
        #print (img_params.counts[:,cp.newaxis, cp.newaxis].shape)
        #print (stepsizes[:,cp.newaxis,1:].shape)
        #print (cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).shape)
        A[:,:Npix, 3:] = img_params.counts[:,cp.newaxis, cp.newaxis] / stepsizes[:,cp.newaxis,1:] * cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1))

        #A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
        A[:,:Npix,:] *= img_params.ie.reshape((img_params.Nimages, Npix))[:,:,cp.newaxis]
        del G

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))
        del mod0

        #print ("Stepsizes", stepsizes)
        #print ("Counts", img_params.counts[0])
        #cp.savetxt('mod0.txt', mod0.ravel())
        #cp.savetxt("gmod.txt", (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis]).ravel())
        #cp.savetxt("gie.txt", img_params.ie.ravel())
        #cp.savetxt("gpix.txt", img_params.pix.ravel())
        #cp.savetxt("ga1.txt", A.ravel())
        #cp.savetxt("gb1.txt", B.ravel())

        # B should be of shape (Nimages, :)                           
        #B = cp.append(((pix - counts*mod0) * ie).ravel(),
        #                cp.zeros(Npriors, cp.float32))


        # Append priors --do priors depend on which image I am looking at?
        #TODO not sure if this is correct for priors? 

        if priorVals is not None:
            print ("Using PRIORS")
            rA, cA, vA, pb, mub = priorVals
            #print (priorVals)
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                if fit_pos[0] == False:
                    ci += 2
                    print (f"Updating ci to {ci=}")
                for rij,vij,bij in zip(ri, vi, bi):
                    #print (f'{vij=} {bij=} {rij=} {ci=}')
                    A[:,Npix + rij, ci] = vij
                    B[:,Npix + rij] += bij
        else:
            print ("NO PRIORS")

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)
        #print ("Xicov", Xicov)

        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        colscales[colscales == 0] = 1.
        #print ("COLSCALES", colscales.shape)
        #cp.savetxt("gcols.txt", colscales.ravel())
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        A /= colscales[:,cp.newaxis, :]
        #cp.savetxt("ga2.txt", A.ravel())
        #cp.savetxt("gb1.txt", B.ravel())

        # Solve the least-squares problem!
        #X,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        #A_T_dot_A = cp.einsum("...ji,...jk", A, A)
        #A_T_dot_B = cp.einsum("...ji,...j", A, B)
        #X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        #print ("X DOT = ", X)
        X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)
        #print ("ATA", A_T_dot_A)
        #print ("ATB", A_T_dot_B)
        #print ("X", X)
        #X[fit_pos,0:2] = 0
        #print ("OUTPUT SHAPES GPU ", A.shape, B.shape, X.shape)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)

        if self.ps is not None:

            N,_,_ = A.shape
            xx = []
            for i in range(N):
                xi,_,_,_ = cp.linalg.lstsq(A[i,:,:], B[i,:], rcond=None)
                xx.append(xi)
            print('X:', X)
            print('xx:', xx)

            import pylab as plt
            plt.clf()
            myA = A.get()
            myA = myA[0,:,:]
            print('A:', myA.shape)
            myX = X.get()
            print('X', myX.shape)
            myX = myX[0,:]
            myB = B.get()
            print('B', myB.shape)
            myB = myB[0,:]

            import fitsio
            fitsio.write('gpu-x.fits', myX, clobber=True)
            fitsio.write('gpu-x-scaled.fits', myX / colscales.get(), clobber=True)
            fitsio.write('gpu-a.fits', myA.reshape((64,64,-1)) * colscales.get()[np.newaxis,:], clobber=True)
            fitsio.write('gpu-a-scaled.fits', myA.reshape((64,64,-1)), clobber=True)
            fitsio.write('gpu-b.fits', myB.reshape((64,64)), clobber=True)
            
            ax = np.dot(myA, myX)
            print('AX', ax.shape)
            print('AX', np.percentile(np.abs(ax), [1,99]))
            print('B ', np.percentile(np.abs(myB), [1,99]))
            #lo,hi = np.percentile(np.abs(myB), [1,99])
            lo,hi = myB.min(), myB.max()
            plt.clf()
            plt.subplot(1,3,1)
            imx = dict(interpolation='nearest', origin='lower', vmin=lo, vmax=hi)
            plt.imshow(ax.reshape((64,64)), **imx)
            plt.colorbar()
            plt.title('AX')
            plt.subplot(1,3,2)
            plt.imshow(myB.reshape((64,64)), **imx)
            plt.colorbar()
            plt.title('B')
            plt.subplot(1,3,3)
            plt.imshow(ax.reshape((64,64)), interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('AX')
            self.ps.savefig()

        # Undo pre-scaling
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        X /= colscales
        #print ("NANs", cp.isnan(X).sum())
        X[cp.isnan(X)] = 0
        #print ("X NORM", X)
        #else:
        #    X[cp.isnan(X)] = 0
        # del A, B
        #Have to corectly make Xic a list of tuples
        for i in range(img_params.Nimages):
            #print ("I", i, X[i], "Xicov", Xicov[i])
            Xic.append((X[i], Xicov[i]))

        """
        for img_i, imderiv in enumerate(img_params.img_derivs):
            pix, ie, mw, mh, counts, cdi, roi = imderiv.mmpix, imderiv.mmie, imderiv.mw, imderiv.mh, imderiv.counts, imderiv.cdi, imderiv.roi 
            assert(pix.shape == (mh,mw))
            # We're going to build a tall matrix A, whose number of
            # rows = number of pixels and cols = number of parameters
            # to update.  We special-case the two spatial derivatives,
            # the rest are in the 'img_derivs' list.

            # number of derivatives
            #Nd = len(img_derivs)
            #Nd = img_derivs_batch.N
            Nd = imderiv.N
            if use_roi:
                (rx0,ry0,rw,rh) = roi
                roi_slice = slice(ry0, ry0+rh), slice(rx0, rx0+rw)
                pix = pix[roi_slice]
                ie = ie[roi_slice]
                Npix = rh*rw
            else:
                Npix = mh*mw

            print (f'{Npix=}, {Npriors=}, {Nd=}')
            A = cp.zeros((Npix + Npriors, Nd+2), cp.float32)

            mod0 = None

            #assert(img_derivs_batch.mogs is None)
            #assert(img_derivs_batch.ffts is not None)
            ## "img_derivs_batch.ffts" is a BatchMixtureOfGaussians (see batch_mixture_profiles.py)
            #Fsum = img_derivs_batch.ffts.getFourierTransformBatchGPU(v, w, zero_mean=True)
            #print ("FSUM", Fsum.shape, "P", P.shape)
            ## Copy P to GPU
            ## TODO P is created in psf.getFourierTransform - this should be done in batch on GPU
            ## resulting in P already being a CuPy array
            #P = cp.asarray(P)
            #G = cp.fft.irfft2(Fsum*P[img_i,:,:])
            ## lanczos_shift_image_batch_gpu has a Python implementation in psf.py
            #print ("G", G.shape, mux, muy)
            ### TODO G should be (nimages, nx, ny) and mux and muy should be 1d vectors
            #lanczos_shift_image_batch_gpu(G, mux, muy)
            #del Fsum
            Gi = G[img_i]
            print (f'{Nd=}, {mh=}, {mw=}', Gi.shape)
            if use_roi:
                assert(Gi.shape == (Nd, rh, rw))
            else:
                assert (Gi.shape == (Nd,mh,mw))
            #if use_roi:
            #    #Gi = Gi[:,roi_slice]
            #    Gi = Gi[:,ry0:ry0+rh, rx0:rx0+rw]

            # The first element in img_derivs is the current galaxy model parameters.
            mod0 = Gi[0,:,:]
            print ("MOD0", mod0.shape, mod0)

            # Shift this initial model image to get X,Y pixel derivatives
            dx = cp.zeros_like(mod0)
            # X derivative -- difference between shifted-left and shifted-right arrays
            dx[:,1:-1] = mod0[:, 2:] - mod0[:, :-2]
            # Y derivative -- difference between shifted-down and shifted-up arrays
            dy = cp.zeros_like(mod0)
            dy[1:-1, :] = mod0[2:, :] - mod0[:-2, :]
            # Push through local WCS transformation to get to RA,Dec param derivatives
            assert(cdi.shape == (2,2))
            # divide by 2 because we did +- 1 pixel
            # negative because we shifted the *image*, which is opposite
            # from shifting the *model*
            A[:Npix, 0] = -((dx * cdi[0, 0] + dy * cdi[1, 0]) * counts / 2).ravel()
            A[:Npix, 1] = -((dx * cdi[0, 1] + dy * cdi[1, 1]) * counts / 2).ravel()
            del dx,dy

            # The derivative with respect to flux (brightness) = the unit-flux current model
            A[:Npix, 2] = mod0.ravel()

            #stepsizes = [s for _,s,_,_ in img_derivs]
            stepsizes = imderiv.steps
            #print ("STEPSIZES", len(stepsizes), Nd, imderiv.N)
            # The first element is mod0, so the stepped parameters start at 1 here.
            for i in range(1, Nd):
                # For other parameters, compute the numerical derivative.
                # mod0 is the unit-brightness model at the current position
                # G[i,*] is the unit-brightness model after stepping the parameter
                # The i+2 here is because the first two params are the spatial derivs
                # (A[:,0] and A[:,1] are filled above)
                # And the next param is the flux deriv
                # (A[:,2] is filled)
                # So the first time through this loop, i=1 and we fill column A[:,3]
                A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()

            # We want to minimize:
            #   || chi + (d(chi)/d(params)) * dparams ||^2
            # So  b = chi
            #     A = -d(chi)/d(params)
            #     x = dparams
            #
            # chi = (data - model) / std = (data - model) * inverr
            # derivs = d(model)/d(param)
            # A matrix = -d(chi)/d(param)
            #          = + (derivs) * inverr
            #
            # Parameters to optimize go in the columns of matrix A
            # Pixels go in the rows.

            # Scale by IE (inverse-error) weighting to get to units of chi
            A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
            # The current residuals = the observed image "pix" minus the current model (counts*mod0),
            # weighted by the inverse-errors.
            print (f'{Npriors=}')
            print ("A", A.shape, pix.shape, ie.shape)
            B = cp.append(((pix - counts*mod0) * ie).ravel(),
                          cp.zeros(Npriors, cp.float32))
            print ("B", B.shape)

            # Append priors
            if priorVals is not None:
                rA, cA, vA, pb, mub = priorVals
                for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                    for rij,vij,bij in zip(ri, vi, bi):
                        A[Npix + rij, ci] = vij
                        B[Npix + rij] += bij

            if False:
                import pylab as plt
                n,m = A.shape
                for i in range(m):
                    plt.clf()
                    plt.imshow(A[:Npix,i].get().reshape((mh,mw)), interpolation='nearest', origin='lower')
                    plt.savefig('gpu-img%i-d%i.png' % (img_i, i))
                for i in range(Nd):
                    plt.clf()
                    plt.imshow(Fsum[img_i,i,:,:].get(), interpolation='nearest', origin='lower')
                    plt.savefig('gpu-imgF%i-d%i.png' % (img_i, i))

            # Compute the covariance matrix
            Xicov = cp.matmul(A.T, A)

            # Pre-scale the columns of A
            colscales = cp.sqrt(cp.diag(Xicov))
            A /= colscales[cp.newaxis, :]

            # Solve the least-squares problem!
            X,_,_,_ = cp.linalg.lstsq(A, B, rcond=None)

            # Undo pre-scaling
            X /= colscales

            if self.ps is not None:
                import pylab as plt
                plt.clf()
                ima = dict(interpolation='nearest', origin='lower')
                rr,cc = 3,4
                plt.subplot(rr,cc,1)
                plt.imshow(mod0.get(), **ima)
                plt.title('mod0')
                sh = mod0.shape
                plt.subplot(rr,cc,2)
                mx = max(cp.abs(B))
                imx = ima.copy()
                imx.update(vmin=-mx, vmax=+mx)
                plt.imshow(B[:Npix].get().reshape(sh), **imx)
                plt.title('B')
                AX = np.dot(A, X)
                plt.subplot(rr,cc,3)
                plt.imshow(AX[:Npix].get().reshape(sh), **imx)
                plt.title('A X')
                for i in range(min(Nd+2, 8)):
                    plt.subplot(rr,cc,5+i)
                    plt.imshow(A[:Npix,i].get().reshape(sh), **ima)
                    if i == 0:
                        plt.title('dx')
                    elif i == 1:
                        plt.title('dy')
                    elif i == 2:
                        plt.title('dflux')
                    else:
                        #plt.title(img_derivs[i-2][0])
                        plt.title("I = "+str(i))
                #plt.suptitle('GPU: image %i/%i' % (img_i+1, img_params.N))
                self.ps.savefig()

            del A,B
            Xic.append((X, Xicov))
            print ("X2", X.shape, Xicov.shape)
        """
        del img_params, A, B, X
        #print_timer()
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

        t1 = time.time()
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
        #roimask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1
        ###############

        # boolean vector, are we fitting for positions?
        fit_pos = img_params.fit_pos
        #print ("FITPOS V", fit_pos)
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

        #A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()
        stepsizes = img_params.steps
        #print ("A", img_params.counts[:,cp.newaxis, cp.newaxis].shape, stepsizes[:,cp.newaxis,1:].shape, cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1)).shape)
        #print (A[:,:Npix, 3:].shape)
        A[:,:Npix, 3:] = img_params.counts[:,cp.newaxis, cp.newaxis] / stepsizes[:,cp.newaxis,1:] * cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1))

        #A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
        A[:,:Npix,:] *= img_params.ie.reshape((img_params.Nimages, Npix))[:,:,cp.newaxis]
        del G

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))
        del mod0, img_params
        #B[:,:Npix] = ((img_params.pix - img_params.counts[:,cp.newaxis, cp.newaxis]*mod0) * img_params.ie).reshape((img_params.Nimages, Npix))
        # B should be of shape (Nimages, :)                           
        #B = cp.append(((pix - counts*mod0) * ie).ravel(),
        #                cp.zeros(Npriors, cp.float32))

        #cp.savetxt('vmod0.txt', mod0.ravel())
        #cp.savetxt("vgmod.txt", (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis]).ravel())
        #cp.savetxt("vgie.txt", img_params.ie.ravel())
        #cp.savetxt("vgpix.txt", img_params.pix.ravel())
        #cp.savetxt("vga1.txt", A.ravel())
        #cp.savetxt("vgb1.txt", B.ravel())

        # Append priors --do priors depend on which image I am looking at?
        #TODO not sure if this is correct for priors? 

        """
        Priors should NOT be added in here - this will be removed
        if priorVals is not None:
            print ("Using PRIORS")
            rA, cA, vA, pb, mub = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                if fit_pos[0] == False:
                    ci += 2
                    print (f"Updating ci to {ci=}")
                for rij,vij,bij in zip(ri, vi, bi):
                    A[:,Npix + rij, ci] = vij
                    B[:,Npix + rij] += bij
        else:
            print ("NO PRIORS")
        """

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)
        #print ("Xicov V", Xicov)

        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        colscales[colscales == 0] = 1.
        ###  TAG: ISSUE 1 4/9/25 - uncomment and move divide within block to get rid of NaNs 
        #if fit_pos[0] is True:
        A /= colscales[:,cp.newaxis, :]

        # Solve the least-squares problem!
        #X,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        #A_T_dot_A = cp.einsum("...ji,...jk", A, A)
        #A_T_dot_B = cp.einsum("...ji,...j", A, B)
        #X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        #print ("X DOT V", X)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)
        #X[fit_pos,0:2] = 0
        #print ("A", A)
        #print ("B", B)
        #print ("XV", X)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)

        ## TAG: indidious hang issue 10/17/25 - pinv can occasionally hang.
        # But we had the issue where if fit_pos is False for all images, and
        # first two columns of A are all zero, the ATA ATB method fails with
        # NaNs.  But we can strip out those rows and zero-pad the output!
        if not np.any(fit_pos):
            #print ("WARNING: fit_pos is False for all images, stripping off rows to prevent NaNs before fitting")
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
        #print ("X NORM", X.shape)
        #    print ("X norm", X)
        #else:
        #    X[cp.isnan(X)] = 0
        # del A, B
        #Have to corectly make Xic a list of tuples
        #for i in range(img_params.Nimages):
        #    Xic.append((X[i], Xicov[i]))
        ##Change return args to keep as cupy arrays
        Xic = (X, Xicov)
        del A, B, X
        return Xic

    def computeGalaxyModelsVectorized(self, img_params):
        # Note, this does *not* scale by *counts*, it produces unit-flux models
        t1 = time.time()
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
        G = cp.fft.irfft2(Fsum*P).astype(cp.float32)
        del Fsum, P
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #cp.savetxt('vgg.txt', G.ravel())
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
            det = conv_mog.var[:,:,:,0,0] * conv_mog.var[:,:,:,1,1] - conv_mog.var[:,:,:,0,1] * conv_mog.var[:,:,:,1,0]
            iv0 = conv_mog.var[:,:,:,1,1] / det
            iv1 = -(conv_mog.var[:,:,:,0,1] + conv_mog.var[:,:,:,1,0]) / det
            iv2 = conv_mog.var[:,:,:,0,0] / det
            scale = conv_mog.amp / (2.*cp.pi*cp.sqrt(det))

            #print('conv_mog.mean shape:', conv_mog.mean.shape)
            #print('xx shape:', xx.shape)

            #print('conv_mog.means:', conv_mog.mean)
            #print('img_params.mux,muy:', img_params.mux, img_params.muy)

            # conv_mog.mean is, eg, (13 x 4 x 2 x 2)
            # (nimages x nderivs x nmog x 2), where the 2 is x,y coordinates.
            # BUT, it's really only (nimages x 2), the values for all the derivs and mogs are equal!
            ###  TAG: ISSUE 2 4/9/25 - uncomment assert to see crash
            #assert(np.all(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] == conv_mog.mean))
            #means = conv_mog.mean[:, 0, 0, :].copy()
            means = conv_mog.mean.copy()
            # now "means" is (nimages x 2)

            # xx, yy are each 64 elements long.

            #print('img_derivs length:', len(img_params.img_derivs))

            #dx = xx - means[:,0]
            #dy = yy - means[:,1]
            #TODO - @Dustin - if I make the following changes, I then need to add another axis to distsq below
            # conv_mog.mean[:,:,:,0].shape == (13,4,1) in one pass through run-one-blob.py and (13,4,2) in another
            #dx = xx - conv_mog.mean[:,:,:,cp.newaxis,0]
            #dy = yy - conv_mog.mean[:,:,:,cp.newaxis,1]
            #if use_roi:

                #print ("MEANS", means[:,0].shape, means[:,1].shape, img_params.roi.shape)
                ## rois: (rx0,ry0, rw,rh)
                #means[:,0] += cp.array([d.roi[0] for d in img_params.img_derivs])
                #means[:,1] += cp.array([d.roi[1] for d in img_params.img_derivs])
                #means[:,0] += img_params.roi[:,0]
                #means[:,1] += img_params.roi[:,1]

            #means[:,0] -= img_params.dx
            #means[:,1] -= img_params.dy
            #print ("MEANS", means.shape, img_params.dx.shape, means[:,:,:,0].shape)
            means[:,:,:,0] -= img_params.dx[:,None,None]
            means[:,:,:,1] -= img_params.dy[:,None,None]

            # (13,4,1) = (nimages, nderivs, nmog)
            #print('scale:', scale.shape)
            #print('iv shapes:', iv0.shape, iv1.shape, iv2.shape)

            # (13,4,64,64) = (nimages, nderivs, ny,nx)
            #print('G:', G.shape)

            # The distsq array is going to be nimages x nderivs x nmog x ny=64 x nx=64
            n = cp.newaxis
            #distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2 +
            #          iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]) +
            #          iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:])**2 +
                      iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,:,:,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n]) +
                      iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,:,:,n,1][:,:,:,:,n])**2)
            # t1 = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2)
            # print('t1', t1.shape)
            # t3 = (iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # print('t3', t3.shape)
            # t2 = (iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]))
            # print('t2', t2.shape)
            # Sum over the nmog
            mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(-0.5*distsq), axis=2).astype(cp.float32)
            del means, distsq, varcopy
            #cp.savetxt('vgmogpatch.txt',mog_g.ravel())
            G += mog_g

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
        if self._gpumode == 0:
            return super().tryUpdates(tractor, X, alphas=alphas)

        t = time.time()
        #mempool = cp.get_default_memory_pool()
        #used_bytes = mempool.used_bytes()
        #tot_bytes = mempool.total_bytes()
        #print (f'{used_bytes=} {tot_bytes=}')
        #mempool.free_all_blocks()
        #used_bytes = mempool.used_bytes()
        #tot_bytes = mempool.total_bytes()
        #print (f'After free {used_bytes=} {tot_bytes=}')
        return super().tryUpdates(tractor, X, alphas=alphas, check_step=check_step)
        print ("GPU FACTORED TRY UPDATES")

        # Dustin's FIXME improvement ideas
        # - pass in / cache initial logprob / model?
        # - pass in / cache BatchPixelizedPSF ??

        if not ((len(tractor.catalog) == 1) and
                isinstance(tractor.catalog[0], ProfileGalaxy) and
                tractor.isParamFrozen('images')):
            print('Calling superclass tryUpdates')
            return super().tryUpdates(tractor, X, alphas=alphas)

        print('number of images:', len(tractor.images))
        Nimages = len(tractor.images)
        src = tractor.catalog[0]
        steps = self.getParameterSteps(tractor, X, alphas)
        print('number of steps (alphas) to try:', len(steps))
        print ("Alphas", alphas)
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

        # pixel position of the source in each image
        xy = [tim.getWcs().positionToPixel(src.pos) for tim in tractor.images]
        px, py = np.array(xy).T
        # pixel region to render
        masks = [tractor._getModelMaskFor(tim, src) for tim in tractor.images]
        img_sky = [tim.getSky().getConstant() for tim in tractor.images]
        img_pix = [tim.getImage(use_gpu=True) for tim in tractor.images]
        img_ie  = [tim.getInvError(use_gpu=True) for tim in tractor.images]
        #print ("MAXES", img_pix[0].max(), img_ie[0].max())
        #print ("SUMS", img_pix[0].sum(), img_ie[0].sum())

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

        img_params, cx,cy,pW,pH = self._getBatchImageParams(tractor, masks, xy)

        gals = self._getBatchGalaxyProfiles(profiles, masks, px, py, cx, cy, pW, pH,
                                            fluxes[0], img_sky, img_pix, img_ie)
        img_params.addBatchGalaxyProfiles(gals)
        G = self.computeGalaxyModelsVectorized(img_params)
        t = time.time()
        assert(G.shape == (Nimages, len(steps), pH, pW))
        mmpix = gals.mmpix
        mmie = gals.mmie
        fluxes = cp.asarray(fluxes, dtype=cp.float32)
        sky = cp.asarray(img_sky, dtype=cp.float32)
        #print ("MMPIX", mmpix.shape, mmie.shape, mmpix.max(), mmpix.sum())
        #print ("FLUXES", fluxes.shape, fluxes)
        #print ("SKY", sky.shape, sky)

        #print ("G", G.shape)
        #chisq = cp.sum([((G[i_image] * fluxes[i_step, i_image] - mmpix[i_image]) * mmie[i_image])**2 for i_image in range(Nimages)])
        chisq = self.calculate_chi2_cupy(G, fluxes, mmpix, mmie, sky) 
        del G, img_params
        mpool = cp.get_default_memory_pool()
        mpool.free_all_blocks()
        #print ("CHISQ", chisq.shape)

        #print ("PRIOR", logpriors)
        logprob = np.array(logpriors)-0.5*chisq.get()
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
        
        if max_idx == len(logprob)-1:
            print ("Best is previous")
            return (0., 0.)

        lp_best = logprob[max_idx]
        lp_last = logprob[-1]
        if lp_best > lp_last:
            # Best we've found so far -- accept this step!
            self.last_step_hit_limit = hit_limit
        return (lp_best-lp_last, alphas[max_idx])


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
        """
        if np.any(px < 0):
            print (f"Warning: {px=} is outside of image")
            px[px < 0] = 0
        if np.any(py < 0):
            print (f"Warning: {py=} is outside of image")
            py[py < 0] = 0
        if np.any(px > imgW):
            print (f"Warning: {px=} is outside of image")
            px[px > imgW] = imgW[px > imgW]
        if np.any(py > imgH):
            print (f"Warning: {py=} is outside of image")
            py[py > imgH] = imgH[py > imgH]
        """
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
