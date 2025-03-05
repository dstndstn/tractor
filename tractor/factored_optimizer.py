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
from tractor.batch_mixture_profiles import ImageDerivs, BatchImageParams, BatchMixtureOfGaussians, BatchImageDerivs
from tractor.batch_galaxy import getShearedProfileGPU, getDerivativeShearedProfilesGPU, print_tbs
from tractor.patch import ModelMask
import cupy as cp
import time
from tractor.galaxy import print_ts

tx = np.zeros(12)
ty = np.zeros(4)
tc = np.zeros(4, np.int32)
tz = np.zeros(10)
th = np.zeros(10)
tg = np.zeros(10)


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

    def getSingleImageUpdateDirection(self, tr, **kwargs):
        allderivs = tr.getDerivs()
        r = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, **kwargs)
        if r is None:
            return None
        x,A,colscales,B = r
        # print('SingeImageUpdateDirection: tr thawed params:')
        # tr.printThawedParams()
        # print('allderivs:', len(allderivs))
        # print('x:', x)

        if False:
            print('Got A matrix:', A.shape)
            global image_counter
            n,m = A.shape
            for i in range(m):
                plt.clf()
                plt.imshow(A[:,i].reshape((50,50)), interpolation='nearest', origin='lower')
                plt.savefig('orig-img%i-d%i.png' % (image_counter, i))
            image_counter += 1

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
        for i,img in enumerate(imgs):
            tr.images = Images(img)
            if mm is not None:
                tr.modelMasks = [mm[i]]
            r = self.getSingleImageUpdateDirection(tr, **kwargs)
            if r is None:
                continue
            x,x_icov = r
            #print('FO: X', x, 'x_icov', x_icov)
            img_opts.append((x,x_icov))
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def getLinearUpdateDirection(self, tr, **kwargs):

        # We can fit for image-based parameters (eg, sky level) and source-based parameters.
        # If image parameters are being fit, use the base code (eg in lsqr_optimizer.py)
        # to fit those, and prepend them to the results below.
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
        x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
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

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            #p = self.ps
            #self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            #self.ps = p
            return R

        print('Running GPU code...')
        try:
            t = time.time()
            R_gpu = self.gpuSingleImageUpdateDirections(tr, **kwargs)
            print ("GPU time:",time.time()-t)
            # return R_gpu
        except:
            import traceback
            print('Exception in GPU code:')
            traceback.print_exc()

            src = tr.catalog[0]
            print('Source:', src)
            print(repr(src))
            f = open('bad.pickle','wb')
            import pickle
            pickle.dump(tr, f)
            f.close()
            sys.exit(-1)

        print ('Running VECTORIZED GPU code...')
        tx[:] = 0
        try:
            t = time.time()
            R_gpuv = self.gpuSingleImageUpdateDirectionsVectorized(tr, **kwargs)
            print ("GPU Vectorized time:", time.time()-t)
            # return R_gpuv
        except:
            import traceback
            print('Exception in GPU Vectorized code:')
            traceback.print_exc()

            src = tr.catalog[0]
            print('Source:', src)
            print(repr(src))
            f = open('bad.pickle','wb')
            import pickle
            pickle.dump(tr, f)
            f.close()
            sys.exit(-1)

        print('Running CPU code for comparison...')
        t = time.time()
        R_cpu = super().getSingleImageUpdateDirections(tr, **kwargs)
        print ("CPU time", time.time()-t)

        xicacc_cpu = 0.
        icacc_cpu = 0.
        for x,ic in R_cpu:
            xicacc_cpu = xicacc_cpu + np.dot(ic, x)
            icacc_cpu = icacc_cpu + ic
        x_cpu,_,_,_ = np.linalg.lstsq(icacc_cpu, xicacc_cpu, rcond=None)
            
        xicacc_gpu = 0.
        icacc_gpu = 0.
        for x,ic in R_gpu:
            xicacc_gpu = xicacc_gpu + np.dot(ic, x)
            icacc_gpu = icacc_gpu + ic
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
        if s < 0.5:
            src = tr.catalog[0]
            print('Source:', src)
            for i,((x_cpu,ic_cpu), (x_gpu,ic_gpu)) in enumerate(zip(R_cpu, R_gpu)):
                print('  CPU:', x_cpu)
                print('  GPU:', x_gpu)
                s = np.sum(x_cpu * x_gpu) / np.sqrt(np.sum(x_cpu**2) * np.sum(x_gpu**2))
                print('  similarity:', s)
                if s < 0.9:
                    tim = tr.images[i]
                    print('Tim:', tim)
                    f = open('bad2.pickle', 'wb')
                    import pickle
                    tr.images = [tim]
                    pickle.dump(tr, f)
                    f.close()
                    sys.exit(-1)
        return R_gpu

    def gpuSingleImageUpdateDirections(self, tr, **kwargs):
        
        #print('Using GpuFriendly code')
        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        #assert(all([tim.sky.getConstant() == 0 for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
        img_pix = [tim.data for tim in tr.images]
        img_ie  = [tim.getInvError() for tim in tr.images]
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
        gpu_px = np.zeros(nimages)
        gpu_py = np.zeros(nimages)
        gpu_halfsize = np.zeros(nimages)
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
            i += 1

        # PSF Fourier transforms
        batch_psf = BatchPixelizedPSF(psfs)
        t1 = time.time()
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(gpu_px, gpu_py, gpu_halfsize)
        add_to_timer(0, time.time()-t1)
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
            if mm.y0 == 0:
                y_delta = 0
            if mm.x0 == 0:
                x_delta = 0
            #mmpix = pix[mm.y0-1:mm.y1, mm.x0-1:mm.x1]
            #mmie =   ie[mm.y0-1:mm.y1, mm.x0-1:mm.x1]
            mmpix = pix[mm.y0-y_delta:mm.y1, mm.x0-x_delta:mm.x1]
            mmie =   ie[mm.y0-y_delta:mm.y1, mm.x0-x_delta:mm.x1]

            # PSF Fourier transforms
            #P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)
            mh,mw = mm.shape

            # sub-pixel shift we have to do at the end...
            dx = px - cx
            dy = py - cy
            mux = dx - x0
            muy = dy - y0
            sx = int(np.round(mux))
            sy = int(np.round(muy))
            # the subpixel portion will be handled with a Lanczos interpolation
            mux -= sx
            muy -= sy
            dxi = x0+sx
            dyi = y0+sy
            assert(np.abs(mux) <= 0.5)
            assert(np.abs(muy) <= 0.5)

            # Embed pix and ie in images the same size as pW,pH.
            padpix = np.zeros((pH,pW), np.float32)
            padie  = np.zeros((pH,pW), np.float32)
            assert(sy <= 0 and sx <= 0)
            #padpix[-sy-1: -sy+mh, -sx-1: -sx+mw] = mmpix
            #padie [-sy-1: -sy+mh, -sx-1: -sx+mw] = mmie
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
            vv = np.zeros((len(amixes), amix.var.shape[0]))
            for i, am in enumerate(amixes):
                vv[i] = am[1].var[:,0,0]+am[1].var[:,1,1]
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
            img_derivs = ImageDerivs(amixes, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, cdi, roi, sky, dxi, dyi)
            img_params.add_image_deriv(img_derivs)
            #Commented out print below
            #img_derivs.tostr()

            assert(sx == 0 and sy == 0)

        #Call collect_params() to finalize BatchImageParams object with all ImageDerivs
        img_params.collect_params()
        add_to_timer(1, time.time()-t1)
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

        if nbands > 1:
            t1 = time.time()
            full_xic = []
            fullN = tr.numberOfParams()
            for iband,(x,ic) in zip(img_bands, Xic):
                assert(fullN == len(x) + nbands - 1)
                x2 = cp.zeros(fullN, cp.float32)
                ic2 = cp.zeros((fullN,fullN), cp.float32)
                # source params are ordered: position, brightness, others
                npos = 2
                nothers = len(x)-3

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

                # aa
                x2[:npos] = x[:npos]
                # b
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
                ic2[npos + iband, npos + iband] = ic[npos, npos]
                # E
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
            Xic = full_xic
            add_to_timer(2, time.time()-t1)

        #
        #print('Calling original version...')
        #sXic = super().getSingleImageUpdateDirections(tr, **kwargs)

        return Xic

    def gpuSingleImageUpdateDirectionsVectorized(self, tr, **kwargs):
        t2 = time.time()
        add_to_t2(0, 0)
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            p = self.ps
            self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            self.ps = p
            return R

        #print('Using GpuFriendly code')
        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tr.images]))
        #assert(all([tim.sky.getConstant() == 0 for tim in tr.images]))
        # Assume single source
        assert(len(tr.catalog) == 1)
        t1 = time.time()
        Nimages = len(tr.images)

        img_pix = [tim.data for tim in tr.images]
        img_ie  = [tim.getInvError() for tim in tr.images]
        # Assume galaxy
        src = tr.catalog[0]
        assert(isinstance(src, ProfileGalaxy))
        psfs = [tim.getPsf() for tim in tr.images]
        th[0] += time.time()-t2
        t1 = time.time()
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))
        th[1] += time.time()-t1
        t1 = time.time()

        # Assume ConstantSky models, grab constant sky levels
        # NOTE - instead of building this list and passing it around in ImageDerivs, etc,
        # we could perhaps just subtract it off img_pix at the start...
        img_sky = [tim.getSky().getConstant() for tim in tr.images]

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskByIdx(i, src) for i in range(len(tr.images))]
        #masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        th[2] += time.time()-t1
        t1 = time.time()
        assert(all([m is not None for m in masks]))

        assert(src.isParamThawed('pos'))
        th[3] += time.time()-t1
        t1 = time.time()

        # FIXME -- must handle priors (ellipticity)!!

        #add_to_timer(10, time.time()-t2)
        #t1 = time.time()
        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        th[4] += time.time()-t1
        t1 = time.time()
        # WCS inv(CD) matrix
        img_cdi = [tim.getWcs().cdInverseAtPosition(src.getPosition(), src=src)
                   for tim in tr.images]
        th[5] += time.time()-t1
        t1 = time.time()
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]
        th[6] += time.time()-t1
        add_to_timer(10, time.time()-t2)
        t1 = time.time()
        bands = src.getBrightness().getParamNames()
        #print('Bands:', bands)
        img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]
        #print('ibands', img_bands)

        # (x0,x1,y0,y1) in image coordinates
        extents = [mm.extent for mm in masks]
        #extents = np.array([mm.extent for mm in masks])

        inner_real_nsigma = 3.
        outer_real_nsigma = 4.

        print ("Calling VECTORIZED version")
        #print (type(tr.images), type(tr))
        px, py = np.array(pxy).T
        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        x0, x1, y0, y1 = np.asarray(extents).T
        gpu_halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2]), axis=0)

        add_to_timer(11, time.time()-t1)
        t1 = time.time()
        # PSF Fourier transforms
        batch_psf = BatchPixelizedPSF(psfs)
        add_to_timer(0, time.time()-t1)
        t1 = time.time()
        P, (cx, cy), (pH, pW), (v, w) = batch_psf.getFourierTransformBatchGPU(px, py, gpu_halfsize)
        #print ("pH", pH, pW)
        #print ("Cx", cx, cy)
        add_to_timer(1, time.time()-t1)
        t1 = time.time()
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        assert(P.shape == (Nimages,len(w),len(v)))

        img_params = BatchImageParams(P, v, w, batch_psf.psf_mogs)
        tz[0] += time.time()-t1
        t9 = time.time()

        #Not optimal but for now go back into loop
        mx0, mx1, my0, my1, mh, mw = np.array([(mm.x0, mm.x1, mm.y0, mm.y1)+mm.shape for mm in masks]).T
        counts = cp.asarray(img_counts)
        cdi = cp.asarray(img_cdi)

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
        padpix = np.zeros((Nimages, pH,pW), np.float32)
        padie  = np.zeros((Nimages, pH,pW), np.float32)
        assert(np.all(sy <= 0) and np.all(sx <= 0))
        tg[0] += time.time()-t9
        t9 = time.time()

        x_delta = np.ones(mx0.shape, np.int32)
        y_delta = np.ones(my0.shape, np.int32)
        x_delta[mx0 == 0] = 0
        y_delta[my0 == 0] = 0
        for i, pix in enumerate(img_pix):
            #padpix[i, -sy[i]-1:-sy[i]+mh[i], -sx[i]-1:-sx[i]+mw[i]] = pix[my0[i]-1:my1[i], mx0[i]-1:mx1[i]]
            padpix[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = pix[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
        for i, ie in enumerate(img_ie):
            #padie[i, -sy[i]-1:-sy[i]+mh[i], -sx[i]-1:-sx[i]+mw[i]] = ie[my0[i]-1:my1[i], mx0[i]-1:mx1[i]]
            padie[i, -sy[i]-y_delta[i]:-sy[i]+mh[i], -sx[i]-x_delta[i]:-sx[i]+mw[i]] = ie[my0[i]-y_delta[i]:my1[i], mx0[i]-x_delta[i]:mx1[i]]
        tg[1] += time.time()-t9
        t9 = time.time()
        roi = cp.asarray([-sx, -sy, mw, mh]).T
        mmpix = cp.asarray(padpix)
        mmie = cp.asarray(padie)
        sky = cp.asarray(img_sky)
        #print("SKY", sky.shape)
        tg[2] += time.time()-t9
        t9 = time.time()

        #print ("DX", dx.shape)
        #print ("MX0", mx0.shape)

        #for tim in tr.images:
        #    print (type(tim), type(tim.getWcs()), type(tim.getWcs().cdInverseAtPixel(px[0], py[0])))
        #    print (tim.getWcs().cdInverseAtPixel(px[0], py[0]))
        #amix_gpu = src._getShearedProfileGPU(tr.images, px, py)
        amix_gpu = getShearedProfileGPU(src, tr.images, px, py)
        amixes_gpu = getDerivativeShearedProfilesGPU(src, tr.images, px, py)
        amixes_gpu = [('current', amix_gpu, 0.)] + amixes_gpu
        tg[3] += time.time()-t9
        t9 = time.time()

        # Split "amix" into terms that we will evaluate using MoG vs FFT.
        # (we'll use that same split for the derivatives also)
        #vv = amix_gpu.var[:,:,0,0] + amix_gpu.var[:,:,1,1]
        vv = cp.zeros((Nimages, len(amixes_gpu), amix_gpu.var.shape[1]))
        for i, am in enumerate(amixes_gpu):
            vv[:,i] = am[1].var[:,:,0,0]+am[1].var[:,:,1,1]

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
        tg[4] += time.time()-t9
        t9 = time.time()
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
        tg[5] += time.time()-t9
        t9 = time.time()
        mh = pH
        mw = pW

        #print ("amixes_gpu", type(amixes_gpu), type(IF), type(IM), type(K), type(D))
        #print ("mogweights", type(mogweights), type(fftweights), fftweights.shape, type(px), type(py), type(mux), type(muy))
        #print ("MH", type(mh), type(mw), type(counts), type(cdi), type(roi))
        img_derivs = BatchImageDerivs(amixes_gpu, IM, IF, K, D, mogweights, fftweights, px, py, mux, muy, mmpix, mmie, mh, mw, counts, cdi, roi, sky, dxi, dyi)
        img_params.addBatchImageDerivs(img_derivs)
        #img_derivs.tostr()
        tg[6] += time.time()-t9

        i = 0
        #print_ts()
        #print_tbs()
        add_to_timer(2, time.time()-t1)
        t1 = time.time()
        #Call collect_params() to finalize BatchImageParams object with all ImageDerivs
        #img_params.collect_params()
        add_to_timer(3, time.time()-t1)
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

        add_to_t2(2, time.time()-t2)
        Xic = self.computeUpdateDirectionsVectorized(img_params, priorVals)
        print (f'{nbands=}')
        t1 = time.time()

        if nbands > 1:
            full_xic = []
            fullN = tr.numberOfParams()
            (X, Xicov) = Xic
            X = X.get()
            Xicov = Xicov.get()
            for iband,x,ic in zip(img_bands, X, Xicov):
                assert(fullN == len(x) + nbands - 1)
                x2 = np.zeros(fullN, np.float32)
                ic2 = np.zeros((fullN,fullN), np.float32)
                # source params are ordered: position, brightness, others
                npos = 2
                nothers = len(x)-3

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

                # aa
                x2[:npos] = x[:npos]
                # b
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
                ic2[npos + iband, npos + iband] = ic[npos, npos]
                # E
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
        add_to_timer(4, time.time()-t1)

        #
        #print('Calling original version...')
        #sXic = super().getSingleImageUpdateDirections(tr, **kwargs)

        print_timer()
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
        t1 = time.time()
        Fsum = img_params.ffts.getFourierTransform(v, w, zero_mean=True)
        add_to_timer(4, time.time()-t1)
        #Fsum shape (Nimages, maxNd, nw, nv)
        # P is created in psf.getFourierTransform - this should be done in batch on GPU
        # resulting in P already being a CuPy array
        #P shape (Nimages, nw, nv)
        P = img_params.P[:,cp.newaxis,:,:]
        #print ("FSUM", Fsum.shape, "P", P.shape)
        t1 = time.time()
        G = cp.fft.irfft2(Fsum*P)
        add_to_timer(5, time.time()-t1)
        t1 = time.time()
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #cp.savetxt('gg.txt', G.ravel())
        add_to_timer(6, time.time()-t1)
        del Fsum
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
            print ("VARCOPY", varcopy.shape, psfmog.var.shape)
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
            assert(np.all(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] == conv_mog.mean))
            means = conv_mog.mean[:, 0, 0, :].copy()
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
                means[i,0] -= deriv.dx
                means[i,1] -= deriv.dy

            # (13,4,1) = (nimages, nderivs, nmog)
            #print('scale:', scale.shape)
            #print('iv shapes:', iv0.shape, iv1.shape, iv2.shape)

            # (13,4,64,64) = (nimages, nderivs, ny,nx)
            #print('G:', G.shape)

            # The distsq array is going to be nimages x nderivs x nmog x ny=64 x nx=64

            # distsq = (iv0[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dx[:,:,cp.newaxis,:] +
            #           iv1[:,:,:,cp.newaxis] * dx[:,:,cp.newaxis,:] * dy[:,:,:,cp.newaxis] +
            #           iv2[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis] * dy[:,:,:,cp.newaxis])

            n = cp.newaxis
            distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2 +
                      iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]) +
                      iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # t1 = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2)
            # print('t1', t1.shape)
            # t3 = (iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # print('t3', t3.shape)
            # t2 = (iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]))
            # print('t2', t2.shape)
            # Sum over the nmog
            mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(-0.5*distsq), axis=2)
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
        roimask = cp.zeros((img_params.Nimages, mod0.shape[1], mod0.shape[2]))
        #print ("ROI", roi)
        for i, ri in enumerate(roi): 
            roimask[i, ri[1]:ri[1]+ri[3], ri[0]:ri[0]+ri[2]] = 1
        ###############

        # Shift this initial model image to get X,Y pixel derivatives
        dx = cp.zeros_like(mod0)
        # dx is of shape (Nimages, nw, nv)
        # X derivative -- difference between shifted-left and shifted-right arrays
        dx[:,:,1:-1] = mod0[:,:, 2:] - mod0[:,:, :-2]
        # Y derivative -- difference between shifted-down and shifted-up arrays
        dy = cp.zeros_like(mod0)
        # dy is of shape (Nimages, nw, nv)
        dy[:,1:-1, :] = mod0[:,2:, :] - mod0[:,:-2, :]
        # Push through local WCS transformation to get to RA,Dec param derivatives
        assert(cdi.shape == (img_params.Nimages,2,2))

        ##Multiply everything by roimask 02-13-25
        dx *= roimask
        dy *= roimask
        mod0 *= roimask
        #Need to reshape for G
        G *= roimask[:,None,:,:]

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

        #A[:Npix, i + 2] = counts / stepsizes[i] * (Gi[i,:,:] - mod0).ravel()
        stepsizes = img_params.steps
        A[:,:Npix, 3:] = img_params.counts[:,cp.newaxis, cp.newaxis] / stepsizes[:,cp.newaxis,1:] * cp.moveaxis((G[:,1:,:,:] - mod0[:,cp.newaxis,:,:]), 1, -1).reshape((img_params.Nimages, Npix, Nd-1))

        #A[:Npix,:] *= ie.ravel()[:, cp.newaxis]
        A[:,:Npix,:] *= img_params.ie.reshape((img_params.Nimages, Npix))[:,:,cp.newaxis]

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))

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
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                for rij,vij,bij in zip(ri, vi, bi):
                    A[:,Npix + rij, ci] = vij
                    B[:,Npix + rij] += bij
        else:
            print ("NO PRIORS")

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)

        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        A /= colscales[:,cp.newaxis, :]

        # Solve the least-squares problem!
        #X,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        A_T_dot_A = cp.einsum("...ji,...jk", A, A)
        A_T_dot_B = cp.einsum("...ji,...j", A, B)
        X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
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
        X /= colscales
        # del A, B
        #Have to corectly make Xic a list of tuples
        for i in range(img_params.Nimages):
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
        add_to_timer(7, time.time()-t1)
        print_timer()
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
        t2 = time.time()
        add_to_t2(1, 0)
        t1 = time.time()
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
        add_to_timer(5, time.time()-t1)
        t1 = time.time()
        #Fsum shape (Nimages, maxNd, nw, nv)
        #print ("Fsum", Fsum.shape, img_params.Nimages, img_params.maxNd, w.shape, v.shape)
        # P is created in psf.getFourierTransform - this should be done in batch on GPU
        # resulting in P already being a CuPy array
        #P shape (Nimages, nw, nv)
        P = img_params.P[:,cp.newaxis,:,:]
        #print ("FSUM", Fsum.shape, "P", P.shape)
        G = cp.fft.irfft2(Fsum*P)
        add_to_timer(6, time.time()-t1)
        t1 = time.time()
        #Do Lanczos shift
        G = lanczos_shift_image_batch_gpu(G, img_params.mux, img_params.muy)
        #cp.savetxt('vgg.txt', G.ravel())
        add_to_timer(7, time.time()-t1)
        t1 = time.time()
        del Fsum
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
            assert(np.all(conv_mog.mean[:, 0, 0, :][:,cp.newaxis,cp.newaxis,:] == conv_mog.mean))
            means = conv_mog.mean[:, 0, 0, :].copy()
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

            means[:,0] -= img_params.dx
            means[:,1] -= img_params.dy

            # (13,4,1) = (nimages, nderivs, nmog)
            #print('scale:', scale.shape)
            #print('iv shapes:', iv0.shape, iv1.shape, iv2.shape)

            # (13,4,64,64) = (nimages, nderivs, ny,nx)
            #print('G:', G.shape)

            # The distsq array is going to be nimages x nderivs x nmog x ny=64 x nx=64
            n = cp.newaxis
            distsq = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2 +
                      iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]) +
                      iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # t1 = (iv0[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:])**2)
            # print('t1', t1.shape)
            # t3 = (iv2[:,:,:,n,n] * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n])**2)
            # print('t3', t3.shape)
            # t2 = (iv1[:,:,:,n,n] * (xx[n,n,n,n,:] - means[:,n,n,n,0][:,:,:,n,:]) * (yy[n,n,n,:,n] - means[:,n,n,n,1][:,:,:,:,n]))
            # print('t2', t2.shape)
            # Sum over the nmog
            mog_g = cp.sum(scale[:,:,:,cp.newaxis,cp.newaxis] * cp.exp(-0.5*distsq), axis=2)
            #cp.savetxt('vgmogpatch.txt',mog_g.ravel())
            G += mog_g

        #Do no use roi since images are padded to be (mh, mw)
        use_roi = False

        add_to_timer(8, time.time()-t1)
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
        #print (img_params.roi.shape)
        roi = img_params.roi
        roimask = cp.zeros((img_params.Nimages, mod0.shape[1], mod0.shape[2]))
        for i, ri in enumerate(roi):
            roimask[i, ri[1]:ri[1]+ri[3], ri[0]:ri[0]+ri[2]] = 1
        #roimask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1
        ###############

        # Shift this initial model image to get X,Y pixel derivatives
        dx = cp.zeros_like(mod0)
        # dx is of shape (Nimages, nw, nv)
        # X derivative -- difference between shifted-left and shifted-right arrays
        dx[:,:,1:-1] = mod0[:,:, 2:] - mod0[:,:, :-2]
        # Y derivative -- difference between shifted-down and shifted-up arrays
        dy = cp.zeros_like(mod0)
        # dy is of shape (Nimages, nw, nv)
        dy[:,1:-1, :] = mod0[:,2:, :] - mod0[:,:-2, :]
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

        B = cp.zeros((img_params.Nimages, Npix + Npriors), cp.float32)
        B[:,:Npix] = ((img_params.pix - (img_params.counts[:,cp.newaxis, cp.newaxis]*mod0 + img_params.sky[:, cp.newaxis, cp.newaxis])) * img_params.ie).reshape((img_params.Nimages, Npix))
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

        if priorVals is not None:
            print ("Using PRIORS")
            rA, cA, vA, pb, mub = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                for rij,vij,bij in zip(ri, vi, bi):
                    A[:,Npix + rij, ci] = vij
                    B[:,Npix + rij] += bij
        else:
            print ("NO PRIORS")

        # Compute the covariance matrix
        Xicov = cp.matmul(A.swapaxes(-1,-2), A)


        # Pre-scale the columns of A
        colscales = cp.sqrt(cp.diagonal(Xicov, axis1=1, axis2=2))
        A /= colscales[:,cp.newaxis, :]

        # Solve the least-squares problem!
        #X,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        A_T_dot_A = cp.einsum("...ji,...jk", A, A)
        A_T_dot_B = cp.einsum("...ji,...j", A, B)
        X = cp.linalg.solve(A_T_dot_A, A_T_dot_B)
        #X = cp.einsum("ijk,ik->ij", cp.linalg.pinv(A), B)

        # Undo pre-scaling
        X /= colscales
        # del A, B
        #Have to corectly make Xic a list of tuples
        #for i in range(img_params.Nimages):
        #    Xic.append((X[i], Xicov[i]))
        ##Change return args to keep as cupy arrays
        Xic = (X, Xicov)
        add_to_timer(9, time.time()-t1)
        add_to_t2(3, time.time()-t2)
        return Xic


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

def add_to_timer(i, tm):
    global tx
    tx[i] += tm

def print_timer():
    global tx
    print ("TX:", tx)

def add_to_t2(i, tm):
    global ty, tc
    ty[i] += tm
    tc[i] += 1

