import sys
from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF
from tractor import mixture_profiles as mp
from tractor.psf import lanczos_shift_image
from astrometry.util.miscutils import get_overlapping_region
import numpy as np
import scipy
import scipy.fft
import time

from data_recorder import DataRecorder

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
        x,A = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, **kwargs)

        if False:
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
            x,x_icov = self.getSingleImageUpdateDirection(tr, **kwargs)
            #print('FO: X', x, 'x_icov', x_icov)
            img_opts.append((x,x_icov))
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def getLinearUpdateDirection(self, tr, **kwargs):
        #print('getLinearUpdateDirection( kwargs=', kwargs, ')')
        img_opts = self.getSingleImageUpdateDirections(tr, **kwargs)
        if len(img_opts) == 1:
            x,ic = img_opts[0]
        else:
            # ~ inverse-covariance-weighted sum of img_opts...
            xicsum = 0
            icsum = 0
            for x,ic in img_opts:
                print('x:', ', '.join(['%.5g' % xx for xx in x]))
                xicsum = xicsum + np.dot(ic, x)
                print('ic:', ic)
                icsum = icsum + ic
            C = np.linalg.inv(icsum)
            x = np.dot(C, xicsum)

        #print('icsum:')
        #print(icsum)
        print('Total opt:')
        print(x)

        return x


class FactoredDenseOptimizer(FactoredOptimizer, ConstrainedDenseOptimizer):
    pass



class GPUFriendlyOptimizer(FactoredDenseOptimizer):

    def getSingleImageUpdateDirections(self, tr, **kwargs):
        if not (tr.isParamFrozen('images') and
                (len(tr.catalog) == 1) and
                isinstance(tr.catalog[0], ProfileGalaxy)):
            p = self.ps
            self.ps = None
            R = super().getSingleImageUpdateDirections(tr, **kwargs)
            self.ps = p
            return R

        print('Using GpuFriendly code')
        print('modelMasks:', tr.modelMasks)
        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
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
        print('Source:', src)

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
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
        img_bands = [bands.index(tim.getPhotoCal().band) for tim in tr.images]
        #print('ibands', img_bands)

        # (x0,x1,y0,y1) in image coordinates
        extents = [mm.extent for mm in masks]

        inner_real_nsigma = 3.
        outer_real_nsigma = 4.

        imgs = []
        for mm,(px,py),(x0,x1,y0,y1),psf,pix,ie,counts,cdi,tim in zip(
                masks, pxy, extents, psfs, img_pix, img_ie, img_counts, img_cdi, tr.images):

            mmpix = pix[mm.y0:mm.y1, mm.x0:mm.x1]
            mmie =   ie[mm.y0:mm.y1, mm.x0:mm.x1]

            psfH,psfW = psf.shape
            halfsize = max([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2])
            # PSF Fourier transforms
            P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)
            mh,mw = mm.shape
            assert(pW % 2 == 0)
            assert(pH % 2 == 0)
            assert(P.shape == (len(w),len(v)))
            del v,w

            # sub-pixel shift we have to do at the end...
            dx = px - cx
            dy = py - cy
            mux = dx - x0
            muy = dy - y0
            sx = int(np.round(mux))
            sy = int(np.round(muy))
            print('GPU: sx,sy', sx,sy)
            # the subpixel portion will be handled with a Lanczos interpolation
            mux -= sx
            muy -= sy
            assert(np.abs(mux) <= 0.5)
            assert(np.abs(muy) <= 0.5)

            # Embed pix and ie in images the same size as pW,pH.
            padpix = np.zeros((pH,pW), np.float32)
            padie  = np.zeros((pH,pW), np.float32)
            assert(sy <= 0 and sx <= 0)
            padpix[-sy: -sy+mh, -sx: -sx+mw] = mmpix
            padie [-sy: -sy+mh, -sx: -sx+mw] = mmie
            roi = (-sx, -sy, mw, mh)
            print('ROI:', roi)
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
            vv = amix.var[:, 0, 0] + amix.var[:, 1, 1]
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
            mogweights = 1.
            fftweights = 1.
            if ramp:
                # ramp
                ns = (pW/2) / np.maximum(1e-6, np.sqrt(vv))
                mogweights = np.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
                fftweights = np.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
                assert(np.all(mogweights > 0.))
                assert(np.all(mogweights <= 1.))
                assert(np.all(fftweights > 0.))
                assert(np.all(fftweights <= 1.))

            img_derivs = []

            for name,mix,step in amixes:
                mogs = None
                ffts = None
                if np.any(IM):
                    mogs = mp.MixtureOfGaussians(
                        mix.amp[IM] * mogweights,
                        mix.mean[IM, :] + np.array([px, py])[np.newaxis, :],
                        mix.var[IM, :, :], quick=True)
                if np.any(IF):
                    ffts = mp.MixtureOfGaussians(
                        mix.amp[IF] * fftweights,
                        mix.mean[IF, :],
                        mix.var[IF, :, :], quick=True)

                img_derivs.append((name, step, mogs, ffts))

            assert(sx == 0 and sy == 0)

            imgs.append((img_derivs, mmpix, mmie, P, mux, muy, mh, mw, counts, cdi, roi))

        #nbands = 1 + max(img_bands)
        nbands = len(bands)

        #print(len(imgs), 'images to process, with a total of', np.sum([len(x[0]) for x in imgs]), 'derivatives')
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

        #Xic = self.computeUpdateDirections(imgs, priorVals, tr)
        Xic = self.computeUpdateDirections(imgs, priorVals, tr, src, **kwargs)

        if nbands > 1:
            full_xic = []
            fullN = tr.numberOfParams()
            for iband,(x,ic) in zip(img_bands, Xic):
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
                
                full_xic.append((x2,ic2))
            Xic = full_xic

        # print('Calling original version...')
        # ps = self.ps
        # self.ps = self.ps_orig
        # sXic = super().getSingleImageUpdateDirections(tr, **kwargs)
        # self.ps = ps

        # for (orig_x,orig_cx),(gpu_x,gpu_cx) in zip(Xic, sXic):
        #     print('Orig x:', orig_x)
        #     print('GPU  x:', gpu_x)

        return Xic

    def computeUpdateDirections(self, imgs, priorVals, tr, src, **kwargs):
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
        #use_roi = True
        Xic = []

        Npriors = 0
        if priorVals is not None:
            rA, cA, vA, pb, mub = priorVals
            Npriors = max(Npriors, max([1+max(r) for r in rA]))
            print('Prior vals:', priorVals)

        for img_i, (img_derivs, pix, ie, P, mux, muy, mw, mh, counts, cdi, roi) in enumerate(imgs):
            assert(pix.shape == (mh,mw))
            # We're going to build a tall matrix A, whose number of
            # rows = number of pixels and cols = number of parameters
            # to update.  We special-case the two spatial derivatives,
            # the rest are in the 'img_derivs' list.

            # number of derivatives
            Nd = len(img_derivs)
            if use_roi:
                (rx0,ry0,rw,rh) = roi
                roi_slice = slice(ry0, ry0+rh), slice(rx0, rx0+rw)
                pix = pix[roi_slice]
                ie = ie[roi_slice]
                Npix = rh*rw
            else:
                Npix = mh*mw
                # HACK -- temp
                (rx0,ry0,rw,rh) = roi
                roi_slice = slice(ry0, ry0+rh), slice(rx0, rx0+rw)
                roi_Npix = rh*rw
                roi_shape = rh,rw
                print('roi_shape:', roi_shape)

            A = np.zeros((Npix + Npriors, Nd+2), np.float32)

            Nw,Nv = P.shape
            mod0 = None
            mix0 = img_derivs[0][3]
            # number of components in the Gaussian mixture
            Nc = len(mix0.amp)
            fftamps = np.zeros((Nd, Nc), np.float32)
            fftvars = np.zeros((Nd, Nc, 3), np.float32)
            for i,(_,_,_,mix) in enumerate(img_derivs):
                fftamps[i,:] = mix.amp
                fftvars[i,:,0] = mix.var[:,0,0]
                fftvars[i,:,1] = mix.var[:,0,1]
                fftvars[i,:,2] = mix.var[:,1,1]

            # from mixture_profiles : getFourierTransform2() with zero_mean=True
            Fsums = np.zeros((Nd, Nw, Nv), np.float32)
            w = np.fft.fftfreq(Nw)
            # Note, this "Nw" looks like it might be a bug (should be "Nv"?) but it's not,
            # I promise.  "v" will end up having length "Nv".  eg, Nw=64, Nv=33.
            v = np.fft.rfftfreq(Nw)
            for i in range(Nd):
                for k in range(Nc):
                    amp = fftamps[i,k]
                    a,b,d = fftvars[i, k, :]
                    Fsums[i, :, :] += amp * np.exp(
                        -2. * np.pi**2 *
                        (a * v[np.newaxis, :]**2 +
                         d * w[:, np.newaxis]**2 +
                         2 * b * v[np.newaxis, :] * w[:, np.newaxis]))

            Gs = scipy.fft.irfftn(Fsums * P[np.newaxis,:,:], axes=(1,2))
            print('Gs shape:', Gs.shape) #--> 4,64,64

            if use_roi:
                Gs = Gs[:, roi_slice[0], roi_slice[1]]
                print('Gs shape:', Gs.shape) #--> eg 2,35,33

            del Fsums
            for i in range(Nd):
                # lanczos_shift_image has a Python implementation in psf.py, or
                # there is a C implementation in mp_fourier.i : lanczos_shift_3f
                print('factored_opt Lanczos-shift:', Gs[i,:,:].shape)
                DataRecorder.get().add('lanczos-factored-before', Gs[i,:,:].copy())
                lanczos_shift_image(Gs[i,:,:], mux, muy, inplace=True)
                DataRecorder.get().add('lanczos-factored-after', Gs[i,:,:].copy())

            # The first element in img_derivs is the current galaxy model parameters.
            mod0 = Gs[0,:,:]

            # Shift this initial model image to get X,Y pixel derivatives
            dx = np.zeros_like(mod0)
            dy = np.zeros_like(mod0)
            # X derivative -- difference between shifted-left and shifted-right arrays
            #dx[:,1:-1] = mod0[:, 2:] - mod0[:, :-2]
            # Y derivative -- difference between shifted-down and shifted-up arrays
            #dy[1:-1, :] = mod0[2:, :] - mod0[:-2, :]
            # Omit a one-pixel boundary on all directions in both arrays!
            dx[1:-1,1:-1] = mod0[1:-1, 2:] - mod0[1:-1, :-2]
            dy[1:-1,1:-1] = mod0[2:, 1:-1] - mod0[:-2, 1:-1]
            
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

            stepsizes = [s for _,s,_,_ in img_derivs]
            # The first element is mod0, so the stepped parameters start at 1 here.
            for i in range(1, Nd):
                # For other parameters, compute the numerical derivative.
                # mod0 is the unit-brightness model at the current position
                # Gs[i,*] is the unit-brightness model after stepping the parameter
                # The i+2 here is because the first two params are the spatial derivs
                # (A[:,0] and A[:,1] are filled above)
                # And the next param is the flux deriv
                # (A[:,2] is filled)
                # So the first time through this loop, i=1 and we fill column A[:,3]
                A[:Npix, i + 2] = counts / stepsizes[i] * (Gs[i,:,:] - mod0).ravel()

            del Gs
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
            A[:Npix,:] *= ie.ravel()[:, np.newaxis]
            # The current residuals = the observed image "pix" minus the current model (counts*mod0),
            # weighted by the inverse-errors.
            B = np.append(((pix - counts*mod0) * ie).ravel(),
                          np.zeros(Npriors, np.float32))

            # Append priors
            if priorVals is not None:
                rA, cA, vA, pb, mub = priorVals
                for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                    for rij,vij,bij in zip(ri, vi, bi):
                        A[Npix + rij, ci] = vij
                        B[Npix + rij] += bij

            if False:
                n,m = A.shape
                for i in range(m):
                    plt.clf()
                    plt.imshow(A[:,i].reshape((mh,mw)), interpolation='nearest', origin='lower')
                    plt.savefig('gpu-img%i-d%i.png' % (img_i, i))

            # Compute the covariance matrix
            Xicov = np.matmul(A.T, A)

            # Pre-scale the columns of A
            colscales = np.sqrt(np.diag(Xicov))
            A /= colscales[np.newaxis, :]

            # Solve the least-squares problem!
            X,_,_,_ = np.linalg.lstsq(A, B, rcond=None)

            # (we undo pre-scaling on X, after this plotting...)

            if self.ps is not None:

                def new_img(X):
                    return X[:Npix].reshape(mh,mw)[roi_slice]
            
                cb = True
                import pylab as plt
                plt.clf()
                ima = dict(interpolation='nearest', origin='lower')
                rr,cc = 3,4
                plt.subplot(rr,cc,1)
                plt.imshow(new_img(mod0), **ima)
                if cb:
                    plt.colorbar()
                plt.title('mod0')

                plt.subplot(rr,cc,4)
                plt.imshow(np.log10(np.maximum(1e-6, new_img(mod0))), **ima)
                if cb:
                    plt.colorbar()
                plt.title('log mod0')

                #sh = new_img(mod0).shape
                plt.subplot(rr,cc,2)
                mx = max(np.abs(B))
                imx = ima.copy()
                imx.update(vmin=-mx, vmax=+mx)
                plt.imshow(new_img(B[:Npix]), **imx)
                if cb:
                    plt.colorbar()
                plt.title('B')
                AX = np.dot(A, X)
                plt.subplot(rr,cc,3)
                plt.imshow(new_img(AX[:Npix]), **imx)
                if cb:
                    plt.colorbar()
                plt.title('A X')
                for i in range(min(Nd+2, 8)):
                    plt.subplot(rr,cc,5+i)
                    plt.imshow(new_img(A[:Npix,i]), **ima)
                    if cb:
                        plt.colorbar()
                    if i == 0:
                        plt.title('dx')
                    elif i == 1:
                        plt.title('dy')
                    elif i == 2:
                        plt.title('dflux')
                    else:
                        plt.title(img_derivs[i-2][0])
                plt.suptitle('GPU: image %i/%i' % (img_i+1, len(imgs)))
                self.ps.savefig()

            # Undo pre-scaling
            X /= colscales

            print('ROI slice:', roi_slice)

            # Call orig version
            print('Calling orig version...')# kwargs=', kwargs)
            tims = tr.images
            mms = tr.modelMasks
            from tractor import Images
            from legacypipe.oneblob import _get_subtim
            timshape = tr.images[img_i].shape
            mm = tr.modelMasks[img_i][src]
            print('orig: tim shape', timshape)
            print('orig: mm shape', (mm.h,mm.w))
            print('orig: mm origin', mm.x0, mm.y0)
            mmslice = mm.slice
            orig_Npix = timshape[0]*timshape[1]

            def orig_img(X):
                return X[:orig_Npix].reshape(timshape)[mmslice]
            
            tr.images = Images(tr.images[img_i])
            tr.modelMasks = [tr.modelMasks[img_i]]

            print('Getting orig_mod0...')
            orig_mod0 = tr.getModelImage(0)[mmslice] / counts
            print('Got orig_mod0')
            # #print('model masks:', tr.modelmasks)
            # mm = tr.modelmasks[img_i][src]
            # (x0,x1,y0,y1) = mm.extent
            # print('extent', x0,x1,y0,y1)
            # x0,x1,y0,y1 = int(x0),int(x1),int(y0),int(y1)
            # tr.images = images(_get_subtim(tr.images[img_i], x0,x1,y0,y1))
            # tr.setmodelmasks(None)
            print('Getting orig derivs...')
            allderivs = tr.getDerivs()
            print('allderivs:', allderivs)
            orig_x,orig_A,orig_B = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, get_B_vector=True, **kwargs)
            tr.images = tims
            tr.modelMasks = mms

            orig_cols = []
            nr,nc = orig_A.shape
            for c in range(nc):
                if np.all(orig_A[:,c] == 0):
                    continue
                orig_cols.append(c)
            orig_cols = np.array(orig_cols)

            orig_A = orig_A[:, orig_cols]
            orig_x = orig_x[orig_cols]
            print('cut orig a to', len(orig_cols), 'of', nc, 'columns')

            if self.ps is not None:
                import pylab as plt
                plt.clf()
                ima = dict(interpolation='nearest', origin='lower')
                rr,cc = 3,4

                plt.subplot(rr,cc,1)
                plt.imshow(orig_mod0, **ima)
                plt.title('mod0')
                if cb:
                    plt.colorbar()
                plt.subplot(rr,cc,4)
                plt.imshow(np.log10(np.maximum(1e-6, orig_mod0)), **ima)
                if cb:
                    plt.colorbar()
                plt.title('log mod0')

                sh = timshape
                mx = max(np.abs(B))
                imx = ima.copy()
                imx.update(vmin=-mx, vmax=+mx)
                plt.subplot(rr,cc,2)
                plt.imshow(orig_img(orig_B), **imx)
                if cb:
                    plt.colorbar()
                plt.title('B')
                AX = np.dot(orig_A, orig_x)
                plt.subplot(rr,cc,3)
                plt.imshow(orig_img(AX), **imx)
                if cb:
                    plt.colorbar()
                plt.title('A X')
                ond = orig_A.shape[1]
                for i in range(ond):
                    plt.subplot(rr,cc,5+i)
                    plt.imshow(orig_img(orig_A[:,i]), **ima)
                    if cb:
                        plt.colorbar()

                plt.suptitle('Orig version: image %i/%i' % (img_i+1, len(imgs)))
                self.ps.savefig()

            if self.ps is not None:
                import pylab as plt

                plt.clf()
                plt.subplot(2,4,1)
                ima = dict(interpolation='nearest', origin='lower')
                plt.imshow(mod0, **ima)
                plt.colorbar()
                plt.title('mod0')
                plt.subplot(2,4,2)
                plt.imshow(mod0[roi_slice], **ima)
                plt.title('mod0')
                plt.colorbar()
                plt.subplot(2,4,3)
                plt.imshow(orig_mod0, **ima)
                plt.title('orig model')
                plt.colorbar()
                plt.subplot(2,4,4)
                plt.imshow(orig_mod0 - mod0[roi_slice], **ima)
                plt.colorbar()
                plt.title('diff')

                plt.subplot(2,4,5)
                om = tr.getModelImage(0) / counts
                plt.imshow(om, **ima)
                plt.colorbar()
                plt.title('orig model (full)')

                plt.subplot(2,4,6)
                plt.imshow(np.log10(np.maximum(1e-6, mod0[roi_slice])), **ima)
                plt.title('log mod0')
                plt.colorbar()

                plt.subplot(2,4,7)
                plt.imshow(np.log10(np.maximum(1e-6, orig_mod0)), **ima)
                plt.title('log orig mod0')
                plt.colorbar()

                plt.subplot(2,4,8)
                plt.imshow(np.log10(np.maximum(1e-6, om)), **ima)
                plt.title('log orig mod0 (full)')
                plt.colorbar()

                self.ps.savefig()

                
                plt.clf()
                ima = dict(interpolation='nearest', origin='lower')
                rr,cc = 3,4
                d = orig_mod0 - mod0[roi_slice]
                mx = np.max(np.abs(d))
                plt.subplot(rr,cc,1)
                plt.imshow(d, vmin=-mx, vmax=mx, **ima)
                if cb:
                    plt.colorbar()
                plt.title('mod0')
                #sh = mod0.shape
                sh = roi_shape
                mx = max(np.abs(B))
                imx = ima.copy()
                imx.update(vmin=-mx, vmax=+mx)
                plt.subplot(rr,cc,2)
                oldval = orig_img(orig_B)
                newval = new_img(B)
                d = newval - oldval
                mx = np.max(np.abs(d))
                print('B max diff:', mx, 'rel diff', np.mean(np.abs(d / np.maximum(1e-16, ((np.abs(oldval) + np.abs(newval))/2.)))))
                plt.imshow(d.reshape(sh), vmin=-mx, vmax=mx, **ima)
                if cb:
                    plt.colorbar()
                plt.title('B')
                AX = new_img(np.dot(A, X * colscales))
                oAX = orig_img(np.dot(orig_A, orig_x))
                plt.subplot(rr,cc,3)
                d = AX - oAX
                mx = np.max(np.abs(d))
                print('AX max diff:', mx, 'rel diff', np.mean(np.abs(d / np.maximum(1e-16, ((np.abs(AX) + np.abs(oAX))/2.)))))
                plt.imshow(d.reshape(sh), vmin=-mx, vmax=mx, **ima)
                if cb:
                    plt.colorbar()
                plt.title('A X')
                oNd = orig_A.shape[1]
                for i in range(oNd):
                    plt.subplot(rr,cc,5+i)
                    #print('     A rms:', np.sqrt(np.mean(A[:Npix,i]**2)))
                    #print('orig_A rms:', np.sqrt(np.mean(orig_A[:Npix,i]**2)))
                    #print('colscales:', colscales)
                    oA = orig_img(orig_A[:,i])
                    nA = new_img(A[:,i] * colscales[i])
                    d = nA - oA
                    mx = np.max(np.abs(d))
                    print('A[:,%i] diff:' % i, mx, 'rel diff', np.mean(np.abs(d / np.maximum(1e-16, ((np.abs(nA) + np.abs(oA))/2)))))
                    plt.imshow(d.reshape(sh), vmin=-mx, vmax=mx, **ima)
                    if cb:
                        plt.colorbar()

                plt.suptitle('Difference: image %i/%i' % (img_i+1, len(imgs)))
                self.ps.savefig()

                
            #j = np.flatnonzero(
            j = np.all(orig_A == 0, axis=1)
            from collections import Counter
            print('all zero rows in orig A:', Counter(j))

            j = np.all(A == 0, axis=1)
            print('all zero rows in A:', Counter(j))
            
            print('X     :', X)
            print('orig x:', orig_x)

            print('orig A', orig_A.shape)
            print('A     ', A.shape)

            # for i,(tag,data) in enumerate(DataRecorder.get().all()):
            #     print('saving', tag)
            #     import fitsio
            #     fitsio.write('data-%02i-%s.fits' % (i,tag), data, overwrite=True)
            # 
            # data = DataRecorder.get().all()
            # 
            # _,before_f = data[0]
            # _,before_g = data[4]
            # 
            # _,after_f = data[1]
            # _,after_g = data[5]
            # 
            # for name,(ot,oldval),(nt,newval) in [
            #         ('Before L', data[0], data[4]),
            #         ('After  L', data[1], data[5])]:
            #     d = newval - oldval
            #     print(name, ': max diff', np.max(np.abs(d)), 'cosine dist',
            #           3600. * np.rad2deg(np.arccos(np.sum(oldval * newval) / (np.sqrt(np.sum(oldval**2)) * np.sqrt(np.sum(newval**2))))), 'arcsec')
            #     if self.ps is not None:
            #         plt.clf()
            #         mx = np.max(oldval)
            #         dmx = np.max(np.abs(d))
            #         ima = dict(interpolation='nearest', origin='lower', vmin=0, vmax=mx)
            #         imd = dict(interpolation='nearest', origin='lower',
            #                    vmin=-dmx, vmax=dmx)
            #         plt.subplot(1,3,1)
            #         plt.imshow(oldval, **ima)
            #         if cb:
            #             plt.colorbar()
            #         plt.title(ot)
            #         plt.subplot(1,3,2)
            #         plt.imshow(newval, **ima)
            #         if cb:
            #             plt.colorbar()
            #         plt.title(nt)
            #         plt.subplot(1,3,3)
            #         plt.imshow(d, **imd)
            #         if cb:
            #             plt.colorbar()
            #         plt.title('diff')
            #         plt.suptitle(name)
            #         self.ps.savefig()
            # 
            # import sys
            # sys.exit(0)

            del A,B,mod0
            Xic.append((X, Xicov))
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
