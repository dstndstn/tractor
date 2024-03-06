from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF
from tractor import mixture_profiles as mp
import numpy as np

'''
A mixin class for LsqrOptimizer that does the linear update direction step
by factorizing over images -- it solves the linear problem for each image
independently, and then combines those results (via their covariances) into
the overall result.
'''
class FactoredOptimizer(object):

    def getSingleImageUpdateDirection(self, tr, **kwargs):
        allderivs = tr.getDerivs()
        x,A = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, **kwargs)
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
            img_opts.append((x,x_icov))
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def getLinearUpdateDirection(self, tr, **kwargs):
        #print('getLinearUpdateDirection( kwargs=', kwargs, ')')
        img_opts = self.getSingleImageUpdateDirections(tr, **kwargs)
        # ~ inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        for x,ic in img_opts:
            xicsum = xicsum + np.dot(ic, x)
            icsum = icsum + ic
        C = np.linalg.inv(icsum)
        x = np.dot(C, xicsum)
        # print('Total opt:')
        # print(x)
        return x


class FactoredDenseOptimizer(FactoredOptimizer, ConstrainedDenseOptimizer):
    pass



class GPUFriendlyOptimizer(FactoredDenseOptimizer):

    def getSingleImageUpdateDirections(self, tr, **kwargs):

        # Assume we're not fitting any of the image parameters.
        assert(tr.isParamFrozen('images'))
        
        img_pix = [tim.data for tim in tr.images]
        img_ie  = [tim.getInvError() for tim in tr.images]
        # Assume single source
        assert(len(tr.catalog) == 1)
        # Assume galaxy
        src = tr.catalog[0]
        assert(isinstance(src, ProfileGalaxy))
        psfs = [tim.getPsf() for tim in tr.images]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        assert(all([m is not None for m in masks]))

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src) for tim in tr.images]

        # (x0,x1,y0,y1) in image coordinates
        extents = [mm.extent for mm in masks]

        inner_real_nsigma = 3.
        outer_real_nsigma = 4.

        mogs = []
        ffts = []
        
        #halfsizes = []
        for mm,(px,py),(x0,x1,y0,y1),psf,tim in zip(masks, pxy, extents, psfs, tr.images):
            psfH,psfW = psf.shape
            halfsize = max([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2])
            #halfsizes.append()
            # PSF Fourier transforms
            P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)
            #print('PSF fourier transform size:', pW,pH)
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
            assert(np.abs(mux) <= 0.5)
            assert(np.abs(muy) <= 0.5)

            # Compute the mixture-of-Gaussian components for this galaxy model
            # (at its current parameter values)
            amix = src._getShearedProfile(tim, px, py)

            # Get derivatives for each galaxy parameter.
            # pos0 = src.getPosition()
            # (px0, py0) = tim.getWcs().positionToPixel(pos0, src)
            # counts = tim.getPhotoCal().brightnessToCounts(src.brightness)

            # if counts == 0 ......

            # Position derivatives -- get initial model, then build patchdx, patchdy
            # patches from it, then use
            # wcs.pixelDerivsToPositionDerivs(pos, src, counts0, patch0,
            #                                 patchdx, patchdy))

            # Brightness derivative(s) -- see galaxy.py:186

            amixes = src.getDerivativeShearedProfiles(tim, px, py)
            amixes = [('current', amix, 0.)] + amixes

            #print('Sheared profiles for derivatives:', amixes)

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

            fftweights = 1.
            mogweights = 1.
            if np.any(IM):
                if ramp:
                    ns = (pW/2) / np.maximum(1e-6, np.sqrt(vv))
                    mogweights = np.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
                    fftweights = np.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
                    assert(np.all(mogweights > 0.))
                    assert(np.all(mogweights <= 1.))
                    assert(np.all(fftweights > 0.))
                    assert(np.all(fftweights <= 1.))

                for name,mix,step in amixes:
                    mogs.append(mp.MixtureOfGaussians(
                        mix.amp[IM] * mogweights,
                        mix.mean[IM, :] + np.array([px, py])[np.newaxis, :],
                        mix.var[IM, :, :], quick=True))

            if np.any(IF):
                if ramp:
                    amps *= fftweights

                for name,mix,step in amixes:
                    ffts.append(mp.MixtureOfGaussians(
                        mix.amp[IF] * fftweights,
                        mix.mean[IF, :], amix.var[IF, :, :], quick=True))

        print(len(ffts), 'FFT mixtures and', len(mogs), 'MoG mixtures')
            
        return super().getSingleImageUpdateDirections(tr, **kwargs)

if __name__ == '__main__':

    import pylab as plt
    from tractor import Image, PixPos, Flux, Tractor, NullWCS, NCircularGaussianPSF, PointSource
    from tractor import ExpGalaxy, PixelizedPSF, HybridPixelizedPSF, GaussianMixturePSF
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor import ModelMask
    n_ims = 10
    sig1s = [3., 10.] * 5
    psf_sigmas = [2., 1.] * 5
    fluxes = [1000., 1000.] * 5
    shape = [3., 0.5, 0.3] * 5
    H,W = 50,50
    cx,cy = 23,27
    
    tims = []
    for i in range(n_ims):
        #x = np.arange(W)
        #y = np.arange(H)
        #data = np.exp(-0.5 * ((x[np.newaxis,:] - cx)**2 + (y[:,np.newaxis] - cy)**2) /
        #              psf_sigmas[i]**2)
        #data *= fluxes[i] / (2. * np.pi * psf_sigmas[i]**2)
        data = np.random.normal(size=(50,50)) * sig1s[i]

        pW = pH = 63
        pp = np.arange(pW)
        psf_stamp = np.exp(-0.5 * ((pp[np.newaxis,:] - pW//2)**2 + (pp[:,np.newaxis] - pH//2)**2) /
                           psf_sigmas[i]**2) / (2. * np.pi * psf_sigmas[i]**2)
        gpsf = GaussianMixturePSF(np.array([1.]), np.array([[0.,0.]]),
                                  np.eye(2)[np.newaxis,:,:] * psf_sigmas[i]**2)
        pix = PixelizedPSF(psf_stamp)
        psf = HybridPixelizedPSF(pix, gauss=gpsf)
        # psf=NCircularGaussianPSF([psf_sigmas[i]], [1.])
        tims.append(Image(data=data, inverr=np.ones_like(data) / sig1s[i],
                          psf=psf,
                          wcs=NullWCS()))
        true_src = ExpGalaxy(PixPos(cx, cy), Flux(fluxes[i]), EllipseE(*shape))
        tr = Tractor([tims[i]], [true_src])
        true_mod = tr.getModelImage(0)
        data += true_mod

    #src = PointSource(PixPos(W//2, H//2), Flux(100.))
    e = EllipseE(2., 0., 0.)
    src = ExpGalaxy(PixPos(W//2, H//2), Flux(100.), EllipseESoft.fromEllipseE(e))

    #opt = FactoredDenseOptimizer()
    opt = GPUFriendlyOptimizer()

    opt2 = ConstrainedDenseOptimizer()

    tr = Tractor(tims, [src], optimizer=opt)
    tr.setModelMasks([{src: ModelMask(0, 0, W, H)} for tim in tims])

    tr2 = Tractor(tims, [src], optimizer=opt2)
    tr.freezeParam('images')
    tr2.freezeParam('images')

    mods = list(tr.getModelImages())

    fit_kwargs = dict(shared_params=False, priors=False)
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **fit_kwargs)
    up2 = tr2.optimizer.getLinearUpdateDirection(tr2, **fit_kwargs)
    print('Update directions:')
    print(up1)
    print(up2)
    
    tr.optimize_loop(**fit_kwargs)
    mods2 = list(tr.getModelImages())

    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods[i], **ima)
    plt.savefig('1.png')

    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods2[i], **ima)
    plt.savefig('2.png')
    
