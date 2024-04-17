import sys
from tractor.dense_optimizer import ConstrainedDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF
from tractor import mixture_profiles as mp
from tractor.psf import lanczos_shift_image
from astrometry.util.miscutils import get_overlapping_region
import numpy as np

image_counter = 0

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

        if False:
            print('Got A matrix:', A.shape)
            global image_counter
            n,m = A.shape
            for i in range(m):
                plt.clf()
                plt.imshow(A[:,i].reshape((50,50)), interpolation='nearest', origin='lower')
                plt.savefig('orig-img%i-d%i.png' % (image_counter, i))
            image_counter += 1


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

        assert(src.isParamThawed('pos'))
        # TODO -- ASSERT no priors!

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        # WCS inv(CD) matrix
        img_cdi = [tim.getWcs().cdInverseAtPosition(src.getPosition(), src=src)
                   for tim in tr.images]
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]

        # (x0,x1,y0,y1) in image coordinates
        extents = [mm.extent for mm in masks]

        inner_real_nsigma = 3.
        outer_real_nsigma = 4.

        imgs = []

        #halfsizes = []
        for mm,(px,py),(x0,x1,y0,y1),psf,pix,ie,counts,cdi,tim in zip(
                masks, pxy, extents, psfs, img_pix, img_ie, img_counts, img_cdi, tr.images):

            mmpix = pix[mm.y0:mm.y1, mm.x0:mm.x1]
            mmie =   ie[mm.y0:mm.y1, mm.x0:mm.x1]

            psfH,psfW = psf.shape
            halfsize = max([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py,
                            psfH//2, psfW//2])
            #halfsizes.append()
            # PSF Fourier transforms
            P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)
            mh,mw = mm.shape
            #print('PSF fourier transform size:', pW,pH)
            #print('P', P.shape)
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

            # print('px,py', px,py)
            # print('cx,cy', cx,cy)
            # print('dx,dy', dx,dy)
            # print('mux,muy', mux,muy)
            # print('sx,sy', sx,sy)
            
            # Compute the mixture-of-Gaussian components for this galaxy model
            # (at its current parameter values)
            amix = src._getShearedProfile(tim, px, py)

            # Get derivatives for each galaxy parameter.

            # Position derivatives -- get initial model, then build patchdx, patchdy
            # patches from it, then use
            # wcs.pixelDerivsToPositionDerivs(pos, src, counts0, patch0,
            #                                 patchdx, patchdy))
            # Brightness derivative(s) -- see galaxy.py:186

            amixes = src.getDerivativeShearedProfiles(tim, px, py)
            amixes = [('current', amix, 0.)] + amixes

            #print('Sheared profiles for derivatives:', [(n,am.var.ravel()) for n,am,_ in amixes])

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

            imgs.append((img_derivs, mmpix, mmie, v, w, P, mux, muy, sx, sy, mh, mw,
                         counts, cdi))

        #print(len(imgs), 'images to process, with a total of', np.sum([len(x[0]) for x in imgs]), 'derivatives')

        Xic = []
        
        for img_i, (img_derivs, pix, ie, v, w, P, mux, muy, sx, sy, mw, mh, counts,cdi) in enumerate(imgs):
            mod0 = None
            A = np.zeros((mh*mw, len(img_derivs)+2), np.float32)

            for ifft, (name, step, mogs, fftmix) in enumerate(img_derivs):
                assert(mogs is None)
                Fsum = fftmix.getFourierTransform(v, w, zero_mean=True)
                G = np.fft.irfftn(Fsum * P)
                G = G.astype(np.float32)
                if mux != 0.0 or muy != 0.0:
                    lanczos_shift_image(G, mux, muy, inplace=True)

                # This is ugly.... we *could* embed the image pixels and inverse-errors
                # in a postage stamp the size of the FFT patch we're using
                    
                # Cut out the portion of the Fourier-transformed result that we
                # care about
                if sx != 0 or sy != 0:
                    gh,gw = G.shape
                    if sx <= 0 and sy <= 0:
                        G = G[-sy:, -sx:]
                    else:
                        # Yuck... can we avoid this in practice??
                        yi, yo = get_overlapping_region(-sy, -sy + mh - 1, 0, gh - 1)
                        xi, xo = get_overlapping_region(-sx, -sx + mw - 1, 0, gw - 1)
                        shG = np.zeros((mh, mw), G.dtype)
                        shG[yo, xo] = G[yi, xi]
                        G = shG
                gh,gw = G.shape
                if gh > mh or gw > mw:
                    G = G[:mh, :mw]
                assert(G.shape == (mh,mw))

                # First set of params = current galaxy model
                if ifft == 0:
                    mod0 = G
                    # Shift to get X,Y pixel derivatives
                    dx = np.zeros_like(G)
                    # X derivative
                    dx[:,1:-1] = G[:, 2:] - G[:, :-2]
                    # Y derivative
                    dy = np.zeros_like(G)
                    dy[1:-1, :] = G[2:, :] - G[:-2, :]
                    # Push through local WCS transformation to get to param derivs
                    assert(cdi.shape == (2,2))
                    #print('CD-inverse:', cdi)
                    i = 0
                    for i in range(2):
                        # divide by 2 because we did +- 1 pixel
                        # negative because we shifted the *image*, which is opposite
                        # from shifting the *model*
                        A[:,i] = -((dx * cdi[0, i] + dy * cdi[1, i]) * counts / 2).ravel()

                    # Flux derivative = current model
                    A[:, 2] = G.ravel()
                else:
                    A[:, ifft + 2] = counts / step * (G - mod0).ravel()

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

            # IE weighting to get to units of chi
            A *= ie.ravel()[:, np.newaxis]

            B = ((pix - counts*mod0) * ie).ravel()

            if False:
                n,m = A.shape
                for i in range(m):
                    plt.clf()
                    plt.imshow(A[:,i].reshape((mh,mw)), interpolation='nearest', origin='lower')
                    plt.savefig('gpu-img%i-d%i.png' % (img_i, i))

            X,_,_,_ = np.linalg.lstsq(A, B, rcond=None)
            Xicov = np.matmul(A.T, A)
            del A
            Xic.append((X, Xicov))

        realX = super().getSingleImageUpdateDirections(tr, **kwargs)

        # for x,realx in zip(Xic, realX):
        #     (x,xic) = x
        #     (realx,realic) = realx
        #     print('     x', x)
        #     print('real x', realx)

        #return realX
        return Xic

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
        ps = pixscales[i] / 3600.
        tan = Tan(ra, dec, float(cx+i), float(cy), c*ps, s*ps, -s*ps, c*ps,
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
    opt2 = GPUFriendlyOptimizer()

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

    #R = tr.optimize_loop(**fit_kwargs)
    #print('Normal fitter: took', R['steps'], 'steps')

    #R2 = tr2.optimize_loop(**fit_kwargs)
    #print('GPU-friendly fitter: took', R2['steps'], 'steps')

    for step in range(20):
        dlnp1,x1,alpha1 = tr.optimize(**fit_kwargs)
        dlnp2,x2,alpha2 = tr2.optimize(**fit_kwargs)
        print('Step', step)
        print('  dlnp1:', dlnp1)
        print('  dlnp2:', dlnp2)
        print('  alpha1:', alpha1)
        print('  alpha2:', alpha2)
        print('  x1:', x1)
        print('  x2:', x2)
        print('  step1:', x1*alpha1)
        print('  step2:', x2*alpha2)
        print('  Source:', src)
        print('  Source2:', src2)
        
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
