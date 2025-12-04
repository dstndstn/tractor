import sys
import time
import os
import gc

import numpy as np
import scipy
import scipy.fft

from tractor.dense_optimizer import ConstrainedDenseOptimizer
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

    def one_image_update(self, tr, max_size=0, **kwargs):
        '''
        Called by all_image_updates, where "tr" has been modified to have only
        one image, this method returns its update direction and inverse-covariance.
        '''
        allderivs = tr.getDerivs()
        r = self.getUpdateDirection(tr, allderivs, get_A_matrix=True,
                                    max_size=max_size, **kwargs)
        if r is None:
            return None
        x,A,colscales,B,Ao = r

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
        return x, icov, colscales

    def all_image_updates(self, tr, priors=False, **kwargs):
        from tractor import Images
        img_opts = []
        imgs = tr.images
        mm = tr.modelMasks

        max_size = 0
        for i,img in enumerate(imgs):
            tr.images = Images(img)
            if mm is not None:
                tr.modelMasks = [mm[i]]
            # Run with PRIORS = FALSE
            r = self.one_image_update(tr, priors=False, max_size=max_size, **kwargs)
            if r is None:
                continue
            x,x_icov,colscales = r
            max_size = max(max_size, len(x))
            img_opts.append((x,x_icov,colscales))
        tr.images = imgs
        tr.modelMasks = mm
        return img_opts

    def getLinearUpdateDirection(self, tr, **kwargs):
        '''
        This is the function in LsqrOptimizer that is being overridden.
        '''

        # We can be fitting for image-based parameters (eg, sky level) and
        # source-based parameters.  If image parameters are being fit, use the
        # base code (eg in lsqr_optimizer.py) to fit those, and prepend them to
        # the source-based parameters computed below.
        x_imgs = None
        image_thawed = tr.isParamThawed('images')
        if image_thawed:
            cat_frozen = tr.isParamFrozen('catalog')
            if not cat_frozen:
                tr.freezeParam('catalog')
            # call superclass...
            x_imgs = super().getLinearUpdateDirection(tr, **kwargs)
            if cat_frozen:
                # no need to do anything beyond the superclass...
                return x_imgs
            else:
                tr.thawParam('catalog')
            # Freeze the images for the rest of the call (until the end...)
            tr.freezeParam('images')

        if len(tr.images) == 0:
            if x_imgs is not None:
                return x_imgs
            return None

        img_opts = self.all_image_updates(tr, **kwargs)
        if len(img_opts) == 0:
            if x_imgs is not None:
                return x_imgs
            return None

        # Compute inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        for x,ic,colscales in img_opts:
            xicsum = xicsum + np.dot(ic, x)
            icsum = icsum + ic

        # Add the priors if needed.
        if kwargs.get('priors', False):
            priors_ATA, priors_ATB = self.getPriorsHessianAndGradient(tr)
            # Add the raw priors to the sums
            if priors_ATA.shape == icsum.shape:
                icsum += priors_ATA
                xicsum += priors_ATB
            elif np.all(priors_ATA == 0) and np.all(priors_ATB == 0):
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=} but priors are zero so ignorning.")
            else:
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=}; using CPU mode instead.")
                if image_thawed:
                    tr.thawParam('images')
                return super().getLinearUpdateDirection(tr, **kwargs)

        #x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
        # cheap preconditioning to reduce the condition number from column scaling
        scale = np.sqrt(np.diag(icsum))
        icsum /= (scale[:,np.newaxis] * scale[np.newaxis,:])
        xicsum /= scale
        x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
        x /= scale

        if x_imgs is not None:
            x = np.append(x_imgs, x)
        if image_thawed:
            tr.thawParam('images')
        return x

from tractor.smarter_dense_optimizer import SmarterDenseOptimizer

class FactoredDenseOptimizer(FactoredOptimizer, SmarterDenseOptimizer):
    pass


if __name__ == '__main__':
    from tractor.galaxy import ExpGalaxy
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor.basics import PixPos, Flux, ConstantSky
    from tractor.psfex import PixelizedPsfEx
    from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
    from tractor import Image, NullWCS, Tractor
    from tractor.utils import _GaussianPriors
    import os
    import pylab as plt
    
    h,w = 100,100
    gal = ExpGalaxy(PixPos(h/2., w/2+.7), Flux(2000.), EllipseE(10., 0.1, 0.4))

    psf = PixelizedPsfEx(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'test',
                                      'psfex-decam-00392360-S31.fits'))
    psf = HybridPixelizedPSF(psf, #cx=w/2., cy=h/2.,
                             gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))
    print('psf', psf)

    sig1 = 1.0
    sig2 = 1.0
    tim1 = Image(np.zeros((h,w), np.float32),
                 inverr=np.ones((h,w), np.float32) / sig1,
                 psf=psf, sky=ConstantSky(0.),
                 wcs=NullWCS(),
                )

    tr = Tractor([tim1], [gal])

    mod = tr.getModelImage(0)

    noisy1 = mod + np.random.normal(scale=sig1, size=(h,w))
    noisy2 = mod + np.random.normal(scale=sig2, size=(h,w))

    tim1.data = noisy1
    tim2 = Image(noisy2,
                 inverr=np.ones((h,w), np.float32) / sig2,
                 psf=psf, sky=ConstantSky(0.),
                 wcs=NullWCS(),
                )

    tr = Tractor([tim1, tim2], [gal])
    tr.freezeParam('images')

    true_params = np.array(tr.getParams())
    
    g = gal.shape.getParams()
    g[0] = 15
    g[2] = 0.
    gal.shape.setParams(g)

    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(noisy1)
    plt.subplot(2,2,2)
    plt.imshow(noisy2)
    plt.subplot(2,2,3)
    mod = tr.getModelImage(0)
    plt.imshow(mod)
    plt.savefig('mod.png')

    p0 = tr.getParams()
    
    print('Opt', tr.optimizer)

    optargs = dict(shared_params=False)

    orig_opt = tr.optimizer

    up = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Update:', up)

    tr.setParams(p0)

    facopt = FactoredDenseOptimizer()
    print('Factored...')
    tr.optimizer = facopt
    up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Update:', up2)

    print('Fractional difference in update directions:', np.sum(np.abs(up - up2) / (np.abs(up) + np.abs(up2)) / 2.))

    print('Optimizing with priors...')
    print('Tractor images:', len(tr.images))

    # From legacypipe, a simplified EllipseESoft object with priors on the ellipticities.
    class EllipseWithPriors(EllipseESoft):
        ellipticityStd = 0.25
        ellipsePriors = None
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.ellipsePriors is None:
                ellipsePriors = _GaussianPriors(None)
                ellipsePriors.add('ee1', 0., self.ellipticityStd,
                                param=EllipseESoft(1.,0.,0.))
                ellipsePriors.add('ee2', 0., self.ellipticityStd,
                                param=EllipseESoft(1.,0.,0.))
                self.__class__.ellipsePriors = ellipsePriors
            self.gpriors = self.ellipsePriors
        @classmethod
        def getName(cls):
            return "EllipseWithPriors(%g)" % cls.ellipticityStd

    shape = gal.shape
    shape2 = EllipseWithPriors(np.log(shape.re), shape.e1, shape.e2)

    gal.shape = shape2

    p0 = tr.getParams()
    optargs = dict(shared_params=False, priors=True)

    tr.optimizer = orig_opt
    up = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Update:', up)

    tr.setParams(p0)
    tr.optimizer = facopt
    up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Update:', up2)

    print('Fractional difference in update directions:', np.sum(np.abs(up - up2) / (np.abs(up) + np.abs(up2)) / 2.))
