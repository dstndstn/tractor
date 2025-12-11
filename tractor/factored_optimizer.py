import sys
import time
import os
import gc

import numpy as np
import scipy
import scipy.fft

from tractor.dense_optimizer import ConstrainedDenseOptimizer

def shownonzero(A):
    if len(A.shape) == 2:
        r,c = A.shape
        for i in range(r):
            print('[ ' + ' '.join('*' if x != 0 else ' ' for x in A[i,:]) + ' ]')
    elif len(A.shape) == 1:
        print('[ ' + ' '.join('*' if x != 0 else ' ' for x in A) + ' ]')

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
        r = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, **kwargs)
        if r is None:
            return None
        #x,A,colscales,B,Ao = r
        x,A,colscales,B = r

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
        #atb = np.matmul(A.T, B)
        #atb = None
        #del A
        #print('norm Ax-B:', np.linalg.norm(np.dot(A, x) - B))
        
        return x, icov, colscales, A, B

    def all_image_updates(self, tr, priors=False, **kwargs):
        from tractor import Images
        img_opts = []
        imgs = tr.images
        mm = tr.modelMasks

        for i,img in enumerate(imgs):
            tr.images = Images(img)
            if mm is not None:
                tr.modelMasks = [mm[i]]
            # Run with PRIORS = FALSE
            r = self.one_image_update(tr, priors=False, **kwargs)
            if r is None:
                continue
            #x,x_icov,colscales,atb = r
            x = r[0]
            img_opts.append(r)#(x,x_icov,colscales))
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
        #for x,ic,colscales,atb in img_opts:
        for x,ic,colscales,A,B in img_opts:
            # print('ic x:', np.dot(ic, x))
            # print('ic:', ic)
            # print('ic x:')
            # shownonzero(np.dot(ic,x))
            # print('ic:')
            # shownonzero(ic)
            #print('atb:', atb)
            #xicsum = xicsum + atb
            xicsum = xicsum + np.dot(ic, x)
            icsum = icsum + ic

        # Add the priors if needed.
        if kwargs.get('priors', False):
            priors_ATA, priors_ATB = self.getPriorsHessianAndGradient(tr)
            # Add the raw priors to the sums
            if priors_ATA.shape == icsum.shape:
                icsum += priors_ATA
                xicsum += priors_ATB
                #print('Priors: xic:', priors_ATB)
                #print('Priors: ic:', priors_ATA)
            elif np.all(priors_ATA == 0) and np.all(priors_ATB == 0):
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=} but priors are zero so ignorning.")
            else:
                print (f"WARNING: Prior shape mismatch {icsum.shape=} {xicsum.shape=} {priors_ATA.shape=} {priors_ATB.shape=}; using CPU mode instead.")
                if image_thawed:
                    tr.thawParam('images')
                return super().getLinearUpdateDirection(tr, **kwargs)

        # cheap preconditioning to reduce the condition number from column scaling
        # just estimate the scale from the sqrt(diagonal)
        scale = np.sqrt(np.diag(icsum))

        #print('FO scales:', scale)

        scale[scale == 0.] = 1.
        icsum /= (scale[:,np.newaxis] * scale[np.newaxis,:])
        xicsum /= scale

        # print('FO Final icsum:')
        # print(icsum)
        # print('FO Final xicsum:')
        # print(xicsum)
        # FIXME -- in theory you could get zeros in other ways, but in practice
        # this is fine.
        # BUT, we already know which parameters are actually active, and we should use
        # that instead!
        if np.any(xicsum == 0):
            Norig = len(xicsum)
            NZ = np.flatnonzero(xicsum != 0)
            icsum = icsum[NZ, :][:, NZ]
            xicsum = xicsum[NZ]
            print('FO: cond', np.linalg.cond(icsum))
            x2,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
            x2 /= scale[NZ]
            x = np.zeros(Norig, np.float32)
            x[NZ] = x2
        else:
            print('FO: cond', np.linalg.cond(icsum))
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
    from tractor import NanoMaggies, LinearPhotoCal
    import os
    import pylab as plt

    def difference(x1, x2):
        #return np.abs(x1 - x2) / np.maximum(1e-16, (np.abs(x1) + np.abs(x2)) / 2.)
        return np.sum(np.abs(x1 - x2) / np.maximum(1e-16, (np.abs(x1) + np.abs(x2)) / 2.))

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
    print('Galaxy shape:', gal.shape)

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
    print('LSQR Update:', up)

    tr.setParams(p0)

    sm_opt = SmarterDenseOptimizer()
    tr.optimizer = sm_opt
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Smarter Update:', up1)

    tr.setParams(p0)

    fac_opt = FactoredDenseOptimizer()
    print('Factored...')
    tr.optimizer = fac_opt
    up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Update:', up2)

    print('Fractional difference in update directions:', difference(up, up2))

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

    orig_shape = gal.shape
    shape = orig_shape
    shape2 = EllipseWithPriors(np.log(shape.re), shape.e1, shape.e2)

    gal.shape = shape2

    p0 = tr.getParams()
    optargs = dict(shared_params=False, priors=True)

    tr.optimizer = orig_opt
    up = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('LSQR Update:', up)

    tr.setParams(p0)
    tr.optimizer = sm_opt
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Smarter Update:', up1)

    tr.setParams(p0)
    tr.optimizer = fac_opt
    up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Update:', up2)

    print('Fractional difference in update directions (LSQR - Fac):', difference(up, up2))

    print('Fractional difference (LSQR - SM):', difference(up, up1))
    print('Fractional difference (SM - Fac):', difference(up1, up2))

    # Give the source fluxes in multiple bands; set each tim's band.
    print()
    print('Multi-band fluxes')
    print()

    tim1.band = 'g'
    tim2.band = 'r'
    tim1.photocal = LinearPhotoCal(1.0, band='g')
    tim2.photocal = LinearPhotoCal(1.0, band='r')

    # -- two images, gr, priors -> ok (1e-5)
    # Galaxy: ExpGalaxy at pixel (50.00, 50.70) with NanoMaggies: g=22.5, r=22.5 and EllipseWithPriors(0.25): log r_e=2.70805, ee1=0.1, ee2=0
    #gal.brightness = NanoMaggies(g=1., r=1.)

    # --> ok
    gal.brightness = NanoMaggies(g=1., r=1., z=1.)

    print('Galaxy:', gal)

    p0 = tr.getParams()

    for priors in [False, True]:
        print()
        print('Priors:', priors)
        print()

        optargs.update(priors=priors)

        tr.setParams(p0)
        tr.optimizer = orig_opt
        up = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
        print('LSQR Update:', up)

        tr.setParams(p0)
        tr.optimizer = sm_opt
        up1 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
        print('Smarter Update:', up1)

        tr.setParams(p0)
        tr.optimizer = fac_opt
        up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)

        print('LSQR Update    :', up)
        print('Smarter Update :', up1)
        print('Factored Update:', up2)

        print('Fractional difference in update directions (LSQR - Fac):', difference(up, up2))
        print('Fractional difference (LSQR - SM):', difference(up, up1))
        print('Fractional difference (Fac - SM):', difference(up1, up2))


    
