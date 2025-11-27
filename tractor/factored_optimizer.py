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

    def one_image_update(self, tr, max_size=0, *kwargs):
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
        return x, icov

    def all_image_updates(self, tr, priors=False, *kwargs):
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
            x,x_icov = r
            max_size = max(max_size, len(x))
            #print('FO: X', x, 'x_icov', x_icov)
            img_opts.append((x,x_icov))
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

        # FIXME
        img_opts = self.all_image_updates(tr, **kwargs)
        if len(img_opts) == 0:
            if x_imgs is not None:
                return x_imgs
            return None

        # Compute inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        for x,ic in img_opts:
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

        x,_,_,_ = np.linalg.lstsq(icsum, xicsum, rcond=None)
        if x_imgs is not None:
            x = np.append(x_imgs, x)
        if image_thawed:
            tr.thawParam('images')
        return x

from tractor.smarter_dense_optimizer import SmarterDenseOptimizer

class FactoredDenseOptimizer(FactoredOptimizer, SmarterDenseOptimizer):
    pass
