'''
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`engine.py`
===========

Core image modeling and fitting.
'''
from __future__ import print_function
import logging

import numpy as np

from tractor.utils import MultiParams, _isint, get_class_from_name
from tractor.patch import Patch, ModelMask
from tractor.image import Image
from tractor.utils import savetxt_cpu_append

logger = logging.getLogger('tractor.engine')
def logverb(*args):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(' '.join(map(str, args)))
def logmsg(*args):
    logger.info(' '.join(map(str, args)))
def isverbose():
    return logger.isEnabledFor(logging.DEBUG)

def set_fp_err():
    '''Cause all floating-point errors to raise exceptions.
    Returns the current error state so you can revert via:

        olderr = set_fp_err()
        # do stuff
        np.seterr(**olderr)
    '''
    return np.seterr(all='raise')


class Catalog(MultiParams):
    '''
    A list of Source objects.  This class allows the Tractor to treat
    a set of astronomical sources as a single object with a bunch of
    parameters.  Most of the functionality comes from the base class.


    Constructor syntax:

    cat = Catalog(src1, src2, src3)

    so if you have a list of sources,

    srcs = [src1, src2, src3]
    cat = Catalog(*srcs)

    '''
    deepcopy = MultiParams.copy

    def __str__(self):
        return ('Catalog: %i sources, %i parameters' %
                (len(self), self.numberOfParams()))

    def printLong(self):
        print('Catalog with %i sources:' % len(self))
        for i, x in enumerate(self):
            print('  %i:' % i, x)

    def getThawedSources(self):
        return self._getActiveSubs()

    def getFrozenSources(self):
        return self._getInactiveSubs()

    def getNamedParamName(self, j):
        return 'source%i' % j


class Images(MultiParams):
    """
    This is a class for holding a list of `Image` objects, each which
    contains data and metadata.  This class allows the Tractor to
    treat a list of `Image`s as a single object that has a set of
    parameters.  Basically all the functionality comes from the base
    class.
    """

    def getNamedParamName(self, j):
        return 'image%i' % j


class OptResult():
    # quack
    pass


class Tractor(MultiParams):
    '''
    Heavy farm machinery.

    As you might guess from the name, this is the main class of the
    Tractor framework.  A Tractor has a set of Images and a set of
    Sources, and has methods to optimize the parameters of those
    Images and Sources.

    '''
    @staticmethod
    def getName():
        return 'Tractor'

    @staticmethod
    def getNamedParams():
        return dict(images=0, catalog=1)

    def __init__(self, images=None, catalog=None, optimizer=None,
                 model_kwargs=None):
        '''
        - `images:` list of Image objects (data)
        - `catalog:` list of Source objects
        '''
        if images is None:
            images = []
        if catalog is None:
            catalog = []
        if not isinstance(images, Images):
            images = Images(*images)
        if not isinstance(catalog, Catalog):
            catalog = Catalog(*catalog)
        super(Tractor, self).__init__(images, catalog)
        self.modtype = np.float32
        self.modelMasks = None
        self.expectModelMasks = False
        if optimizer is None:
            from .lsqr_optimizer import LsqrOptimizer
            self.optimizer = LsqrOptimizer()
        else:
            self.optimizer = optimizer

        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs

    def __str__(self):
        s = ('%s with %i sources and %i images' % (
            self.getName(), len(self.catalog), len(self.images)))
        names = []
        for im in self.images:
            if im.name is None:
                names.append('[unnamed]')
            else:
                names.append(im.name)
        s += ' (' + ', '.join(names) + ')'
        return s

    # For use from emcee
    def __call__(self, X):
        self.setParams(X)
        return self.getLogProb()

    # For pickling
    def __getstate__(self):
        version = 1
        S = (version, self.getImages(), self.getCatalog(), self.liquid,
             self.modtype, self.modelMasks, self.expectModelMasks,
             self.optimizer)
        return S

    def __setstate__(self, state):
        if len(state) == 6:
            # "backwards compat"
            (images, catalog, self.liquid, self.modtype, self.modelMasks,
             self.expectModelMasks) = state
            from .lsqr_optimizer import LsqrOptimizer
            self.optimizer = LsqrOptimizer()
        elif len(state) == 8:
            (ver, images, catalog, self.liquid, self.modtype, self.modelMasks,
             self.expectModelMasks, self.optimizer) = state
        self.subs = [images, catalog]

    def getNImages(self):
        return len(self.images)

    def getImage(self, imgi):
        return self.images[imgi]

    def getImages(self):
        return self.images

    def getCatalog(self):
        return self.catalog

    def setCatalog(self, srcs):
        # FIXME -- ensure that "srcs" is a Catalog?  Or duck-type it?
        self.catalog = srcs

    def setImages(self, ims):
        self.images = ims

    def addImage(self, img):
        self.images.append(img)

    def addSource(self, src):
        self.catalog.append(src)

    def addSources(self, srcs):
        self.catalog.extend(srcs)

    def removeSource(self, src):
        self.catalog.remove(src)

    def optimize_forced_photometry(self, **kwargs):
        '''
        Returns an "OptResult" duck with fields:

        .ims0, .ims1         (if wantims=True)
        .IV                  (if variance=True)
        .fitstats            (if fitstats=True)

        ims0, ims1:
        [ (img_data, mod, ie, chi, roi), ... ]


        ASSUMES linear brightnesses!

        ASSUMES all source parameters except brightness are frozen.

        If sky=False,
        ASSUMES image parameters are frozen.
        If sky=True,
        ASSUMES only the sky parameters are unfrozen

        ASSUMES the PSF and Sky models are position-independent!!

        PRIORS probably don't work because we don't setParams() when
        evaluating likelihood or prior!
        '''
        kw = self.model_kwargs.copy()
        kw.update(kwargs)
        return self.optimizer.forced_photometry(self, **kw)

    # alphas=None, damp=0, priors=True, scale_columns=True,
    # shared_params=True, variance=False, just_variance=False):
    def optimize(self, **kwargs):
        '''
        Performs *one step* of optimization.

        (Exactly what that entails depends on the optimizer; by
        default (LsqrOptimizer) it means one linearized least-squares
        + line search iteration.)

        Returns (delta-logprob, parameter update X, alpha stepsize)

        If variance=True,

        Returns (delta-logprob, parameter update X, alpha stepsize, variance)

        If just_variance=True,
        Returns variance.

        '''
        '''
        If rois is not None, it must be a list of [x0,x1,y0,y1] the
        same length as the number of images, giving the ROI in which
        the chi value (and derivatives) will be evaluated.
        '''
        kw = self.model_kwargs.copy()
        kw.update(kwargs)
        return self.optimizer.optimize(self, **kw)

    def optimize_loop(self, **kwargs):
        '''
        Performs multiple steps of optimization until convergence.

        Returns a dict of results (exact contents varying by optimizer).
        '''
        kw = self.model_kwargs.copy()
        kw.update(kwargs)
        return self.optimizer.optimize_loop(self, **kw)

    def getDerivs(self, **kwargs):
        '''
        Computes model-image derivatives for each parameter.

        Returns a nested list of tuples:

        allderivs: [
           (param0:)  [  (deriv, img), (deriv, img), ... ],
           (param1:)  [],
           (param2:)  [  (deriv, img), ],
        ]

        Where the *derivs* are *Patch* objects and *imgs* are *Image*
        objects.
        '''
        allderivs = []

        if self.isParamFrozen('catalog'):
            srcs = []
        else:
            srcs = list(self.catalog.getThawedSources())

        allsrcs = self.catalog

        kw = self.model_kwargs.copy()
        kw.update(kwargs)

        if not self.isParamFrozen('images'):
            for i in self.images.getThawedParamIndices():
                img = self.images[i]
                derivs = img.getParamDerivatives(self, allsrcs, **kw)
                mod0 = None
                for di, deriv in enumerate(derivs):
                    if deriv is False:
                        if mod0 is None:
                            mod0 = self.getModelImage(img, **kwargs)
                            p0 = img.getParams()
                            stepsizes = img.getStepSizes()
                            paramnames = img.getParamNames()
                        oldval = img.setParam(di, p0[di] + stepsizes[di])
                        mod = self.getModelImage(img, **kwargs)
                        img.setParam(di, oldval)
                        deriv = Patch(0, 0, (mod - mod0) / stepsizes[di])
                        deriv.name = 'd(im%i)/d(%s)' % (i, paramnames[di])
                    allderivs.append([(deriv, img)])
                del mod0

        for src in srcs:
            srcderivs = [[] for i in range(src.numberOfParams())]
            for img in self.images:
                derivs = self._getSourceDerivatives(src, img, **kwargs)
                for k, deriv in enumerate(derivs):
                    if deriv is None:
                        continue
                    srcderivs[k].append((deriv, img))
            allderivs.extend(srcderivs)
        #print('allderivs:', len(allderivs))
        #print('N params:', self.numberOfParams())

        assert(len(allderivs) == self.numberOfParams())
        return allderivs

    def setModelMasks(self, masks, assumeMasks=True):
        '''
        A "model mask" is used to define the pixels that are evaluated
        when computing the model patch for a source in an image.  This
        allows for consistent computation of derivatives and
        optimization, without introducing errors due to approximating
        the profiles differently given different parameter settings.

        *masks*: if None, this masking is disabled, and normal
        approximation rules apply.

        Otherwise, *masks* must be a list, with length equal to the
        number of images.  Each list element must be a dictionary with
        Source objects for keys and ModelMask objects for values.
        Sources that do not touch the image should not exist in the
        dictionary; all the ModelMask objects should be non-None and
        non-empty.
        '''
        self.modelMasks = masks
        assert((masks is None) or (len(masks) == len(self.images)))
        self.expectModelMasks = (masks is not None) and assumeMasks

    def _getModelMaskByIdx(self, idx, src):
        if self.modelMasks is None:
            return None
        try:
            return self.modelMasks[idx][src]
        except KeyError:
            return None

    def _getModelMaskFor(self, image, src):
        if self.modelMasks is None:
            return None
        i = self.images.index(image)
        try:
            return self.modelMasks[i][src]
        except KeyError:
            return None

    def _checkModelMask(self, patch, mask):
        if self.expectModelMasks:
            if patch is not None:
                assert(mask is not None)

        if patch is not None and mask is not None:
            # not strictly required?  but a good idea!
            assert(patch.patch.shape == mask.patch.shape)

        if patch is not None and mask is not None and patch.patch is not None:
            nonzero = Patch(patch.x0, patch.y0, patch.patch != 0)
            #print('nonzero type:', nonzero.patch.dtype)
            unmasked = Patch(mask.x0, mask.y0, np.logical_not(mask.mask))
            #print('unmasked type:', unmasked.patch.dtype)
            bad = nonzero.performArithmetic(unmasked, '__iand__', otype=bool)
            assert(np.all(bad.patch == False))

    def _getSourceDerivatives(self, src, img, **kwargs):
        mask = self._getModelMaskFor(img, src)

        # HACK! -- assume no modelMask -> no overlap
        if self.expectModelMasks and mask is None:
            return [None] * src.numberOfParams()
        derivs = src.getParamDerivatives(img, modelMask=mask, **kwargs)

        # HACK -- auto-add?
        # if self.expectModelMasks:
        #     for d in derivs:
        #         if d is not None and mask is None:
        #             # add to modelMasks
        #             i = self.images.index(img)
        #             # set 'mask' so the assertion in _checkModelMask doesn't fire
        #             mask = Patch(d.x0, d.y0, d.patch != 0)
        #             self.modelMasks[i][src] = mask

        # HACK -- check 'em
        # for d in derivs:
        #     if d is not None:
        #         self._checkModelMask(d, mask)

        return derivs

    def getModelPatch(self, img, src, **kwargs):
        mask = self._getModelMaskFor(img, src)
        # HACK -- assume no mask -> no overlap
        if self.expectModelMasks and mask is None:
            return None
        kw = self.model_kwargs.copy()
        kw.update(kwargs)
        mod = src.getModelPatch(img, modelMask=mask, **kw)
        return mod

    def getModelImage(self, img, srcs=None, sky=True, minsb=None, **kwargs):
        '''
        Create a model image for the given "tractor image", including
        the sky level.  If "srcs" is specified (a list of sources),
        then only those sources will be rendered into the image.
        Otherwise, the whole catalog will be.
        '''
        if _isint(img):
            img = self.getImage(img)
        mod = np.zeros(img.getModelShape(), self.modtype)
        if sky:
            img.getSky().addTo(mod)
        if srcs is None:
            srcs = self.catalog
        for src in srcs:
            if src is None:
                continue
            patch = self.getModelPatch(img, src, minsb=minsb, **kwargs)
            if patch is None:
                continue
            patch.addTo(mod)
        return mod

    def getModelImages(self, **kwargs):
        for img in self.images:
            yield self.getModelImage(img, **kwargs)

    def getChiImages(self, **kwargs):
        for img in self.images:
            yield self.getChiImage(img=img, **kwargs)

    def getChiImage(self, imgi=-1, img=None, srcs=None, minsb=0., **kwargs):
        if img is None:
            img = self.getImage(imgi)
        mod = self.getModelImage(img, srcs=srcs, minsb=minsb, **kwargs)
        chi = (img.getImage() - mod) * img.getInvError()
        #savetxt_cpu_append('cmod.txt', mod)
        #savetxt_cpu_append('cie.txt', img.getInvError())
        #savetxt_cpu_append('cpix.txt', img.getImage())
        if not np.all(np.isfinite(chi)):
            print('Chi not finite')
            print('Image finite?', np.all(np.isfinite(img.getImage())))
            print('Mod finite?', np.all(np.isfinite(mod)))
            print('InvErr finite?', np.all(np.isfinite(img.getInvError())))
            print('Current thawed parameters:')
            self.printThawedParams()
            print('Current sources:')
            for src in self.getCatalog():
                print('  ', src)
            print('Image:', img)
            print('sky:', img.getSky())
            print('psf:', img.getPsf())
        return chi

    def getLogLikelihood(self, **kwargs):
        chisq = 0.
        for i, chi in enumerate(self.getChiImages(**kwargs)):
            chisq += (chi.astype(float) ** 2).sum()
        return -0.5 * chisq

    def getLogProb(self, **kwargs):
        '''
        Return the posterior log PDF, evaluated at the current parameters.
        '''
        lnprior = self.getLogPrior()
        if lnprior == -np.inf:
            return lnprior
        lnl = self.getLogLikelihood(**kwargs)
        lnp = lnprior + lnl
        if np.isnan(lnp):
            print('Tractor.getLogProb() returning NaN.')
            print('Params:')
            self.printThawedParams()
            print('log likelihood:', lnl)
            print('log prior:', lnprior)
            return -np.inf
        return lnp
