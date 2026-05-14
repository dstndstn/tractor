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

import time
tchi = np.zeros(10)
tmod = np.zeros(6)
teng = np.zeros(6)
tlog = np.zeros(6)

def printTiming():
    print (f'EngTimes: {tchi=} {tmod=} {teng=} {tlog=}')

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

def get_global_extent(tims, ux0, uy0, ux1, uy1):
    """
    Standardizes the image info based on a pre-calculated global box.
    This ensures we don't 'shrink' our view based on masks.
    """
    image_info = []
    for tim in tims:
        img_h, img_w = tim.shape
        # The intersection of the global box and this specific image
        y_start = max(uy0, 0)
        y_end   = min(uy1, img_h)
        x_start = max(ux0, 0)
        x_end   = min(ux1, img_w)

        # We store the absolute image coords (ly) and the stack offsets (gy)
        # th, tw here represent the 'actual' overlapping area
        th = max(0, y_end - y_start)
        tw = max(0, x_end - x_start)
        image_info.append((x_start, y_start, th, tw))

    return image_info

def get_global_extent_mm(tims, modelMasks=None):
    all_x0, all_x1, all_y0, all_y1 = [], [], [], []
    image_info = []

    if modelMasks is None:
        modelMasks = [None] * len(tims)

    for tim, mask in zip(tims, modelMasks):
        if mask is not None:
            tx0, ty0 = mask.x0, mask.y0
            th, tw = mask.shape
        else:
            # Full image fallback: No x0/y0 on tim, so it's 0,0 relative to itself
            tx0, ty0 = 0, 0
            th, tw = tim.shape

        all_x0.append(tx0)
        all_x1.append(tx0 + tw)
        all_y0.append(ty0)
        all_y1.append(ty0 + th)
        image_info.append((tx0, ty0, th, tw))

    ux0, ux1 = min(all_x0), max(all_x1)
    uy0, uy1 = min(all_y0), max(all_y1)
    ux0 = int(ux0)
    ux1 = int(ux1)
    uy0 = int(uy0)
    uy1 = int(uy1)

    return (ux0, ux1, uy0, uy1), image_info


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
        self.blobid = None
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

    def getImagesGPU(self):
        import cupy as cp
        return [cp.asarray(im) for im in self.images]

    def getCatalog(self):
        return self.catalog

    def setBlobid(self, blobid):
        self.blobid = blobid

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
        #print ("OPTIMIZER = ", self.optimizer, self.optimizer.optimize_loop)
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
        t = time.time()
        allderivs = []

        if self.isParamFrozen('catalog'):
            srcs = []
        else:
            srcs = list(self.catalog.getThawedSources())

        allsrcs = self.catalog

        kw = self.model_kwargs.copy()
        kw.update(kwargs)
        #print ("TEST1")

        if not self.isParamFrozen('images'):
            for i in self.images.getThawedParamIndices():
                img = self.images[i]
                #print ("IMG", i, img)
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
                    #np.savetxt('cpu_d_'+str(di)+'.txt', deriv.patch)
                del mod0

        #print ("TEST2", len(self.images))
        for src in srcs:
            srcderivs = [[] for i in range(src.numberOfParams())]
            for img in self.images:
                #print ("IMG2", img)
                derivs = self._getSourceDerivatives(src, img, **kwargs)
                for k, deriv in enumerate(derivs):
                    #print ("K", k)
                    if deriv is None:
                        #print ("Deriv is None")
                        continue
                    srcderivs[k].append((deriv, img))
                    #np.savetxt('cpu_sd_'+str(k)+'.txt', deriv.patch)
            allderivs.extend(srcderivs)
        #print('allderivs:', len(allderivs))
        #for i in range(len(allderivs)):
        #    print ("I", i, len(allderivs[i]))
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
        #print ("D1", src.getParamDerivatives)
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
        t0 = time.time()
        if _isint(img):
            img = self.getImage(img)
        mod = np.zeros(img.getModelShape(), self.modtype)
        if sky:
            img.getSky().addTo(mod)
            #print ("SKYC", mod.max())
        if srcs is None:
            srcs = self.catalog
        for src in srcs:
            if src is None:
                continue
            t = time.time()
            patch = self.getModelPatch(img, src, minsb=minsb, **kwargs)
            tmod[1] += time.time()-t
            if patch is None:
                continue
            #print (f'{patch.patch.shape=} {patch.patch.max()=} {mod.shape=}')
            patch.addTo(mod)
            #print ("MOD CPU ", mod.max()) 
        tmod[0] += time.time()-t0
        return mod

    def getModelSubImageOneSource(self, img, src=None, sky=True, minsb=None, **kwargs):
        t = time.time()
        if src is None:
            src = self.catalog[0]
        patch = self.getModelPatch(img, src, minsb=minsb, **kwargs)
        if patch is None:
            #teng[4] += time.time()-t
            return None
        if sky:
            img.getSky().addTo(patch.patch)
        #teng[4] += time.time()-t
        return patch.patch

    def getModelImages(self, **kwargs):
        for img in self.images:
            yield self.getModelImage(img, **kwargs)

    def getChiImagesGPU(self, **kwargs):
        for img in self.images:
            yield self.getChiImageGPU(img=img, **kwargs)

    def getChiImages(self, **kwargs):
        for img in self.images:
            yield self.getChiImage(img=img, **kwargs)

    def getChiSquaredsBatch(self, candidate_params, use_gpu=True, **kwargs):
        import cupy as cp
        n_cand = len(candidate_params)
        chisqs = cp.zeros(n_cand, dtype=cp.float64)
        
        # Store current state to restore later
        p0 = self.getParams()
        # Assume we are optimizing the first source for the line search
        src = self.getCatalog()[0]
        
        # Pre-fetch modelMask if provided in kwargs
        modelMask = kwargs.get('modelMask', None)

        for img in self.getImages():
            # Get data and inverse error on GPU
            data_gpu = img.getImage(use_gpu=use_gpu)
            ie_gpu = img.getInvError(use_gpu=use_gpu)
            wcs = img.getWcs()
            photocal = img.getPhotoCal()
            psf = img.getPsf()
            
            # 1. Collect pixel coords and counts for all candidates
            pxs, pys, counts = [], [], []
            for p in candidate_params:
                self.setParams(p)
                px, py = wcs.positionToPixel(src.getPosition(), src)
                cnt = photocal.brightnessToCounts(src.getBrightness())
                pxs.append(px)
                pys.append(py)
                counts.append(cnt)
                
            # 2. Call the batched PSF method
            # This returns (N, H, W) patches and the origin (x0, y0)
            patches, x0, y0 = psf.getBatchPointSourcePatch(pxs, pys, counts, 
                                                           modelMask=modelMask)
            
            # 3. Compute Chi^2 contribution for this image
            ph, pw = patches.shape[1], patches.shape[2]
            
            # Extract corresponding ROI from the data/ie images
            # If modelMask was used, x0/y0 are already the mask's origin
            d_cut = data_gpu[y0:y0+ph, x0:x0+pw]
            ie_cut = ie_gpu[y0:y0+ph, x0:x0+pw]
            
            # Vectorized chi calculation: (Data - Model) * InvErr
            # CuPy handles the (H, W) vs (N, H, W) broadcasting
            diffs = (d_cut - patches) * ie_cut
            chisqs += cp.sum(diffs**2, axis=(1, 2))
            
        self.setParams(p0)
        return chisqs

    def get_3d_stack_InvErrors(self, tims=None):
        if tims is None:
            tims = self.images
        
        # 1. Automatically find the canvas size needed
        all_h = [t.shape[0] for t in tims]
        all_w = [t.shape[1] for t in tims]
        max_h, max_w = max(all_h), max(all_w)
        
        # 2. Call your existing method with the 'Full' bounds
        # Since ux0, uy0 = 0, the stack offsets gy0, gx0 will just be ty0, tx0
        stack = self.getBatchInvErrors(ux0=0, uy0=0, ux1=max_w, uy1=max_h, tims=tims)
        #for i in range(len(tims)):
        #    print (f'{i=} ie {stack[i].max()=}')
        
        return stack

    def get_3d_stack_Data(self, tims=None):
        if tims is None:
            tims = self.images

        # 1. Automatically find the canvas size needed
        all_h = [t.shape[0] for t in tims]
        all_w = [t.shape[1] for t in tims]
        max_h, max_w = max(all_h), max(all_w)
            
        # 2. Call your existing method with the 'Full' bounds
        # Since ux0, uy0 = 0, the stack offsets gy0, gx0 will just be ty0, tx0
        stack = self.getBatchData(ux0=0, uy0=0, ux1=max_w, uy1=max_h, tims=tims)
        #for i in range(len(tims)):
        #    print (f'{i=} data {stack[i].max()=}')
            
        return stack 

    def get_3d_stack_Model(self, tims=None, src=None, sky=True, minsb=None, **kwargs):
        import cupy as cp
        from tractor.sky import NullSky, ConstantSky
        if tims is None:
            tims = self.images
        if src is None:
            src = self.catalog[0]

        # 1. Automatically find the canvas size needed
        all_h = [t.shape[0] for t in tims]
        all_w = [t.shape[1] for t in tims]
        max_h, max_w = max(all_h), max(all_w)

        """
        stack = np.zeros((max_h, max_w), dtype=tims[0].data.dtype)
        for i, img in enumerate(tims):
            cmodi = self.getModelImage(img)
            #patch = self.getModelPatch(img, src, minsb=minsb, **kwargs)
            #if patch is None:
            #    continue
            #print ("SHAPES", img.shape, cmodi.shape, patch.patch.shape)
            #if sky:
            #    thisSky = img.getSky()
            #    if not isinstance(thisSky, NullSky):
            #        thisSky.addTo(patch.patch)
            #stack[i:all_h[i], :all_w[i]] = patch.patch
            stack[i:all_h[i], :all_w[i]] = cmodi
        """
        stack = self.getBatchModel(ux0=0, uy0=0, ux1=max_w, uy1=max_h, tims=tims)
        return cp.asarray(stack)

    def get_3d_stack_Chi(self, tims=None, ie=None, im=None, mod=None):
        import cupy as cp
        if tims is None:
            tims = self.images

        if ie is None:
            ie = self.get_3d_stack_InvErrors(tims)
        if im is None:
            im = self.get_3d_stack_Data(tims)
        if mod is None:
            mod = self.get_3d_stack_Model(tims)

        stack = (im-mod)*ie
        return stack

    def getChiImageStack(self, use_gpu=True, modelMask=None):
        """
        Returns a 3D CuPy tensor (N_images, H_mask, W_mask) of chi values.
        """
        import cupy as cp
        from tractor.batch_psf import BatchPixelizedPSF
        
        # 1. Gather metadata for the stack
        tims = self.getImages()
        psfs = [tim.getPsf() for tim in tims]
        src = self.getCatalog()[0] # Focusing on the point source
        
        # 2. Initialize the Batch PSF with the stack of PSFs
        batch_psf = BatchPixelizedPSF(psfs)
        
        # 3. Get positions and counts for all images
        # We do this on CPU/NumPy and then move to GPU
        pxs, pys, counts = [], [], []
        for tim in tims:
            px, py = tim.getWcs().positionToPixel(src.getPosition(), src)
            cnt = tim.getPhotoCal().brightnessToCounts(src.getBrightness())
            pxs.append(px)
            pys.append(py)
            counts.append(cnt)
        
        # 4. Get the 3D Model Patch Stack
        # This calls your Lanczos-shifted batch logic
        model_stack, mx0, my0 = batch_psf.getBatchPointSourcePatch(
            cp.array(pxs), cp.array(pys), cp.array(counts), modelMask=modelMask
        )
        
        # 5. Extract Data and InvErr stacks from the images
        # We crop the global images to match the model_stack's footprint
        mh, mw = model_stack.shape[1:]
        
        # Stack the crops: (N_images, H_mask, W_mask)
        data_stack = cp.stack([tim.getImage(use_gpu=False)[my0:my0+mh, mx0:mx0+mw] for tim in tims])
        ie_stack = cp.stack([tim.getInvError(use_gpu=True)[my0:my0+mh, mx0:mx0+mw] for tim in tims])
        
        # 6. Calculate Chi Stack: (Data - Model) * InvError
        chi_stack = (data_stack - model_stack) * ie_stack
        
        return chi_stack

    def getChiImageGPU(self, imgi=-1, img=None, srcs=None, minsb=0., **kwargs):
        import cupy as cp
        t = time.time()
        gi = cp.asarray(self.getChiImage(imgi, img, srcs, minsb, **kwargs))
        tchi[6] += time.time()-t
        """
        TODO: In future use factored_optimizer helpers to get chi2
        if img is None:
            img = self.getImage(imgi)
        gtl[0] += time.time()-t
        t = time.time()
        mod = self.getModelImage(img, srcs=srcs, minsb=minsb, **kwargs)
        gtl[1] += time.time()-t
        t = time.time()
        chi = (img.getImage(use_gpu=True) - cp.asarray(mod)) * img.getInvError(use_gpu=True)
        gtl[2] += time.time()-t
        if not np.all(np.isfinite(chi)):
            print('ERROR: Chi not finite')
        return chi
        """
        return gi

    def getChiImage(self, imgi=-1, img=None, srcs=None, minsb=0., use_gpu=False, **kwargs):
        t = time.time()
        if img is None:
            img = self.getImage(imgi)
        mod = self.getModelImage(img, srcs=srcs, minsb=minsb, **kwargs)
        tchi[7] += time.time()-t
        t = time.time()
        if use_gpu:
            import cupy as cp
            chi = (img.getImage(use_gpu=True)-cp.asarray(mod))*img.getInvError(use_gpu=True)
            tchi[8] += time.time()-t
            return chi
        chi = (img.getImage() - mod) * img.getInvError()
        tchi[9] += time.time()-t
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

    def getLogLikelihoodBatch(self, candidate_params, bounds, use_gpu=True, **kwargs):
        """
        candidate_params: List of parameter arrays (one for each alpha)
        Returns: cp.array of shape (len(candidate_params),)
        """
        import cupy as cp
        # 1. Get the 4D Chi tensor from your new batch engine
        # Expected shape: (N_candidates, N_images, H, W) or (N_candidates, Total_Pixels)
        t = time.time()
        (ux0, ux1, uy0, uy1) = bounds 
        chi_batch = self.getBatchChiImages(
            ux0=ux0, uy0=uy0, ux1=ux1, uy1=uy1,
            candidate_params=candidate_params, 
            use_gpu=use_gpu, 
            #batch_psf=self.batch_psf,
            **kwargs
        )
        """
        print (f'{chi_batch.shape=}')
        for i in range(chi_batch.shape[0]):
            print (f'{candidate_params[i]=}')
            for j in range(chi_batch.shape[1]):
                print (f'{i=} {j=} {chi_batch[i][j].max()=} {chi_batch[i][j].shape=}') 
        """
        
        # 2. Compute -0.5 * sum(chi^2) across all images/pixels for EACH candidate
        # We sum across all axes EXCEPT the first one (the candidate index)
        axes_to_sum = tuple(range(1, chi_batch.ndim))
        chisq = cp.sum(chi_batch**2, axis=axes_to_sum, dtype=cp.float64)
        #print ("LOGL", chisq.shape, chi_batch.shape)
        #print (f'{chisq=}')
        tlog[1] += time.time()-t
        
        return -0.5 * chisq

    def getLogLikelihood(self, **kwargs):
        t = time.time()
        chisq = 0.
        #print ("LEN IMAGES", len(self.images))
        for i, chi in enumerate(self.getChiImages(**kwargs)):
            chisq += (chi.astype(float) ** 2).sum()
            #print (f'{i=} {chisq=} {chi.max()=} {chi.shape=}')
        tlog[0] += time.time()-t
        return -0.5 * chisq

    def getLogProbGPU(self, **kwargs):
        '''
        Return the posterior log PDF, evaluated at the current parameters.
        '''
        import cupy as cp
        lnprior = self.getLogPrior()
        if lnprior == -np.inf:
            return lnprior
        t = time.time()
        lnl = self.getLogLikelihoodGPU(**kwargs)
        #print ("GTL:", gtl, gcl, cl, "GI:", gi)
        lnp = lnprior + lnl
        if cp.isnan(lnp):
            print('Tractor.getLogProb() returning NaN.')
            print('Params:')
            self.printThawedParams()
            print('log likelihood:', lnl)
            print('log prior:', lnprior)
            return -np.inf
        return lnp

    def getLogProbBatch(self, candidate_params, bounds=None, use_gpu=True, ie_stack=None, **kwargs):
        '''
        Return the posterior log PDF for a batch of candidates.
        Evaluates LogPrior (CPU) + LogLikelihood (GPU).
        '''
        import cupy as cp
        t = time.time()
        if bounds is None:
            bounds = get_global_extent_mm(self.images)[0] 
        if ie_stack is None:
            (ux0, uy0, ux1, uy1) = bounds
            tims = self.images
            ie_stack = self.getBatchInvErrors(ux0, uy0, ux1, uy1, use_gpu=use_gpu, tims=tims)
        n_cand = len(candidate_params)
        #print (f'{candidate_params=}')
        
        # 1. Evaluate Priors (CPU-bound, usually very fast)
        p0 = self.getParams()
        priors = []
        for p in candidate_params:
            self.setParams(p)
            priors.append(self.getLogPrior())
        priors = np.array(priors)
        self.setParams(p0) # Restore immediately
        
        # Identify candidates that are physically possible
        valid_mask = np.isfinite(priors)
        if not np.any(valid_mask):
            print ("INVALID")
            return cp.asarray(priors)

        # 2. Evaluate Likelihood for valid candidates only
        # (Optional: You could filter candidate_params here to save GPU time,
        # but for small N like 13, it's often faster to just run the batch)
        lnl = self.getLogLikelihoodBatch(candidate_params, bounds=bounds, use_gpu=use_gpu, ie_stack=ie_stack, **kwargs)
        #print (f'{lnl=}')
        
        # 3. Combine
        lnp = cp.asarray(priors)
        lnp[valid_mask] += lnl
        #print (f'{lnp=}')
        tlog[3] += time.time()-t
        
        return lnp

    def getLogProb(self, **kwargs):
        '''
        Return the posterior log PDF, evaluated at the current parameters.
        '''
        t = time.time()
        lnprior = self.getLogPrior()
        #print (f'{lnprior=}')
        if lnprior == -np.inf:
            return lnprior
        lnl = self.getLogLikelihood(**kwargs)
        #print (f'{lnl=}')
        #print ("TL:", tl, gcl, cl, "GI", gi)
        lnp = lnprior + lnl
        #print ("LP", lnprior, "LNL", lnl, "LNP", lnp)
        if np.isnan(lnp):
            print('Tractor.getLogProb() returning NaN.')
            print('Params:')
            self.printThawedParams()
            print('log likelihood:', lnl)
            print('log prior:', lnprior)
            return -np.inf
        tlog[2] += time.time()-t
        #print (f'{teng=}')
        #print (f'{tchi=}')
        #print (f'{tmod=}')
        #print (f'{tlog=}')
        return lnp

#    def getBatchInvErrors(self, use_gpu=True, modelMasks=None, tims=None):
    def getBatchInvErrors(self, ux0, uy0, ux1, uy1, use_gpu=True, tims=None, modelMasks=None):
        import cupy as cp
        t = time.time()
        if tims is None:
            tims = self.getImages()
        #if modelMasks is None:
        #    modelMasks = [self._getModelMaskFor(tim, self.catalog[0]) for tim in tims]

        #(ux0, ux1, uy0, uy1), img_info = get_global_extent(tims, modelMasks)
        #mm_coords, mm_info = get_global_extent_mm(tims, modelMasks)
        img_info = get_global_extent(tims, ux0, uy0, ux1, uy1)

        #print (f'{ux0=} {ux1=} {uy0=} {uy1=} {img_info=}')
        #print (f'{mm_coords=} {mm_info=}')
        uH, uW = uy1 - uy0, ux1 - ux0
        stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)
        #print (f'{stack.shape=}')

        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            ie = tim.getInvError(use_gpu=True)
            if not isinstance(ie, cp.ndarray): ie = cp.asarray(ie)

            # Local indices: since tim has no x0/y0, we pull from its (0,0)
            # but if it's a mask, tx0/ty0 ARE the local offsets we need
            # relative to the tim's internal buffer.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw

            # Global indices relative to the 3D stack
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            #print (f"IE {i=} {ie.max()=} {ie.shape=} {gy0=} {gy1=} {gx0=} {gx1=} {ly0=} {ly1=} {lx0=} {lx1=}")
            #print (cp.where(ie == ie.max()))

            stack[i, gy0:gy1, gx0:gx1] = ie[ly0:ly1, lx0:lx1]
            #print (f"IE {stack[i].max()=}")
        teng[0] += time.time()-t
        #print (f'{teng=}')
        return stack

    def getBatchData(self, ux0, uy0, ux1, uy1, use_gpu=True, tims=None, modelMasks=None):
        import cupy as cp
        t = time.time()
        if tims is None:
            tims = self.getImages()
        #if modelMasks is None:
        #    modelMasks = [self._getModelMaskFor(tim, self.catalog[0]) for tim in tims]

        #(ux0, ux1, uy0, uy1), img_info = get_global_extent(tims, modelMasks)
        #mm_coords, mm_info = get_global_extent_mm(tims, modelMasks)
        img_info = get_global_extent(tims, ux0, uy0, ux1, uy1)

        #print (f'{ux0=} {ux1=} {uy0=} {uy1=} {img_info=}')
        #print (f'{mm_coords=} {mm_info=}')
        uH, uW = uy1 - uy0, ux1 - ux0
        stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)
        #print (f'{stack.shape=}')

        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            im = tim.getImage(use_gpu=True)
            if not isinstance(im, cp.ndarray): im = cp.asarray(im)

            # Local indices: since tim has no x0/y0, we pull from its (0,0)
            # but if it's a mask, tx0/ty0 ARE the local offsets we need
            # relative to the tim's internal buffer.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw

            # Global indices relative to the 3D stack
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            stack[i, gy0:gy1, gx0:gx1] = im[ly0:ly1, lx0:lx1]
            #print (f"DATA {stack[i].max()=}")
        #teng[0] += time.time()-t
        #print (f'{teng=}')
        return stack

    def getBatchModel(self, ux0, uy0, ux1, uy1, use_gpu=True, tims=None, modelMasks=None):
        import cupy as cp
        t = time.time()
        if tims is None:
            tims = self.getImages()
        #if modelMasks is None:
        #    modelMasks = [self._getModelMaskFor(tim, self.catalog[0]) for tim in tims]
                
        #(ux0, ux1, uy0, uy1), img_info = get_global_extent(tims, modelMasks)
        #mm_coords, mm_info = get_global_extent_mm(tims, modelMasks)
        img_info = get_global_extent(tims, ux0, uy0, ux1, uy1)

        #print (f'{ux0=} {ux1=} {uy0=} {uy1=} {img_info=}')
        #print (f'{mm_coords=} {mm_info=}')
        uH, uW = uy1 - uy0, ux1 - ux0
        stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)
        #print (f'{stack.shape=}')

        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            cmodi = self.getModelImage(tim)
            if not isinstance(cmodi, cp.ndarray): cmodi = cp.asarray(cmodi)

            # Local indices: since tim has no x0/y0, we pull from its (0,0)
            # but if it's a mask, tx0/ty0 ARE the local offsets we need
            # relative to the tim's internal buffer.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw
            
            # Global indices relative to the 3D stack
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            stack[i, gy0:gy1, gx0:gx1] = cmodi[ly0:ly1, lx0:lx1]
            #print (f"DATA {stack[i].max()=}")
        #teng[0] += time.time()-t
        #print (f'{teng=}')
        return stack


#    def getBatchModelImage(self, use_gpu=True, batch_psf=None, modelMasks=None, tims=None):
    def getBatchModelImage(self, ux0, uy0, ux1, uy1, use_gpu=True, batch_psf=None, tims=None):
        import cupy as cp
        import numpy as np
        from tractor.sky import NullSky, ConstantSky

        t = time.time()
        #if tims is None:
        #    tims = self.getImages()
        #if modelMasks is None:
        #    modelMasks = [self._getModelMaskFor(tim, self.catalog[0]) for tim in tims]
        #(ux0, ux1, uy0, uy1), img_info = get_global_extent(tims, modelMasks)
        img_info = get_global_extent(tims, ux0, uy0, ux1, uy1)
        uH, uW = uy1 - uy0, ux1 - ux0
        # Initialize 3D stack on GPU
        model_stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)

        # 1. Add Sky
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            sky = tim.getSky()
            if isinstance(sky, NullSky):
                continue

            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0
            #print (f'SKY {gx0=} {gx1=} {gy0=} {gy1=} {th=} {tw=} {ux0=} {uy0=}')

            if isinstance(sky, ConstantSky):
                val = sky.getConstant()
                if val != 0:
                    model_stack[i, gy0:gy1, gx0:gx1] += val
            else:
                temp_sky = np.zeros(tim.shape, dtype=np.float32)
                sky.addTo(temp_sky)
                model_stack[i, gy0:gy1, gx0:gx1] += cp.asarray(temp_sky[ty0:ty0+th, tx0:tx0+tw])
        tmod[2] += time.time()-t
        #print ("MOD SKY", model_stack.max(), model_stack.sum())

        # 2. Add Sources
        for src in self.catalog:
            t = time.time()
            gx_list, gy_list, counts = [], [], cp.zeros(len(tims), dtype=cp.float32)
            for i, tim in enumerate(tims):
                # Pixel position on the tim
                px, py = tim.getWcs().positionToPixel(src.getPosition())
                #print (f'GPUX {i=} {px=} {py=} {tim=} {src.getPosition()=}')

                # Global coordinate is just px/py because tim starts at 0,0
                gx_list.append(px)
                gy_list.append(py)

                if (0 <= px < tim.shape[1] and 0 <= py < tim.shape[0]):
                    counts[i] = tim.getPhotoCal().brightnessToCounts(src.getBrightness())

            tmod[3] += time.time()-t
            t = time.time()
            if cp.all(counts == 0): continue

            #print (f'{gx_list=} {gy_list=}')
            #print (f'SRC {ux0=} {uy0=} {uW=} {uH=}')
            src_3d = batch_psf.getBatchPointSourcePatch(
                cp.array(gx_list), cp.array(gy_list),
                ux0, uy0, uW, uH
            )
            model_stack += src_3d * counts[:, cp.newaxis, cp.newaxis]
            tmod[4] += time.time()-t
            """
            for i, tim in enumerate(tims):
                mod1 = self.getModelImage(img=tim)
                print (f'MOD1 {mod1.max()=} {mod1.shape=} {mod1.sum()=}')
                print (np.where(mod1 == mod1.max()))
                print (f'MOD STACK {i=} {model_stack[i].max()=} {model_stack[i].shape=} {model_stack[i].sum()=}')
                print (cp.where(model_stack[i] == model_stack[i].max()))
            """

        return model_stack

    def getBatchChiImages(self, ux0, uy0, ux1, uy1, use_gpu=True, batch_psf=None, tims=None, ie_stack=None, candidate_params=None):
        import cupy as cp
        import time
        t = time.time()
        tx = time.time()
        if tims is None:
            tims = self.getImages()
        Nimages = len(tims)

        # 2. Get the global canvas bounds and the per-image offsets/shapes
        #(ux0, ux1, uy0, uy1), img_info = get_global_extent(tims, modelMasks)
        img_info = get_global_extent(tims, ux0, uy0, ux1, uy1)
        uH, uW = uy1 - uy0, ux1 - ux0

        # 3. Initialize stacks for Data and Inverse Variance
        data_stack = cp.zeros((len(tims), uH, uW), dtype=cp.float64)
        #ie_stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)

        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            # Global indices relative to the 3D stack origin (ux0, uy0)
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            # Local indices relative to the tim's internal buffer (starting at 0,0)
            # If tx0/ty0 came from a mask, they represent the offset within the tim.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw
            # Populate the stacks
            data_stack[i, gy0:gy1, gx0:gx1] = cp.asarray(tim.getImage(use_gpu=True)[ly0:ly1, lx0:lx1])

        tchi[1] += time.time()-t
        t = time.time()
        if ie_stack is None:
            ie_stack = self.getBatchInvErrors(ux0, uy0, ux1, uy1, use_gpu=use_gpu, tims=tims)
        tchi[2] += time.time()-t
        t = time.time()

        # 2. Model Stack Generation
        if candidate_params is not None:
            # 4D PATH: (N_alphas, N_images, uH, uW)
            n_alphas = len(candidate_params)
            p_initial = self.getParams()
            
            # Pre-allocate 4D on CPU for the fast fill
            cmod_stack_4d = np.zeros((n_alphas, Nimages, uH, uW), dtype=np.float64)
            
            for a_idx, p_trial in enumerate(candidate_params):
                self.setParams(p_trial) # Temporary set to render
                for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
                    gy0, gy1 = ty0 - uy0, ty0 + th - uy0
                    gx0, gx1 = tx0 - ux0, tx0 + tw - ux0
                    cmod_stack_4d[a_idx, i, gy0:gy1, gx0:gx1] = self.getModelImage(tim)[ty0:ty0+th, tx0:tx0+tw]
            
            self.setParams(p_initial) # Restore state
            model_stack = cp.asarray(cmod_stack_4d)
            
            # Broadcoast Data and IE over the Alpha dimension
            # (1, N_img, H, W) is broadcast against (N_alpha, N_img, H, W)
            chi = (data_stack[cp.newaxis, ...] - model_stack) * ie_stack[cp.newaxis, ...]
            tchi[3] += time.time()-t
            tchi[0] += time.time()-tx
            return chi

        t = time.time()
        cmod_stack = np.zeros((len(tims), uH, uW), dtype=np.float64)
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            # Local indices relative to the tim's internal buffer (starting at 0,0)
            # If tx0/ty0 came from a mask, they represent the offset within the tim.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw

            #print (f'{ux0=} {ux1=} {uy0=} {uy1=} {gy0=} {gy1=} {gx0=} {gx1=} {ly0=} {ly1=} {lx0=} {lx1=}')
            # Populate the stacks
            cmodi = self.getModelImage(tim)
            #print (f'{cmodi.max()=} ', np.where(cmodi == cmodi.max()))
            cmod_stack[i, gy0:gy1, gx0:gx1] = cmodi[ly0:ly1, lx0:lx1]
            #print (f'{cmodi.shape=} {cmodi.max()=} {cmod_stack[i].max()=}')
        cmod_stack = cp.asarray(cmod_stack)
        tchi[4] += time.time()-t

        #print (f'{cmod_stack.shape=}')

        # 4. Generate the Model stack using the same extent
        #model_stack = self.getBatchModelImage(use_gpu=use_gpu, batch_psf=batch_psf, modelMasks=modelMasks, tims=tims)
        #model_stack = self.getBatchModelImage(ux0, uy0, ux1, uy1, use_gpu=use_gpu, batch_psf=batch_psf, tims=tims)
        #print (f'{model_stack.shape=}')
        #teng[3] += time.time()-t
        model_stack = cmod_stack

        tchi[0] += time.time()-tx
        # 5. Final Chi calculation: (Data - Model) * Weight
        # Weight is usually Inverse Error (1/sigma)
        #print (f'{tchi=}')
        #print (f'{teng=}')
        #print (f'{tmod=}')
        #print (f'{tlog=}')
        return (data_stack - model_stack) * ie_stack



#    def getBatchChiImages(self, use_gpu=True, batch_psf=None, modelMasks=None, tims=None):
    def getBatchChiImages_save(self, ux0, uy0, ux1, uy1, use_gpu=True, batch_psf=None, tims=None, ie_stack=None):
        import cupy as cp
        import time
        t = time.time()
        if tims is None:
            tims = self.getImages()
        #if modelMasks is None:
        #    modelMasks = [self._getModelMaskFor(tim, self.catalog[0]) for tim in tims]
        
        # 2. Get the global canvas bounds and the per-image offsets/shapes
        #(ux0, ux1, uy0, uy1), img_info = get_global_extent(tims, modelMasks)
        img_info = get_global_extent(tims, ux0, uy0, ux1, uy1)
        uH, uW = uy1 - uy0, ux1 - ux0

        # 3. Initialize stacks for Data and Inverse Variance
        data_stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)
        #ie_stack = cp.zeros((len(tims), uH, uW), dtype=cp.float32)

        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            # Global indices relative to the 3D stack origin (ux0, uy0)
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            # Local indices relative to the tim's internal buffer (starting at 0,0)
            # If tx0/ty0 came from a mask, they represent the offset within the tim.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw

            # Populate the stacks
            data_stack[i, gy0:gy1, gx0:gx1] = cp.asarray(tim.getImage(use_gpu=True)[ly0:ly1, lx0:lx1])
            #ie_stack[i, gy0:gy1, gx0:gx1] = cp.asarray(tim.getInvError(use_gpu=True)[ly0:ly1, lx0:lx1])
            #print (f"Chi Batch {i=} {ie_stack[i].shape=} {ie_stack[i].max()=} {data_stack[i].shape=} {data_stack[i].max()=} {gy0=} {gy1=} {gx0=} {gx1=} {ly0=} {ly1=} {lx0=} {lx1=}")

        teng[2] += time.time()-t
        t = time.time()

        if ie_stack is None:
            ie_stack = self.getBatchInvErrors(ux0, uy0, ux1, uy1, use_gpu=use_gpu, tims=tims)

        teng[5] += time.time()-t
        t = time.time()

        """
        cdata = np.zeros((len(tims), uH, uW), dtype=np.float32)
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0
                
            # Local indices relative to the tim's internal buffer (starting at 0,0)
            # If tx0/ty0 came from a mask, they represent the offset within the tim.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw
                        
            # Populate the stacks
            cdata[i, gy0:gy1, gx0:gx1] = tim.getImage()[ly0:ly1, lx0:lx1]
            #print (f'{i=} {cdata[i].max()=} {data_stack[i].max()=}')
        cdata = cp.asarray(cdata)
        teng[1] += time.time()-t
        t = time.time()
        """

        cmod_stack = np.zeros((len(tims), uH, uW), dtype=np.float32)
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            # Local indices relative to the tim's internal buffer (starting at 0,0)
            # If tx0/ty0 came from a mask, they represent the offset within the tim.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw

            #print (f'{ux0=} {ux1=} {uy0=} {uy1=} {gy0=} {gy1=} {gx0=} {gx1=} {ly0=} {ly1=} {lx0=} {lx1=}')
            # Populate the stacks
            cmodi = self.getModelImage(tim)
            #print (f'{cmodi.max()=} ', np.where(cmodi == cmodi.max()))
            cmod_stack[i, gy0:gy1, gx0:gx1] = cmodi[ly0:ly1, lx0:lx1]
            #print (f'{cmodi.shape=} {cmodi.max()=} {cmod_stack[i].max()=}')
        cmod_stack = cp.asarray(cmod_stack)

        teng[0] += time.time()-t
        t = time.time()
        """
        cmod_stack2 = np.zeros((len(tims), uH, uW), dtype=np.float32)
        for i, (tim, (tx0, ty0, th, tw)) in enumerate(zip(tims, img_info)):
            gy0, gy1 = ty0 - uy0, ty0 + th - uy0
            gx0, gx1 = tx0 - ux0, tx0 + tw - ux0

            # Local indices relative to the tim's internal buffer (starting at 0,0)
            # If tx0/ty0 came from a mask, they represent the offset within the tim.
            ly0, ly1 = ty0, ty0 + th
            lx0, lx1 = tx0, tx0 + tw

            #print (f'{ux0=} {ux1=} {uy0=} {uy1=} {gy0=} {gy1=} {gx0=} {gx1=} {ly0=} {ly1=} {lx0=} {lx1=}')
            # Populate the stacks
            cmodi = self.getModelSubImageOneSource(tim)
            #print (f'2 {cmodi.max()=} ', np.where(cmodi == cmodi.max()))
            if cmodi is None:
                continue
            #cmod_stack2[i, gy0:gy1, gx0:gx1] = cmodi
        cmod_stack2 = cp.asarray(cmod_stack2)
        """

        #print (f'{cmod_stack.shape=}')
        #print ("ALLCLOSE", cp.allclose(cmod_stack, cmod_stack2))
        #teng[6] += time.time()-t
        t = time.time()

        # 4. Generate the Model stack using the same extent
        #model_stack = self.getBatchModelImage(use_gpu=use_gpu, batch_psf=batch_psf, modelMasks=modelMasks, tims=tims)
        #model_stack = self.getBatchModelImage(ux0, uy0, ux1, uy1, use_gpu=use_gpu, batch_psf=batch_psf, tims=tims)
        #print (f'{model_stack.shape=}')
        #teng[3] += time.time()-t
        model_stack = cmod_stack

        # 5. Final Chi calculation: (Data - Model) * Weight
        # Weight is usually Inverse Error (1/sigma)
        print (f'{teng=}')
        return (data_stack - model_stack) * ie_stack
