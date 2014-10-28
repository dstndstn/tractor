'''
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`engine.py`
===========

Core image modeling and fitting.
'''

from math import ceil, floor, pi, sqrt, exp
import time
import logging
import random
import os
import resource
import gc

import numpy as np

# Scipy deps -- pushed down to where they are used.
# from scipy.sparse import csr_matrix, csc_matrix
# from scipy.sparse.linalg import lsqr
# from scipy.ndimage.morphology import binary_dilation
# from scipy.ndimage.measurements import label

from astrometry.util.multiproc import *
from astrometry.util.ttime import *

from .utils import MultiParams, _isint, listmax, get_class_from_name
from .cache import *
from .patch import *

def logverb(*args):
    msg = ' '.join([str(x) for x in args])
    logging.debug(msg)
def logmsg(*args):
    msg = ' '.join([str(x) for x in args])
    logging.info(msg)
def isverbose():
    # Ugly logging interface...
    return (logging.getLogger().level <= logging.DEBUG)

def set_fp_err():
    '''Cause all floating-point errors to raise exceptions.
    Returns the current error state so you can revert via:

        olderr = set_fp_err()
        # do stuff
        np.seterr(**olderr)
    '''
    return np.seterr(all='raise')

class Image(MultiParams):
    '''
    An image plus its calibration information.  An ``Image`` has
    pixels, inverse-variance map, WCS, PSF, photometric calibration
    information, and sky level.  All these things are ``Params``
    instances, and ``Image`` is a ``MultiParams`` so that the Tractor
    can optimize them.
    '''
    def __init__(self, data=None, invvar=None, inverr=None,
                 psf=None, wcs=None, sky=None,
                 photocal=None, name=None, time=None, **kwargs):
        '''
        Args:
          * *data*: numpy array: the image pixels
          * *invvar*: numpy array: the image inverse-variance
          * *inverr*: numpy array: the image inverse-error
          * *psf*: a :class:`tractor.PSF` duck
          * *wcs*: a :class:`tractor.WCS` duck
          * *sky*: a :class:`tractor.Sky` duck
          * *photocal*: a :class:`tractor.PhotoCal` duck
          * *name*: string name of this image.
          * *zr*: plotting range ("vmin"/"vmax" in matplotlib.imshow)

        Only one of *invvar* and *inverr* should be given.  If both
        are given, inverr takes precedent.
          
        '''
        self.data = data
        if inverr is not None:
            self.inverr = inverr
        elif invvar is not None:
            self.inverr = np.sqrt(invvar)
            
        self.name = name
        self.zr = kwargs.pop('zr', None)
        self.time = time

        # Fill in defaults, if necessary.
        if wcs is None:
            from .basics import NullWCS
            wcs = NullWCS()
        if sky is None:
            from .basics import NullSky
            sky = NullSky()
        if photocal is None:
            from .basics import NullPhotoCal
            photocal = NullPhotoCal()

        # acceptable approximation level when rendering this model
        # image
        self.modelMinval = 0.
            
        super(Image, self).__init__(psf, wcs, photocal, sky)

    def __str__(self):
        return 'Image ' + str(self.name)

    @staticmethod
    def getNamedParams():
        return dict(psf=0, wcs=1, photocal=2, sky=3)

    def getTime(self):
        return self.time
    
    def getParamDerivatives(self, tractor, srcs):
        '''
        Returns a list of Patch objects, one per numberOfParams().
        Note that this means you have to pay attention to the
        frozen/thawed state.

        Can return None for no derivative, or False if you want the
        Tractor to compute the derivatives for you.
        '''
        derivs = []
        for s in self._getActiveSubs():
            if hasattr(s, 'getParamDerivatives'):
                #print 'Calling getParamDerivatives on', s
                sd = s.getParamDerivatives(tractor, self, srcs)
                assert(len(sd) == s.numberOfParams())
                derivs.extend(sd)
            else:
                derivs.extend([False] * s.numberOfParams())
        # print 'Image.getParamDerivatives: returning', derivs
        return derivs

    def getSky(self):
        return self.sky

    def setSky(self, sky):
        self.sky = sky

    def setPsf(self, psf):
        self.psf = psf

    @property
    def shape(self):
        return self.getShape()

    @property
    def invvar(self):
        return self.inverr**2
    
    # Numpy arrays have shape H,W
    def getWidth(self):
        return self.getShape()[1]
    def getHeight(self):
        return self.getShape()[0]
    def getShape(self):
        if 'shape' in self.__dict__:
            return self.shape
        return self.data.shape

    def getModelShape(self):
        return self.getShape()
    
    def hashkey(self):
        return ('Image', id(self.data), id(self.inverr), self.psf.hashkey(),
                self.sky.hashkey(), self.wcs.hashkey(),
                self.photocal.hashkey())

    def numberOfPixels(self):
        (H,W) = self.data.shape
        return W*H

    def getInvError(self):
        return self.inverr
    def getInvvar(self):
        return self.inverr**2

    def getImage(self):
        return self.data
    def getPsf(self):
        return self.psf
    def getWcs(self):
        return self.wcs
    def getPhotoCal(self):
        return self.photocal

    @staticmethod
    def readFromFits(fits, prefix=''):
        hdr = fits[0].read_header()
        pix = fits[1].read()
        iv = fits[2].read()
        assert(pix.shape == iv.shape)

        def readObject(prefix):
            k = prefix
            objclass = hdr[k]
            clazz = get_class_from_name(objclass)
            fromfits = getattr(clazz, 'fromFitsHeader')
            print 'fromFits:', fromfits
            obj = fromfits(hdr, prefix=prefix + '_')
            print 'Got:', obj
            return obj

        psf = readObject(prefix + 'PSF')
        wcs = readObject(prefix + 'WCS')
        sky = readObject(prefix + 'SKY')
        pcal = readObject(prefix + 'PHO')

        return Image(data=pix, invvar=iv, psf=psf, wcs=wcs, sky=sky,
                     photocal=pcal)
        
    def toFits(self, fits, prefix='', primheader=None, imageheader=None,
               invvarheader=None):
        psf = self.getPsf()
        wcs = self.getWcs()
        sky = self.getSky()
        pcal = self.getPhotoCal()
        
        if primheader is None:
            import fitsio
            hdr = fitsio.FITSHDR()
        else:
            hdr = primheader
        tt = type(psf)
        psf_type = '%s.%s' % (tt.__module__, tt.__name__)
        tt = type(wcs)
        wcs_type = '%s.%s' % (tt.__module__, tt.__name__)
        tt = type(sky)
        sky_type = '%s.%s' % (tt.__module__, tt.__name__)
        tt = type(pcal)
        pcal_type = '%s.%s' % (tt.__module__, tt.__name__)
        hdr.add_record(dict(name=prefix + 'PSF', value=psf_type,
                            comment='PSF class'))
        hdr.add_record(dict(name=prefix + 'WCS', value=wcs_type,
                            comment='WCS class'))
        hdr.add_record(dict(name=prefix + 'SKY', value=sky_type,
                            comment='Sky class'))
        hdr.add_record(dict(name=prefix + 'PHO', value=pcal_type,
                            comment='PhotoCal class'))
        psf.toFitsHeader(hdr,  prefix + 'PSF_')
        wcs.toFitsHeader(hdr,  prefix + 'WCS_')
        sky.toFitsHeader(hdr,  prefix + 'SKY_')
        pcal.toFitsHeader(hdr, prefix + 'PHO_')

        fits.write(None, header=hdr)
        fits.write(self.getImage(), header=imageheader)
        fits.write(self.getInvvar(), header=invvarheader)

    
        
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
        print 'Catalog with %i sources:' % len(self)
        for i,x in enumerate(self):
            print '  %i:' % i, x

    # inherited from MultiParams:
    # def __len__(self):
    #  ''' Returns the number of sources in this catalog'''
    # def numberOfParams(self):
    #  '''Returns the number of active parameters in all sources'''

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

# These are free functions for multiprocessing in "getderivs2()"
def getmodelimagestep(X):
    (tr, j, k, p0, step) = X
    im = tr.getImage(j)
    #print 'Setting param', p0, step, p0+step
    im.setParam(k, p0 + step)
    mod = tr.getModelImage(im)
    im.setParam(k, p0)
    return mod
def getmodelimagefunc(X):
    (tr, imj) = X
    #print 'getmodelimagefunc(): imj', imj, 'pid', os.getpid()
    return tr.getModelImage(imj)
def getsrcderivs(X):
    (src, img) = X
    return src.getParamDerivatives(img)
def getimagederivs(X):
    (imj, img, tractor, srcs) = X
    ## FIXME -- avoid shipping all images...
    return img.getParamDerivatives(tractor, srcs)
def getmodelimagefunc2(X):
    (tr, im) = X
    #print 'getmodelimagefunc2(): im', im, 'pid', os.getpid()
    #tr.images = Images(im)
    try:
        return tr.getModelImage(im)
    except:
        import traceback
        print 'Exception in getmodelimagefun2:'
        traceback.print_exc()
        raise

class OptResult():
    # quack
    pass

class Tractor(MultiParams):
    """
    Heavy farm machinery.

    As you might guess from the name, this is the main class of the
    Tractor framework.  A Tractor has a set of Images and a set of
    Sources, and has methods to optimize the parameters of those
    Images and Sources.

    """
    @staticmethod
    def getName():
        return 'Tractor'
    
    @staticmethod
    def getNamedParams():
        return dict(images=0, catalog=1)

    def __init__(self, images=[], catalog=[], mp=None):
        '''
        - `images:` list of Image objects (data)
        - `catalog:` list of Source objects
        '''
        if not isinstance(images, Images):
            images = Images(*images)
        if not isinstance(catalog, Catalog):
            catalog = Catalog(*catalog)
        super(Tractor,self).__init__(images, catalog)
        self._setup(mp=mp)

    def disable_cache(self):
        self.cache = None

    def _setup(self, mp=None, cache=None, pickleCache=False):
        if mp is None:
            mp = multiproc()
        self.mp = mp
        self.modtype = np.float32
        if cache is None:
            cache = Cache()
        self.cache = cache
        self.pickleCache = pickleCache

    def __str__(self):
        s = '%s with %i sources and %i images' % (self.getName(), len(self.catalog), len(self.images))
        names = []
        for im in self.images:
            if im.name is None:
                names.append('[unnamed]')
            else:
                names.append(im.name)
        s += ' (' + ', '.join(names) + ')'
        return s

    def is_multiproc(self):
        return self.mp.pool is not None

    def _map(self, func, iterable):
        return self.mp.map(func, iterable)
    def _map_async(self, func, iterable):
        return self.mp.map_async(func, iterable)

    # For use from emcee
    def __call__(self, X):
        self.setParams(X)
        return self.getLogProb()

    # For pickling
    def __getstate__(self):
        S = (self.getImages(), self.getCatalog(), self.liquid)
        if self.pickleCache:
            S = S + (self.cache,)
        return S
    def __setstate__(self, state):
        args = {}
        if len(state) == 3:
            (images, catalog, liquid) = state
        elif len(state) == 4:
            (images, catalog, liquid, cache) = state
            args.update(cache=cache, pickleCache=pickleCache)
        self.subs = [images, catalog]
        self.liquid = liquid
        self._setup(**args)

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

    def computeParameterErrors(self, symmetric=False):
        if not symmetric:
            return self._param_errors_1()

        e1 = self._param_errors_1(1.)
        e2 = self._param_errors_1(-1.)
        sigs = []
        for s1,s2 in zip(e1,e2):
            if s1 is None:
                sigs.append(s2)
            elif s2 is None:
                sigs.append(s1)
            else:
                sigs.append((s1 + s2) / 2.)
        return sigs

    def _param_errors_1(self, sign=1.):
        # Try to compute 1-sigma error bars on each parameter by
        # sweeping the parameter (in the "getStepSizes()" direction)
        # until we find delta-chi-squared of 1.
        # That's a delta-logprob of 0.5
        stepsizes = np.array(self.getStepSizes())
        pp0 = np.array(self.getParams())
        lnp0 = self.getLogProb()
        nms = self.getParamNames()
        sigmas = []
        target = lnp0 - 0.5
        for i,(p0,step,nm) in enumerate(zip(pp0, stepsizes, nms)):
            self.setParams(pp0)
            # Take increasingly large steps until we find one with
            # logprob < target
            p1 = None
            #print 'Looking for error bars on', nm, 'around', p0
            for j in range(20):
                tryp1 = p0 + sign * step * (2. ** j)
                self.setParam(i, tryp1)
                lnp1 = self.getLogProb()
                #print '  stepping to', tryp1, 'for dlnp', lnp1 - lnp0
                # FIXME -- could also track the largest lnp < target,
                # to narrow the binary search range later...
                if lnp1 < target:
                    p1 = tryp1
                    break
            if p1 is None:
                sigmas.append(None)
                continue
            # Binary search until the range is small enough.
            lo,hi = min(p0, p1), max(p0, p1)
            lnplo,lnphi = lnp0,lnp1
            #print 'Binary searching in', lo, hi
            sigma = None
            for j in range(20):
                assert(lo <= hi)
                if np.abs(lnplo - target) < 1e-3:
                    sigma = np.abs(lo - p0)
                    break
                mid = (lo + hi) / 2.
                self.setParam(i, mid)
                lnpmid = self.getLogProb()
                #print '  trying', mid, 'for dlnp', lnpmid - lnp0
                if lnpmid < target:
                    hi = mid
                    lnphi = lnpmid
                else:
                    lo = mid
                    lnplo = lnpmid
            sigmas.append(sigma)
        return np.array(sigmas)

    def optimize_lbfgsb(self, hessian_terms=10, plotfn=None):

        XX = []
        OO = []
        def objective(x, tractor, stepsizes, lnp0):
            res = lnp0 - tractor(x * stepsizes)
            print 'LBFGSB objective:', res
            if plotfn:
                XX.append(x.copy())
                OO.append(res)
            return res

        from scipy.optimize import fmin_l_bfgs_b

        stepsizes = np.array(self.getStepSizes())
        p0 = np.array(self.getParams())
        lnp0 = self.getLogProb()

        print 'Active parameters:', len(p0)

        print 'Calling L-BFGS-B ...'
        X = fmin_l_bfgs_b(objective, p0 / stepsizes, fprime=None,
                          args=(self, stepsizes, lnp0),
                          approx_grad=True, bounds=None, m=hessian_terms,
                          epsilon=1e-8, iprint=0)
        p1,lnp1,d = X
        print d
        print 'lnp0:', lnp0
        self.setParams(p1 * stepsizes)
        print 'lnp1:', self.getLogProb()

        if plotfn:
            import pylab as plt
            plt.clf()
            XX = np.array(XX)
            OO = np.array(OO)
            print 'XX shape', XX.shape
            (N,D) = XX.shape
            for i in range(D):
                OO[np.abs(OO) < 1e-8] = 1e-8
                neg = (OO < 0)
                plt.semilogy(XX[neg,i], -OO[neg], 'bx', ms=12, mew=2)
                pos = np.logical_not(neg)
                plt.semilogy(XX[pos,i], OO[pos], 'rx', ms=12, mew=2)
                I = np.argsort(XX[:,i])
                plt.plot(XX[I,i], np.abs(OO[I]), 'k-', alpha=0.5)
                plt.ylabel('Objective value')
                plt.xlabel('Parameter')
            plt.twinx()
            for i in range(D):
                plt.plot(XX[:,i], np.arange(N), 'r-')
                plt.ylabel('L-BFGS-B iteration number')
            plt.savefig(plotfn)

    def getDynamicScales(self):
        '''
        Returns parameter step sizes that will result in changes in
        chi^2 of about 1.0
        '''
        scales = np.zeros(self.numberOfParams())
        for i in range(self.getNImages()):
            derivs = self._getOneImageDerivs(i)
            for j,x0,y0,der in derivs:
                scales[j] += np.sum(der**2)
        scales = np.sqrt(scales)
        I = (scales != 0)
        if any(I):
            scales[I] = 1./scales[I]
        I = (scales == 0)
        if any(I):
            scales[I] = np.array(self.getStepSizes())[I]
        return scales
        
    def _ceres_opt(self, variance=False, scale_columns=True,
                   numeric=False, scaled=True, numeric_stepsize=0.1,
                   dynamic_scale=True,
                   dlnp = 1e-3, max_iterations=0):
        from ceres import ceres_opt

        pp = self.getParams()
        if len(pp) == 0:
            return None

        if scaled:
            p0 = np.array(pp)

            if dynamic_scale:
                scales = self.getDynamicScales()
                # print 'Dynamic scales:', scales

            else:
                scales = np.array(self.getStepSizes())
            
            # Offset all the parameters so that Ceres sees them all
            # with value 1.0
            p0 -= scales
            params = np.ones_like(p0)
        
            scaler = ScaledTractor(self, p0, scales)
            tractor = scaler

        else:
            params = np.array(pp)
            tractor = self

        variance_out = None
        if variance:
            variance_out = np.zeros_like(params)

        R = ceres_opt(tractor, self.getNImages(), params, variance_out,
                      (1 if scale_columns else 0),
                      (1 if numeric else 0), numeric_stepsize,
                      dlnp, max_iterations)
        if variance:
            R['variance'] = variance_out

        if scaled:
            print 'Opt. in scaled space:', params
            self.setParams(p0 + params * scales)
            variance_out *= scales**2
            R['params0'] = p0
            R['scales'] = scales

        return R
        
    # This function is called-back by _ceres_opt; it is called from
    # ceres-tractor.cc via ceres.i .
    def _getOneImageDerivs(self, imgi):
        # Returns:
        #     [  (param-index, deriv_x0, deriv_x0, deriv), ... ]
        # not necessarily in order of param-index
        #
        # NOTE, this scales the derivatives by inverse-error and -1 to
        # yield derivatives of CHI with respect to PARAMs; NOT the
        # model image wrt params.
        #
        allderivs = []

        # First, derivs for Image parameters (because 'images' comes
        # first in the tractor's parameters)
        parami = 0
        img = self.images[imgi]
        cat = self.catalog
        if not self.isParamFrozen('images'):
            for i in self.images.getThawedParamIndices():
                if i == imgi:
                    # Give the image a chance to compute its own derivs
                    derivs = img.getParamDerivatives(self, cat)
                    needj = []
                    for j,deriv in enumerate(derivs):
                        if deriv is None:
                            continue
                        if deriv is False:
                            needj.append(j)
                            continue
                        allderivs.append((parami + j, deriv))

                    if len(needj):
                        mod0 = self.getModelImage(i)
                        p0 = img.getParams()
                        ss = img.getStepSizes()
                    for j in needj:
                        step = ss[j]
                        img.setParam(j, p0[j]+step)
                        modj = self.getModelImage(i)
                        img.setParam(j, p0[j])
                        deriv = Patch(0, 0, (modj - mod0) / step)
                        allderivs.append((parami + j, deriv))

                parami += self.images[i].numberOfParams()

            assert(parami == self.images.numberOfParams())
            
        srcs = list(self.catalog.getThawedSources())
        for src in srcs:
            derivs = src.getParamDerivatives(img)
            for j,deriv in enumerate(derivs):
                if deriv is None:
                    continue
                allderivs.append((parami + j, deriv))
            parami += src.numberOfParams()

        assert(parami == self.numberOfParams())
        # Clip and unpack the (x0,y0,patch) elements for ease of use from C (ceres)
        # Also scale by -1 * inverse-error to get units of dChi here.
        ie = img.getInvError()
        H,W = img.shape
        chiderivs = []
        for ind,d in allderivs:
            d.clipTo(W,H)
            if d.patch is None:
                continue
            deriv = -1. * d.patch.astype(np.float64) * ie[d.getSlice()]
            chiderivs.append((ind, d.x0, d.y0, deriv))
            
        return chiderivs
    
            
    def _ceres_forced_photom(self, result, umodels,
                             imlist, mods0, scales,
                             skyderivs, minFlux,
                             BW, BH,
                             ceresType = np.float32,
                             nonneg = False,
                             wantims0 = True,
                             wantims1 = True,
                             negfluxval = None,
                             ):
        '''
        negfluxval: when 'nonneg' is set, the flux value to give sources that went
        negative in an unconstrained fit.
        '''
        from ceres import ceres_forced_phot

        t0 = Time()
        blocks = []
        blockstart = {}
        if BW is None:
            BW = 50
        if BH is None:
            BH = 50
        usedParamMap = {}
        nextparam = 0
        # umodels[ imagei, srci ] = Patch
        Nsky = 0
        Z = []
        if skyderivs is not None:
            # skyderivs = [ (param0:)[ (deriv,img), ], (param1:)[ (deriv,img), ], ...]
            # Reorg them to be in img-major order
            skymods = [ [] for im in imlist ]
            for skyd in skyderivs:
                for (deriv,img) in skyd:
                    imi = imlist.index(img)
                    skymods[imi].append(deriv)

            for mods,im,mod0 in zip(skymods, imlist, mods0):
                Z.append((mods, im, 1., mod0, Nsky))
                Nsky += len(mods)

        Z.extend(zip(umodels, imlist, scales, mods0, np.zeros(len(imlist),int)+Nsky))

        sky = (skyderivs is not None)

        for zi,(umods,img,scale,mod0, paramoffset) in enumerate(Z):
            H,W = img.shape
            if img in blockstart:
                (b0,nbw,nbh) = blockstart[img]
            else:
                # Dice up the image
                nbw = int(np.ceil(W / float(BW)))
                nbh = int(np.ceil(H / float(BH)))
                b0 = len(blocks)
                blockstart[img] = (b0, nbw, nbh)
                for iy in range(nbh):
                    for ix in range(nbw):
                        x0 = ix * BW
                        y0 = iy * BH
                        slc = (slice(y0, min(y0+BH, H)),
                               slice(x0, min(x0+BW, W)))
                        data = (x0, y0,
                                img.getImage()[slc].astype(ceresType),
                                mod0[slc].astype(ceresType),
                                img.getInvError()[slc].astype(ceresType))
                        blocks.append((data, []))

            for modi,umod in enumerate(umods):
                if umod is None:
                    continue
                # DEBUG
                if len(umod.shape) != 2:
                    print 'zi', zi
                    print 'modi', modi
                    print 'umod', umod
                umod.clipTo(W,H)
                umod.trimToNonZero()
                if umod.patch is None:
                    continue
                # Dice up the model
                ph,pw = umod.shape
                bx0 = np.clip(int(np.floor( umod.x0       / float(BW))),
                              0, nbw-1)
                bx1 = np.clip(int(np.ceil ((umod.x0 + pw) / float(BW))),
                              0, nbw-1)
                by0 = np.clip(int(np.floor( umod.y0       / float(BH))),
                              0, nbh-1)
                by1 = np.clip(int(np.ceil ((umod.y0 + ph) / float(BH))),
                              0, nbh-1)

                parami = paramoffset + modi
                if parami in usedParamMap:
                    ceresparam = usedParamMap[parami]
                else:
                    usedParamMap[parami] = nextparam
                    ceresparam = nextparam
                    nextparam += 1

                cmod = (umod.patch * scale).astype(ceresType)
                for by in range(by0, by1+1):
                    for bx in range(bx0, bx1+1):
                        bi = by * nbw + bx
                        #if type(umod.x0) != int or type(umod.y0) != int:
                        #    print 'umod:', umod.x0, umod.y0, type(umod.x0), type(umod.y0)
                        #    print 'umod:', umod
                        dd = (ceresparam, int(umod.x0), int(umod.y0), cmod)
                        blocks[b0 + bi][1].append(dd)
        logverb('forced phot: dicing up', Time()-t0)
                        
        rtn = []
        if wantims0:
            t0 = Time()
            params = self.getParams()
            result.ims0 = self._getims(params, imlist, umodels, mods0, scales,
                                       sky, minFlux, None)
            logverb('forced phot: ims0', Time()-t0)

        t0 = Time()
        fluxes = np.zeros(len(usedParamMap))
        print 'Ceres forced phot:'
        print len(blocks), ('image blocks (%ix%i), %i params' % (BW, BH, len(fluxes)))
        # init fluxes passed to ceres
        p0 = self.getParams()
        for i,k in usedParamMap.items():
            fluxes[k] = p0[i]

        nonneg = int(nonneg)
        if nonneg:
            # Initial run with nonneg=False, to get in the ballpark
            x = ceres_forced_phot(blocks, fluxes, 0)
            assert(x == 0)
            logverb('forced phot: ceres initial run', Time()-t0)
            t0 = Time()
            if negfluxval is not None:
                fluxes = np.maximum(fluxes, negfluxval)

        x = ceres_forced_phot(blocks, fluxes, nonneg)
        #print 'Ceres forced phot:', x
        logverb('forced phot: ceres', Time()-t0)

        t0 = Time()
        params = np.zeros(len(p0))
        for i,k in usedParamMap.items():
            params[i] = fluxes[k]
        self.setParams(params)
        logverb('forced phot: unmapping params:', Time()-t0)

        if wantims1:
            t0 = Time()
            result.ims1 = self._getims(params, imlist, umodels, mods0, scales,
                                       sky, minFlux, None)
            logverb('forced phot: ims1:', Time()-t0)
        return x

    def _get_fitstats(self, imsBest, srcs, imlist, umodsforsource,
                      umodels, scales, nilcounts, extras=[]):
        '''
        extras: [ ('key', [eim0,eim1,eim2]), ... ]

        Extra fields to add to the "FitStats" object, populated by
        taking the profile-weighted sum of 'eim*'.  The number of
        these images *must* match the number and shape of Tractor
        images.
        '''
        if extras is None:
            extras = []

        class FitStats(object):
            pass
        fs = FitStats()

        # Per-image stats:
        imchisq = []
        imnpix = []
        for img,mod,ie,chi,roi in imsBest:
            imchisq.append((chi**2).sum())
            imnpix.append((ie > 0).sum())
        fs.imchisq = np.array(imchisq)
        fs.imnpix = np.array(imnpix)

        # Per-source stats:
        
        # profile-weighted chi-squared (unit-model weighted chi-squared)
        fs.prochi2 = np.zeros(len(srcs))
        # profile-weighted number of pixels
        fs.pronpix = np.zeros(len(srcs))
        # profile-weighted sum of (flux from other sources / my flux)
        fs.profracflux = np.zeros(len(srcs))
        fs.proflux = np.zeros(len(srcs))
        # total number of pixels touched by this source
        fs.npix = np.zeros(len(srcs), int)

        for key,x in extras:
            setattr(fs, key, np.zeros(len(srcs)))

        # subtract sky from models before measuring others' flux
        # within my profile
        skies = []
        for tim,(img,mod,ie,chi,roi) in zip(imlist, imsBest):
            tim.getSky().addTo(mod, scale=-1.)
            skies.append(tim.getSky().val)
        fs.sky = np.array(skies)

        # Some fancy footwork to convert from umods to sources
        # (eg, composite galaxies that can have multiple umods)

        # keep reusing these arrays
        srcmods = [np.zeros_like(chi) for (img,mod,ie,chi,roi) in imsBest]
        
        # for each source:
        for si,uis in enumerate(umodsforsource):
            #print 'fit stats for source', si, 'of', len(umodsforsource)
            src = self.catalog[si]
            # for each image
            for imi,(umods,scale,tim,(img,mod,ie,chi,roi)) in enumerate(
                zip(umodels, scales, imlist, imsBest)):
                # just use 'scale'?
                pcal = tim.getPhotoCal()
                cc = [pcal.brightnessToCounts(b) for b in src.getBrightnesses()]
                csum = sum(cc)
                if csum == 0:
                    continue
                # Still want to measure objects with negative flux
                # if csum < nilcounts:
                #     continue
                
                srcmod = srcmods[imi]
                xlo,xhi,ylo,yhi = None,None,None,None
                # for each component (usually just one)
                for ui,counts in zip(uis, cc):
                    if counts == 0:
                        continue
                    um = umods[ui]
                    if um is None:
                        continue
                    # track total ROI.
                    x0,y0 = um.x0,um.y0
                    uh,uw = um.shape
                    if xlo is None or x0 < xlo:
                        xlo = x0
                    if xhi is None or x0 + uw > xhi:
                        xhi = x0 + uw
                    if ylo is None or y0 < ylo:
                        ylo = y0
                    if yhi is None or y0 + uh > yhi:
                        yhi = y0 + uh
                    # accumulate this unit-flux model into srcmod
                    (um * counts).addTo(srcmod)
                # Divide by total flux, not flux within this image; sum <= 1.
                if xlo is None or xhi is None or ylo is None or yhi is None:
                    continue
                slc = slice(ylo,yhi),slice(xlo,xhi)
                
                srcmod[slc] /= csum

                nz = np.flatnonzero((srcmod[slc] != 0) * (ie[slc] > 0))
                if len(nz) == 0:
                    srcmod[slc] = 0.
                    continue

                fs.prochi2[si] += np.sum(np.abs(srcmod[slc].flat[nz]) * chi[slc].flat[nz]**2)
                fs.pronpix[si] += np.sum(np.abs(srcmod[slc].flat[nz]))
                # (mod - srcmod*csum) is the model for everybody else
                fs.profracflux[si] += np.sum((np.abs(mod[slc] / csum - srcmod[slc]) * np.abs(srcmod[slc])).flat[nz])
                # scale to nanomaggies, weight by profile
                fs.proflux[si] += np.sum((np.abs((mod[slc] - srcmod[slc]*csum) / scale) * np.abs(srcmod[slc])).flat[nz])
                fs.npix[si] += len(nz)

                for key,extraims in extras:
                    x = getattr(fs, key)
                    x[si] += np.sum(np.abs(srcmod[slc].flat[nz]) * extraims[imi][slc].flat[nz])

                srcmod[slc] = 0.

        # re-add sky
        for tim,(img,mod,ie,chi,roi) in zip(imlist, imsBest):
            tim.getSky().addTo(mod)

        return fs
    
    def _get_iv(self, sky, skyvariance, Nsky, skyderivs, srcs,
                imlist, umodels, scales):
        if sky and skyvariance:
            NS = Nsky
        else:
            NS = 0
        IV = np.zeros(len(srcs) + NS)
        if sky and skyvariance:
            for di,(dsky,tim) in enumerate(skyderivs):
                ie = tim.getInvError()
                if dsky.shape == tim.shape:
                    dchi2 = np.sum((dsky.patch * ie)**2)
                else:
                    mm = np.zeros(tim.shape)
                    dsky.addTo(mm)
                    dchi2 = np.sum((mm * ie)**2)
                IV[di] = dchi2
        for i,(tim,umods,scale) in enumerate(zip(imlist, umodels, scales)):
            mm = np.zeros(tim.shape)
            ie = tim.getInvError()
            for ui,um in enumerate(umods):
                if um is None:
                    continue
                #print 'deriv: sum', um.patch.sum(), 'scale', scale, 'shape', um.shape,
                um.addTo(mm)
                x0,y0 = um.x0,um.y0
                uh,uw = um.shape
                slc = slice(y0, y0+uh), slice(x0,x0+uw)
                dchi2 = np.sum((mm[slc] * scale * ie[slc]) ** 2)
                IV[NS + ui] += dchi2
                mm[slc] = 0.
        return IV
    
    def _get_umodels(self, srcs, imgs, minsb, rois):
        #
        # Here we build up the "umodels" nested list, which has shape
        # (if it were a numpy array) of (len(images), len(srcs))
        # where each element is None, or a Patch with the unit-flux model
        # of that source in that image.
        #
        umodels = []
        umodtosource = {}
        umodsforsource = [[] for s in srcs]

        for i,img in enumerate(imgs):
            umods = []
            pcal = img.getPhotoCal()
            nvalid = 0
            nallzero = 0
            nzero = 0
            if rois is not None:
                roi = rois[i]
                y0 = roi[0].start
                x0 = roi[1].start
            else:
                x0 = y0 = 0
            for si,src in enumerate(srcs):
                counts = sum([pcal.brightnessToCounts(b) for b in src.getBrightnesses()])
                if counts <= 0:
                    mv = 1e-3
                else:
                    # we will scale the PSF by counts and we want that
                    # scaled min val to be less than minsb
                    mv = minsb / counts
                ums = src.getUnitFluxModelPatches(img, minval=mv)

                isvalid = False
                isallzero = False

                for ui,um in enumerate(ums):
                    if um is None:
                        continue
                    if um.patch is None:
                        continue
                    um.x0 -= x0
                    um.y0 -= y0
                    isvalid = True
                    if um.patch.sum() == 0:
                        nzero += 1
                    else:
                        isallzero = False

                    if not np.all(np.isfinite(um.patch)):
                        print 'Non-finite patch for source', src
                        print 'In image', img
                        assert(False)

                # first image only:
                if i == 0:
                    for ui in range(len(ums)):
                        umodtosource[len(umods) + ui] = si
                        umodsforsource[si].append(len(umods) + ui)

                umods.extend(ums)

                if isvalid:
                    nvalid += 1
                    if isallzero:
                        nallzero += 1
            #print 'Img', i, 'has', nvalid, 'of', len(srcs), 'sources'
            #print '  ', nallzero, 'of which are all zero'
            #print '  ', nzero, 'components are zero'
            umodels.append(umods)
        return umodels, umodtosource, umodsforsource

    def _getims(self, fluxes, imgs, umodels, mod0, scales, sky, minFlux, rois):
        ims = []
        for i,(img,umods,m0,scale
               ) in enumerate(zip(imgs, umodels, mod0, scales)):
            roi = None
            if rois:
                roi = rois[i]
            mod = m0.copy()
            assert(np.all(np.isfinite(mod)))
            if sky:
                img.getSky().addTo(mod)
                assert(np.all(np.isfinite(mod)))
            for f,um in zip(fluxes,umods):
                if um is None:
                    continue
                if um.patch is None:
                    continue
                if minFlux is not None:
                    f = max(f, minFlux)
                counts = f * scale
                if counts == 0.:
                    continue
                if not np.isfinite(counts):
                    print 'Warning: counts', counts, 'f', f, 'scale', scale
                assert(np.isfinite(counts))
                assert(np.all(np.isfinite(um.patch)))
                #print 'Adding umod', um, 'with counts', counts, 'to mod', mod.shape
                (um * counts).addTo(mod)

            ie = img.getInvError()
            im = img.getImage()
            if roi is not None:
                ie = ie[roi]
                im = ie[roi]
            chi = (im - mod) * ie

            # DEBUG
            if not np.all(np.isfinite(chi)):
                print 'Chi has non-finite pixels:'
                print np.unique(chi[np.logical_not(np.isfinite(chi))])
                print 'Inv error range:', ie.min(), ie.max()
                print 'All finite:', np.all(np.isfinite(ie))
                print 'Mod range:', mod.min(), mod.max()
                print 'All finite:', np.all(np.isfinite(mod))
                print 'Img range:', im.min(), im.max()
                print 'All finite:', np.all(np.isfinite(im))
            assert(np.all(np.isfinite(chi)))
            ims.append((im, mod, ie, chi, roi))
        return ims
    
    def _lnp_for_update(self, mod0, imgs, umodels, X, alpha, p0, rois,
                        scales, p0sky, Xsky, priors, sky, minFlux):
        if X is None:
            pa = p0
        else:
            pa = [p + alpha * d for p,d in zip(p0, X)]
        chisq = 0.
        chis = []
        if Xsky is not None:
            self.images.setParams([p + alpha * d for p,d in zip(p0sky, Xsky)])
        # Recall that "umodels" is a full matrix (shape (Nimage,
        # Nsrcs)) of patches, so we just go through each image,
        # ignoring None entries and building up model images from
        # the scaled unit-flux patches.

        ims = self._getims(pa, imgs, umodels, mod0, scales, sky, minFlux, rois)
        for nil,nil,nil,chi,roi in ims:
            chis.append(chi)
            chisq += (chi.astype(np.float64)**2).sum()
        lnp = -0.5 * chisq
        if priors:
            lnp += self.getLogPrior()
        return lnp, chis, ims

    def _lsqr_forced_photom(self, result, derivs, mod0, imgs, umodels, rois, scales,
                            priors, sky, minFlux, justims0, subimgs,
                            damp, alphas, Nsky, mindlnp, shared_params,
                            use_tsnnls):
        # About rois and derivs: we call
        #   getUpdateDirection(derivs, ..., chiImages=[chis])
        # And this uses the "img" objects in "derivs" to decide on the region
        # that is being optimized; the number of rows = total number of pixels.
        # We have to make sure that "chiImages" matches that size.
        #
        # We shift the unit-flux models (above, um.x0 -= x0) to be
        # relative to the ROI.

        # debugging images
        ims0 = None
        imsBest = None

        lnp0 = None
        chis0 = None
        quitNow = False

        ## FIXME -- this should depend on the PhotoCal scalings!
        damp0 = 1e-3
        damping = damp

        while True:
            # A flag to try again even if the lnprob got worse
            tryAgain = False

            p0 = self.getParams()
            if sky:
                p0sky = p0[:Nsky]
                p0 = p0[Nsky:]

            if lnp0 is None:
                t0 = Time()
                lnp0,chis0,ims0 = self._lnp_for_update(
                    mod0, imgs, umodels, None, None, p0, rois, scales,
                    None, None, priors, sky, minFlux)
                logverb('forced phot: initial lnp = ', lnp0, 'took', Time()-t0)

            if justims0:
                result.lnp0 = lnp0
                result.chis0 = chis0
                result.ims0 = ims0
                return

            # print 'Starting opt loop with'
            # print '  p0', p0
            # print '  lnp0', lnp0
            # print '  chisqs', [(chi**2).sum() for chi in chis0]
            # print 'chis0:', chis0

            # Ugly: getUpdateDirection calls self.getImages(), and
            # ASSUMES they are the same as the images referred-to in
            # the "derivs", to figure out which chi image goes with
            # which image.  Temporarily set images = subimages
            if rois is not None:
                realims = self.images
                self.images = subimgs

            logverb('forced phot: getting update with damp=', damping)
            t0 = Time()
            X = self.getUpdateDirection(derivs, damp=damping, priors=priors,
                                        scale_columns=False, chiImages=chis0,
                                        shared_params=shared_params,
                                        use_tsnnls=use_tsnnls)
            topt = Time()-t0
            logverb('forced phot: opt:', topt)
            #print 'forced phot: update', X
            if rois is not None:
                self.images = realims

            if len(X) == 0:
                print 'Error getting update direction'
                break
            
            ## tryUpdates():
            if alphas is None:
                # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
                alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

            if sky:
                # Split the sky-update parameters from the source parameters
                Xsky = X[:Nsky]
                X = X[Nsky:]
            else:
                p0sky = Xsky = None

            # Check whether the update produces all fluxes above the
            # minimums: if so we should be able to take the step with
            # alpha=1 and quit.

            if (minFlux is None) or np.all((p0 + X) >= minFlux):
                #print 'Update produces non-negative fluxes; accepting with alpha=1'
                alphas = [1.]
                quitNow = True
            else:
                print 'Some too-negative fluxes requested:'
                print 'Fluxes:', p0
                print 'Update:', X
                print 'Total :', p0+X
                print 'MinFlux:', minFlux
                if damp == 0.0:
                    damping = damp0
                    damp0 *= 10.
                    print 'Setting damping to', damping
                    if damp0 < 1e3:
                        tryAgain = True

            lnpBest = lnp0
            alphaBest = None
            chiBest = None

            for alpha in alphas:
                t0 = Time()
                lnp,chis,ims = self._lnp_for_update(
                    mod0, imgs, umodels, X, alpha, p0, rois, scales,
                    p0sky, Xsky, priors, sky, minFlux)
                logverb('Forced phot: stepped with alpha', alpha,
                        'for lnp', lnp, ', dlnp', lnp-lnp0)
                logverb('Took', Time()-t0)
                if lnp < (lnpBest - 1.):
                    logverb('lnp', lnp, '< lnpBest-1', lnpBest-1.)
                    break
                if lnp > lnpBest:
                    alphaBest = alpha
                    lnpBest = lnp
                    chiBest = chis
                    imsBest = ims

            if alphaBest is not None:
                # Clamp fluxes up to zero
                if minFlux is not None:
                    pa = [max(minFlux, p + alphaBest * d) for p,d in zip(p0, X)]
                else:
                    pa = [p + alphaBest * d for p,d in zip(p0, X)]
                self.catalog.setParams(pa)

                if sky:
                    self.images.setParams([p + alpha * d for p,d
                                           in zip(p0sky, Xsky)])

                dlogprob = lnpBest - lnp0
                alpha = alphaBest
                lnp0 = lnpBest
                chis0 = chiBest
                # print 'Accepting alpha =', alpha
                # print 'new lnp0', lnp0
                # print 'new chisqs', [(chi**2).sum() for chi in chis0]
                # print 'new params', self.getParams()
            else:
                dlogprob = 0.
                alpha = 0.

                ### ??
                if sky:
                    # Revert -- recall that we change params while probing in
                    # lnpForUpdate()
                    self.images.setParams(p0sky)

            #tstep = Time() - t0
            #print 'forced phot: line search:', tstep
            #print 'forced phot: alpha', alphaBest, 'for delta-lnprob', dlogprob
            if dlogprob < mindlnp:
                if not tryAgain:
                    break

            if quitNow:
                break
        result.ims0 = ims0
        result.ims1 = imsBest
    
    def optimize_forced_photometry(self, alphas=None, damp=0, priors=False,
                                   minsb=0.,
                                   mindlnp=1.,
                                   rois=None,
                                   sky=False,
                                   minFlux=None,
                                   fitstats=False,
                                   fitstat_extras=None,
                                   justims0=False,
                                   variance=False,
                                   skyvariance=False,
                                   shared_params=True,
                                   use_tsnnls=False,
                                   use_ceres=False,
                                   BW=None, BH=None,
                                   nonneg=False,
                                   #nilcounts=1e-6,
                                   nilcounts=-1e30,
                                   wantims=True,
                                   negfluxval=None,
                                   ):
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

        PRIORS probably don't work because we don't setParams() when evaluating
        likelihood or prior!
        '''
        from basics import LinearPhotoCal, ShiftedWcs

        result = OptResult()

        assert(not priors)
        scales = []
        imgs = self.getImages()
        for img in imgs:
            assert(isinstance(img.getPhotoCal(), LinearPhotoCal))
            scales.append(img.getPhotoCal().getScale())

        if rois is not None:
            assert(len(rois) == len(imgs))

        # HACK -- if sky=True, assume we are fitting the sky in ALL images.
        # We could ask which ones are thawed...
        if sky:
            for img in imgs:
                # FIXME -- would be nice to allow multi-param linear sky models
                assert(img.getSky().numberOfParams() == 1)

        Nsourceparams = self.catalog.numberOfParams()
        srcs = list(self.catalog.getThawedSources())

        t0 = Time()
        (umodels, umodtosource, umodsforsource
         )= self._get_umodels(srcs, imgs, minsb, rois)
        for umods in umodels:
            assert(len(umods) == Nsourceparams)
        tmods = Time()-t0
        logverb('forced phot: getting unit-flux models:', tmods)

        subimgs = []
        if rois is not None:
            for i,img in enumerate(imgs):
                roi = rois[i]
                y0 = roi[0].start
                x0 = roi[1].start
                subwcs = ShiftedWcs(img.wcs, x0, y0)
                subimg = Image(data=img.data[roi], inverr=img.inverr[roi],
                               psf=img.psf, wcs=subwcs, sky=img.sky,
                               photocal=img.photocal, name=img.name)
                subimgs.append(subimg)
            imlist = subimgs
        else:
            imlist = imgs
        
        t0 = Time()
        fsrcs = list(self.catalog.getFrozenSources())
        mod0 = []
        for img in imlist:
            # "sky = not sky": I'm not just being contrary :)
            # If we're fitting sky, we'll do a setParams() and get the
            # sky models to render themselves when evaluating lnProbs,
            # rather than pre-computing the nominal value here and
            # then computing derivatives.
            mod0.append(self.getModelImage(img, fsrcs, minsb=minsb, sky=not sky))
        tmod = Time() - t0
        logverb('forced phot: getting frozen-source model:', tmod)

        skyderivs = None
        if sky:
            t0 = Time()
            # build the derivative list as required by getUpdateDirection:
            #    (param0) ->  [  (deriv, img), (deriv, img), ...   ], ... ],
            skyderivs = []
            for img in imlist:
                dskys = img.getSky().getParamDerivatives(self, img, None)
                for dsky in dskys:
                    skyderivs.append([(dsky, img)])
            Nsky = len(skyderivs)
            assert(Nsky == self.images.numberOfParams())
            assert(Nsky + Nsourceparams == self.numberOfParams())
            logverb('forced phot: sky derivs', Time()-t0)
        else:
            Nsky = 0

        wantims0 = wantims1 = wantims
        if fitstats:
            wantims1 = True

        if use_ceres:
            x = self._ceres_forced_photom(result, umodels, imlist, mod0, 
                                          scales, skyderivs, minFlux, BW, BH,
                                          nonneg=nonneg, wantims0=wantims0,
                                          wantims1=wantims1, negfluxval=negfluxval)
            result.ceres_status = x

        else:
            t0 = Time()
            derivs = [[] for i in range(Nsourceparams)]
            for i,(tim,umods,scale) in enumerate(zip(imlist, umodels, scales)):
                for um,dd in zip(umods, derivs):
                    if um is None:
                        continue
                    dd.append((um * scale, tim))
            logverb('forced phot: derivs', Time()-t0)
            if sky:
                # print 'Catalog params:', self.catalog.numberOfParams()
                # print 'Image params:', self.images.numberOfParams()
                # print 'Total # params:', self.numberOfParams()
                # print 'cat derivs:', len(derivs)
                # print 'sky derivs:', len(skyderivs)
                # print 'total # derivs:', len(derivs) + len(skyderivs)
                # Sky derivatives are part of the image derivatives, so go first in
                # the derivative list.
                derivs = skyderivs + derivs
            assert(len(derivs) == self.numberOfParams())
            self._lsqr_forced_photom(
                result, derivs, mod0, imgs, umodels, rois, scales, priors, sky,
                minFlux, justims0, subimgs, damp, alphas, Nsky, mindlnp,
                shared_params, use_tsnnls)
                
        if variance:
            # Inverse variance
            t0 = Time()
            result.IV = self._get_iv(sky, skyvariance, Nsky, skyderivs, srcs,
                                     imlist, umodels, scales)
            logverb('forced phot: variance:', Time()-t0)

        imsBest = getattr(result, 'ims1', None)
        if fitstats and imsBest is None:
            print 'Warning: fit stats not computed because imsBest is None'
            result.fitstats = None
        elif fitstats:
            t0 = Time()
            result.fitstats = self._get_fitstats(imsBest, srcs, imlist, umodsforsource,
                                                 umodels, scales, nilcounts, extras=fitstat_extras)
            logverb('forced phot: fit stats:', Time()-t0)
        return result


    def optimize(self, alphas=None, damp=0, priors=True, scale_columns=True,
                 shared_params=True, variance=False, just_variance=False):
        '''
        Performs *one step* of linearized least-squares + line search.
        
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
        logverb(self.getName()+': Finding derivs...')
        t0 = Time()
        allderivs = self.getDerivs()
        tderivs = Time()-t0
        #print Time() - t0
        #print 'allderivs:', allderivs
        #for d in allderivs:
        #   for (p,im) in d:
        #       print 'patch mean', np.mean(p.patch)
        logverb('Finding optimal update direction...')
        t0 = Time()
        X = self.getUpdateDirection(allderivs, damp=damp, priors=priors,
                                    scale_columns=scale_columns,
                                    shared_params=shared_params,
                                    variance=variance)
        if variance:
            if len(X) == 0:
                return 0, X, 0, None
            X,var = X
            if just_variance:
                return var
        #print Time() - t0
        topt = Time()-t0
        #print 'X:', X
        if len(X) == 0:
            return 0, X, 0.
        logverb('X: len', len(X), '; non-zero entries:', np.count_nonzero(X))
        logverb('Finding optimal step size...')
        t0 = Time()
        (dlogprob, alpha) = self.tryUpdates(X, alphas=alphas)
        tstep = Time() - t0
        logverb('Finished opt2.')
        logverb('  alpha =',alpha)
        logverb('  Tderiv', tderivs)
        logverb('  Topt  ', topt)
        logverb('  Tstep ', tstep)
        if variance:
            return dlogprob, X, alpha, var
        return dlogprob, X, alpha

    def getParameterScales(self):
        print self.getName()+': Finding derivs...'
        allderivs = self.getDerivs()
        print 'Finding column scales...'
        s = self.getUpdateDirection(allderivs, scales_only=True)
        return s

    def tryUpdates(self, X, alphas=None):
        if alphas is None:
            # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        pBefore = self.getLogProb()
        logverb('  log-prob before:', pBefore)
        pBest = pBefore
        alphaBest = None
        p0 = self.getParams()
        for alpha in alphas:
            logverb('  Stepping with alpha =', alpha)
            pa = [p + alpha * d for p,d in zip(p0, X)]
            self.setParams(pa)
            pAfter = self.getLogProb()
            logverb('  Log-prob after:', pAfter)
            logverb('  delta log-prob:', pAfter - pBefore)

            if not np.isfinite(pAfter):
                logmsg('  Got bad log-prob', pAfter)
                break

            if pAfter < (pBest - 1.):
                break

            if pAfter > pBest:
                alphaBest = alpha
                pBest = pAfter
        
        if alphaBest is None or alphaBest == 0:
            print "Warning: optimization is borking"
            print "Parameter direction =",X
            print "Parameters, step sizes, updates:"
            for n,p,s,x in zip(self.getParamNames(), self.getParams(), self.getStepSizes(), X):
                print n, '=', p, '  step', s, 'update', x
        if alphaBest is None:
            self.setParams(p0)
            return 0, 0.

        logmsg('  Stepping by', alphaBest, 'for delta-logprob', pBest - pBefore)
        pa = [p + alphaBest * d for p,d in zip(p0, X)]
        self.setParams(pa)
        return pBest - pBefore, alphaBest


    def getDerivs(self):
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
        if self.is_multiproc():
            # First, derivs for Image parameters (because 'images'
            # comes first in the tractor's parameters)
            if self.isParamFrozen('images'):
                ims = []
                imjs = []
            else:
                imjs = [i for i in self.images.getThawedParamIndices()]
                ims = [self.images[j] for j in imjs]
            imderivs = self._map(getimagederivs, [(imj, im, self, allsrcs)
                                                  for im,imj in zip(ims, imjs)])

            needimjs = []
            needims = []
            needparams = []
    
            for derivs,im,imj in zip(imderivs, ims, imjs):
                need = []
                for k,d in enumerate(derivs):
                    if d is False:
                        need.append(k)
                if len(need):
                    needimjs.append(imj)
                    needims.append(im)
                    needparams.append(need)
                        
            # initial models...
            logverb('Getting', len(needimjs), 'initial models for image derivatives')
            mod0s = self._map_async(getmodelimagefunc, [(self, imj) for imj in needimjs])
            # stepping each (needed) param...
            args = []
            # for j,im in enumerate(ims):
            #   p0 = im.getParams()
            #   #print 'Image', im
            #   #print 'Step sizes:', im.getStepSizes()
            #   #print 'p0:', p0
            #   for k,step in enumerate(im.getStepSizes()):
            #       args.append((self, j, k, p0[k], step))
            for im,imj,params in zip(needims, needimjs, needparams):
                p0 = im.getParams()
                ss = im.getStepSizes()
                for i in params:
                    args.append((self, imj, i, p0[i], ss[i]))
            # reverse the args so we can pop() below.
            logverb('Stepping in', len(args), 'model parameters for derivatives')
            mod1s = self._map_async(getmodelimagestep, reversed(args))

            # Next, derivs for the sources.
            args = []
            for j,src in enumerate(srcs):
                for i,img in enumerate(self.images):
                    args.append((src, img))
            sderivs = self._map_async(getsrcderivs, reversed(args))
    
            # Wait for and unpack the image derivatives...
            mod0s = mod0s.get()
            mod1s = mod1s.get()
            # convert to a imj->mod0 map
            assert(len(mod0s) == len(needimjs))
            mod0s = dict(zip(needimjs, mod0s))
    
            for derivs,im,imj in zip(imderivs, ims, imjs):
                for k,d in enumerate(derivs):
                    if d is False:
                        mod0 = mod0s[imj]
                        nm = im.getParamNames()[k]
                        step = im.getStepSizes()[k]
                        mod1 = mod1s.pop()
                        d = Patch(0, 0, (mod1 - mod0) / step)
                        d.name = 'd(im%i)/d(%s)' % (j,nm)
                    allderivs.append([(d, im)])
            # popped all
            assert(len(mod1s) == 0)
    
            # for i,(j,im) in enumerate(zip(imjs,ims)):
            #   mod0 = mod0s[i]
            #   p0 = im.getParams()
            #   for k,(nm,step) in enumerate(zip(im.getParamNames(), im.getStepSizes())):
            #       mod1 = mod1s.pop()
            #       deriv = Patch(0, 0, (mod1 - mod0) / step)
            #       deriv.name = 'd(im%i)/d(%s)' % (j,nm)
            #       allderivs.append([(deriv, im)])
    
            # Wait for source derivs...
            sderivs = sderivs.get()
    
            for j,src in enumerate(srcs):
                srcderivs = [[] for i in range(src.numberOfParams())]
                for i,img in enumerate(self.images):
                    # Get derivatives (in this image) of params
                    derivs = sderivs.pop()
                    # derivs is a list of Patch objects or None, one per parameter.
                    assert(len(derivs) == src.numberOfParams())
                    for k,deriv in enumerate(derivs):
                        if deriv is None:
                            continue
                        # if deriv is False:
                        #     # compute it now
                        #   (THIS IS WRONG; mod0s only has initial models for images
                        #    that need it -- those that are unfrozen)
                        #     mod0 = mod0s[i]
                        #     nm = src.getParamNames()[k]
                        #     step = src.getStepSizes()[k]
                        #     mod1 = tra.getModelImage(img)
                        #     d = Patch(0, 0, (mod1 - mod0) / step)
                        #     d.name = 'd(src(im%i))/d(%s)' % (i, nm)
                        #     deriv = d
                            
                        if not np.all(np.isfinite(deriv.patch.ravel())):
                            print 'Derivative for source', src
                            print 'deriv index', i
                            assert(False)
                        srcderivs[k].append((deriv, img))
                allderivs.extend(srcderivs)
            
        else:
            
            if not self.isParamFrozen('images'):
                for i in self.images.getThawedParamIndices():
                    img = self.images[i]
                    derivs = img.getParamDerivatives(self, allsrcs)
                    mod0 = None
                    for di,deriv in enumerate(derivs):
                        if deriv is False:
                            if mod0 is None:
                                mod0 = self.getModelImage(img)
                                p0 = img.getParams()
                                stepsizes = img.getStepSizes()
                                paramnames = img.getParamNames()
                            oldval = img.setParam(di, p0[di] + stepsizes[di])
                            mod = self.getModelImage(img)
                            img.setParam(di, oldval)
                            deriv = Patch(0, 0, (mod - mod0) / stepsizes[di])
                            deriv.name = 'd(im%i)/d(%s)' % (i, paramnames[di])
                        allderivs.append([(deriv, img)])
                    del mod0

            for src in srcs:
                srcderivs = [[] for i in range(src.numberOfParams())]
                for img in self.images:
                    derivs = src.getParamDerivatives(img)
                    for k,deriv in enumerate(derivs):
                        if deriv is None:
                            continue
                        srcderivs[k].append((deriv, img))
                allderivs.extend(srcderivs)
            #print 'allderivs:', len(allderivs)
            #print 'N params:', self.numberOfParams()

        assert(len(allderivs) == self.numberOfParams())
        return allderivs

    def getUpdateDirection(self, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           use_tsnnls=False,
                           use_ceres=False,
                           get_A_matrix=False):

        # allderivs: [
        #    (param0:)  [  (deriv, img), (deriv, img), ... ],
        #    (param1:)  [],
        #    (param2:)  [  (deriv, img), ],
        # ]
        # The "img"s may repeat
        # "deriv" are Patch objects.

        # Each position in the "allderivs" array corresponds to a
        # model parameter that we are optimizing

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

        # Parameters to optimize go in the columns of matrix A
        # Pixels go in the rows.

        if shared_params:
            # Find shared parameters
            p0 = self.getParams()
            self.setParams(np.arange(len(p0)))
            p1 = self.getParams()
            self.setParams(p0)
            U,I = np.unique(p1, return_inverse=True)
            logverb(len(p0), 'params;', len(U), 'unique')
            paramindexmap = I
            #print 'paramindexmap:', paramindexmap
            #print 'p1:', p1
            
        # Build the sparse matrix of derivatives:
        sprows = []
        spcols = []
        spvals = []

        # Keep track of row offsets for each image.
        imgoffs = {}
        nextrow = 0
        for param in allderivs:
            for deriv,img in param:
                if img in imgoffs:
                    continue
                imgoffs[img] = nextrow
                #print 'Putting image', img.name, 'at row offset', nextrow
                nextrow += img.numberOfPixels()
        Nrows = nextrow
        del nextrow
        Ncols = len(allderivs)

        # FIXME -- shared_params should share colscales!
        
        colscales = np.ones(len(allderivs))
        for col, param in enumerate(allderivs):
            RR = []
            VV = []
            WW = []
            for (deriv, img) in param:
                inverrs = img.getInvError()
                (H,W) = img.shape
                row0 = imgoffs[img]
                deriv.clipTo(W, H)
                pix = deriv.getPixelIndices(img)
                if len(pix) == 0:
                    #print 'This param does not influence this image!'
                    continue

                assert(np.all(pix < img.numberOfPixels()))
                # (grab non-zero indices)
                dimg = deriv.getImage()
                nz = np.flatnonzero(dimg)
                #print '  source', j, 'derivative', p, 'has', len(nz), 'non-zero entries'
                if len(nz) == 0:
                    continue
                rows = row0 + pix[nz]
                #print 'Adding derivative', deriv.getName(), 'for image', img.name
                vals = dimg.flat[nz]
                w = inverrs[deriv.getSlice(img)].flat[nz]
                assert(vals.shape == w.shape)
                if not scales_only:
                    RR.append(rows)
                    VV.append(vals)
                    WW.append(w)

            # massage, re-scale, and clean up matrix elements
            if len(VV) == 0:
                continue
            rows = np.hstack(RR)
            VV = np.hstack(VV)
            WW = np.hstack(WW)
            #vals = np.hstack(VV) * np.hstack(WW)
            #print 'VV absmin:', np.min(np.abs(VV))
            #print 'WW absmin:', np.min(np.abs(WW))
            #print 'VV type', VV.dtype
            #print 'WW type', WW.dtype
            vals = VV * WW
            #print 'vals absmin:', np.min(np.abs(vals))
            #print 'vals absmax:', np.max(np.abs(vals))
            #print 'vals type', vals.dtype

            # shouldn't be necessary since we check len(nz)>0 above
            #if len(vals) == 0:
            #   continue
            mx = np.max(np.abs(vals))
            if mx == 0:
                logmsg('mx == 0:', len(np.flatnonzero(VV)), 'of', len(VV), 'non-zero derivatives,',
                       len(np.flatnonzero(WW)), 'of', len(WW), 'non-zero weights;',
                       len(np.flatnonzero(vals)), 'non-zero products')
                continue
            # MAGIC number: near-zero matrix elements -> 0
            # 'mx' is the max value in this column.
            FACTOR = 1.e-10
            I = (np.abs(vals) > (FACTOR * mx))
            rows = rows[I]
            vals = vals[I]
            scale = np.sqrt(np.dot(vals, vals))
            colscales[col] = scale
            #logverb('Column', col, 'scale:', scale)
            if scales_only:
                continue

            sprows.append(rows)
            spcols.append(col)
            #c = np.empty_like(rows)
            #c[:] = col
            #spcols.append(c)
            if scale_columns:
                spvals.append(vals / scale)
            else:
                spvals.append(vals)
                
        if scales_only:
            return colscales

        b = None
        if priors:
            # We don't include the priors in the "colscales"
            # computation above, mostly because the priors are
            # returned as sparse additions to the matrix, and not
            # necessarily column-oriented the way the other params
            # are.  It would be possible to make it work, but dstn is
            # not convinced it's worth the effort right now.
            X = self.getLogPriorDerivatives()
            if X is not None:
                rA,cA,vA,pb = X

                sprows.extend([ri + Nrows for ri in rA])
                spcols.extend(cA)
                spvals.extend([vi / colscales[ci] for vi,ci in zip(vA,cA)])
                oldnrows = Nrows
                nr = listmax(rA, -1) + 1
                Nrows += nr
                logverb('Nrows was %i, added %i rows of priors => %i' % (oldnrows, nr, Nrows))
                # if len(cA) == 0:
                #     Ncols = 0
                # else:
                #     Ncols = 1 + max(cA)

                b = np.zeros(Nrows)
                b[oldnrows:] = np.hstack(pb)

        if len(spcols) == 0:
            logverb("len(spcols) == 0")
            return []

        # 'spcols' has one integer per 'sprows' block.
        # below we hstack the rows, but before doing that, remember how
        # many rows are in each chunk.
        spcols = np.array(spcols)
        nrowspercol = np.array([len(x) for x in sprows])
        
        if shared_params:
            # Apply shared parameter map
            #print 'Before applying shared parameter map:'
            #print 'spcols:', len(spcols), 'elements'
            #print '  ', len(set(spcols)), 'unique'
            spcols = paramindexmap[spcols]
            #print 'After:'
            #print 'spcols:', len(spcols), 'elements'
            #print '  ', len(set(spcols)), 'unique'
            Ncols = np.max(spcols) + 1
            logverb('Set Ncols=', Ncols)

        # b = chi
        #
        # FIXME -- we could be much smarter here about computing
        # just the regions we need!
        #
        if b is None:
            b = np.zeros(Nrows)

        chimap = {}
        if chiImages is not None:
            for img,chi in zip(self.getImages(), chiImages):
                chimap[img] = chi
                
        # iterating this way avoids setting the elements more than once
        for img,row0 in imgoffs.items():
            chi = chimap.get(img, None)
            if chi is None:
                #print 'computing chi image'
                chi = self.getChiImage(img=img)
            chi = chi.ravel()
            NP = len(chi)
            # we haven't touched these pix before
            assert(np.all(b[row0 : row0 + NP] == 0))
            assert(np.all(np.isfinite(chi)))
            #print 'Setting [%i:%i) from chi img' % (row0, row0+NP)
            b[row0 : row0 + NP] = chi
        ###### Zero out unused rows -- FIXME, is this useful??
        # print 'Nrows', Nrows, 'vs len(urows)', len(urows)
        # bnz = np.zeros(Nrows)
        # bnz[urows] = b[urows]
        # print 'b', len(b), 'vs bnz', len(bnz)
        # b = bnz
        assert(np.all(np.isfinite(b)))

        use_lsqr = True

        if use_ceres:
            # Solver::Options::linear_solver_type to SPARSE_NORMAL_CHOLESKY 
            pass
        
        if use_tsnnls:
            use_lsqr = False
            from tsnnls import tsnnls_lsqr
            #logmsg('TSNNLS: %i cols (%i unique), %i elements' %
            #       (Ncols, len(ucols), len(spvals)))
            print 'spcols:', spcols.shape, spcols.dtype
            #print 'spvals:', spvals.shape, spvals.dtype
            print 'spvals:', len(spvals), 'chunks'
            print '  total', sum(len(x) for x in spvals), 'elements'
            print 'b:', b.shape, b.dtype
            #print 'sprows:', sprows.shape, sprows.dtype
            print 'sprows:', len(sprows), 'chunks'
            print '  total', sum(len(x) for x in sprows), 'elements'

            ucols,colI = np.unique(spcols, return_inverse=True)
            J = np.argsort(colI)

            sorted_cols = colI[J]
            nel = [len(sprows[j]) for j in J]
            sorted_rows = np.hstack([sprows[j].astype(np.int32) for j in J])
            sorted_vals = np.hstack([spvals[j] for j in J])
            #Nelements = sum(len(x) for x in spvals)
            Nelements = sum(nel)
            
            colinds = np.zeros(len(ucols)+1, np.int32)
            for c,n in zip(sorted_cols, nel):
                colinds[c+1] += n
            colinds = np.cumsum(colinds).astype(np.int32)
            assert(colinds[-1] == Nelements)
            #colinds = colinds[:-1]
            
            # print 'sorted_cols:', sorted_cols
            # print 'column inds:', colinds
            # print 'sorted_rows:', sorted_rows
            # print 'sorted_vals:', sorted_vals
            print 'colinds:', colinds.shape, colinds.dtype
            print 'rows:', sorted_rows.shape, sorted_rows.dtype
            print 'vals:', sorted_vals.shape, sorted_vals.dtype

            # compress b and rows?
            urows,K = np.unique(sorted_rows, return_inverse=True)
            bcomp = b[urows]
            rowcomp = K.astype(np.int32)

            # print 'Compressed rows:', rowcomp
            # print 'Compressed b:', bcomp
            # for c,(i0,i1) in enumerate(zip(colinds, colinds[1:])):
            #     print 'Column', c, 'goes from', i0, 'to', i1
            #     print 'rows:', rowcomp[i0:i1]
            #     print 'vals:', sorted_vals[i0:i1]
            
            nrcomp = len(urows)
            
            #tsnnls_lsqr(colinds, sorted_rows, sorted_vals,
            #            b, Nrows, Nelements)

            X = tsnnls_lsqr(colinds, rowcomp, sorted_vals,
                            bcomp, nrcomp, Nelements)
            print 'Got TSNNLS result:', X

            # Undo the column mappings
            X2 = np.zeros(len(allderivs))
            #X2[colI] = X
            X2[ucols] = X
            X = X2
            del X2
            
        if use_lsqr:
            from scipy.sparse import csr_matrix, csc_matrix
            from scipy.sparse.linalg import lsqr

            spvals = np.hstack(spvals)
            assert(np.all(np.isfinite(spvals)))
    
            sprows = np.hstack(sprows) # hogg's lovin' hstack *again* here
            assert(len(sprows) == len(spvals))
                
            # For LSQR, expand 'spcols' to be the same length as 'sprows'.
            cc = np.empty(len(sprows))
            i = 0
            for c,n in zip(spcols, nrowspercol):
                cc[i : i+n] = c
                i += n
            spcols = cc
            assert(i == len(sprows))
            assert(len(sprows) == len(spcols))

            logverb('  Number of sparse matrix elements:', len(sprows))
            urows = np.unique(sprows)
            ucols = np.unique(spcols)
            logverb('  Unique rows (pixels):', len(urows))
            logverb('  Unique columns (params):', len(ucols))
            if len(urows) == 0 or len(ucols) == 0:
                return []
            logverb('  Max row:', urows[-1])
            logverb('  Max column:', ucols[-1])
            logverb('  Sparsity factor (possible elements / filled elements):', float(len(urows) * len(ucols)) / float(len(sprows)))
            
            # FIXME -- does it make LSQR faster if we remap the row and column
            # indices so that no rows/cols are empty?
    
            # FIXME -- we could probably construct the CSC matrix ourselves!
    
            # Build sparse matrix
            #A = csc_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))
            A = csr_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))

            if get_A_matrix:
                return A

            lsqropts = dict(show=isverbose(), damp=damp)
            if variance:
                lsqropts.update(calc_var=True)
    
            # lsqr can trigger floating-point errors
            #np.seterr(all='warn')
            
            # Run lsqr()
            logmsg('LSQR: %i cols (%i unique), %i elements' %
                   (Ncols, len(ucols), len(spvals)-1))
    
            # print 'A matrix:'
            # print A.todense()
            # print
            # print 'vector b:'
            # print b
            
            t0 = time.clock()
            (X, istop, niters, r1norm, r2norm, anorm, acond,
             arnorm, xnorm, var) = lsqr(A, b, **lsqropts)
            t1 = time.clock()
            logmsg('  %.1f seconds' % (t1-t0))

            del A
            del b
    
            # print 'LSQR results:'
            # print '  istop =', istop
            # print '  niters =', niters
            # print '  r1norm =', r1norm
            # print '  r2norm =', r2norm
            # print '  anorm =', anorm
            # print '  acord =', acond
            # print '  arnorm =', arnorm
            # print '  xnorm =', xnorm
            # print '  var =', var
    
            #olderr = set_fp_err()
        
        logverb('scaled  X=', X)
        X = np.array(X)

        if shared_params:
            # Unapply shared parameter map -- result is duplicated
            # result elements.
            logverb('shared_params: before, X len', len(X), 'with', np.count_nonzero(X), 'non-zero entries')
            logverb('paramindexmap: len', len(paramindexmap), 'range', paramindexmap.min(), paramindexmap.max())
            X = X[paramindexmap]
            logverb('shared_params: after, X len', len(X), 'with', np.count_nonzero(X), 'non-zero entries')

        if scale_columns:
            X /= colscales
        logverb('  X=', X)

        #np.seterr(**olderr)
        #print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]

        if variance:
            if shared_params:
                # Unapply shared parameter map.
                var = var[paramindexmap]
            
            if scale_columns:
                var /= colscales**2
            return X,var

        return X

    # def changeInvvar(self, Q2=None):
    #     '''
    #     run one iteration of iteratively reweighting the invvars for IRLS
    #     '''
    #     if Q2 is None:
    #         return
    #     assert(Q2 > 0.5)
    #     for img in self.getImages():
    #         resid = img.getImage() - self.getModelImage(img)
    #         oinvvar = img.getOrigInvvar()
    #         smask = img.getStarMask()
    #         chi2 = oinvvar * resid**2
    #         factor = Q2 / (Q2 + chi2)
    #         img.setInvvar(oinvvar * factor * smask)

    def getModelPatchNoCache(self, img, src, **kwargs):
        return src.getModelPatch(img, **kwargs)

    def getModelPatch(self, img, src, minsb=None, **kwargs):
        if self.cache is None:
            # shortcut
            return src.getModelPatch(img, **kwargs)

        deps = (img.hashkey(), src.hashkey())
        deps = hash(deps)
        mv,mod = self.cache.get(deps, (0.,None))
        if minsb is None:
            minsb = img.modelMinval
        if mv > minsb:
            mod = None
        if mod is not None:
            pass
        else:
            mod = self.getModelPatchNoCache(img, src, minsb=minsb, **kwargs)
            self.cache.put(deps, (minsb,mod))
        return mod

    def getModelImage(self, img, srcs=None, sky=True, minsb=None):
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
            img.sky.addTo(mod)
        if srcs is None:
            srcs = self.catalog
        for src in srcs:
            patch = self.getModelPatch(img, src, minsb=minsb)
            if patch is None:
                continue
            patch.addTo(mod)
        return mod

    def getOverlappingSources(self, img, srcs=None, minsb=0.):
        from scipy.ndimage.morphology import binary_dilation
        from scipy.ndimage.measurements import label

        if _isint(img):
            img = self.getImage(img)
        mod = np.zeros(img.getShape(), self.modtype)
        if srcs is None:
            srcs = self.catalog
        patches = []
        for src in srcs:
            patch = self.getModelPatch(img, src, minsb=minsb)
            #print 'Source', src
            #print '--> patch', patch
            patches.append(patch)
            if patch is not None:
                #print '    sum', patch.patch.sum()
                #print '    max', patch.patch.max()
                #print '    min > 0', patch.patch[patch.patch > 0].min()
                assert(np.all(patch.patch >= 0))
                patch.addTo(mod)

        #mod = self.getModelImage(img, srcs=srcs, minsb=minsb, sky=False)
        bigmod = (mod > minsb)
        # print 'Mod: nonzero pixels', len(np.flatnonzero(mod))
        # print 'BigMod: nonzero pixels', len(np.flatnonzero(bigmod))
        bigmod = binary_dilation(bigmod)
        L,n = label(bigmod, structure=np.ones((3,3), int))
        #L,n = label(mod > minsb, structure=np.ones((5,5), int))
        #print 'Found', n, 'groups of sources'
        assert(L.shape == mod.shape)

        srcgroups = {}
        #equivgroups = {}
        H,W = mod.shape
        for i,modpatch in enumerate(patches):
            #for i,src in enumerate(srcs):
            # modpatch = self.getModelPatch(img, src, minsb=minsb)
            #print 'modpatch', modpatch
            if modpatch is None or not modpatch.clipTo(W,H):
                # no overlap with image
                continue
            #print 'modpatch', modpatch
            lpatch = L[modpatch.getSlice(mod)]
            #print 'mp', modpatch.shape
            #print 'lpatch', lpatch.shape
            assert(lpatch.shape == modpatch.shape)
            ll = np.unique(lpatch[modpatch.patch > minsb])
            #print 'labels:', ll, 'for source', src
            # Not sure how this happens, but it does...
            #ll = [l for l in ll if l > 0]
            if len(ll) == 0:
                # this sources contributes very little!
                continue

            # if len(ll) != 1:
            #   import pylab as plt
            #   plt.clf()
            #   plt.subplot(2,2,1)
            #   plt.imshow(lpatch, origin='lower', interpolation='nearest')
            #   plt.subplot(2,2,2)
            #   plt.imshow(modpatch.patch, origin='lower', interpolation='nearest')
            #   plt.subplot(2,2,3)
            #   plt.imshow(modpatch.patch > minsb, origin='lower', interpolation='nearest')
            #   plt.subplot(2,2,4)
            #   plt.imshow(lpatch * (modpatch.patch > minsb), origin='lower', interpolation='nearest')
            #   plt.savefig('ll.png')
            #   sys.exit(-1)

            if len(ll) > 1:
                # This can happen when a source has two peaks (eg, a PSF)
                # that are separated by a pixel and the saddle is below minval.
                pass
            assert(len(ll) == 1)
            ll = ll[0]
            if not ll in srcgroups:
                srcgroups[ll] = []
            srcgroups[ll].append(i)
        #return srcgroups.values() #, L
        return srcgroups, L, mod

    def getModelImages(self):
        if self.is_multiproc():
            # avoid shipping my images...
            allimages = self.getImages()
            self.images = Images()
            args = [(self, im) for im in allimages]
            #print 'Calling _map:', getmodelimagefunc2
            #print 'args:', args
            mods = self._map(getmodelimagefunc2, args)
            self.images = allimages
        else:
            mods = [self.getModelImage(img) for img in self.images]
        return mods

    def clearCache(self):
        self.cache.clear() # = Cache()

    def getChiImages(self):
        mods = self.getModelImages()
        chis = []
        for img,mod in zip(self.images, mods):
            chi = (img.getImage() - mod) * img.getInvError()
            if not np.all(np.isfinite(chi)):
                print 'Chi not finite'
                print 'Image finite?', np.all(np.isfinite(img.getImage()))
                print 'Mod finite?', np.all(np.isfinite(mod))
                print 'InvErr finite?', np.all(np.isfinite(img.getInvError()))
            chis.append(chi)
        return chis

    def getChiImage(self, imgi=-1, img=None, srcs=None, minsb=0.):
        if img is None:
            img = self.getImage(imgi)
        mod = self.getModelImage(img, srcs=srcs, minsb=minsb)
        chi = (img.getImage() - mod) * img.getInvError()
        if not np.all(np.isfinite(chi)):
            print 'Chi not finite'
            print 'Image finite?', np.all(np.isfinite(img.getImage()))
            print 'Mod finite?', np.all(np.isfinite(mod))
            print 'InvErr finite?', np.all(np.isfinite(img.getInvError()))
            print 'Current thawed parameters:'
            self.printThawedParams()
            print 'Current sources:'
            for src in self.getCatalog():
                print '  ', src
            print 'Image:', img
            print 'sky:', img.getSky()
            print 'psf:', img.getPsf()
        return chi

    def getNdata(self):
        count = 0
        for img in self.images:
            InvError = img.getInvError()
            count += len(np.ravel(InvError > 0.0))
        return count

    def getLogLikelihood(self):
        chisq = 0.
        for i,chi in enumerate(self.getChiImages()):
            chisq += (chi.astype(float) ** 2).sum()
        return -0.5 * chisq

    def getLogProb(self):
        '''
        return the posterior PDF, evaluated at the parametrs
        '''
        lnprior = self.getLogPrior()
        if lnprior == -np.inf:
            return lnprior
        lnl = self.getLogLikelihood()
        lnp = lnprior + lnl
        if np.isnan(lnp):
            print 'Tractor.getLogProb() returning NaN.'
            print 'Params:'
            self.printThawedParams()
            print 'log likelihood:', lnl
            print 'log prior:', lnprior
            return -np.inf
        return lnp

    def getBbox(self, img, srcs):
        nzsum = None
        # find bbox
        for src in srcs:
            p = self.getModelPatch(img, src)
            if p is None:
                continue
            nz = p.getNonZeroMask()
            nz.patch = nz.patch.astype(np.int)
            if nzsum is None:
                nzsum = nz
            else:
                nzsum += nz
            # ie = tim.getInvError()
            # p2 = np.zeros_like(ie)
            # p.addTo(p2)
            # effect = np.sum(p2)
            # print 'Source:', src
            # print 'Total chi contribution:', effect, 'sigma'
        nzsum.trimToNonZero()
        roi = nzsum.getExtent()
        return roi
    


    
class ScaledTractor(object):
    def __init__(self, tractor, p0, scales):
        self.tractor = tractor
        self.offset = p0
        self.scale = scales
    def getImage(self, i):
        return self.tractor.getImage(i)
    def getChiImage(self, i):
        return self.tractor.getChiImage(i)
    def _getOneImageDerivs(self, i):
        derivs = self.tractor._getOneImageDerivs(i)
        for (ind, x0, y0, der) in derivs:
            der *= self.scale[ind]
            #print 'Derivative', ind, 'has RSS', np.sqrt(np.sum(der**2))
        return derivs
    def setParams(self, p):
        #print 'ScaledTractor: setParams', p
        return self.tractor.setParams(self.offset + self.scale * p)
        
