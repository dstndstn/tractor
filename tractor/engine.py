'''
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`engine.py`
===========

Core image modeling and fitting.
'''

import logging

import numpy as np

from astrometry.util.ttime import Time

from .utils import MultiParams, _isint, listmax, get_class_from_name
from .patch import Patch

def logverb(*args):
    msg = ' '.join([str(x) for x in args])
    logging.debug(msg)
def logmsg(*args):
    msg = ' '.join([str(x) for x in args])
    logging.info(msg)
def isverbose():
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

        If *wcs* is not given, assumes pixel space.

        If *sky* is not given, assumes zero sky.
        
        If *photocal* is not given, assumes count units.
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
        hdr = self.getFitsHeader(header=primheader, prefix=prefix)
        fits.write(None, header=hdr)
        fits.write(self.getImage(), header=imageheader)
        fits.write(self.getInvvar(), header=invvarheader)

    def getFitsHeader(self, header=None, prefix=''):
        psf = self.getPsf()
        wcs = self.getWcs()
        sky = self.getSky()
        pcal = self.getPhotoCal()
        
        if header is None:
            import fitsio
            hdr = fitsio.FITSHDR()
        else:
            hdr = header
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
        return hdr

    def getStandardFitsHeader(self, header=None):
        if header is None:
            import fitsio
            hdr = fitsio.FITSHDR()
        else:
            hdr = header
        psf = self.getPsf()
        wcs = self.getWcs()
        sky = self.getSky()
        pcal = self.getPhotoCal()
        psf.toStandardFitsHeader(hdr)
        wcs.toStandardFitsHeader(hdr)
        sky.toStandardFitsHeader(hdr)
        pcal.toStandardFitsHeader(hdr)
        return hdr

    
        
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


class OptResult():
    # quack
    pass

class TractorBase(MultiParams):
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

    def __init__(self, images=[], catalog=[]):
        '''
        - `images:` list of Image objects (data)
        - `catalog:` list of Source objects
        '''
        if not isinstance(images, Images):
            images = Images(*images)
        if not isinstance(catalog, Catalog):
            catalog = Catalog(*catalog)
        super(Tractor,self).__init__(images, catalog)
        self.modtype = np.float32
        self.modelMasks = None
        self.expectModelMasks = False

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

    # For use from emcee
    def __call__(self, X):
        self.setParams(X)
        return self.getLogProb()

    # For pickling
    def __getstate__(self):
        S = (self.getImages(), self.getCatalog(), self.liquid,
             self.modtype, self.modelMasks, self.expectModelMasks)
        return S
    def __setstate__(self, state):
        (images, catalog, self.liquid, self.modtype, self.modelMasks,
         self.expectModelMasks
         ) = state
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
                mask = self._getModelMaskFor(img, src)
                ums = src.getUnitFluxModelPatches(img, minval=mv, modelMask=mask)

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

        # Render unit-flux models for each source.
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

        _optimize_forcedphot_core(
            result, umodels, imlist, mod0, scales, skyderivs, minFlux,
            BW, BH,
            nonneg=nonneg, wantims0=wantims0, wantims1=wantims1,
            negfluxval=negfluxval, rois=rois, priors=priors, sky=sky,
            justims0=justims0, subimgs=subimgs, damp=damp, alphas=alphas,
            Nsky=Nsky, mindlnp=mindlnp, shared_params=shared_params)
        
                
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


    def _optimize_forcedphot_core(
            self,
            result, umodels, imlist, mod0, scales, skyderivs, minFlux,
            BW, BH,
            nonneg=None, wantims0=None, wantims1=None,
            negfluxval=None, rois=None, priors=None, sky=None,
            justims0=None, subimgs=None, damp=None, alphas=None,
            Nsky=None, mindlnp=None, shared_params=None):
        raise RuntimeError('Unimplemented')
    

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
        raise RuntimeError('Unimplemented')

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
        
        # if alphaBest is None or alphaBest == 0:
        #     print "Warning: optimization is borking"
        #     print "Parameter direction =",X
        #     print "Parameters, step sizes, updates:"
        #     for n,p,s,x in zip(self.getParamNames(), self.getParams(), self.getStepSizes(), X):
        #         print n, '=', p, '  step', s, 'update', x
        if alphaBest is None:
            self.setParams(p0)
            return 0, 0.

        logverb('  Stepping by', alphaBest, 'for delta-logprob', pBest - pBefore)
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
                derivs = self._getSourceDerivatives(src, img)
                for k,deriv in enumerate(derivs):
                    if deriv is None:
                        continue
                    srcderivs[k].append((deriv, img))
            allderivs.extend(srcderivs)
        #print 'allderivs:', len(allderivs)
        #print 'N params:', self.numberOfParams()

        assert(len(allderivs) == self.numberOfParams())
        return allderivs

    def setModelMasks(self, masks, assumeMasks=True):

        '''
        A "model mask" is used to define the pixels that are evaluated
        when computing the model patch for a source in an image.  This
        allows for consistent computation of derivatives and
        optimization, without introducing errors due to approximating
        the profiles differently given different parameter settings.

        If *masks* is None, this masking is disabled, and normal
        approximation rules apply.

        Otherwise, *masks* must be a list, with length equal to the
        number of images.  Each list element must be a dictionary with
        Source objects for keys and Patch objects for values, where
        the Patch images are binary masks (True for pixels that should
        be evaluated).  Sources that do not touch the image should not
        exist in the dictionary; all the Patches should be non-None
        and non-empty.
        '''
        self.modelMasks = masks
        assert((masks is None) or (len(masks) == len(self.images)))
        self.expectModelMasks = (masks is not None) and assumeMasks

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
            #print 'nonzero type:', nonzero.patch.dtype
            unmasked = Patch(mask.x0, mask.y0, np.logical_not(mask.patch))
            #print 'unmasked type:', unmasked.patch.dtype
            bad = nonzero.performArithmetic(unmasked, '__iand__', otype=bool)
            assert(np.all(bad.patch == False))

    def _getSourceDerivatives(self, src, img, **kwargs):
        mask = self._getModelMaskFor(img, src)

        # HACK! -- assume no modelMask -> no overlap
        if self.expectModelMasks and mask is None:
            return [None] * src.numberOfParams()

        #print 'getting param derivs for', src
        derivs = src.getParamDerivatives(img, modelMask=mask, **kwargs)
        #print 'done getting param derivs for', src

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

        mod = src.getModelPatch(img, modelMask=mask, **kwargs)
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
            img.getSky().addTo(mod)
        if srcs is None:
            srcs = self.catalog
        for src in srcs:
            if src is None:
                continue
            patch = self.getModelPatch(img, src, minsb=minsb)
            if patch is None:
                continue
            patch.addTo(mod)
        return mod

    def getModelImages(self):
        return [self.getModelImage(img) for img in self.images]

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

    

from lsqr_mixin import TractorLsqrMixin
    
class Tractor(TractorLsqrMixin, TractorBase):
    pass

    
class ScaledTractor(object):
    def __init__(self, tractor, p0, scales):
        self.tractor = tractor
        self.offset = p0
        self.scale = scales
    def getImage(self, i):
        return self.tractor.getImage(i)
    def getChiImage(self, i):
        #return self.tractor.getChiImage(i)
        return self.tractor.getChiImage(i).astype(float)

    def _getOneImageDerivs(self, i):
        derivs = self.tractor._getOneImageDerivs(i)
        for (ind, x0, y0, der) in derivs:
            der *= self.scale[ind]
            #print 'Derivative', ind, 'has RSS', np.sqrt(np.sum(der**2))
        return derivs
    def setParams(self, p):
        #print 'ScaledTractor: setParams', p
        return self.tractor.setParams(self.offset + self.scale * p)
        
