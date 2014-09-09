"""
This file is part of the Tractor project.
Copyright 2011, 2012, 2013 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`galaxy.py`
================

Exponential and deVaucouleurs galaxy model classes.

These use the slightly modified versions of the exp and dev profiles
from the SDSS /Photo/ software; we use multi-Gaussian approximations
of these.
"""
import numpy as np

from . import mixture_profiles as mp
from .engine import *
from .utils import *
from .cache import *

_galcache = Cache(maxsize=10000)
def get_galaxy_cache():
    return _galcache

def set_galaxy_cache_size(N):
    global _galcache
    _galcache = Cache(maxsize=N)

def disable_galaxy_cache():
    global _galcache
    _galcache = None

class GalaxyShape(ParamList):
    '''
    A naive representation of an ellipse (describing a galaxy shape),
    using effective radius (in arcsec), axis ratio, and position angle.

    For better ellipse parameterizations, see ellipses.py
    '''
    @staticmethod
    def getName():
        return "Galaxy Shape"

    @staticmethod
    def getNamedParams():
        '''
        re: arcsec
        ab: axis ratio, dimensionless, in [0,1]
        phi: deg, "E of N", 0=direction of increasing Dec, 90=direction of increasing RA
        '''
        return dict(re=0, ab=1, phi=2)

    def __repr__(self):
        return 're=%g, ab=%g, phi=%g' % (self.re, self.ab, self.phi)

    def __str__(self):
        return '%s: re=%.2f, ab=%.2f, phi=%.1f' % (self.getName(), self.re, self.ab, self.phi)

    # def getAllStepSizes(self, *args, **kwargs):
    #     abstep = 0.01
    #     if self.ab >= (1 - abstep):
    #         abstep = -abstep
    #     return [ 0.1, abstep, 1. ]

    def getRaDecBasis(self):
        '''
        Returns a transformation matrix that takes vectors in r_e
        to delta-RA, delta-Dec vectors.
        '''
        # convert re, ab, phi into a transformation matrix
        phi = np.deg2rad(90 - self.phi)
        # convert re to degrees
        # HACK -- bring up to a minimum size to prevent singular matrix inversions
        re_deg = max(1./30, self.re) / 3600.
        cp = np.cos(phi)
        sp = np.sin(phi)
        # Squish, rotate, and scale into degrees.
        # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
        G = re_deg * np.array([[ cp, sp * self.ab],
                               [-sp, cp * self.ab]])
        return G

    def getTensor(self, cd):
        # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
        G = self.getRaDecBasis()
        # "cd" takes pixels to degrees (intermediate world coords)
        # T takes pixels to unit vectors.
        print 'Shape', self
        print 'Basis', G, 'cd', cd
        T = np.dot(np.linalg.inv(G), cd)
        return T

class Galaxy(MultiParams):
    '''
    Generic Galaxy profile with position, brightness, and shape.
    '''
    def __init__(self, *args):
        super(Galaxy, self).__init__(*args)
        self.name = self.getName()
        self.dname = self.getDName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2)

    def getName(self):
        return 'Galaxy'

    def getDName(self):
        '''
        Name used in labeling the derivative images d(Dname)/dx, eg
        '''
        return 'gal'
        
    def getSourceType(self):
        return self.name

    def getPosition(self):
        return self.pos
    def getShape(self):
        return self.shape
    def getBrightness(self):
        return self.brightness
    def getBrightnesses(self):
        return [self.getBrightness()]
    def setBrightness(self, brightness):
        self.brightness = brightness

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness)
                + ' and ' + str(self.shape))
    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', shape=' + repr(self.shape))

    def copy(self):
        return None

    def getUnitFluxModelPatch(self, img, **kwargs):
        raise RuntimeError('getUnitFluxModelPatch unimplemented in' +
                           self.getName())

    def getModelPatch(self, img, minsb=0.):
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None
        minval = minsb / abs(counts)
        p1 = self.getUnitFluxModelPatch(img, minval=minval)
        if p1 is None:
            return None
        return p1 * counts

    # returns [ Patch, Patch, ... ] of length numberOfParams().
    # Galaxy.
    def getParamDerivatives(self, img):
        pos0 = self.getPosition()
        (px0,py0) = img.getWcs().positionToPixel(pos0, self)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        patch0 = self.getUnitFluxModelPatch(img, px0, py0)
        if patch0 is None:
            return [None] * self.numberOfParams()
        derivs = []

        # derivatives wrt position
        psteps = pos0.getStepSizes()
        if not self.isParamFrozen('pos'):
            params = pos0.getParams()
            if counts == 0:
                derivs.extend([None] * len(params))
                psteps = []
            for i,pstep in enumerate(psteps):
                oldval = pos0.setParam(i, params[i]+pstep)
                (px,py) = img.getWcs().positionToPixel(pos0, self)
                pos0.setParam(i, oldval)
                patchx = self.getUnitFluxModelPatch(img, px, py)
                if patchx is None or patchx.getImage() is None:
                    derivs.append(None)
                    continue
                dx = (patchx - patch0) * (counts / pstep)
                dx.setName('d(%s)/d(pos%i)' % (self.dname, i))
                derivs.append(dx)

        # derivatives wrt brightness
        bsteps = self.brightness.getStepSizes()
        if not self.isParamFrozen('brightness'):
            params = self.brightness.getParams()
            for i,bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, params[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts) / bstep)
                df.setName('d(%s)/d(bright%i)' % (self.dname, i))
                derivs.append(df)

        # derivatives wrt shape
        gsteps = self.shape.getStepSizes()
        if not self.isParamFrozen('shape'):
            gnames = self.shape.getParamNames()
            oldvals = self.shape.getParams()
            # print 'Galaxy.getParamDerivatives:', self.getName()
            # print '  oldvals:', oldvals
            if counts == 0:
                derivs.extend([None] * len(oldvals))
                gsteps = []
            for i,gstep in enumerate(gsteps):
                oldval = self.shape.setParam(i, oldvals[i]+gstep)
                #print '  stepped', gnames[i], 'by', gsteps[i],
                #print 'to get', self.shape
                patchx = self.getUnitFluxModelPatch(img, px0, py0)
                self.shape.setParam(i, oldval)
                if patchx is None:
                    print 'patchx is None:'
                    print '  ', self
                    print '  stepping galaxy shape', self.shape.getParamNames()[i]
                    print '  stepped', gsteps[i]
                    print '  to', self.shape.getParams()[i]
                    derivs.append(None)

                dx = (patchx - patch0) * (counts / gstep)
                dx.setName('d(%s)/d(%s)' % (self.dname, gnames[i]))
                derivs.append(dx)
        return derivs


class CompositeGalaxy(MultiParams):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''
    def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2, brightnessDev=3, shapeDev=4)

    def getName(self):
        return 'CompositeGalaxy'

    def getPosition(self):
        return self.pos

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' ' + str(self.shapeExp)
                + ' and deV ' + str(self.brightnessDev) + ' ' + str(self.shapeDev))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) + 
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev))

    def getBrightness(self):
        ''' This makes some assumptions about the ``Brightness`` / ``PhotoCal`` and
        should be treated as approximate.'''
        return self.brightnessExp + self.brightnessDev

    def getBrightnesses(self):
        return [self.brightnessExp, self.brightnessDev]
    
    def getModelPatch(self, img, minsb=0.):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if minsb == 0.:
            kw = {}
        else:
            kw = dict(minsb=minsb/2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        pe = e.getModelPatch(img, **kw)
        pd = d.getModelPatch(img, **kw)
        if pe is None:
            return pd
        if pd is None:
            return pe
        return pe + pd

    def getUnitFluxModelPatches(self, img, minval=0.):
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        return (e.getUnitFluxModelPatches(img, minval=minval) +
                d.getUnitFluxModelPatches(img, minval=minval))

    def getUnitFluxModelPatch(self, img, px=None, py=None):
        # this code is un-tested
        assert(False)
        fe = self.brightnessExp / (self.brightnessExp + self.brightnessDev)
        fd = 1. - fe
        assert(fe >= 0.)
        assert(fe <= 1.)
        e = ExpGalaxy(self.pos, fe, self.shapeExp)
        d = DevGalaxy(self.pos, fd, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        pe = e.getModelPatch(img, px, py)
        pd = d.getModelPatch(img, px, py)
        if pe is None:
            return pd
        if pd is None:
            return pe
        return pe + pd

    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img):
        #print 'CompositeGalaxy: getParamDerivatives'
        #print '  Exp brightness', self.brightnessExp, 'shape', self.shapeExp
        #print '  Dev brightness', self.brightnessDev, 'shape', self.shapeDev
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        d.dname = 'comp.dev'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessDev'):
            d.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')

        de = e.getParamDerivatives(img)
        dd = d.getParamDerivatives(img)

        if self.isParamFrozen('pos'):
            derivs = de + dd
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes())
            for i in range(npos):
                dp = add_patches(de[i], dd[i])
                if dp is not None: 
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de[npos:])
            derivs.extend(dd[npos:])

        return derivs

class ProfileGalaxy(object):
    '''
    A mix-in class that renders itself based on a Mixture-of-Gaussians
    profile.
    '''
    def getName(self):
        return 'ProfileGalaxy'

    def getProfile(self):
        return None

    # Here's the main method to override;
    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        return None
    
    def _getUnitFluxDeps(self, img, px, py):
        return None

    def _getUnitFluxPatchSize(self, img, minval):
        return 0
    
    def getUnitFluxModelPatch(self, img, px=None, py=None, minval=0.0):
        if px is None or py is None:
            (px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
        #
        if _galcache is None:
            return self._realGetUnitFluxModelPatch(img, px, py, minval)
        
        deps = self._getUnitFluxDeps(img, px, py)
        try:
            (cached,mv) = _galcache.get(deps)
            if mv <= minval:
                return cached
        except KeyError:
            pass
        patch = self._realGetUnitFluxModelPatch(img, px, py, minval)
        _galcache.put(deps, (patch,minval))
        return patch

    def getUnitFluxModelPatches(self, img, minval=0.):
        return [self.getUnitFluxModelPatch(img, minval=minval)]

    def _realGetUnitFluxModelPatch(self, img, px, py, minval):
        # now choose the patch size
        halfsize = self._getUnitFluxPatchSize(img, px, py, minval)
        
        # find overlapping pixels to render
        (outx, inx) = get_overlapping_region(
            int(floor(px-halfsize)), int(ceil(px+halfsize+1)),
            0, img.getWidth())
        (outy, iny) = get_overlapping_region(
            int(floor(py-halfsize)), int(ceil(py+halfsize+1)),
            0, img.getHeight())
        if inx == [] or iny == []:
            # no overlap
            return None

        x0,x1 = outx.start, outx.stop
        y0,y1 = outy.start, outy.stop
        psf = img.getPsf()
        if hasattr(psf, 'getMixtureOfGaussians'):
            amix = self._getAffineProfile(img, px, py)
            # now convolve with the PSF, analytically
            psfmix = psf.getMixtureOfGaussians()
            #psfmix.normalize()
            cmix = amix.convolve(psfmix)
            psfconvolvedimg = mp.mixture_to_patch(cmix, x0, x1, y0, y1, minval)
            return Patch(x0, y0, psfconvolvedimg)
        else:
            P,(px0,py0),(pH,pW) = psf.getFourierTransform(halfsize)
            w = np.fft.rfftfreq(pW)
            v = np.fft.fftfreq(pH)

            dx = px - px0
            dy = py - py0
            # Put the integer portion of the offset into Patch x0,y0
            ix0 = int(np.round(dx))
            iy0 = int(np.round(dy))
            # Put the subpixel portion into the galaxy FFT.
            mux = dx - ix0
            muy = dy - iy0

            amix = self._getAffineProfile(img, mux, muy)
            Fsum = amix.getFourierTransform(w, v)
                
            # FIXME -- could adjust the ifft shape...
            G = np.fft.irfft2(Fsum * P, s=(pH,pW))
            # Clip down to suggested "halfsize"
            if x0 > ix0:
                G = G[:,x0 - ix0:]
                ix0 = x0
            if y0 > iy0:
                G = G[y0 - iy0:, :]
                iy0 = y0
            gh,gw = G.shape
            if gw+ix0 > x1:
                G = G[:,:x1-ix0]
            if gh+iy0 > y1:
                G = G[:y1-iy0,:]
            return Patch(ix0, iy0, G)
                    

class HoggGalaxy(ProfileGalaxy, Galaxy):

    def getName(self):
        return 'HoggGalaxy'

    def copy(self):
        return HoggGalaxy(self.pos.copy(), self.brightness.copy(),
                          self.shape.copy())

    def getRadius(self):
        return self.nre * self.shape.re
    
    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        # shift and squash
        cd = img.getWcs().cdAtPixel(px, py)
        galmix = self.getProfile()
        Tinv = np.linalg.inv(self.shape.getTensor(cd))
        amix = galmix.apply_affine(np.array([px,py]), Tinv.T)
        amix.symmetrize()
        return amix

    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(), px, py,
                     img.getWcs().hashkey(),
                     img.getPsf().hashkey(), self.shape.hashkey()))

    def _getUnitFluxPatchSize(self, img, px, py, minval):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        cd = img.getWcs().cdAtPixel(px, py)
        pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
        halfsize = max(1., self.getRadius() / 3600. / pixscale)
        psf = img.getPsf()
        halfsize += psf.getRadius()
        return halfsize

class ExpGalaxy(HoggGalaxy):
    nre = 4.
    profile = mp.get_exp_mixture()
    profile.normalize()
    @staticmethod
    def getExpProfile():
        return ExpGalaxy.profile
    def __init__(self, *args, **kwargs):
        self.nre = ExpGalaxy.nre
        super(ExpGalaxy,self).__init__(*args, **kwargs)
    def getName(self):
        return 'ExpGalaxy'
    def getProfile(self):
        return ExpGalaxy.getExpProfile()
    def getShape(self):
        return self.shape
    def copy(self):
        return ExpGalaxy(self.pos.copy(), self.brightness.copy(),
                         self.shape.copy())

class DevGalaxy(HoggGalaxy):
    nre = 8.
    profile = mp.get_dev_mixture()
    profile.normalize()
    @staticmethod
    def getDevProfile():
        return DevGalaxy.profile
    def __init__(self, *args, **kwargs):
        self.nre = DevGalaxy.nre
        super(DevGalaxy,self).__init__(*args, **kwargs)
    def getName(self):
        return 'DevGalaxy'
    def getProfile(self):
        return DevGalaxy.getDevProfile()
    def copy(self):
        return DevGalaxy(self.pos.copy(), self.brightness.copy(),
                         self.shape.copy())


class FracDev(ScalarParam):
    stepsize = 0.01
    def getClippedValue(self):
        f = self.getValue()
        return np.clip(f, 0., 1.)

class SoftenedFracDev(FracDev):
    '''
    Implements a "softened" version of the deV-to-total fraction.

    The sigmoid function is scaled and shifted so that S(0) ~ 0.1 and
    S(1) ~ 0.9.

    Use the 'getClippedValue()' function to get the un-softened
    fracDev in [0,1].
    '''
    def getClippedValue(self):
        f = self.getValue()
        return 1./(1 + np.exp(4.*(0.5 - f)))
    
class FixedCompositeGalaxy(MultiParams, ProfileGalaxy):
    '''
    A galaxy with Exponential and deVaucouleurs components where the
    brightnesses of the deV and exp components are defined in terms of
    a total brightness and a fraction of that total that goes to the
    deV component.
    
    The two components share a position (ie the centers are the same),
    but have different shapes.  The galaxy has a single brightness
    that is split between the components.

    This is like CompositeGalaxy, but more useful for getting
    consistent colors from forced photometry, because one can freeze
    the fracDev to keep the deV+exp profile fixed.
    '''
    def __init__(self, pos, brightness, fracDev, shapeExp, shapeDev):
        # handle passing fracDev as a float
        if not isinstance(fracDev, BaseParams):
            fracDev = FracDev(fracDev)
        MultiParams.__init__(self, pos, brightness, fracDev,
                             shapeExp, shapeDev)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, fracDev=2, shapeExp=3, shapeDev=4)

    def getName(self):
        return 'FixedCompositeGalaxy'

    def getPosition(self):
        return self.pos

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness) 
                + ', ' + str(self.fracDev)
                + ', exp ' + str(self.shapeExp)
                + ', deV ' + str(self.shapeDev))
    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', fracDev=' + repr(self.fracDev) + 
                ', shapeExp=' + repr(self.shapeExp) +
                ', shapeDev=' + repr(self.shapeDev) + ')')
    def copy(self):
        return FixedCompositeGalaxy(self.pos.copy(), self.brightness.copy(),
                                    self.fracDev.copy(),
                                    self.shapeExp.copy(),
                                    self.shapeDev.copy())

    def getBrightness(self):
        return self.brightness

    def getBrightnesses(self):
        return [self.brightness]

    def setBrightness(self, b):
        self.brightness = b

    def getModelPatch(self, img, minsb=0.):
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None
        minval = minsb / abs(counts)
        p1 = self.getUnitFluxModelPatch(img, minval=minval)
        if p1 is None:
            return None
        return p1 * counts

    def _getAffineProfile(self, img, px, py):
        f = self.fracDev.getClippedValue()
        profs = []
        if f > 0.:
            profs.append((f, DevGalaxy.profile, self.shapeDev))
        if f < 1.:
            profs.append((1.-f, ExpGalaxy.profile, self.shapeExp))

        cd = img.getWcs().cdAtPixel(px, py)
        mix = []
        for f,p,s in profs:
            Tinv = np.linalg.inv(s.getTensor(cd))
            amix = p.apply_affine(np.array([px,py]), Tinv.T)
            amix.symmetrize()
            amix.amp *= f
            mix.append(amix)
        if len(mix) == 1:
            return mix[0]
        return mix[0] + mix[1]

    def _getUnitFluxPatchSize(self, img, px, py, minval):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        cd = img.getWcs().cdAtPixel(px, py)
        pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
        s = self.shapeExp
        r = ExpGalaxy.nre * s.re
        s = self.shapeDev
        r = max(r, DevGalaxy.nre * s.re)
        halfsize = max(1., r) / 3600. / pixscale
        psf = img.getPsf()
        halfsize += psf.getRadius()
        return halfsize
    
    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(),
                     px, py, img.getWcs().hashkey(),
                     img.getPsf().hashkey(),
                     self.shapeDev.hashkey(),
                     self.shapeExp.hashkey(),
                     self.fracDev.hashkey()))
    
    # WARNING, this code has not been tested.
    def getParamDerivatives(self, img):
        ### FIXME -- minsb would be useful here!
        #pos0 = self.getPosition()
        #(px0,py0) = img.getWcs().positionToPixel(pos0, self)
        e = ExpGalaxy(self.pos, self.brightness, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightness, self.shapeDev)
        e.dname = 'fcomp.exp'
        d.dname = 'fcomp.dev'

        f = self.fracDev.getClippedValue()

        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')

        de = e.getParamDerivatives(img)
        dd = d.getParamDerivatives(img)

        # fracDev scaling
        for deriv in de:
            if deriv is not None:
                deriv *= (1.-f)
        for deriv in dd:
            if deriv is not None:
                deriv *= f
        
        derivs = []
        i0 = 0
        if not self.isParamFrozen('pos'):
            # "pos" is shared between the models, so add the derivs.
            npos = self.pos.numberOfParams()
            for i in range(npos):
                ii = i0+i
                df = add_patches(de[ii], dd[ii])
                if df is not None: 
                    df.setName('d(fcomp)/d(pos%i)' % i)
                derivs.append(df)
            i0 += npos

        if not self.isParamFrozen('brightness'):
            # shared between the models, so add the derivs.
            nb = self.brightness.numberOfParams()
            for i in range(nb):
                ii = i0+i
                df = add_patches(de[ii], dd[ii])
                if df is not None: 
                    df.setName('d(fcomp)/d(bright%i)' % i)
                derivs.append(df)
            i0 += nb
                
        if not self.isParamFrozen('fracDev'):
            counts = img.getPhotoCal().brightnessToCounts(self.brightness)
            if counts == 0.:
                derivs.append(None)
            else:
                ue = e.getUnitFluxModelPatch(img)
                ud = d.getUnitFluxModelPatch(img)
                if ue is not None:
                    ue *= -1
                df = add_patches(ud, ue)
                if df is None:
                    derivs.append(None)
                else:
                    df *= counts
                    df.setName('d(fcomp)/d(fracDev)')
                    derivs.append(df)

        if not self.isParamFrozen('shapeExp'):
            derivs.extend(de[i0:])
        if not self.isParamFrozen('shapeDev'):
            derivs.extend(dd[i0:])
            
        return derivs
    

if __name__ == '__main__':
    from astrometry.util.plotutils import PlotSequence
    import matplotlib
    from basics import GaussianMixturePSF, PixPos, Flux, NullPhotoCal, NullWCS, ConstantSky
    from engine import Image
    matplotlib.use('Agg')
    import pylab as plt
    ps = PlotSequence('gal')
    
    # example PSF (from WISE W1 fit)
    w = np.array([ 0.77953706,  0.16022146,  0.06024237])
    mu = np.array([[-0.01826623, -0.01823262],
                   [-0.21878855, -0.0432496 ],
                   [-0.83365747, -0.13039277]])
    sigma = np.array([[[  7.72925584e-01,   5.23305564e-02],
                       [  5.23305564e-02,   8.89078473e-01]],
                       [[  9.84585869e+00,   7.79378820e-01],
                       [  7.79378820e-01,   8.84764455e+00]],
                       [[  2.02664489e+02,  -8.16667434e-01],
                        [ -8.16667434e-01,   1.87881670e+02]]])
    
    psf = GaussianMixturePSF(w, mu, sigma)

    shape = GalaxyShape(10., 0.5, 30.)
    pos = PixPos(100, 50)
    bright = Flux(1000.)
    egal = ExpGalaxy(pos, bright, shape)

    data = np.zeros((100, 200))
    tim = Image(data=data, psf=psf)

    p0 = egal.getModelPatch(tim)
    
    p1 = egal.getModelPatch(tim, 1e-3)

    bright.setParams([100.])

    p2 = egal.getModelPatch(tim, 1e-3)

    print 'p0', p0.patch.sum()
    print 'p1', p1.patch.sum()
    print 'p2', p2.patch.sum()
    
    plt.clf()
    ima = dict(interpolation='nearest', origin='lower')
    plt.subplot(2,2,1)
    plt.imshow(np.log10(np.maximum(1e-16, p0.patch)), **ima)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.log10(np.maximum(1e-16, p1.patch)), **ima)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(np.log10(np.maximum(1e-16, p2.patch)), **ima)
    plt.colorbar()
    ps.savefig()
