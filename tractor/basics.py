"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`basics.py`
===========

Generally useful generic implementations of things the Tractor needs:
Magnitudes, (RA,Dec) positions, FITS WCS, and so on.

"""
from math import ceil, floor, pi, sqrt, exp

from .engine import *
from .utils import *
#from . import ducks
import ducks

import mixture_profiles as mp
import numpy as np

from astrometry.util.starutil_numpy import *
from astrometry.util.miscutils import *

class TractorWCSWrapper(object):
    '''
    Wraps a Tractor WCS object to look like an
    astrometry.util.util.Tan/Sip object.
    '''
    def __init__(self, wcs, w, h, x0=0, y0=0):
        self.wcs = wcs
        self.imagew = w
        self.imageh = h
        self.x0 = x0
        self.y0 = y0
    def pixelxy2radec(self, x, y):
        # FITS pixels x,y
        rd = self.wcs.pixelToPosition(x+self.x0-1, y+self.y0-1)
        return rd.ra, rd.dec
    def radec2pixelxy(self, ra, dec):
        # Vectorized?
        if hasattr(ra, '__len__') or hasattr(dec, '__len__'):
            try:
                b = np.broadcast(ra, dec)
                ok = np.ones(b.shape, bool)
                x  = np.zeros(b.shape)
                y  = np.zeros(b.shape)
                rd = RaDecPos(0.,0.)
                for i,(r,d) in enumerate(b):
                    rd.ra = r
                    rd.dec = d
                    xi,yi = self.wcs.positionToPixel(rd)
                    x.flat[i] = xi
                    y.flat[i] = yi
                x += 1 - self.x0
                y += 1 - self.y0
                return ok,x,y
            except:
                pass
            
        x,y = self.wcs.positionToPixel(RaDecPos(ra, dec))
        return True, x-self.x0+1, y-self.y0+1


def getParamTypeTree(param):
    mytype = str(type(param))
    if isinstance(param, MultiParams):
        return [mytype] + [getParamTypeTree(s) for s in param._getActiveSubs()]
    return [mytype]


class TAITime(ScalarParam, ArithmeticParams):
    '''
    This is TAI as used in the SDSS 'frame' headers; eg

    TAI     =        4507681767.55 / 1st row - 
    Number of seconds since Nov 17 1858

    And http://mirror.sdss3.org/datamodel/glossary.html#tai
    says:

    MJD = TAI/(24*3600)
    '''
    equinox = 53084.28 # mjd
    daysperyear = 365.25

    mjd2k = datetomjd(J2000)

    def __init__(self, t, mjd=None):
        if t is None and mjd is not None:
            t = mjd * 24. * 3600.
        super(TAITime, self).__init__(t)

    def toMjd(self):
        return self.getValue() / (24.*3600.)

    def getSunTheta(self):
        mjd = self.toMjd()
        th = 2. * np.pi * (mjd - TAITime.equinox) / TAITime.daysperyear
        th = np.fmod(th, 2.*np.pi)
        return th

    def toYears(self):
        ''' years since Nov 17, 1858 ?'''
        return float(self.getValue() / (24.*3600. * TAITime.daysperyear))

    def toYear(self):
        ''' to proper year '''
        return self.toYears() - TAITime(None, mjd=TAITime.mjd2k).toYears() + 2000.0


class Mag(ScalarParam):
    '''
    An implementation of `Brightness` that stores a single magnitude.
    '''
    stepsize = -0.01
    strformat = '%.3f'

class Flux(ScalarParam):
    '''
    A `Brightness` implementation that stores raw counts.
    '''
    def __mul__(self, factor):
        new = self.copy()
        new.val *= factor
        return new
    __rmul__ = __mul__

class MultiBandBrightness(ParamList, ducks.Brightness):
    '''
    An implementation of `Brightness` that stores an independent
    brightness in a set of named bands.  The PhotoCal for an image
    must know its band, and then it can retrieve the appropriate
    brightness for the image in question.

    This is the base class for Mags and Fluxes.
    '''
    def __init__(self, **kwargs):
        '''
        MultiBandBrightness(r=14.3, g=15.6, order=['r','g'])

        The `order` parameter is optional; it determines the ordering
        of the bands in the parameter vector (eg, `getParams()`).
        '''
        keys = kwargs.pop('order', None)
        if keys is None:
            keys = kwargs.keys()
            keys.sort()
        assert(len(kwargs) == len(keys))
        assert(set(kwargs.keys()) == set(keys))
        vals = []
        for k in keys:
            vals.append(kwargs[k])
        super(MultiBandBrightness, self).__init__(*vals)
        self.order = keys
        self.addNamedParams(**dict((k,i) for i,k in enumerate(keys)))
    
    def __setstate__(self, state):
        '''For pickling.'''
        self.__dict__ = state
        self.addNamedParams(**dict((k,i)
                                   for i,k in enumerate(self.order)))

    def copy(self):
        return self*1.
        
    def getBand(self, band):
        return getattr(self, band)

    def setBand(self, band, value):
        return setattr(self, band, value)
        
        
class Mags(MultiBandBrightness):
    '''
    An implementation of `Brightness` that stores magnitudes in
    multiple bands.
    '''
    def __init__(self, **kwargs):
        '''
        Mags(r=14.3, g=15.6, order=['r','g'])

        The `order` parameter is optional; it determines the ordering
        of the bands in the parameter vector (eg, `getParams()`).
        '''
        super(Mags,self).__init__(**kwargs)
        self.stepsizes = [-0.01] * self.numberOfParams()

    def getMag(self, bandname):
        '''
        Bandname: string
        Returns: mag in the given band.
        '''
        return self.getBand(bandname)

    def setMag(self, bandname, mag):
        return self.setBand(bandname, mag)

    def __add__(self, other):
        # mags + 0.1
        if np.isscalar(other):
            kwargs = {}
            for band in self.order:
                m1 = self.getMag(band)
                kwargs[band] = m1 + other
            return Mags(order=self.order, **kwargs)

        # ASSUME some things about calibration here...
        kwargs = {}
        for band in self.order:
            m1 = self.getMag(band)
            m2 = other.getMag(band)
            msum = -2.5 * np.log10( 10.**(-m1/2.5) + 10.**(-m2/2.5) )
            kwargs[band] = msum
        return Mags(order=self.order, **kwargs)

    def __mul__(self, factor):
        # Return the magnitude that corresponds to the flux rescaled
        # by factor.  Flux is positive (and log(-ve) is not
        # permitted), so we take the abs of the input scale factor to
        # prevent embarrassment.  Negative magnifications appear in
        # gravitational lensing, but they just label the "parity" of
        # the source, not its brightness. So, we treat factor=-3 the
        # same as factor=3, for example.
        kwargs = {}
        dmag = -2.5 * np.log10(np.abs(factor))
        for band in self.order:
            m = self.getMag(band)
            mscaled = m + dmag
            kwargs[band] = mscaled
        return self.__class__(order=self.order, **kwargs)


class Fluxes(MultiBandBrightness):
    '''
    An implementation of `Brightness` that stores fluxes in multiple
    bands.
    '''
    def __add__(self, other):
        kwargs = {}
        for band in self.order:
            m1 = self.getBand(band)
            m2 = other.getBand(band)
            kwargs[band] = m1 + m2
        return self.__class__(order=self.order, **kwargs)
    def __mul__(self, factor):
        kwargs = {}
        for band in self.order:
            m = self.getFlux(band)
            kwargs[band] = m * factor
        return self.__class__(order=self.order, **kwargs)

    def getFlux(self, bandname):
        return self.getBand(bandname)
    def setFlux(self, bandname, value):
        return self.setBand(bandname, value)
    

class NanoMaggies(Fluxes):
    '''
    A `Brightness` implementation that stores nano-maggies (ie,
    calibrated flux units), which have the advantage of being linear
    and easily convertible to mags.
    '''
    def __repr__(self):
        return str(self)
    def __str__(self):
        s = getClassName(self) + ': '
        ss = []
        for b in self.order:
            f = self.getFlux(b)
            if f <= 0:
                ss.append('%s=(flux %.3g)' % (b,f))
            else:
                m = self.getMag(b)
                ss.append('%s=%.3g' % (b,m))
        s += ', '.join(ss)
        return s

    def getMag(self, band):
        ''' Convert to mag.'''
        flux = self.getFlux(band)
        mag = NanoMaggies.nanomaggiesToMag(flux)
        return mag

    @staticmethod
    def fromMag(mag):
        order = mag.order
        return NanoMaggies(
            order=order,
            **dict([(k,NanoMaggies.magToNanomaggies(mag.getMag(k)))
                    for k in order]))

    @staticmethod
    def magToNanomaggies(mag):
        nmgy = 10. ** ((mag - 22.5) / -2.5)
        return nmgy

    @staticmethod
    def nanomaggiesToMag(nmgy):
        mag = -2.5 * (np.log10(nmgy) - 9)
        return mag

    @staticmethod
    def zeropointToScale(zp):
        '''
        Converts a traditional magnitude zeropoint to a scale factor
        by which nanomaggies should be multiplied to produce image
        counts.
        '''
        return 10.**((zp - 22.5)/2.5)

    @staticmethod
    def scaleToZeropoint(zpscale):
        '''
        Converts a scale factor (by which nanomaggies should be
        multiplied to produce image counts) into a traditional
        magnitude zeropoint.
        '''
        return 22.5 + 2.5 * np.log10(zpscale)
    
    @staticmethod
    def fluxErrorsToMagErrors(flux, flux_invvar):
        flux = np.atleast_1d(flux)
        flux_invvar = np.atleast_1d(flux_invvar)
        dflux = np.zeros(len(flux))
        okiv = (flux_invvar > 0)
        dflux[okiv] = (1./np.sqrt(flux_invvar[okiv]))
        okflux = (flux > 0)
        mag = np.zeros(len(flux))
        mag[okflux] = (NanoMaggies.nanomaggiesToMag(flux[okflux]))
        dmag = np.zeros(len(flux))
        ok = (okiv * okflux)
        dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dflux[ok] / flux[ok]))
        mag[np.logical_not(okflux)] = np.nan
        dmag[np.logical_not(ok)] = np.nan
        return mag.astype(np.float32), dmag.astype(np.float32)


    
class FluxesPhotoCal(BaseParams, ducks.ImageCalibration):
    def __init__(self, band):
        self.band = band
        BaseParams.__init__(self)
    def copy(self):
        return FluxesPhotoCal(self.band)
    def brightnessToCounts(self, brightness):
        flux = brightness.getFlux(self.band)
        return flux
    def __str__(self):
        return 'FluxesPhotoCal(band=%s)' % (self.band)


class MagsPhotoCal(ParamList, ducks.ImageCalibration):
    '''
    A `PhotoCal` implementation to be used with zeropoint-calibrated
    `Mags`.
    '''
    def __init__(self, band, zeropoint):
        '''
        Create a new ``MagsPhotoCal`` object with a *zeropoint* in a
        *band*.

        The ``Mags`` objects you use must have *band* as one of their
        available bands.
        '''
        self.band = band
        # MAGIC
        self.maxmag = 50.
        ParamList.__init__(self, zeropoint)

    def copy(self):
        return MagsPhotoCal(self.band, self.zp)

    @staticmethod
    def getNamedParams():
        return dict(zp=0)

    def getStepSizes(self, *args, **kwargs):
        return [0.01]

    def brightnessToCounts(self, brightness):
        mag = brightness.getMag(self.band)
        if not np.isfinite(mag):
            return 0.
        if mag > self.maxmag:
            return 0.
        return 10.**(0.4 * (self.zp - mag))

    def countsToMag(self, counts):
        return self.zp - 2.5 * np.log10(counts)

    def __str__(self):
        return 'MagsPhotoCal(band=%s, zp=%.3f)' % (self.band, self.zp)

class NullPhotoCal(BaseParams, ducks.ImageCalibration):
    '''
    The "identity" `PhotoCal`, to be used with `Flux` -- the
    `Brightness` objects are in units of `Image` counts.
    '''
    def brightnessToCounts(self, brightness):
        return brightness.getValue()

class LinearPhotoCal(ScalarParam, ducks.ImageCalibration):
    '''
    A `PhotoCal`, to be used with `Flux` or `Fluxes` brightnesses,
    that simply scales the flux by a fixed factor; the brightness
    units are proportional to image counts.
    '''

    def __init__(self, scale, band=None):
        '''
        Creates a new LinearPhotoCal object that scales the Fluxes by
        the given factor to produce image counts.

        If 'band' is not None, will retrieve that band from a `Fluxes` object.
        '''
        super(LinearPhotoCal, self).__init__(scale)
        self.band = band

    def getScale(self):
        return self.val

    def brightnessToCounts(self, brightness):
        if self.band is None:
            counts = brightness.getValue() * self.val
        else:
            counts = brightness.getFlux(self.band) * self.val
        return counts

    def toStandardFitsHeader(self, hdr):
        hdr.add_record(
            dict(name='MAGZP',
                 value=NanoMaggies.scaleToZeropoint(self.getScale()),
                 comment='Zeropoint magnitude'))
    

class NullWCS(BaseParams, ducks.ImageCalibration):
    '''
    The "identity" WCS -- useful when you are using raw pixel
    positions rather than RA,Decs.
    '''
    def __init__(self, pixscale=1., dx=0., dy=0.):
        '''
        pixscale: [arcsec/pix]
        '''
        self.pixscale = pixscale
        self.dx = dx
        self.dy = dy
    def hashkey(self):
        return ('NullWCS', self.dx, self.dy)
    def positionToPixel(self, pos, src=None):
        return pos.x + self.dx, pos.y + self.dy
    def pixelToPosition(self, x, y, src=None):
        return x - self.dx, y - self.dy
    def cdAtPixel(self, x, y):
        return np.array([[1.,0.],[0.,1.]]) * self.pixscale / 3600.

class WcslibWcs(BaseParams, ducks.ImageCalibration):
    '''
    A WCS implementation that wraps a FITS WCS object (with a pixel
    offset), delegating to wcslib.

    FIXME: we could use the "wcssub()" functionality to handle subimages
    rather than x0,y0.

    FIXME: we could implement anwcs_copy() using wcscopy().
    
    '''
    def __init__(self, filename, hdu=0, wcs=None):
        self.x0 = 0.
        self.y0 = 0.
        if wcs is not None:
            self.wcs = wcs
        else:
            from astrometry.util.util import anwcs
            wcs = anwcs(filename, hdu)
            self.wcs = wcs

    # pickling
    def __getstate__(self):
        #print 'pickling wcslib header...'
        s = self.wcs.getHeaderString()
        #print 'pickled wcslib header: len', len(s)
        # print 'Pickling WcslibWcs: string'
        # print '------------------------------------------------'
        # print s
        # print '------------------------------------------------'
        return (self.x0, self.y0, s)

    def __setstate__(self, state):
        (x0,y0,hdrstr) = state
        self.x0 = x0
        self.y0 = y0
        from astrometry.util.util import anwcs_from_string
        # print 'Creating WcslibWcs from header string:'
        # print '------------------------------------------------'
        # print hdrstr
        # print '------------------------------------------------'
        #print 'Unpickling: wcslib header string length:', len(hdrstr)
        self.wcs = anwcs_from_string(hdrstr)
        #print 'unpickling done'
        
    def copy(self):
        raise RuntimeError('unimplemented')

    def __str__(self):
        return ('WcslibWcs: x0,y0 %.3f,%.3f' % (self.x0,self.y0))

    def debug(self):
        from astrometry.util.util import anwcs_print_stdout
        print 'WcslibWcs:'
        anwcs_print_stdout(self.wcs)

    def pixel_scale(self):
        #from astrometry.util.util import anwcs_pixel_scale
        #return anwcs_pixel_scale(self.wcs)
        cd = self.cdAtPixel(self.x0, self.y0)
        return np.sqrt(np.abs(cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0])) * 3600.

    def setX0Y0(self, x0, y0):
        '''
        Sets the pixel offset to apply to pixel coordinates before putting
        them through the wrapped WCS.  Useful when using a cropped image.
        '''
        self.x0 = x0
        self.y0 = y0

    def positionToPixel(self, pos, src=None):
        ok,x,y = self.wcs.radec2pixelxy(pos.ra, pos.dec)
        # MAGIC: 1 for FITS coords.
        return x - 1. - self.x0, y - 1. - self.y0

    def pixelToPosition(self, x, y, src=None):
        # MAGIC: 1 for FITS coords.
        ok,ra,dec = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 1. + self.y0)
        return RaDecPos(ra, dec)

    def cdAtPixel(self, x, y):
        '''
        Returns the ``CD`` matrix at the given ``x,y`` pixel position.

        (Returns the constant ``CD`` matrix elements)
        '''
        ok,ra0,dec0 = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 1. + self.y0)
        ok,ra1,dec1 = self.wcs.pixelxy2radec(x + 2. + self.x0, y + 1. + self.y0)
        ok,ra2,dec2 = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 2. + self.y0)

        cosdec = np.cos(np.deg2rad(dec0))

        return np.array([[(ra1 - ra0)*cosdec, (ra2 - ra0)*cosdec],
                 [dec1 - dec0,        dec2 - dec0]])


class ConstantFitsWcs(ParamList, ducks.ImageCalibration):
    '''
    A WCS implementation that wraps a FITS WCS object (with a pixel
    offset).
    '''
    def __init__(self, wcs):
        '''
        Creates a new ``ConstantFitsWcs`` given an underlying WCS object.
        '''
        self.x0 = 0
        self.y0 = 0
        super(ConstantFitsWcs, self).__init__()
        self.wcs = wcs

    def hashkey(self):
        return (self.x0, self.y0, id(self.wcs))

    def copy(self):
        return ConstantFitsWcs(self.wcs)
        
    def __str__(self):
        return ('%s: x0,y0 %.3f,%.3f, WCS ' % (getClassName(self), self.x0,self.y0)
                + str(self.wcs))

    def getX0Y0(self):
        return self.x0,self.y0

    def setX0Y0(self, x0, y0):
        '''
        Sets the pixel offset to apply to pixel coordinates before putting
        them through the wrapped WCS.  Useful when using a cropped image.
        '''
        self.x0 = x0
        self.y0 = y0

    def positionToPixel(self, pos, src=None):
        '''
        Converts an :class:`tractor.RaDecPos` to a pixel position.
        Returns: tuple of floats ``(x, y)``
        '''
        X = self.wcs.radec2pixelxy(pos.ra, pos.dec)
        # handle X = (ok,x,y) and X = (x,y) return values
        if len(X) == 3:
            ok,x,y = X
        else:
            assert(len(X) == 2)
            x,y = X
        # MAGIC: subtract 1 to convert from FITS to zero-indexed pixels.
        return x - 1 - self.x0, y - 1 - self.y0

    def pixelToPosition(self, x, y, src=None):
        '''
        Converts floats ``x``, ``y`` to a
        :class:`tractor.RaDecPos`.
        '''
        # MAGIC: add 1 to convert from zero-indexed to FITS pixels.
        r,d = self.wcs.pixelxy2radec(x + 1 + self.x0, y + 1 + self.y0)
        return RaDecPos(r,d)

    def cdAtPixel(self, x, y):
        '''
        Returns the ``CD`` matrix at the given ``x,y`` pixel position.

        (Returns the constant ``CD`` matrix elements)
        '''
        cd = self.wcs.get_cd()
        return np.array([[cd[0], cd[1]], [cd[2],cd[3]]])

    def pixel_scale(self):
        return self.wcs.pixel_scale()

    def toStandardFitsHeader(self, hdr):
        if self.x0 != 0 or self.y0 != 0:
            wcs = self.wcs.get_subimage(self.x0, self.y0,
                                        wcs.get_width()-x0,
                                        wcs.get_height()-y0)
        else:
            wcs = self.wcs
        wcs.add_to_header(hdr)

    
### FIXME -- this should be called TanWcs!
class FitsWcs(ConstantFitsWcs, ducks.ImageCalibration):
    '''
    The WCS object must be an astrometry.util.util.Tan object, or a
    convincingly quacking duck.

    You can also give it a filename and HDU.
    '''

    @staticmethod
    def getNamedParams():
        return dict(crval1=0, crval2=1, crpix1=2, crpix2=3,
                    cd1_1=4, cd1_2=5, cd2_1=6, cd2_2=7)

    def __init__(self, wcs, hdu=0):
        '''
        Creates a new ``FitsWcs`` given a :class:`~astrometry.util.util.Tan`
        object.  To create one of these from a filename and FITS HDU extension,

        ::
            from astrometry.util.util import Tan

            fn = 'my-file.fits'
            ext = 0
            FitsWcs(Tan(fn, ext))

        To create one from WCS parameters,

            tanwcs = Tan(crval1, crval2, crpix1, crpix2,
                cd11, cd12, cd21, cd22, imagew, imageh)
            FitsWcs(tanwcs)

        '''
        if hasattr(self, 'x0'):
            print 'FitsWcs has an x0 attr:', self.x0
        self.x0 = 0
        self.y0 = 0

        if isinstance(wcs, basestring):
            from astrometry.util.util import Tan
            wcs = Tan(wcs, hdu)

        super(FitsWcs, self).__init__(wcs)
        # ParamList keeps its params in a list; we don't want to do that.
        del self.vals

    def copy(self):
        from astrometry.util.util import Tan
        wcs = FitsWcs(Tan(self.wcs))
        wcs.setX0Y0(self.x0, self.y0)
        return wcs

    # Here we ASSUME TAN WCS!
    # oi vey...
    def _setThing(self, i, val):
        w = self.wcs
        if i in [0,1]:
            crv = w.crval
            crv[i] = val
            w.set_crval(*crv)
        elif i in [2,3]:
            crp = w.crpix
            crp[i-2] = val
            w.set_crpix(*crp)
        elif i in [4,5,6,7]:
            cd = list(w.get_cd())
            cd[i-4] = val
            w.set_cd(*cd)
        elif i == 8:
            self.x0 = val
        elif i == 9:
            self.y0 = val
        else:
            raise IndexError
    def _getThing(self, i):
        w = self.wcs
        if i in [0,1]:
            crv = w.crval
            return crv[i]
        elif i in [2,3]:
            crp = w.crpix
            return crp[i-2]
        elif i in [4,5,6,7]:
            cd = w.get_cd()
            return cd[i-4]
        elif i == 8:
            return self.x0
        elif i == 9:
            return self.y0
        else:
            raise IndexError
    def _getThings(self):
        w = self.wcs
        return list(w.crval) + list(w.crpix) + list(w.cd) + [self.x0, self.y0]
    def _numberOfThings(self):
        return 10
    def getStepSizes(self, *args, **kwargs):
        pixscale = self.wcs.pixel_scale()
        # we set everything ~ 1 pixel
        dcrval = pixscale / 3600.
        # CD matrix: ~1 pixel over image size (call it 1000)
        dcd = pixscale / 3600. / 1000.
        ss = [dcrval, dcrval, 1., 1., dcd, dcd, dcd, dcd, 1., 1.]
        return list(self._getLiquidArray(ss))

class PixPos(ParamList):
    '''
    A Position implementation using pixel positions.
    '''
    @staticmethod
    def getNamedParams():
        return dict(x=0, y=1)
    def __init__(self, *args):
        super(PixPos, self).__init__(*args)
        self.stepsize = [0.1, 0.1]
    def __str__(self):
        return 'pixel (%.2f, %.2f)' % (self.x, self.y)
    def getDimension(self):
        return 2

class RaDecPos(ArithmeticParams, ParamList):
    '''
    A Position implementation using RA,Dec positions, in degrees.

    Attributes:
      * ``.ra``
      * ``.dec``
    '''
    @staticmethod
    def getName():
        return "RaDecPos"
    @staticmethod
    def getNamedParams():
        return dict(ra=0, dec=1)
    def __str__(self):
        return '%s: RA, Dec = (%.5f, %.5f)' % (self.getName(), self.ra, self.dec)
    def __init__(self, *args, **kwargs):
        super(RaDecPos,self).__init__(*args,**kwargs)
        #self.setStepSizes(1e-4)
        delta = 1e-4
        self.setStepSizes([delta / np.cos(np.deg2rad(self.dec)), delta])
    def getDimension(self):
        return 2
    #def setStepSizes(self, delta):
    #    self.stepsizes = [delta / np.cos(np.deg2rad(self.dec)),delta]

    def distanceFrom(self, pos):
        from astrometry.util.starutil_numpy import degrees_between
        return degrees_between(self.ra, self.dec, pos.ra, pos.dec)


class NullSky(BaseParams, ducks.Sky):
    '''
    A Sky implementation that does nothing; the background level is
    zero.
    '''
    pass
    
class ConstantSky(ScalarParam, ducks.ImageCalibration):
    '''
    A simple constant sky level across the whole image.

    This sky object has one parameter, the constant level.
    
    The sky level is specified in the same units as the image
    ("counts").
    '''
    def getParamDerivatives(self, tractor, img, srcs):
        p = Patch(0, 0, np.ones_like(img.getImage()))
        p.setName('dsky')
        return [p]
    def addTo(self, img, scale=1.):
        if self.val == 0:
            return
        img += (self.val * scale)
    def getParamNames(self):
        return ['sky']

    def getConstant(self):
        return self.val

    def subtract(self, con):
        self.val -= con

    def toStandardFitsHeader(self, hdr):
        hdr.add_record(dict(name='SKY', comment='Sky value in Tractor model',
                            value=self.val))

# class OffsetConstantSky(ConstantSky):
#     '''
#     Presents an offset *in the parameter interface only*.
#     '''
#     def __init__(self, val):
#         super(OffsetConstantSky, self).__init__(val)
#         self.offset = 0.
# 
#     def getParams(self):
#         return [self.val + self.offset]
#     def setParams(self, p):
#         assert(len(p) == 1)
#         self._set(p[0] - self.offset)
#     def setParam(self, i, p):
#         assert(i == 0)
#         oldval = self.val
#         self._set(p - self.offset)
#         return oldval + self.offset


class BasicSource(ducks.Source):
    def getPosition(self):
        return self.pos
    def setPosition(self, position):
        self.pos = position


class SingleProfileSource(BasicSource):
    '''
    A mix-in class for Source objects that have a single profile, eg, PointSources,
    Dev, Exp, and Sersic galaxies, and also FixedCompositeGalaxy (surprising but true)
    but not CompositeGalaxy.
    '''

    def getBrightness(self):
        return self.brightness
    def setBrightness(self, brightness):
        self.brightness = brightness

    def getBrightnesses(self):
        return [self.getBrightness()]

    def getUnitFluxModelPatches(self, *args, **kwargs):
        return [self.getUnitFluxModelPatch(*args, **kwargs)]

    def getModelPatch(self, img, minsb=None, modelMask=None):
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None
        if minsb is None:
            minsb = img.modelMinval
        minval = minsb / counts
        upatch = self.getUnitFluxModelPatch(img, minval=minval, modelMask=modelMask)
        if upatch is None:
            return None
        return upatch * counts



class PointSource(MultiParams, SingleProfileSource):
    '''
    An implementation of a point source, characterized by its position
    and brightness.

    '''
    def __init__(self, pos, br):
        '''
        PointSource(pos, brightness)
        '''
        super(PointSource, self).__init__(pos, br)
        # if not None, fixedRadius determines the size of unit-flux
        # model Patches produced for this PointSource.
        self.fixedRadius = None
        # if not None, minradius determines the minimum size of unit-flux
        # models
        self.minRadius = None
    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)
    def getSourceType(self):
        return 'PointSource'
    def __str__(self):
        return (self.getSourceType() + ' at ' + str(self.pos) +
                ' with ' + str(self.brightness))
    def __repr__(self):
        return (self.getSourceType() + '(' + repr(self.pos) + ', ' +
                repr(self.brightness) + ')')

    def getUnitFluxModelPatch(self, img, minval=0., derivs=False, modelMask=None):
        (px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
        H,W = img.shape
        psf = self._getPsf(img)
        # quit early if the requested position is way outside the image bounds
        r = self.fixedRadius
        if r is None:
            r = psf.getRadius()
        if px + r < 0 or px - r > W or py + r < 0 or py - r > H:
            return None
        patch = psf.getPointSourcePatch(px, py, minval=minval, extent=[0,W,0,H],
                                        radius=self.fixedRadius, derivs=derivs,
                                        minradius=self.minRadius, modelMask=modelMask)
        return patch

    def _getPsf(self, img):
        return img.getPsf()

    def getParamDerivatives(self, img, fastPosDerivs=True, modelMask=None):
        '''
        returns [ Patch, Patch, ... ] of length numberOfParams().
        '''
        # Short-cut the case where we're only fitting fluxes, and the
        # band of the image is not being fit.
        counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
        if self.isParamFrozen('pos') and not self.isParamFrozen('brightness'):
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            allzero = True
            for i,bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                if countsi != counts0:
                    allzero = False
                    break
            if allzero:
                return [None]*self.numberOfParams()

        pos = self.getPosition()
        wcs = img.getWcs()

        minsb = img.modelMinval
        if counts0 > 0:
            minval = minsb / counts0
        else:
            minval = None

        derivs = (not self.isParamFrozen('pos')) and fastPosDerivs
        patchdx,patchdy = None,None
        
        if derivs:
            patches = self.getUnitFluxModelPatch(img, minval=minval, derivs=True,
                                                 modelMask=modelMask)
            #print 'minval=', minval, 'Patches:', patches
            if patches is None:
                return [None]*self.numberOfParams()
            if not isinstance(patches, tuple):
                patch0 = patches
                #print 'img:', img
                #print 'counts0:', counts0
            else:
                patch0, patchdx, patchdy = patches

        else:
            patch0 = self.getUnitFluxModelPatch(img, minval=minval, modelMask=modelMask)

        if patch0 is None:
            return [None]*self.numberOfParams()
        # check for intersection of patch0 with img
        H,W = img.shape
        if not patch0.overlapsBbox((0, W, 0, H)):
            return [None]*self.numberOfParams()
        
        derivs = []

        # Position
        if not self.isParamFrozen('pos'):

            if patchdx is not None and patchdy is not None:

                # Convert x,y derivatives to Position derivatives

                px,py = wcs.positionToPixel(pos, self)
                cd = wcs.cdAtPixel(px, py)
                cdi = np.linalg.inv(cd)
                # Get thawed Position parameter indices
                thawed = pos.getThawedParamIndices()
                for i,pname in zip(thawed, pos.getParamNames()):
                    deriv = (patchdx * cdi[0,i] + patchdy * cdi[1,i]) * counts0
                    deriv.setName('d(ptsrc)/d(pos.%s)' % pname)
                    derivs.append(deriv)

            elif counts0 == 0:
                derivs.extend([None] * pos.numberOfParams())
            else:
                psteps = pos.getStepSizes(img)
                pvals = pos.getParams()
                for i,pstep in enumerate(psteps):
                    oldval = pos.setParam(i, pvals[i] + pstep)
                    patchx = self.getUnitFluxModelPatch(img, minval=minval,
                                                        modelMask=modelMask)
                
                    pos.setParam(i, oldval)
                    if patchx is None:
                        dx = patch0 * (-1 * counts0 / pstep)
                    else:
                        dx = (patchx - patch0) * (counts0 / pstep)
                    dx.setName('d(ptsrc)/d(pos%i)' % i)
                    derivs.append(dx)

        # Brightness
        if not self.isParamFrozen('brightness'):
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            for i,bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts0) / bstep)
                df.setName('d(ptsrc)/d(bright%i)' % i)
                derivs.append(df)
        return derivs

class PixelizedPSF(BaseParams, ducks.ImageCalibration):
    '''
    A PSF model based on an image postage stamp, which will be
    sinc-shifted to subpixel positions.

    Galaxies will be rendering using FFT convolution.
    
    FIXME -- currently this class claims to have no params.
    '''
    def __init__(self, img, Lorder=3):
        self.img = img
        H,W = img.shape
        assert((H % 2) == 1)
        assert((W % 2) == 1)
        self.Lorder = Lorder
        self.fftcache = {}
        
    def __str__(self):
        return 'PixelizedPSF'

    def hashkey(self):
        return ('PixelizedPSF', tuple(self.img.ravel()))

    def copy(self):
        return PixelizedPSF(self.img.copy())

    def getRadius(self):
        H,W = self.img.shape
        return np.hypot(H,W)/2.

    def getPointSourcePatch(self, px, py, minval=0., modelMask=None, **kwargs):

        ## FIXME!
        assert(modelMask is None)

        from scipy.ndimage.filters import correlate1d
        H,W = self.img.shape
        ix = int(np.round(px))
        iy = int(np.round(py))
        dx = px - ix
        dy = py - iy
        x0 = ix - W/2
        y0 = iy - H/2
        L = self.Lorder
        Lx = lanczos_filter(L, np.arange(-L, L+1) + dx)
        Ly = lanczos_filter(L, np.arange(-L, L+1) + dy)
        sx      = correlate1d(self.img, Lx, axis=1, mode='constant')
        shifted = correlate1d(sx,       Ly, axis=0, mode='constant')
        #shifted /= (Lx.sum() * Ly.sum())
        #print 'Shifted PSF: range', shifted.min(), shifted.max()

        ### ???
        #shifted = np.maximum(shifted, 0.)

        shifted /= shifted.sum()
        return Patch(x0, y0, shifted)

    def getFourierTransformSize(self, radius):
        ## FIXME -- power-of-2 MINUS one to keep things odd...?
        sz = 2**int(np.ceil(np.log2(radius*2.))) - 1
        return sz
    
    def getFourierTransform(self, radius):
        sz = self.getFourierTransformSize(radius)
        if sz in self.fftcache:
            return self.fftcache[sz]
        H,W = self.img.shape
        subimg = self.img
        pad = np.zeros((sz,sz))
        if sz > H:
            y0 = (sz - H)/2
        else:
            y0 = 0
            d = (H - sz)/2
            subimg = subimg[d:-d, :]
        if sz > W:
            x0 = (sz - W)/2
        else:
            x0 = 0
            d = (W - sz)/2
            subimg = subimg[:, d:-d]
        sh,sw = subimg.shape
        pad[y0:y0+sh, x0:x0+sw] = subimg
        P = np.fft.rfft2(pad)
        rtn = P, (sz/2, sz/2), pad.shape
        self.fftcache[sz] = rtn
        return rtn
    
class GaussianMixturePSF(ParamList, ducks.ImageCalibration):
    '''
    A PSF model that is a mixture of general 2-D Gaussians
    (characterized by amplitude, mean, covariance)
    '''
    def __init__(self, *args):
        '''
        GaussianMixturePSF(amp, mean, var)

        or

        GaussianMixturePSF(a0,a1,a2, mx0,my0,mx1,my1,mx2,my2,
                           vxx0,vyy0,vxy0, vxx1,vyy1,vxy1, vxx2,vyy2,vxy2)
        
        amp:  np array (size K) of Gaussian amplitudes
        mean: np array (size K,2) of means
        var:  np array (size K,2,2) of variances
        '''
        if len(args) == 3:
            amp, mean, var = args
        else:
            assert(len(args) % 6 == 0)
            K = len(args) / 6
            amp  = np.array(args[:K])
            mean = np.array(args[K:3*K]).reshape((K,2))
            args = args[3*K:]
            var  = np.zeros((K,2,2))
            var[:,0,0] = args[::3]
            var[:,1,1] = args[1::3]
            var[:,0,1] = var[:,1,0] = args[2::3]

        self.mog = mp.MixtureOfGaussians(amp, mean, var)
        assert(self.mog.D == 2)
        self.radius = 25
        super(GaussianMixturePSF, self).__init__()

        del self.vals
        
        K = self.mog.K
        self.stepsizes = [0.01]*K + [0.01]*(K*2) + [0.1]*(K*3)
        self._set_param_names(K)

    def _set_param_names(self, K):
        # ordering: A0, A1, ... Ak, mux0, muy0, mux1, muy1, mux2, muy2, ...
        #   var0xx,var0yy,var0xy, var1xx, var1yy, var1xy
        names = {}
        for k in range(K):
            names['amp%i'%k] = k
            names['meanx%i'%k] = K+(k*2)
            names['meany%i'%k] = K+(k*2)+1
            names['varxx%i'%k] = K*3 + (k*3)
            names['varyy%i'%k] = K*3 + (k*3)+1
            names['varxy%i'%k] = K*3 + (k*3)+2
        # print 'Setting param names:', names
        self.addNamedParams(**names)
        
    def get_wmuvar(self):
        return (self.mog.amp, self.mog.mean, self.mog.var)
        
    @classmethod
    def fromFitsHeader(clazz, hdr, prefix=''):
        params = []
        for i in range(100):
            k = prefix + 'P%i' % i
            print 'Key', k
            if not k in hdr:
                break
            params.append(hdr.get(k))
        print 'PSF Params:', params
        if len(params) == 0 or (len(params) % 6 != 0):
            raise RuntimeError('Failed to create %s from FITS header: expected '
                               'factor of 6 parameters, got %i' % 
                               (str(clazz), len(params)))
        K = len(params) / 6
        psf = clazz(np.zeros(K), np.zeros((K,2)), np.zeros((K,2,2)))
        psf.setParams(params)
        return psf
    
    def getMixtureOfGaussians(self, px=None, py=None):
        return self.mog

    def applyTo(self, image):
        raise
    
    def scaleBy(self, factor):
        # Use not advised, ever
        amp = self.mog.amp
        mean = self.mog.mean * factor
        var = self.mog.var * factor**2
        return GaussianMixturePSF(amp, mean, var)

    def shiftBy(self, dx, dy):
        self.mog.mean[:,0] += dx
        self.mog.mean[:,1] += dy
    
    def computeRadius(self):
        import numpy.linalg
        # ?
        meig = max([max(abs(numpy.linalg.eigvalsh(v)))
                    for v in self.mog.var])
        return self.getNSigma() * np.sqrt(meig)
        
    def getNSigma(self):
        # MAGIC -- N sigma for rendering patches
        return 5.

    def getRadius(self):
        return self.radius

    # returns a Patch object.
    def getPointSourcePatch(self, px, py, minval=0., extent=None, radius=None,
                            derivs=False, minradius=None, modelMask=None, **kwargs):
        '''
        extent = [x0,x1,y0,y1], clip to [x0,x1), [y0,y1).
        '''

        ## FIXME!
        assert(modelMask is None)

        if minval is None:
            minval = 0.
        if minval > 0. or minradius is not None:
            if radius is not None:
                rr = radius
            elif self.radius is not None:
                rr = self.radius
            else:
                r = 0.
                for v in self.mog.var:
                    # overestimate
                    vv = (v[0,0] + v[1,1])
                    norm = 2. * np.pi * np.linalg.det(v)
                    r2 = vv * -2. * np.log(minval * norm)
                    if r2 > 0:
                        r = max(r, np.sqrt(r2))
                rr = int(np.ceil(r))

            x0 = int(floor(px - rr))
            x1 = int(ceil (px + rr)) + 1
            y0 = int(floor(py - rr))
            y1 = int(ceil (py + rr)) + 1

            if extent is not None:
                [xl,xh,yl,yh] = extent
                # clip
                x0 = max(x0, xl)
                x1 = min(x1, xh)
                y0 = max(y0, yl)
                y1 = min(y1, yh)

            if x0 >= x1:
                return None
            if y0 >= y1:
                return None

            kwa = {}
            if minradius is not None:
                kwa['minradius'] = minradius
            
            return self.mog.evaluate_grid_approx3(
                x0, x1, y0, y1, px, py, minval, derivs=derivs, **kwa)
            

        if radius is None:
            r = self.getRadius()
        else:
            r = radius
        x0,x1 = int(floor(px-r)), int(ceil(px+r)) + 1
        y0,y1 = int(floor(py-r)), int(ceil(py+r)) + 1
        return self.mog.evaluate_grid(x0, x1, y0, y1, px, py)

    def __str__(self):
        return (
            'GaussianMixturePSF: amps=' + str(tuple(self.mog.amp.ravel())) +
            ', means=' + str(tuple(self.mog.mean.ravel())) +
            ', var=' + str(tuple(self.mog.var.ravel())))

    def _numberOfThings(self):
        K = self.mog.K
        return K * (1 + 2 + 3)
    def _getThings(self):
        p = list(self.mog.amp) + list(self.mog.mean.ravel())
        for v in self.mog.var:
            p += (v[0,0], v[1,1], v[0,1])
        return p
    def _getThing(self, i):
        return self._getThings()[i]
    def _setThings(self, p):
        K = self.mog.K
        self.mog.amp = np.atleast_1d(p[:K])
        pp = p[K:]
        self.mog.mean = np.atleast_2d(pp[:K*2]).reshape(K,2)
        pp = pp[K*2:]
        self.mog.var[:,0,0] = pp[::3]
        self.mog.var[:,1,1] = pp[1::3]
        self.mog.var[:,0,1] = self.mog.var[:,1,0] = pp[2::3]
    def _setThing(self, i, p):
        K = self.mog.K
        if i < K:
            old = self.mog.amp[i]
            self.mog.amp[i] = p
            return old
        i -= K
        if i < K*2:
            old = self.mog.mean.ravel()[i]
            self.mog.mean.ravel()[i] = p
            return old
        i -= K*2
        j = i / 3
        k = i % 3
        if k in [0,1]:
            old = self.mog.var[j,k,k]
            self.mog.var[j,k,k] = p
            return old
        old = self.mog.var[j,0,1]
        self.mog.var[j,0,1] = p
        self.mog.var[j,1,0] = p
        return old

    @staticmethod
    def fromStamp(stamp, N=3, P0=None, xy0=None, alpha=0.,
                  emsteps=1000, v2=False, approx=1e-30,
                  v3=False):
        '''
        optional P0 = (w,mu,var): initial parameter guess.

        w has shape (N,)
        mu has shape (N,2)
        var (variance) has shape (N,2,2)

        optional xy0 = int x0,y0 origin of stamp.
        '''
        from emfit import em_fit_2d, em_fit_2d_reg
        from fitpsf import em_init_params
        if P0 is not None:
            w,mu,var = P0
        else:
            w,mu,var = em_init_params(N, None, None, None)
        stamp = stamp.copy()
        
        if xy0 is None:
            xm, ym = -(stamp.shape[1]/2), -(stamp.shape[0]/2)
        else:
            xm, ym = xy0

        if v3:
            tpsf = GaussianMixturePSF(w, mu, var)
            tim = Image(data=stamp, invvar=1e6*np.ones_like(stamp),
                        psf=tpsf)
            h,w = tim.shape
            src = PointSource(PixPos(w/2, h/2), Flux(1.))
            tr = Tractor([tim],[src])
            tr.freezeParam('catalog')
            tim.freezeAllBut('psf')
            tim.modelMinval = approx
            for step in range(20):
                dlnp,X,alpha = tr.optimize(shared_params=False)
                print 'dlnp', dlnp
                if dlnp < 1e-6:
                    break
            return tpsf
                
        elif v2:
            from emfit import em_fit_2d_reg2
            print 'stamp sum:', np.sum(stamp)
            #stamp *= 1000.
            ok,skyamp = em_fit_2d_reg2(stamp, xm, ym, w, mu, var, alpha,
                                       emsteps, approx)
            #print 'sky amp:', skyamp
            #print 'w sum:', sum(w)
            tpsf = GaussianMixturePSF(w, mu, var)
            return tpsf,skyamp
        else:

            stamp /= stamp.sum()
            stamp = np.maximum(stamp, 0)

            em_fit_2d_reg(stamp, xm, ym, w, mu, var, alpha, emsteps)

        tpsf = GaussianMixturePSF(w, mu, var)
        return tpsf


class GaussianMixtureEllipsePSF(GaussianMixturePSF):
    '''
    A variant of GaussianMixturePSF that uses EllipseESoft to describe
    the covariance ellipse.
    '''

    def __init__(self, *args):
        '''
        args = (amp, mean, ell)

        or

        args = (a0,a1,..., mx0,my0,mx1,my1,..., logr0,ee1-0,ee2-0,logr1,ee1-2,...)

        
        amp:  np array (size K) of Gaussian amplitudes
        mean: np array (size K,2) of means
        ell:  list (length K) of EllipseESoft objects
        '''
        if len(args) == 3:
            amp, mean, ell = args
        else:
            from .ellipses import EllipseESoft
            assert(len(args) % 6 == 0)
            K = len(args) / 6
            amp  = np.array(args[:K])
            mean = np.array(args[K:3*K]).reshape((K,2))
            args = args[3*K:]
            ell = [EllipseESoft(*args[3*k: 3*(k+1)]) for k in range(K)]

        K = len(amp)
        var = np.zeros((K,2,2))
        for k in range(K):
            var[k,:,:] = self.ellipseToVariance(ell[k])
        self.ellipses = [e.copy() for e in ell]
        super(GaussianMixtureEllipsePSF, self).__init__(amp, mean, var)
        self.stepsizes = [0.001]*K + [0.001]*(K*2) + [0.001]*(K*3)

    def ellipseToVariance(self, ell):
        return ell.getCovariance()
        
    def _set_param_names(self, K):
        names = {}
        for k in range(K):
            names['amp%i'%k] = k
            names['meanx%i'%k] = K+(k*2)
            names['meany%i'%k] = K+(k*2)+1
            names['logr%i'%k] = K*3 + (k*3)
            names['ee1-%i'%k] = K*3 + (k*3)+1
            names['ee2-%i'%k] = K*3 + (k*3)+2
        self.addNamedParams(**names)

    def toMog(self):
        return GaussianMixturePSF(self.mog.amp, self.mog.mean, self.mog.var)
        
    def __str__(self):
        return (
            'GaussianMixtureEllipsePSF: amps=' +
            '['+', '.join(['%.3f' % a for a in self.mog.amp.ravel()]) + ']' +
            ', means=[' + ', '.join([
                '(%.3f, %.3f)' % (x,y) for x,y in self.mog.mean]) + ']' +
            ', ellipses=' + ', '.join(str(e) for e in self.ellipses) +
            ', var=' + str(tuple(self.mog.var.ravel())))
    
    def _getThings(self):
        p = list(self.mog.amp) + list(self.mog.mean.ravel())
        for e in self.ellipses:
            p += e.getAllParams()
        return p
    def _setThings(self, p):
        K = self.mog.K
        self.mog.amp = np.atleast_1d(p[:K])
        pp = p[K:]
        self.mog.mean = np.atleast_2d(pp[:K*2]).reshape(K,2)
        pp = pp[K*2:]
        for i,e in enumerate(self.ellipses):
            e.setAllParams(pp[:3])
            pp = pp[3:]
            self.mog.var[i,:,:] = self.ellipseToVariance(e)
    def _setThing(self, i, p):
        ## hack
        things = self._getThings()
        old = things[i]
        things[i] = p
        self._setThings(things)
        return old

    @staticmethod
    def fromStamp(stamp, N=3, P0=None, approx=1e-6, damp=0.):
        '''
        optional P0 = (list of floats): initial parameter guess.

        (parameters of a GaussianMixtureEllipsePSF)
        '''
        from .ellipses import EllipseESoft
        w = np.ones(N) / float(N)
        mu = np.zeros((N,2))
        ell = [EllipseESoft(np.log(2*r), 0., 0.) for r in range(1, N+1)]
        psf = GaussianMixtureEllipsePSF(w, mu, ell)
        if P0 is not None:
            psf.setAllParams(P0)
        tim = Image(data=stamp, invvar=1e6*np.ones_like(stamp), psf=psf)
        H,W = stamp.shape
        src = PointSource(PixPos(W/2, H/2), Flux(1.))
        tr = Tractor([tim],[src])
        tr.freezeParam('catalog')
        tim.freezeAllBut('psf')
        #print 'Fitting:'
        #tr.printThawedParams()
        tim.modelMinval = approx
        alphas = [0.1, 0.3, 1.0]
        for step in range(50):
            dlnp,X,alpha = tr.optimize(shared_params=False, alphas=alphas,
                                       damp=damp)
            #print 'dlnp', dlnp, 'alpha', alpha
            #print 'X', X
            if dlnp < 1e-6:
                break
            #print 'psf', psf
        return psf
    
class NCircularGaussianPSF(MultiParams, ducks.ImageCalibration):
    '''
    A PSF model using N concentric, circular Gaussians.
    '''
    def __init__(self, sigmas, weights):
        '''
        Creates a new N-Gaussian (concentric, isotropic) PSF.

        sigmas: (list of floats) standard deviations of the components

        weights: (list of floats) relative weights of the components;
        given two components with weights 0.9 and 0.1, the total mass
        due to the second component will be 0.1.  These values will be
        normalized so that the total mass of the PSF is 1.0.

        eg,   NCircularGaussianPSF([1.5, 4.0], [0.8, 0.2])
        '''
        assert(len(sigmas) == len(weights))
        psigmas = ParamList(*sigmas)
        psigmas.stepsizes = [0.01] * len(sigmas)
        pweights = ParamList(*weights)
        pweights.stepsizes = [0.01] * len(weights)
        super(NCircularGaussianPSF, self).__init__(psigmas, pweights)
        self.minradius = 1.

    @property
    def amp(self):
        return self.weights

    @property
    def mog(self):
        return self.getMixtureOfGaussians()
    
    @staticmethod
    def getNamedParams():
        return dict(sigmas=0, weights=1)

    def __str__(self):
        return ('NCircularGaussianPSF: sigmas [ ' +
                ', '.join(['%.3f'%s for s in self.sigmas]) +
                ' ], weights [ ' +
                ', '.join(['%.3f'%w for w in self.weights]) +
                ' ]')

    def __repr__(self):
        return ('NCircularGaussianPSF: sigmas [ ' +
                ', '.join(['%.3f'%s for s in self.sigmas]) +
                ' ], weights [ ' +
                ', '.join(['%.3f'%w for w in self.weights]) +
                ' ]')

    # Get the real underlying ones without paying attention to frozen state
    @property
    def mysigmas(self):
        return self.sigmas.vals
    @property
    def myweights(self):
        return self.weights.vals

    def scale(self, factor):
        ''' Returns a new PSF that is *factor* times wider. '''
        return NCircularGaussianPSF(np.array(self.mysigmas) * factor, self.myweights)

    def getMixtureOfGaussians(self, px=None, py=None):
        K = len(self.myweights)
        amps = np.array(self.myweights)
        means = np.zeros((K,2))
        vars = np.zeros((K,2,2))
        for k in range(K):
            vars[k,0,0] = vars[k,1,1] = self.mysigmas[k]**2
        return mp.MixtureOfGaussians(amps, means, vars)
        
    def hashkey(self):
        hk = ('NCircularGaussianPSF', tuple(self.sigmas), tuple(self.weights))
        return hk
    
    def copy(self):
        return NCircularGaussianPSF(list([s for s in self.sigmas]),
                                    list([w for w in self.weights]))

    def applyTo(self, image):
        from scipy.ndimage.filters import gaussian_filter
        # gaussian_filter normalizes the Gaussian; the output has ~ the
        # same sum as the input.
        res = np.zeros_like(image)
        for s,w in zip(self.sigmas, self.weights):
            res += w * gaussian_filter(image, s)
        res /= sum(self.weights)
        return res

    def getNSigma(self):
        # HACK - MAGIC -- N sigma for rendering patches
        return 5.

    def getRadius(self):
        if hasattr(self, 'radius'):
            return self.radius
        return max(self.minradius, max(self.mysigmas) * self.getNSigma())

    # returns a Patch object.
    def getPointSourcePatch(self, px, py, minval=0., radius=None,
                            modelMask=None, **kwargs):

        ## FIXME!
        assert(modelMask is None)

        ix = int(round(px))
        iy = int(round(py))
        if radius is None:
            rad = int(ceil(self.getRadius()))
        else:
            rad = radius
        x0 = ix - rad
        x1 = ix + rad + 1
        y0 = iy - rad
        y1 = iy + rad + 1
        mix = self.getMixtureOfGaussians()
        mix.mean[:,0] += px
        mix.mean[:,1] += py
        return mp.mixture_to_patch(mix, x0, x1, y0, y1, minval=minval)

# class SubImage(Image):
#   def __init__(self, im, roi,
#                skyclass=SubSky,
#                psfclass=SubPsf,
#                wcsclass=SubWcs):
#       (x0,x1,y0,y1) = roi
#       slc = (slice(y0,y1), slice(x0,x1))
#       data = im.getImage[slc]
#       invvar = im.getInvvar[slc]
#       sky = skyclass(im.getSky(), roi)
#       psf = psfclass(im.getPsf(), roi)
#       wcs = wcsclass(im.getWcs(), roi)
#       pcal = im.getPhotoCal()
#       super(SubImage, self).__init__(data=data, invvar=invvar, psf=psf,
#                                      wcs=wcs, sky=sky, photocal=photocal,
#                                      name='sub'+im.name)
# 
# class SubSky(object):
#   def __init__(self, sky, roi):
#       self.sky = sky
#       self.roi = roi
# 
#   #def getParamDerivatives(self, img):
#   def addTo(self, mod):

class ParamsWrapper(BaseParams):
    def __init__(self, real):
        self.real = real
    def hashkey(self):
        return self.real.hashkey()
    def getLogPrior(self):
        return self.real.getLogPrior()
    def getLogPriorDerivatives(self):
        return self.real.getLogPriorDerivatives()
    def getParams(self):
        return self.real.getParams()
    def setParams(self, x):
        return self.real.setParams(x)
    def setParam(self, i, x):
        return self.real.setParam(i, x)
    def numberOfParams(self):
        return self.real.numberOfParams()
    def getParamNames(self):
        return self.real.getParamNames()
    def getStepSizes(self, *args, **kwargs):
        return self.real.getStepSizes(*args, **kwargs)


class ShiftedPsf(ParamsWrapper, ducks.ImageCalibration):
    def __init__(self, psf, x0, y0):
        super(ShiftedPsf, self).__init__(psf)
        self.psf = psf
        self.x0 = x0
        self.y0 = y0
    def __str__(self):
        return ('ShiftedPsf: %i,%i + ' % (self.x0,self.y0)) + str(self.psf)
    def hashkey(self):
        return ('ShiftedPsf', self.x0, self.y0) + self.psf.hashkey()
    def getPointSourcePatch(self, px, py, extent=None, derivs=False, **kwargs):
        if extent is not None:
            (ex0,ex1,ey0,ey1) = extent
            extent = (ex0+self.x0, ex1+self.x0, ey0+self.y0, ey1+self.y0)
        p = self.psf.getPointSourcePatch(self.x0 + px, self.y0 + py,
                                         extent=extent, derivs=derivs, **kwargs)
        # Now we have to shift the patch back too
        if p is None:
            return None
        if derivs and isinstance(p, tuple):
            p,dx,dy = p
            p.x0 -= self.x0
            p.y0 -= self.y0
            dx.x0 -= self.x0
            dx.y0 -= self.y0
            dy.x0 -= self.x0
            dy.y0 -= self.y0
            return p,dx,dy
        p.x0 -= self.x0
        p.y0 -= self.y0
        return p

    def getRadius(self):
        return self.psf.getRadius()

    def getMixtureOfGaussians(self, px=None, py=None, **kwargs):
        if px is not None:
            px = px + self.x0
        if py is not None:
            py = py + self.y0
        return self.psf.getMixtureOfGaussians(px=px, py=py, **kwargs)
    
class ScaledPhotoCal(ParamsWrapper, ducks.ImageCalibration):
    def __init__(self, photocal, factor):
        super(ScaledPhotoCal,self).__init__(photocal)
        self.pc = photocal
        self.factor = factor
    def hashkey(self):
        return ('ScaledPhotoCal', self.factor) + self.pc.hashkey()
    def brightnessToCounts(self, brightness):
        return self.factor * self.pc.brightnessToCounts(brightness)

class ScaledWcs(ParamsWrapper, ducks.ImageCalibration):
    def __init__(self, wcs, factor):
        super(ScaledWcs,self).__init__(wcs)
        self.factor = factor
        self.wcs = wcs

    def hashkey(self):
        return ('ScaledWcs', self.factor) + tuple(self.wcs.hashkey())

    def cdAtPixel(self, x, y):
        x,y = (x + 0.5) / self.factor - 0.5, (y + 0.5) * self.factor - 0.5
        cd = self.wcs.cdAtPixel(x, y)
        return cd / self.factor

    def positionToPixel(self, pos, src=None):
        x,y = self.wcs.positionToPixel(pos, src=src)
        # Or somethin'
        return ((x + 0.5) * self.factor - 0.5,
                (y + 0.5) * self.factor - 0.5)

class ShiftedWcs(ParamsWrapper, ducks.ImageCalibration):
    '''
    Wraps a WCS in order to use it for a subimage.
    '''
    def __init__(self, wcs, x0, y0):
        super(ShiftedWcs,self).__init__(wcs)
        self.x0 = x0
        self.y0 = y0
        self.wcs = wcs

    def toFitsHeader(self, hdr, prefix=''):
        tt = type(self.wcs)
        sub_type = '%s.%s' % (tt.__module__, tt.__name__)
        hdr.add_record(dict(name=prefix + 'SUB', value=sub_type,
                            comment='ShiftedWcs sub-type'))
        hdr.add_record(dict(name=prefix + 'X0', value=self.x0,
                            comment='ShiftedWcs x0'))
        hdr.add_record(dict(name=prefix + 'Y0', value=self.y0,
                            comment='ShiftedWcs y0'))
        print 'Sub wcs:', self.wcs
        self.wcs.toFitsHeader(hdr, prefix=prefix)
        
    def hashkey(self):
        return ('ShiftedWcs', self.x0, self.y0) + tuple(self.wcs.hashkey())

    def cdAtPixel(self, x, y):
        return self.wcs.cdAtPixel(x + self.x0, y + self.y0)

    def positionToPixel(self, pos, src=None):
        x,y = self.wcs.positionToPixel(pos, src=src)
        return (x - self.x0, y - self.y0)

    def pixelToPosition(self, x, y, src=None):
        pos = self.wcs.pixelToPosition(x+self.x0, y+self.y0, src=src)
        return pos

    
