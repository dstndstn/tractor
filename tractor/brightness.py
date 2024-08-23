import numpy as np
from tractor.utils import ScalarParam, ParamList, BaseParams, getClassName
from tractor import ducks


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
            keys = list(kwargs.keys())
            keys.sort()
        assert(len(kwargs) == len(keys))
        assert(set(kwargs.keys()) == set(keys))
        vals = []
        for k in keys:
            vals.append(kwargs[k])
        super(MultiBandBrightness, self).__init__(*vals)
        self.order = keys
        self.addNamedParams(**dict((k, i) for i, k in enumerate(keys)))

    def __setstate__(self, state):
        '''For pickling.'''
        self.__dict__ = state
        self.addNamedParams(**dict((k, i)
                                   for i, k in enumerate(self.order)))

    def copy(self):
        return self * 1.

    def getBand(self, band):
        return getattr(self, band)

    def setBand(self, band, value):
        return setattr(self, band, value)


class Mags(MultiBandBrightness):
    '''
    An implementation of `Brightness` that stores magnitudes in
    multiple bands.

    Works with MagsPhotoCal.
    '''

    def __init__(self, **kwargs):
        '''
        Mags(r=14.3, g=15.6, order=['r','g'])

        The `order` parameter is optional; it determines the ordering
        of the bands in the parameter vector (eg, `getParams()`).
        '''
        super(Mags, self).__init__(**kwargs)
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
            msum = -2.5 * np.log10(10.**(-m1 / 2.5) + 10.**(-m2 / 2.5))
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
                ss.append('%s=(flux %.5g)' % (b, f))
            else:
                m = self.getMag(b)
                ss.append('%s=%.5g' % (b, m))
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
            **dict([(k, NanoMaggies.magToNanomaggies(mag.getMag(k)))
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
        return 10.**((zp - 22.5) / 2.5)

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
        dflux[okiv] = (1. / np.sqrt(flux_invvar[okiv]))
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

    def copy(self):
        return self.__class__(self.getValue(), band=self.band)

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
