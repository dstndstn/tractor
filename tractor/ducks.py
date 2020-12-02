"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`ducks.py`
===========

Duck-type definitions of types used by the Tractor.

Most of this code is not actually used at all.  It's here for
documentation purposes.
"""


class Params(object):
    '''
    A set of parameters that can be optimized by the Tractor.

    This is a duck-type definition.
    '''

    def copy(self):
        return None

    def hashkey(self):
        '''
        Returns a tuple containing the state of this `Params` object
        for use as a cache key.

        All elements must be hashable: see
        http://docs.python.org/glossary.html#term-hashable
        '''
        return ()

    # def __hash__(self):
    #    ''' Params must be hashable. '''
    #    return None
    # def __eq__(self, other):

    def getParamNames(self):
        ''' Returns a list of strings: the names of the parameters. '''
        return []

    def numberOfParams(self):
        ''' Returns the number of parameters (ie, number of scalar
        values).'''
        return len(self.getParams())

    def getParams(self):
        ''' Returns a *copy* of the current parameter values as an
        iterable (eg, list)'''
        return []

    def getAllParams(self):
        return self.getParams()

    def getAllStepSizes(self, *args, **kwargs):
        return self.getStepSizes(*args, **kwargs)

    def getStepSizes(self, *args, **kwargs):
        ''' Returns "reasonable" step sizes for the parameters.'''
        return []

    def setAllStepSizes(self, ss):
        self.setStepSizes(ss)

    def setStepSizes(self, ss):
        assert(len(ss) == self.numberOfParams())
        pass

    def setParams(self, p):
        '''
        Sets the parameter values to the values in the given
        iterable `p`.  The length of `p` will be equal to
        `numberOfParams()`.
        '''
        assert(len(p) == self.numberOfParams())

    def setAllParams(self, p):
        return self.setParams(p)

    def setParam(self, i, p):
        '''
        Sets parameter index 'i' to new value 'p'.

        i: integer in the range [0, numberOfParams()).
        p: float

        Returns the old value.
        '''
        return None

    def getLowerBounds(self):
        return []

    def getUpperBounds(self):
        return []

    def getMaxStep(self):
        '''
        Returns the largest step we should take in this parameter.  Use for nonlinear
        params where making a large change will take us outside the linear optimization
        regime.
        '''
        return None

    def getGaussianPriors(self):
        '''
        Returns a list of
        (index, mu, sigma)
        of Gaussian priors on this set of parameters.
        '''
        return []

    def getLogPrior(self):
        '''
        Returns the prior, evaluated at the current values of
        the parameters.
        '''
        return 0.

    def getLogPriorDerivatives(self):
        '''
        Returns a "chi-like" approximation to the log-prior at the
        current parameter values.

        This will go into the least-squares fitting (each term in the
        prior acts like an extra "pixel" in the fit).

        Returns (rowA, colA, valA, pb, mub), where:

        rowA, colA, valA: describe a sparse matrix pA

        pA: has shape N x numberOfParams
        pb: has shape N
        mub: has shape N

        rowA: list of iterables of ints
        colA: list of ints
        valA: list of iterables of floats
        pb:   list of iterables of floats
        mub:  list of iterables of floats

        where "N" is the number of "pseudo-pixels" or Gaussian terms.
        "pA" will be appended to the least-squares "A" matrix, and
        "pb" will be appended to the least-squares "b" vector, and the
        least-squares problem is minimizing

        || A * (delta-params) - b ||^2

        We also require *mub*, which is like "pb" but not shifted
        relative to the current parameter values; ie, it's the mean of
        the Gaussian.

        This function must take frozen-ness of parameters into account
        (this is implied by the "numberOfParams" shape requirement).
        '''
        return None


class ImageCalibration(object):
    def toFitsHeader(self, hdr, prefix=''):
        params = self.getAllParams()
        names = self.getParamNames()
        for i,(name,val) in enumerate(zip(names, params)):
            k = prefix + 'P%i' % i
            hdr.add_record(dict(name=k, value=val, comment=name))

    def toStandardFitsHeader(self, hdr):
        pass

    @classmethod
    def fromFitsHeader(clazz, hdr, prefix=''):
        args = []
        for i in range(100):
            k = prefix + 'A%i' % i
            if not k in hdr:
                break
            args.append(hdr.get(k))
        obj = clazz(*args)
        params = []
        for i in range(100):
            k = prefix + 'P%i' % i
            if not k in hdr:
                break
            params.append(hdr.get(k))
        obj.setAllParams(params)
        return obj


class Sky(ImageCalibration, Params):
    '''
    Duck-type definition for a sky model.
    '''

    def getParamDerivatives(self, tractor, img, srcs):
        '''
        Returns [ Patch, Patch, ... ], of length numberOfParams(),
        containing the derivatives in the given `Image` for each
        parameter.
        '''
        return []

    def addTo(self, mod, scale=1.):
        '''
        Add the sky to the input synthetic image `mod`, a 2-D numpy
        array.
        '''
        pass

    def getConstant(self):
        '''
        Returns an unspecified constant value, eg the mean, median, etc.
        '''
        return 0.

    def subtract(self, con):
        '''
        Subtracts a constant value from this sky model.
        '''
        raise RuntimeError('Unimplemented: Sky.subtract()')

    def shift(self, x0, y0):
        '''
        Shifts this sky model so that it applies to the subimage starting at x0,y0.
        '''
        pass

    def shifted(self, x0, y0):
        s = self.copy()
        s.shift(x0, y0)
        return s


class Source(Params):
    '''
    This is the duck-type definition of a Source (star, galaxy, etc)
    that the Tractor uses.
    '''

    def getModelPatch(self, img, minsb=0., modelMask=None):
        '''
        Returns a Patch object containing a rendering of this Source
        into the given `Image` object.  This will probably use the
        calibration information of the `Image`: the WCS, PSF, and
        photometric calibration.

        *minsb*: the allowable approximation error per pixel; we are
        asking the source to render itself out to this surface
        brightness.

        *modelMask*: a ModelMask object describing the rectangular
         region of interest (image pixels).
        '''
        pass

    def getParamDerivatives(self, img, modelMask=None):
        '''
        Returns [ Patch, Patch, ... ], of length numberOfParams(),
        containing the derivatives in the given `Image` for each
        parameter.
        '''
        return []

    def getBrightnesses(self):
        return []

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        '''
        Returns a list the same length as getBrightnesses(), each
        containing a Patch whose sum is ~ unity.

        Like getModelPatch(), but ignore the brightness of the object
        and just return a patch whose sum is unity.  Like "minsb",
        "minval" gives the allowable per-pixel value at which the
        profile can be truncated.  The patch may therefore not sum to
        1 exactly.
        '''
        pass


class Brightness(Params):
    '''
    Duck-type definition of the brightness of an astronomical source.

    Only used as an input to `PhotoCal`.  `Source`s have
    `Brightness`es; `PhotoCal`s convert these into counts in a
    specific `Image`.
    '''
    pass


class PhotoCal(ImageCalibration, Params):
    '''
    Duck-type definition of photometric calibration.

    A `PhotoCal` belongs to an `Image`; it converts `Brightness`
    values into counts ("data numbers", ADU, etc) in the data space
    (synthetic image) of the `Image`.  It also contains the parameters
    of that conversion so they can be optimized along with everything
    else.

    This relationship need not be linear: the `Brightness` could be an
    astronomical magnitude, for example.  In general, there is a lot
    of freedom in the definition of the `Brightness` object, and
    `PhotoCal` has to be kept consistent with that.
    '''

    def brightnessToCounts(self, brightness):
        '''Converts `brightness`, a `Brightness` duck, into counts.

        Returns: float
        '''
        pass


class Position(Params):
    '''
    Duck-type definition of the position of an astronomical object.

    Only used as an input to a `WCS` object; `Sources` have
    `Positions`, and `WCS` objects convert them into pixel coordinates
    in a specific `Image`.
    '''
    pass


class Time(Params):
    '''
    Objects of type "Time" should define arithmetic operators (at least
    __sub__, __add__, __isub__, __iadd__)
    '''
    # def __sub__(self, other):
    #   pass

    def getSunTheta(self):
        ''' Returns the angle of the Earth's (mean?) anomaly at this time;
        ie, the time of year expressed as an angle in radians.'''
        pass

    def toYears(self):
        pass


class WCS(ImageCalibration, Params):
    '''
    Duck-type definition of World Coordinate System.

    Converts between Position objects and Image pixel coordinates.

    In general, there is a lot of freedom in the definition of the
    `Position` object, and `WCS` has to be kept consistent with that.
    For instance, if the `Positions` used are image-based x-y
    positions (`PixPos`), then `WCS` has to be null (or close to
    that); `NullWCS`.
    '''

    def positionToPixel(self, pos, src=None):
        '''
        Converts a :class:`tractor.Position` *pos* into ``x,y`` pixel
        coordinates.

        Note that the :class:`tractor.Source` may be passed in; your
        :class:`tractor.WCS` could have color-specific behavior, for
        example.

        Returns tuple `(x, y)`, where `x` and `y` are floats, and `0,
        0` is the first pixel.

        Pixels are funny things.  Our convention is shifted by 1 from
        the FITS convention, so 0,0 is the *center* of the first
        ("zeroth", says Hogg) pixel, if you think of pixels as little
        boxes.  (What is the emoticon for "point and laugh"?)
        '''
        return None

    def cdAtPixel(self, x, y):
        '''
        Returns a local affine relationship between `Position` and
        (x,y) pixel coordinates.  This is used, for example, to
        convert tensor shapes of galaxies from `Position` space to
        image space.

        Returns a numpy array of shape (2,2).

        In FITS celestial coordinates language, this is the CD matrix
        at pixel x,y:

        [ [ dRA/dx * cos(Dec), dRA/dy * cos(Dec) ],
          [ dDec/dx          , dDec/dy           ] ]

        in FITS these are called:

        [ [ CD11             , CD12              ],
          [ CD21             , CD22              ] ]

        The units of these things are degrees per pixel.
        '''
        return None

    def cdInverseAtPixel(self, x, y):
        import numpy as np
        cd = self.cdAtPixel(x, y)
        cdi = np.linalg.inv(cd)
        return cdi

    def cdInverseAtPosition(self, pos, src=None):
        px, py = self.positionToPixel(pos, src=src)
        return self.cdInverseAtPixel(px, py)

    def pixelDerivsToPositionDerivs(self, pos, src, counts0, patch0, patchdx, patchdy):
        # Convert x,y derivatives to Position derivatives
        cdi = self.cdInverseAtPosition(pos, src=src)
        # Get thawed Position parameter indices
        derivs = []
        for i,pname in pos.getThawedParamIndicesAndNames():
            deriv = (patchdx * cdi[0, i] +
                     patchdy * cdi[1, i]) * counts0
            deriv.setName('d(ptsrc)/d(pos.%s)' % pname)
            derivs.append(deriv)
        return derivs

    def pixscale_at(self, x, y):
        '''
        Returns the local pixel scale at the given *x*,*y* pixel coords,
        in *arcseconds* per pixel.
        '''
        import numpy as np
        return 3600. * np.sqrt(np.abs(np.linalg.det(self.cdAtPixel(x, y))))

    def shifted(self, dx, dy):
        '''
        Returns a new WCS object appropriate for the subimage starting
        at (dx,dy) with respect to the current WCS origin.
        '''
        return None


class PSF(ImageCalibration, Params):
    '''
    Duck-type definition of a point-spread function.
    '''

    def getPointSourcePatch(self, px, py, minval=0., modelMask=None):
        '''
        Returns a `Patch`, a rendering of a point source at the given
        pixel coordinates.

        The returned `Patch` should have unit "counts".

        *minval* says that we are willing to accept an approximation
        such that pixels with counts < minval can be omitted.

        *modelMask* describes the pixels to be evaluated.  If the
         *modelMask* includes a pixel-by-pixel mask, this overrides
         *minval*.
        '''
        pass

    def getRadius(self):
        '''
        Returns the size of the support of this PSF.

        This is required because the Tractor has to decide what size
        to make the ``Patch``es.
        '''
        return 0

    def getShifted(self, x0, y0):
        '''
        Returns a PSF model for the subimage starting at x0,y0.
        '''
        return None

    # Optional: Allows galaxy models to render via analytic convolution:
    # def getMixtureOfGaussians(self, px=None, py=None, **kwargs):
    #     '''
    #     Returns a mixture_profiles.MixtureOfGaussians object approximating this
    #     PSF at the given px,py position.  The mean of the MoG is NOT set to px,py;
    #     it is 0,0.
    #     '''
