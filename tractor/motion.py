from .utils import *
from .basics import *


class Parallax(ArithmeticParams, ScalarParam):
    ''' in arcesc '''
    stepsize = 1e-3

    def __str__(self):
        return 'Parallax: %.3f arcsec' % (self.getValue())


class ParallaxWithPrior(Parallax):
    def getLogPrior(self):
        p = self.getValue()
        if p < 0:
            return -np.inf
        # Lutz & Kelker (1973) PASP 85 573
        # in the introduction, yo!
        return -4. * np.log(p)

    def isLegal(self):
        p = self.getValue()
        return (p >= 0)

    #### FIXME -- cos(Dec)


class PMRaDec(RaDecPos):
    @staticmethod
    def getName():
        return "PMRaDec"

    def __str__(self):
        return '%s: (%.2f, %.2f) mas/yr' % (self.getName(),
                                            1000. * self. getRaArcsecPerYear(),
                                            1000. * self.getDecArcsecPerYear())

    def __init__(self, *args, **kwargs):
        self.addParamAliases(ra=0, dec=1)
        super(PMRaDec, self).__init__(*args, **kwargs)
        self.setStepSizes([1e-6] * 2)

    @staticmethod
    def getNamedParams():
        return dict(pmra=0, pmdec=1)

    def getRaArcsecPerYear(self):
        return self.pmra * 3600.

    def getDecArcsecPerYear(self):
        return self.pmdec * 3600.

    # def getParamDerivatives(self, img, modelMask=None):
    #    return [None]*self.numberOfParams()


class MovingPointSource(PointSource):
    def __init__(self, pos, brightness, pm, parallax, epoch=0.):
        # Assume types...
        assert(type(pos) is RaDecPos)
        assert(type(pm) is PMRaDec)
        super(PointSource, self).__init__(pos, brightness, pm,
                                          Parallax(parallax))
        self.epoch = epoch

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, pm=2, parallax=3)

    def getSourceType(self):
        return 'MovingPointSource'

    def __str__(self):
        return (self.getSourceType() + ' at ' + str(self.pos) +
                ' with ' + str(self.brightness) + ', pm ' + str(self.pm) +
                ', parallax ' + str(self.parallax))

    def __repr__(self):
        return (self.getSourceType() + '(' + repr(self.pos) + ', ' +
                repr(self.brightness) + ', ' + repr(self.pm) + ', ' +
                repr(self.parallax) + ')')

    def getPositionAtTime(self, t):
        from astrometry.util.starutil_numpy import radectoxyz, arcsecperrad, axistilt, xyztoradec

        dt = (t - self.epoch).toYears()
        # Assume "pos" is an RaDecPos
        p = self.pos + dt * self.pm
        suntheta = t.getSunTheta()

        # print 'dt', dt, 'pos', self.pos, 'pm', self.pm, 'dt*pm:', dt * self.pm
        # print 'p0: (%.8f, %.8f)' % (self.pos.ra, self.pos.dec)
        # print 'p1: (%.8f, %.8f)' % (p.ra, p.dec)

        xyz = radectoxyz(p.ra, p.dec)
        xyz = xyz[0]
        # d(celestial coords)/d(parallax)
        # - takes numerical derivatives when it could take analytic ones
        # output is in [degrees / arcsec].  Yep.    Crazy but true.
        # HACK: fmods dRA when it should do something continuous.
        # rd2xyz(0,0) is a unit vector; 1/arcsecperrad is (a good approximation to)
        # the distance on the unit sphere spanned by an angle of 1 arcsec.
        # We take a step of that length and return the change in RA,Dec.
        # It's about 1e-5 so we don't renormalize the xyz unit vector.
        dxyz1 = radectoxyz(0., 0.) / arcsecperrad
        dxyz1 = dxyz1[0]
        # - imprecise angle of obliquity
        # - implicitly assumes circular orbit
        # output is in [degrees / arcsec].  Yep.    Crazy but true.
        dxyz2 = radectoxyz(90., axistilt) / arcsecperrad
        dxyz2 = dxyz2[0]
        xyz += self.parallax.getValue() * (dxyz1 * np.cos(suntheta) +
                                           dxyz2 * np.sin(suntheta))
        r, d = xyztoradec(xyz)
        return RaDecPos(r, d)

    def getUnitFluxModelPatch(self, img, minval=0., modelMask=None):
        pos = self.getPositionAtTime(img.getTime())
        (px, py) = img.getWcs().positionToPixel(pos)
        patch = img.getPsf().getPointSourcePatch(
            px, py, minval=minval, modelMask=modelMask)
        return patch

    def getParamDerivatives(self, img, modelMask=None):
        '''
        MovingPointSource derivatives.

        returns [ Patch, Patch, ... ] of length numberOfParams().
        '''

        t = img.getTime()
        pos0 = self.getPositionAtTime(t)
        (px0, py0) = img.getWcs().positionToPixel(pos0, self)
        patch0 = img.getPsf().getPointSourcePatch(px0, py0, modelMask=modelMask)
        counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
        derivs = []

        # print 'MovingPointSource.getParamDerivs:'
        # print 'initial pixel pos', px0, py0

        # Position

        # FIXME -- could just compute positional derivatives once and
        # reuse them, but have to be careful about frozen-ness -- eg,
        # if RA were frozen but not Dec.
        # OR, could compute dx,dy and then use CD matrix to convert
        # dpos to derivatives.
        # pderivs = []
        # if ((not self.isParamFrozen('pos')) or
        #   (not self.isParamFrozen('pm')) or
        #   (not self.isParamFrozen('parallax'))):
        #
        #   psteps = pos0.getStepSizes(img)
        #   pvals = pos0.getParams()
        #   for i,pstep in enumerate(psteps):
        #       oldval = pos0.setParam(i, pvals[i] + pstep)
        #       (px,py) = img.getWcs().positionToPixel(pos0, self)
        #       patchx = img.getPsf().getPointSourcePatch(px, py)
        #       pos0.setParam(i, oldval)
        #       dx = (patchx - patch0) * (counts0 / pstep)
        #       dx.setName('d(ptsrc)/d(pos%i)' % i)
        #       pderivs.append(dx)
        # if not self.isParamFrozen('pos'):
        #   derivs.extend(pderivs)

        def _add_posderivs(p, name):
            # uses "globals": t, patch0, counts0
            psteps = p.getStepSizes(img)
            pvals = p.getParams()
            for i, pstep in enumerate(psteps):
                oldval = p.setParam(i, pvals[i] + pstep)
                tpos = self.getPositionAtTime(t)
                (px, py) = img.getWcs().positionToPixel(tpos, self)

                # print 'stepping param', name, i, '-->', p, '--> pos', tpos, 'pix pos', px,py
                patchx = img.getPsf().getPointSourcePatch(px, py, modelMask=modelMask)
                p.setParam(i, oldval)
                dx = (patchx - patch0) * (counts0 / pstep)
                dx.setName('d(ptsrc)/d(%s%i)' % (name, i))
                # print 'deriv', dx.patch.min(), dx.patch.max()
                derivs.append(dx)

        # print 'Finding RA,Dec derivatives'
        if not self.isParamFrozen('pos'):
            _add_posderivs(self.pos, 'pos')

        # Brightness
        # print 'Finding Brightness derivatives'
        if not self.isParamFrozen('brightness'):
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            for i, bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts0) / bstep)
                df.setName('d(ptsrc)/d(bright%i)' % i)
                derivs.append(df)

        # print 'Finding Proper Motion derivatives'
        if not self.isParamFrozen('pm'):
            #   # ASSUME 'pm' is the same type as 'pos'
            #   dt = (t - self.epoch).toYears()
            #   for d in pderivs:
            #       dd = d * dt
            #       derivs.append(dd)
            _add_posderivs(self.pm, 'pm')

        # print 'Finding Parallax derivatives'
        if not self.isParamFrozen('parallax'):
            _add_posderivs(self.parallax, 'parallax')

        return derivs
