from tractor.utils import *
from tractor.galaxy import *
from tractor.pointsource import PointSource

class DevCoreGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with deVauc and central point source components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses.
    '''
    def __init__(self, pos, brightnessDev, shapeDev, brightnessPsf):
        MultiParams.__init__(self, pos, brightnessDev, shapeDev, brightnessPsf)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessDev=1, shapeDev=2, brightnessPsf=3)

    def getName(self):
        return 'DevCoreGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with deV ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev) + ' and PSF ' + str(self.brightnessPsf))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev) +
                ', brightnessPsf=' + repr(self.brightnessPsf))

    def getBrightness(self):
        # assume linear
        return self.brightnessDev + self.brightnessPsf

    def getBrightnesses(self):
        return [self.brightnessDev, self.brightnessPsf]

    def _getModelPatches(self, img, minsb=0., modelMask=None):
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        p = PointSource(self.pos, self.brightnessPsf)
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            d.halfsize = self.halfsize
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        pp = p.getModelPatch(img, modelMask=modelMask, **kw)
        return (pd, pp)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        pd, pp = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pd, pp)

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        p = PointSource(self.pos, self.brightnessPsf)
        if hasattr(self, 'halfsize'):
            d.halfsize = self.halfsize
        return (d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask) +
                p.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    def getParamDerivatives(self, img, modelMask=None):
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        p = PointSource(self.pos, self.brightnessPsf)
        if hasattr(self, 'halfsize'):
            d.halfsize = self.halfsize
        d.dname = 'devcore.dev'
        p.dname = 'devcore.psf'
        if self.isParamFrozen('pos'):
            d.freezeParam('pos')
            p.freezeParam('pos')
        if self.isParamFrozen('brightnessDev'):
            d.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')
        if self.isParamFrozen('brightnessPsf'):
            p.freezeParam('brightness')

        dd = d.getParamDerivatives(img, modelMask=modelMask)
        dp = p.getParamDerivatives(img, modelMask=modelMask)

        if self.isParamFrozen('pos'):
            derivs = dd + dp
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes())
            for i in range(npos):
                dpos = add_patches(dd[i], dp[i])
                if dpos is not None:
                    dpos.setName('d(devcore)/d(pos%i)' % i)
                derivs.append(dpos)
            derivs.extend(dd[npos:])
            derivs.extend(dp[npos:])

        return derivs

    
