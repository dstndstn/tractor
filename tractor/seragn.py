from tractor.utils import *
from tractor.galaxy import *
from tractor.pointsource import PointSource
from tractor.sersic import SersicGalaxy, SersicIndex

class SersicCoreGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Sersic plus central point source components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses.
    '''
    def __init__(self, pos, brightness, shape, sersicindex, brightnessPsf):
        MultiParams.__init__(self, pos, brightness, shape, sersicindex, brightnessPsf)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2, sersicindex=3, brightnessPsf=4)

    def getName(self):
        return 'SersicCoreGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Sersic ' + str(self.brightness) + ', '
                + str(self.shape) + ', ' + str(self.sersicindex)
                + ' and PSF ' + str(self.brightnessPsf))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', shape=' + repr(self.shape) +
                ', sersicindex=' + repr(self.sersicindex) +
                ', brightnessPsf=' + repr(self.brightnessPsf))

    def getBrightness(self):
        # assume linear
        return self.brightness + self.brightnessPsf

    def getBrightnesses(self):
        return [self.brightness, self.brightnessPsf]

    def _getModelPatches(self, img, minsb=0., modelMask=None):
        s = SersicGalaxy(self.pos, self.brightness, self.shape, self.sersicindex)
        p = PointSource(self.pos, self.brightnessPsf)
        if minsb == 0. or minsb is None:
            kw = {}
        else:
            kw = dict(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            s.halfsize = self.halfsize
        ps = s.getModelPatch(img, modelMask=modelMask, **kw)
        pp = p.getModelPatch(img, modelMask=modelMask, **kw)
        return (ps, pp)

    def getModelPatch(self, img, minsb=0., modelMask=None):
        ps, pp = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(ps, pp)

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        s = SersicGalaxy(self.pos, self.brightness, self.shape, self.sersicindex)
        p = PointSource(self.pos, self.brightnessPsf)
        if hasattr(self, 'halfsize'):
            s.halfsize = self.halfsize
        return (s.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask) +
                p.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask))

    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    def getParamDerivatives(self, img, modelMask=None):
        s = SersicGalaxy(self.pos, self.brightness, self.shape, self.sersicindex)
        p = PointSource(self.pos, self.brightnessPsf)
        if hasattr(self, 'halfsize'):
            s.halfsize = self.halfsize
        s.dname = 'sercore.ser'
        p.dname = 'sercore.psf'
        if self.isParamFrozen('pos'):
            s.freezeParam('pos')
            p.freezeParam('pos')
        if self.isParamFrozen('brightness'):
            s.freezeParam('brightness')
        if self.isParamFrozen('shape'):
            s.freezeParam('shape')
        if self.isParamFrozen('sersicindex'):
            s.freezeParam('sersicindex')
        if self.isParamFrozen('brightnessPsf'):
            p.freezeParam('brightness')

        ds = s.getParamDerivatives(img, modelMask=modelMask)
        dp = p.getParamDerivatives(img, modelMask=modelMask)

        if self.isParamFrozen('pos'):
            derivs = ds + dp
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes())
            for i in range(npos):
                dpos = add_patches(ds[i], dp[i])
                if dpos is not None:
                    dpos.setName('d(sercore)/d(pos%i)' % i)
                derivs.append(dpos)
            derivs.extend(ds[npos:])
            derivs.extend(dp[npos:])

        return derivs

    
