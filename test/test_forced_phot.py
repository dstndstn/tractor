import unittest

import numpy as np

from tractor import *
from tractor.patch import ModelMask, Patch
from tractor.pointsource import SingleProfileSource
from tractor.ducks import Source
from tractor.utils import _GaussianPriors, GaussianPriorsMixin
from tractor.dense_optimizer import ConstrainedDenseOptimizer

class ConstantSource(MultiParams, Source):
    def __init__(self, br):
        super().__init__(br)
    def getBrightness(self):
        return self.brightness
    def setBrightness(self, brightness):
        self.brightness = brightness
    def getUnitFluxModelPatches(self, *args, **kwargs):
        return [self.getUnitFluxModelPatch(*args, **kwargs)]
    def getUnitFluxModelPatch(self, img, modelMask=None, **kwargs):
        if modelMask is None:
            return Patch(0,0,np.ones(img.shape, np.float32))
        return Patch(0,0, np.ones(modelMask.shape, np.float32))
    def getModelPatch(self, img, modelMask=None, **kwargs):
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None
        # HACK
        if not np.isfinite(np.float32(counts)):
            return None
        upatch = self.getUnitFluxModelPatch(img, modelMask=modelMask, **kwargs)
        if upatch is None:
            return None
        if upatch.patch is not None:
            assert(np.all(np.isfinite(upatch.patch)))
        p = upatch * counts
        if p.patch is not None:
            assert(np.all(np.isfinite(p.patch)))
        return p
    @staticmethod
    def getNamedParams():
        return dict(brightness=0)

class ConstantSourceWithPrior(GaussianPriorsMixin, ConstantSource):
    priorStd = 0.
    priors = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set class "priors" object
        if self.priors is None:
            p = _GaussianPriors(None)
            p.add('brightness', 0., self.priorStd,
                       param=ConstantSource(0.))
            self.__class__.priors = p
        self.gpriors = self.priors


class ForcedPhotTest(unittest.TestCase):
    def setUp(self):
        pass

    def singlePixelTim(self, val):
        tim = Image(data=np.zeros((1,1),np.float32) + val,
                    inverr=np.ones((1,1),np.float32),
                    photocal=LinearPhotoCal(1.))
        return tim

    def test_pixel(self):
        # A single-pixel test
        val = 42.
        tim = self.singlePixelTim(val)

        src = ConstantSource(Flux(10.))

        p = src.getModelPatch(tim)
        print('Patch:', p)
        print('Patch:', p, p.patch)
        print('Source params:', src.getParams())

        tr = Tractor([tim], [src])
        tr.freezeParam('images')
        tr.optimize_forced_photometry()

        print('Optimized:', src.getParams())
        self.assertAlmostEqual(src.getParams()[0], val, 6)

    def test_pixel_with_prior(self):
        val = 4.
        tim = self.singlePixelTim(val)

        src = ConstantSource(Flux(0.))

        p = src.getModelPatch(tim)
        print('Source params:', src.getParams())

        tr = Tractor([tim], [src])
        tr.freezeParam('images')
        tr.optimize_forced_photometry()

        print('Optimized:', src.getParams())
        self.assertAlmostEqual(src.getParams(), [val], 6)

        fluxSigma = 1.
        ConstantSourceWithPrior.priorStd = fluxSigma
        src = ConstantSourceWithPrior(Flux(0.))
        p = src.getModelPatch(tim)
        print('Source params:', src.getParams())
        print('Source priors:', src.gpriors)
        
        tr = Tractor([tim], [src])
        tr.freezeParam('images')
        print('LogPriorDerivs:', tr.getLogPriorDerivatives())
        tr.optimize_forced_photometry(priors=True)
        print('Optimized:', src.getParams())
        self.assertAlmostEqual(src.getParams()[0], val/2., 6)

        src.setParams([1.])
        tr.optimize_forced_photometry(priors=True)
        print('Optimized:', src.getParams())
        self.assertAlmostEqual(src.getParams()[0], val/2., 6)

        print()
        tim.inverr[:,:] = 2.
        src.setParams([0.])
        tr.optimize_forced_photometry(priors=True)
        print('Optimized:', src.getParams())

        sumiv = 0.
        sumval = 0.
        iv = tim.getInvvar()
        sumiv += np.sum(iv)
        sumval += np.sum(iv * val)
        iv = 1./fluxSigma**2
        sumiv += iv
        sumval += iv * 0.
        pval = sumval / sumiv
        print('Predicted val:', pval)
        
        self.assertAlmostEqual(src.getParams()[0], pval, 6)
        
if __name__ == '__main__':
    # import sys
    # if '--plots' in sys.argv:
    #     sys.argv.remove('--plots')
    #     from astrometry.util.plotutils import PlotSequence
    #     ps = PlotSequence('gal')
    unittest.main()

