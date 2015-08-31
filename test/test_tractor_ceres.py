import matplotlib
matplotlib.use('Agg')
import pylab as plt

import unittest

from tractor import *
from tractor.sdss import *
from tractor.galaxy import *

class TractorCeresTest(unittest.TestCase):
    def test_ceres(self):
        W,H = 100,100
        tim1 = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                     psf=NCircularGaussianPSF([2.], [1.]),
                     photocal=LinearPhotoCal(1.))
        tim2 = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                     psf=NCircularGaussianPSF([3.], [1.]),
                     photocal=LinearPhotoCal(1.))

        star = PointSource(PixPos(W/2, H/2), Flux(100.))

        from tractor.ceres_mixin import TractorCeresMixin
        class CeresTractor(TractorCeresMixin, TractorBase):
            pass
        
        tr = CeresTractor([tim1,tim2], [star])

        mods = tr.getModelImages()
        print 'mods:', mods
        print list(mods)
        chis = tr.getChiImages()
        print 'chis:', chis
        print list(chis)
        lnp = tr.getLogProb()
        print 'lnp', lnp

        tr.freezeParam('images')
        #dlnp,x,a = tr.optimize()
        #print 'dlnp', dlnp
        #print 'x', x
        #print 'a', a

        star.brightness.setParams([100.])
        star.freezeAllBut('brightness')
        print 'star', star
        tr.optimize_forced_photometry()
        print 'star', star



if __name__ == '__main__':
    unittest.main()
