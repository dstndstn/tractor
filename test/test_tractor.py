import matplotlib
matplotlib.use('Agg')
import pylab as plt

import unittest

from tractor import *
from tractor.sdss import *
from tractor.galaxy import *
from tractor.ceres_optimizer import CeresOptimizer

class TractorTest(unittest.TestCase):
    def test_expgal(self):
        ra,dec = 123., 45.
        pos = RaDecPos(ra, dec)
        flux = 6.
        sflux = SdssFlux(flux)
        re,ab,phi = 7, 0.8, 9
        shape = GalaxyShape(re, ab, phi)
        gal = ExpGalaxy(pos, sflux, shape)
        # harsh
        self.assertEqual(str(gal),
                         'ExpGalaxy at RaDecPos: RA, Dec = (123.00000, 45.00000) with SdssFlux: 6000000.0 and Galaxy Shape: re=7.00, ab=0.80, phi=9.0')
        self.assertEqual(str(gal.shape), str(shape))
        self.assertEqual(shape.re, re)
        self.assertEqual(shape.ab, ab)
        self.assertEqual(shape.phi, phi)
        self.assertEqual(gal.shape.re, re)
        self.assertEqual(gal.shape.ab, ab)
        self.assertEqual(gal.shape.phi, phi)
        self.assertEqual(gal.getParams(), [ra, dec, flux, re, ab, phi])
        self.assertEqual(shape.getParams(), [re, ab, phi])

        re2 = 7.7
        gal.shape.re = re2
        print gal
        self.assertEqual(shape.re, re2)
        self.assertEqual(gal.shape.re, re2)
        print gal.subs
        print shape.vals
        self.assertEqual(gal.getParams(), [ra, dec, flux, re2, ab, phi])
        self.assertEqual(shape.getParams(), [re2, ab, phi])

        re3 = 7.77
        gal.shape = GalaxyShape(re3, ab, phi)
        self.assertEqual(gal.shape.re, re3)

        # However:
        self.assertNotEqual(gal.shape, shape)
        self.assertNotEqual(shape.re, re3)
        self.assertEqual(shape.re, re2)

    def test_model_images(self):
        W,H = 100,100
        tim1 = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                     psf=NCircularGaussianPSF([2.], [1.]),
                     photocal=LinearPhotoCal(1.))
        tim2 = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                     psf=NCircularGaussianPSF([3.], [1.]),
                     photocal=LinearPhotoCal(1.))

        star = PointSource(PixPos(W/2, H/2), Flux(100.))
        

        for opt in [None]: #, CeresOptimizer()]:
            tr = Tractor([tim1,tim2], [star], optimizer=opt)
            mods = tr.getModelImages()
            print 'mods:', mods
            print list(mods)
            chis = tr.getChiImages()
            print 'chis:', chis
            print list(chis)
            lnp = tr.getLogProb()
            print 'lnp', lnp
    
            tr.freezeParam('images')
            X = tr.optimize()
            # dlnp,x,a
            # print 'dlnp', dlnp
            # print 'x', x
            # print 'a', a
            print 'opt result:', X
            
            star.brightness.setParams([100.])
            star.freezeAllBut('brightness')
            print 'star', star
            tr.optimize_forced_photometry()
            print 'star', star

        
if __name__ == '__main__':
    unittest.main()
