from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import unittest

from tractor import *
#from tractor.sdss import *
from tractor.galaxy import *

class TractorTest(unittest.TestCase):
    def test_expgal(self):
        ra,dec = 123., 45.
        pos = RaDecPos(ra, dec)
        flux = 6.
        sflux = NanoMaggies(r=flux)
        re,ab,phi = 7, 0.8, 9
        shape = GalaxyShape(re, ab, phi)
        gal = ExpGalaxy(pos, sflux, shape)
        # harsh
        self.assertEqual(str(gal),
                         'ExpGalaxy at RaDecPos: RA, Dec = (123.00000, 45.00000) with NanoMaggies: r=20.6 and Galaxy Shape: re=7.00, ab=0.80, phi=9.0')
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
        print(gal)
        self.assertEqual(shape.re, re2)
        self.assertEqual(gal.shape.re, re2)
        print(gal.subs)
        print(shape.vals)
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
        psf_sig1 = 2.
        psf_sig2 = 3.
        tim1 = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                     psf=NCircularGaussianPSF([psf_sig1], [1.]),
                     photocal=LinearPhotoCal(1.))
        tim2 = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                     psf=NCircularGaussianPSF([psf_sig2], [1.]),
                     photocal=LinearPhotoCal(1.))

        trueflux = 100.
        star = PointSource(PixPos(W/2, H/2), Flux(trueflux))

        tr = Tractor([tim1,tim2], [star])

        # Create model images with known source
        mods = list(tr.getModelImages())
        print('Mod sums:', [np.sum(m) for m in mods])

        # Sum of fluxes ~= trueflux
        self.assertLess(np.abs(np.sum(mods[0]) - trueflux), 1e-3)
        self.assertLess(np.abs(np.sum(mods[1]) - trueflux), 1e-3)

        # Max: flux * peak of gaussian = 1./(2.*pi*sigma**2)
        self.assertTrue(np.abs(np.max(mods[0]) -
                               trueflux / (2.*np.pi*psf_sig1**2)) < 1e-3)
        self.assertTrue(np.abs(np.max(mods[1]) -
                               trueflux / (2.*np.pi*psf_sig2**2)) < 1e-3)

        chis = list(tr.getChiImages())
        #print('chis:', chis)
        lnp = tr.getLogProb()
        #print('lnp', lnp)

        self.assertTrue(np.abs(lnp - -143.681) < 1e-3)
        
        tr.freezeParam('images')
        X = tr.optimize()
        print('opt result:', X)
            
        # Image data is still zero -- forced phot should set flux to ~ zero.
        star.brightness.setParams([100.])
        star.freezeAllBut('brightness')
        print('star', star)
        tr.optimize_forced_photometry()
        print('star', star)

        self.assertTrue(np.abs(star.getBrightness().getValue()) < 1e-3)

        # Now set data to model images
        tim1.data = mods[0]
        tim2.data = mods[1]
        # And re-forced phot.
        tr.optimize_forced_photometry()
        print('star', star)

        self.assertTrue(np.abs(star.getBrightness().getValue() - trueflux)
                        < 1e-3)

        star.getBrightness().setValue(0.)
        
        # Now set data to noisy model images.
        np.random.seed(42)
        tim1.data = mods[0] + np.random.normal(size=tim1.shape)
        tim2.data = mods[1] + np.random.normal(size=tim2.shape)

        # And re-forced phot.
        tr.optimize_forced_photometry()
        print('star', star)

        self.assertTrue(np.abs(star.getBrightness().getValue() - trueflux)
                        < 5.)

        
if __name__ == '__main__':
    unittest.main()
