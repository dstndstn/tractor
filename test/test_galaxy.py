from __future__ import print_function

import unittest
import os

import numpy as np

from tractor import *
from tractor.galaxy import *

ps = None

class GalaxyTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_gal(self):

        if ps is not None:
            import pylab as plt

        #pos = RaDecPos(0., 1.)
        pos = PixPos(49.5, 50.)
        bright = NanoMaggies(g=1., r=2.)
        shape = GalaxyShape(2., 0.5, 45.)

        H,W = 100,100
        tim = Image(data=np.zeros((H,W), np.float32),
                    inverr=np.ones((H,W), np.float32),
                    psf=GaussianMixturePSF(1., 0., 0., 4., 4., 0.),
                    photocal=LinearPhotoCal(1., band='r'),
            )
        
        # base class
        #gal1 = Galaxy(pos, bright, shape)

        gal1 = ExpGalaxy(pos, bright, shape)

        self.assertEqual(gal1.shape.ab, 0.5)
        print('gal:', gal1)
        print('gal:', str(gal1))
        # harsh
        self.assertEqual(str(gal1), 'ExpGalaxy at pixel (49.50, 50.00) with NanoMaggies: g=22.5, r=21.7 and Galaxy Shape: re=2.00, ab=0.50, phi=45.0')
        self.assertEqual(repr(gal1), 'ExpGalaxy(pos=PixPos[49.5, 50.0], brightness=NanoMaggies: g=22.5, r=21.7, shape=re=2, ab=0.5, phi=45)')
        
        derivs = gal1.getParamDerivatives(tim)
        print('Derivs:', derivs)

        self.assertEqual(len(derivs), 7)
        self.assertEqual(len(derivs), gal1.numberOfParams())
        self.assertEqual(len(derivs), len(gal1.getParams()))
        for d in derivs:
            self.assertIsNotNone(d)

        # Set one of the fluxes to zero.
        gal1.brightness.r = 0.

        derivs = gal1.getParamDerivatives(tim)
        print('Derivs:', derivs)
        self.assertEqual(len(derivs), 7)
        for i,d in enumerate(derivs):
            # flux derivatives still non-None
            if i in [2,3]:
                self.assertIsNotNone(derivs[i])
            else:
                # other derivatives should be None
                self.assertIsNone(derivs[i])

        gal1.brightness.r = 100.

        mod = np.zeros((H,W), np.float32)
        
        p1 = gal1.getModelPatch(tim)
        print('Model patch:', p1)
        print('patch sum:', p1.patch.sum())

        # specific...
        self.assertEqual(p1.x0, 16)
        self.assertEqual(p1.y0, 17)
        self.assertEqual(p1.shape, (68,69))
        self.assertTrue(np.abs(p1.patch.sum() - 100.) < 1e-3)

        p1.addTo(mod)
        print('Mod sum:', mod.sum())

        self.assertTrue(np.abs(mod.sum() - 100.) < 1e-3)

        if ps is not None:
            plt.clf()
            plt.imshow(mod, interpolation='nearest', origin='lower')
            plt.colorbar()
            ps.savefig()

        mm = Patch(25, 25, np.ones((50,50), bool))

        # up2 = gal1.getUnitFluxModelPatch(tim, modelMask=mm)
        # print('Unit-flux patch:', up2)
        # 
        # px,py = tim.getWcs().positionToPixel(gal1.getPosition(), gal1)
        # rup2 = gal1._realGetUnitFluxModelPatch(tim, px, py, None,
        #                                        modelMask=mm)
        # print('Real unit-flux patch:', rup2)
        # 
        # deps = gal1._getUnitFluxDeps(tim, px, py)
        # print('Unit-flux deps:', deps)
        
        p2 = gal1.getModelPatch(tim, modelMask=mm)
        print('Patch:', p2)
        self.assertEqual(p2.x0, 25)
        self.assertEqual(p2.y0, 25)
        self.assertEqual(p2.shape, (50,50))

        print('patch sum:', p2.patch.sum())

        mod2 = np.zeros((H,W), np.float32)
        p2.addTo(mod2)
        print('Mod sum:', mod2.sum())
        self.assertTrue(np.abs(mod2.sum() - 100.) < 1e-3)

        if ps is not None:
            plt.clf()
            plt.imshow(mod2, interpolation='nearest', origin='lower')
            plt.colorbar()
            ps.savefig()

            plt.clf()
            plt.imshow(mod2-mod, interpolation='nearest', origin='lower')
            plt.colorbar()
            ps.savefig()
        
        print('Diff between mods:', np.abs(mod - mod2).max())
        self.assertTrue(np.abs(mod - mod2).max() < 1e-6)


        p3 = gal1.getModelPatch(tim, extent=[30,70,29,69])
        print('Patch:', p3)
        self.assertEqual(p3.x0, 30)
        self.assertEqual(p3.y0, 29)
        self.assertEqual(p3.shape, (40,40))
        print('patch sum:', p3.patch.sum())
        mod3 = np.zeros((H,W), np.float32)
        p3.addTo(mod3)
        print('Mod sum:', mod3.sum())
        self.assertTrue(np.abs(mod3.sum() - 100.) < 1e-3)
        print('Diff between mods:', np.abs(mod3 - mod).max())
        self.assertTrue(np.abs(mod3 - mod).max() < 1e-6)

        

if __name__ == '__main__':
    import sys
    if '--plots' in sys.argv:
        sys.argv.remove('--plots')
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('gal')

    unittest.main()
