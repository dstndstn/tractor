from __future__ import print_function

import unittest
import os

import numpy as np

from tractor import *
from tractor.galaxy import *
from tractor.patch import ModelMask
#from tractor.sersic import SersicGalaxy, SersicIndex

#from astrometry.util.plotutils import PlotSequence
#ps = PlotSequence('gal')
ps = None

class GalaxyTest(unittest.TestCase):
    def setUp(self):
        pass
        
    def test_gal(self):

        if ps is not None:
            import pylab as plt

        pos = PixPos(49.5, 50.)
        pos0 = pos
        bright = NanoMaggies(g=1., r=2.)
        shape = GalaxyShape(2., 0.5, 45.)

        #psf = GaussianMixturePSF(1., 0., 0., 4., 4., 0.)
        psf = GaussianMixturePSF(1., 0., 0., 6., 6., -1.)
        #psf = GaussianMixturePSF(1., 0., 0., 9., 9., -1.)
        #psf = GaussianMixturePSF(1., 0., 0., 16., 16., -1.)

        H,W = 100,100
        tim = Image(data=np.zeros((H,W), np.float32),
                    inverr=np.ones((H,W), np.float32),
                    psf=psf,
                    photocal=LinearPhotoCal(1., band='r'),
            )

        psf0 = tim.psf
        
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
        for i,d in enumerate(derivs):
            if i == 2:
                self.assertIsNone(d)
            else:
                self.assertIsNotNone(d)

        # Set one of the fluxes to zero.
        gal1.brightness.r = 0.

        derivs = gal1.getParamDerivatives(tim)
        print('Derivs:', derivs)
        self.assertEqual(len(derivs), 7)
        for i,d in enumerate(derivs):
            # flux derivative for this image's band should still be non-None
            if i == 3:
                self.assertIsNotNone(derivs[i])
            else:
                # other derivatives should be None
                self.assertIsNone(derivs[i])

        gal1.brightness.r = 100.

        mod = np.zeros((H,W), np.float32)
        
        p1 = gal1.getModelPatch(tim)
        print('Model patch:', p1)
        print('patch sum:', p1.patch.sum())

        # Very specific tests...
        self.assertEqual(p1.x0, 16)
        self.assertEqual(p1.y0, 17)
        self.assertEqual(p1.shape, (68,69))
        self.assertTrue(np.abs(p1.patch.sum() - 100.) < 1e-3)

        p1.addTo(mod)
        print('Mod sum:', mod.sum())

        mh,mw = mod.shape
        xx,yy = np.meshgrid(np.arange(mw), np.arange(mh))
        cx,cy = np.sum(xx*mod)/np.sum(mod), np.sum(yy*mod)/np.sum(mod)
        mxx = np.sum((xx-cx)**2*mod)/np.sum(mod)
        myy = np.sum((yy-cy)**2*mod)/np.sum(mod)
        mxy = np.sum((xx-cx)*(yy-cy)*mod)/np.sum(mod)
        print('mod centroid:', cx,cy)
        print('moments:', mxx,myy,mxy)
        
        self.assertTrue(np.abs(mod.sum() - 100.) < 1e-3)

        if ps is not None:
            plt.clf()
            plt.imshow(mod, interpolation='nearest', origin='lower')
            plt.title('mod')
            plt.colorbar()
            ps.savefig()

        def show_model(modN, mod, name):
            plt.clf()
            sy,sx = slice(40,60),slice(40,60)
            plt.subplot(2,3,1)
            plt.imshow(modN[sy,sx], interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title(name)

            if mod is not None:
                plt.subplot(2,3,2)
                plt.imshow(mod[sy,sx], interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.title('mod')

                plt.subplot(2,3,3)
                mx = np.abs(modN-mod).max()
                plt.imshow((modN-mod)[sy,sx],
                           interpolation='nearest', origin='lower', vmin=-mx,vmax=mx)
                plt.colorbar()
                plt.title('%s - mod' % name)
            
                plt.subplot(2,3,6)
                plt.imshow(modN-mod, interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.title('%s - mod' % name)

            plt.subplot(2,3,4)
            mx = modN.max()
            plt.imshow(np.log10(modN),
                       vmin=np.log10(mx)-6,
                       vmax=np.log10(mx),
                       interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('log %s' % name)

            if mod is not None:
                plt.subplot(2,3,5)
                plt.imshow(np.log10(mod),
                           vmin=np.log10(mx)-6,
                           vmax=np.log10(mx),
                           interpolation='nearest', origin='lower')
                plt.colorbar()
                plt.title('log mod')
            ps.savefig()

            
        # Test with ModelMask
        mm = ModelMask(25, 25, 50, 50)
        p2 = gal1.getModelPatch(tim, modelMask=mm)
        mod2 = np.zeros((H,W), np.float32)
        p2.addTo(mod2)
        print('Patch:', p2)
        self.assertEqual(p2.x0, 25)
        self.assertEqual(p2.y0, 25)
        self.assertEqual(p2.shape, (50,50))
        print('patch sum:', p2.patch.sum())
        print('Mod sum:', mod2.sum())
        self.assertTrue(np.abs(mod2.sum() - 100.) < 1e-3)

        if ps is not None:
            show_model(mod2, mod, 'mod2')
        
        print('Diff between mods:', np.abs(mod - mod2).max())
        self.assertTrue(np.abs(mod - mod2).max() < 1e-6)

        # Test with a ModelMask with a binary map of pixels of interest
        mm3 = ModelMask(30, 29, np.ones((40,40), bool))
        p3 = gal1.getModelPatch(tim, modelMask=mm3)
        mod3 = np.zeros((H,W), np.float32)
        p3.addTo(mod3)
        print('Patch:', p3)
        self.assertEqual(p3.x0, 30)
        self.assertEqual(p3.y0, 29)
        self.assertEqual(p3.shape, (40,40))
        print('patch sum:', p3.patch.sum())
        print('Mod sum:', mod3.sum())
        self.assertTrue(np.abs(mod3.sum() - 100.) < 1e-3)
        print('Diff between mods:', np.abs(mod3 - mod).max())
        self.assertTrue(np.abs(mod3 - mod).max() < 1e-6)

        if ps is not None:
            show_model(mod3, mod, 'mod3')
        
        # Test with a PixelizedPSF (FFT method), created from the Gaussian PSF
        # image (so we can check consistency)
        psfpatch = tim.psf.getPointSourcePatch(
            24., 24., modelMask=ModelMask(0, 0, 50, 50))
        print('PSF patch:', psfpatch)
        tim.psf = PixelizedPSF(psfpatch.patch[:49,:49])
        pixpsf = tim.psf
        
        # No modelmask
        print()
        print('Rendering mod4')
        p4 = gal1.getModelPatch(tim)
        mod4 = np.zeros((H,W), np.float32)
        p4.addTo(mod4)
        print('Patch:', p4)

        cx,cy = np.sum(xx*mod4)/np.sum(mod4), np.sum(yy*mod4)/np.sum(mod4)
        mxx = np.sum((xx-cx)**2*mod4)/np.sum(mod4)
        myy = np.sum((yy-cy)**2*mod4)/np.sum(mod4)
        mxy = np.sum((xx-cx)*(yy-cy)*mod4)/np.sum(mod4)
        print('mod centroid:', cx,cy)
        print('moments:', mxx,myy,mxy)

        if ps is not None:
            show_model(mod4, mod, 'mod4')

        # These assertions are fairly arbitrary...
        self.assertEqual(p4.x0, 6)
        self.assertEqual(p4.y0, 7)
        self.assertEqual(p4.shape, (88,89))
        print('patch sum:', p4.patch.sum())
        print('Mod sum:', mod4.sum())
        self.assertTrue(np.abs(mod4.sum() - 100.) < 1e-3)
        print('Diff between mods:', np.abs(mod4 - mod).max())
        #self.assertTrue(np.abs(mod4 - mod).max() < 1e-6)
        self.assertTrue(np.abs(mod4 - mod).max() < 2e-3)

        import tractor.galaxy
        
        if ps is not None:
            #pp = [49.0, 49.1, 49.2, 49.3, 49.4, 49.5, 49.6, 49.7, 49.8, 49.9, 50.]
            #pp = np.arange(48, 51, 0.1)

            plt.clf()
            
            for L in [3,5,7]:
                tractor.galaxy.fft_lanczos_order = L
                CX = []
                pp = np.arange(49, 50.1, 0.1)
                gal1copy = gal1.copy()
                for p in pp:
                    gal1copy.pos = PixPos(p, 50.)
                    tim.psf = psf
                    newmod = np.zeros((H,W), np.float32)
                    p1 = gal1copy.getModelPatch(tim)
                    p1.addTo(newmod)
                    #mod[:,:] = newmod
        
                    tim.psf = pixpsf
                    modX = np.zeros((H,W), np.float32)
                    p1 = gal1copy.getModelPatch(tim)
                    p1.addTo(modX)
        
                    print('p=', p)
                    cx,cy = np.sum(xx*modX)/np.sum(modX), np.sum(yy*modX)/np.sum(modX)
                    mxx = np.sum((xx-cx)**2*modX)/np.sum(modX)
                    myy = np.sum((yy-cy)**2*modX)/np.sum(modX)
                    mxy = np.sum((xx-cx)*(yy-cy)*modX)/np.sum(modX)
                    print('mod centroid:', cx,cy)
                    print('moments:', mxx,myy,mxy)
                    CX.append(cx)

                plt.plot(pp, np.array(CX)-np.array(pp), '-',
                         label='Lanczos-%i' % L)


                #show_model(modX, newmod, 'mod4(%.1f)' % p)

                # plt.clf()
                # plt.subplot(2,1,1)
                # plt.plot(mod[H/2,:], 'k-')
                # plt.plot(modX[H/2,:], 'r-')
                # plt.subplot(2,1,2)
                # plt.plot(modX[H/2,:] - mod[H/2,:], 'r-')
                # plt.suptitle('mod4(%.1f)' % p)
                # ps.savefig()

            # plt.clf()
            # plt.plot(pp, CX, 'b-')
            # plt.plot(pp, pp, 'k-', alpha=0.25)
            # plt.xlabel('Pixel position')
            # plt.ylabel('Centroid')
            # plt.title('Lanczos-3 interpolation of galaxy profile')
            # ps.savefig()

            plt.axhline(0, color='k', alpha=0.25)
            plt.xlabel('Pixel position')
            plt.ylabel('Centroid - Pixel position')
            plt.title('Lanczos interpolation of galaxy profile')
            plt.legend(loc='upper left')
            ps.savefig()
            
            from astrometry.util.miscutils import lanczos_filter
            plt.clf()
            xx = np.linspace(-(7+1), 7+1, 300)
            for L in [3,5,7]:
                plt.plot(xx, lanczos_filter(L, xx), '-', label='Lancoz-%i' % L)
            plt.title('Lanczos')
            ps.savefig()
            
                                    
        tractor.galaxy.fft_lanczos_order = 3
            
        # Test with ModelMask with "mm"
        p5 = gal1.getModelPatch(tim, modelMask=mm)
        mod5 = np.zeros((H,W), np.float32)
        p5.addTo(mod5)
        print('Patch:', p5)

        if ps is not None:
            show_model(mod5, mod, 'mod5')

        self.assertEqual(p5.x0, 25)
        self.assertEqual(p5.y0, 25)
        self.assertEqual(p5.shape, (50,50))
        print('patch sum:', p5.patch.sum())
        print('Mod sum:', mod5.sum())
        self.assertTrue(np.abs(mod5.sum() - 100.) < 1e-3)
        print('Diff between mods:', np.abs(mod5 - mod).max())
        #self.assertTrue(np.abs(mod5 - mod).max() < 1e-6)
        self.assertTrue(np.abs(mod5 - mod).max() < 2e-3)

        # Test with a source center outside the ModelMask.
        # Way outside the ModelMask -> model is None
        gal1.pos = PixPos(200, -50.)
        p6 = gal1.getModelPatch(tim, modelMask=mm)
        self.assertIsNone(p6)

        # Slightly outside the ModelMask
        gal1.pos = PixPos(20., 25.)
        p7 = gal1.getModelPatch(tim, modelMask=mm)
        mod7 = np.zeros((H,W), np.float32)
        p7.addTo(mod7)
        print('Patch:', p7)

        if ps is not None:
            show_model(mod7, mod, 'mod7')

        self.assertEqual(p7.x0, 25)
        self.assertEqual(p7.y0, 25)
        self.assertEqual(p7.shape, (50,50))
        print('patch sum:', p7.patch.sum())
        print('Mod sum:', mod7.sum())
        #self.assertTrue(np.abs(mod7.sum() - 1.362) < 1e-3)
        self.assertTrue(np.abs(mod7.sum() - 1.963) < 1e-3)

        # Test a HybridPSF
        tim.psf = HybridPixelizedPSF(tim.psf)
        hybridpsf = tim.psf
        
        # Slightly outside the ModelMask
        gal1.pos = PixPos(20., 25.)
        p8 = gal1.getModelPatch(tim, modelMask=mm)
        mod8 = np.zeros((H,W), np.float32)
        p8.addTo(mod8)
        print('Patch:', p8)

        if ps is not None:
            show_model(mod8, mod, 'mod8')

        self.assertEqual(p8.x0, 25)
        self.assertEqual(p8.y0, 25)
        self.assertEqual(p8.shape, (50,50))
        print('patch sum:', p8.patch.sum())
        print('Mod sum:', mod8.sum())
        #self.assertTrue(np.abs(mod8.sum() - 1.362) < 1e-3)
        self.assertTrue(np.abs(mod7.sum() - 1.963) < 1e-3)

        if ps is not None:
            plt.clf()
            plt.imshow(mod7, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('Source outside mask')
            ps.savefig()

            plt.clf()
            plt.imshow(mod8, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('Source outside mask, Hybrid PSF')
            ps.savefig()

        # Put the source close to the image edge.
        gal1.pos = PixPos(5., 5.)
        # No model mask
        p9 = gal1.getModelPatch(tim)
        mod9 = np.zeros((H,W), np.float32)
        p9.addTo(mod9)
        print('Patch:', p9)

        if ps is not None:
            show_model(mod9, None, 'mod9')

        #self.assertEqual(p8.x0, 25)
        #self.assertEqual(p8.y0, 25)
        #self.assertEqual(p8.shape, (50,50))
        print('Mod sum:', mod9.sum())
        #self.assertTrue(np.abs(mod9.sum() - 96.98) < 1e-2)
        self.assertTrue(np.abs(mod9.sum() - 94.33) < 1e-2)

        # Zero outside (0,50),(0,50)
        self.assertEqual(np.sum(np.abs(mod9[50:,:])), 0.)
        self.assertEqual(np.sum(np.abs(mod9[:,50:])), 0.)
        
        if ps is not None:
            plt.clf()
            plt.imshow(mod9, interpolation='nearest', origin='lower',
                       vmin=0, vmax=1e-12)
            plt.colorbar()
            plt.title('Source near image edge')
            ps.savefig()

        # Source back in the middle.
        # Tight ModelMask
        gal1.pos = pos0
        mm10 = ModelMask(45, 45, 10, 10)
        p10 = gal1.getModelPatch(tim, modelMask=mm10)
        mod10 = np.zeros((H,W), np.float32)
        p10.addTo(mod10)
        print('Patch:', p10)
        self.assertEqual(p10.x0, 45)
        self.assertEqual(p10.y0, 45)
        self.assertEqual(p10.shape, (10,10))
        print('Mod sum:', mod10.sum())
        #self.assertTrue(np.abs(mod10.sum() - 96.98) < 1e-2)

        # Larger modelMask
        mm11 = ModelMask(30, 30, 40, 40)
        p11 = gal1.getModelPatch(tim, modelMask=mm11)
        mod11 = np.zeros((H,W), np.float32)
        p11.addTo(mod11)
        print('Patch:', p11)
        print('Mod sum:', mod11.sum())
        self.assertTrue(np.abs(mod11.sum() - 100.) < 1e-3)
        
        if ps is not None:
            plt.clf()
            plt.imshow(mod10, interpolation='nearest', origin='lower')
            plt.colorbar()
            ps.savefig()

            plt.clf()
            plt.imshow(mod11, interpolation='nearest', origin='lower')
            plt.colorbar()
            ps.savefig()

        diff = (mod11 - mod10)[45:55, 45:55]
        print('Max diff:', np.abs(diff).max())
        self.assertTrue(np.abs(diff).max() < 1e-6)


        # DevGalaxy test
        gal1.pos = pos0

        bright2 = bright
        shape2 = GalaxyShape(3., 0.4, 60.)
        gal2 = DevGalaxy(pos, bright2, shape2)

        p12 = gal2.getModelPatch(tim, modelMask=mm)
        mod12 = np.zeros((H,W), np.float32)
        p12.addTo(mod12)
        print('Patch:', p12)
        print('patch sum:', p12.patch.sum())
        print('Mod sum:', mod12.sum())
        self.assertTrue(np.abs(mod12.sum() - 99.95) < 1e-2)

        # Test FixedCompositeGalaxy
        shapeExp = shape
        shapeDev = shape2

        #modExp = mod2
        modExp = mod4
        modDev = mod12
        
        # Set FracDev = 0 --> equals gal1 in patch2.
        gal3 = FixedCompositeGalaxy(pos, bright, FracDev(0.), shapeExp,shapeDev)
        print('Testing galaxy:', gal3)
        print('repr', repr(gal3))
        p13 = gal3.getModelPatch(tim, modelMask=mm)
        mod13 = np.zeros((H,W), np.float32)
        p13.addTo(mod13)
        print('Patch:', p13)
        print('patch sum:', p13.patch.sum())
        print('Mod sum:', mod13.sum())

        if ps is not None:
            show_model(mod13, modExp, 'mod13')

        self.assertTrue(np.abs(mod13.sum() - 100.00) < 1e-2)
        print('SAD:', np.sum(np.abs(mod13 - modExp)))
        self.assertTrue(np.sum(np.abs(mod13 - modExp)) < 2e-5)

        # Set FracDev = 1 --> equals gal2 in patch12.
        gal3.fracDev.setValue(1.)
        p14 = gal3.getModelPatch(tim, modelMask=mm)
        mod14 = np.zeros((H,W), np.float32)
        p14.addTo(mod14)
        print('Patch:', p14)
        print('patch sum:', p14.patch.sum())
        print('Mod sum:', mod14.sum())
        self.assertTrue(np.abs(mod14.sum() - 99.95) < 1e-2)
        print('SAD:', np.sum(np.abs(mod14 - modDev)))
        #self.assertTrue(np.sum(np.abs(mod14 - modDev)) < 1e-8)
        self.assertTrue(np.sum(np.abs(mod14 - modDev)) < 1e-5)

        # Set FracDev = 0.5 --> equals mean
        gal3.fracDev = SoftenedFracDev(0.5)
        p15 = gal3.getModelPatch(tim, modelMask=mm)
        mod15 = np.zeros((H,W), np.float32)
        p15.addTo(mod15)
        print('Patch:', p15)
        print('patch sum:', p15.patch.sum())
        print('Mod sum:', mod15.sum())
        self.assertTrue(np.abs(mod15.sum() - 99.98) < 1e-2)
        target = (modDev + modExp) / 2.
        print('SAD:', np.sum(np.abs(mod15 - target)))
        self.assertTrue(np.sum(np.abs(mod15 - target)) < 2e-5)

        derivs = gal3.getParamDerivatives(tim)
        print('Derivs:', derivs)
        self.assertEqual(len(derivs), 11)

        # CompositeGalaxy
        
        gal4 = CompositeGalaxy(pos, bright, shapeExp, bright, shapeDev)
        print('Testing galaxy:', gal4)
        print('repr', repr(gal4))
        
        p16 = gal4.getModelPatch(tim, modelMask=mm)
        mod16 = np.zeros((H,W), np.float32)
        p16.addTo(mod16)
        print('Patch:', p16)
        print('patch sum:', p16.patch.sum())
        print('Mod sum:', mod16.sum())
        self.assertTrue(np.abs(mod16.sum() - 199.95) < 1e-2)
        target = (modDev + modExp)
        print('SAD:', np.sum(np.abs(mod16 - target)))
        self.assertTrue(np.sum(np.abs(mod16 - target)) < 2e-5)

        derivs = gal4.getParamDerivatives(tim)
        print('Derivs:', derivs)
        self.assertEqual(len(derivs), 12)

        p17,p18 = gal4.getUnitFluxModelPatches(tim, modelMask=mm)
        mod17 = np.zeros((H,W), np.float32)
        mod18 = np.zeros((H,W), np.float32)
        p17.addTo(mod17)
        p18.addTo(mod18)
        print('SAD', np.sum(np.abs(mod17 * 100. - modExp)))
        print('SAD', np.sum(np.abs(mod18 * 100. - modDev)))
        self.assertTrue(np.abs(mod17 * 100. - modExp).sum() < 2e-5)
        self.assertTrue(np.abs(mod18 * 100. - modDev).sum() < 2e-5)

        # SersicGalaxy
        # gal5 = SersicGalaxy(pos, bright, shapeExp, SersicIndex(1.))
        # 
        # tim.psf = psf0
        # 
        # p19 = gal5.getModelPatch(tim, modelMask=mm)
        # mod19 = np.zeros((H,W), np.float32)
        # p19.addTo(mod19)
        # 
        # if ps is not None:
        #     plt.clf()
        #     plt.imshow(mod19, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.title('Sersic n=1')
        #     ps.savefig()
        # 
        #     plt.clf()
        #     plt.imshow(mod19 - modExp, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.title('Sersic n=1 - EXP')
        #     ps.savefig()
        # 
        # print('Patch:', p19)
        # print('patch sum:', p19.patch.sum())
        # print('Mod sum:', mod19.sum())
        # self.assertTrue(np.abs(mod19.sum() - 100.00) < 1e-2)
        # target = modExp
        # # print('SAD:', np.sum(np.abs(mod19 - target)))
        # # self.assertTrue(np.sum(np.abs(mod19 - target)) < 1e-5)
        # 
        # gal5.sersicindex.setValue(4.)
        # gal5.shape = shapeDev
        # 
        # p20 = gal5.getModelPatch(tim, modelMask=mm)
        # mod20 = np.zeros((H,W), np.float32)
        # p20.addTo(mod20)
        # print('Patch:', p20)
        # print('patch sum:', p20.patch.sum())
        # print('Mod sum:', mod20.sum())
        # 
        # if ps is not None:
        #     plt.clf()
        #     plt.imshow(mod20, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.title('Sersic n=4')
        #     ps.savefig()
        # 
        #     plt.clf()
        #     plt.imshow(mod20 - modDev, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.title('Sersic n=4 - DEV')
        #     ps.savefig()
        # 
        # self.assertTrue(np.abs(mod20.sum() - 99.95) < 1e-2)
        # target = modDev
        # print('SAD:', np.sum(np.abs(mod20 - target)))
        # self.assertTrue(np.sum(np.abs(mod20 - target)) < 1e-5)


        # A galaxy that will wrap around
        from tractor.ellipses import EllipseE
        mmX = ModelMask(20, 20, 60, 60)
        shapeX = EllipseE(20., 0.7, 0.7)
        tim.psf = pixpsf
        gal6 = DevGalaxy(pos0, bright, shapeX)

        p21 = gal6.getModelPatch(tim, modelMask=mmX)
        mod21 = np.zeros((H,W), np.float32)
        p21.addTo(mod21)
        
        if ps is not None:
            plt.clf()
            plt.imshow(mod21, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('Pixelized PSF')
            ps.savefig()

            
        tim.psf = hybridpsf
            
        p22 = gal6.getModelPatch(tim, modelMask=mmX)
        mod22 = np.zeros((H,W), np.float32)
        p22.addTo(mod22)

        # Horizontal slice through the middle of the galaxy
        m21 = mod21[49, :]
        m22 = mod22[49, :]
        
        if ps is not None:
            plt.clf()
            plt.imshow(mod22, interpolation='nearest', origin='lower')
            plt.colorbar()
            plt.title('Hybrid PSF')
            ps.savefig()

            #print('m22', m22)
            
            plt.clf()
            plt.plot(m21, 'r-')
            plt.plot(m22, 'b-')
            plt.yscale('symlog', linthreshy=1e-8)
            ps.savefig()

        imx = np.argmax(m22)
        diff = np.diff(m22[:imx+1])
        # Assert monotonic up to the peak (from the left)
        # (low-level jitter/wrap-around is allowed)
        self.assertTrue(np.all(np.logical_or(np.abs(diff) < 2e-9,
                                             diff > 0)))

        diff = np.diff(m22[imx:])
        # Assert monotonic decreasing after to the peak
        # (low-level jitter/wrap-around is allowed)
        self.assertTrue(np.all(np.logical_or(np.abs(diff) < 2e-9,
                                             diff < 0)))

        # Assert that wrap-around exists for PixelizedPsf model
        
        diff = np.diff(m21[:imx+1])
        self.assertFalse(np.all(np.logical_or(np.abs(diff) < 1e-9, diff > 0)))

        diff = np.diff(m21[imx:])
        self.assertFalse(np.all(np.logical_or(np.abs(diff) < 1e-9, diff < 0)))

        
            
        
if __name__ == '__main__':
    import sys
    if '--plots' in sys.argv:
        sys.argv.remove('--plots')
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('gal')

    unittest.main()
