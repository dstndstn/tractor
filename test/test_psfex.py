from __future__ import print_function
import os
import numpy as np

import unittest

from tractor import *
from tractor.galaxy import ExpGalaxy, disable_galaxy_cache
from tractor.psfex import PixelizedPsfEx

ps = None

class PsfExTest(unittest.TestCase):

    def test_psfex(self):

        if ps is not None:
            from astrometry.util.plotutils import dimshow
            import pylab as plt

        fn = os.path.join(os.path.dirname(__file__),
                          'psfex-decam-00392360-S31.fits')
        psf = PixelizedPsfEx(fn)
    
        H,W = 100,100
        cx,cy = W/2., H/2.
    
        pixpsf = psf.constantPsfAt(cx, cy)

        ph,pw = pixpsf.shape
        xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
        im = pixpsf.img.copy()
        im /= np.sum(im)
        cenx,ceny = np.sum(im * xx), np.sum(im * yy)
        print('Pixpsf centroid:', cenx,ceny)
        print('shape:', ph,pw)
        
        dx,dy = cenx - pw/2, ceny - ph/2
        print('dx,dy', dx,dy)
        
        # gpsf = GaussianMixturePSF.fromStamp(im, N=1)
        # print('Fit gpsf:', gpsf)
        # self.assertTrue(np.abs(gpsf.mog.mean[0,0] - dx) < 0.1)
        # self.assertTrue(np.abs(gpsf.mog.mean[0,1] - dy) < 0.1)
        # self.assertTrue(np.abs(gpsf.mog.var[0,0,0] - 15.5) < 1.)
        # self.assertTrue(np.abs(gpsf.mog.var[0,1,1] - 13.5) < 1.)
        # self.assertTrue(np.abs(gpsf.mog.var[0,1,0] -   -1) < 1.)

        gpsf = GaussianMixturePSF.fromStamp(im, N=2)
        print('Fit gpsf:', gpsf)
        print('Params:', ', '.join(['%.1f' % p for p in gpsf.getParams()]))

        pp = np.array([0.8, 0.2, 0.1, -0.0, 1.2, 0.2, 7.6, 6.0, -1.0, 51.6, 49.1, -1.3])
        self.assertTrue(np.all(np.abs(np.array(gpsf.getParams()) - pp) < 0.1))
        
        tim = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                    psf=psf)
    
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    
        star = PointSource(PixPos(cx, cy), Flux(100.))
        gal = ExpGalaxy(PixPos(cx, cy), Flux(100.), EllipseE(1., 0., 0.))
    
        tr1 = Tractor([tim], [star])
        tr2 = Tractor([tim], [gal])
    
        disable_galaxy_cache()
        
        tim.psf = psf
        mod = tr1.getModelImage(0)
        mod1 = mod
    
        im = mod.copy()
        im /= im.sum()
        cenx,ceny = np.sum(im * xx), np.sum(im * yy)
        print('Star model + PsfEx centroid', cenx, ceny)

        self.assertTrue(np.abs(cenx - (cx+dx)) < 0.1)
        self.assertTrue(np.abs(ceny - (cy+dy)) < 0.1)

        if ps is not None:
            plt.clf()
            dimshow(mod)
            plt.title('Star model, PsfEx')
            ps.savefig()

        tim.psf = pixpsf
    
        mod = tr1.getModelImage(0)

        if ps is not None:
            plt.clf()
            dimshow(mod)
            plt.title('Star model, pixpsf')
            ps.savefig()
    
        tim.psf = gpsf
    
        mod = tr1.getModelImage(0)
        mod2 = mod
        
        if ps is not None:
            plt.clf()
            dimshow(mod)
            plt.title('Star model, gpsf')
            plt.colorbar()
            ps.savefig()
        
            plt.clf()
            dimshow(mod1 - mod2)
            plt.title('Star model, PsfEx - gpsf')
            plt.colorbar()
            ps.savefig()

        # range ~ -0.15 to +0.25
        self.assertTrue(np.all(np.abs(mod1 - mod2) < 0.25))
        
        tim.psf = psf
        mod = tr2.getModelImage(0)
        mod1 = mod

        im = mod.copy()
        im /= im.sum()
        cenx,ceny = np.sum(im * xx), np.sum(im * yy)
        print('Gal model + PsfEx centroid', cenx, ceny)

        self.assertTrue(np.abs(cenx - (cx+dx)) < 0.1)
        self.assertTrue(np.abs(ceny - (cy+dy)) < 0.1)
        
        if ps is not None:
            plt.clf()
            dimshow(mod)
            plt.title('Gal model, PsfEx')
            ps.savefig()
    
        tim.psf = pixpsf
        mod = tr2.getModelImage(0)
    
        # plt.clf()
        # dimshow(mod)
        # plt.title('Gal model, pixpsf')
        # ps.savefig()

        tim.psf = gpsf
        mod = tr2.getModelImage(0)
        mod2 = mod
    
        if ps is not None:
            plt.clf()
            dimshow(mod)
            plt.title('Gal model, gpsf')
            ps.savefig()

            plt.clf()
            dimshow(mod1 - mod2)
            plt.title('Gal model, PsfEx - gpsf')
            plt.colorbar()
            ps.savefig()

        # range ~ -0.1 to +0.2
        self.assertTrue(np.all(np.abs(mod1 - mod2) < 0.2))
        

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--plots', action='store_true', help='Create plots?')
    # opt = parser.parse_args()
    #if opt.plots:
    import sys
    if '--plots' in sys.argv:
        sys.argv.remove('--plots')
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('psfex')

    unittest.main()
