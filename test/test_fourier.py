import sys
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import optparse

import numpy as np

from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

from astrometry.util.plotutils import *
from astrometry.util.ttime import *


def test_fft(ps):
    # DevGalaxy(pos=RaDecPos[240.14431131092763, 6.1210541865974211], brightness=NanoMaggies: g=22.4, r=21.2, z=20.4, shape=log r_e=3.51358, ee1=-0.0236628, ee2=0.125124
    # img 00347334-S31 r
    # img.shape (93, 89)
    # modelMask.shape (129, 94)

    # (Pdb) p gh
    # 128
    # (Pdb) p gw
    # 128
    # (Pdb) p mh
    # 129
    # (Pdb) p mw
    # 94
    # p halfsize
    # 64
    # p px,py
    # (58.41863462372021, 28.101390778484983)

    gpsf = GaussianMixturePSF(0.8, 0.1, 0.1, 0., 0., 0., 0., 0., 0.,
                              4., 4., 0., 6., 6., 0., 8., 8., 0.)
    gpsf.radius = 15
    psfimg = gpsf.getPointSourcePatch(0., 0., radius=15)
    print 'PSF image size', psfimg.shape
    pixpsf = PixelizedPSF(psfimg.patch)

    data=np.zeros((H,W), np.float32)
    img = Image(data=data, invvar=np.ones_like(data), psf=gpsf)

    modelmasks = dict()
    gal = ExpGalaxy(PixPos(x,y), Flux(100.),
                    EllipseESoft(3., x/float(W), y/float(H)))
    mmsz = 100
    modelmasks[gal] = Patch(int(x-mmsz/2), int(y-mmsz/2),
                            np.ones((mmsz,mmsz), bool))
    tr = Tractor([img], cat)
    tr.disable_cache()

    
def test_galaxy_grid(ps, args):
    W,H = 800,800

    if len(args):
        testname = args[0]
    else:
        testname = None
    
    xx = np.linspace(20, W-20, 30)
    yy = np.linspace(20, H-20, 30)
    #xx = np.linspace(20, W-20, 10)
    #yy = np.linspace(20, H-20, 10)

    #gpsf = NCircularGaussianPSF([2.], [1.])
    # two components
    #gpsf = GaussianMixturePSF(0.9, 0.1, 0., 0., 0., 0.,
    #                          4., 4., 0., 6., 6., 0.)
    # three components
    gpsf = GaussianMixturePSF(0.8, 0.1, 0.1, 0., 0., 0., 0., 0., 0.,
                              4., 4., 0., 6., 6., 0., 8., 8., 0.)
    gpsf.radius = 15
    psfimg = gpsf.getPointSourcePatch(0., 0., radius=15)
    print 'PSF image size', psfimg.shape
    pixpsf = PixelizedPSF(psfimg.patch)

    data=np.zeros((H,W), np.float32)
    img = Image(data=data, invvar=np.ones_like(data), psf=gpsf)

    modelmasks = dict()
    mmsz = 100

    cat = []
    for y in yy:
        for x in xx:
            gal = ExpGalaxy(PixPos(x,y), Flux(100.),
                            EllipseESoft(3., x/float(W), y/float(H)))
            cat.append(gal)
            modelmasks[gal] = Patch(int(x-mmsz/2), int(y-mmsz/2),
                                    np.ones((mmsz,mmsz), bool))

            
    tr = Tractor([img], cat)
    tr.disable_cache()

    if testname is not None:
        t0 = Time()
        if testname == 'gauss1':
            img.psf = gpsf
            mod = tr.getModelImage(0)
        elif testname == 'fft1':
            img.psf = pixpsf
            mod = tr.getModelImage(0)

        elif testname == 'gauss2':
            tr.setModelMasks([modelmasks])
            img.psf = gpsf
            mod = tr.getModelImage(0)
        elif testname == 'fft2':
            tr.setModelMasks([modelmasks])
            img.psf = pixpsf
            mod = tr.getModelImage(0)

        t1 = Time()
        print t1 - t0
        return
    
    t0 = Time()
    img.psf = gpsf
    gmod = tr.getModelImage(0)
    t1 = Time()
    img.psf = pixpsf
    fmod = tr.getModelImage(0)
    t2 = Time()

    print 'Gaussian convolution:', t1-t0
    print 'FFT      convolution:', t2-t1

    mx = gmod.max()
    ima = dict(vmin=-0.01*mx, vmax=0.95*mx)
    dfrac = 0.01
    diffa = dict(vmin=-dfrac*mx, vmax=dfrac*mx)

    plt.clf()
    dimshow(gmod, **ima)
    plt.title('Gaussian convolution')
    ps.savefig()

    plt.clf()
    dimshow(fmod, **ima)
    plt.title('FFT convolution')
    ps.savefig()

    plt.clf()
    dimshow(gmod - fmod, **diffa)
    plt.title('Gaussian - FFT convolution')
    ps.savefig()

    tr.setModelMasks([modelmasks])
    t0 = Time()
    img.psf = gpsf
    gmod = tr.getModelImage(0)
    t1 = Time()
    img.psf = pixpsf
    fmod = tr.getModelImage(0)
    t2 = Time()
    
    print 'With modelMasks:'
    print 'Gaussian convolution:', t1-t0
    print 'FFT      convolution:', t2-t1

    plt.clf()
    dimshow(gmod, **ima)
    plt.title('Gaussian convolution')
    ps.savefig()

    plt.clf()
    dimshow(fmod, **ima)
    plt.title('FFT convolution')
    ps.savefig()

    plt.clf()
    dimshow(gmod - fmod, **diffa)
    plt.title('Gaussian - FFT convolution')
    ps.savefig()
    


def test_model_masks(ps):
    W,H = 50,50
    cx,cy = W/2., H/2.
    gal = ExpGalaxy(PixPos(cx,cy), Flux(100.), EllipseESoft(1., 0., 0.5))
    #gal = ExpGalaxy(PixPos(cx,cy), Flux(100.), EllipseESoft(-1., 0., 0.5))
    halfsize = 25
    
    gpsf = NCircularGaussianPSF([2.], [1.])
    gpsf.radius = halfsize
    psfimg = gpsf.getPointSourcePatch(0., 0., radius=15)
    print 'PSF image size', psfimg.shape
    pixpsf = PixelizedPSF(psfimg.patch)
    #psf = GaussianMixturePSF([1.], [0., 0.], 2.*np.array([[[1.,0.],[0.,1.]],]))
    #psf.radius = 15
    #psfimg = psf.
    data=np.zeros((H,W), np.float32)
    img = Image(data=data, invvar=np.ones_like(data), psf=gpsf)
    
    px,py = cx,cy
    
    mod_mog = gal.getModelPatch(img)
    
    img.psf = pixpsf
    mod_fft = gal.getModelPatch(img)
    
    print 'MoG model:', mod_mog.shape, mod_mog.patch.min(), mod_mog.patch.max()
    print 'FFT model:', mod_fft.shape, mod_fft.patch.min(), mod_fft.patch.max()
    
    dfrac = 0.01
    
    mx = mod_mog.patch.max()
    ima = dict(vmin=0.05*mx, vmax=0.95*mx)
    diffa = dict(vmin=-dfrac*mx, vmax=dfrac*mx)
    
    plt.clf()
    plt.subplot(1,3,1)
    dimshow(mod_mog.patch, **ima)
    plt.title('MoG')
    plt.subplot(1,3,2)
    dimshow(mod_fft.patch, **ima)
    plt.title('FFT')
    if mod_mog.shape == mod_fft.shape:
        plt.subplot(1,3,3)
        dimshow(mod_mog.patch - mod_fft.patch, **diffa)
        plt.title('Diff')
    ps.savefig()
    
    
    tr = Tractor([img], [gal])
    tr.disable_cache()
    
    img.psf = gpsf
    mod_mog = tr.getModelImage(0)
    img.psf = pixpsf
    mod_fft = tr.getModelImage(0)
    
    mx = mod_mog.max()
    ima = dict(vmin=0.05*mx, vmax=0.95*mx)
    diffa = dict(vmin=-dfrac*mx, vmax=dfrac*mx)
    
    plt.clf()
    plt.subplot(1,3,1)
    dimshow(mod_mog, **ima)
    plt.title('MoG')
    plt.subplot(1,3,2)
    dimshow(mod_fft, **ima)
    plt.title('FFT')
    plt.subplot(1,3,3)
    dimshow(mod_mog - mod_fft, **diffa)
    plt.title('Diff')
    ps.savefig()
    
    mask1 = Patch(10, 10, np.ones((30,30), bool))
    mask2 = Patch(15, 10, np.ones((29,30), bool))
    mask3 = Patch(0, 0, np.ones((20,20), bool))
    mask4 = Patch(30, 30, np.ones((20,20), bool))
    
    mask5 = Patch(20, 20, np.ones((10,10), bool))
    mask6 = Patch(-25, -25, np.ones((100,100), bool))
    
    print
    print 'DIY modelMask'
    
    for mask in [mask1, mask2, mask3, mask4, mask5, mask6]:
    
        print
        print 'MoG:'
    
        img.psf = gpsf
        mod_mog = gal.getModelPatch(img, modelMask=mask)
        assert(mod_mog.x0 == mask.x0)
        assert(mod_mog.y0 == mask.y0)
        assert(mod_mog.shape == mask.shape)
    
        print
        print 'FFT:'
    
        # pad,cx,cy = pixpsf._padInImage(*mask5.shape)
        # print 'Padded PSF image:', pad.shape, 'center', cx,cy
        # plt.clf()
        # dimshow(pad)
        # ps.savefig()
        # sys.exit(0)
        
        img.psf = pixpsf
        mod_fft = gal.getModelPatch(img, modelMask=mask)
        assert(mod_fft.x0 == mask.x0)
        assert(mod_fft.y0 == mask.y0)
        assert(mod_fft.shape == mask.shape)
    
        mx = mod_mog.patch.max()
        ima = dict(vmin=0.05*mx, vmax=0.95*mx)
        diffa = dict(vmin=-dfrac*mx, vmax=dfrac*mx)
    
        plt.clf()
        plt.subplot(1,3,1)
        dimshow(mod_mog.patch, **ima)
        plt.title('MoG')
        plt.subplot(1,3,2)
        dimshow(mod_fft.patch, **ima)
        plt.title('FFT')
        plt.subplot(1,3,3)
        dimshow(mod_mog.patch - mod_fft.patch, **diffa)
        plt.title('Diff')
        ps.savefig()
    
        
    print
    print 'setModelMasks'
    print
    print 'MoG:'
    
    modmask = [{gal: mask}]
    tr.setModelMasks(modmask)
    
    img.psf = gpsf
    mod_mog = tr.getModelImage(0)
    
    print
    print 'FFT:'
    
    img.psf = pixpsf
    mod_fft = tr.getModelImage(0)
    
    mx = mod_mog.max()
    ima = dict(vmin=0.05*mx, vmax=0.95*mx)
    diffa = dict(vmin=-dfrac*mx, vmax=dfrac*mx)
    
    plt.clf()
    plt.subplot(1,3,1)
    dimshow(mod_mog, **ima)
    plt.title('MoG')
    plt.subplot(1,3,2)
    dimshow(mod_fft, **ima)
    plt.title('FFT')
    plt.subplot(1,3,3)
    dimshow(mod_mog - mod_fft, **diffa)
    plt.title('Diff')
    ps.savefig()
    
    


def OLD_STUFF():
    P,(px0,py0),(pH,pW) = pixpsf.getFourierTransform(halfsize)
    
    w = np.fft.rfftfreq(pW)
    v = np.fft.fftfreq(pH)
    
    dx = px - px0
    dy = py - py0
    # Put the integer portion of the offset into Patch x0,y0
    ix0 = int(np.round(dx))
    iy0 = int(np.round(dy))
    # Put the subpixel portion into the galaxy FFT.
    mux = dx - ix0
    muy = dy - iy0
    
    amix = gal._getAffineProfile(img, mux, muy)
    Fsum = amix.getFourierTransform(w, v)
    #print 'Fsum:', Fsum
    print type(Fsum)
    
    plt.clf()
    plt.subplot(1,2,1)
    dimshow(Fsum.real)
    plt.subplot(1,2,2)
    dimshow(Fsum.imag)
    plt.savefig('fsum.png')
    
    ig = np.fft.irfft2(Fsum, s=(pH,pW))
    
    plt.clf()
    dimshow(ig)
    plt.savefig('ig.png')
    
    # Fake image with tiny PSF -- what does it look like if we render out the
    # galaxy and FT it?
    tinypsf = NCircularGaussianPSF([0.01], [1.])
    tinyimg = Image(data=data, invvar=np.ones_like(data), psf=tinypsf)
    gal.halfsize = 25
    tinyp = gal.getModelPatch(tinyimg)
    
    plt.clf()
    dimshow(tinyp.patch)
    plt.savefig('tiny.png')
    
    print 'tinyp shape', tinyp.shape
    tpsf = PixelizedPSF(tinyp.patch)
    Ftiny,nil,nil = tpsf.getFourierTransform(25)
    
    plt.clf()
    plt.subplot(1,2,1)
    dimshow(Ftiny.real)
    plt.subplot(1,2,2)
    dimshow(Ftiny.imag)
    plt.savefig('ftiny.png')
    
    
    
    G = np.fft.irfft2(Fsum * P, s=(pH,pW))
    galpatch = Patch(ix0, iy0, G)
    
    
    mod = np.zeros(img.shape, np.float32)
    galpatch.addTo(mod)
    
    plt.clf()
    dimshow(mod)
    plt.savefig('conv.png')





if __name__ == '__main__':
    disable_galaxy_cache()
    ps = PlotSequence('diff')

    parser = optparse.OptionParser()
    opt,args = parser.parse_args()

    #test_fft(ps)
    
    test_model_masks(ps)
    sys.exit(0)

    test_galaxy_grid(ps, args)
    
