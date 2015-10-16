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

def test_mixture_profiles(ps):

    from tractor.mixture_profiles import MixtureOfGaussians

    W,H = 32,32
    cx,cy = W/2 + 0.5, H/2
    
    vx,vy = 4., 16.
    vxy = 2.
    mp = MixtureOfGaussians(1., [cx, cy], np.array([[[vx,vxy],[vxy,vy]]]))
    print mp

    grid = mp.evaluate_grid_dstn(0, W, 0, H, 0., 0.)
    print grid
    grid = grid.patch
    
    plt.clf()
    dimshow(grid)
    ps.savefig()

    F = np.fft.rfft2(grid)
    v = np.fft.rfftfreq(W)
    w = np.fft.fftfreq(H)

    #print 'F', F
    #print 'w', w
    #print 'v', v
    
    F2 = mp.getFourierTransform(v, w)
    
    F3 = mp.getFourierTransform(v, w, use_mp_fourier=False)

    mx = np.absolute(F).max()

    ima = dict(vmin=-mx, vmax=mx)

    for f in [F, F2, F3]:
        plt.clf()
        plt.subplot(2,2,1)
        dimshow(f.real, **ima)
        plt.subplot(2,2,2)
        dimshow(f.imag, **ima)
        plt.subplot(2,2,3)
        dimshow(np.hypot(f.real, f.imag), **ima)
        plt.subplot(2,2,4)
        #dimshow(np.arctan2(F.real, F.imag))
        dimshow(np.angle(f))
        ps.savefig()

    I1 = np.fft.irfft2(F)
    I2 = np.fft.irfft2(F2)
    I3 = np.fft.irfft2(F3)

    for I in [I1,I2,I3]:
        plt.clf()
        plt.subplot(1,2,1)
        dimshow(I)
        plt.subplot(1,2,2)
        dimshow(np.log10(I))
        ps.savefig()

    for I in [I2,I3]:
        plt.clf()
        dimshow(I-I1)
        ps.savefig()

        
    

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
    #tr.disable_cache()

    
def test_galaxy_grid(ps, args):
    W,H = 800,800

    if len(args):
        testname = args[0]
    else:
        testname = None
    
    xx = np.linspace(20, W-20, 30)
    yy = np.linspace(20, H-20, 30)
    #xx = np.linspace(20, W-20, 2)
    #yy = np.linspace(20, H-20, 2)

    #gpsf = NCircularGaussianPSF([2.], [1.])
    # one component
    #gpsf = GaussianMixturePSF(1.0, 0., 0., 4., 4., 0.)

    # two components
    gpsf = GaussianMixturePSF(0.9, 0.1, 0., 0., 0., 0.,
                              4., 4., 0., 6., 6., 0.)
    # three components
    #gpsf = GaussianMixturePSF(0.8, 0.1, 0.1, 0., 0., 0., 0., 0., 0.,
    #                          4., 4., 0., 6., 6., 0., 8., 8., 0.)
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
    #tr.disable_cache()

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

    # ugal = ExpGalaxy(PixPos(x,y), Flux(100.),
    #                  EllipseESoft(1., 0., 0.))
    # 
    # ugal2 = ExpGalaxy(PixPos(x,y), Flux(100.),
    #                   EllipseESoft(2., 0., 0.))
    # 
    # utr = Tractor([img], [ugal,ugal2])
    # utr.disable_cache()
    # 
    # img.psf = pixpsf
    # m1 = utr.getModelImage(0)
    # img.psf = gpsf
    # m2 = utr.getModelImage(0)
    # print 'm1 rms', np.sqrt(np.sum(m1**2))
    # print 'm2 rms', np.sqrt(np.sum(m2**2))
    # print 'm1-m2 rms', np.sqrt(np.sum((m1-m2)**2))
    # sys.exit(0)
    
    t0 = Time()
    img.psf = pixpsf
    fmod = tr.getModelImage(0)
    t1 = Time()
    tfft = t1 - t0

    t0 = Time()
    img.psf = gpsf
    gmod = tr.getModelImage(0)
    t1 = Time()
    tgauss = t1 - t0

    print 'Gaussian convolution:', tgauss
    print 'FFT      convolution:', tfft

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

    return
    
    plt.clf()
    dimshow(gmod - fmod, **diffa)
    plt.title('Gaussian - FFT convolution')
    ps.savefig()

    tr.setModelMasks([modelmasks])

    t0 = Time()
    img.psf = pixpsf
    fmod = tr.getModelImage(0)
    t1 = Time()
    tfft = t1 - t0

    t0 = Time()
    img.psf = gpsf
    gmod = tr.getModelImage(0)
    t1 = Time()
    tgauss = t1 - t0

    print 'With modelMasks:'
    print 'Gaussian convolution:', tgauss
    print 'FFT      convolution:', tfft

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
    #cx,cy = W/2., H/2.
    cx,cy = 25.0, 24.7
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
    #tr.disable_cache()
    
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

    mask7 = Patch(0,0, np.ones((31,31), bool))
    mask8 = Patch(5,7, np.ones((40,40), bool))
    mask9 = Patch(15,17, np.ones((40,40), bool))
    mask10 = Patch(5,17, np.ones((40,40), bool))
    mask11 = Patch(15,7, np.ones((40,40), bool))

    print
    print 'DIY modelMask'
    
    for mask in [mask7, mask8, mask9, mask10,mask11]: #mask1, mask2, mask3, mask4, mask5, mask6, mask7]:
        #for mx in range(-15,30,2):
        mask = Patch(mx, 7, np.ones((33,33), bool))
    
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


def test_psfex(ps):

    psf = PixelizedPsfEx('psfex-decam-00392360-S31.fits')

    H,W = 100,100

    cx,cy = W/2., H/2.

    pixpsf = psf.constantPsfAt(cx, cy)

    ph,pw = pixpsf.shape
    xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
    im = pixpsf.img.copy()
    im /= np.sum(im)

    cenx,ceny = np.sum(im * xx), np.sum(im * yy)
    print 'Pixpsf centroid:', cenx,ceny
    print 'shape:', ph,pw
    
    dx,dy = cenx - pw/2, ceny - ph/2

    vxx = np.sum(im * (xx - cenx)**2)
    vxy = np.sum(im * (xx - cenx)*(yy - ceny))
    vyy = np.sum(im * (yy - ceny)**2)

    gpsf = GaussianMixturePSF(1., dx, dy, vxx, vyy, vxy)
    
    tim = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                psf=psf)

    xx,yy = np.meshgrid(np.arange(W), np.arange(H))

    star = PointSource(PixPos(cx, cy), Flux(100.))
    gal = ExpGalaxy(PixPos(cx, cy), Flux(100.), EllipseE(1., 0., 0.))

    tr1 = Tractor([tim], [star])
    tr2 = Tractor([tim], [gal])

    disable_galaxy_cache()
    # tr1.disable_cache()
    # tr2.disable_cache()
    
    tim.psf = psf
    mod = tr1.getModelImage(0)

    im = mod.copy()
    im /= im.sum()
    cenx,ceny = np.sum(im * xx), np.sum(im * yy)
    print 'Star model + PsfEx centroid', cenx, ceny
    
    
    plt.clf()
    dimshow(mod)
    plt.title('Star model, PsfEx')
    ps.savefig()

    tim.psf = pixpsf

    mod = tr1.getModelImage(0)
    plt.clf()
    dimshow(mod)
    plt.title('Star model, pixpsf')
    ps.savefig()

    tim.psf = gpsf

    mod = tr1.getModelImage(0)
    plt.clf()
    dimshow(mod)
    plt.title('Star model, gpsf')
    ps.savefig()

    
    tim.psf = psf
    mod = tr2.getModelImage(0)

    im = mod.copy()
    im /= im.sum()
    cenx,ceny = np.sum(im * xx), np.sum(im * yy)
    print 'Gal model + PsfEx centroid', cenx, ceny
    
    plt.clf()
    dimshow(mod)
    plt.title('Gal model, PsfEx')
    ps.savefig()

    tim.psf = pixpsf
    mod = tr2.getModelImage(0)

    plt.clf()
    dimshow(mod)
    plt.title('Gal model, pixpsf')
    ps.savefig()
    

if __name__ == '__main__':
    disable_galaxy_cache()
    ps = PlotSequence('diff')

    parser = optparse.OptionParser()
    opt,args = parser.parse_args()

    test_model_masks(ps)
    sys.exit(0)
    test_galaxy_grid(ps, args)

    test_mixture_profiles(ps)
    
    test_psfex(ps)

    test_fft(ps)

    
