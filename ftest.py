import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.plotutils import *

from tractor import *
from tractor.sdss_galaxy import *

ps = PlotSequence('ftest')

psfsig = 2.
#W,H = 51,51
W,H = 101,101
cx,cy = W/2, H/2

psfim = np.exp(((np.arange(W)-cx)[np.newaxis,:]**2 +
                (np.arange(H)-cy)[:,np.newaxis]**2)/(-2.*psfsig**2))
psfim /= psfim.sum()
print 'psfim shape', psfim.shape

P = np.fft.rfft2(psfim)
print 'P shape', P.shape

ima = dict(interpolation='nearest', origin='lower')

plt.clf()
plt.subplot(1,3,1)
plt.imshow(psfim, **ima)
plt.subplot(1,3,2)
plt.imshow(P.real, **ima)
plt.subplot(1,3,3)
plt.imshow(P.imag, **ima)
ps.savefig()

gal = ExpGalaxy(PixPos(cx,cy), Flux(1.), GalaxyShape(5., 0.5, 30.))

tim = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
            wcs=NullWCS(), psf=NCircularGaussianPSF([0.5], [1.]),
            photocal=LinearPhotoCal(1.), sky=ConstantSky(0.))

tractor = Tractor([tim], [gal])

mod = tractor.getModelImage(0)
print 'mod', mod.shape

# print 'Galaxy:', gal
# mx,my = tim.getWcs().positionToPixel(gal.getPosition())
# mog = tim.getPsf().getMixtureOfGaussians()
# print 'Galaxy pix pos:', mx, my
# print 'MoG PSF:', mog
# pro = gal._getAffineProfile(tim, mx, my)
# print 'Galaxy affine profile:', pro

G = np.fft.rfft2(mod)
print 'G', G.shape

plt.clf()
plt.subplot(1,3,1)
plt.imshow(mod, **ima)
plt.subplot(1,3,2)
plt.imshow(G.real, **ima)
plt.subplot(1,3,3)
plt.imshow(G.imag, **ima)
ps.savefig()


tim = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
            wcs=NullWCS(), psf=NCircularGaussianPSF([psfsig], [1.]),
            photocal=LinearPhotoCal(1.), sky=ConstantSky(0.))
tractor = Tractor([tim], [gal])
mod2 = tractor.getModelImage(0)
G2 = np.fft.rfft2(mod2)

print 'mod2', mod2.shape
print 'G2', G2.shape

mx,my = 0.,0.
gmog = gal._getAffineProfile(tim, mx, my)

w = np.fft.rfftfreq(W)
v = np.fft.fftfreq(W)
print 'Frequencies:', len(w), 'x', len(v)

ww,vv = np.meshgrid(w, v)

print 'ww,vv', ww.shape, vv.shape

Fsum = None
for k in range(gmog.K):
    V = gmog.var[k,:,:]
    iv = np.linalg.inv(V)
    mu = gmog.mean[k,:]
    amp = gmog.amp[k]
    #print 'Variance', V
    #print 'iv', iv
    a,b,d = iv[0,0], iv[0,1], iv[1,1]

    assert(np.all(mu == 0))
    
    a *= 0.5
    b *= 0.5
    d *= 0.5
    
    det = a*d - b**2
    #F = (np.pi / np.sqrt(det) *
    F = np.exp(-np.pi**2/det * (a * vv**2 + d*ww**2 - 2*b*vv*ww))
    # * np.exp(2.*np.pi* 1j * (mu[0]*vv + mu[1]*ww)))

    #F /= (2. * np.pi * np.sqrt(np.linalg.det(V)))

    #print 'det V', np.linalg.det(V)
    #print 'det iv', np.linalg.det(iv)
    
    print 'F', F.shape
    
    if Fsum is None:
        Fsum = amp * F
    else:
        Fsum += amp * F
    
    if False:
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(F.real, **ima)
        plt.subplot(2,2,2)
        plt.imshow(F.imag, **ima)
        plt.subplot(2,2,3)
        plt.imshow(Fsum.real, **ima)
        plt.subplot(2,2,4)
        plt.imshow(Fsum.imag, **ima)
        ps.savefig()

    FF = np.fft.irfft2(Fsum, s=(H,W))

    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(FF, **ima)
    plt.subplot(1,3,2)
    plt.imshow(Fsum.real, **ima)
    plt.subplot(1,3,3)
    plt.imshow(Fsum.imag, **ima)
    ps.savefig()
    
        
print 'abs max Fsum.imag:', np.abs(Fsum.imag).max()
    
IG = np.fft.irfft2(Fsum, s=(H,W))
print 'IG', IG.shape

plt.clf()
plt.subplot(1,3,1)
plt.imshow(IG, **ima)
plt.subplot(1,3,2)
plt.imshow(Fsum.real, **ima)
plt.subplot(1,3,3)
plt.imshow(Fsum.imag, **ima)
ps.savefig()

CF = Fsum * P
C = np.fft.irfft2(CF, s=(H,W))
print 'CF', CF.shape
print 'C', C.shape

plt.clf()
plt.subplot(1,3,1)
plt.imshow(C, **ima)
plt.subplot(1,3,2)
plt.imshow(CF.real, **ima)
plt.subplot(1,3,3)
plt.imshow(CF.imag, **ima)
ps.savefig()

plt.clf()
plt.subplot(1,3,1)
plt.imshow(mod2, **ima)
plt.subplot(1,3,2)
plt.imshow(G2.real, **ima)
plt.subplot(1,3,3)
plt.imshow(G2.imag, **ima)
ps.savefig()


plt.clf()
mx = np.hypot(G2.real, G2.imag).max()
plt.subplot(2,3,1)
plt.imshow(np.log10(np.abs(CF.real) / mx), vmin=-6, vmax=0, **ima)
plt.subplot(2,3,4)
plt.imshow(np.log10(np.abs(CF.imag) / mx), vmin=-6, vmax=0, **ima)
plt.subplot(2,3,2)
plt.imshow(np.log10(np.abs(G2.real) / mx), vmin=-6, vmax=0, **ima)
plt.subplot(2,3,5)
plt.imshow(np.log10(np.abs(G2.imag) / mx), vmin=-6, vmax=0, **ima)

plt.subplot(2,3,3)
plt.imshow(np.log10(np.abs(CF.real - G2.real) / mx), vmin=-12, vmax=0, **ima)
plt.subplot(2,3,6)
plt.imshow(np.log10(np.abs(CF.imag - G2.imag) / mx), vmin=-12, vmax=0, **ima)

ps.savefig()


print 'C', C.sum()
print 'mod2:', mod2.sum()

plt.clf()
mx = mod2.max()
plt.subplot(2,2,1)
plt.imshow(np.log10(mod2/mx), vmin=-6, vmax=0, **ima)
plt.subplot(2,2,2)
plt.imshow(np.log10(C/mx), vmin=-6, vmax=0, **ima)
plt.subplot(2,2,3)
resid = C - mod2
mx = np.abs(resid).max()
plt.imshow(resid, vmin=-mx, vmax=mx, **ima)

plt.subplot(2,2,4)
plt.imshow(np.log10(resid/mx), vmin=-6, vmax=0, **ima)

ps.savefig()

print 'Residual: max', mx

R = np.fft.rfft2(resid)
mx = max(np.abs(R.real).max(), np.abs(R.imag).max())

plt.clf()
plt.subplot(1,3,1)
plt.imshow(resid, **ima)
plt.subplot(1,3,2)
plt.imshow(R.real, vmin=-mx, vmax=mx, **ima)
plt.subplot(1,3,3)
plt.imshow(R.imag, vmin=-mx, vmax=mx, **ima)
ps.savefig()


