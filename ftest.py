from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import sys

from astrometry.util.plotutils import *

from tractor import *
from tractor.galaxy import *

def video():
    ps = PlotSequence('ftest', format='%04i')
    psfsig = 2.
    W,H = 101,81
    cx,cy = W/2, H/2
    psfim = np.exp(((np.arange(W)-cx)[np.newaxis,:]**2 +
                    (np.arange(H)-cy)[:,np.newaxis]**2)/(-2.*psfsig**2))
    psfim /= psfim.sum()
    P = np.fft.rfft2(psfim)
    ima = dict(interpolation='nearest', origin='lower')
    gx,gy = W/2 + 0.3, H/2 + 0.5
    gal = ExpGalaxy(PixPos(gx,gy), Flux(1.), GalaxyShape(5., 0.5, 30.))
    tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)),
                wcs=NullWCS(), psf=NCircularGaussianPSF([psfsig], [1.]),
                photocal=LinearPhotoCal(1.), sky=ConstantSky(0.))
    tractor = Tractor([tim], [gal])
    mod2 = tractor.getModelImage(0)
    G2 = np.fft.rfft2(mod2)
    P = np.fft.rfft2(psfim)
    w = np.fft.rfftfreq(W)
    v = np.fft.fftfreq(H)
    plt.figure(figsize=(5.1,4.05))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    #for phi in np.linspace(30., 30.00001, 100):
    #for re in np.linspace(5., 5.001, 100):
    #    gal.shape.re = re
    for phi in np.linspace(30., 30.0001, 1000):
        gal.shape.phi = phi
        mod1 = tractor.getModelImage(0)
        F1 = np.fft.rfft2(mod1)
        mx,my = gx - cx, gy - cy
        gmog = gal._getAffineProfile(tim, mx, my)
        Fsum = gmog.getFourierTransform(w, v)
        F2 = Fsum * P
        mod2 = np.fft.irfft2(F2, s=mod1.shape)
        plt.clf()
        mx = np.hypot(F1.real, F1.imag).max()
        lim = np.fliplr(np.log10(np.abs(F1.real - F2.real) / mx))
        rim = np.log10(np.abs(F1.imag - F2.imag) / mx)
        M = np.hstack((lim, rim))
        vmin,vmax = 7, 11
        plt.imshow(-M, vmin=vmin, vmax=vmax, cmap='hot', **ima)
        plt.xticks([]); plt.yticks([])
        ps.savefig()
        # avconv -r 20 -i ftest-%04d.png -b:v 2000k /tmp/err.mp4




    
ps = PlotSequence('ftest', format='%04i')

psfsig = 2.
#W,H = 51,51
W,H = 101,81
cx,cy = W/2, H/2


#psfmodel = NCircularGaussianPSF([psfsig], [1.])

#V = np.array([ [[ 2., -0.3 ], [-0.3, 4.]] ])
#V = np.array([ [[ 2., -1. ], [-1., 4.]] ])
#psfmodel = GaussianMixturePSF(np.array([1.]), np.zeros((1,2)), V)

# asymmetric multi-component Gaussian
V = np.array([ [[ 2., -1. ], [-1., 4.]],
               [[ 1., 0.], [0., 1.]] ])
mus = np.array([ [0.,0.], [1.,1.]])
psfmodel = GaussianMixturePSF(np.array([0.6, 0.4]), mus, V)

psfpatch = psfmodel.getPointSourcePatch(cx, cy, radius=max(W/2, H/2))
psfim = np.zeros((H,W))
psfpatch.addTo(psfim)

psfim /= psfim.sum()
print('psfim shape', psfim.shape)

P = np.fft.rfft2(psfim)
print('P shape', P.shape)

ima = dict(interpolation='nearest', origin='lower')

# psfim2 = np.exp(((np.arange(W)-cx)[np.newaxis,:]**2 +
#                  (np.arange(H)-cy)[:,np.newaxis]**2)/(-2.*psfsig**2))
# psfim2 /= psfim2.sum()

if False:
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(psfim, **ima)
    #plt.subplot(2,2,2)
    #plt.imshow(psfim2, **ima)
    #plt.imshow(psfim - psfim2, **ima)
    plt.subplot(2,2,3)
    plt.imshow(P.real, **ima)
    plt.subplot(2,2,4)
    plt.imshow(P.imag, **ima)
    plt.suptitle('PSF')
    ps.savefig()

# padpsf = np.zeros((H*3, W*3))
# padpsf[H:2*H, W:2*W] = psfim
# P2 = np.fft.rfft2(padpsf)
# plt.clf()
# plt.subplot(1,3,1)
# plt.imshow(padpsf, **ima)
# plt.subplot(1,3,2)
# plt.imshow(P2.real, **ima)
# plt.subplot(1,3,3)
# plt.imshow(P2.imag, **ima)
# ps.savefig()

gx,gy = W/2 + 0.3, H/2 + 0.5

gal = ExpGalaxy(PixPos(gx,gy), Flux(1.), GalaxyShape(5., 0.5, 30.))

tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)),
            wcs=NullWCS(), psf=NCircularGaussianPSF([0.5], [1.]),
            photocal=LinearPhotoCal(1.), sky=ConstantSky(0.))


psf2 = PixelizedPSF(psfim[cy-20:cy+20+1, cx-20:cx+20+1])
tim.psf = psf2
#1mod = gal.getModelPatch(tim)

#from tractor.mixture_profiles import *

mux,muy = 0.4, 0.5
#halfsize = 4
halfsize = 50
amix = gal._getAffineProfile(tim, mux, muy)
#amix = MixtureOfGaussians(amix.amp[:2], amix.mean[:2,:], amix.var[:2,:,:])
P,(px0,py0),(pH,pW) = tim.getPsf().getFourierTransform(halfsize)
w = np.fft.rfftfreq(pW)
v = np.fft.fftfreq(pH)
print('w', w)
print('v', v)
F1 = amix.getFourierTransform(w, v, use_mp_fourier=False)
F2 = amix.getFourierTransform(w, v, use_mp_fourier=True)

print(F1[:5,:5])
print(F2[:5,:5])

print('F1:', F1.dtype, F1.shape)
print('F2:', F2.dtype, F2.shape)
diff = F1 - F2
print('max diff', np.max(np.abs(diff)))

mx = max(np.hypot(F1.real, F1.imag).max(), np.hypot(F2.real, F2.imag).max())
plt.clf()
plt.subplot(2,2,1)
plt.imshow(F1.real, vmin=-mx, vmax=mx, **ima)
plt.title('True real')
plt.subplot(2,2,2)
plt.imshow(F1.imag, vmin=-mx, vmax=mx, **ima)
plt.title('True imag')

plt.subplot(2,2,3)
plt.imshow(F2.real, vmin=-mx, vmax=mx, **ima)
plt.subplot(2,2,4)
plt.imshow(F2.imag, vmin=-mx, vmax=mx, **ima)
ps.savefig()
sys.exit(0)
        

tractor = Tractor([tim], [gal])

mod = tractor.getModelImage(0)
print('mod', mod.shape)

G = np.fft.rfft2(mod)
print('G', G.shape)

if False:
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(mod, **ima)
    plt.subplot(1,3,2)
    plt.imshow(G.real, **ima)
    plt.subplot(1,3,3)
    plt.imshow(G.imag, **ima)
    plt.suptitle('Galaxy (conv narrow PSF)')
    ps.savefig()

tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)),
            wcs=NullWCS(), psf=psfmodel,
            photocal=LinearPhotoCal(1.), sky=ConstantSky(0.))
tractor = Tractor([tim], [gal])
mod2 = tractor.getModelImage(0)
G2 = np.fft.rfft2(mod2)
print('mod2', mod2.shape)
print('G2', G2.shape)

print()
print()

psf1 = tim.getPsf()
u1 = gal.getUnitFluxModelPatch(tim)

psf2 = PixelizedPSF(psfim[cy-20:cy+20+1, cx-20:cx+20+1])

print('Pixelized PSF shape:', psf2.img.shape)

tim.psf = psf2

halfsize = 80.
P,(px0,py0),pshape = psf2.getFourierTransform(halfsize)

print('PSF x0,y0', px0, py0)

pad = np.fft.irfft2(P, s=pshape)
plt.clf()
plt.subplot(1,3,1)
plt.imshow(P.real, **ima)
plt.subplot(1,3,2)
plt.imshow(P.imag, **ima)
plt.subplot(1,3,3)
plt.imshow(pad, **ima)
plt.suptitle('padded PSF')
ps.savefig()

print('PSF shape', pshape)
pH,pW = pshape
print('F(PSF) shape:', P.shape)
w = np.fft.rfftfreq(pW)
v = np.fft.fftfreq(pH)

dx = gx - px0
dy = gy - py0
ix0 = int(np.round(dx))
iy0 = int(np.round(dy))
mux = dx - ix0
muy = dy - iy0
print('ix0,iy0', ix0,iy0)
print('mux,muy', mux,muy)

amix = gal._getAffineProfile(tim, mux, muy)
print('w', len(w), 'v', len(v))
Fsum = None
for k in range(amix.K):
    V = amix.var[k,:,:]
    iv = np.linalg.inv(V)
    mu = amix.mean[k,:]

    mux = -mu[0]
    muy = -mu[1]
    
    amp = amix.amp[k]
    a,b,d = 0.5 * iv[0,0], 0.5 * iv[0,1], 0.5 * iv[1,1]
    det = a*d - b**2
    F = (np.exp(-np.pi**2/det *
                (a * v[:,np.newaxis]**2 +
                 d * w[np.newaxis,:]**2 -
                 2*b*v[:,np.newaxis]*w[np.newaxis,:]))
                 * np.exp(2.*np.pi* 1j * (mux*w[np.newaxis,:] + 
                                          muy*v[:,np.newaxis])))
    if Fsum is None:
        Fsum = amp * F
    else:
        Fsum += amp * F

y1 = 100
x1 = 120

print('Fsum shape', Fsum.shape)
G = np.fft.irfft2(Fsum, s=pshape)
gh,gw = G.shape

plt.clf()
plt.subplot(1,3,1)
plt.imshow(Fsum.real, **ima)
plt.subplot(1,3,2)
plt.imshow(Fsum.imag, **ima)
plt.suptitle('Galaxy')
plt.subplot(1,3,3)
plt.imshow(G, **ima)#extent=[ix0,ix0+gw, iy0,iy0+gh], **ima)
ps.savefig()

FC = Fsum * P

C = np.fft.irfft2(FC, s=pshape)
ch,cw = C.shape
print('C shape', C.shape)

plt.clf()
plt.subplot(1,3,1)
plt.imshow(FC.real, **ima)
plt.subplot(1,3,2)
plt.imshow(FC.imag, **ima)
plt.subplot(1,3,3)
plt.imshow(C, extent=[ix0,ix0+cw, iy0,iy0+ch], **ima)
plt.suptitle('Galaxy conv PSF')
ps.savefig()

# plt.clf()
# plt.imshow(C, extent=[ix0,ix0+cw, iy0,iy0+ch], **ima)
# plt.suptitle('Galaxy conv PSF')
# ps.savefig()

if ix0 < 0:
    C = C[:,-ix0:]
    ix0 = 0
if iy0 < 0:
    C = C[-iy0:,:]
    iy0 = 0
ch,cw = C.shape

# plt.clf()
# plt.imshow(C, extent=[ix0,ix0+cw, iy0,iy0+ch], **ima)
# plt.suptitle('Galaxy conv PSF')
# ps.savefig()

ihalfsize = int(np.round(halfsize))
if cw > (gx + ihalfsize):
    C = C[:,:-(cw - (gx+ihalfsize))]
if ch > (gy + ihalfsize):
    C = C[:-(ch - (gy+ihalfsize)),:]
ch,cw = C.shape

plt.clf()
plt.imshow(C, extent=[ix0,ix0+cw, iy0,iy0+ch], **ima)
plt.suptitle('Galaxy conv PSF')
ps.savefig()

mh,mw = mod2.shape
C = C[:mh,:mw]
plt.clf()
plt.imshow(C-mod2, **ima)
plt.suptitle('Galaxy conv PSF - model')
ps.savefig()


u2 = gal.getUnitFluxModelPatch(tim)

plt.clf()
plt.subplot(2,3,1)
x0,y0,im = u1.x0, u1.y0, u1.patch
ph,pw = im.shape
print('u1:', im.sum())
plt.imshow(im, extent=[x0,x0+pw, y0,y0+ph], **ima)
plt.title('Gauss')
ax = plt.axis()
mx = im.max()

plt.subplot(2,3,4)
plt.imshow(np.log10(np.abs(im/mx)), vmin=-10, vmax=0, extent=[x0,x0+pw, y0,y0+ph], **ima)
plt.title('Gauss')
plt.axis(ax)

plt.subplot(2,3,2)
x0,y0,im = u2.x0, u2.y0, u2.patch
ph,pw = im.shape
print('u2:', im.sum())
plt.imshow(im, extent=[x0,x0+pw, y0,y0+ph], **ima)
plt.axis(ax)
plt.title('Fourier')

plt.subplot(2,3,5)
plt.imshow(np.log10(np.abs(im/mx)), vmin=-10, vmax=0, extent=[x0,x0+pw, y0,y0+ph], **ima)
plt.title('Fourier')
plt.axis(ax)

x0 = min(u1.x0, u2.x0)
y0 = min(u1.x0, u2.x0)
x1 = max(u1.x1, u2.x1)
y1 = max(u1.y1, u2.y1)
diff = np.zeros((y1-y0, x1-x0))
u1.addTo(diff)
u2.addTo(diff, scale=-1)
mx = np.abs(diff).max()
plt.subplot(2,3,3)
plt.imshow(diff, extent=[x0,x1,y0,y1], vmin=-mx, vmax=mx, **ima)
plt.axis(ax)
plt.title('diff: %.3g' % mx)
plt.suptitle('galaxy unit flux models')
ps.savefig()

tim.psf = psf1

print()
print()


#mx,my = 0.,0.
mx,my = gx,gy
gmog = gal._getAffineProfile(tim, mx, my)

w = np.fft.rfftfreq(W)
v = np.fft.fftfreq(H)
print('Frequencies:', len(w), 'x', len(v))

#print 'w', w
#print 'v', v
#ww,vv = np.meshgrid(w, v)
#print 'ww,vv', ww.shape, vv.shape

Fsum = None
for k in range(gmog.K):
    V = gmog.var[k,:,:]
    iv = np.linalg.inv(V)
    mu = gmog.mean[k,:]
    amp = gmog.amp[k]
    a,b,d = iv[0,0], iv[0,1], iv[1,1]

    mux = -(mu[0] - cx)
    muy = -(mu[1] - cy)
    print('mu', mux, muy)
    
    a *= 0.5
    b *= 0.5
    d *= 0.5
    
    det = a*d - b**2
    #F = (np.exp(-np.pi**2/det * (a * vv**2 + d*ww**2 - 2*b*vv*ww))
    #     * np.exp(2.*np.pi* 1j * (mux*ww + muy*vv)))
    F = (np.exp(-np.pi**2/det *
                (a * v[:,np.newaxis]**2 +
                 d * w[np.newaxis,:]**2 -
                 2*b*v[:,np.newaxis]*w[np.newaxis,:]))
         * np.exp(2.*np.pi* 1j * (mux*w[np.newaxis,:] + muy*v[:,np.newaxis])))
    
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

    if False:
        FF = np.fft.irfft2(Fsum, s=(H,W))
        plt.clf()
        plt.subplot(1,3,1)
        plt.imshow(FF, **ima)
        plt.subplot(1,3,2)
        plt.imshow(Fsum.real, **ima)
        plt.subplot(1,3,3)
        plt.imshow(Fsum.imag, **ima)
        ps.savefig()
        
print('abs max Fsum.imag:', np.abs(Fsum.imag).max())
    
IG = np.fft.irfft2(Fsum, s=(H,W))
print('IG', IG.shape)

plt.clf()
plt.subplot(1,3,1)
plt.imshow(IG, **ima)
plt.subplot(1,3,2)
plt.imshow(Fsum.real, **ima)
plt.subplot(1,3,3)
plt.imshow(Fsum.imag, **ima)
plt.suptitle('Fsum (galaxy)')
ps.savefig()

print('Fsum:', Fsum.shape)

P = np.fft.rfft2(psfim)
print('P:', P.shape)

CF = Fsum * P
C = np.fft.irfft2(CF, s=(H,W))
print('CF', CF.shape)
print('C', C.shape)

plt.clf()
plt.subplot(1,3,1)
plt.imshow(C, **ima)
plt.subplot(1,3,2)
plt.imshow(CF.real, **ima)
plt.subplot(1,3,3)
plt.imshow(CF.imag, **ima)
plt.suptitle('C (galaxy conv PSF)')
ps.savefig()

plt.clf()
plt.subplot(1,3,1)
plt.imshow(mod2, **ima)
plt.subplot(1,3,2)
plt.imshow(G2.real, **ima)
plt.subplot(1,3,3)
plt.imshow(G2.imag, **ima)
plt.suptitle('Gaussian model')
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


print('C', C.sum())
print('mod2:', mod2.sum())

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
log = np.log10(np.abs(resid)/mx)
#plt.imshow(np.where(np.sign(resid) > 0, log, -log), vmin=-6, vmax=6, **ima)
plt.imshow(log, vmin=-6, vmax=0, **ima)

ps.savefig()

print('Residual: max', mx)

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



