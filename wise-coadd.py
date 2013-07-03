import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

from scipy.ndimage.morphology import binary_dilation

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from runslice import *

from tractor import *
from tractor.ttime import *


#X = unpickle_from_file('eboss-s82-tst-r00-d00-w1.pickle')
#X = unpickle_from_file('ebossw3-tst-r47-d00-w1.pickle')
X = unpickle_from_file('eboss-w3-tst-r48-d00-w1.pickle')
wfn = 'wise-objs-w3.fits'


r0,r1,d0,d1 = X['ralo'],X['rahi'],X['declo'],X['dechi']
ra  = (r0+r1) / 2.
dec = (d0+d1) / 2.
T = X['T']
print 'T'
T.about()
R = X['res1']
Res1 = X['res1']
sdss = X['S']
cat = X['cat']
bandnum = X['bandnum']

print 'SDSS sources:', len(sdss)
#sdss.about()

ps = PlotSequence('co')

S = 100
pixscale = 2.75 / 3600.
targetwcs = Tan(ra, dec, (S+1)/2., (S+1)/2.,
                pixscale, 0., 0., pixscale,
                S, S)
print 'Target WCS:', targetwcs

class Duck(object):
    pass

def _resample_one((ti, tim, mod)):
    print ti.tag, tim.name
    x0,x1,y0,y1 = ti.extents

    tim.setInvvar(tim.invvar)

    # Create sub-WCS
    wcs = tim.getWcs().wcs
    wcs2 = Sip(wcs)
    cpx,cpy = wcs2.crpix
    wcs2.set_crpix((cpx - x0, cpy - y0))
    h,w = tim.shape
    wcs2.set_width(w)
    wcs2.set_height(h)
    #print 'wcs copy', wcs2
    
    yo,xo,yi,xi,nil = resample_with_wcs(targetwcs, wcs2, [],[])
    if yo is None:
        return None

    nnim = np.zeros((S,S))
    nnim[yo,xo] = tim.data[yi,xi]

    iv = np.zeros((S,S))
    iv[yo,xo] = tim.invvar[yi,xi]

    # Patch masked pixels so we can interpolate
    patchmask = (tim.invvar > 0)
    patchimg = tim.data.copy()
    rdmask = tim.rdmask
    Nlast = -1
    while True:
        I = np.flatnonzero(rdmask * (patchmask == 0))
        print len(I), 'pixels need patching'
        if len(I) == 0:
            break
        assert(len(I) != Nlast)
        Nlast = len(I)
        iy,ix = np.unravel_index(I, tim.data.shape)
        psum = np.zeros(len(I), patchimg.dtype)
        pn = np.zeros(len(I), int)

        ok = (iy > 0)
        psum[ok] += patchimg[iy[ok]-1, ix[ok]]
        pn[ok] += 1

        ok = (iy < (h-1))
        psum[ok] += patchimg[iy[ok]+1, ix[ok]]
        pn[ok] += 1

        ok = (ix > 0)
        psum[ok] += patchimg[iy[ok], ix[ok]-1]
        pn[ok] += 1

        ok = (ix < (w-1))
        psum[ok] += patchimg[iy[ok], ix[ok]+1]
        pn[ok] += 1

        patchimg.flat[I] = (psum / np.maximum(pn, 1)).astype(patchimg.dtype)
        patchmask.flat[I] = (pn > 0)

    # Resample
    Lorder = 3
    yo,xo,yi,xi,rpix = resample_with_wcs(targetwcs, wcs2, [patchimg, mod], Lorder)
    if yo is None:
        return None

    rpatch = np.zeros((S,S))
    rpatch[yo,xo] = rpix[0]

    rmod = np.zeros((S,S))
    rmod[yo,xo] = rpix[1]

    print 'sig1:', tim.sigma1
    sig1 = tim.sigma1
    sky = tim.getSky().getValue()
    scale = tim.getPhotoCal().getScale()
    sig1 = sig1 * scale
    print 'scale', scale, 'scaled sig1:', sig1
    w = (1. / sig1**2)
    ww = w * (iv > 0)


    d = Duck()

    d.nnimg = (nnim   - sky) * scale
    d.rimg  = (rpatch - sky) * scale
    d.rmod  = (rmod   - sky) * scale
    d.ww = ww
    d.mask = (iv > 0)
    d.sig1 = sig1
    d.name = tim.name
    d.mod = mod
    d.img = tim.data
    d.invvar = tim.invvar
    d.sky = sky
    d.scale = scale
    d.lnp1 = np.sum(((mod - tim.getImage()) * tim.getInvError())**2)
    d.npix1 = np.sum(tim.getInvError() > 0)
    d.lnp2 = np.sum(((rmod - rpatch)**2 * iv))
    d.npix2 = np.sum(iv > 0)
    
    return d



def _resample_mod((ti, tim, mod)):
    print ti.tag, tim.name
    x0,x1,y0,y1 = ti.extents
    # Create sub-WCS
    wcs = tim.getWcs().wcs
    wcs2 = Sip(wcs)
    cpx,cpy = wcs2.crpix
    wcs2.set_crpix((cpx - x0, cpy - y0))
    h,w = tim.shape
    wcs2.set_width(w)
    wcs2.set_height(h)
    # Resample
    Lorder = 3
    yo,xo,yi,xi,rpix = resample_with_wcs(targetwcs, wcs2, [mod], Lorder)
    if yo is None:
        return None
    rmod = np.zeros((S,S))
    rmod[yo,xo] = rpix[0]

    iv = np.zeros((S,S))
    iv[yo,xo] = tim.invvar[yi,xi]
    sig1 = tim.sigma1
    w = (1. / sig1**2)
    ww = w * (iv > 0)

    return rmod,ww


def _rev_resample_mask((ti, tim, mask)):
    print ti.tag, tim.name
    x0,x1,y0,y1 = ti.extents
    # Create sub-WCS
    wcs = tim.getWcs().wcs
    wcs2 = Sip(wcs)
    cpx,cpy = wcs2.crpix
    wcs2.set_crpix((cpx - x0, cpy - y0))
    h,w = tim.shape
    wcs2.set_width(w)
    wcs2.set_height(h)
    yo,xo,yi,xi,nil = resample_with_wcs(wcs2, targetwcs, [],[])
    if yo is None:
        return None
    rmask = np.zeros((h,w))
    rmask[yo,xo] = mask[yi, xi]
    return rmask

#ims.append((nnim, rpatch, (iv>0), sig1, tim.name, mod, rmod, tim.data, sky, tim.invvar, scale))


args = []
for i,(ti,(tim,mod,nil)) in enumerate(zip(T,R)):
    #if i == 9:
    #    break
    args.append((ti, tim, mod))

mp = multiproc(16)
ims = mp.map(_resample_one, args)

nnsum    = np.zeros((S,S))
lancsum  = np.zeros((S,S))
lancsum2 = np.zeros((S,S))
modsum   = np.zeros((S,S))
wsum     = np.zeros((S,S))

lnp1 = 0.
lnp2 = 0.
npix1 = 0
npix2 = 0

for d in ims:
    nnsum    += (d.nnimg   * d.ww)
    lancsum  += (d.rimg    * d.ww)
    lancsum2 += (d.rimg**2 * d.ww)
    modsum   += (d.rmod    * d.ww)
    wsum     += d.ww
    lnp1  += d.lnp1
    npix1 += d.npix1
    lnp2  += d.lnp2
    npix2 += d.npix2

nnimg   = (nnsum   / np.maximum(wsum, 1e-6))
coimg   = (lancsum / np.maximum(wsum, 1e-6))
comod   = (modsum  / np.maximum(wsum, 1e-6))
coinvvar = wsum
cochi = (coimg - comod) * np.sqrt(coinvvar)

sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
print 'Coadd sig:', sig

lnp3 = np.sum(cochi**2)
npix3 = np.sum(coinvvar > 0)

print 'lnp1 (orig)  ', lnp1
print '         npix', npix1
print 'lnp2 (resamp)', lnp2
print '         npix', npix2
print 'lnp3 (coadd) ', lnp3
print '         npix', npix3

lvar = lancsum2 / (np.maximum(wsum, 1e-6)) - coimg**2
coppstd = np.sqrt(lvar)

plt.figure(figsize=(8,8))

ima = dict(interpolation='nearest', origin='lower',
           vmin=-2*sig, vmax=10*sig)


R,C = 2,3

plt.clf()
plt.suptitle('First-round Coadds')
plt.subplot(R,C,1)
plt.imshow(nnimg, **ima)
#plt.colorbar()
plt.title('NN data')
plt.subplot(R,C,2)
plt.imshow(coimg, **ima)
#plt.colorbar()
plt.title('Data')
plt.subplot(R,C,3)
plt.imshow(comod, **ima)
#plt.colorbar()
plt.title('Model')
plt.subplot(R,C,4)
plt.imshow(coppstd, interpolation='nearest', origin='lower')
#plt.colorbar()
plt.title('Coadd std')
plt.subplot(R,C,5)
plt.imshow(cochi, interpolation='nearest', origin='lower',
           vmin=-5., vmax=5.)
#plt.colorbar()
plt.title('Chi')
plt.subplot(R,C,6)
plt.imshow(cochi, interpolation='nearest', origin='lower',
           vmin=-20., vmax=20.)
#plt.colorbar()
plt.title('Chi (b)')
ps.savefig()


lancsum  = np.zeros((S,S))
wsum     = np.zeros_like(lancsum)
nnsum    = np.zeros_like(lancsum)
lancsum2 = np.zeros_like(lancsum)
modsum   = np.zeros((S,S))

rchis = []

for d in ims:
    rchi = (d.rimg - coimg) * d.mask / np.maximum(coppstd, 1e-6)
    badpix = (np.abs(rchi) >= 5.)
    # grow by a small margin
    badpix = binary_dilation(badpix)
    notbad = np.logical_not(badpix)
    rchis.append(rchi)
    d.mask *= notbad
    w = (1. / d.sig1**2)
    ww = w * d.mask
    # update d.ww?
    nnsum    += (d.nnimg   * ww)
    lancsum  += (d.rimg    * ww)
    lancsum2 += (d.rimg**2 * ww)
    modsum   += (d.rmod    * ww)
    wsum     += ww

nn    = (nnsum   / np.maximum(wsum, 1e-6))
coimg = (lancsum / np.maximum(wsum, 1e-6))
comod = (modsum  / np.maximum(wsum, 1e-6))
coinvvar = wsum
cochi = (coimg - comod) * np.sqrt(coinvvar)

print 'Second-round coadd:'
sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
print 'Coadd sig:', sig
    
lnp3 = np.sum(cochi**2)
npix3 = np.sum(coinvvar > 0)
print 'lnp3 (coadd) ', lnp3
print '         npix', npix3

# per-pixel variance
lvar = lancsum2 / (np.maximum(wsum, 1e-6)) - coimg**2
coppstd = np.sqrt(lvar)

ima = dict(interpolation='nearest', origin='lower',
           vmin=-2*sig, vmax=10*sig)

R,C = 2,3

plt.clf()
plt.suptitle('Second-round Coadds')
plt.subplot(R,C,1)
plt.imshow(nn, **ima)
# plt.colorbar()
plt.title('NN data')
plt.subplot(R,C,2)
plt.imshow(coimg, **ima)
# plt.colorbar()
plt.title('Data')
plt.subplot(R,C,3)
plt.imshow(comod, **ima)
# plt.colorbar()
plt.title('Model')
plt.subplot(R,C,4)
plt.imshow(coppstd, interpolation='nearest', origin='lower')
# plt.colorbar()
plt.title('Coadd std')
plt.subplot(R,C,5)
plt.imshow(cochi, interpolation='nearest', origin='lower',
           vmin=-5., vmax=5.)
# plt.colorbar()
plt.title('Chi')
plt.subplot(R,C,6)
plt.imshow(cochi, interpolation='nearest', origin='lower',
           vmin=-20., vmax=20.)
# plt.colorbar()
plt.title('Chi (b)')
ps.savefig()




W = fits_table(wfn) #, columns=['ra','dec'])
print 'Read', len(W), 'from', wfn
W.cut((W.ra > r0) * (W.ra < r1) * (W.dec > d0) * (W.dec < d1))
print 'Cut to', len(W), 'in RA,Dec box'

# Find WISE objs with no SDSS counterpart
I,J,d = match_radec(W.ra, W.dec, sdss.ra, sdss.dec, 4./3600.)
unmatched = np.ones(len(W), bool)
unmatched[I] = False
UW = W[unmatched]

# Plot SDSS objects and WISE objects on residual image
plt.clf()
plt.imshow(cochi, interpolation='nearest', origin='lower',
           vmin=-10., vmax=10., cmap='gray')
ax = plt.axis()
oxy = [targetwcs.radec2pixelxy(r,d) for r,d in zip(W.ra, W.dec)]
X = np.array([x for ok,x,y in oxy])
Y = np.array([y for ok,x,y in oxy])
p1 = plt.plot(X-1, Y-1, 'r+')
oxy = [targetwcs.radec2pixelxy(r,d) for r,d in zip(sdss.ra, sdss.dec)]
X = np.array([x for ok,x,y in oxy])
Y = np.array([y for ok,x,y in oxy])
p2 = plt.plot(X-1, Y-1, 'bx')

oxy = [targetwcs.radec2pixelxy(r,d) for r,d in zip(UW.ra, UW.dec)]
X = np.array([x for ok,x,y in oxy])
Y = np.array([y for ok,x,y in oxy])
p3 = plt.plot(X-1, Y-1, 'r+', lw=2, ms=12)

plt.axis(ax)
plt.legend((p1,p2,p3), ('WISE', 'SDSS', 'WISE-only'))
ps.savefig()

plt.clf()
plt.imshow(coppstd, interpolation='nearest', origin='lower')
#           vmin=0, vmax=5.*sig)
plt.title('Coadd per-pixel std')
ps.savefig()




# 1. Create tractor PointSource objects for each WISE-only object

print 'Tractor catalog:'
for src in cat:
    print '  ', src

band = 'W%i' % bandnum

wcat = []
for i in range(len(UW)):
    mag = UW.get('w%impro' % bandnum)[i]
    nm = NanoMaggies.magToNanomaggies(mag)
    src = PointSource(RaDecPos(UW.ra[i], UW.dec[i]),
                      NanoMaggies(**{band: nm}))
    wcat.append(src)

# 2. Apply rchi masks to individual images
R = Res1
args = []
for i,(ti,(tim,mod,nil),d) in enumerate(zip(T,R,ims)):
    args.append((ti, tim, d.mask))
rmasks = mp.map(_rev_resample_mask, args)
tims = []
for mask,(tim,nil,nil) in zip(rmasks, R):
    tims.append(tim)
    tim.setInvvar(tim.invvar * mask)

# 3. Re-run forced photometry on individual images
srcs = [src for src in cat] + wcat
tr = Tractor(tims, srcs)
print 'Created Tractor:', tr

tr.freezeParamsRecursive('*')
tr.thawPathsTo('sky')
tr.thawPathsTo(band)

minsb = 0.005
minFlux = None

t0 = Time()
ims0,ims1,IV,fs = tr.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
                                                sky=True, minFlux=minFlux,
                                                fitstats=True,
                                                variance=True)
print 'Forced phot took', Time()-t0

args = []
for i,(ti,(tim,mod,nil)) in enumerate(zip(T,R)):
    args.append((ti, tim, mod))
mims = mp.map(_resample_mod, args)

modsum2 = np.zeros((S,S))
wsum2 = np.zeros((S,S))
for rmod,ww in mims:
    modsum2 += (rmod * ww)
    wsum2   += ww
comod2 = modsum2 / np.maximum(wsum2, 1e-12)
cochi2 = (coimg - comod2) * np.sqrt(coinvvar)

plt.clf()
plt.imshow(cochi, interpolation='nearest', origin='lower',
           vmin=-10., vmax=10., cmap='gray')
plt.title('Chi (before)')
plt.savefig()

plt.clf()
plt.imshow(cochi2, interpolation='nearest', origin='lower',
           vmin=-10., vmax=10., cmap='gray')
plt.title('Chi (after)')
plt.savefig()

plt.clf()
plt.imshow(comod, **ima)
plt.title('Model (before)')
plt.savefig()

plt.clf()
plt.imshow(comod2, **ima)
plt.title('Model (after)')
plt.savefig()



# 4. Run forced photometry on coadd






for i,d in enumerate(ims):
    sig1 = d.sig1

    R,C = 2,3

    plt.clf()
    plt.subplot(R,C,1)
    plt.imshow(d.rimg, interpolation='nearest', origin='lower',
               vmin=-2*sig1, vmax=5*sig1)
    plt.title('data')
    
    plt.subplot(R,C,2)
    plt.imshow(d.rmod, interpolation='nearest', origin='lower',
               vmin=-2*sig1, vmax=5*sig1)
    plt.title('mod')
    
    plt.subplot(R,C,3)
    chi = (d.rimg - d.rmod) * d.mask / sig1
    plt.imshow(chi, interpolation='nearest', origin='lower',
               vmin=-5, vmax=5, cmap='gray')
    plt.title('chi: %.1f' % np.sum(chi**2))

    # grab original rchi
    rchi = rchis[i]
    
    rchi2 = np.sum(rchi**2) / np.sum(d.mask)

    plt.subplot(R,C,4)
    plt.imshow(rchi, interpolation='nearest', origin='lower',
               vmin=-5, vmax=5, cmap='gray')
    plt.title('rchi2 vs coadd: %.2f' % rchi2)

    plt.subplot(R,C,5)
    plt.imshow(np.abs(rchi) > 5, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('rchi > 5')

    plt.subplot(R,C,6)
    plt.imshow(d.mask, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('mask')

    plt.suptitle(d.name)
    ps.savefig()
