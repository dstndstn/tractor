import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from runslice import *


#X = unpickle_from_file('eboss-s82-tst-r00-d00-w1.pickle')
#X = unpickle_from_file('ebossw3-tst-r47-d00-w1.pickle')
X = unpickle_from_file('eboss-w3-tst-r48-d00-w1.pickle')

r0,r1,d0,d1 = X['ralo'],X['rahi'],X['declo'],X['dechi']
ra  = (r0+r1) / 2.
dec = (d0+d1) / 2.
T = X['T']
print 'T'
T.about()
R = X['res1']

ps = PlotSequence('co')

S = 100
pixscale = 2.75 / 3600.
targetwcs = Tan(ra, dec, (S+1)/2., (S+1)/2.,
                pixscale, 0., 0., pixscale,
                S, S)
print 'Target WCS:', targetwcs

lancsum  = np.zeros((S,S))
lancwsum = np.zeros_like(lancsum)
nnsum    = np.zeros_like(lancsum)
nnwsum   = np.zeros_like(lancsum)
lancsum2 = np.zeros_like(lancsum)
modsum  = np.zeros((S,S))

ims = []

lnp1 = 0.
lnp2 = 0.
npix1 = 0
npix2 = 0




for i,(ti,(tim,mod,nil)) in enumerate(zip(T,R)):

    #if i == 9:
    #    break
    
    print ti.tag, tim.name
    x0,x1,y0,y1 = ti.extents

    tim.setInvvar(tim.invvar)
    lnp1 += np.sum(((mod - tim.getImage()) * tim.getInvError())**2)
    npix1 += np.sum(tim.getInvError() > 0)

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
        continue

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
        continue

    rpatch = np.zeros((S,S))
    rpatch[yo,xo] = rpix[0]

    rmod = np.zeros((S,S))
    rmod[yo,xo] = rpix[1]

    lnp2 += np.sum(((rmod - rpatch)**2 * iv))
    npix2 += np.sum(iv > 0)

    #print 'Photocal:', tim.photocal
    print 'sig1:', tim.sigma1
    sig1 = tim.sigma1

    #print 'Sky:', tim.sky
    sky = tim.getSky().getValue()

    scale = tim.getPhotoCal().getScale()
    sig1 = sig1 * scale
    print 'scale', scale, 'scaled sig1:', sig1
    rpatch = (rpatch - sky) * scale
    rmod = (rmod - sky) * scale
    nnim = (nnim - sky) * scale
    w = (1. / sig1**2)
    ww = w * (iv > 0)

    lancsum  += (rpatch * ww)
    lancwsum += ww
    lancsum2 += ww * (rpatch **2)

    modsum += (rmod * ww)

    nnsum  += (nnim * ww)
    nnwsum += ww

    ims.append((nnim, rpatch, (iv>0), sig1, tim.name, mod, rmod, tim.data, sky, tim.invvar, scale))


nn      = (nnsum   / np.maximum(nnwsum,   1e-6))
lanczos = (lancsum / np.maximum(lancwsum, 1e-6))
model   = (modsum  / np.maximum(lancwsum, 1e-6))

sig = 1./np.sqrt(np.median(lancwsum[lancwsum > 0]))
print 'Coadd sig:', sig

lnp3 = np.sum((model - lanczos)**2 * lancwsum)
npix3 = np.sum(lancwsum > 0)

print 'lnp1 (orig)  ', lnp1
print '         npix', npix1
print 'lnp2 (resamp)', lnp2
print '         npix', npix2
print 'lnp3 (coadd) ', lnp3
print '         npix', npix3

lvar = lancsum2 / (np.maximum(lancwsum, 1e-6)) - lanczos**2
lstd = np.sqrt(lvar)

plt.figure(figsize=(8,8))

ima = dict(interpolation='nearest', origin='lower',
           vmin=-2*sig, vmax=10*sig)

plt.clf()
plt.suptitle('Coadds')
plt.subplot(2,2,1)
plt.imshow(nn, **ima)
plt.colorbar()
plt.title('nearest neighbor')

plt.subplot(2,2,2)
plt.imshow(lanczos, **ima)
plt.colorbar()
plt.title('Lanczos3')

plt.subplot(2,2,3)
plt.imshow(model, **ima)
plt.colorbar()
plt.title('Model')

plt.subplot(2,2,4)
plt.imshow(lstd, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('std')
ps.savefig()



plt.clf()

plt.subplot(2,2,1)
plt.imshow(lanczos, **ima)
plt.colorbar()
plt.title('Data (Lanczos3)')

plt.subplot(2,2,2)
plt.imshow(model, **ima)
plt.colorbar()
plt.title('Model')

plt.subplot(2,2,3)
plt.imshow((lanczos - model) * np.sqrt(lancwsum), interpolation='nearest', origin='lower',
           vmin=-5., vmax=5.)
plt.colorbar()
plt.title('Chi')

plt.subplot(2,2,4)
plt.imshow((lanczos - model) * np.sqrt(lancwsum), interpolation='nearest', origin='lower',
           vmin=-20., vmax=20.)
plt.colorbar()
plt.title('Chi (b)')

ps.savefig()


for (nnim, lancim, mask, sig1, name, mod, rmod, img, sky, oiv, scale) in ims:

    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(lancim, interpolation='nearest', origin='lower',
               vmin=-2*sig1, vmax=5*sig1)
    plt.title('data')
    
    plt.subplot(2,2,2)
    plt.imshow(rmod, interpolation='nearest', origin='lower',
               vmin=-2*sig1, vmax=5*sig1)
    plt.title('mod')
    
    plt.subplot(2,2,3)
    chi = (lancim - rmod) * mask / sig1
    plt.imshow(chi, interpolation='nearest', origin='lower',
               vmin=-5, vmax=5, cmap='gray')
    plt.title('chi: %.1f' % np.sum(chi**2))

    rchi2 = (np.sum(((lancim - lanczos) * mask / np.maximum(lstd, 1e-6)) ** 2) /
             np.sum(mask))

    plt.subplot(2,2,4)
    plt.imshow((lancim - lanczos) * mask / np.maximum(lstd, 1e-6), interpolation='nearest', origin='lower',
               vmin=-5, vmax=5, cmap='gray')
    plt.title('rchi2 vs coadd: %.2f' % rchi2)

    plt.suptitle(name)
    ps.savefig()
