import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from runslice import *


X = unpickle_from_file('eboss-s82-tst-r00-d00-w1.pickle')
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

ims = []

for ti,(tim,nil,nil) in zip(T,R):
    print ti.tag
    x0,x1,y0,y1 = ti.extents
    print ti.pixinroi, (x1-x0)*(y1-y0),
    I = np.flatnonzero(tim.invvar)
    print len(I), 'non-zero invvar',
    print '      ', (100. * len(I) / ti.pixinroi)

    #print 'wcs', tim.getWcs()
    wcs = tim.getWcs().wcs
    #print 'wcs', wcs
    print type(wcs)

    wcs2 = Sip(wcs)
    #print 'wcs copy', wcs2
    cpx,cpy = wcs2.crpix
    wcs2.set_crpix((cpx - x0, cpy - y0))
    h,w = tim.shape
    wcs2.set_width(w)
    wcs2.set_height(h)
    print 'wcs copy', wcs2
    
    yo,xo,yi,xi,nil = resample_with_wcs(targetwcs, wcs2, [],[])
    if yo is None:
        continue

    nnim = np.zeros((S,S))
    nnim[yo,xo] = tim.data[yi,xi]

    iv = np.zeros((S,S))
    iv[yo,xo] = tim.invvar[yi,xi]

    print 'tim.maskplane:', tim.maskplane.dtype, tim.maskplane.shape
    #I = np.flatnonzero(tim.invvar > 0)
    #M = tim.maskplane.flat[I]
    M = tim.maskplane.ravel()
    npix = len(M)
    print npix, 'mask pixels'
    for bit in range(32):
        I = np.flatnonzero(M & (1 << bit))
        if len(I) == 0:
            continue
        print '# with bit', bit, 'set:', len(I), ': %.1f %%' % (100. * len(I)/npix)

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

    Lorder = 3
    yo,xo,yi,xi,rpix = resample_with_wcs(targetwcs, wcs2, [patchimg], Lorder)
    if yo is None:
        continue

    rpatch = np.zeros((S,S))
    rpatch[yo,xo] = rpix[0]

    print 'Photocal:', tim.photocal
    print 'sig1:', tim.sigma1
    sig1 = tim.sigma1

    print 'Sky:', tim.sky
    sky = tim.getSky().getValue()

    rpatch = (rpatch - sky) * tim.getPhotoCal().getScale()
    w = (1. / sig1**2)

    ww = w * (iv > 0)

    lancsum  += (rpatch * ww)
    lancwsum += ww
    lancsum2 += ww * (rpatch **2)

    nnim = (nnim - sky) * tim.getPhotoCal().getScale()

    nnsum  += (nnim * ww)
    nnwsum += ww

    ims.append((nnim, rpatch, (iv>0), sig1 * tim.getPhotoCal().getScale(), tim.name))


nn = (nnsum / np.maximum(nnwsum, 1e-6))

lanczos = (lancsum / np.maximum(lancwsum, 1e-6))

sig = 1./np.sqrt(np.median(lancwsum[lancwsum > 0]))

lvar = lancsum2 / (np.maximum(lancwsum, 1e-6)) - lanczos**2
lstd = np.sqrt(lvar)

plt.figure(figsize=(8,8))

plt.clf()
plt.subplot(2,2,1)
plt.imshow(nn, interpolation='nearest', origin='lower',
           vmin=-2*sig, vmax=5*sig)
plt.colorbar()
plt.title('nearest neighbor')

plt.subplot(2,2,2)
plt.imshow(lanczos, interpolation='nearest', origin='lower',
           vmin=-2*sig, vmax=5*sig)
plt.colorbar()
plt.title('Lanczos3')

plt.subplot(2,2,4)
plt.imshow(lstd, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('std')

ps.savefig()


plt.clf()
plt.subplot(2,2,1)
plt.imshow(nn, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('nearest neighbor')

plt.subplot(2,2,2)
plt.imshow(lanczos, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('Lanczos3')

plt.subplot(2,2,4)
plt.imshow(lstd, interpolation='nearest', origin='lower')
plt.colorbar()
plt.title('std')

ps.savefig()



for (nnim, lancim, mask, sig1, name) in ims:

    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(nnim, interpolation='nearest', origin='lower',
               vmin=-2*sig1, vmax=5*sig1)
    plt.title('nearest neighbor')

    #plt.subplot(2,2,2)
    #plt.imshow(iv, interpolation='nearest', origin='lower')

    plt.subplot(2,2,2)
    plt.imshow(lancim, interpolation='nearest', origin='lower',
               vmin=-2*sig1, vmax=5*sig1)
    plt.title('Lanczos3')

    plt.subplot(2,2,3)
    plt.imshow((lancim - lanczos) * mask, interpolation='nearest', origin='lower',
               vmin=-3*sig1, vmax=3*sig1, cmap='gray')
    plt.title('Lanczos3 - avg')

    rchi2 = (np.sum(((lancim - lanczos) * mask / np.maximum(lstd, 1e-6)) ** 2) /
             np.sum(mask))

    plt.subplot(2,2,4)
    plt.imshow((lancim - lanczos) * mask / np.maximum(lstd, 1e-6), interpolation='nearest', origin='lower',
               vmin=-5, vmax=5, cmap='gray')
    plt.title('rchi2(Lanczos3): %.2f' % rchi2)

    plt.suptitle(name)

    ps.savefig()

