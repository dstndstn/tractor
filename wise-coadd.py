import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

from scipy.ndimage.morphology import binary_dilation

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from runslice import *
from astrometry.util.starutil_numpy import *

from tractor import *
from tractor.ttime import *



T = fits_table('wise_allsky_4band_p3as_cdd.fits')

ps = PlotSequence('co')

plt.clf()
plt.plot(T.ra, T.dec, 'r.', ms=4, alpha=0.5)
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.title('Atlas tile centers')
plt.axis([360,0,-90,90])
ps.savefig()

T.row = np.arange(len(T))

# W3
r0,r1 = 210.593,  219.132
d0,d1 =  51.1822,  54.1822

margin = 1.

T.cut((T.ra + margin > r0) *
      (T.ra - margin < r1) *
      (T.dec + margin > d0) *
      (T.dec - margin < d1))
print 'Cut to', len(T), 'near RA,Dec box'
#T.xyz = radectoxyz(T.ra, T.dec)

wisedir = 'wise-frames'
WISE = fits_table(os.path.join(wisedir, 'WISE-index-L1b.fits'))
print 'Read', len(WISE), 'WISE L1b frames'
#WISE.about()

print 'frame range', WISE.frame_num.min(), WISE.frame_num.max()

margin = 2.
WISE.cut((WISE.ra + margin > r0) *
         (WISE.ra - margin < r1) *
         (WISE.dec + margin > d0) *
         (WISE.dec - margin < d1))
print 'Cut to', len(WISE), 'near RA,Dec box'
#WISE.xyz = radectoxyz(WISE.ra, WISE.dec)

outdir = 'wise-coadds'

pixscale = 2.75 / 3600.
W,H = 2048, 2048

#from wise3 import get_l1b_file

band = 1
L = 3

mask_gz = True
unc_gz = True

for ti in T:
    print 'RA,Dec', ti.ra, ti.dec
    cowcs = Tan(ti.ra, ti.dec, (W+1)/2., (H+1)/2.,
                -pixscale, 0., 0., pixscale, W, H)

    copoly = np.array([cowcs.pixelxy2radec(x,y) for x,y in [(1,1), (W,1), (W,H), (1,H)]])
    print 'copoly', copoly

    margin = 2.
    WI = np.flatnonzero((degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec) < margin) *
                        WISE.band == band)
    print 'Found', len(WI), 'WISE frames in range'

    # zz = list([a+'%03i'%b for a,b in zip(WISE.scan_id[WI], WISE.frame_num[WI])])
    # nil,I = np.unique(zz, return_index=True)
    # print 'Found', len(I), 'unique scan/frames'
    # print nil
    # WI = WI[I]

    coimg = np.zeros((H,W))
    con   = np.zeros((H,W))
    cowimg = np.zeros((H,W))
    cow    = np.zeros((H,W))
    
    for nwi,wi in enumerate(WISE[WI]):
        print
        print (nwi+1), 'of', len(WI)
        scangrp = wi.scan_id[-2:]
        basefn = os.path.join(wisedir, 'p1bm_frm', scangrp, wi.scan_id,
                              '%03i' % wi.frame_num, '%s%03i-w%i' % (wi.scan_id, wi.frame_num, band))
        intfn  = basefn + '-int-1b.fits'
        uncfn  = basefn + '-unc-1b.fits'
        if unc_gz:
            uncfn = uncfn + '.gz'
        maskfn = basefn + '-msk-1b.fits'
        if mask_gz:
            maskfn = maskfn + '.gz'

        print 'intfn', intfn
        print 'uncfn', uncfn
        print 'maskfn', maskfn

        wcs = Sip(intfn)
        print 'Wcs:', wcs
        h,w = wcs.get_height(), wcs.get_width()
        poly = np.array([wcs.pixelxy2radec(x,y) for x,y in [(1,1), (w,1), (w,h), (1,h)]])
        if not polygons_intersect(copoly, poly):
            print 'Image does not intersect target'
            continue

        P = pyfits.open(intfn)
        img = P[0].data
        ihdr = P[0].header

        mask = pyfits.open(maskfn)[0].data
        unc = pyfits.open(uncfn)[0].data

        zp = ihdr['MAGZP']
        zpscale = NanoMaggies.zeropointToScale(zp)

        goodmask = ((mask & sum([1<<bit for bit in [0,1,2,3,4,5,6,7, 9,
                                                    10,11,12,13,14,15,16,17,18,
                                                    21,26,27,28]])) == 0)
        iv = np.zeros_like(img)
        iv[goodmask] = 1. / (unc[goodmask])**2

        # Patch masked pixels so we can interpolate
        (h,w) = img.shape
        patchmask = (iv > 0)
        patchimg = img.copy()
        Nlast = -1
        while True:
            I = np.flatnonzero(patchmask == 0)
            print len(I), 'pixels need patching'
            if len(I) == 0:
                break
            assert(len(I) != Nlast)
            Nlast = len(I)
            iy,ix = np.unravel_index(I, img.shape)
            psum = np.zeros(len(I), patchimg.dtype)
            pn = np.zeros(len(I), int)
            ok = (iy > 0)
            psum[ok] += (patchimg [iy[ok]-1, ix[ok]] *
                         patchmask[iy[ok]-1, ix[ok]])
            pn[ok] +=    patchmask[iy[ok]-1, ix[ok]]
            ok = (iy < (h-1))
            psum[ok] += (patchimg [iy[ok]+1, ix[ok]] *
                         patchmask[iy[ok]+1, ix[ok]])
            pn[ok] +=    patchmask[iy[ok]+1, ix[ok]]
            ok = (ix > 0)
            psum[ok] += (patchimg [iy[ok], ix[ok]-1] *
                         patchmask[iy[ok], ix[ok]-1])
            pn[ok] +=    patchmask[iy[ok], ix[ok]-1]
            ok = (ix < (w-1))
            psum[ok] += (patchimg [iy[ok], ix[ok]+1] *
                         patchmask[iy[ok], ix[ok]+1])
            pn[ok] +=    patchmask[iy[ok], ix[ok]+1]
            patchimg.flat[I] = (psum / np.maximum(pn, 1)).astype(patchimg.dtype)
            patchmask.flat[I] = (pn > 0)

        #name = 'WISE ' + ihdr['COADDID'] + ' W%i' % band
        Yo,Xo,Yi,Xi,rims = resample_with_wcs(cowcs, wcs, [patchimg], L)
        if Yo is None:
            continue
        rim = rims[0]

        riv = np.zeros_like(coimg)
        riv[Yo,Xo] = iv[Yi,Xi]
        
        coimg[Yo,Xo] += rim * zpscale
        con  [Yo,Xo] += 1

        cowimg[Yo,Xo] += rim * zpscale * riv[Yo,Xo]
        cow += riv

        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(coimg / np.maximum(con, 1), interpolation='nearest', origin='lower')
        plt.subplot(2,2,2)
        plt.imshow(cowimg / np.maximum(cow, 1e-16), interpolation='nearest', origin='lower')
        plt.subplot(2,2,3)
        plt.imshow(con, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.suptitle('%i images' % nwi)
        ps.savefig()

    coimg /= np.maximum(con, 1)
    cowimg /= np.maximum(cow, 1e-16)

    plt.clf()
    plt.imshow(coimg, interpolation='nearest', origin='lower')
    ps.savefig()
    
    plt.clf()
    plt.imshow(cowimg, interpolation='nearest', origin='lower')
    ps.savefig()
