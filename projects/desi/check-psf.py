import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import sys
import os

import fitsio

from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence, dimshow

from astrometry.util.util import *

from tractor import *

from common import *

if __name__ == '__main__':
    decals = Decals()

    ps = PlotSequence('psf')

    #B = decals.get_bricks()

    T = decals.get_ccds()
    T.cut(T.extname == 'S1')
    print 'Cut to', len(T)
    #print 'Expnums:', T[:10]
    T.cut(T.expnum == 348233)
    print 'Cut to', len(T)
    T.about()

    band = T.filter[0]
    print 'Band:', band

    im = DecamImage(T[0])
    print 'Reading', im.imgfn

    # Get approximate image center for astrometry
    hdr = im.read_image_header()
    wcs = Tan(hdr['CRVAL1'], hdr['CRVAL2'], hdr['CRPIX1'], hdr['CRPIX2'],
              hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2'],
              hdr['NAXIS1'], hdr['NAXIS2'])
    print 'WCS:', wcs
    r,d = wcs.pixelxy2radec(wcs.imagew/2, wcs.imageh/2)

    pixscale = wcs.pixel_scale()/3600.
    run_calibs(im, r, d, pixscale, astrom=True, morph=False, se2=False)

    iminfo = im.get_image_info()
    print 'img:', iminfo
    H,W = iminfo['dims']
    psfex = PsfEx(im.psffn, W, H, nx=6)

    S = im.read_sdss()
    print len(S), 'SDSS sources'
    S.cut(S.objc_type == 6)
    print len(S), 'SDSS stars'
    S.flux = S.get('%s_psfflux' % band)

    wcs = im.read_wcs()

    img = im.read_image()
    sky = np.median(img)
    img -= sky
    invvar = im.read_invvar(clip=True)
    sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
    sigoff = 3
    img += sig1 * sigoff
    # convert to sigmas
    img /= sig1
    invvar *= sig1**2

    H,W = img.shape
    sz = 20
    
    ok,S.x,S.y = wcs.radec2pixelxy(S.ra, S.dec)
    S.x -= 1
    S.y -= 1
    S.ix, S.iy = np.round(S.x).astype(int), np.round(S.y).astype(int)
    S.cut((S.ix >= sz) * (S.iy >= sz) * (S.ix < W-sz) * (S.iy < H-sz))
    print len(S), 'SDSS stars in bounds'

    S.cut(invvar[S.iy, S.ix] > 0)
    print len(S), 'SDSS stars not in masked regions'
    
    rows,cols = 4,5
    plt.clf()
    for i,sdssi in enumerate(np.argsort(-S.flux)):
        if i >= rows*cols:
            break
        s = S[sdssi]
        plt.subplot(rows, cols, 1+i)
        subimg = img[s.iy - sz : s.iy + sz+1, s.ix - sz : s.ix + sz+1]
        dimshow(subimg, ticks=False)
    ps.savefig()
    
    maxes = []
    plt.clf()
    for i,sdssi in enumerate(np.argsort(-S.flux)):
        if i >= rows*cols:
            break
        s = S[sdssi]
        plt.subplot(rows, cols, 1+i)
        subimg = img[s.iy - sz : s.iy + sz+1, s.ix - sz : s.ix + sz+1]
        mx = subimg.max()
        maxes.append(mx)
        logmx = np.log10(mx)
        dimshow(np.log10(np.maximum(subimg, mx*1e-16)), vmin=0, vmax=logmx,
                ticks=False)
    ps.savefig()

    plt.clf()
    for i,sdssi in enumerate(np.argsort(-S.flux)):
        if i >= rows*cols:
            break
        s = S[sdssi]
        plt.subplot(rows, cols, 1+i)

        subimg = img[s.iy - sz : s.iy + sz+1, s.ix - sz : s.ix + sz+1]
        ss = 5
        flux = np.sum(img[s.iy-ss:s.iy+ss+1, s.ix-ss:s.ix+ss+1])
        # subtract off the 3*sig we added
        flux -= (2*ss+1)**2 * sigoff

        psfimg = psfex.instantiateAt(s.x, s.y)
        psfimg = psfimg * flux + sigoff
        
        mx = maxes[i]
        #mx = psfimg.max()
        logmx = np.log10(mx)
        dimshow(np.log10(np.maximum(psfimg, mx*1e-16)), vmin=0, vmax=logmx,
                ticks=False)
    ps.savefig()

    
