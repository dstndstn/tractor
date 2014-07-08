import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *

import fitsio

from tractor.basics import NanoMaggies

from scipy.ndimage.filters import gaussian_filter

from decam import sky_subtract

ps = PlotSequence('det')

def get_rgb_image(g, r, z,
                  alpha = 1.5
                  ):
    scale_g = 2.5
    scale_r = 1.0
    #scale_z = 0.7
    scale_z = 0.5
    #m = -0.02
    m = 0.0

    # Watch the ordering here -- "r" aliasing!
    b = np.maximum(0, g * scale_g - m)
    g = np.maximum(0, r * scale_r - m)
    r = np.maximum(0, z * scale_z - m)
    I = (r+g+b)/3.

    Q = 20
    m2 = 0.
    fI = np.arcsinh(alpha * Q * (I - m2)) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    R = fI * r / I
    G = fI * g / I
    B = fI * b / I
    maxrgb = reduce(np.maximum, [R,G,B])
    J = (maxrgb > 1.)
    R[J] = R[J]/maxrgb[J]
    G[J] = G[J]/maxrgb[J]
    B[J] = B[J]/maxrgb[J]
    return np.clip(np.dstack([R,G,B]), 0., 1.)

def _det_one((cowcs, fn, wcsfn)):
    print 'Image', fn
    F = fitsio.FITS(fn)[0]
    imginf = F.get_info()
    hdr = F.read_header()
    H,W = imginf['dims']

    wcs = Sip(wcsfn)
    depix = wcs.pixel_scale()
    seeing = hdr['SEEING']
    print 'Seeing', seeing
    psf_sigma = seeing / depix / 2.35
    print 'Sigma:', psf_sigma
    psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)

    if False:
        # Compare PSF models with Peter's "SEEING" card
        # (units: arcsec FWHM)
        from tractor.psfex import *
        psffn = fn.replace('.p.w.fits', '.p.w.cat.psf')
        print 'PSF', psffn
        psfmod = PsfEx(psffn, W, H)
        psfim = psfmod.instantiateAt(W/2, H/2)
        print 'psfim', psfim.shape
        ph,pw = psfim.shape
        plt.clf()
        plt.plot(psfim[ph/2,:], 'r-')
        plt.plot(psfim[:,pw/2], 'b-')
        xx = np.arange(pw)
        cc = pw/2
        G = np.exp(-0.5 * (xx - cc)**2 / sig**2)
        plt.plot(G * sum(psfim[:,pw/2]) / sum(G), 'k-')
        ps.savefig()

    # Read full image and estimate noise
    img = F.read()
    print 'Image', img.shape

    # Estimate and subtract background
    bg = sky_subtract(img, 512, gradient=False)
    img -= bg
    
    diffs = img[:-5:10,:-5:10] - img[5::10,5::10]
    mad = np.median(np.abs(diffs).ravel())
    sig1 = 1.4826 * mad / np.sqrt(2.)
    print 'Per-pixel noise estimate:', sig1

    # Read bad pixel mask
    maskfn = fn.replace('.p.w.fits', '.p.w.bpm.fits')
    mask = fitsio.read(maskfn)
    # FIXME -- patch image?
    img[mask != 0] = 0.

    # Get image zeropoint
    for zpkey in ['MAG_ZP', 'UB1_ZP']:
        zp = hdr.get(zpkey, None)
        if zp is not None:
            break
    zpscale = NanoMaggies.zeropointToScale(zp)
    # Scale image to nanomaggies
    img /= zpscale
    sig1 /= zpscale

    # Produce detection map
    detmap = gaussian_filter(img, psf_sigma) / psfnorm**2
    detmap_sig1 = sig1 / psfnorm
    print 'Detection map sig1', detmap_sig1

    # Lanczos resample
    L = 3
    try:
        Yo,Xo,Yi,Xi,[rim] = resample_with_wcs(cowcs, wcs, [detmap], L)
    except OverlapError:
        return None

    detmap_iv = (mask[Yo,Xo] == 0) * 1./detmap_sig1**2
    #detmap_iv = 1./detmap_sig1**2

    return Yo,Xo,rim,detmap_iv
    


def main():
    mp = multiproc(8)
    #mp = multiproc()

    wcsfn = 'unwise/352/3524p000/unwise-3524p000-w1-img-m.fits'
    wcs = Tan(wcsfn)
    W,H = wcs.get_width(), wcs.get_height()

    bounds = wcs.radec_bounds()
    print 'RA,Dec bounds', bounds
    print 'pixscale', wcs.pixel_scale()

    # Tweak to DECam pixel scale and number of pixels.
    depix = 0.27
    D = int(np.ceil((W * wcs.pixel_scale() / depix) / 4)) * 4
    DW,DH = D,D
    wcs.set_crpix(DW/2 + 1.5, DH/2 + 1.5)
    pixscale = depix / 3600.
    wcs.set_cd(-pixscale, 0., 0., pixscale)
    wcs.set_imagesize(DW, DH)
    W,H = wcs.get_width(), wcs.get_height()

    print 'Detmap patch size:', wcs.get_width(), wcs.get_height()
    bounds = wcs.radec_bounds()
    print 'RA,Dec bounds', bounds
    print 'pixscale', wcs.pixel_scale()

    # 1/4 x 1/4 subimage
    nsub = 4
    subx, suby = 0, 3
    chips = [30, 29, 24, 23, 22, 21, 43, 42]

    # 1/16 x 1/16 for faster testing
    nsub = 16
    subx, suby = 0, 12
    chips = [22, 43]

    subw, subh = W/nsub, H/nsub
    subwcs = Tan(wcs)
    subwcs.set_crpix(wcs.crpix[0] - subx*subw, wcs.crpix[1] - suby*subh)
    subwcs.set_imagesize(subw, subh)

    bounds = subwcs.radec_bounds()
    print 'RA,Dec bounds', bounds
    print 'Sub-image patch size:', subwcs.get_width(), subwcs.get_height()

    cowcs = subwcs

    # Insert imaging database here...
    paths = []
    for dirpath,dirs,files in os.walk('data/desi', followlinks=True):
        for fn in files:
            path = os.path.join(dirpath, fn)
            if path.endswith('.p.w.fits'):
                if any([('/C%02i/' % chip) in path for chip in chips]):
                    paths.append(path)
    print 'Found', len(paths), 'images'

    coadds = []

    for band in ['g','r','z']:

        print
        print 'Band', band
        fns = [fn for fn in paths if '%sband' % band in fn]
        #print 'Found', fns

        coH,coW = cowcs.get_height(), cowcs.get_width()
        coadd = np.zeros((coH,coW))
        coadd_iv = np.zeros((coH,coW))

        wcsdir = 'data/decam/astrom'
        args = [(cowcs, fn,
                 os.path.join(wcsdir, os.path.basename(fn).replace('.fits','.wcs')))
                 for fn in fns]
        for i,A in enumerate(mp.map(_det_one, args)):
            if A is None:
                print 'Skipping input', fns[i]
                continue
            Yo,Xo,rim,detmap_iv = A
            coadd[Yo,Xo] += rim * detmap_iv
            coadd_iv[Yo,Xo] += detmap_iv

        coadd /= np.maximum(coadd_iv, 1e-16)

        mn,mx = [np.percentile(coadd, p) for p in [20,99]]

        coadds.append((coadd, coadd_iv, mn, mx))

        plt.clf()
        plt.imshow(coadd, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=mn, vmax=mx)
        plt.title('Coadd detmap: %s' % band)
        plt.colorbar()
        ps.savefig()

        plt.clf()
        plt.imshow(coadd_iv, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('Coadd detmap invvar: %s' % band)
        plt.colorbar()
        ps.savefig()

        fitsio.write('detmap-%s.fits' % band, coadd.astype(np.float32), clobber=True)
        # no clobber -- append to file
        fitsio.write('detmap-%s.fits' % band, coadd_iv.astype(np.float32))


    rgb = get_rgb_image(coadds[0][0], coadds[1][0], coadds[2][0])
    plt.clf()
    plt.imshow(rgb, interpolation='nearest', origin='lower')
    ps.savefig()


if __name__ == '__main__':
    main()
    
    g,r,z = [fitsio.read('detmap-%s.fits' % band) for band in 'grz']

    rgb = get_rgb_image(g,r,z, alpha=0.5)
    ps.skipto(6)
    plt.clf()
    plt.imshow(rgb, interpolation='nearest', origin='lower')
    ps.savefig()
