import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import tempfile

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *
from astrometry.libkd.spherematch import *

import fitsio

from tractor.basics import NanoMaggies

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing

from decam import sky_subtract

ps = PlotSequence('det')

def get_rgb_image(g, r, z,
                  alpha = 1.5,
                  m = 0.0,
                  m2 = 0.0,
                  scale_g = 2.0,
                  scale_r = 1.0,
                  scale_z = 0.5,
                  Q = 20
                  ):
    #scale_g = 2.5
    #scale_z = 0.7
    #m = -0.02

    # Watch the ordering here -- "r" aliasing!
    b = np.maximum(0, g * scale_g - m)
    g = np.maximum(0, r * scale_r - m)
    r = np.maximum(0, z * scale_z - m)
    I = (r+g+b)/3.

    #m2 = 0.
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

def _det_one((cowcs, fn, wcsfn, do_img, wise)):
    print 'Image', fn
    F = fitsio.FITS(fn)[0]
    imginf = F.get_info()
    hdr = F.read_header()
    H,W = imginf['dims']

    wcs = Sip(wcsfn)

    if wise:
        # HACK -- assume, incorrectly, single Gaussian ~diffraction limited
        psf_sigma = 0.873
    else:
        pixscale = wcs.pixel_scale()
        seeing = hdr['SEEING']
        print 'Seeing', seeing
        psf_sigma = seeing / pixscale / 2.35
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

    if wise:
        ivfn = fn.replace('img-m.fits', 'invvar-m.fits.gz')
        iv = fitsio.read(ivfn)
        sig1 = 1./np.sqrt(np.median(iv))
        print 'Per-pixel noise estimate:', sig1
        mask = (iv == 0)
    else:
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

        # FIXME -- mask edge pixels -- some seem to be bad and unmasked
        mask[:2 ,:] = 1
        mask[-2:,:] = 1
        mask[:, :2] = 1
        mask[:,-2:] = 1

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
    detmap = gaussian_filter(img, psf_sigma, mode='constant') / psfnorm**2
    detmap_sig1 = sig1 / psfnorm
    print 'Detection map sig1', detmap_sig1

    # Lanczos resample
    print 'Resampling...'
    L = 3
    try:
        lims = [detmap]
        if do_img:
            lims.append(img)
        Yo,Xo,Yi,Xi,rims = resample_with_wcs(cowcs, wcs, lims, L)
        rdetmap = rims[0]
    except OverlapError:
        return None
    print 'Resampled'

    detmap_iv = (mask[Yo,Xo] == 0) * 1./detmap_sig1**2

    if do_img:
        rimg = rims[1]
    else:
        rimg = None

    return Yo,Xo,rdetmap,detmap_iv,rimg
    


def create_detmaps(bands):
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

    # Plug the WCS header cards into the output coadd files.
    f,tmpfn = tempfile.mkstemp()
    os.close(f)
    cowcs.write_to(tmpfn)
    hdr = fitsio.read_header(tmpfn)
    os.remove(tmpfn)

    for band in bands:

        print
        print 'Band', band

        wise = band.startswith('W')
        if not wise:
            fns = [fn for fn in paths if '%sband' % band in fn]
            wcsdir = 'data/decam/astrom'
            wcsfns = [os.path.join(wcsdir, os.path.basename(fn).replace('.fits','.wcs'))
                      for fn in fns]
        else:
            # HACK
            wisefn = 'unwise/352/3524p000/unwise-3524p000-%s-img-m.fits' % band.lower()
            fns = [wisefn]
            wcsfns = fns
            
        # resample image too (not just detection map?)
        do_img = True

        coH,coW = cowcs.get_height(), cowcs.get_width()
        codet = np.zeros((coH,coW))
        codet_iv = np.zeros((coH,coW))
        if do_img:
            coadd = np.zeros((coH, coW))
            coadd_iv = np.zeros((coH,coW))

        args = [(cowcs, fn, wcsfn, do_img, wise) for fn,wcsfn in zip(fns,wcsfns)]
        for i,A in enumerate(mp.map(_det_one, args)):
            if A is None:
                print 'Skipping input', fns[i]
                continue
            Yo,Xo,rdetmap,detmap_iv,rimg = A
            codet[Yo,Xo] += rdetmap * detmap_iv
            codet_iv[Yo,Xo] += detmap_iv
            if do_img:
                coadd[Yo,Xo] += rimg * detmap_iv
                coadd_iv[Yo,Xo] += detmap_iv

        codet /= np.maximum(codet_iv, 1e-16)
        fitsio.write('detmap-%s.fits' % band, codet.astype(np.float32), header=hdr, clobber=True)
        # no clobber -- append to file
        fitsio.write('detmap-%s.fits' % band, codet_iv.astype(np.float32), header=hdr)

        if do_img:
            coadd /= np.maximum(coadd_iv, 1e-16)
            fitsio.write('coadd-%s.fits' % band, coadd.astype(np.float32), header=hdr, clobber=True)
            # no clobber -- append to file
            fitsio.write('coadd-%s.fits' % band, coadd_iv.astype(np.float32), header=hdr)




def main2(bands):

    detmaps = [fitsio.read('detmap-%s.fits' % b) for b in bands]
    for b,d in zip(bands,detmaps):
        print 'Band', b, 'detmap peak:', np.max(d)
    cowcs = Tan('detmap-%s.fits' % bands[0])
    coH,coW = cowcs.get_height(), cowcs.get_width()


    # Read SExtractor catalogs
    fns = [
        'data/desi/imaging/redux/decam/proc/20130804/C22/gband/dec095705.22.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C22/rband/dec095702.22.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C22/zband/dec095704.22.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C43/gband/dec095705.43.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C43/rband/dec095702.43.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C43/zband/dec095704.43.p.w.cat.fits',
        #'data/desi/imaging/redux/decam/proc/20130805/C22/gband/dec096107.22.p.w.cat.fits',
        #'data/desi/imaging/redux/decam/proc/20130805/C22/rband/dec096104.22.p.w.cat.fits',
        #'data/desi/imaging/redux/decam/proc/20130805/C22/zband/dec096106.22.p.w.cat.fits',
        ]

    cats = {}
    for fn in fns:
        T = fits_table(fn, hdu=2,
                       column_map={'alpha_j2000':'ra', 'delta_j2000':'dec'},
                       columns=['x_image', 'y_image', 'alpha_j2000', 'delta_j2000',
                                'mag_auto', 'flux_auto'])
        T.cut(T.mag_auto < 99)
        imfn = fn.replace('.cat.fits','.fits')
        hdr = fitsio.read_header(imfn)
        for zpkey in ['MAG_ZP', 'UB1_ZP']:
            zp = hdr.get(zpkey, None)
            if zp is not None:
                break
        print 'ZP', zp
        print 'Mags', T.mag_auto.min(), T.mag_auto.max()
        print 'Fluxes', T.flux_auto.min(), T.flux_auto.max()

        wcsfn = os.path.basename(fn).replace('.cat.fits','.wcs')
        wcsdir = 'data/decam/astrom'
        wcsfn = os.path.join(wcsdir, wcsfn)
        print 'WCS fn', wcsfn
        wcs = Sip(wcsfn)
        T.ra,T.dec = wcs.pixelxy2radec(T.x_image, T.y_image)
        # Cut to sources within the coadd.
        ok,T.cox,T.coy = cowcs.radec2pixelxy(T.ra, T.dec)
        T.cut((T.cox >= 1) * (T.cox <= coW) * (T.coy >= 1) * (T.coy <= coH))

        band = None
        for b in bands:
            if '%sband' % b in fn:
                band = b
                break

        mag2 = -2.5 * np.log10(T.flux_auto) + zp

        T.set('mag', mag2)
        if not band in cats:
            cats[band] = []
        cats[band].append(T)
    for k in cats.keys():
        cats[k] = merge_tables(cats[k])

    g,r,z = [cats[b].copy() for b in 'grz']

    R = 0.5 / 3600.
    I,J,d = match_radec(g.ra, g.dec, r.ra, r.dec, R, nearest=True)
    print len(I), 'g-r matches'
    # CUT to g-r matches only!
    g.cut(I)
    r.cut(J)

    I,J,d = match_radec(g.ra, g.dec, z.ra, z.dec, R, nearest=True)
    print len(I), 'g-r-z matches'
    # CUT to g-r-z matches only!
    g.cut(I)
    r.cut(I)
    z.cut(J)

    plt.clf()
    plt.plot(g.mag - r.mag, r.mag - z.mag, 'k.')
    plt.xlabel('g - r')
    plt.ylabel('r - z')
    ps.savefig()

    # Match to AllWISE catalog to find typical colors.
    wise = fits_table('wise-sources-3524p000.fits')
    print len(wise), 'WISE sources'

    ok,wise.cox,wise.coy = cowcs.radec2pixelxy(wise.ra, wise.dec)
    wise.cut((wise.cox >= 1) * (wise.cox <= coW) * (wise.coy >= 1) * (wise.coy <= coH))
    print 'Cut to', len(wise), 'within coadd'

    print 'Min W1:', np.min(wise.w1mpro)
    print 'Min W2:', np.min(wise.w2mpro)

    I,J,d = match_radec(r.ra, r.dec, wise.ra, wise.dec, 4./3600)
    print len(I), 'matches'

    plt.clf()
    plt.plot(g.mag[I] - r.mag[I], r.mag[I] - wise.w1mpro[J], 'k.')
    plt.xlabel('g - r')
    plt.ylabel('r - W1')
    ps.savefig()

    plt.clf()
    plt.plot(g.mag[I] - r.mag[I], r.mag[I] - wise.w2mpro[J], 'k.')
    plt.xlabel('g - r')
    plt.ylabel('r - W2')
    ps.savefig()


    # Check detection map values -- do they correspond to catalog fluxes?
    ix,iy = np.round(wise.cox - 1).astype(int), np.round(wise.coy - 1).astype(int)
    for b in ['W1','W2']:
        dmap = detmaps[bands.index(b)]
        dval = dmap[iy, ix]
        mag = wise.get('%smpro' % b.lower())
        
        plt.clf()
        plt.plot(mag, -2.5 * (np.log10(dval) - 9.), 'k.')
        plt.xlabel('WISE mpro')
        plt.ylabel('detection map')
        lo,hi = 13,18
        plt.plot([lo,hi],[lo,hi],'b-', alpha=0.5)
        plt.axis([hi,lo,hi,lo])
        plt.title(b)
        ps.savefig()

    for b in 'grz':
        cat = cats[b].copy()
        ix,iy = np.round(cat.cox - 1).astype(int), np.round(cat.coy - 1).astype(int)
        dmap = detmaps[bands.index(b)]
        dval = dmap[iy, ix]
        
        plt.clf()
        plt.plot(cat.mag, -2.5 * (np.log10(dval) - 9.), 'k.')
        plt.xlabel('SourceExtractor mag_auto')
        plt.ylabel('detection map')
        lo,hi = 16,24
        #lo,hi = 8,30
        plt.plot([lo,hi],[lo,hi],'b-', alpha=0.5)
        plt.axis([hi,lo,hi,lo])
        plt.title(b)
        ps.savefig()




def main3(bands):
    import detmap.detection as detmapdet

    detmaps = [fitsio.read('detmap-%s.fits' % b) for b in bands]
    detivs  = [fitsio.read('detmap-%s.fits' % b, ext=1) for b in bands]
    sig1s = [np.sqrt(1./np.median(iv[iv>0])) for iv in detivs]

    # WISE Vega to AB
    for b,dmag in [('W1', 2.683), ('W2', 3.319)]:
        i = bands.index(b)
        scale = 10. ** (dmag / 2.5)
        detmaps[i] /= scale
        sig1s  [i] /= scale
        detivs [i] *= scale**2

    if False:
        for detmap,detiv,band,sig1 in zip(detmaps, detivs, bands, sig1s):
            #mn,mx = [np.percentile(codet, p) for p in [20,99]]
            mn,mx = 0., 10.*sig1
            plt.clf()
            plt.imshow(detmap, interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            plt.title('Coadd detmap: %s' % band)
            plt.colorbar()
            ps.savefig()
    
            plt.clf()
            plt.imshow(detiv, interpolation='nearest', origin='lower', cmap='gray')
            plt.title('Coadd detmap invvar: %s' % band)
            plt.colorbar()
            ps.savefig()
    else:
        ps.skip(2*len(bands))


    cowcs = Tan('detmap-%s.fits' % bands[0])
    coH,coW = cowcs.get_height(), cowcs.get_width()

    # Read SExtractor catalogs
    fns = [
        'data/desi/imaging/redux/decam/proc/20130804/C22/gband/dec095705.22.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C22/rband/dec095702.22.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C22/zband/dec095704.22.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C43/gband/dec095705.43.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C43/rband/dec095702.43.p.w.cat.fits',
        'data/desi/imaging/redux/decam/proc/20130804/C43/zband/dec095704.43.p.w.cat.fits',
        ]
    catbands = []   # kitties playing guitars... look out youtubes, here we come
    for fn in fns:
        band = None
        for b in bands:
            if ('%sband' % b) in fn:
                band = b
                break
        catbands.append(band)
    fns = zip(fns, catbands)

    cats = {}
    for fn,band in fns:
        T = fits_table(fn, hdu=2,
                       columns=['x_image', 'y_image', 'mag_auto', 'flux_auto'])
        T.cut(T.mag_auto < 99)
        wcsfn = os.path.basename(fn).replace('.cat.fits','.wcs')
        wcsdir = 'data/decam/astrom'
        wcsfn = os.path.join(wcsdir, wcsfn)
        print 'WCS fn', wcsfn
        wcs = Sip(wcsfn)
        T.ra,T.dec = wcs.pixelxy2radec(T.x_image, T.y_image)
        # Cut to sources within the coadd.
        ok,T.cox,T.coy = cowcs.radec2pixelxy(T.ra, T.dec)
        T.cut((T.cox >= 1) * (T.cox <= coW) * (T.coy >= 1) * (T.coy <= coH))

        T.gmag = np.zeros(len(T))
        T.rmag = np.zeros(len(T))
        T.zmag = np.zeros(len(T))
        T.set('%smag' % band, T.mag_auto)
        if not band in cats:
            cats[band] = []
        cats[band].append(T)
    for k in cats.keys():
        cats[k] = merge_tables(cats[k])
        print len(cats[k]), 'SourceExtractor sources for', k, 'band'

    match_pix = 2.5
    match_radius = 0.27 * match_pix / 3600.
    # Match r to g,z, dropping g,z detections within radius and
    # keeping g- and z-only.
    rr = cats['r']
    rr.band = np.array(['r'] * len(rr))
    gg = cats['g']
    gg.band = np.array(['g'] * len(gg))
    zz = cats['z']
    zz.band = np.array(['z'] * len(zz))
    for oo in (gg,zz):
        I,J,d = match_radec(rr.ra, rr.dec, oo.ra, oo.dec, match_radius)
        magcol = '%smag' % oo.band[0]
        mag = rr.get(magcol)
        mag[I] = oo.get(magcol)[J]
        keep = np.ones(len(oo), bool)
        keep[J] = False
        oo.cut(keep)
    secat = merge_tables([gg, rr, zz])
    print 'Total of', len(secat), 'SExtractor sources'

    # Cut SExtractor detections near the CCD edges
    # FIXME -- hard-coded DECam CCD sizes!
    secat.cut((secat.x_image > 2.) * (secat.x_image < 2047) *
              (secat.y_image > 2.) * (secat.y_image < 4095))
    print 'Cut to', len(secat), 'SExtractor sources not near edges'

    #noptical = sednames.index('W1-only')
    noptical = 6
    seds = [
        ('g-only', (1., 0., 0.)),
        ('r-only', (0., 1., 0.)),
        ('z-only', (0., 0., 1.)),
        ('Flat',   (1., 1., 1.)),
        ('Red',    [2.5**x for x in [1, 0, -2]]),
        ('loc2',   [2.5**x for x in [1, 0, -0.7]]),
        #('FlatW1',   (1., 1., 1., 1.)),
        # ('W1-only', (0., 0., 0., 1., 0.)),
        # ('W2-only', (0., 0., 0., 0., 1.)),
        # ('W12', (0., 0., 0., 1., 1.)),
        # ('FlatW12',   (1., 1., 1., 1., 1.)),
        #('RedW1',    (2.5 **  1, 1., 2.5 ** -2, 2.5**-3)),
        #('RedW12',    (2.5 **  1, 1., 2.5 ** -2, 2.5**-3, 2.5**-3)),
        #('Mine',   [2.5**x for x in [1, 0, -0.5, -2, -2]]),
        # ('loc1', [2.5 ** c for c in [0.5, 0., -0.3]]),
        # ('loc3', [2.5 ** c for c in [1.4, 0., -1.1]]),
        # ('loc4', [2.5 ** c for c in [1.4, 0., -1.8]]),
        # ('Redder', (2.5 **  1.5, 1., 2.5 ** -3)),
        ]

    H,W = detmaps[0].shape

    # bitmasks for detection blobs in each SED.
    detmask = np.zeros((H,W), np.uint16)
    peakmask = np.zeros((H,W), np.uint16)

    # unique peaks
    upx,upy = None,None

    rgb = fitsio.read('rgb.fits')
    print 'RGB', rgb.shape

    # smooth the rgb image a little
    for i in range(3):
        rgb[:,:,i] = gaussian_filter(rgb[:,:,i], 0.5)

    ig  = bands.index('g')
    ir  = bands.index('r')
    iz  = bands.index('z')
    iW1 = bands.index('W1')
    iW2 = bands.index('W2')

    sednames = [nm for nm,sed in seds]

    alldetmaps = []

    for ised,(name,sed) in enumerate(seds):
        print 'SED:', name
        mdet, msig, msig1 = detmapdet.sed_matched_filter(sed, detmaps, detivs, sig1s)

        blobs,blobslices,P,Px,Py,peaks = detmapdet.get_detections(mdet / msig, 1., mdet,
                                                                  fill_holes=True,
                                                                  nsigma=5.)
        detmask  |= (blobs != 0) * (1 << ised)
        peakmask |= peaks        * (1 << ised)

        alldetmaps.append(mdet / msig)

        if upx is None:
            upx = Px
            upy = Py
            newpeaks = Px,Py
        else:
            keep = np.ones(len(Px), bool)
            I,J,d = match_xy(upx, upy, Px, Py, match_pix)
            keep[J] = False
            print 'Keeping', sum(keep), 'new peaks'
            upx = np.append(upx, Px[keep])
            upy = np.append(upy, Py[keep])
            newpeaks = Px[keep],Py[keep]
        print 'Total of', len(upx), 'peaks'

        px,py = newpeaks
        if len(px) == 0:
            continue
        peaksn = mdet[py,px] / msig[py,px]
        I = np.argsort(-peaksn)

        if ised >= noptical:
            print 'New peaks: S/N vs Flat S/N:'
            for i in I[:25]:
                optsn = [m[py[i],px[i]] for m in alldetmaps[:noptical]]
                optimax = np.argmax(optsn)
                print '  % 8.1f' % (peaksn[i]), '  % 8.1f' % optsn[optimax], 'in', sednames[optimax]

        # Plot most significant new peaks
        plt.clf()
        for j,i in enumerate(I[:25]):
            S = 20
            plt.subplot(5,5, j+1)
            plt.imshow(rgb[max(0, py[i]-S) : min(coH, py[i]+S+1),
                           max(0, px[i]-S) : min(coW, px[i]+S+1), :],
                       interpolation='nearest', origin='lower')
            plt.xticks([])
            plt.yticks([])
        plt.suptitle('New peaks for %s' % name)
        ps.savefig()

        # Plot detmap values in r,W1 for all peaks
        magg  = -2.5*(np.log10(detmaps[ig ][Py,Px])-9)
        magr  = -2.5*(np.log10(detmaps[ir ][Py,Px])-9)
        magz  = -2.5*(np.log10(detmaps[iz ][Py,Px])-9)
        magW1 = -2.5*(np.log10(detmaps[iW1][Py,Px])-9)
        magW2 = -2.5*(np.log10(detmaps[iW2][Py,Px])-9)

        plt.clf()
        p1 = plt.plot(magr, magW1, 'k.', alpha=0.5)
        p2 = plt.plot(magr, magW2, 'm.', alpha=0.5)
        p3 = plt.plot(magr, magg, 'g.', alpha=0.5)
        p4 = plt.plot(magr, magz, 'r.', alpha=0.5)
        plt.legend([p[0] for p in [p3,p4,p1,p2]], ['g','z','W1','W2'],
                   loc='upper left')
        plt.xlabel('r mag')
        plt.ylabel('X mag')
        lo,hi = 12,26
        plt.plot([lo,hi],[lo,hi], 'b-', alpha=0.5)
        plt.plot([lo,hi],[lo+1,hi+1], 'b-', alpha=0.25)
        plt.plot([lo,hi],[lo-1,hi-1], 'b-', alpha=0.25)
        plt.plot([lo,hi],[lo-2,hi-2], 'b-', alpha=0.25)
        plt.plot([lo,hi],[lo-3,hi-3], 'b-', alpha=0.25)
        plt.plot([lo,hi],[lo-4,hi-4], 'b-', alpha=0.25)
        plt.axis([lo,hi,lo,hi])
        plt.title(name)
        ps.savefig()


        plt.clf()
        p1 = plt.plot(magr, magW1 - magr, 'k.', alpha=0.5)
        p2 = plt.plot(magr, magW2 - magr, 'm.', alpha=0.5)
        p3 = plt.plot(magr, magg  - magr, 'g.', alpha=0.5)
        p4 = plt.plot(magr, magz  - magr, 'r.', alpha=0.5)
        plt.legend([p[0] for p in [p3,p4,p1,p2]], ['g','z','W1','W2'],
                   loc='upper left')
        plt.xlabel('r mag')
        plt.ylabel('X - r mag')
        lo,hi = 12,26
        for y in [1,0,0,-1,-2,-3,-4]:
            plt.axhline(y, color='b', alpha=0.25)
        plt.axis([lo,hi,-5,5])
        plt.title(name)
        ps.savefig()


        # plt.clf()
        # plt.imshow(blobs, interpolation='nearest', origin='lower')
        # plt.title('blobs: %s' % name)
        # ps.savefig()

        # plt.clf()
        # plt.imshow(peaks, interpolation='nearest', origin='lower')
        # plt.title('peaks: %s' % name)
        # ps.savefig()

        # for iblob,slc in enumerate(blobslices):
        #     mblob = mdet[slc]
        #     msigblob = msig[slc]
        #     pk = peaks[slc]
        #     y0,x0 = [s.start for s in slc]
        #     y1,x1 = [s.stop  for s in slc]
        #     pi = np.flatnonzero(pk)
        #     py,px = np.unravel_index(pi, pk.shape)
        #     px = px.astype(np.int32)
        #     py = py.astype(np.int32)

    hot = (detmask > 0)
    blobs,nblobs = label(hot, np.ones((3,3), int))
    blobslices = find_objects(blobs)

    plt.clf()
    plt.imshow(blobs, interpolation='nearest', origin='lower')
    plt.title('blobs: all')
    ps.savefig()

    I,J,d = match_xy(upx, upy, secat.cox-1, secat.coy-1, match_pix)
    print 'Of', len(upx), 'SED-matched and', len(secat), 'SExtractor,'
    print 'Matched', len(I)

    only = np.ones(len(secat), bool)
    only[J] = False
    seonly = secat[only]
    only = np.ones(len(upx), bool)
    only[I] = False
    sedonly = upx[only],upy[only]

    sematch = secat[J]

    # What about SDSS sources?
    sdss = fits_table('photoobjs-3524p000.fits')
    # Cut to sources within the coadd.
    ok,sdss.cox,sdss.coy = cowcs.radec2pixelxy(sdss.ra, sdss.dec)
    sdss.cut((sdss.cox >= 1) * (sdss.cox <= coW) * (sdss.coy >= 1) * (sdss.coy <= coH))
    print len(sdss), 'SDSS sources within coadd'
    

    plt.clf()
    plt.imshow(rgb, interpolation='nearest', origin='lower')
    ax = plt.axis()
    plt.plot(sematch.cox-1, sematch.coy-1, 'o', mec='w', mfc='none', ms=6, mew=2)
    plt.plot(seonly.cox-1,  seonly.coy-1,  'o', mec='r', mfc='none', ms=6, mew=2)
    plt.plot(sdss.cox-1, sdss.coy-1, 'x', color=(0,1,0))
    x,y = sedonly
    plt.plot(x, y, 'w+', ms=8, mew=1)
    plt.axis(ax)
    
    ps.savefig()


    for iblob,slc in enumerate(blobslices):
        pk = peakmask[slc] * hot[slc]
        y0,x0 = [s.start for s in slc]
        y1,x1 = [s.stop  for s in slc]
        pi = np.flatnonzero(pk)
        py,px = np.unravel_index(pi, pk.shape)
        px = px.astype(np.int32) + x0
        py = py.astype(np.int32) + y0

        sy,sx = slc
        
        if iblob % 25 == 0:
            plt.clf()
        plt.subplot(5, 5, 1 + (iblob % 25))
        plt.imshow(rgb[sy,sx,:] * hot[sy,sx,np.newaxis],
                   interpolation='nearest', origin='lower',
                   extent=[x0-0.5,x1-0.5,y0-0.5,y1-0.5])
        ax = plt.axis()
        #plt.plot(px, py, 'k+', ms=15, mew=3)
        #plt.plot(px, py, 'w+', ms=10, mew=2)
        #plt.plot(upx, upy, 'k+', ms=15, mew=3)
        #plt.plot(upx, upy, 'w+', ms=15, mew=1)
        x,y = sedonly
        plt.plot(x, y, 'w+', ms=15, mew=1)
        
        #for b,cc in zip(bands, 'cgr'):
        #    plt.plot(cats[b].cox-1, cats[b].coy-1, 'o', mec=cc, mfc='none', ms=10, mew=2)
        #plt.plot(secat.cox-1, secat.coy-1, 'o', mec='r', mfc='none', ms=10, mew=2)

        plt.plot(sematch.cox-1, sematch.coy-1, 'o', mec='w', mfc='none', ms=10, mew=2)
        plt.plot(seonly.cox-1,  seonly.coy-1,  'o', mec='r', mfc='none', ms=10, mew=2)

        plt.axis(ax)
        if (iblob+1) % 25 == 0:
            ps.savefig()
        if iblob > 150:
            break


def write_rgb():
    #g,r,z = [fitsio.read('detmap-%s.fits' % band) for band in 'grz']
    g,r,z = [fitsio.read('coadd-%s.fits' % band) for band in 'grz']

    plt.figure(figsize=(10,10))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    plt.clf()
    for (im1,cc),scale in zip([(g,'b'),(r,'g'),(z,'r')],
                             [2.0, 1.2, 0.4]):
        im = im1 * scale
        im = im[im != 0]
        plt.hist(im.ravel(), histtype='step', color=cc,
                 range=[np.percentile(im, p) for p in (1,98)], bins=50)
    ps.savefig()
        
    #rgb = get_rgb_image(g,r,z, alpha=0.8, m=0.02)
    #rgb = get_rgb_image(g,r,z, alpha=16., m=0.005, m2=0.002,
    #rgb = get_rgb_image(g,r,z, alpha=32., m=0.01, m2=0.002,
    rgb = get_rgb_image(g,r,z, alpha=8., m=0.0, m2=0.0,
        scale_g = 2.,
        scale_r = 1.1,
        scale_z = 0.5,
        Q = 10)


    #for im in g,r,z:
    #    mn,mx = [np.percentile(im, p) for p in [20,99]]
    #    print 'mn,mx:', mn,mx
    
    plt.clf()
    plt.imshow(rgb, interpolation='nearest', origin='lower')
    ps.savefig()

    fitsio.write('rgb.fits', rgb)


if __name__ == '__main__':
    bands = ['g','r','z', 'W1','W2']

    missing = []
    for b in bands:
        if not os.path.exists('detmap-%s.fits' % b):
            missing.append(b)
    if len(missing):
        create_detmaps(missing)

    #ps.skipto(10)
    if not os.path.exists('rgb.fits'):
        write_rgb()

    ps.skipto(2)
    #main2(bands)
    #sys.exit(0)
    main3(bands)



