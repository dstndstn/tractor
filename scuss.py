if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np

import os
import sys
import logging

import fitsio

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *

from tractor import *

'''
scp -r scuss@202.127.24.6:todustin .
mv todustin scuss

for x in scuss/*.pos; do
  text2fits.py -H "ra dec x y objid sdss_psfmag_u sdss_psfmagerr_u" -f ssddsff $x $x.fits
done

'''
outfn = 'tractor-scuss.fits'
imstatsfn = 'tractor-scuss-imstats.fits'

ps = PlotSequence('scuss')

if os.path.exists(outfn):

    T = fits_table(outfn)
    print 'read', len(T)

    mag = T.sdss_psfmag_u
    nm = NanoMaggies.magToNanomaggies(mag)

    counts = T.tractor_u_counts
    dcounts = T.tractor_u_counts_invvar
    dcounts = 1./np.sqrt(dcounts)


    plt.clf()
    plt.errorbar(nm, counts, yerr=dcounts, fmt='o', ms=5)
    plt.xlabel('SDSS nanomaggies')
    plt.ylabel('Tractor counts')
    plt.title('Tractor forced photometry of SCUSS data')
    ps.savefig()

    plt.clf()
    plt.errorbar(np.maximum(1e-2, nm), np.maximum(1e-3, counts), yerr=dcounts, fmt='o', ms=5, alpha=0.5)
    plt.xlabel('SDSS nanomaggies')
    plt.ylabel('Tractor counts')
    plt.title('Tractor forced photometry of SCUSS data')
    plt.xscale('log')
    plt.yscale('log')
    ps.savefig()

    plt.clf()
    plt.loglog(np.maximum(1e-2, nm), np.maximum(1e-2, counts), 'b.', ms=5, alpha=0.5)
    plt.xlabel('SDSS nanomaggies')
    plt.ylabel('Tractor counts')
    plt.title('Tractor forced photometry of SCUSS data')
    ax = plt.axis()
    plt.axhline(1e-2, color='r', alpha=0.5)
    plt.axvline(1e-2, color='r', alpha=0.5)
    plt.xlim(0.8e-2, ax[1])
    plt.ylim(0.8e-2, ax[3])
    ps.savefig()


    I = np.flatnonzero((nm > 1e-2) * (counts > 1e-2))

    J = np.flatnonzero((nm > 1) * (counts > 1e-2))
    med = np.median(counts[J] / nm[J])

    plt.clf()
    plt.loglog(nm[I], counts[I]/nm[I], 'b.', ms=5, alpha=0.5)
    plt.xlabel('SDSS nanomaggies')
    plt.ylabel('Tractor counts / SDSS nanomaggies')
    plt.title('Tractor forced photometry of SCUSS data')
    ax = plt.axis()
    plt.axhline(med, color='k', alpha=0.5)
    plt.axis(ax)
    ps.savefig()


    sys.exit(0)


img = fitsio.read('scuss/p0214_0099_1.fits')
print 'Read img', img.shape, img.dtype
H,W = img.shape

posfn = 'scuss/p0214_0099_1.pos.fits'
if not os.path.exists(posfn):
    from astrometry.util.fits import streaming_text_table
    postxt = posfn.replace('.fits','')
    assert(os.path.exists(postxt))
    hdr = 'ra dec x y objid sdss_psfmag_u sdss_psfmagerr_u'
    d = np.float64
    f = np.float32
    types = [str,str,d,d,str,f,f]
    T = streaming_text_table(postxt, headerline=hdr, coltype=types)
    T.writeto(posfn)
    
T = fits_table(posfn)
print 'Read sources', len(T)

flag = fitsio.read('scuss/flag_1.fits')
print 'Read flag', flag.shape, flag.dtype

psffn = 'scuss/p0214_0099_1.psf'
psf = PsfEx(psffn, W, H)
print 'Read PSF', psf

picpsffn = psffn + '.pickle'
if not os.path.exists(picpsffn):
    psf.savesplinedata = True
    print 'Fitting PSF model...'
    psf.ensureFit()
    print 'done'
    pickle_to_file(psf.splinedata, picpsffn)
else:
    print 'Reading PSF model parameters from', picpsffn
    data = unpickle_from_file(picpsffn)
    print 'Fitting PSF...'
    psf.fitSavedData(*data)
    print 'done'


plo,phi = [np.percentile(img[flag == 0], p) for p in [25,75]]
# Wikipedia says:  IRQ -> sigma:
sigma = (phi - plo) / (0.6745 * 2)
print 'Sigma:', sigma

invvar = np.zeros_like(img) + (1./sigma**2)
invvar[flag != 0] = 0.

med = np.median(img[flag == 0])

band = 'u'

fullinvvar = invvar
fullimg  = img
fullflag = flag
fullpsf  = psf
fullT = T

margin = 10 # pixels
nx = 20
ny = 20
XX = np.round(np.linspace(0, W, nx)).astype(int)
YY = np.round(np.linspace(0, H, ny)).astype(int)

results = []

imstats = fits_table()
imstats.xlo = np.zeros(((len(YY)-1)*(len(XX)-1)), int)
imstats.xhi = np.zeros_like(imstats.xlo)
imstats.ylo = np.zeros_like(imstats.xlo)
imstats.yhi = np.zeros_like(imstats.xlo)
imstats.ninbox = np.zeros_like(imstats.xlo)
imstats.ntotal = np.zeros_like(imstats.xlo)
imstatkeys = ['imchisq', 'imnpix', 'sky']
for k in imstatkeys:
    imstats.set(k, np.zeros(len(imstats)))


celli = -1
for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
    for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
        celli += 1
        imstats.xlo[celli] = xlo
        imstats.xhi[celli] = xhi
        imstats.ylo[celli] = ylo
        imstats.yhi[celli] = yhi
        # We will fit for sources in the [xlo,xhi), [ylo,yhi) box.
        # We add a margin in the image around that ROI
        # Beyond that, we add a margin of extra sources

        # image region:
        ix0 = max(0, xlo - margin)
        ix1 = min(W, xhi + margin)
        iy0 = max(0, ylo - margin)
        iy1 = min(H, yhi + margin)
        S = (slice(iy0, iy1), slice(ix0, ix1))

        img = fullimg[S]
        invvar = fullinvvar[S]
        psf = ShiftedPsf(fullpsf, ix0, iy0)

        # sources nearby
        x0 = max(0, xlo - margin*2)
        x1 = min(W, xhi + margin*2)
        y0 = max(0, ylo - margin*2)
        y1 = min(H, yhi + margin*2)
        
        # (SCUSS uses FITS pixel indexing)
        J = np.flatnonzero((fullT.x-1 >= x0) * (fullT.x-1 < x1) *
                           (fullT.y-1 >= y0) * (fullT.y-1 < y1))
        T = fullT[J].copy()
        T.row = J

        T.inbounds = ((T.x-1 >= xlo) * (T.x-1 < xhi) *
                      (T.y-1 >= ylo) * (T.y-1 < yhi))
        # Put in-bounds ones first
        T.cut(np.argsort(-1 * T.inbounds))

        # Adjust for subimage
        T.x -= ix0
        T.y -= iy0

        imstats.ninbox[celli] = sum(T.inbounds)
        imstats.ntotal[celli] = len(T)

        print 'Image subregion:', img.shape
        print 'Number of sources in ROI:', sum(T.inbounds)
        print 'Number of source in ROI + margin:', len(T)

        print 'Source positions: x', T.x.min(), T.x.max(), 'y', T.y.min(), T.y.max()

        tim = Image(data=img, invvar=invvar, psf=psf, wcs=NullWCS(),
                    sky=ConstantSky(med), photocal=LinearPhotoCal(1., band=band),
                    name='scuss 1', domask=False)

        cat = []
        for i in range(len(T)):
            # -1: SCUSS, apparently, uses FITS pixel conventions.
            src = PointSource(PixPos(T.x[i] - 1, T.y[i] - 1),
                              Fluxes(**{band:100.}))
            cat.append(src)
        
        tractor = Tractor([tim], cat)
        
        print 'All params:'
        tractor.printThawedParams()
        
        t0 = Time()
        tractor.freezeParamsRecursive('*')
        tractor.thawPathsTo('sky')
        tractor.thawPathsTo(band)
        
        print 'Fitting params:'
        tractor.printThawedParams()
        
        ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
            minsb=1e-3*sigma, mindlnp=1., sky=True, minFlux=None, variance=True,
            fitstats=True)
        
        print 'Forced phot took', Time()-t0
        
        print 'Fit params:'
        tractor.printThawedParams()

        T.set('tractor_%s_counts' % band, np.array([src.getBrightness().getBand(band) for src in cat]))
        T.set('tractor_%s_counts_invvar' % band, IV)
        T.cell = np.zeros(len(T), int) + celli
        if fs is not None:
            # Per-source stats
            for k in ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
                T.set(k, getattr(fs, k))

            # Per-image stats
            for k in imstatkeys:
                X = getattr(fs, k)
                #print 'image stats', k, '=', X
                imstats.get(k)[celli] = X[0]

        results.append(T)

        if celli >= 10:
            continue

        mod = tractor.getModelImage(0)
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=med + -2. * sigma, vmax=med + 5. * sigma)
        plt.clf()
        plt.imshow(img, **ima)
        plt.title('Data')
        ps.savefig()
        
        plt.clf()
        plt.imshow(mod, **ima)
        plt.title('Model')
        ps.savefig()
        
        noise = np.random.normal(scale=sigma, size=img.shape)
        plt.clf()
        plt.imshow(mod + noise, **ima)
        plt.title('Model + noise')
        ps.savefig()
        
        chi = (img - mod) * tim.getInvError()
        plt.clf()
        plt.imshow(chi, interpolation='nearest', origin='lower', vmin=-5, vmax=5)
        plt.title('Chi')
        ps.savefig()

TT = merge_tables(results)
TT.cut(TT.inbounds)
TT.cut(np.argsort(TT.row))
TT.writeto(outfn)

imstats.writeto(imstatsfn)


if False:
    print 'Fitting PSF model...'
    psf.ensureFit()
    print 'done'
    
    plt.clf()
    i=0
    nw,nh = 3,3
    for y in np.linspace(0, H, nh):
        for x in np.linspace(0, W, nw):
            print 'PSF X,Y', x,y
            mog = psf.mogAt(x, y)
            print 'MoG:', mog
            patch = mog.getPointSourcePatch(x, y)
            print 'Patch: sum', patch.patch.sum(), 'max', patch.patch.max()
            plt.subplot(nh,nw, i+1)
            plt.imshow(patch.patch, interpolation='nearest', origin='lower')
            plt.colorbar()
            i += 1
    ps.savefig()

# The image pixel PDF looks clean
# plo,phi = [np.percentile(img.ravel(), p) for p in [1,99]]
# plt.clf()
# plt.hist(img.ravel(), 100, range=(plo,phi))
# ps.savefig()



