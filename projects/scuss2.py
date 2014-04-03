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
from astrometry.util.util import *
from astrometry.sdss import *
from astrometry.util.ttime import *

median_f = flat_median_f
percentile_f = flat_percentile_f

from tractor import *

'''
sex data/scuss-w1-images/stacked/a0073.fit -c CS82.sex -CATALOG_NAME a0073.se.fits -WEIGHT_IMAGE data/scuss-w1-images/stacked/b_a0073.fit -CHECKIMAGE_TYPE NONE
psfex a0073.se.fits -c CS82.psfex
--> a0073.se.psf
'''


def main():
    import optparse

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-o', dest='outfn', help='Output filename (FITS table)')
    parser.add_option('-i', dest='imgfn', help='Image input filename')
    parser.add_option('-f', dest='flagfn', help='Flags input filename')
    parser.add_option('-z', dest='flagzero', help='Flag image: zero = 0', action='store_true')
    parser.add_option('-p', dest='psffn', help='PsfEx input filename')
    #parser.add_option('-s', dest='postxt', help='Source positions input text file')
    parser.add_option('-S', dest='statsfn', help='Output image statistis filename (FITS table); optional')

    parser.add_option('-g', dest='gaussianpsf', action='store_true',
                      default=False,
                      help='Use multi-Gaussian approximation to PSF?')
    
    parser.add_option('-P', dest='plotbase', default='scuss',
                      help='Plot base filename (default: %default)')
    parser.add_option('-l', dest='local', action='store_true', default=False,
                      help='Use local SDSS tree?')
    opt,args = parser.parse_args()

    # Check command-line arguments
    if len(args):
        print 'Extra arguments:', args
        parser.print_help()
        sys.exit(-1)
    for fn,name,exists in [(opt.outfn, 'output filename (-o)', False),
                           (opt.imgfn, 'image filename (-i)', True),
                           (opt.flagfn, 'flag filename (-f)', True),
                           (opt.psffn, 'PSF filename (-p)', True),
                           #(opt.postxt, 'Source positions filename (-s)', True),
                           ]:
        if fn is None:
            print 'Must specify', name
            sys.exit(-1)
        if exists and not os.path.exists(fn):
            print 'Input file', fn, 'does not exist'
            sys.exit(-1)

    sdss = DR9(basedir='data/unzip')
    if opt.local:
        sdss.useLocalTree()
        sdss.saveUnzippedFiles('data/unzip')

    # Read inputs
    print 'Reading input image', opt.imgfn
    img = fitsio.read(opt.imgfn)
    print 'Read img', img.shape, img.dtype
    H,W = img.shape
    img = img.astype(np.float32)
    
    wcs = anwcs(opt.imgfn)
    print 'WCS pixel scale:', wcs.pixel_scale()
    
    print 'Reading flags', opt.flagfn
    flag = fitsio.read(opt.flagfn)
    print 'Read flag', flag.shape, flag.dtype

    print 'Reading PSF', opt.psffn
    psf = PsfEx(opt.psffn, W, H)

    if opt.gaussianpsf:
        picpsffn = opt.psffn + '.pickle'
        if not os.path.exists(picpsffn):
            psf.savesplinedata = True
            print 'Fitting PSF model...'
            psf.ensureFit()
            pickle_to_file(psf.splinedata, picpsffn)
            print 'Wrote', picpsffn
        else:
            print 'Reading PSF model parameters from', picpsffn
            data = unpickle_from_file(picpsffn)
            print 'Fitting PSF...'
            psf.fitSavedData(*data)
            
    print 'Computing image sigma...'
    if opt.flagzero:
        bad = np.flatnonzero((flag == 0))
        good = (flag != 0)
    else:
        bad = np.flatnonzero((flag != 0))
        good = (flag == 0)

    igood = img[good]
    plo,med,phi = [percentile_f(igood, p) for p in [25, 50, 75]]
    # Wikipedia says:  IRQ -> sigma:
    sigma = (phi - plo) / (0.6745 * 2)
    print 'Sigma:', sigma
    invvar = np.zeros_like(img) + (1./sigma**2)
    invvar.flat[bad] = 0.

    del bad
    del good
    del igood
    
    # Estimate sky level from median -- not actually necessary, since
    # we will fit it below.
    
    band = 'u'

    # Get SDSS sources within the image...

    print 'Reading SDSS objects...'
    T = read_photoobjs_in_wcs(wcs, 1./60., sdss=sdss)
    print 'Got', len(T), 'SDSS objs'

    ok,T.x,T.y = wcs.radec2pixelxy(T.ra, T.dec)
    
    # We will break the image into cells for speed -- save the
    # original full-size inputs here.
    fullinvvar = invvar
    fullimg  = img
    fullpsf  = psf
    fullT = T

    # We add a margin around each cell -- we want sources within the
    # cell, we need to include a margin of image pixels touched by
    # those sources, and also an additional margin of sources that
    # touch those pixels.
    margin = 10 # pixels
    # Number of cells to split the image into
    #nx = 10
    #ny = 10
    nx = ny = 1
    # cell positions
    XX = np.round(np.linspace(0, W, nx+1)).astype(int)
    YY = np.round(np.linspace(0, H, ny+1)).astype(int)
    
    results = []

    # Image statistics
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
    
    # Plots:
    ps = PlotSequence(opt.plotbase)
    
    # Loop over cells...
    celli = -1
    for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
        for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
            celli += 1
            imstats.xlo[celli] = xlo
            imstats.xhi[celli] = xhi
            imstats.ylo[celli] = ylo
            imstats.yhi[celli] = yhi
            print
            print 'Doing image cell %i: x=[%i,%i), y=[%i,%i)' % (celli, xlo,xhi,ylo,yhi)
            # We will fit for sources in the [xlo,xhi), [ylo,yhi) box.
            # We add a margin in the image around that ROI
            # Beyond that, we add a margin of extra sources
    
            # image region: [ix0,ix1)
            ix0 = max(0, xlo - margin)
            ix1 = min(W, xhi + margin)
            iy0 = max(0, ylo - margin)
            iy1 = min(H, yhi + margin)
            S = (slice(iy0, iy1), slice(ix0, ix1))

            img = fullimg[S]
            invvar = fullinvvar[S]

            if not opt.gaussianpsf:
                # Instantiate pixelized PSF at this cell center.
                pixpsf = fullpsf.instantiateAt((xlo+xhi)/2., (ylo+yhi)/2.)
                print 'Pixpsf:', pixpsf.shape
                psf = PixelizedPSF(pixpsf)
            else:
                psf = fullpsf
            psf = ShiftedPsf(fullpsf, ix0, iy0)
            
            # sources nearby
            x0 = max(0, xlo - margin*2)
            x1 = min(W, xhi + margin*2)
            y0 = max(0, ylo - margin*2)
            y1 = min(H, yhi + margin*2)
            
            # (SCUSS uses FITS pixel indexing, so -1)
            J = np.flatnonzero((fullT.x-1 >= x0) * (fullT.x-1 < x1) *
                               (fullT.y-1 >= y0) * (fullT.y-1 < y1))
            T = fullT[J].copy()
            T.row = J
    
            # Remember which sources are within the cell (not the margin)
            T.inbounds = ((T.x-1 >= xlo) * (T.x-1 < xhi) *
                          (T.y-1 >= ylo) * (T.y-1 < yhi))
            # Shift source positions so they are correct for this subimage (cell)
            T.x -= ix0
            T.y -= iy0
    
            imstats.ninbox[celli] = sum(T.inbounds)
            imstats.ntotal[celli] = len(T)
    
            # print 'Image subregion:', img.shape
            print 'Number of sources in ROI:', sum(T.inbounds)
            print 'Number of sources in ROI + margin:', len(T)
            #print 'Source positions: x', T.x.min(), T.x.max(), 'y', T.y.min(), T.y.max()

            # Create tractor.Image object
            tim = Image(data=img, invvar=invvar, psf=psf, wcs=NullWCS(),
                        sky=ConstantSky(med), photocal=LinearPhotoCal(1., band=band),
                        name=opt.imgfn, domask=False)
    
            # Create tractor catalog objects
            cat = []
            for i in range(len(T)):
                # -1: SCUSS, apparently, uses FITS pixel conventions.
                src = PointSource(PixPos(T.x[i] - 1, T.y[i] - 1),
                                  Fluxes(**{band:100.}))
                cat.append(src)

            # Create Tractor object.
            tractor = Tractor([tim], cat)

            # print 'All params:'
            # tractor.printThawedParams()
            t0 = Time()
            tractor.freezeParamsRecursive('*')
            tractor.thawPathsTo('sky')
            tractor.thawPathsTo(band)
            # print 'Fitting params:'
            # tractor.printThawedParams()

            # Forced photometry
            X = tractor.optimize_forced_photometry(
                minsb=1e-3*sigma, mindlnp=1., sky=True, minFlux=None, variance=True,
                fitstats=True, shared_params=False)
            IV = X.IV
            fs = X.fitstats

            print 'Forced photometry took', Time()-t0
            
            # print 'Fit params:'
            # tractor.printThawedParams()

            # Record results
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
                    imstats.get(k)[celli] = X[0]
            results.append(T)

            # Make plots for the first N cells
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
    

    # Merge results from the cells
    TT = merge_tables(results)
    # Cut to just the sources within the cells
    TT.cut(TT.inbounds)
    TT.delete_column('inbounds')
    # Sort them back into original order
    TT.cut(np.argsort(TT.row))
    #TT.delete_column('row')
    TT.writeto(opt.outfn)
    print 'Wrote results to', opt.outfn
    
    if opt.statsfn:
        imstats.writeto(opt.statsfn)
        print 'Wrote image statistics to', opt.statsfn

    plot_results(opt.outfn, ps)


    
def plot_results(outfn, ps):
    T = fits_table(outfn)
    print 'read', len(T)

    # SDSS measurements
    mag = T.sdss_psfmag_u
    nm = NanoMaggies.magToNanomaggies(mag)

    # Tractor measurements
    counts = T.tractor_u_counts
    dcounts = T.tractor_u_counts_invvar
    dcounts = 1./np.sqrt(dcounts)

    # plt.clf()
    # plt.errorbar(nm, counts, yerr=dcounts, fmt='o', ms=5)
    # plt.xlabel('SDSS nanomaggies')
    # plt.ylabel('Tractor counts')
    # plt.title('Tractor forced photometry of SCUSS data')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.errorbar(np.maximum(1e-2, nm), np.maximum(1e-3, counts), yerr=dcounts, fmt='o', ms=5, alpha=0.5)
    # plt.xlabel('SDSS nanomaggies')
    # plt.ylabel('Tractor counts')
    # plt.title('Tractor forced photometry of SCUSS data')
    # plt.xscale('log')
    # plt.yscale('log')
    # ps.savefig()

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

    # Cut to valid/bright ones
    I = np.flatnonzero((nm > 1e-2) * (counts > 1e-2))
    J = np.flatnonzero((nm > 1) * (counts > 1e-2))
    # Estimate zeropoint
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
    
if __name__ == '__main__':
    main()
    
