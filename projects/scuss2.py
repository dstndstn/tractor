from __future__ import print_function
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
from tractor.sdss import *

from sequels import treat_as_pointsource

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

    parser.add_option('--sky', dest='fitsky', action='store_true',
                      help='Fit sky level as well as fluxes?')
    parser.add_option('--band', '-b', dest='band', default='r',
                      help='Which SDSS band to use for forced photometry profiles: default %default')

    parser.add_option('-g', dest='gaussianpsf', action='store_true',
                      default=False,
                      help='Use multi-Gaussian approximation to PSF?')
    
    parser.add_option('-P', dest='plotbase', default='scuss',
                      help='Plot base filename (default: %default)')
    parser.add_option('-l', dest='local', action='store_true', default=False,
                      help='Use local SDSS tree?')

    # TESTING
    parser.add_option('--sub', dest='sub', action='store_true',
                      help='Cut to small sub-image for testing')
    parser.add_option('--res', dest='res', action='store_true',
                      help='Just plot results from previous run')

    opt,args = parser.parse_args()

    # Check command-line arguments
    if len(args):
        print('Extra arguments:', args)
        parser.print_help()
        sys.exit(-1)
    for fn,name,exists in [(opt.outfn, 'output filename (-o)', False),
                           (opt.imgfn, 'image filename (-i)', True),
                           (opt.flagfn, 'flag filename (-f)', True),
                           (opt.psffn, 'PSF filename (-p)', True),
                           #(opt.postxt, 'Source positions filename (-s)', True),
                           ]:
        if fn is None:
            print('Must specify', name)
            sys.exit(-1)
        if exists and not os.path.exists(fn):
            print('Input file', fn, 'does not exist')
            sys.exit(-1)

    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.res:
        ps = PlotSequence(opt.plotbase)
        plot_results(opt.outfn, ps)
        sys.exit(0)

    sdss = DR9(basedir='.')#data/unzip')
    if opt.local:
        sdss.useLocalTree(pobj='photoObjs-new')
        sdss.saveUnzippedFiles('data/unzip')

    # Read inputs
    print('Reading input image', opt.imgfn)
    img,hdr = fitsio.read(opt.imgfn, header=True)
    print('Read img', img.shape, img.dtype)
    H,W = img.shape
    img = img.astype(np.float32)

    sky = hdr['SKYADU']
    print('Sky:', sky)

    cal = hdr['CALIA73']
    print('Zeropoint cal:', cal)
    zpscale = 10.**((2.5 + cal) / 2.5)
    print('Zp scale', zpscale)
    
    wcs = anwcs(opt.imgfn)
    print('WCS pixel scale:', wcs.pixel_scale())
    
    print('Reading flags', opt.flagfn)
    flag = fitsio.read(opt.flagfn)
    print('Read flag', flag.shape, flag.dtype)

    imslice = None
    if opt.sub:
        imslice = (slice(0, 800), slice(0, 800))
    if imslice is not None:
        img = img[imslice]
        H,W = img.shape
        flag = flag[imslice]
        wcs.set_width(W)
        wcs.set_height(H)

    print('Reading PSF', opt.psffn)
    psf = PsfEx(opt.psffn, W, H)

    if opt.gaussianpsf:
        picpsffn = opt.psffn + '.pickle'
        if not os.path.exists(picpsffn):
            psf.savesplinedata = True
            print('Fitting PSF model...')
            psf.ensureFit()
            pickle_to_file(psf.splinedata, picpsffn)
            print('Wrote', picpsffn)
        else:
            print('Reading PSF model parameters from', picpsffn)
            data = unpickle_from_file(picpsffn)
            print('Fitting PSF...')
            psf.fitSavedData(*data)

    #
    x = psf.instantiateAt(0., 0.)
    print('PSF', x.shape)
    x = x.shape[0]
    #psf.radius = (x+1)/2.
    psf.radius = 20
    
    print('Computing image sigma...')
    if opt.flagzero:
        bad = np.flatnonzero((flag == 0))
        good = (flag != 0)
    else:
        bad = np.flatnonzero((flag != 0))
        good = (flag == 0)

    igood = img[good]
    #plo,med,phi = [percentile_f(igood, p) for p in [25, 50, 75]]
    #sky = med
    plo,phi = [percentile_f(igood, p) for p in [25, 75]]
    # Wikipedia says:  IRQ -> sigma:
    sigma = (phi - plo) / (0.6745 * 2)
    print('Sigma:', sigma)
    invvar = np.zeros_like(img) + (1./sigma**2)
    invvar.flat[bad] = 0.
    del bad
    del good
    del igood
    
    band = 'u'

    # Get SDSS sources within the image...

    print('Reading SDSS objects...')
    cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
            'devflux', 'expflux',
            'resolve_status', 'nchild', 'flags', 'objc_flags',
            'run','camcol','field','id',
            'psfflux', 'psfflux_ivar', 'cmodelflux', 'cmodelflux_ivar',
            'modelflux', 'modelflux_ivar',
            'extinction']
    T = read_photoobjs_in_wcs(wcs, 1./60., sdss=sdss, cols=cols)
    print('Got', len(T), 'SDSS objs')

    T.treated_as_pointsource = treat_as_pointsource(T, band_index(opt.band))

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
    imh,imw = img.shape
    nx = int(np.round(imw / 400.))
    ny = int(np.round(imh / 400.))
    #nx = ny = 20
    #nx = ny = 1

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
            print()
            print('Doing image cell %i: x=[%i,%i), y=[%i,%i)' % (celli, xlo,xhi,ylo,yhi))
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
                print('Pixpsf:', pixpsf.shape)
                psf = PixelizedPSF(pixpsf)
            else:
                psf = fullpsf
            psf = ShiftedPsf(fullpsf, ix0, iy0)
            
            # sources nearby
            x0 = max(0, xlo - margin*2)
            x1 = min(W, xhi + margin*2)
            y0 = max(0, ylo - margin*2)
            y1 = min(H, yhi + margin*2)
            
            # FITS pixel indexing, so -1
            J = np.flatnonzero((fullT.x-1 >= x0) * (fullT.x-1 < x1) *
                               (fullT.y-1 >= y0) * (fullT.y-1 < y1))
            T = fullT[J].copy()
            T.row = J
    
            # Remember which sources are within the cell (not the margin)
            T.inbounds = ((T.x-1 >= xlo) * (T.x-1 < xhi) *
                          (T.y-1 >= ylo) * (T.y-1 < yhi))

            # Shift source positions so they are correct for this subimage (cell)
            #T.x -= ix0
            #T.y -= iy0
    
            imstats.ninbox[celli] = sum(T.inbounds)
            imstats.ntotal[celli] = len(T)
    
            # print 'Image subregion:', img.shape
            print('Number of sources in ROI:', sum(T.inbounds))
            print('Number of sources in ROI + margin:', len(T))
            #print 'Source positions: x', T.x.min(), T.x.max(), 'y', T.y.min(), T.y.max()

            twcs = WcslibWcs(None, wcs=wcs)
            twcs.setX0Y0(ix0, iy0)

            # Create tractor.Image object
            tim = Image(data=img, invvar=invvar, psf=psf, wcs=twcs,
                        sky=ConstantSky(sky),
                        photocal=LinearPhotoCal(zpscale, band=band),
                        name=opt.imgfn, domask=False)
    
            # Create tractor catalog objects
            cat,catI = get_tractor_sources_dr9(
                None, None, None, bandname=opt.band,
                sdss=sdss, objs=T.copy(), bands=[band],
                nanomaggies=True, fixedComposites=True, useObjcType=True,
                getobjinds=True)
            print('Got', len(cat), 'Tractor sources')

            assert(len(cat) == len(catI))

            # for r,d,src in zip(T.ra[catI], T.dec[catI], cat):
            #     print 'Source', src.getPosition()
            #     print '    vs', r, d
            
            # Create Tractor object.
            tractor = Tractor([tim], cat)

            # print 'All params:'
            # tractor.printThawedParams()
            t0 = Time()
            tractor.freezeParamsRecursive('*')
            tractor.thawPathsTo(band)
            if opt.fitsky:
                tractor.thawPathsTo('sky')
            # print 'Fitting params:'
            # tractor.printThawedParams()

            minsig = 0.1

            # making plots?
            #if celli <= 10:
            #    mod0 = tractor.getModelImage(0)

            # Forced photometry
            X = tractor.optimize_forced_photometry(
                #minsb=minsig*sigma, mindlnp=1., minFlux=None,
                variance=True, fitstats=True, shared_params=False,
                sky=opt.fitsky,
                use_ceres=True, BW=8, BH=8)
            IV = X.IV
            fs = X.fitstats

            print('Forced photometry took', Time()-t0)
            
            # print 'Fit params:'
            # tractor.printThawedParams()

            # Record results
            X = np.zeros(len(T), np.float32)
            X[catI] = np.array([src.getBrightness().getBand(band) for src in cat]).astype(np.float32)
            T.set('tractor_%s_nanomaggies' % band, X)
            X = np.zeros(len(T), np.float32)
            X[catI] = IV.astype(np.float32)
            T.set('tractor_%s_nanomaggies_invvar' % band, X)
            X = np.zeros(len(T), bool)
            X[catI] = True
            T.set('tractor_%s_has_phot' % band, X)

            # DEBUG
            X = np.zeros(len(T), np.float64)
            X[catI] = np.array([src.getPosition().ra for src in cat])
            T.tractor_ra = X
            X = np.zeros(len(T), np.float64)
            X[catI] = np.array([src.getPosition().dec for src in cat])
            T.tractor_dec = X

            T.cell = np.zeros(len(T), int) + celli
            if fs is not None:
                # Per-source stats
                for k in ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
                    T.set(k, getattr(fs, k))
                # Per-image stats
                for k in imstatkeys:
                    X = getattr(fs, k)
                    imstats.get(k)[celli] = X[0]

            #T.about()
            # DEBUG
            ## KK = np.flatnonzero(T.tractor_u_nanomaggies[catI] > 3.)
            ## T.cut(catI[KK])
            ## cat = [cat[k] for k in KK]
            ## catI = np.arange(len(cat))
            ## #T.about()
            ## print T.tractor_u_nanomaggies
            ## print T.psfflux[:,0]

            results.append(T.copy())

            # tc = T.copy()
            # print 'tc'
            # print tc.tractor_u_nanomaggies
            # print tc.psfflux[:,0]
            # plot_results(None, ps, tc)
            # mc = merge_tables([x.copy() for x in results])
            # print 'Results:'
            # for x in results:
            #     print x.tractor_u_nanomaggies
            #     print x.psfflux[:,0]
            # print 'Merged'
            # print mc.tractor_u_nanomaggies
            # print mc.psfflux[:,0]
            # plot_results(None, ps, mc)

            # Make plots for the first N cells
            if celli >= 10:
                continue
    
            mod = tractor.getModelImage(0)
            ima = dict(interpolation='nearest', origin='lower',
                       vmin=sky + -2. * sigma, vmax=sky + 5. * sigma,
                       cmap='gray', extent=[ix0-0.5, ix1-0.5, iy0-0.5, iy1-0.5])

            ok,rc,dc = wcs.pixelxy2radec((ix0+ix1)/2., (iy0+iy1)/2.)

            plt.clf()
            plt.imshow(img, **ima)
            plt.title('Data: ~ (%.3f, %.3f)' % (rc,dc))
            #ps.savefig()

            ax = plt.axis()
            plt.plot(T.x-1, T.y-1, 'o', mec='r', mfc='none', ms=10)
            plt.axis(ax)
            plt.title('Data + SDSS sources ~ (%.3f, %.3f)' % (rc,dc))
            ps.savefig()

            flim = 2.5
            I = np.flatnonzero(T.psfflux[catI,0] > flim)
            for ii in I:
                tind = catI[ii]
                src = cat[ii]
                fluxes = [T.psfflux[tind,0], src.getBrightness().getBand(band)]
                print('Fluxes', fluxes)
                mags = [-2.5*(np.log10(flux)-9) for flux in fluxes]
                print('Mags', mags)

                t = ''
                if type(src) == ExpGalaxy:
                    t = 'E'
                elif type(src) == DevGalaxy:
                    t = 'D'
                elif type(src) == PointSource:
                    t = 'S'
                elif type(src) == FixedCompositeGalaxy:
                    t = 'C'
                else:
                    t = str(type(src))

                plt.text(T.x[tind], T.y[tind]+3, '%.1f / %.1f %s' % (mags[0], mags[1], t), color='r',
                         va='bottom',
                         bbox=dict(facecolor='k', alpha=0.5))
                plt.plot(T.x[tind]-1, T.y[tind]-1, 'rx')

            for i,src in enumerate(cat):
                flux = src.getBrightness().getBand(band)
                if flux < flim:
                    continue
                tind = catI[i]
                fluxes = [T.psfflux[tind,0], flux]
                print('RA,Dec', T.ra[tind],T.dec[tind])
                print(src.getPosition())
                print('Fluxes', fluxes)
                mags = [-2.5*(np.log10(flux)-9) for flux in fluxes]
                print('Mags', mags)
                plt.text(T.x[tind], T.y[tind]-3, '%.1f / %.1f' % (mags[0], mags[1]), color='g',
                         va='top', bbox=dict(facecolor='k', alpha=0.5))
                plt.plot(T.x[tind]-1, T.y[tind]-1, 'g.')
                         
            plt.axis(ax)
            ps.savefig()

            # plt.clf()
            # plt.imshow(mod0, **ima)
            # plt.title('Initial Model')
            # #plt.colorbar()
            # ps.savefig()

            # plt.clf()
            # plt.imshow(mod0, interpolation='nearest', origin='lower',
            #            cmap='gray', extent=[ix0-0.5, ix1-0.5, iy0-0.5, iy1-0.5])
            # plt.title('Initial Model')
            # plt.colorbar()
            # ps.savefig()

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
            plt.imshow(chi, interpolation='nearest', origin='lower',
                       cmap='RdBu', vmin=-5, vmax=5)
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
    print('Wrote results to', opt.outfn)
    
    if opt.statsfn:
        imstats.writeto(opt.statsfn)
        print('Wrote image statistics to', opt.statsfn)

    plot_results(opt.outfn, ps)


    
def plot_results(outfn, ps, T=None):
    if T is None:
        T = fits_table(outfn)
        print('read', len(T))

    I = np.flatnonzero(T.tractor_u_has_phot)
    print('Plotting', len(I), 'with phot')

    stars = (T.objc_type[I] == 6)
    gals  = (T.objc_type[I] == 3)

    # SDSS measurements
    nm = np.zeros(len(I))
    nm[stars] = T.psfflux  [I[stars],0]
    nm[gals ] = T.modelflux[I[gals ],0]

    # Tractor measurements
    counts  = T.tractor_u_nanomaggies[I]
    dcounts = 1./np.sqrt(T.tractor_u_nanomaggies_invvar[I])

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

    xx, dc = [],[]
    xxmags = []
    for mag in [24,23,22,21,20,19,18,17,16,15,14,13,12]:
        tnm = 10.**((mag - 22.5)/-2.5)
        if (mag > 12):
            nmlo = tnm / np.sqrt(2.5)
            nmhi = tnm * np.sqrt(2.5)
            K = np.flatnonzero((counts > nmlo) * (counts < nmhi))
            xx.append(tnm)
            xxmags.append(mag)
            dc.append(np.median(dcounts[K]))
    xx = np.array(xx)
    dc = np.array(dc)

    if False:
        plt.clf()
        plt.loglog(np.maximum(1e-2, nm[stars]), np.maximum(1e-2, counts[stars]), 'b.', ms=5, alpha=0.5)
        plt.loglog(np.maximum(1e-2, nm[gals] ), np.maximum(1e-2, counts[gals ]), 'g.', ms=5, alpha=0.5)
        plt.xlabel('SDSS nanomaggies')
        plt.ylabel('Tractor nanomaggies')
        plt.title('Tractor forced photometry of SCUSS data')
        ax = plt.axis()
        plt.axhline(1e-2, color='r', alpha=0.5)
        plt.axvline(1e-2, color='r', alpha=0.5)
        mx = max(ax[1],ax[3])
        plt.plot([1e-2,mx], [1e-2,mx], 'b-', alpha=0.25, lw=2)
    
        for tnm,mag in zip(xx, xxmags):
            plt.axvline(tnm, color='k', alpha=0.25)
            plt.text(tnm*1.05, 3e4, '%i mag' % mag, ha='left', rotation=90, color='0.5')
        plt.errorbar(xx, xx, yerr=dc, fmt=None, ecolor='r', elinewidth=2,
                     capsize=3) #, capthick=2)
        plt.plot([xx,xx],[xx-dc, xx+dc], 'r-')
    
        mm = np.arange(11, 27)
        nn = 10.**((mm - 22.5)/-2.5)
        plt.xticks(nn, ['%i' % i for i in mm])
        plt.yticks(nn, ['%i' % i for i in mm])
    
        plt.xlim(0.8e-2, ax[1])
        plt.ylim(0.8e-2, ax[3])
        ps.savefig()



    lo,hi = 11, 26
    smag = np.clip(-2.5 * (np.log10(nm)     - 9), lo,hi)
    tmag = np.clip(-2.5 * (np.log10(counts) - 9), lo,hi)
    dt = np.abs((-2.5 / np.log(10.)) * dc / xx)
    xxmag = -2.5 * (np.log10(xx)-9)

    plt.clf()
    p1 = plt.plot(smag[stars], tmag[stars], 'b.', ms=5, alpha=0.5)
    p2 = plt.plot(smag[gals] , tmag[gals ], 'g.', ms=5, alpha=0.5)
    plt.xlabel('SDSS mag')
    plt.ylabel('Tractor mag')
    plt.title('Tractor forced photometry of SCUSS data')
    plt.plot([lo,hi],[lo,hi], 'b-', alpha=0.25, lw=2)
    plt.axis([hi,lo,hi,lo])
    plt.legend((p1[0],p2[0]), ('Stars','Galaxies'), loc='lower right')
    plt.errorbar(xxmag, xxmag, dt, fmt=None, ecolor='r', elinewidth =2, capsize=3)
    plt.plot([xxmag,xxmag],[xxmag-dt, xxmag+dt], 'r-')
    ps.savefig()


    plt.clf()
    p1 = plt.plot(smag[stars], tmag[stars] - smag[stars], 'b.', ms=5, alpha=0.5)
    p2 = plt.plot(smag[gals] , tmag[gals ] - smag[gals ], 'g.', ms=5, alpha=0.5)
    plt.xlabel('SDSS mag')
    plt.ylabel('Tractor mag - SDSS mag')
    plt.title('Tractor forced photometry of SCUSS data')
    plt.axhline(0, color='b', alpha=0.25, lw=2)
    plt.axis([hi,lo,-1,1])
    plt.legend((p1[0],p2[0]), ('Stars','Galaxies'), loc='lower right')
    plt.errorbar(xxmag, np.zeros_like(xxmag), dt, fmt=None, ecolor='r', elinewidth=2, capsize=3)
    plt.plot([xxmag,xxmag],[-dt, +dt], 'r-')
    ps.savefig()


    # lo,hi = -2,5
    # plt.clf()
    # loghist(np.clip(np.log10(nm),lo,hi), np.clip(np.log10(counts), lo, hi), 200,
    #         range=((lo-0.1,hi+0.1),(lo-0.1,hi+0.1)))
    # plt.xlabel('SDSS nanomaggies')
    # plt.ylabel('Tractor nanomaggies')
    # plt.title('Tractor forced photometry of SCUSS data')
    # ps.savefig()

    if True:
        # Cut to valid/bright ones
        I = np.flatnonzero((nm > 1e-2) * (counts > 1e-2))
        J = np.flatnonzero((nm > 1) * (counts > 1e-2))
        # Estimate zeropoint
        med = np.median(counts[J] / nm[J])
    
        plt.clf()
        plt.loglog(nm[I], counts[I]/nm[I], 'b.', ms=5, alpha=0.25)
        plt.xlabel('SDSS nanomaggies')
        plt.ylabel('Tractor nanomaggies / SDSS nanomaggies')
        plt.title('Tractor forced photometry of SCUSS data')
        ax = plt.axis()
        #plt.axhline(med, color='k', alpha=0.5)
        plt.axhline(1, color='k', alpha=0.5)
        plt.axis(ax)
        plt.ylim(0.1, 10.)
        ps.savefig()


    C = fits_table('data/scuss-w1-images/photozCFHTLS-W1_270912.fits',
                   columns=['alpha','delta','u','eu', 'g', 'stargal', 'ebv'])
    Cfull = C
    Tfull = T

    if False:
        # Johan's mag vs error plots
        #plt.clf()
        #ha = dict(range=((19,26),(-3,0)))#, doclf=False)
        ha = dict(range=((19,26),(np.log10(5e-2), np.log10(0.3))))
        #plt.subplot(2,2,1)
        loghist(Cfull.u, np.log10(Cfull.eu), 100, **ha)
        plt.xlabel('u (mag)')
        plt.ylabel('u error (mag)')
        plt.title('CFHT')
        yt = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]
        plt.yticks(np.log10(yt), ['%g'%y for y in yt])
        ps.savefig()
        #plt.subplot(2,2,2)
        su = -2.5*(np.log10(Tfull.modelflux[:,0])-9)
        se = np.abs((-2.5 / np.log(10.)) * (1./np.sqrt(Tfull.modelflux_ivar[:,0])) / Tfull.modelflux[:,0])
        loghist(su, np.log10(se), 100, **ha)
        plt.xlabel('u (mag)')
        plt.ylabel('u error (mag)')
        plt.yticks(np.log10(yt), ['%g'%y for y in yt])
        plt.title('SDSS')
        ps.savefig()
        c = Tfull.tractor_u_nanomaggies
        d = 1./np.sqrt(Tfull.tractor_u_nanomaggies_invvar)
        tu = -2.5 * (np.log10(c) - 9)
        te = np.abs((-2.5 / np.log(10.)) * d / c)
        #plt.subplot(2,2,3)
        loghist(tu, np.log10(te), 100, **ha)
        plt.xlabel('u (mag)')
        plt.ylabel('u error (mag)')
        plt.yticks(np.log10(yt), ['%g'%y for y in yt])
        plt.title('SCUSS')
        ps.savefig()
    

    # (don't use T.cut(): const T)
    T = T[T.tractor_u_has_phot]

    print('C stargal:', np.unique(C.stargal))

    I,J,d = match_radec(T.ra, T.dec, C.alpha, C.delta, 1./3600.)

    C = C[J]
    T = T[I]

    stars = (T.objc_type == 6)
    gals  = (T.objc_type == 3)

    #stars = (C.stargal == 1)
    #gals  = (C.stargal == 0)

    counts  = T.tractor_u_nanomaggies
    dcounts = 1./np.sqrt(T.tractor_u_nanomaggies_invvar)

    sdssu = -2.5*(np.log10(T.modelflux[:,0])-9)
    tmag = -2.5 * (np.log10(counts) - 9)
    dt = np.abs((-2.5 / np.log(10.)) * dcounts / counts)

    #cmag = C.u + 0.241 * (C.u - C.g)

    sdssu = -2.5*(np.log10(T.psfflux[:,0])-9)
    sdssg = -2.5*(np.log10(T.psfflux[:,1])-9)

    sdssugal = -2.5*(np.log10(T.modelflux[:,0])-9)

    #sdssu = -2.5*(np.log10(T.modelflux[:,0])-9)
    #sdssg = -2.5*(np.log10(T.modelflux[:,1])-9)
    #cmag = C.u
    #cmag = C.u + 0.241 * (sdssu - sdssg)
    #cmag += (4.705 * C.ebv)

    def _comp_plot(smag, cmag, tt, xname='SDSS', yname='CFHTLS'):
        plt.clf()
        lo,hi = 13, 26
        # p1 = plt.plot(cmag[stars], smag[stars], 'b.', ms=5, alpha=0.5)
        # p2 = plt.plot(cmag[gals] , smag[gals ], 'g.', ms=5, alpha=0.5)
        # plt.xlabel('CFHTLS u mag')
        # plt.ylabel('SDSS u mag')
        p1 = plt.plot(smag[stars], cmag[stars] - smag[stars], 'b.', ms=5, alpha=0.5)
        p2 = plt.plot(smag[gals ], cmag[gals]  - smag[gals ], 'g.', ms=5, alpha=0.5)
        plt.xlabel('%s u mag' % xname)
        plt.ylabel('%s u mag - %s u mag' % (yname,xname))
        plt.title(tt)
        #plt.plot([lo,hi],[lo,hi], 'b-', alpha=0.25, lw=2)
        #plt.axis([hi,lo,hi,lo])
        plt.axhline(0, color='b', alpha=0.25, lw=2)
        plt.axis([hi,lo,-2,2])
        plt.legend((p1[0],p2[0]), ('Stars','Galaxies'), loc='lower right')
        ps.savefig()

    smag = sdssu
    cmag = C.u

    eu = 4.705 * C.ebv
    ct = 0.241 * (sdssu - sdssg)

    sboth = np.zeros_like(smag)
    sboth[stars] = sdssu[stars]
    sboth[gals] = sdssugal[gals]

    _comp_plot(smag, cmag, 'CFHTLS vs SDSS -- raw (PSF)')

    _comp_plot(sdssugal, cmag, 'CFHTLS vs SDSS -- raw (model)')

    _comp_plot(sboth, cmag, 'CFHTLS vs SDSS -- raw')

    _comp_plot(sboth, cmag + eu, 'CFHTLS vs SDSS -- un-extincted')

    _comp_plot(sboth, cmag + ct, 'CFHTLS vs SDSS -- color term')

    #_comp_plot(sboth, cmag + ct + eu, 'CFHTLS vs SDSS -- un-extincted, color term')
    #_comp_plot(sboth, cmag - eu, 'CFHTLS vs SDSS -- -un-extincted')
    #_comp_plot(sboth, cmag + ct - eu, 'CFHTLS vs SDSS -- -un-extincted, color term')

    _comp_plot(cmag + ct, tmag, 'CFHTLS+ct vs Tractor(SCUSS)',
               xname='CFHTLS', yname='Tractor(SCUSS)')
    _comp_plot(cmag, tmag, 'CFHTLS vs Tractor(SCUSS)',
               xname='CFHTLS', yname='Tractor(SCUSS)')
    _comp_plot(sboth, tmag, 'SDSS vs Tractor(SCUSS)',
               xname='SDSS', yname='Tractored SCUSS')

    plt.clf()
    keep = ((cmag > 17) * (cmag < 22))
    plt.plot((sdssu-sdssg)[keep * stars], (tmag - cmag)[keep * stars], 'b.',
             ms=5, alpha=0.5)
    plt.plot((sdssu-sdssg)[keep *  gals], (tmag - cmag)[keep *  gals], 'g.',
             ms=5, alpha=0.5)
    plt.xlabel('SDSS u-g')
    plt.ylabel('Tractor(SCUSS) - CFHTLS')
    plt.axis([-1,4,-1,1])
    ps.savefig()
    
    # I = np.flatnonzero((sboth < 16) * ((cmag + ct - sboth) > 0.25))
    # for r,d in zip(T.ra[I], T.dec[I]):
    #     print 'RA,Dec', r,d
    #     print 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=%.5f&dec=%.5f&width=100&height=100' % (r,d)

    # ???
    # sdssu -= T.extinction[:,0]
    # sdssg -= T.extinction[:,1]
    # tmag -= T.extinction[:,0]

    xxmag = np.arange(13, 26)
    dx = []
    dc = []
    for xx in xxmag:
        ii = np.flatnonzero((tmag > xx-0.5) * (tmag < xx+0.5))
        dx.append(np.median(dt[ii]))
        ii = np.flatnonzero((cmag > xx-0.5) * (cmag < xx+0.5))
        dc.append(np.median(C.eu[ii]))
    dc = np.array(dc)
    dx = np.array(dx)
    
    plt.clf()
    lo,hi = 13, 26
    p1 = plt.plot(cmag[stars], tmag[stars], 'b.', ms=5, alpha=0.5)
    p2 = plt.plot(cmag[gals] , tmag[gals ], 'g.', ms=5, alpha=0.5)
    plt.xlabel('CFHTLS mag')
    plt.ylabel('Tractor mag')
    plt.title('Tractor forced photometry of SCUSS data')
    plt.plot([lo,hi],[lo,hi], 'b-', alpha=0.25, lw=2)
    plt.axis([hi,lo,hi,lo])
    plt.legend((p1[0],p2[0]), ('Stars','Galaxies'), loc='lower right')
    plt.errorbar(xxmag, xxmag, dx, fmt=None, ecolor='r', elinewidth =2, capsize=3)
    plt.plot([xxmag,xxmag],[xxmag-dx, xxmag+dx], 'r-')
    dd = 0.1
    plt.errorbar(xxmag+dd, xxmag, dc, fmt=None, ecolor='m', elinewidth =2, capsize=3)
    plt.plot([xxmag+dd,xxmag+dd],[xxmag-dc, xxmag+dc], 'm-')
    ps.savefig()

    plt.clf()
    p1 = plt.plot(cmag[stars], tmag[stars] - cmag[stars], 'b.', ms=5, alpha=0.5)
    p2 = plt.plot(cmag[gals] , tmag[gals ] - cmag[gals ], 'g.', ms=5, alpha=0.5)
    plt.xlabel('CFHTLS mag')
    plt.ylabel('Tractor mag - CFHTLS mag')
    plt.title('Tractor forced photometry of SCUSS data')
    plt.axhline(0, color='b', alpha=0.25, lw=2)
    plt.axis([hi,lo,-1,1.5])
    plt.legend((p1[0],p2[0]), ('Stars','Galaxies'), loc='lower right')
    plt.errorbar(xxmag, np.zeros_like(xxmag), dx, fmt=None, ecolor='r', elinewidth=2, capsize=3)
    plt.plot([xxmag,xxmag],[-dx, +dx], 'r-')
    plt.errorbar(xxmag+dd, np.zeros_like(xxmag), dc, fmt=None, ecolor='m', elinewidth =2, capsize=3)
    plt.plot([xxmag+dd,xxmag+dd],[-dc, +dc], 'm-')
    ps.savefig()

    


    
if __name__ == '__main__':
    main()
    
