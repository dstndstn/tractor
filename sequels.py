#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
from glob import glob

import fitsio

# qsub -d $(pwd) -N sequels -l "cput=2:00:00" -l "nodes=1:ppn=1" -l "pvmem=4gb" -o sequels.log -t 0-99 ./sequels.py

if __name__ == '__main__':
    arr = os.environ.get('PBS_ARRAYID')
    d = os.environ.get('PBS_O_WORKDIR')
    if arr is not None and d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *
from astrometry.util.sdss_radec_to_rcf import *

from tractor import *
from tractor.ttime import *
from tractor.sdss import *

from wisecat import wise_catalog_radecbox

import logging


'''
ln -s /clusterfs/riemann/raid007/ebosswork/eboss/photoObj photoObjs-new
ln -s /clusterfs/riemann/raid006/bosswork/boss/resolve/2013-07-29 photoResolve-new
'''

tiledir = 'wise-coadds'

photoobjdir = 'photoObjs-new'
resolvedir = 'photoResolve-new'

outdir = 'sequels-phot'
tempoutdir = 'sequels-phot-temp'
pobjoutdir = 'sequels-pobj'

Time.add_measurement(MemMeas)

def get_photoobj_filename(rr, run, camcol, field):
    fn = os.path.join(photoobjdir, rr, '%i'%run, '%i'%camcol,
                      'photoObj-%06i-%i-%04i.fits' % (run, camcol, field))
    return fn

def estimate_sky(img, iv):
    sim = np.sort(img.ravel())
    sigest = sim[int(0.5 * len(sim))] - sim[int(0.16 * len(sim))]
    #print 'sig est', sigest
    nsig = 0.1 * sigest

    I = np.linspace(0.3*len(sim), 0.55*len(sim), 15).astype(int)
    sumn = []
    for ii in I:
        X = sim[ii]
        sumn.append(sum((sim > X-nsig) * (sim < X+nsig)))
    sumn = np.array(sumn)

    iscale = 0.5 * len(sim)
    i0 = I[len(I)/2]
    xi = (I - i0) / iscale

    A = np.zeros((len(I), 3))
    A[:,0] = 1.
    A[:,1] = xi
    A[:,2] = xi**2

    b = sumn

    res = np.linalg.lstsq(A, b)
    X = res[0]
    #print 'X', X
    Imax = -X[1] / (2. * X[2])
    Imax = (Imax * iscale) + i0
    i = int(np.round(Imax))
    #print 'Imax', Imax
    mu = sim[i]
    #print 'mu', mu
    return mu


def read_photoobjs(r0, r1, d0, d1, margin):

    wfn = os.path.join(resolvedir, 'window_flist.fits')
    #W = fits_table(wfn)

    ra,dec = (r0+r1)/2., (d0+d1)/2.

    rad = degrees_between(ra,dec, r0,d0)
    rad += np.hypot(13., 9.)/60.
    # a little extra margin
    rad += margin

    #I = np.flatnonzero(distsq_between_radecs(ra, dec, W.ra, W.dec) <= rad)
    #print 'Found', len(I), 'fields possibly in range'

    RCF = radec_to_sdss_rcf(ra, dec, radius=rad*60., tablefn=wfn)
    print 'Found', len(RCF), 'fields possibly in range'

    ddec = margin
    dra  = margin / min([np.cos(np.deg2rad(x)) for x in [d0,d1]])

    TT = []
    sdss = DR9()
    for run,camcol,field,r,d in RCF:
        print 'RCF', run, camcol, field
        rr = sdss.get_rerun(run, field=field)
        if rr in [None, '157']:
            print 'Rerun 157'
            continue

        fn = get_photoobj_filename(rr, run, camcol, field)

        cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type', 'modelflux',
                'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
                'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
                'resolve_status', 'nchild', 'flags', 'objc_flags',
                'run','camcol','field','id'
                ]
        T = fits_table(fn, columns=cols)
        if T is None:
            print 'read 0 from', fn
            continue
        print 'read', len(T), 'from', fn
        T.cut((T.ra  >= (r0-dra )) * (T.ra  <= (r1+dra)) *
              (T.dec >= (d0-ddec)) * (T.dec <= (d1+ddec)) *
              ((T.resolve_status & 256) > 0))
        print 'cut to', len(T), 'in RA,Dec box and PRIMARY.'
        if len(T) == 0:
            continue
        TT.append(T)
    T = merge_tables(TT)
    return T

def one_tile(tile, opt, savepickle):
    bands = opt.bands
    outfn = opt.output % (tile.coadd_id)

    sband = 'r'
    bandnum = 'ugriz'.index(sband)

    tt0 = Time()
    print
    print 'Coadd tile', tile.coadd_id

    fn = os.path.join(tiledir, 'coadd-%s-w%i-img.fits' % (tile.coadd_id, bands[0]))
    print 'Reading', fn
    wcs = Tan(fn)
    r0,r1,d0,d1 = wcs.radec_bounds()
    print 'RA,Dec bounds:', r0,r1,d0,d1
    H,W = wcs.get_height(), wcs.get_width()

    objfn = os.path.join(tempoutdir, 'photoobjs-%s.fits' % tile.coadd_id)
    if os.path.exists(objfn):
        print 'Reading', objfn
        T = fits_table(objfn)
    else:
        T = read_photoobjs(r0, r1, d0, d1, 1./60.)
        T.writeto(objfn)

    if opt.photoObjsOnly:
        return

    # Cut galaxies based on signal-to-noise of theta (effective
    # radius) measurement.
    b = bandnum
    gal = (T.objc_type == 3)
    dev = gal * (T.fracdev[:,b] >= 0.5)
    exp = gal * (T.fracdev[:,b] <  0.5)
    stars = (T.objc_type == 6)
    print sum(dev), 'deV,', sum(exp), 'exp, and', sum(stars), 'stars'
    print 'Total', len(T), 'sources'

    thetasn = np.zeros(len(T))
    T.theta_deverr[dev,b] = np.maximum(1e-6, T.theta_deverr[dev,b])
    T.theta_experr[exp,b] = np.maximum(1e-5, T.theta_experr[exp,b])
    # theta_experr nonzero: 1.28507e-05
    # theta_deverr nonzero: 1.92913e-06
    thetasn[dev] = T.theta_dev[dev,b] / T.theta_deverr[dev,b]
    thetasn[exp] = T.theta_exp[exp,b] / T.theta_experr[exp,b]

    aberrzero = np.zeros(len(T), bool)
    aberrzero[dev] = (T.ab_deverr[dev,b] == 0.)
    aberrzero[exp] = (T.ab_experr[exp,b] == 0.)

    maxtheta = np.zeros(len(T), bool)
    maxtheta[dev] = (T.theta_dev[dev,b] >= 29.5)
    maxtheta[exp] = (T.theta_exp[exp,b] >= 59.0)

    # theta S/N > modelflux for dev, 10*modelflux for exp
    bigthetasn = (thetasn > (T.modelflux[:,b] * (1.*dev + 10.*exp)))

    print sum(gal * (thetasn < 3.)), 'have low S/N in theta'
    print sum(gal * (T.modelflux[:,b] > 1e4)), 'have big flux'
    print sum(aberrzero), 'have zero a/b error'
    print sum(maxtheta), 'have the maximum theta'
    print sum(bigthetasn), 'have large theta S/N vs modelflux'
    
    badgals = gal * reduce(np.logical_or,
                           [thetasn < 3.,
                            T.modelflux[:,b] > 1e4,
                            aberrzero,
                            maxtheta,
                            bigthetasn,
                            ])
    print 'Found', sum(badgals), 'bad galaxies'
    T.treated_as_pointsource = badgals
    T.objc_type[badgals] = 6

    defaultflux = 100.

    # hack
    T.psfflux    = np.zeros((len(T),5), np.float32) + defaultflux
    T.cmodelflux = T.psfflux
    T.devflux    = T.psfflux
    T.expflux    = T.psfflux

    ok,T.x,T.y = wcs.radec2pixelxy(T.ra, T.dec)
    T.x = (T.x - 1.).astype(np.float32)
    T.y = (T.y - 1.).astype(np.float32)
    margin = 20.
    I = np.flatnonzero((T.x >= -margin) * (T.x < W+margin) *
                       (T.y >= -margin) * (T.y < H+margin))
    T.cut(I)
    print 'N objects:', len(T)

    wanyband = wband = 'w'
    print 'Creating tractor sources...'
    cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                  objs=T, bands=[], nanomaggies=True,
                                  extrabands=[wband],
                                  fixedComposites=True,
                                  useObjcType=True)
    print 'Created', len(T), 'sources'
    assert(len(cat) == len(T))

    pixscale = wcs.pixel_scale()
    # crude intrinsic source radii, in pixels
    sourcerad = np.zeros(len(cat))
    for i in range(len(cat)):
        src = cat[i]
        if isinstance(src, PointSource):
            continue
        elif isinstance(src, HoggGalaxy):
            sourcerad[i] = (src.nre * src.re / pixscale)
            #src.halfsize = sourcerad[i]
        elif isinstance(src, FixedCompositeGalaxy):
            sourcerad[i] = max(src.shapeExp.re * ExpGalaxy.nre,
                               src.shapeDev.re * DevGalaxy.nre) / pixscale
            #src.halfsize = sourcerad[i]
    print 'sourcerad range:', min(sourcerad), max(sourcerad)

    wfn = os.path.join(tempoutdir, 'wise-sources-%s.fits' % (tile.coadd_id))
    print 'looking for', wfn
    if os.path.exists(wfn):
        WISE = fits_table(wfn)
        print 'Read', len(WISE), 'WISE sources nearby'
    else:
        cols = ['ra','dec'] + ['w%impro'%band for band in [1,2,3,4]]
        WISE = wise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
        WISE.writeto(wfn)
        print 'Found', len(WISE), 'WISE sources nearby'

    for band in bands:
        WISE.set('w%inm' % band,
                 NanoMaggies.magToNanomaggies(WISE.get('w%impro' % band)))

    unmatched = np.ones(len(WISE), bool)
    I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 4./3600.)
    unmatched[I] = False
    UW = WISE[unmatched]
    print 'Got', len(UW), 'unmatched WISE sources'

    # Record WISE fluxes for catalog matches.
    # (this provides decent initialization for 'minsb' approx.)
    wiseflux = {}
    for band in bands:
        wiseflux[band] = np.zeros(len(T))
        # X[I] += Y[J] with duplicate I doesn't work.
        #wiseflux[band][J] += WISE.get('w%inm' % band)[I]
        lhs = wiseflux[band]
        rhs = WISE.get('w%inm' % band)[I]
        for j,f in zip(J, rhs):
            lhs[j] += f

    ok,UW.x,UW.y = wcs.radec2pixelxy(UW.ra, UW.dec)
    UW.x -= 1.
    UW.y -= 1.

    T.coadd_id = np.array([tile.coadd_id] * len(T))
    T.cell = np.zeros(len(T), np.int16)
    T.cell_x0 = np.zeros(len(T), np.int16)
    T.cell_y0 = np.zeros(len(T), np.int16)
    T.cell_x1 = np.zeros(len(T), np.int16)
    T.cell_y1 = np.zeros(len(T), np.int16)

    inbounds = np.flatnonzero((T.x >= -0.5) * (T.x < W-0.5) *
                              (T.y >= -0.5) * (T.y < H-0.5))

    for band in bands:
        tb0 = Time()
        print
        print 'Coadd tile', tile.coadd_id
        print 'Band', band
        wband = 'w%i' % band

        imfn = os.path.join(tiledir, 'coadd-%s-w%i-img-w.fits'    % (tile.coadd_id, band))
        ivfn = os.path.join(tiledir, 'coadd-%s-w%i-invvar-w.fits' % (tile.coadd_id, band))

        print 'Reading', imfn
        wcs = Tan(imfn)
        r0,r1,d0,d1 = wcs.radec_bounds()
        print 'RA,Dec bounds:', r0,r1,d0,d1
        ra,dec = wcs.radec_center()
        print 'Center:', ra,dec
        img = fitsio.read(imfn)
        print 'Reading', ivfn
        iv = fitsio.read(ivfn)

        minsb = getattr(opt, 'minsb%i' % band)
        print 'Minsb:', minsb

        # Load the average PSF model
        P = fits_table('wise-psf-avg.fits', hdu=band)
        # Instantiate a (non-varying) mixture-of-Gaussians PSF
        psf = GaussianMixturePSF(P.amp, P.mean, P.var)

        # Render the PSF profile for figuring out source radii for
        # approximation purposes.
        R = 100
        psf.radius = R
        pat = psf.getPointSourcePatch(0., 0.)
        assert(pat.x0 == pat.y0)
        assert(pat.x0 == -R)
        psfprofile = pat.patch[R, R:]
        #print 'Profile:', psfprofile

        # Set WISE source radii based on flux
        UW.rad = np.zeros(len(UW))
        wnm = UW.get('w%inm' % band)
        for r,pro in enumerate(psfprofile):
            flux = minsb / pro
            UW.rad[wnm > flux] = r
        # plt.clf()
        # plt.semilogy(UW.rad, wnm, 'b.')
        # plt.xlabel('radius')
        # plt.ylabel('flux')
        # ps.savefig()
        UW.rad = np.maximum(UW.rad + 1, 3.)

        # Increase SDSS source radii based on WISE catalog-matched fluxes.
        wf = wiseflux[band]
        I = np.flatnonzero(wf > defaultflux)
        wfi = wf[I]
        print 'Initializing', len(I), 'fluxes based on catalog matches'
        for i,flux in zip(I, wf[I]):
            assert(np.isfinite(flux))
            cat[i].getBrightness().setBand(wanyband, flux)
        rad = np.zeros(len(I))
        drad = 0.
        for r,pro in enumerate(psfprofile):
            flux = minsb / pro
            rad[wfi > flux] = r
            if defaultflux > flux:
                drad = r
        print 'default source radius:', drad
        # these are radii the SDSS sources would have based on their WISE
        # catalog-matched PSF-source size.
        srad2 = np.zeros(len(cat))
        srad2[I] = rad
        #sourcerad = np.maximum(drad, np.maximum(sourcerad, srad2))
        for i in range(len(cat)):
            src = cat[i]
            R = max([drad, sourcerad[i], srad2[i]])
            if isinstance(src, PointSource):
                src.radius = R
            elif (isinstance(src, HoggGalaxy) or
                  isinstance(src, FixedCompositeGalaxy)):
                src.halfsize = R

        # We're going to dice the image up into cells for
        # photometry... remember the whole image and initialize
        # whole-image results.
        fullimg = img
        fullinvvar = iv
        fullIV = np.zeros(len(cat))
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
        fitstats = dict([(k, np.zeros(len(cat))) for k in fskeys])

        twcs = ConstantFitsWcs(wcs)

        sky = estimate_sky(img, iv)
        print 'Estimated sky', sky
        tsky = ConstantSky(sky)

        # cell positions
        XX = np.round(np.linspace(0, W, opt.blocks+1)).astype(int)
        YY = np.round(np.linspace(0, H, opt.blocks+1)).astype(int)

        if savepickle:
            mods = []
            cats = []

        celli = -1
        for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
            for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
                celli += 1

                if opt.cell is not None and opt.cell != celli:
                    print 'Skipping cell', celli
                    continue

                print
                print 'Cell', celli, 'of', (opt.blocks**2), 'for', tile.coadd_id, 'band', wband

                imargin = 12
                # SDSS and WISE source margins beyond the image margins ( + source radii )
                smargin = 1
                wmargin = 1

                # image region: [ix0,ix1)
                ix0 = max(0, xlo - imargin)
                ix1 = min(W, xhi + imargin)
                iy0 = max(0, ylo - imargin)
                iy1 = min(H, yhi + imargin)
                slc = (slice(iy0, iy1), slice(ix0, ix1))
                print 'Image ROI', ix0, ix1, iy0, iy1
                img    = fullimg   [slc]
                invvar = fullinvvar[slc]
                twcs.setX0Y0(ix0, iy0)

                #I = np.flatnonzero(np.logical_not(np.isfinite(img)))
                #invvar.flat[I] = 0.
                #img.flat[I] = 0.

                tim = Image(data=img, invvar=invvar, psf=psf, wcs=twcs,
                            sky=tsky, photocal=LinearPhotoCal(1., band=wanyband),
                            name='Coadd %s W%i (%i,%i)' % (tile.coadd_id, band, xi,yi),
                            domask=False)

                # Relevant SDSS sources:
                m = smargin + sourcerad
                I = np.flatnonzero(((T.x+m) >= (ix0-0.5)) * ((T.x-m) < (ix1-0.5)) *
                                   ((T.y+m) >= (iy0-0.5)) * ((T.y-m) < (iy1-0.5)))
                inbox = ((T.x[I] >= (xlo-0.5)) * (T.x[I] < (xhi-0.5)) *
                         (T.y[I] >= (ylo-0.5)) * (T.y[I] < (yhi-0.5)))
                # Inside this cell
                srci = I[inbox]
                # In the margin
                margi = I[np.logical_not(inbox)]

                # sources in the ROI box
                subcat = [cat[i] for i in srci]

                # include *copies* of sources in the margins
                # (that way we automatically don't save the results)
                subcat.extend([cat[i].copy() for i in margi])
                assert(len(subcat) == len(I))

                # add WISE-only sources in the expanded region
                m = wmargin + UW.rad
                J = np.flatnonzero(((UW.x+m) >= (ix0-0.5)) * ((UW.x-m) < (ix1-0.5)) *
                                   ((UW.y+m) >= (iy0-0.5)) * ((UW.y-m) < (iy1-0.5)))
                wnm = UW.get('w%inm' % band)
                for j in J:
                    if not np.isfinite(wnm[j]):
                        continue
                    #assert(np.isfinite(wnm[j]))
                    ps = PointSource(RaDecPos(UW.ra[j], UW.ra[j]),
                                              NanoMaggies(**{wanyband: wnm[j]}))
                    ps.radius = UW.rad[j]
                    subcat.append(ps)
                print 'Sources:', len(srci), 'in the box,', len(I)-len(srci), 'in the margins, and', len(J), 'WISE-only'

                print 'Creating a Tractor with image', tim.shape, 'and', len(subcat), 'sources'
                tractor = Tractor([tim], subcat)

                print 'Running forced photometry...'
                t0 = Time()
                tractor.freezeParamsRecursive('*')
                # tractor.thawPathsTo('sky')
                tractor.thawPathsTo(wanyband)

                # DEBUG
                #p0 = tractor.getParams()

                ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
                    minsb=minsb, mindlnp=1., sky=False, minFlux=None,
                    fitstats=True, variance=True, shared_params=False)
                print 'That took', Time()-t0

                # tractor.setParams(p0)
                # 
                # t0 = Time()
                # ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
                #     minsb=minsb, mindlnp=1., sky=False, minFlux=None,
                #     fitstats=True, variance=False, shared_params=False,
                #     use_tsnnls=True)
                # print 'TSNNLS took', Time()-t0

                # Rinse sources with negative flux and repeat!
                # subcat2 = [src for src in subcat if src.getBrightness().getBand(wanyband) > 0.]
                # print 'Cut from', len(subcat), 'to', len(subcat2), 'non-neg sources'
                # tractor = Tractor([tim], subcat2)
                # print 'Running forced photometry...'
                # t0 = Time()
                # tractor.freezeParamsRecursive('*')
                # # tractor.thawPathsTo('sky')
                # tractor.thawPathsTo(wanyband)
                # ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
                #     minsb=minsb, mindlnp=1., sky=False, minFlux=None,
                #     fitstats=True, variance=True, shared_params=False)
                # print 'That took', Time()-t0
                # im,mod,ie,chi,roi = ims1[0]

                if savepickle:
                    if ims1 is None:
                        mod = None
                    else:
                        im,mod,ie,chi,roi = ims1[0]
                    mods.append(mod)
                    cats.append((
                        srci, margi, UW.x[J], UW.y[J],
                        T.x[srci], T.y[srci], T.x[margi], T.y[margi],
                        [src.copy() for src in cat],
                        [src.copy() for src in subcat]))

                if opt.pickle2:
                    fn = opt.output % (tile.coadd_id)
                    fn = fn.replace('.fits','-cell%02i.pickle' % celli)
                    pickle_to_file((ims0, ims1, cat, subcat), fn)
                    print 'Pickled', fn
                    

                if len(srci):
                    T.cell[srci] = celli
                    T.cell_x0[srci] = ix0
                    T.cell_x1[srci] = ix1
                    T.cell_y0[srci] = iy0
                    T.cell_y1[srci] = iy1
                    # Save fit stats
                    fullIV[srci] = IV[:len(srci)]
                    for k in fskeys:
                        x = getattr(fs, k)
                        fitstats[k][srci] = np.array(x)

                cpu0 = tb0.meas[0]
                t = Time()
                cpu = t.meas[0]
                dcpu = (cpu.cpu - cpu0.cpu)
                print 'So far:', Time()-tb0, '-> predict CPU time', (dcpu * (opt.blocks**2) / float(celli+1))

        nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
        nm_ivar = fullIV
        T.set(wband + '_nanomaggies', nm.astype(np.float32))
        T.set(wband + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
        dnm = np.zeros(len(nm_ivar), np.float32)
        okiv = (nm_ivar > 0)
        dnm[okiv] = (1./np.sqrt(nm_ivar[okiv])).astype(np.float32)
        okflux = (nm > 0)
        mag = np.zeros(len(nm), np.float32)
        mag[okflux] = (NanoMaggies.nanomaggiesToMag(nm[okflux])).astype(np.float32)
        dmag = np.zeros(len(nm), np.float32)
        ok = (okiv * okflux)
        dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dnm[ok] / nm[ok])).astype(np.float32)

        mag[np.logical_not(okflux)] = np.nan
        dmag[np.logical_not(ok)] = np.nan
        
        T.set(wband + '_mag', mag)
        T.set(wband + '_mag_err', dmag)
        for k in fskeys:
            T.set(wband + '_' + k, fitstats[k].astype(np.float32))

        print 'Tile', tile.coadd_id, 'band', wband, 'took', Time()-tb0

    T.cut(inbounds)

    T.delete_column('psfflux')
    T.delete_column('cmodelflux')
    T.delete_column('devflux')
    T.delete_column('expflux')
    #T.delete_column('index')

    T.writeto(outfn)

    if savepickle:
        fn = opt.output % (tile.coadd_id)
        fn = fn.replace('.fits','.pickle')
        pickle_to_file((mods, cats, T, sourcerad), fn)
        print 'Pickled', fn

    print 'Tile', tile.coadd_id, 'took', Time()-tt0
                

def main():
    import optparse

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--minsb1', dest='minsb1', default=0.1, type=float)
    parser.add_option('--minsb2', dest='minsb2', default=0.1, type=float)
    parser.add_option('--minsb3', dest='minsb3', default=0.1, type=float)
    parser.add_option('--minsb4', dest='minsb4', default=0.1, type=float)
    parser.add_option('--blocks', dest='blocks', default=10, type=int,
                      help='NxN number of blocks to cut the image into')
    parser.add_option('-o', dest='output', default=os.path.join(outdir, 'phot-%s.fits'))
    parser.add_option('-b', dest='bands', action='append', type=int, default=[],
                      help='Add WISE band (default: 1,2)')

    parser.add_option('--photoobjs-only', dest='photoObjsOnly',
                      action='store_true', default=False,
                      help='Ensure photoobjs file exists and then quit?')

    parser.add_option('-p', dest='pickle', default=False, action='store_true',
                      help='Save .pickle file for debugging purposes')
    parser.add_option('--pp', dest='pickle2', default=False, action='store_true',
                      help='Save .pickle file for each cell?')

    parser.add_option('--finish', dest='finish', default=False, action='store_true')

    parser.add_option('--summary', dest='summary', default=False, action='store_true')

    parser.add_option('--cell', dest='cell', default=None, type=int,
                      help='Just run one cell?')

    parser.add_option('-v', dest='verbose', default=False, action='store_true')

    opt,args = parser.parse_args()

    if len(opt.bands) == 0:
        opt.bands = [1,2]

    lvl = logging.INFO
    if opt.verbose:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    # sequels-atlas.fits: written by wise-coadd.py
    dataset = 'sequels'
    fn = '%s-atlas.fits' % dataset
    print 'Reading', fn
    T = fits_table(fn)

    # Read Atlas Image table
    # A = fits_table('wise_allsky_4band_p3as_cdd.fits', columns=['coadd_id', 'ra', 'dec'])
    # print len(A), 'atlas tiles'
    # D = np.unique(A.dec)
    # print len(D), 'unique Decs'
    # print D
    # print 'diffs:', np.diff(D)
    # for d in D:
    #     I = np.flatnonzero(A.dec == d)
    #     R = np.unique(A.ra[I])
    #     print 'Dec', d, 'has', len(R), 'unique RA:', R
    #     print 'diffs', np.diff(R)

    ps = PlotSequence(dataset + '-phot')

    if opt.summary:
        A = T

        plt.clf()
        missing = []
        for i in range(len(A)):
            r,d = A.ra[i], A.dec[i]
            dd = 1024 * 2.75 / 3600.
            dr = dd / np.cos(np.deg2rad(d))
            outfn = opt.output % (A.coadd_id[i])
            rr,dd = [r-dr,r-dr,r+dr,r+dr,r-dr], [d-dd,d+dd,d+dd,d-dd,d-dd]
            if not os.path.exists(outfn):
                missing.append((i,rr,dd,r,d))
            plt.plot(rr, dd, 'k-')
        for i,rr,dd,r,d in missing:
            plt.plot(rr, dd, 'r-')
            plt.text(r, d, '%i' % i, rotation=90, color='b', va='center', ha='center')
        plt.title('missing tiles')
        plt.axis([118, 212, 44,61])
        ps.savefig()

        print 'Missing tiles:', [m[0] for m in missing]


        rdfn = 'rd.fits'
        if not os.path.exists(rdfn):
            fns = glob(os.path.join(tempoutdir, 'photoobjs-*.fits'))
            fns.sort()
            TT = []
            for fn in fns:
                T = fits_table(fn, columns=['ra','dec'])
                print len(T), 'from', fn
                TT.append(T)
            T = merge_tables(TT)
            print 'Total of', len(T)
            T.writeto(rdfn)
        else:
            T = fits_table(rdfn)
            print 'Got', len(T), 'from', rdfn
        
        plt.clf()
        loghist(T.ra, T.dec, 500, range=((118,212),(44,61)))
        plt.xlabel('RA')
        plt.ylabel('Dec')
        ps.savefig()

        ax = plt.axis()
        for i in range(len(A)):
            r,d = A.ra[i], A.dec[i]
            dd = 1024 * 2.75 / 3600.
            dr = dd / np.cos(np.deg2rad(d))
            plt.plot([r-dr,r-dr,r+dr,r+dr,r-dr], [d-dd,d+dd,d+dd,d-dd,d-dd], 'r-')
        plt.axis(ax)
        ps.savefig()

        sys.exit(0)

    if opt.finish:
        # Find all *-phot.fits outputs
        # Determine which photoObj files are involved
        # Collate and resolve objs measured in multiple tiles
        # Expand into photoObj-parallel files
        if len(args):
            fns = args
        else:
            fns = glob(os.path.join(outdir, 'phot-*.fits'))
            fns.sort()
            print 'Found', len(fns), 'photometry output files'
        fieldmap = {}
        for ifn,fn in enumerate(fns):
            print 'Reading', (ifn+1), 'of', len(fns), fn
            cols = ['ra','dec',
                    'objid', 'index', 'x','y', 
                    'treated_as_pointsource', 'coadd_id']
            for band in opt.bands:
                for k in ['nanomaggies', 'nanomaggies_ivar', 'mag', 'mag_err',
                          'prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
                    cols.append('w%i_%s' % (band, k))
            rcfcols = ['run','camcol','field','id',]
            try:
                T = fits_table(fn, columns=cols + rcfcols)
            except:
                print 'Run,camcol,field columns not found; reading photoobjs file to get them.'
                print fn
                T = fits_table(fn, columns=cols)
                print 'Got', len(T), 'from', fn
                pfn = fn.replace('phot-', 'photoobjs-').replace(outdir, tempoutdir)
                print 'Reading', pfn
                P = fits_table(pfn, columns=rcfcols + ['objid'])
                print 'Got', len(P), 'from', pfn
                print 'Applying objid map...'
                objidmap = dict([(oid,i) for i,oid in enumerate(P.objid)])
                I = np.array([objidmap[oid] for oid in T.objid])
                P.cut(I)
                assert(len(P) == len(T))
                assert(np.all(P.objid == T.objid))
                for k in rcfcols:
                    T.set(k, P.get(k))
                
            print 'Read', len(T), 'entries'
            rcf = np.unique(zip(T.run, T.camcol, T.field))
            for run,camcol,field in rcf:
                if not (run,camcol,field) in fieldmap:
                    fieldmap[(run,camcol,field)] = []
                Tsub = T[(T.run == run) * (T.camcol == camcol) * (T.field == field)]
                #print len(Tsub), 'in', (run,camcol,field)
                for col in ['run','camcol','field']:
                    Tsub.delete_column(col)
                fieldmap[(run,camcol,field)].append(Tsub)

        for (run,camcol,field),TT in fieldmap.items():
            print len(TT), 'tiles for', (run,camcol,field)
            # HACK
            rr = '301'
            pofn = get_photoobj_filename(rr, run,camcol,field)
            F = fitsio.FITS(pofn)
            N = F[1].get_nrows()
            #print pofn, 'has', N, 'rows'

            # DEBUG
            #POBJ = fits_table(pofn, columns=['objid'])
            #assert(len(POBJ) == N)

            P = fits_table()
            P.has_wise_phot = np.zeros(N, bool)
            if len(TT) > 1:
                # Resolve duplicate measurements (in multiple tiles)
                # based on || (x,y) - center ||^2
                P.R2 = np.empty(N, np.float32)
                P.R2[:] = 1e9
                # WISE coadd tile CRPIX-1 (x,y in the phot-*.fits files are 0-indexed)
                cx,cy = 1023.5, 1023.5
            for T in TT:
                coadd = T.coadd_id[0]
                if len(TT) > 1:
                    I = T.id - 1
                    R2 = (T.x - cx)**2 + (T.y - cy)**2
                    J = (R2 < P.R2[I])
                    I = I[J]
                    P.R2[I] = R2[J].astype(np.float32)
                    #print len(I), 'are closest'
                    T.cut(J)
                print '  ', len(T), 'from', coadd
                if len(T) == 0:
                    continue
                I = T.id - 1
                P.has_wise_phot[I] = True
                pcols = P.get_columns()
                for col in T.get_columns():
                    if col in pcols:
                        pval = P.get(col)
                        print '  ', col, pval.dtype
                        pval[I] = (T.get(col)).astype(pval.dtype)
                    else:
                        tval = T.get(col)
                        X = np.zeros(N, tval.dtype)
                        X[I] = tval
                        P.set(col, X)

                #assert(np.all(POBJ.objid[P.has_wise_phot] == P.objid[P.has_wise_phot]))

            P.delete_column('index')
            P.delete_column('id')
            if len(TT) > 1:
                P.delete_column('R2')
            
            myoutdir = os.path.join(pobjoutdir, rr, '%i'%run, '%i'%camcol)
            if not os.path.exists(myoutdir):
                os.makedirs(myoutdir)
            outfn = os.path.join(myoutdir, 'photoWiseForced-%06i-%i-%04i.fits' % (run, camcol, field))
            P.writeto(outfn)
            print 'Wrote', outfn

            #assert(np.all(POBJ.objid[P.has_wise_phot] == P.objid[P.has_wise_phot]))

        sys.exit(0)

    tiles = []
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        tiles.append(arr)
    else:
        if len(args) == 0:
            tiles.append(0)
        else:
            for a in args:
                if '-' in a:
                    aa = a.split('-')
                    if len(aa) != 2:
                        print 'With arg containing a dash, expect two parts'
                        print aa
                        sys.exit(-1)
                    start = int(aa[0])
                    end = int(aa[1])
                    for i in range(start, end+1):
                        tiles.append(i)
                else:
                    tiles.append(int(a))

    for i in tiles:
        one_tile(T[i], opt, opt.pickle)

if __name__ == '__main__':
    main()

