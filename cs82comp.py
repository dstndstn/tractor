if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
import logging
import tempfile
import tractor
import pyfits
import pylab as plt
import numpy as np
import sys
import re
from glob import glob

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import *
from astrometry.util.util import *
from astrometry.blind.plotstuff import *
from astrometry.util.resample import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

from cs82 import *

def comp1(T1, T2, ps, r0,r1,d0,d1,raa,deca,rab,decb):
    plt.clf()
    lo,hi = -10,100
    loghist(T1.sdss_i_nanomaggies, T2.sdss_i_nanomaggies, 200,
            range=((lo,hi),(lo,hi)))
    ps.savefig()
    
    plt.clf()
    dnm1 = 1./np.sqrt(T1.sdss_i_nanomaggies_invvar)
    dnm2 = 1./np.sqrt(T2.sdss_i_nanomaggies_invvar)
    loghist(T1.sdss_i_nanomaggies,
            (T1.sdss_i_nanomaggies - T2.sdss_i_nanomaggies) / np.hypot(dnm1,dnm2),
            200, range=((lo,hi),(-10,10)))
    plt.xlabel('T1 nm')
    plt.ylabel('T1 - T2 nm / sigma')
    ps.savefig()

    nm1 = T1.sdss_i_nanomaggies
    nm2 = T2.sdss_i_nanomaggies
    m1 = NanoMaggies.nanomaggiesToMag(np.maximum(nm1, 1e-3))
    m2 = NanoMaggies.nanomaggiesToMag(np.maximum(nm2, 1e-3))

    for mlo,mhi in [(22,23),(21,22),(20,21),(19,20),(18,19),(17,18),(0,17)]:
        I = np.flatnonzero((m1 >= mlo) * (m1 < mhi))
        dm = 0.5
        plt.clf()
        plt.hist(np.clip(m1[I] - m2[I], -dm, dm), 50, range=(-dm,dm))
        plt.xlim(-dm, dm)
        plt.xlabel('delta-mag')
        plt.title('Mag1 in [%.1f, %.1f]' % (mlo,mhi))
        ps.savefig()
        
    dnm = (T1.sdss_i_nanomaggies - T2.sdss_i_nanomaggies) / np.hypot(dnm1,dnm2)
    I = np.flatnonzero(dnm > 3.)
    plt.clf()
    plt.plot(T1.ra[I], T1.dec[I], 'r.')
    #ax = plt.axis()
    for x in rab:
        plt.axvline(x, color='r', alpha=0.1)
    for x in decb:
        plt.axhline(x, color='r', alpha=0.1)
    for x in raa:
        plt.axvline(x, color='b', alpha=0.1)
    for x in deca:
        plt.axhline(x, color='b', alpha=0.1)
    plt.axis([r0,r1,d0,d1])
    plt.title('large dnm')
    ps.savefig()

    if False:
        for f in ['prochi2_i', 'pronpix_i', 'profracflux_i',
                  'proflux_i', 'npix_i']:
    
            loghist(T1.get(f), T2.get(f), 100, imshowargs=dict(cmap=antigray),
                    hot=False)
            ax = plt.axis()
            plt.plot(T1.get(f)[I], T2.get(f)[I], 'r.')
            plt.axis(ax)
            plt.title(f)
            ps.savefig()
    


if __name__ == '__main__':

    ps = PlotSequence('cs82comp')
    
    #T1 = fits_table('cs82a-phot-S82p18p.fits')
    #T2 = fits_table('cs82b-phot-S82p18p.fits')
    #nra1,ndec1 = (50, 1)
    #nra2,ndec2 = (10, 10)
    
    # T1 = fits_table('cs82c-phot-S82p18p-slice4.fits')
    # T2 = fits_table('cs82d-phot-S82p18p-slice21.fits')
    # nra1,ndec1 = (50, 1)
    # nra2,ndec2 = (50, 4)

    cs82field = 'S82p18p'

    bands = 'ugriz'
    colors = 'bgrmk'
    T = None
    for band in bands:
        Ti = fits_table('cs82-%s--phot-%s.fits' % (band, cs82field))
        if T is None:
            T = Ti
        else:
            T.add_columns_from(Ti)
    T.writeto('cs82-phot-%s.fits' % cs82field)

    print len(T), 'sources'
    T.cut(T.phot_done)
    print len(T), 'with phot done'

    plt.clf()
    lp = []
    for band,cc in zip(bands, colors):
        x = T.get('sdss_%s_nanomaggies' % band)
        n,b,p = plt.hist(x, 100, color=cc, histtype='step', range=(-1, 5))
        lp.append(p[0])
    plt.xlabel('Flux (nanomaggies)')
    mags = [21, 22, 23]
    for m in mags:
        nm = NanoMaggies.magToNanomaggies(m)
        plt.axvline(nm, color='k', alpha=0.2)
        plt.text(nm, 6000, ' %i' % m, ha='left')
    plt.axvline(0, color='k', alpha=0.2)
    # xl = plt.xlim()
    # a = plt.twiny()
    # plt.xticks([NanoMaggies.magToNanomaggies(m) for m in mags],
    #            ['%i' % m for m in mags])
    # plt.xlim(xl)
    plt.legend(lp, [b for b in bands], loc='upper right')
    plt.title('CS82/SDSS forced photometry')
    ps.savefig()


    plt.clf()
    lp = []
    for band,cc in zip(bands, colors):
        x = T.get('sdss_%s_mag' % band)
        x = x[np.isfinite(x)]
        n,b,p = plt.hist(x, 30, color=cc, histtype='step', range=(20,26))
        lp.append(p[0])
    plt.xlabel('Mag')
    plt.legend(lp, [b for b in bands], loc='upper right')
    plt.title('CS82/SDSS forced photometry')
    ps.savefig()

    plt.clf()
    Ipsf = np.flatnonzero(T.chi2_psf < T.chi2_model)
    sdss = T.sdss_i_mag[Ipsf]
    cfht = T.mag_psf[Ipsf]
    loghist(cfht, sdss - cfht, 100, range=((18, 24),(-1,1)))
    plt.xlabel('CS82 i-band (mag)')
    plt.ylabel('SDSS i - CS82 i (mag)')
    plt.title('CS82/SDSS forced photometry: PSFs')
    ps.savefig()

    plt.clf()
    Imod = np.flatnonzero(T.chi2_model < T.chi2_psf)
    sdss = T.sdss_i_mag[Imod]
    cfht = NanoMaggies.nanomaggiesToMag(
        NanoMaggies.magToNanomaggies(T.mag_disk[Imod]) +
        NanoMaggies.magToNanomaggies(T.mag_spheroid[Imod]))
    loghist(cfht, sdss - cfht, 100, range=((18, 24),(-1,1)))
    plt.xlabel('CS82 i-band (mag)')
    plt.ylabel('SDSS i - CS82 i (mag)')
    plt.title('CS82/SDSS forced photometry: Galaxies')
    ps.savefig()


    for band,yr in zip('ugrz', [(-1,6), (-1,3), (-1,2), (-2,1)]):
        yl,yh = yr
        yt = np.arange(yl, yh+1)
        plt.clf()
        Ipsf = np.flatnonzero(T.chi2_psf < T.chi2_model)
        sdss = T.get('sdss_%s_mag' % band)[Ipsf]
        cfht = T.mag_psf[Ipsf]
        plt.subplot(2,1,1)
        loghist(cfht, sdss - cfht, 100, range=((18, 24),yr), doclf=False)
        plt.yticks(yt)
        plt.ylabel('SDSS %s - CS82 i (mag)' % band)
        plt.title('PSFs')
        plt.subplot(2,1,2)
        Imod = np.flatnonzero(T.chi2_model < T.chi2_psf)
        sdss = T.get('sdss_%s_mag' % band)[Imod]
        cfht = NanoMaggies.nanomaggiesToMag(
            NanoMaggies.magToNanomaggies(T.mag_disk[Imod]) +
            NanoMaggies.magToNanomaggies(T.mag_spheroid[Imod]))
        loghist(cfht, sdss - cfht, 100, range=((18, 24),yr), doclf=False)
        plt.yticks(yt)
        plt.ylabel('SDSS %s - CS82 i (mag)' % band)
        plt.xlabel('CS82 i-band (mag)')
        plt.title('Galaxies')
        ps.savefig()
    

    g = T.sdss_g_mag
    r = T.sdss_r_mag
    i = T.sdss_i_mag
    z = T.sdss_z_mag
    # plt.clf()
    # loghist(g-r, r-i, 100, range=((-2,4),(-2,4)))
    # plt.xlabel('SDSS g-r (mag)')
    # plt.ylabel('SDSS r-i (mag)')
    # ps.savefig()

    plt.clf()
    I = np.flatnonzero(r < 21)
    loghist((g-r)[I], (r-i)[I], 100, #range=((-2,4),(-2,4)))
            range=((-1,3),(-1,3)))
    plt.xlabel('SDSS g-r (mag)')
    plt.ylabel('SDSS r-i (mag)')
    plt.title('r < 21')
    ps.savefig()

    plt.clf()
    I = np.flatnonzero((r > 22) * (r < 23))
    loghist((g-r)[I], (r-i)[I], 100, #range=((-2,4),(-2,4)))
            range=((-1,3),(-1,3)))
    plt.xlabel('SDSS g-r (mag)')
    plt.ylabel('SDSS r-i (mag)')
    plt.title('r in [22,23]')
    ps.savefig()

    # plt.clf()
    # loghist(r-i, i-z, 100, range=((-2,4),(-2,4)))
    # plt.xlabel('SDSS r-i (mag)')
    # plt.ylabel('SDSS i-z (mag)')
    # ps.savefig()
    # 
    # 
    # sys.exit(0)

    
    T1 = fits_table('cs82-i--phot-S82p18p.fits')
    T2 = fits_table('cs82-i2--phot-S82p18p.fits')
    nra1,ndec1 = (50, 1)
    nra2,ndec2 = (50, 4)

    # RA,Dec boundaries used in these photometry runs:
    r0,r1 = 15.783169965, 16.7623649628
    d0,d1 = -0.0919737024789, 0.939468218607
    raa  = np.linspace(r0, r1, nra1  +1)
    deca = np.linspace(d0, d1, ndec1 +1)
    rab  = np.linspace(r0, r1, nra2  +1)
    decb = np.linspace(d0, d1, ndec2 +1)

    def gridlines():
        for x in rab:
            plt.axvline(x, color='b', alpha=0.5)
        for x in decb:
            plt.axhline(x, color='b', alpha=0.5)
        for x in raa:
            plt.axvline(x, color=(0,1,0), alpha=0.5)
        for x in deca:
            plt.axhline(x, color=(0,1,0), alpha=0.5)
        
    
    print 'phot_done:', np.unique(T1.phot_done)
    # assert(np.all(T1.phot_done == T2.phot_done))
    # T1.cut(T1.phot_done)
    # T2.cut(T2.phot_done)

    I = np.logical_and(T1.phot_done, T2.phot_done)
    T1.cut(I)
    T2.cut(I)

    print len(T1), len(T2), 'with photometry done'

    print np.unique(T1.fit_ok_i.astype(np.uint8))
    print np.unique(T1.fit_ok_i)
    print sum(T1.fit_ok_i), 'with fit_ok'
    print sum(T2.fit_ok_i), 'with fit_ok'

    print 'RA,Dec bounds of sources with photometry done:', T1.ra.min(), T1.ra.max(), T1.dec.min(), T1.dec.max()

    comp1(T1, T2, ps, r0,r1,d0,d1,raa,deca,rab,decb)

    plt.figure(figsize=(12,12))

    # wcs of area to look at
    pixscale = 0.4 / 3600.
    ra0,dec0 = 16.4, 0.1
    W,H = 1800,1800

    # ra0  = (T1.ra.min() + T1.ra.max()) / 2.
    # dec0 = (T1.dec.min()+ T1.dec.max())/ 2.
    # W = int((T1.ra.max()  - T1.ra.min() ) / pixscale)
    # H = int((T1.dec.max() - T1.dec.min()) / pixscale)
    
    wcs = Tan(ra0, dec0, W/2+1, H/2+1, pixscale, 0., 0., pixscale, W, H)

    r0,r1,d0,d1 = wcs.radec_bounds()
    print 'RA,Dec bounds of coadd:', r0,r1,d0,d1
    
    # coadd of SDSS images
    T = fits_table('sdssfield-%s.fits' % cs82field)
    I = np.flatnonzero((T.ra1  >= r0) * (T.ra0  <= r1) *
                       (T.dec1 >= d0) * (T.dec0 <= d1))
    T.cut(I)
    print 'Cut to', len(T), 'SDSS fields in range'
                       
    sdss = DR9(basedir='data/unzip')
    sdss.saveUnzippedFiles('data/unzip')

    coadd  = np.zeros((wcs.imageh, wcs.imagew), np.float32)
    ncoadd = np.zeros((wcs.imageh, wcs.imagew), np.int32)

    band = 'i'
    tims = []
    dgpsfs = []
    for i,(r,c,f) in enumerate(zip(T.run, T.camcol, T.field)):
        print 'Reading', (i+1), 'of', len(T), ':', r,c,f,band
        tim,inf = get_tractor_image_dr9(
            r, c, f, band, sdss=sdss,
            nanomaggies=True, zrange=[-2,5], roiradecbox=[r0,r1,d0,d1],
            invvarIgnoresSourceFlux=True, psf='dg')
        if tim is None:
            continue
        dgpsfs.append(inf['dgpsf'])
        (tH,tW) = tim.shape
        print 'Tim', tim.shape
        tim.wcs.setConstantCd(tW/2., tH/2.)
        del tim.origInvvar
        del tim.starMask
        del tim.mask
        tims.append(tim)
        try:
            wcswrap = AsTransWrapper(tim.wcs.astrans, tW,tH,
                                     tim.wcs.x0, tim.wcs.y0)
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, wcswrap, [], 3)
        except:
            import traceback
            print 'Failed to resample:'
            traceback.print_exc()
            continue
        coadd[Yo,Xo] += tim.getImage()[Yi,Xi]
        ncoadd[Yo,Xo] += 1
    coadd = coadd / np.maximum(1, ncoadd).astype(np.float32)
    print len(tims), 'tims; ncoadd range %i %i; coadd range %g, %g' % (ncoadd.min(), ncoadd.max(), coadd.min(), coadd.max())

    ii = np.argsort([s1 for (a,s1,b,s2) in dgpsfs])
    ii = ii[len(ii)/2]
    print 'Median PSF width:', dgpsfs[ii]
    (a,s1, b,s2) = dgpsfs[ii]
    medpsf = NCircularGaussianPSF([s1, s2], [a, b])
    
    plt.clf()
    coa = dict(interpolation='nearest', origin='lower',
               extent=[r0,r1,d0,d1], vmin=-0.05, vmax=0.25)
    plt.imshow(coadd, **coa)
    plt.hot()
    gridlines()
    setRadecAxes(r0,r1,d0,d1)
    plt.title('SDSS coadd: %s band' % band)
    ps.savefig()
    
    fn = 'data/cs82/cats/masked.%s_y.V2.7A.swarp.cut.deVexp.fit' % cs82field
    extra_cols = []
    T = fits_table(fn, hdu=2,
            column_map={'ALPHA_J2000':'ra', 'DELTA_J2000':'dec'},
            columns=[x.upper() for x in
                     ['ALPHA_J2000', 'DELTA_J2000',
                      'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
                      'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
                      'disk_theta_world', 'spheroid_reff_world',
                      'spheroid_aspect_world', 'spheroid_theta_world',
                      'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])

    print 'Read', len(T), 'CS82 sources'
    T.cut((T.ra >= r0) * (T.ra <= r1) * (T.dec >= d0) * (T.dec <= d1))
    print 'Cut to', len(T), 'within RA,Dec box'
    
    srcs,isrcs = get_cs82_sources(T, bands=['i'])
    print 'Got', len(srcs), 'sources'
    Ti = T[isrcs]

    tim = Image(data=np.zeros((H,W), np.float32),
                invvar=np.ones((H,W), np.float32),
                psf=medpsf,
                wcs=ConstantFitsWcs(wcs),
                sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1., band=band),
                domask=False)
    
    tractor = Tractor([tim], srcs)
    mod = tractor.getModelImage(0)

    plt.clf()
    plt.imshow(mod, **coa)
    gridlines()
    setRadecAxes(r0,r1,d0,d1)
    plt.title('CS82 catalog model')
    ps.savefig()

    #ras = [src.pos.ra  for src in srcs]
    #I,J,d = match_radec(ras, [src.pos.dec for src in srcs],
    #                    T1.ra, T1.dec, 1./3600.)
    I,J,d = match_radec(Ti.ra, Ti.dec,
                        T1.ra, T1.dec, 1./3600.)
    print 'CS82:', len(Ti), 'and matched', len(I)
    nm = T1.sdss_i_nanomaggies
    for i,j in zip(I,J):
        setattr(srcs[i].getBrightness(), band, nm[j])
    mod1 = tractor.getModelImage(0)
    plt.clf()
    plt.imshow(mod1, **coa)
    gridlines()
    setRadecAxes(r0,r1,d0,d1)
    plt.title('Tractor model 1')
    ps.savefig()

    I,J,d = match_radec(Ti.ra, Ti.dec,
                        T2.ra, T2.dec, 1./3600.)
    print 'CS82:', len(Ti), 'and matched', len(I)
    nm = T2.sdss_i_nanomaggies
    for i,j in zip(I,J):
        setattr(srcs[i].getBrightness(), band, nm[j])
    mod2 = tractor.getModelImage(0)
    plt.clf()
    plt.imshow(mod2, **coa)
    gridlines()
    setRadecAxes(r0,r1,d0,d1)
    plt.title('Tractor model 2')
    ps.savefig()


    
    plt.clf()
    plt.imshow(mod1 - mod2, interpolation='nearest', origin='lower',
               extent=[r0,r1,d0,d1], vmin=-0.01, vmax=0.01)
    plt.gray()
    gridlines()
    setRadecAxes(r0,r1,d0,d1)
    plt.title('mod1 - mod2')
    ps.savefig()


    
