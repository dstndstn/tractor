import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from glob import glob

from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import * #PlotSequence, dimshow
from astrometry.libkd.spherematch import match_radec
from tractor import *
from tractor.galaxy import *
from common import *

if __name__ == '__main__':

    ps = PlotSequence('kick')
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)

    # A catalog of sources overlapping one DECaLS CCD, arbitrarily:
    # python projects/desi/forced-photom-decam.py decam/CP20140810_g_v2/c4d_140816_032035_ooi_g_v2.fits.fz 1 DR1 f.fits

    T = fits_table('cat.fits')
    
    I = np.flatnonzero(T.type == 'EXP ')
    J = np.flatnonzero(T.type == 'DEV ')

    plt.clf()
    p1 = plt.semilogy(T.shapeexp_e1[I], T.shapeexp_e1_ivar[I], 'b.')
    p2 = plt.semilogy(T.shapeexp_e2[I], T.shapeexp_e2_ivar[I], 'r.')
    plt.xlabel('Ellipticity e')
    plt.ylabel('Ellipticity inverse-variance e_ivar')
    plt.title('EXP galaxies')
    plt.legend([p1[0],p2[0]], ['e1','e2'])
    ps.savefig()

    plt.clf()
    p1 = plt.semilogy(T.shapeexp_e1[I], 1./np.sqrt(T.shapeexp_e1_ivar[I]), 'b.')
    p2 = plt.semilogy(T.shapeexp_e2[I], 1./np.sqrt(T.shapeexp_e2_ivar[I]), 'r.')
    plt.xlabel('Ellipticity e')
    plt.ylabel('Ellipticity error e_err')
    plt.title('EXP galaxies')
    plt.legend([p1[0],p2[0]], ['e1','e2'])
    ps.savefig()
    
    plt.clf()
    p1 = plt.semilogy(T.shapedev_e1[J], T.shapedev_e1_ivar[J], 'b.')
    p2 = plt.semilogy(T.shapedev_e2[J], T.shapedev_e2_ivar[J], 'r.')
    plt.xlabel('Ellipticity e')
    plt.ylabel('Ellipticity inverse-variance e_ivar')
    plt.title('DEV galaxies')
    plt.legend([p1[0],p2[0]], ['e1','e2'])
    ps.savefig()

    plt.clf()
    p1 = plt.semilogy(T.shapedev_e1[J], 1./np.sqrt(T.shapedev_e1_ivar[J]), 'b.')
    p2 = plt.semilogy(T.shapedev_e2[J], 1./np.sqrt(T.shapedev_e2_ivar[J]), 'r.')
    plt.xlabel('Ellipticity e')
    plt.ylabel('Ellipticity error e_err')
    plt.title('DEV galaxies')
    plt.legend([p1[0],p2[0]], ['e1','e2'])
    ps.savefig()



    sys.exit(0)

    
    print 'RA', T.ra.min(), T.ra.max()
    print 'Dec', T.dec.min(), T.dec.max()

    # Uhh, how does *this* happen?!  Fitting gone wild I guess
    # T.cut((T.ra > 0) * (T.ra < 360) * (T.dec > -90) * (T.dec < 90))
    # print 'RA', T.ra.min(), T.ra.max()
    # print 'Dec', T.dec.min(), T.dec.max()
    # rlo,rhi = [np.percentile(T.ra,  p) for p in [1,99]]
    # dlo,dhi = [np.percentile(T.dec, p) for p in [1,99]]
    # print 'RA', rlo,rhi
    # print 'Dec', dlo,dhi
    # plt.clf()
    # plothist(T.ra, T.dec, 100, range=((rlo,rhi),(dlo,dhi)))
    # plt.xlabel('RA')
    # plt.ylabel('Dec')
    # ps.savefig()
    
    # decals = Decals()
    # B = decals.get_bricks()
    # #B.about()
    # brick = B[B.brickid == brickid]
    # assert(len(brick) == 1)
    # brick = brick[0]
    # wcs = wcs_for_brick(brick)
    # ccds = decals.get_ccds()
    # ccds.cut(ccds_touching_wcs(wcs, ccds))
    # print len(ccds), 'CCDs'
    # #ccds.about()
    # ccds.cut(ccds.filter == 'r')
    # print len(ccds), 'CCDs'
    # S = []
    # for ccd in ccds:
    #     im = DecamImage(ccd)
    #     S.append(fits_table(im.sdssfn))
    # S = merge_tables(S)
    # print len(S), 'total SDSS'
    # #nil,I = np.unique(S.ra, return_index=True)
    # nil,I = np.unique(['%.5f %.5f' % (r,d) for r,d in zip(S.ra,S.dec)], return_index=True)
    # S.cut(I)
    # print len(S), 'unique'
    # 
    # I,J,d = match_radec(T.ra, T.dec, S.ra, S.dec, 1./3600.)
    # print len(I), 'matches'
    # 
    # plt.clf()
    # plt.loglog(S.r_psfflux[J], T.decam_r_nanomaggies[I], 'r.')
    # ps.savefig()

    #plt.clf()
    #plt.loglog(T.sdss_modelflux[:,2], T.decam_r_nanomaggies, 'r.')
    #ps.savefig()

    
    for bindex,band in [(1,'g'), (2,'r'), (4,'z')]:
        sflux = T.sdss_modelflux[:,bindex]
        dflux = T.get('decam_%s_nanomaggies' % band)
        I = np.flatnonzero(sflux > 10.)
        med = np.median(dflux[I] / sflux[I])
        # plt.clf()
        # plt.loglog(sflux, dflux / sflux, 'ro', mec='r', ms=4, alpha=0.1)
        # plt.axhline(med, color='k')
        # plt.ylim(0.5, 2.)
        # ps.savefig()

        corr = dflux / med
        T.set('decam_%s_nanomaggies_corr' % band, corr)
        T.set('decam_%s_mag_corr' % band, NanoMaggies.nanomaggiesToMag(corr))

        dflux = T.get('decam_%s_nanomaggies_corr' % band)
        plt.clf()
        #plt.loglog(sflux, dflux / sflux, 'o', mec='b', ms=4, alpha=0.1)
        plt.loglog(sflux, dflux / sflux, 'b.', alpha=0.01)
        plt.xlim(1e-1, 3e3)
        plt.axhline(1., color='k')
        plt.ylim(0.5, 2.)
        plt.xlabel('SDSS flux (nmgy)')
        plt.ylabel('DECam flux / SDSS flux')
        plt.title('%s band' % band)
        ps.savefig()
        
    bands = 'grz'

    # for band in bands:
    #     plt.clf()
    #     sn = T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
    #     mag = T.get('decam_%s_mag_corr' % band)
    #     plt.semilogy(mag, sn, 'b.')
    #     plt.axis([20, 26, 1, 100])
    #     ps.savefig()

    ccmap = dict(g='g', r='r', z='m')
    plt.clf()
    for band in bands:
        sn = T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
        mag = T.get('decam_%s_mag_corr' % band)
        cc = ccmap[band]
        #plt.semilogy(mag, sn, '.', color=cc, alpha=0.2)
        plt.semilogy(mag, sn, '.', color=cc, alpha=0.01, mec='none')
    plt.xlabel('mag')
    plt.ylabel('Flux Signal-to-Noise')
    tt = [1,2,3,4,5,10,20,30,40,50]
    plt.yticks(tt, ['%i' % t for t in tt])
    plt.axhline(5., color='k')
    plt.axis([21, 26, 1, 20])
    plt.title('DECaLS depth')
    ps.savefig()



    [gsn,rsn,zsn] = [T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
                     for band in bands]
    TT = T[(gsn > 5.) * (rsn > 5.) * (zsn > 5.)]
    
    # plt.clf()
    # plt.plot(g-r, r-z, 'k.', alpha=0.2)
    # plt.xlabel('g - r (mag)')
    # plt.ylabel('r - z (mag)')
    # plt.xlim(-0.5, 2.5)
    # plt.ylim(-0.5, 3)
    # ps.savefig()

    plt.clf()
    lp = []
    cut = (TT.sdss_objc_type == 6)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_psfflux[:,i])
             for i in [1,2,4]]
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.3, color='b')
    lp.append(p[0])
    cut = (TT.sdss_objc_type == 3)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_modelflux[:,i])
             for i in [1,2,4]]
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.3, color='r')
    lp.append(p[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    plt.legend(lp, ['stars', 'galaxies'])
    plt.title('SDSS')
    ps.savefig()


    g = TT.decam_g_mag_corr
    r = TT.decam_r_mag_corr
    z = TT.decam_z_mag_corr

    plt.clf()
    lt,lp = [],[]
    for cut,cc,tt in [(TT.sdss_objc_type == 6, 'b', 'stars'),
                      (TT.sdss_objc_type == 3, 'r', 'galaxies'),
                      (TT.sdss_objc_type == 0, 'g', 'faint')]:
        p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.3, color=cc)
        lt.append(tt)
        lp.append(p[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    plt.legend(lp, lt)
    plt.title('DECaLS')
    ps.savefig()

    

    # Stars/galaxies in subplots

    plt.clf()
    lp = []
    cut = (TT.sdss_objc_type == 6)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_psfflux[:,i])
             for i in [1,2,4]]
    plt.subplot(1,2,1)
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.02, color='b')
    px = plt.plot(100, 100, '.', color='b')
    lp.append(px[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    cut = (TT.sdss_objc_type == 3)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_modelflux[:,i])
             for i in [1,2,4]]
    plt.subplot(1,2,2)
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.02, color='r')
    px = plt.plot(100, 100, '.', color='r')
    lp.append(px[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    plt.figlegend(lp, ['stars', 'galaxies'], 'upper right')
    plt.suptitle('SDSS')
    ps.savefig()

    g = TT.decam_g_mag_corr
    r = TT.decam_r_mag_corr
    z = TT.decam_z_mag_corr

    plt.clf()
    lt,lp = [],[]
    for i,(cut,cc,tt) in enumerate([
        (TT.sdss_objc_type == 6, 'b', 'stars'),
        (TT.sdss_objc_type == 3, 'r', 'galaxies'),
        #(TT.sdss_objc_type == 0, 'g', 'faint'),
        ]):
        plt.subplot(1,2,i+1)
        p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.02, color=cc)
        lt.append(tt)
        px = plt.plot(100, 100, '.', color=cc)
        lp.append(px[0])
        plt.xlabel('g - r (mag)')
        plt.ylabel('r - z (mag)')
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 3)
    plt.figlegend(lp, lt, 'upper right')
    plt.suptitle('DECaLS')
    ps.savefig()

    


