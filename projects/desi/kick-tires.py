import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.libkd.spherematch import match_radec
from tractor import *
from tractor.galaxy import *
from common import *

if __name__ == '__main__':

    brickid = 371589
    ps = PlotSequence('kick')
    ps.suffixes = ['png','pdf']

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
    
    fn = 'tractor-phot-b%06i.fits' % brickid
    T = fits_table(fn)
    print 'Read', len(T)
    T.about()
    T.cut(T.blob > 0)
    print 'Cut to', len(T)
    
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
        plt.loglog(sflux, dflux / sflux, 'o', mec='b', ms=4, alpha=0.1)
        plt.axhline(1., color='k')
        plt.ylim(0.5, 2.)
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
        plt.semilogy(mag, sn, '.', color=cc, alpha=0.2)
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

    
