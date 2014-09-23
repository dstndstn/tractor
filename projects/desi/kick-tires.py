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

    fn = 'tractor-phot-b%06i.fits' % brickid
    T = fits_table(fn)
    print 'Read', len(T)
    T.about()

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
        plt.clf()
        sflux = T.sdss_modelflux[:,bindex]
        dflux = T.get('decam_%s_nanomaggies' % band)
        plt.loglog(sflux, dflux / sflux, 'ro', mec='r', ms=4, alpha=0.1)
        I = np.flatnonzero(sflux > 10.)
        med = np.median(dflux[I] / sflux[I])
        plt.axhline(med, color='k')
        plt.ylim(0.5, 2.)
        ps.savefig()

        corr = dflux / med
        T.set('decam_%s_nanomaggies_corr' % band, corr)
        T.set('decam_%s_mag_corr' % band, NanoMaggies.nanomaggiesToMag(corr))

    bands = 'grz'

    for band in bands:
        plt.clf()
        sn = T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
        mag = T.get('decam_%s_mag_corr' % band)
        plt.semilogy(mag, sn, 'b.')
        ps.savefig()


