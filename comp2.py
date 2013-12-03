import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *

import sys
import fitsio

ps = PlotSequence('comp')
ps.suffixes = ['png', 'pdf']

#  All-Sky, via polygon search for full SEQUELS area
# text2fits.py -S 56 -H "ra dec sigra sigdec sigradec w1mpro w1sigmpro w1snr w1rchi2 w2mpro w2sigmpro w2snr w2rchi2 nb na cc_flags ext_flg var_flg moon_lev w1nm w1m w2nm w2m" wise_allsky.wise_allsky_4band_p3as_psd28572.tbl sequels-wise-cat.fits -f "ddfffffffffffjjssssjjjj"
W = fits_table('sequels-wise-cat.fits')
print len(W), 'WISE catalog sources'
T = fits_table('data/sequels-phot-v5.fits')
print len(T), 'Tractor sources'

I,J,d = match_radec(W.ra, W.dec, T.ra, T.dec, 4./3600.)
print len(I), 'matches'

plt.figure(figsize=(5,3.5))
plt.subplots_adjust(left=0.15, bottom=0.15, top=0.95, right=0.95)

for band in [1,2]:
    w = W.get('w%impro' % band)
    t = T.get('w%i_mag' % band)

    lo,hi = 10,25
    cathi = 18
    ha = dict(bins=100, histtype='step', range=(lo,hi), log=True)
    tsty = dict(color=(0.8,0.8,1.0), lw=3)
    csty = dict(color='b')
    a = ha.copy()
    a.update(tsty)
    plt.clf()
    n,b,p1 = plt.hist(t, **a)
    a = ha.copy()
    a.update(csty)
    n,b,p2 = plt.hist(w, **a)
    # legend only
    p1 = plt.plot([1,1],[1,1], **tsty)
    p2 = plt.plot([1,1],[1,1], **csty)
    plt.xlabel('W%i mag (Vega)' % band)
    plt.ylabel('Number of sources')
    #plt.title('WISE catalog vs Tractor forced photometry depths')
    plt.legend((p1[0],p2[0]), ('W%i (Tractor)' % band, 'W%i (WISE catalog)' % band), loc='lower right')
    #plt.ylim(1., 2e4)
    plt.ylim(1e2, 2e6)
    plt.xlim(lo,hi)
    ps.savefig()

    w = w[I]
    t = t[J]
    ok = np.logical_and(np.isfinite(w), np.isfinite(t))
    w = w[ok]
    t = t[ok]
    print 'Band', band, ': cut to', len(w), 'valid'
    
    ha = dict(bins=200,
              )#imshowargs=dict(cmap=antigray), hot=False)
    
    plt.clf()
    loghist(w, t, range=((lo,cathi),(lo,cathi)), **ha)
    plt.xlabel('WISE W%i mag' % band)
    plt.ylabel('Tractor W%i mag' % band)
    #plt.title('WISE catalog vs Tractor forced photometry')
    plt.axis([cathi,lo,cathi,lo])
    ps.savefig()

    # Tractor CMD
    t2 = T.get('w%i_mag' % band)
    ok2 = np.isfinite(t2)
    t2 = t2[ok2]
    rmag2 = T.modelflux[:,2]
    rmag2 = rmag2[ok2]
    rmag2 = -2.5 * (np.log10(rmag2) - 9.)

    plt.clf()
    H,xe,ye = loghist(rmag2 - t2, rmag2, range=((-5,10),(12,25)), **ha)
    plt.xlabel('r - W%i (mag)' % band)
    plt.ylabel('r (mag)')
    #plt.title('SDSS/WISE Tractor forced-photometry')
    plt.axis([-5,10,25,12])
    ps.savefig()

    # Catalog-match CMD
    rmag = T.modelflux[:,2]
    rmag = rmag[J][ok]
    rmag = -2.5 * (np.log10(rmag) - 9.)
    
    plt.clf()
    loghist(rmag - w, rmag, range=((-5,10),(12,25)),
            imshowargs=dict(vmax=np.log10(np.max(H))), **ha)
    plt.xlabel('r - W%i (mag)' % band)
    plt.ylabel('r (mag)')
    #plt.title('SDSS/WISE catalog matches')
    plt.axis([-5,10,25,12])
    ps.savefig()


    







sys.exit(0)

for band in []:#[1]: #,2,3,4]:
    coadd_id = '1384p454'
    for phase in ['a','d']:
        fn1 = 'c/phot-%s-%i%s.fits' % (coadd_id, band, phase)
        T1 = fits_table(fn1)
    
        W = fits_table('sequels-phot-temp/wise-sources-%s.fits' % coadd_id)
        print len(W), 'WISE'
    
        P = T1
        plt.clf()
        ok = np.isfinite(P.w1_mag)
        lo,hi = 10,25
        cathi = 18
        ha = dict(bins=100, histtype='step', range=(lo,hi), log=True)
        tsty = dict(color=(0.8,0.8,1.0), lw=3)
        csty = dict(color='b')
        a = ha.copy()
        a.update(tsty)
        n,b,p1 = plt.hist(P.w1_mag[ok], **a)
        a = ha.copy()
        a.update(csty)
        n,b,p2 = plt.hist(W.w1mpro, **a)
        # legend only
        p1 = plt.plot([1,1],[1,1], **tsty)
        p2 = plt.plot([1,1],[1,1], **csty)
        plt.xlabel('W1 mag (Vega)')
        plt.ylabel('Number of sources')
        plt.title('WISE catalog vs Tractor forced photometry depths')
        plt.legend((p1[0],p2[0]), ('W1 (Tractor)', 'W1 (WISE catalog)'), loc='lower right')
        plt.ylim(1., 2e4)
        plt.xlim(lo,hi)
        ps.savefig()
    
        I,J,d = match_radec(P.ra, P.dec, W.ra, W.dec, 4./3600.)
        print len(I), 'matches'
    
        ha = dict(bins=200,
                  )#imshowargs=dict(cmap=antigray), hot=False)
    
        plt.clf()
        loghist(W.w1mpro[J], P.w1_mag[I], range=((lo,cathi),(lo,cathi)), **ha)
        plt.xlabel('WISE W1 mag')
        plt.ylabel('Tractor W1 mag')
        plt.title('WISE catalog vs Tractor forced photometry')
        plt.axis([cathi,lo,cathi,lo])
        ps.savefig()
    
        plt.clf()
        P.r_mag = -2.5 * (np.log10(P.modelflux[:,2]) - 9.)
        loghist(P.r_mag - P.w1_mag, P.r_mag, range=((-5,10),(12,25)), **ha)
        plt.xlabel('r - W1 (mag)')
        plt.ylabel('r (mag)')
        plt.title('Tractor forced-photometered SDSS/WISE')
        ps.savefig()



for band in [1]:#,2,3,4]:
    coadd_id = '1384p454'

    pairs = [('a','e'), ('a','f'), ('a','g'), ('f','e')]

    for c1,c2 in pairs:
        fn1 = 'c/phot-%s-%i%s.fits' % (coadd_id, band, c1)
        T1 = fits_table(fn1)

        fn2 = 'c/phot-%s-%i%s.fits' % (coadd_id, band, c2)
        if not os.path.exists(fn1) or not os.path.exists(fn2):
            print 'not found:', fn1, fn2
            continue
        T1 = fits_table(fn1)
        T2 = fits_table(fn2)

        plt.clf()
        plt.plot(T1.get('w%i_nanomaggies' % band), T2.get('w%i_nanomaggies' % band), 'b.',
                 alpha=0.2)
        plt.xscale('symlog')
        plt.yscale('symlog')
        plt.title('%s - %s' % (fn1,fn2))
        ps.savefig()

sys.exit(0)

for band in [1,2,3,4]:
    fn1 = 'c/phot-1384p454-%ib.fits' % band
    fn2 = 'c/phot-1384p454-%ic.fits' % band
    if not os.path.exists(fn1) or not os.path.exists(fn2):
        print 'not found:', fn1, fn2
        continue
    T1 = fits_table(fn1)
    T2 = fits_table(fn2)

    plt.clf()
    plt.plot(T1.get('w%i_nanomaggies' % band), T2.get('w%i_nanomaggies' % band), 'b.',
             alpha=0.2)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.title('%s - %s' % (fn1,fn2))
    ps.savefig()
    
