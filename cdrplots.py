import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', serif='computer modern roman')
matplotlib.rc('font', **{'sans-serif': 'computer modern sans serif'})
import sys

import pyfits
import numpy as np
import pylab as plt

from astrometry.util.plotutils import *
from astrometry.util.fits import *
from astrometry.libkd.spherematch import *

ps = PlotSequence('cdr', suffixes=['png','pdf'])

plt.figure(figsize=(5,4))
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.92)

A = fits_table('sequels-atlas.fits')

WW = []
for coadd_id in A.coadd_id[:208]:
    fn = 'sequels-phot-temp/wise-sources-%s.fits' % coadd_id
    if not os.path.exists(fn):
        print 'does not exist:', fn
        continue
    W = fits_table(fn)
    print len(W), 'WISE'
    WW.append(W)
print 'Read', len(WW)
W = merge_tables(WW)
print len(W), 'total sources'

lo,hi = 10,23
ha = dict(bins=100, histtype='step', range=(lo,hi), log=True)

plt.clf()
n1,b,p1 = plt.hist(W.w1mpro + 2.7, color='b', **ha)
#n2,b,p2 = plt.hist(W.w2mpro + 3.34, color='g', lw=3, alpha=0.5, **ha)
n2,e = np.histogram(W.w2mpro + 3.34, bins=100, range=(lo,hi))
p2 = plt.plot(e.repeat(2)[1:-1], n2.repeat(2), color='g', lw=2, alpha=0.5)
#n,b,p3 = plt.hist(W.w3mpro + 5.17, color='r', **ha)
#n,b,p4 = plt.hist(W.w4mpro + 6.62, color='m', **ha)
p1 = plt.plot([1,1],[1,1], color='b')
#p2 = plt.plot([1,1],[1,1], color='g', lw=3, alpha=0.5)
#p3 = plt.plot([1,1],[1,1], color='r')
#p4 = plt.plot([1,1],[1,1], color='m')
plt.xlabel('WISE mag (AB)')
plt.ylabel('Number of sources')
plt.title('WISE All-Sky Data Release catalog depth')
#plt.legend((p1[0],p2[0],p3[0],p4[0]), ('W1', 'W2', 'W3', 'W4'), loc='lower right')
plt.legend((p1[0],p2[0]), ('W1', 'W2'), loc='lower right')
plt.ylim(20., 1.3 * max(max(n1), max(n2)))
plt.xlim(lo,hi)
ps.savefig()

sys.exit(0)



PP = []
WW = []
for coadd_id in A.coadd_id[:5]:
    P = fits_table('sequels-phot/phot-%s.fits' % coadd_id)
    print len(P), 'SDSS'
    W = fits_table('sequels-phot-temp/wise-sources-%s.fits' % coadd_id)
    print len(W), 'WISE'

    PP.append(P)
    WW.append(W)
P = merge_tables(PP)
W = merge_tables(WW)
                
lo,hi = 10,25
cathi = 18

ha = dict(bins=100, histtype='step', range=(lo,hi), log=True)
tsty = dict(color=(0.8,0.8,1.0), lw=3)
csty = dict(color='b')

plt.clf()
ok = np.isfinite(P.w1_mag)
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

plt.clf()
loghist(W.w1mpro[J], P.w1_mag[I], range=((lo,cathi),(lo,cathi)), bins=200,
        imshowargs=dict(cmap=antigray), hot=False)
#plt.gray()        
#plt.colorbar(antigray)
plt.xlabel('WISE W1 mag')
plt.ylabel('Tractor W1 mag')
plt.title('WISE catalog vs Tractor forced photometry')
plt.axis([cathi,lo,cathi,lo])
ps.savefig()

plt.clf()
P.r_mag = -2.5 * (np.log10(P.modelflux[:,2]) - 9.)
loghist(P.r_mag - P.w1_mag, P.r_mag, range=((-5,10),(12,25)), bins=100,
        imshowargs=dict(cmap=antigray), hot=False)
#plt.colorbar(antigray)
#plt.gray()
plt.xlabel('r - W1 (mag)')
plt.ylabel('r (mag)')
plt.title('Tractor forced-photometered SDSS/WISE')
ps.savefig()

plt.clf()
loghist(P.r_mag[I] - W.w1mpro[J], P.r_mag[I], range=((-5,10),(12,25)), bins=100,
        imshowargs=dict(cmap=antigray), hot=False)
plt.xlabel('r - W1 (mag)')
plt.ylabel('r (mag)')
plt.title('Catalog matched SDSS/WISE')
ps.savefig()

