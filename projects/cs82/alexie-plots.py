import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *
from tractor import *

ps = PlotSequence('cs82')

C = merge_tables([fits_table('data/cs82/cats/masked.S82p%ip_y.V2.7A.swarp.cut.deVexp.fit' % i, hdu=2)
                  for i in [40,41,42,43]])
print len(C), 'CS82 catalog'
C.about()
C.ptsrc = (C.spread_model < 0.0036)

F = merge_tables([fits_table('cs82-phot-S82p%ip.fits' % i)
                  for i in [40,41,42,43]])
print len(F), 'Forced-phot catalog'

S = fits_table('cas-p4043p.fits')
print len(S), 'SDSS CASJobs'

I = (F.phot_done * F.fit_ok_i)
J = (C.ptsrc)

K = np.logical_and(I, J)

sdss = F.sdss_i_mag[K]
cfht = C.mag_psf[K]
plt.clf()
loghist(cfht, sdss - cfht, 200, range=((18, 24),(-1,1)))
plt.xlabel('CS82 i-band (mag)')
plt.ylabel('SDSS i - CS82 i (mag)')
plt.title('CS82/SDSS forced photometry: PSFs')
ps.savefig()

K = np.logical_and(I, np.logical_not(J))

sdss = F.sdss_i_mag[K]
cfht = NanoMaggies.nanomaggiesToMag(
    NanoMaggies.magToNanomaggies(C.mag_disk[K]) +
    NanoMaggies.magToNanomaggies(C.mag_spheroid[K]))
plt.clf()
loghist(cfht, sdss - cfht, 200, range=((18, 24),(-1,1)))
plt.xlabel('CS82 i-band (mag)')
plt.ylabel('SDSS i - CS82 i (mag)')
plt.title('CS82/SDSS forced photometry: Galaxies')
ps.savefig()


FI = F[F.phot_done]

I,J,d = match_radec(S.ra, S.dec, FI.ra, FI.dec, 1./3600.)
print len(I), 'matches'

for bi,band in enumerate('ugriz'):
    plt.clf()
    n,b,p1 = plt.hist(S.get('modelmag_%s' % band),
                      100, range=(18,28),
                      histtype='step', color=(0.8,0.8,1), lw=3)
    n,b,p3 = plt.hist(S.get('psfmag_%s' % band),
                      100, range=(18,28),
                      histtype='step', color=(0.8,1,0.8), lw=3)
    n,b,p2 = plt.hist(FI.get('sdss_%s_mag' % band),
                      100, range=(18,28), histtype='step', color='b')
    p1 = plt.plot([0],[0],'-', color=(0.8,0.8,1), lw=3)
    p2 = plt.plot([0],[0],'-', color='b')
    #p3 = plt.plot([0],[0],'-', color=(0.8,1,0.8), lw=3)
    plt.xlabel('%s (mag)' % band)
    plt.legend((p2[0], p1[0]), ('CS82 Forced phot', 'Annis et al coadds'))
    plt.xlim(18,28)
    ps.savefig()



for bi,band in enumerate('ugriz'):
    plt.clf()
    annis = S.get('modelmag_%s' % band)[I]
    forced = FI.get('sdss_%s_mag' % band)[J]
    loghist(annis, forced - annis, 200, range=((18,25),(-1,1)))
    plt.xlabel('Annis et al coadds: %s (mag)' % band)
    plt.ylabel('CS82 forced photometry - Annis: %s (mag)' % band)
    plt.title('CS82 forced photometry vs Annis et al coadds: %s band' % band)
    ps.savefig()

