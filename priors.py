if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *

ps = PlotSequence('priors')

S = fits_table('stars.fits')
G = fits_table('gals.fits')

# plt.clf()
# loghist(S.psfmag_r - S.psfmag_i, S.psfmag_r, 200, range=((-5,5),(8,25)))
# plt.ylim(25,8)
# plt.xlabel('r-i (mag)')
# plt.ylabel('r (mag)')
# ps.savefig()

plt.clf()
plt.subplot(2,1,1)
loghist(S.psfmag_r - S.psfmag_i, S.psfmag_g - S.psfmag_r, 200, range=((-4,4),(-2,3)),
        doclf=False)
plt.xlabel('r-i (mag)')
plt.ylabel('g-r (mag)')
plt.title('Stars')
plt.subplot(2,1,2)
loghist(G.cmodelmag_r - G.cmodelmag_i, G.cmodelmag_g - G.cmodelmag_r, 200, range=((-4,4),(-2,3)),
        doclf=False)
plt.xlabel('r-i (mag)')
plt.ylabel('g-r (mag)')
plt.title('Galaxies')
plt.suptitle('Distribution of stars and galaxies in color space')
ps.savefig()

plt.clf()
plt.subplot(2,1,1)
plothist(S.psfmag_r - S.psfmag_i, S.psfmag_g - S.psfmag_r, 200, range=((-4,4),(-2,3)),
        doclf=False)
plt.xlabel('r-i (mag)')
plt.ylabel('g-r (mag)')
plt.title('Stars')
plt.subplot(2,1,2)
plothist(G.cmodelmag_r - G.cmodelmag_i, G.cmodelmag_g - G.cmodelmag_r, 200, range=((-4,4),(-2,3)),
        doclf=False)
plt.xlabel('r-i (mag)')
plt.ylabel('g-r (mag)')
plt.title('Galaxies')
plt.suptitle('Distribution of stars and galaxies in color space')
ps.savefig()


plt.clf()
ha = dict(bins=100, range=(8,25), histtype='step')#, log=True)
n1,b,p1 = plt.hist(S.psfmag_r, color='r', **ha)
n2,b,p2 = plt.hist(G.cmodelmag_r, color='b', **ha)
y0,y1 = plt.ylim()
plt.ylim(0.3, y1)
plt.yscale('log')
plt.legend((p1[0],p2[0]), ('Stars', 'Galaxies'), loc='upper left')
plt.xlabel('r-band magnitude')
plt.title('Distribution of stars and galaxies vs brightness')
plt.xlim(8,25)
ps.savefig()
# print 'n1:', n1
# print 'n2:', n2

plt.clf()
plt.subplot(2,1,1)
plothist(180 + (S.ra-180) * np.cos(np.deg2rad(S.dec)), S.dec, (200,100),
         imshowargs=dict(aspect='equal', vmax=1000), doclf=False)
plt.title('Stars')
plt.subplot(2,1,2)
plothist(180 + (G.ra-180) * np.cos(np.deg2rad(G.dec)), G.dec, (200,100),
         imshowargs=dict(aspect='equal', vmax=1000), doclf=False)
plt.title('Galaxies')
plt.suptitle('Distribution of stars and galaxies vs position')
ps.savefig()

