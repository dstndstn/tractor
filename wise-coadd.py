import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

from scipy.ndimage.morphology import binary_dilation

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from runslice import *

from tractor import *
from tractor.ttime import *



T = fits_table('wise_allsky_4band_p3as_cdd.fits')

ps = PlotSequence('co')

plt.clf()
plt.plot(T.ra, T.dec, 'r.', ms=4, alpha=0.5)
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.title('Atlas tile centers')
plt.axis([360,0,-90,90])
ps.savefig()




# for i,d in enumerate(ims):
#     sig1 = d.sig1
# 
#     R,C = 2,3
# 
#     plt.clf()
#     plt.subplot(R,C,1)
#     plt.imshow(d.rimg, interpolation='nearest', origin='lower',
#                vmin=-2*sig1, vmax=5*sig1)
#     plt.title('data')
#     
#     plt.subplot(R,C,2)
#     plt.imshow(d.rmod, interpolation='nearest', origin='lower',
#                vmin=-2*sig1, vmax=5*sig1)
#     plt.title('mod')
#     
#     plt.subplot(R,C,3)
#     chi = (d.rimg - d.rmod) * d.mask / sig1
#     plt.imshow(chi, interpolation='nearest', origin='lower',
#                vmin=-5, vmax=5, cmap='gray')
#     plt.title('chi: %.1f' % np.sum(chi**2))
# 
#     # grab original rchi
#     rchi = rchis[i]
#     
#     rchi2 = np.sum(rchi**2) / np.sum(d.mask)
# 
#     plt.subplot(R,C,4)
#     plt.imshow(rchi, interpolation='nearest', origin='lower',
#                vmin=-5, vmax=5, cmap='gray')
#     plt.title('rchi2 vs coadd: %.2f' % rchi2)
# 
#     plt.subplot(R,C,5)
#     plt.imshow(np.abs(rchi) > 5, interpolation='nearest', origin='lower', cmap='gray')
#     plt.title('rchi > 5')
# 
#     plt.subplot(R,C,6)
#     plt.imshow(d.mask, interpolation='nearest', origin='lower', cmap='gray')
#     plt.title('mask')
# 
#     plt.suptitle(d.name)
#     ps.savefig()
