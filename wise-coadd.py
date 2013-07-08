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


