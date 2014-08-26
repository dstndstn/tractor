import os
import numpy as np
from astrometry.util.fits import fits_table
from common import decals_dir

'''
python projects/desi/queue-calibs.py  | qdo load decals -
'''

if __name__ == '__main__':
    ccdsfn = os.path.join(decals_dir, 'decals-ccds.fits')
    T = fits_table(ccdsfn)

    #I = np.flatnonzero(T.expnum == 349664)
    #I = np.flatnonzero(T.expnum == 349667)
    I = np.flatnonzero(T.expnum == 349589)

    #print len(I), 'in cut'
    for i in I:
        print 'python projects/desi/run-calib.py %i' % i

