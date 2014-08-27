import sys
import os
import numpy as np
from astrometry.util.fits import fits_table
from common import Decals, wcs_for_brick, ccds_touching_wcs

'''
python projects/desi/queue-calibs.py  | qdo load decals -
qdo launch decals 1 --batchopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial"
'''

if __name__ == '__main__':
    #ccdsfn = os.path.join(decals_dir, 'decals-ccds.fits')
    #T = fits_table(ccdsfn)

    D = Decals()
    T = D.get_ccds()

    # g,r,z full focal planes, 2014-08-18
    #I = np.flatnonzero(T.expnum == 349664)
    #I = np.flatnonzero(T.expnum == 349667)
    #I = np.flatnonzero(T.expnum == 349589)

    #for im in T.cpimage[:10]:
    #    print >>sys.stderr, 'im >>%s<<' % im, im.startswith('CP20140818')
    #I = np.flatnonzero(np.array([im.startswith('CP20140818') for im in T.cpimage]))

    # images touching brick X
    B = D.get_bricks()
    ii = 380155
    targetwcs = wcs_for_brick(B[ii])
    I = ccds_touching_wcs(targetwcs, T)
    #print len(I), 'CCDs touching'

    print >>sys.stderr, len(I), 'in cut'
    for i in I:
        print 'python projects/desi/run-calib.py %i' % i

