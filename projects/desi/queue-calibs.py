import sys
import os
import numpy as np
from astrometry.util.fits import fits_table
from common import Decals, wcs_for_brick, ccds_touching_wcs

'''
python projects/desi/queue-calibs.py  | qdo load cal -
qdo launch cal 1 --pbsopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial"

python projects/desi/queue-calibs.py  | qdo load bricks -
qdo launch bricks 1 --pbsopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial -o pipebrick-logs -j oe -l pvmem=6GB" \
    --script projects/desi/pipebrick.sh
'''

from astrometry.libkd.spherematch import *

import matplotlib
matplotlib.use('Agg')
import pylab as plt

if __name__ == '__main__':
    D = Decals()
    T = D.get_ccds()
    B = D.get_bricks()

    # I,J,d,counts = match_radec(B.ra, B.dec, T.ra, T.dec, 0.2, nearest=True, count=True)
    # plt.clf()
    # plt.hist(counts, counts.max()+1)
    # plt.savefig('bricks.png')
    # B.cut(I[counts >= 9])
    # plt.clf()
    # plt.plot(B.ra, B.dec, 'b.')
    # #plt.scatter(B.ra[I], B.dec[I], c=counts)
    # plt.savefig('bricks2.png')

    #B.cut((B.ra > 240) * (B.ra < 250) * (B.dec > 5) * (B.dec < 12))

    #B.cut((B.ra > 240) * (B.ra < 242) * (B.dec > 5) * (B.dec < 7))
    B.cut((B.ra > 240) * (B.ra < 244) * (B.dec > 5) * (B.dec < 9))
    #print len(B), 'bricks in range'

    for b in B.brickid:
        fn = 'pipebrick-cats/tractor-phot-b%06i.fits' % b
        if os.path.exists(fn):
            print >> sys.stderr, 'exists:', fn
            continue
        print b
    sys.exit(0)

    allI = set()
    for b in B:
        wcs = wcs_for_brick(b)
        I = ccds_touching_wcs(wcs, T)
        allI.update(I)
    #print 'Total of', len(allI), 'CCDs touch'
    #T.cut(np.array(list(allI)))

    print >>sys.stderr, len(B), 'bricks,', len(allI), 'CCDs'

    for i in list(allI):
        print 'python projects/desi/run-calib.py %i' % i

    sys.exit(0)

    # g,r,z full focal planes, 2014-08-18
    #I = np.flatnonzero(T.expnum == 349664)
    #I = np.flatnonzero(T.expnum == 349667)
    #I = np.flatnonzero(T.expnum == 349589)

    #for im in T.cpimage[:10]:
    #    print >>sys.stderr, 'im >>%s<<' % im, im.startswith('CP20140818')
    #I = np.flatnonzero(np.array([im.startswith('CP20140818') for im in T.cpimage]))

    # images touching brick X
    B = D.get_bricks()
    #ii = 380155
    ii = 377305
    targetwcs = wcs_for_brick(B[ii])
    I = ccds_touching_wcs(targetwcs, T)
    #print len(I), 'CCDs touching'

    print >>sys.stderr, len(I), 'in cut'
    for i in I:
        print 'python projects/desi/run-calib.py %i' % i

