import glob as glob
import os
import datetime

import numpy as np

from astrometry.util.util import *
from astrometry.util.starutil import *
from astrometry.util.fits import *

import fitsio

from common import exposure_metadata

'''
python -u projects/desi/make-exposure-list.py ~/cosmo/data/staging/decam/CP*/c4d_*_ooi*.fits.fz > log 2> err &
python -u projects/desi/make-exposure-list.py -o ccds-20140810.fits ~/cosmo/data/staging/decam/CP20140810/c4d_*_ooi*.fits.fz > log-0810 2>&1 &

python -u projects/desi/make-exposure-list.py --trim /global/homes/d/dstn/cosmo/staging/decam/ -o 1.fits ~/cosmo/staging/decam/CP20140810/c4d_140809_04*_ooi*.fits.fz > log 2> err &
'''

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%prog [options] <frame frame frame>')
    parser.add_option('-o', dest='outfn', help='Output filename', default='ccds.fits')
    parser.add_option('--trim', help='Trim prefix from filename')
    opt,args = parser.parse_args()

    T = exposure_metadata(args)
    T.about()

    print 'Converting datatypes...'
    for k in T.get_columns():
        print
        print k
        #print T.get(k)
        # Convert doubles to floats
        if k.startswith('ra') or k.startswith('dec') or k.startswith('cr'):
            continue
        X = T.get(k)
        print X.dtype
        if X.dtype == np.float64:
            T.set(k, X.astype(np.float32))
        elif X.dtype == np.int64:
            T.set(k, X.astype(np.int32))

    T.about()

    T.writeto(opt.outfn)
    print 'Wrote', opt.outfn
                
