import glob as glob
import os

import fitsio


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%prog [options] <frame frame frame>')
    opt,args = parser.parse_args()

    for fn in args:
        print 'Reading', fn
        F = fitsio.FITS(fn)

    
