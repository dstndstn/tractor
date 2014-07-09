import fitsio

from astrometry.util.util import *
from astrometry.sdss.fields import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *





if __name__ == '__main__':
    import optparse
    import sys

    parser = optparse.OptionParser('%prog [options] <WISE tile name>')
    parser.add_option('-n', type=int, default=4,
                      help='Number of sub-tiles; default %default')
    parser.add_option('-x', type=int, help='Sub-tile x', default=0)
    parser.add_option('-y', type=int, help='Sub-tile y', default=0)
    parser.add_option('--atlas', default='allsky-atlas.fits', help='WISE tile list')
    opt,args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)
        
    tile = args[0]

    print 'Reading', opt.atlas
    T = fits_table(opt.atlas)
    print 'Read', len(T), 'WISE tiles'
    I = np.flatnonzero(tile == T.coadd_id)
    if len(I) != 1:
        print 'Failed to find WISE tile', tile
        sys.exit(-1)
    I = I[0]
    tra,tdec = T.ra[I],T.dec[I]
    del T


    
