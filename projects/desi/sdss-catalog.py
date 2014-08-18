import fitsio

from astrometry.util.util import *
from astrometry.sdss.fields import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

from desi_common import *


if __name__ == '__main__':
    import optparse
    import sys
    import desi_common

    parser = optparse.OptionParser('%prog [options] <WISE tile name> <catalog output name>')
    parser.add_option('-n', type=int, default=desi_common.N_subtiles,
                      help='Number of sub-tiles; default %default')
    parser.add_option('-x', type=int, help='Sub-tile x', default=0)
    parser.add_option('-y', type=int, help='Sub-tile y', default=0)
    parser.add_option('--atlas', default='allsky-atlas.fits', help='WISE tile list')
    parser.add_option('--bands', default=[], action='append', help='Bands to include in output catalog, default g,r,z')
    opt,args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
        sys.exit(-1)
        
    tile,outfn = args

    if opt.n != desi_common.N_subtiles:
        desi_common.N_subtiles = opt.n

    if len(opt.bands) == 0:
        opt.bands = ['g','r','z']

    wcs = get_subtile_wcs(tile, opt.x, opt.y)
    print 'WCS:', wcs

    # FIXME
    margin = 0.
    photoobjdir = 'photoObjs-new'

    sdss = DR9(basedir=photoobjdir)
    sdss.useLocalTree()

    cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
            'resolve_status', 'nchild', 'flags', 'objc_flags',
            'run','camcol','field','id',
            'psfflux', 'psfflux_ivar',
            'cmodelflux', 'cmodelflux_ivar',
            'modelflux', 'modelflux_ivar',
            'devflux', 'expflux']

    objs = read_photoobjs_in_wcs(wcs, margin, sdss=sdss, cols=cols)
    print 'Got', len(objs), 'photoObjs'

    srcs = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss,
        bands=opt.bands,
        nanomaggies=True, fixedComposites=True,
        useObjcType=True,
        ellipse=EllipseESoft.fromRAbPhi)
    print 'Got', len(srcs), 'Tractor sources'

    cat = Catalog(*srcs)
    N = cat.numberOfParams()
    var = np.zeros(N)

    T,hdr = prepare_fits_catalog(cat, var, None, None, opt.bands, None)
    T.writeto(outfn, header=hdr)
    print 'Wrote to', outfn

