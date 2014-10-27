from runbrick import *
import sys
from astrometry.util.ttime import Time, MemMeas
from astrometry.util.plotutils import PlotSequence
import optparse
import logging

import runbrick

if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] brick-number')
    parser.add_option('--threads', type=int, help='Run multi-threaded')
    opt,args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)
    brick = int(args[0], 10)

    Time.add_measurement(MemMeas)

    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.threads and opt.threads > 1:
        from astrometry.util.multiproc import multiproc
        # ?? global
        runbrick.mp = multiproc(opt.threads)

    catalogfn = 'tractor-phot-b%06i.fits' % brick

    P = dict(W=3600, H=3600, brickid=brick)
    R = stage0(**P)
    P.update(R)
    R = stage1(**P)
    P.update(R)
    stage202(catalogfn=catalogfn, **P)

    # Plots
    ps = PlotSequence('brick-%06i' % brick)
    stage102(ps=ps, **P)
