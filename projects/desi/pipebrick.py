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
    parser.add_option('--stamp', action='store_true', help='Run a tiny postage-stamp')
    opt,args = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)
    brick = int(args[0], 10)

    Time.add_measurement(MemMeas)

    lvl = logging.WARNING
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.threads and opt.threads > 1:
        from astrometry.util.multiproc import multiproc
        # ?? global
        runbrick.mp = multiproc(opt.threads)

    P = dict(W=3600, H=3600, brickid=brick, pipe=True)

    if opt.stamp:
        catalogfn = 'tractor-phot-b%06i-stamp.fits' % brick
        P.update(W=100, H=100)
    else:
        catalogfn = 'tractor-phot-b%06i.fits' % brick

    t0 = Time()
    R = stage0(**P)
    t1 = Time()
    print 'Stage0:', t1-t0
    P.update(R)
    R = stage1(**P)
    t2 = Time()
    print 'Stage1:', t2-t1
    P.update(R)
    P.update(catalogfn=catalogfn)
    stage202(**P)
    t202 = Time()
    print 'Stage202:', t202-t2

    # Plots
    ps = PlotSequence('brick-%06i' % brick)
    P.update(ps=ps)
    stage102(**P)
    t102 = Time()
    print 'Stage102:', t102-t202
    print 'Total:', t102 - t0
