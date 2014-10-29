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
        #runbrick.mp = multiproc(opt.threads)

        from utils.debugpool import DebugPool, DebugPoolMeas
        dpool = DebugPool(opt.threads, taskqueuesize=2*opt.threads)
        mp = multiproc(pool=dpool)
        Time.add_measurement(DebugPoolMeas(dpool))
        runbrick.mp = mp

    P = dict(W=3600, H=3600, brickid=brick, pipe=True)

    if opt.stamp:
        catalogfn = 'tractor-phot-b%06i-stamp.fits' % brick
        pspat = 'pipebrick-plots/brick-%06i-stamp' % brick
        P.update(W=100, H=100)
    else:
        catalogfn = 'pipebrick-cats/tractor-phot-b%06i.fits' % brick
        pspat = 'pipebrick-plots/brick-%06i' % brick

    t0 = Time()
    R = stage_tims(**P)
    P.update(R)
    t1 = Time()
    print 'Stage tims:', t1-t0

    R = stage_srcs(**P)
    P.update(R)
    t2 = Time()
    print 'Stage srcs:', t2-t1

    R = stage_fitblobs(**P)
    P.update(R)
    t2b = Time()
    print 'Stage fitblobs:', t2b-t2

    P.update(catalogfn=catalogfn)
    stage_writecat(**P)
    t3 = Time()
    print 'Stage writecat:', t3-t2b

    # Plots
    
    ps = PlotSequence(pspat)
    P.update(ps=ps, outdir='pipebrick-plots')
    stage_fitplots(**P)
    t4 = Time()
    print 'Stage fitplots:', t4-t3
    print 'Total:', t4 - t0
