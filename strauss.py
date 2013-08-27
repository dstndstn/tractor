if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import logging

import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
from astrometry.util.multiproc import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.libkd.spherematch import *
from astrometry.sdss import *

from wise3 import *
from tractor import *

class myopts(object):
    pass


basedir = 'wise-frames'
wisedatadirs = [(basedir, 'merged'),]

#indexfn = None
indexfn = 'WISE-index-L1b.fits'

def _run_one((dataset, band, rlo, rhi, dlo, dhi, T, sfn)):
    opt = myopts()
    opt.wisedatadirs = wisedatadirs
    opt.minflux = None
    opt.sources = sfn
    opt.nonsdss = True
    opt.wsources = 'wise-objs-%s.fits' % dataset
    opt.osources = None
    opt.minsb = 0.005
    opt.ptsrc = False
    opt.pixpsf = False
    opt.force = []
    #opt.force = [104]
    opt.write = True
    opt.ri = None
    opt.di = None
    opt.bandnum = band
    opt.name = '%s-w%i' % (dataset, band)
    opt.picklepat = opt.name + '-stage%0i.pickle'

    # No plots
    #opt.ps = opt.name
    opt.ps = None

    mp = multiproc()

    try:
        #runtostage(110, opt, mp, rlo,rhi,dlo,dhi)
        runtostage(108, opt, mp, rlo,rhi,dlo,dhi, indexfn=indexfn)
        #runtostage(700, opt, mp, rlo,rhi,dlo,dhi)
    except:
        import traceback
        print
        traceback.print_exc()
        print
