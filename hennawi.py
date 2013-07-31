if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import logging

import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
from astrometry.util.multiproc import *

from wise3 import *

class myopts(object):
    pass

'''
text2fits.py -S 1 agn_coords.txt agn.fits
'''
if __name__ == '__main__':
    T = fits_table('agn.fits')
    T.ra  = np.array([hmsstring2ra(s) for s in T.final_ra])
    T.dec = np.array([dmsstring2dec(s) for s in T.final_dec])

    print 'RA', T.ra
    print 'Dec', T.dec

    ps = PlotSequence('hennawi')

    plt.clf()
    plt.plot(T.ra, T.dec, 'r.')
    ps.savefig()

    r0,r1 = T.ra.min(),  T.ra.max()
    d0,d1 = T.dec.min(), T.dec.max()
    print 'RA range', r0,r1
    print 'Dec range', d0,d1

    margin = 0.001
    dr = margin / np.cos(np.deg2rad((d0+d1)/2.))
    rlo = r0 - dr
    rhi = r1 + dr
    dlo = d0 - margin
    dhi = d1 + margin

    sfn = 'agn2.fits'

    T.writeto(sfn)

    opt = myopts()

    #mp = multiproc(8)
    mp = multiproc(1)

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise_frames'
    wisedatadirs = [(basedir, 'merged'),]
    opt.wisedatadirs = wisedatadirs
    opt.minflux = None
    opt.sources = sfn
    opt.nonsdss = True
    opt.wsources = 'wise-objs-hennawi.fits'
    opt.osources = None
    opt.minsb = 0.005
    opt.ptsrc = False
    opt.pixpsf = False
    #opt.force = []
    opt.force = [106, 107, 108]

    opt.write = True
    opt.ri = None
    opt.di = None

    for band in [1,2,3,4]:
        opt.bandnum = band
        opt.name = 'hennawi-w%i' % band
        opt.picklepat = opt.name + '-stage%0i.pickle'
        opt.ps = opt.name

        try:
            runtostage(108, opt, mp, rlo,rhi,dlo,dhi)
            runtostage(700, opt, mp, rlo,rhi,dlo,dhi)
        except:
            import traceback
            print
            traceback.print_exc()
            print
            pass
    
