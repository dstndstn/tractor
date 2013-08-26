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

'''
Type II quasars in SDSS to forced-photometer in WISE, from Michael
Strauss

email of 2013-08-23 from Strauss: attach dustin.lis

text2fits.py -H "plate fiber mjd something ra dec" -f jjjddd dustin.lis strauss.fits
'''
if __name__ == '__main__':
    sfn = 'strauss.fits'
    fulldataset = 'strauss'
    TT = fits_table(sfn)
    ps = PlotSequence(fulldataset)

    margin = 0.003

    #resfn = 'strauss-results.fits'

    #mp = multiproc(8)
    mp = multiproc(1)

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    basedir = 'wise-frames'
    wisedatadirs = [(basedir, 'merged'),]

    # Cut to single objects

    for i in range(len(TT)):
        T = TT[np.array([i])]

        dataset = '%s-%03i' % (fulldataset, i)

        r0,r1 = T.ra.min(),  T.ra.max()
        d0,d1 = T.dec.min(), T.dec.max()

        dr = margin / np.cos(np.deg2rad((d0+d1)/2.))
        rlo = r0 - dr
        rhi = r1 + dr
        dlo = d0 - margin
        dhi = d1 + margin

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
        #opt.force = [104, 105, 106, 107, 108]
        #opt.force = [104]
        #opt.force = range(100, 109)
        opt.write = True
        opt.ri = None
        opt.di = None
    
        for band in [1,2,3,4]:
            opt.bandnum = band
            opt.name = '%s-w%i' % (dataset, band)
            opt.picklepat = opt.name + '-stage%0i.pickle'
            opt.ps = opt.name
    
            try:
                #runtostage(110, opt, mp, rlo,rhi,dlo,dhi)
                runtostage(108, opt, mp, rlo,rhi,dlo,dhi)
                #runtostage(700, opt, mp, rlo,rhi,dlo,dhi)
            except:
                import traceback
                print
                traceback.print_exc()
                print
                pass
    
        for band in [1,2,3,4]:
            pfn = '%s-w%i-stage106.pickle' % (dataset, band)
            X = unpickle_from_file(pfn)
            R = X['R']
            assert(len(R) == len(T))
            nm = R.get('w%i' % band)
            nm_ivar = R.get('w%i_ivar' % band)
            T.set('w%i_nanomaggies' % band, nm)
            T.set('w%i_nanomaggies_invvar' % band, nm_ivar)
            dnm = 1./np.sqrt(nm_ivar)
            mag = NanoMaggies.nanomaggiesToMag(nm)
            dmag = np.abs((-2.5 / np.log(10.)) * dnm / nm)
            T.set('w%i_mag' % band, mag)
            T.set('w%i_mag_err' % band, dmag)
        T.writeto(resfn)
        print 'Wrote', resfn
