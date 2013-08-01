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

from wise3 import *
from tractor import *

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

    margin = 0.003
    dr = margin / np.cos(np.deg2rad((d0+d1)/2.))
    rlo = r0 - dr
    rhi = r1 + dr
    dlo = d0 - margin
    dhi = d1 + margin

    sfn = 'agn2.fits'
    resfn = 'agn3.fits'

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
    opt.force = []
    #opt.force = [104, 105, 106, 107, 108]
    #opt.force = [104]
    opt.force = range(104, 109)
    
    opt.write = True
    opt.ri = None
    opt.di = None

    #for band in [1,2,3,4]:
    for band in [4]:
        opt.bandnum = band
        opt.name = 'hennawi-w%i' % band
        opt.picklepat = opt.name + '-stage%0i.pickle'
        opt.ps = opt.name

        try:
            #runtostage(110, opt, mp, rlo,rhi,dlo,dhi)
            runtostage(108, opt, mp, rlo,rhi,dlo,dhi)
            runtostage(700, opt, mp, rlo,rhi,dlo,dhi)
        except:
            import traceback
            print
            traceback.print_exc()
            print
            pass

    alltims = []
    for band in [1,2,3,4]:
        pfn = 'hennawi-w%i-stage101.pickle' % band
        X = unpickle_from_file(pfn)
        alltims.append(X['tims'])

    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, right=0.9,
                        bottom=0.1, top=0.9)

    plt.clf()
    mjd2k = datetomjd(J2000)
    y0 = TAITime(None, mjd=mjd2k).toYears()
    lt,lp = [],[]
    for band,tims,cc in zip([1,2,3,4], alltims, 'mbgr'):
        times = np.sort(np.array([(tim.time.toYears() - y0) + 2000. for tim in tims]))
        nobs = np.arange(len(times)+1).repeat(2)[1:-1]
        tt = times.repeat(2)
        tt[0] += - 1.
        tt[-1] += 1.
        
        #p1 = plt.plot(times - 2010, np.zeros(len(times)) + band, 'o', color=cc)
        #plt.axhline(band, color=cc, alpha=0.5)
        p1 = plt.plot(tt - 2010, nobs, '-', color=cc)
        lp.append(p1)
        lt.append('W%i' % band)
    plt.xlabel('Date of observation (years - 2010.0)')
    #plt.yticks([])
    #plt.ylim(-1, 5)
    plt.ylabel('Cumulative number of observations')
    plt.xlim(0,1)
    plt.legend(lp, lt, loc='upper left')
    ps.savefig()

    for band in [1,2,3,4]:
        pfn = 'hennawi-w%i-stage106.pickle' % band
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

        # cat = X['cat2']
        # nm = []
        # assert(len(cat) == len(T))
        # for src in cat:
        #     nm.append(src.getBrightness())
        # nm = np.array(nm)
        # T.set('w%i_nanomaggies' % band, nm)
        # T.set('w%i_mag' % band, NanoMaggies.nanomaggiesToMag(nm))
    T.writeto(resfn)
    print 'Wrote', resfn
