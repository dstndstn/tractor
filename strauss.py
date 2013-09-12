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
indexfn = os.path.join(basedir, 'WISE-index-L1b.fits')

datadir = 'strauss-data'

def _run_one((dataset, band, rlo, rhi, dlo, dhi, T, sfn)):
    opt = myopts()
    opt.wisedatadirs = wisedatadirs
    opt.minflux = None
    opt.sources = sfn
    opt.nonsdss = True
    opt.wsources = os.path.join(datadir, 'wise-objs-%s.fits' % dataset)
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
    opt.picklepat = os.path.join(datadir, opt.name + '-stage%0i.pickle')

    # Plots?
    opt.ps = opt.name
    #opt.ps = None

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
    return None
    


'''
Type II quasars in SDSS to forced-photometer in WISE, from Michael
Strauss

email of 2013-08-23 from Strauss: attach dustin.lis

text2fits.py -H "plate fiber mjd something ra dec" -f jjjddd dustin.lis strauss.fits

+ from Jenny Greene 2013-09-12:

>>      SDSS1309_0205.dat 2.2325 197.8264 2.09655
>>      SDSS2252_0108.dat 2.537 343.21960 1.14157

cat > strauss2.txt <<EOF
# z ra dec
2.2325 197.8264 2.09655
2.537 343.21960 1.14157
EOF
text2fits.py -f ddd strauss2.txt strauss2.fits

'''
if __name__ == '__main__':

    #sfn = 'strauss.fits'
    #fulldataset = 'strauss'

    sfn = 'strauss2.fits'
    fulldataset = 'strauss2'

    TT = fits_table(sfn)

    #mp = multiproc(8)
    mp = multiproc(1)

    margin = 0.003

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    # Cut to single objects

    args = []

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

        if True:
            # HACK -- got the WISE catalogs on riemann and the WISE exposures on NERSC...
            from wisecat import wise_catalog_radecbox
            cols=['cntr', 'ra', 'dec', 'sigra', 'sigdec', 'cc_flags',
                  'ext_flg', 'var_flg', 'moon_lev', 'ph_qual',
                  'w1mpro', 'w1sigmpro', 'w1sat', 'w1nm', 'w1m', 
                  'w1snr', 'w1cov', 'w1mag', 'w1sigm', 'w1flg',
                  'w2mpro', 'w2sigmpro', 'w2sat', 'w2nm', 'w2m',
                  'w2snr', 'w2cov', 'w2mag', 'w2sigm', 'w2flg',
                  'w3mpro', 'w3sigmpro', 'w3sat', 'w3nm', 'w3m', 
                  'w3snr', 'w3cov', 'w3mag', 'w3sigm', 'w3flg',
                  'w4mpro', 'w4sigmpro', 'w4sat', 'w4nm', 'w4m',
                  'w4snr', 'w4cov', 'w4mag', 'w4sigm', 'w4flg', ]
            W = wise_catalog_radecbox(rlo, rhi, dlo, dhi, cols=cols)
            if W is None:
                W = fits_table()
                for c in cols:
                    W.set(c, np.array([]))
            wfn = os.path.join(datadir, 'wise-objs-%s.fits' % dataset)
            W.writeto(wfn)
            print 'Wrote', wfn
            continue
    
        for band in [1,2,3,4]:
            pfn = os.path.join(datadir, '%s-w%i-stage108.pickle' % (dataset, band))
            if os.path.exists(pfn):
                print 'Output exists:', pfn, '; skipping'
                continue
            args.append((dataset, band, rlo, rhi, dlo, dhi, T, sfn))

    mp.map(_run_one, args)


    # Collate results
    results = []
    for i in range(len(TT)):
        T = TT[np.array([i])]
        dataset = '%s-%03i' % (fulldataset, i)

        resfn = os.path.join(datadir, '%s.fits' % dataset)

        gotall = True

        for band in [1,2,3,4]:
            pfn = os.path.join(datadir, '%s-w%i-stage106.pickle' % (dataset, band))
            if not os.path.exists(pfn):
                print 'File does not exist:', pfn, '; skipping'
                gotall = False
                break
            X = unpickle_from_file(pfn)
            R = X['R']

            nwise = 0
            nsdss = 0
            mpro = np.nan
            if len(R) > len(T):
                # print 'T:'
                # T.about()
                # print 'R:'
                # R.about()
                # print 'R.sdss?', R.sdss
                UW = X['UW']
                # print 'UW:'
                # UW.about()
                S = X['S']
                # print 'S:'
                # S.about()
                nwise = len(UW)
                nsdss = len(R) - len(UW) - len(T)

                if nwise:
                    mpro = UW.get('w%impro' % band)[0]
                    # UW items are at the end
                    R = R[:len(R)-nwise]

                if nsdss:
                    # Which one is the target T?
                    assert(len(T) == 1)
                    ir = np.argmin(np.hypot(T.ra[0] - R.ra, T.dec[0] - R.dec))
                    R = R[np.array([ir])]

            T.set('wise_n_near_w%i' % band, np.zeros(len(T), int) + nwise)
            T.set('other_sdss_near_w%i' % band, np.zeros(len(T), int) + nsdss)
            T.set('wise_near0_w%impro' % band, np.zeros(len(T), np.float32) + mpro)
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

        if not gotall:
            continue

        T.writeto(resfn)
        print 'Wrote', resfn

        results.append((i,T))

    I = np.array([i for i,T in results])
    R = merge_tables([T for i,T in results])

    origcols = TT.get_columns()
    for k in R.get_columns():
        if k in origcols:
            print 'Skipping original column', k
            continue
        print 'Adding column', k
        r = R.get(k)
        X = np.zeros(len(TT), r.dtype)
        X[I] = r
        TT.set(k, X)

    TT.writeto('%s-results.fits' % fulldataset)
