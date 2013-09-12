if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import logging
import sys

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

def h2():
    T = fits_table('qso_BOSS_SDSS_MYERS_v5_6_0.fits')
    print 'Read', len(T), 'targets'
    # Match to SEQUELS region.
    T.cut((T.ra > 118) * (T.ra < 212) * (T.dec > 44) * (T.dec < 61))
    print 'Cut to', len(T)

    kd1 = tree_build_radec(T.ra, T.dec)

    fns = glob(os.path.join('sequels-phot', 'phot-*.fits'))
    fns.sort()
    print 'Found', len(fns), 'SEQUELS photometry output files'

    TT = []
    for fn in fns:
        print 'Reading', fn
        P = fits_table(fn, columns=['ra','dec','treated_as_pointsource',
                                    'x','y','coadd_id',
                                    'w1_nanomaggies','w1_nanomaggies_ivar',
                                    'w1_mag','w1_mag_err','w1_prochi2','w1_pronpix',
                                    'w1_profracflux', 'w1_proflux', 'w1_npix',
                                    'w2_nanomaggies','w2_nanomaggies_ivar',
                                    'w2_mag','w2_mag_err','w2_prochi2','w2_pronpix',
                                    'w2_profracflux', 'w2_proflux', 'w2_npix',
                                    ])
        print 'Got', len(P)
        kd2 = tree_build_radec(P.ra, P.dec)

        r = deg2dist(1. / 3600.)
        I,J,d = trees_match(kd1, kd2, r, nearest=True)
        print 'Matched', len(I)
        tree_free(kd2)

        P.cut(J)
        P.match_dist = dist2arcsec(d)
        P.qso_table_index = I
        TT.append(P)
    P = merge_tables(TT)
    P.writeto('h2.fits')
    


class myopts(object):
    pass

'''
text2fits.py -S 1 agn_coords.txt agn.fits
'''
if __name__ == '__main__':
    h2()
    sys.exit(0)


    T = fits_table('agn.fits')
    T.ra  = np.array([hmsstring2ra(s) for s in T.final_ra])
    T.dec = np.array([dmsstring2dec(s) for s in T.final_dec])

    print 'RA', T.ra
    print 'Dec', T.dec

    ps = PlotSequence('hennawi')

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

    #mp = multiproc(8)
    mp = multiproc(1)

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    r = np.hypot(dhi - dlo, (rhi - rlo) * np.cos(np.deg2rad((d0+d1)/2.))) / 2.
    RCF = radec_to_sdss_rcf((r0+r1)/2., (d0+d1)/2.,
                            tablefn='window_flist.fits')
    sdss = DR9()

    bands = 'ugriz'
    #bands = ['r']

    SS = []

    btims = {}
    for run,camcol,field,r,d in RCF:
        print 'RCF', run, camcol, field
        rr = sdss.get_rerun(run, field=field)
        if rr in [None, '157']:
            continue

        SS.append(get_tractor_sources_dr9(run, camcol, field, sdss=sdss,
                                          radecroi=[rlo,rhi,dlo,dhi],
                                          nanomaggies=True, fixedComposites=True,
                                          useObjcType=True, getsourceobjs=True))

        for band in bands:
            tim,inf = get_tractor_image_dr9(run, camcol, field, band, sdss=sdss,
                                            roiradecbox=[rlo,rhi,dlo,dhi],
                                            zrange=(-2,5), invvarIgnoresSourceFlux=True,
                                            nanomaggies=True)
            print 'Got', tim
            print 'shape:', tim.shape
            print 'Observed:', tim.time.toYear()

            print 'ROI', inf['roi']

            # HACK -- remove images entirely in 128-pix overlap region
            x0,x1,y0,y1 = inf['roi']
            if y0 > 1361:
                print 'Skipping image in overlap region'
                continue

            # Mask pixels outside ROI
            h,w = tim.shape
            wcs = tim.getWcs()
            XX,YY = np.meshgrid(np.arange(w), np.arange(h))
            xy = []
            for r,d in [(rlo,dlo),(rlo,dhi),(rhi,dhi),(rhi,dlo)]:
                x,y = wcs.positionToPixel(RaDecPos(r,d))
                xy.append((x,y))
            xy = np.array(xy)
            J = point_in_poly(XX.ravel(), YY.ravel(), xy)
            iv = tim.getInvvar()
            K = J.reshape(iv.shape)
            tim.setInvvar(iv * K)
            tim.rdmask = K

            if not band in btims:
                btims[band] = []
            btims[band].append(tim)

    cat = []
    for r,d in zip(T.ra, T.dec):
        cat.append(PointSource(RaDecPos(r,d),
                               NanoMaggies(order=bands,
                                           **dict([(band,1) for band in bands]))))

    # merge SDSS objects
    Tsdss = merge_tables([S for srcs,S in SS])
    Tsdss.index = np.arange(len(Tsdss))
    ss = []
    for srcs,S in SS:
        ss.extend(srcs)
    SS = ss
    print 'Got total of', len(SS), 'SDSS sources'
    # Remove duplicates
    I,J,d = match_radec(Tsdss.ra, Tsdss.dec, Tsdss.ra, Tsdss.dec,
                        1./3600., notself=True)
    keep = np.ones(len(Tsdss), bool)
    keep[np.maximum(I,J)] = False
    print 'Keeping', sum(keep), 'non-dup SDSS sources'
    Tsdss.cut(keep)
    # Remove matches with the target list
    I,J,d = match_radec(Tsdss.ra, Tsdss.dec, T.ra, T.dec, 1./3600.)
    print len(I), 'SDSS sources matched with targets'
    keep = np.ones(len(Tsdss), bool)
    keep[I] = False
    sdssmatch = Tsdss[I]
    Tsdss.cut(keep)
    print 'Kept', len(Tsdss), 'SDSS sources'
    # source objects
    sdssobjs = [SS[i] for i in Tsdss.index]

    # Record SDSS catalog mags
    print 'Matched sources:'
    for j in J:
        print '  ', SS[Tsdss.index[j]]

    for band in bands:
        iband = band_index(band)
        nm = np.zeros(len(T))
        nm[J] = sdssmatch.psfflux[:,iband]
        nm_ivar = np.zeros(len(T))
        nm_ivar[J] = sdssmatch.psfflux_ivar[:,iband]
        dnm = 1./np.sqrt(nm_ivar)
        mag = NanoMaggies.nanomaggiesToMag(nm)
        dmag = np.abs((-2.5 / np.log(10.)) * dnm / nm)

        # Luptitudes
        # mag2 = np.zeros(len(T))
        # dmag2 = np.zeros(len(T))
        # mag2[J] = sdssmatch.psfmag[:,iband]
        # dmag2[J] = sdssmatch.psfmagerr[:,iband]

        T.set('sdss_cat_%s' % band, mag)
        T.set('sdss_cat_%s_err' % band, dmag)
        #T.set('sdss_cat2_%s' % band, mag2)
        #T.set('sdss_cat2_%s_err' % band, dmag2)

    
    cat.extend(sdssobjs)
    print 'Total of', len(cat), 'sources'
    for src in cat:
        print '  ', src
    

    for band in bands:
        tims = btims[band]
        nims = len(tims)
        cols = int(np.ceil(np.sqrt(nims)))
        rows = int(np.ceil(nims / float(cols)))
        plt.clf()
        for i,tim in enumerate(tims):
            plt.subplot(rows, cols, i+1)
            ima = dict(interpolation='nearest', origin='lower',
                       vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
            plt.imshow(tim.getImage(), **ima)
            plt.xticks([]); plt.yticks([])
            plt.title(tim.name)
        plt.suptitle('Individual exposures: SDSS %s' % band)
        ps.savefig()
        for i,tim in enumerate(tims):
            plt.subplot(rows, cols, i+1)
            ax = plt.axis()
            xy = np.array([tim.getWcs().positionToPixel(RaDecPos(r,d))
                           for r,d in zip(T.ra, T.dec)])
            plt.plot(xy[:,0], xy[:,1], 'r+', ms=15, mew=1.)

            xy = np.array([tim.getWcs().positionToPixel(s.getPosition())
                           for s in sdssobjs])
            plt.plot(xy[:,0], xy[:,1], 'gx', ms=10, mew=1.)

            plt.axis(ax)
        ps.savefig()

        # ivmax = np.max([tim.getInvvar().max() for tim in tims])
        # plt.clf()
        # for i,tim in enumerate(tims):
        #     plt.subplot(rows, cols, i+1)
        #     ima = dict(interpolation='nearest', origin='lower',
        #                vmin=0, vmax=ivmax, cmap='gray')
        #     plt.imshow(tim.getInvvar(), **ima)
        #     plt.xticks([]); plt.yticks([])
        #     plt.title(tim.name)
        # plt.suptitle('Individual invvars: SDSS %s' % band)
        # ps.savefig()

        tractor = Tractor(tims, cat)
        t0 = Time()

        # print 'Tractor: all params', tractor.numberOfParams()
        # for nm,v in zip(tractor.getParamNames(), tractor.getParams()):
        #     print '  ', nm, v

        tractor.freezeParamsRecursive('*')
        tractor.thawPathsTo('sky')
        tractor.thawPathsTo(band)

        # print 'Tractor: all params', tractor.numberOfParams()
        # for nm,v in zip(tractor.getParamNames(), tractor.getParams()):
        #     print '  ', nm, v

        ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
            minsb=1e-3, mindlnp=1., sky=True, minFlux=None,
            fitstats=True, variance=True)
        print 'Forced phot took', Time()-t0

        plt.clf()
        for i,tim in enumerate(tims):
            plt.subplot(rows, cols, i+1)
            ima = dict(interpolation='nearest', origin='lower',
                       vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
            mod = tractor.getModelImage(i)
            plt.imshow(mod, **ima)
            plt.xticks([]); plt.yticks([])
            plt.title(tim.name)
        plt.suptitle('Models: SDSS %s' % band)
        ps.savefig()
        for i,tim in enumerate(tims):
            plt.subplot(rows, cols, i+1)
            ax = plt.axis()
            xy = np.array([tim.getWcs().positionToPixel(RaDecPos(r,d))
                           for r,d in zip(T.ra, T.dec)])
            plt.plot(xy[:,0], xy[:,1], 'r+', ms=15, mew=1.)
            xy = np.array([tim.getWcs().positionToPixel(s.getPosition())
                           for s in sdssobjs])
            plt.plot(xy[:,0], xy[:,1], 'gx', ms=10, mew=1.)
            plt.axis(ax)
        ps.savefig()

        # Trim off just the targets (not the extra SDSS objects)
        NT = len(T)
        nm = np.array([src.getBrightness().getBand(band)
                       for src in tractor.getCatalog()[:NT]])
        nm_ivar = IV[:NT]

        T.set('sdss_%s_nanomaggies' % band, nm)
        T.set('sdss_%s_nanomaggies_invvar' % band, nm_ivar)
        dnm = 1./np.sqrt(nm_ivar)
        mag = NanoMaggies.nanomaggiesToMag(nm)
        dmag = np.abs((-2.5 / np.log(10.)) * dnm / nm)
        T.set('sdss_%s_mag' % band, mag)
        T.set('sdss_%s_mag_err' % band, dmag)



    basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise_frames'
    wisedatadirs = [(basedir, 'merged'),]
    opt = myopts()
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
    #opt.force = range(100, 109)
    
    opt.write = True
    opt.ri = None
    opt.di = None

    for band in [1,2,3,4]:
        opt.bandnum = band
        opt.name = 'hennawi-w%i' % band
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
        p1 = plt.plot(tt - 2010, nobs, '-', color=cc)
        lp.append(p1)
        lt.append('W%i' % band)
    plt.xlabel('Date of observation (years - 2010.0)')
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
    T.writeto(resfn)
    print 'Wrote', resfn
