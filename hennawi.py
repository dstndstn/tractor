#! /usr/bin/env python

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

import fitsio

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
    

def h3():
    from sequels import read_photoobjs
    from wisecat import wise_catalog_radecbox

    '''
    collect results via:
    cp h3/h3-merged-r120-d0.fits h3.fits
    for ((r=121;r<240; r++)); do echo $r; tabmerge h3/h3-merged-r${r}-d0.fits+1 h3.fits+1; done
    for ((d=1;d<10;d++)); do for ((r=121;r<240; r++)); do echo $d $r; tabmerge h3/h3-merged-r${r}-d${d}.fits+1 h3.fits+1; done; done
    liststruc h3.fits
    '''
    
    d0 = 19.
    d1 = 20.
    r0 = 120.
    r1 = 121.

    lvl = logging.INFO
    #lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    dirname = 'h3'

    # d0 = 19.
    # d1 = 20.
    # r0 = 120.
    # r1 = 121.
    DD = np.arange(0, 10+1)
    RR = np.arange(120, 240+1)

    RRDD = []
    for d0,d1 in zip(DD, DD[1:]):
        for r0,r1 in zip(RR, RR[1:]):
            RRDD.append((r0,r1,d0,d1))

    for r0,r1,d0,d1 in RRDD:
        print 'Reading photoObjs...'
        fn = os.path.join(dirname, 'h3-sdss-r%.0f-d%.0f.fits' % (r0, d0))
        if not os.path.exists(fn):
            cols = ['objid', 'ra', 'dec', 'objc_type',
                    #'modelflux', 'modelflux_ivar',
                    #'psfflux', 'psfflux_ivar',
                    'modelmag', 'modelmagerr',
                    'psfmag', 'psfmagerr',
                    'resolve_status', 'nchild', 'flags', 'objc_flags',
                    'run','camcol','field','id']
            SDSS = read_photoobjs(r0, r1, d0, d1, 0.01, cols=cols)
            print 'Got', len(SDSS)
            SDSS.writeto(fn)
        else:
            print 'Reading', fn
            SDSS = fits_table(fn)
            print 'Got', len(SDSS)
    
        wfn = os.path.join(dirname, 'h3-wise-r%.0f-d%.0f.fits' % (r0,d0))
        print 'looking for', wfn
        if os.path.exists(wfn):
            WISE = fits_table(wfn)
            print 'Read', len(WISE), 'WISE sources nearby'
        else:
            cols = (['ra','dec','cntr'] + ['w%impro'%band for band in [1,2,3,4]] +
                    ['w%isigmpro'%band for band in [1,2,3,4]] +
                    ['w%isnr'%band for band in [1,2,3,4]])
            WISE = wise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
            WISE.writeto(wfn)
            print 'Found', len(WISE), 'WISE sources nearby'
    
        WISE.cut(WISE.w4snr >= 5)
        print 'Cut to', len(WISE), 'on W4snr'
        WISE.cut((WISE.w1snr >= 5) * (WISE.w2snr >= 5))
        print 'Cut to', len(WISE), 'on W[12]snr'
        
        I,J,d = match_radec(WISE.ra, WISE.dec, SDSS.ra, SDSS.dec, 1./3600., nearest=True)
        print 'Matched', len(I)
    
        WISE.match_dist = np.zeros(len(WISE))
        WISE.match_dist[I] = d
        WISE.matched = np.zeros(len(WISE), bool)
        WISE.matched[I] = True
        for c in SDSS.get_columns():
            print 'SDSS column', c
            S = SDSS.get(c)
            sh = (len(WISE),)
            if len(S.shape) > 1:
                sh = sh + S.shape[1:]
            X = np.zeros(sh, S.dtype)
            X[I] = S[J]
            WISE.set('sdss_' + c, X)
    
        fn = os.path.join(dirname, 'h3-merged-r%.0f-d%.0f.fits' % (r0,d0))
        WISE.writeto(fn)
        print 'Wrote', fn



def sdss_forced_phot(r0,r1,d0,d1, rlo, rhi, dlo, dhi, T, ps,
                     bands = 'ugriz',
                     sdss = DR9(),
                     fitsky=False):
    dec = (dlo + dhi)/2.
    r = np.hypot(dhi - dlo, (rhi - rlo) * np.cos(np.deg2rad(dec))) / 2.
    RCF = radec_to_sdss_rcf((r0+r1)/2., (d0+d1)/2.,
                            tablefn='window_flist.fits')
    print 'Run,Camcol,Fields:', RCF

    SS = []

    btims = {}
    for run,camcol,field,r,d in RCF:
        print 'RCF', run, camcol, field
        rr = sdss.get_rerun(run, field=field)
        print 'Rerun', rr
        if rr in [None, '157']:
            continue

        s = get_tractor_sources_dr9(run, camcol, field, sdss=sdss,
                                    radecroi=[rlo,rhi,dlo,dhi],
                                    nanomaggies=True, fixedComposites=True,
                                    useObjcType=True, getsourceobjs=True)
        if s == []:
            continue
        SS.append(s)

        for band in bands:
            tim,inf = get_tractor_image_dr9(run, camcol, field, band, sdss=sdss,
                                            roiradecbox=[rlo,rhi,dlo,dhi],
                                            zrange=(-2,5),
                                            invvarIgnoresSourceFlux=True,
                                            invvarAtCenterImage=True,
                                            nanomaggies=True,
                                            psf='dg')
            print 'Got tim', tim
            if tim is None:
                continue
            #print 'shape:', tim.shape
            #print 'Observed:', tim.time.toYear()
            #print 'ROI', inf['roi']

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

    if len(SS) == 0:
        return
    
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
    if len(SS):
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

    else:
        sdssobjs = []

    print 'Total of', len(cat), 'sources'
    for src in cat:
        print '  ', src
    

    for band in bands:

        if not band in btims:
            continue

        tims = btims[band]
        if ps:
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
            #ps.savefig()
            for i,tim in enumerate(tims):
                plt.subplot(rows, cols, i+1)
                ax = plt.axis()
                xy = np.array([tim.getWcs().positionToPixel(RaDecPos(r,d))
                               for r,d in zip(T.ra, T.dec)])
                plt.plot(xy[:,0], xy[:,1], 'r+', ms=15, mew=1.)
    
                if len(sdssobjs):
                    xy = np.array([tim.getWcs().positionToPixel(s.getPosition())
                                   for s in sdssobjs])
                    plt.plot(xy[:,0], xy[:,1], 'gx', ms=10, mew=1.)
    
                plt.axis(ax)
            ps.savefig()


            # invvar
            plt.clf()
            for i,tim in enumerate(tims):
                plt.subplot(rows, cols, i+1)
                ima = dict(interpolation='nearest', origin='lower',
                           vmin=0, cmap='gray')
                iv = tim.getInvvar()
                plt.imshow(iv, **ima)
                plt.xticks([]); plt.yticks([])
                plt.title(tim.name)
                plt.colorbar()
            plt.suptitle('Invvars: SDSS %s' % band)
            ps.savefig()

        tractor = Tractor(tims, cat)
        t0 = Time()

        tractor.freezeParamsRecursive('*')
        if fitsky:
            tractor.thawPathsTo('sky')
        tractor.thawPathsTo(band)

        # Note that we have 5-band NanoMaggies objects, so the bands
        # don't affect each other (ie, no need to worry about
        # re-setting them to a default value)

        # ceres block size
        sz = 10

        wantims = (ps is not None)

        for fiti, tag in enumerate(['', '_b']):

            pargs = dict(minsb=1e-3, mindlnp=1., minFlux=None,
                         sky=fitsky, fitstats=True, variance=True,
                         shared_params=False, wantims=wantims)

            t0 = Time()
            if fiti == 0:
                R = tractor.optimize_forced_photometry(
                    use_ceres=True,
                    BW=sz, BH=sz, **pargs)
            else:
                optworked = False
                try:
                    R = tractor.optimize_forced_photometry(**pargs)
                    optworked = True
                except:
                    import traceback
                    print 'WARNING: optimize_forced_photometry failed:'
                    traceback.print_exc()
                    print
                    
            print 'Forced phot took', Time()-t0

            IV = R.IV
            fitstats = R.fitstats
            if wantims:
                ims0 = R.ims0
                ims1 = R.ims1

            if ps:
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
                #ps.savefig()
                for i,tim in enumerate(tims):
                    plt.subplot(rows, cols, i+1)
                    ax = plt.axis()
                    xy = np.array([tim.getWcs().positionToPixel(RaDecPos(r,d))
                                   for r,d in zip(T.ra, T.dec)])
                    plt.plot(xy[:,0], xy[:,1], 'r+', ms=15, mew=1.)
                    if len(sdssobjs):
                        xy = np.array([tim.getWcs().positionToPixel(s.getPosition())
                                       for s in sdssobjs])
                        plt.plot(xy[:,0], xy[:,1], 'gx', ms=10, mew=1.)
                    plt.axis(ax)
                ps.savefig()


    
            # Trim off just the targets (omit the extra SDSS objects)
            NT = len(T)
            nm = np.array([src.getBrightness().getBand(band)
                           for src in tractor.getCatalog()[:NT]])
            nm_ivar = IV[:NT]
    
            T.set('sdss_%s_nanomaggies%s' % (band, tag), nm)
            T.set('sdss_%s_nanomaggies_invvar%s' % (band, tag), nm_ivar)
            dnm = 1./np.sqrt(nm_ivar)
            mag = NanoMaggies.nanomaggiesToMag(nm)
            dmag = np.abs((-2.5 / np.log(10.)) * dnm / nm)
            T.set('sdss_%s_mag%s' % (band, tag), mag)
            T.set('sdss_%s_mag_err%s' % (band, tag), dmag)
    
            if fitstats is not None:
                fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
                for k in fskeys:
                    T.set(k + '_' + band + tag, getattr(fitstats, k).astype(np.float32))

            if fitsky:
                skies = np.array([tim.getSky().val for tim in tims])
                z = np.zeros(len(T), int)
                T.set('sky_%s_%s_n' % (band, tag), z + len(skies))
                z = np.zeros(len(T), np.float32)
                T.set('sky_%s_%s_mean' % (band, tag), z + np.mean(skies))
                T.set('sky_%s_%s_min'  % (band, tag), z + np.min(skies))
                T.set('sky_%s_%s_max'  % (band, tag), z + np.max(skies))
                T.set('sky_%s_%s_med'  % (band, tag), z + np.median(skies))

            # ceres
            if fiti == 0:
                stat = R.ceres_status
                func_tol = (stat['termination'] == 2)
                steps = stat['steps_successful']
                T.set('fit_ok_%s%s' % (band, tag), np.array([(func_tol and steps > 0)] * len(T)))
            else:
                T.set('fit_ok_%s%s' % (band, tag), np.array([optworked] * len(T)))




def redqsos():
    # W4 detections without SDSS matches.
    T = fits_table('w4targets.fits')
    ps = None
    #ps = PlotSequence('redqso')

    arr = os.environ.get('PBS_ARRAYID')
    tag = '-b'
    if arr is not None:
        arr = int(arr)
        #chunk = 100
        chunk = 50
        T = T[arr * chunk: (arr+1) * chunk]
        print 'Cut to chunk', (arr * chunk)
        tag = '-%03i' % arr
    
    sdss = DR9()
    sdss.useLocalTree()
    sdss.saveUnzippedFiles('data/unzip')

    mp = multiproc(1)

    #lvl = logging.DEBUG
    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    newcols = {}
    origcols = T.get_columns()
    
    T.done = np.zeros(len(T), np.uint8)

    version = get_svn_version()
    print 'SVN version info:', version

    hdr = fitsio.FITSHDR()
    hdr.add_record(dict(name='PHOT_VER', value=version['Revision'],
                        comment='SVN revision'))
    hdr.add_record(dict(name='PHOT_URL', value=version['URL'], comment='SVN URL'))
    hdr.add_record(dict(name='PHOT_DAT', value=datetime.datetime.now().isoformat(),
                        comment='forced phot run time'))

    for i,(ra,dec) in enumerate(zip(T.ra, T.dec)):
        print
        print i
        print 'RA,Dec', ra, dec
        r0,r1 = ra,ra
        d0,d1 = dec,dec

        #margin = 0.003
        # ~9 SDSS pixel half-box
        margin = 0.001
        
        dr = margin / np.cos(np.deg2rad((d0+d1)/2.))
        rlo = r0 - dr
        rhi = r1 + dr
        dlo = d0 - margin
        dhi = d1 + margin

        sky = True
        #sky = False

        t = T[np.array([i])]
        sdss_forced_phot(r0,r1,d0,d1, rlo, rhi, dlo, dhi, t, ps,
                         sdss=sdss, fitsky=sky)
        #print 'Columns:', t.get_columns()
        #t.about()
        
        for key in t.get_columns():
            if key in origcols:
                continue
            val = t.get(key)
            if not key in newcols:
                newcols[key] = np.zeros(len(T), val.dtype)
                T.set(key, newcols[key])
            print 'set', key, i, '=', val[0]
            newcols[key][i] = val[0]
        T.done[i] = 1

        #if i and i % 100 == 0:
        if False:
            fn = 'wisew4phot-interim%s.fits' % tag
            T.writeto(fn, header=hdr)
            print 'Wrote', fn
            
    T.writeto('wisew4phot%s.fits' % tag, header=hdr)

class myopts(object):
    pass

def finish_redqsos():
    TT = [fits_table('wisew4phot-%03i.fits' % i) for i in range(642)]
    T = merge_tables(TT, columns='fillzero')
    T.writeto('wisew4phot.fits')

'''
text2fits.py -S 1 agn_coords.txt agn.fits
'''
if __name__ == '__main__':
    #h2()
    #h3()

    #redqsos()
    finish_redqsos()

    sys.exit(0)

    # merge
    TT = []
    allcols = set()
    coltypes = dict()
    for i in range(132):
        fn = 'wisew4phot-%03i.fits' % i
        T = fits_table(fn)
        print 'read', len(T), 'from', fn
        #T.about()
        TT.append(T)
        newcols = set(T.get_columns()) - allcols
        for c in newcols:
            coltypes[c] = T.get(c).dtype
        allcols.update(newcols)

    for i,T in enumerate(TT):
        diff = allcols - set(T.get_columns())
        if len(diff) == 0:
            continue
        print
        print 'File', i
        for c in diff:
            print 'Set', c, 'to all zero'
            T.set(c, np.zeros(len(T), coltypes[c]))

    T = merge_tables(TT)
    T.writeto('wisew4phot-2.fits')

    sys.exit(0)

    # (from w4.py)
    from astrometry.util.starutil_numpy import *
    T = fits_table('wisew4_nomatch.fits')
    l,b = radectolb(T.ra, T.dec)
    T.cut(np.abs(b) > 10.)
    l,b = radectolb(T.ra, T.dec)
    gdist = np.hypot(l, b)
    T.cut(gdist > 30.)
    u,v = radectoecliptic(T.ra, T.dec)
    T.cut(np.abs(v) > 5)
    T.cut((T.ra > 125) * (T.ra < 225) * (T.dec > 0) * (T.dec < 60))



    # T = fits_table('agn.fits')
    # T.ra  = np.array([hmsstring2ra(s) for s in T.final_ra])
    # T.dec = np.array([dmsstring2dec(s) for s in T.final_dec])
    # sfn = 'agn2.fits'
    # T.writeto(sfn)
    # do_sdss = True
    # resfn = 'agn3.fits'
    # wsources = 'wise-objs-hennawi.fits'

    sfn = 'elgordo.fits'
    resfn = 'elgordo-wise.fits'

    wsources = 'wise-objs-elgordo.fits'
    do_sdss = False
    ceres = False

    indexfn = 'WISE-index-L1b.fits'
    #indexfn = None
    basedir = 'wise-frames'
    #basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise_frames'
    wisedatadirs = [(basedir, 'merged'),]


    T = fits_table(sfn)
    #print 'RA', T.ra
    #print 'Dec', T.dec

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

    #mp = multiproc(8)
    mp = multiproc(1)

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if do_sdss:
        sdss_forced_phot(r0,r1,d0,d1, rlo, rhi, dlo, dhi, T)


    opt = myopts()
    opt.wisedatadirs = wisedatadirs
    opt.minflux = None
    opt.sources = sfn
    opt.nonsdss = True
    opt.wsources = wsources
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
            runtostage(108, opt, mp, rlo,rhi,dlo,dhi, indexfn=indexfn, ceres=ceres)
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
        print 'R', len(R)
        print 'T', len(T)
        assert(sum(R.sdss) == len(T))
        R.cut(R.sdss > 0)
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
