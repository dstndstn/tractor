import matplotlib
matplotlib.use('Agg')
import pylab as plt
import sys

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *

from sequels import get_photoobj_filename

F = fits_table('window_flist.fits')
print len(F), 'fields'
rr = '301'
F.cut(F.rerun == rr)
print len(F), 'rerun', rr

if False:
    # Start from the back.
    I = np.arange(len(F)-1, -1, -1)
    F = F[I]


    if True:
        # ra,dec = 3.1, 14.2
        # run,camcol,field = 2566, 2, 329
        # coadd = '0031p136'

        # ra,dec = 162.5, -24.7
        # run,camcol,field = 5999, 2, 11
        # coadd = '1618m243'

        # ra,dec = 46.7, 9.9
        # run, camcol, field = 4334, 3, 15
        # coadd = '0461p106'

        # ra,dec = 41.4, 0.4
        # run, camcol, field = 4874, 4, 698
        # coadd = '0408p000'

        # ra,dec = 149.4, 9.6
        # run, camcol, field = 3630, 2, 220
        # coadd = '1501p090'

        ra,dec = 58.1, -0.4
        #run, camcol, field = 4136, 3, 206
        run, camcol, field = 4136, 5, 206
        coadd = '0574p000'

        rr = '301'
        dirnm = 'data/sequels-phot-5'
        ps = PlotSequence('check')

        pfn = get_photoobj_filename(rr, run, camcol, field)
        T = fits_table(pfn)
        T.star_type = (T.objc_type == 6)
        T.gal_type = (T.objc_type == 3)
        T.primary = ((T.resolve_status & 256) > 0)
        T.shouldphot = np.logical_and(T.primary,
                                      np.logical_or(T.star_type, T.gal_type))

        wfn = os.path.join('data/redo3', rr, '%i'%run, '%i'%camcol,
                           'photoWiseForced-%06i-%i-%04i.fits' % (run, camcol, field))
        W = fits_table(wfn)
        wisephot = W.has_wise_phot.astype(np.uint8)
        W.wisephot = (wisephot == ord('T'))
        W.delete_column('has_wise_phot')

        A = fits_table('sdss2-atlas.fits')
        mI,mJ,d = match_radec(A.ra, A.dec, np.array([ra]), np.array([dec]), 3.)
        A.cut(mI)
        print 'Nearby coadd tiles in sdss2-atlas:', A.coadd_id

        A = fits_table('wise_allsky_4band_p3as_cdd.fits')
        mI,mJ,d = match_radec(A.ra, A.dec, np.array([ra]), np.array([dec]), 3.)
        A.cut(mI)
        rds = []
        print 'Nearby coadd tiles:', A.coadd_id
        for a in A:
            ww,hh = a.naxis1, a.naxis2
            wcs = Tan(a.ra, a.dec, a.crpix1, a.crpix2, a.cdelt1, 0., 0., a.cdelt2,
                      float(ww), float(hh))
            rd = np.array([wcs.pixelxy2radec(x,y) for x,y in [(1,1),(1,hh),(ww,hh),(ww,1),(1,1)]])
            rds.append(rd)

        plt.clf()
        plt.plot(T.ra, T.dec, 'bo', mec='b', mfc='none')
        plt.plot(T.ra[T.shouldphot], T.dec[T.shouldphot], 'gs', mec='g', mfc='none')
        plt.plot(W.ra[W.ra != 0], W.dec[W.dec != 0], 'r.')
        if np.sum(W.wisephot):
            plt.plot(W.ra[W.wisephot], W.dec[W.wisephot], 'mx', ms=8)
        #ax1 = plt.axis()
        ax1 = [ra - 0.2, ra + 0.2, dec - 0.2, dec + 0.2]
        fn = os.path.join(dirnm, 'phot-%s.fits' % coadd)
        print 'opening', fn
        try:
            P = fits_table(fn)
            plt.plot(P.ra, P.dec, 'k.', alpha=0.1, zorder=1)
        except:
            P = None
        #fn = mfn.replace('.fits','.png')
        for i,rd in enumerate(rds):
            plt.plot(rd[:,0], rd[:,1], 'r-')
            plt.text(np.mean(rd[:,0]), np.mean(rd[:,1]), A.coadd_id[i].replace('_ab41',""), fontsize=8,
                     ha='center')
        ps.savefig()
        ax = [ra - 0.5, ra + 0.5, dec-0.5, dec+0.5]
        plt.axis(ax)
        ps.savefig()
    
        plt.axis(ax1)
        ps.savefig()
    
        try:
            plt.clf()
            U = fits_table('data/redo3/phot-unsplit-%s.fits' % coadd)
            plt.plot(U.ra, U.dec, 'k.', alpha=0.1)
            plt.axis(ax)
            plt.title('unsplit')
            ps.savefig()
        except:
            U = None
            pass
    
        plt.clf()
        #U = fits_table('data/sequels-phot-5/phot-1043p000.fits')
        if P:
            plt.plot(P.ra, P.dec, 'k.', alpha=0.1)

        if U:
            rcf = np.unique(zip(U.run, U.camcol, U.field))
            print 'Unique rcf:', rcf
            for r,c,f in rcf:
                ii = np.flatnonzero((F.run == r) * (F.camcol == c) * (F.field == f))
                ii = ii[0]
                plt.text(F.ra[ii], F.dec[ii], '%i/%i/%i' % (r,c,f), rotation=90, color='r')
        plt.axis(ax)
        plt.title('phot')
        ps.savefig()
    
        plt.clf()
        U = fits_table('sdss-phot-temp/photoobjs-%s.fits' % coadd)
        plt.plot(U.ra, U.dec, 'k.', alpha=0.1)
        plt.axis(ax)
        plt.title('photoobjs')
        ps.savefig()

        sys.exit(0)


F.cut(F.run == 4136)

for run,camcol,field in zip(F.run, F.camcol, F.field):

    dirnm = 'data/redo3'

    #if run < 211:
    #    continue
    # if not ((run,camcol,field) in [(307,1,186),
    #                                #(308,1,11),
    #                                #(1000,6,179),
    #                                #(1000,6,180),
    #                                ]):
    #     continue
    # if not ((run,camcol,field) in [(4334, 3, 13),
    #                                (4334, 3, 14),
    #                                (4334, 3, 15),
    #                                (4334, 3, 16),
    #                                (4334, 3, 17),
    #                                (4334, 4, 11),
    #                                (4334, 4, 18),
    #                                (4334, 4, 19),
    #                                (4334, 5, 20),
    #                                (4334, 6, 11),
    #                                (4334, 6, 18),
    #                                (4334, 6, 19),
    #                                (4334, 6, 20),
    #                                ]):
    #     continue
    # if not ((run,camcol,field) in [(4874, 4, 698),]):
    #         continue
    # if not ((run,camcol,field) in [(3630, 2, 220),]):
    #     continue
    if not ((run,camcol,field) in [(4136, 3, 206),
                                   (4136, 5, 206)]):
        continue
    dirnm = 'data/redo5'
    
    pfn = get_photoobj_filename(rr, run, camcol, field)
    T = fits_table(pfn, columns=['resolve_status', 'objc_type'])
    if T is None:
        print 'WARNING:', run, camcol, field, ': empty file', pfn
        continue
    print 'Read', len(T), 'from', pfn
    T.primary = ((T.resolve_status & 256) > 0)
    nprim = np.sum(T.primary)
    print nprim, 'PRIMARY'
    if nprim == 0:
        continue

    T.star_type = (T.objc_type == 6)
    T.gal_type = (T.objc_type == 3)
    T.shouldphot = np.logical_and(T.primary,
                                  np.logical_or(T.star_type, T.gal_type))
    ns = np.sum(T.shouldphot)
    print ns, 'should be photometered'
    if ns == 0:
        continue

    wfn = os.path.join(dirnm, rr, '%i'%run, '%i'%camcol,
                      'photoWiseForced-%06i-%i-%04i.fits' % (run, camcol, field))
    cols = ['has_wise_phot']
    if os.path.exists(wfn):
        W = fits_table(wfn, columns=cols)
    else:
        print 'WARNING:', run, camcol, field, ': missing file', wfn
        wfn = os.path.join('data/sdss-pobj', rr, '%i'%run, '%i'%camcol,
                          'photoWiseForced-%06i-%i-%04i.fits' % (run, camcol, field))
        if not os.path.exists(wfn):
            print 'WARNING:', run, camcol, field, ': missing file', wfn
            continue
        W = fits_table(wfn, columns=cols)

    print 'Read', len(W), 'from', wfn

    print 'has_wise_phot:', np.unique(W.has_wise_phot)
    wisephot = W.has_wise_phot.astype(np.uint8)
    print 'wisephot:', np.unique(wisephot)
    # HACK -- fitsio issues with booleans
    W.wisephot = (wisephot == ord('T'))
    print 'W.wisephot:', np.unique(W.wisephot)

    T.star_type = (T.objc_type == 6)
    T.gal_type = (T.objc_type == 3)
    T.shouldphot = np.logical_and(T.primary,
                                  np.logical_or(T.star_type, T.gal_type))

    print np.sum(T.shouldphot), 'SDSS should be photometered;', np.sum(W.wisephot), 'with WISE photometry'

    I = np.flatnonzero(np.logical_xor(W.wisephot, T.shouldphot))

    if len(I) == 0:
        print 'OK:', np.sum(W.wisephot), np.sum(T.shouldphot)
        continue

    T = fits_table(pfn)
    T.star_type = (T.objc_type == 6)
    T.gal_type = (T.objc_type == 3)
    T.primary = ((T.resolve_status & 256) > 0)
    T.shouldphot = np.logical_and(T.primary,
                                  np.logical_or(T.star_type, T.gal_type))
    print 'ERROR:', run, camcol, field, ':', len(I), 'unmatched'

    W = fits_table(wfn)
    W.wisephot = (wisephot == ord('T'))
    W.delete_column('has_wise_phot')

    mfn = 'missing-%06i-%i-%04i.fits' % (run,camcol,field)

        # coadd = '1043p000'
        # #imfn = os.path.join('data/unwise/unwise', coadd[:3], coadd, 'unwise-%s-w1-img-m.fits' % coadd)
        # #wcs = Tan(imfn)
        # #rd = np.array([wcs.pixelxy2radec(x,y) for x,y in [(1,1),(1,2048),(2048,2048),(2048,1),(1,1)]])
        # #print 'rd', rd.shape
        # for rd in rds:
        #     plt.plot(rd[:,0], rd[:,1], 'r-')
        # #ax = plt.axis()
        # ax = [103.5, 105.2, -0.9, 0.8]
        # plt.axis(ax)
        # plt.savefig(fn.replace('.png','-2.png'))
        # print 'Wrote', fn
    
        # plt.clf()
        # P = fits_table('data/sequels-phot-5/phot-1043p000.fits')
        # plt.plot(P.ra, P.dec, 'k.', alpha=0.1)
        # plt.plot(rd[:,0], rd[:,1], 'r-')
        # plt.axis(ax)
        # plt.savefig(fn.replace('.png','-3.png'))
    
    T.cut(I)
    W.cut(I)

    W.add_columns_from(T, dup='sdss_')
    W.writeto(mfn)
    print 'Wrote', mfn


