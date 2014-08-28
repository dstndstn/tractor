import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *

from common import *

if __name__ == '__main__':

    ps = PlotSequence('se')

    D = Decals()
    B = D.get_bricks()
    brick = B[377305]
    #wcs = wcs_for_brick(brick, W=400,H=400)
    wcs = wcs_for_brick(brick)
    T = D.get_ccds()
    T.cut(ccds_touching_wcs(wcs, T))
    print len(T), 'CCDs touching brick'

    MM = []
    SS = []
    #SDSS = []
    for t in T:
        #if t.filter != 'r':
        #    continue
        im = DecamImage(t)

        #run_calibs(im, t.ra, t.dec, 0.262/3600., morph=False)
        if not os.path.exists(im.morphfn):
            continue

        wcs = im.read_wcs()

        M = fits_table(im.morphfn, hdu=2)
        print len(M), 'morphs', im.morphfn
        M.filter = np.array([t.filter] * len(M))
        MM.append(M)

        S = fits_table(im.sexfn, hdu=2)
        print len(S), 'se', im.sexfn
        S.ra,S.dec = wcs.pixelxy2radec(S.x_image, S.y_image)
        zp = D.get_zeropoint_for(im)
        print 'zp', zp

        flux = S.flux_auto
        mag = -2.5 * np.log10(flux)
        print 'Median mag', np.median(mag)
        #S.set('mag_auto_%s' % im.band, mag + zp)
        S.mag_auto = mag + zp

        for ap in range(3):
            mag = -2.5 * np.log10(S.flux_aper[:,ap])
            print 'Median mag', np.median(mag)
            #S.set('mag_ap%i_%s' % (ap,im.band), mag + zp)
            S.set('mag_ap%i' % (ap), mag + zp)

        #sdss = fits_table(im.sdssfn)
        #print len(sdss), 'SDSS', im.sdssfn
        #flux = sdss.get('%s_psfflux' % im.band)
        #I,J,d = match_radec(S.ra, S.dec, sdss.ra, sdss.dec, 0.5/3600.)
        #SDSS.append(sdss)
        
        S.filter = np.array([t.filter] * len(S))
        SS.append(S)
    S = merge_tables(SS, columns='fillzero')
    M = merge_tables(MM)

    # Merge sources within each filter
    Sfilt = {}
    for band in 'grz':
        SF = S[S.filter == band]
        print len(SF), 'in', band
        I,J,d = match_radec(SF.ra, SF.dec, SF.ra, SF.dec, 0.5/3600., notself=True)
        keep = np.ones(len(SF), bool)
        keep[np.minimum(I,J)] = False
        SF.cut(keep)
        print 'Cut to', len(SF), 'un-matched'
        Sfilt[band] = SF

    I,J,d = match_radec(Sfilt['r'].ra, Sfilt['r'].dec, Sfilt['g'].ra, Sfilt['g'].dec,
                        0.5/3600.)
    SS = Sfilt['r'][I]
    for col in SS.get_columns():
        if not col in ['ra','dec']:
            SS.rename(col, col+'_r')
    SG = Sfilt['g'][J]
    for col in SG.get_columns():
        SG.rename(col, col + '_g')
    SS.add_columns_from(SG)
    print len(SS), 'matched gr'
    I,J,d = match_radec(SS.ra, SS.dec, Sfilt['z'].ra, Sfilt['z'].dec, 0.5/3600.)
    SS.cut(I)
    SZ = Sfilt['z'][J]
    for col in SZ.get_columns():
        SZ.rename(col, col + '_z')
    SS.add_columns_from(SZ)
    #SS.add_columns_from(Sfilt['z'][J])
    print len(SS), 'matched grz'

    for col in ['mag_auto', 'mag_ap0', 'mag_ap1', 'mag_ap2']:
        g = SS.get(col + '_g')
        r = SS.get(col + '_r')
        z = SS.get(col + '_z')

        plt.clf()
        plt.plot(g - r, r - z, 'b.')
        plt.xlabel('g - r')
        plt.ylabel('r - z')
        plt.title(col)
        ps.savefig()

    plt.clf()
    plt.loglog(M.chi2_psf, M.chi2_model, 'b.')
    plt.xlabel('chi2_psf')
    plt.ylabel('chi2_model')
    ps.savefig()

    plt.clf()
    plt.semilogy(M.class_star, M.chi2_psf / M.chi2_model, 'b.')
    plt.xlabel('class_star')
    plt.ylabel('chi2_psf / chi2_model')
    ps.savefig()

    for col in ['x2_world', 'y2_world', 'xy_world', 'a_world', 'b_world', 'theta_world',
                'class_star']:
        plt.clf()
        X = S.get(col)
        mn,mx = [np.percentile(X, p) for p in [1,99]]
        plt.hist(X, 100, range=(mn,mx))
        plt.xlabel(col)
        ps.savefig()

    plt.clf()
    plt.semilogy(S.class_star, S.a_world, 'b.', alpha=0.5)
    plt.xlabel('class_star')
    plt.ylabel('a_world')
    ps.savefig()

