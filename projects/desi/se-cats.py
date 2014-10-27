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

    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.95,
                        hspace=0.05, wspace=0.05)

    D = Decals()
    B = D.get_bricks()
    ii = 377305
    #ii = 380155
    brick = B[ii]
    #targetwcs = wcs_for_brick(brick, W=400,H=400)
    targetwcs = wcs_for_brick(brick)
    T = D.get_ccds()
    T.cut(ccds_touching_wcs(targetwcs, T))
    print len(T), 'CCDs touching brick'

    MM = []
    SS = []
    #SDSS = []
    for t in T:
        #if t.filter != 'r':
        #    continue
        im = DecamImage(t)

        run_calibs(im, t.ra, t.dec, 0.262, morph=False)

        if not os.path.exists(im.morphfn):
            continue

        wcs = im.read_wcs()

        M = fits_table(im.morphfn, hdu=2)
        print len(M), 'morphs', im.morphfn
        M.filter = np.array([t.filter] * len(M))
        MM.append(M)

        #S = fits_table(im.sefn, hdu=2)
        S = fits_table(im.se2fn, hdu=2)
        print len(S), 'se2', im.se2fn
        S.ra,S.dec = wcs.pixelxy2radec(S.x_image, S.y_image)
        zp = D.get_zeropoint_for(im)
        print 'zp', zp

        flux = S.flux_auto
        mag = -2.5 * np.log10(flux)
        print 'Median mag', np.median(mag)
        #S.set('mag_auto_%s' % im.band, mag + zp)
        S.mag_auto = mag + zp

        for ap in range(5):
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

    Sr = Sfilt['r']
    Sg = Sfilt['g']
    Sz = Sfilt['z']
    I,J,d = match_radec(Sr.ra, Sr.dec, Sg.ra, Sg.dec, 0.5/3600.)
    Sgrz = Sr[I]
    for col in Sgrz.get_columns():
        if not col in ['ra','dec']:
            Sgrz.rename(col, col+'_r')
    Sg.cut(J)
    for col in Sg.get_columns():
        Sg.rename(col, col + '_g')
    Sgrz.add_columns_from(Sg)
    print len(Sgrz), 'matched gr'
    I,J,d = match_radec(Sgrz.ra, Sgrz.dec, Sz.ra, Sz.dec, 0.5/3600.)
    Sgrz.cut(I)
    Sz.cut(J)
    for col in Sz.get_columns():
        Sz.rename(col, col + '_z')
    Sgrz.add_columns_from(Sz)
    print len(Sgrz), 'matched grz'

    del Sg
    del Sr
    del Sz

    for col in ['mag_auto', 'mag_ap0', 'mag_ap1', 'mag_ap2', 'mag_ap3', 'mag_ap4']:
        g = Sgrz.get(col + '_g')
        r = Sgrz.get(col + '_r')
        z = Sgrz.get(col + '_z')

        plt.clf()
        plt.plot(g - r, r - z, 'bo', mec='none', mfc='b', alpha=0.5, ms=4)
        plt.xlabel('g - r')
        plt.ylabel('r - z')
        plt.title(col)
        ps.savefig()

        loghist(g - r, r - z, 100, range=((-0.5, 2.5),(-1,3)))
        plt.xlabel('g - r')
        plt.ylabel('r - z')
        plt.title(col)
        ps.savefig()

        X = Sgrz.flux_auto_r / Sgrz.fluxerr_auto_r
        I = np.flatnonzero(np.logical_and(np.isfinite(X), X > 20))
        loghist(g[I] - r[I], r[I] - z[I], 100, range=((-0.5, 2.5),(-1,3)))
        plt.xlabel('g - r')
        plt.ylabel('r - z')
        plt.title(col + ' SNR(r) > 20')
        ps.savefig()


    rgb = fitsio.read('rgb.fits')
    print 'rgb', rgb.shape
    h,w,three = rgb.shape

    def plot_rgbs(I):
        nplot = 0
        plt.clf()
        rows,cols = 6,8
        for i in I:
            ok,x,y = targetwcs.radec2pixelxy(Sgrz.ra[i], Sgrz.dec[i])
            x -= 1
            y -= 1
            x = int(x)
            y = int(y)
            sz = 25
            if x < sz or y < sz or x+sz > w or y+sz > h:
                continue
            plt.subplot(rows,cols, 1+nplot)
            dimshow(rgb[y-sz:y+sz+1, x-sz:x+sz+1, :])
            plt.xticks([]); plt.yticks([])
            nplot += 1
            if nplot >= rows*cols:
                break

    g = Sgrz.mag_ap4_g
    r = Sgrz.mag_ap4_r
    z = Sgrz.mag_ap4_z
    I = np.flatnonzero(((r-z) > (0.5 + ((g-r) - 0.5) * (1.0-0.5)/(1.25-0.5))) * ((g-r) < 1.25))
    print len(I), 'away from stellar locus'

    loghist(g[I] - r[I], r[I] - z[I], 100, range=((-0.5, 2.5),(-1,3)))
    plt.xlabel('g - r')
    plt.ylabel('r - z')
    plt.title('Away from stellar locus?')
    ps.savefig()

    plot_rgbs(I)
    plt.suptitle('Away from stellar locus')
    ps.savefig()

    I = np.flatnonzero(Sgrz.class_star_r < 0.1)
    loghist(g[I] - r[I], r[I] - z[I], 100, range=((-0.5, 2.5),(-1,3)))
    plt.xlabel('g - r')
    plt.ylabel('r - z')
    plt.title('class_star < 0.1')
    ps.savefig()

    plot_rgbs(I)
    plt.suptitle('class_star < 0.1')
    ps.savefig()

    I = np.flatnonzero((Sgrz.class_star_r < 0.5) * (Sgrz.class_star_r > 0.1))
    loghist(g[I] - r[I], r[I] - z[I], 100, range=((-0.5, 2.5),(-1,3)))
    plt.xlabel('g - r')
    plt.ylabel('r - z')
    plt.title('0.1 < class_star < 0.5')
    ps.savefig()

    plot_rgbs(I)
    plt.suptitle('0.1 < class_star < 0.5')
    ps.savefig()

    I = np.flatnonzero((Sgrz.class_star_r > 0.5) * (Sgrz.class_star_r < 0.9))
    loghist(g[I] - r[I], r[I] - z[I], 100, range=((-0.5, 2.5),(-1,3)))
    plt.xlabel('g - r')
    plt.ylabel('r - z')
    plt.title('0.5 < class_star < 0.9')
    ps.savefig()

    plot_rgbs(I)
    plt.suptitle('0.5 < class_star < 0.9')
    ps.savefig()

    I = np.flatnonzero(Sgrz.class_star_r > 0.9)
    loghist(g[I] - r[I], r[I] - z[I], 100, range=((-0.5, 2.5),(-1,3)))
    plt.xlabel('g - r')
    plt.ylabel('r - z')
    plt.title('class_star > 0.9')
    ps.savefig()

    plot_rgbs(I)
    plt.suptitle('class_star > 0.9')
    ps.savefig()


    I = np.flatnonzero(Sgrz.class_star_r < 0.1)

    plt.clf()
    plt.loglog(S.fluxerr_auto, S.flux_auto, 'b.')
    plt.xlabel('fluxerr_auto')
    plt.ylabel('flux_auto')
    plt.title('SE2')
    ps.savefig()

    plt.clf()
    X = S.flux_auto / S.fluxerr_auto
    X = X[np.isfinite(X)]
    mn,mx = [np.percentile(X, p) for p in [1,99]]
    print 'min,max SNR', mn,mx
    mx = 50.
    plt.hist(X, 50, range=(mn,mx))
    plt.xlabel('flux_auto SNR')
    plt.title('SE2')
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

