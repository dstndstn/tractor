from __future__ import print_function
import sys
import os
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from glob import glob

from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import * #PlotSequence, dimshow
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import match_radec
from tractor import *
from tractor.galaxy import *
from common import *

def plot_objects(objs, allobjs, img, brickwcs, S=25, rows=4, cols=6,
                 imshowargs={}, fracflux=False, labels=None):
    objs.x0 = np.maximum(0, np.round(objs.bx - S).astype(int))
    objs.y0 = np.maximum(0, np.round(objs.by - S).astype(int))

    for i,o in enumerate(objs):
        if i >= rows*cols:
            break
        plt.subplot(rows, cols, i+1)
        H,W,three = img.shape
        dimshow(img[o.y0:min(H, o.by+S),
                    o.x0:min(W, o.bx+S), :], ticks=False)

        if fracflux:
            plt.text(0, 0, '%i/%i/%i' % 
                     (int(100. * o.decam_fracflux[1]),
                      int(100. * o.decam_fracflux[2]),
                      int(100. * o.decam_fracflux[4])),
                      color='red', size='small',
                      ha='left', va='bottom')

            #plt.text(0, 2*S, '%.3f, %.3f' % (o.ra, o.dec),
            #         color='red', size='small', ha='left', va='top')

        if labels is not None:
            txt = labels[i]
            if txt is not None:
                plt.text(0, 0, txt, color='red', size='small',
                         ha='left', va='bottom')
        
    if allobjs is None:
        return

    for i,o in enumerate(objs):
        if i >= rows*cols:
            break
        plt.subplot(rows, cols, i+1)

        ax = plt.axis()
        # plot sources nearby...
        near = allobjs[(np.abs(allobjs.bx - o.bx) < S) *
                       (np.abs(allobjs.by - o.by) < S)]

        # print len(near), 'nearby sources'
        # print '  type', near.type
        x0 = np.maximum(0, np.round(o.bx - S).astype(int))
        y0 = np.maximum(0, np.round(o.by - S).astype(int))
        
        for n in near:
            e = None
            cc = None
            typ = n.type.strip()
            green = (0.2,1,0.2)
            if typ == 'PSF':
                plt.plot(n.bx-x0, n.by-y0, 'o', mfc=green, mec='k', alpha=0.6)
            elif typ == 'EXP':
                e = EllipseE(n.shapeexp_r, n.shapeexp_e1, n.shapeexp_e2)
                cc = '0.8'
            elif typ == 'DEV':
                e = EllipseE(n.shapedev_r, n.shapedev_e1, n.shapedev_e2)
                cc = green
            elif typ == 'COMP':
                plt.plot(n.bx-x0, n.by-y0, 'yo', alpha=0.6)

            if e is not None:
                G = e.getRaDecBasis()
                angle = np.linspace(0, 2.*np.pi, 60)
                xy = np.vstack((np.append([0,0,1], np.sin(angle)),
                                np.append([0,1,0], np.cos(angle)))).T
                rd = np.dot(G, xy.T).T
                ra  = n.ra  + rd[:,0] * np.cos(np.deg2rad(n.dec))
                dec = n.dec + rd[:,1]
                ok,xx,yy = brickwcs.radec2pixelxy(ra, dec)
                x1,x2,x3 = xx[:3]
                y1,y2,y3 = yy[:3]
                plt.plot([x3 - x0, x1 - x0, x2 - x0],
                         [y3 - y0, y1 - y0, y2 - y0], '-', color=cc)
                plt.plot(x1 - x0, y1 - y0, '.', color=cc, ms=3, alpha=0.6)
                xx = xx[3:]
                yy = yy[3:]
                plt.plot(xx - x0, yy - y0, '-', color=cc)
        plt.axis(ax)


def forced_phot():
    # Forced phot
    ps = PlotSequence('kick')
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)

    ff = [
        ('decam-348227-S15-g-forced.fits', 348227, 'S15', 'g'),
        ('decam-348248-S16-g-forced.fits', 348248, 'S16', 'g'),
        ('decam-348271-S14-g-forced.fits', 348271, 'S14', 'g'),
        ('decam-348660-S15-r-forced.fits', 348660, 'S15', 'r'),
        ('decam-348684-S16-r-forced.fits', 348684, 'S16', 'r'),
        ('decam-348712-S14-r-forced.fits', 348712, 'S14', 'r'),
        ('decam-349154-S15-z-forced.fits', 349154, 'S15', 'z'),
        ('decam-349183-S14-z-forced.fits', 349183, 'S14', 'z'),
        ('decam-346630-S16-z-forced.fits', 346630, 'S16', 'z'),
    ]
    FF = []
    for fn,expnum,extname,band in ff:
        F = fits_table(fn)
        print(len(F), 'from', fn)
        F.expnum  = np.array([expnum ] * len(F))
        F.extname = np.array([extname] * len(F))
        FF.append(F)
    F = merge_tables(FF)
    bricks = np.unique(F.brickname)
    print('Bricks:', bricks)
    T = merge_tables([fits_table(os.path.join('dr1','tractor',b[:3],
                                              'tractor-%s.fits' % b))
                                              for b in bricks])
    print('Total of', len(T), 'sources')
    T.cut(T.brick_primary == 1)
    print('Cut to', len(T), 'brick_primary')

    Tall = T.copy()

    cutouts = []

    I,J,d = match_radec(Tall.ra, Tall.dec, Tall.ra, Tall.dec, 5./3600.,
                        notself=True)
    K = np.flatnonzero(I < J)
    I = I[K]
    J = J[K]
    # randomize
    K = np.random.randint(2, size=len(I))
    I,J = I*K + J*(1-K), I*(1-K) + J*K

    plt.clf()
    plothist(3600. *(Tall.ra[I] - Tall.ra[J]) * np.cos(np.deg2rad(Tall.dec[I])),
            3600. * (Tall.dec[I] - Tall.dec[J]), 200) #, range=((-2,2),(-2,2)))
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.title('%i sources, %i matches' % (len(Tall), len(I)))
    ps.savefig()

    dist = 3600. * np.hypot((Tall.ra[I] - Tall.ra[J]) *
                            np.cos(np.deg2rad(Tall.dec[I])),
                            Tall.dec[I] - Tall.dec[J])

    cutouts.append((Tall, I[dist < 0.5], 'Close pairs', None))
    
    plt.clf()
    plt.hist(dist, 50, histtype='step', color='b')
    plt.xlabel('Match distance (arcsec)')
    plt.title('%i sources, %i matches' % (len(Tall), len(I)))
    ps.savefig()

    plt.clf()
    plothist(Tall.bx0[I] - Tall.bx0[J], Tall.by0[I] - Tall.by0[J],
             200, range=((-20,20),(-20,20)))
    plt.xlabel('d(x0)')
    plt.ylabel('d(y0)')
    plt.title('%i sources, %i matches' % (len(Tall), len(I)))
    ps.savefig()

    # plt.clf()
    # plt.plot(np.vstack((Tall.bx0[I], Tall.bx0[J])),
    #          np.vstack((Tall.by0[I], Tall.by0[J])), 'b-')
    # ps.savefig()
    
    T.srcid = (T.brickid.astype(np.int64) << 32 | T.objid)

    F.srcid = (F.brickid.astype(np.int64) << 32 | F.objid)
    print('Total of', len(F), 'forced')
    print(len(np.unique(F.srcid)), 'unique source in forced')

    tmap = dict([(s,i) for i,s in enumerate(T.srcid)])

    T.forced_g = [ [] for i in range(len(T)) ]
    T.forced_r = [ [] for i in range(len(T)) ]
    T.forced_z = [ [] for i in range(len(T)) ]

    for f in F:
        i = tmap[f.srcid]
        forced = T.get('forced_%s' % f.filter)[i]
        forced.append(f.flux)

    allbands = 'ugrizY'
        
    # pack into arrays
    bands = 'grz'
    for band in bands:
        flist = T.get('forced_%s' % band)
        nmax = max([len(f) for f in flist])
        arr = np.zeros((len(T), nmax), np.float32)
        for i,f in enumerate(flist):
            arr[i,:len(f)] = f
        T.set('forced_%s' % band, arr)

        ib = allbands.index(band)
        flux = T.decam_flux[:,ib]
        arr = np.zeros(len(T), np.float32)
        for i,f in enumerate(flist):
            arr[i] = np.mean([(fi - flux[i])**2 for fi in f])
        arr = np.sqrt(arr)
        T.set('forced_rms_%s' % band, arr)

        T.set('forced_n_%s' % band, np.array([len(f) for f in flist]))
        
        
    # Cut to sources with forced phot
    T.cut(np.flatnonzero(reduce(np.logical_or, [
        np.any(T.get('forced_%s'%band) > 0, axis=1) for band in bands])))
    print('Cut to', len(T), 'sources with forced phot')

    flux = T.decam_flux[:,1]
    forced = T.forced_g
    rms = T.forced_rms_g
    N = T.forced_n_g
    I = np.flatnonzero((N > 1) * (flux > 1e3))
    print(len(I), 'with flux > 10^3')
    #print 'flux', flux[I]
    #print 'forced'
    for f,ff,r,n in zip(flux[I], forced[I,:], rms[I], N[I]):
        print('flux', f, 'n', n, 'forced RMS', r, 'forced', ff)

    # Compute some summary stats
    # allbands = 'ugrizY'
    # for b in bands:
    #     ib = allbands.index(b)
    #     forced = T.get('forced_%s' % b)
    #     flux = T.decam_flux[:,ib]
    #     T.set('forced_rms_%s' % b, np.sqrt([np.mean([(fi - fl)**2 for fi in flist if fi != 0])
    #                                         for flist,fl in zip(forced, flux)]))
    
    imgs = {}
    
    allbands = 'ugrizY'
    for b in bands:
        ib = allbands.index(b)
        plt.clf()
        f = T.get('forced_%s' % b)
        n,nf = f.shape
        flux = T.decam_flux[:,ib]
        lo,hi = 1e-2, 1e5
        plt.plot([lo,hi], [lo,hi], 'r-')
        plt.errorbar(flux, flux, fmt='none',
                     yerr=1./np.sqrt(T.decam_flux_ivar[:,ib]), ecolor='r')
        for i in range(nf):
            plt.plot(T.decam_flux[:,ib], f[:,i], 'b.', alpha=0.25)
        plt.axis([lo,hi,lo,hi])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Combined flux')
        plt.ylabel('Forced-photometry flux')
        plt.title('Forced phot: %s band' % b)
        ps.savefig()

    for b in bands:
        ib = allbands.index(b)
        plt.clf()
        f = T.get('forced_%s' % b)
        n,nf = f.shape
        flux = T.decam_flux[:,ib]
        lo,hi = -0.5, 4
        plt.axhline(1., color='r')
        plt.axhline(0., color='k', alpha=0.25)
        for i in range(nf):
            plt.plot(flux, np.clip(f[:,i] / flux, lo, hi), 'b.', alpha=0.25)
        plt.ylim(lo-0.1, hi+0.1)
        plt.xlim(1e-2, 1e5)
        plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Combined flux')
        plt.ylabel('Relative forced-photometry flux')
        plt.title('Forced phot: %s band' % b)
        ps.savefig()

        # Large relative flux
        I = np.flatnonzero((flux > 0.) * ((f[:,i] / flux) > 4.))
        #print 'Relative fluxes:', (f[:,i]/flux)[I]
        #print 'Fluxes:', flux[I]
        I = I[np.argsort(-flux[I])]
        #print 'Sorted fluxes:', flux[I]
        labels = []
        for t in T[I]:
            fluxes = t.get('forced_%s' % b)
            ib = allbands.index(b)
            flux = t.decam_flux[ib]
            txt = ('%.1f / %.1f / %.1f | %.1f' % 
                   (fluxes[0], fluxes[1], fluxes[2], flux))
            labels.append(txt)

        cutouts.append((T, I, 'Large relative flux: %s band' % b, labels))



    for b in bands:
        ib = allbands.index(b)
        plt.clf()
        rms = T.get('forced_rms_%s' % b)
        N = T.get('forced_n_%s' % b)
        flux = T.decam_flux[:,ib]
        lo,hi = -0.5, 4
        #plt.axhline(1., color='r')
        #plt.axhline(0., color='k', alpha=0.25)
        I = np.flatnonzero(N > 1)
        print(len(I), 'of', len(rms), 'have >1 exposure')
        plt.plot(flux[I], rms[I], 'b.', alpha=0.25)
        #plt.ylim(lo-0.1, hi+0.1)
        plt.xlim(1e-2, 1e5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Combined flux')
        plt.ylabel('Forced-photometry flux RMS')
        plt.title('Forced phot: %s band' % b)
        ps.savefig()

        # Large relative RMS
        I = np.flatnonzero((flux > 10.) * (N > 1) * ((rms / flux) > 0.1))
        I = I[np.argsort(-flux[I])]
        cutouts.append((T, I, 'Large relative RMS: %s band' % b, None))

        T[I].writeto('large-rms-%s.fits' % b)
        

    # Create a fake "brick" WCS bounding the forced-phot objects
    # rlo = T.ra.min()
    # rhi = T.ra.max()
    # dlo = T.dec.min()
    # dhi = T.dec.max()

    rlo = Tall.ra.min()
    rhi = Tall.ra.max()
    dlo = Tall.dec.min()
    dhi = Tall.dec.max()

    if rhi - rlo > 180:
        print('No RA wrap-around support')
        sys.exit(0)

    dec = (dlo + dhi) / 2.
    ra  = (rlo + rhi) / 2.
    pixscale = 0.262
    ddec = (dhi - dlo)
    H = ddec * 3600. / pixscale
    dra  = (rhi - rlo)
    W = dra  * np.cos(np.deg2rad(dec)) * 3600. / pixscale
    H = int(np.ceil(H))
    W = int(np.ceil(W))
    print('Target image shape', (H,W))
    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale/3600., 0., 0., pixscale/3600.,
                    float(W), float(H))
    img = np.zeros((H,W,3), np.uint8)
    print('img', img.shape)
    decals = Decals()
    for brickname in bricks:
        brick = decals.get_brick_by_name(brickname)
        brickwcs = wcs_for_brick(brick)

        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, brickwcs, [], 3)
        except:
            continue
        brickimg = plt.imread(os.path.join('dr1', 'coadd', brickname[:3],
                                           brickname, 'decals-%s-image.jpg' % brickname))
        print('brick image', brickimg.shape, brickimg.dtype)
        brickimg = brickimg[::-1,:,:]
        img[Yo,Xo,:] = brickimg[Yi,Xi,:]

    # Now fake the "bx,by" coords to refer to 'targetwcs' / 'img'
    ok,x,y = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.bx = x - 1
    T.by = y - 1
    ok,x,y = targetwcs.radec2pixelxy(Tall.ra, Tall.dec)
    Tall.bx = x - 1
    Tall.by = y - 1

    print('Tall:', len(Tall), 'sources')

    # plt.clf()
    # dimshow(img)
    # ps.savefig()
    # 
    # ax = plt.axis()
    # plt.plot(Tall.bx, Tall.by, 'r.')
    # plt.axis(ax)
    # ps.savefig()
    
    for TT,I,desc,labels in cutouts:
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                            hspace=0.05, wspace=0.05)

        kwa = dict(rows=6, cols=9)

        plt.clf()
        plot_objects(TT[I], None, img, targetwcs, **kwa)
        plt.suptitle(desc)
        ps.savefig()
        
        plt.clf()
        plot_objects(TT[I], Tall, img, targetwcs, labels=labels, **kwa)
        plt.suptitle(desc)
        ps.savefig()


    ## Initial positions
    # ok,x,y = targetwcs.radec2pixelxy(T.ra, T.dec)
    # T.bx = x - 1
    # T.by = y - 1
    # ok,x,y = targetwcs.radec2pixelxy(Tall.ra, Tall.dec)
    # Tall.bx = x - 1
    # Tall.by = y - 1

        

        
def galaxies():
    ps = PlotSequence('kick')
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95)
    brick = '3166p025'

    decals = Decals()
    b = decals.get_brick_by_name(brick)
    brickwcs = wcs_for_brick(b)
    
    # A catalog of sources overlapping one DECaLS CCD, arbitrarily:
    # python projects/desi/forced-photom-decam.py decam/CP20140810_g_v2/c4d_140816_032035_ooi_g_v2.fits.fz 1 DR1 f.fits
    #T = fits_table('cat.fits')

    T = fits_table(os.path.join('dr1', 'tractor', brick[:3],
                                'tractor-%s.fits' % brick))
    print(len(T), 'catalog sources')
    print(np.unique(T.brick_primary))
    T.cut(T.brick_primary)
    print(len(T), 'primary')

    print('Out of bounds:', np.unique(T.out_of_bounds))
    print('Left blob:', np.unique(T.left_blob))
    
    img = plt.imread(os.path.join('dr1', 'coadd', brick[:3], brick,
                                  'decals-%s-image.jpg' % brick))
    img = img[::-1,:,:]
    print('Image:', img.shape)

    if False:
        resid = plt.imread(os.path.join('dr1', 'coadd', brick[:3], brick,
                                      'decals-%s-resid.jpg' % brick))
        resid = resid[::-1,:,:]

    
    T.shapeexp_e1_err = 1./np.sqrt(T.shapeexp_e1_ivar)
    T.shapeexp_e2_err = 1./np.sqrt(T.shapeexp_e2_ivar)
    T.shapeexp_r_err  = 1./np.sqrt(T.shapeexp_r_ivar)
    T.shapedev_e1_err = 1./np.sqrt(T.shapedev_e1_ivar)
    T.shapedev_e2_err = 1./np.sqrt(T.shapedev_e2_ivar)
    T.shapedev_r_err  = 1./np.sqrt(T.shapedev_r_ivar)

    T.gflux = T.decam_flux[:,1]
    T.rflux = T.decam_flux[:,2]
    T.zflux = T.decam_flux[:,4]
    
    I = np.flatnonzero(T.type == 'EXP ')
    J = np.flatnonzero(T.type == 'DEV ')

    E = T[I]
    D = T[J]

    cutobjs = []

    cut = np.logical_or(E.shapeexp_e1_err > 1., E.shapeexp_e2_err > 1.)
    I = np.flatnonzero(cut)
    print(len(I), 'EXP with large ellipticity error')
    cutobjs.append((E[I], 'EXP ellipticity error > 1'))

    E.cut(np.logical_not(cut))
    
    I = np.flatnonzero(np.logical_or(D.shapedev_e1_err > 1., D.shapedev_e2_err > 1.))
    print(len(I), 'DEV with large ellipticity error')
    cutobjs.append((D[I], 'DEV ellipticity error > 1'))

    I = np.flatnonzero(np.logical_or(np.abs(E.shapeexp_e1) > 0.5,
                                     np.abs(E.shapeexp_e2) > 0.5))
    cutobjs.append((E[I], 'EXP with ellipticity > 0.5'))

    I = np.flatnonzero(np.logical_or(np.abs(D.shapedev_e1) > 0.5,
                                     np.abs(D.shapedev_e2) > 0.5))
    cutobjs.append((E[I], 'DEV with ellipticity > 0.5'))

    I = np.flatnonzero(np.logical_or(E.shapeexp_e1_err < 3e-3,
                                     E.shapeexp_e2_err < 3e-3))
    cutobjs.append((E[I], 'EXP with small ellipticity errors (<3e-3)'))

    I = np.flatnonzero(np.logical_or(D.shapedev_e1_err < 3e-3,
                                     D.shapedev_e2_err < 3e-3))
    cutobjs.append((D[I], 'DEV with small ellipticity errors (<3e-3)'))

    I = np.flatnonzero(D.shapedev_r > 10.)
    cutobjs.append((D[I], 'DEV with large radius (>10")'))

    I = np.flatnonzero(D.shapedev_r_err < 2e-3)
    cutobjs.append((D[I], 'DEV with small radius errors (<2e-3)'))

    I = np.flatnonzero((D.rflux > 100.) * (D.shapedev_r < 5.))
    cutobjs.append((D[I], 'DEV, small & bright'))

    I = np.flatnonzero((E.rflux > 100.) * (E.shapeexp_r < 5.))
    cutobjs.append((E[I], 'EXP, small & bright'))
    
    # I = np.argsort(-T.decam_flux[:,2])
    # cutobjs.append((T[I], 'brightest objects'))

    I = np.flatnonzero(np.logical_or(D.rflux < -5., D.gflux < -5))
    cutobjs.append((D[I], 'DEV with neg g or r flux'))

    I = np.flatnonzero(np.logical_or(E.rflux < -5., E.gflux < -5))
    cutobjs.append((E[I], 'EXP with neg g or r flux'))
    
    I = np.flatnonzero(T.decam_rchi2[:,2] > 5.)
    cutobjs.append((T[I], 'rchi2 > 5'))

    
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95,
                        hspace=0.05, wspace=0.05)

        
    # plt.clf()
    # p1 = plt.semilogy(T.shapeexp_e1[I], T.shapeexp_e1_ivar[I], 'b.')
    # p2 = plt.semilogy(T.shapeexp_e2[I], T.shapeexp_e2_ivar[I], 'r.')
    # plt.xlabel('Ellipticity e')
    # plt.ylabel('Ellipticity inverse-variance e_ivar')
    # plt.title('EXP galaxies')
    # plt.legend([p1[0],p2[0]], ['e1','e2'])
    # ps.savefig()

    plt.clf()
    p1 = plt.semilogy(E.shapeexp_e1, E.shapeexp_e1_err, 'b.')
    p2 = plt.semilogy(E.shapeexp_e2, E.shapeexp_e2_err, 'r.')
    plt.xlabel('Ellipticity e')
    plt.ylabel('Ellipticity error e_err')
    plt.title('EXP galaxies')
    plt.legend([p1[0],p2[0]], ['e1','e2'])
    ps.savefig()
    
    # plt.clf()
    # p1 = plt.semilogy(T.shapedev_e1[J], T.shapedev_e1_ivar[J], 'b.')
    # p2 = plt.semilogy(T.shapedev_e2[J], T.shapedev_e2_ivar[J], 'r.')
    # plt.xlabel('Ellipticity e')
    # plt.ylabel('Ellipticity inverse-variance e_ivar')
    # plt.title('DEV galaxies')
    # plt.legend([p1[0],p2[0]], ['e1','e2'])
    # ps.savefig()

    plt.clf()
    p1 = plt.semilogy(D.shapedev_e1, D.shapedev_e1_err, 'b.')
    p2 = plt.semilogy(D.shapedev_e2, D.shapedev_e2_err, 'r.')
    plt.xlabel('Ellipticity e')
    plt.ylabel('Ellipticity error e_err')
    plt.title('DEV galaxies')
    plt.legend([p1[0],p2[0]], ['e1','e2'])
    ps.savefig()


    plt.clf()
    p1 = plt.loglog(D.shapedev_r, D.shapedev_r_err, 'b.')
    p2 = plt.loglog(E.shapeexp_r, E.shapeexp_r_err, 'r.')
    plt.xlabel('Radius r')
    plt.ylabel('Radius error r_err')
    plt.title('DEV, EXP galaxies')
    plt.legend([p1[0],p2[0]], ['deV','exp'])
    ps.savefig()



    plt.clf()
    p1 = plt.loglog(D.rflux, D.shapedev_r, 'b.')
    p2 = plt.loglog(E.rflux, E.shapeexp_r, 'r.')
    plt.xlabel('r-band flux')
    plt.ylabel('Radius r')
    plt.title('DEV, EXP galaxies')
    plt.legend([p1[0],p2[0]], ['deV','exp'])
    ps.savefig()



    plt.clf()
    p1 = plt.loglog(-D.rflux, D.shapedev_r, 'b.')
    p2 = plt.loglog(-E.rflux, E.shapeexp_r, 'r.')
    plt.xlabel('Negative r-band flux')
    plt.ylabel('Radius r')
    plt.title('DEV, EXP galaxies')
    plt.legend([p1[0],p2[0]], ['deV','exp'])
    ps.savefig()

    plt.clf()
    plt.loglog(D.rflux, D.decam_rchi2[:,2], 'b.')
    plt.loglog(E.rflux, E.decam_rchi2[:,2], 'r.')
    plt.xlabel('r-band flux')
    plt.ylabel('rchi2 in r')
    plt.title('DEV, EXP galaxies')
    plt.legend([p1[0],p2[0]], ['deV','exp'])
    ps.savefig()



    for objs,desc in cutobjs:
        if len(objs) == 0:
            print('No objects in cut', desc)
            continue

        rows,cols = 4,6
        objs = objs[:rows*cols]

        if False:
            plt.clf()
            dimshow(img)
            plt.plot(objs.bx, objs.by, 'rx')
            ps.savefig()

        plt.clf()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                            hspace=0.05, wspace=0.05)

        plot_objects(objs, None, img, brickwcs)
        plt.suptitle(desc)
        ps.savefig()

        plt.clf()
        plot_objects(objs, T, img, brickwcs, fracflux=True)
        plt.suptitle(desc)
        ps.savefig()
        
        if False:
            plt.clf()
            for i,o in enumerate(objs):
                plt.subplot(rows, cols, i+1)
                H,W,three = img.shape
                dimshow(resid[o.y0:min(H, o.by+S),
                              o.x0:min(W, o.bx+S), :], ticks=False)
            plt.suptitle(desc)
            ps.savefig()

    
    

    sys.exit(0)

    
    print('RA', T.ra.min(), T.ra.max())
    print('Dec', T.dec.min(), T.dec.max())

    # Uhh, how does *this* happen?!  Fitting gone wild I guess
    # T.cut((T.ra > 0) * (T.ra < 360) * (T.dec > -90) * (T.dec < 90))
    # print 'RA', T.ra.min(), T.ra.max()
    # print 'Dec', T.dec.min(), T.dec.max()
    # rlo,rhi = [np.percentile(T.ra,  p) for p in [1,99]]
    # dlo,dhi = [np.percentile(T.dec, p) for p in [1,99]]
    # print 'RA', rlo,rhi
    # print 'Dec', dlo,dhi
    # plt.clf()
    # plothist(T.ra, T.dec, 100, range=((rlo,rhi),(dlo,dhi)))
    # plt.xlabel('RA')
    # plt.ylabel('Dec')
    # ps.savefig()
    
    # decals = Decals()
    # B = decals.get_bricks()
    # #B.about()
    # brick = B[B.brickid == brickid]
    # assert(len(brick) == 1)
    # brick = brick[0]
    # wcs = wcs_for_brick(brick)
    # ccds = decals.get_ccds()
    # ccds.cut(ccds_touching_wcs(wcs, ccds))
    # print len(ccds), 'CCDs'
    # #ccds.about()
    # ccds.cut(ccds.filter == 'r')
    # print len(ccds), 'CCDs'
    # S = []
    # for ccd in ccds:
    #     im = DecamImage(ccd)
    #     S.append(fits_table(im.sdssfn))
    # S = merge_tables(S)
    # print len(S), 'total SDSS'
    # #nil,I = np.unique(S.ra, return_index=True)
    # nil,I = np.unique(['%.5f %.5f' % (r,d) for r,d in zip(S.ra,S.dec)], return_index=True)
    # S.cut(I)
    # print len(S), 'unique'
    # 
    # I,J,d = match_radec(T.ra, T.dec, S.ra, S.dec, 1./3600.)
    # print len(I), 'matches'
    # 
    # plt.clf()
    # plt.loglog(S.r_psfflux[J], T.decam_r_nanomaggies[I], 'r.')
    # ps.savefig()

    #plt.clf()
    #plt.loglog(T.sdss_modelflux[:,2], T.decam_r_nanomaggies, 'r.')
    #ps.savefig()

    
    for bindex,band in [(1,'g'), (2,'r'), (4,'z')]:
        sflux = T.sdss_modelflux[:,bindex]
        dflux = T.get('decam_%s_nanomaggies' % band)
        I = np.flatnonzero(sflux > 10.)
        med = np.median(dflux[I] / sflux[I])
        # plt.clf()
        # plt.loglog(sflux, dflux / sflux, 'ro', mec='r', ms=4, alpha=0.1)
        # plt.axhline(med, color='k')
        # plt.ylim(0.5, 2.)
        # ps.savefig()

        corr = dflux / med
        T.set('decam_%s_nanomaggies_corr' % band, corr)
        T.set('decam_%s_mag_corr' % band, NanoMaggies.nanomaggiesToMag(corr))

        dflux = T.get('decam_%s_nanomaggies_corr' % band)
        plt.clf()
        #plt.loglog(sflux, dflux / sflux, 'o', mec='b', ms=4, alpha=0.1)
        plt.loglog(sflux, dflux / sflux, 'b.', alpha=0.01)
        plt.xlim(1e-1, 3e3)
        plt.axhline(1., color='k')
        plt.ylim(0.5, 2.)
        plt.xlabel('SDSS flux (nmgy)')
        plt.ylabel('DECam flux / SDSS flux')
        plt.title('%s band' % band)
        ps.savefig()
        
    bands = 'grz'

    # for band in bands:
    #     plt.clf()
    #     sn = T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
    #     mag = T.get('decam_%s_mag_corr' % band)
    #     plt.semilogy(mag, sn, 'b.')
    #     plt.axis([20, 26, 1, 100])
    #     ps.savefig()

    ccmap = dict(g='g', r='r', z='m')
    plt.clf()
    for band in bands:
        sn = T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
        mag = T.get('decam_%s_mag_corr' % band)
        cc = ccmap[band]
        #plt.semilogy(mag, sn, '.', color=cc, alpha=0.2)
        plt.semilogy(mag, sn, '.', color=cc, alpha=0.01, mec='none')
    plt.xlabel('mag')
    plt.ylabel('Flux Signal-to-Noise')
    tt = [1,2,3,4,5,10,20,30,40,50]
    plt.yticks(tt, ['%i' % t for t in tt])
    plt.axhline(5., color='k')
    plt.axis([21, 26, 1, 20])
    plt.title('DECaLS depth')
    ps.savefig()



    [gsn,rsn,zsn] = [T.get('decam_%s_nanomaggies' % band) * np.sqrt(T.get('decam_%s_nanomaggies_invvar' % band))
                     for band in bands]
    TT = T[(gsn > 5.) * (rsn > 5.) * (zsn > 5.)]
    
    # plt.clf()
    # plt.plot(g-r, r-z, 'k.', alpha=0.2)
    # plt.xlabel('g - r (mag)')
    # plt.ylabel('r - z (mag)')
    # plt.xlim(-0.5, 2.5)
    # plt.ylim(-0.5, 3)
    # ps.savefig()

    plt.clf()
    lp = []
    cut = (TT.sdss_objc_type == 6)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_psfflux[:,i])
             for i in [1,2,4]]
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.3, color='b')
    lp.append(p[0])
    cut = (TT.sdss_objc_type == 3)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_modelflux[:,i])
             for i in [1,2,4]]
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.3, color='r')
    lp.append(p[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    plt.legend(lp, ['stars', 'galaxies'])
    plt.title('SDSS')
    ps.savefig()


    g = TT.decam_g_mag_corr
    r = TT.decam_r_mag_corr
    z = TT.decam_z_mag_corr

    plt.clf()
    lt,lp = [],[]
    for cut,cc,tt in [(TT.sdss_objc_type == 6, 'b', 'stars'),
                      (TT.sdss_objc_type == 3, 'r', 'galaxies'),
                      (TT.sdss_objc_type == 0, 'g', 'faint')]:
        p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.3, color=cc)
        lt.append(tt)
        lp.append(p[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    plt.legend(lp, lt)
    plt.title('DECaLS')
    ps.savefig()

    

    # Stars/galaxies in subplots

    plt.clf()
    lp = []
    cut = (TT.sdss_objc_type == 6)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_psfflux[:,i])
             for i in [1,2,4]]
    plt.subplot(1,2,1)
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.02, color='b')
    px = plt.plot(100, 100, '.', color='b')
    lp.append(px[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    cut = (TT.sdss_objc_type == 3)
    g,r,z = [NanoMaggies.nanomaggiesToMag(TT.sdss_modelflux[:,i])
             for i in [1,2,4]]
    plt.subplot(1,2,2)
    p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.02, color='r')
    px = plt.plot(100, 100, '.', color='r')
    lp.append(px[0])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - z (mag)')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 3)
    plt.figlegend(lp, ['stars', 'galaxies'], 'upper right')
    plt.suptitle('SDSS')
    ps.savefig()

    g = TT.decam_g_mag_corr
    r = TT.decam_r_mag_corr
    z = TT.decam_z_mag_corr

    plt.clf()
    lt,lp = [],[]
    for i,(cut,cc,tt) in enumerate([
        (TT.sdss_objc_type == 6, 'b', 'stars'),
        (TT.sdss_objc_type == 3, 'r', 'galaxies'),
        #(TT.sdss_objc_type == 0, 'g', 'faint'),
        ]):
        plt.subplot(1,2,i+1)
        p = plt.plot((g-r)[cut], (r-z)[cut], '.', alpha=0.02, color=cc)
        lt.append(tt)
        px = plt.plot(100, 100, '.', color=cc)
        lp.append(px[0])
        plt.xlabel('g - r (mag)')
        plt.ylabel('r - z (mag)')
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 3)
    plt.figlegend(lp, lt, 'upper right')
    plt.suptitle('DECaLS')
    ps.savefig()

    


if __name__ == '__main__':
    forced_phot()


    
