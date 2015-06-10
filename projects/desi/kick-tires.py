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
from astrometry.libkd.spherematch import match_radec
from tractor import *
from tractor.galaxy import *
from common import *

if __name__ == '__main__':

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
    print len(T), 'catalog sources'
    print np.unique(T.brick_primary)
    T.cut(T.brick_primary)
    print len(T), 'primary'

    print 'Out of bounds:', np.unique(T.out_of_bounds)
    print 'Left blob:', np.unique(T.left_blob)
    
    img = plt.imread(os.path.join('dr1', 'coadd', brick[:3], brick,
                                  'decals-%s-image.jpg' % brick))
    img = img[::-1,:,:]
    print 'Image:', img.shape

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
    print len(I), 'EXP with large ellipticity error'
    cutobjs.append((E[I], 'EXP ellipticity error > 1'))

    E.cut(np.logical_not(cut))
    
    I = np.flatnonzero(np.logical_or(D.shapedev_e1_err > 1., D.shapedev_e2_err > 1.))
    print len(I), 'DEV with large ellipticity error'
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
            print 'No objects in cut', desc
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

        S = 25

        objs.x0 = np.maximum(0, np.round(objs.bx - S).astype(int))
        objs.y0 = np.maximum(0, np.round(objs.by - S).astype(int))

        for i,o in enumerate(objs):
            plt.subplot(rows, cols, i+1)
            H,W,three = img.shape
            dimshow(img[o.y0:min(H, o.by+S),
                        o.x0:min(W, o.bx+S), :], ticks=False)
            #print 'Cutout pixels range:'
            

        plt.suptitle(desc)
        ps.savefig()

        for i,o in enumerate(objs):
            plt.subplot(rows, cols, i+1)

            ax = plt.axis()
            # plot sources nearby...
            near = T[(np.abs(T.bx - o.bx) < S) * (np.abs(T.by - o.by) < S)]

            # print len(near), 'nearby sources'
            # print '  type', near.type
            # print '  shapeexp_e1', near.shapeexp_e1
            # print '  shapeexp_e2', near.shapeexp_e2
            
            x0 = np.maximum(0, np.round(o.bx - S).astype(int))
            y0 = np.maximum(0, np.round(o.by - S).astype(int))

            # print '  bx', near.bx
            # print '  by', near.by
            # print '  bx-x0', near.bx - x0
            # print '  by-y0', near.by - y0
            
            for n in near:
                e = None
                cc = None
                if n.type.strip() == 'PSF':
                    plt.plot(n.bx-x0, n.by-y0, 'r.')
                elif n.type.strip() == 'EXP':
                    e = EllipseE(n.shapeexp_r, n.shapeexp_e1, n.shapeexp_e2)
                    cc = '1'
                elif n.type.strip() == 'DEV':
                    e = EllipseE(n.shapedev_r, n.shapedev_e1, n.shapedev_e2)
                    cc = 'r'
                elif n.type.strip() == 'COMP':
                    plt.plot(n.bx-x0, n.by-y0, 'ro')
                    
                if e is not None:
                    G = e.getRaDecBasis()
                    angle = np.linspace(0, 2.*np.pi, 60)
                    xy = np.vstack((np.append([0,0,1], np.sin(angle)),
                                    np.append([0,1,0], np.cos(angle)))).T
                    # print 'G', G
                    #print 'G', G.shape
                    #print 'xy', xy.shape
                    rd = np.dot(G, xy.T).T
                    ## print 'rd', rd.shape
                    ra  = n.ra  + rd[:,0] * np.cos(np.deg2rad(n.dec))
                    dec = n.dec + rd[:,1]
                    ok,xx,yy = brickwcs.radec2pixelxy(ra, dec)
                    #print 'xx,yy', xx.shape, yy.shape
                    x1,x2,x3 = xx[:3]
                    y1,y2,y3 = yy[:3]
                    # print '  x123', xx[:3]-x0
                    # print '  y123', yy[:3]-y0
                    plt.plot([x3 - x0, x1 - x0, x2 - x0],
                             [y3 - y0, y1 - y0, y2 - y0], '-', color=cc)
                    plt.plot(x1 - x0, y1 - y0, '.', color=cc, ms=3)
                    xx = xx[3:]
                    yy = yy[3:]
                    plt.plot(xx - x0, yy - y0, '-', color=cc)
            plt.axis(ax)

            plt.text(0, 0, '%i/%i/%i' % 
                    (int(100. * o.decam_fracflux[1]),
                     int(100. * o.decam_fracflux[2]),
                     int(100. * o.decam_fracflux[4])),
                     color='red', size='small',
                     ha='left', va='bottom')

            #plt.text(0, 2*S, '%.3f, %.3f' % (o.ra, o.dec),
            #         color='red', size='small', ha='left', va='top')
            
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

    
    print 'RA', T.ra.min(), T.ra.max()
    print 'Dec', T.dec.min(), T.dec.max()

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

    


