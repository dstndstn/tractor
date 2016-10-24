from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from astrometry.util.plotutils import *

'''
This is a little script to investigate the expected depth of the
z-band images in DECaLS DR1.

Plots sent to decam-obs on 2015-06-09.

'''

from nightlystrategy import *
from astrometry.util.fits import *
from astrometry.libkd.spherematch import *
from astrometry.util.stages import *

def stage_read():
    C = fits_table('decals/decals-ccds.fits')
    band = 'z'
    C.cut(C.filter == band)
    print(len(C), 'z-band CCDs')
    C.cut(np.argsort(C.expnum))
    
    parser,gvs = getParserAndGlobals()
    opt,args = parser.parse_args(args='--date 2015-01-15 --portion 1 --pass 1'.split())
    decam = setupGlobals(opt, gvs)
    
    expfactors = np.zeros(len(C))
    
    Z = fits_table('zeropoints.fits')
    Z.expnum = np.array([int(x) for x in Z.expnum])
    Z.filter = np.array([f[0] for f in Z.filter])
    Z.ccdname = np.array([s.strip() for s in Z.ccdname])
    
    obstatus = fits_table('../obs-trunk/obstatus/decam-tiles_obstatus.fits')
    print(len(obstatus), 'obstatus tiles')
    obskd = tree_build_radec(obstatus.ra, obstatus.dec)
    
    R = fits_table()
    R.zp = []
    R.zpRaw = []
    R.skyflux = []
    R.sky = []
    R.ef = []
    R.et = []
    R.sat = []
    R.realet = []
    R.mjd = []
    R.fwhm = []
    R.photometric = []
    
    # unique dates
    ui = []
    udates = []
    
    # print 'DR1:', np.unique(C.dr1)
    # print 'Number of exposures marked non-photometric: ',
    # I = np.flatnonzero(C.dr1 == 0)
    # print len(np.unique(C.expnum[I]))
    # print 'Total number of exposures:', len(C)

    expnums = np.unique(C.expnum)

    for expnum in expnums:
        I = np.flatnonzero(C.expnum == expnum)
        c = C[I[0]]
        if 'COSMOS' in c.cpimage:
            continue
        if 'DES' in c.cpimage:
            continue
        if c.exptime <= 30:
            continue

        CC = C[I]
        
        # Record the first time a new date is seen
        # NOTE the date_obs values are in UTC, so this is dodgy!
        date = c.date_obs[:10]
        if not date in udates:
            udates.append(date)
            ui.append(len(R.zp))

        print('  ', c.cpimage)
        print('  exptime', c.exptime)
            
        J = []
        ZZ = Z[(Z.expnum == expnum)]
        for c in CC:
            ccdname = c.extname.strip()
            i = np.flatnonzero((ZZ.ccdname == ccdname))
            assert(len(i) == 1)
            J.append(i[0])
        ZZ = ZZ[np.array(J)]
        
        # If there is a mix of photometric and non-photometric CCDs,
        # cut to just the photometric ones.
        if len(np.unique(CC.dr1)) == 2:
            keep = np.flatnonzero(CC.dr1 == 1)
            CC.cut(keep)
            ZZ.cut(keep)
            print('Cut to', len(keep), 'photometric CCDs in this exposure')

        #airmass = np.median(CC.airmass)
        airmass = 1.0
        
        zp = np.median(ZZ.ccdzpt)
        fwhm = np.median(ZZ.seeing)
        skylev = np.median(c.avsky)
        gain = np.median(c.arawgain)
        skyflux = skylev / c.exptime
        
        print('  zeropoint', zp, 'in CP')
        print('  fwhm', fwhm)
        print('  sky level', skylev, 'in CP')
        print('  sky flux', skyflux, 'per second in CP units')
        skyflux /= (0.27**2)
        print('  sky flux', skyflux, 'per second per square arcsecond in CP')
        skyflux *= gain
        print('  sky flux', skyflux, 'e- per second per square arcsec')
        skysb = -2.5 * np.log10(skyflux)
    
        zpRaw = zp + 2.5*np.log10(gain)
        print('  zpRaw:', zpRaw)
        
        skyRaw = skysb + zpRaw
        print('  skyRaw', skyRaw)
    
        tsat = np.floor((30000.0/(0.27)**2)*np.power(10.0,-0.4*(26.484 - skyRaw)))
        print('  sat_max:', tsat)
        
        J = tree_search_radec(obskd, c.ra_bore, c.dec_bore, 1)
        print('  ', len(J), 'obstatus matches')
        ebv = obstatus[J[0]].ebv_med
        print('  ebv', ebv)

        ef = ExposureFactor(band, airmass, ebv, fwhm, zpRaw, skyRaw, gvs)
        print('  Exposure factor', ef)
    
        et = np.floor(ef * gvs.base_exptimes[band])
        print('  Exposure time', et)
        et = np.clip(et, gvs.floor_exptimes[band], gvs.ceil_exptimes[band])
        print('  Clipped exposure time', et)
        print('  Actual  exposure time', c.exptime)
    
        R.zp.append(zp)
        R.zpRaw.append(zpRaw)
        R.skyflux.append(skyflux)
        R.sky.append(skyRaw)
        R.ef.append(ef)
        R.et.append(et)
        R.realet.append(c.exptime)
        R.sat.append(tsat)
        R.mjd.append(c.mjd_obs)
        R.fwhm.append(fwhm)
        R.photometric.append(c.dr1)
    
    R.to_np_arrays()
    return dict(R=R, ui=ui, udates=udates)


    # for i,c in enumerate(C):
    #     if 'COSMOS' in c.cpimage:
    #         continue
    #     if 'DES' in c.cpimage:
    #         continue
    # 
    #     # Record the first time a new date is seen
    #     # NOTE the date_obs values are in UTC, so this is dodgy!
    #     date = c.date_obs[:10]
    #     if not date in udates:
    #         udates.append(date)
    #         ui.append(len(R.zp))
    # 
    #     ccdname = c.extname.strip()
    #     if ccdname != 'N4':
    #         continue
    #     if c.exptime <= 30:
    #         continue
    # 
    #     #if c.dr1 == 0:
    #     #    nonphotom.append(len(zps))
    # 
    #     I = np.flatnonzero((Z.expnum == c.expnum) * (Z.ccdname == ccdname))
    #     print 'Expnum', c.expnum, 'ccdname', ccdname, 'matched', len(I), 'in zeropoints file'
    #     print '  ', c.cpimage
    #     print '  exptime', c.exptime
    #     assert(len(I) == 1)
    #     z = Z[I[0]]
    #     zp = z.ccdzpt
    #     print '  zeropoint', zp, 'in CP'
    #     airmass = c.airmass
    #     fwhm = z.seeing
    #     print '  fwhm', fwhm
    # 
    #     print '  sky level', c.avsky, 'in CP'
    #     skyflux = c.avsky / c.exptime
    #     print '  sky flux', skyflux, 'per second in CP units'
    #     skyflux /= (0.27**2)
    #     print '  sky flux', skyflux, 'per second per square arcsecond in CP'
    #     skyflux *= c.arawgain
    #     print '  sky flux', skyflux, 'e- per second per square arcsec'
    #     skysb = -2.5 * np.log10(skyflux)
    # 
    #     zpRaw = zp + 2.5*np.log10(c.arawgain)
    #     print '  zpRaw:', zpRaw
    #     
    #     skyRaw = skysb + zpRaw
    #     print '  skyRaw', skyRaw
    # 
    #     tsat = np.floor((30000.0/(0.27)**2)*np.power(10.0,-0.4*(26.484 - skyRaw)))
    #     print '  sat_max:', tsat
    #     
    #     J = tree_search_radec(obskd, c.ra_bore, c.dec_bore, 1)
    #     print '  ', len(J), 'obstatus matches'
    #     ebv = obstatus[J[0]].ebv_med
    #     print '  ebv', ebv
    #     
    #     ef = ExposureFactor(band, airmass, ebv, fwhm, zpRaw, skyRaw, gvs)
    #     print '  Exposure factor', ef
    # 
    #     et = np.floor(ef * gvs.base_exptimes[band])
    #     print '  Exposure time', et
    #     et = np.clip(et, gvs.floor_exptimes[band], gvs.ceil_exptimes[band])
    #     print '  Clipped exposure time', et
    #     print '  Actual  exposure time', c.exptime
    # 
    #     R.zp.append(zp)
    #     R.zpRaw.append(zpRaw)
    #     R.skyflux.append(skyflux)
    #     R.sky.append(skyRaw)
    #     R.ef.append(ef)
    #     R.et.append(et)
    #     R.realet.append(c.exptime)
    #     R.sat.append(tsat)
    #     R.mjd.append(c.mjd_obs)
    #     R.fwhm.append(fwhm)
    #     R.photometric.append(c.dr1)
    # 
    # R.to_np_arrays()
    # return dict(R=R, ui=ui, udates=udates)

def stage_plot(R=None, ui=None, udates=None):
    plt.figure(figsize=(12,16))
    #plt.figure(figsize=(12,12))
    #plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.2)
    plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.2)

    # Effective exposure time
    R.teff = R.realet / R.ef
    
    for cut in [False,True]:
        if cut:
            keep = (R.photometric == 1)
    
            ii = np.flatnonzero(keep)
            ui = np.array([np.sum(keep[:i]) for i in ui])
    
            R.cut(keep)
            
        rows,cols = 6,1
        plt.clf()
    
        for s in range(rows):
            plt.subplot(rows, cols, s+1)
            nonphotom = np.flatnonzero(R.photometric == 0)
            for i in nonphotom:
                plt.axvline(i, color='y', alpha=0.2)
    
        zpnom = 26.484 - 2.5*np.log10(4.3)
    
        plt.subplot(rows, cols, 1)
        plt.plot(R.zp, 'b')
        plt.ylabel('Zeropoint')
        plt.ylim(24.5, 25.5)
        plt.axhline(zpnom, color='k', alpha=0.2)
    
        for i,d in zip(ui, udates):
            plt.axvline(i, color='k', alpha=0.2)
            plt.text(i, 25, d, rotation='vertical', size='smaller')
    
        plt.subplot(rows, cols, 2)
    
        #plt.plot(skyfluxes, 'b')
        #plt.ylabel('Sky flux')
        #plt.ylim(0, 8000)
    
        skynom = 18.46
        
        plt.plot(R.sky, 'b')
        plt.ylabel('Sky Surf Bright')
        plt.axhline(skynom, color='k', alpha=0.2)
        plt.yticks(range(16,21))
        plt.ylim(16, 20)
    
        plt.subplot(rows, cols, 3)
        plt.plot(R.fwhm, 'b')
        plt.ylabel('FWHM')
        plt.axhline(1.3, color='k', alpha=0.2)
        plt.ylim(0.5, 2.5)
        
        plt.subplot(rows, cols, 4)
        plt.semilogy(R.ef, 'b')
        plt.ylabel('Exposure factor')
        plt.axhline(1., color='k', alpha=0.2)
        plt.axhline(0.8, color='k', ls='--', alpha=0.2)
        plt.axhline(2.5, color='k', ls='--', alpha=0.2)
        plt.semilogy(R.sat / 100., 'k', alpha=0.2)
        #plt.ylim(0.2, 10)
        plt.ylim(0.5, 5)
    
        plt.subplot(rows, cols, 5)
        plt.semilogy(R.realet, 'b')
        plt.ylabel('Actual exposure time')
        plt.axhline(100., color='k', alpha=0.2)
        plt.axhline(80, color='k', ls='--', alpha=0.2)
        plt.axhline(250, color='k', ls='--', alpha=0.2)
        plt.semilogy(R.sat, 'k', alpha=0.2)
        #plt.ylim(0.2 * 100., 10 * 100.)
        plt.ylim(0.5 * 100., 5 * 100.)

        plt.subplot(rows, cols, 6)
        plt.plot(R.teff, 'b')
        plt.ylabel('Effective exposure time')
        plt.axhline(100., color='k', alpha=0.2)
        plt.ylim(25, 300)
        
        # plt.subplot(rows, cols, 5)
        # plt.plot(ets, 'b')
        # #plt.plot(sats, 'r')
        # plt.plot(realets, 'g')
        # plt.ylabel('times')
        # plt.ylim(0, 250)
    
        for i in range(rows):
            plt.subplot(rows, cols, 1+i)
            plt.xlim(0, len(R))
            plt.xticks([])
    
            if i > 0:
                for ii,d in zip(ui, udates):
                    plt.axvline(ii, color='k', alpha=0.2)

        plt.xticks(ui, udates, rotation='vertical')
                    
        plt.suptitle('DECaLS DR1 Observing Stats')
        ps.savefig()
    
    
    plt.figure(figsize=(8,6))
    plt.clf()
    #plt.scatter(R.et, R.realet, c=(R.sat > R.et), s=9, alpha=0.4)
    I = np.flatnonzero(R.sat < R.et)
    p1 = plt.plot(R.et[I], R.realet[I], 'r.', alpha=0.5)
    I = np.flatnonzero(R.sat >= R.et)
    p2 = plt.plot(R.et[I], R.realet[I], 'b.', alpha=0.5)
    plt.axis([0,260,0,260])
    plt.xlabel('Computed exposure time')
    plt.ylabel('Actual exposure time')
    plt.title('DECaLS DR1 Observing Stats')
    plt.legend([p2[0],p1[0]], ('Normal', 'Sat-clipped'),
               loc='upper left')
    ps.savefig()
    
    hstyle = dict(histtype='step', color='b')
    
    plt.clf()
    lo,hi = 0, 260
    plt.hist(np.clip(R.et, lo,hi), 50, range=(lo, hi), **hstyle)
    plt.xlabel('Computed exposure time (s)')
    plt.axvline(100., color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    plt.xlim(lo, hi)
    ps.savefig()

    plt.clf()
    lo,hi = 0, 260
    plt.hist(np.clip(R.realet, lo,hi), 50, range=(lo, hi), **hstyle)
    plt.xlabel('Actual exposure time (s)')
    plt.axvline(100., color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    plt.xlim(lo, hi)
    ps.savefig()

    plt.clf()
    lo,hi = 0, 260
    plt.hist(np.clip(R.teff, lo, hi), 50, range=(lo,hi), **hstyle)
    plt.xlabel('Effective exposure time (s)')
    plt.axvline(100, color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    plt.xlim(lo, hi)
    ps.savefig()
    
    plt.clf()
    lo,hi = 24.5, 25.5
    plt.hist(np.clip(R.zp, lo, hi), 50, range=(lo, hi), **hstyle)
    plt.xlabel('Zeropoint')
    plt.axvline(zpnom, color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    plt.xlim(lo,hi)
    ps.savefig()
        
    plt.clf()
    lo,hi = 16,20
    plt.hist(np.clip(R.sky, lo,hi), 50, range=(lo, hi), **hstyle)
    plt.xlabel('Sky surface brightness')
    plt.axvline(skynom, color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    ps.savefig()
    
    fwhmnom = 1.3
    plt.clf()
    lo,hi = 0.5, 2.5
    plt.hist(np.clip(R.fwhm, lo,hi), 50, range=(lo, hi), **hstyle)
    plt.xlabel('FWHM (arcsec)')
    plt.axvline(fwhmnom, color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    ps.savefig()
    
    plt.clf()
    lo,hi = 0, 10
    plt.hist(np.clip(R.ef, lo,hi), 50, range=(lo, hi), **hstyle)
    plt.xlabel('Exposure factors')
    plt.axvline(1., color='r', alpha=0.5, lw=2)
    plt.title('DECaLS DR1 Observing Stats')
    ps.savefig()
    
    # plt.clf()
    # plt.hist(realets, 50)
    # plt.xlabel('Actual exposure times')
    # ps.savefig()


if __name__ == '__main__':
    ps = PlotSequence('et')

    stagefunc = CallGlobal('stage_%s', globals())

    runstage('plot', 'exptime-%(stage)s.pickle', stagefunc,
             prereqs={ 'plot': 'read', 'read':None },
        force=['plot'])
    
    
