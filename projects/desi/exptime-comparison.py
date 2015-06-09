import matplotlib
matplotlib.use('Agg')
import pylab as plt
from astrometry.util.plotutils import *

'''
This is a little script to investigate the expected depth of the z-band images in DECaLS DR1.

Plots sent to decam-obs on 2015-06-09.

'''

ps = PlotSequence('et')

from nightlystrategy import *
from astrometry.util.fits import *
from astrometry.libkd.spherematch import *
#from tractor.sfd import *

C = fits_table('decals/decals-ccds.fits')
band = 'z'
C.cut(C.filter == band)
print len(C), 'z-band CCDs'
C.cut(np.argsort(C.expnum))

parser,gvs = getParserAndGlobals()
opt,args = parser.parse_args(args='--date 2015-01-15 --portion 1 --pass 1'.split())
decam = setupGlobals(opt, gvs)

expfactors = np.zeros(len(C))

Z = fits_table('zeropoints.fits')
Z.expnum = np.array([int(x) for x in Z.expnum])
Z.filter = np.array([f[0] for f in Z.filter])
Z.ccdname = np.array([s.strip() for s in Z.ccdname])

#sfd = SFDMap()

obstatus = fits_table('../obs-trunk/obstatus/decam-tiles_obstatus.fits')
print len(obstatus), 'obstatus tiles'

zps = []
zpRaws = []
skyfluxes = []
skys = []
efs = []
ets = []
sats = []
realets = []
mjds = []
fwhms = []

ui = []
udates = []

nonphotom = []

obskd = tree_build_radec(obstatus.ra, obstatus.dec)

print 'DR1:', np.unique(C.dr1)

print 'Number of exposures marked non-photometric: ',
I = np.flatnonzero(C.dr1 == 0)
print len(np.unique(C.expnum[I]))
print 'Total number of exposures:', len(C)

for i,c in enumerate(C):
    if 'COSMOS' in c.cpimage:
        continue
    if 'DES' in c.cpimage:
        continue

    date = c.date_obs[:10]
    if not date in udates:
        udates.append(date)
        ui.append(len(zps))

    ccdname = c.extname.strip()
    #if ccdname != 'S29':
    if ccdname != 'N4':
        continue
    if c.exptime < 30:
        continue

    if c.dr1 == 0:
        nonphotom.append(len(zps))

    I = np.flatnonzero((Z.expnum == c.expnum) * (Z.ccdname == ccdname))
    print 'Expnum', c.expnum, 'ccdname', ccdname, 'matched', len(I), 'in zeropoints file'
    print '  ', c.cpimage
    print '  exptime', c.exptime
    assert(len(I) == 1)
    z = Z[I[0]]
    zp = z.ccdzpt
    print '  zeropoint', zp, 'in CP'
    #fid_exptime = gvs.base_exptimes[band]
    #zp += 2.5 * np.log10(fid_exptime)
    #print '  zeropoint at fiducial exptime', zp
    airmass = c.airmass
    fwhm = z.seeing
    print '  fwhm', fwhm

    print '  sky level', c.avsky, 'in CP'
    skyflux = c.avsky / c.exptime
    print '  sky flux', skyflux, 'per second in CP units'
    skyflux /= (0.27**2)
    print '  sky flux', skyflux, 'per second per square arcsecond in CP'
    skyflux *= c.arawgain
    print '  sky flux', skyflux, 'e- per second per square arcsec'

    skysb = -2.5 * np.log10(skyflux)
    #print '  sky inst. mag', skysb

    zpRaw = zp + 2.5*np.log10(c.arawgain)
    print '  zpRaw:', zpRaw
    
    skyRaw = skysb + zpRaw
    print '  skyRaw', skyRaw

    tsat = np.floor((30000.0/(0.27)**2)*np.power(10.0,-0.4*(26.484 - skyRaw)))
    print '  sat_max:', tsat
    
    #I,J,d = match_radec(c.ra_bore, c.dec_bore, obstatus.ra, obstatus.dec, 1)
    J = tree_search_radec(obskd, c.ra_bore, c.dec_bore, 1)
    print '  ', len(J), 'obstatus matches'
    ebv = obstatus[J[0]].ebv_med
    print '  ebv', ebv
    
    ef = ExposureFactor(band, airmass, ebv, fwhm, zpRaw, skyRaw, gvs)
    print '  Exposure factor', ef

    et = np.floor(ef * gvs.base_exptimes[band])
    print '  Exposure time', et
    et = np.clip(et, gvs.floor_exptimes[band], gvs.ceil_exptimes[band])
    print '  Clipped exposure time', et
    print '  Actual  exposure time', c.exptime

    zps.append(zp)
    zpRaws.append(zpRaw)
    skyfluxes.append(skyflux)
    skys.append(skyRaw)
    efs.append(ef)
    ets.append(et)
    realets.append(c.exptime)
    sats.append(tsat)
    mjds.append(c.mjd_obs)
    fwhms.append(fwhm)
    
#plt.figure(figsize=(12,16))
plt.figure(figsize=(12,12))
plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.2)

for cut in [False,True]:
    if cut:
        keep = np.ones(len(zps), bool)
        keep[np.array(nonphotom)] = False
        ii = np.flatnonzero(keep)
        ui = np.array([np.sum(keep[:i]) for i in ui])
        zps = np.array(zps)[keep]
        nonphotom = []
        skys = np.array(skys)[keep]
        efs = np.array(efs)[keep]
        fwhms = np.array(fwhms)[keep]

        sats = np.array(sats)[keep]
        realets = np.array(realets)[keep]
        ets = np.array(ets)[keep]
        
    rows,cols = 4,1
    plt.clf()


    for s in range(rows):
        plt.subplot(rows, cols, s+1)
        for i in nonphotom:
            plt.axvline(i, color='y', alpha=0.2)

    zpnom = 26.484 - 2.5*np.log10(4.3)

    plt.subplot(rows, cols, 1)
    plt.plot(zps, 'b')
    #plt.plot(zpRaws, 'r')
    plt.ylabel('Zeropoint')
    #plt.ylim(23, 27)
    #plt.ylim(24, 26)
    plt.ylim(24.5, 25.5)
    plt.axhline(zpnom, color='k', alpha=0.2)

    #plt.xticks(ui, udates)
    for i,d in zip(ui, udates):
        plt.axvline(i, color='k', alpha=0.2)
        plt.text(i, 25, d, rotation='vertical', size='smaller')
    plt.xticks([])

    plt.subplot(rows, cols, 2)

    #plt.plot(skyfluxes, 'b')
    #plt.ylabel('Sky flux')
    #plt.ylim(0, 8000)

    #plt.subplot(rows, cols, 3)
    skynom = 18.46
    
    plt.plot(skys, 'b')
    plt.ylabel('Sky Surf Bright')
    plt.axhline(skynom, color='k', alpha=0.2)
    plt.yticks(range(16,21))
    plt.xticks([])
    plt.ylim(16, 20)

    plt.subplot(rows, cols, 3)
    plt.plot(fwhms, 'b')
    plt.ylabel('FWHM')
    plt.axhline(1.3, color='k', alpha=0.2)
    plt.ylim(0.5, 2.5)
    plt.xticks([])
    
    plt.subplot(rows, cols, 4)
    plt.semilogy(efs, 'b')
    plt.ylabel('Exposure factor')
    plt.axhline(1., color='k', alpha=0.2)
    plt.ylim(0.2, 10)

    # plt.subplot(rows, cols, 5)
    # plt.plot(ets, 'b')
    # #plt.plot(sats, 'r')
    # plt.plot(realets, 'g')
    # plt.ylabel('times')
    # plt.ylim(0, 250)

    for i in range(rows):
        plt.subplot(rows, cols, 1+i)
        plt.xlim(0, len(zps))

        if i > 0:
            for ii,d in zip(ui, udates):
                plt.axvline(ii, color='k', alpha=0.2)

    plt.suptitle('DECaLS DR1 Observing Stats')
    ps.savefig()


plt.figure(figsize=(8,6))
plt.clf()
plt.scatter(ets, realets, c=(sats > ets), s=9, alpha=0.4)
plt.axis([0,260,0,260])
plt.xlabel('Computed exposure time')
plt.ylabel('Actual exposure time')
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()

# for i,(arr,name,lo,hi, sub, cc) in enumerate([
# 
#         (zpRaws, 'ZP(raw)', 24.5, 27.5, 1, 'r'),
#         (skyfluxes, 'Sky flux', 0, 8000, 2, 'b'),
#         (skys, 'Sky', 15, 20, 3, 'b'),
#         (efs, 'Exp factor', 0, 10,),
#         (ets, 'Exp time', 50, 250),
#         (realets, 'Real exp time', 0, 250),
#         (sats, 'Sat time', 0, 250),]):
#     plt.subplot(8, 1, 1+i)
#     plt.plot(arr)
#     #plt.xlabel('MJD')
#     plt.ylabel(name)
#     plt.xlim(0, len(arr))
#     plt.ylim(lo, hi)
# ps.savefig()

# Effective exposure time
teff = realets / efs

hstyle = dict(histtype='step', color='b')

plt.clf()
lo,hi = 0, 250
plt.hist(np.clip(ets, lo,hi), 50, range=(lo, hi), **hstyle)
plt.xlabel('Exposure times')
plt.axvline(100., color='r', alpha=0.5, lw=2)
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()

plt.clf()
lo,hi = 0, 250
plt.hist(np.clip(teff, lo, hi), 50, range=(lo,hi), **hstyle)
plt.xlabel('Effective exposure time (s)')
plt.axvline(100, color='r', alpha=0.5, lw=2)
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()


plt.clf()
lo,hi = 24.5, 25.5
plt.hist(np.clip(zps, lo, hi), 50, range=(lo, hi), **hstyle)
plt.xlabel('Zeropoint')
plt.axvline(zpnom, color='r', alpha=0.5, lw=2)
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()
    
plt.clf()
lo,hi = 16,20
plt.hist(np.clip(skys, lo,hi), 50, range=(lo, hi), **hstyle)
plt.xlabel('Sky surface brightness')
plt.axvline(skynom, color='r', alpha=0.5, lw=2)
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()

fwhmnom = 1.3
plt.clf()
lo,hi = 0.5, 3.0
plt.hist(np.clip(fwhms, lo,hi), 50, range=(lo, hi), **hstyle)
plt.xlabel('FWHM (arcsec)')
plt.axvline(fwhmnom, color='r', alpha=0.5, lw=2)
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()

plt.clf()
lo,hi = 0, 10
plt.hist(np.clip(efs, lo,hi), 50, range=(lo, hi), **hstyle)
plt.xlabel('Exposure factors')
plt.axvline(1., color='r', alpha=0.5, lw=2)
plt.title('DECaLS DR1 Observing Stats')
ps.savefig()


# plt.clf()
# plt.hist(realets, 50)
# plt.xlabel('Actual exposure times')
# ps.savefig()
