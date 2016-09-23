from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.libkd.spherematch import *
from astrometry.util.miscutils import *
from tractor import *

ps = PlotSequence('cs82')

from cs82 import *

cs82field = 'S82p40p'
T,F = getTables(cs82field, enclosed=False)

T.about()
##
T = fits_table('cs82-phot-%s.fits' % cs82field)

T.alphamodel_j2000 = T.ra
T.deltamodel_j2000 = T.dec

T.phot_done = np.zeros(len(T), bool)
T.marginal = np.zeros(len(T), bool)
T.margindist = np.zeros(len(T), np.float32)

masks = get_cs82_masks(cs82field)
sdss = DR9(basedir='data/unzip')
sdss.useLocalTree()
sdss.saveUnzippedFiles('data/unzip')

print('Score range:', F.score.min(), F.score.max())
print('Before score cut:', len(F))
F.cut(F.score > 0.5)
print('Cut on score:', len(F))

ra0 = T.ra.min()
ra1 = T.ra.max()
dec0 = T.dec.min()
dec1 = T.dec.max()

# decs = np.linspace(dec0, dec1, 50)
# ras  = np.linspace(ra0,  ra1,  50)
# dlo,dhi = decs[10:12]
# rlo,rhi = ras [10:12]

# decs = np.linspace(dec0, dec1, 25)
# ras  = np.linspace(ra0,  ra1,  25)
# dlo,dhi = decs[10:12]
# rlo,rhi = ras [10:12]

rlo,rhi = 36.21, 36.235
#dlo,dhi = 0.34, 0.36
dlo,dhi = 0.342, 0.362

margin = 8. / 3600

# Cut to masks overlapping this slice.
masksi = []
for m in masks:
    mr0,md0 = np.min(m, axis=0)
    mr1,md1 = np.max(m, axis=0)
    if mr1 < rlo or mr0 > rhi or md1 < dlo or md0 > dhi:
        continue
    masksi.append(m)
print(len(masksi), 'masks overlap this slice')

# Cut to CS82 sources overlapping
Ibox = np.flatnonzero(
    ((T.dec + margin) >= dlo) * ((T.dec - margin) <= dhi) *
    ((T.ra  + margin) >= rlo) * ((T.ra  - margin) <= rhi))
T.marginal[:] = False
T.marginal[Ibox] = np.logical_not(
    (T.dec[Ibox] >= dlo) * (T.dec[Ibox] <= dhi) *
    (T.ra [Ibox] >= rlo) * (T.ra [Ibox] <= rhi))
print(len(Ibox), 'sources in RA,Dec slice')
print(len(np.flatnonzero(T.marginal)), 'are in the margins')

# Cut to SDSS fields overlapping
Fi = F[np.logical_not(np.logical_or(F.dec0 > dhi, F.dec1 < dlo)) *
       np.logical_not(np.logical_or(F.ra0  > rhi, F.ra1  < rlo))]
print(len(Fi), 'fields in RA,Dec slice')

band = 'r'
bands = 'r'

print('Creating Tractor sources...')
maglim = 24
cat,icat = get_cs82_sources(T[Ibox], maglim=maglim, bands=bands)
print('Got', len(cat), 'sources')
# Icat: index into T, row-parallel to cat
Icat = Ibox[icat]
del icat
print(len(Icat), 'sources created')

cat.freezeParamsRecursive('*')
cat.thawPathsTo(band)
cat.setParams(T.get('sdss_%s_nanomaggies' % band)[Icat])

pfn = 'tims.pickle'
if os.path.exists(pfn):
    tims,sigs,npix = unpickle_from_file(pfn)
else:
    # Read SDSS images...
    tims = []
    sigs = []
    npix = 0
    for i,(r,c,f) in enumerate(zip(Fi.run, Fi.camcol, Fi.field)):
        print('Reading', (i+1), 'of', len(Fi), ':', r,c,f,band)
        tim,inf = get_tractor_image_dr9(
            r, c, f, band, sdss=sdss,
            nanomaggies=True, zrange=[-2,5],
            roiradecbox=[rlo,rhi,dlo,dhi],
            invvarIgnoresSourceFlux=True)
        if tim is None:
            continue
        (H,W) = tim.shape
        print('Tim', tim.shape)
        tim.wcs.setConstantCd(W/2., H/2.)
        del tim.origInvvar
        del tim.starMask
        del tim.mask
        tim.domask = False
        tims.append(tim)
        sigs.append(1./np.sqrt(np.median(tim.invvar)))
        npix += (H*W)
        print('got', (H*W), 'pixels, total', npix)
        #print 'Read image', i+1, 'in band', band, ':', Time()-tb0
    
        #tm0 = Time()
        # Apply CS82 masks to SDSS images
        #astrans = tim.wcs.astrans
        #xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        # masked = np.zeros((H,W), bool)
        # for m in masksi:
        #     mx,my = astrans.radec_to_pixel(m[:,0], m[:,1])
        #     mx -= tim.wcs.x0
        #     my -= tim.wcs.y0
        #     mx0,my0 = int(np.floor(np.min(mx))), int(np.floor(np.min(my)))
        #     if mx0 > W or my0 > H:
        #         continue
        #     mx1,my1 = int(np.ceil(np.max(mx))), int(np.ceil(np.max(my)))
        #     if mx1 < 0 or my1 < 0:
        #         continue
        # 
        #     xlo = max(0, mx0)
        #     xhi = min(W, mx1)
        #     ylo = max(0, my0)
        #     yhi = min(H, my1)
        #     slc = slice(ylo,yhi), slice(xlo,xhi)
        # 
        #     masked[slc] |= point_in_poly(xx[slc], yy[slc],
        #                                  np.vstack((mx,my)).T)
        # print 'Masking', np.sum(masked), 'pixels'
        # tim.invvar[masked] = 0.
        # tim.setInvvar(tim.invvar)
        # print 'Total of', np.sum(tim.invvar > 0), 'unmasked pixels'
        # print 'Masking took:', Time()-tm0

    pickle_to_file((tims,sigs,npix), pfn)

print('Read', len(tims), 'images')
print('total of', npix, 'pixels')

# Create a fake WCS for this subregion -- for plots only
pixscale = 0.396 / 3600.
decpix = int(np.ceil((dhi - dlo) / pixscale))
# HACK -- ignoring cos(dec)
rapix = int(np.ceil((rhi - rlo) / pixscale))
wcs = Tan((rlo+rhi)/2., (dlo+dhi)/2., rapix/2 + 1, decpix/2 + 1,
          pixscale, 0., 0., pixscale, rapix, decpix)

coadd = np.zeros((wcs.imageh, wcs.imagew), np.float32)
ncoadd = np.zeros((wcs.imageh, wcs.imagew), np.int32)
coiv = np.zeros((wcs.imageh, wcs.imagew), np.float32)
for tim in tims:
    (H,W) = tim.shape
    try:
        wcswrap = AsTransWrapper(tim.wcs.astrans, W,H,
                                 tim.wcs.x0, tim.wcs.y0)
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(
            wcs, wcswrap, [], 3)
    except:
        import traceback
        print('Failed to resample:')
        traceback.print_exc()
        continue
    coadd[Yo,Xo] += tim.getImage()[Yi,Xi]
    coiv[Yo,Xo] += tim.getInvvar()[Yi,Xi]
    ncoadd[Yo,Xo] += 1
coadd = coadd / np.maximum(1, ncoadd).astype(np.float32)
print(len(tims), 'tims; ncoadd range %i %i; coadd range %g, %g' % (ncoadd.min(), ncoadd.max(), coadd.min(), coadd.max()))
plt.clf()
#coa = dict(interpolation='nearest', origin='lower',
#           extent=[rlo,rhi,dlo,dhi], vmin=-0.05, vmax=0.5)
coa = dict(interpolation='nearest', origin='lower',
           extent=[rlo,rhi,dlo,dhi], vmin=-0.02, vmax=0.1,
           cmap='gray')
plt.imshow(coadd, **coa)
#m = 0.003
m = 0.
plt.title('Coadd')
setRadecAxes(rlo-m,rhi+m,dlo-m,dhi+m)
ps.savefig()

# plt.clf()
# plt.imshow(coiv, interpolation='nearest', origin='lower',
#            extent=[rlo,rhi,dlo,dhi], vmin=0)
# plt.title('sum of invvars')
# setRadecAxes(rlo-m,rhi+m,dlo-m,dhi+m)

# Choose surface-brightness approximation level
sig1 = np.median(sigs)
minsig = 0.1
minsb = minsig * sig1
print('Sigma1:', sig1, 'minsig', minsig, 'minsb', minsb)
                
tractor = Tractor(tims, cat)

imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
             vmin=-3, vmax=3)
ima = dict(interpolation='nearest', origin='lower',
           vmin=-0.04, vmax=0.2, cmap='gray')

mcoadd = np.zeros((wcs.imageh, wcs.imagew), np.float32)
noisycoadd = np.zeros((wcs.imageh, wcs.imagew), np.float32)
ncoadd = np.zeros((wcs.imageh, wcs.imagew), np.int32)
for i,tim in enumerate(tims):
    mod = tractor.getModelImage(i)
    sig1 = sigs[i]
    noise = np.random.normal(size=mod.shape) * sig1
    img = tim.getImage()
        
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(np.rot90(img, k=3), **ima)
    plt.title('SDSS image')
    plt.subplot(2,2,2)
    plt.imshow(np.rot90(mod, k=3), **ima)
    plt.title('Model')
    plt.subplot(2,2,3)
    plt.imshow(np.rot90(mod+noise, k=3), **ima)
    plt.title('Model+Noise')
    plt.subplot(2,2,4)
    plt.imshow(np.rot90(-(img - mod)/sig1, k=3), **imchi)
    plt.title('Chi')
    plt.suptitle(tim.name)
    ps.savefig()

    (H,W) = tim.shape
    try:
        wcswrap = AsTransWrapper(tim.wcs.astrans, W,H,
                                 tim.wcs.x0, tim.wcs.y0)
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(
            wcs, wcswrap, [], 3)
    except:
        import traceback
        print('Failed to resample:')
        traceback.print_exc()
        continue
    mcoadd[Yo,Xo] += mod[Yi,Xi]
    noisycoadd[Yo,Xo] += (mod+noise)[Yi,Xi]
    ncoadd[Yo,Xo] += 1

mcoadd = mcoadd / np.maximum(1, ncoadd).astype(np.float32)
noisycoadd = noisycoadd / np.maximum(1, ncoadd).astype(np.float32)

plt.clf()
plt.imshow(mcoadd, **coa)
plt.title('Coadd of models')
setRadecAxes(rlo-m,rhi+m,dlo-m,dhi+m)
ps.savefig()

plt.clf()
plt.imshow(noisycoadd, **coa)
plt.title('Noisy coadd of models')
setRadecAxes(rlo-m,rhi+m,dlo-m,dhi+m)
ps.savefig()

W,H = rapix, decpix
fakeimg = Image(data=np.zeros((H,W)), invvar=np.ones((H,W)),
                psf=NCircularGaussianPSF([1.0], [1.0]),
                wcs=ConstantFitsWcs(wcs),
                sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1., band=band),
                domask=False)

tr2 = Tractor([fakeimg], cat)
mod = tr2.getModelImage(0)
plt.clf()
plt.imshow(mod, **coa)
plt.title('Model of coadd')
setRadecAxes(rlo-m,rhi+m,dlo-m,dhi+m)
ps.savefig()
                

sys.exit(0)



C = merge_tables([fits_table('data/cs82/cats/masked.S82p%ip_y.V2.7A.swarp.cut.deVexp.fit' % i, hdu=2)
                  for i in [40,41,42,43]])
print(len(C), 'CS82 catalog')
#C.about()
C.ptsrc = (C.spread_model < 0.0036)

F = merge_tables([fits_table('cs82-phot-S82p%ip.fits' % i)
                  for i in [40,41,42,43]])
print(len(F), 'Forced-phot catalog')

S = fits_table('cas-p4043p.fits')
print(len(S), 'SDSS CASJobs')

S.cut(S.nchild == 0)
print(len(S), 'child objects')

if not 'inmask' in S.get_columns():
    from cs82 import get_cs82_masks
    masks = [get_cs82_masks('S82p%ip' % i) for i in [40,41,42,43]]
    print('masks', masks[0][0])
    inmask = np.zeros(len(S), bool)
    kd = tree_build(np.vstack((S.ra, S.dec)).T)
    for mm in masks:
        for mi,m in enumerate(mm):
            if mi % 1000 == 0:
                print(mi)
            # mr0,md0 = np.min(m, axis=0)
            # mr1,md1 = np.max(m, axis=0)
            # I = np.flatnonzero((S.ra > mr0) * (S.ra < mr1) * (S.dec > md0) * (S.dec < md1))
    
            mc = np.mean(m, axis=0)
            mr2 = np.max(np.sum((m - mc)**2, axis=1))
            I = tree_search(kd, mc, np.sqrt(mr2))
    
            inside = point_in_poly(S.ra[I], S.dec[I], m)
            inmask[I[inside]] = True
            
    tree_free(kd)
    S.inmask = inmask
    S.writeto('cas-p4043p.fits')

print(sum(S.inmask), 'of', len(S), 'sources from CAS are inside masks')
S.cut(np.logical_not(S.inmask))
print(len(S), 'CAS after mask cut')

Idet = ((S.flags_i & 0x30000000) > 0)# binned1 | binned2
#S.cut(
#print 'Cut to', len(S), 'CAS detected in i-band'

I = (F.phot_done * F.fit_ok_i)
J = (C.ptsrc)

plt.clf()
pha = dict(range=((35.8, 39.5),(-0.09, 0.93)), imshowargs=dict(vmin=0, vmax=35))
plothist(F.ra[I], F.dec[I], 200, **pha)
plt.title('CS82 forced photometry')
ps.savefig()

plt.clf()
plothist(S.ra, S.dec, 200, **pha)
plt.title('CAS photometry')
ps.savefig()

K = np.logical_and(I, J)

sdss = F.sdss_i_mag[K]
cfht = C.mag_psf[K]
plt.clf()
loghist(cfht, sdss - cfht, 200, range=((18, 24),(-1,1)))
plt.xlabel('CS82 i-band (mag)')
plt.ylabel('SDSS i - CS82 i (mag)')
plt.title('CS82/SDSS forced photometry: PSFs')
ps.savefig()

K = np.logical_and(I, np.logical_not(J))

sdss = F.sdss_i_mag[K]
cfht = NanoMaggies.nanomaggiesToMag(
    NanoMaggies.magToNanomaggies(C.mag_disk[K]) +
    NanoMaggies.magToNanomaggies(C.mag_spheroid[K]))
plt.clf()
loghist(cfht, sdss - cfht, 200, range=((18, 24),(-1,1)))
plt.xlabel('CS82 i-band (mag)')
plt.ylabel('SDSS i - CS82 i (mag)')
plt.title('CS82/SDSS forced photometry: Galaxies')
ps.savefig()


FI = F[F.phot_done]

I,J,d = match_radec(S.ra, S.dec, FI.ra, FI.dec, 1./3600.)
print(len(I), 'matches')

for bi,band in enumerate('ugriz'):
    plt.clf()
    n1,b,p1 = plt.hist(S.get('modelmag_%s' % band),
                      100, range=(18,28),
                      histtype='step', color=(0.8,0.8,1), lw=3)
    n1b,b,p1b = plt.hist(S.get('modelmag_%s' % band)[Idet],
                      100, range=(18,28),
                      histtype='step', color=(0.7,0.9,0.7), lw=3)
    #n,b,p3 = plt.hist(S.get('psfmag_%s' % band),
    #                  100, range=(18,28),
    #                  histtype='step', color=(0.8,1,0.8), lw=3)
    n2,b,p2 = plt.hist(FI.get('sdss_%s_mag' % band),
                      100, range=(18,28), histtype='step', color='b')
    p1 = plt.plot([0],[0],'-', color=(0.8,0.8,1), lw=3)
    p1b = plt.plot([0],[0],'-', color=(0.7,0.9,0.7), lw=3)
    p2 = plt.plot([0],[0],'-', color='b')
    #p3 = plt.plot([0],[0],'-', color=(0.8,1,0.8), lw=3)
    plt.xlabel('%s (mag)' % band)
    plt.legend((p2[0], p1[0], p1b[0]), ('CS82 Forced phot', 'Annis et al coadds',
                                        'Annis, detected in i-band'),
               loc='upper left')
    plt.xlim(18,28)
    plt.ylim(0, 1.25*max(max(n1), max(n2)))
    ps.savefig()



for bi,band in enumerate('ugriz'):
    plt.clf()
    annis = S.get('modelmag_%s' % band)[I]
    forced = FI.get('sdss_%s_mag' % band)[J]
    loghist(annis, forced - annis, 200, range=((18,25),(-1,1)))
    plt.xlabel('Annis et al coadds: %s (mag)' % band)
    plt.ylabel('CS82 forced photometry - Annis: %s (mag)' % band)
    plt.title('CS82 forced photometry vs Annis et al coadds: %s band' % band)
    ps.savefig()

