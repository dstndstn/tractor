import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *

import urllib2

import fitsio

#  Ta = fits_table('phot-1384p454-w1-b16.fits')
#  Tb = fits_table('phot-1384p454-w1-b17.fits')
#  blocksa = 16
#  blocksb = 17

def reassemble_chunks(mods, blocks, imargin):
    rows = []
    j = 0
    while len(mods):
        #print 'row start'
        modrow = mods[:blocks]
        for i in range(1, len(modrow)):
            modrow[i] = modrow[i][:,imargin:]
        for i in range(0, len(modrow)-1):
            modrow[i] = modrow[i][:,:-imargin]
        if j > 0:
            for i in range(len(modrow)):
                modrow[i] = modrow[i][imargin:,:]
        if j < blocks-1:
            for i in range(len(modrow)):
                modrow[i] = modrow[i][:-imargin,:]
        j += 1
        mods = mods[blocks:]
        #for m in modrow:
        #    print 'shape:', m.shape
        row = np.hstack(modrow)
        #print 'row:', row.shape
        rows.append(row)
    mod = np.vstack(rows)
    #print 'moda:', moda.shape
    return mod




ps = PlotSequence('comp')

fna = 'phot-1384p454-b8.fits'
fnb = 'phot-1384p454-b6.fits'
Ta = fits_table(fna)
Tb = fits_table(fnb)
blocksa = 8
blocksb = 6

modsa,catsa,sxa,sya,srada = unpickle_from_file(fna.replace('.fits','.pickle'))
modsb,catsb,sxb,syb,sradb = unpickle_from_file(fnb.replace('.fits','.pickle'))

imargin = 12

band = 1
coadd_id = '1384p454'
tiledir = 'wise-coadds'
fn = os.path.join(tiledir, 'coadd-%s-w%i-img.fits' % (coadd_id, band))
print 'Reading', fn
wcs = Tan(fn)
H,W = wcs.get_height(), wcs.get_width()
print 'Shape', H,W
H,W = 1024,1024

img = fitsio.read(fn)

# cell positions
Xa = np.round(np.linspace(0, W, blocksa+1)).astype(int)
Ya = np.round(np.linspace(0, H, blocksa+1)).astype(int)
Xb = np.round(np.linspace(0, W, blocksb+1)).astype(int)
Yb = np.round(np.linspace(0, H, blocksb+1)).astype(int)

#moda = reassemble_chunks(modsa, blocksa, imargin)
#modb = reassemble_chunks(modsb, blocksb, imargin)

ima = dict(interpolation='nearest', origin='lower',
           vmin=-0.01, vmax=0.5, cmap='gray')

plt.clf()
plt.hist(img.ravel(), 100, range=(-0.5,1))
plt.xlim(-0.5,1)
ps.savefig()

G = fits_table('sweeps-%s-gals.fits' % coadd_id)
print len(G), 'galaxies'
G.cut((G.theta_dev[:,2] > 0) * (G.theta_exp[:,2] > 0))
print len(G), 'galaxies with positive thetas'
G.cut(G.modelflux[:,2] > 0)
print len(G), 'galaxies with positive flux'

###
G.modelflux[:,2] = np.maximum(1e-2, G.modelflux[:,2])
G.modelflux[:,2] = np.minimum(1e4,  G.modelflux[:,2])
G.theta_dev[:,2] = np.maximum(1e-2, G.theta_dev[:,2])
G.theta_exp[:,2] = np.maximum(1e-2, G.theta_exp[:,2])

D = G[G.fracdev[:,2] > 0.5]
E = G[G.fracdev[:,2] <= 0.5]
print len(D), 'dev', len(E), 'exp'
D.cut(D.theta_dev[:,2] > 0)
E.cut(E.theta_exp[:,2] > 0)
print len(D), 'dev', len(E), 'exp with positive theta'

# plt.clf()
# plt.hist(G.theta_dev[:,2], 50, histtype='step', color='r')
# plt.hist(D.theta_dev[:,2], 50, histtype='step', color='r')
# plt.hist(G.theta_exp[:,2], 50, histtype='step', color='b')
# plt.hist(E.theta_exp[:,2], 50, histtype='step', color='b')
# plt.xlabel('theta')
# ps.savefig()

plt.clf()
plt.hist(np.log10(G.theta_dev[:,2]), 50, histtype='step', color='r')
plt.hist(np.log10(D.theta_dev[:,2]), 50, histtype='step', color='r')
plt.hist(np.log10(G.theta_exp[:,2]), 50, histtype='step', color='b')
plt.hist(np.log10(E.theta_exp[:,2]), 50, histtype='step', color='b')
plt.xlabel('log theta')
ps.savefig()

plt.clf()
plt.hist(G.ab_dev[:,2], 100, histtype='step', color='r')
plt.hist(G.ab_exp[:,2], 100, histtype='step', color='b')
plt.hist(D.ab_dev[:,2], 100, histtype='step', color='r')
plt.hist(E.ab_exp[:,2], 100, histtype='step', color='b')
plt.xlabel('ab')
ps.savefig()

plt.clf()
plt.hist(np.log10(G.modelflux[:,2]), 100, histtype='step', color='r')
plt.xlabel('log modelflux')
ps.savefig()

# plt.clf()
# loghist(D.theta_dev[:,2], D.modelflux[:,2], 100)
# plt.xlabel('theta_dev')
# plt.ylabel('modelflux')
# ps.savefig()

plt.clf()
loghist(D.theta_dev[:,2], np.log10(D.modelflux[:,2]), 100)
plt.xlabel('theta_dev')
plt.ylabel('log modelflux')
ps.savefig()

# plt.clf()
# loghist(E.theta_exp[:,2], E.modelflux[:,2], 100)
# plt.xlabel('theta_exp')
# plt.ylabel('modelflux')
# ps.savefig()

plt.clf()
loghist(E.theta_exp[:,2], np.log10(E.modelflux[:,2]), 100)
plt.xlabel('theta_exp')
plt.ylabel('log modelflux')
ps.savefig()

# plt.clf()
# loghist(E.ab_exp[:,2], E.modelflux[:,2], 100)
# plt.xlabel('ab_exp')
# plt.ylabel('modelflux')
# ps.savefig()

plt.clf()
loghist(E.ab_exp[:,2], np.log10(E.modelflux[:,2]), 100)
plt.xlabel('ab_exp')
plt.ylabel('log modelflux')
ps.savefig()


plt.clf()
loghist(D.theta_dev[:,2], D.ab_dev[:,2], 100)
plt.xlabel('theta_dev')
plt.ylabel('ab_dev')
ps.savefig()

plt.clf()
loghist(E.theta_exp[:,2], E.ab_exp[:,2], 100)
plt.xlabel('theta_exp')
plt.ylabel('ab_exp')
ps.savefig()

plt.clf()
loghist(E.theta_exp[:,2], np.minimum(60., E.theta_experr[:,2]), 100)
plt.xlabel('theta_exp')
plt.ylabel('theta_exp err')
ps.savefig()

plt.clf()
loghist(D.theta_dev[:,2], np.minimum(30., D.theta_deverr[:,2]), 100)
plt.xlabel('theta_dev')
plt.ylabel('theta_dev err')
ps.savefig()

plt.clf()
loghist(E.ab_exp[:,2], np.minimum(1., E.ab_experr[:,2]), 100)
plt.xlabel('ab_exp')
plt.ylabel('ab_exp err')
ps.savefig()

plt.clf()
loghist(D.ab_dev[:,2], np.minimum(1., D.ab_deverr[:,2]), 100)
plt.xlabel('ab_dev')
plt.ylabel('ab_dev err')
ps.savefig()


# Plot some wacky objects
I = np.flatnonzero(D.modelflux[:,2] >= 1e4)
T = D[I]
print 'Bright dev: theta_dev=', T.theta_dev[:,2]
rows,cols = 4,6
plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.1, right=0.9,
                    bottom=0.1, top=0.9)
plt.clf()
for i in range(min(len(T), rows * cols)):
    ra,dec = T.ra[i], T.dec[i]
    url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=%g&dec=%g&scale=1&width=128&height=128' % (ra,dec)
    fn = 'cutout-%g-%g.jpg' % (ra,dec)
    if not os.path.exists(fn):
        cmd = 'wget "%s" -O "%s"' % (url, fn)
        print cmd
        os.system(cmd)
    I = plt.imread(fn)
    plt.subplot(rows, cols, i+1)
    plt.imshow(I, interpolation='nearest', origin='lower')
ps.savefig()






hha,wwa = modsa[0].shape
hhb,wwb = modsb[0].shape
hh = min(hha,hhb)
ww = min(wwa,wwb)

print 'A:', wwa,hha
print 'B:', wwb,hhb

print 'sxa', sxa.shape

HH = max(hha,hhb)
WW = max(wwa,wwb)

imd = dict(interpolation='nearest', origin='lower',
           vmin=-1e-2, vmax=1e-2, cmap='gray')

ra,da,srcsa,tim = catsa[0]
rb,db,srcsb,tim = catsb[0]

print 'Catalog A:'
for src in srcsa:
    print '  ', src

print
print 'Catalog B:'
for src in srcsb:
    print '  ', src


print
I,J,d = match_radec(np.array(ra),np.array(da), np.array(rb),np.array(db), 1./3600.)
print len(I), 'matches'

for i,j in zip(I,J):
    print
    print srcsa[i]
    print srcsb[j]

print

for im,imargs,tt,cat in [(modsa[0], ima, 'Model A (%i), first block' % blocksa, catsa[0]),
                         (modsb[0], ima, 'Model B (%i), first block' % blocksb, catsb[0]),
                         (img, ima, 'Data', None),
                         (modsa[0][:hh,:ww] - modsb[0][:hh,:ww], imd, 'Model A-B', catsa[0])]:

    plt.clf()
    #plt.imshow(im[:hh,:ww], **ima)
    plt.imshow(im, **imargs)
    h,w = im.shape
    plt.axhline(h - imargin, color='b')
    plt.axvline(w - imargin, color='b')
    plt.title(tt)

    ax = [-10, WW+10, -10, HH+10]
    plt.axis(ax)

    ps.savefig()

    I = np.flatnonzero((sxa > -20) * (sxa < WW+20) * (sya > -20) * (sya < HH+20))
    x = sxa[I]
    y = sya[I]
    r = srada[I]

    plt.plot(x, y, 'r.')
    plt.plot(np.vstack([x-r, x+r]), np.vstack([y,y]), 'r-')
    plt.plot(np.vstack([x,x]), np.vstack([y-r, y+r]), 'r-')

    if cat is not None:
        r,d,srcs,tim = cat
        r = np.array(r)
        d = np.array(d)
        ok,x,y = wcs.radec2pixelxy(r,d)
        x -= 1
        y -= 1
        plt.plot(x, y, 'o', mec='g', mfc='none')

    plt.axis(ax)
    ps.savefig()



# plt.clf()
# plt.imshow(modsb[0][:hh,:ww], **ima)
# plt.title('Model B (%i), first block' % blocksb)
# ps.savefig()
# 
# plt.clf()
# plt.imshow(img[:hh,:ww], **ima)
# plt.title('Data A, one block')
# ps.savefig()


plt.imshow(modsa[0][:hh,:ww] - modsb[0][:hh,:ww],
           interpolation='nearest', origin='lower',
           vmin=-1e-2, vmax=1e-2, cmap='gray')
plt.title('Model A - Model B')
ps.savefig()


# for img,a in [(moda,ima),(modb,ima),(moda-modb, dict(interpolation='nearest', origin='lower',
#                                                      vmin=-1e-3, vmax=1.e-3, cmap='gray'))]:
#     plt.clf()
#     plt.imshow(img, **a)
#     ax = plt.axis()
#     for x in Xa:
#         plt.axvline(x, color='b', alpha=0.5)
#     for y in Ya:
#         plt.axhline(y, color='b', alpha=0.5)
#     for x in Xb:
#         plt.axvline(x, color='r', alpha=0.5)
#     for y in Yb:
#         plt.axhline(y, color='r', alpha=0.5)
#     plt.axis(ax)
#     ps.savefig()
#     plt.axis([0,400,0,400])
#     ps.savefig()


# plt.clf()
# plt.plot(Ta.ra, Ta.dec, 'r.')
# plt.xlabel('RA')
# plt.ylabel('Dec')
# ps.savefig()

# plt.clf()
# ha = dict(bins=100, histtype='step')
# plt.hist(Ta.w1_nanomaggies, color='b', **ha)
# # plt.hist(Ta.w2_nanomaggies, color='g', **ha)
# # plt.hist(Ta.w3_nanomaggies, color='r', **ha)
# # plt.hist(Ta.w4_nanomaggies, color='m', **ha)
# ps.savefig()

plt.clf()
plt.plot(Ta.w1_nanomaggies, Tb.w1_nanomaggies, 'b.')
# plt.plot(Ta.w2_nanomaggies, Tb.w2_nanomaggies, 'g.')
# plt.plot(Ta.w3_nanomaggies, Tb.w3_nanomaggies, 'r.')
# plt.plot(Ta.w4_nanomaggies, Tb.w4_nanomaggies, 'm.')
plt.xlabel('run A flux (nanomaggies)')
plt.ylabel('run B flux (nanomaggies)')
plt.xscale('symlog')
plt.yscale('symlog')
ps.savefig()

plt.clf()
plt.plot(Ta.w1_nanomaggies, Tb.w1_nanomaggies / Ta.w1_nanomaggies, 'b.')

I = np.random.randint(len(Tb), size=(100,))
d = 1./np.sqrt(Tb.w1_nanomaggies_ivar[I])
y = Tb.w1_nanomaggies[I]
x = Ta.w1_nanomaggies[I]
plt.plot(np.vstack([x, x]), np.vstack([y-d, y+d]) / x, 'b-', alpha=0.5)

I = np.flatnonzero(np.logical_or(y / x > 2, y/x < 0.5))
d = 1./np.sqrt(Tb.w1_nanomaggies_ivar[I])
y = Tb.w1_nanomaggies[I]
x = Ta.w1_nanomaggies[I]
plt.plot(np.vstack([x, x]), np.vstack([y-d, y+d]) / x, 'b-', alpha=0.5)

plt.xlabel('a (nm)')
plt.ylabel('b/a (nm)')
plt.xscale('symlog')
#plt.yscale('symlog')
ps.savefig()

ratio = Tb.w1_nanomaggies / Ta.w1_nanomaggies
O = np.flatnonzero(np.logical_or(ratio > 2., ratio < 0.5))
print len(O), 'outliers'

# where are the outliers?
plt.clf()
plt.plot(Ta.ra, Ta.dec, 'r.', alpha=0.1)
plt.plot(Ta.ra[O], Ta.dec[O], 'b.', alpha=0.5)
plt.xlabel('RA')
plt.ylabel('Dec')
ps.savefig()

ok,X,Y = wcs.radec2pixelxy(Ta.ra, Ta.dec)
X -= 1.
Y -= 1.
Ta.x = X
Ta.y = Y

plt.clf()
#plt.plot(Ta.x[Ta.w1_ntimes == 1], Ta.y[Ta.w1_ntimes == 1], 'r.', alpha=0.1)
plt.plot(Ta.x, Ta.y, 'r.', alpha=0.1)
plt.plot(Ta.x[O], Ta.y[O], 'b.', alpha=0.5)
#plt.plot(Ta.x[Ta.w1_ntimes == 0], Ta.y[Ta.w1_ntimes == 0], 'g.', alpha=0.5)
ax = plt.axis()
for x in Xa:
    plt.axvline(x, color='k', alpha=0.5)
for y in Ya:
    plt.axhline(y, color='k', alpha=0.5)
for x in Xb:
    plt.axvline(x, color='g', alpha=0.5)
for y in Yb:
    plt.axhline(y, color='g', alpha=0.5)
plt.axis(ax)
plt.xlabel('x')
plt.ylabel('y')
ps.savefig()


