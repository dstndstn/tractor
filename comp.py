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

modsa = unpickle_from_file(fna.replace('.fits','.pickle'))
modsb = unpickle_from_file(fnb.replace('.fits','.pickle'))

imargin = 8

band = 1
coadd_id = '1384p454'
tiledir = 'wise-coadds'
fn = os.path.join(tiledir, 'coadd-%s-w%i-img.fits' % (coadd_id, band))
print 'Reading', fn
wcs = Tan(fn)
H,W = wcs.get_height(), wcs.get_width()
print 'Shape', H,W

H,W = 1024,1024

# cell positions
Xa = np.round(np.linspace(0, W, blocksa+1)).astype(int)
Ya = np.round(np.linspace(0, H, blocksa+1)).astype(int)
Xb = np.round(np.linspace(0, W, blocksb+1)).astype(int)
Yb = np.round(np.linspace(0, H, blocksb+1)).astype(int)



moda = reassemble_chunks(modsa, blocksa, imargin)
modb = reassemble_chunks(modsb, blocksb, imargin)

ima = dict(interpolation='nearest', origin='lower',
           vmin=0, vmax=1., cmap='gray')

plt.clf()
plt.imshow(modsa[0], **ima)
ps.savefig()

for img,a in [(moda,ima),(modb,ima),(moda-modb, dict(interpolation='nearest', origin='lower',
                                                     vmin=-1e-3, vmax=1.e-3, cmap='gray'))]:
    plt.clf()
    plt.imshow(img, **a)
    ax = plt.axis()
    for x in Xa:
        plt.axvline(x, color='b', alpha=0.5)
    for y in Ya:
        plt.axhline(y, color='b', alpha=0.5)
    for x in Xb:
        plt.axvline(x, color='r', alpha=0.5)
    for y in Yb:
        plt.axhline(y, color='r', alpha=0.5)
    plt.axis(ax)
    ps.savefig()
    plt.axis([0,400,0,400])
    ps.savefig()


plt.clf()
plt.plot(Ta.ra, Ta.dec, 'r.')
plt.xlabel('RA')
plt.ylabel('Dec')
ps.savefig()

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
plt.xlabel('a (nm)')
plt.ylabel('b/a (nm)')
plt.xscale('symlog')
plt.yscale('symlog')
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


