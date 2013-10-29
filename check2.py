import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.fits import *
from astrometry.util.util import *

ps = PlotSequence('check')
coadd_id = '1497p015'
band = 1

fn = '%s-w%i-data.fits' % (coadd_id, band)
data = fitsio.read(fn)
print 'Read data', data.shape, data.dtype
fn = '%s-w%i-mod.fits' % (coadd_id, band)
mod = fitsio.read(fn)
print 'Read mod', mod.shape, mod.dtype
fn = '%s-w%i-ierr.fits' % (coadd_id, band)
ierr = fitsio.read(fn)
print 'Read ierr', ierr.shape, ierr.dtype
fn = '%s-w%i-chi.fits' % (coadd_id, band)
chi = fitsio.read(fn)
print 'Read chi', chi.shape, chi.dtype
sig1 = 1./np.median(ierr)
H,W = data.shape

T = fits_table('cosmos-phot-%s.fits' % coadd_id)
N = 20
T.cut((T.x > N) * (T.x < W-N) * (T.y > N) * (T.y < H-N))
T.cut(T.get('w%i_mag'%band) > 10.)

W = fits_table('phot-temp/wise-sources-%s.fits' % coadd_id)
wcs = Tan('wise-coadds/149/1497p015/unwise-1497p015-w1-img-m.fits')
ok,W.x,W.y = wcs.radec2pixelxy(W.ra, W.dec)
W.x -= 1.
W.y -= 1.

I = np.argsort(-T.get('w%i_prochi2' % band) / T.get('w%i_pronpix' % band))
print 'T:', len(T), 'I', len(I), 'max', I.max()

rows,cols = 8,8

for pnum in range(3):
    plt.clf()
    plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99, wspace=0.1, hspace=0.1)
    for k in range(rows*cols):
        plt.subplot(rows, cols, k+1)
        ii = I[k]
        print 'k', k, 'ii', ii
        x,y = T.x[ii], T.y[ii]
        slc = (slice(y-N, y+N+1), slice(x-N, x+N+1))

        imga = dict(interpolation='nearest', origin='lower',
                   vmin=-2.*sig1, vmax=10.*sig1, cmap='gray')
        ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                   extent=[x-N, x+N, y-N, y+N])

        if pnum == 0:
            #plt.imshow(data[slc], **ima)
            plt.imshow(np.sqrt(np.maximum(0, data[slc] / sig1)), vmin=0., vmax=10., **ima)
            #plt.imshow(np.log10(np.maximum(1e-6, 2.+(data[slc]/sig1))), vmin=0., vmax=3., **ima)
        elif pnum == 1:
            #plt.imshow(mod[slc], **ima)
            plt.imshow(np.sqrt(np.maximum(0, mod[slc] / sig1)), vmin=0., vmax=10., **ima)
            #plt.imshow(np.log10(np.maximum(1e-6, 2.+(mod[slc]/sig1))), vmin=0., vmax=3., **ima)
        elif pnum == 2:
            plt.imshow(chi[slc], interpolation='nearest', origin='lower',
                       vmin=-5, vmax=5, cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.text(N/2, N/2, '%.2f' % T.get('w%i_mag' % band)[ii], va='center', ha='center',
                 color='r')
        ax = plt.axis()
        S = T[(T.x > x-N) * (T.x < x+N) * (T.y > y-N) * (T.y < y+N)]
        plt.plot(S.x, S.y, 'r+')
        S = W[(W.x > x-N) * (W.x < x+N) * (W.y > y-N) * (W.y < y+N)]
        plt.plot(S.x, S.y, 'gx')#, mec='g', mfc='none')
        plt.axis(ax)
    ps.savefig()
    
