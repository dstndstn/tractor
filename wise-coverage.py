#! /usr/bin/env python

import os
import sys

if __name__ == '__main__':
    d = os.environ.get('PBS_O_WORKDIR')
    if d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.blind.plotstuff import *


def image_way():
    W,H = 4000,2000
    plot = Plotstuff(size=(W,H), outformat='png')
    plot.wcs = anwcs_create_allsky_hammer_aitoff(180., 0., W, H)
    out = plot.outline
    out.stepsize = 2000
    out.fill = 1
    
    wcs = Tan()
    out.wcs = anwcs_new_tan(wcs)
    wcs = anwcs_get_sip(out.wcs)
    wcs = wcs.wcstan

    totals = [np.zeros((H,W), int) for b in range(4)]
    
    ps = PlotSequence('cov')
    for nbands in [4,3,2]:
        bb = [1,2,3,4][:nbands]
        for band in bb:

            ofn = 'cov-n%i-b%i.fits' % (nbands, band)
            if os.path.exists(ofn):
                print 'Exists:', ofn
                count = fitsio.read(ofn)
                print 'Read', count.shape, count.dtype, 'max', count.max()
                totals[band-1] += count

                plt.clf()
                plt.imshow(count, interpolation='nearest', origin='lower',
                           vmin=0, vmax=100, cmap='gray')
                plt.colorbar()
                ps.savefig()
                continue

            count = np.zeros((H,W), np.int16)
            fn = 'wise-frames/WISE-l1b-metadata-%iband.fits' % nbands
            cols = [('w%i'%band)+c for c in
                    ['crval1','crval2','crpix1','crpix2',
                     'cd1_1','cd1_2','cd2_1','cd2_2', 'naxis1','naxis2']]
            print 'Reading', fn
            T = fits_table(fn, columns=cols)
            print 'Read', len(T), 'from', fn
            arrs = [T.get(c).astype(float) for c in cols]

            plot.clear()
            plot.color = 'white'
            plot.alpha = 1./255.
            plot.op = CAIRO_OPERATOR_ADD

            N = len(T)
            for i in xrange(N):
                if arrs[-1][i] == -1:
                    continue
                wcs.set(*[a[i] for a in arrs])
                plot.plot('outline')

                if i and i % 10000 == 0 or i == N-1:
                    print 'exposure', i, 'of', N
                    im = plot.get_image_as_numpy()
                    print 'max:', im[:,:,0].max()
                    count += im[:,:,0]
                    del im
                    print 'total max:', count.max()
                    plot.clear()
                    
            fitsio.write(ofn, count, clobber=True)
            print 'Wrote', ofn

            totals[band-1] += count

            plt.clf()
            plt.imshow(count, interpolation='nearest', origin='lower',
                       vmin=0, vmax=100, cmap='gray')
            plt.colorbar()
            ps.savefig()
            
            del T
            del arrs




    M = reduce(np.logical_or, [t > 0 for t in totals])

    for tot in totals:
        plt.clf()
        plt.imshow(tot, interpolation='nearest', origin='lower',
                   vmin=0, vmax=100, cmap='gray')
        plt.colorbar()
        ps.savefig()

    plt.clf()
    mx = 60
    for tot,cc in zip(totals, 'bgrm'):
        plt.hist(np.minimum(tot[M], mx), range=(0,mx),
                 bins=mx+1, histtype='step', color=cc)
    ps.savefig()

        
def healpix_way():
    Nside = 200
    NHP = 12 * Nside**2
    r0,r1,d0,d1 = [np.zeros(NHP) for i in range(4)]

    ra,dec = [np.zeros(NHP) for i in range(2)]
    
    counts = [np.zeros(NHP) for i in range(4)]
    
    print 'Healpix ranges for', NHP
    for hp in range(NHP):
        r0[hp],r1[hp],d0[hp],d1[hp] = healpix_radec_bounds(hp, Nside)
        ra[hp],dec[hp] = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
        
    wcs = Tan()
    for nbands in [4,3,2]:
        bb = [1,2,3,4][:nbands]

        fn = 'wise-frames/WISE-l1b-metadata-%iband.fits' % nbands
        cols = 'ra','dec'
        print 'Reading', fn
        T = fits_table(fn, columns=cols)
        print 'Read', len(T), 'from', fn

        I,J,d = match_radec(T.ra, T.dec, ra, dec, 1.)
        print 'Matched', len(I)
        
        for band in bb:
            #fn = 'wise-frames/WISE-l1b-metadata-%iband.fits' % nbands
            cols = [('w%i'%band)+c for c in
                    ['crval1','crval2','crpix1','crpix2',
                     'cd1_1','cd1_2','cd2_1','cd2_2', 'naxis1','naxis2']]
            print 'Reading', fn
            T = fits_table(fn, columns=cols, rows=I)
            print 'Read', len(T), 'from', fn
            arrs = [T.get(c).astype(float) for c in cols]

            N = len(T)
            for i in xrange(N):
                if arrs[-1][i] == -1:
                    continue
                wcs.set(*[a[i] for a in arrs])
                #rlo,rhi,dlo,dhi = wcs.radec_bounds()
                #I = np.flatnonzero(

                JJ = np.unique(J[I == i])
                print 'WCS', i, ':', len(JJ), 'matched'
                for j in JJ:
                    if wcs.is_inside(ra[j], dec[j]):
                        counts[band-1][j] += 1

    for i,c in enumerate(counts):
        fn = 'coverage-hp-w%i.fits' % (i+1)
        fitsio.write(fn, c, clobber=True)
        print 'Wrote', fn

                
                
    
if __name__ == '__main__':
    healpix_way()
    
