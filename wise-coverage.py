#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.blind.plotstuff import *

if __name__ == '__main__':
    W,H = 4000,2000
    plot = Plotstuff(size=(W,H), outformat='png')
    plot.wcs = anwcs_create_allsky_hammer_aitoff(0., 0., W, H)
    out = plot.outline
    out.stepsize = 2000
    out.fill = 1

    ps = PlotSequence('cov')
    for nbands in [4,3,2]:
        bb = [1,2,3,4][:nbands]
        for band in bb:
            count = np.zeros((H,W), int)
            fn = 'wise-frames/WISE-l1b-metadata-%iband.fits' % nbands
            # cols = [('w%i'%band)+c for c in
            #         ['crval1','crval2','crpix1','crpix2',
            #          'cd1_1','cd1_2','cd2_1','cd2_2', 'naxis1','naxis2']]

            cols = [('w%i'%band)+c for c in
                    ['ra1','dec1','ra2','dec2','ra3','dec3','ra4','dec4']]

            T = fits_table(fn, columns=cols)
            print 'Read', len(T), 'from', fn
            wcs = Tan()
            #out.wcs = anwcs_new_tan(wcs)

            arrs = [T.get(c).astype(float) for c in cols]
            r1,d1,r2,d2,r3,d3,r4,d4 = arrs

            plot.clear()
            plot.color = 'white'
            plot.alpha = 1./255.
            plot.op = CAIRO_OPERATOR_ADD

            N = len(T)
            for i in xrange(N):
                #wcs.set(*[a[i] for a in arrs])
                #plot.plot('outline')

                # plot.move_to_radec(T.ra1[i], T.dec1[i])
                # plot.line_to_radec(T.ra2[i], T.dec2[i])
                # plot.line_to_radec(T.ra3[i], T.dec3[i])
                # plot.line_to_radec(T.ra4[i], T.dec4[i])
                plot.move_to_radec(r1[i], d1[i])
                plot.line_to_radec(r2[i], d2[i])
                plot.line_to_radec(r3[i], d3[i])
                plot.line_to_radec(r4[i], d4[i])
                plot.close_path()
                plot.fill()

                if i and i % 1000 == 0 or i == N-1:
                    print 'exposure', i, 'of', N
                    im = plot.get_image_as_numpy()
                    #print 'im:', im.shape
                    print 'max:', im.max()
                    im = im[:,:,0]
                    count += im
                    print 'total max:', count.max()
                    plot.clear()

                    # plot.op = CAIRO_OPERATOR_OVER
                    # plot.color = 'black'
                    # plot.plot('fill')
                    # plot.color = 'white'
                    # plot.op = CAIRO_OPERATOR_ADD

            fitsio.write('cov-n%i-b%i.fits' % (nbands, band), count)
