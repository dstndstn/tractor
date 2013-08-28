import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *

from tractor import *
from tractor.ttime import *

import logging
lvl = logging.INFO
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

if __name__ == '__main__':
    '''
    ln -s /clusterfs/riemann/raid006/dr10/boss/sweeps/dr9 sweeps
    
    '''
    dataset = 'sequels'
    fn = '%s-atlas.fits' % dataset
    print 'Reading', fn
    T = fits_table(fn)

    tiledir = 'wise-coadds'
    bands = [1,2,3,4]
    
    ps = PlotSequence(dataset)

    # SEQUELS
    R0,R1 = 120.0, 200.0
    D0,D1 =  45.0,  60.0
    
    gsweeps = fits_table('sweeps/datasweep-index-star.fits')
    ssweeps = fits_table('sweeps/datasweep-index-gal.fits')
    print 'Read', len(gsweeps), 'galaxy sweep entries'
    print 'Read', len(ssweeps), 'star sweep entries'
    gsweeps.cut(gsweeps.nprimary > 0)
    ssweeps.cut(ssweeps.nprimary > 0)
    print 'Cut to', len(gsweeps), 'gal and', len(ssweeps), 'star on NPRIMARY'
    margin = 1
    gsweeps.cut((gsweeps.ra  > (R0-margin)) * (gsweeps.ra  < (R1+margin)) *
                (gsweeps.dec > (D0-margin)) * (gsweeps.dec < (D1+margin)))
    ssweeps.cut((ssweeps.ra  > (R0-margin)) * (ssweeps.ra  < (R1+margin)) *
                (ssweeps.dec > (D0-margin)) * (ssweeps.dec < (D1+margin)))
    print 'Cut to', len(gsweeps), 'gal and', len(ssweeps), 'star on RA,Dec box'
    gsweeps.isgal = np.ones(len(gsweeps))
    ssweeps.isgal = np.zeros(len(ssweeps))
    sweeps = merge_tables([gsweeps, ssweeps])
    print 'Merged:', len(sweeps)
    
    for tile in T:
        for band in bands:
            print
            print 'Coadd tile', tile.coadd_id
            print 'Band', band

            fn = os.path.join(tiledir, 'coadd-%s-w%i-img.fits' % (tile.coadd_id, band))
            print 'Reading', fn
            
            wcs = Tan(fn)
            r0,r1,d0,d1 = wcs.radec_bounds()
            print 'RA,Dec bounds:', r0,r1,d0,d1

            ra,dec = wcs.radec_center()
            print 'Center:', ra,dec

            margin = 0.

            # Add approx SDSS field size margin
            margin += np.hypot(13., 9.)/60.

            cosd = np.cos(np.deg2rad(sweeps.dec))            
            S = sweeps[(sweeps.ra  > (r0-margin/cosd)) * (sweeps.ra  < (r1+margin/cosd)) *
                       (sweeps.dec > (d0-margin))      * (sweeps.dec < (d1+margin))]
            print 'Cut to', len(S), 'datasweeps in this tile'
            
            
