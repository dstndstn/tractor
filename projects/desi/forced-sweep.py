from __future__ import print_function
import sys
import os
import numpy as np
import fitsio
from glob import glob
from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.file import *

if __name__ == '__main__':

    #fns = glob('forced/*/*/decam-*-forced.fits')
    fns = glob('forced/003*/*/decam-*-forced.fits')
    #fns = glob('forced/*/*/decam-*-N4-forced.fits')
    fns.sort()
    #fns = fns[:100]
    print(len(fns), 'forced-phot results files')


    TT = []
    for fn in fns:
        T = fits_table(fn, columns=['brickid', 'objid', 'flux', 'flux_ivar', 'mjd', 'filter', 'brickname'])
        T.srcid = (T.brickid.astype(np.int64) << 32 | T.objid)
        T.delete_column('brickid')
        T.delete_column('objid')

        print('Read', len(T), 'from', fn)
        # hdr = fitsio.read_header(fn)
        # expnum = hdr['EXPNUM']
        # ccdname = hdr['CCDNAME']
        # f = hdr['FILTER']
        # print('expnum', expnum, 'CCD', ccdname, 'Filter', f)
        TT.append(T)
    T = merge_tables(TT)
    del TT

    bands = 'grz'

    ubricks = np.unique(T.brickname)
    print('bricks:', ubricks)
    for brick in ubricks:
        bfn = os.path.join('/project/projectdirs/cosmo/data/legacysurvey/dr1/tractor', brick[:3], 'tractor-%s.fits' % brick)
        B = fits_table(bfn)
        B.srcid = (B.brickid.astype(np.int64) << 32 | B.objid)

        tmap = dict([(s,i) for i,s in enumerate(B.srcid)])
        
        for band in bands:
            flux   = [ [] for i in range(len(B)) ]
            fluxiv = [ [] for i in range(len(B)) ]
            mjd    = [ [] for i in range(len(B)) ]

            J = np.flatnonzero((T.brickname == brick) * (T.filter == band))
            print(len(J), 'in brick', brick, 'band', band)
            if len(J) == 0:
                continue
            for j in J:
                i = tmap[T.srcid[j]]
                flux[i].append(T.flux[j])
                fluxiv[i].append(T.flux_ivar[j])
                mjd[i].append(T.mjd[j])

            nmax = max([len(lst) for lst in flux])
            print('Band:', band, 'Nmax:', nmax)
            if nmax == 0:
                continue
            for flist,col,dt in [(flux,   'forced_flux_%s'      % band, np.float32),
                                 (fluxiv, 'forced_flux_ivar_%s' % band, np.float32),
                                 (mjd,    'forced_mjd_%s'       % band, np.float64),]:
                arr = np.zeros((len(B), nmax), dt)
                for i,f in enumerate(flist):
                    arr[i,:len(f)] = f
                B.set(col, arr)

        outfn = os.path.join('fsweep', brick[:3], 'tractor-%s.fits' % brick)
        trymakedirs(outfn, dir=True)
        B.writeto(outfn)
