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
    fns = glob('forced/*/*/decam-*-N4-forced.fits')
    fns.sort()
    print(len(fns), 'forced-phot results files')

    fns = fns[:100]

    bricks = {}
    TT = []
    for fn in fns:
        T = fits_table(fn)
        T.srcid = (T.brickid.astype(np.int64) << 32 | T.objid)

        print('Read', len(T), 'from', fn)
        hdr = fitsio.read_header(fn)
        expnum = hdr['EXPNUM']
        ccdname = hdr['CCDNAME']
        f = hdr['FILTER']
        print('expnum', expnum, 'CCD', ccdname, 'Filter', f)

        bands = 'grz'

        ubricks = np.unique(T.brickname)
        print('bricks:', ubricks)
        for b in ubricks:
            if not b in bricks:
                bfn = os.path.join('/project/projectdirs/cosmo/data/legacysurvey/dr1/tractor', b[:3], 'tractor-%s.fits' % b)
                B = fits_table(bfn)
                B.srcid = (B.brickid.astype(np.int64) << 32 | B.objid)
                bricks[b] = B
                B.tmap = dict([(s,i) for i,s in enumerate(B.srcid)])
                for band in bands:
                    B.set('forced_flux_%s' % band, [ [] for i in range(len(B)) ])
                    B.set('forced_flux_ivar_%s' % band, [ [] for i in range(len(B)) ])
                    B.set('forced_mjd_%s' % band, [ [] for i in range(len(B)) ])

            else:
                B = bricks[b]

            band = T.filter[0]

            flux = B.get('forced_flux_%s' % band)
            fluxiv = B.get('forced_flux_ivar_%s' % band)
            mjd = B.get('forced_mjd_%s' % band)

            for j in np.flatnonzero(T.brickname == b):
                #try:
                i = B.tmap[T.srcid[j]]
                #except KeyError:
                #    continue
                flux[i].append(T.flux[j])
                fluxiv[i].append(T.flux_ivar[j])
                mjd[i].append(T.mjd[j])



    for brickname,B in bricks.items():
        print('Brick', brickname)
        outfn = os.path.join('fsweep', brickname[:3], 'tractor-%s.fits' % brickname)
        B.delete_column('tmap')
        for band in bands:
            nmax = max([len(lst) for lst in B.get('forced_flux_%s' % band)])
            print('Band:', band, 'Nmax:', nmax)
            for col,dt in [('forced_flux_%s' % band, np.float32),
                           ('forced_flux_ivar_%s' % band, np.float32),
                           ('forced_mjd_%s' % band, np.float64),]:
                if nmax == 0:
                    B.delete_column(col)
                    continue
                arr = np.zeros((len(B), nmax), dt)
                flist = B.get(col)
                for i,f in enumerate(flist):
                    arr[i,:len(f)] = f
                B.set(col, arr)

        
        trymakedirs(outfn, dir=True)
        B.writeto(outfn)

