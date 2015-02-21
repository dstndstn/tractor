from astrometry.util.fits import *
import numpy as np
import os
import fitsio

T = fits_table('cosmos-ccds.fits')
print len(T), 'CCDs in COSMOS'
en,I = np.unique(T.expnum, return_index=True)
T.cut(I)
print len(T), 'exposures'
print 'filters', np.unique(T.filter)
bands = 'grz'
T.cut(np.array([f in bands for f in T.filter]))
print len(T), bands

minexptime = dict(g=60, r=60, z=100)

T.exptime = np.round(T.exptime).astype(int)

for band in bands:
    print
    Ti = T[T.filter == band]
    print len(Ti), 'in', band
    et = np.unique(Ti.exptime)
    print 'Exposure times:', et
    for t in et:
        Tt = Ti[Ti.exptime == t]
        print len(Tt), 'with exptime', t
        #en = np.unique(Tt.expnum)

    print
    mint = minexptime[band]
    Ti.cut(Ti.exptime >= mint)
    print len(Ti), 'with exptime >=', mint

    print 'g_seeing values:', np.unique(Ti.g_seeing)
    for fn in Ti.cpimage:
        print 'file', fn
        fn = os.path.join('decals/images/decam', fn)
        if not os.path.exists(fn):
            print 'Not found:', fn
            fn = fn.replace('.fits.fz', '.fits')
            if not os.path.exists(fn):
                print 'Not found:', fn
                continue
        print 'Reading', fn
        hdr = fitsio.read_header(fn)
        
        phot = hdr['PHOT_FLAG']
        print 'PHOT:', phot, '(photometric=1)'
        
