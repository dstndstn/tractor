import glob as glob
import os

import numpy as np

from astrometry.util.starutil import *
from astrometry.util.fits import *

import fitsio


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%prog [options] <frame frame frame>')
    opt,args = parser.parse_args()

    nan = np.nan
    primkeys = [('FILTER',''),
                ('RA', nan),
                ('DEC', nan),
                ('AIRMASS', nan),
                ('DATE-OBS', ''),
                ('G-SEEING', nan),
                ('EXPTIME', nan),
                ('EXPNUM', 0),
                ]
    hdrkeys = [('AVSKY', nan),
               ('ARAWGAIN', nan),
               ('FWHM', nan),
               ]

    vals = dict([(k,[]) for k,d in primkeys + hdrkeys])

    for fn in args:
        print 'Reading', fn
        F = fitsio.FITS(fn)

        print F
        print len(F)
        primhdr = F[0].read_header()
        print primhdr

        # filt = primhdr['FILTER'].split()[0]
        # ra  = hmsstring2ra (primhdr['RA'])
        # dec = dmsstring2dec(primhdr['DEC'])
        # airmass = primhdr['AIRMASS']
        # date    = primhdr['DATE-OBS']
        # gseeing = primhdr['G-SEEING']
        # exptime = primhdr['EXPTIME']
        # 
        # print 'Filt', filt, 'RA,Dec', ra,dec
        # print 'exptime', exptime
        # print 'airmass', airmass
        # print 'date', date
        # print 'gsee', gseeing

        for hdu in range(1, len(F)):
            hdr = F[hdu].read_header()
            
            for k,d in primkeys:
                vals[k].append(primhdr.get(k, d))
            for k,d in hdrkeys:
                vals[k].append(hdr.get(k, d))

    T = fits_table()
    for k,v in vals.items():
        T.set(k.lower().replace('-','_'), np.array(v))
    T.about()

    T.filter = np.array([s.split()[0] for s in T.filter])
    T.ra  = np.array([hmsstring2ra (s) for s in T.ra ])
    T.dec = np.array([dmsstring2dec(s) for s in T.dec])
    T.about()

    for k in T.get_columns():
        print
        print k
        print T.get(k)

    T.writeto('ccds.fits')
    
                
