import glob as glob
import os

import numpy as np

from astrometry.util.util import *
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
               ('ZNAXIS1',0),
               ('ZNAXIS2',0),
               ('CRPIX1',nan),
               ('CRPIX2',nan),
               ('CRVAL1',nan),
               ('CRVAL2',nan),
               ('CD1_1',nan),
               ('CD1_2',nan),
               ('CD2_1',nan),
               ('CD2_2',nan),
               ]

    vals = dict([(k,[]) for k,d in
                 primkeys + hdrkeys + [('FILENAME',''), ('HDU',0)]])

    for fn in args:
        print 'Reading', fn
        F = fitsio.FITS(fn)
        #print F
        #print len(F)
        primhdr = F[0].read_header()
        #print primhdr

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

            vals['FILENAME'].append(fn)
            vals['HDU'].append(hdu)

    T = fits_table()
    #for k,v in vals.items():
    for k,d in primkeys + hdrkeys:
        T.set(k.lower().replace('-','_'), np.array(vals[k]))
    #T.about()

    T.filter = np.array([s.split()[0] for s in T.filter])
    T.ra_bore  = np.array([hmsstring2ra (s) for s in T.ra ])
    T.dec_bore = np.array([dmsstring2dec(s) for s in T.dec])

    T.ra  = np.zeros(len(T))
    T.dec = np.zeros(len(T))
    for i in range(len(T)):
        # FIXME -- is this the right way around?
        W,H = T.znaxis1[i], T.znaxis2[i]

        wcs = Tan(T.crval1[i], T.crval2[i], T.crpix1[i], T.crpix2[i],
                  T.cd1_1[i], T.cd1_2[i], T.cd2_1[i], T.cd2_2[i], float(W), float(H))
        
        xc,yc = W/2.+0.5, H/2.+0.5
        rc,dc = wcs.pixelxy2radec(xc,yc)
        T.ra [i] = rc
        T.dec[i] = dc

    T.about()

    for k in T.get_columns():
        print
        print k
        print T.get(k)

    T.writeto('ccds.fits')
    
                
