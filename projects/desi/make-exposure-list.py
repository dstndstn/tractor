import glob as glob
import os

import numpy as np

from astrometry.util.util import *
from astrometry.util.starutil import *
from astrometry.util.fits import *

import fitsio

'''
python -u projects/desi/make-exposure-list.py ~/cosmo/data/staging/decam/CP*/c4d_*_ooi*.fits.fz > log 2> err &
python -u projects/desi/make-exposure-list.py -o ccds-20140810.fits ~/cosmo/data/staging/decam/CP20140810/c4d_*_ooi*.fits.fz > log-0810 2>&1 &
'''

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%prog [options] <frame frame frame>')
    parser.add_option('-o', dest='outfn', help='Output filename', default='ccds.fits')
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
               ('EXTNAME',''),
               ]

    otherkeys = [('FILENAME',''), ('HDU',0),
                 ('HEIGHT',0),('WIDTH',0),
                 ]

    allkeys = primkeys + hdrkeys + otherkeys

    vals = dict([(k,[]) for k,d in allkeys])

    for i,fn in enumerate(args):
        print 'Reading', (i+1), 'of', len(args), ':', fn
        F = fitsio.FITS(fn)
        #print F
        #print len(F)
        primhdr = F[0].read_header()
        #print primhdr

        for hdu in range(1, len(F)):
            hdr = F[hdu].read_header()

            info = F[hdu].get_info()
            #'extname': 'S1', 'dims': [4146L, 2160L]
            H,W = info['dims']

            for k,d in primkeys:
                vals[k].append(primhdr.get(k, d))
            for k,d in hdrkeys:
                vals[k].append(hdr.get(k, d))

            vals['FILENAME'].append(fn)
            vals['HDU'].append(hdu)
            vals['WIDTH'].append(int(W))
            vals['HEIGHT'].append(int(H))

    T = fits_table()
    for k,d in allkeys:
        T.set(k.lower().replace('-','_'), np.array(vals[k]))
    #T.about()

    T.filter = np.array([s.split()[0] for s in T.filter])
    T.ra_bore  = np.array([hmsstring2ra (s) for s in T.ra ])
    T.dec_bore = np.array([dmsstring2dec(s) for s in T.dec])

    T.ra  = np.zeros(len(T))
    T.dec = np.zeros(len(T))
    for i in range(len(T)):
        W,H = T.width[i], T.height[i]

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

    T.writeto(opt.outfn)
    print 'Wrote', opt.outfn
                
