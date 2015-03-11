import numpy as np
from astrometry.util.fits import *

'''
dstn wrote make-exposure-list.py to read header information from a set of images.
DJS  wrote code to process Arjun's zeropoints file, making the photometric cuts.

Merge for DR1!
'''

P = fits_table('/global/project/projectdirs/cosmo/work/decam/cats/ZeroPoints/ZeroPoints-DR1.fits')
print 'Read', len(P), 'of DJS zeropoints'
C = fits_table('decals/decals-ccds.fits')
print 'Read', len(C), 'of dstn CCDs'

imap = dict([((int(enum),ccdname.strip()),i) for i,(enum,ccdname) in enumerate(zip(P.expnum,P.ccdname))])
C.I = np.array([imap.get((expnum,ccdname.strip()),-1) for expnum,ccdname in zip(C.expnum, C.extname)])
C.cut(C.I >= 0)
print 'Cut to', len(C), 'matched on expnum,ccdname'

print 'Filenames:', C.cpimage[:10]


fns = P.filename[C.I]
C.dr1 = P.dr1[C.I]

#print 'Filenames:', C.fn[:10]
for fn in fns:
    print fn

C.cpimage = np.array([fn
                 .replace('/global/homes/a/arjundey/cats/CPDESY1_Stripe82/',
                          'decam/CPDES82/')
                 .replace('data/', 'decam/')
                 .replace('cats/', 'decam/')
                 for fn in fns])
C.delete_column('I')
C.writeto('ccds.fits')
