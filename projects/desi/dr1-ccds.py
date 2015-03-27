import numpy as np
from astrometry.util.fits import *

'''
dstn wrote make-exposure-list.py to read header information from a set of images.
DJS  wrote code to process Arjun's zeropoints file, making the photometric cuts.

Merge for DR1!
'''

P = fits_table('/global/project/projectdirs/cosmo/work/decam/cats/ZeroPoints/ZeroPoints-DR1.fits')
print 'Read', len(P), 'of DJS zeropoints'
C = fits_table('decals/decals-ccds-dr1-dstn.fits')
print 'Read', len(C), 'of dstn CCDs'

imap = dict([((int(enum),ccdname.strip()),i) for i,(enum,ccdname) in enumerate(zip(P.expnum,P.ccdname))])
C.I = np.array([imap.get((expnum,ccdname.strip()),-1) for expnum,ccdname in zip(C.expnum, C.extname)])
C.cut(C.I >= 0)
print 'Cut to', len(C), 'matched on expnum,ccdname'

print 'Filenames:', C.cpimage[:10]

fns = P.filename[C.I]
C.dr1 = P.dr1[C.I]

#print 'Filenames:', C.fn[:10]
#for fn in fns:
#    print fn

oldcpimage = C.cpimage

C.cpimage = np.array([fn
                      .replace('/global/homes/a/arjundey/cats/CPDESY1_Stripe82/',
                               'decam/CPDES82/')
                      .replace('data/', 'decam/')
                      .replace('cats/', 'decam/')
                      .replace('.fits', '.fits.fz')
                      for fn in fns])

#### NOTE, the cpimage_hdu column is BOGUS (for DES, because of CP vs DES)
import fitsio
fits = {}
hdus = {}
dirnm = os.path.join(os.environ['DECALS_DIR'], 'images')
for i,fn in enumerate(C.cpimage):

    print
    print 'CCD', (i+1), 'of', len(C)
    print oldcpimage[i]
    old = oldcpimage[i].replace('decals/images/', '').strip()
    fn = fn.strip()
    if (not old.endswith('.fz')) and fn.endswith('.fz'):
        old = old + '.fz'
    print fn
    print old
    if fn == old:
        continue

    if fn in fits:
        F = fits[fn]
    else:
        pth = os.path.join(dirnm, fn.strip())
        if not os.path.exists(pth):
            p2 = pth.replace('.fits.fz', '.fits')
            if os.path.exists(p2):
                pth = p2
                #print 'Using', p2, 'instead of', pth
        F = fitsio.FITS(pth)
        fits[fn] = F

    if len(fits) > 100:
        fits = {}

    ff = F[C.cpimage_hdu[i]].read_header()
    ee = ff['EXTNAME'].strip()
    extname = C.extname[i].strip()
    if ee != extname:
        print 'HDU wrong for', fn
        if not (fn,extname) in hdus:
            for hdu in range(len(F)):
                hdr = F[hdu].read_header()
                if not 'EXTNAME' in hdr:
                    continue
                ext = hdr['EXTNAME'].strip()
                hdus[(fn,ext)] = hdu
        hdu = hdus[(fn,extname)]
        print 'Found hdu', hdu, 'for extname', extname
        C.cpimage_hdu[i] = hdu
    else:
        print 'ok:', ee, C.extname[i].strip(), fn

C.delete_column('I')
C.writeto('ccds.fits')
print 'Finished.'
