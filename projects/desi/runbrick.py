import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import fitsio

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.resample import *
from astrometry.util.starutil_numpy import *

if __name__ == '__main__':

    B = fits_table('bricks.fits')

    # brick index...
    ii = 377305

    brick = B[ii]

    ra,dec = brick.ra, brick.dec
    W,H = 3600,3600
    pixscale = 0.27 / 3600.

    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    float(W), float(H))

    T = fits_table('ccds.fits')
    sz = 0.25
    T.cut(np.abs(T.dec - dec) < sz)
    T.cut(degrees_between(T.ra, T.dec, ra, dec) < sz)
    print len(T), 'CCDs nearby'

    for band in 'grz':
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        print 'filenames,hdus:', zip(TT.filename, TT.hdu)

        for fn,hdu in zip(TT.filename, TT.hdu):
            print
            imgfn = fn
            print 'Image file', fn, 'hdu', hdu
            #img = fitsio.FITS(fn)[hdu].read()
            img,imghdr = fitsio.read(fn, ext=hdu, header=True)

            sky = imghdr['SKYBRITE']
            print 'SKYBRITE:', sky
            medsky = np.median(img)
            print 'Image median:', medsky
            #img -= sky
            img -= medsky
            print 'Image median:', np.median(img)

            dqfn = fn.replace('_ooi_', '_ood_')
            print 'DQ', dqfn

            wcsfn = imgfn
            wcsfn = wcsfn.replace('/project/projectdirs/cosmo/staging/decam',
                                  'calib/astrom')
            wcsfn = wcsfn.replace('.fits.fz', '.ext%02i.wcs' % hdu)
            corrfn = wcsfn.replace('.wcs', '.corr')

            sexfn = imgfn
            sexfn = wcsfn.replace('calib/astrom', 'calib/sextractor').replace('.wcs', '.cat')

            psffn = wcsfn.replace('calib/astrom', 'calib/psf').replace('.wcs','.psf')

            morphfn = wcsfn.replace('calib/astrom', 'calib/morph').replace('.wcs','.fits')

            for dirnm in [os.path.dirname(fn) for fn in [wcsfn,corrfn,sexfn,psffn,morphfn]]:
                if not os.path.exists(dirnm):
                    try:
                        os.makedirs(dirnm)
                    except:
                        pass

            print 'WCS filename', wcsfn
            print 'Corr', corrfn
            print 'SExtractor', sexfn
            print 'PSFEx', psffn
            print 'Morph', morphfn

            run_funpack = False
            run_astrom = False
            run_psfex = False
            run_morph = False

            if not all([os.path.exists(fn) for fn in [sexfn,morphfn,wcsfn]]):
                # we run the first sextractor via solve-field
                run_funpack = True
            if not all([os.path.exists(fn) for fn in [wcsfn,sexfn]]):
                run_astrom = True
            if not os.path.exists(psffn):
                run_psfex = True
            if not os.path.exists(morphfn):
                run_morph = True
            
            if run_funpack:
                #
                cmd = 'rm -f 1.fits flags.fits'
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

                tmpimgfn = '1.fits'
                #cmd = 'funpack -E 0,%i -O %s %s' % (hdu, tmpimgfn, imgfn)
                cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, imgfn)
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

                tmpmaskfn = 'flags.fits'
                cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, dqfn)
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

            if run_astrom or run_morph:
                # grab header values...
                #primhdr = fitsio.read_header(tmpimgfn)
                #hdr = fitsio.read_header(tmpimgfn, ext=1)
                primhdr = fitsio.read_header(imgfn)
                hdr = fitsio.read_header(tmpimgfn)

                magzp = primhdr['MAGZERO']
                seeing = pixscale * 3600 * hdr['FWHM']

            if run_astrom:
                #
                sexcmd = 'sex -FLAG_IMAGE %s -SEEING_FWHM %f -MAG_ZEROPOINT %f' % (tmpmaskfn, seeing, magzp)
                cmd = ' '.join([
                    'solve-field --config ~/desi-dstn/sdss-astrometry-index/r2/cfg',
                    '-D . --temp-dir tmp',
                    '--ra 244 --dec 8 --radius 1 -L 0.25 -H 0.29 -u app',
                    '--continue --no-plots',
                    '--sextractor-config /project/projectdirs/desi/imaging/code/cats/DECaLS-v2.sex',
                    '--sextractor-path "%s"' % sexcmd,
                    '-X x_image -Y y_image -s flux_auto',
                    '--crpix-center',
                    '-N none -U none -S none -M none -R none',
                    '--keep-xylist', sexfn, '--temp-axy',
                    '--corr', corrfn, '--tag-all',
                    '--wcs', wcsfn,
                    '--no-remove-lines --uniformize 0 --no-fits2fits',
                    tmpimgfn])
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

            if run_psfex:
                cmd = 'psfex -c ~/desi/imaging/code/cats/DECaLS-v2.psfex -PSF_DIR %s %s' % (os.path.dirname(psffn), sexfn)
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

            if run_morph:
                cmd = ' '.join(['sex -c ~/desi/imaging/code/cats/CS82_MF.sex',
                                '-FLAG_IMAGE', tmpmaskfn,
                                '-SEEING_FWHM %f' % seeing,
                                '-MAG_ZEROPOINT %f' % magzp,
                                '-PSF_NAME', psffn,
                                '-CATALOG_NAME', morphfn,
                                tmpimgfn])
                print cmd
                if os.system(cmd):
                    sys.exit(-1)



            wcs = Sip(wcsfn)

            dq = fitsio.FITS(dqfn)[hdu].read()
            
            wtfn = imgfn.replace('_ooi_', '_oow_')
            print 'Weight', wtfn
            wt = fitsio.FITS(wtfn)[hdu].read()

