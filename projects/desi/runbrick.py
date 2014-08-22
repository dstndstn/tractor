import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import tempfile
import os

import fitsio

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.resample import *
from astrometry.util.starutil_numpy import *

from tractor import *
from tractor.galaxy import *

tempdir = os.environ['TMPDIR']
calibdir = os.environ.get('DECALS_CALIB', 'calib')
print 'calibdir', calibdir

def create_temp(**kwargs):
    f,fn = tempfile.mkstemp(dir=tempdir, **kwargs)
    os.close(f)
    return fn

class DecamImage(object):
    def __init__(self, imgfn, hdu):
        self.imgfn = imgfn
        self.hdu   = hdu
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        base = os.path.basename(imgfn)
        base = base.replace('.fits.fz', '')
        dirname = os.path.basename(os.path.dirname(imgfn))
        self.name = base + '/' + dirname

        print 'dir,base', dirname, base
        print 'calibdir', calibdir
        
        extnm = '.ext%02i' % hdu
        self.wcsfn = os.path.join(calibdir, 'astrom', dirname, base + extnm + '.wcs')
        self.corrfn = self.wcsfn.replace('.wcs', '.corr')
        self.sexfn = os.path.join(calibdir, 'sextractor', dirname, base + extnm + '.cat')
        self.psffn = os.path.join(calibdir, 'psf', dirname, base + extnm + '.psf')
        self.morphfn = os.path.join(calibdir, 'morph', dirname, base + extnm + '.fits')

    def makedirs(self):
        for dirnm in [os.path.dirname(fn) for fn in
                      [self.wcsfn, self.corrfn, self.sexfn, self.psffn, self.morphfn]]:
            if not os.path.exists(dirnm):
                try:
                    os.makedirs(dirnm)
                except:
                    pass

    def read_image(self, **kwargs):
        return fitsio.read(self.imgfn, ext=self.hdu, **kwargs)

    def read_image_primary_header(self, **kwargs):
        return fitsio.read_header(self.imgfn)

    def read_image_header(self, **kwargs):
        return fitsio.read_header(self.imgfn, ext=self.hdu)

    def read_dq(self, **kwargs):
        return fitsio.FITS(self.dqfn)[self.hdu].read()

    def read_invvar(self, **kwargs):
        return fitsio.FITS(self.wtfn)[self.hdu].read()


def run_calibs(im):
    print 'wcs', im.wcsfn
    print 'se', im.sexfn
    print 'psf', im.psffn
    print 'morph', im.morphfn

    im.makedirs()

    run_funpack = False
    run_se = False
    run_astrom = False
    run_psfex = False
    run_morph = False

    if not all([os.path.exists(fn) for fn in [im.sexfn, im.psffn, im.wcsfn]]):
        run_se = True
    if not all([os.path.exists(fn) for fn in [im.sexfn, im.morphfn]]):
        run_funpack = True
    if not all([os.path.exists(fn) for fn in [im.wcsfn,im.corrfn]]):
        run_astrom = True
    if not os.path.exists(im.psffn):
        run_psfex = True
    if not os.path.exists(im.morphfn):
        run_morph = True
    
    if run_funpack:
        tmpimgfn  = create_temp(suffix='.fits')
        tmpmaskfn = create_temp(suffix='.fits')

        cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, im.imgfn)
        print cmd
        if os.system(cmd):
            sys.exit(-1)

        cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, im.dqfn)
        print cmd
        if os.system(cmd):
            sys.exit(-1)

    if run_astrom or run_morph:
        # grab header values...
        primhdr = im.read_image_prim_header()
        hdr     = im.read_image_header()

        magzp  = primhdr['MAGZERO']
        seeing = pixscale * 3600 * hdr['FWHM']

    if run_se:
        cmd = ' '.join([
            'sex',
            '-c', '/project/projectdirs/desi/imaging/code/cats/DECaLS-v2.sex',
            '-FLAG_IMAGE', tmpmaskfn, '-SEEING_FWHM %f' % seeing,
            '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', im.sexfn,
            tmpimgfn])
        print cmd
        if os.system(cmd):
            sys.exit(-1)

    if run_astrom:
        cmd = ' '.join([
            'solve-field --config ~/desi-dstn/sdss-astrometry-index/r2/cfg',
            '-D . --temp-dir tmp',
            '--ra %f --dec %f' % (ra,dec), '--radius 1 -L 0.25 -H 0.29 -u app',
            '--continue --no-plots --no-remove-lines --uniformize 0 --no-fits2fits',
            '-X x_image -Y y_image -s flux_auto --extension 2',
            '--width 2048 --height 4096',
            '--crpix-center',
            '-N none -U none -S none -M none -R none',
            '--temp-axy', '--corr', im.corrfn, '--tag-all',
            '--wcs', im.wcsfn, im.sexfn])
        print cmd
        if os.system(cmd):
            sys.exit(-1)

    if run_psfex:
        cmd = 'psfex -c ~/desi/imaging/code/cats/DECaLS-v2.psfex -PSF_DIR %s %s' % (os.path.dirname(im.psffn), im.sexfn)
        print cmd
        if os.system(cmd):
            sys.exit(-1)

    if run_morph:
        cmd = ' '.join(['sex -c ~/desi/imaging/code/cats/CS82_MF.sex',
                        '-FLAG_IMAGE', tmpmaskfn,
                        '-SEEING_FWHM %f' % seeing,
                        '-MAG_ZEROPOINT %f' % magzp,
                        '-PSF_NAME', im.psffn,
                        '-CATALOG_NAME', im.morphfn,
                        tmpimgfn])
        print cmd
        if os.system(cmd):
            sys.exit(-1)


def main():
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

    tims = []

    for band in 'grz':
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        print 'filenames,hdus:', zip(TT.filename, TT.hdu)

        for fn,hdu in zip(TT.filename, TT.hdu):
            print
            print 'Image file', fn, 'hdu', hdu

            im = DecamImage(fn, hdu)

            run_calibs(im, ra, dec)

            img,imghdr = im.read_image(header=True)
            #dq = im.read_dq()
            invvar = im.read_invvar()
            wcs = Sip(im.wcsfn)

            primhdr = im.read_image_primary_header()
            magzp  = primhdr['MAGZERO']
            zpscale = NanoMaggies.zeropointToScale(magzp)

            sky = imghdr['SKYBRITE']
            print 'SKYBRITE:', sky
            medsky = np.median(img)
            print 'Image median:', medsky
            img -= medsky

            twcs = ConstantFitsWcs(wcs)
            #if x0 or y0:
            #    twcs.setX0Y0(x0,y0)

            imh,imw = img.shape
            psf = PsfEx(im.psffn, imw, imh, scale=False, nx=9, ny=17)

            # Scale images to Nanomaggies
            img /= zpscale
            invvar *= zpscale**2
            orig_zpscale = zpscale
            zpscale = 1.

            tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                        photocal=LinearPhotoCal(zpscale, band=band),
                        sky=ConstantSky(0.), name=im.name + ' ' + band)
            tims.append(tim)

    print 'Tims:', tims


if __name__ == '__main__':
    main()

