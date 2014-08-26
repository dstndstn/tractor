import os
import tempfile

import fitsio


tempdir = os.environ['TMPDIR']
decals_dir = os.environ.get('DECALS_DIR')
calibdir = os.path.join(decals_dir, 'calib', 'decam')
sedir    = os.path.join(decals_dir, 'calib', 'se-config')
an_config= os.path.join(decals_dir, 'calib', 'an-config', 'cfg')


def create_temp(**kwargs):
    f,fn = tempfile.mkstemp(dir=tempdir, **kwargs)
    os.close(f)
    os.unlink(fn)
    return fn

class DecamImage(object):
    def __init__(self, imgfn, hdu, band, expnum, extname, calname, exptime):
        self.imgfn = os.path.join(decals_dir, 'images', 'decam', imgfn)
        self.hdu   = hdu
        self.expnum = expnum
        self.extname = extname
        self.band  = band
        self.exptime = exptime
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        ibase = os.path.basename(imgfn)
        ibase = ibase.replace('.fits.fz', '')
        idirname = os.path.basename(os.path.dirname(imgfn))
        #self.name = dirname + '/' + base + ' + %02i' % hdu
        print 'dir,base', idirname, ibase
        #print 'calibdir', calibdir

        self.calname = calname
        self.name = '%08i-%s' % (expnum, extname)
        print 'Calname', calname
        
        extnm = '.ext%02i' % hdu
        self.wcsfn = os.path.join(calibdir, 'astrom', calname + '.wcs.fits')
        self.corrfn = self.wcsfn.replace('.wcs.fits', '.corr.fits')
        self.sdssfn = self.wcsfn.replace('.wcs.fits', '.sdss.fits')
        self.sexfn = os.path.join(calibdir, 'sextractor', calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', calname + '.fits')
        self.morphfn = os.path.join(calibdir, 'morph', calname + '.fits')

    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

    def makedirs(self):
        for dirnm in [os.path.dirname(fn) for fn in
                      [self.wcsfn, self.corrfn, self.sdssfn, self.sexfn, self.psffn, self.morphfn]]:
            if not os.path.exists(dirnm):
                try:
                    os.makedirs(dirnm)
                except:
                    pass

    def _read_fits(self, fn, hdu, slice=None, header=None, **kwargs):
        if slice is not None:
            f = fitsio.FITS(fn)[hdu]
            img = f[slice]
            rtn = img
            if header:
                hdr = f.read_header()
                return (img,hdr)
            return img
        return fitsio.read(fn, ext=hdu, header=header, **kwargs)

    def read_image(self, **kwargs):
        return self._read_fits(self.imgfn, self.hdu, **kwargs)

    def get_image_info(self, **kwargs):
        return fitsio.FITS(self.imgfn)[self.hdu].get_info()
    
    def read_image_primary_header(self, **kwargs):
        return fitsio.read_header(self.imgfn)

    def read_image_header(self, **kwargs):
        return fitsio.read_header(self.imgfn, ext=self.hdu)

    def read_dq(self, **kwargs):
        return self._read_fits(self.dqfn, self.hdu, **kwargs)
    #return fitsio.FITS(self.dqfn)[self.hdu].read()

    def read_invvar(self, **kwargs):
        return self._read_fits(self.wtfn, self.hdu, **kwargs)
    #return fitsio.FITS(self.wtfn)[self.hdu].read()


def bounce_run_calibs(X):
    return run_calibs(*X)

def run_calibs(im, ra, dec, pixscale):
    for fn in [im.wcsfn,im.sexfn,im.psffn,im.morphfn,im.corrfn,im.sdssfn]:
        print 'exists?', os.path.exists(fn), fn
        
    im.makedirs()

    run_funpack = False
    run_se = False
    run_astrom = False
    run_psfex = False
    run_morph = False

    if not all([os.path.exists(fn) for fn in [im.sexfn]]):
        run_se = True
        run_funpack = True
    if not all([os.path.exists(fn) for fn in [im.wcsfn,im.corrfn,im.sdssfn]]):
        run_astrom = True
    if not os.path.exists(im.psffn):
        run_psfex = True
    if not os.path.exists(im.morphfn):
        run_morph = True
        run_funpack = True
    
    if run_funpack:
        tmpimgfn  = create_temp(suffix='.fits')
        tmpmaskfn = create_temp(suffix='.fits')

        cmd = 'funpack -E %i -O %s %s' % (im.hdu, tmpimgfn, im.imgfn)
        print cmd
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)

        cmd = 'funpack -E %i -O %s %s' % (im.hdu, tmpmaskfn, im.dqfn)
        print cmd
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)

    if run_astrom or run_morph:
        # grab header values...
        primhdr = im.read_image_primary_header()
        hdr     = im.read_image_header()

        magzp  = primhdr['MAGZERO']
        seeing = pixscale * 3600 * hdr['FWHM']

    if run_se:
        cmd = ' '.join([
            'sex',
            '-c', os.path.join(sedir, 'DECaLS-v2.sex'),
            '-FLAG_IMAGE', tmpmaskfn, '-SEEING_FWHM %f' % seeing,
            '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', im.sexfn,
            tmpimgfn])
        print cmd
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)

    if run_astrom:
        cmd = ' '.join([
            'solve-field --config', an_config, '-D . --temp-dir', tempdir,
            '--ra %f --dec %f' % (ra,dec), '--radius 1 -L 0.25 -H 0.29 -u app',
            '--continue --no-plots --no-remove-lines --uniformize 0',
            '--no-fits2fits',
            '-X x_image -Y y_image -s flux_auto --extension 2',
            '--width 2048 --height 4096',
            '--crpix-center',
            '-N none -U none -S none -M none --rdls', im.sdssfn,
            '--corr', im.corrfn, '--wcs', im.wcsfn, 
            '--temp-axy', '--tag-all', im.sexfn])
        print cmd
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)

    if run_psfex:
        cmd = ('psfex -c %s -PSF_DIR %s %s' %
               (os.path.join(sedir, 'DECaLS-v2.psfex'),
                os.path.dirname(im.psffn), im.sexfn))
        print cmd
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)

    if run_morph:
        cmd = ' '.join(['sex -c', os.path.join(sedir, 'CS82_MF.sex'),
                        '-FLAG_IMAGE', tmpmaskfn,
                        '-SEEING_FWHM %f' % seeing,
                        '-MAG_ZEROPOINT %f' % magzp,
                        '-PSF_NAME', im.psffn,
                        '-CATALOG_NAME', im.morphfn,
                        tmpimgfn])
        print cmd
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)


