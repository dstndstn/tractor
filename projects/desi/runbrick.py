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
from astrometry.util.miscutils import *
from astrometry.util.resample import *
from astrometry.util.starutil_numpy import *
from astrometry.libkd.spherematch import *

from tractor import *
from tractor.galaxy import *
from tractor.source_extractor import *

tempdir = os.environ['TMPDIR']
calibdir = os.environ.get('DECALS_CALIB', 'calib')
print 'calibdir', calibdir

def create_temp(**kwargs):
    f,fn = tempfile.mkstemp(dir=tempdir, **kwargs)
    os.close(f)
    os.unlink(fn)
    return fn

class DecamImage(object):
    def __init__(self, imgfn, hdu, band):
        self.imgfn = imgfn
        self.hdu   = hdu
        self.band  = band
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

    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

    def makedirs(self):
        for dirnm in [os.path.dirname(fn) for fn in
                      [self.wcsfn, self.corrfn, self.sexfn, self.psffn, self.morphfn]]:
            if not os.path.exists(dirnm):
                try:
                    os.makedirs(dirnm)
                except:
                    pass

    def read_image(self, slice=None, **kwargs):
        if slice is not None:
            f = fitsio.FITS(self.imgfn)[self.hdu]
            img = f[slice]
            rtn = img
            if 'header' in kwargs and kwargs['header']:
                hdr = f.read_header()
                return (img,hdr)
            return rtn
        
        return fitsio.read(self.imgfn, ext=self.hdu, **kwargs)

    def read_image_primary_header(self, **kwargs):
        return fitsio.read_header(self.imgfn)

    def read_image_header(self, **kwargs):
        return fitsio.read_header(self.imgfn, ext=self.hdu)

    def read_dq(self, **kwargs):
        return fitsio.FITS(self.dqfn)[self.hdu].read()

    def read_invvar(self, **kwargs):
        return fitsio.FITS(self.wtfn)[self.hdu].read()


def run_calibs(im, ra, dec, pixscale):
    #print 'wcs', im.wcsfn
    #print 'se', im.sexfn
    #print 'psf', im.psffn
    #print 'morph', im.morphfn

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

        cmd = 'funpack -E %i -O %s %s' % (im.hdu, tmpimgfn, im.imgfn)
        print cmd
        if os.system(cmd):
            sys.exit(-1)

        cmd = 'funpack -E %i -O %s %s' % (im.hdu, tmpmaskfn, im.dqfn)
        print cmd
        if os.system(cmd):
            sys.exit(-1)

    if run_astrom or run_morph:
        # grab header values...
        primhdr = im.read_image_primary_header()
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
    ps = PlotSequence('brick')
    
    B = fits_table('bricks.fits')

    # brick index...
    ii = 377305

    brick = B[ii]

    ra,dec = brick.ra, brick.dec
    W,H = 3600,3600
    pixscale = 0.27 / 3600.

    bands = ['g','r','z']
    catband = 'r'

    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    float(W), float(H))

    T = fits_table('ccds.fits')
    sz = 0.25
    T.cut(np.abs(T.dec - dec) < sz)
    T.cut(degrees_between(T.ra, T.dec, ra, dec) < sz)
    print len(T), 'CCDs nearby'

    ims = []
    for band in bands:
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        print 'filenames,hdus:', zip(TT.filename, TT.hdu)
        for fn,hdu in zip(TT.filename, TT.hdu):
            print
            print 'Image file', fn, 'hdu', hdu
            im = DecamImage(fn, hdu, band)
            ims.append(im)

    for im in ims:
        band = im.band
        run_calibs(im, ra, dec, pixscale)

    catims = [im for im in ims if im.band == catband]
    print 'Reference catalog files:', catims

    cats = []
    extra_cols = []
    for im in catims:
        cat = fits_table(
            im.morphfn,
            hdu=2, #column_map={'ALPHA_J2000':'ra', 'DELTA_J2000':'dec'},
            columns=[x.upper() for x in
                     [#'ALPHA_J2000', 'DELTA_J2000',
                      'x_image', 'y_image',
                      'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
                      'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
                      'disk_theta_world', 'spheroid_reff_world',
                      'spheroid_aspect_world', 'spheroid_theta_world',
                      'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
        print 'Read', len(cat), 'from', im.morphfn
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)
        cats.append(cat)

    plt.clf()
    for cat in cats:
        plt.plot(cat.ra, cat.dec, 'o', mec='none', mfc='b', alpha=0.5)
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    ps.savefig()

    for cat in cats:
        ok,x,y = targetwcs.radec2pixelxy(cat.ra, cat.dec)
        cat.cut((x > 0.5) * (x < (W+0.5)) * (y > 0.5) * (y < (H+0.5)))

    merged = cats[0]
    for cat in cats[1:]:
        I,J,d = match_radec(merged.ra, merged.dec, cat.ra, cat.dec, 0.5/3600.)
        keep = np.ones(len(cat), bool)
        keep[J] = False
        if sum(keep):
            merged = merge_tables([merged, cat[keep]])
    
    plt.clf()
    plt.plot(merged.ra, merged.dec, 'o', mec='none', mfc='b', alpha=0.5)
    plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    ps.savefig()

    del cats

    cat,isrcs = get_se_modelfit_cat(merged, maglim=90, bands=bands)
    print 'Tractor sources:', cat

    T = merged[isrcs]
    T.about()
    # for c in T.get_columns():
    #     plt.clf()
    #     plt.hist(T.get(c), 50)
    #     plt.xlabel(c)
    #     ps.savefig()

    zz = T[T.spheroid_reff_world == 0.]
    print 'Zero spheroid reff:', len(zz)
    zz.writeto('zsph.fits')
    
    zz = T[T.disk_scale_world == 0.]
    print 'Zero disk reff:', len(zz)
    zz.writeto('zdisk.fits')


    tims = []
    for im in ims:
        band = im.band

        wcs = Sip(im.wcsfn)
        print 'Image shape', wcs.imagew, wcs.imageh
        
        imh,imw = wcs.imageh,wcs.imagew
        imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
        ok,tx,ty = wcs.radec2pixelxy(targetrd[:-1,0], targetrd[:-1,1])
        tpoly = zip(tx,ty)

        ## FIXME -- it seems I got lucky and the cross product is negative -- clockwise
        ## One could check this and reverse the polygon vertex order.
        # dx0,dy0 = tx[1]-tx[0], ty[1]-ty[0]
        # dx1,dy1 = tx[2]-tx[1], ty[2]-ty[1]
        # cross = dx0*dy1 - dx1*dy0
        # print 'Cross:', cross

        plt.clf()
        imp = np.array(imgpoly)
        #print 'imp', imp
        ii = np.array([0,1,2,3,0])
        imp = imp[ii,:]
        plt.plot(imp[:,0], imp[:,1], 'b-')
        plt.plot(tx[ii], ty[ii], 'r-')
        clip = clip_polygon(imgpoly, tpoly)
        #print 'Clip:', clip
        clip = np.array(clip)
        plt.plot(clip[ii,0], clip[ii,1], 'g-')
        ps.savefig()

        x0,y0 = np.floor(clip.min(axis=0)).astype(int)
        x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
        print 'Image range:', x0,x1, y0,y1

        slc = slice(y0,y1+1), slice(x0,x1+1)

        img,imghdr = im.read_image(header=True)
        #dq = im.read_dq()
        invvar = im.read_invvar()

        psf_fwhm = imghdr['FWHM']

        primhdr = im.read_image_primary_header()
        magzp  = primhdr['MAGZERO']
        zpscale = NanoMaggies.zeropointToScale(magzp)
        print 'Magzp', magzp
        print 'zpscale', zpscale

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
        sig1 = 1./np.sqrt(np.median(invvar))

        # plt.clf()
        # plt.imshow(invvar, interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.title('weight map: ' + im.name)
        # ps.savefig()
        # 
        # plt.clf()
        # plt.hist(invvar.ravel(), 100)
        # plt.xlabel('invvar')
        # ps.savefig()

        # The weight maps have values < 0
        invvar = np.maximum(invvar, 0.)

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=ConstantSky(0.), name=im.name + ' ' + band)
        tim.zr = [-3. * sig1, 10. * sig1]
        tims.append(tim)

        # HACK
        tim.psf = NCircularGaussianPSF([psf_fwhm / 2.35],[1.])

        tractor = Tractor([tim], cat)

        mn,mx = tim.zr
        ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                   vmin=mn, vmax=mx)
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(tim.getImage(), **ima)
        mod = tractor.getModelImage(tim)
        plt.subplot(1,2,2)
        plt.imshow(mod, **ima)
        ps.savefig()

    print 'Tims:', tims

    # tractor = Tractor(tims, cat)
    # for i,tim in enumerate(tims):
    #     plt.clf()
    #     mn,mx = tim.zr
    #     ima = dict(interpolation='nearest', origin='lower', cmap='gray',
    #                vmin=mn, vmax=mx)
    # 
    #     plt.subplot(1,2,1)
    #     plt.imshow(tim.getImage(), **ima)
    #     mod = tractor.getModelImage(i)
    #     plt.subplot(1,2,2)
    #     plt.imshow(mod, **ima)
    #     ps.savefig()
        


if __name__ == '__main__':
    main()

