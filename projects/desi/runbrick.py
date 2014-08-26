import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import tempfile
import os

import fitsio

from scipy.ndimage.filters import *
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing

from astrometry.util.fits import *
from astrometry.util.file import *
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

#calibdir = os.environ.get('DECALS_CALIB', 'calib')
#imgdir   = os.environ.get('DECALS_IMG', None)
#sedir    = os.environ.get('DECALS_SE',
#                          '/project/projectdirs/desi/imaging/code/cats')

decals_dir = os.environ.get('DECALS_DIR')

calibdir = os.path.join(decals_dir, 'calib', 'decam')
sedir    = os.path.join(decals_dir, 'calib', 'se-config')
an_config= os.path.join(decals_dir, 'calib', 'an-config', 'cfg')

print 'calibdir', calibdir
#

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
        self.corrfn = self.wcsfn.replace('.wcs', '.corr.fits')
        self.sdssfn = self.wcsfn.replace('.wcs', '.sdss.fits')
        self.sexfn = os.path.join(calibdir, 'sextractor', calname + '.cat.fits')
        # PsfEx hard-codes the .psf suffix
        self.psffn = os.path.join(calibdir, 'psfex', calname + '.psf')
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


def run_calibs(im, ra, dec, pixscale):
    #print 'wcs', im.wcsfn
    #print 'se', im.sexfn
    #print 'psf', im.psffn
    #print 'morph', im.morphfn

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
            '-c', os.path.join(sedir, 'DECaLS-v2.sex'),
            '-FLAG_IMAGE', tmpmaskfn, '-SEEING_FWHM %f' % seeing,
            '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', im.sexfn,
            tmpimgfn])
        print cmd
        if os.system(cmd):
            sys.exit(-1)

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
            sys.exit(-1)

    if run_psfex:
        cmd = ('psfex -c %s -PSF_DIR %s %s' %
               (os.path.join(sedir, 'DECaLS-v2.psfex'),
                os.path.dirname(im.psffn), im.sexfn))
        print cmd
        if os.system(cmd):
            sys.exit(-1)

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
            sys.exit(-1)


def main():
    ps = PlotSequence('brick')
    plt.figure(figsize=(12,9));
    #plt.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.95,
    #                    hspace=0.05, wspace=0.05)

    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        #hspace=0.05, wspace=0.05)
                        hspace=0.2, wspace=0.05)
    
    B = fits_table(os.path.join(decals_dir, 'decals-bricks.fits'))

    # brick index...
    ii = 377305

    brick = B[ii]

    ra,dec = brick.ra, brick.dec
    #W,H = 3600,3600
    W,H = 400,400
    pixscale = 0.27 / 3600.

    bands = ['g','r','z']
    catband = 'r'

    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    float(W), float(H))

    ccdsfn = os.path.join(decals_dir, 'decals-ccds.fits')
    T = fits_table(ccdsfn)
    sz = 0.25
    T.cut(np.abs(T.dec - dec) < sz)
    T.cut(degrees_between(T.ra, T.dec, ra, dec) < sz)
    print len(T), 'CCDs nearby'

    ims = []
    for band in bands:
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        for t in TT:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(t.cpimage, t.cpimage_hdu, band, t.expnum, t.extname.strip(),
                            t.calname.strip(), t.exptime)
            ims.append(im)
            
    for im in ims:
        run_calibs(im, ra, dec, pixscale)

    zpfn = os.path.join(calibdir, 'photom', 'zeropoints.fits')
    print 'Reading zeropoints:', zpfn
    ZP = fits_table(zpfn)

    # Check photometric calibrations
    lastband = None

    for im in ims:
        band = im.band
        cat = fits_table(im.morphfn, hdu=2, columns=[
            'mag_psf','x_image', 'y_image', 'mag_disk', 'mag_spheroid', 'flags',
            'flux_psf' ])
        print 'Read', len(cat), 'from', im.morphfn
        cat.cut(cat.flags == 0)
        print '  Cut to', len(cat), 'with no flags set'
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)

        sdss = fits_table(im.sdssfn)


        I = np.flatnonzero(ZP.expnum == im.expnum)
        if len(I) > 1:
            I = np.flatnonzero((ZP.expnum == im.expnum) * (ZP.extname == im.extname))
        assert(len(I) == 1)
        I = I[0]
        magzp = ZP.zpt[I]
        print 'magzp', magzp
        exptime = ZP.exptime[I]
        magzp += 2.5 * np.log10(exptime)
        print 'magzp', magzp

        primhdr = im.read_image_primary_header()
        magzp0  = primhdr['MAGZERO']
        print 'header magzp:', magzp0

        I,J,d = match_radec(cat.ra, cat.dec, sdss.ra, sdss.dec, 1./3600.)

        flux = sdss.get('%s_psfflux' % band)
        mag = NanoMaggies.nanomaggiesToMag(flux)

        # plt.clf()
        # plt.plot(mag[J], cat.mag_psf[I] - mag[J], 'b.')
        # plt.xlabel('SDSS %s psf mag' % band)
        # plt.ylabel('SDSS - DECam mag')
        # plt.title(im.name)
        # plt.axhline(0, color='k', alpha=0.5)
        # plt.ylim(-2,2)
        # plt.xlim(15, 23)
        # ps.savefig()

        if band != lastband:
            if lastband is not None:
                ps.savefig()
            off = 0
            plt.clf()

        plt.subplot(2,4, off+1)
        mag2 = -2.5 * np.log10(cat.flux_psf)
        p = plt.plot(mag[J], mag[J] - mag2[I], 'b.')
        plt.xlabel('SDSS %s psf mag' % band)
        if off in [0,4]:
            plt.ylabel('SDSS - DECam instrumental mag')
        plt.title(im.name)

        med = np.median(mag[J] - mag2[I])
        plt.axhline(med, color='k', alpha=0.25)

        plt.ylim(29,32)
        plt.xlim(15, 22)
        plt.axhline(magzp, color='r', alpha=0.5)
        plt.axhline(magzp0, color='b', alpha=0.5)

        off += 1
        lastband = band
    ps.savefig()
        
        
    # FIXME -- we're only reading 'catband'-band catalogs, and all the fluxes
    # are initialized at that band's flux... should really read all bands!
        
    # Select SE catalogs to read
    catims = [im for im in ims if im.band == catband]
    print 'Reference catalog files:', catims
    # ... and read 'em
    cats = []
    extra_cols = []
    for im in catims:
        cat = fits_table(
            im.morphfn, hdu=2,
            columns=[x.upper() for x in
                     ['x_image', 'y_image', 'flags',
                      'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
                      'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
                      'disk_theta_world', 'spheroid_reff_world',
                      'spheroid_aspect_world', 'spheroid_theta_world',
                      'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
        print 'Read', len(cat), 'from', im.morphfn
        cat.cut(cat.flags == 0)
        print '  Cut to', len(cat), 'with no flags set'
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)
        cats.append(cat)

        
        
    # Plot all catalog sources and ROI
    # plt.clf()
    # for cat in cats:
    #     plt.plot(cat.ra, cat.dec, 'o', mec='none', mfc='b', alpha=0.5)
    # plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    # ps.savefig()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    # Cut catalogs to ROI
    for cat in cats:
        ok,x,y = targetwcs.radec2pixelxy(cat.ra, cat.dec)
        cat.cut((x > 0.5) * (x < (W+0.5)) * (y > 0.5) * (y < (H+0.5)))

    # Merge catalogs by keeping sources > 0.5" away from previous ones
    merged = cats[0]
    for cat in cats[1:]:
        if len(merged) == 0:
            merged = cat
            continue
        if len(cat) == 0:
            continue
        I,J,d = match_radec(merged.ra, merged.dec, cat.ra, cat.dec, 0.5/3600.)
        keep = np.ones(len(cat), bool)
        keep[J] = False
        if sum(keep):
            merged = merge_tables([merged, cat[keep]])
    
    # plt.clf()
    # plt.plot(merged.ra, merged.dec, 'o', mec='none', mfc='b', alpha=0.5)
    # plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    # ps.savefig()

    del cats
    # Create Tractor sources
    cat,isrcs = get_se_modelfit_cat(merged, maglim=90, bands=bands)
    print 'Tractor sources:', cat
    T = merged[isrcs]

    # record coordinates in target brick image
    ok,T.tx,T.ty = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.tx -= 1
    T.ty -= 1
    T.itx = np.clip(np.round(T.tx).astype(int), 0, W-1)
    T.ity = np.clip(np.round(T.ty).astype(int), 0, H-1)

    nstars = sum([1 for src in cat if isinstance(src, PointSource)])
    print 'Number of point sources:', nstars

    #T.about()
    # for c in T.get_columns():
    #     plt.clf()
    #     plt.hist(T.get(c), 50)
    #     plt.xlabel(c)
    #     ps.savefig()

    zpfn = os.path.join(calibdir, 'photom', 'zeropoints.fits')
    print 'Reading zeropoints:', zpfn
    ZP = fits_table(zpfn)

    # Read images, clip to ROI
    tims = []
    for im in ims:
        band = im.band
        wcs = Sip(im.wcsfn)
        #print 'Image shape', wcs.imagew, wcs.imageh
        imh,imw = wcs.imageh,wcs.imagew
        imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
        ok,tx,ty = wcs.radec2pixelxy(targetrd[:-1,0], targetrd[:-1,1])
        tpoly = zip(tx,ty)
        clip = clip_polygon(imgpoly, tpoly)
        clip = np.array(clip)
        print 'Clip', clip
        if len(clip) == 0:
            continue
        x0,y0 = np.floor(clip.min(axis=0)).astype(int)
        x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
        slc = slice(y0,y1+1), slice(x0,x1+1)

        ## FIXME -- it seems I got lucky and the cross product is
        ## negative -- clockwise One could check this and reverse the
        ## polygon vertex order.
        # dx0,dy0 = tx[1]-tx[0], ty[1]-ty[0]
        # dx1,dy1 = tx[2]-tx[1], ty[2]-ty[1]
        # cross = dx0*dy1 - dx1*dy0
        # print 'Cross:', cross

        img,imghdr = im.read_image(header=True, slice=slc)
        invvar = im.read_invvar(slice=slc)
        #print 'Image ', img.shape

        # header 'FWHM' is in pixels
        psf_fwhm = imghdr['FWHM']
        primhdr = im.read_image_primary_header()

        I = np.flatnonzero(ZP.expnum == im.expnum)
        if len(I) > 1:
            I = np.flatnonzero((ZP.expnum == im.expnum) * (ZP.extname == im.extname))
        assert(len(I) == 1)
        I = I[0]
        magzp = ZP.zpt[I]
        print 'magzp', magzp
        exptime = ZP.exptime[I]
        magzp += 2.5 * np.log10(exptime)
        print 'magzp', magzp

        magzp0  = primhdr['MAGZERO']
        print 'header magzp:', magzp0

        zpscale = NanoMaggies.zeropointToScale(magzp)
        print 'zpscale', zpscale

        #sky = imghdr['SKYBRITE']
        medsky = np.median(img)
        #print 'SKYBRITE:', sky
        #print 'Image median:', medsky
        img -= medsky

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        # get full image size for PsfEx
        info = im.get_image_info()
        print 'Image info:', info
        fullh,fullw = info['dims']
        psfex = PsfEx(im.psffn, fullw, fullh, scale=False, nx=9, ny=17)
        #psfex = ShiftedPsf(psfex, x0, y0)
        # HACK!!
        psf_sigma = psf_fwhm / 2.35
        psf = NCircularGaussianPSF([psf_sigma],[1.])

        # Scale images to Nanomaggies
        img /= zpscale
        invvar *= zpscale**2
        orig_zpscale = zpscale
        zpscale = 1.
        sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))

        # Clamp near-zero (incl negative!) invvars to zero
        thresh = 0.2 * (1./sig1**2)
        invvar[invvar < thresh] = 0

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=ConstantSky(0.), name=im.name + ' ' + band)
        tim.zr = [-3. * sig1, 10. * sig1]
        tim.sig1 = sig1
        tim.band = band
        tim.psf_fwhm = psf_fwhm
        tim.psf_sigma = psf_sigma
        tim.sip_wcs = wcs
        tim.x0,tim.y0 = int(x0),int(y0)
        tim.psfex = psfex
        mn,mx = tim.zr
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        tims.append(tim)

        # tractor = Tractor([tim], cat)
        # plt.clf()
        # plt.subplot(1,2,1)
        # plt.imshow(tim.getImage(), **ima)
        # mod = tractor.getModelImage(tim)
        # plt.subplot(1,2,2)
        # plt.imshow(mod, **ima)
        # plt.suptitle(tim.name)
        # ps.savefig()
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

        # plt.clf()
        # plt.imshow(tim.getImage(), **ima)
        # plt.suptitle(tim.name)
        # ps.savefig()

    # save resampling params
    for tim in tims:
        wcs = tim.sip_wcs
        x0,y0 = int(tim.x0),int(tim.y0)
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(x0, y0, subw, subh)
        tim.subwcs = subwcs
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue
        tim.resamp = (Yo,Xo,Yi,Xi)

    # Produce per-band coadds and an RGB image, for plots
    rgbim = np.zeros((H,W,3))
    coimgs = []
    coimas = []
    for ib,band in enumerate(bands):
        coimg = np.zeros((H,W))
        con   = np.zeros((H,W))
        for tim in tims:
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            nn = (tim.getInvvar()[Yi,Xi] > 0)
            coimg[Yo,Xo] += tim.getImage ()[Yi,Xi] * nn
            con  [Yo,Xo] += nn
            mn,mx = tim.zr
        coimg /= np.maximum(con,1)
        c = 2-ib
        rgbim[:,:,c] = np.clip((coimg - mn) / (mx - mn), 0., 1.)
        coimgs.append(coimg)
        coimas.append(dict(interpolation='nearest', origin='lower', cmap='gray',
                           vmin=mn, vmax=mx))
    imx = dict(interpolation='nearest', origin='lower')
    imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
                vmin=-5, vmax=5)
        
    # Render the detection maps
    detmaps = dict([(b, np.zeros((H,W))) for b in bands])
    detivs  = dict([(b, np.zeros((H,W))) for b in bands])
    for tim in tims:
        psf_sigma = tim.psf_sigma
        band = tim.band
        iv = tim.getInvvar()
        psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        detim = tim.getImage().copy()
        detim[iv == 0] = 0.
        detim = gaussian_filter(detim, psf_sigma) / psfnorm**2
        detsig1 = tim.sig1 / psfnorm
        subh,subw = tim.shape
        detiv = np.zeros((subh,subw)) + (1. / detsig1**2)
        detiv[iv == 0] = 0.
        (Yo,Xo,Yi,Xi) = tim.resamp
        detmaps[band][Yo,Xo] += detiv[Yi,Xi] * detim[Yi,Xi]
        detivs [band][Yo,Xo] += detiv[Yi,Xi]

    # find significant peaks in the per-band detection maps and SED-matched (hot)
    # segment into blobs
    # blank out blobs containing a catalog source
    # create sources for any remaining peaks
    hot = np.zeros((H,W), bool)
    sedmap = np.zeros((H,W))
    sediv  = np.zeros((H,W))
    for band in bands:
        detmap = detmaps[band] / np.maximum(1e-16, detivs[band])
        detsn = detmap * np.sqrt(detivs[band])
        hot |= (detsn > 5.)
        sedmap += detmaps[band]
        sediv  += detivs [band]
        detmaps[band] = detmap
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
    hot |= (sedsn > 5.)
    peaks = hot.copy()

    plt.clf()
    plt.imshow(np.round(sedsn), interpolation='nearest', origin='lower',
               vmin=0, vmax=10, cmap='hot')
    plt.title('SED-matched detection filter (flat SED)')
    ps.savefig()

    crossa = dict(ms=10, mew=1.5)
    plt.clf()
    plt.imshow(peaks, cmap='gray', **imx)
    ax = plt.axis()
    plt.plot(T.itx, T.ity, 'r+', **crossa)
    plt.axis(ax)
    plt.title('Detection blobs')
    ps.savefig()
    
    blobs,nblobs = label(hot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    for x,y in zip(T.itx, T.ity):
        # blob number
        bb = blobs[y,x]
        if bb == 0:
            continue
        # un-set 'peaks' within this blob
        slc = blobslices[bb-1]
        peaks[slc][blobs[slc] == bb] = 0

    plt.clf()
    plt.imshow(peaks, cmap='gray', **imx)
    ax = plt.axis()
    plt.plot(T.itx, T.ity, 'r+', **crossa)
    plt.axis(ax)
    plt.title('Detection blobs minus SE catalog sources')
    ps.savefig()
        
    # zero out the edges(?)
    peaks[0 ,:] = peaks[:, 0] = 0
    peaks[-1,:] = peaks[:,-1] = 0
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,2:  ])
    pki = np.flatnonzero(peaks)
    peaky,peakx = np.unravel_index(pki, peaks.shape)
    print len(peaky), 'peaks'
    
    plt.clf()
    plt.imshow(coimgs[1], **coimas[1])
    ax = plt.axis()
    plt.plot(T.tx, T.ty, 'r+', **crossa)
    plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
    plt.axis(ax)
    plt.title('SE Catalog + SED-matched detections')
    ps.savefig()

    plt.clf()
    plt.imshow(rgbim, **imx)
    ax = plt.axis()
    plt.plot(T.tx, T.ty, 'r+', **crossa)
    plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
    plt.axis(ax)
    plt.title('SE Catalog + SED-matched detections')
    ps.savefig()
    
    if False:
        # RGB detection map
        rgbdet = np.zeros((H,W,3))
        for iband,band in enumerate(bands):
            c = 2-iband
            detsn = detmaps[band] * np.sqrt(detivs[band])
            rgbdet[:,:,c] = np.clip(detsn / 10., 0., 1.)
        plt.clf()
        plt.imshow(rgbdet, **imx)
        ax = plt.axis()
        plt.plot(T.tx, T.ty, 'r+', **crossa)
        plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
        plt.axis(ax)
        plt.title('SE Catalog + SED-matched detections')
        ps.savefig()

    # Grow the 'hot' pixels by dilating by a few pixels
    rr = 2.0
    RR = int(np.ceil(rr))
    S = 2*RR+1
    struc = (((np.arange(S)-RR)**2)[:,np.newaxis] +
             ((np.arange(S)-RR)**2)[np.newaxis,:]) <= rr**2
    hot = binary_dilation(hot, structure=struc)
    #iterations=int(np.ceil(2. * psf_sigma)))

    # Add sources for the new peaks we found
    # make their initial fluxes ~ 5-sigma
    fluxes = dict([(b,[]) for b in bands])
    for tim in tims:
        psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
        fluxes[tim.band].append(5. * tim.sig1 / psfnorm)
    fluxes = dict([(b, np.mean(fluxes[b])) for b in bands])
    pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
    print 'Adding', len(pr), 'new sources'
    # Also create FITS table for new sources
    Tnew = fits_table()
    Tnew.ra  = pr
    Tnew.dec = pd
    Tnew.tx = peakx
    Tnew.ty = peaky
    Tnew.itx = np.clip(np.round(Tnew.tx).astype(int), 0, W-1)
    Tnew.ity = np.clip(np.round(Tnew.ty).astype(int), 0, H-1)
    for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
        cat.append(PointSource(RaDecPos(r,d),
                               NanoMaggies(order=bands, **fluxes)))
    T = merge_tables([T, Tnew], columns='fillzero')

    # Segment, and record which sources fall into each blob
    blobs,nblobs = label(hot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]
    blobsrcs = []
    blobflux = []
    for blob in range(1, nblobs+1):
        blobsrcs.append(np.flatnonzero(T.blob == blob))
        # not really 'flux' per se...
        bslc = blobslices[blob-1]
        blobflux.append(np.sum(sedsn[bslc][blobs[bslc] == blob]))

    if False:
        plt.clf()
        plt.imshow(hot, cmap='gray', **imx)
        plt.title('Segmentation')
        ps.savefig()


    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]

    # Fit in order of flux
    for iblob in np.argsort(-np.array(blobflux)):
        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        if len(Isrcs) == 0:
            continue

        cat.freezeAllParams()
        print 'Fitting:'
        for i in Isrcs:
            cat.thawParams(i)
            print cat[i]
            
        #print 'Fitting:'
        #tractor.printThawedParams()

        # before-n-after plots
        mod0 = [tractor.getModelImage(tim) for tim in tims]
        print 'Initial chi-squared:', tractor.getLogLikelihood()

        # blob bbox in target coords
        sy,sx = bslc
        y0,y1 = sy.start, sy.stop
        x0,x1 = sx.start, sx.stop

        rr,dd = targetwcs.pixelxy2radec([x0,x0,x1,x1],[y0,y1,y1,y0])

        ###
        # FIXME -- We create sub-image for each blob here.
        # What wo don't do, though, is mask out the invvar pixels
        # that are within the blob bounding-box but not within the
        # blob itself.  Does this matter?
        ###
        
        subtims = []
        for i,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            if sx1 < 0 or sy1 < 0 or sx1 > w or sy1 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            
            print 'image subregion', sx0,sx1,sy0,sy1

            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subiv  = tim.getInvvar()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[i]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            # FIXME --
            #subpsf = tim.psfex.mogAt(ox0+(x0+x1)/2., oy0+(y0+y1)/2.)
            #subpsf = tim.getPsf()

            psfimg = tim.psfex.instantiateAt(ox0+(x0+x1)/2., oy0+(y0+y1)/2.,
                                             nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg)

            subtim = Image(data=subimg, invvar=subiv, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky())
            subtims.append(subtim)
            
        subtr = Tractor(subtims, cat)
        subtr.freezeParam('images')
        print 'Optimizing:', subtr
        subtr.printThawedParams()
        
        for step in range(10):
            #dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        # Try fitting sources one at a time?
        # if len(Isrcs) > 1:
        #     for i in Isrcs:
        #         print 'Fitting source', i
        #         cat.freezeAllBut(i)
        #         for step in range(5):
        #             dlnp,X,alpha = subtr.optimize(priors=False,
        #                                           shared_params=False)
        #             print 'dlnp:', dlnp
        #             if dlnp < 0.1:
        #                 break
            
        mod1 = [tractor.getModelImage(tim) for tim in tims]
        print 'First fit chi-squared:', tractor.getLogLikelihood()

        # Forced-photometer bands individually
        for band in bands:
            cat.freezeAllRecursive()
            for i in Isrcs:
                cat.thawParam(i)
                cat[i].thawPathsTo(band)
            #cat.thawPathsTo(band)
            print
            print 'Fitting', band, 'band:'
            subtr.printThawedParams()
            B = 8
            X = subtr.optimize_forced_photometry(shared_params=False, use_ceres=True,
                                                 BW=B, BH=B, wantims=False)

        # Try to forced-photometer bands simultaneously -- doesn't work.
        # cat.freezeAllRecursive()
        # for i in Isrcs:
        #     cat.thawParam(i)
        #     cat[i].thawPathsTo(*bands)
        # print
        # print 'Forced phot:'
        # subtr.printThawedParams()
        # B = 8
        # X = subtr.optimize_forced_photometry(shared_params=False, use_ceres=True,
        #                                      BW=B, BH=B, wantims=False)
        # cat.thawAllRecursive()

        mod2 = [tractor.getModelImage(tim) for tim in tims]
        print 'Forced-phot chi-squared:', tractor.getLogLikelihood()

        rgbm0 = np.zeros((H,W,3))
        rgbm1 = np.zeros((H,W,3))
        rgbm2 = np.zeros((H,W,3))
        rgbchi0 = np.zeros((H,W,3))
        rgbchi1 = np.zeros((H,W,3))
        rgbchi2 = np.zeros((H,W,3))
        subims0 = []
        subims1 = []
        subims2 = []
        chis = dict([(b,[]) for b in bands])
        
        for iband,band in enumerate(bands):
            coimg = coimgs[iband]
            com0  = np.zeros((H,W))
            com1  = np.zeros((H,W))
            com2  = np.zeros((H,W))
            cochi0 = np.zeros((H,W))
            cochi1 = np.zeros((H,W))
            cochi2 = np.zeros((H,W))
            for tim,m0,m1,m2 in zip(tims, mod0, mod1,mod2):
                if tim.band != band:
                    continue
                (Yo,Xo,Yi,Xi) = tim.resamp

                chi0 = ((tim.getImage()[Yi,Xi] - m0[Yi,Xi]) *
                        tim.getInvError()[Yi,Xi])
                chi1 = ((tim.getImage()[Yi,Xi] - m1[Yi,Xi]) *
                        tim.getInvError()[Yi,Xi])
                chi2 = ((tim.getImage()[Yi,Xi] - m2[Yi,Xi]) *
                        tim.getInvError()[Yi,Xi])

                rechi = np.zeros((H,W))
                rechi[Yo,Xo] = chi0
                rechi0 = rechi[bslc].copy()
                rechi[Yo,Xo] = chi1
                rechi1 = rechi[bslc].copy()
                rechi[Yo,Xo] = chi2
                rechi2 = rechi[bslc].copy()
                chis[band].append((rechi0,rechi1,rechi2))
                
                cochi0[Yo,Xo] += chi0
                cochi1[Yo,Xo] += chi1
                cochi2[Yo,Xo] += chi2
                com0 [Yo,Xo] += m0[Yi,Xi]
                com1 [Yo,Xo] += m1[Yi,Xi]
                com2 [Yo,Xo] += m2[Yi,Xi]
                mn,mx = tim.zr
            com0  /= np.maximum(con,1)
            com1  /= np.maximum(con,1)
            com2  /= np.maximum(con,1)

            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            c = 2-iband
            rgbm0[:,:,c] = np.clip((com0  - mn) / (mx - mn), 0., 1.)
            rgbm1[:,:,c] = np.clip((com1  - mn) / (mx - mn), 0., 1.)
            rgbm2[:,:,c] = np.clip((com2  - mn) / (mx - mn), 0., 1.)

            mn,mx = -5,5
            rgbchi0[:,:,c] = np.clip((cochi0 - mn) / (mx - mn), 0, 1)
            rgbchi1[:,:,c] = np.clip((cochi1 - mn) / (mx - mn), 0, 1)
            rgbchi2[:,:,c] = np.clip((cochi2 - mn) / (mx - mn), 0, 1)

            subims0.append((coimg[bslc], com0[bslc], ima, cochi0[bslc]))
            subims1.append((coimg[bslc], com1[bslc], ima, cochi1[bslc]))
            subims2.append((coimg[bslc], com2[bslc], ima, cochi2[bslc]))

        # Plot per-band chi coadds, and RGB images for before & after
        for subims,rgbm in [(subims0,rgbm0), (subims1,rgbm1), (subims2,rgbm2)]:
            plt.clf()
            for j,(im,m,ima,chi) in enumerate(subims):
                plt.subplot(3,4,1 + j + 0)
                plt.imshow(im, **ima)
                plt.subplot(3,4,1 + j + 4)
                plt.imshow(m, **ima)
                plt.subplot(3,4,1 + j + 8)
                plt.imshow(-chi, **imchi)
            plt.subplot(3,4,4)
            plt.imshow(np.dstack([rgbim[:,:,c][bslc] for c in [0,1,2]]), **imx)
            plt.subplot(3,4,8)
            plt.imshow(np.dstack([rgbm[:,:,c][bslc] for c in [0,1,2]]), **imx)
            plt.subplot(3,4,12)
            plt.imshow(rgbim, **imx)
            ax = plt.axis()
            plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
            plt.axis(ax)
            ps.savefig()

        # Plot per-image chis
        cols = max(len(v) for v in chis.values())
        rows = len(bands)
        for i in [0,1,2]:
            plt.clf()
            for row,band in enumerate(bands):
                sp0 = 1 + cols*row
                for col,cc in enumerate(chis[band]):
                    chi = cc[i]
                    plt.subplot(rows, cols, sp0 + col)
                    plt.imshow(-chi, **imchi)
            ps.savefig()
    
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

