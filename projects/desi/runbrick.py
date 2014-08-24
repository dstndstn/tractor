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
imgdir   = os.environ.get('DECALS_IMG', None)
sedir    = os.environ.get('DECALS_SE',
                          '/project/projectdirs/desi/imaging/code/cats')
print 'calibdir', calibdir

#

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

    for fn in [im.wcsfn,im.sexfn,im.psffn,im.morphfn,im.corrfn]:
        print 'exists?', os.path.exists(fn), fn
        
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
            '-c', os.path.join(sedir, 'DECaLS-v2.sex'),
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
    
    B = fits_table('bricks.fits')

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
            #continue
            if imgdir:
                fn = fn.replace('/project/projectdirs/cosmo/staging',
                                imgdir)
            
            im = DecamImage(fn, hdu, band)
            ims.append(im)

            # fns = ','.join([im.imgfn, im.dqfn, im.wtfn])
            # cmd = 'rsync --progress -arvz carver:"{%s}" .' % fns
            # print
            # if os.system(cmd):
            #     sys.exit(-1)
            # continue
            

            
    for im in ims:
        band = im.band
        run_calibs(im, ra, dec, pixscale)

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

    # Merge catalogs by keeping ones > 0.5" away from previous ones
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
    #T.about()
    # for c in T.get_columns():
    #     plt.clf()
    #     plt.hist(T.get(c), 50)
    #     plt.xlabel(c)
    #     ps.savefig()

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

        # in pixels
        psf_fwhm = imghdr['FWHM']
        primhdr = im.read_image_primary_header()
        magzp  = primhdr['MAGZERO']
        zpscale = NanoMaggies.zeropointToScale(magzp)
        print 'Magzp', magzp
        print 'zpscale', zpscale

        #sky = imghdr['SKYBRITE']
        medsky = np.median(img)
        #print 'SKYBRITE:', sky
        #print 'Image median:', medsky
        img -= medsky

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        ### FIXME!! -- this is the sliced image size -- should be full
        ### size probably!

        info = im.get_image_info()
        print 'Image info:', info
        fullh,fullw = info['dims']

        psfex = PsfEx(im.psffn, fullw, fullh, scale=False, nx=9, ny=17)
        psfex = ShiftedPsf(psfex, x0, y0)
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

    detmaps = dict([(b, np.zeros((H,W))) for b in bands])
    detivs  = dict([(b, np.zeros((H,W))) for b in bands])
    for tim in tims:
        # Render the detection map
        wcs = tim.sip_wcs
        x0,y0 = tim.x0,tim.y0
        psf_sigma = tim.psf_sigma
        band = tim.band
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(int(x0), int(y0), subw, subh)
        subiv = tim.getInvvar()
        psfnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
        detim = tim.getImage().copy()
        detim[subiv == 0] = 0.
        detim = gaussian_filter(detim, psf_sigma) / psfnorm**2
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
            print 'Resampled', len(Yo), 'pixels'
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue

        detsig1 = tim.sig1 / psfnorm
        detiv = np.zeros((subh,subw)) + (1. / detsig1**2)
        detiv[subiv == 0] = 0.

        detmaps[band][Yo,Xo] += detiv[Yi,Xi] * detim[Yi,Xi]
        detivs [band][Yo,Xo] += detiv[Yi,Xi]

    hot = np.zeros((H,W), bool)
    
    sedmap = np.zeros((H,W))
    sediv  = np.zeros((H,W))
    for band in bands:
        detmap = detmaps[band] / np.maximum(1e-16, detivs[band])
        detsn = detmap * np.sqrt(detivs[band])
        hot |= (detsn > 5.)
        sedmap += detmaps[band]
        sediv  += detivs [band]

        # plt.clf()
        # plt.imshow(detmap, interpolation='nearest', origin='lower', cmap='gray')
        # plt.title('Detection map: %s' % band)
        # plt.colorbar()
        # ps.savefig()

        plt.clf()
        plt.imshow(detsn, interpolation='nearest', origin='lower', cmap='hot',
                   vmin=-2, vmax=10)
        plt.title('Detection map S/N: %s' % band)
        plt.colorbar()
        ps.savefig()


    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)

    hot |= (sedsn > 5.)
    hot = binary_dilation(hot, iterations=int(np.ceil(2. * psf_sigma)))

    # plt.clf()
    # plt.imshow(sedmap, interpolation='nearest', origin='lower', cmap='gray')
    # plt.title('Detection map: flat SED')
    # plt.colorbar()
    # ps.savefig()

    plt.clf()
    plt.imshow(sedsn, interpolation='nearest', origin='lower', cmap='hot',
               vmin=-2, vmax=10)
    plt.title('Detection map S/N: flat SED')
    plt.colorbar()
    ps.savefig()

    plt.clf()
    plt.imshow(hot, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Segmentation')
    ax = plt.axis()
    ok,T.tx,T.ty = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.tx -= 1
    T.ty -= 1
    plt.plot(T.tx, T.ty, 'r+', mec='r', mfc='none', mew=2, ms=10)
    plt.axis(ax)
    ps.savefig()

    blobs,nblobs = label(hot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)

    print 'blobs max', blobs.max()

    T.blob = blobs[np.clip(np.round(T.ty).astype(int), 0, H-1),
                   np.clip(np.round(T.tx).astype(int), 0, W-1)]

    blobsrcs = []
    for blob in range(1, nblobs+1):
        blobsrcs.append(np.flatnonzero(T.blob == blob))

    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')

    # save resampling params
    for tim in tims:
        wcs = tim.sip_wcs

        x0,y0 = int(tim.x0),int(tim.y0)
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(x0, y0, subw, subh)

        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
            print 'Resampled', len(Yo), 'pixels'
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue
        tim.resamp = (Yo,Xo,Yi,Xi)


    for b,I in enumerate(blobsrcs):
        bslc = blobslices[b]
        bsrcs = blobsrcs[b]

        print 'blob slice:', bslc
        print 'sources in blob:', I
        if len(I) == 0:
            continue

        cat.freezeAllParams()
        #cat.thawParams(I)
        for i in I:
            cat.thawParams(i)

        print 'Fitting:'
        tractor.printThawedParams()

        # before-n-after plots
        mod0 = [tractor.getModelImage(tim) for tim in tims]

        for step in range(10):
            dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        mod1 = [tractor.getModelImage(tim) for tim in tims]

        rgbim = np.zeros((H,W,3))
        rgbm0 = np.zeros((H,W,3))
        rgbm1 = np.zeros((H,W,3))

        for ib,band in enumerate(bands):
            coimg = np.zeros((H,W))
            com0  = np.zeros((H,W))
            com1  = np.zeros((H,W))
            con   = np.zeros((H,W))
            for tim,m0,m1 in zip(tims, mod0, mod1):
                if tim.band != band:
                    continue
                (Yo,Xo,Yi,Xi) = tim.resamp
                coimg[Yo,Xo] += tim.getImage()[Yi,Xi]
                com0 [Yo,Xo] += m0[Yi,Xi]
                com1 [Yo,Xo] += m1[Yi,Xi]
                con  [Yo,Xo] += 1

                mn,mx = tim.zr

            coimg /= np.maximum(con,1)
            com0  /= np.maximum(con,1)
            com1  /= np.maximum(con,1)

            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            sy,sx = bslc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop

            c = 2-ib
            rgbim[:,:,c] = np.clip((coimg - mn) / (mx - mn), 0., 1.)
            rgbm0[:,:,c] = np.clip((com0  - mn) / (mx - mn), 0., 1.)
            rgbm1[:,:,c] = np.clip((com1  - mn) / (mx - mn), 0., 1.)

            for m,txt in [(com0,'Before'),(com1,'After')]:
                plt.clf()
                plt.subplot(1,2,1)
                plt.imshow(coimg, **ima)
                plt.subplot(1,2,2)
                plt.imshow(m, **ima)
                ax = plt.axis()
                plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
                plt.axis(ax)
                plt.suptitle('%s optimization: %s band' % (txt, band))
                ps.savefig()

        ima = dict(interpolation='nearest', origin='lower')
        for m,txt in [(rgbm0,'Before'), (rgbm1,'After')]:
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(rgbim, **ima)
            plt.subplot(1,2,2)
            plt.imshow(m, **ima)
            ax = plt.axis()
            plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],'r-')
            plt.axis(ax)
            plt.suptitle('%s optimization: %s band' % (txt, band))
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

