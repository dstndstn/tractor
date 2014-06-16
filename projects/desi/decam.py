if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.sdss.fields import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

from scipy.ndimage.filters import *
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.interpolation import shift

def read_decam_image(basefn, skysubtract=True, slc=None):
    '''
    slc: slice of image to read
    '''
    imgfn  = basefn + '.fits'
    maskfn = basefn + '.bpm.fits'
    psffn  = basefn + '.cat.psf'

    print 'Reading', imgfn, 'and', maskfn
    f = fitsio.FITS(imgfn)
    m = fitsio.FITS(maskfn)
    hdr = f[0].read_header()
    fullimg = None
    if slc is not None:
        if skysubtract:
            # Also read full image for sky estimation
            fullimg = f[0].read()
        img = f[0][slc]
        mask = m[0][slc]
        y0,x0 = slc[0].start, slc[1].start
    else:
        img = f[0].read()
        fullimg = img
        mask = m[0].read()
        y0,x0 = 0,0
    #img,hdr = fitsio.read(imgfn, header=True)
    print 'Got image', img.shape, img.dtype
    print 'Mask', mask.shape, mask.dtype
    print 'values', np.unique(mask)
    print 'Mask=0:', sum(mask == 0)

    name = hdr['FILENAME'].replace('.fits','').replace('_', ' ')
    filt = hdr['FILTER'].split()[0]
    sat = hdr.get('SATURATE', None)
    zp = hdr.get('MAG_ZP', None)
    if zp is None:
        zp = hdr['UB1_ZP']
    zpscale = NanoMaggies.zeropointToScale(zp)
    print 'Name', name, 'filter', filt, 'zp', zp
    
    sip = Sip(imgfn)
    print 'SIP', sip
    print 'RA,Dec bounds', sip.radec_bounds()
    
    H,W = img.shape
    print 'Reading PSF', psffn
    psf = PsfEx(psffn, W, H)
    print 'Got PSF', psf
    if x0 or y0:
        psf = ShiftedPsf(psf, x0, y0)
        
    if skysubtract:
        # Remove x/y gradient estimated in ~500-pixel^2 squares
        fH,fW = fullimg.shape
        nx = int(np.ceil(float(fW) / 512))
        ny = int(np.ceil(float(fH) / 512))
        xx = np.linspace(0, fW, nx+1)
        yy = np.linspace(0, fH, ny+1)
        subs = np.zeros((len(yy)-1, len(xx)-1))
        for iy,(ylo,yhi) in enumerate(zip(yy, yy[1:])):
            for ix,(xlo,xhi) in enumerate(zip(xx, xx[1:])):
                subim = fullimg[ylo:yhi, xlo:xhi]
                subs[iy,ix] = np.median(subim.ravel())

        xx,yy = np.meshgrid(xx[:-1],yy[:-1])
        A = np.zeros((len(xx.ravel()), 3))
        A[:,0] = 1.
        dx = float(fW) / float(nx)
        dy = float(fH) / float(ny)
        A[:,1] = 0.5*dx + xx.ravel()
        A[:,2] = 0.5*dy + yy.ravel()
        b = subs.ravel()
        X,res,rank,s = np.linalg.lstsq(A, b)

        print 'Sky gradient:', X

        bg = np.zeros_like(img) + X[0]
        bg += (X[1] * (x0 + np.arange(W)))[np.newaxis,:]
        bg += (X[2] * (y0 + np.arange(H)))[:,np.newaxis]

        bx = (X[1] * (x0 + np.arange(W)))
        by = (X[2] * (y0 + np.arange(H)))
        print 'Background x contribution:', bx.shape, bx.min(), bx.max()
        print 'Background y contribution:', by.shape, by.min(), by.max()

        #bg = np.zeros_like(img)
        #ok = median_smooth(img, (mask != 0), 100, bg)
        orig_img = img.copy()
        img -= bg
        sky = 0.
        sky1 = X[0] + X[1]*(x0 + W/2.) + X[2]*(y0 + H/2.)
    else:
        sky = np.median(img.ravel())
        print 'Median:', sky
        sky1 = sky

    if fullimg is not None:
        dim = fullimg
    else:
        dim = img
    diffs = dim[:-5:10,:-5:10] - dim[5::10,5::10]
    mad = np.median(np.abs(diffs).ravel())
    sig1 = 1.4826 * mad / np.sqrt(2.)
    print 'MAD', mad, '-> sigma', sig1
    invvar = np.zeros_like(img) + 1./sig1**2
    invvar[mask != 0] = 0.
    if sat is not None:
        saturated = (img >= sat)
        invvar[saturated] = 0.

    #print 'Image: max', img.max()
    
    wcs = ConstantFitsWcs(sip)
    if x0 or y0:
        wcs.setX0Y0(x0,y0)

    # Scale images to Nanomaggies
    img /= zpscale
    invvar *= zpscale**2
    sky /= zpscale
    sig1 /= zpscale
    sky1 /= zpscale

    orig_zpscale = zpscale
    zpscale = 1.

    tim = Image(img, invvar=invvar, wcs=wcs,
                photocal=LinearPhotoCal(zpscale, band=filt),
                psf=psf, sky=ConstantSky(sky), name=name)
    tim.zr = [sky-3.*sig1, sky+5.*sig1]
    tim.orig_img = orig_img
    tim.filter = filt
    tim.sig1 = sig1
    tim.sky1 = sky1
    tim.zp = zp
    tim.ozpscale = orig_zpscale
    return tim
    

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%prog [options] [base input filename]')
    parser.add_option('--se', action='store_true')
    parser.add_option('--x0', type=int, help='Read sub-image', default=0)
    parser.add_option('--y0', type=int, help='Read sub-image', default=0)
    parser.add_option('-W', type=int, help='Read sub-image', default=0)
    parser.add_option('-H', type=int, help='Read sub-image', default=0)
    opt,args = parser.parse_args()

    if opt.W or opt.H or opt.x0 or opt.y0:
        x1,y1 = -1,-1
        if opt.H:
            y1 = opt.y0 + opt.H
        if opt.W:
            x1 = opt.x0 + opt.W
        slc = slice(opt.y0, y1), slice(opt.x0, x1)
    else:
        slc = None

    if len(args):
        decbase = args[0]
    else:
        decbase = 'proc/20130330/C01/zband/DECam_00192399.01.p.w'

    tim = read_decam_image(decbase, slc=slc)
    print 'Got', tim, tim.shape

    basefn = os.path.basename(decbase).lower().replace('.p.w', '')
    picklefn = basefn + '.pickle'
    secatfn = decbase + '.morph.fits'
    sdssobjfn = basefn + '-sdss.fits'
    #seobjfn = basefn + '-se.fits'
    seobjfn = decbase + '.cat.fits'

    if os.path.exists(picklefn):
        print 'Reading', picklefn
        X = unpickle_from_file(picklefn)
        tim.psf = X['psf']
    else:
        tim.psf.ensureFit()
        X = dict(psf=tim.psf)
        pickle_to_file(X, picklefn)
        print 'Wrote', picklefn

    # SourceExtractor, or SDSS?
    secat = opt.se
    if secat:
        objfn = seobjfn
        catname = 'SExtractor'
        ps = PlotSequence('decam-se')
    else:
        objfn = sdssobjfn
        catname = 'SDSS'
        ps = PlotSequence('decam-sdss')
    ps.format = '%03i'

    sdss = DR9(basedir='dr9')
    sdss.saveUnzippedFiles('dr9')
    if os.environ.get('BOSS_PHOTOOBJ') is not None:
        print 'Using local tree for SDSS files'
        sdss.useLocalTree()

    sip = tim.wcs.wcs
    if secat:
        T = fits_table(secatfn, hdu=2,
                       column_map={'alpha_j2000':'ra', 'delta_j2000':'dec'},)
        print len(T), 'sources in SExtractor catalog'

        T.mag_psf      += tim.zp
        T.mag_spheroid += tim.zp
        T.mag_disk     += tim.zp
        
        from projects.cs82.cs82 import get_cs82_sources
        cat,catI = get_cs82_sources(T, bands=['z'])
        T.cut(catI)
        
    else:
        if not os.path.exists(objfn):
            margin = 5./3600.

            print 'SIP:', sip

            objs = read_photoobjs_in_wcs(sip, margin, sdss=sdss)
            objs.writeto(objfn)
        else:
            objs = fits_table(objfn)

        #print 'Zip:', zip(objs.run, objs.camcol, objs.field)
        #rcfs = np.unique(zip(objs.run, objs.camcol, objs.field))
        #print 'RCF', rcfs

        print 'run', objs.run.min(), objs.run.max()
        print 'camcol', objs.camcol.min(), objs.camcol.max()
        print 'field', objs.field.min(), objs.field.max()
        
        rcfnum = (objs.run.astype(np.int32) * 10000 +
                  objs.camcol.astype(np.int32) * 1000 +
                  objs.field)
        rcfnum = np.unique(rcfnum)
        rcfs = zip(rcfnum / 10000, rcfnum % 10000 / 1000, rcfnum % 1000)
        print 'RCF', rcfs

        
        # SEcat = fits_table(seobjfn, hdu=2)
        # SEcat.ra  = SEcat.alpha_j2000
        # SEcat.dec = SEcat.delta_j2000
        # I,J,d = match_radec(SEcat.ra, SEcat.dec, objs.ra, objs.dec, 1.0/3600.)
        # plt.clf()
        # plt.plot(3600.*(SEcat.ra[I] - objs.ra[J]), 3600.*(SEcat.dec[I] - objs.dec[J]), 'b.')
        # plt.xlabel('dRA (arcsec)')
        # plt.ylabel('dDec (arcsec)')
        # plt.axis([-0.6,0.6,-0.6,0.6])
        # ps.savefig()

        
        print len(objs), 'SDSS photoObjs'
        r0,r1,d0,d1 = sip.radec_bounds()
        cat = get_tractor_sources_dr9(
            None, None, None, objs=objs, sdss=sdss,
            radecroi=[r0,r1,d0,d1], bands=[tim.filter],
            nanomaggies=True, fixedComposites=True,
            useObjcType=True)

        #radec0 = np.array([(src.getPosition().ra, src.getPosition().dec)
        #                   for src in cat])
        
    print len(cat), 'sources'


    # FIXME -- looks like a small astrometric shift between SourceExtractor
    # catalog and SDSS.
    if True:
        SEcat = fits_table(seobjfn, hdu=2)
        SEcat.ra  = SEcat.alpha_j2000
        SEcat.dec = SEcat.delta_j2000
        SDSScat = fits_table(sdssobjfn)
        I,J,d = match_radec(SEcat.ra, SEcat.dec, SDSScat.ra, SDSScat.dec, 1.0/3600.)
        plt.clf()
        plt.plot(3600.*(SEcat.ra[I] - SDSScat.ra[J]), 3600.*(SEcat.dec[I] - SDSScat.dec[J]), 'b.')
        plt.xlabel('dRA (arcsec)')
        plt.ylabel('dDec (arcsec)')
        plt.axis([-0.6,0.6,-0.6,0.6])
        ps.savefig()
        #sys.exit(0)

        if secat:
            dra = ddec = 0.
        else:
            dra  = np.median(SEcat.ra [I] - SDSScat.ra [J])
            ddec = np.median(SEcat.dec[I] - SDSScat.dec[J])
        


    def imshow(img, **kwargs):
        x = plt.imshow(img.T, **kwargs)
        plt.xticks([])
        plt.yticks([])
        return x

    def sqimshow(img, **kwa):
        mn = kwa.pop('vmin')
        mx = kwa.pop('vmax')
        imshow(np.sqrt(np.maximum(0, img - mn)), vmin=0, vmax=np.sqrt(mx-mn), **kwa)
    
    if secat:
        H,W = tim.shape
        I = np.argsort(T.mag_psf)
        ims = []
        ratios = []
        for i in I[:20]:
            print
            x,y = T.x_image[i], T.y_image[i]
            ix,iy = int(np.round(x)), int(np.round(y))

            psfim = tim.getPsf().getPointSourcePatch(x-1,y-1)
            ph,pw = psfim.shape
            print 'PSF shape', pw,ph
            S = ph/2
            if ix < S or ix > (W-S) or iy < S or iy > (H-S):
                continue
            #subim = tim.getImage()[iy-S:iy+S+1, ix-S:ix+S+1]
            x0,y0 = psfim.x0, psfim.y0
            subim = tim.getImage()[y0:y0+ph, x0:x0+pw]

            pixim = tim.getPsf().instantiateAt(x-1, y-1)
            print 'pixim', pixim.sum()
            
            mn,mx = [np.percentile(subim, p) for p in [25,100]]
            zp = tim.ozpscale
            print 'subim sum', np.sum(subim)
            print 'flux', T.flux_psf[i]
            print 'zpscale', zp
            flux = T.flux_psf[i] / zp
            print 'flux/zpscale', flux
            print 'psfim sum', psfim.patch.sum()

            dy = y - iy
            dx = x - ix
            
            ims.append((subim, psfim, flux, mn,mx, pixim,dx,dy))
            ratios.append(subim.sum() / flux)

        ratio = np.median(ratios)
        for subim, psfim, flux, mn,mx, pixim,dx,dy in ims:
            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
            dd = (mx - mn) * 0.05
            imdiff = dict(interpolation='nearest', origin='lower', cmap='gray',
                          vmin=-dd, vmax=dd)

            mod = psfim.patch * flux * ratio
            pixmod = pixim * flux * ratio

            if dx < 0:
                dx += 1
            if dy < 0:
                dy += 1

            spix = shift(pixmod, (dy,dx))
            ph,pw = spix.shape
            fh,fw = mod.shape
            sx = (fw - pw)/2
            sy = (fh - ph)/2
            xx,yy = np.meshgrid(np.arange(pw), np.arange(ph))
            pixx = np.sum(xx * pixmod) / pixmod.sum()
            pixy = np.sum(yy * pixmod) / pixmod.sum()

            shiftpix = np.zeros_like(mod)
            shiftpix[sy:sy+ph, sx:sx+pw] = spix

            xx,yy = np.meshgrid(np.arange(fw), np.arange(fh))
            modx = np.sum(xx * mod) / mod.sum()
            mody = np.sum(yy * mod) / mod.sum()
            shx = np.sum(xx * shiftpix) / shiftpix.sum()
            shy = np.sum(yy * shiftpix) / shiftpix.sum()

            imx = np.sum(xx * subim) / subim.sum()
            imy = np.sum(yy * subim) / subim.sum()
            
            print
            print 'Dx,Dy', dx,dy
            print 'Model    centroid %.5f, %.5f' % (modx,mody)
            print 'Shiftpix centroid %.5f, %.5f' % (shx, shy )
            print 'Image    centroid %.5f, %.5f' % (imx,imy)
            #print 'Pixim    centroid %.5f, %.5f' % (pixx,pixy)

            print 'Image - Model     %.5f, %.5f' % (imx-shx,imy-shy)

            #shiftpix2 = shift(shiftpix, (imy-shy, imx-shx))
            shiftpix2 = shift(shiftpix,
                              (-(pixy-np.round(pixy)), -(pixx-np.round(pixx))))

            plt.clf()
            plt.suptitle('dx,dy %.2f,%.2f' % (dx,dy))

            plt.subplot(3,4,1)
            imshow(subim, **ima)
            plt.title('image')

            plt.subplot(3,4,5)
            imshow(mod, **ima)
            plt.title('G model')

            plt.subplot(3,4,9)
            imshow(subim - mod, **imdiff)
            plt.title('G resid')

            #plt.subplot(3,4,3)
            #imshow(subim, **ima)
            plt.subplot(3,4,3)
            imshow(pixmod, **ima)
            plt.title('Pixelize mod')
            plt.subplot(3,4,7)
            imshow(shiftpix, **ima)
            plt.title('Shifted pix mod')
            plt.subplot(3,4,11)
            imshow(subim - shiftpix, **imdiff)
            plt.title('Shifted pix resid')

            plt.subplot(3,4,10)
            imshow(subim - shiftpix2, **imdiff)
            plt.title('Shifted pix2 resid')

            
            plt.subplot(3,4,12)
            imshow(shiftpix - mod, **imdiff)
            plt.title('Shifted pix - G mod')

            
            #plt.subplot(3,4,11)
            #imshow(subim - pixmod, **imdiff)

            plt.subplot(3,4,4)
            sqimshow(subim, **ima)
            plt.title('Image')
            plt.subplot(3,4,8)
            sqimshow(pixmod, **ima)
            plt.title('Pix mod')

            
            plt.subplot(3,4,2)
            sqimshow(subim, **ima)
            plt.title('Image')
            plt.subplot(3,4,6)
            sqimshow(mod, **ima)
            plt.title('G mod')
            ps.savefig()
        sys.exit(0)
    
    
    # very rough FWHM
    psfim = tim.getPsf().getPointSourcePatch(0., 0.)
    mx = psfim.patch.max()
    area = np.sum(psfim.patch > 0.5*mx)
    fwhm = 2. * np.sqrt(area / np.pi)
    print 'PSF FWHM', fwhm
    psfsig = fwhm/2.35
    psfnorm = np.sqrt(gaussian_filter(psfim.patch, psfsig).max())

    print 'PSF norm:', psfnorm

    # run rough detection alg on image
    img = tim.getImage().copy()
    img[(tim.getInvError() == 0)] = 0.
    detimg = gaussian_filter(img, psfsig) / psfnorm**2
    nsigma = 4.
    thresh = nsigma * tim.sig1 / psfnorm
    hot = (detimg > thresh)
    # expand by fwhm
    hot = binary_dilation(hot, iterations=int(fwhm))

    ima = dict(interpolation='nearest', origin='lower',
               vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
    imx = dict(interpolation='nearest', origin='lower', cmap='gray')
    
    # plt.clf()
    # imshow(tim.orig_img-tim.sky1, **ima)
    # plt.title('Original image')
    # ps.savefig()
    # 
    # plt.clf()
    # imshow(tim.orig_img-tim.data-tim.sky1, **ima)
    # plt.title('Background estimate')
    # ps.savefig()
    
    plt.clf()
    imshow(tim.data, **ima)
    plt.title('Image')
    ps.savefig()

    plt.clf()
    imshow(hot, **imx)
    plt.title('Detected')
    ps.savefig()
    
    plt.clf()
    imx = dict(interpolation='nearest', origin='lower')
    imshow(tim.getInvError() == 0, cmap='gray', **imx)
    plt.title('Masked')
    ps.savefig()

    tractor = Tractor([tim], cat)

    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod0 = mod
    chi0 = chi

    #fitsio.write(basefn + '-image.fits', tim.data, clobber=True)
    #fitsio.write(basefn + '-mod0.fits', mod0, clobber=True)
    
    plt.clf()
    imshow(mod, **ima)
    plt.title('Tractor model image: Initial')
    ps.savefig()

    imchi = dict(interpolation='nearest', origin='lower',
                 vmin=-5, vmax=5, cmap='RdBu')

    imchi2 = dict(interpolation='nearest', origin='lower',
                  vmin=-100, vmax=100, cmap='RdBu')

    plt.clf()
    imshow(-chi, **imchi)
    plt.title('Image - Model chi: Initial')
    ps.savefig()

    # plt.clf()
    # imshow(-chi, **imchi2)
    # plt.title('Image - Model chi: Initial')
    # ps.savefig()
    
    tractor.freezeParamsRecursive('*')
    tractor.thawPathsTo(tim.filter)

    #print 'Fitting params:'
    #tractor.printThawedParams()
    
    sdssflux = np.array([sum(b.getFlux(tim.filter)
                             for b in src.getBrightnesses())
                         for src in cat])
    
    print 'Opt forced photom...'
    tractor.optimize_forced_photometry(shared_params=False, use_ceres=True,
                                       BW=8,BH=8, wantims=False)

    tflux = np.array([sum(b.getFlux(tim.filter)
                          for b in src.getBrightnesses())
                      for src in cat])

    smag = -2.5 * (np.log10(sdssflux) - 9.)
    tmag = -2.5 * (np.log10(   tflux) - 9.)

    T = fits_table()
    T.ra = np.array([src.getPosition().ra for src in cat])
    T.dec = np.array([src.getPosition().dec for src in cat])
    T.flux = tflux
    T.mag = tmag
    if secat:
        T.writeto(basefn + '-se-phot.fits')
    else:
        T.writeto(basefn + '-phot.fits')
    
    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod1 = mod
    chi1 = chi

    I = np.argsort(-np.abs(chi1.ravel()))
    print 'Worst chi pixels:', chi1.flat[I[:20]]
    
    #fitsio.write(basefn + '-mod1.fits', mod1, clobber=True)

    plt.clf()
    imshow(mod, **ima)
    plt.title('Tractor model image: Forced photom')
    ps.savefig()

    plt.clf()
    imshow(-chi, **imchi)
    plt.title('Image - Model chi: Forced photom')
    ps.savefig()

    plt.clf()
    imshow(-chi, **imchi2)
    plt.title('Image - Model chi: Forced photom')
    ps.savefig()
    
    plt.clf()
    lo,hi = -1e-1, 1e5
    plt.plot(sdssflux, tflux, 'b.', alpha=0.5)
    plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    plt.xlabel('%s z flux (nanomaggies)' % catname)
    plt.ylabel('DECam z flux (nanomaggies)')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.axis([1e-1, 1e5, 1e-1, 1e5])
    plt.title('Tractor forced photometry of DECam data')
    ps.savefig()

    plt.clf()
    lo,hi = 10,25
    plt.plot(smag, tmag, 'b.', alpha=0.5)
    plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    plt.xlabel('%s z (mag)' % catname)
    plt.ylabel('DECam z (mag)')
    plt.axis([hi,lo,hi,lo])
    plt.title('Tractor forced photometry of DECam data')
    ps.savefig()

    if False:
        S = fits_table(sdssobjfn)
        # stars
        S.cut(S.objc_type == 6)
    
        I,J,d = match_radec(S.ra, S.dec, T.ra, T.dec, 1./3600)
        S.cut(I)
        g = S.psfmag[:,1]
        r = S.psfmag[:,2]
        z = T.mag[J]
        plt.clf()
        plt.plot(g-r, r-z, 'b.', alpha=0.25)
        plt.axis([0, 2, -1, 3])
        plt.xlabel('SDSS g-r')
        plt.ylabel('SDSS r - DECam z')
        plt.title('Forced photometry using %s catalog (%i stars)' % (catname, len(S)))
        ps.savefig()

    if False:
        H,W = tim.shape
        for y0 in np.arange(0, H, 256):
            for x0 in np.arange(0, W, 256):
                subslc = slice(y0, y0+256), slice(x0, x0+256)
                imsa = ima.copy()
                imsa.update(extent=[x0,x0+256,y0,y0+256])
    
                print 'Slice', x0, y0
                I = np.argsort(-np.abs(chi0.ravel()))
                print 'Largest chi0:', chi0.flat[I[:20]]
                I = np.argsort(-np.abs(chi1.ravel()))
                print 'Largest chi1:', chi1.flat[I[:20]]
                
                plt.clf()
                plt.subplot(2,3,1)
                plt.imshow(tim.data[subslc], **imsa)
                plt.subplot(2,3,2)
                plt.imshow(mod0[subslc], **imsa)
                plt.subplot(2,3,3)
                plt.imshow(mod1[subslc], **imsa)
                plt.subplot(2,3,4)
                n1,b,p = plt.hist(np.clip(chi0[subslc], -6,6).ravel(), 50,
                                  range=(-6,6),
                                  log=True, histtype='step', color='r')
                n2,b,p = plt.hist(np.clip(chi1[subslc], -6,6).ravel(), 50,
                                  range=(-6,6),
                                  log=True, histtype='step', color='b')
                plt.axis([-6.1, 6.1, 0.1, 1.2*max(max(n1),max(n2))])
                plt.subplot(2,3,5)
                plt.imshow(-chi0[subslc], **imchi)
                plt.title('chi2: %g' % np.sum(chi0[subslc]**2))
                plt.subplot(2,3,6)
                plt.imshow(-chi1[subslc], **imchi)
                plt.title('chi2: %g' % np.sum(chi1[subslc]**2))
                ps.savefig()


    # Run detection alg on model image as well
    detimg = gaussian_filter(mod1, psfsig) / psfnorm**2
    modhot = (detimg > thresh)
    # expand by fwhm
    modhot = binary_dilation(modhot, iterations=int(fwhm))

    # plt.clf()
    # imshow(modhot, interpolation='nearest', origin='lower', cmap='gray')
    # plt.title('Mod Detected')
    # ps.savefig()

    uhot = np.logical_or(hot, modhot)
    
    # plt.clf()
    # imshow(uhot, interpolation='nearest', origin='lower', cmap='gray')
    # plt.title('Union Detected')
    # ps.savefig()

    blobs,nblobs = label(uhot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)

    # Also find the sources *within* each blob.
    #ra  = np.array([src.getPosition().ra  for src in cat])
    #dec = np.array([src.getPosition().dec for src in cat])
    wcs = tim.getWcs()
    xy = np.array([wcs.positionToPixel(src.getPosition()) for src in cat])
    xy = np.round(xy).astype(int)
    x = xy[:,0]
    y = xy[:,1]
    print 'x,y', x.shape, x.dtype, y.shape, y.dtype
    
    # Sort by chi-squared contributed by each blob.
    blobchisq = []
    blobsrcs = []
    for b,bslc in enumerate(blobslices):
        sy,sx = bslc
        y0,y1 = sy.start, sy.stop
        x0,x1 = sx.start, sx.stop
        bl = blobs[bslc]
        # chisq contributed by this blob
        chisq = np.sum((bl == (b+1)) * chi1[bslc]**2)
        blobchisq.append(chisq)
        # sources within this blob.
        I = np.flatnonzero((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
        if len(I):
            #I = I[bl[y[I],x[I]] == (b+1)]
            I = I[blobs[y[I],x[I]] == (b+1)]
        if len(I):
            blobsrcs.append([cat[i] for i in I])
        else:
            # this should be surprising...
            blobsrcs.append([])
    blobchisq = np.array(blobchisq)

    class ChattyTractor(Tractor):
        def setParams(self, p):
            #print 'SetParams:', ', '.join(['%.5f' % pp for pp in p])
            super(ChattyTractor, self).setParams(p)

        def _getOneImageDerivs(self, i):
            print 'GetOneImageDerivs:', i
            X = super(ChattyTractor, self)._getOneImageDerivs(i)
            print
            if X is None:
                print 'Got', X
            else:
                for ind,x0,y0,der in X:
                    print '  Ind', ind, 'x0y0', x0,y0, 'der', der.shape, der.dtype
            print
            return X

    radec0 = []
    radec1 = []

    imsq = dict(interpolation='nearest', origin='lower',
                vmin=tim.zr[0], vmax=25.*tim.zr[1], cmap='gray')

    #for ii,b in [(1330, 1492)]:
    for ii,b in enumerate(np.argsort(-blobchisq)):
        bslc = blobslices[b]
        bsrcs = blobsrcs[b]

        #if ii >= 50:
        #    break
        
        print
        print 'Blob', ii, 'of', len(blobchisq), 'index', b
        print 
        
        subiv = tim.getInvvar()[bslc]
        subiv[blobs[bslc] != (b+1)] = 0.

        subimg = tim.getImage()[bslc]
        
        # plt.clf()
        # plt.subplot(2,2,1)
        # imshow(subimg, **ima)
        # plt.subplot(2,2,2)
        # imshow(subiv, **imx)
        # plt.subplot(2,2,3)
        # imshow(mod[bslc], **ima)
        # plt.subplot(2,2,4)
        # imshow(chi[bslc] * (subiv > 0), **imchi)
        # ps.savefig()

        sy,sx = bslc
        y0,y1 = sy.start, sy.stop
        x0,x1 = sx.start, sx.stop

        subpsf = tim.getPsf().mogAt((x0+x1)/2., (y0+y1)/2.)
        subwcs = ShiftedWcs(tim.getWcs(), x0, y0)

        ###
        if ii < 25:
            subh,subw = subimg.shape
            rwcs = TractorWCSWrapper(subwcs, subw,subh)
            rerun = '301'
            resams = []
            for band in ['z','r']:
                resam = np.zeros_like(subimg)
                nresam = np.zeros(subimg.shape, int)
                s1s = []
                for run,camcol,field in rcfs:
                    #print 'Run, camcol, field', run, camcol, field
                    sdss.retrieve('frame', run, camcol, field=field, band=band,
                                  rerun=rerun)
                    frame = sdss.readFrame(run, camcol, field, band)
                    #print 'Got frame', frame
                    sw,sh = 2048,1489
                    ast = frame.getAsTrans()
                    swcs = AsTransWrapper(ast, sw,sh)
                    try:
                        Yo,Xo,Yi,Xi,nil = resample_with_wcs(rwcs, swcs, [], 3)
                    except OverlapError:
                        continue
                    simg = frame.getImage()
                    sh,sw = simg.shape
                    
                    resam[Yo,Xo] += simg[Yi,Xi]
                    nresam[Yo,Xo] += 1
    
                    sdss.retrieve('psField', run, camcol, field=field, band=band,
                                  rerun=rerun)
                    psfield = sdss.readPsField(run, camcol, field)
                    bandnum = band_index(band)
                    iv = frame.getInvvar(psfield, bandnum, ignoreSourceFlux=True,
                                         constantSkyAt=(sh/2,sw/2))
                    #print 'invvar', iv
                    s1s.append(np.sqrt(1./iv))
                resam /= nresam
                s1 = np.median(s1s)
                resams.append((nresam, resam, band, s1))
                if band == 'z':
                    z1 = s1
                
            imsdss = dict(interpolation='nearest', origin='lower', cmap='gray',
                          vmin=-3.*z1, vmax=5.*z1)
            plt.clf()
            plt.subplot(2,3,1)
            imshow(subimg, **ima)
            plt.title('DECam z')
            plt.subplot(2,3,4)
            imshow(subimg, **imsdss)
            plt.title('DECam z')
            for i,(nresam, resam, band, s1) in enumerate(resams):
                if np.all(nresam == 0):
                    continue
                mn,mx = [np.percentile(resam[nresam>0], p) for p in [25,98]]
                plt.subplot(2,3, 2+i)
                imshow(resam, **ima)
                plt.title('SDSS %s' % band)
                plt.subplot(2,3, 5+i)
                imshow(resam, **imsdss)
                plt.title('SDSS %s' % band)
            ps.savefig()

        subtim = Image(data=subimg, invvar=subiv, psf=subpsf, wcs=subwcs,
                       sky=tim.getSky(), photocal=tim.getPhotoCal())
        #subtr = ChattyTractor([subtim], blobsrcs[b])
        subtr = Tractor([subtim], blobsrcs[b])
        subtr.modtype = np.float64
        subtr.freezeParam('images')
        subtr.catalog.thawAllRecursive()

        print 'Calling ceres optimization on subimage of size', subtim.shape,
        print 'and', len(blobsrcs[b]), 'sources'
        print 'Fitting params:'
        subtr.printThawedParams()

        submod = subtr.getModelImage(0)
        subchi = (subtim.getImage() - submod) * np.sqrt(subiv)

        if ii < 25:
            # plt.clf()
            # plt.subplot(2,2,1)
            # imshow(subtim.getImage(), **ima)
            # plt.subplot(2,2,2)
            # imshow(subiv, **imx)
            # plt.subplot(2,2,3)
            # imshow(submod, **ima)
            # plt.subplot(2,2,4)
            # imshow(subchi, **imchi)
            # ps.savefig()
            plt.clf()
            plt.subplot(2,3,1)
            imshow(subtim.getImage(), **ima)
            plt.subplot(2,3,2)
            imshow(submod, **ima)
            plt.subplot(2,3,3)
            imshow(subchi, **imchi)
            plt.subplot(2,3,4)
            sqimshow(subtim.getImage(), **imsq)
            plt.subplot(2,3,5)
            sqimshow(submod, **imsq)
            ps.savefig()

        
        radec0.extend([(src.getPosition().ra, src.getPosition().dec)
                       for src in bsrcs])
        
        subtr._ceres_opt()

        radec1.extend([(src.getPosition().ra, src.getPosition().dec)
                       for src in bsrcs])
        
        submod = subtr.getModelImage(0)
        subchi = (subtim.getImage() - submod) * np.sqrt(subiv)

        # plt.clf()
        # plt.subplot(2,2,1)
        # imshow(subtim.getImage(), **ima)
        # plt.subplot(2,2,2)
        # imshow(subiv, **imx)
        # plt.subplot(2,2,3)
        # imshow(submod, **ima)
        # plt.subplot(2,2,4)
        # imshow(subchi, **imchi)
        # ps.savefig()

        if ii < 25:
            plt.clf()
            plt.subplot(2,3,1)
            imshow(subtim.getImage(), **ima)
            plt.subplot(2,3,2)
            imshow(submod, **ima)
            plt.subplot(2,3,3)
            imshow(subchi, **imchi)
            plt.subplot(2,3,4)
            sqimshow(subtim.getImage(), **imsq)
            plt.subplot(2,3,5)
            sqimshow(submod, **imsq)
            ps.savefig()
        
    radec0 = np.array(radec0)
    radec1 = np.array(radec1)

    plt.clf()
    plt.plot(3600. * (radec1[:,0] - radec0[:,0]),
             3600. * (radec1[:,1] - radec0[:,1]), 'b.')
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.axis([-0.6,0.6,-0.6,0.6])
    ps.savefig()
    
             
    tflux = np.array([sum(b.getFlux(tim.filter)
                          for b in src.getBrightnesses())
                      for src in cat])
    tmag = -2.5 * (np.log10(   tflux) - 9.)

    plt.clf()
    lo,hi = 10,25
    plt.plot(smag, tmag, 'b.', alpha=0.5)
    plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    plt.xlabel('%s z (mag)' % catname)
    plt.ylabel('DECam z (mag)')
    plt.axis([hi,lo,hi,lo])
    plt.title('Tractor forced photometry of DECam data')
    ps.savefig()
