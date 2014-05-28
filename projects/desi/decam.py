if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.util import *
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
    sat = hdr['SATURATE']
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
        
    saturated = (img >= sat)
    
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
    invvar[saturated] = 0.

    #print 'Image: max', img.max()
    
    wcs = ConstantFitsWcs(sip)
    if x0 or y0:
        wcs.setX0Y0(x0,y0)
        
    tim = Image(img, invvar=invvar, wcs=wcs,
                photocal=LinearPhotoCal(zpscale, band=filt),
                psf=psf, sky=ConstantSky(sky), name=name)
    tim.zr = [sky-3.*sig1, sky+5.*sig1]
    tim.orig_img = orig_img
    tim.filter = filt
    tim.sig1 = sig1
    tim.sky1 = sky1
    tim.zp = zp
    return tim
    

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser('%prog [options]')
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
    decbase = 'proc/20130330/C01/zband/DECam_00192399.01.p.w'
    tim = read_decam_image(decbase, slc=slc)
    print 'Got', tim, tim.shape

    basefn = 'decam-00192399.01'
    picklefn = basefn + '.pickle'
    secatfn = decbase + '.morph.fits'
    sdssobjfn = basefn + '-sdss.fits'
    seobjfn = basefn + '-se.fits'

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

    sdss = DR9(basedir='dr9')
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

    else:
        if not os.path.exists(objfn):
            margin = 5./3600.
            objs = read_photoobjs_in_wcs(sip, margin, sdss=sdss)
            objs.writeto(objfn)
        else:
            objs = fits_table(objfn)

            print len(objs), 'SDSS photoObjs'
            r0,r1,d0,d1 = sip.radec_bounds()
            cat = get_tractor_sources_dr9(
                None, None, None, objs=objs, sdss=sdss,
                radecroi=[r0,r1,d0,d1], bands=['z'],
                nanomaggies=True, fixedComposites=True,
                useObjcType=True)

    print len(cat), 'sources'
        
    # very rough FWHM
    psfim = tim.getPsf().getPointSourcePatch(0., 0.)
    mx = psfim.patch.max()
    area = np.sum(psfim.patch > 0.5*mx)
    fwhm = 2. * np.sqrt(area / np.pi)
    print 'PSF FWHM', fwhm
    psfsig = fwhm/2.35
    
    #ph,pw = psfim.shape
    #pimg = (np.exp(-0.5 * (np.arange(pw)-pw/2)**2 / psfsig**2)[np.newaxis,:] *
    #        np.exp(-0.5 * (np.arange(ph)-ph/2)**2 / psfsig**2)[:,np.newaxis])
    psfnorm = np.sqrt(gaussian_filter(psfim.patch, psfsig).max())
    
    # run rough detection alg on image
    detimg = gaussian_filter(tim.getImage(), psfsig)
    nsigma = 5.
    thresh = nsigma * tim.sig1 / psfnorm
    hot = (detimg > thresh)
    # expand by fwhm
    hot = binary_dilation(hot, iterations=int(fwhm))

    def imshow(img, **kwargs):
        #x = plt.imshow(np.rot90(img), **kwargs)
        x = plt.imshow(img.T, **kwargs)
        return x
        
    #slc = slice(0,2000),slice(0,1000)
    #slc = slice(0,2000),slice(1000,2000)
    
    # plt.clf()
    # plt.imshow(tim.orig_img, interpolation='nearest', origin='lower',
    #            vmin=tim.zr[0]+med, vmax=tim.zr[1]+med)
    # plt.title('Original image')
    # ps.savefig()

    ima = dict(interpolation='nearest', origin='lower',
               vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
    
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
    imshow(hot, interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Detected')
    ps.savefig()
    
    plt.clf()
    imx = dict(interpolation='nearest', origin='lower')
    imshow(tim.getInvError() == 0, cmap='gray', **imx)
    plt.title('Masked')
    ps.savefig()

    sys.exit(0)
    
    # plt.clf()
    # imshow(tim.data[slc], **ima)
    # plt.title('Image')
    # ps.savefig()

    # plt.clf()
    # imshow(tim.getInvError(), interpolation='nearest', origin='lower',
    #        vmin=0, vmax=1.1/tim.sig1)
    # plt.colorbar()
    # plt.title('Inverse-error')
    # ps.savefig()
    # 
    # plt.clf()
    # imshow(tim.getInvError()[slc], interpolation='nearest', origin='lower',
    #        vmin=0, vmax=1.1/tim.sig1)
    # plt.colorbar()
    # plt.title('Inverse-error')
    # ps.savefig()

        
    
    tractor = Tractor([tim], cat)

    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod0 = mod
    chi0 = chi

    fitsio.write(basefn + '-image.fits', tim.data, clobber=True)
    fitsio.write(basefn + '-mod0.fits', mod0, clobber=True)
    
    plt.clf()
    imshow(mod, **ima)
    plt.title('Tractor model image: Initial')
    ps.savefig()

    # plt.clf()
    # imshow(mod[slc], **ima)
    # plt.title('Tractor model image: Initial')
    # ps.savefig()

    imchi = dict(interpolation='nearest', origin='lower',
                 vmin=-5, vmax=5, cmap='RdBu')

    imchi2 = dict(interpolation='nearest', origin='lower',
                  vmin=-100, vmax=100, cmap='RdBu')

    plt.clf()
    imshow(-chi, **imchi)
    plt.title('Image - Model chi: Initial')
    ps.savefig()

    plt.clf()
    imshow(-chi, **imchi2)
    plt.title('Image - Model chi: Initial')
    ps.savefig()
    
    # plt.clf()
    # imshow(-chi[slc], **imchi)
    # plt.title('Image - Model chi: Initial')
    # ps.savefig()
    
    tractor.freezeParamsRecursive('*')
    tractor.thawPathsTo(tim.filter)

    print 'Fitting params:'
    tractor.printThawedParams()
    
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
    
    fitsio.write(basefn + '-mod1.fits', mod1, clobber=True)

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
    
    # plt.clf()
    # imshow(mod[slc], **ima)
    # plt.title('Tractor model image: Forced photom')
    # ps.savefig()
    # 
    # plt.clf()
    # imshow(-chi[slc], **imchi)
    # plt.title('Image - Model chi: Forced photom')
    # ps.savefig()
    
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
                slc = slice(y0, y0+256), slice(x0, x0+256)
                imsa = ima.copy()
                imsa.update(extent=[x0,x0+256,y0,y0+256])
    
                print 'Slice', x0, y0
                I = np.argsort(-np.abs(chi0.ravel()))
                print 'Largest chi0:', chi0.flat[I[:20]]
                I = np.argsort(-np.abs(chi1.ravel()))
                print 'Largest chi1:', chi1.flat[I[:20]]
                
                plt.clf()
                plt.subplot(2,3,1)
                plt.imshow(tim.data[slc], **imsa)
                plt.subplot(2,3,2)
                plt.imshow(mod0[slc], **imsa)
                plt.subplot(2,3,3)
                plt.imshow(mod1[slc], **imsa)
                plt.subplot(2,3,4)
                n1,b,p = plt.hist(np.clip(chi0[slc], -6,6).ravel(), 50,
                                  range=(-6,6),
                                  log=True, histtype='step', color='r')
                n2,b,p = plt.hist(np.clip(chi1[slc], -6,6).ravel(), 50,
                                  range=(-6,6),
                                  log=True, histtype='step', color='b')
                plt.axis([-6.1, 6.1, 0.1, 1.2*max(max(n1),max(n2))])
                plt.subplot(2,3,5)
                plt.imshow(-chi0[slc], **imchi)
                plt.title('chi2: %g' % np.sum(chi0[slc]**2))
                plt.subplot(2,3,6)
                plt.imshow(-chi1[slc], **imchi)
                plt.title('chi2: %g' % np.sum(chi1[slc]**2))
                ps.savefig()
