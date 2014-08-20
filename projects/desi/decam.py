
'''
    2014-August run #1: g,r,z coverage:
    240 < RA < 251, 6 < dec < 11



    -exposure/CCD database?
    -calibration database?  Or file tree?
    -tractor on tiles
    -images -- CP or Nugent calibrations?
    -astrometry w/ SDSS index files
    -catalogs -- 
       -- source extractor: simple (union of single exposures?  coadd?)
       -- source extractor with galaxy photometry ("")
       -- SDSS
       -- Pan-STARRS
       -- SED-matched filter
    -segmentation
    -brightest to faintest?  largest residuals to smallest?
    -add new sources?  Point source <-> galaxy swaps?
    -moving sources?

    -catalog formats -- internal / output
    -output values -- uncertainties; metrics

'''

    



if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

from astrometry.libkd.spherematch import *
from astrometry.util.util import *
from astrometry.util.ttime import *
from astrometry.util.resample import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.sdss.fields import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

from scipy.ndimage.filters import *
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.interpolation import shift

from scipy.interpolate import RectBivariateSpline

#from .em import *
from em import *

def tweak_astrometry(SEcat, SDSScat, sip, ps):
    SEcat = fits_table(seobjfn, hdu=2)
    SEcat.ra  = SEcat.alpha_j2000
    SEcat.dec = SEcat.delta_j2000
    SDSScat = fits_table(sdssobjfn)
    ok,xx,yy = sip.radec2pixelxy(SDSScat.ra, SDSScat.dec)
    I,J,d = match_xy(xx, yy, SEcat.x_image, SEcat.y_image, 5.0)
    print len(I), 'matches'

    # Try doing the EM tune-up thing.

    X = np.vstack((xx[I] - SEcat.x_image[J], yy[I] - SEcat.y_image[J])).T
    weights = [1.]
    sigma = 2.
    mu = np.array([0.,0.])
    bg = 0.01
    B = 0.5

    for i in range(20):
        weights,mu,sigma,B,Q,fore = em_step(X, weights, mu, sigma, bg, B)
    print 'Sigma', sigma
    print 'Mu', mu
    print 'B', B

    if ps:
        plt.clf()
        plt.scatter(xx[I] - SEcat.x_image[J], yy[I] - SEcat.y_image[J], c=fore,
                    edgecolors='none', alpha=0.5)
        plt.colorbar()
        plt.plot(mu[0], mu[1], 'kx', ms=15, mew=3)
        angle = np.linspace(0, 2.*np.pi, 200)
        plt.plot(mu[0] + sigma * np.sin(angle),
                 mu[1] + sigma * np.cos(angle), 'k-')
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.axhline(0, color='k', alpha=0.5)
        plt.axvline(0, color='k', alpha=0.5)
        plt.axis('scaled')
        plt.axis([-2,2,-2,2])
        ps.savefig()

    # UPDATE 'sip'
    crpix = sip.crpix
    print 'CRPIX', crpix
    sip.set_crpix((crpix[0] - mu[0], crpix[1] - mu[1]))

    ok,xx,yy = sip.radec2pixelxy(SDSScat.ra, SDSScat.dec)
    I,J,d = match_xy(xx, yy, SEcat.x_image, SEcat.y_image, 5.0)
    print len(I), 'matches'

    if ps:
        plt.clf()
        plt.plot(xx[I] - SEcat.x_image[J], yy[I] - SEcat.y_image[J], 'b.')
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.axhline(0, color='k', alpha=0.5)
        plt.axvline(0, color='k', alpha=0.5)
        plt.title('SE cat x,y - SIP(SDSS)')
        plt.axis('scaled')
        plt.axis([-2,2,-2,2])
        ps.savefig()

def set_source_radii(psf, cat, minsb):
    # Render PSF profile to determine good source radii
    R = 100
    print 'PSF type:', type(psf)
    psf.radius = R
    pat = psf.getPointSourcePatch(0., 0.)
    print 'PSF patch: x0,y0', pat.x0,pat.y0, 'shape', pat.patch.shape
    assert(pat.x0 == pat.y0)
    assert(pat.x0 == -R)
    # max of +dx, -dx, +dy, -dy directions.
    psfprofile = reduce(np.maximum, [pat.patch[R, R:],
                                     pat.patch[R, R::-1],
                                     pat.patch[R:, R],
                                     pat.patch[R::-1, R]])
    # Set minimum flux to correspond to minimum radius
    defaultflux = minsb / psfprofile[minradius]
    print 'Setting default flux', defaultflux
    
    # Set source radii based on initial fluxes
    rad = np.zeros(len(cat), int)
    for r,pro in enumerate(psfprofile):
        flux = minsb / pro
        rad[sdssflux > flux] = r
    # Set radii
    for i in range(len(cat)):
        src = cat[i]
        # set fluxes
        b = src.getBrightness()
        if b.getFlux(tim.filter) <= defaultflux:
            b.setFlux(tim.filter, defaultflux)
        R = max(minradius, rad[i])
        if isinstance(src, PointSource):
            src.fixedRadius = R
        elif isinstance(src, (HoggGalaxy, FixedCompositeGalaxy)):
            src.halfsize = R

def detection_map(psf, img, inverr, sig1, nsigma=4, dilate_fwhm=1.):
    # rough FWHM
    psfim = psf.getPointSourcePatch(0., 0.)
    mx = psfim.patch.max()
    area = np.sum(psfim.patch > 0.5*mx)
    fwhm = 2. * np.sqrt(area / np.pi)
    print 'PSF FWHM', fwhm
    psfsig = fwhm/2.35
    psfnorm = np.sqrt(gaussian_filter(psfim.patch, psfsig).max())
    print 'PSF norm:', psfnorm

    # run rough detection alg on image
    img = img.copy()
    if inverr is not None:
        img[inverr == 0] = 0.
    detimg = gaussian_filter(img, psfsig) / psfnorm**2
    thresh = nsigma * sig1 / psfnorm
    hot = (detimg > thresh)
    # expand by fwhm
    hot = binary_dilation(hot, iterations=int(fwhm * dilate_fwhm))
    return hot

def sky_subtract(img, cellsize, fullimg=None, x0=0, y0=0, gradient=True):
    # Remove x/y gradient estimated in ~"cellsize"-pixel^2 squares
    if fullimg is None:
        fullimg = img
    H,W = img.shape
    fH,fW = fullimg.shape
    nx = int(np.ceil(float(fW) / cellsize))
    ny = int(np.ceil(float(fH) / cellsize))
    if gradient:
        xx = np.linspace(0, fW, nx+1)
        yy = np.linspace(0, fH, ny+1)
    else:
        # Make half-size boxes at the edges to improve the spatial
        # response
        xx = np.linspace(0., fW, nx+1)
        xx += (xx[1]-xx[0])/2.
        xx = np.append(0, xx.astype(int))
        xx[-1] = fW
        print 'xx', xx
        yy = np.linspace(0., fH, ny+1)
        yy += (yy[1]-yy[0])/2.
        yy = np.append(0, yy.astype(int))
        yy[-1] = fH
        
    subs = np.zeros((len(yy)-1, len(xx)-1))
    for iy,(ylo,yhi) in enumerate(zip(yy, yy[1:])):
        for ix,(xlo,xhi) in enumerate(zip(xx, xx[1:])):
            subim = fullimg[ylo:yhi, xlo:xhi]
            subs[iy,ix] = np.median(subim.ravel())

    if gradient:
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
        # bx = (X[1] * (x0 + np.arange(W)))
        # by = (X[2] * (y0 + np.arange(H)))
        # print 'Background x contribution:', bx.shape, bx.min(), bx.max()
        # print 'Background y contribution:', by.shape, by.min(), by.max()
    else:
        sx,sy = (xx[1:] + xx[:-1])/2., (yy[1:]+yy[:-1])/2.
        spl = RectBivariateSpline(sx, sy, subs.T)
        bg = spl(x0 + np.arange(W), y0 + np.arange(H)).T

    return bg

def read_decam_image(basefn, skysubtract=True, slc=None):
    '''
    slc: slice of image to read
    '''
    imgfn  = basefn + '.fits'
    maskfn = basefn + '.bpm.fits'
    psffn  = basefn + '.cat.psf'
    print 'Reading', imgfn, 'and', maskfn

    wcsfn = os.path.join('data/decam/astrom', os.path.basename(imgfn).replace('.fits','.wcs'))
    print 'Reading WCS from', wcsfn

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
    
    #sip = Sip(imgfn)
    sip = Sip(wcsfn)
    print 'SIP', sip
    print 'RA,Dec bounds', sip.radec_bounds()
    
    H,W = img.shape
    print 'Reading PSF', psffn
    psf = PsfEx(psffn, W, H, scale=False, nx=9, ny=17)
    print 'Got PSF', psf
    if x0 or y0:
        psf = ShiftedPsf(psf, x0, y0)
        
    if skysubtract:
        bg = sky_subtract(img, 512, fullimg=fullimg, x0=0,y0=0)
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
    tim.skyval = sky
    tim.zp = zp
    tim.ozpscale = orig_zpscale
    return tim
    

def imshow(img, **kwargs):
    x = plt.imshow(img.T, **kwargs)
    plt.xticks([])
    plt.yticks([])
    return x

def sqimshow(img, **kwa):
    mn = kwa.pop('vmin')
    mx = kwa.pop('vmax')
    imshow(np.sqrt(np.maximum(0, img - mn)), vmin=0, vmax=np.sqrt(mx-mn), **kwa)




        
if __name__ == '__main__':
    import optparse
    import logging
    parser = optparse.OptionParser('%prog [options] [base input filename]')
    parser.add_option('--se', action='store_true')
    parser.add_option('--x0', type=int, help='Read sub-image', default=0)
    parser.add_option('--y0', type=int, help='Read sub-image', default=0)
    parser.add_option('-W', type=int, help='Read sub-image', default=0)
    parser.add_option('-H', type=int, help='Read sub-image', default=0)
    parser.add_option('-v', dest='verbose', default=False, action='store_true')
    opt,args = parser.parse_args()

    lvl = logging.WARN
    if opt.verbose:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    
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

    ps = PlotSequence('decam')
    ps.format = '%03i'
        
    #if True:
    for timi in range(25):
        X = unpickle_from_file('subtim-%04i.pickle' % timi)
        subtim = X['tim']
        bsrcs = X['cat']

        subtr = Tractor([subtim], bsrcs)
        subtr.modtype = np.float64
        subtr.freezeParam('images')
        subtr.catalog.thawAllRecursive()

        print 'subtr', subtr
        if len(bsrcs) == 0:
            continue
            
        #subtr.catalog.freezeAllParams()
        #subtr.catalog.thawParam(0)
        
        if False:
            derivs = subtr._getOneImageDerivs(0)
            pnames = subtr.getParamNames()
            ss = subtr.getStepSizes()
            
            pp = subtr.getParams()
            p0 = np.array(pp)
            scales = 10. * np.array(subtr.getStepSizes())
    
            p0 -= scales
            params = np.ones_like(p0)
            scaler = ScaledTractor(subtr, p0, scales)
            #scaler.setParams(params)
            sderivs = scaler._getOneImageDerivs(0)
    
            for (i,x0,y0,deriv),(si,sx0,sy0,sderiv) in zip(derivs, sderivs):
                assert(i == si)
    
                plt.clf()
                plt.subplot(2,3,1)
                mx = np.abs(deriv).max()
                plt.imshow(deriv, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='RdBu')
                plt.colorbar()
                plt.subplot(2,3,4)
                mx = np.abs(sderiv).max()
                plt.imshow(sderiv, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='RdBu')
                #plt.title('Scaled deriv for %s; %f' % (pnames[si], scales[si]))
                plt.title('%g' % scales[si])
                plt.colorbar()
    
                plt.suptitle('Deriv for %s' % pnames[i])
    
                chi0 = subtr.getChiImage(0)
                subtr.setParam(i, pp[i] + ss[i])
                chi1 = subtr.getChiImage(0)
                subtr.setParam(i, pp[i])
                mx = np.abs(chi0).max()
                plt.subplot(2,3,2)
                plt.imshow(chi0, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='RdBu')
                plt.colorbar()
                plt.subplot(2,3,3)
                dchi = chi1 - chi0
                mx = np.abs(dchi).max()
                plt.imshow(dchi, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='RdBu')
                plt.colorbar()
    
    
                chi0 = scaler.getChiImage(0)
                params[i] += 0.1
                scaler.setParams(params)
                chi1 = scaler.getChiImage(0)
                params[i] -= 0.1
                scaler.setParams(params)
                mx = np.abs(chi0).max()
                plt.subplot(2,3,5)
                plt.imshow(chi0, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='RdBu')
                plt.colorbar()
                plt.subplot(2,3,6)
                dchi = chi1 - chi0
                mx = np.abs(dchi).max()
                plt.imshow(dchi, interpolation='nearest', origin='lower',
                           vmin=-mx, vmax=mx, cmap='RdBu')
                plt.colorbar()
                
                ps.savefig()
    
            sys.exit(0)

        p0 = subtr.getParams()

        print 'Calling ceres optimization on subimage of size', subtim.shape,
        print 'and', len(bsrcs), 'sources'
        #print 'Fitting params:'
        #subtr.printThawedParams()
        #subtr._ceres_opt()

        print
        print '----------Dynamic LSQR ---------------------'
        print
        subtr.setParams(p0)

        ss = subtr.getDynamicScales()
        print 'Dynamic scales:', ss
        subtr.catalog.setAllStepSizes(ss)
        ss2 = subtr.catalog.getAllStepSizes()
        print 'ss2', ss2
        ss = np.array(ss)
        ss2 = np.array(ss2)
        print 'diff', ss - ss2

        I = np.flatnonzero(ss != ss2)
        nm = subtr.catalog.getParamNames()
        print 'Different:', [nm[i] for i in I]

        assert(np.all(np.array(ss) == np.array(ss2)))
        
        while True:
            dlnp,X,alpha,var = subtr.optimize(variance=True, shared_params=False,
                                              alphas=[0.01, 0.1, 1., 2., 4., 10.])
            print 'dlnp', dlnp
            if dlnp < 1e-3:
                break

        dyn_lsqr_lnp = subtr.getLogProb()

        p0 = subtr.getParams()
        lnp0 = subtr.getLogProb()
        print 'Check vars:'
        for i,(nm,val,vvar) in enumerate(zip(subtr.getParamNames(),
                                             subtr.getParams(), var)):
            lnpx = subtr.getLogProb()
            assert(lnpx == lnp0)
            dvar = np.sqrt(vvar)
            subtr.setParam(i, p0[i] + dvar)
            lnp1 = subtr.getLogProb()
            subtr.setParam(i, p0[i] - dvar)
            lnp2 = subtr.getLogProb()
            subtr.setParam(i, p0[i])
            print '  ', nm, val, '+-', dvar, '-> dlnp', (lnp1-lnp0), (lnp2-lnp0)
        
        print
        print '------------------- Ceres ---------------------'
        print
        subtr.setParams(p0)
        R = subtr._ceres_opt(variance=True, max_iterations=100)

        print 'Report:', R['full_report']

        var = R['variance']
        print 'Params:'
        for nm,val,vvar in zip(subtr.getParamNames(),
                               subtr.getParams(), var):
            print '  ', nm, '=', val, '+-', np.sqrt(vvar)


        # print
        # print '------------------- variance=True, scaled=False ----------------'
        # print
        # subtr.setParams(p0)
        # 
        # R = subtr._ceres_opt(variance=True, scaled=False)
        # var = R['variance']
        # print 'Params:'
        # for nm,val,vvar in zip(subtr.getParamNames(),
        #                        subtr.getParams(), var):
        #     print '  ', nm, '=', val, '+-', np.sqrt(vvar)
        # 
        # 
        # print
        # print
        # print '------------------- variance=True, numeric=True -----------------'
        # print
        # subtr.setParams(p0)
        # 
        # print 'Ceres with numeric differentiation:'
        # R = subtr._ceres_opt(variance=True, numeric=True)
        # var = R['variance']
        # print 'Params:'
        # for nm,val,vvar in zip(subtr.getParamNames(),
        #                        subtr.getParams(), var):
        #     print '  ', nm, '=', val, '+-', np.sqrt(vvar)
        # print

        ceres_lnp = subtr.getLogProb()
        
        if True:
            pp = subtr.getParams()
            lnp0 = subtr.getLogProb()
            print 'Check vars:'
            for i,(nm,val,vvar) in enumerate(zip(subtr.getParamNames(),
                                                 subtr.getParams(), var)):
                lnpx = subtr.getLogProb()
                assert(lnpx == lnp0)
                dvar = np.sqrt(vvar)
                subtr.setParam(i, pp[i] + dvar)
                lnp1 = subtr.getLogProb()
                subtr.setParam(i, pp[i] - dvar)
                lnp2 = subtr.getLogProb()
                subtr.setParam(i, pp[i])
                print '  ', nm, val, '+-', dvar, '-> dlnp', (lnp1-lnp0), (lnp2-lnp0)

        print
        print '------------------- LSQR ---------------------'
        print
        subtr.setParams(p0)

        while True:
            dlnp,X,alpha,var = subtr.optimize(variance=True, shared_params=False,
                                              alphas=[0.01, 0.1, 1., 2., 4., 10.])
            print 'dlnp', dlnp
            if dlnp < 1e-3:
                break

        lsqr_lnp = subtr.getLogProb()

        p0 = subtr.getParams()
        lnp0 = subtr.getLogProb()
        print 'Check vars:'
        for i,(nm,val,vvar) in enumerate(zip(subtr.getParamNames(),
                                             subtr.getParams(), var)):
            lnpx = subtr.getLogProb()
            assert(lnpx == lnp0)
            dvar = np.sqrt(vvar)
            subtr.setParam(i, p0[i] + dvar)
            lnp1 = subtr.getLogProb()
            subtr.setParam(i, p0[i] - dvar)
            lnp2 = subtr.getLogProb()
            subtr.setParam(i, p0[i])
            print '  ', nm, val, '+-', dvar, '-> dlnp', (lnp1-lnp0), (lnp2-lnp0)




        print 'Log-probs: ceres', ceres_lnp, 'vs', lsqr_lnp, '(delta: %g)' % (lsqr_lnp - ceres_lnp)

        print 'vs Dynamic LSQR:', dyn_lsqr_lnp, '(delta: %g)' % (dyn_lsqr_lnp - max(lsqr_lnp, ceres_lnp))
                
    sys.exit(0)
    

    tim = read_decam_image(decbase, slc=slc)
    print 'Got', tim, tim.shape

    basefn = os.path.basename(decbase).lower().replace('.p.w', '')
    picklefn = basefn + '.pickle'
    objfn = basefn + '-sdss.fits'
    # SourceExtractor catalog
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
    
    catname = 'SDSS'
    sdss = DR9(basedir='dr9')
    sdss.saveUnzippedFiles('dr9')
    if os.environ.get('BOSS_PHOTOOBJ') is not None:
        print 'Using local tree for SDSS files'
        sdss.useLocalTree()

    # read_source_extractor_catalog(secatfn, tim.zp
        
    sip = tim.wcs.wcs
    if not os.path.exists(objfn):
        margin = 5./3600.
        cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
                'modelflux', 'modelflux_ivar',
                'psfflux', 'psfflux_ivar',
                'cmodelflux', 'cmodelflux_ivar',
                'devflux', 'expflux',
                'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr',
                'phi_dev_deg', 'phi_exp_deg',
                'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr',
                'resolve_status', 'nchild', 'flags', 'objc_flags',
                'run','camcol','field','id' ]
        objs = read_photoobjs_in_wcs(sip, margin, sdss=sdss, cols=cols)
        objs.writeto(objfn)
    else:
        objs = fits_table(objfn)

    print len(objs), 'SDSS photoObjs'
    # FIXME -- RA=0 wrap-around issues
    r0,r1,d0,d1 = sip.radec_bounds()
    cat = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss,
        radecroi=[r0,r1,d0,d1], bands=[tim.filter],
        nanomaggies=True, fixedComposites=True,
        useObjcType=True,
        ellipse=EllipseESoft.fromRAbPhi)
    catsources = objs
    print len(cat), 'sources'

    rcfnum = (objs.run.astype(np.int32) * 10000 +
              objs.camcol.astype(np.int32) * 1000 +
              objs.field)
    rcfnum = np.unique(rcfnum)
    rcfs = zip(rcfnum / 10000, rcfnum % 10000 / 1000, rcfnum % 1000)
    print 'RCF', rcfs

    secat = fits_table(seobjfn, hdu=2,
                       column_map={'alpha_j2000':'ra', 'delta_j2000':'dec'})

    # FIXME -- check astrometry
    tweak_astrometry(secat, objs, sip)

    ima = dict(interpolation='nearest', origin='lower',
               vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
    imx = dict(interpolation='nearest', origin='lower', cmap='gray')
    
    plt.clf()
    imshow(tim.data, **ima)
    plt.title('Image')
    ps.savefig()

    # plt.clf()
    # imshow(hot, **imx)
    # plt.title('Detected')
    # ps.savefig()
    # 
    # plt.clf()
    # imx = dict(interpolation='nearest', origin='lower')
    # imshow(tim.getInvError() == 0, cmap='gray', **imx)
    # plt.title('Masked')
    # ps.savefig()

    for src in cat:
        pos = src.getPosition()
        pos.stepsizes = [1e-5, 1e-5]
    
    tractor = Tractor([tim], cat)

    print 'Rendering initial model image (no optimization)...'
    t0 = Time()
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
    print 'That took', Time()-t0

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
                             for b in src.getBrightnesses()) for src in cat])
    catsources.set('sdss_%s_nanomaggies' % tim.filter, sdssflux)

    # Number of sigma to render profiles to
    minsig = 0.1
    minradius = 3
    # -> min surface brightness
    minsb = tim.sig1 * minsig
    print 'Sigma1:', tim.sig1, 'minsig', minsig, 'minsb', minsb
    set_source_radii(tim.getPsf(), cat, minsb, minradius)

    print 'Opt forced photom...'
    R = tractor.optimize_forced_photometry(
        minsb=minsb,
        shared_params=False, wantims=False, fitstats=True, variance=True,
        use_ceres=True, BW=8,BH=8)
    flux_iv,fs = R.IV, R.fitstats

    T,hdr = prepare_fits_catalog(cat, flux_iv, catsources.copy(), None,
                             [tim.filter], fs)
    #get_tractor_fits_values(T, cat, 'tractor_%s_init')
    T.writeto(basefn + '-phot-1.fits', header=hdr)
        
    smag = -2.5 * (np.log10(sdssflux) - 9.)
    tmag = T.get('decam_%s_mag' % tim.filter)
    tflux = T.get('decam_%s_nanomaggies' % tim.filter)
    
    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod1 = mod
    chi1 = chi

    fitsio.write(basefn + '-mod1.fits', mod1, clobber=True)
    fitsio.write(basefn + '-chi1.fits', chi1, clobber=True)

    plt.clf()
    imshow(mod, **ima)
    plt.title('Tractor model image: Forced photom')
    ps.savefig()

    plt.clf()
    imshow(-chi, **imchi)
    plt.title('Image - Model chi: Forced photom')
    ps.savefig()

    # plt.clf()
    # imshow(-chi, **imchi2)
    # plt.title('Image - Model chi: Forced photom')
    # ps.savefig()
    
    # plt.clf()
    # lo,hi = -1e-1, 1e5
    # plt.plot(sdssflux, tflux, 'b.', alpha=0.5)
    # plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    # plt.xlabel('%s %s flux (nanomaggies)' % (catname, tim.filter))
    # plt.ylabel('DECam %s flux (nanomaggies)' % tim.filter)
    # plt.xscale('symlog')
    # plt.yscale('symlog')
    # plt.axis([1e-1, 1e5, 1e-1, 1e5])
    # plt.title('Tractor forced photometry of DECam data')
    # ps.savefig()

    plt.clf()
    lo,hi = 10,25
    plt.plot(smag, tmag, 'b.', alpha=0.5)
    plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    plt.xlabel('%s %s (mag)' % (catname, tim.filter))
    plt.ylabel('DECam %s (mag)' % tim.filter)
    plt.axis([hi,lo,hi,lo])
    plt.title('Tractor forced photometry of DECam data')
    ps.savefig()

    # Run detection algorithm on image
    hot = detection_map(tim.getPsf(), tim.getImage(), tim.getInvError(),
                        tim.sig1)
    # Run detection alg on model image as well
    modhot = detection_map(tim.getPsf(), mod1, None, tim.sig1)
    uhot = np.logical_or(hot, modhot)
    
    # plt.clf()
    # imshow(uhot, interpolation='nearest', origin='lower', cmap='gray')
    # plt.title('Union Detected')
    # ps.savefig()

    blobs,nblobs = label(uhot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)

    # Find the sources *within* each blob.
    wcs = tim.getWcs()
    xy = np.array([wcs.positionToPixel(src.getPosition()) for src in cat])
    xy = np.round(xy).astype(int)
    x = xy[:,0]
    y = xy[:,1]
    
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
        # sources within this blob's rectangular bounding-box
        I = np.flatnonzero((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
        if len(I):
            # sources within this blob proper
            I = I[blobs[y[I],x[I]] == (b+1)]

        if len(I):
            blobsrcs.append([cat[i] for i in I])
        else:
            # No sources within blob... DECam-only sources
            blobsrcs.append([])
    blobchisq = np.array(blobchisq)

    radec0 = []
    radec1 = []

    imsq = dict(interpolation='nearest', origin='lower',
                vmin=tim.zr[0], vmax=25.*tim.zr[1], cmap='gray')

    ndecamonly = 0

    for ii,b in enumerate(np.argsort(-blobchisq)):
        bslc = blobslices[b]
        bsrcs = blobsrcs[b]

        print
        print 'Blob', ii, 'of', len(blobchisq), 'index', b
        print 

        print 'blob slice:', bslc

        subiv = tim.getInvvar()[bslc]
        subiv[blobs[bslc] != (b+1)] = 0.
        subimg = tim.getImage()[bslc]
        sy,sx = bslc
        y0,y1 = sy.start, sy.stop
        x0,x1 = sx.start, sx.stop
        subpsf = tim.getPsf().mogAt((x0+x1)/2., (y0+y1)/2.)
        subwcs = ShiftedWcs(tim.getWcs(), x0, y0)

        doplot = (ii < 25) or (len(bsrcs) == 0 and ndecamonly < 25)

        if doplot:
            if ii >= 25:
                ndecamonly += 1
            subh,subw = subimg.shape
            rwcs = TractorWCSWrapper(subwcs, subw,subh)
            rerun = '301'
            resams = []
            for band in ['g', 'r', 'z']:
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
            plt.title('DECam %s' % tim.filter)
            plt.subplot(2,3,2)
            imshow(subimg, **imsdss)
            plt.title('DECam %s'% tim.filter)
            for i,(nresam, resam, band, s1) in enumerate(resams):
                if np.all(nresam == 0):
                    continue
                mn,mx = [np.percentile(resam[nresam>0], p) for p in [25,98]]
                plt.subplot(2,3, 4+i)
                #imshow(resam, **ima)
                #plt.title('SDSS %s' % band)
                imshow(resam, **imsdss)
                plt.title('SDSS %s' % band)
            ps.savefig()

        subtim = Image(data=subimg, invvar=subiv, psf=subpsf, wcs=subwcs,
                       sky=tim.getSky(), photocal=tim.getPhotoCal())

        pfn = 'subtim-%04i.pickle' % ii
        pickle_to_file(dict(tim=subtim, cat=bsrcs, bslc=bslc), pfn)
        print 'Wrote', pfn
        
        subtr = Tractor([subtim], bsrcs)
        subtr.modtype = np.float64
        subtr.freezeParam('images')
        subtr.catalog.thawAllRecursive()

        print 'Calling ceres optimization on subimage of size', subtim.shape,
        print 'and', len(bsrcs), 'sources'
        print 'Fitting params:'
        subtr.printThawedParams()

        submod = subtr.getModelImage(0)
        subchi = (subtim.getImage() - submod) * np.sqrt(subiv)

        if doplot:
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

        if len(subtr.getParams()) == 0:
            continue
            
        radec0.extend([(src.getPosition().ra, src.getPosition().dec)
                       for src in bsrcs])

        R2 = subtr._ceres_opt(variance=True, scale_columns=False)
        var2 = R2['variance']
        print 'Params2:'
        for nm,val,vvar in zip(subtr.getParamNames(),
                               subtr.getParams(), var2):
            print '  ', nm, '=', val, '+-', np.sqrt(vvar)

        #subtr._ceres_opt()
        R = subtr._ceres_opt(variance=True)
        var = R['variance']
        print 'Params:'
        for nm,val,vvar in zip(subtr.getParamNames(),
                               subtr.getParams(), var):
            print '  ', nm, '=', val, '+-', np.sqrt(vvar)

        print
        print 'Ceres with numeric differentiation:'
        R = subtr._ceres_opt(variance=True, numeric=True)
        var = R['variance']
        print 'Params:'
        for nm,val,vvar in zip(subtr.getParamNames(),
                               subtr.getParams(), var):
            print '  ', nm, '=', val, '+-', np.sqrt(vvar)
        print

            
        p0 = subtr.getParams()
        lnp0 = subtr.getLogProb()
        print 'Check vars:'
        for i,(nm,val,vvar) in enumerate(zip(subtr.getParamNames(),
                                             subtr.getParams(), var)):
            lnpx = subtr.getLogProb()
            assert(lnpx == lnp0)
            dvar = np.sqrt(vvar)
            subtr.setParam(i, p0[i] + dvar)
            lnp1 = subtr.getLogProb()
            subtr.setParam(i, p0[i] - dvar)
            lnp2 = subtr.getLogProb()
            subtr.setParam(i, p0[i])
            print '  ', nm, val, '+-', dvar, '-> dlnp', (lnp1-lnp0), (lnp2-lnp0)

        radec1.extend([(src.getPosition().ra, src.getPosition().dec)
                       for src in bsrcs])
        
        if ii < 25:
            submod = subtr.getModelImage(0)
            subchi = (subtim.getImage() - submod) * np.sqrt(subiv)
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


        while True:
            dlnp,x,alpha,vars3 = subtr.optimize(
                shared_params=False, variance=True)
            print 'Opt: dlnp', dlnp
            if dlnp < 1e-3:
                break
        print 'Params3:'
        for nm,val,vvar in zip(subtr.getParamNames(),
                               subtr.getParams(), vars3):
            print '  ', nm, '=', val, '+-', np.sqrt(vvar)
            
        subtr.catalog.freezeAllParams()
        vars2 = []
        for i,src in enumerate(bsrcs):
            subtr.catalog.thawParam(i)
            print 'Fitting source', i
            for nm,val in zip(subtr.getParamNames(), subtr.getParams()):
                print '  ', nm, '=', val
            
            while True:
                dlnp,X,alpha,svar = subtr.optimize(
                    shared_params=False, variance=True)
                print 'Opt: dlnp', dlnp
                if svar is None:
                    svar = np.zeros_like(X)
                if dlnp < 1e-3:
                    break
            vars2.append(svar)
            subtr.catalog.freezeParam(i)

        vars2 = np.hstack(vars2)
        subtr.catalog.thawAllParams()
        print 'Params4:'
        for nm,val,vvar in zip(subtr.getParamNames(),
                               subtr.getParams(), vars2):
            print '  ', nm, '=', val, '+-', np.sqrt(vvar)


        p0 = subtr.getParams()
        lnp0 = subtr.getLogProb()
        print 'Check vars 2:'
        for i,(nm,val,vvar) in enumerate(zip(subtr.getParamNames(),
                                             subtr.getParams(), vars2)):
            dvar = np.sqrt(vvar)
            subtr.setParam(i, p0[i] + dvar)
            lnp1 = subtr.getLogProb()
            subtr.setParam(i, p0[i] - dvar)
            lnp2 = subtr.getLogProb()
            subtr.setParam(i, p0[i])
            print '  ', nm, val, '+-', dvar, '-> dlnp', (lnp1-lnp0), (lnp2-lnp0)

            
        if ii < 25:
            submod = subtr.getModelImage(0)
            subchi = (subtim.getImage() - submod) * np.sqrt(subiv)
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

    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod2 = mod
    chi2 = chi

    fitsio.write(basefn + '-mod2.fits', mod2, clobber=True)
    fitsio.write(basefn + '-chi2.fits', chi2, clobber=True)

    # FIXME -- We use the flux inverse-variance from before --
    # incorrect if the profiles have changed.

    flux = np.array([sum(b.getFlux(tim.filter) for b in src.getBrightnesses())
                     for src in cat])
    mag,dmag = NanoMaggies.fluxErrorsToMagErrors(flux, flux_iv)

    T = catsources.copy()
    T.set('sdss_%s_nanomaggies' % tim.filter, sdssflux)
    T.set('decam_%s_nanomaggies' % tim.filter, flux)
    T.set('decam_%s_nanomaggies_invvar' % tim.filter, flux_iv)
    T.set('decam_%s_mag'  % tim.filter, mag)
    T.set('decam_%s_mag_err'  % tim.filter, dmag)

    # fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
    # for k in fskeys:
    #     x = getattr(fs, k)
    #     x = np.array(x).astype(np.float32)
    #     T.set('decam_%s_%s' % (tim.filter, k), x.astype(np.float32))
    get_tractor_fits_values(T, cat, 'tractor_%s')

    T.writeto(basefn + '-phot-2.fits', header=hdr)

    plt.clf()
    imshow(mod, **ima)
    plt.title('Tractor model image: Full opt')
    ps.savefig()

    plt.clf()
    imshow(-chi, **imchi)
    plt.title('Image - Model chi: Full opt')
    ps.savefig()
             
    tflux = np.array([sum(b.getFlux(tim.filter) for b in src.getBrightnesses())
                      for src in cat])
    tmag = -2.5 * (np.log10(   tflux) - 9.)

    plt.clf()
    lo,hi = 10,25
    plt.plot(smag, tmag, 'b.', alpha=0.5)
    plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    plt.xlabel('%s %s (mag)' % (catname, tim.filter))
    plt.ylabel('DECam %s (mag)' % tim.filter)
    plt.axis([hi,lo,hi,lo])
    plt.title('Tractor forced photometry of DECam data')
    ps.savefig()

