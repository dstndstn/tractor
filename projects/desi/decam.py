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


def gauss2d(X, mu, sigma):
    N,two = X.shape
    assert(two == 2)
    C,two = mu.shape
    assert(two == 2)
    rtn = np.zeros((N,C))
    for c in range(C):
        e = -np.sum((X - mu[c,:])**2, axis=1) / (2.*sigma[c]**2)
        I = (e > -700)  # -> ~ 1e-304
        rtn[I,c] = 1./(2.*pi*sigma[c]**2) * np.exp(e[I])
    return rtn

def em_step(X, weights, mu, sigma, background, B):
    '''
    mu: shape (C,2) or (2,)
    sigma: shape (C,) or scalar
    weights: shape (C,) or 1.
    C: number of Gaussian components

    X: (N,2)
    '''
    mu_orig = mu

    mu = np.atleast_2d(mu)
    sigma = np.atleast_1d(sigma)
    weights = np.atleast_1d(weights)
    weights /= np.sum(weights)

    print '    em_step: weights', weights, 'mu', mu, 'sigma', sigma, 'background fraction', B
    # E:
    # fg = p( Y, Z=f | theta ) = p( Y | Z=f, theta ) p( Z=f | theta )
    fg = gauss2d(X, mu, sigma) * (1. - B) * weights
    # fg shape is (N,C)
    # bg = p( Y, Z=b | theta ) = p( Y | Z=b, theta ) p( Z=b | theta )
    bg = background * B
    assert(all(np.isfinite(fg.ravel())))
    assert(all(np.isfinite(np.atleast_1d(bg))))
    # normalize:
    sfg = np.sum(fg, axis=1)
    # fore = p( Z=f | Y, theta )
    fore = fg / (sfg + bg)[:,np.newaxis]
    # back = p( Z=b | Y, theta )
    back = bg / (sfg + bg)
    assert(all(np.isfinite(fore.ravel())))
    assert(all(np.isfinite(back.ravel())))

    # M:
    # maximize mu, sigma:
    #mu = np.sum(fore[:,np.newaxis] * X, axis=0) / np.sum(fore)
    mu = np.dot(fore.T, X) / np.sum(fore)
    # 2.*sum(fore) because X,mu are 2-dimensional.
    #sigma = np.sqrt(np.sum(fore[:,np.newaxis] * (X - mu)**2) / (2.*np.sum(fore)))
    C = len(sigma)
    for c in range(C):
        sigma[c] = np.sqrt(np.sum(fore[:,c][:,np.newaxis] * (X - mu[c,:])**2)
                           / (2. * np.sum(fore[:,c])))
    #print 'mu', mu, 'sigma', sigma
    if np.min(sigma) == 0:
        return (mu, sigma, B, -1e6, np.zeros(len(X)))
    assert(np.all(sigma > 0))

    # maximize weights:
    weights = np.mean(fore, axis=0)
    weights /= np.sum(weights)

    # maximize B.
    # B = p( Z=b | theta )
    B = np.mean(back)

    # avoid multiplying 0 * -inf = NaN
    I = (fg > 0)
    lfg = np.zeros_like(fg)
    lfg[I] = np.log(fg[I])

    lbg = np.log(bg * np.ones_like(fg))
    lbg[np.flatnonzero(np.isfinite(lbg) == False)] = 0.

    # Total expected log-likelihood
    Q = np.sum(fore*lfg + back[:,np.newaxis]*lbg)

    print 'Fore', fore.shape
    if len(mu_orig.shape) == 1:
        return (1., mu[0,:], sigma[0], B, Q, fore[:,0])
    return (weights, mu, sigma, B, Q, fore)


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

def get_tractor_params(T, cat, pat):
    typemap = { PointSource: 'S', ExpGalaxy: 'E', DevGalaxy: 'D',
                FixedCompositeGalaxy: 'C' }
    T.set(pat % 'type', np.array([typemap[type(src)] for src in cat]))

    T.set(pat % 'ra',  np.array([src.getPosition().ra  for src in cat]))
    T.set(pat % 'dec', np.array([src.getPosition().dec for src in cat]))

    shapeExp = np.zeros((len(T), 3))
    shapeDev = np.zeros((len(T), 3))
    fracDev  = np.zeros(len(T))

    for i,src in enumerate(cat):
        if isinstance(src, ExpGalaxy):
            shapeExp[i,:] = src.shape.getAllParams()
        elif isinstance(src, DevGalaxy):
            shapeDev[i,:] = src.shape.getAllParams()
            fracDev[i] = 1.
        elif isinstance(src, FixedCompositeGalaxy):
            shapeExp[i,:] = src.shapeExp.getAllParams()
            shapeDev[i,:] = src.shapeDev.getAllParams()
            fracDev[i] = src.fracDev.getValue()

    T.set(pat % 'shapeExp', shapeExp)
    T.set(pat % 'shapeDev', shapeDev)
    T.set(pat % 'fracDev', fracDev)
    return

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
        catsources = T
        
    else:
        if not os.path.exists(objfn):
            margin = 5./3600.

            print 'SIP:', sip

            cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
                    'modelflux', 'modelflux_ivar',
                    'psfflux', 'psfflux_ivar',
                    'cmodelflux', 'cmodelflux_ivar',
                    'devflux', 'expflux',
                    'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
                    'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
                    'resolve_status', 'nchild', 'flags', 'objc_flags',
                    'run','camcol','field','id'
                    ]
            objs = read_photoobjs_in_wcs(sip, margin, sdss=sdss, cols=cols)
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
        
            #ellipse=EllipseESoft.fromRAbPhi)

        catsources = objs
        
    print len(cat), 'sources'


    typemap = { PointSource: 'S', ExpGalaxy: 'E', DevGalaxy: 'D',
                FixedCompositeGalaxy: 'C' }
    #T.tractor_type = np.array([typemap[type(src)] for src in cat])

    hdr = fitsio.FITSHDR()
    # Find a source of each type and query its parameter names
    for t,ts in typemap.items():
        for src in cat:
            if type(src) == t:
                print 'Parameters for', t, src
                sc = src.copy()
                #print 'Copy is', sc
                #print type(sc)
                #print dir(sc)
                sc.thawAllRecursive()
                for i,nm in enumerate(sc.getParamNames()):
                    hdr.add_record(dict(name='TR_%s_P%i' % (ts, i), value=nm,
                                        comment='Tractor param name'))

                def flatten_node(node):
                    return reduce(lambda x,y: x+y, [flatten_node(c) for c in node[1:]],
                                  [node[0]])

                tree = getParamTypeTree(sc)
                print 'Source param types:', tree
                types = flatten_node(tree)
                print 'Flat:', types
                for i,t in enumerate(types):
                    hdr.add_record(dict(name='TR_%s_T%i' % (ts, i), value=t.replace("'", '"'),
                                        comment='Tractor param types'))
                break
    print 'Header:', hdr



    # FIXME -- check astrometry
    if True:
        SEcat = fits_table(seobjfn, hdu=2)
        SEcat.ra  = SEcat.alpha_j2000
        SEcat.dec = SEcat.delta_j2000
        SDSScat = fits_table(sdssobjfn)
        I,J,d = match_radec(SEcat.ra, SEcat.dec, SDSScat.ra, SDSScat.dec, 4.0/3600.)
        print len(I), 'matches'

        # The SExtractor catalogs are way wrong
        # plt.clf()
        # plt.plot(3600.*(SEcat.ra[I] - SDSScat.ra[J]), 3600.*(SEcat.dec[I] - SDSScat.dec[J]), 'b.')
        # plt.xlabel('dRA (arcsec)')
        # plt.ylabel('dDec (arcsec)')
        # #plt.axis([-0.6,0.6,-0.6,0.6])
        # plt.title('SE cat RA,Dec')
        # ps.savefig()

        # rr,dd = sip.pixelxy2radec(SEcat.x_image, SEcat.y_image)
        # I,J,d = match_radec(rr, dd, SDSScat.ra, SDSScat.dec, 4.0/3600.)
        # print len(I), 'matches'
        # 
        # # SE x,y coords seem to be = FITS convention (1-indexed)
        # 
        # plt.clf()
        # plt.plot(3600.*(rr[I] - SDSScat.ra[J]), 3600.*(dd[I] - SDSScat.dec[J]), 'b.')
        # plt.xlabel('dRA (arcsec)')
        # plt.ylabel('dDec (arcsec)')
        # plt.title('SIP(SE cat x,y) - SDSS RA,Dec')
        # plt.axhline(0, color='k', alpha=0.5)
        # plt.axvline(0, color='k', alpha=0.5)
        # plt.axis('scaled')
        # plt.axis([-0.6,0.6,-0.6,0.6])
        # ps.savefig()

        ok,xx,yy = sip.radec2pixelxy(SDSScat.ra, SDSScat.dec)
        I,J,d = match_xy(xx, yy, SEcat.x_image, SEcat.y_image, 5.0)
        print len(I), 'matches'

        # plt.clf()
        # plt.plot(xx[I] - SEcat.x_image[J], yy[I] - SEcat.y_image[J], 'b.')
        # plt.xlabel('dx (pix)')
        # plt.ylabel('dy (pix)')
        # plt.axhline(0, color='k', alpha=0.5)
        # plt.axvline(0, color='k', alpha=0.5)
        # plt.title('SE cat x,y - SIP(SDSS)')
        # plt.axis('scaled')
        # plt.axis([-2,2,-2,2])
        # ps.savefig()

        # Push the SE x,y -> rr,dd coords back through to xx',yy'
        # Differences are milli-pixels
        # ok,xx,yy = sip.radec2pixelxy(rr, dd)
        # plt.clf()
        # plt.plot(SEcat.x_image - xx, SEcat.y_image - yy, 'b.')
        # plt.xlabel('dx (pix)')
        # plt.ylabel('dy (pix)')
        # plt.title('SIP xy->rd->xy residuals')
        # ps.savefig()



        # Try doing the EM tune-up thing.
        #plt.plot(xx[I] - SEcat.x_image[J], yy[I] - SEcat.y_image[J], 'b.')

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


        plt.clf()
        plt.scatter(xx[I] - SEcat.x_image[J], yy[I] - SEcat.y_image[J], c=fore,
                    edgecolors='none', alpha=0.5)
        plt.colorbar()
        plt.plot(mu[0], mu[1], 'kx', ms=15, mew=3)
        angle = np.linspace(0, 2.*np.pi, 200)
        plt.plot(mu[0] + sigma * np.sin(angle), mu[1] + sigma * np.cos(angle), 'k-')
        plt.xlabel('dx (pix)')
        plt.ylabel('dy (pix)')
        plt.axhline(0, color='k', alpha=0.5)
        plt.axvline(0, color='k', alpha=0.5)
        plt.axis('scaled')
        plt.axis([-2,2,-2,2])
        ps.savefig()

        crpix = sip.crpix
        print 'CRPIX', crpix
        sip.set_crpix((crpix[0] - mu[0], crpix[1] - mu[1]))

        ok,xx,yy = sip.radec2pixelxy(SDSScat.ra, SDSScat.dec)
        I,J,d = match_xy(xx, yy, SEcat.x_image, SEcat.y_image, 5.0)
        print len(I), 'matches'

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

        if False:
            # Show zoom-ins of images + sources
            ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=tim.zr[0], vmax=tim.skyval + 10.*tim.sig1)
            plt.clf()
            plt.imshow(tim.getImage(), **ima)
            ax = plt.axis()
            plt.plot(SEcat.x_image-1, SEcat.y_image-1, 'o', mec='r', mfc='none', ms=12)
            plt.axis(ax)
            ps.savefig()
            #plt.plot(xx, yy, 'g+', mfc='none')
            plt.plot(xx, yy, '+', mec=(0,1,0), mfc='none', ms=12)
            ps.savefig()
            plt.axis([0,1000,0,500])
            ps.savefig()
            plt.axis([1000,2000,0,500])
            ps.savefig()
            plt.axis([0,1000,500,1000])
            ps.savefig()
            plt.axis([1000,2000,500,1000])
            ps.savefig()


    # Check out the PSF models.
    psf = tim.psf

    print 'PSF', psf
    print type(psf)
    print 'PSF scale:', psf.scale
    print 'PSF sampling:', psf.sampling

    # H,W = tim.shape
    # xx,yy = np.meshgrid(np.linspace(0, W, 20), np.linspace(0, H, 40))
    # mogs = [psf.mogAt(x,y) for x,y in zip(xx.ravel(), yy.ravel())]
    # for i,nm in enumerate(mogs[0].getParamNames()):
    #     pp = [mog.getParams()[i] for mog in mogs]
    #     pp = np.array(pp).reshape(xx.shape)
    #     plt.clf()
    #     plt.imshow(pp,  interpolation='nearest', origin='lower')
    #     plt.colorbar()
    #     plt.title('Parameter: %s' % nm)
    #     ps.savefig()
    # for i,spl in enumerate(psf.splines):
    #     plt.clf()
    #     #vals = spl(xx,yy)
    #     vals = spl(np.linspace(0, W, 20), np.linspace(0, H, 40))
    #     #print 'xx,yy,vals', xx.shape, yy.shape, vals.shape
    #     print 'vals', vals.shape
    #     plt.imshow(vals, interpolation='nearest', origin='lower')
    #     plt.colorbar()
    #     plt.title('Parameter %i' % i)
    #     ps.savefig()

    #flux = np.array([tim.photocal.brightnessToCounts(src.getBrightness())
    #                 for src in cat])
    flux = SEcat.flux_auto

    ima = dict(interpolation='nearest', origin='lower', cmap='gray')
               #vmin=tim.zr[0], vmax=tim.skyval + 20.*tim.sig1)

    H,W = tim.shape
    S = 20
    I = np.argsort(-flux)
    while False:
    #for i in I[:20]:
        #x,y = tim.wcs.positionToPixel(cat[i].getPosition())
        x,y = SEcat.x_image[i] - 1, SEcat.y_image[i] - 1
        if x < S or y < S or x+S >= W or y+S >= H:
            continue
        ix,iy = int(np.round(x)), int(np.round(y))
        subim = tim.getImage()[iy-S:iy+S+1, ix-S:ix+S+1]
        ext = [ix-S, ix+S, iy-S, iy+S]
        subiv = tim.getInvvar()[iy-S:iy+S+1, ix-S:ix+S+1]

        print 'Subimage max', subim.max()

        psfimg = psf.instantiateAt(ix, iy, nativeScale=True)

        subim /= subim.sum()
        psfimg /= psfimg.sum()
        mx = max(subim.max(), psfimg.max())
        ima.update(vmin=-0.05*mx, vmax=mx)

        sh,sw = subim.shape
        #subrgb = np.zeros((h,w,3))
        subrgb = plt.cm.gray((subim - ima['vmin']) / (ima['vmax'] - ima['vmin']))
        print 'subrgb', subrgb.shape
        bad = (1,0,0)
        for i in range(3):
            subrgb[:,:,i][subiv == 0] = bad[i]

        plt.clf()
        plt.subplot(2,4,1)
        #plt.imshow(subim, extent=ext, **ima)
        plt.imshow(subrgb, extent=ext, **ima)
        ax = plt.axis()
        plt.plot(x, y, 'o', mfc='none', mec='r', ms=12)
        plt.axis(ax)
        plt.title('Image')
        plt.subplot(2,4,2)
        #plt.imshow(psfimg, **ima)
        #plt.title('Image')

        pixpsf = PixelizedPSF(psfimg)
        patch = pixpsf.getPointSourcePatch(x - (ix-S), y - (iy-S))
        print 'Patch', patch.x0, patch.y0, patch.patch.shape

        psfsub = np.zeros_like(subim)
        patch.addTo(psfsub)
        psfsub /= psfsub.sum()
        print 'Pix sum', patch.patch.sum()
        print 'Pix max', psfsub.max()

        plt.imshow(psfsub, **ima)
        plt.title('PSF pix')

        mog = psf.mogAt(x, y)
        print 'PSF MOG:', mog
        patch = mog.getPointSourcePatch(x, y)
        print 'Patch', patch.x0, patch.y0, patch.patch.shape
        patch.x0 -= (ix - S)
        patch.y0 -= (iy - S)
        psfg = np.zeros_like(subim)
        patch.addTo(psfg)
        psfg /= psfg.sum()

        print 'Gauss sum', patch.patch.sum()
        print 'Gauss max', psfg.max()

        # Re-fit the PSF image as MoG
        # im = np.maximum(psfimg, 0)
        # PS = im.shape[0]
        # xm,ym = -(PS/2), -(PS/2)
        # K = 3
        # w,mu,var = em_init_params(K, None, None, None)
        # em_fit_2d(im, xm, ym, w, mu, var)
        # #print 'Re-fit params:', w, mu, var
        # repsf = GaussianMixturePSF(w, mu, var)
        # print 'Re-fit MOG:', repsf
        # patch = repsf.getPointSourcePatch(x, y)
        # print 'Patch', patch.x0, patch.y0, patch.patch.shape
        # patch.x0 -= (ix - S)
        # patch.y0 -= (iy - S)
        # psfg2 = np.zeros_like(subim)
        # patch.addTo(psfg2)
        # psfg2 /= psfg2.sum()

        plt.subplot(2,4,3)
        plt.imshow(psfg, **ima)
        plt.title('PSF Gaussian')


        plt.subplot(2,4,7)
        plt.imshow(-(subim - psfsub), interpolation='nearest', origin='lower',
                   cmap='RdBu')
        plt.title('Image - PsfPix')

        plt.subplot(2,4,8)
        plt.imshow(-(subim - psfg), interpolation='nearest', origin='lower',
                   cmap='RdBu')
        plt.title('Image - PsfG')
                   
        ima.update(vmin=0, vmax=np.sqrt(mx * 1.05))

        plt.subplot(2,4,5)
        plt.imshow(np.sqrt(subim + 0.05*mx), extent=ext, **ima)
        plt.title('sqrt Image')
        plt.subplot(2,4,6)
        #plt.imshow(np.sqrt(psfimg + 0.05*mx), **ima)
        plt.imshow(np.sqrt(psfsub + 0.05*mx), **ima)
        plt.title('sqrt PSF pix')
        
        ps.savefig()
        

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
                             for b in src.getBrightnesses())
                         for src in cat])

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

    # Number of sigma to render profiles to
    minsig = 0.1
    # -> min surface brightness
    minsb = tim.sig1 * minsig
    print 'Sigma1:', tim.sig1, 'minsig', minsig, 'minsb', minsb

    minradius = 3
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
    
    print 'Opt forced photom...'
    R = tractor.optimize_forced_photometry(
        minsb=minsb,
        shared_params=False, wantims=False, fitstats=True, variance=True,
        use_ceres=True, BW=8,BH=8)
    flux_iv,fs = R.IV, R.fitstats

    flux = np.array([sum(b.getFlux(tim.filter)
                         for b in src.getBrightnesses())
                     for src in cat])
    mag,dmag = NanoMaggies.fluxErrorsToMagErrors(flux, flux_iv)

    T = catsources.copy()
    T.set('sdss_%s_nanomaggies' % tim.filter, sdssflux)
    T.set('decam_%s_nanomaggies' % tim.filter, flux)
    T.set('decam_%s_nanomaggies_invvar' % tim.filter, flux_iv)
    T.set('decam_%s_mag'  % tim.filter, mag)
    T.set('decam_%s_mag_err'  % tim.filter, dmag)

    fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
    for k in fskeys:
        x = getattr(fs, k)
        x = np.array(x).astype(np.float32)
        T.set('decam_%s_%s' % (tim.filter, k), x.astype(np.float32))
                    
    smag = -2.5 * (np.log10(sdssflux) - 9.)
    tmag = mag
    tflux = flux

    get_tractor_params(T, cat, 'tractor_%s_init')

    if secat:
        T.writeto(basefn + '-se-phot-1.fits', header=hdr)
    else:
        T.writeto(basefn + '-phot-1.fits', header=hdr)
    
    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod1 = mod
    chi1 = chi

    fitsio.write(basefn + '-mod1.fits', mod1, clobber=True)
    fitsio.write(basefn + '-chi1.fits', chi1, clobber=True)

    I = np.argsort(-np.abs(chi1.ravel()))
    print 'Worst chi pixels:', chi1.flat[I[:20]]
    
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
    plt.xlabel('%s %s flux (nanomaggies)' % (catname, tim.filter))
    plt.ylabel('DECam %s flux (nanomaggies)' % tim.filter)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.axis([1e-1, 1e5, 1e-1, 1e5])
    plt.title('Tractor forced photometry of DECam data')
    ps.savefig()

    plt.clf()
    lo,hi = 10,25
    plt.plot(smag, tmag, 'b.', alpha=0.5)
    plt.plot([lo,hi], [lo,hi], 'k-', alpha=0.5)
    plt.xlabel('%s %s (mag)' % (catname, tim.filter))
    plt.ylabel('DECam %s (mag)' % tim.filter)
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

    ndecamonly = 0

    for ii,b in enumerate(np.argsort(-blobchisq)):
        bslc = blobslices[b]
        bsrcs = blobsrcs[b]

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

        doplot = (ii < 25) or (len(blobsrcs[b]) == 0 and ndecamonly < 25)

        ###
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

        if doplot:
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

    mod = tractor.getModelImage(0)
    chi = (tim.data - mod) * tim.getInvError()
    mod2 = mod
    chi2 = chi

    fitsio.write(basefn + '-mod2.fits', mod2, clobber=True)
    fitsio.write(basefn + '-chi2.fits', chi2, clobber=True)

    # FIXME -- We use the flux inverse-variance from before --
    # incorrect if the profiles have changed.

    flux = np.array([sum(b.getFlux(tim.filter)
                         for b in src.getBrightnesses())
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
    get_tractor_params(T, cat, 'tractor_%s')

    if secat:
        T.writeto(basefn + '-se-phot-2.fits', header=hdr)
    else:
        T.writeto(basefn + '-phot-2.fits', header=hdr)

    plt.clf()
    imshow(mod, **ima)
    plt.title('Tractor model image: Full opt')
    ps.savefig()

    plt.clf()
    imshow(-chi, **imchi)
    plt.title('Image - Model chi: Full opt')
    ps.savefig()
             
    tflux = np.array([sum(b.getFlux(tim.filter)
                          for b in src.getBrightnesses())
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
