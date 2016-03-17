from __future__ import print_function

import pylab as plt

import numpy as np
import fitsio
from collections import Counter

from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence

from tractor.cfht import CfhtLinearPhotoCal, parse_head_file
from tractor.image import Image
from tractor.sky import ConstantSky
from tractor.psf import GaussianMixturePSF
from tractor.psfex import PixelizedPsfEx
from tractor.wcs import ConstantFitsWcs, RaDecPos
from tractor.brightness import NanoMaggies
from tractor.galaxy import GalaxyShape, ExpGalaxy, DevGalaxy
from tractor.pointsource import PointSource
from tractor import Tractor

from tractor.source_extractor import get_se_modelfit_cat

def gim2d_catalog(cat, band):
    '''
    http://irsa.ipac.caltech.edu/data/COSMOS/tables/morphology/cosmos_morph_zurich_colDescriptions.html
    '''
    '''
    ACS_MU_CLASS 	float 	  	Type of object.
    1 = galaxy
    2 = star
    3 = spurious
    '''
    '''
    ACS_CLEAN 	float 	  	Object useable flag.
    0 = do not use this object
    1 = use this object
    '''
    '''
    FLUX_GIM2D 	float 	counts 	GIM2D total flux
    R_GIM2D 	float 	arcseconds 	GIM2D psf-convolved half-light radius of object
    ELL_GIM2D 	float 	  	GIM2D ellipticity = 1-b/a of object
    PA_GIM2D 	float 	degrees 	GIM2D position angle of object - cw from +y-axis
    DX_GIM2D 	float 	arcseconds 	x-offset of GIM2D-model center from ACS-coordinate center
    DY_GIM2D 	float 	arcseconds 	y-offset of GIM2D-model center from ACS-coordinate center
    SERSIC_N_GIM2D 	float 	  	GIM2D Sersic index
    R_0P5_GIM2D 	float 	arcseconds 	GIM2D half-light radius of object without PSF convolution

    TYPE 	float 	  	ZEST Type CLASS
    1 = Early type
    2 = Disk
    3 = Irregular Galaxy
    9 = no classification
    '''

    '''
    BULG 	float 	  	ZEST "Bulgeness" CLASS - only for Type 2 (disk) galaxies.
    0 = bulge dominated galaxy
    1,2 = intermediate-bulge galaxies
    3 = pure disk galaxy
    9 = no classification
    '''

    '''
    STELLARITY 	float 	  	Visual Stellarity flag.

    0 if ACS_CLASS_STAR<0.6 (object is ASSUMED to be a galaxy; no visual inspection)
    0 if ACS_CLASS_STAR>=0.6 AND object visually identified as a galaxy.
    1 if ACS_CLASS_STAR>=0.6 AND visually identified as a star.
    2 if ACS_CLASS_STAR>=0.8 (object is assumed to be a star and was not visually inspected)
    3 if ACS_CLASS_STAR<0.6 but object is visually identified as a star (e.g. saturated star, etc)

    JUNKFLAG 	float 	  	
    0 = good object
    1 = spurious
    '''
    
    print('Classifications:', Counter(cat.type).most_common())
    
    cat.is_galaxy = (cat.stellarity == 0)
    srcs = []
    for t in cat:
        pos = RaDecPos(t.ra, t.dec)
        bright = NanoMaggies(**{band:NanoMaggies.magToNanomaggies(t.acs_mag_auto)})
        shape = GalaxyShape(t.r_0p5_gim2d, 1. - t.ell_gim2d, 90. + t.pa_gim2d)

        is_galaxy = (t.is_galaxy * (shape.re >= 0 ) * (shape.ab <= 1.) *
                     (shape.phi > -999))
        
        if is_galaxy and t.type == 1:
            # deV
            src = DevGalaxy(pos, bright, shape)
        elif is_galaxy and t.type == 2:
            # exp
            src = ExpGalaxy(pos, bright, shape)
        else:
            src = PointSource(pos, bright)
        srcs.append(src)
    return srcs
                

if __name__ == '__main__':

    ps = PlotSequence('cfht')

    if False:
        # Don't start looking at the ACS image.  Don't.
        imgfn = 'cfht/acs_I_030mas_065_sci.VISRES.fits'
        wtfn = 'cfht/acs_I_030mas_065_wht.VISRES.fits'
        flagfn = 'cfht/acs_I_030mas_065_flg.VISRES.fits'

        catfn = 'cfht/acs_I_030mas_065_sci.VISRES.ldac'
        cat = fits_table(catfn, hdu=2)
        cat.ra = cat.alpha_j2000
        cat.dec = cat.delta_j2000
    
        img,imghdr = fitsio.read(imgfn, ext=ext, header=True)
        print('Read image', img.shape, img.dtype)
        img = img.astype(np.float32)
        H,W = img.shape
        import sys
        sys.exit(0)
    
    imgfn = 'cfht/1624827p.fits'
    headfn = 'cfht/1624827p.head'
    psffn = 'cfht/1624827p.psf'
    ext = 14

    # wget http://irsa.ipac.caltech.edu/data/COSMOS/tables/morphology/cosmos_morph_zurich_1.0.tbl
    # text2fits.py -H "SequentialID RA DEC CAPAK_ID CAPAK_RA CAPAK_DEC ACS_MAG_AUTO ACS_MAGERR_AUTO ACS_X_IMAGE ACS_Y_IMAGE ACS_XPEAK_IMAGE ACS_YPEAK_IMAGE ACS_ALPHAPEAK_ ACS_DELTAPEAK_ ACS_A_IMAGE ACS_B_IMAGE ACS_THETA_IMAGE ACS_ELONGATION ACS_CLASS_STAR ACS_IDENT ACS_SE ACS_MU_CLASS ACS_OVERLAP ACS_NEARSTAR ACS_MASK ACS_MASKED ACS_CLEAN ACS_UNIQUE GG M20 CC AA R20 R50 R80 RPET FLAGRPET FLUX_GIM2D LE_FLUX_GIM2D UE_FLUX_GIM2D R_GIM2D LE_R_GIM2D UE_R_GIM2D ELL_GIM2D LE_ELL_GIM2D UE_ELL_GIM2D PA_GIM2D LE_PA_GIM2D UE_PA_GIM2D DX_GIM2D LE_DX_GIM2D UE_DX_GIM2D DY_GIM2D LE_DY_GIM2D UE_DY_GIM2D SERSIC_N_GIM2D LE_N_GIM2D UE_N_GIM2D R_0P5_GIM2D CHI_GIM2D ITER_GIM2D PC_1 PC_2 PC_3 TYPE BULG IRRE ELLI STELLARITY JUNKFLAG ACSTile" -f dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddds cosmos_morph_zurich_1.0.tbl cosmos_morph_zurich_1.0.fits -S 4
    
    catfn = 'cfht/cosmos_morph_zurich_1.0.fits'
    cat = fits_table(catfn)
    print('Read', len(cat), 'catalog entries')
    cat = cat[cat.junkflag == 0]
    print(len(cat), 'not junk')
    
    img,imghdr = fitsio.read(imgfn, ext=ext, header=True)
    print('Read image', img.shape, img.dtype)
    img = img.astype(np.float32)
    H,W = img.shape
    
    band = imghdr['FILTER'][0]
    print('Band:', band)
    photocal = CfhtLinearPhotoCal(imghdr, band)
    
    headers = parse_head_file(headfn)
    print('Read headers:', len(headers))
    
    wcshdr = headers[ext-1]
    wcshdr['IMAGEW'] = W
    wcshdr['IMAGEH'] = H
    wcs = wcs_pv2sip_hdr(wcshdr)
    print('WCS pixel scale:', wcs.pixel_scale())
    
    ok,xx,yy = wcs.radec2pixelxy(cat.ra, cat.dec)
    print('Ext', ext, 'x range', int(xx.min()), int(xx.max()), 'y range', int(yy.min()), int(yy.max()))

    cat.xx = xx
    cat.yy = yy
    
    # # Estimate per-pixel noise via Blanton's 5-pixel MAD
    slice1 = (slice(0,-5,10),slice(0,-5,10))
    slice2 = (slice(5,None,10),slice(5,None,10))
    mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
    sig1 = 1.4826 * mad / np.sqrt(2.)
    print('sig1 estimate:', sig1)
    inverr = np.ones_like(img) / sig1

    # PSF...
    #psfex = PixelizedPsfEx(psffn)
    psfex = GaussianMixturePSF(1., 0., 0., 4., 4., 0.)
    #psfex = GaussianMixturePSF(1., 0., 0., 1., 1., 0.)

    sky = np.median(img)
    img -= sky
    
    tim = Image(data=img, inverr=inverr, wcs=ConstantFitsWcs(wcs),
                photocal=photocal,
                psf=psfex, sky=ConstantSky(0.))

    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-2.*sig1, vmax=10.*sig1)

    plt.clf()
    plt.hist((img * inverr).ravel(), 100, range=(-5,5))
    plt.xlabel('Image sigma')
    ps.savefig()
    
    plt.clf()
    plt.imshow(img, **ima)
    ax = plt.axis()
    plt.plot(xx-1, yy-1, 'r.')
    plt.axis(ax)
    ps.savefig()

    x0,y0 = 0,0
    x1,y1 = 600,600
    
    plt.axis([x0, x1, y0, y1])
    ps.savefig()

    if False:
        plt.clf()
        plt.hist(cat.acs_mag_auto, 100)
        plt.xlabel('ACS MAG_AUTO')
        ps.savefig()

        plt.clf()
        R = cat.r_0p5_gim2d
        R = R[R >= 0]
        plt.hist(R, 100)
        plt.xlabel('r_0p5_gim2d')
        ps.savefig()

        plt.clf()
        ab = 1. - cat.ell_gim2d
        ab = ab[(ab >= 0) * (ab < 99)]
        plt.hist(ab, 100)
        plt.xlabel('ab gim2d')
        ps.savefig()

    subtim = tim.subimage(x0, x1, y0, y1)

    #mn,mx = np.percentile(subtim.getImage().ravel(), [25, 98])
    # plt.clf()
    # plt.imshow(subtim.getImage(), interpolation='nearest', origin='lower',
    #            cmap='gray', vmin=mn, vmax=mx)
    # ps.savefig()

    cat.cut((cat.xx >= x0) * (cat.xx <= x1) * (cat.yy >= y0) * (cat.yy <= y1))
    print('Cut to', len(cat), 'sources in subimage')
    
    srcs = gim2d_catalog(cat, band)

    tractor = Tractor([subtim], srcs)

    if False:
        plt.clf()
        plt.imshow(img, **ima)
        ax = plt.axis()
        I = np.flatnonzero(cat.type == 1)
        plt.plot(cat.xx[I], cat.yy[I], 'r.')
        I = np.flatnonzero(cat.type == 2)
        plt.plot(cat.xx[I], cat.yy[I], 'b.')
        I = np.flatnonzero(cat.type == 3)
        plt.plot(cat.xx[I], cat.yy[I], 'm.')
        plt.axis(ax)
        ps.savefig()
    
        plt.clf()
        plt.imshow(subtim.getImage(), **ima)
        ax = plt.axis()
        plt.plot(cat.xx, cat.yy, 'r.')
        for t in cat:
            if t.type in [1,2]:
                r = t.r_0p5_gim2d
                if r > 0:
                    rtxt = '%.1f' % r
                else:
                    rtxt = 'X'
                ab = 1.-t.ell_gim2d
                if ab >= 0 and ab < 1:
                    abtxt = '%.2f' % ab
                else:
                    abtxt = 'X'
                
                plt.text(t.xx, t.yy, 'T%i, r %s ab %s' % (t.type, rtxt, abtxt),
                         color='r')
        plt.axis(ax)
        ps.savefig()
    
    tractor.freezeParam('images')
    for src in srcs:
        src.freezeAllBut('brightness')

    from tractor.ceres_optimizer import CeresOptimizer
    B = 8
    opti = CeresOptimizer(BW=B, BH=B)
    tractor.optimizer = opti

    print('Forced phot...')
    kwa = {}
    R = tractor.optimize_forced_photometry(shared_params=False, variance=True, **kwa)

    print('R:', R, dir(R))
    
    mod = tractor.getModelImage(0)
        
    plt.clf()
    plt.imshow(mod, **ima)
    plt.title('Forced photometry model')
    ps.savefig()

    plt.clf()
    plt.imshow(subtim.getImage(), **ima)
    plt.title('CFHT data')
    ps.savefig()

    print('Tim photocal:', tim.photocal)
    print('Subtim photocal:', subtim.photocal)
    
    # # Order by flux
    I = np.argsort([-src.getBrightness().getFlux(band) for src in srcs])
    for i in I:
        src = srcs[i]
        print('Source:', src)
        print('-> counts', subtim.photocal.brightnessToCounts(src.getBrightness()))

    # flux in nanomaggies
    flux = np.array([src.getBrightness().getFlux(band) for src in srcs])
    fluxiv = R.IV
    mag, magerr = NanoMaggies.fluxErrorsToMagErrors(flux, fluxiv)

    typemap = { ExpGalaxy:'E', DevGalaxy:'D', PointSource:'P' }
    
    cat.tractor_type = np.array([typemap[type(src)] for src in srcs])
    cat.set('cfht_forced_mag_%s'    % band, mag)
    cat.set('cfht_forced_magerr_%s' % band, magerr)
    cat.writeto('cfht-forced.fits')

    
    plt.clf()
    plt.plot(cat.acs_mag_auto, mag, 'b.')
    plt.xlabel('ACS I-band (mag)')
    plt.ylabel('CFHT %s-band forced phot (mag)' % band)
    plt.title('CFHT forced phot')
    ps.savefig()
