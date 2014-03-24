if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import tempfile
import pylab as plt
import numpy as np
import sys
from glob import glob
import logging

import pyfits
import fitsio

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import match_radec
from astrometry.util.util import Sip, anwcs, Tan

from astrometry.util.sdss_radec_to_rcf import *

import tractor
from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *

from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

#from matplotlib.nxutils import points_inside_poly

logger = logging.getLogger('wise')

def read_wise_level1b(basefn, radecroi=None, radecrad=None, filtermap={},
                      nanomaggies=False, mask_gz=False, unc_gz=False,
                      sipwcs=False, constantInvvar=False,
                      roi = None,
                      zrsigs = [-3,10],
                      ):
    intfn  = basefn + '-int-1b.fits'
    maskfn = basefn + '-msk-1b.fits'

    if mask_gz:
        maskfn = maskfn + '.gz'
    uncfn  = basefn + '-unc-1b.fits'
    if unc_gz:
        uncfn = uncfn + '.gz'

    logger.debug('intensity image   %s' % intfn)
    logger.debug('mask image        %s' % maskfn)
    logger.debug('uncertainty image %s' % uncfn)

    if sipwcs:
        wcs = Sip(intfn, 0)
        twcs = tractor.FitsWcs(wcs)
    else:
        twcs = tractor.FitsWcs(intfn)

    # Read enough of the image to get its size
    Fint = fitsio.FITS(intfn)
    H,W = Fint[0].get_info()['dims']

    if radecrad is not None:
        r,d,rad = radecrad
        x,y = twcs.positionToPixel(tractor.RaDecPos(r,d))
        pixrad = rad / (twcs.pixel_scale() / 3600.)
        print 'Tractor WCS:', twcs
        print 'RA,Dec,rad', r,d,rad, 'becomes x,y,pixrad', x,y,pixrad
        roi = (x-pixrad, x+pixrad+1, y-pixrad, y+pixrad+1)

    if radecroi is not None:
        ralo,rahi, declo,dechi = radecroi
        xy = [twcs.positionToPixel(tractor.RaDecPos(r,d))
              for r,d in [(ralo,declo), (ralo,dechi), (rahi,declo), (rahi,dechi)]]
        xy = np.array(xy)
        x0,x1 = xy[:,0].min(), xy[:,0].max()
        y0,y1 = xy[:,1].min(), xy[:,1].max()
        print 'RA,Dec ROI', ralo,rahi, declo,dechi, 'becomes x,y ROI', x0,x1,y0,y1
        roi = (x0,x1+1, y0,y1+1)

    if roi is not None:
        x0,x1,y0,y1 = roi
        x0 = int(np.floor(x0))
        x1 = int(np.ceil (x1))
        y0 = int(np.floor(y0))
        y1 = int(np.ceil (y1))
        roi = (x0,x1, y0,y1)
        # Clip to image size...
        x0 = np.clip(x0, 0, W)
        x1 = np.clip(x1, 0, W)
        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        if x0 == x1 or y0 == y1:
            print 'ROI is empty'
            return None
        assert(x0 < x1)
        assert(y0 < y1)
        #roi = (x0,x1,y0,y1)
        twcs.setX0Y0(x0,y0)

    else:
        x0,x1,y0,y1 = 0,W, 0,H
        roi = (x0,x1,y0,y1)

    ihdr = Fint[0].read_header()
    data = Fint[0][y0:y1, x0:x1]
    logger.debug('Read %s intensity' % (str(data.shape)))
    band = ihdr['BAND']

    F = fitsio.FITS(uncfn)
    assert(F[0].get_info()['dims'] == [H,W])
    unc = F[0][y0:y1, x0:x1]

    F = fitsio.FITS(maskfn)
    assert(F[0].get_info()['dims'] == [H,W])
    mask = F[0][y0:y1, x0:x1]


    # HACK -- circular Gaussian PSF of fixed size...
    # in arcsec 
    fwhms = { 1: 6.1, 2: 6.4, 3: 6.5, 4: 12.0 }
    # -> sigma in pixels
    sig = fwhms[band] / 2.35 / twcs.pixel_scale()
    #print 'PSF sigma', sig, 'pixels'
    tpsf = tractor.NCircularGaussianPSF([sig], [1.])


    filter = 'w%i' % band
    if filtermap:
        filter = filtermap.get(filter, filter)
    zp = ihdr['MAGZP']
    if nanomaggies:
        photocal = tractor.LinearPhotoCal(tractor.NanoMaggies.zeropointToScale(zp),
                                          band=filter)
    else:
        photocal = tractor.MagsPhotoCal(filter, zp)

    #print 'Image median:', np.median(data)
    #print 'unc median:', np.median(unc)

    sky = np.median(data)
    tsky = tractor.ConstantSky(sky)

    name = 'WISE ' + ihdr['FRSETID'] + ' W%i' % band

    # Mask bits, from
    # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4a.html#maskdef
    # 0 from static mask: excessively noisy due to high dark current alone
    # 1 from static mask: generally noisy [includes bit 0]
    # 2 from static mask: dead or very low responsivity
    # 3 from static mask: low responsivity or low dark current
    # 4 from static mask: high responsivity or high dark current
    # 5 from static mask: saturated anywhere in ramp
    # 6 from static mask: high, uncertain, or unreliable non-linearity
    # 7 from static mask: known broken hardware pixel or excessively noisy responsivity estimate [may include bit 1]
    # 8 reserved
    # 9 broken pixel or negative slope fit value (downlink value = 32767)
    # 10 saturated in sample read 1 (down-link value = 32753)
    # 11 saturated in sample read 2 (down-link value = 32754)
    # 12 saturated in sample read 3 (down-link value = 32755)
    # 13 saturated in sample read 4 (down-link value = 32756)
    # 14 saturated in sample read 5 (down-link value = 32757)
    # 15 saturated in sample read 6 (down-link value = 32758)
    # 16 saturated in sample read 7 (down-link value = 32759)
    # 17 saturated in sample read 8 (down-link value = 32760)
    # 18 saturated in sample read 9 (down-link value = 32761)
    # 19 reserved
    # 20 reserved
    # 21 new/transient bad pixel from dynamic masking
    # 22 reserved
    # 23 reserved
    # 24 reserved
    # 25 reserved
    # 26 non-linearity correction unreliable
    # 27 contains cosmic-ray or outlier that cannot be classified (from temporal outlier rejection in multi-frame pipeline)
    # 28 contains positive or negative spike-outlier
    # 29 reserved
    # 30 reserved
    # 31 not used: sign bit

    goodmask = ((mask & sum([1<<bit for bit in [0,1,2,3,4,5,6,7, 9,
                                                10,11,12,13,14,15,16,17,18,
                                                21,26,27,28]])) == 0)
    sigma1 = np.median(unc[goodmask])
    zr = np.array(zrsigs) * sigma1 + sky

    # constant
    cinvvar = np.zeros_like(data)
    cinvvar[goodmask] = 1. / (sigma1**2)
    # varying
    vinvvar = np.zeros_like(data)
    vinvvar[goodmask] = 1. / (unc[goodmask])**2
    
    bad = np.flatnonzero(np.logical_not(np.isfinite(vinvvar)))
    if len(bad):
        vinvvar.flat[bad] = 0.
        cinvvar.flat[bad] = 0.
        data.flat[bad] = sky

    if constantInvvar:
        invvar = cinvvar
    else:
        invvar = vinvvar

    # avoid NaNs
    data[np.logical_not(goodmask)] = sky

    mjd = ihdr['MJD_OBS']
    time = TAITime(None, mjd=mjd)

    tim = tractor.Image(data=data, invvar=invvar, psf=tpsf, wcs=twcs,
                        sky=tsky, photocal=photocal, time=time, name=name, zr=zr,
                        domask=False)
    tim.extent = [x0,x1,y0,y1]
    tim.sigma1 = sigma1
    #tim.roi = roi

    # FIXME
    tim.maskplane = mask
    tim.uncplane = unc
    tim.goodmask = goodmask

    # carry both around for retrofitting
    tim.vinvvar = vinvvar
    tim.cinvvar = cinvvar

    return tim


def read_wise_level3(basefn, radecroi=None, filtermap={},
                     nanomaggies=False):
    intfn = basefn + '-int-3.fits'
    uncfn = basefn + '-unc-3.fits'

    print 'intensity image', intfn
    print 'uncertainty image', uncfn

    P = pyfits.open(intfn)
    ihdr = P[0].header
    data = P[0].data
    print 'Read', data.shape, 'intensity'
    band = ihdr['BAND']

    P = pyfits.open(uncfn)
    uhdr = P[0].header
    unc = P[0].data
    print 'Read', unc.shape, 'uncertainty'

    ''' cov:
    BAND    =                    1 / wavelength band number
    WAVELEN =                3.368 / [microns] effective wavelength of band
    COADDID = '3342p000_ab41'      / atlas-image identifier
    MAGZP   =                 20.5 / [mag] relative photometric zero point
    MEDINT  =      4.0289044380188 / [DN] median of intensity pixels
    '''
    ''' int:
    BUNIT   = 'DN      '           / image pixel units
    CTYPE1  = 'RA---SIN'           / Projection type for axis 1
    CTYPE2  = 'DEC--SIN'           / Projection type for axis 2
    CRPIX1  =          2048.000000 / Axis 1 reference pixel at CRVAL1,CRVAL2
    CRPIX2  =          2048.000000 / Axis 2 reference pixel at CRVAL1,CRVAL2
    CDELT1  =  -0.0003819444391411 / Axis 1 scale at CRPIX1,CRPIX2 (deg/pix)
    CDELT2  =   0.0003819444391411 / Axis 2 scale at CRPIX1,CRPIX2 (deg/pix)
    CROTA2  =             0.000000 / Image twist: +axis2 W of N, J2000.0 (deg)
    '''
    ''' unc:
    FILETYPE= '1-sigma uncertainty image' / product description
    '''

    twcs = tractor.WcslibWcs(intfn)
    print 'WCS', twcs
    #twcs.debug()
    print 'pixel scale', twcs.pixel_scale()

    # HACK -- circular Gaussian PSF of fixed size...
    # in arcsec 
    fwhms = { 1: 6.1, 2: 6.4, 3: 6.5, 4: 12.0 }
    # -> sigma in pixels
    sig = fwhms[band] / 2.35 / twcs.pixel_scale()
    print 'PSF sigma', sig, 'pixels'
    tpsf = tractor.NCircularGaussianPSF([sig], [1.])

    if radecroi is not None:
        ralo,rahi, declo,dechi = radecroi
        xy = [twcs.positionToPixel(tractor.RaDecPos(r,d))
              for r,d in [(ralo,declo), (ralo,dechi), (rahi,declo), (rahi,dechi)]]
        xy = np.array(xy)
        x0,x1 = xy[:,0].min(), xy[:,0].max()
        y0,y1 = xy[:,1].min(), xy[:,1].max()
        print 'RA,Dec ROI', ralo,rahi, declo,dechi, 'becomes x,y ROI', x0,x1,y0,y1

        # Clip to image size...
        H,W = data.shape
        x0 = max(0, min(x0, W-1))
        x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H-1))
        y1 = max(0, min(y1, H))
        print ' clipped to', x0,x1,y0,y1

        data = data[y0:y1, x0:x1]
        unc = unc[y0:y1, x0:x1]
        twcs.setX0Y0(x0,y0)
        print 'Cut data to', data.shape

    else:
        H,W = data.shape
        x0,x1,y0,y1 = 0,W, 0,H

    filter = 'w%i' % band
    if filtermap:
        filter = filtermap.get(filter, filter)
    zp = ihdr['MAGZP']

    if nanomaggies:
        photocal = tractor.LinearPhotoCal(tractor.NanoMaggies.zeropointToScale(zp),
                                          band=filter)
    else:
        photocal = tractor.MagsPhotoCal(filter, zp)

    print 'Image median:', np.median(data)
    print 'unc median:', np.median(unc)

    sky = np.median(data)
    tsky = tractor.ConstantSky(sky)

    sigma1 = np.median(unc)
    zr = np.array([-3,10]) * sigma1 + sky

    name = 'WISE ' + ihdr['COADDID'] + ' W%i' % band

    tim = tractor.Image(data=data, invvar=1./(unc**2), psf=tpsf, wcs=twcs,
                        sky=tsky, photocal=photocal, name=name, zr=zr,
                        domask=False)
    tim.extent = [x0,x1,y0,y1]
    return tim

read_wise_coadd = read_wise_level3
read_wise_image = read_wise_level1b


def get_psf_model(band, pixpsf=False, xy=None, positive=True, cache={}):

    if xy is not None:
        x,y = xy
        gx = np.clip((int(x)/100) * 100 + 50, 50, 950)
        gy = np.clip((int(x)/100) * 100 + 50, 50, 950)
        assert(gx % 100 == 50)
        assert(gy % 100 == 50)
        xy = (gx,gy)

    key = (band, pixpsf, xy, positive)
    if key in cache:
        return cache[key]

    if xy is None:
        psf = pyfits.open('wise-psf-w%i-500-500.fits' % band)[0].data
    else:
        ## ASSUME existence of wise-psf/wise-psf-w%i-%i-%i.fits on a grid 50,150,...,950
        fn = 'wise-psf/wise-psf-w%i-%03i-%03i.fits' % (band, gx, gy)
        psf = pyfits.open(fn)[0].data

    if pixpsf:
        #print 'Read PSF image:', psf.shape, 'range', psf.min(), psf.max()
        if positive:
            psf = np.maximum(psf, 0.)
        psf = PixelizedPSF(psf)
        cache[key] = psf
        return psf

    S = psf.shape[0]
    # number of Gaussian components
    K = 3
    w,mu,sig = em_init_params(K, None, None, None)
    II = psf.copy()
    II = np.maximum(II, 0)
    II /= II.sum()
    xm,ym = -(S/2), -(S/2)
    res = em_fit_2d(II, xm, ym, w, mu, sig)
    if res != 0:
        raise RuntimeError('Failed to fit PSF')
    print 'W1 PSF:'
    print '  w', w
    print '  mu', mu
    print '  sigma', sig
    psf = GaussianMixturePSF(w, mu, sig)
    psf.computeRadius()
    cache[key] = psf
    return psf


def main():
    # from tractor.sdss_galaxy import *
    # import sys
    # 
    # ell = GalaxyShape(10., 0.5, 45)
    # print ell
    # S = 20./3600.
    # dra,ddec = np.meshgrid(np.linspace(-S, S, 200),
    #                        np.linspace(-S, S, 200))
    # overlap = []
    # for r,d in zip(dra.ravel(),ddec.ravel()):
    #     overlap.append(ell.overlapsCircle(r,d, 0.))
    # overlap = np.array(overlap).reshape(dra.shape)
    # print 'overlap', overlap.min(), overlap.max()
    # 
    # import matplotlib
    # matplotlib.use('Agg')
    # import pylab as plt
    # plt.clf()
    # plt.imshow(overlap)
    # plt.savefig('overlap.png')
    # 
    # sys.exit(0)

    from bigboss_test import *

    #filtermap = { 'w1':'i', 'w2':'i', 'w3':'i', 'w4':'i' }
    filtermap = None
    
    import matplotlib
    matplotlib.use('Agg')

    import logging
    import sys
    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    from astrometry.util.util import *
    from astrometry.util.pyfits_utils import fits_table
    from astrometry.libkd.spherematch import *
    import numpy as np

    #bandnums = [1,2,3,4]
    bandnums = [1,]
    bands = ['w%i' % n for n in bandnums]

    (ra0,ra1, dec0,dec1) = radecroi

    # cfht = fits_table('/project/projectdirs/bigboss/data/cs82/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
    # print 'Read', len(cfht), 'sources'
    # # Cut to ROI
    # cfht.ra  = cfht.alpha_j2000
    # cfht.dec = cfht.delta_j2000
    # cfht.cut((cfht.ra > ra0) * (cfht.ra < ra1) * (cfht.dec > dec0) * (cfht.dec < dec1))
    # print 'Cut to', len(cfht), 'objects in ROI.'
    # cfht.cut((cfht.mag_psf < 25.))
    # print 'Cut to', len(cfht), 'bright'
    #srcs = get_cfht_catalog(mags=bands, maglim=25.)

    srcs,T = get_cfht_catalog(mags=['i'] + bands, maglim=25., returnTable=True)
    print 'Got', len(srcs), 'CFHT sources'


    wise = fits_table('/project/projectdirs/bigboss/data/wise/catalogs/wisecat.fits')
    print 'Read', len(wise), 'WISE sources'
    #(ra0,ra1, dec0,dec1) = radecroi
    wise.cut((wise.ra > ra0) * (wise.ra < ra1) * (wise.dec > dec0) * (wise.dec < dec1))
    print 'Cut to', len(wise), 'objects in ROI.'

    I,J,D = match_radec(T.ra, T.dec, wise.ra, wise.dec, 1./3600.)
    print len(I), 'matches'
    print len(np.unique(I)), 'unique CFHT sources in matches'
    print len(np.unique(J)), 'unique WISE sources in matches'

    for j in np.unique(J):
        K = np.flatnonzero(J == j)
        # UGH, just assign to the nearest
        i = np.argmin(D[K])
        K = K[i]
        i = I[K]

        for band in bands:
            if isinstance(srcs[i], CompositeGalaxy):
                mag = wise.get(band+'mag')[j]
                half = mag + 0.75
                setattr(srcs[i].brightnessExp, band, half)
                setattr(srcs[i].brightnessDev, band, half)
            else:
                setattr(srcs[i].brightness, band, wise.get(band+'mag')[j])
        print 'Plugged in WISE mags for source:', srcs[i]
    
    #### Cut to just sources that had a match to the WISE catalog
    # ### uhh, why don't we just use the WISE catalog then?
    # JJ,KK = np.unique(J, return_index=True)
    # keep = Catalog()
    # for i in I[KK]:
    # keep.append(srcs[i])
    # srcs = keep
    # print 'Kept:'
    # for src in srcs:
    # print src
    

    ims = []


    basedir = '/project/projectdirs/bigboss/data/wise/level3'
    pat = '3342p000_ab41-w%i'
    for band in bandnums:
        base = pat % band
        basefn = os.path.join(basedir, base)
        im = read_wise_coadd(basefn, radecroi=radecroi, filtermap=filtermap)
        tr = tractor.Tractor(tractor.Images(im), srcs)
        make_plots('wise-%i-' % band, im, tr=tr)
        ims.append(im)


    basedir = '/project/projectdirs/bigboss/data/wise/level1b'
    pat = '04933b137-w%i'
    for band in bandnums:
        base = pat % band
        basefn = os.path.join(basedir, base)
        im = read_wise_image(basefn, radecroi=radecroi, filtermap=filtermap)
        tr = tractor.Tractor(tractor.Images(im), srcs)
        make_plots('wise-%i-' % band, im, tr=tr)
        ims.append(im)
        im.freezeAllBut('psf', 'sky')
        tr.freezeParam('catalog')

        j = 1
        while True:
            dlnp,X,alpha = tr.optimize(damp=1.)
            make_plots('wise-%i-psfsky-%i-' % (bandnum,j), im, tr=tr, plots=['model','chi'])
            j += 1

    sys.exit(0)

    tr = tractor.Tractor(tractor.Images(*ims), srcs)

    # fix all calibration
    tr.freezeParam('images')

    # freeze source positions, shapes
    tr.freezeParamsRecursive('pos', 'shape', 'shapeExp', 'shapeDev')

    # freeze all sources
    tr.catalog.freezeAllParams()
    # also freeze all bands
    tr.catalog.freezeParamsRecursive(*bands)

    for im,bandnum in zip(ims,bandnums):
        tr.setImages(tractor.Images(im))
        band = im.photocal.band
        print 'band', band
        # thaw this band
        tr.catalog.thawParamsRecursive(band)

        # sweep across the image, optimizing in circles.
        # we'll use the healpix grid for circle centers.
        # how big? in arcmin
        R = 1.
        Rpix = R / 60. / np.sqrt(np.abs(np.linalg.det(im.wcs.cdAtPixel(0,0))))
        nside = int(healpix_nside_for_side_length_arcmin(R/2.))
        print 'Nside', nside
        print 'radius in pixels:', Rpix

        # start in one corner.
        pos = im.wcs.pixelToPosition(0, 0)
        hp = radecdegtohealpix(pos.ra, pos.dec, nside)

        hpqueue = [hp]
        hpdone = []

        j = 1

        while len(hpqueue):
            hp = hpqueue.pop()
            hpdone.append(hp)
            print 'looking at healpix', hp
            ra,dec = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
            print 'RA,Dec center', ra,dec
            x,y = im.wcs.positionToPixel(tractor.RaDecPos(ra,dec))
            H,W  = im.shape
            if x < -Rpix or y < -Rpix or x >= W+Rpix or y >= H+Rpix:
                print 'pixel', x,y, 'out of bounds'
                continue

            # add neighbours
            nn = healpix_get_neighbours(hp, nside)
            print 'healpix neighbours', nn
            for ni in nn:
                if ni in hpdone:
                    continue
                if ni in hpqueue:
                    continue
                hpqueue.append(ni)
                print 'enqueued neighbour', ni

            # FIXME -- add PSF-sized margin to radius
            #ra,dec = (radecroi[0]+radecroi[1])/2., (radecroi[2]+radecroi[3])/2.

            tr.catalog.thawSourcesInCircle(tractor.RaDecPos(ra, dec), R/60.)

            for step in range(10):
                print 'Optimizing:'
                for nm in tr.getParamNames():
                    print nm
                (dlnp,X,alpha) = tr.optimize(damp=1.)
                print 'dlnp', dlnp
                print 'alpha', alpha
            
                if True:
                    print 'plotting', j
                    make_plots('wise-%i-step%03i-' % (bandnum,j), im, tr=tr, plots=['model','chi'])
                    j += 1

                    ###################### profiling
                    #if j == 10:
                    #    sys.exit(0)
                    ######################

                if dlnp < 1:
                    break

            tr.catalog.freezeAllParams()

        # re-freeze this band
        tr.catalog.freezeParamsRecursive(band)



    # Optimize sources one at a time, and one image at a time.
    #sortmag = 'w1'
    #I = np.argsort([getattr(src.getBrightness(), sortmag) for src in srcs])
    # for j,i in enumerate(I):
    #     srci = int(i)
    #     # 
    #     tr.catalog.thawParam(srci)
    #     while True:
    #         print 'optimizing source', j+1, 'of', len(I)
    #         print 'Optimizing:'
    #         for nm in tr.getParamNames():
    #             print nm
    #         (dlnp,X,alpha) = tr.optimize()
    #         if dlnp < 1:
    #             break
    # 
    #     tr.catalog.freezeParam(srci)
    # 
    #     #if ((j+1) % 10 == 0):
    #     if True:
    #         print 'plotting', j
    #         make_plots('wise-%i-step%03i-' % (bandnum,j), im, tr=tr, plots=['model','chi'])



    #print 'Optimizing:'
    #for n in tr.getParamNames():
    #    print n
    #tr.optimize()

    for band,im in zip([1,2,3,4], ims):
        make_plots('wise-%i-opt1-' % band, im, tr=tr, plots=['model','chi'])



def forcedphot():
    T1 = fits_table('cs82data/cas-primary-DR8.fits')
    print len(T1), 'primary'
    T1.cut(T1.nchild == 0)
    print len(T1), 'children'

    rl,rh = T1.ra.min(), T1.ra.max()
    dl,dh = T1.dec.min(), T1.dec.max()

    tims = []

    # Coadd
    basedir = os.path.join('cs82data', 'wise', 'level3')
    basefn = os.path.join(basedir, '3342p000_ab41-w1')
    tim = read_wise_coadd(basefn, radecroi=[rl,rh,dl,dh], nanomaggies=True)
    tims.append(tim)

    # Individuals
    basedir = os.path.join('cs82data', 'wise', 'level1b')
    for fn in [ '04933b137-w1', '04937b137-w1', '04941b137-w1', '04945b137-w1', '04948a112-w1',
                '04949b137-w1', '04952a112-w1', '04953b137-w1', '04956a112-w1', '04960a112-w1',
                '04964a112-w1', '04968a112-w1', '05204a106-w1' ]:
        basefn = os.path.join(basedir, '04933b137-w1')
        tim = read_wise_image(basefn, radecroi=[rl,rh,dl,dh], nanomaggies=True)
        tims.append(tim)

    # tractor.Image's setMask() does a binary dilation on bad pixels!
    for tim in tims:
        #   tim.mask = np.zeros(tim.shape, dtype=bool)
        tim.invvar = tim.origInvvar
        tim.mask = np.zeros(tim.shape, dtype=bool)
    print 'tim:', tim

    ps = PlotSequence('forced')

    plt.clf()
    plt.plot(T1.ra, T1.dec, 'r.')
    for tim in tims:
        wcs = tim.getWcs()
        H,W = tim.shape
        rr,dd = [],[]
        for x,y in zip([1,1,W,W,1], [1,H,H,1,1]):
            rd = wcs.pixelToPosition(x,y)
            rr.append(rd.ra)
            dd.append(rd.dec)
        plt.plot(rr, dd, 'k-', alpha=0.5)
    #setRadecAxes(rl,rh,dl,dh)
    ps.savefig()

    T2 = fits_table('wise-cut.fits')
    T2.w1 = T2.w1mpro
    R = 1./3600.
    I,J,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
    print len(I), 'matches'

    refband = 'r'
    #bandnum = band_index('r')

    Lstar = (T1.probpsf == 1) * 1.0
    Lgal  = (T1.probpsf == 0)
    fracdev = T1.get('fracdev_%s' % refband)
    Ldev = Lgal * fracdev
    Lexp = Lgal * (1. - fracdev)

    ndev, nexp, ncomp,nstar = 0, 0, 0, 0
    cat = Catalog()
    #for i,t in enumerate(T1):

    jmatch = np.zeros(len(T1))
    jmatch[:] = -1
    jmatch[I] = J

    for i in range(len(T1)):
        j = jmatch[i]
        if j >= 0:
            # match source: grab WISE catalog mag
            w1 = T2.w1[j]
        else:
            # unmatched: set it faint
            w1 = 18.

        bright = NanoMaggies(w1=NanoMaggies.magToNanomaggies(w1))

        pos = RaDecPos(T1.ra[i], T1.dec[i])
        if Lstar[i] > 0:
            # Star
            star = PointSource(pos, bright)
            cat.append(star)
            nstar += 1
            continue

        hasdev = (Ldev[i] > 0)
        hasexp = (Lexp[i] > 0)
        iscomp = (hasdev and hasexp)
        if iscomp:
            dbright = bright * Ldev[i]
            ebright = bright * Lexp[i]
        elif hasdev:
            dbright = bright
        elif hasexp:
            ebright = bright
        else:
            assert(False)
                                             
        if hasdev:
            re  = T1.get('devrad_%s' % refband)[i]
            ab  = T1.get('devab_%s'  % refband)[i]
            phi = T1.get('devphi_%s' % refband)[i]
            dshape = GalaxyShape(re, ab, phi)
        if hasexp:
            re  = T1.get('exprad_%s' % refband)[i]
            ab  = T1.get('expab_%s'  % refband)[i]
            phi = T1.get('expphi_%s' % refband)[i]
            eshape = GalaxyShape(re, ab, phi)

        if iscomp:
            gal = CompositeGalaxy(pos, ebright, eshape, dbright, dshape)
            ncomp += 1
        elif hasdev:
            gal = DevGalaxy(pos, dbright, dshape)
            ndev += 1
        elif hasexp:
            gal = ExpGalaxy(pos, ebright, eshape)
            nexp += 1

        cat.append(gal)
    print 'Created', ndev, 'pure deV', nexp, 'pure exp and',
    print ncomp, 'composite galaxies',
    print 'and', nstar, 'stars'

    tractor = Tractor(tims, cat)

    for i,tim in enumerate(tims):
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=tim.zr[0], vmax=tim.zr[1])

        mod = tractor.getModelImage(i)

        plt.clf()
        plt.imshow(mod, **ima)
        plt.gray()
        plt.title('model: %s' % tim.name)
        ps.savefig()

        plt.clf()
        plt.imshow(tim.getImage(), **ima)
        plt.gray()
        plt.title('data: %s' % tim.name)
        ps.savefig()

        plt.clf()
        plt.imshow(tim.getInvvar(), interpolation='nearest', origin='lower')
        plt.gray()
        plt.title('invvar')
        ps.savefig()

        #for tim in tims:
        wcs = tim.getWcs()
        H,W = tim.shape
        poly = []
        for r,d in zip([rl,rl,rh,rh,rl], [dl,dh,dh,dl,dl]):
            x,y = wcs.positionToPixel(RaDecPos(r,d))
            poly.append((x,y))
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        xy = np.vstack((xx.flat, yy.flat)).T
        grid = points_inside_poly(xy, poly)
        grid = grid.reshape((H,W))

        tim.setInvvar(tim.getInvvar() * grid)

        # plt.clf()
        # plt.imshow(grid, interpolation='nearest', origin='lower')
        # plt.gray()
        # ps.savefig()

        plt.clf()
        plt.imshow(tim.getInvvar(), interpolation='nearest', origin='lower')
        plt.gray()
        plt.title('invvar')
        ps.savefig()

        if i == 1:
            plt.clf()
            plt.imshow(tim.goodmask, interpolation='nearest', origin='lower')
            plt.gray()
            plt.title('goodmask')
            ps.savefig()

            miv = (1./(tim.uncplane)**2)
            for bit in range(-1,32):
                if bit >= 0:
                    miv[(tim.maskplane & (1 << bit)) != 0] = 0.
                if bit == 31:
                    plt.clf()
                    plt.imshow(miv, interpolation='nearest', origin='lower')
                    plt.gray()
                    plt.title('invvar with mask bits up to %i blanked out' % bit)
                    ps.savefig()

            # for bit in range(32):
            #   plt.clf()
            #   plt.imshow(tim.maskplane & (1 << bit),
            #              interpolation='nearest', origin='lower')
            #   plt.gray()
            #   plt.title('mask bit %i' % bit)
            #   ps.savefig()


def wisemap():
    from bigboss_test import radecroi
    from astrometry.blind.plotstuff import *

    basedir = '/project/projectdirs/bigboss'
    wisedatadir = os.path.join(basedir, 'data', 'wise')

    (ra0,ra1, dec0,dec1) = radecroi
    ra = (ra0 + ra1) / 2.
    dec = (dec0 + dec1) / 2.
    width = 2.

    rfn = 'wise-roi.fits'
    if not os.path.exists(rfn):
        TT = []
        for part in range(1, 7):
            fn = 'index-allsky-astr-L1b-part%i.fits' % part
            catfn = os.path.join(wisedatadir, fn)
            print 'Reading', catfn
            T = fits_table(catfn)
            print 'Read', len(T)
            I,J,d = match_radec(ra, dec, T.ra, T.dec, width)
            print 'Found', len(I), 'RA,Dec matches'
            if len(I) == 0:
                del T
                continue
            T.cut(J)
            newhdr = []
            for i in range(len(T)):
                hdr = T.header[i]
                hdr = [str(s) for s in hdr]
                hdr = (['SIMPLE  =                    T',
                    'BITPIX  =                    8',
                    'NAXIS   =                    0',
                    ] + hdr +
                       ['END'])
                hdr = [x + (' ' * (80-len(x))) for x in hdr]
                hdrstr = ''.join(hdr)
                newhdr.append(hdrstr)
            T.delete_column('header')
            T.headerstr = np.array(newhdr)
            TT.append(T)

        T = merge_tables(TT)
        T.about()

        sid = np.array([np.sum([float(1 << (8*(6-i))) * ord(s[i]) for i in range(6)])
                for s in T.scan_id])
        I = np.lexsort((T.frame_num, sid))
        T.cut(I)
        T.writeto(rfn)

    T = fits_table(rfn)

    print 'Scan/Frame:'
    for s,f in zip(T.scan_id, T.frame_num):
        print '  ', s, f

    plot = Plotstuff(outformat='png', ra=ra, dec=dec, width=width, size=(800,800))
    out = plot.outline
    plot.color = 'white'
    plot.alpha = 0.1
    plot.apply_settings()

    for i in range(len(T)):
        hdrstr = T.headerstr[i]
        wcs = anwcs(hdrstr)
        out.wcs = wcs
        out.fill = False
        plot.plot('outline')
        out.fill = True
        plot.plot('outline')

    plot.color = 'gray'
    plot.alpha = 1.0
    plot.plot_grid(1, 1, 1, 1)
    plot.write('wisemap.png')

    I,J,d = match_radec(ra, dec, T.ra, T.dec, width)
    print 'Found', len(I), 'RA,Dec matches'
    i = np.argmin(d)
    i = J[i]
    print 'ra,dec', ra,dec, 'closest', T.ra[i], T.dec[i]
    hdrstr = T.headerstr[i]
    if len(hdrstr) % 80:
        hdrstr = hdrstr + (' ' * (80 - (len(hdrstr)%80)))
    wcs = anwcs(hdrstr)
    plot.color = 'blue'
    plot.alpha = 0.2
    plot.apply_settings()
    out.wcs = wcs
    out.fill = False
    plot.plot('outline')
    out.fill = True
    plot.plot('outline')
    plot.write('wisemap2.png')


def wise_psf_plots():
    # Plot Aaron's PSF models
    for i in range(1,5):
        P=pyfits.open('psf%i.fits' % i)[0].data
        print P.min(), P.max()
        P/=P.max()
        plt.clf()
        plt.imshow(np.log10(np.maximum(P,1e-8)), interpolation='nearest', origin='lower', vmax=0.01)
        plt.colorbar()
        plt.savefig('psf-w%i.png' % i)
        
    plt.clf()
    for i,y in enumerate([0,500,1000]):
        for j,x in enumerate([0,500,1000]):
            P=pyfits.open('psf-1-%i-%i.fits' % (x,y))[0].data
            P/=P.max()
            plt.subplot(3, 3, 3*i+j+1)
            plt.imshow(np.log10(np.maximum(P,1e-8)), interpolation='nearest', origin='lower', vmax=0.01)
            #plt.colorbar()
    plt.savefig('psf-w1-xy.png')

    psf = pyfits.open('psf%i.fits' % 1)[0].data
    S = psf.shape[0]
    # number of Gaussian components
    for K in range(1,6):
        w,mu,sig = em_init_params(K, None, None, None)
        II = psf.copy()
        II /= II.sum()
        II = np.maximum(II, 0)
        xm,ym = -(S/2), -(S/2)
        res = em_fit_2d(II, xm, ym, w, mu, sig)
        print 'em_fit_2d result:', res
        if res != 0:
            raise RuntimeError('Failed to fit PSF')
        print 'w,mu,sig', w,mu,sig
        mypsf = GaussianMixturePSF(w, mu, sig)
        mypsf.computeRadius()

        #
        mypsf.radius = S/2
        mod = mypsf.getPointSourcePatch(0., 0.)
        mod = mod.patch
        mod /= mod.sum()

        plt.clf()
        plt.subplot(1,2,1)
        ima = dict(interpolation='nearest', origin='lower', vmax=0.01 + np.log10(II.max()))
        plt.imshow(np.log10(np.maximum(II, 1e-8)), **ima)
        plt.subplot(1,2,2)
        plt.imshow(np.log10(np.maximum(mod, 1e-8)), **ima)
        plt.savefig('psf-k%i.png' % K)


def plot_unmatched():
    from bigboss_test import radecroi
    '''
    select
      run, rerun, camcol, field, nChild, probPSF,
      psfFlux_u, psfFlux_g, psfFlux_r, psfFlux_i, psfFlux_z,
      deVRad_u, deVRad_g, deVRad_r, deVRad_i, deVRad_z,
      deVAB_u, deVAB_g, deVAB_r, deVAB_i, deVAB_z,
      deVPhi_u, deVPhi_g, deVPhi_r, deVPhi_i, deVPhi_z,
      deVFlux_u, deVFlux_g, deVFlux_r, deVFlux_i, deVFlux_z,
      expRad_u, expRad_g, expRad_r, expRad_i, expRad_z,
      expAB_u, expAB_g, expAB_r, expAB_i, expAB_z,
      expPhi_u, expPhi_g, expPhi_r, expPhi_i, expPhi_z,
      expFlux_u, expFlux_g, expFlux_r, expFlux_i, expFlux_z,
      fracDeV_u, fracDeV_g, fracDeV_r, fracDeV_i, fracDeV_z,
      flags_u, flags_g, flags_r, flags_i, flags_z,
      probPSF_u, probPSF_g, probPSF_r, probPSF_i, probPSF_z,
      ra, dec
      from PhotoPrimary
      where ra between 333.5 and 335.5 and dec between -0.5 and 1.5
        '''

    ''' -> 124k rows.  (sdss-cas-testarea.fits)  Distinct runs:
    2585, 2728, 7712, 4203, 2583, 4192, 4207, 4184, 2662, 7717

    Run  #sources
    2583 663
    2585 675
    2662 4
    2728 156
    4184 762
    4192 36135
    4203 5
    4207 44078
    7712 12047
    7717 29911
    '''

    '''
    select
      run, rerun, camcol, field, nChild, probPSF,
      psfFlux_u, psfFlux_g, psfFlux_r, psfFlux_i, psfFlux_z,
      deVRad_u, deVRad_g, deVRad_r, deVRad_i, deVRad_z,
      deVAB_u, deVAB_g, deVAB_r, deVAB_i, deVAB_z,
      deVPhi_u, deVPhi_g, deVPhi_r, deVPhi_i, deVPhi_z,
      deVFlux_u, deVFlux_g, deVFlux_r, deVFlux_i, deVFlux_z,
      expRad_u, expRad_g, expRad_r, expRad_i, expRad_z,
      expAB_u, expAB_g, expAB_r, expAB_i, expAB_z,
      expPhi_u, expPhi_g, expPhi_r, expPhi_i, expPhi_z,
      expFlux_u, expFlux_g, expFlux_r, expFlux_i, expFlux_z,
      fracDeV_u, fracDeV_g, fracDeV_r, fracDeV_i, fracDeV_z,
      flags_u, flags_g, flags_r, flags_i, flags_z,
      probPSF_u, probPSF_g, probPSF_r, probPSF_i, probPSF_z,
      ra, dec, resolveStatus, score into mydb.wisetest from PhotoObjAll
      where ra between 333.5 and 335.5 and dec between -0.5 and 1.5
        and (resolveStatus & (
                dbo.fResolveStatus('SURVEY_PRIMARY') |
                dbo.fResolveStatus('SURVEY_BADFIELD') |
                dbo.fResolveStatus('SURVEY_EDGE'))) != 0

        --> sdss-cas-testarea-2.fits

    select
    run, rerun, camcol, field, nChild, probPSF,
    psfFlux_u, psfFlux_g, psfFlux_r, psfFlux_i, psfFlux_z,
    deVRad_u, deVRad_g, deVRad_r, deVRad_i, deVRad_z,
    deVAB_u, deVAB_g, deVAB_r, deVAB_i, deVAB_z,
    deVPhi_u, deVPhi_g, deVPhi_r, deVPhi_i, deVPhi_z,
    deVFlux_u, deVFlux_g, deVFlux_r, deVFlux_i, deVFlux_z,
    expRad_u, expRad_g, expRad_r, expRad_i, expRad_z,
    expAB_u, expAB_g, expAB_r, expAB_i, expAB_z,
    expPhi_u, expPhi_g, expPhi_r, expPhi_i, expPhi_z,
    expFlux_u, expFlux_g, expFlux_r, expFlux_i, expFlux_z,
    fracDeV_u, fracDeV_g, fracDeV_r, fracDeV_i, fracDeV_z,
    flags_u, flags_g, flags_r, flags_i, flags_z,
    probPSF_u, probPSF_g, probPSF_r, probPSF_i, probPSF_z,
    ra, dec, resolveStatus, score into mydb.wisetest from PhotoObjAll
    where ra between 333.5 and 335.5 and dec between -0.5 and 1.5
    and (resolveStatus & (
        dbo.fResolveStatus('SURVEY_PRIMARY') |
    dbo.fResolveStatus('SURVEY_BADFIELD') |
    dbo.fResolveStatus('SURVEY_EDGE') |
    dbo.fResolveStatus('SURVEY_BEST')
    )) != 0
    
    --> sdss-cas-testarea-3.fits
    '''

    ''' Check it out: spatial source density looks fine.  No overlap between runs.
    '''

    rng = ((333.5, 335.5), (-0.5, 1.5))

    ps = PlotSequence('sdss')

    if False:
        r,d = np.mean(rng[0]), np.mean(rng[1])
    
        RCF = radec_to_sdss_rcf(r, d, radius=2. * 60., tablefn='window_flist-DR9.fits')
        print 'Found', len(RCF), 'fields in range'
        for run,c,f,ra,dec in RCF:
            print '  ',run,c,f, 'at', ra,dec
    
        from astrometry.blind.plotstuff import *
        plot = Plotstuff(rdw=(r,d, 10), size=(1000,1000), outformat='png')
        plot.color = 'white'
        plot.alpha = 0.5
        plot.apply_settings()
        T = fits_table('window_flist-DR9.fits')
        I,J,d = match_radec(T.ra, T.dec, r,d, 10)
        T.cut(I)
        print 'Plotting', len(T)
        for i,(m0,m1,n0,n1,node,incl) in enumerate(zip(T.mu_start, T.mu_end, T.nu_start, T.nu_end, T.node, T.incl)):
            #rr,dd = [],[]
            for j,(m,n) in enumerate([(m0,n0),(m0,n1),(m1,n1),(m1,n0),(m0,n0)]):
                ri,di = munu_to_radec_deg(m, n, node, incl)
                #rr.append(ri)
                #dd.append(di)
                if j == 0:
                    plot.move_to_radec(ri,di)
                else:
                    plot.line_to_radec(ri,di)
            plot.stroke()
        plot.plot_grid(2, 2, 5, 5)
        plot.write('fields.png')

    # CAS PhotoObjAll.resolveStatus bits
    sprim = 0x100
    sbad  = 0x800
    sedge = 0x1000
    sbest = 0x200

    if False:
        T = fits_table('sdss-cas-testarea.fits')
        plt.clf()
        plothist(T.ra, T.dec, 200, range=rng)
        plt.title('PhotoPrimary')
        ps.savefig()

        T = fits_table('sdss-cas-testarea-2.fits')
        plt.clf()
        plothist(T.ra, T.dec, 200, range=rng)
        plt.title('PhotoObjAll: SURVEY_PRIMARY | SURVEY_BADFIELD | SURVEY_EDGE')
        ps.savefig()

        T = fits_table('sdss-cas-testarea-3.fits')
        plt.clf()
        plothist(T.ra, T.dec, 200, range=rng)
        plt.title('PhotoObjAll: SURVEY_PRIMARY | SURVEY_BADFIELD | SURVEY_EDGE | SURVEY_BEST')
        ps.savefig()

        for j,(flags,txt) in enumerate([ (sprim, 'PRIM'), (sbad, 'BAD'), (sedge, 'EDGE'),
                         (sbest, 'BEST')]):
            I = np.flatnonzero((T.resolvestatus & (sprim | sbad | sedge | sbest)) == flags)
            print len(I), 'with', txt
            if len(I) == 0:
                continue
            plt.clf()
            plothist(T.ra[I], T.dec[I], 200, range=rng)
            plt.title('%i with %s' % (len(I), txt))
            ps.savefig()

        for j,(flags,txt) in enumerate([ (sprim | sbad, 'PRIM + BAD'),
                         (sprim | sedge, 'PRIM + EDGE'),
                         (sprim | sbest, 'PRIM + BEST') ]):
            I = np.flatnonzero((T.resolvestatus & flags) > 0)
            print len(I), 'with', txt
            if len(I) == 0:
                continue
            plt.clf()
            plothist(T.ra[I], T.dec[I], 200, range=rng)
            plt.title('%i with %s' % (len(I), txt))
            ps.savefig()


        # for run in np.unique(T.run):
        #   I = (T.run == run)
        #   plt.clf()
        #   plothist(T.ra[I], T.dec[I], 200, range=rng)
        #   plt.title('Run %i' % run)
        #   ps.savefig()

        R = 1./3600.
        I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, R, notself=True)
        print len(I), 'matches'
        plt.clf()
        loghist((T.ra[I]-T.ra[J])*3600., (T.dec[I]-T.dec[J])*3600., 200, range=((-1,1),(-1,1)))
        ps.savefig()

    ps = PlotSequence('forced')
    
    basedir = os.environ.get('BIGBOSS_DATA', '/project/projectdirs/bigboss')
    wisedatadir = os.path.join(basedir, 'data', 'wise')
    l1bdir = os.path.join(wisedatadir, 'level1b')

    wisecat = fits_table(os.path.join(wisedatadir, 'catalogs', 'wisecat2.fits'))
    #plt.clf()
    #plothist(wisecat.ra, wisecat.dec, 200, range=rng)
    #plt.savefig('wisecat.png')

    (ra0,ra1, dec0,dec1) = radecroi
    ra = (ra0 + ra1) / 2.
    dec = (dec0 + dec1) / 2.

    #cas = fits_table('sdss-cas-testarea.fits')
    #cas = fits_table('sdss-cas-testarea-2.fits')
    cas = fits_table('sdss-cas-testarea-3.fits')
    print 'Read', len(cas), 'CAS sources'

    cas.cut((cas.resolvestatus & sedge) == 0)
    print 'Cut to ', len(cas), 'without SURVEY_EDGE set'


    # Check out WISE / SDSS matches.
    wise = wisecat
    sdss = cas
    print len(sdss), 'SDSS sources'
    print len(wise), 'WISE sources'
    R = 10.
    I,J,d = match_radec(wise.ra, wise.dec, sdss.ra, sdss.dec,
                        R/3600., nearest=True)
    print len(I), 'matches'

    print 'max dist:', d.max()

    plt.clf()
    plt.hist(d * 3600., 100, range=(0,R), log=True)
    plt.xlabel('Match distance (arcsec)')
    plt.ylabel('Number of matches')
    plt.title('SDSS-WISE astrometric matches')
    ps.savefig()
    
    plt.clf()
    loghist((wise.ra[I] - sdss.ra[J])*3600., (wise.dec[I] - sdss.dec[J])*3600.,
            200, range=((-R,R),(-R,R)))
    plt.title('SDSS-WISE astrometric matches')
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    ps.savefig()

    R = 4.

    I,J,d = match_radec(wise.ra, wise.dec, sdss.ra, sdss.dec,
                        R/3600., nearest=True)
    print len(I), 'matches'

    unmatched = np.ones(len(wise), bool)
    unmatched[I] = False
    wun = wise[unmatched]

    plt.clf()
    plothist(sdss.ra, sdss.dec, 200, range=rng)
    plt.title('SDSS source density')
    ps.savefig()

    plt.clf()
    plothist(wise.ra, wise.dec, 200, range=rng)
    plt.title('WISE source density')
    ps.savefig()

    plt.clf()
    #plt.plot(wun.ra, wun.dec, 'r.')
    #plt.axis(rng[0] + rng[1])
    plothist(wun.ra, wun.dec, 200, range=rng)
    plt.title('Unmatched WISE sources')
    ps.savefig()

    for band in 'ugriz':
        sdss.set('psfmag_'+band,
                 NanoMaggies.nanomaggiesToMag(sdss.get('psfflux_'+band)))
    
    # plt.clf()
    # loghist(wise.w1mpro[I], sdss.psfmag_r[J], 200)
    # plt.xlabel('WISE w1mpro')
    # plt.ylabel('SDSS psfflux_r')
    # ps.savefig()
    
    for band in 'riz':
        ax = [0, 10, 25, 5]
        plt.clf()
        mag = sdss.get('psfmag_'+band)[J]
        loghist(mag - wise.w1mpro[I], mag, 200,
                range=((ax[0],ax[1]),(ax[3],ax[2])))
        plt.xlabel('SDSS %s - WISE w1' % band)
        plt.ylabel('SDSS '+band)
        plt.axis(ax)
        ps.savefig()

    for w,t in [(wise[I], 'Matched'), (wun, 'Unmatched')]:
        plt.clf()
        w1 = w.get('w1mpro')
        w2 = w.get('w2mpro')
        ax = [-1, 3, 18, 6]
        loghist(w1-w2, w1, 200,
                range=((ax[0],ax[1]),(ax[3],ax[2])))
        plt.xlabel('W1 - W2')
        plt.ylabel('W1')
        plt.title('WISE CMD for %s sources' % t)
        plt.axis(ax)
        ps.savefig()

    sdssobj = DR9()
    band = 'r'
    RCF = np.unique(zip(sdss.run, sdss.camcol, sdss.field))
    wcses = []
    fns = []

    pfn = 'wcses.pickle'
    if os.path.exists(pfn):
        print 'Reading', pfn
        wcses,fns = unpickle_from_file(pfn)
    else:
        for r,c,f in RCF:
            fn = sdssobj.retrieve('frame', r, c, f, band)
            print 'got', fn
            fns.append(fn)
            wcs = Tan(fn, 0)
            print 'got wcs', wcs
            wcses.append(wcs)
        pickle_to_file((wcses,fns), pfn)
        print 'Wrote to', pfn

    wisefns = glob(os.path.join(wisedatadir, 'level3', '*w1-int-3.fits'))
    wisewcs = []
    for fn in wisefns:
        print 'Reading', fn
        wcs = anwcs(fn, 0)
        print 'Got', wcs
        wisewcs.append(wcs)
        
    I = np.argsort(wun.w1mpro)
    wun.cut(I)
    for i in range(len(wun)):
        ra,dec = wun.ra[i], wun.dec[i]
        insdss = -1
        for j,wcs in enumerate(wcses):
            if wcs.is_inside(ra,dec):
                insdss = j
                break
        inwise = -1
        for j,wcs in enumerate(wisewcs):
            if wcs.is_inside(ra,dec):
                inwise = j
                break
        N = 0
        if insdss != -1:
            N += 1
        if inwise != -1:
            N += 1
        if N == 0:
            continue

        if N != 2:
            continue
        
        plt.clf()
        ss = 1
        plt.subplot(2, N, ss)
        ss += 1
        M = 0.02
        I = np.flatnonzero((sdss.ra  > (ra  - M)) * (sdss.ra  < (ra  + M)) *
                           (sdss.dec > (dec - M)) * (sdss.dec < (dec + M)))
        sdssnear = sdss[I]
        plt.plot(sdss.ra[I], sdss.dec[I], 'b.', alpha=0.7)
        I = np.flatnonzero((wise.ra  > (ra  - M)) * (wise.ra  < (ra  + M)) *
                           (wise.dec > (dec - M)) * (wise.dec < (dec + M)))
        wisenear = wise[I]
        plt.plot(wise.ra[I], wise.dec[I], 'rx', alpha=0.7)
        if insdss:
            wcs = wcses[j]
            w,h = wcs.imagew, wcs.imageh
            rd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                           [(1,1), (w,1), (w,h), (1,h), (1,1)]])
            plt.plot(rd[:,0], rd[:,1], 'b-', alpha=0.5)
        if inwise:
            wcs = wisewcs[j]
            w,h = wcs.imagew, wcs.imageh
            rd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                           [(1,1), (w,1), (w,h), (1,h), (1,1)]])
            plt.plot(rd[:,0], rd[:,1], 'r-', alpha=0.5)
        plt.plot([ra], [dec], 'o', mec='k', mfc='none', mew=3, ms=20, alpha=0.5)
        #plt.axis([ra+M, ra-M, dec-M, dec+M])
        plt.axis([ra-M, ra+M, dec-M, dec+M])
        plt.xticks([ra], ['RA = %0.3f' % ra])
        plt.yticks([dec], ['Dec = %0.3f' % dec])

        SW = 20
        
        ss = N+1
        plt.subplot(2, N, ss)
        if insdss != -1:
            ss += 1
            j = insdss
            wcs = wcses[j]
            xc,yc = wcs.radec2pixelxy(ra,dec)
            r,c,f = RCF[j]
            frame = sdssobj.readFrame(r,c,f, band)
            #S = 50
            S = SW * 3.472
            im = frame.image
            H,W = im.shape
            y0,x0 = max(0, yc-S), max(0, xc-S)
            y1,x1 = min(H, yc+S), min(W, xc+S)
            subim = im[y0:y1, x0:x1]

            #plt.imshow(subim, interpolation='nearest', origin='lower',
            #          vmax=0.3, extent=[x0,x1,y0,y1])
            plt.imshow(subim.T, interpolation='nearest', origin='lower',
                       vmax=0.3, extent=[y0,y1,x0,x1])
            #vmax=subim.max()*1.01)
            ax = plt.axis()
            sx,sy = wcs.radec2pixelxy(sdssnear.ra, sdssnear.dec)
            #plt.plot(x, y, 'o', mec='b', mfc='none', ms=15)
            plt.plot(sy, sx, 'o', mec='b', mfc='none', ms=15)

            x,y = wcs.radec2pixelxy(wisenear.ra, wisenear.dec)
            #plt.plot(x, y, 'rx', ms=10)
            plt.plot(y, x, 'rx', ms=10)

            # Which way up?
            x,y = wcs.radec2pixelxy(np.array([ra, ra]), np.array([0.5,2.0]) * S * 0.396/3600. + dec)
            #plt.plot(x, y, 'b-', alpha=0.5, lw=2)
            plt.plot(y, x, 'b-', alpha=0.5, lw=2)
            plt.axis(ax)
            plt.gray()
            plt.title('SDSS %s (%i/%i/%i)' % (band, r,c,f))

            # Try to guess the PRIMARY run from the nearest object.
            for I in np.argsort((sx-xc)**2 + (sy-yc)**2):
                r,c,f = sdssnear.run[I], sdssnear.camcol[I], sdssnear.field[I]
                jj = None
                for j,(ri,ci,fi) in enumerate(RCF):
                    if ri == r and ci == c and fi == f:
                        jj = j
                        break
                assert(jj is not None)
                wcs = wcses[jj]
                xc,yc = wcs.radec2pixelxy(ra,dec)
                frame = sdssobj.readFrame(r,c,f, band)
                S = SW * 3.472
                im = frame.image
                H,W = im.shape
                y0,x0 = max(0, yc-S), max(0, xc-S)
                y1,x1 = min(H, yc+S), min(W, xc+S)
                subim = im[y0:y1, x0:x1]
                if np.prod(subim.shape) == 0:
                    continue
                print 'subim shape', subim.shape
                plt.subplot(2,N,2)
                plt.imshow(subim.T, interpolation='nearest', origin='lower',
                       vmax=0.3, extent=[y0,y1,x0,x1])
                plt.gray()
                plt.title('SDSS %s (%i/%i/%i)' % (band, r,c,f))
                break


        if inwise != -1:
            plt.subplot(2, N, ss)
            ss += 1
            j = inwise
            wcs = wisewcs[j]
            ok,x,y = wcs.radec2pixelxy(ra,dec)
            im = pyfits.open(wisefns[j])[0].data
            S = SW
            H,W = im.shape
            y0,x0 = max(0, y-S), max(0, x-S)
            subim = im[y0 : min(H, y+S), x0 : min(W, x+S)]

            plt.imshow(subim, interpolation='nearest', origin='lower',
                       vmax=subim.max()*1.01)
            ax = plt.axis()
            x,y = [],[]
            for r,d in zip(wisenear.ra, wisenear.dec):
                ok,xi,yi = wcs.radec2pixelxy(r,d)
                x.append(xi)
                y.append(yi)
            x = np.array(x)
            y = np.array(y)
            plt.plot(x-x0, y-y0, 'rx', ms=15)

            x,y = [],[]
            for r,d in zip(sdssnear.ra, sdssnear.dec):
                ok,xi,yi = wcs.radec2pixelxy(r,d)
                x.append(xi)
                y.append(yi)
            x = np.array(x)
            y = np.array(y)
            plt.plot(x-x0, y-y0, 'o', mec='b', mfc='none', ms=10)

            # Which way up?
            pixscale = 1.375 / 3600.
            ok,x1,y1 = wcs.radec2pixelxy(ra, dec + 0.5 * S * pixscale)
            ok,x2,y2 = wcs.radec2pixelxy(ra, dec + 2.0 * S * pixscale)
            plt.plot([x1-x0,x2-x0], [y1-y0,y2-y0], 'r-', alpha=0.5, lw=2)

            plt.axis([ax[1],ax[0],ax[2],ax[3]])
            #plt.axis(ax)
            plt.gray()
            plt.title('WISE W1 (coadd)')

        plt.suptitle('WISE unmatched source: w1=%.1f, RA,Dec = (%.3f, %.3f)' %
                     (wun.w1mpro[i], ra, dec))
            
        ps.savefig()

        rcfs = zip(sdssnear.run, sdssnear.camcol, sdssnear.field)
        print 'Nearby SDSS sources are from:', np.unique(rcfs)

    return



def forced2():
    from bigboss_test import radecroi
    ps = PlotSequence('forced')

    basedir = os.environ.get('BIGBOSS_DATA', '/project/projectdirs/bigboss')
    wisedatadir = os.path.join(basedir, 'data', 'wise')
    l1bdir = os.path.join(wisedatadir, 'level1b')
    wisecat = fits_table(os.path.join(wisedatadir, 'catalogs', 'wisecat2.fits'))

    # CAS PhotoObjAll.resolveStatus bits
    sprim = 0x100
    sbad  = 0x800
    sedge = 0x1000
    sbest = 0x200

    (ra0,ra1, dec0,dec1) = radecroi
    ra = (ra0 + ra1) / 2.
    dec = (dec0 + dec1) / 2.

    cas = fits_table('sdss-cas-testarea-3.fits')
    print 'Read', len(cas), 'CAS sources'
    cas.cut((cas.resolvestatus & sedge) == 0)
    print 'Cut to ', len(cas), 'without SURVEY_EDGE set'

    # Drop "sbest" sources that have an "sprim" nearby.
    Ibest = (cas.resolvestatus & (sprim | sbest)) == sbest
    Iprim = (cas.resolvestatus & (sprim | sbest)) == sprim
    I,J,d = match_radec(cas.ra[Ibest], cas.dec[Ibest], cas.ra[Iprim], cas.dec[Iprim], 2./3600.)

    Ibest[np.flatnonzero(Ibest)[I]] = False
    #Ikeep = np.ones(len(Ibest), bool)
    #Ikeep[I] = False
    cas.cut(np.logical_or(Ibest, Iprim))
    print 'Cut to', len(cas), 'PRIMARY + BEST-not-near-PRIMARY'

    I,J,d = match_radec(cas.ra, cas.dec, cas.ra, cas.dec, 2./3600., notself=True)
    plt.clf()
    loghist((cas.ra[I]-cas.ra[J])*3600., (cas.dec[I]-cas.dec[J])*3600., 200)
    plt.title('CAS self-matches')
    ps.savefig()
    
    psf = pyfits.open('wise-psf-w1-500-500.fits')[0].data
    S = psf.shape[0]
    # number of Gaussian components
    K = 3
    w,mu,sig = em_init_params(K, None, None, None)
    II = psf.copy()
    II = np.maximum(II, 0)
    II /= II.sum()
    xm,ym = -(S/2), -(S/2)
    res = em_fit_2d(II, xm, ym, w, mu, sig)
    if res != 0:
        raise RuntimeError('Failed to fit PSF')
    print 'W1 PSF:'
    print '  w', w
    print '  mu', mu
    print '  sigma', sig
    w1psf = GaussianMixturePSF(w, mu, sig)
    w1psf.computeRadius()

    print 'PSF radius:', w1psf.getRadius(), 'pixels'
    
    T = fits_table('wise-roi.fits')
    for i in range(len(T)):
        basefn = os.path.join(l1bdir, '%s%i-w1' % (T.scan_id[i], T.frame_num[i]))
        fn = basefn + '-int-1b.fits'
        print 'Looking for', fn
        if not os.path.exists(fn):
            continue
        print '  -> Found it!'

        tim = read_wise_image(basefn, nanomaggies=True)
        tim.psf = w1psf
        
        wcs = tim.wcs.wcs
        r0,r1,d0,d1 = wcs.radec_bounds()
        print 'RA,Dec bounds:', r0,r1, d0,d1
        
        w,h = wcs.imagew, wcs.imageh
        rd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                       [(1,1), (w,1), (w,h), (1,h), (1,1)]])

        I = np.flatnonzero((cas.ra > r0) * (cas.ra < r1) *
                           (cas.dec > d0) * (cas.dec < d1))
        J = point_in_poly(cas.ra[I], cas.dec[I], rd)
        I = I[J]
        cashere = cas[I]
        # 10-20k sources...

        wbands = ['w1']
        sdssband = 'i'
        tsrcs = get_tractor_sources_cas_dr9(cashere, nanomaggies=True,
                                            bandname=sdssband, bands=[sdssband],
                                            extrabands=wbands)
        #keepsrcs = []
        for src in tsrcs:
            for br in src.getBrightnesses():
                f = br.getBand(sdssband)
                #if f < 0:
                #   continue
                for wb in wbands:
                    br.setBand(wb, f)
                #keepsrcs.append(src)
        #tsrcs = keepsrcs
                
        print 'Created', len(tsrcs), 'tractor sources in this image'

        I = np.flatnonzero((wisecat.ra > r0) * (wisecat.ra < r1) *
                           (wisecat.dec > d0) * (wisecat.dec < d1))
        J = point_in_poly(wisecat.ra[I], wisecat.dec[I], rd)
        I = I[J]
        print 'Found', len(I), 'WISE catalog sources in this image'

        wc = wisecat[I]
        tra  = np.array([src.getPosition().ra  for src in tsrcs])
        tdec = np.array([src.getPosition().dec for src in tsrcs])

        R = 4.
        I,J,d = match_radec(wc.ra, wc.dec, tra, tdec,
                            R/3600., nearest=True)
        # cashere.ra, cashere.dec, 
        print 'Found', len(I), 'SDSS-WISE matches within', R, 'arcsec'

        for i,j in zip(I,J):
            w1 = wc.w1mpro[i]
            w1 = NanoMaggies.magToNanomaggies(w1)
            bb = tsrcs[j].getBrightnesses()
            for b in bb:
                b.setBand('w1', w1 / float(len(bb)))

        keepsrcs = []
        for src in tsrcs:
            #for b in src.getBrightness():
            b = src.getBrightness()
            if b.getBand(sdssband) > 0 or b.getBand(wbands[0]) > 0:
                keepsrcs.append(src)
        tsrcs = keepsrcs
        print 'Keeping', len(tsrcs), 'tractor sources from SDSS'

        unmatched = np.ones(len(wc), bool)
        unmatched[I] = False
        wun = wc[unmatched]
        print len(wun), 'unmatched WISE sources'
        for i in range(len(wun)):
            pos = RaDecPos(wun.ra[i], wun.dec[i])
            nm = NanoMaggies.magToNanomaggies(wun.w1mpro[i])
            br = NanoMaggies(i=25., w1=nm)
            tsrcs.append(PointSource(pos, br))
        
        plt.clf()
        plt.plot(rd[:,0], rd[:,1], 'k-')
        plt.plot(cashere.ra, cashere.dec, 'r.', alpha=0.1)
        plt.plot(wc.ra, wc.dec, 'bx', alpha=0.1)
        setRadecAxes(r0,r1,d0,d1)
        ps.savefig()

        zlo,zhi = tim.zr
        ima = dict(interpolation='nearest', origin='lower', vmin=zlo, vmax=zhi)
        imchi = dict(interpolation='nearest', origin='lower',
                 vmin=-5, vmax=5)

        plt.clf()
        plt.imshow(tim.getImage(), **ima)
        plt.hot()
        plt.title(tim.name + ': data')
        ps.savefig()

        wsrcs = []
        for i in range(len(wc)):
            pos = RaDecPos(wc.ra[i], wc.dec[i])
            nm = NanoMaggies.magToNanomaggies(wc.w1mpro[i])
            br = NanoMaggies(i=25., w1=nm)
            wsrcs.append(PointSource(pos, br))

        tr = Tractor([tim], wsrcs)
        tr.freezeParam('images')

        for jj in range(2):
            print 'Rendering WISE model image...'
            wmod = tr.getModelImage(0)
            plt.clf()
            plt.imshow(wmod, **ima)
            plt.hot()
            plt.title(tim.name + ': WISE sources only')
            ps.savefig()

            assert(np.all(np.isfinite(wmod)))
            assert(np.all(np.isfinite(tim.getInvError())))
            assert(np.all(np.isfinite(tim.getImage())))

            wchi = tr.getChiImage(0)
            plt.clf()
            plt.imshow(wchi, **imchi)
            plt.title(tim.name + ': chi, WISE sources only')
            plt.gray()
            ps.savefig()

            if jj == 1:
                break

            tr.optimize()


        tr = Tractor([tim], tsrcs)
        print 'Rendering model image...'
        mod = tr.getModelImage(0)

        plt.clf()
        plt.imshow(mod, **ima)
        plt.title(tim.name + ': SDSS + WISE sources')
        ps.savefig()

        print 'tim', tim
        print 'tim.photocal:', tim.photocal
        
        wsrcs = []
        for i in range(len(wc)):
            pos = RaDecPos(wc.ra[i], wc.dec[i])
            nm = NanoMaggies.magToNanomaggies(wc.w1mpro[i])
            br = NanoMaggies(i=25., w1=nm)
            wsrcs.append(PointSource(pos, br))

        tr = Tractor([tim], wsrcs)
        print 'Rendering WISE model image...'
        wmod = tr.getModelImage(0)

        plt.clf()
        plt.imshow(wmod, **ima)
        plt.title(tim.name + ': WISE sources only')
        ps.savefig()
        
        

if __name__ == '__main__':
    forced2()
    sys.exit(0)

    wise_psf_plots()
    wisemap()

    forcedphot()

    from astrometry.util.fits import *
    from astrometry.libkd.spherematch import *
    import pylab as plt

    T1 = fits_table('cs82data/cas-primary-DR8.fits')
    print len(T1), 'SDSS'
    print '  RA', T1.ra.min(), T1.ra.max()

    cutfn = 'wise-cut.fits'
    if not os.path.exists(cutfn):
        T2 = fits_table('wise-27-tag.fits')
        print len(T2), 'WISE'
        print '  RA', T2.ra.min(), T2.ra.max()
        T2.cut((T2.ra  > T1.ra.min())  * (T2.ra < T1.ra.max()) *
               (T2.dec > T1.dec.min()) * (T2.dec < T1.dec.max()))
        print 'Cut WISE to same RA,Dec region:', len(T2)
        T2.writeto('wise-cut.fits')
    else:
        T2 = fits_table(cutfn)
        print len(T2), 'WISE (cut)'

    R = 1./3600.
    I,J,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
    print len(I), 'matches'

    plt.clf()
    #loghist(T1.r[I] - T1.i[I], T1.r[I] - T2.w1mpro[J], 200, range=((0,2),(-2,5)))
    #plt.plot(T1.r[I] - T1.i[I], T1.r[I] - T2.w1mpro[J], 'r.')
    for cc,lo,hi in [('r', 8, 14), ('y', 14, 15), ('g', 15, 16), ('b', 16, 17), ('m', 17,20)]:
        w1 = T2.w1mpro[J]
        K = (w1 >= lo) * (w1 < hi)
        plt.plot(T1.r[I[K]] - T1.i[I[K]], T1.r[I[K]] - T2.w1mpro[J[K]], '.', color=cc)
        #plt.plot(T1.r[I] - T1.i[I], T1.r[I] - T2.w1mpro[J], 'r.')
    plt.xlabel('r - i')
    plt.ylabel('r - W1 3.4 micron')
    plt.axis([0,2,0,8])
    plt.savefig('wise1.png')

    plt.clf()
    plt.plot(T1.ra, T1.dec, 'bx')
    plt.plot(T2.ra, T2.dec, 'o', mec='r', mfc='none')
    plt.plot(T2.ra[J], T2.dec[J], '^', mec='g', mfc='none')
    plt.savefig('wise2.png')

    R = 2./3600.
    I2,J2,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
    print len(I), 'matches'
    plt.plot(T2.ra[J2], T2.dec[J2], '^', mec='g', mfc='none', ms=10)
    plt.savefig('wise3.png')

    plt.clf()
    plt.plot(3600.*(T1.ra [I2] - T2.ra [J2]),
             3600.*(T1.dec[I2] - T2.dec[J2]), 'r.')
    plt.xlabel('dRA (arcsec)')
    plt.ylabel('dDec (arcsec)')
    plt.savefig('wise4.png')

    plt.clf()
    plt.subplot(2,1,1)
    plt.hist(T1.r, 100, range=(10,25), histtype='step', color='k')
    plt.hist(T1.r[I], 100, range=(10,25), histtype='step', color='r')
    plt.hist(T1.r[I2], 100, range=(10,25), histtype='step', color='m')
    plt.xlabel('r band')
    plt.axhline(0, color='k', alpha=0.5)
    plt.ylim(-5, 90)
    plt.xlim(10,25)

    plt.subplot(2,1,2)
    plt.hist(T2.w1mpro, 100, range=(10,25), histtype='step', color='k')
    plt.hist(T2.w1mpro[J], 100, range=(10,25), histtype='step', color='r')
    plt.hist(T2.w1mpro[J2], 100, range=(10,25), histtype='step', color='m')
    plt.xlabel('W1 band')
    plt.axhline(0, color='k', alpha=0.5)
    plt.ylim(-5, 60)
    plt.xlim(10,25)

    plt.savefig('wise5.png')

    sys.exit(0)

    #import cProfile
    #from datetime import tzinfo, timedelta, datetime
    #cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
    main()



