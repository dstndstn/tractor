import os
import tempfile
import tractor
import pyfits
import numpy as np

def read_wise_coadd(basefn, radecroi=None, filtermap={}):
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
    photocal = tractor.MagsPhotoCal(filter, zp)

    print 'Image median:', np.median(data)
    print 'unc median:', np.median(unc)

    sky = np.median(data)
    tsky = tractor.ConstantSky(sky)

    sigma1 = np.median(unc)
    zr = np.array([-3,10]) * sigma1 + sky

    name = 'WISE ' + ihdr['COADDID'] + ' W%i' % band

    tim = tractor.Image(data=data, invvar=1./(unc**2), psf=tpsf, wcs=twcs,
                        sky=tsky, photocal=photocal, name=name, zr=zr)
    tim.extent = [x0,x1,y0,y1]
    return tim


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

    basedir = '/project/projectdirs/bigboss/data/wise/level3'
    #filtermap = { 'w1':'i', 'w2':'i', 'w3':'i', 'w4':'i' }
    filtermap = None
    
    pat = '3342p000_ab41-w%i'

    import matplotlib
    matplotlib.use('Agg')

    import logging
    import sys
    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)


    #bandnums = [1,2,3,4]
    bandnums = [1,]
    bands = ['w%i' % n for n in bandnums]

    srcs = get_cfht_catalog(mags=bands, maglim=25.)

    ims = []
    for band in bandnums:
        base = pat % band
        basefn = os.path.join(basedir, base)
        im = read_wise_coadd(basefn, radecroi=radecroi, filtermap=filtermap)
        tr = tractor.Tractor(tractor.Images(im), srcs)
        make_plots('wise-%i-' % band, im, tr=tr)

        ims.append(im)

    tr = tractor.Tractor(tractor.Images(*ims), srcs)

    # fix all calibration
    tr.freezeParam('images')

    # freeze source positions, shapes
    tr.freezeParamsRecursive('pos', 'shape', 'shapeExp', 'shapeDev')

    # freeze all sources
    tr.catalog.freezeAllParams()
    # also freeze all bands
    tr.catalog.freezeParamsRecursive(*bands)

    # Optimize sources one at a time, and one image at a time.
    sortmag = 'w1'
    I = np.argsort([getattr(src.getBrightness(), sortmag) for src in srcs])
    for im,bandnum in zip(ims,bandnums):
	tr.setImages(tractor.Images(im))
        band = im.photocal.band
        print 'band', band
        # thaw this band
        tr.catalog.thawParamsRecursive(band)

        # have to add PSF-sized margin to radius
        ra,dec = (radecroi[0]+radecroi[1])/2., (radecroi[2]+radecroi[3])/2.
        radius = 1./60.
        tr.catalog.thawSourcesInCircle(tractor.RaDecPos(ra, dec), radius)

        j = 1
        while True:
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

            if dlnp < 1:
                break




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


        # re-freeze this band
        tr.catalog.freezeParamsRecursive(band)


    #print 'Optimizing:'
    #for n in tr.getParamNames():
    #    print n
    #tr.optimize()

    for band,im in zip([1,2,3,4], ims):
        make_plots('wise-%i-opt1-' % band, im, tr=tr, plots=['model','chi'])



if __name__ == '__main__':
    import cProfile
    from datetime import tzinfo, timedelta, datetime
    cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))

