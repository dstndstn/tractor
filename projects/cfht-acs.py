from __future__ import print_function

import pylab as plt

import numpy as np
import fitsio

from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.fits import fits_table

from tractor.cfht import CfhtLinearPhotoCal, parse_head_file
from tractor.image import Image
from tractor.sky import ConstantSky
from tractor.psf import GaussianMixturePSF
from tractor.psfex import PixelizedPsfEx

from tractor.source_extractor import get_se_modelfit_cat

if __name__ == '__main__':

    imgfn = 'cfht/1624827p.fits'
    headfn = 'cfht/1624827p.head'
    psffn = 'cfht/1624827p.psf'
    ext = 14

    catfn = 'cfht/acs_I_030mas_065_sci.VISRES.ldac'
    cat = fits_table(catfn, hdu=2)
    print('Read', len(cat), 'catalog entries')

    img,imghdr = fitsio.read(imgfn, ext=ext, header=True)
    print('Read image', img.shape, img.dtype)
    img = img.astype(np.float32)
    H,W = img.shape
    
    band = imghdr['FILTER'][0]
    print('Band:', band)
    photocal = CfhtLinearPhotoCal(imghdr, band)
    
    headers = parse_head_file(headfn)
    print('Read headers:', len(headers))
    
    #for ext in range(1, 36):
    wcshdr = headers[ext-1]
    wcshdr['IMAGEW'] = W
    wcshdr['IMAGEH'] = H
    wcs = wcs_pv2sip_hdr(wcshdr)
    #print('Got WCS:', wcs)
    
    ok,xx,yy = wcs.radec2pixelxy(cat.alpha_j2000, cat.delta_j2000)
    print('Ext', ext, 'x range', int(xx.min()), int(xx.max()), 'y range', int(yy.min()), int(yy.max()))
    
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

    sky = np.median(img)
    img -= sky
    
    tim = Image(data=img, inverr=inverr, wcs=wcs, photocal=photocal,
                psf=psfex, sky=ConstantSky(0.))
    

    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower', cmap='gray',
               vmin=-2.*sig1, vmax=10.*sig1)
    ax = plt.axis()
    plt.plot(xx-1, yy-1, 'r.')
    plt.axis(ax)
    plt.savefig('cfht.png')
    
