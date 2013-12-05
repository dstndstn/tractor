if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.plotutils import *

from tractor import *

if __name__ == '__main__':
    ps = PlotSequence('great')
    
    img = fitsio.read('image-000-0.fits')
    print 'Image size', img.shape
    stars = fitsio.read('starfield_image-000-0.fits')
    print 'Starfield size', stars.shape
    # first star is "centered"
    star = stars[:48,:48]
    print 'Star shape', star.shape
    
    # estimate noise in image
    diffs = np.abs(img[:-5:10,:-5:10] - img[5::10,5::10]).ravel()
    mad = np.median(diffs)
    sig1 = 1.4826 * mad
    print 'MAD', mad, '-> sigma', sig1

    img = img[:48,:48]

    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower', cmap='gray')
    ps.savefig()
    
    # plt.clf()
    # plt.hist(img.ravel(), 100, range=(-0.1, 0.1))
    # ps.savefig()

    star /= star.sum()
    
    psf = GaussianMixturePSF.fromStamp(star)
    print 'PSF', psf
    psf.shiftBy(0.5, 0.5)
    print 'PSF', psf
    #print 'mean', psf.mog.mean
    #print 'mean', psf.mog.mean.shape
    
    psfmodel = psf.getPointSourcePatch(0., 0., radius=24)
    print 'PSF model', psfmodel.shape
    psfmodel /= psfmodel.patch.sum()
    
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(np.log10(star), interpolation='nearest', origin='lower',
               cmap='gray', vmin=-6, vmax=0)
    plt.title('PSF stamp')
    plt.subplot(1,2,2)
    plt.imshow(np.log10(psfmodel.patch), interpolation='nearest',
               origin='lower', cmap='gray', vmin=-6, vmax=0)
    plt.title('PSF model')
    ps.savefig()


    # create tractor Image object.
    tim = Image(data=img, invvar=np.ones_like(img) * 1./sig1**2,
                psf=psf, wcs=NullWCS(), sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1.), name='dimsum 000-0-0-0',
                domask=False, zr=[-2.*sig1, 3.*sig1])
    

    
