if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import numpy as np
import pyfits

from astrometry.util.util import Tan

from sdsstractor import *

if __name__ == '__main__':
	W,H = 200,200
	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1.

	ra,dec = 0,0
	# arcsec/pix
	pixscale = 1.
	wcs1 = Tan(ra, dec, W/2, H/2, -pixscale/3600., 0, 0, -pixscale/3600., W, H)
	wcs = FitsWcs(wcs1)
	photocal = NullPhotoCal()
	psf = NGaussianPSF([1.], [1.])
	sky = ConstantSky(0.)
	
	noise = (np.random.normal(size=image.shape) *
			 np.sqrt(1. / invvar))
	img = Image(data=image + noise, invvar=invvar, psf=psf,
				wcs=wcs, sky=sky, photocal=photocal)

	rd1 = wcs.pixelToPosition(None, (90,100))
	rd2 = wcs.pixelToPosition(None, (110,100))
	
	g1 = HoggExpGalaxy(rd1, Flux(1000.), 20., 0.5, 0.)
	g2 = HoggExpGalaxy(rd2, Flux(1000.), 20., 0.5, 90.)

	tractor = SDSSTractor([img])
	tractor.catalog.append(g1)
	tractor.catalog.append(g2)

	mods = tractor.getModelImages()
	mod = mods[0]

	args = dict(interpolation='nearest', origin='lower')
	plt.clf()
	plt.imshow(mod, **args)
	plt.gray()
	plt.colorbar()
	plt.savefig('test1.png')

	wcs1.write_to('wcs.fits')
	hdr = pyfits.open('wcs.fits')[0].header

	pyfits.writeto('t_img.fits', mod, header=hdr, clobber=True)
	mask = np.zeros((H,W), np.int16)
	pyfits.writeto('t_msk.fits', mask, clobber=True)
	var = 1./invvar
	pyfits.writeto('t_var.fits', var, clobber=True)

	hdus = pyfits.HDUList([pyfits.PrimaryHDU(header=hdr),
						   pyfits.ImageHDU(mod, header=hdr),
						   pyfits.ImageHDU(mask),
						   pyfits.ImageHDU(var)])
	hdus.writeto('t.fits', clobber=True)
	
			
