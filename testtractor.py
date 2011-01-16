if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import pylab as plt
import numpy as np
from math import pi, sqrt

from tractor import *

class TestTractor(Tractor):
	pass

def main():
	W,H = 300,200

	psf = NGaussianPSF([2.], [1.])
	cx,cy,flux = 100., 80., 1000.

	image = np.zeros((H,W))
	err = np.zeros_like(image) + 1.
	invvar = 1./(err**2)
	src = PointSource(PixPos([cx, cy]), Flux(flux))
	photocal = NullPhotoCal()
	wcs = NullWCS()

	data = Image(data=image, invvar=invvar,
				 psf=psf, wcs=wcs, photocal=photocal)

	# add perfect point source to image.
	patch = src.getModelPatch(data)
	patch.addTo(image)

	assert(abs(image.sum() - flux) < 1.)

	mn,mx = image.min(),image.max()

	imargs1 = dict(interpolation='nearest', origin='lower')
	imargs = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)

	plt.clf()
	plt.imshow(image, **imargs)
	plt.colorbar()
	plt.savefig('test-img.png')
	
	# Create new Image with the synthetic image.
	data = Image(data=image, invvar=invvar,
				 psf=psf, wcs=wcs, photocal=photocal)

	# Create a Tractor with a source at the right location but with not enough
	# flux.
	src2 = PointSource(PixPos([cx, cy]), Flux(flux / 2.))

	tractor = TestTractor([data], catalog=[src2])

	mod = tractor.getModelImage(data)

	plt.clf()
	plt.imshow(mod, **imargs)
	plt.colorbar()
	plt.savefig('test-mod0.png')

	derivs = src.getFitParamDerivatives(data)
	print len(derivs), 'derivatives'
	for i,deriv in enumerate(derivs):
		plt.clf()
		plt.imshow(deriv.getImage(), **imargs1)
		plt.colorbar()
		plt.title('derivative ' + deriv.getName())
		plt.savefig('test-deriv%i-0a.png' % i)

		plt.clf()
		(H,W) = data.shape
		deriv.clipTo(W, H)
		plt.imshow(deriv.getImage(), **imargs1)
		plt.colorbar()
		plt.title('derivative ' + deriv.getName())
		plt.savefig('test-deriv%i-0b.png' % i)

		di = deriv.getImage()
		I = deriv.getPixelIndices(data)
		dimg = np.zeros_like(image).ravel()
		dimg[I] = di
		dimg = dimg.reshape(image.shape)

		plt.clf()
		plt.imshow(dimg, **imargs1)
		plt.colorbar()
		plt.title('derivative ' + deriv.getName())
		plt.savefig('test-deriv%i-0c.png' % i)

		plt.clf()
		slc = deriv.getSlice(data)
		plt.imshow(image[slc], **imargs)
		plt.colorbar()
		plt.title('image slice')
		plt.savefig('test-deriv%i-0d.png' % i)

	tractor.optimizeCatalogAtFixedComplexityStep()

	mod = tractor.getModelImage(data)

	plt.clf()
	plt.imshow(mod, **imargs)
	plt.colorbar()
	plt.savefig('test-mod1.png')


if __name__ == '__main__':
	main()
