if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import pyfits
import pylab as plt

from tractor import *

import numpy as np
from math import pi


class SDSSTractor(Tractor):

	def createNewSource(self, img, x, y, ht):
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		#patch = img.getPsf().getPointSourcePatch(x, y)
		#print 'psf patch:', patch.shape
		#print 'psf patch: max', patch.max(), 'sum', patch.sum()
		#print 'new source peak height:', ht, '-> flux', ht/patch.max()
		#ht /= patch.max()
		return PointSource(PixPos([x,y]), Flux(ht))



def main():
	# image
	# invvar
	# sky
	# psf

	psf = NGaussianPSF([2.0], [1.0])

	# We'll start by working in pixel coords
	wcs = NullWCS()
	# And counts
	photocal = NullPhotoCal()

	data = Image(data=img, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
				 photocal=photocal)
	
	tractor = SDSSTractor([data])

	
	zrange = np.array([-3.,+10.]) * skysig + sky

	plt.clf()
	plt.imshow(img, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1]) #vmin=50, vmax=500)
	plt.hot()
	plt.colorbar()
	plt.savefig('img.png')

	plt.clf()
	plt.imshow((img-sky) * np.sqrt(invvar),
			   interpolation='nearest', origin='lower')
	plt.hot()
	plt.colorbar()
	plt.savefig('chi.png')

	plt.clf()
	plt.imshow((img-skymed) * np.sqrt(invvar),
			   interpolation='nearest', origin='lower',
			   vmin=-3, vmax=10.)
	plt.hot()
	plt.colorbar()
	plt.savefig('chi2.png')


	Nsrc = 10
	steps = (['plots'] + ['source']*Nsrc + ['plots'] + ['psf'] + ['plots'] + ['psf2'])*3 + ['plots'] + ['break']

	chiArange = None

	for i,step in enumerate(steps):

		if step == 'plots':
			print 'Making plots...'
			NS = len(tractor.getCatalog())

			chis = tractor.getChiImages()
			chi = chis[0]

			tt = 'sources: %i, chi^2 = %g' % (NS, np.sum(chi**2))

			mods = tractor.getModelImages()
			mod = mods[0]

			plt.clf()
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin=zrange[0], vmax=zrange[1])
			plt.hot()
			plt.colorbar()
			ax = plt.axis()
			img = tractor.getImage(0)
			wcs = img.getWcs()
			x = []
			y = []
			for src in tractor.getCatalog():
				pos = src.getPosition()
				px,py = wcs.positionToPixel(pos)
				x.append(px)
				y.append(py)
			plt.plot(x, y, 'b+')
			plt.axis(ax)
			plt.title(tt)
			plt.savefig('mod-%02i.png' % i)

			if chiArange is None:
				chiArange = (chi.min(), chi.max())

			plt.clf()
			plt.imshow(chi, interpolation='nearest', origin='lower',
					   vmin=chiArange[0], vmax=chiArange[1])
			plt.hot()
			plt.colorbar()
			plt.title(tt)
			plt.savefig('chiA-%02i.png' % i)

			plt.clf()
			plt.imshow(chi, interpolation='nearest', origin='lower',
					   vmin=-3, vmax=10.)
			plt.hot()
			plt.colorbar()
			plt.title(tt)
			plt.savefig('chiB-%02i.png' % i)
		

		#print 'Driving the tractor...'

		elif step == 'break':
			break

		elif step == 'source':
			print
			print 'Before createSource, catalog is:',
			tractor.getCatalog().printLong()
			print
			rtn = tractor.createSource()
			print
			print 'After  createSource, catalog is:',
			tractor.getCatalog().printLong()
			print

			if False:
				(sm,tryxy) = rtn[0]
				plt.clf()
				plt.imshow(sm, interpolation='nearest', origin='lower')
				plt.hot()
				plt.colorbar()
				ax = plt.axis()
				plt.plot([x for x,y in tryxy], [y for x,y in tryxy], 'b+')
				plt.axis(ax)
				plt.savefig('create-%02i.png' % i)

		elif step == 'psf':
			baton = (i,)
			tractor.optimizeAllPsfAtFixedComplexityStep(
				derivCallback=(psfDerivCallback, baton))

		elif step == 'psf2':
			tractor.increaseAllPsfComplexity()

		print 'Tractor cache has', len(tractor.cache), 'entries'


if __name__ == '__main__':
	main()
