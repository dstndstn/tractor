# Copyright 2011 Dustin Lang and David W. Hogg.  All rights reserved.

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import pyfits
import pylab as plt

from tractor import *

import numpy as np
from math import pi

class HSTTractor(Tractor):

	def createNewSource(self, img, x, y, ht):
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		#patch = img.getPsf().getPointSourcePatch(x, y)
		#print 'psf patch:', patch.shape
		#print 'psf patch: max', patch.max(), 'sum', patch.sum()
		#print 'new source peak height:', ht, '-> flux', ht/patch.max()
		#ht /= patch.max()
		return PointSource(PixPos([x,y]), Flux(ht))

def measure_sky_variance(img):
	x0,y0 = 5,7
	d = img[y0:,x0:] - img[:-y0,:-x0]
	mad = abs(d.ravel())
	I = np.argsort(mad)
	mad = mad[I[len(I)/2]]
	print 'median abs diff:', mad
	sigmasq = mad**2 * pi / 4.
	print 'sigma', np.sqrt(sigmasq)
	return sigmasq


def main():

	if True:
		P = pyfits.open('jbf108bzq_flt.fits')
		img = P[1].data
		err = P[2].data
		#dq  = P[3].data

		P = pyfits.open('jbf108bzq_dq1.fits')
		dq  = P[0].data

		#cut = [slice(900,1200), slice(2300,2600)]
		cut = [slice(1400,1700), slice(2300,2600)]
		img = img[cut]
		err = err[cut]
		dq  = dq[cut]

		skyvar = measure_sky_variance(img)
		skymed = np.median(img.ravel())
		skysig = np.sqrt(skyvar)
		print 'Estimate sky value', skymed, 'and sigma', skysig
		zrange = np.array([-3.,+10.]) * skysig + skymed
		invvar = 1. / (err**2)
		invvar[dq > 0] = 0

	else:
		# Dealing with cosmic rays is a PITA so use DRZ for now...
		P = pyfits.open('jbf108020_drz.fits')
		img = P[1].data
		wht = P[2].data
		ctx = P[3].data
		cut = [slice(1000,1300), slice(2300,2600)]
		img = img[cut]
		wht = wht[cut]

		# HACKs abound in what follows...
		skyvar = measure_sky_variance(img)
		invvar = wht / np.median(wht) / skyvar

		skymed = np.median(img.ravel())
		skysig = np.sqrt(skyvar)
		zrange = np.array([-3.,+10.]) * skysig + skymed

		# add in source noise to variance map
		# problem for the reader:  why *divide* by wht?
		srcvar = np.maximum(0, (img - skymed) / np.maximum(wht, np.median(wht)*1e-6))
		invvar = invvar / (1.0 + invvar * srcvar)


	plt.clf()
	plt.hist(img.ravel(), bins=np.linspace(zrange[0], zrange[1], 100))
	plt.savefig('hist.png')

	plt.clf()
	plt.imshow(img, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1]) #vmin=50, vmax=500)
	plt.hot()
	plt.colorbar()
	plt.savefig('img.png')
	
	plt.clf()
	plt.imshow(invvar, interpolation='nearest', origin='lower',
			  vmin=0., vmax=2./(skysig**2))
	plt.hot()
	plt.colorbar()
	plt.savefig('invvar.png')
	
	plt.clf()
	plt.imshow((img-skymed) * np.sqrt(invvar), interpolation='nearest', origin='lower')
	#vmin=-3, vmax=10.)
	plt.hot()
	plt.colorbar()
	plt.savefig('chi.png')

	plt.clf()
	plt.imshow((img-skymed) * np.sqrt(invvar), interpolation='nearest', origin='lower',
			  vmin=-3, vmax=10.)
	plt.hot()
	plt.colorbar()
	plt.savefig('chi2.png')

	# Initialize with a totally bogus Gaussian PSF model.
	psf = NCircularGaussianPSF([2.0], [1.0])

	# test it...
	if False:
		for i,x in enumerate(np.arange(17, 18.7, 0.1)):
			p = psf.getPointSourcePatch(x, x)
			img = p.getImage()
			print 'x', x, '-> sum', img.sum()
			plt.clf()
			x0,y0 = p.getX0(),p.getY0()
			h,w = img.shape
			plt.imshow(img, extent=[x0, x0+w, y0, y0+h],
					   origin='lower', interpolation='nearest')
			plt.axis([10,25,10,25])
			plt.title('x=%.1f' % x)
			plt.savefig('psf-%02i.png' % i)


	# We'll start by working in pixel coords
	wcs = NullWCS()

	# And counts
	photocal = NullPhotoCal()

	data = Image(data=img, invvar=invvar, psf=psf, wcs=wcs, sky=skymed,
				 photocal=photocal)
	
	tractor = HSTTractor([data])

	if False:
		X = tractor.getChiImages()
		chi = X[0]
		plt.clf()
		plt.imshow(chi, interpolation='nearest', origin='lower')
		plt.hot()
		plt.colorbar()
		plt.savefig('chi3.png')

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


ploti = 0

def psfDerivCallback(tractor, imagei, img, psf, steps, mod0, derivs, baton):
	(stepi,) = baton
	for i,deriv in enumerate(derivs):
		img = deriv.getImage()
		print 'derivate bounds:', img.min(), img.max()
		plt.clf()
		plt.imshow(deriv.getImage(), interpolation='nearest', origin='lower',
				   vmin=-1000, vmax=1000)
		plt.colorbar()
		plt.savefig('psfderiv-%02i-%i.png' % (stepi, i))


if __name__ == '__main__':
	main()

