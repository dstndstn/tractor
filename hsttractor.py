
from tractor import *

import numpy as np
from math import pi

class Flux(float):
	def copy(self):
		return Flux(self)

class PixPos(tuple):
	def copy(self):
		return PixPos(self[0],self[1])


class HSTTractor(Tractor):

	def createNewSource(self, img, x, y, ht):
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		x0,y0,patch = img.getPsf().getPointSourcePatch(x, y)
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

if __name__ == '__main__':
	import pyfits

	import matplotlib
	matplotlib.use('Agg')
	import pylab as pl

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


	pl.clf()
	pl.hist(img.ravel(), bins=np.linspace(zrange[0], zrange[1], 100))
	pl.savefig('hist.png')

	pl.clf()
	pl.imshow(img, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1]) #vmin=50, vmax=500)
	pl.hot()
	pl.colorbar()
	pl.savefig('img.png')
	
	pl.clf()
	pl.imshow(invvar, interpolation='nearest', origin='lower',
			  vmin=0., vmax=2./(skysig**2))
	pl.hot()
	pl.colorbar()
	pl.savefig('invvar.png')
	
	pl.clf()
	pl.imshow((img-skymed) * np.sqrt(invvar), interpolation='nearest', origin='lower')
	#vmin=-3, vmax=10.)
	pl.hot()
	pl.colorbar()
	pl.savefig('chi.png')

	pl.clf()
	pl.imshow((img-skymed) * np.sqrt(invvar), interpolation='nearest', origin='lower',
			  vmin=-3, vmax=10.)
	pl.hot()
	pl.colorbar()
	pl.savefig('chi2.png')


	# Initialize with a totally bogus Gaussian PSF model.
	psf = NGaussianPSF([2.0], [1.0])

	# We'll start by working in pixel coords
	wcs = NullWCS()

	data = Image(data=img, invvar=invvar, psf=psf, wcs=wcs, sky=skymed)
	
	tractor = HSTTractor([data])

	if False:
		X = tractor.getChiImages()
		chi = X[0]
		pl.clf()
		pl.imshow(chi, interpolation='nearest', origin='lower')
		pl.hot()
		pl.colorbar()
		pl.savefig('chi3.png')

	mods = tractor.getModelImages()
	mod = mods[0]

	pl.clf()
	pl.imshow(mod, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1])
	pl.hot()
	pl.colorbar()
	pl.savefig('mod0.png')

	tractor.createSource()

	mods = tractor.getModelImages()
	mod = mods[0]

	pl.clf()
	pl.imshow(mod, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1])
	pl.hot()
	pl.colorbar()
	pl.savefig('mod1.png')
