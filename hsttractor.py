
from tractor import *

import numpy as np
from math import pi

class Flux(object):
	def __init__(self, val):
		self.val = val

	def __repr__(self):
		return 'Flux(%g)' % self.val
	def __str__(self):
		return 'Flux: %g' % self.val

	def getValue(self):
		return self.val

	def copy(self):
		return Flux(self.val)

	def numberOfFitParams(self):
		return 1
	def getFitStepSizes(self, img):
		#return [(Tractor.LOG, 0.1)]
		return [0.1]
	def stepParam(self, parami, delta):
		assert(parami == 0)
		#self *= exp(delta)
		self.val += delta
		#print 'Flux: stepping by', delta
		#self += delta
	def stepParams(self, params):
		assert(len(params) == self.numberOfFitParams())
		for i in range(len(params)):
			self.stepParam(i, params[i])



#class PixPos(tuple):

class PixPos(list):
	def copy(self):
		return PixPos([self[0],self[1]])

	def getDimension(self):
		return 2
	def getFitStepSizes(self, img):
		#return [(Tractor.LINEAR, 0.1), (Tractor.LINEAR, 0.1)]
		return [0.1, 0.1]
	def stepParam(self, parami, delta):
		assert(parami >= 0)
		assert(parami <= 1)
		self[parami] += delta
	def stepParams(self, params):
		assert(len(params) == self.getDimension())
		for i in range(len(params)):
			self[i] += params[i]
		#print 'PixPos: stepping by', params

	def __hash__(self):
		# convert to tuple
		return hash((self[0], self[1]))

	#def __iadd__(self, X):
	#  	self[0] += X[0]
	#	self[1] += X[1]


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

if __name__ == '__main__':
	import pyfits

	import matplotlib
	matplotlib.use('Agg')
	import pylab as plt

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
	psf = NGaussianPSF([2.0], [1.0])

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

	Nsrc = 50
	for i in range(Nsrc+1):
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
		plt.savefig('mod-%02i.png' % i)

		chis = tractor.getChiImages()
		chi = chis[0]

		plt.clf()
		plt.imshow(chi, interpolation='nearest', origin='lower')
		plt.hot()
		plt.colorbar()
		plt.savefig('chiA-%02i.png' % i)

		plt.clf()
		plt.imshow(chi, interpolation='nearest', origin='lower',
				   vmin=-3, vmax=10.)
		plt.hot()
		plt.colorbar()
		plt.savefig('chiB-%02i.png' % i)

		if i == Nsrc:
			break

		print
		print 'Before createSource, catalog is:', #tractor.getCatalog()
		tractor.getCatalog().printLong()
		print

		tractor.createSource()

		print
		print 'After  createSource, catalog is:', #tractor.getCatalog()
		tractor.getCatalog().printLong()
		print

