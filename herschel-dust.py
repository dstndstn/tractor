"""
This code is part of the Tractor.

Copyright 2012 David W. Hogg

This code builds a simultaneous emitting dust model for a set of
aligned Herschel maps.

It has various things related to Herschel data hard-coded.
"""

import matplotlib
matplotlib.use('Agg')
import pylab as plt

import numpy as np
from tractor import *
from tractor.cache import Cache
import pyfits
from astrometry.util.util import Tan, tan_wcs_resample, log_init
from astrometry.util.multiproc import multiproc
from astrometry.util.file import *
import multiprocessing

class Physics(object):
	# all the following from physics.nist.gov
	hh = 6.62606957e-34 # J s
	cc = 299792458. # m s^{-1}
	kk = 1.3806488e-23 # J K^{-1}
	#logtwohcsq = np.log(2. * hh * cc ** 2)

	@staticmethod
	def black_body(lam, lnT):
		"""
		Compute the black-body formula, for a given lnT.
		"""
		return (2. * Physics.hh * Physics.cc ** 2 * lam ** -5 /
				(np.exp(Physics.hh * Physics.cc / (lam * Physics.kk * np.exp(lnT))) - 1.))

	# @staticmethod
	# def log_black_body(lam, lnT):
	# 	"""
	# 	Compute the black-body formula, for a given lnT.
	# 	"""
	# 	return (Physics.logtwohcsq +
	# 			-5.0 * np.log(lam) +
	# 			-np.log(np.exp(Physics.hh * Physics.cc / (lam * Physics.kk * np.exp(lnT))) - 1.))

class DustBrightness(ParamList):
	@staticmethod
	def getNamedParams():
		return dict(logsolidangle=0, logtemperature=1, emissivity=2)
	def getStepSizes(self, *args):
		#return [1., 1., 1.]
		return [1e-2, 1e-2, 1e-2]

class DustPhotoCal(ParamList):
	# MAGIC number: wavelength scale for emissivity model
	lam0 = 1.e-4 # m

	def __init__(self, lam, Mjypersrperdn):
		'''
		lam: central wavelength of filter in microns

		Mjypersrperdn: photometric calibration in mega-Jansky per
		steradian per DN.


		self.cal is in SI units (should be the same as the black_body
		formula).  CHECK this.
		'''
		self.lam = lam
		self.cal = Mjypersrperdn / (1e-20)
		# No (adjustable) params
		super(DustPhotoCal,self).__init__()
		
	def brightnessToCounts(self, brightness):
		br = (brightness.logsolidangle +
			  np.log(Physics.black_body(self.lam, brightness.logtemperature)) +
			  brightness.emissivity * np.log(self.lam / DustPhotoCal.lam0))
		return self.cal * np.exp(br)


class CachingPointSource(PointSource):

	cache = Cache(100000)

	def getModelPatch(self, img):
		patch = self._getUnitPatch(img)
		counts = img.getPhotoCal().brightnessToCounts(self.brightness)
		return patch * counts

	def _getUnitPatch(self, img):
		##key = hash((img.getPsf().hashkey(), img.getWcs().hashkey(), self.getPosition()))

		### ASSUME no change in image PSF and WCS or PointSource position!!!
		#key = hash((img.getPsf().hashkey(), img.getWcs().hashkey(), self.getPosition()))
		key = hash((id(img), id(self)))
		
		patch = CachingPointSource.cache.get(key, None)
		if patch is None:
			(px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
			patch = img.getPsf().getPointSourcePatch(px, py)
			CachingPointSource.cache.put(key, patch)
		return patch

	def getParamDerivatives(self, img):
		patch0 = self._getUnitPatch(img)
		#(px0,py0) = img.getWcs().positionToPixel(pos0, self)
		#patch0 = img.getPsf().getPointSourcePatch(px0, py0)
		counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
		derivs = []

		# Position
		if not self.isParamFrozen('pos'):
			pos0 = self.getPosition()
			psteps = pos0.getStepSizes(img)
			pvals = pos0.getParams()
			for i,pstep in enumerate(psteps):
				oldval = pos0.setParam(i, pvals[i] + pstep)
				(px,py) = img.getWcs().positionToPixel(pos0, self)
				patchx = img.getPsf().getPointSourcePatch(px, py)
				pos0.setParam(i, oldval)
				dx = (patchx - patch0) * (counts0 / pstep)
				dx.setName('d(ptsrc)/d(pos%i)' % i)
				derivs.append(dx)
		# Brightness
		if not self.isParamFrozen('brightness'):
			bsteps = self.brightness.getStepSizes(img)
			bvals = self.brightness.getParams()

			bnames = self.brightness.getParamNames()

			for i,bstep in enumerate(bsteps):
				oldval = self.brightness.setParam(i, bvals[i] + bstep)
				countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
				print 'Stepping param', bnames[i], 'by', bstep, 'takes counts from', counts0, 'to', countsi
				self.brightness.setParam(i, oldval)
				df = patch0 * ((countsi - counts0) / bstep)
				df.setName('d(ptsrc)/d(bright%i)' % i)
				derivs.append(df)
		return derivs



class NpArrayParams(ParamList):
	'''
	An implementation of Params that holds values in an np.ndarray
	'''
	def __init__(self, a):
		self.a = np.array(a)
		super(NpArrayParams, self).__init__()
		del self.vals
		# from NamedParams...
		# active/inactive
		self.liquid = [True] * self._numberOfThings()

	def __getattr__(self, name):
		if name == 'vals':
			return self.a.ravel()
		if name in ['shape',]:
			return getattr(self.a, name)
		raise AttributeError(name + ': no such attribute in NpArrayParams.__getattr__')


class DustSheet(MultiParams):
	'''
	A Source composed of a a grid of dust parameters
	'''
	@staticmethod
	def getNamedParams():
		return dict(logsolidangle=0, logtemperature=1, emissivity=2)

	def __init__(self, logsolidangle, logtemperature, emissivity, wcs):
		assert(logsolidangle.shape == logtemperature.shape)
		assert(logsolidangle.shape == emissivity.shape)
		assert(logsolidangle.shape == (wcs.get_height(), wcs.get_width()))
		super(DustSheet, self).__init__(NpArrayParams(logsolidangle),
										NpArrayParams(logtemperature),
										NpArrayParams(emissivity))
		self.wcs = wcs

	def __getattr__(self, name):
		if name == 'shape':
			return self.logsolidangle.shape

	def getModelPatch(self, img):
		# Compute emission in the native grid
		# Resample onto img grid + do PSF convolution in one shot
		H,W = self.shape

		# This takes advantage of the coincidence that our DustPhotoCal does the right thing
		# with numpy arrays.
		counts = img.getPhotoCal().brightnessToCounts(self)
		# Thanks to NpArrayParams they get ravel()'d down to 1-d, so
		# reshape back to 2d
		counts = counts.reshape(self.shape).astype(np.float32)

		#print 'Counts:', counts.shape

		# I am weak... do resampling + convolution in separate shots
		iH,iW = img.shape
		rim = np.zeros((iH,iW), np.float32)
		imwcs = img.getWcs()

		## ASSUME TAN WCS on both the DustSheet and img.
		res = tan_wcs_resample(self.wcs, imwcs.wcs, counts, rim, 2)
		assert(res == 0)

		## ASSUME the PSF has "applyTo"
		outimg = img.getPsf().applyTo(rim)

		return Patch(0, 0, outimg)

	def getStepSizes(self, *args, **kwargs):
		return ([0.1] * len(self.logsolidangle) +
				[0.1] * len(self.logtemperature) +
				[0.1] * len(self.emissivity))

	def getParamDerivatives(self, img):
		# Super-naive!!
		p0 = self.getParams()
		mod0 = self.getModelPatch(img)
		derivs = []
		for i,step in enumerate(self.getStepSizes()):
			print 'Img', img.name, 'deriv', i
			oldval = self.setParam(i, p0[i] + step)
			modi = self.getModelPatch(img)
			self.setParam(i, oldval)
			d = (modi - mod0) * (1./step)
			derivs.append(d)
		return derivs

def makeplots(tractor, step):
	print 'Rendering synthetic images...'

	mods = tractor.getModelImages()
	for i,mod in enumerate(mods):
		tim = tractor.getImage(i)
		ima = dict(interpolation='nearest', origin='lower',
				   vmin=tim.zr[0], vmax=tim.zr[1])
		plt.clf()
		plt.imshow(mod, **ima)
		plt.gray()
		plt.colorbar()
		plt.title(tim.name)
		plt.savefig('model-%i-%02i.png' % (i, step))

		if step == 0:
			plt.clf()
			plt.imshow(tim.getImage(), **ima)
			plt.gray()
			plt.colorbar()
			plt.title(tim.name)
			plt.savefig('data-%i.png' % (i))

		plt.clf()
		# tractor.getChiImage(i), 
		plt.imshow((tim.getImage() - mod) * tim.getInvError(),
				   interpolation='nearest', origin='lower',
				   vmin=-5, vmax=+5)
		plt.gray()
		plt.colorbar()
		plt.title(tim.name)
		plt.savefig('chi-%i-%02i.png' % (i, step))

def main():
	import optparse
	import logging
	import sys

	###
	log_init(3)
	
	parser = optparse.OptionParser()
	parser.add_option('--threads', dest='threads', default=1, type=int, help='Use this many concurrent processors')
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')
	opt,args = parser.parse_args()

	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	mp = multiproc(opt.threads, wrap_all=True)

	"""
	Brittle function to read Groves data sample and make the
	things we need for fitting.
	
	# issues:
	- Need to read the WCS too.
	- Need, for each image, to construct the rectangular
	nearest-neighbor interpolation indices to image 0.
	- Can I just blow up the SPIRE images with interpolation?
	"""
	dataList = [
		#('m31_brick15_PACS100.fits',   7.23,  7.7),
		('m31_brick15_PACS160.fits',   3.71, 12.0),
		('m31_brick15_SPIRE250.fits',  None, 18.0),
		('m31_brick15_SPIRE350.fits',  None, 25.0),
		('m31_brick15_SPIRE500.fits',  None, 37.0),
		]

	print 'Reading images...'
	tims = []
	for i, (fn, noise, fwhm) in enumerate(dataList):
		print
		print 'Reading', fn
		P = pyfits.open(fn)
		image = P[0].data.astype(np.float32)
		#wcs = Tan(fn, 0)
		H,W = image.shape
		hdr = P[0].header
		wcs = Tan(hdr['CRVAL1'], hdr['CRVAL2'],
				  hdr['CRPIX1'], hdr['CRPIX2'],
				  hdr['CDELT1'], 0., 0., hdr['CDELT2'], W, H)
		assert(hdr['CROTA2'] == 0.)
		if noise is None:
			noise = float(hdr['NOISE'])
		print 'Noise', noise
		print 'Median image value', np.median(image)
		invvar = np.ones_like(image) / (noise**2)

		skyval = np.percentile(image, 5)
		sky = ConstantSky(skyval)

		lam = float(hdr['FILTER'])
		# calibrated, yo
		print 'Lambda', lam
		assert(hdr['BUNIT'] == 'MJy/Sr')
		pcal = DustPhotoCal(lam, 1.)
		#nm = '%s %i' % (hdr['INSTRUME'], lam)
		nm = fn.replace('.fits', '')
		#zr = noise * np.array([-3, 10]) + skyval
		zr = np.array([np.percentile(image.ravel(), p) for p in [1, 99]])
		print 'Pixel scale:', wcs.pixel_scale()
		# meh
		sigma = fwhm / wcs.pixel_scale() / 2.35
		print 'PSF sigma', sigma, 'pixels'
		psf = NCircularGaussianPSF([sigma], [1.])
		twcs = FitsWcs(wcs)
		tim = Image(data=image, invvar=invvar, psf=psf, wcs=twcs,
					sky=sky, photocal=pcal, name=nm)
		tim.zr = zr
		tims.append(tim)

		# plt.clf()
		# plt.hist(image.ravel(), 100)
		# plt.title(nm)
		# plt.savefig('im%i-hist.png' % i)

	plt.clf()
	for tim,c in zip(tims, ['b','g','y',(1,0.5,0),'r']):
		H,W = tim.shape
		twcs = tim.getWcs()
		rds = []
		for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5),(0.5,0.5)]:
			rd = twcs.pixelToPosition(x,y)
			rds.append(rd)
		rds = np.array(rds)
		plt.plot(rds[:,0], rds[:,1], '-', color=c, lw=2, alpha=0.5)
	plt.savefig('radec.png')

	print 'Creating dust sheet...'
	N = 5

	# Build a WCS for the dust sheet to match the first image (assuming it's square and axis-aligned)
	
	wcs = tims[0].getWcs().wcs
	r,d = wcs.radec_center()
	H,W = tims[0].shape
	scale = wcs.pixel_scale()
	scale *= float(W)/N / 3600.
	c = float(N)/2. + 0.5
	dwcs = Tan(r, d, c, c, scale, 0, 0, scale, N, N)

	rds = []
	H,W = N,N
	for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5),(0.5,0.5)]:
		r,d = dwcs.pixelxy2radec(x,y)
		rds.append((r,d))
	rds = np.array(rds)
	plt.plot(rds[:,0], rds[:,1], 'k-', lw=1, alpha=1)
	plt.savefig('radec2.png')

	pixscale = dwcs.pixel_scale()
	logsa = np.log(1e-3 * ((pixscale / 3600 / (180./np.pi))**2))

	logsa = np.zeros((H,W)) + logsa
	logt = np.zeros((H,W)) + np.log(17.)
	emis = np.zeros((H,W)) + 2.

	#X,Y = np.meshgrid(np.arange(W), np.arange(H))
	#logsa += X*0.1
	#logt += Y*0.1
	#emis += Y*0.1
	
	ds = DustSheet(logsa, logt, emis, dwcs)

	print 'DustSheet:', ds
	#print 'np', ds.numberOfParams()
	#print 'pn', ds.getParamNames()
	#print 'p', ds.getParams()

	cat = Catalog()
	cat.append(ds)
	
	tractor = Tractor(Images(*tims), cat)
	tractor.mp = mp

	makeplots(tractor, 0)

	for im in tractor.getImages():
		im.freezeAllBut('sky')

	for i in range(10):
		#tractor.optimize(damp=10.)
		tractor.optimize(damp=1., alphas=[1e-3, 1e-2, 0.1, 0.3, 1., 3., 10., 30., 100.])
		makeplots(tractor, 1 + i)

	pickle_to_file(tractor, 'herschel.pickle')

	

def old():
	print 'Creating point sources...'
	# Create grid of PointSources of dust.
	twcs = tims[0].wcs
	wcs = twcs.wcs
	# arcsec/pix
	pixscale = wcs.pixel_scale()
	S = 5
	logsa = np.log(1e-3 * ((pixscale * S / 3600 / (180./np.pi))**2))
	XX,YY = np.meshgrid(np.arange(0, wcs.get_width(), S),
						np.arange(0, wcs.get_height(), S))
	srcgridh,srcgridw = XX.shape
	print 'Source grid:', srcgridw, 'x', srcgridh
	cat = Catalog()
	for i,(x,y) in enumerate(zip(XX.ravel(),YY.ravel())):
		pos = twcs.pixelToPosition(x,y)
		bright = DustBrightness(logsa, np.log(17.), 2.)
		ps = CachingPointSource(pos, bright)
		ps.freezeParam('pos')
		if i == 0:
			print 'pos', pos
			print 'brightness', bright
			print 'ps', ps
			print 'params', zip(ps.getParamNames(), ps.getParams())
			for tim in tims:
				print 'Counts in', tim.name, '=', tim.getPhotoCal().brightnessToCounts(bright)

		cat.append(ps)

	tractor = Tractor(Images(*tims), cat)

	tractor.mp = mp
	tractor.cache = Cache(maxsize=5*len(cat))

	for im in tractor.getImages():
		im.freezeAllBut('sky')

	def makeplots(tractor, step):
		print 'Rendering synthetic images...'

		mods = tractor.getModelImages()
		for i,mod in enumerate(mods):
			tim = tractor.getImage(i)
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=tim.zr[0], vmax=tim.zr[1])
			plt.clf()
			plt.imshow(mod, **ima)
			plt.gray()
			plt.colorbar()
			plt.title(tim.name)
			plt.savefig('model-%i-%02i.png' % (i, step))

			if step == 0:
				plt.clf()
				plt.imshow(tim.getImage(), **ima)
				plt.gray()
				plt.colorbar()
				plt.title(tim.name)
				plt.savefig('data-%i.png' % (i))

			plt.clf()
			# tractor.getChiImage(i), 
			plt.imshow((tim.getImage() - mod) * tim.getInvError(),
					   interpolation='nearest', origin='lower',
					   vmin=-5, vmax=+5)
			plt.gray()
			plt.colorbar()
			plt.title(tim.name)
			plt.savefig('chi-%i-%02i.png' % (i, step))

	makeplots(tractor, 0)

	print 'Tcache', tractor.cache.printStats()
	print 'CPS', CachingPointSource.cache.printStats()

	j = 1

	# Fit sky
	if False:
		print 'Optimizing sky...'
		tractor.freezeParam('catalog')
		print 'Fitting:', tractor.getParamNames()
		tractor.optimize(alphas=[1.])
		makeplots(tractor, 1)
		tractor.thawParam('catalog')

		print 'Tcache', tractor.cache.printStats()
		print 'CPS', CachingPointSource.cache.printStats()

	#print 'Fitting:', tractor.getParamNames()
	#tractor.optimize()
	#makeplots(tractor, j)
	#j+=1
	#return

	print 'Optimizing %i sources...' % len(tractor.catalog)
	tractor.thawParam('catalog')
	tractor.freezeParam('images')
	tractor.catalog.freezeAllParams()
	N = 3
	# Fit in NxN blocks of sources.
	for i0 in range(0, srcgridh, N):
		II = range(i0, min(srcgridh, i0+N))
		for j0 in range(0, srcgridw, N):
			JJ = range(j0, min(srcgridw, j0+N))
			mi,mj = np.meshgrid(II,JJ)
			srcs = [ii * srcgridw + jj for ii,jj in zip(mi.ravel(), mj.ravel())]
			print 'Fitting sources', srcs
			for ii in srcs:
				print 'type', type(ii)
				tractor.catalog.thawParam(int(ii))

			print 'Pre:'
			v0 = tractor.getParams()
			for nm,val in zip(tractor.getParamNames(), tractor.getParams()):
				print '  ', nm, '=', val

			tractor.optimize(alphas=[1e-3, 1e-2, 0.1, 0.3, 1., 3., 10., 30., 100.], damp=10.)

			print 'Post:'
			for nm,val0,val in zip(tractor.getParamNames(), v0, tractor.getParams()):
				print '  ', nm, ':', val0, ' -> ', val

			for ii in srcs:
				tractor.catalog.freezeParam(int(ii))

			makeplots(tractor, j)
			j += 1


	#for i in range(len(tractor.catalog)):
	# 	print 'Fitting source', i
	# 	tractor.catalog.thawParam(i)
	# 	print 'Fitting:', tractor.getParamNames()
	# 	print 'Source:', tractor.catalog[i]
	# 	tractor.optimize(alphas=[1e-3, 1e-2, 0.1, 0.3, 1., 3., 10., 30., 100.], damp=10.)
	# 	print 'After fitting:', tractor.catalog[i]
	# 	tractor.catalog.freezeParam(i)
	# 
	# 	print 'Tcache', tractor.cache.printStats()
	# 	print 'CPS', CachingPointSource.cache.printStats()
	# 
	# 	#if (i+1) % 10 == 0:
	# 	if True:
	# 		makeplots(tractor, j)
	# 		j += 1


if __name__ == '__main__':
	#main()
	import cProfile
	import sys
	from datetime import tzinfo, timedelta, datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	#sys.exit(0)
