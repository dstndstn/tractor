from math import ceil, floor, pi, sqrt, exp
import numpy as np
import random
#import scipy.sparse.linalg as sparse
#import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

'''
Duck types:

class Position(object):
	def setElement(self, i, val):
		pass
	def getElement(self, i):
		return 1.0


source 42, adjust position elements 0 (stepsize=1e-6), 1 (stepsize=1e-6)
source 43, adjust position elements 0, 1, 2

class Catalog(object):
	[sources]

	def getSources(self, image):
		return [source]

class Band(object):
	pass

class Source(object):
	'
	Position
	Flux
	SourceType
	(sourcetype-specific parameters)
	'
	def getParameter(self, i):
		if i < self.Nparams:
			return (val, step)
		return None

	def setParameter(self, i, val):
		pass

	def isInImage(self, image):
		return True



	def getFlux(self, band):
		return 42.0



class WCS(object):
	def positionToPixel(self, position, image):
		'
		position: list of Position objects
		image: Image object

		returns: xy: a list of pixel locations -- (x,y) doubles
		:            eg, [ (1.0, 2.0), (3.0, 4.0) ]
		
		'
		return xy



class Image(object):
	-invvar
	-wcs
	-psf
	-flux calib

	
'''


class Source(object):
	'''
	MUST BE HASHABLE!
	http://docs.python.org/glossary.html#term-hashable
	'''

	def getPosition(self):
		pass
	def getModelPatch(self, img):
		pass

	def numberOfFitParams(self):
		return 0
	# returns [ Patch, Patch, ... ] of length numberOfFitParams().
	def getFitParamDerivatives(self, img):
		return []
	# update parameters in this direction with this step size.
	def stepParams(self, dparams, alpha):
		pass
	

class PointSource(Source):
	def __init__(self, pos, flux):
		self.pos = pos
		self.flux = flux
	def __str__(self):
		return 'PointSource at ' + str(self.pos) + ' with flux ' + str(self.flux)
	def __repr__(self):
		return 'PointSource(' + repr(self.pos) + ', ' + repr(self.flux) + ')'

	def copy(self):
		return PointSource(self.pos.copy(), self.flux.copy())

	def __hash__(self):
		return (self.pos, self.flux).__hash__()
	def __eq__(self, other):
		return hash(self) == hash(other)

	def getPosition(self):
		return self.pos

	def getModelPatch(self, img):
		(px,py) = img.getWcs().positionToPixel(self.getPosition())
		patch = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		return patch * counts

	# [pos], flux
	def numberOfFitParams(self):
		return self.pos.dimension() + 1

	fluxstep = 0.1

	# returns [ Patch, Patch, ... ] of length numberOfFitParams().
	def getFitParamDerivatives(self, img):
		pos0 = self.getPosition()
		steps = pos0.getFitStepSizes(img)

		(px,py) = img.getWcs().positionToPixel(pos0)
		patch0 = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().fluxToCounts(self.flux)

		derivs = []
		for i in range(len(steps)):
			posx = pos0.copy()
			posx[i] += steps[i]
			(px,py) = img.getWcs().positionToPixel(posx)
			patchx = img.getPsf().getPointSourcePatch(px, py)
			dx = (patchx - patch0) * counts
			derivs.append(dx)
			
		df = (patch0 * counts) * PointSource.fluxstep
		derivs.append(df)
		return derivs

	# update parameters in this direction with this step size.
	def stepParams(self, dparams, alpha, img):
		pos = self.getPosition()
		assert(len(dparams) == (pos.dimension() + 1))
		pos += (dparams[:-1] * alpha)
		# for i,dp in dparams:
		#		pos[i] += dp * alpha

		dc = dparams[-1]
		newcounts = exp(PointSource.fluxstep * dc)
		self.flux = img.getPhotoCal().countsToFlux(newcounts)


def randomint():
	return int(random.random() * (2**32)) #(2**48))

class Image(object):
	def __init__(self, data=None, invvar=None, psf=None, sky=0, wcs=None,
				 photocal=None):
		self.data = data
		self.invvar = invvar
		self.psf = psf
		self.sky = sky
		self.wcs = wcs
		self.photocal = photocal

	def __getattr__(self, name):
		if name == 'shape':
			return self.data.shape

	# Any time an attribute is changed, update the "version" number to a random value.
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
		#object.__setattr__(self, 'version', randomint())
		self.setversion(randomint())
	def setversion(self, ver):
		object.__setattr__(self, 'version', ver)
	def getVersion(self):
		return self.version

	def numberOfPixels(self):
		(H,W) = self.data.shape
		return W*H
		
	def getError(self):
		return np.sqrt(self.invvar)
	def getImage(self):
		return self.data
	def getPsf(self):
		return self.psf
	def getWcs(self):
		return self.wcs
	def getPhotoCal(self):
		return self.photocal

class PhotoCal(object):
	def fluxToCounts(self, flux):
		pass
	def countsToFlux(self, counts):
		pass

class NullPhotoCal(object):
	def fluxToCounts(self, flux):
		return flux
	def countsToFlux(self, counts):
		return counts

class WCS(object):
	def positionToPixel(self, pos):
		pass

# useful when you're using raw pixel positions rather than RA,Decs
class NullWCS(WCS):
	def positionToPixel(self, pos):
		return pos

class Patch(object):
	def __init__(self, x0, y0, patch):
		self.x0 = x0
		self.y0 = y0
		self.patch = patch
	def getOrigin(self):
		return (self.x0,self.y0)
	def getPatch(self):
		return self.patch
	def getImage(self):
		return self.patch
	def getX0(self):
		return self.x0
	def getY0(self):
		return self.y0

	def getSlice(self):
		(ph,pw) = self.patch.shape
		return (slice(self.y0, self.y0+ph),
				slice(self.x0, self.x0+pw))

	def getPixelIndices(self, parent):
		(h,w) = self.shape
		(H,W) = parent.shape
		X,Y = np.meshgrid(np.arange(w), np.arange(h))
		return (Y.ravel() + self.y0) * W + X.ravel()

	def addTo(self, img, scale=1.):
		(ph,pw) = self.patch.shape
		img[self.y0:self.y0+ph, self.x0:self.x0+pw] += self.getImage() * scale

	def __getattr__(self, name):
		if name == 'shape':
			return self.patch.shape

	def __mul__(self, flux):
		return Patch(self.x0, self.y0, self.patch * flux)

	def __sub__(self, other):
		assert(isinstance(other, Patch))
		assert(self.x0 == other.getX0())
		assert(self.y0 == other.getY0())
		assert(self.shape == other.shape)
		return Patch(self.x0, self.y0, self.patch - other.patch)


class PSF(object):
	def applyTo(self, image):
		pass

	# return Patch, a rendering of a point source at the given pixel
	# coordinate.
	def getPointSourcePatch(self, px, py):
		pass

class NGaussianPSF(PSF):
	def __init__(self, sigmas, weights):
		'''
		Creates a new N-Gaussian (concentric, isotropic) PSF.

		sigmas: (list of floats) standard deviations of the components

		weights: (list of floats) relative weights of the components;
		given two components with weights 0.9 and 0.1, the total mass
		due to the second component will be 0.1.  These values will be
		normalized so that the total mass of the PSF is 1.0.

		eg,   NGaussianPSF([1.5, 4.0], [0.8, 0.2])
		'''
		self.sigmas = sigmas
		self.weights = weights

	def applyTo(self, image):
		from scipy.ndimage.filters import gaussian_filter
		# gaussian_filter normalizes the Gaussian; the output has ~ the
		# same sum as the input.
		
		res = np.zeros_like(image)
		for s,w in zip(self.sigmas, self.weights):
			res += w * gaussian_filter(image, s)
		res /= sum(self.weights)
		return res

	# returns a Patch object.
	def getPointSourcePatch(self, px, py):
		ix = int(round(px))
		iy = int(round(py))
		dx = px - ix
		dy = py - iy
		# HACK - MAGIC -- N sigma for rendering patches
		rad = int(ceil(max(self.sigmas) * 5.))
		sz = 2*rad + 1
		X,Y = np.meshgrid(np.arange(sz), np.arange(sz))
		X -= dx + rad
		Y -= dy + rad
		patch = np.zeros((sz,sz))
		x0 = ix - rad
		y0 = iy - rad
		R2 = (X**2 + Y**2)
		for s,w in zip(self.sigmas, self.weights):
			patch += w / (2.*pi*s**2) * np.exp(-0.5 * R2 / s**2)
		patch /= sum(self.weights)
		print 'sum of PSF patch:', patch.sum()
		return Patch(x0, y0, patch)

class Cache(dict):
	pass

class Catalog(list):
	def __hash__(self):
		return hash(tuple([hash(x) for x in self]))
	def deepcopy(self):
		return Catalog([x.copy() for x in self])


class Tractor(object):

	def __init__(self, image, catalog=[]):
		'''
		image: list of Image objects (data)
		catalog: list of Source objects
		'''
		self.images = image
		self.catalog = Catalog(catalog)

		self.cache = Cache()
		self.cachestack = []

	#def getImage(self, imgi):
	#eturn self.images[imgi]

	def optimizeCatalogAtFixedComplexityStep(self):
		'''
		-synthesize images

		-get all derivatives
		(taking numerical derivatives itself?)

		-build matrix

		-take step (try full step, back off)
		'''
		print 'Optimizing at fixed complexity'
		mods = self.getModelImages()
 
		# need all derivatives  dChi / dparam
		# for each pixel in each image
		#  and each parameter in each source

		nparams = [src.numberOfFitParams() for src in self.catalog]
		row0 = np.cumsum([0] + nparams)

		npixels = [img.numberOfPixels() for img in self.images]
		col0 = np.cumsum([0] + npixels)
		# [ 0, (W0*H0), (W0*H0 + W1*H1), ... ]

		sprows = []
		spcols = []
		spvals = []

		for j,src in enumerate(self.catalog):
			#params = src.getFitParams()
			#assert(len(params) == nparams[j])

			for i,img in enumerate(self.images):
				#
				#patch = self.getModelPatch(img, src)
				#if patch is None:
				#	continue

				# Now we know that this src/img interact
				# Get derivatives (in this image) of params
				derivs = src.getFitParamDerivatives(img)
				# derivs = [ Patch, Patch, ... ] (list of length len(params))
				assert(len(derivs) == nparams[j])

				print 'Got derivatives:', derivs

				errs = img.getError()

				# Add to the sparse matrix of derivatives:
				#cols = col0[i] + pix
				for p,deriv in enumerate(derivs):
					pix = deriv.getPixelIndices(img)
					# (in the parent image)
					assert(all(pix < npixels[i]))

					# (grab non-zero indices)
					dimg = deriv.getImage()
					nz = np.flatnonzero(dimg)
					cols = col0[i] + pix[nz]
					rows = np.zeros_like(cols) + row0[j] + p
					#rows = np.zeros(len(cols), int) + row0[j] + p
					vals = dimg.ravel()[nz]
					w = errs[deriv.getSlice()].ravel()[nz]
					assert(vals.shape == w.shape)

					sprows.append(rows)
					spcols.append(cols)
					spvals.append(vals * w)

		sprows = np.hstack(sprows)
		spcols = np.hstack(spcols)
		spvals = np.hstack(spvals)

		print 'Matrix elements:', len(sprows)
		urows = np.unique(sprows)
		print 'Unique rows:', len(urows)

		# Build sparse matrix
		A = csr_matrix((spvals, (sprows, spcols)))

		# b = -weighted residuals
		#b = ((data - image) * sqrt(invvar)).ravel()
		b = np.zeros(np.max(spcols))

		lsqropts = {}

		# Run lsqr()
		(X, istop, niters, r1norm, r2norm, anorm, acond,
		 arnorm, xnorm, var) = lsqr(A, b, show=not quiet, **lsqropts)

		# Unpack.

		pBefore = self.getLogProb()
		print 'log-prob before:', pBefore

		for alpha in [2.**-np.arange(5)]:
			print 'Stepping with alpha =', alpha

			oldcat = self.catalog.deepcopy()
			self.pushCache()

			for j,src in enumerate(self.catalog):
				dparams = X[row0[j] : row0[j] + nparams[j]]
				assert(len(dparams) == nparams[j])
				src.stepParams(dparams, alpha)

			pAfter = self.getLogProb()
			print 'log-prob after:', pAfter

			if pAfter > pBefore:
				print 'Accepting step!'
				self.mergeCache()
				break

			print 'Rejecting step!'
			self.popCache()
			# revert the catalog
			self.catalog = oldcat

		# if we get here, this step was rejected.


	def getModelPatch(self, img, src):
		#pixpos = img.getWcs().positionToPixel(src.getPosition())
		return src.getModelPatch(img)#, pixpos)

	# ??
	def getModelImageNoCache(self, img):
		mod = np.zeros_like(img.getImage())
		mod += img.sky
		# HACK -- add sources...
		for src in self.catalog:
			# get model patch for this src in this img?
			# point sources vs extended
			# extended sources -- might want to render pre-psf then apply psf in one shot?
			patch = self.getModelPatch(img, src)
			patch.addTo(mod)

		return mod

	def getModelImage(self, img):
		# dependencies of this model image:
		# img.sky, img.psf, img.wcs, sources that overlap.
		#deps = hash((img.getVersion(), hash(self.catalog)))
		deps = (img.getVersion(), hash(self.catalog))
		print 'Model image:'
		print '  catalog', self.catalog
		print '  -> deps', deps
		mod = self.cache.get(deps, None)
		if mod is not None:
			print '  Cache hit!'
		else:
			mod = self.getModelImageNoCache(img)
			print 'Caching model image'
			self.cache[deps] = mod
		return mod

	def getModelImages(self):
		mods = []
		for img in self.images:
			mod = self.getModelImage(img)
			mods.append(mod)
		return mods

	def getChiImages(self):
		mods = self.getModelImages()
		chis = []
		for img,mod in zip(self.images, mods):
			chis.append((img.getImage() - mod) * img.getError())
		return chis

	#def findPeaks(self, img, thresh):

	def createNewSource(self, img, x, y, height):
		return None

	def getLogProb(self):
		chisq = 0.
		for i,chi in enumerate(self.getChiImages()):
			chisq += (chi ** 2).sum()
		return -0.5 * chisq

	def pushCache(self):
		self.cachestack.append(self.cache)
		self.cache = self.cache.copy()

	def mergeCache(self):
		# drop the top of the stack.
		self.cachestack.pop()

	def popCache(self):
		self.cache = self.cachestack.pop()

	#
	#def startTryUpdate(self):

	def createSource(self):
		'''
		-synthesize images
		-look for "promising" Positions with "positive" residuals
		---chi image, PSF smooth, propose positions?
		-instantiate new source (Position, flux, PSFType)
		-local optimizeAtFixedComplexity
		'''
		for i,chi in enumerate(self.getChiImages()):
			img = self.images[i]
			# PSF-correlate
			sm = img.getPsf().applyTo(chi)
			# find peaks, create sources
			#return sm

			# HACK -- magic value 10
			#pks = self.findPeaks(sm, 10)

			# find single peak pixel, create source
			I = np.argmax(sm)
			(H,W) = sm.shape
			ix = I%W
			iy = I/W
			# this is just the peak pixel height...
			#ht = img.getImage()[iy,ix] - img.getSky()
			ht = (img.getImage() - self.getModelImage(img))[iy,ix]
			#ht = img.getImage()[iy,ix] - img.getSky()
			src = self.createNewSource(img, ix, iy, ht)

			# try adding the new source...
			#self.startTryUpdate()
			#self.catalog.append(src)
			#self.finishTryUpdate()

			pBefore = self.getLogProb()
			print 'log-prob before:', pBefore

			self.pushCache()
			oldcat = self.catalog.deepcopy()

			self.catalog.append(src)
			self.optimizeCatalogAtFixedComplexityStep()

			pAfter = self.getLogProb()
			print 'log-prob after:', pAfter

			if pAfter > pBefore:
				print 'Keeping new source'
				self.mergeCache()
				
			else:
				print 'Rejecting new source'
				self.popCache()
				# revert the catalog
				self.catalog = oldcat

			pEnd = self.getLogProb()
			print 'log-prob at finish:', pEnd


		#ims = self.getSyntheticImages()
		#for model,data in zip(ims, self.images):
		#		chi

	def modifyComplexity(self):
		'''
		-synthesize images
		-for all sources?
		---for all sourceTypes (including None)
		-----change source.sourceType -> sourceType
		-----local optimizeAtFixedComplexity
		'''
		pass
	
	def step(self):
		'''
		'''
		pass

