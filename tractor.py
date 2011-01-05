from math import ceil, floor, pi, sqrt
import numpy as np
import random

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
		x0,y0,patch = img.getPsf().getPointSourcePatch(px, py)
		return (x0,y0, self.flux * patch)


def randomint():
	return int(random.random() * (2**32)) #(2**48))

class Image(object):
	def __init__(self, data=None, invvar=None, psf=None, sky=0, wcs=None):
		self.data = data
		self.invvar = invvar
		self.psf = psf
		self.sky = sky
		self.wcs = wcs

	# Any time an attribute is changed, update the "version" number to a random value.
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
		#object.__setattr__(self, 'version', randomint())
		self.setversion(randomint())
	def setversion(self, ver):
		object.__setattr__(self, 'version', ver)
	def getVersion(self):
		return self.version
		
	def getError(self):
		return np.sqrt(self.invvar)
	def getImage(self):
		return self.data
	def getPsf(self):
		return self.psf
	def getWcs(self):
		return self.wcs

class WCS(object):
	def positionToPixel(self, pos):
		pass

# useful when you're using raw pixel positions rather than RA,Decs
class NullWCS(WCS):
	def positionToPixel(self, pos):
		return pos
	
class PSF(object):
	def applyTo(self, image):
		pass

	# return (x0, y0, patch), a rendering of a point source at the given pixel
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
		return x0,y0,patch

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
		self.data = image
		self.catalog = Catalog(catalog)

		self.cache = Cache()
		self.cachestack = []

	#def getImage(self, imgi):
	#eturn self.data[imgi]

	def optimizeAtFixedComplexityStep(self):
		'''
		-synthesize images

		-get all derivatives
		(taking numerical derivatives itself?)

		-build matrix

		-take step (try full step, back off)
		'''
		print 'Optimizing at fixed complexity'

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
			(x0,y0,patch) = self.getModelPatch(img, src)
			(ph,pw) = patch.shape
			mod[y0:y0+ph, x0:x0+pw] += patch

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
		for img in self.data:
			mod = self.getModelImage(img)
			mods.append(mod)
		return mods

	def getChiImages(self):
		mods = self.getModelImages()
		chis = []
		for img,mod in zip(self.data, mods):
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
			img = self.data[i]
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
			self.optimizeAtFixedComplexityStep()

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
		#for model,data in zip(ims, self.data):
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

