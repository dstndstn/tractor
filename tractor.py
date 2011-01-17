from math import ceil, floor, pi, sqrt, exp
import numpy as np
import random
#import scipy.sparse.linalg as sparse
#import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from astrometry.util.miscutils import get_overlapping_region

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

	def __hash__(self):
		return hash(self.hashkey())
	def hashkey(self):
		return ('Source',)

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
	def getParams(self):
		return []
	def setParams(self, p):
		pass
	
class PointSource(Source):
	def __init__(self, pos, flux):
		self.pos = pos
		self.flux = flux
	def __str__(self):
		return 'PointSource at ' + str(self.pos) + ' with ' + str(self.flux)
	def __repr__(self):
		return 'PointSource(' + repr(self.pos) + ', ' + repr(self.flux) + ')'

	def copy(self):
		return PointSource(self.pos.copy(), self.flux.copy())

	#def __hash__(self):
	#	return hash((self.pos, self.flux))
	def hashkey(self):
		return ('PointSource', self.pos.hashkey(), self.flux.hashkey())

	def __eq__(self, other):
		return hash(self) == hash(other)

	def getPosition(self):
		return self.pos
	def getFlux(self):
		return self.flux

	def getModelPatch(self, img):
		(px,py) = img.getWcs().positionToPixel(self.getPosition())
		patch = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		return patch * counts

	# [pos], [flux]
	def numberOfFitParams(self):
		return self.pos.getDimension() + self.flux.numberOfFitParams()

	# returns [ Patch, Patch, ... ] of length numberOfFitParams().
	def getFitParamDerivatives(self, img):
		pos0 = self.getPosition()
		psteps = pos0.getFitStepSizes(img)

		(px,py) = img.getWcs().positionToPixel(pos0)
		patch0 = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().fluxToCounts(self.flux)

		derivs = []
		for i in range(len(psteps)):
			posx = pos0.copy()
			posx.stepParam(i, psteps[i])
			(px,py) = img.getWcs().positionToPixel(posx)
			patchx = img.getPsf().getPointSourcePatch(px, py)
			dx = (patchx - patch0) * (counts / psteps[i])
			dx.setName('d(src)/d(pos%i)' % i)
			derivs.append(dx)

		fsteps = self.flux.getFitStepSizes(img)
		for i in range(len(fsteps)):
			fi = self.flux.copy()
			fi.stepParam(i, fsteps[i])
			countsi = img.getPhotoCal().fluxToCounts(fi)
			df = patch0 * ((countsi - counts) / fsteps[i])
			df.setName('d(src)/d(flux%i)' % i)
			derivs.append(df)

		return derivs

	# update parameters in this direction
	def stepParams(self, dparams):
		pos = self.getPosition()
		np = pos.getDimension()
		nf = self.flux.numberOfFitParams()
		assert(len(dparams) == (np + nf))
		dp = dparams[:np]
		pos.stepParams(dp)
		df = dparams[np:]
		self.flux.stepParams(df)

	def getParams(self):
		pp = self.getPosition().getParams()
		pf = self.flux.getParams()
		np = self.getPosition().getDimension()
		assert(len(pp) == np)
		p = pp + pf
		assert(len(p) == (len(pp) + len(pf)))
		return p

	def setParams(self, p):
		pos = self.getPosition()
		np = pos.getDimension()
		nf = self.flux.numberOfFitParams()
		assert(len(p) == (np + nf))
		pp = p[:np]
		pos.setParams(pp)
		pf = p[np:]
		self.flux.setParams(pf)


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

	def setPsf(self, psf):
		self.psf = psf

	def __getattr__(self, name):
		if name == 'shape':
			return self.data.shape

	def __hash__(self):
		return hash(self.hashkey())

	def hashkey(self):
		#return (id(self.data), id(self.invvar), hash(self.psf),
		#		hash(self.sky), hash(self.wcs), hash(self.photocal))
		return ('Image', id(self.data), id(self.invvar), self.psf.hashkey(),
				hash(self.sky), hash(self.wcs), hash(self.photocal))

	# Any time an attribute is changed, update the "version" number to a random value.
	# FIXME -- should probably hash all my members instead!
	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
		self.setversion(randomint())
	def setversion(self, ver):
		object.__setattr__(self, 'version', ver)
	def getVersion(self):
		return self.version

	def numberOfPixels(self):
		(H,W) = self.data.shape
		return W*H
		
	def getInvError(self):
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

	#def getDimension(self):
	#	return 0
	def numberOfFitParams(self):
		return 0
	def getFitStepSizes(self, img):
		return []

class NullPhotoCal(object):
	def fluxToCounts(self, flux):
		return flux.getValue()
	def countsToFlux(self, counts):
		return counts.getValue()

	def numberOfFitParams(self):
		return 0
	#def getDimension(self):
	#	return 0
	def getFitStepSizes(self, img):
		return []

class Flux(object):
	def __init__(self, val):
		self.val = val

	def hashkey(self):
		return ('Flux', self.val)

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
	def getParams(self):
		return [self.val]
	def setParams(self, p):
		assert(len(p) == 1)
		self.val = p[0]


class WCS(object):
	def positionToPixel(self, pos):
		pass

# useful when you're using raw pixel positions rather than RA,Decs
class NullWCS(WCS):
	def positionToPixel(self, pos):
		return pos

class PixPos(list):
	def __str__(self):
		return 'pixel (%.2f, %.2f)' % (self[0], self[1])
	def __repr__(self):
		return 'PixPos(%.4f, %.4f)' % (self[0], self[1])

	def copy(self):
		return PixPos([self[0],self[1]])

	def hashkey(self):
		return ('PixPos', self[0], self[1])

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

	def getParams(self):
		return [self[0], self[1]]
	def setParams(self, p):
		assert(len(p) == len(self))
		for i,pval in enumerate(p):
			self[i] = pval

	def __hash__(self):
		# convert to tuple
		return hash((self[0], self[1]))

	#def __iadd__(self, X):
	#  	self[0] += X[0]
	#	self[1] += X[1]

class Patch(object):
	def __init__(self, x0, y0, patch):
		self.x0 = x0
		self.y0 = y0
		self.patch = patch

	def setName(self, name):
		self.name = name
	def getName(self):
		return self.name

	def copy(self):
		if self.patch is None:
			return Patch(self.x0, self.y0, None)
		return Patch(self.x0, self.y0, self.patch.copy())

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

	def clipTo(self, W, H):
		if self.patch is None:
			return False
		if self.x0 < 0:
			self.patch = self.patch[:, -self.x0:]
			self.x0 = 0
		if self.y0 < 0:
			self.patch = self.patch[-self.y0:, :]
			self.y0 = 0
		if self.x0 >= W:
			# empty
			self.patch = None
			return False
		if self.y0 >= H:
			self.patch = None
			return False
		(h,w) = self.shape
		if (self.x0 + w) > W:
			self.patch = self.patch[:, :(W - self.x0)]
		if (self.y0 + h) > H:
			self.patch = self.patch[:(H - self.y0), :]

		assert(self.x0 >= 0)
		assert(self.y0 >= 0)
		(h,w) = self.shape
		assert(w <= W)
		assert(h <= H)
		assert(self.shape == self.patch.shape)
		return True

	'''
	def clipTo(self, x0, y0, W, H):
		(h,w) = self.shape
		newx0 = self.x0
		if x0 > self.x0:
			self.patch = self.patch[:, x0 - self.x0:]
			newx0 = x0
		newy0 = self.y0
		if y0 > self.y0:
			self.patch = self.patch[y0 - self.y0:, :]
			newy0 = y0
		newx1 = self.x0 + w
		if W < newx1:
			self.patch = self.patch
		newy1 = self.
	'''		

	def getSlice(self, parent=None):
		if self.patch is None:
			return ([],[])
		(ph,pw) = self.patch.shape
		if parent is not None:
			(H,W) = parent.shape
			return (slice(np.clip(self.y0, 0, H), np.clip(self.y0+ph, 0, H)),
					slice(np.clip(self.x0, 0, W), np.clip(self.x0+pw, 0, W)))
		return (slice(self.y0, self.y0+ph),
				slice(self.x0, self.x0+pw))

	def getPixelIndices(self, parent):
		if self.patch is None:
			return np.array([])
		(h,w) = self.shape
		(H,W) = parent.shape
		X,Y = np.meshgrid(np.arange(w), np.arange(h))
		return (Y.ravel() + self.y0) * W + (X.ravel() + self.x0)

	def addTo(self, img, scale=1.):
		if self.patch is None:
			return
		(ih,iw) = img.shape
		(ph,pw) = self.patch.shape

		(outx, inx) = get_overlapping_region(self.x0, self.x0+pw-1, 0, iw-1)
		(outy, iny) = get_overlapping_region(self.y0, self.y0+ph-1, 0, ih-1)
		if inx == [] or iny == []:
			return
		x0 = outx.start
		y0 = outy.start
		p = self.patch[iny,inx]
		img[outy, outx] += self.getImage()[iny, inx] * scale

	def __getattr__(self, name):
		if name == 'shape':
			if self.patch is None:
				return (0,0)
			return self.patch.shape

	def __mul__(self, flux):
		if self.patch is None:
			return Patch(self.x0, self.y0, None)
		return Patch(self.x0, self.y0, self.patch * flux)

	def __sub__(self, other):
		assert(isinstance(other, Patch))
		if (self.x0 == other.getX0() and self.y0 == other.getY0() and
			self.shape == other.shape):
			assert(self.x0 == other.getX0())
			assert(self.y0 == other.getY0())
			assert(self.shape == other.shape)
			if self.patch is None or other.patch is None:
				return Patch(self.x0, self.y0, None)
			return Patch(self.x0, self.y0, self.patch - other.patch)

		(ph,pw) = self.patch.shape
		(ox0,oy0) = other.getX0(), other.getY0()
		(oh,ow) = other.shape

		# Find the union of the regions.
		ux0 = min(ox0, self.x0)
		uy0 = min(oy0, self.y0)
		ux1 = max(ox0 + ow, self.x0 + pw)
		uy1 = max(oy0 + oh, self.y0 + ph)

		p = np.zeros((uy1 - uy0, ux1 - ux0))
		p[self.y0 - uy0 : self.y0 - uy0 + ph,
		  self.x0 - ux0 : self.x0 - ux0 + pw] = self.patch
		p[oy0 - uy0 : oy0 - uy0 + oh,
		  ox0 - ux0 : ox0 - ux0 + ow] -= other.getImage()
		return Patch(ux0, uy0, p)


class PSF(object):
	def applyTo(self, image):
		pass

	# return Patch, a rendering of a point source at the given pixel
	# coordinate.
	def getPointSourcePatch(self, px, py):
		pass

	def copy(self):
		return PSF()

	def __hash__(self):
		return hash(self.hashkey())

	def hashkey(self):
		return ('PSF',)

	# Returns a new PSF object that is a more complex version of self.
	def proposeIncreasedComplexity(self, img):
		return PSF()
	
	def numberOfFitParams(self):
		return 0
	def getFitStepSizes(self, img):
		return []
	def stepParam(self, parami, delta):
		pass
	def stepParams(self, dparams):
		assert(len(dparams) == self.numberOfFitParams())
		for i,dp in enumerate(dparams):
			self.stepParam(i, dp)
	def isValidParamStep(self, dparam):
		return True
		
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
		assert(len(sigmas) == len(weights))
		self.sigmas = sigmas
		self.weights = weights

	def __str__(self):
		return ('NGaussianPSF: sigmas [ ' +
				', '.join(['%.3f'%s for s in self.sigmas]) +
				' ], weights [ ' +
				', '.join(['%.3f'%w for w in self.weights]) +
				' ]')

	def __repr__(self):
		return ('NGaussianPSF: sigmas [ ' +
				', '.join(['%.3f'%s for s in self.sigmas]) +
				' ], weights [ ' +
				', '.join(['%.3f'%w for w in self.weights]) +
				' ]')

	def proposeIncreasedComplexity(self, img):
		maxs = np.max(self.sigmas)
		# MAGIC
		news = self.sigmas + [maxs + 1.]
		return NGaussianPSF(news, self.weights + [0.05])

	def numberOfFitParams(self):
		return 2 * len(self.sigmas)
	def getFitStepSizes(self, img):
		#return [0.1]*len(self.sigmas) + [0.1]*len(self.weights)
		return [0.01]*len(self.sigmas) + [0.01]*len(self.weights)
	def stepParam(self, parami, delta):
		assert(parami >= 0)
		assert(parami < 2*len(self.sigmas))
		NS = len(self.sigmas)
		if parami < NS:
			self.sigmas[parami] += delta
			#print 'NGaussianPSF: setting sigma', parami, 'to', self.sigmas[parami]
		else:
			self.weights[parami - NS] += delta
			#print 'NGaussianPSF: setting weight', (parami-NS), 'to', self.weights[parami-NS]

	def isValidParamStep(self, dparam):
		NS = len(self.sigmas)
		assert(len(dparam) == 2*NS)
		dsig = dparam[:NS]
		dw = dparam[NS:]
		for s,ds in zip(self.sigmas, dsig):
			# MAGIC
			if s + ds < 0.1:
				return False
		for w,dw in zip(self.weights, dw):
			if w + dw < 0:
				return False
		return True
		#return all(self.sigmas + dsig > 0.1) and all(self.weights + dw > 0)

	def normalize(self):
		mx = max(self.weights)
		self.weights = [w/mx for w in self.weights]

	def hashkey(self):
		return ('NGaussianPSF', tuple(self.sigmas), tuple(self.weights))
	
	def copy(self):
		return NGaussianPSF(list([s for s in self.sigmas]),
							list([w for w in self.weights]))

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
		X,Y = np.meshgrid(np.arange(sz).astype(float), np.arange(sz).astype(float))
		X -= dx + rad
		Y -= dy + rad
		patch = np.zeros((sz,sz))
		x0 = ix - rad
		y0 = iy - rad
		R2 = (X**2 + Y**2)
		for s,w in zip(self.sigmas, self.weights):
			patch += w / (2.*pi*s**2) * np.exp(-0.5 * R2 / (s**2))
		patch /= sum(self.weights)
		#print 'sum of PSF patch:', patch.sum()
		return Patch(x0, y0, patch)

class Cache(dict):
	pass

class Catalog(list):
	def __hash__(self):
		return hash(self.hashkey())
	def hashkey(self):
		#return tuple([hash(x) for x in self])
		return tuple([x.hashkey() for x in self])

	def deepcopy(self):
		return Catalog([x.copy() for x in self])

	def printLong(self):
		print 'Catalog:'
		for i,x in enumerate(self):
			print '  %i:' % i, x

	def getAllParams(self):
		return [src.getParams() for src in self]
	def setAllParams(self, p):
		for src,pp in zip(self, p):
			src.setParams(pp)


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

	def getImage(self, imgi):
		return self.images[imgi]

	def getCatalog(self):
		return self.catalog

	'''
	LINEAR = 'linear'
	LOG = 'log'

	@staticmethod
	def getDerivative(paramtype, stepsize, img0, img1):
		if paramtype == Tractor.LINEAR:
			return (img1 - img0) / stepsize
		elif paramtype == Tractor.LOG:
			return ()
	'''

	def increasePsfComplexity(self, imagei):
		print 'Increasing complexity of PSF in image', imagei
		img = self.getImage(imagei)
		psf = img.getPsf()
		#npixels = img.numberOfPixels()
		# For the PSF model, we render out the whole image.
		#mod0 = self.getModelImage(img)
		pBefore = self.getLogProb()

		psfk = psf.proposeIncreasedComplexity(img)

		print 'Trying to increase PSF complexity'
		print 'from:', psf
		print 'to  :', psfk

		img.setPsf(psfk)
		#modk = self.getModelImage(img)
		pAfter = self.getLogProb()

		print 'Before increasing PSF complexity: log-prob', pBefore
		print 'After  increasing PSF complexity: log-prob', pAfter

		self.optimizePsfAtFixedComplexityStep(imagei)

		pAfter2 = self.getLogProb()

		print 'Before increasing PSF complexity: log-prob', pBefore
		print 'After  increasing PSF complexity: log-prob', pAfter
		print 'After  tuning:                    log-prob', pAfter2

		if pAfter2 > pBefore:
			print 'Accepting PSF change!'
		else:
			print 'Rejecting PSF change!'
			img.setPsf(psf)

		print 'PSF is', img.getPsf()

	def increaseAllPsfComplexity(self):
		for i in range(len(self.images)):
			self.increasePsfComplexity(i)

	def changeSource(self, source):
		'''
		Proposes a list of alternatives, where each is a lists of new
		Sources that the given Source could be changed into.
		'''
		return []

	def changeSourceTypes(self, srcs=None, altCallback=None):
		print
		print 'changeSourceTypes'
		pBefore = self.getLogProb()
		print 'log-prob before:', pBefore
		print

		#if srcs is None:
		#	srcs = self.catalog

		oldcat = self.catalog

		# We can't just loop over "srcs" -- because when we accept a
		# change, the catalog changes!
		#donesrcs = []
		#addedsrcs = []

		#i = -1
		#for src in srcs:
		#	i += 1
			# for i,src in enumerate(srcs):
			#i = 0
			#while True:
			#for ii in len(srcs):
			#if i >= len(srcs):
			#	break
			#src = srcs[i]
			#if src in 

		i = -1
		ii = -1
		while True:
			i += 1
			print
			print 'changeSourceTypes: source', i
			print
			self.catalog = oldcat
			self.catalog.printLong()

			if srcs is None:
				# go through self.catalog using "ii" as the index.
				# (which is updated within the loop when self.catalog is mutated)
				ii += 1
				if ii >= len(self.catalog):
					break
				print '  changing source index', ii
				src = self.catalog[ii]
			else:
				if i >= len(srcs):
					break
				src = srcs[i]

			pBefore = self.getLogProb()
			print 'log-prob before:', pBefore
			#print 'catalog hash-code', hash(self.catalog)

			bestlogprob = pBefore
			bestalt = -1
			bestparams = None

			alts = self.changeSource(src)
			#print 'changing source', src, 'into alternatives:', alts
			for j,newsrcs in enumerate(alts):
				#print 'trying alternative', j, ':', newsrcs
				newcat = oldcat.deepcopy()
				newcat.remove(src)
				newcat.extend(newsrcs)
				print 'Replacing:'
				print '  from', src
				print '  to  ', newsrcs
				self.catalog = newcat
				print 'Before optimizing:', self.getLogProb()
				# first try individually optimizing the newly-added
				# source...
				for ostep in range(20):
					print 'Optimizing the new sources (step %i)...' % (ostep+1)
					dlnprob = self.optimizeCatalogAtFixedComplexityStep(srcs=newsrcs)
					print 'delta-log-prob', dlnprob
					if dlnprob < 1.:
						print 'failed to improve the new source enough (d lnprob = %g)' % dlnprob
						break
					print 'Changed to', newsrcs
				print 'After optimizing new sources:', self.getLogProb()
				self.optimizeCatalogAtFixedComplexityStep()
				print 'After optimizing all with new sources:', self.getLogProb()

				if altCallback is not None:
					altCallback(self, src, newsrcs, i, j)

				pAfter = self.getLogProb()
				print 'log-prob after:', pAfter
				print 'delta-log-prob:', pAfter - pBefore

				if pAfter > bestlogprob:
					print 'Best change so far!'
					bestlogprob = pAfter
					bestalt = j
					bestparams = newcat.getAllParams()

			if bestparams is not None:
				print 'Switching to new catalog!'
				# We want to update "oldcat" in-place (rather than
				# setting "self.catalog = bestcat") so that the source
				# object identities don't change -- so that the outer
				# loop "for src in self.catalog" still works.  We need
				# to updated the structure and params.
				oldcat.remove(src)
				ii -= 1
				oldcat.extend(alts[bestalt])
				oldcat.setAllParams(bestparams)
				self.catalog = oldcat
				pBefore = bestlogprob
				print 'New catalog:'
				self.catalog.printLong()
				assert(self.getLogProb() == pBefore)

		self.catalog = oldcat


	def optimizePsfAtFixedComplexityStep(self, imagei,
										 derivCallback=None):
		print 'Optimizing PSF in image', imagei, 'at fixed complexity'
		img = self.getImage(imagei)
		psf = img.getPsf()
		nparams = psf.numberOfFitParams()
		npixels = img.numberOfPixels()

		if nparams == 0:
			raise RuntimeError('No PSF parameters to optimize')

		# For the PSF model, we render out the whole image.
		mod0 = self.getModelImage(img)

		steps = psf.getFitStepSizes(img)
		assert(len(steps) == nparams)
		derivs = []
		print 'Computing PSF derivatives around PSF:', psf
		for k,s in enumerate(steps):
			if False:
				psfk = psf.copy()
				psfk.stepParam(k, s)
				print '  step param', k, 'by', s, 'to get', psfk
				img.setPsf(psfk)
				modk = self.getModelImage(img)
				# to reuse code, wrap this in a Patch...
				dk = Patch(0, 0, (modk - mod0) / s)
			else:
				# symmetric finite differences
				psfk1 = psf.copy()
				psfk2 = psf.copy()
				psfk1.stepParam(k, -s)
				psfk2.stepParam(k, +s)
				print '  step param', k, 'by', -s, 'to get', psfk1
				print '  step param', k, 'by',  s, 'to get', psfk2
				img.setPsf(psfk1)
				modk1 = self.getModelImage(img)
				img.setPsf(psfk2)
				modk2 = self.getModelImage(img)
				dk = Patch(0, 0, (modk2 - modk1) / (s*2))

			derivs.append(dk)
		img.setPsf(psf)

		assert(len(derivs) == nparams)

		if derivCallback:
			(func, baton) = derivCallback
			func(self, imagei, img, psf, steps, mod0, derivs, baton)

		inverrs = img.getInvError()

		# Build the sparse matrix of derivatives:
		sprows = []
		spcols = []
		spvals = []

		for p,deriv in enumerate(derivs):
			(H,W) = img.shape
			deriv.clipTo(W, H)
			pix = deriv.getPixelIndices(img)
			assert(all(pix < npixels))
			# (grab non-zero indices)
			dimg = deriv.getImage()
			nz = np.flatnonzero(dimg)
			#print '  psf derivative', p, 'has', len(nz), 'non-zero entries'
			if len(nz) == 0:
				continue
			rows = pix[nz]
			cols = np.zeros_like(rows) + p
			vals = dimg.ravel()[nz]
			w = inverrs[deriv.getSlice()].ravel()[nz]
			assert(vals.shape == w.shape)
			sprows.append(rows)
			spcols.append(cols)
			spvals.append(vals * w)

		########################### FIXME
		######### see the other lsqr() call to make sure the params are right
		# here.

		# ensure the sparse matrix is full up to the number of columns we expect
		spcols.append([nparams-1])
		sprows.append([0])
		spvals.append([0])

		sprows = np.hstack(sprows)
		spcols = np.hstack(spcols)
		spvals = np.hstack(spvals)

		print 'Number of sparse matrix elements:', len(sprows)
		urows = np.unique(sprows)
		print 'Unique rows (pixels):', len(urows)
		print 'Max row:', max(sprows)
		ucols = np.unique(spcols)
		print 'Unique columns (params):', len(ucols)

		# Build sparse matrix
		A = csr_matrix((spvals, (sprows, spcols)))

		# b = -weighted residuals
		NP = img.numberOfPixels()
		data = img.getImage()
		inverr = img.getInvError()
		mod = mod0
		assert(np.product(mod.shape) == NP)
		assert(mod.shape == data.shape)
		assert(mod.shape == inverr.shape)
		b = ((data - mod) * inverr).ravel()
		b = b[:urows.max() + 1]
		
		lsqropts = dict(show=True)

		# Run lsqr()
		(X, istop, niters, r1norm, r2norm, anorm, acond,
		 arnorm, xnorm, var) = lsqr(A, b, **lsqropts)
		# (PSF)
		print 'lsqr stopped for reason', istop, 'after', niters, 'iterations'

		# Unpack.

		print 'Parameter changes:', X

		pBefore = self.getLogProb()
		print 'log-prob before:', pBefore

		for alpha in 2.**-np.arange(10):
			print 'Stepping with alpha =', alpha
			dparams = X
			assert(len(dparams) == nparams)
			if not psf.isValidParamStep(dparams * alpha):
				print 'Changing PSF params by', (dparams*alpha), 'is not valid!'
				continue

			self.pushCache()

			newpsf = psf.copy()
			newpsf.stepParams(dparams * alpha)
			print 'Stepped to', newpsf
			img.setPsf(newpsf)

			pAfter = self.getLogProb()
			print 'log-prob before:', pBefore
			print 'log-prob after :', pAfter

			if pAfter > pBefore:
				print 'Accepting step!'
				newpsf.normalize()
				self.mergeCache()
				break

			print 'Rejecting step!'
			self.popCache()
			# revert
			img.setPsf(psf)

		# if we get here, this step was rejected.

	def optimizeAllPsfAtFixedComplexityStep(self, **kwargs):
		for i in range(len(self.images)):
			self.optimizePsfAtFixedComplexityStep(i, **kwargs)

	def optimizeCatalogAtFixedComplexityStep(self, srcs=None):
		'''
		-synthesize images

		-get all derivatives
		(taking numerical derivatives itself?)

		-build matrix

		-take step (try full step, back off)
		'''
		print 'Optimizing at fixed complexity'

 		if srcs is None:
			srcs = self.catalog

		# need all derivatives  dChi / dparam
		# for each pixel in each image
		#  and each parameter in each source
		nparams = [src.numberOfFitParams() for src in srcs]
		col0 = np.cumsum([0] + nparams)

		npixels = [img.numberOfPixels() for img in self.images]
		row0 = np.cumsum([0] + npixels)
		# [ 0, (W0*H0), (W0*H0 + W1*H1), ... ]

		sprows = []
		spcols = []
		spvals = []

		for j,src in enumerate(srcs):
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

				#print 'Got derivatives:', derivs
				inverrs = img.getInvError()

				# Add to the sparse matrix of derivatives:
				for p,deriv in enumerate(derivs):
					(H,W) = img.shape
					#print 'Before clipping:'
					#print 'deriv shape is', deriv.shape
					#print 'deriv slice is', deriv.getSlice()
					deriv.clipTo(W, H)
					pix = deriv.getPixelIndices(img)
					#print 'After clipping:'
					#print 'deriv shape is', deriv.shape
					#print 'deriv slice is', deriv.getSlice()
					#print 'image shape is', img.shape
					#print 'parent pix', (W*H), npixels[i]
					#print 'pix range:', pix.min(), pix.max()
					# (in the parent image)
					if len(pix) == 0:
						print 'This source does not influence this image!'
						continue
					
					if pix.max() >= npixels[i]:
						print 'maximum pixel index:', pix.max()
						print 'number of pixels in this image:', npixels[i]
						print 'W,H', (W,H)
						print 'patch origin', deriv.getOrigin()
						print 'patch shape', deriv.shape
					assert(all(pix < npixels[i]))
					# (grab non-zero indices)
					dimg = deriv.getImage()
					nz = np.flatnonzero(dimg)
					#print '  source', j, 'derivative', p, 'has', len(nz), 'non-zero entries'
					rows = row0[i] + pix[nz]
					cols = np.zeros_like(rows) + col0[j] + p
					#rows = np.zeros(len(cols), int) + row0[j] + p
					vals = dimg.ravel()[nz]
					#print 'inverrs shape is', inverrs.shape
					w = inverrs[deriv.getSlice(img)].ravel()[nz]
					assert(vals.shape == w.shape)

					# We want to minimize:
					#   || chi + (d(chi)/d(params)) * dparams ||
					# So  b = chi
					#     A = -d(chi)/d(params)
					#     x = dparams
					#
					# chi = (data - model) / std = (data - model) * inverr
					# derivs = d(model)/d(param)
					# A matrix = -d(chi)/d(param)
					#          = + (derivs) * inverr

					sprows.append(rows)
					spcols.append(cols)
					spvals.append(vals * w)

		# ensure the sparse matrix is full up to the number of columns we expect
		spcols.append([np.sum(nparams) - 1])
		sprows.append([0])
		spvals.append([0])

		sprows = np.hstack(sprows)
		spcols = np.hstack(spcols)
		spvals = np.hstack(spvals)

		print '  Number of sparse matrix elements:', len(sprows)
		urows = np.unique(sprows)
		print '  Unique rows (pixels):', len(urows)
		print '  Max row:', max(sprows)
		ucols = np.unique(spcols)
		print '  Unique columns (params):', len(ucols)

		# Build sparse matrix
		A = csr_matrix((spvals, (sprows, spcols)))

		# b = chi
		b = np.zeros(np.sum(npixels))
		for i,img in enumerate(self.images):
			NP = npixels[i]
			mod = self.getModelImage(img)
			data = img.getImage()
			inverr = img.getInvError()
			assert(np.product(mod.shape) == NP)
			assert(mod.shape == data.shape)
			assert(mod.shape == inverr.shape)
			b[row0[i] : row0[i] + NP] = ((data - mod) * inverr).ravel()
		#print 'b shape', b.shape
		b = b[:urows.max() + 1]
		#print 'b shape', b.shape
		
		lsqropts = dict(show=False)

		# Run lsqr()
		(X, istop, niters, r1norm, r2norm, anorm, acond,
		 arnorm, xnorm, var) = lsqr(A, b, **lsqropts)

		# Unpack.
		print '  X=', X

		pBefore = self.getLogProb()
		print 'log-prob before:', pBefore

		for alpha in 2.**-np.arange(10):
			print '  Stepping with alpha =', alpha

			#oldcat = self.catalog.deepcopy()
			#self.pushCache()

			oldparams = []

			for j,src in enumerate(srcs):
				dparams = X[col0[j] : col0[j] + nparams[j]]
				assert(len(dparams) == nparams[j])
				#print 'Applying parameter update', dparams, 'to source', src
				oldparams.append(src.getParams())
				src.stepParams(dparams * alpha)

			pAfter = self.getLogProb()
			#print 'log-prob before:', pBefore
			#print 'log-prob after :', pAfter
			print '  delta log-prob:', pAfter - pBefore

			if pAfter > pBefore:
				print 'Accepting step!'
				#self.mergeCache()
				return pAfter - pBefore

			print '  Rejecting step!'
			#self.popCache()
			# revert the catalog
			#self.catalog = oldcat

			assert(len(srcs) == len(oldparams))
			for j,src in enumerate(srcs):
				src.setParams(oldparams[j])

		# if we get here, this step was rejected.
		return -1
	
	def getModelPatchNoCache(self, img, src):
		return src.getModelPatch(img)

	def getModelPatch(self, img, src):
		deps = (img.hashkey(), src.hashkey())
		deps = hash(deps)
		mod = self.cache.get(deps, None)
		if mod is not None:
			#print '  Cache hit!'
			pass
		else:
			mod = self.getModelPatchNoCache(img, src)
			#print 'Caching model image'
			self.cache[deps] = mod
		return mod

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
		#deps = (img.getVersion(), hash(self.catalog))

		#deps = (hash(img), hash(self.catalog))
		deps = (img.hashkey(), self.catalog.hashkey())
		#print 'deps:', deps
		deps = hash(deps)

		#print 'Model image:'
		#print '  catalog', self.catalog
		#print '  -> deps', deps
		mod = self.cache.get(deps, None)
		if mod is not None:
			#print '  Cache hit!'
			mod = mod.copy()
		else:
			mod = self.getModelImageNoCache(img)
			#print 'Caching model image'
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
			chis.append((img.getImage() - mod) * img.getInvError())
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
		print
		print 'Tractor.createSource'
		'''
		-synthesize images
		-look for "promising" Positions with "positive" residuals
		- (not near existing sources)
		---chi image, PSF smooth, propose positions?
		-instantiate new source (Position, flux, PSFType)
		-local optimizeAtFixedComplexity
		'''

		rtn = []
		
		for i,chi in enumerate(self.getChiImages()):
			img = self.images[i]

			# block out regions around existing Sources.
			for j,src in enumerate(self.catalog):
				patch = self.getModelPatch(self.images[i], src)
				(H,W) = img.shape
				if not patch.clipTo(W, H):
					continue
				chi[patch.getSlice()] = 0.

			# PSF-correlate
			sm = img.getPsf().applyTo(chi)
			# find peaks, create sources

			# HACK -- magic value 10
			#pks = self.findPeaks(sm, 10)

			# Try to create sources in the highest-value pixels.
			II = np.argsort(-sm.ravel())

			tryxy = []

			# MAGIC: number of pixels to try.
			for ii,I in enumerate(II[:10]):
				# find peak pixel, create source
				#I = np.argmax(sm)
				(H,W) = sm.shape
				ix = I%W
				iy = I/W
				# this is just the peak pixel height...
				ht = (img.getImage() - self.getModelImage(img))[iy,ix]
				print 'creating new source at x,y', (ix,iy)
				src = self.createNewSource(img, ix, iy, ht)

				tryxy.append((ix,iy))

				# try adding the new source...
				pBefore = self.getLogProb()
				print 'log-prob before:', pBefore

				self.pushCache()
				oldcat = self.catalog.deepcopy()

				self.catalog.append(src)
				#print 'added source, catalog is:'
				#print self.catalog

				# first try individually optimizing the newly-added
				# source...
				for ostep in range(20):
					print 'Optimizing the new source (step %i)...' % (ostep+1)
					dlnprob = self.optimizeCatalogAtFixedComplexityStep(srcs=[src])
					if dlnprob < 1.:
						print 'failed to improve the new source enough (d lnprob = %g)' % dlnprob
						break

				# Try changing the newly-added source type?
				#print 'Trying to change the source type of the newly-added source'
				#self.changeSourceTypes(srcs=[src])
					
				# then the whole catalog
				print 'Optimizing the catalog with the new source...'
				self.optimizeCatalogAtFixedComplexityStep()

				pAfter = self.getLogProb()
				print 'log-prob before:', pBefore
				print 'log-prob after :', pAfter
				print 'd log-prob:', (pAfter - pBefore)

				if pAfter > pBefore:
					print 'Keeping new source'
					self.mergeCache()
					break

				else:
					print 'Rejecting new source'
					self.popCache()
					# revert the catalog
					self.catalog = oldcat

			rtn.append((sm, tryxy))

			pEnd = self.getLogProb()
			print 'log-prob at finish:', pEnd

		return rtn

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

