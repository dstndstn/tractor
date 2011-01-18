from math import ceil, floor, pi, sqrt, exp
import numpy as np
import random
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from astrometry.util.miscutils import get_overlapping_region

import pylab as plt

class Params(object):
	def __hash__(self):
		return hash(self.hashkey())
	def __eq__(self, other):
		return hash(self) == hash(other)
	def hashkey(self):
		return ('ParamSet',)
	def numberOfParams(self):
		return 0
	def stepParam(self, parami, delta):
		pass
	def stepParams(self, dparams):
		assert(len(dparams) == self.numberOfParams())
		for i,dp in enumerate(dparams):
			self.stepParam(i, dp)
	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		return []
	def setParams(self, p):
		pass
	def getStepSizes(self, *args, **kwargs):
		return []

# An implementation of Params that holds values in a list.
class ParamList(Params):
	def __init__(self, *args):
		self.vals = list(args)
		self.namedparams = self.getNamedParams()
	def getNamedParams(self):
		return []
	def __getattr__(self, name):
		for n,i in self.namedparams:
			if name == n:
				return self.vals[i]
		raise AttributeError('ParamList (%s): unknown attribute "%s"' %
							 (str(type(self)), name))
	def hashkey(self):
		return ('ParamList',) + tuple(self.vals)
	def numberOfParams(self):
		return len(self.vals)
	def stepParam(self, parami, delta):
		self.vals[parami] += delta
	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		return list(self.vals)
	def setParams(self, p):
		assert(len(p) == len(self.vals))
		for i,pp in enumerate(p):
			self.vals[i] = pp
	def getStepSizes(self, *args, **kwargs):
		return [1 for x in self.vals]

	# len()
	def __len__(self):
		return self.numberOfParams()
	# []
	def __getitem__(self, i):
		#print 'ParamList.__getitem__', i, 'returning', self.vals[i]
		return self.vals[i]

	# iterable
	class ParamListIter(object):
		def __init__(self, pl):
			self.pl = pl
			self.i = 0
		def __iter__(self):
			return self
		def next(self):
			if self.i >= len(self.pl):
				raise StopIteration
			rtn = self.pl[self.i]
			#print 'paramlistiter: returning element', self.i, '=', rtn
			self.i += 1
			return rtn
	def __iter__(self):
		return ParamList.ParamListIter(self)

# An implementation of Params that combines component sub-Params.
class MultiParams(Params):
	def __init__(self, *args):
		self.subs = args
		self.namedparams = self.getNamedParams()
	def getNamedParams(self):
		return []
	def __getattr__(self, name):
		for n,i in self.namedparams:
			if name == n:
				return self.subs[i]
		raise AttributeError('MultiParam (%s): unknown attribute "%s"' %
							 (str(type(self)), name))

	def hashkey(self):
		t = ('MultiParams',)
		for s in self.subs:
			t = t + s.hashkey()
		return t

	def numberOfParams(self):
		return sum(s.numberOfParams() for s in self.subs)

	def stepParam(self, parami, delta):
		for s in self.subs:
			n = s.numberOfParams()
			if parami < n:
				s.stepParam(parami, delta)
				return
			parami -= n

	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		p = []
		for s in self.subs:
			p.extend(s.getParams())
		return p

	def setParams(self, p):
		i = 0
		for s in self.subs:
			n = s.numberOfParams()
			s.setParams(p[i:i+n])
			i += n

	def getStepSizes(self):
		p = []
		for s in self.subs:
			p.extend(s.getStepSizes())
		return p


# This is just the duck-type definition
class Source(Params):
	'''
	Must be hashable: see
	  http://docs.python.org/glossary.html#term-hashable
	'''
	def hashkey(self):
		return ('Source',)
	#def getPosition(self):
	#	pass
	def getModelPatch(self, img):
		pass
	# returns [ Patch, Patch, ... ] of length numberOfParams().
	def getParamDerivatives(self, img):
		return []


class PointSource(MultiParams):
	def __init__(self, pos, flux):
		MultiParams.__init__(self, pos, flux)
		#print 'PointSource constructor: nparams = ', self.numberOfParams()
	def getNamedParams(self):
		return [('pos', 0), ('flux', 1)]
	def getPosition(self):
		return self.pos
	def getFlux(self):
		return self.flux
	def __str__(self):
		return 'PointSource at ' + str(self.pos) + ' with ' + str(self.flux)
	def __repr__(self):
		return 'PointSource(' + repr(self.pos) + ', ' + repr(self.flux) + ')'
	def copy(self):
		return PointSource(self.pos.copy(), self.flux.copy())
	def hashkey(self):
		return ('PointSource', self.pos.hashkey(), self.flux.hashkey())

	def getModelPatch(self, img):
		(px,py) = img.getWcs().positionToPixel(self.getPosition())
		patch = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		return patch * counts

	# returns [ Patch, Patch, ... ] of length numberOfParams().
	def getParamDerivatives(self, img):
		pos0 = self.getPosition()
		(px,py) = img.getWcs().positionToPixel(pos0)
		patch0 = img.getPsf().getPointSourcePatch(px, py)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		derivs = []
		psteps = pos0.getStepSizes(img)
		for i in range(len(psteps)):
			posx = pos0.copy()
			posx.stepParam(i, psteps[i])
			(px,py) = img.getWcs().positionToPixel(posx)
			patchx = img.getPsf().getPointSourcePatch(px, py)
			dx = (patchx - patch0) * (counts / psteps[i])
			dx.setName('d(ptsrc)/d(pos%i)' % i)
			derivs.append(dx)
		fsteps = self.flux.getStepSizes(img)
		for i in range(len(fsteps)):
			fi = self.flux.copy()
			fi.stepParam(i, fsteps[i])
			countsi = img.getPhotoCal().fluxToCounts(fi)
			df = patch0 * ((countsi - counts) / fsteps[i])
			df.setName('d(ptsrc)/d(flux%i)' % i)
			derivs.append(df)
		return derivs

class Flux(ParamList):
	def hashkey(self):
		return ('Flux', self.val)
	def getNamedParams(self):
		return [('val', 0)]
	def __repr__(self):
		return 'Flux(%g)' % self.val
	def __str__(self):
		return 'Flux: %g' % self.val
	def copy(self):
		return Flux(self.val)
	def getValue(self):
		return self.val

	def getStepSizes(self, img):
		return [0.1]

class PixPos(ParamList):
	def getNamedParams(self):
		return [('x', 0), ('y', 1)]
	def __str__(self):
		return 'pixel (%.2f, %.2f)' % (self.x, self.y)
	def __repr__(self):
		return 'PixPos(%.4f, %.4f)' % (self.x, self.y)
	def copy(self):
		return PixPos(self.x, self.y)
	def hashkey(self):
		return ('PixPos', self.x, self.y)

	def getDimension(self):
		return 2
	def getStepSizes(self, img):
		return [0.1, 0.1]


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
		raise AttributeError('Image: unknown attribute "%s"' % name)

	def __hash__(self):
		return hash(self.hashkey())

	def hashkey(self):
		return ('Image', id(self.data), id(self.invvar), self.psf.hashkey(),
				hash(self.sky), hash(self.wcs), hash(self.photocal))

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

	#def numberOfParams(self):
	#	return 0
	#def getStepSizes(self, img):
	#	return []

class NullPhotoCal(object):
	def fluxToCounts(self, flux):
		return flux.getValue()
	def countsToFlux(self, counts):
		return counts.getValue()

	#def numberOfParams(self):
	#	return 0
	#def getStepSizes(self, img):
	#	return []

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

	plotnum = 0

	def addTo(self, img, scale=1.):
		if self.patch is None:
			return
		(ih,iw) = img.shape
		(ph,pw) = self.shape
		(outx, inx) = get_overlapping_region(self.x0, self.x0+pw-1, 0, iw-1)
		(outy, iny) = get_overlapping_region(self.y0, self.y0+ph-1, 0, ih-1)
		if inx == [] or iny == []:
			return
		p = self.patch[iny,inx]
		img[outy, outx] += p * scale

		if False:
			tmpimg = np.zeros_like(img)
			tmpimg[outy,outx] = p * scale
			plt.clf()
			plt.imshow(tmpimg, interpolation='nearest', origin='lower')
			plt.hot()
			plt.colorbar()
			fn = 'addto-%03i.png' % Patch.plotnum
			plt.savefig(fn)
			print 'Wrote', fn

			plt.clf()
			plt.imshow(p, interpolation='nearest', origin='lower')
			plt.hot()
			plt.colorbar()
			fn = 'addto-%03i-p.png' % Patch.plotnum
			plt.savefig(fn)
			print 'Wrote', fn

			Patch.plotnum += 1

	def __getattr__(self, name):
		if name == 'shape':
			if self.patch is None:
				return (0,0)
			return self.patch.shape
		raise AttributeError('Patch: unknown attribute "%s"' % name)

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


# This is just the duck-type definition
class PSF(Params):
	def applyTo(self, image):
		pass

	# Returns the number of pixels in the support of this PSF.
	def getRadius(self):
		return 0

	# return Patch, a rendering of a point source at the given pixel
	# coordinate.
	def getPointSourcePatch(self, px, py):
		pass

	def copy(self):
		return PSF()

	def hashkey(self):
		return ('PSF',)

	# Returns a new PSF object that is a more complex version of self.
	def proposeIncreasedComplexity(self, img):
		return PSF()
	
	def getStepSizes(self, img):
		return []
	def isValidParamStep(self, dparam):
		return True


class NGaussianPSF(MultiParams):
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
		MultiParams.__init__(self, ParamList(*sigmas), ParamList(*weights))

	def getNamedParams(self):
		return [('sigmas', 0), ('weights', 1)]

	def __str__(self):
		print 'sigmas:', self.sigmas
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
		# MAGIC -- make new Gaussian with variance bigger than the biggest
		# so far
		return NGaussianPSF(list(self.sigmas) + [maxs + 1.],
							list(self.weights) + [0.05])

	def getStepSizes(self, img):
		N = len(self.sigmas)
		return [0.01]*N + [0.01]*N

	'''
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
		'''

	def normalize(self):
		mx = max(self.weights)
		self.weights.setParams([w/mx for w in self.weights])

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

	def getNSigma(self):
		# HACK - MAGIC -- N sigma for rendering patches
		return 5.

	def getRadius(self):
		return max(self.sigmas) * self.getNSigma()

	# returns a Patch object.
	def getPointSourcePatch(self, px, py):
		ix = int(round(px))
		iy = int(round(py))
		dx = px - ix
		dy = py - iy

		rad = int(ceil(self.getRadius()))
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

	def increasePsfComplexity(self, imagei):
		print 'Increasing complexity of PSF in image', imagei
		pBefore = self.getLogProb()
		img = self.getImage(imagei)
		psf = img.getPsf()
		psfk = psf.proposeIncreasedComplexity(img)

		print 'Trying to increase PSF complexity'
		print 'from:', psf
		print 'to  :', psfk

		img.setPsf(psfk)
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

		oldcat = self.catalog

		# We can't just loop over "srcs" -- because when we accept a
		# change, the catalog changes!
		# FIXME -- with this structure, we try to change new sources that
		# we have just added.
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

			bestlogprob = pBefore
			bestalt = -1
			bestparams = None

			alts = self.changeSource(src)
			for j,newsrcs in enumerate(alts):
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
		nparams = psf.numberOfParams()
		npixels = img.numberOfPixels()
		if nparams == 0:
			raise RuntimeError('No PSF parameters to optimize')

		# For the PSF model, we render out the whole image.
		mod0 = self.getModelImage(img)

		steps = psf.getStepSizes(img)
		assert(len(steps) == nparams)
		derivs = []
		print 'Computing PSF derivatives around PSF:', psf
		for k,s in enumerate(steps):
			if True:
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

		# (PSF)
		X = self.optimize([img], [derivs])

		print 'PSF Parameter changes:', X
		dlogprob = self.tryParamUpdates([psf], X)
		print 'After:', psf
		print 'Log-prob improvement:', dlogprob

		return dlogprob

	#if not psf.isValidParamStep(dparams * alpha):
	#	print 'Changing PSF params by', (dparams*alpha), 'is not valid!'
	#	continue

	def optimizeAllPsfAtFixedComplexityStep(self, **kwargs):
		for i in range(len(self.images)):
			self.optimizePsfAtFixedComplexityStep(i, **kwargs)

	def optimize(self, allimgs, allderivs):
		# Build the sparse matrix of derivatives:
		sprows = []
		spcols = []
		spvals = []

		# Keep track of row offsets for each image.
		imgoffs = {}
		nextrow = 0

		nextcol = 0

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

		for img, derivs in zip(allimgs, allderivs):
			inverrs = img.getInvError()
			(H,W) = img.shape

			row0 = imgoffs.get(img, -1)
			if row0 == -1:
				row0 = nextrow
				imgoffs[img] = row0
				nextrow += img.numberOfPixels()

			# Add to the sparse matrix of derivatives:
			for p,deriv in enumerate(derivs):
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
					#print 'This source does not influence this image!'
					continue

				assert(all(pix < img.numberOfPixels()))
				# (grab non-zero indices)
				dimg = deriv.getImage()
				nz = np.flatnonzero(dimg)
				#print '  source', j, 'derivative', p, 'has', len(nz), 'non-zero entries'
				rows = row0 + pix[nz]
				cols = np.zeros_like(rows) + nextcol + p
				vals = dimg.ravel()[nz]
				w = inverrs[deriv.getSlice(img)].ravel()[nz]
				assert(vals.shape == w.shape)
				sprows.append(rows)
				spcols.append(cols)
				spvals.append(vals * w)

			nextcol += len(derivs)

		if len(spcols) == 0:
			return []

		# ensure the sparse matrix is full up to the number of columns we expect
		spcols.append([nextcol - 1])
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
		# FIXME -- we could be much smarter here about computing just the regions
		# we need!
		b = np.zeros(nextrow)
		for img,row0 in imgoffs.items():
			NP = img.numberOfPixels()
			mod = self.getModelImage(img)
			data = img.getImage()
			inverr = img.getInvError()
			assert(np.product(mod.shape) == NP)
			assert(mod.shape == data.shape)
			assert(mod.shape == inverr.shape)
			b[row0 : row0 + NP] = ((data - mod) * inverr).ravel()
		b = b[:urows.max() + 1]
		#print 'b shape', b.shape

		# FIXME -- does it make LSQR faster if we remap the row and column
		# indices so that no rows/cols are empty?
		
		lsqropts = dict(show=False)

		# Run lsqr()
		(X, istop, niters, r1norm, r2norm, anorm, acond,
		 arnorm, xnorm, var) = lsqr(A, b, **lsqropts)

		print '  X=', X
		return X

	# X: delta-params
	def tryParamUpdates(self, srcs, X, alphas=None, accept='best'):
		assert(accept in ['best', 'first'])
		if alphas is None:
			# 1/1024 to 1 in factors of 2
			alphas = 2.**-(np.arange(10,0,-1)-1)

		pBefore = self.getLogProb()
		print '  log-prob before:', pBefore

		pBest = pBefore
		bestAlpha = None

		for alpha in alphas:
			print '  Stepping with alpha =', alpha
			oldparams = []
			par0 = 0
			for j,src in enumerate(srcs):
				npar = src.numberOfParams()
				dparams = X[par0 : par0 + npar]
				par0 += npar
				assert(len(dparams) == src.numberOfParams())
				oldparams.append(src.getParams())
				src.stepParams(dparams * alpha)

			pAfter = self.getLogProb()
			print '  delta log-prob:', pAfter - pBefore

			if accept == 'best':
				if pAfter > pBest:
					pBest = pAfter
					bestAlpha = alpha
			elif accept == 'first':
				if pAfter > pBefore:
					print 'Accepting step!'
					return pAfter - pBefore
				print '  Rejecting step!'

			assert(len(srcs) == len(oldparams))
			for j,src in enumerate(srcs):
				src.setParams(oldparams[j])

		if accept == 'best':
			if bestAlpha is None:
				return -1
			print '  Stepping by', bestAlpha, 'for delta-logprob', pBest - pBefore
			par0 = 0
			for j,src in enumerate(srcs):
				npar = src.numberOfParams()
				dparams = X[par0 : par0 + npar]
				par0 += npar
				assert(len(dparams) == src.numberOfParams())
				src.stepParams(dparams * bestAlpha)
			return pBest - pBefore

		# if we get here, this step was rejected.
		return -1
		

	def optimizeCatalogAtFixedComplexityStep(self, srcs=None):
		'''
		-synthesize images
		-get all derivatives
		-build matrix
		-take step (try full step, back off)
		'''
		print 'Optimizing at fixed complexity'

 		if srcs is None:
			srcs = self.catalog

		allderivs = []
		allimgs = []
		for j,src in enumerate(srcs):
			for i,img in enumerate(self.images):
				# Get derivatives (in this image) of params
				derivs = src.getParamDerivatives(img)
				assert(len(derivs) == src.numberOfParams())
				allderivs.append(derivs)
				allimgs.append(img)

		X = self.optimize(allimgs, allderivs)

		dlogprob = self.tryParamUpdates(srcs, X)

		return dlogprob
	
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

	# the real deal
	def getModelImageNoCache(self, img):
		mod = np.zeros_like(img.getImage())
		mod += img.sky
		for src in self.catalog:
			patch = self.getModelPatch(img, src)
			patch.addTo(mod)
		return mod

	def getModelImage(self, img):
		# dependencies of this model image:
		# img.sky, img.psf, img.wcs, sources that overlap.
		#deps = (hash(img), hash(self.catalog))
		deps = (img.hashkey(), self.catalog.hashkey())
		#print 'deps:', deps
		deps = hash(deps)
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

	def createNewSource(self, img, x, y, height):
		return None

	def getLogLikelihood(self):
		chisq = 0.
		for i,chi in enumerate(self.getChiImages()):
			chisq += (chi ** 2).sum()
		return -0.5 * chisq

	def getLogPrior(self):
		return -sum([len(p) for p in self.catalog.getAllParams()])

	# posterior
	def getLogProb(self):
		return self.getLogLikelihood() + self.getLogPrior()

	def pushCache(self):
		self.cachestack.append(self.cache)
		self.cache = self.cache.copy()

	def mergeCache(self):
		# drop the top of the stack.
		self.cachestack.pop()

	def popCache(self):
		self.cache = self.cachestack.pop()

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
				(H,W) = sm.shape
				ix = I%W
				iy = I/W
				# this is just the peak pixel height...
				ht = (img.getImage() - self.getModelImage(img))[iy,ix]
				print 'creating new source at x,y', (ix,iy)
				src = self.createNewSource(img, ix, iy, ht)
				print 'Got:', src
				
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
					print 'After:', src
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

