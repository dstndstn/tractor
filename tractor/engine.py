'''
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`engine.py`
===========

Core image modeling and fitting.
'''

from math import ceil, floor, pi, sqrt, exp
import time
import logging
import random
import os
import resource
import gc

import numpy as np
#import pylab as plt

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label

from astrometry.util.miscutils import get_overlapping_region
from astrometry.util.multiproc import *
#from .utils import MultiParams, _isint, listmax
#from .cache import *
#from .ttime import Time
from utils import MultiParams, _isint, listmax
from cache import *
from ttime import Time

def logverb(*args):
	msg = ' '.join([str(x) for x in args])
	logging.debug(msg)
def logmsg(*args):
	msg = ' '.join([str(x) for x in args])
	logging.info(msg)
def isverbose():
	# Ugly logging interface...
	return (logging.getLogger().level <= logging.DEBUG)

def set_fp_err():
	'''Cause all floating-point errors to raise exceptions.
	Returns the current error state so you can revert via:

	    olderr = set_fp_err()
	    # do stuff
	    np.seterr(**olderr)
	'''
	return np.seterr(all='raise')

class Image(MultiParams):
	'''
	An image plus its calibration information.  An ``Image`` has
	pixels, inverse-variance map, WCS, PSF, photometric calibration
	information, and sky level.  All these things are ``Params``
	instances, and ``Image`` is a ``MultiParams`` so that the Tractor
	can optimize them.
	'''
	def __init__(self, data=None, invvar=None, psf=None, wcs=None, sky=None,
				 photocal=None, name=None, domask=True, time=None, **kwargs):
		'''
		Args:
		  * *data*: numpy array: the image pixels
		  * *invvar*: numpy array: the image inverse-variance
		  * *psf*: a :class:`tractor.PSF` duck
		  * *wcs*: a :class:`tractor.WCS` duck
		  * *sky*: a :class:`tractor.Sky` duck
		  * *photocal*: a :class:`tractor.PhotoCal` duck
		  * *name*: string name of this image.
		  * *zr*: plotting range ("vmin"/"vmax" in matplotlib.imshow)

		'''
		self.data = data
		self.origInvvar = 1. * np.array(invvar)
		kwa = dict()
		if 'dilation' in kwargs:
			kwa.update(dilation=kwargs.pop('dilation'))
		self.domask = domask
		if domask:
			self.setMask(**kwa)
		self.setInvvar(self.origInvvar)
		self.name = name
		self.starMask = np.ones_like(self.data)
		self.zr = kwargs.pop('zr', None)
		self.time = time
		super(Image, self).__init__(psf, wcs, photocal, sky)

	def __str__(self):
		return 'Image ' + str(self.name)

	@staticmethod
	def getNamedParams():
		return dict(psf=0, wcs=1, photocal=2, sky=3)

	def getTime(self):
		return self.time
	
	def getParamDerivatives(self, tractor, srcs):
		'''
		Returns a list of Patch objects, one per numberOfParams().
		Note that this means you have to pay attention to the
		frozen/thawed state.

		Can return None for no derivative, or False if you want the
		Tractor to compute the derivatives for you.
		'''
		derivs = []
		for s in self._getActiveSubs():
			if hasattr(s, 'getParamDerivatives'):
				print 'Calling getParamDerivatives on', s
				sd = s.getParamDerivatives(tractor, self, srcs)
				assert(len(sd) == s.numberOfParams())
				derivs.extend(sd)
			else:
				derivs.extend([False] * s.numberOfParams())
		print 'Image.getParamDerivatives: returning', derivs
		return derivs

	def getSky(self):
		return self.sky

	def setSky(self, sky):
		self.sky = sky

	def setPsf(self, psf):
		self.psf = psf

	def __getattr__(self, name):
		if name == 'shape':
			return self.getShape()
		raise AttributeError('Image: unknown attribute "%s"' % name)

	# Numpy arrays have shape H,W
	def getWidth(self):
		return self.getShape()[1]
	def getHeight(self):
		return self.getShape()[0]
	def getShape(self):
		if 'shape' in self.__dict__:
			return self.shape
		return self.data.shape

	def hashkey(self):
		return ('Image', id(self.data), id(self.invvar), self.psf.hashkey(),
				self.sky.hashkey(), self.wcs.hashkey(),
				self.photocal.hashkey())

	def numberOfPixels(self):
		(H,W) = self.data.shape
		return W*H

	def getInvError(self):
		return self.inverr
	def getInvvar(self):
		return self.invvar
	def setInvvar(self,invvar):
		self.invvar = 1. * invvar
		if self.domask:
			self.invvar[self.mask] = 0. 
		self.inverr = np.sqrt(self.invvar)

	def getMedianPixelNoise(self, nz=True):
		if nz:
			iv = self.invvar[self.invvar != 0.]
		else:
			iv = self.invvar
		return 1./np.sqrt(np.median(iv))

	def getMask(self):
		return self.mask

	def setMask(self, dilation=3):
		self.mask = (self.origInvvar <= 0.)
		if dilation > 0:
			self.mask = binary_dilation(self.mask,iterations=dilation)

	def getStarMask(self):
		return self.starMask
	def setStarMask(self,mask):
		self.starMask = mask


	def getOrigInvvar(self):
		return self.origInvvar
	def getImage(self):
		return self.data
	def getPsf(self):
		return self.psf
	def getWcs(self):
		return self.wcs
	def getPhotoCal(self):
		return self.photocal


# Adds two patches, handling the case when one is None
def add_patches(pa, pb):
	p = pa
	if pb is not None:
		if p is None:
			p = pb
		else:
			p += pb
	return p
	
class Patch(object):
	'''
	An image patch; a subimage.  In the Tractor we use these to hold
	synthesized (ie, model) images.  The patch is a rectangular grid
	of pixels and it knows its offset (2-d position) in some larger
	image.

	This class overloads arithmetic operations (like add and multiply)
	relevant to synthetic image patches.
	'''
	def __init__(self, x0, y0, patch):
		self.x0 = x0
		self.y0 = y0
		self.patch = patch
		self.name = ''
		if patch is not None:
			try:
				H,W = patch.shape
				self.size = H*W
			except:
				pass

	def __str__(self):
		s = 'Patch: '
		name = getattr(self, 'name', '')
		if len(name):
			s += name + ' '
		s += 'origin (%i,%i) ' % (self.x0, self.y0)
		if self.patch is not None:
			(H,W) = self.patch.shape
			s += 'size (%i x %i)' % (W, H)
		else:
			s += '(no image)'
		return s

	def set(self, other):
		self.x0 = other.x0
		self.y0 = other.y0
		self.patch = other.patch
		self.name = other.name
	
	def trimToNonZero(self):
		if self.patch is None:
			return
		H,W = self.patch.shape
		for x in range(W):
			if not np.all(self.patch[:,x] == 0):
				break
		x0 = x
		for x in range(W, 0, -1):
			if not np.all(self.patch[:,x-1] == 0):
				break
		x1 = x

		for y in range(H):
			if not np.all(self.patch[y,:] == 0):
				break
		y0 = y
		for y in range(H, 0, -1):
			if not np.all(self.patch[y-1,:] == 0):
				break
		y1 = y

		if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
			return

		self.patch = self.patch[y0:y1, x0:x1]
		self.x0 += x0
		self.y0 += y0

	def overlapsBbox(self, bbox):
		ext = self.getExtent()
		(x0,x1,y0,y1) = ext
		(ox0,ox1,oy0,oy1) = bbox
		if x0 >= ox1 or ox0 >= x1 or y0 >= oy1 or oy0 >= y1:
			return False
		return True

	def hasBboxOverlapWith(self, other):
		oext = other.getExtent()
		return self.overlapsBbox(oext)
		
	def hasNonzeroOverlapWith(self, other):
		if not self.hasBboxOverlapWith(other):
			return False
		ext = self.getExtent()
		(x0,x1,y0,y1) = ext
		oext = other.getExtent()
		(ox0,ox1,oy0,oy1) = oext
		ix,ox = get_overlapping_region(ox0, ox1-1, x0, x1-1)
		iy,oy = get_overlapping_region(oy0, oy1-1, y0, y1-1)
		ix = slice(ix.start -  x0, ix.stop -  x0)
		iy = slice(iy.start -  y0, iy.stop -  y0)
		sub = self.patch[iy,ix]
		osub = other.patch[oy,ox]
		assert(sub.shape == osub.shape)
		return np.sum(sub * osub) > 0.

	def getNonZeroMask(self):
		nz = (self.patch != 0)
		return Patch(self.x0, self.y0, nz)
	
	def __repr__(self):
		return str(self)
	def setName(self, name):
		self.name = name
	def getName(self):
		return self.name

	# for Cache
	#def size(self):
	#	(H,W) = self.patch.shape
	#	return H*W
	def copy(self):
		if self.patch is None:
			return Patch(self.x0, self.y0, None)
		return Patch(self.x0, self.y0, self.patch.copy())

	def getExtent(self, margin=0.):
		''' Return (x0, x1, y0, y1) '''
		(h,w) = self.shape
		return (self.x0-margin, self.x0 + w + margin,
				self.y0-margin, self.y0 + h + margin)

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
		if self.x0 >= W:
			# empty
			self.patch = None
			return False
		if self.y0 >= H:
			self.patch = None
			return False
		if self.x0 < 0:
			self.patch = self.patch[:, -self.x0:]
			self.x0 = 0
		if self.y0 < 0:
			self.patch = self.patch[-self.y0:, :]
			self.y0 = 0
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


	#### WARNing, this function has not been tested
	def clipToRoi(self, x0,x1,y0,y1):
		if self.patch is None:
			return False
		if ((self.x0 >= x1) or (self.x1 <= x0) or
			(self.y0 >= y1) or (self.y1 <= y0)):
			# empty
			self.patch = None
			return False

		if self.x0 < x0:
			self.patch = self.patch[:, x0-self.x0:]
			self.x0 = x0
		if self.y0 < y0:
			self.patch = self.patch[(y0-self.y0):, :]
			self.y0 = y0
		(h,w) = self.shape
		if (self.x0 + w) > x1:
			self.patch = self.patch[:, :(x1 - self.x0)]
		if (self.y0 + h) > y1:
			self.patch = self.patch[:(y1 - self.y0), :]
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
			return np.array([], np.int)
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

		# if False:
		# 	tmpimg = np.zeros_like(img)
		# 	tmpimg[outy,outx] = p * scale
		# 	plt.clf()
		# 	plt.imshow(tmpimg, interpolation='nearest', origin='lower')
		# 	plt.hot()
		# 	plt.colorbar()
		# 	fn = 'addto-%03i.png' % Patch.plotnum
		# 	plt.savefig(fn)
		# 	print 'Wrote', fn
		# 
		# 	plt.clf()
		# 	plt.imshow(p, interpolation='nearest', origin='lower')
		# 	plt.hot()
		# 	plt.colorbar()
		# 	fn = 'addto-%03i-p.png' % Patch.plotnum
		# 	plt.savefig(fn)
		# 	print 'Wrote', fn
		# 
		# 	Patch.plotnum += 1

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
	def __div__(self, x):
		if self.patch is None:
			return Patch(self.x0, self.y0, None)
		return Patch(self.x0, self.y0, self.patch / x)

	def performArithmetic(self, other, opname, otype=float):
		assert(isinstance(other, Patch))
		if (self.x0 == other.getX0() and self.y0 == other.getY0() and
			self.shape == other.shape):
			assert(self.x0 == other.getX0())
			assert(self.y0 == other.getY0())
			assert(self.shape == other.shape)
			if self.patch is None or other.patch is None:
				return Patch(self.x0, self.y0, None)
			pcopy = self.patch.copy()
			op = getattr(pcopy, opname)
			return Patch(self.x0, self.y0, op(other.patch))

		(ph,pw) = self.patch.shape
		(ox0,oy0) = other.getX0(), other.getY0()
		(oh,ow) = other.shape

		# Find the union of the regions.
		ux0 = min(ox0, self.x0)
		uy0 = min(oy0, self.y0)
		ux1 = max(ox0 + ow, self.x0 + pw)
		uy1 = max(oy0 + oh, self.y0 + ph)

		p = np.zeros((uy1 - uy0, ux1 - ux0), dtype=otype)
		p[self.y0 - uy0 : self.y0 - uy0 + ph,
		  self.x0 - ux0 : self.x0 - ux0 + pw] = self.patch

		psub = p[oy0 - uy0 : oy0 - uy0 + oh,
				 ox0 - ux0 : ox0 - ux0 + ow]
		op = getattr(psub, opname)
		op(other.getImage())
		return Patch(ux0, uy0, p)

	def __add__(self, other):
		return self.performArithmetic(other, '__iadd__')

	def __sub__(self, other):
		return self.performArithmetic(other, '__isub__')



class Catalog(MultiParams):
	'''
	A list of Source objects.  This class allows the Tractor to treat
	a set of astronomical sources as a single object with a bunch of
	parameters.  Most of the functionality comes from the base class.


	Constructor syntax:

	cat = Catalog(src1, src2, src3)

	so if you have a list of sources,

	srcs = [src1, src2, src3]
	cat = Catalog(*srcs)

	'''
	deepcopy = MultiParams.copy

	def __str__(self):
		return ('Catalog: %i sources, %i parameters' %
				(len(self), self.numberOfParams()))

	def printLong(self):
		print 'Catalog with %i sources:' % len(self)
		for i,x in enumerate(self):
			print '	 %i:' % i, x

	# inherited from MultiParams:
	# def __len__(self):
	#  ''' Returns the number of sources in this catalog'''
	# def numberOfParams(self):
	#  '''Returns the number of active parameters in all sources'''

	def getThawedSources(self):
		return self._getActiveSubs()

	def getFrozenSources(self):
		return self._getInactiveSubs()

	def getNamedParamName(self, j):
		return 'source%i' % j

	def thawSourcesInCircle(self, pos, radius):
		for i,src in enumerate(self):
			if src.overlapsCircle(pos, radius):
				self.thawParam(i)
			


class Images(MultiParams):
	"""
	This is a class for holding a list of `Image` objects, each which
	contains data and metadata.  This class allows the Tractor to
	treat a list of `Image`s as a single object that has a set of
	parameters.  Basically all the functionality comes from the base
	class.
	"""
	def getNamedParamName(self, j):
		return 'image%i' % j

# These are free functions for multiprocessing in "getderivs2()"
def getmodelimagestep((tr, j, k, p0, step)):
	im = tr.getImage(j)
	#print 'Setting param', p0, step, p0+step
	im.setParam(k, p0 + step)
	mod = tr.getModelImage(im)
	im.setParam(k, p0)
	return mod
def getmodelimagefunc((tr, imj)):
	print 'getmodelimagefunc(): imj', imj, 'pid', os.getpid()
	return tr.getModelImage(imj)
def getsrcderivs((src, img)):
	return src.getParamDerivatives(img)

def getimagederivs((imj, img, tractor, srcs)):
	## FIXME -- avoid shipping all images...
	return img.getParamDerivatives(tractor, srcs)

def getmodelimagefunc2((tr, im)):
	#print 'getmodelimagefunc2(): im', im, 'pid', os.getpid()
	#tr.images = Images(im)
	try:
		return tr.getModelImage(im)
	except:
		import traceback
		print 'Exception in getmodelimagefun2:'
		traceback.print_exc()
		raise

class Tractor(MultiParams):
	"""
	Heavy farm machinery.

	As you might guess from the name, this is the main class of the
	Tractor framework.  A Tractor has a set of Images and a set of
	Sources, and has methods to optimize the parameters of those
	Images and Sources.

	"""
	@staticmethod
	def getName():
		return 'Tractor'
	
	@staticmethod
	def getNamedParams():
		return dict(images=0, catalog=1)

	def __init__(self, images=[], catalog=[], mp=None):
		'''
		- `images:` list of Image objects (data)
		- `catalog:` list of Source objects
		'''
		if not isinstance(images, Images):
			images = Images(*images)
		if not isinstance(catalog, Catalog):
			catalog = Catalog(*catalog)
		super(Tractor,self).__init__(images, catalog)
		self._setup(mp=mp)

	# def __del__(self):
	# 	# dstn is not sure this is necessary / useful
	# 	if self.cache is not None:
	# 		self.cache.clear()
	# 	del self.cache

	def _setup(self, mp=None, cache=None, pickleCache=False):
		if mp is None:
			mp = multiproc()
		self.mp = mp
		self.modtype = np.float32
		if cache is None:
			cache = Cache()
		self.cache = cache
		self.pickleCache = pickleCache

	def __str__(self):
		s = '%s with %i sources and %i images' % (self.getName(), len(self.catalog), len(self.images))
		s += ' (' + ', '.join([im.name for im in self.images]) + ')'
		return s

	def is_multiproc(self):
		return self.mp.pool is not None

	def _map(self, func, iterable):
		return self.mp.map(func, iterable)
	def _map_async(self, func, iterable):
		return self.mp.map_async(func, iterable)

	# For use from emcee
	def __call__(self, X):
		# print self.getName()+'.__call__: I am pid', os.getpid()
		self.setParams(X)
		lnp = self.getLogProb()
		# print self.cache
		# from .sdss_galaxy import get_galaxy_cache
		# print 'Galaxy cache:', get_galaxy_cache()
		# print 'Items:'
		# self.cache.printItems()
		# print
		return lnp

	# For pickling
	def __getstate__(self):
		#print 'pickling tractor in pid', os.getpid()
		S = (self.getImages(), self.getCatalog(), self.liquid)
		if self.pickleCache:
			S = S + (self.cache,)
		#print 'Tractor.__getstate__:', S
		return S
	def __setstate__(self, state):
		#print 'unpickling tractor in pid', os.getpid()
		args = {}
		if len(state) == 3:
			(images, catalog, liquid) = state
		elif len(state) == 4:
			(images, catalog, liquid, cache) = state
			args.update(cache=cache, pickleCache=pickleCache)
		self.subs = [images, catalog]
		self.liquid = liquid
		self._setup(**args)

	def getNImages(self):
		return len(self.images)

	def getImage(self, imgi):
		return self.images[imgi]

	def getImages(self):
		return self.images

	def getCatalog(self):
		return self.catalog

	def setCatalog(self, srcs):
		# FIXME -- ensure that "srcs" is a Catalog?	 Or duck-type it?
		self.catalog = srcs

	def setImages(self, ims):
		self.images = ims

	def addImage(self, img):
		self.images.append(img)

	def addSource(self, src):
		self.catalog.append(src)

	def addSources(self, srcs):
		self.catalog.extend(srcs)

	def removeSource(self, src):
		self.catalog.remove(src)

	def computeParameterErrors(self, symmetric=False):
		if not symmetric:
			return self._param_errors_1()

		e1 = self._param_errors_1(1.)
		e2 = self._param_errors_1(-1.)
		sigs = []
		for s1,s2 in zip(e1,e2):
			if s1 is None:
				sigs.append(s2)
			elif s2 is None:
				sigs.append(s1)
			else:
				sigs.append((s1 + s2) / 2.)
		return sigs

	def _param_errors_1(self, sign=1.):
		# Try to compute 1-sigma error bars on each parameter by
		# sweeping the parameter (in the "getStepSizes()" direction)
		# until we find delta-chi-squared of 1.
		# That's a delta-logprob of 0.5
		stepsizes = np.array(self.getStepSizes())
		pp0 = np.array(self.getParams())
		lnp0 = self.getLogProb()
		nms = self.getParamNames()
		sigmas = []
		target = lnp0 - 0.5
		for i,(p0,step,nm) in enumerate(zip(pp0, stepsizes, nms)):
			self.setParams(pp0)
			# Take increasingly large steps until we find one with
			# logprob < target
			p1 = None
			#print 'Looking for error bars on', nm, 'around', p0
			for j in range(20):
				tryp1 = p0 + sign * step * (2. ** j)
				self.setParam(i, tryp1)
				lnp1 = self.getLogProb()
				#print '  stepping to', tryp1, 'for dlnp', lnp1 - lnp0
				# FIXME -- could also track the largest lnp < target,
				# to narrow the binary search range later...
				if lnp1 < target:
					p1 = tryp1
					break
			if p1 is None:
				sigmas.append(None)
				continue
			# Binary search until the range is small enough.
			lo,hi = min(p0, p1), max(p0, p1)
			lnplo,lnphi = lnp0,lnp1
			#print 'Binary searching in', lo, hi
			sigma = None
			for j in range(20):
				assert(lo <= hi)
				if np.abs(lnplo - target) < 1e-3:
					sigma = np.abs(lo - p0)
					break
				mid = (lo + hi) / 2.
				self.setParam(i, mid)
				lnpmid = self.getLogProb()
				#print '  trying', mid, 'for dlnp', lnpmid - lnp0
				if lnpmid < target:
					hi = mid
					lnphi = lnpmid
				else:
					lo = mid
					lnplo = lnpmid
			sigmas.append(sigma)
		return np.array(sigmas)

	def optimize_lbfgsb(self, hessian_terms=10, plotfn=None):

		XX = []
		OO = []
		def objective(x, tractor, stepsizes, lnp0):
			res = lnp0 - tractor(x * stepsizes)
			print 'LBFGSB objective:', res
			if plotfn:
				XX.append(x.copy())
				OO.append(res)
			return res

		from scipy.optimize import fmin_l_bfgs_b

		stepsizes = np.array(self.getStepSizes())
		p0 = np.array(self.getParams())
		lnp0 = self.getLogProb()

		print 'Active parameters:', len(p0)

		print 'Calling L-BFGS-B ...'
		X = fmin_l_bfgs_b(objective, p0 / stepsizes, fprime=None,
						  args=(self, stepsizes, lnp0),
						  approx_grad=True, bounds=None, m=hessian_terms,
						  epsilon=1e-8, iprint=0)
		p1,lnp1,d = X
		print d
		print 'lnp0:', lnp0
		self.setParams(p1 * stepsizes)
		print 'lnp1:', self.getLogProb()

		if plotfn:
			import pylab as plt
			plt.clf()
			XX = np.array(XX)
			OO = np.array(OO)
			print 'XX shape', XX.shape
			(N,D) = XX.shape
			for i in range(D):
				OO[np.abs(OO) < 1e-8] = 1e-8
				neg = (OO < 0)
				plt.semilogy(XX[neg,i], -OO[neg], 'bx', ms=12, mew=2)
				pos = np.logical_not(neg)
				plt.semilogy(XX[pos,i], OO[pos], 'rx', ms=12, mew=2)
				I = np.argsort(XX[:,i])
				plt.plot(XX[I,i], np.abs(OO[I]), 'k-', alpha=0.5)
				plt.ylabel('Objective value')
				plt.xlabel('Parameter')
			plt.twinx()
			for i in range(D):
				plt.plot(XX[:,i], np.arange(N), 'r-')
				plt.ylabel('L-BFGS-B iteration number')
			plt.savefig(plotfn)

	def optimize_forced_photometry(self, alphas=None, damp=0, priors=False,
								   minsb=0.,
								   mindlnp=1.,
								   rois=None,
								   sky=False,
								   minFlux=None,
								   fitstats=False,
								   justims0=False,
								   variance=False,
								   skyvariance=False):
		'''
		ASSUMES linear brightnesses!

		ASSUMES all source parameters except brightness are frozen.

		If sky=False,
		ASSUMES image parameters are frozen.
		If sky=True,
		ASSUMES only the sky parameters are unfrozen

		ASSUMES the PSF and Sky models are position-independent!!

		PRIORS probably don't work because we don't setParams() when evaluating
		likelihood or prior!
		'''
		from basics import LinearPhotoCal, ShiftedWcs

		assert(not priors)

		scales = []
		imgs = self.getImages()
		for img in imgs:
			assert(isinstance(img.getPhotoCal(), LinearPhotoCal))
			scales.append(img.getPhotoCal().getScale())

		# HACK -- if sky=True, assume we are fitting the sky in ALL images.
		# We could ask which ones are thawed...
		if sky:
			for img in imgs:
				# FIXME -- would be nice to allow multi-param linear sky models
				assert(img.getSky().numberOfParams() == 1)

		Nsourceparams = self.catalog.numberOfParams()

		if rois is not None:
			assert(len(rois) == len(imgs))

		#
		# Here we build up the "umodels" nested list, which has shape
		# (if it were a numpy array) of (len(images), len(srcs))
		# where each element is None, or a Patch with the unit-flux model
		# of that source in that image.
		#
		t0 = Time()
		umodels = []
		subimgs = []
		srcs = list(self.catalog.getThawedSources())

		umodtosource = {}
		umodsforsource = [[] for s in srcs]

		for i,img in enumerate(imgs):
			umods = []
			pcal = img.getPhotoCal()

			nvalid = 0
			nallzero = 0
			nzero = 0
			
			if rois is not None:
				roi = rois[i]
				y0 = roi[0].start
				x0 = roi[1].start
				subwcs = ShiftedWcs(img.wcs, x0, y0)
				subimg = Image(data=img.data[roi], invvar=img.invvar[roi],
							   psf=img.psf, wcs=subwcs, sky=img.sky,
							   photocal=img.photocal, name=img.name)
				subimgs.append(subimg)
			else:
				x0 = y0 = 0

			for si,src in enumerate(srcs):
				cc = [pcal.brightnessToCounts(b) for b in src.getBrightnesses()]
				csum = sum(cc)
				if csum == 0:
					mv = 0.
				else:
					mv = minsb / csum
				ums = src.getUnitFluxModelPatches(img, minval=mv)

				isvalid = False
				isallzero = False

				for ui,um in enumerate(ums):
					if um is None:
						continue
					um.x0 -= x0
					um.y0 -= y0
					isvalid = True
					if um.patch.sum() == 0:
						nzero += 1
					else:
						isallzero = False


				# first image only:
				if i == 0:
					for ui in range(len(ums)):
						umodtosource[len(umods) + ui] = si
						umodsforsource[si].append(len(umods) + ui)

				umods.extend(ums)

				if isvalid:
					nvalid += 1
					if isallzero:
						nallzero += 1

			#print 'Img', i, 'has', nvalid, 'of', len(srcs), 'sources'
			#print '  ', nallzero, 'of which are all zero'
			#print '  ', nzero, 'components are zero'

			assert(len(umods) == Nsourceparams)
			umodels.append(umods)
		tmods = Time()-t0
		print 'forced phot: getting unit-flux models:', tmods

		if rois is None:
			imlist = imgs
		else:
			imlist = subimgs
		
		t0 = Time()
		fsrcs = list(self.catalog.getFrozenSources())
		mod0 = []
		for img in imlist:
			# "sky = not sky": I'm not just being contrary :)
			# If we're fitting sky, we'll do a setParams() and get the sky models
			# to render themselves when evaluating lnProbs, rather than pre-computing
			# the nominal value here and then computing derivatives.
			mod0.append(self.getModelImage(img, fsrcs, minsb=minsb, sky=not sky))
		tmod = Time() - t0
		print 'forced phot: getting frozen-source model:', tmod

		if sky:
			t0 = Time()
			skyderivs = []
			# build the derivative list as required by getUpdateDirection:
			#    (param0) ->  [  (deriv, img), (deriv, img), ...   ], ... ],
			for img in imlist:
				dskys = img.getSky().getParamDerivatives(self, img, None)
				#print 'Image', img
				#print 'Derivatives', dskys
				for dsky in dskys:
					skyderivs.append([(dsky, img)])
			Nsky = len(skyderivs)
			#print 'Nsky:', Nsky
			#print 'Image params:'
			#self.images.printThawedParams()
			assert(Nsky == self.images.numberOfParams())
			assert(Nsky + Nsourceparams == self.numberOfParams())
			print 'forced phot: sky derivs', Time()-t0
		else:
			Nsky = 0

		t0 = Time()
		derivs = [[] for i in range(Nsourceparams)]
		for i,(tim,umods,scale) in enumerate(zip(imlist, umodels, scales)):
			for um,dd in zip(umods, derivs):
				if um is None:
					continue
				dd.append((um * scale, tim))
		print 'forced phot: derivs', Time()-t0


		#### NOTE, this is actually INVERSE variance
		V = None

		if variance:
			NSV = 0
			if sky and skyvariance:
				NS = Nsky
			else:
				NS = 0
			V = np.zeros(len(srcs) + NS)
			if sky and skyvariance:
				for di,(dsky,tim) in enumerate(skyderivs):
					ie = tim.getInvError()
					if dsky.shape == tim.shape:
						dchi2 = np.sum((dsky.patch * ie)**2)
					else:
						mm = np.zeros(tim.shape)
						dsky.addTo(mm)
						dchi2 = np.sum((mm * ie)**2)
					V[di] = dchi2
				
			for i,(tim,umods,scale) in enumerate(zip(imlist, umodels, scales)):
				mm = np.zeros(tim.shape)
				ie = tim.getInvError()
				for ui,um in enumerate(umods):
					if um is None:
						continue
					#print 'deriv: sum', um.patch.sum(), 'scale', scale, 'shape', um.shape,
					um.addTo(mm)
					dchi2 = np.sum((mm * scale * ie) ** 2)
					#print 'dchi2', dchi2
					V[NS + ui] += dchi2
					mm[:,:] = 0.
			#print 'var1:', V

		if sky:
			# print 'Catalog params:', self.catalog.numberOfParams()
			# print 'Image params:', self.images.numberOfParams()
			# print 'Total # params:', self.numberOfParams()
			# print 'cat derivs:', len(derivs)
			# print 'sky derivs:', len(skyderivs)
			# print 'total # derivs:', len(derivs) + len(skyderivs)
			# Sky derivatives are part of the image derivatives, so go first in
			# the derivative list.
			derivs = skyderivs + derivs

		assert(len(derivs) == self.numberOfParams())

		# About rois and derivs: we call
		#   getUpdateDirection(derivs, ..., chiImages=[chis])
		# And this uses the "img" objects in "derivs" to decide on the region
		# that is being optimized; the number of rows = total number of pixels.
		# We have to make sure that "chiImages" matches that size.
		#
		# We shift the unit-flux models (above, um.x0 -= x0) to be relative to the
		# ROI.

		def lnpForUpdate(mod0, imgs, umodels, X, alpha, p0, tractor, rois, scales,
						 p0sky, Xsky, priors, sky, minFlux):
			ims = []
			if X is None:
				pa = p0
			else:
				pa = [p + alpha * d for p,d in zip(p0, X)]
			chisq = 0.
			chis = []

			if Xsky is not None:
				self.images.setParams([p + alpha * d for p,d in zip(p0sky, Xsky)])

			# Recall that "umodels" is a full matrix (shape (Nimage,
			# Nsrcs)) of patches, so we just go through each image,
			# ignoring None entries and building up model images from
			# the scaled unit-flux patches.
			
			for i,(img,umods,m0,scale) in enumerate(zip(imgs, umodels, mod0, scales)):
				roi = None
				if rois:
					roi = rois[i]
				mod = m0.copy()
				if sky:
					img.getSky().addTo(mod)
				for b,um in zip(pa,umods):
					if um is None:
						continue
					if minFlux is not None:
						b = max(b, minFlux)
					counts = b * scale
					if counts == 0.:
						continue
					(um * counts).addTo(mod)

				ie = img.getInvError()
				im = img.getImage()
				if roi is not None:
					ie = ie[roi]
					im = ie[roi]
				chi = (im - mod) * ie
				ims.append((im, mod, ie, chi, roi))

				chis.append(chi)
				chisq += (chi**2).sum()
			lnp = -0.5 * chisq
			if priors:
				lnp += tractor.getLogPrior()
			return lnp,	chis, ims

		# debugging images
		ims0 = None
		imsBest = None

		lnp0 = None
		chis0 = None
		quitNow = False

		## FIXME -- this should depend on the PhotoCal scalings!
		damp0 = 1e-3
		damping = damp

		while True:
			# A flag to try again even if the lnprob got worse
			tryAgain = False

			p0 = self.getParams()
			if sky:
				p0sky = p0[:Nsky]
				p0 = p0[Nsky:]

			if lnp0 is None:
				t0 = Time()
				lnp0,chis0,ims0 = lnpForUpdate(mod0, imgs, umodels, None, None, p0,
											   self, rois, scales, None, None, priors,
											   sky, minFlux)
				print 'forced phot: initial lnp', Time()-t0

			if justims0:
				return lnp0,chis0,ims0

			# print 'Starting opt loop with'
			# print '  p0', p0
			# print '  lnp0', lnp0
			# print '  chisqs', [(chi**2).sum() for chi in chis0]
			# print 'chis0:', chis0

			# Ugly: getUpdateDirection calls self.getImages(), and
			# ASSUMES they are the same as the images referred-to in
			# the "derivs", to figure out which chi image goes with
			# which image.  Temporarily set images = subimages
			if rois is not None:
				realims = self.images
				self.images = subimgs

			print 'forced phot: getting update with damp=', damping
			t0 = Time()
			# X,V = self.getUpdateDirection(derivs, damp=damping, priors=priors,
			# 							  scale_columns=False, chiImages=chis0,
			# 							  variance=True)
			# print 'Source variance:', V[Nsky:]
			# if not skyvariance: ...FIXME...

			X = self.getUpdateDirection(derivs, damp=damping, priors=priors,
										scale_columns=False, chiImages=chis0)
			# print 'Source variance:', V[Nsky:]

			topt = Time()-t0
			print 'forced phot: opt:', topt
			#print 'forced phot: update', X
			if rois is not None:
				self.images = realims

			if len(X) == 0:
				print 'Error getting update direction'
				break
			
			## tryUpdates():
			if alphas is None:
				# 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
				alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

			if sky:
				# Split the sky-update parameters from the source parameters
				Xsky = X[:Nsky]
				X = X[Nsky:]
			else:
				p0sky = Xsky = None

			# Check whether the update produces all fluxes above the
			# minimums: if so we should be able to take the step with
			# alpha=1 and quit.

			#print 'p0:', p0
			#print 'X:', X
			if (minFlux is None) or np.all((p0 + X) >= minFlux):
				#print 'Update produces non-negative fluxes; accepting with alpha=1'
				alphas = [1.]
				quitNow = True
			else:
				print 'Some too-negative fluxes requested:'
				print 'Fluxes:', p0
				print 'Update:', X
				print 'Total :', p0+X
				print 'MinFlux:', minFlux
				if damp == 0.0:
					damping = damp0
					damp0 *= 10.
					print 'Setting damping to', damping
					if damp0 < 1e3:
						tryAgain = True

			lnpBest = lnp0
			alphaBest = None
			chiBest = None

			for alpha in alphas:
				t0 = Time()
				lnp,chis,ims = lnpForUpdate(mod0, imgs, umodels, X, alpha, p0, self,
											rois, scales, p0sky, Xsky, priors, sky,
											minFlux)
				print 'Forced phot: stepped with alpha', alpha, 'for dlnp', lnp-lnp0
				print 'Took', Time()-t0
				if lnp < (lnpBest - 1.):
					break
				if lnp > lnpBest:
					alphaBest = alpha
					lnpBest = lnp
					chiBest = chis
					imsBest = ims

			#logmsg('  Stepping by', alphaBest, 'for delta-logprob', lnpBest - lnp0)
			if alphaBest is not None:
				# Clamp fluxes up to zero
				if minFlux is not None:
					pa = [max(minFlux, p + alphaBest * d) for p,d in zip(p0, X)]
				else:
					pa = [p + alphaBest * d for p,d in zip(p0, X)]
				self.catalog.setParams(pa)

				if sky:
					self.images.setParams([p + alpha * d for p,d in zip(p0sky, Xsky)])

				dlogprob = lnpBest - lnp0
				alpha = alphaBest
				lnp0 = lnpBest
				chis0 = chiBest
				# print 'Accepting alpha =', alpha
				# print 'new lnp0', lnp0
				# print 'new chisqs', [(chi**2).sum() for chi in chis0]
				# print 'new params', self.getParams()
			else:
				dlogprob = 0.
				alpha = 0.

				### ??
				if sky:
					# Revert -- recall that we change params while probing in lnpForUpdate()
					self.images.setParams(p0sky)

			#tstep = Time() - t0
			#print 'forced phot: line search:', tstep
			#print 'forced phot: alpha', alphaBest, 'for delta-lnprob', dlogprob
			if dlogprob < mindlnp:
				if not tryAgain:
					break

			if quitNow:
				break

		rtn = (ims0,imsBest,V)

		if fitstats and imsBest is None:
			rtn = rtn + (None,)
		elif fitstats:
			class FitStats(object):
				pass
			fs = FitStats()

			# Per-image stats:
			imchisq = []
			imnpix = []
			for img,mod,ie,chi,roi in imsBest:
				imchisq.append((chi**2).sum())
				imnpix.append((ie > 0).sum())
			fs.imchisq = np.array(imchisq)
			fs.imnpix = np.array(imnpix)

			# Per-source stats:

			# Weak!  Assume one component per source
			#assert(len(umodels[0]) == len(srcs))
			
			# profile-weighted chi-squared (unit-model weighted chi-squared)
			fs.prochi2 = np.zeros(len(srcs))

			# profile-weighted number of pixels
			fs.pronpix = np.zeros(len(srcs))

			# profile-weighted sum of (flux from other sources / my flux)
			fs.profracflux = np.zeros(len(srcs))
			fs.proflux = np.zeros(len(srcs))

			# total number of pixels touched by this source
			fs.npix = np.zeros(len(srcs), int)

			# subtract sky from models before measuring others' flux within my profile
			skies = []
			for tim,(img,mod,ie,chi,roi) in zip(imlist, imsBest):
				tim.getSky().addTo(mod, scale=-1.)
				skies.append(tim.getSky().val)
			fs.sky = np.array(skies)

			# Some fancy footwork to convert from umods to sources
			# (eg, composite galaxies that can have multiple umods)
			# for each source
			for si,uis in enumerate(umodsforsource):
				src = self.catalog[si]
				# for each image
				for umods,scale,tim,(img,mod,ie,chi,roi) in zip(umodels, scales, imlist, imsBest):
					# just use 'scale'?
					pcal = tim.getPhotoCal()
					cc = [pcal.brightnessToCounts(b) for b in src.getBrightnesses()]
					csum = sum(cc)
					if csum == 0:
						continue

					srcmod = np.zeros_like(chi)
					# for each component (usually just one)
					for ui,counts in zip(uis, cc):
						um = umods[ui]
						if um is None:
							continue
						# accumulate this unit-flux model into srcmod
						(um * counts).addTo(srcmod)

					#ssum = srcmod.sum()
					# Divide by total flux, not flux within this image; sum <= 1.
					srcmod /= csum

					nz = np.flatnonzero((srcmod > 0) * (ie > 0))
					if len(nz) == 0:
						continue

					fs.prochi2[si] += np.sum(srcmod.flat[nz] * chi.flat[nz]**2)
					fs.pronpix[si] += np.sum(srcmod.flat[nz])
					# (mod - srcmod*csum) is the model for everybody else
					fs.profracflux[si] += np.sum(((mod/csum - srcmod) * srcmod).flat[nz])
					# scale to nanomaggies, weight by profile
					fs.proflux[si] += np.sum((((mod - srcmod*csum) / scale) * srcmod).flat[nz])
					fs.npix[si] += len(nz)

			# re-add sky
			for tim,(img,mod,ie,chi,roi) in zip(imlist, imsBest):
				tim.getSky().addTo(mod)

			rtn = rtn + (fs,)

		return rtn


	def optimize(self, alphas=None, damp=0, priors=True, scale_columns=True):
		'''
		Performs *one step* of linearized least-squares + line search.
		
		Returns (delta-logprob, parameter update X, alpha stepsize)
		'''
		print self.getName()+': Finding derivs...'
		t0 = Time()
		allderivs = self.getDerivs()
		tderivs = Time()-t0
		#print Time() - t0
		#print 'allderivs:', allderivs
		#for d in allderivs:
		#	for (p,im) in d:
		#		print 'patch mean', np.mean(p.patch)
		print 'Finding optimal update direction...'
		t0 = Time()
		X = self.getUpdateDirection(allderivs, damp=damp, priors=priors,
									scale_columns=scale_columns)
		#print Time() - t0
		topt = Time()-t0
		#print 'X:', X
		if len(X) == 0:
			return 0, X, 0.
		print 'Finding optimal step size...'
		t0 = Time()
		(dlogprob, alpha) = self.tryUpdates(X, alphas=alphas)
		tstep = Time() - t0
		print 'Finished opt2.'
		print '  alpha =',alpha
		print '  Tderiv', tderivs
		print '  Topt  ', topt
		print '  Tstep ', tstep
		return dlogprob, X, alpha

	def getParameterScales(self):
		print self.getName()+': Finding derivs...'
		allderivs = self.getDerivs()
		print 'Finding column scales...'
		s = self.getUpdateDirection(allderivs, scales_only=True)
		return s

	def tryUpdates(self, X, alphas=None):
		if alphas is None:
			# 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
			alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

		pBefore = self.getLogProb()
		logverb('  log-prob before:', pBefore)
		pBest = pBefore
		alphaBest = None
		p0 = self.getParams()
		for alpha in alphas:
			logverb('  Stepping with alpha =', alpha)
			pa = [p + alpha * d for p,d in zip(p0, X)]
			self.setParams(pa)
			pAfter = self.getLogProb()
			logverb('  Log-prob after:', pAfter)
			logverb('  delta log-prob:', pAfter - pBefore)
			if pAfter < (pBest - 1.):
				break

			if pAfter > pBest:
				alphaBest = alpha
				pBest = pAfter
		
		if alphaBest is None or alphaBest == 0:
			print "Warning: optimization is borking"
			print "Parameter direction =",X
			print "Parameters and step sizes:"
			for n,p,s in zip(self.getParamNames(), self.getParams(), self.getStepSizes()):
				print n, p, s
		if alphaBest is None:
			self.setParams(p0)
			return 0, 0.

		logmsg('  Stepping by', alphaBest, 'for delta-logprob', pBest - pBefore)
		pa = [p + alphaBest * d for p,d in zip(p0, X)]
		self.setParams(pa)
		return pBest - pBefore, alphaBest


	def getDerivs(self):
		# Returns:
		# allderivs: [
		#	 (param0:)	[  (deriv, img), (deriv, img), ... ],
		#	 (param1:)	[],
		#	 (param2:)	[  (deriv, img), ],
		# ]
		allderivs = []

		# First, derivs for Image parameters (because 'images' comes first in the
		# tractor's parameters)
		if self.isParamFrozen('images'):
			ims = []
			imjs = []
		else:
			imjs = [i for i in self.images.getThawedParamIndices()]
			ims = [self.images[j] for j in imjs]

		if self.isParamFrozen('catalog'):
			srcs = []
		else:
			srcs = list(self.catalog.getThawedSources())
			#print len(srcs), 'thawed sources'
			
		# FIXME -- don't we want Sky, PSF, etc to be able to provide
		# their own derivatives?

		imderivs = self._map(getimagederivs, [(imj, im, self, srcs)
											  for im,imj in zip(ims, imjs)])

		needimjs = []
		needims = []
		needparams = []

		for derivs,im,imj in zip(imderivs, ims, imjs):
			need = []
			for k,d in enumerate(derivs):
				if d is False:
					need.append(k)
			if len(need):
				needimjs.append(imj)
				needims.append(im)
				needparams.append(need)
					
		# initial models...
		print 'Getting', len(needimjs), 'initial models for image derivatives'
		mod0s = self._map_async(getmodelimagefunc, [(self, imj) for imj in needimjs])
		# stepping each (needed) param...
		args = []
		# for j,im in enumerate(ims):
		# 	p0 = im.getParams()
		# 	#print 'Image', im
		# 	#print 'Step sizes:', im.getStepSizes()
		# 	#print 'p0:', p0
		# 	for k,step in enumerate(im.getStepSizes()):
		# 		args.append((self, j, k, p0[k], step))
		for im,imj,params in zip(needims, needimjs, needparams):
		 	p0 = im.getParams()
			ss = im.getStepSizes()
			for i in params:
				args.append((self, imj, i, p0[i], ss[i]))
		# reverse the args so we can pop() below.
		print 'Stepping in', len(args), 'model parameters for derivatives'
		mod1s = self._map_async(getmodelimagestep, reversed(args))

		# Next, derivs for the sources.
		args = []
		for j,src in enumerate(srcs):
			for i,img in enumerate(self.images):
				args.append((src, img))
		sderivs = self._map_async(getsrcderivs, reversed(args))

		# Wait for and unpack the image derivatives...
		mod0s = mod0s.get()
		mod1s = mod1s.get()
		# convert to a imj->mod0 map
		assert(len(mod0s) == len(needimjs))
		mod0s = dict(zip(needimjs, mod0s))

		for derivs,im,imj in zip(imderivs, ims, imjs):
			for k,d in enumerate(derivs):
				if d is False:
					mod0 = mod0s[imj]
					nm = im.getParamNames()[k]
					step = im.getStepSizes()[k]
					mod1 = mod1s.pop()
					d = Patch(0, 0, (mod1 - mod0) / step)
					d.name = 'd(im%i)/d(%s)' % (j,nm)
				allderivs.append([(d, im)])
		# popped all
		assert(len(mod1s) == 0)

		# for i,(j,im) in enumerate(zip(imjs,ims)):
		# 	mod0 = mod0s[i]
		# 	p0 = im.getParams()
		# 	for k,(nm,step) in enumerate(zip(im.getParamNames(), im.getStepSizes())):
		# 		mod1 = mod1s.pop()
		# 		deriv = Patch(0, 0, (mod1 - mod0) / step)
		# 		deriv.name = 'd(im%i)/d(%s)' % (j,nm)
		# 		allderivs.append([(deriv, im)])

		# Wait for source derivs...
		sderivs = sderivs.get()

		for j,src in enumerate(srcs):
			srcderivs = [[] for i in range(src.numberOfParams())]
			for i,img in enumerate(self.images):
				# Get derivatives (in this image) of params
				derivs = sderivs.pop()
				# derivs is a list of Patch objects or None, one per parameter.
				assert(len(derivs) == src.numberOfParams())
				for k,deriv in enumerate(derivs):
					if deriv is None:
						continue
					if not np.all(np.isfinite(deriv.patch.ravel())):
						print 'Derivative for source', src
						print 'deriv index', i
						assert(False)
					srcderivs[k].append((deriv, img))
			allderivs.extend(srcderivs)

		assert(len(allderivs) == self.numberOfParams())
		return allderivs

	def getUpdateDirection(self, allderivs, damp=0., priors=True,
						   scale_columns=True, scales_only=False,
						   chiImages=None, variance=False):

		# allderivs: [
		#	 (param0:)	[  (deriv, img), (deriv, img), ... ],
		#	 (param1:)	[],
		#	 (param2:)	[  (deriv, img), ],
		# ]
		# The "img"s may repeat
		# "deriv" are Patch objects.

		# Each position in the "allderivs" array corresponds to a
		# model parameter that we are optimizing

		# We want to minimize:
		#	|| chi + (d(chi)/d(params)) * dparams ||^2
		# So  b = chi
		#	  A = -d(chi)/d(params)
		#	  x = dparams
		#
		# chi = (data - model) / std = (data - model) * inverr
		# derivs = d(model)/d(param)
		# A matrix = -d(chi)/d(param)
		#		   = + (derivs) * inverr

		# Parameters to optimize go in the columns of matrix A
		# Pixels go in the rows.

		# Build the sparse matrix of derivatives:
		sprows = []
		spcols = []
		spvals = []

		# Keep track of row offsets for each image.
		imgoffs = {}
		nextrow = 0
		for param in allderivs:
			for deriv,img in param:
				if img in imgoffs:
					continue
				imgoffs[img] = nextrow
				#print 'Putting image', img.name, 'at row offset', nextrow
				nextrow += img.numberOfPixels()
		Nrows = nextrow
		del nextrow
		Ncols = len(allderivs)

		colscales = []
		for col, param in enumerate(allderivs):
			RR = []
			VV = []
			WW = []
			for (deriv, img) in param:
				inverrs = img.getInvError()
				(H,W) = img.shape
				row0 = imgoffs[img]
				deriv.clipTo(W, H)
				pix = deriv.getPixelIndices(img)
				if len(pix) == 0:
					#print 'This param does not influence this image!'
					continue

				assert(np.all(pix < img.numberOfPixels()))
				# (grab non-zero indices)
				dimg = deriv.getImage()
				nz = np.flatnonzero(dimg)
				#print '  source', j, 'derivative', p, 'has', len(nz), 'non-zero entries'
				if len(nz) == 0:
					continue
				rows = row0 + pix[nz]
				#print 'Adding derivative', deriv.getName(), 'for image', img.name
				vals = dimg.ravel()[nz]
				w = inverrs[deriv.getSlice(img)].ravel()[nz]
				assert(vals.shape == w.shape)
				if not scales_only:
					RR.append(rows)
					VV.append(vals)
					WW.append(w)

			# massage, re-scale, and clean up matrix elements
			if len(VV) == 0:
				colscales.append(1.)
				continue
			rows = np.hstack(RR)
			VV = np.hstack(VV)
			WW = np.hstack(WW)
			#vals = np.hstack(VV) * np.hstack(WW)
			#print 'VV absmin:', np.min(np.abs(VV))
			#print 'WW absmin:', np.min(np.abs(WW))
			#print 'VV type', VV.dtype
			#print 'WW type', WW.dtype
			vals = VV * WW
			#print 'vals absmin:', np.min(np.abs(vals))
			#print 'vals absmax:', np.max(np.abs(vals))
			#print 'vals type', vals.dtype

			# shouldn't be necessary since we check len(nz)>0 above
			#if len(vals) == 0:
			#	colscales.append(1.)
			#	continue
			mx = np.max(np.abs(vals))
			if mx == 0:
				print 'mx == 0:', len(np.flatnonzero(VV)), 'of', len(VV), 'non-zero derivatives,',
				print len(np.flatnonzero(WW)), 'of', len(WW), 'non-zero weights;',
				print len(np.flatnonzero(vals)), 'non-zero products'
				colscales.append(1.)
				continue
			# MAGIC number: near-zero matrix elements -> 0
			# 'mx' is the max value in this column.
			FACTOR = 1.e-10
			I = (np.abs(vals) > (FACTOR * mx))
			rows = rows[I]
			vals = vals[I]
			scale = np.sqrt(np.dot(vals, vals))
			colscales.append(scale)
			assert(len(colscales) == (col+1))
			logverb('Column', col, 'scale:', scale)
			if scales_only:
				continue
			sprows.append(rows)
			c = np.empty_like(rows)
			c[:] = col
			spcols.append(c)
			if scale_columns:
				spvals.append(vals / scale)
			else:
				spvals.append(vals)
				
		colscale = np.array(colscales)
		if scales_only:
			return colscale

		b = None
		if priors:
			# We don't include the priors in the "colscale"
			# computation above, mostly because the priors are
			# returned as sparse additions to the matrix, and not
			# necessarily column-oriented the way the other params
			# are.  It would be possible to make it work, but dstn is
			# not convinced it's worth the effort right now.
			X = self.getLogPriorDerivatives()
			if X is not None:
				rA,cA,vA,pb = X
				sprows.extend([ri + Nrows for ri in rA])
				spcols.extend(cA)
				spvals.extend([vi / colscale[ci] for vi,ci in zip(vA,cA)])
				oldnrows = Nrows
				nr = listmax(rA, -1) + 1
				Nrows += nr
				logverb('Nrows was %i, added %i rows of priors => %i' % (oldnrows, nr, Nrows))
				#print 'cA', cA
				#print 'max', np.max(cA)
				#print 'max', np.max(cA)+1
				#print 'Ncols', Ncols
				Ncols = max(Ncols, listmax(cA, -1) + 1)
				b = np.zeros(Nrows)
				b[oldnrows:] = np.hstack(pb)

		if len(spcols) == 0:
			logverb("len(spcols) == 0")
			return []

		sprows = np.hstack(sprows) # hogg's lovin' hstack *again* here
		spcols = np.hstack(spcols)
		spvals = np.hstack(spvals)
		assert(len(sprows) == len(spcols))
		assert(len(sprows) == len(spvals))

		logverb('  Number of sparse matrix elements:', len(sprows))
		urows = np.unique(sprows)
		ucols = np.unique(spcols)
		logverb('  Unique rows (pixels):', len(urows))
		logverb('  Unique columns (params):', len(ucols))
		if len(urows) == 0 or len(ucols) == 0:
			return []
		logverb('  Max row:', urows[-1])
		logverb('  Max column:', ucols[-1])
		logverb('  Sparsity factor (possible elements / filled elements):', float(len(urows) * len(ucols)) / float(len(sprows)))

		assert(np.all(np.isfinite(spvals)))

		# FIXME -- does it make LSQR faster if we remap the row and column
		# indices so that no rows/cols are empty?

		# FIXME -- we could probably construct the CSC matrix ourselves!

		# Build sparse matrix
		#A = csc_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))
		A = csr_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))

		# b = chi
		#
		# FIXME -- we could be much smarter here about computing
		# just the regions we need!
		#
		if b is None:
			b = np.zeros(Nrows)

		chimap = {}
		if chiImages is not None:
			for img,chi in zip(self.getImages(), chiImages):
				chimap[img] = chi
				
		# iterating this way avoids setting the elements more than once
		for img,row0 in imgoffs.items():
			chi = chimap.get(img, None)
			if chi is None:
				#print 'computing chi image'
				chi = self.getChiImage(img=img)
			chi = chi.ravel()
			NP = len(chi)
			# we haven't touched these pix before
			assert(np.all(b[row0 : row0 + NP] == 0))
			assert(np.all(np.isfinite(chi)))
			#print 'Setting [%i:%i) from chi img' % (row0, row0+NP)
			b[row0 : row0 + NP] = chi

		###### Zero out unused rows -- FIXME, is this useful??
		# print 'Nrows', Nrows, 'vs len(urows)', len(urows)
		# bnz = np.zeros(Nrows)
		# bnz[urows] = b[urows]
		# print 'b', len(b), 'vs bnz', len(bnz)
		# b = bnz
		assert(np.all(np.isfinite(b)))

		lsqropts = dict(show=isverbose(), damp=damp)
		if variance:
			lsqropts.update(calc_var=True)

		# lsqr can trigger floating-point errors
		#np.seterr(all='warn')
		
		# Run lsqr()
		logmsg('LSQR: %i cols (%i unique), %i elements' %
			   (Ncols, len(ucols), len(spvals)-1))

		# print 'A matrix:'
		# print A.todense()
		# print
		# print 'vector b:'
		# print b
		
		t0 = time.clock()
		(X, istop, niters, r1norm, r2norm, anorm, acond,
		 arnorm, xnorm, var) = lsqr(A, b, **lsqropts)
		t1 = time.clock()
		logmsg('  %.1f seconds' % (t1-t0))

		# print 'LSQR results:'
		# print '  istop =', istop
		# print '  niters =', niters
		# print '  r1norm =', r1norm
		# print '  r2norm =', r2norm
		# print '  anorm =', anorm
		# print '  acord =', acond
		# print '  arnorm =', arnorm
		# print '  xnorm =', xnorm
		# print '  var =', var

		#olderr = set_fp_err()
		
		logverb('scaled	 X=', X)
		X = np.array(X)
		if scale_columns:
			X /= colscales
		logverb('  X=', X)

		#np.seterr(**olderr)
		#print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]

		if variance:
			if scale_columns:
				### CHECK!!
				print 'Warning: scale_columns and variance: CHECK THIS'
				var /= colscales**2
			return X,var

		return X

	def changeInvvar(self, Q2=None):
		'''
		run one iteration of iteratively reweighting the invvars for IRLS
		'''
		if Q2 is None:
			return
		assert(Q2 > 0.5)
		for img in self.getImages():
			resid = img.getImage() - self.getModelImage(img)
			oinvvar = img.getOrigInvvar()
			smask = img.getStarMask()
			chi2 = oinvvar * resid**2
			factor = Q2 / (Q2 + chi2)
			img.setInvvar(oinvvar * factor * smask)

	def getModelPatchNoCache(self, img, src, **kwargs):
		return src.getModelPatch(img, **kwargs)

	def getModelPatch(self, img, src, minsb=0., **kwargs):
		# print
		# print 'getModelPatch.'
		# print 'src', src
		# print '-> hashkey', src.hashkey()
		# print 'img', img
		# print '-> hashkey', img.hashkey()
		# print
		deps = (img.hashkey(), src.hashkey())
		deps = hash(deps)
		mv,mod = self.cache.get(deps, (0.,None))
		if mv > minsb:
			mod = None
		if mod is not None:
			#logverb('	Cache hit for model patch: image ' + str(img) +
			#		', source ' + str(src))
			#logverb('	image hashkey ' + str(img.hashkey()))
			#logverb('	source hashkey ' + str(src.hashkey()))
			pass
		else:
			#logverb('	Cache miss for model patch: image ' + str(img) +
			#		', source ' + str(src))
			#logverb('	image hashkey ' + str(img.hashkey()))
			#logverb('	source hashkey ' + str(src.hashkey()))
			mod = self.getModelPatchNoCache(img, src, minsb=minsb, **kwargs)
			#print 'Caching model image'
			self.cache.put(deps, (minsb,mod))
		return mod

	#def getModelImageNoCache(self, img, srcs=None, sky=True):
	def getModelImage(self, img, srcs=None, sky=True, minsb=0.):
		'''
		Create a model image for the given "tractor image", including
		the sky level.	If "srcs" is specified (a list of sources),
		then only those sources will be rendered into the image.
		Otherwise, the whole catalog will be.
		'''
		if _isint(img):
			img = self.getImage(img)
		#print 'getModelImage: for image', img
		mod = np.zeros(img.getShape(), self.modtype)
		if sky:
			img.sky.addTo(mod)
		if srcs is None:
			srcs = self.catalog
		for src in srcs:
			patch = self.getModelPatch(img, src, minsb=minsb)
			if patch is None:
				#print 'None patch: src is', src
				#print 'position is', img.getWcs().positionToPixel(src.pos, src)
				continue
			patch.addTo(mod)
		return mod

	#def getModelImage(self, img, srcs=None, sky=True):
	#	return self.getModelImageNoCache(img, srcs=srcs, sky=sky)
	'''
	def getModelImage(self, img):
		# dependencies of this model image:
		deps = (img.hashkey(), self.catalog.hashkey())
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
	'''

	def getOverlappingSources(self, img, srcs=None, minsb=0.):

		if _isint(img):
			img = self.getImage(img)
		mod = np.zeros(img.getShape(), self.modtype)
		if srcs is None:
			srcs = self.catalog
		patches = []
		for src in srcs:
			patch = self.getModelPatch(img, src, minsb=minsb)
			#print 'Source', src
			#print '--> patch', patch
			patches.append(patch)
			if patch is not None:
				#print '    sum', patch.patch.sum()
				#print '    max', patch.patch.max()
				#print '    min > 0', patch.patch[patch.patch > 0].min()
				assert(np.all(patch.patch >= 0))
				patch.addTo(mod)

		#mod = self.getModelImage(img, srcs=srcs, minsb=minsb, sky=False)
		bigmod = (mod > minsb)
		# print 'Mod: nonzero pixels', len(np.flatnonzero(mod))
		# print 'BigMod: nonzero pixels', len(np.flatnonzero(bigmod))
		bigmod = binary_dilation(bigmod)
		L,n = label(bigmod, structure=np.ones((3,3), int))
		#L,n = label(mod > minsb, structure=np.ones((5,5), int))
		#print 'Found', n, 'groups of sources'
		assert(L.shape == mod.shape)

		srcgroups = {}
		#equivgroups = {}
		H,W = mod.shape
		for i,modpatch in enumerate(patches):
			#for i,src in enumerate(srcs):
			# modpatch = self.getModelPatch(img, src, minsb=minsb)
			#print 'modpatch', modpatch
			if modpatch is None or not modpatch.clipTo(W,H):
				# no overlap with image
				continue
			#print 'modpatch', modpatch
			lpatch = L[modpatch.getSlice(mod)]
			#print 'mp', modpatch.shape
			#print 'lpatch', lpatch.shape
			assert(lpatch.shape == modpatch.shape)
			ll = np.unique(lpatch[modpatch.patch > minsb])
			#print 'labels:', ll, 'for source', src
			# Not sure how this happens, but it does...
			#ll = [l for l in ll if l > 0]
			if len(ll) == 0:
				# this sources contributes very little!
				continue

			# if len(ll) != 1:
			# 	import pylab as plt
			# 	plt.clf()
			# 	plt.subplot(2,2,1)
			# 	plt.imshow(lpatch, origin='lower', interpolation='nearest')
			# 	plt.subplot(2,2,2)
			# 	plt.imshow(modpatch.patch, origin='lower', interpolation='nearest')
			# 	plt.subplot(2,2,3)
			# 	plt.imshow(modpatch.patch > minsb, origin='lower', interpolation='nearest')
			# 	plt.subplot(2,2,4)
			# 	plt.imshow(lpatch * (modpatch.patch > minsb), origin='lower', interpolation='nearest')
			# 	plt.savefig('ll.png')
			# 	sys.exit(-1)

			if len(ll) > 1:
				# This can happen when a source has two peaks (eg, a PSF)
				# that are separated by a pixel and the saddle is below minval.
				pass
   			assert(len(ll) == 1)
			ll = ll[0]
			if not ll in srcgroups:
				srcgroups[ll] = []
			srcgroups[ll].append(i)
		#return srcgroups.values() #, L
		return srcgroups, L, mod

	def getModelImages(self):
		if self.is_multiproc():
			# avoid shipping my images...
			allimages = self.getImages()
			self.images = Images()
			args = [(self, im) for im in allimages]
			#print 'Calling _map:', getmodelimagefunc2
			#print 'args:', args
			mods = self._map(getmodelimagefunc2, args)
			self.images = allimages
		else:
			mods = [self.getModelImage(img) for img in self.images]
		return mods

	def clearCache(self):
		self.cache.clear() # = Cache()

	def getChiImages(self):
		mods = self.getModelImages()
		chis = []
		for img,mod in zip(self.images, mods):
			chis.append((img.getImage() - mod) * img.getInvError())
		return chis

	def getChiImage(self, imgi=-1, img=None, srcs=None, minsb=0.):
		if img is None:
			img = self.getImage(imgi)
		mod = self.getModelImage(img, srcs, minsb=minsb)
		return (img.getImage() - mod) * img.getInvError()

	def getNdata(self):
		count = 0
		for img in self.images:
			InvError = img.getInvError()
			count += len(np.ravel(InvError > 0.0))
		return count

	def getLogLikelihood(self):
		chisq = 0.
		for i,chi in enumerate(self.getChiImages()):
			chisq += (chi.astype(float) ** 2).sum()
		return -0.5 * chisq

	def getLogProb(self):
		'''
		return the posterior PDF, evaluated at the parametrs
		'''
		return self.getLogLikelihood() + self.getLogPrior()

	def getBbox(self, img, srcs):
		nzsum = None
		# find bbox
		for src in srcs:
		 	p = self.getModelPatch(img, src)
		 	if p is None:
		 		continue
		 	nz = p.getNonZeroMask()
		 	nz.patch = nz.patch.astype(np.int)
		 	if nzsum is None:
		 		nzsum = nz
		 	else:
		 		nzsum += nz
			# ie = tim.getInvError()
			# p2 = np.zeros_like(ie)
			# p.addTo(p2)
			# effect = np.sum(p2)
			# print 'Source:', src
			# print 'Total chi contribution:', effect, 'sigma'
		nzsum.trimToNonZero()
		roi = nzsum.getExtent()
		return roi
	
	def createNewSource(self, img, x, y, height):
		return None

	def debugNewSource(self, *args, **kwargs):
		pass

	def createSource(self, nbatch=1, imgi=None, jointopt=False,
					 avoidExisting=True):
		logverb('createSource')
		'''
		-synthesize images
		-look for "promising" x,y image locations with positive residuals
		- (not near existing sources)
		---chi image, PSF smooth, propose positions?
		-instantiate new source (Position, brightness)
		-local optimizeAtFixedComplexity
		'''
		if imgi is None:
			imgi = range(self.getNImages())
		
		for i in imgi:
			for b in range(nbatch):
				chi = self.getChiImage(i)
				img = self.getImage(i)

				if avoidExisting:
					# block out regions around existing Sources.
					for j,src in enumerate(self.catalog):
						patch = self.getModelPatch(img, src)
						(H,W) = img.shape
						if not patch.clipTo(W, H):
							continue
						chi[patch.getSlice()] = 0.

				# PSF-correlate
				sm = img.getPsf().applyTo(chi)
				debugargs = dict(imgi=i, img=img, chiimg=chi, smoothed=sm)
				self.debugNewSource(type='chi-smoothed', **debugargs)

				# Try to create sources in the highest-valued pixels.
				# FIXME -- should do peak-finding (ie, non-maximal rejection)
				II = np.argsort(-sm.ravel())
				# MAGIC: number of pixels to try.
				for ii,I in enumerate(II[:10]):
					(H,W) = sm.shape
					ix = I%W
					iy = I/W
					# this is just the peak pixel height difference...
					ht = (img.getImage() - self.getModelImage(img))[iy,ix]
					logverb('Requesting new source at x,y', (ix,iy))
					src = self.createNewSource(img, ix, iy, ht)
					logverb('Got:', src)
					debugargs['src'] = src
					self.debugNewSource(type='newsrc-0', **debugargs)
					# try adding the new source...
					pBefore = self.getLogProb()
					logverb('log-prob before:', pBefore)
					if jointopt:
						oldcat = self.catalog.deepcopy()

					self.catalog.append(src)

					# individually optimizing the newly-added
					# source...
					for ostep in range(20):
						logverb('Optimizing the new source (step %i)...' % (ostep+1))
						dlnprob,X,alpha = self.optimizeCatalogAtFixedComplexityStep(srcs=[src])
						logverb('After:', src)
						self.debugNewSource(type='newsrc-opt', step=ostep, dlnprob=dlnprob,
											**debugargs)
						if dlnprob < 1.:
							logverb('failed to improve the new source enough (d lnprob = %g)' % dlnprob)
							break

					# Try changing the newly-added source type?
					# print 'Trying to change the source type of the newly-added source'
					# self.changeSourceTypes(srcs=[src])
					
					if jointopt:
						# then the whole catalog
						logverb('Optimizing the catalog with the new source...')
						self.optimizeCatalogAtFixedComplexityStep()

					pAfter = self.getLogProb()
					logverb('delta log-prob:', (pAfter - pBefore))

					if pAfter > pBefore:
						logverb('Keeping new source')
						break

					else:
						logverb('Rejecting new source')
						# revert the catalog
						if jointopt:
							self.catalog = oldcat
						else:
							self.catalog.pop()


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
		print 'After  tuning:					 log-prob', pAfter2

		# HACKY: want to be better, and to have successfully optimized...
		if pAfter2 > pAfter+1. and pAfter2 > pBefore+2.:
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

	def debugChangeSources(self, **kwargs):
		pass

	def changeSourceTypes(self, srcs=None, jointopt=False):
		'''
		Returns a list of booleans of length "srcs": whether the
		sources were changed or not.
		'''
		logverb('changeSourceTypes')
		pBefore = self.getLogProb()
		logverb('log-prob before:', pBefore)

		didchange = []

		oldcat = self.catalog
		ncat = len(oldcat)

		# We can't just loop over "srcs" -- because when we accept a
		# change, the catalog changes!
		# FIXME -- with this structure, we try to change new sources that
		# we have just added.
		i = -1
		ii = -1
		while True:
			i += 1
			logverb('changeSourceTypes: source', i)
			self.catalog = oldcat

			if srcs is None:
				# go through self.catalog using "ii" as the index.
				# (which is updated within the loop when self.catalog is mutated)
				ii += 1
				if ii >= len(self.catalog):
					break
				if ii >= ncat:
					break
				logverb('  changing source index', ii)
				src = self.catalog[ii]
				logmsg('Considering change to source:', src)
			else:
				if i >= len(srcs):
					break
				src = srcs[i]

			# Prevent too-easy switches due to the current source not being optimized.
			pBefore = self.getLogProb()
			logverb('Optimizing source before trying to change it...')
			self.optimizeCatalogLoop(srcs=[src], sky=False)
			logmsg('After optimizing source:', src)
			pAfter = self.getLogProb()
			logverb('delta-log-prob:', pAfter - pBefore)
			pBefore = pAfter

			bestlogprob = pBefore
			bestalt = -1
			bestparams = None

			alts = self.changeSource(src)
			self.debugChangeSources(step='start', src=src, alts=alts)
			srcind = oldcat.index(src)
			for j,newsrcs in enumerate(alts):
				newcat = oldcat.deepcopy()
				rsrc = newcat.pop(srcind)
				newcat.extend(newsrcs)
				logverb('Trying change:')
				logverb('  from', src)
				logverb('  to  ', newsrcs)
				self.catalog = newcat

				self.debugChangeSources(step='init', src=src, newsrcs=newsrcs, alti=j)

				# first try individually optimizing the newly-added
				# sources...
				self.optimizeCatalogLoop(srcs=newsrcs, sky=False)
				logverb('After optimizing new sources:')
				for ns in newsrcs:
					logverb('  ', ns)
				self.debugChangeSources(step='opt0', src=src, newsrcs=newsrcs, alti=j)
				if jointopt:
					self.optimizeCatalogAtFixedComplexityStep(sky=False)

				pAfter = self.getLogProb()
				logverb('delta-log-prob:', pAfter - pBefore)

				self.debugChangeSources(step='opt1', src=src, newsrcs=newsrcs, alti=j, dlnprob=pAfter-pBefore)

				if pAfter > bestlogprob:
					logverb('Best change so far!')
					bestlogprob = pAfter
					bestalt = j
					bestparams = newcat.getParams()

			if bestparams is not None:
				#print 'Switching to new catalog!'
				# We want to update "oldcat" in-place (rather than
				# setting "self.catalog = bestcat") so that the source
				# object identities don't change -- so that the outer
				# loop "for src in self.catalog" still works.  We need
				# to updated the structure and params.
				oldcat.remove(src)
				ii -= 1
				ncat -= 1
				oldcat.extend(alts[bestalt])
				oldcat.setAllParams(bestparams)
				self.catalog = oldcat
				pBefore = bestlogprob

				logmsg('')
				logmsg('Accepted change:')
				logmsg('from:', src)
				if len(alts[bestalt]) == 1:
					logmsg('to:', alts[bestalt][0])
				else:
					logmsg('to:', alts[bestalt])

				assert(self.getLogProb() == pBefore)
				self.debugChangeSources(step='switch', src=src, newsrcs=alts[bestalt], alti=bestalt, dlnprob=bestlogprob)
				didchange.append(True)
			else:
				self.debugChangeSources(step='keep', src=src)
				didchange.append(False)

		self.catalog = oldcat
		return didchange

