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
import pylab as plt

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.ndimage.morphology import binary_dilation

from astrometry.util.miscutils import get_overlapping_region
from astrometry.util.multiproc import *
from .utils import MultiParams
from .cache import *
from .ttime import Time

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
				 photocal=None, name=None, **kwargs):
		'''
		Args:
		  * *data*: numpy array: the image pixels
		  * *invvar*: numpy array: the image inverse-variance
		  * *psf*: a :class:`tractor.PSF` duck
		  * *wcs*: a :class:`tractor.WCS` duck
		  * *sky*: a :class:`tractor.Sky` duck
		  * *photocal*: a :class:`tractor.PhotoCal` duck
		  * *name*: string name of this image.

		'''
		self.data = data
		self.origInvvar = 1. * np.array(invvar)
		self.setMask()
		self.setInvvar(self.origInvvar)
		self.name = name
		self.starMask = np.ones_like(self.data)
		super(Image, self).__init__(psf, wcs, photocal, sky)

	def __str__(self):
		return 'Image ' + str(self.name)

	@staticmethod
	def getNamedParams():
		return dict(psf=0, wcs=1, photocal=2, sky=3)

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
		self.invvar[self.mask] = 0. 
		self.inverr = np.sqrt(self.invvar)

	def getMask(self):
		return self.mask

	def setMask(self):
		self.mask = (self.origInvvar <= 0.)
		self.mask = binary_dilation(self.mask,iterations=3)

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

	def getExtent(self):
		''' Return (x0, x1, y0, y1) '''
		(w,h) = self.shape
		return (self.x0, self.x0 + w, self.y0, self.y0 + h)

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

	def performArithmetic(self, other, opname):
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

		p = np.zeros((uy1 - uy0, ux1 - ux0))
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

	def getNamedParamName(self, j):
		return 'source%i' % j


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
	im.setParam(k, p0 + step)
	mod = tr.getModelImage(im)
	im.setParam(k, p0)
	return mod
def getmodelimagefunc((tr, imj)):
	print 'getmodelimagefunc(): imj', imj, 'pid', os.getpid()
	return tr.getModelImage(imj)
def getsrcderivs((src, img)):
	return src.getParamDerivatives(img)

def getmodelimagefunc2((tr, im)):
	#print 'getmodelimagefunc2(): im', im, 'pid', os.getpid()
	#tr.images = Images(im)
	return tr.getModelImage(im)

class Tractor(MultiParams):
	"""
	Heavy farm machinery.

	As you might guess from the name, this is the main class of the
	Tractor framework.  A Tractor has a set of Images and a set of
	Sources, and has methods to optimize the parameters of those
	Images and Sources.

	"""
	@staticmethod
	def getNamedParams():
		return dict(images=0, catalog=1)

	def __init__(self, images=[], catalog=[], mp=None):
		'''
		- `images:` list of Image objects (data)
		- `catalog:` list of Source objects
		'''
		super(Tractor,self).__init__(Images(*images), Catalog(*catalog))
		self._setup(mp=mp)

	def _setup(self, mp=None, cache=None, pickleCache=False, cachestack=[]):
		if mp is None:
			mp = multiproc()
		self.mp = mp
		self.modtype = np.float32
		if cache is None:
			cache = Cache()
		self.cache = Cache()
		self.cachestack = []
		self.pickleCache = pickleCache

	def __str__(self):
		s = 'Tractor with %i sources and %i images' % (len(self.catalog), len(self.images))
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
		print 'Tractor.__call__: I am pid', os.getpid()
		self.setParams(X)
		lnp = self.getLogProb()
		print self.cache
		from .sdss_galaxy import get_galaxy_cache
		print 'Galaxy cache:', get_galaxy_cache()
		#print 'Items:'
		#self.cache.printItems()
		#print
		return lnp

	# For pickling
	def __getstate__(self):
		#print 'pickling tractor in pid', os.getpid()
		S = (self.getImages(), self.getCatalog(), self.liquid)
		if self.pickleCache:
			S = S + (self.cache,)
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

	def optimize(self, alphas=None):
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		print 'opt2: Finding derivs...'
		t0 = Time()
		allderivs = self.getDerivs()
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		tderivs = Time()-t0
		#print Time() - t0
		#print 'allderivs:', allderivs
		#for d in allderivs:
		#	for (p,im) in d:
		#		print 'patch mean', np.mean(p.patch)
		print 'Finding optimal update direction...'
		t0 = Time()
		X = self.getUpdateDirection(allderivs)
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		#print Time() - t0
		topt = Time()-t0
		#print 'X:', X
		if len(X) == 0:
			return 0, X, 0.
		print 'Finding optimal step size...'
		t0 = Time()
		(dlogprob, alpha) = self.tryUpdates(X, alphas=alphas)
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		tstep = Time() - t0
		print 'Finished opt2.'
		print '  Tderiv', tderivs
		print '  Topt  ', topt
		print '  Tstep ', tstep
		return dlogprob, X, alpha

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
		# initial models...
		mod0s = self._map_async(getmodelimagefunc, [(self, imj) for imj in imjs])
		# stepping each param...
		args = []
		for j,im in enumerate(ims):
			p0 = im.getParams()
			for k,step in enumerate(im.getStepSizes()):
				args.append((self, j, k, p0[k], step))
		# reverse the args so we can pop() below.
		mod1s = self._map_async(getmodelimagestep, reversed(args))

		# Next, derivs for the sources.
		if self.isParamFrozen('catalog'):
			srcs = []
		else:
			srcs = list(self.catalog.getThawedSources())
			#print len(srcs), 'thawed sources'
		args = []
		for j,src in enumerate(srcs):
			for i,img in enumerate(self.images):
				args.append((src, img))
		sderivs = self._map_async(getsrcderivs, reversed(args))

		# Wait for and unpack the image derivatives...
		mod0s = mod0s.get()
		mod1s = mod1s.get()
		for i,(j,im) in enumerate(zip(imjs,ims)):
			mod0 = mod0s[i]
			p0 = im.getParams()
			for k,(nm,step) in enumerate(zip(im.getParamNames(), im.getStepSizes())):
				mod1 = mod1s.pop()
				deriv = Patch(0, 0, (mod1 - mod0) / step)
				deriv.name = 'd(im%i)/d(%s)' % (j,nm)
				allderivs.append([(deriv, im)])

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
					if not all(np.isfinite(deriv.patch.ravel())):
						print 'Derivative for source', src
						print 'deriv index', i
						assert(False)
					srcderivs[k].append((deriv, img))
			allderivs.extend(srcderivs)

		assert(len(allderivs) == self.numberOfParams())
		return allderivs

	def getUpdateDirection(self, allderivs):

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
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]


		print 'imgoffs:', imgoffs

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

				assert(all(pix < img.numberOfPixels()))
				# (grab non-zero indices)
				dimg = deriv.getImage()
				nz = np.flatnonzero(dimg)
				#print '  source', j, 'derivative', p, 'has', len(nz), 'non-zero entries'
				rows = row0 + pix[nz]
				#print 'Adding derivative', deriv.getName(), 'for image', img.name
				vals = dimg.ravel()[nz]
				w = inverrs[deriv.getSlice(img)].ravel()[nz]
				assert(vals.shape == w.shape)
				RR.append(rows)
				VV.append(vals)
				WW.append(w)

			# massage, re-scale, and clean up matrix elements
			if len(VV) == 0:
				colscales.append(1.)
				continue
			rows = np.hstack(RR)
			vals = np.hstack(VV) * np.hstack(WW)
			if len(vals) == 0:
				colscales.append(1.)
				continue
			mx = np.max(np.abs(vals))
			if mx == 0:
				print 'mx == 0'
				colscales.append(1.)
				continue
			# MAGIC number: near-zero matrix elements -> 0
			# 'mx' is the max value in this column.
			FACTOR = 1.e-10
			I = (np.abs(vals) > (FACTOR * mx))
			rows = rows[I]
			vals = vals[I]
			scale = np.sqrt(np.sum(vals * vals))
			colscales.append(scale)
			assert(len(colscales) == (col+1))
			logverb('Column', col, 'scale:', scale)
			sprows.append(rows)
			spcols.append(np.zeros_like(rows) + col)
			spvals.append(vals / scale)

		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		if len(spcols) == 0:
			logverb("len(spcols) == 0")
			return []

		sprows = np.hstack(sprows) # hogg's lovin' hstack *again* here
		spcols = np.hstack(spcols)
		spvals = np.hstack(spvals)
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		assert(len(sprows) == len(spcols))
		assert(len(sprows) == len(spvals))

		logverb('  Number of sparse matrix elements:', len(sprows))
		urows = np.unique(sprows)
		logverb('  Unique rows (pixels):', len(urows))
		logverb('  Max row:', urows[-1])
		ucols = np.unique(spcols)
		logverb('  Unique columns (params):', len(ucols))
		logverb('  Max column:', ucols[-1])
		logverb('  Sparsity factor (possible elements / filled elements):', float(len(urows) * len(ucols)) / float(len(sprows)))

		assert(all(np.isfinite(spvals)))

		# FIXME -- does it make LSQR faster if we remap the row and column
		# indices so that no rows/cols are empty?

		# FIXME -- we could probably construct the CSC matrix ourselves!

		# Build sparse matrix
		#A = csc_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))
		A = csr_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]

		# b = chi
		#
		# FIXME -- we could be much smarter here about computing
		# just the regions we need!
		#
		b = np.zeros(Nrows)
		# iterating this way avoid setting the elements more than once
		for img,row0 in imgoffs.items():
			chi = self.getChiImage(img=img).ravel()
			NP = len(chi)
			# we haven't touched these pix before
			assert(all(b[row0 : row0 + NP] == 0))
			assert(all(np.isfinite(chi)))
			b[row0 : row0 + NP] = chi

		# Zero out unused rows -- FIXME, is this useful??
		bnz = np.zeros(Nrows)
		bnz[urows] = b[urows]
		b = bnz
		assert(all(np.isfinite(b)))

		lsqropts = dict(show=isverbose())

		# lsqr can trigger floating-point errors
		np.seterr(all='warn')
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
		
		# Run lsqr()
		logmsg('LSQR: %i cols (%i unique), %i elements' %
			   (Ncols, len(ucols), len(spvals)-1))
		t0 = time.clock()
		(X, istop, niters, r1norm, r2norm, anorm, acond,
		 arnorm, xnorm, var) = lsqr(A, b, **lsqropts)
		t1 = time.clock()
		logmsg('  %.1f seconds' % (t1-t0))

		olderr = set_fp_err()
		
		logverb('scaled	 X=', X)
		X = np.array(X)
		X /= np.array(colscales)
		logverb('  X=', X)

		np.seterr(**olderr)
		print "RUsage is: ",resource.getrusage(resource.RUSAGE_SELF)[2]
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

	def getModelPatchNoCache(self, img, src):
		return src.getModelPatch(img)

	def getModelPatch(self, img, src):
		deps = (img.hashkey(), src.hashkey())
		deps = hash(deps)
		mod = self.cache.get(deps, None)
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
			mod = self.getModelPatchNoCache(img, src)
			#print 'Caching model image'
			self.cache.put(deps, mod)
		return mod

	def getModelImageNoCache(self, img, srcs=None, sky=True):
		'''
		Create a model image for the given "tractor image", including
		the sky level.	If "srcs" is specified (a list of sources),
		then only those sources will be rendered into the image.
		Otherwise, the whole catalog will be.
		'''
		if type(img) is int:
			img = self.getImage(img)
		mod = np.zeros(img.getShape(), self.modtype)
		if sky:
			img.sky.addTo(mod)
		if srcs is None:
			srcs = self.catalog
		for src in srcs:
			patch = self.getModelPatch(img, src)
			if patch is None:
				#print 'None patch: src is', src
				#print 'position is', img.getWcs().positionToPixel(src.pos, src)
				continue
			patch.addTo(mod)
		return mod

	def getModelImage(self, img, srcs=None, sky=True):
		return self.getModelImageNoCache(img, srcs=srcs, sky=sky)
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
	
	def getModelImages(self):
		if self.is_multiproc():
			# avoid shipping my images...
			allimages = self.getImages()
			self.images = []
			mods = self._map(getmodelimagefunc2, [(self, im) for im in allimages])
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

	def getChiImage(self, imgi=-1, img=None, srcs=None):
		if img is None:
			img = self.getImage(imgi)
		mod = self.getModelImage(img, srcs)
		return (img.getImage() - mod) * img.getInvError()

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

	def pushCache(self):
		self.cachestack.append(self.cache)
		self.cache = self.cache.copy()

	def mergeCache(self):
		# drop the top of the stack.
		self.cachestack.pop()

	def popCache(self):
		self.cache = self.cachestack.pop()

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

