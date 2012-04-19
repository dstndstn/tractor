# Copyright 2011, 2012 Dustin Lang and David W. Hogg.  All rights reserved.

# to-do:
# ------
# -write the paper!
# -release DR9.1

from math import ceil, floor, pi, sqrt, exp
import time
import logging
import random
import os

import numpy as np
import pylab as plt

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from astrometry.util.miscutils import get_overlapping_region
from astrometry.util.multiproc import *
from .utils import MultiParams
from .cache import *
from .ttime import Time

FACTOR = 1.e-10

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
	return np.seterr(all='raise')

def randomint():
	return int(random.random() * (2**32)) #(2**48))

class Image(MultiParams):
	'''
	An image plus its calibration information.	The Tractor handles
	multiple Images.
	'''
	def __init__(self, **kwargs):
		'''
		Expected kwargs:
		(data=None, invvar=None,
		 psf=None, wcs=None, sky=None, photocal=None,
		 name=None)
		
		'''
		self.data = kwargs.pop('data', None)
		self.invvar = kwargs.pop('invvar', None)
		for i,x in enumerate(self.invvar):
			for j,y in enumerate(x):
				if y < 0:
					self.invvar[i][j] = 0
		self.inverr = np.sqrt(self.invvar)
		self.origInvvar = self.invvar

		self.name = kwargs.pop('name', None)

		psf = kwargs.pop('psf', None)
		sky = kwargs.pop('sky', None)
		wcs = kwargs.pop('wcs', None)
		photocal = kwargs.pop('photocal', None)
		super(Image,self).__init__(psf, wcs, photocal, sky)
		#print 'Image:', self.wcs, self.psf, self.sky, self.photocal

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
		self.invvar = invvar
		for i,x in enumerate(self.invvar):
			for j,y in enumerate(x):
				if not np.isfinite(y):
					self.invvar[i][j] = 0
		self.inverr = np.sqrt(invvar)
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
	An image + pixel offset
	'''
	def __init__(self, x0, y0, patch):
		self.x0 = x0
		self.y0 = y0
		self.patch = patch
		self.name = ''

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
	def size(self):
		(H,W) = self.patch.shape
		return H*W
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
	deepcopy = MultiParams.copy

	def __str__(self):
		#self.printLong()
		#return 'Catalog with %i sources' % len(self)
		return 'Catalog: %i sources, %i parameters' % (len(self), self.numberOfParams())

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
	@staticmethod
	def getNamedParams():
		return dict(images=0, catalog=1)

	def __init__(self, images=[], catalog=[], mp=None):
		'''
		image: list of Image objects (data)
		catalog: list of Source objects
		'''
		super(Tractor,self).__init__(Images(*images), Catalog(*catalog))
		self.cache = Cache()
		self.cachestack = []
		if mp is None:
			mp = multiproc()
		self.mp = mp

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
		return lnp

	# For pickling
	def __getstate__(self):
		return (self.getImages(), self.getCatalog(), self.liquid)
	def __setstate__(self, state):
		(images, catalog, liquid) = state
		self.subs = [images, catalog]
		self.liquid = liquid
		self.mp = multiproc()
		self.cache = Cache()
		self.cachestack = []

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

	def optimizeCatalogLoop(self, nsteps=20, **kwargs):
		mindlnprob = kwargs.pop('mindlnprob', 1.e-3)
		for ostep in range(nsteps):
			logmsg('Optimizing sources (step %i)...' % (ostep+1))
			dlnprob,X,alpha = self.optimizeCatalogAtFixedComplexityStep(**kwargs)
			logverb('delta-log-prob', dlnprob)
			if 'srcs' in kwargs:
				srcs = kwargs['srcs']
				if len(srcs) == 1:
					logmsg('-> ', srcs[0])
				else:
					logmsg('-> ', srcs)
			if dlnprob <= mindlnprob:
				logverb('converged to tolerance %g (d lnprob = %g)' % (mindlnprob, dlnprob))
				return False
		return True

	def optimizeCatalogAtFixedComplexityStep(self, srcs=None, brightnessonly=False,
											 alphas=None, sky=True):
		'''
		Returns: (delta-log-prob, delta-parameters, step size alpha)

		-synthesize images
		-get all derivatives
		-build matrix
		-take step (try full step, back off)
		'''
		logverb('Optimizing at fixed complexity')
		alldevirs = self.getAllDerivs(srcs=srcs, brightnessonly=brightnessonly, sky=sky)
		X = self.optimize(alldevirs)
		(dlogprob, alpha) = self.tryParamUpdates(srcs, X, alphas)
		return dlogprob, X, alpha


	### FIXME -- temporary functions, parallel to optimizeCatalogAtFixedComplexityStep()
	### et al, that use the param infrastructure correctly.
	def opt2(self, alphas=None):
		print 'opt2: Finding derivs...'
		t0 = Time()
		allderivs = self.getderivs2()
		print Time() - t0
		#print 'allderivs:', allderivs
		#for d in allderivs:
		#	for (p,im) in d:
		#		print 'patch mean', np.mean(p.patch)
		print 'Finding optimal update direction...'
		t0 = Time()
		X = self.optimize(allderivs)
		print Time() - t0
		#print 'X:', X
		if len(X) == 0:
			return 0, X, 0.
		print 'Finding optimal step size...'
		t0 = Time()
		(dlogprob, alpha) = self.tryupdates2(X, alphas=alphas)
		print Time() - t0
		print 'Finished opt2.'
		return dlogprob, X, alpha

	def tryupdates2(self, X, alphas=None):
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


	def getderivs2(self):
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
			imjs = [i for i in self.getThawedParamIndices()]
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
			print len(srcs), 'thawed sources'
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

	def optimize(self, alldevirs):

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
		for param in alldevirs:
			for deriv,img in param:
				if img in imgoffs:
					continue
				imgoffs[img] = nextrow
				#print 'Putting image', img.name, 'at row offset', nextrow
				nextrow += img.numberOfPixels()
		Nrows = nextrow
		del nextrow
		Ncols = len(alldevirs)

		colscales = []
		for col, param in enumerate(alldevirs):
			RR = []
			CC = []
			VV = []
			WW = []
			for (deriv, img) in param:
				inverrs = img.getInvError()
				(H,W) = img.shape
				row0 = imgoffs[img]
				#print 'Image', img.name, 'has row0=', row0
				#print 'Before clipping:'
				#print 'deriv shape is', deriv.shape
				#print 'deriv slice is', deriv.getSlice()
				deriv.clipTo(W, H)
				pix = deriv.getPixelIndices(img)
				# (in the parent image)
				#print 'After clipping:'
				#print 'deriv shape is', deriv.shape
				#print 'deriv slice is', deriv.getSlice()
				#print 'image shape is', img.shape
				#print 'parent pix', (W*H), npixels[i]
				#print 'pix range:', pix.min(), pix.max()
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
				cols = np.zeros_like(rows) + col
				vals = dimg.ravel()[nz]
				w = inverrs[deriv.getSlice(img)].ravel()[nz]
				assert(vals.shape == w.shape)
				RR.append(rows)
				CC.append(cols)
				VV.append(vals)
				WW.append(w)

			# massage, re-scale, and clean up matrix elements
			if len(VV) == 0:
				colscales.append(1.)
				continue
			rows = np.hstack(RR)
			cols = np.hstack(CC)
			vals = np.hstack(VV) * np.hstack(WW)
			if len(vals) == 0:
				colscales.append(1.)
				continue
			mx = np.max(np.abs(vals))
			if mx == 0:
				print 'mx == 0'
				colscales.append(1.)
				continue
			#print 'mx=', mx
			I = (np.abs(vals) > (FACTOR * mx))
			rows = rows[I]
			cols = cols[I]
			vals = vals[I]
			scale = np.sqrt(np.sum(vals * vals))
			colscales.append(scale)
			assert(len(colscales) == (col+1))
			logverb('Column', col, 'scale:', scale)
			sprows.append(rows)
			spcols.append(cols)
			spvals.append(vals / scale)

		if len(spcols) == 0:
			logverb("len(spcols) == 0")
			return []

		# HACK
		uniquecols = np.unique(np.hstack(spcols))

		# ensure the sparse matrix is full up to the number of columns we expect
		# dstn: what does this do? -hogg
		# hogg: it is a hack. -dstn
		spcols.append([Ncols - 1])
		sprows.append([0])
		spvals.append([0])

		sprows = np.hstack(sprows) # hogg's lovin' hstack *again* here
		spcols = np.hstack(spcols)
		spvals = np.hstack(spvals)
		assert(len(sprows) == len(spcols))
		assert(len(sprows) == len(spvals))

		logverb('  Number of sparse matrix elements:', len(sprows))
		urows = np.unique(sprows)
		logverb('  Unique rows (pixels):', len(urows))
		logverb('  Max row:', max(urows))
		ucols = np.unique(spcols)
		logverb('  Unique columns (params):', len(ucols))
		logverb('  Max column:', max(ucols))
		logverb('  Sparsity factor (possible elements / filled elements):', float(len(urows) * len(ucols)) / float(len(sprows)))

		assert(all(np.isfinite(spvals)))

		# Build sparse matrix
		A = csr_matrix((spvals, (sprows, spcols)))

		# b = chi
		#
		# FIXME -- we could be much smarter here about computing
		# just the regions we need!
		#
		b = np.zeros(Nrows)
		# iterating this way avoid setting the elements more than once
		for img,row0 in imgoffs.items():
			NP = img.numberOfPixels()
			mod = self.getModelImage(img)
			data = img.getImage()
			inverr = img.getInvError()
			assert(np.product(mod.shape) == NP)
			assert(mod.shape == data.shape)
			assert(mod.shape == inverr.shape)
			# we haven't touched these pix before
			assert(all(b[row0 : row0 + NP] == 0))
			assert(all(np.isfinite(data.ravel())))
			assert(all(np.isfinite(mod.ravel())))
			assert(all(np.isfinite(inverr.ravel())))
			b[row0 : row0 + NP] = ((data - mod) * inverr).ravel()
			assert(all(np.isfinite(b[row0 : row0 + NP])))
		b = b[:urows.max() + 1]

		assert(all(np.isfinite(b)))

		# FIXME -- does it make LSQR faster if we remap the row and column
		# indices so that no rows/cols are empty?

		lsqropts = dict(show=isverbose())

		# lsqr can trigger floating-point errors
		np.seterr(all='warn')
		
		# Run lsqr()
		logmsg('LSQR: %i cols (%i unique), %i elements' %
			   (Ncols, len(uniquecols), len(spvals)-1))
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
		return X

	# Hmm, does this make you think Catalog should be a MultiParams?
	def stepParams(self, X, srcs=None, alpha=1.):
		if srcs is None:
			srcs = self.catalog
		oldparams = []
		par0 = 0
		for j,src in enumerate(srcs):
			npar = src.numberOfParams()
			dparams = X[par0 : par0 + npar]
			par0 += npar
			#assert(len(dparams) == src.numberOfParams())
			pars = src.getParams()
			oldparams.append(pars)
			#src.stepParams(dparams * alpha)
			src.setParams(np.array(pars) + dparams * alpha)
		return oldparams
	def revertParams(self, oldparams, srcs=None):
		if srcs is None:
			srcs = self.catalog
		assert(len(srcs) == len(oldparams))
		for j,src in enumerate(srcs):
			src.setParams(oldparams[j])

	# X: delta-params
	#
	# Returns: delta-logprob, alphaBest
	def tryParamUpdates(self, srcs, X, alphas=None):
		if alphas is None:
			# 1/1024 to 1 in factors of 2
			alphas = 2.**-(np.arange(10,0,-1)-1)

		pBefore = self.getLogProb()
		logverb('  log-prob before:', pBefore)

		pBest = pBefore
		alphaBest = None

		for alpha in alphas:
			logverb('  Stepping with alpha =', alpha)
			oldparams = self.stepParams(X, srcs, alpha)
			pAfter = self.getLogProb()
			logverb('  delta log-prob:', pAfter - pBefore)
			self.revertParams(oldparams, srcs)
			# want to improve over last step.
			# (allow some margin though)
			if pAfter < (pBest - 1.):
				break
			if pAfter > pBest:
				alphaBest = alpha
				pBest = pAfter
		if alphaBest is None:
			return 0, 0.
		logmsg('  Stepping by', alphaBest, 'for delta-logprob', pBest - pBefore)
		self.stepParams(X, srcs, alphaBest)
		return pBest - pBefore, alphaBest

	def getAllDerivs(self, srcs=None, brightnessonly=False, sky=True):
		'''
		Returns a list of pairs, D[parami] = (deriv,image)
		'''
		if srcs is None:
			srcs = self.catalog
		alldevirs = []
		for j,src in enumerate(srcs):
			allderivs = [[] for i in range(src.numberOfParams())]
			for i,img in enumerate(self.images):
				# Get derivatives (in this image) of params
				derivs = src.getParamDerivatives(img, brightnessonly=brightnessonly)
				assert(len(derivs) == src.numberOfParams())
				for k,deriv in enumerate(derivs):
					if deriv is None:
						continue
					if not all(np.isfinite(deriv.patch.ravel())):
						print 'Derivative for source', src
						print 'deriv index', i
						assert(False)
					allderivs[k].append((deriv, img))
			alldevirs.extend(allderivs)
		if sky:
			for i,img in enumerate(self.getImages()):
				derivs = img.getSky().getParamDerivatives(img)
				alldevirs.extend([[(d,img) for d in derivs]])
		return alldevirs

	def changeInvvar(self,IWLSscale=None):
		if IWLSscale is None:
			return
		for img in self.getImages():
			data = img.getImage()
			mod = self.getModelImage(img)
			chi = data-mod
			invvar = img.getInvvar()
			img.setInvvar((1./invvar)*((IWLSscale*(chi**2))/(IWLSscale+(chi)**2)))
	
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
			self.cache[deps] = mod
		return mod

	def getModelImageNoCache(self, img, srcs=None):
		'''
		Create a model image for the given "tractor image", including
		the sky level.	If "srcs" is specified (a list of sources),
		then only those sources will be rendered into the image.
		Otherwise, the whole catalog will be.
		'''
		if type(img) is int:
			img = self.getImage(img)
		#mod = np.zeros_like(img.getImage())
		# FIXME -- specify type?? np.float32?
		mod = np.zeros(img.getShape())
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

	def getModelImage(self, img, srcs=None):
		return self.getModelImageNoCache(img, srcs)
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
		self.cache = Cache()

	def getChiImages(self):
		mods = self.getModelImages()
		chis = []
		for img,mod in zip(self.images, mods):
			chis.append((img.getImage() - mod) * img.getInvError())
		return chis

	def getChiImage(self, imgi,srcs=None):
		img = self.getImage(imgi)
		mod = self.getModelImage(img,srcs)
		return (img.getImage() - mod) * img.getInvError()

	def getLogLikelihood(self):
		chisq = 0.
		for i,chi in enumerate(self.getChiImages()):
			chisq += (chi.astype(float) ** 2).sum()
		return -0.5 * chisq

	def getLogPrior(self):
		return -len(self.catalog.getParams())

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

	def optimizeCatalogBrightnesses(self, srcs=None, sky=False):
		return self.optimizeCatalogAtFixedComplexityStep(srcs=srcs, brightnessonly=True,
														 sky=sky)

	def optimizeSkyAtFixedComplexityStep(self, imagei):
		logmsg('Optimize sky at fixed complexity')
		img = self.getImage(imagei)
		sky = img.getSky()
		derivs = sky.getParamDerivatives(img)
		allparams = [[(deriv,img)] for deriv in derivs]
		X = self.optimize(allparams)
		logmsg('Sky paramater changes:', X)
		logmsg('Before:', sky)
		dlnp,alpha = self.tryParamUpdates([sky], X)
		logmsg('After:', sky)
		logverb('Log-prob improvement:', dlnp)
		return dlnp, X, alpha

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
		p0 = psf.getParams()
		for k,s in enumerate(steps):
			if True:

				#psfk = psf.copy()
				#psfk.stepParam(k, s)
				oldval = psf.setParam(k, p0[k]+s)
				#print '  step param', k, 'by', s, 'to get', psfk
				#img.setPsf(psfk)
				modk = self.getModelImage(img)
				psf.setParam(k, oldval)
				# to reuse code, wrap this in a Patch...
				dk = Patch(0, 0, (modk - mod0) / s)
			else:
				# symmetric finite differences
				# psfk1 = psf.copy()
				# psfk2 = psf.copy()
				# psfk1.stepParam(k, -s)
				# psfk2.stepParam(k, +s)
				# print '  step param', k, 'by', -s, 'to get', psfk1
				# print '  step param', k, 'by',  s, 'to get', psfk2
				# img.setPsf(psfk1)
				# modk1 = self.getModelImage(img)
				# img.setPsf(psfk2)
				# modk2 = self.getModelImage(img)
				# dk = Patch(0, 0, (modk2 - modk1) / (s*2))
				pass

			derivs.append(dk)
		img.setPsf(psf)
		assert(len(derivs) == nparams)

		if derivCallback:
			(func, baton) = derivCallback
			func(self, imagei, img, psf, steps, mod0, derivs, baton)

		# (PSF)
		allparams = [[(deriv,img)] for deriv in derivs]
		X = self.optimize(allparams)

		print 'PSF Parameter changes:', X
		dlogprob,alpha = self.tryParamUpdates([psf], X)
		print 'After:', psf
		print 'Log-prob improvement:', dlogprob

		return dlogprob

	def optimizeAllPsfAtFixedComplexityStep(self, **kwargs):
		for i in range(len(self.images)):
			self.optimizePsfAtFixedComplexityStep(i, **kwargs)

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

