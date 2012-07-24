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
from astrometry.util.util import Tan, tan_wcs_resample, log_init, lanczos
from astrometry.util.multiproc import multiproc
from astrometry.util.file import pickle_to_file, unpickle_from_file
from astrometry.util.plotutils import setRadecAxes
import multiprocessing
import os

class Physics(object):
	# all the following from physics.nist.gov
	# http://physics.nist.gov/cuu/Constants/Table/allascii.txt
	hh = 6.62606957e-34 # J s
	cc = 299792458. # m s^{-1}
	kk = 1.3806488e-23 # J K^{-1}

	@staticmethod
	def black_body_lambda(lam, lnT):
		"""
		Compute the black-body formula, for a given lnT.

		'lam' is wavelength in meters
		'lnT' is log-temperature in Kelvin

		Return value is in [J s^{-1} m^{-2} m^{-1} sr^{-1}],
		power radiated per square meter (area), per meter of wavelength, per steradian
		"""
		return (2. * Physics.hh * (Physics.cc ** 2) * (lam ** -5) /
				(np.exp(Physics.hh * Physics.cc / (lam * Physics.kk * np.exp(lnT))) - 1.))

	@staticmethod
	def black_body_nu(lam, lnT):
		"""
		Compute the black-body formula, for a given lnT.

		'lam' is wavelength in meters
		'lnT' is log-temperature in Kelvin

		Return value is in [J s^{-1} m^{-2} Hz^{-1} sr^{-1}],
		power radiated per square meter (area), per Hz of frequency, per steradian
		"""
		return (lam **2 / Physics.cc) * Physics.black_body_lambda(lam, lnT)

	@staticmethod
	def black_body(lam, lnT):
		return black_body_nu(lam, lnT)


class DustPhotoCal(ParamList):
	def __init__(self, lam, pixscale):
		'''
		lam: central wavelength of filter in microns

		Mjypersrperdn: photometric calibration in mega-Jansky per
		steradian per DN.

		Jansky: 1e-26 watts per square meter per hertz.

		self.cal is the calibration factor that scales SI units to
		image pixel values.  The images are in MJy/Sr, so

		ie, [Joules/s/m^2/m] * [self.cal] = [MJy/Sr]

		So self.cal has to be in units of [m s / Sr], which we get from lam^2/c
		and the pixel scale (?)

		'''
		self.lam = lam
		self.cal = 1e20
		# No (adjustable) params
		super(DustPhotoCal,self).__init__()


	# MAGIC number: wavelength scale for emissivity model
	lam0 = 1.e-4 # m

	def dustParamsToCounts(self, logsolidangle, logtemperature, emissivity):
		# see, eg, http://arxiv.org/abs/astro-ph/9902255 for (lam/lam0) ^ -beta
		beta = np.array(emissivity)
		return(
			np.exp(logsolidangle)
			* ((self.lam / DustPhotoCal.lam0) ** (-1. * beta))
			* Physics.black_body_nu(self.lam, logtemperature)
			* self.cal)
		
	def brightnessToCounts(self, brightness):
		return self.dustParamsToCounts(brightness.logsolidangle,
									   brightness.logtemperature,
									   brightness.emissivity)

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
		raise AttributeError() #name + ': no such attribute in NpArrayParams.__getattr__')

	def __getstate__(self): return self.__dict__
	def __setstate__(self, d): self.__dict__.update(d)


class DustSheet(MultiParams):
	'''
	A Source that represents a smooth sheet of dust.  The dust parameters are help in arrays
	that represent sample points on the sky; they are interpolated onto the destination image.
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

		self.Tcache = {}

		# log(T): log-normal, mean 17K sigma 20%
		self.prior_logt_mean = np.log(17.)
		self.prior_logt_std = np.log(1.2)
		# emissivity: normal
		self.prior_emis_mean = 2.
		self.prior_emis_std = 0.5

		# priors on smoothness (stdevs)
		#self.prior_logt_smooth = np.log(1.2)
		#self.prior_logsa_smooth = np.log(1.2)
		#self.prior_emis_smooth = 0.25

		# weak
		self.prior_logt_smooth = np.log(1.5)
		self.prior_logsa_smooth = np.log(1.5)
		self.prior_emis_smooth = 0.5

	def getRaDecCorners(self, margin=0):
		H,W = self.wcs.get_height(), self.wcs.get_width()
		rds = []
		for x,y in [(1.-margin,1.-margin),(W+margin,1.-margin),(W+margin,H+margin),(1.-margin,H+margin)]:
			r,d = self.wcs.pixelxy2radec(x,y)
			rds.append((r,d))
		rds.append(rds[0])
		rds = np.array(rds)
		return rds


	def getArrays(self, ravel=False): #, reshape=True):   DOI, they're already the right shape.
		logsa = self.logsolidangle.a
		logt  = self.logtemperature.a
		emis  = self.emissivity.a
		# if reshape:
		# 	shape = self.shape
		# 	logsa = logsa.reshape(shape)
		# 	logt  = logt.reshape(shape)
		# 	emis  = emis.reshape(shape)
		if ravel:
			logsa = logsa.ravel()
			logt  = logt.ravel()
			emis  = emis.ravel()
		return (logsa, logt, emis)

	def getLogPrior(self):
		logsa, logt, emis = self.getArrays()
		# harsh punishment for invalid values
		if (emis < 0.).any():
			return -1e100 * np.sum(emis < 0.)
		P = (
			-0.5 * np.sum(((logt - self.prior_logt_mean) /
						   self.prior_logt_std)**2) +

			# emissivity: normal, mean 2. sigma 0.5
			-0.5 * np.sum(((emis - self.prior_emis_mean) /
						   self.prior_emis_std) ** 2) +

			# log(surface density): free (flat prior)
			0.
			)
		# Smoothness:
		for p,smooth in [(logt,  self.prior_logt_smooth),
						 (logsa, self.prior_logsa_smooth),
						 (emis,  self.prior_emis_smooth)]:
			P += (-0.5 * np.sum(((p[1:,:] - p[:-1,:]) / smooth)**2) +
				  -0.5 * np.sum(((p[:,1:] - p[:,:-1]) / smooth)**2))

		#print 'Returning log-prior', P
		return P

	def getLogPriorChi(self):
		'''
		Returns a "chi-like" approximation to the log-prior at the
		current parameter values.

		This will go into the least-squares fitting (each term in the
		prior acts like an extra "pixel" in the fit).

		Returns (rowA, colA, valA, pb), where:

		rowA, colA, valA: describe a sparse matrix pA

		pA: has shape N x numberOfParams
		pb: has shape N

		rowA, colA, valA, and pb should be *lists* of np.arrays

		where "N" is the number of "pseudo-pixels"; "pA" will be
		appended to the least-squares "A" matrix, and "pb" will be
		appended to the least-squares "b" vector, and the
		least-squares problem is minimizing

		|| A * (delta-params) - b ||^2

		This function must take frozen-ness of parameters into account
		(this is implied by the "numberOfParams" shape requirement).
		'''

		# We have separable priors on the log-T and emissivity
		# parameters, so the number of unfrozen params will be N.

		rA = []
		cA = []
		vA = []
		pb = []

		logsa, logt, emis = self.getArrays(ravel=True)

		c0 = 0
		if not self.isParamFrozen('logsolidangle'):
			c0 += self.logsolidangle.numberOfParams()
		r0 = 0

		for pname, mn, st, arr in [('logtemperature', self.prior_logt_mean,
									self.prior_logt_std, logt),
								   ('emissivity', self.prior_emis_mean,
									self.prior_emis_std, emis)]:
			if self.isParamFrozen(pname):
				continue
			p = getattr(self, pname)
			I = np.array(list(p.getThawedParamIndices()))
			NI = len(I)
			off = np.arange(NI)
			std = st * np.ones(NI)
			rA.append(r0 + off)
			cA.append(c0 + off)
			vA.append( 1. / std )
			pb.append( -(arr[I] - mn) / std )
			c0 += NI
			r0 += NI

		# Smoothness:
		c0 = 0
		H,W = self.shape
		for pname, smooth, arr in [('logsolidangle',  self.prior_logsa_smooth, logsa),
								   ('logtemperature', self.prior_logt_smooth,  logt),
								   ('emissivity',     self.prior_emis_smooth,  emis)]:
			if self.isParamFrozen(pname):
				continue
			p = getattr(self, pname)
			I = np.array(list(p.getThawedParamIndices()))
			assert(np.all(I >= 0))
			assert(np.all(I < (W*H)))

			# grid coords of thawed parameters
			X,Y = (I % W), (I / W)
			# pixel-to-param map.
			paramindex = np.zeros(H*W, int) - 1
			paramindex[I] = np.arange(len(I))

			for KK, dp in [((X > 0), 1), ((Y > 0), W)]:
				# KK: params to include in this constraint
				# pixel indices 
				II = I[KK]
				# dp: where is the neighbouring pixel in the grid??
				JJ = II - dp
				# is pixel J thawed?
				Jind = paramindex[JJ]
				Jlive = (Jind != -1)
				#
				NI = len(II)
				offI = np.flatnonzero(KK)
				assert(NI == len(offI))
				row = np.arange(NI) + r0
				rA.append(row)
				cA.append(c0 + offI)
				sm = np.ones(NI) / smooth 
				vA.append( sm )
				pb.append( -(arr[II] - arr[JJ]) / sm )
				assert(len(rA[-1]) == len(cA[-1]))
				assert(len(rA[-1]) == len(vA[-1]))
				assert(len(rA[-1]) == len(pb[-1]))

				# include thawed pixels J too.
				rr = row[Jlive]
				rA.append(rr)
				cA.append(c0 + Jind[Jlive])
				vA.append( -np.ones(len(rr)) / smooth )
				r0 += NI
				assert(len(rA[-1]) == len(cA[-1]))
				assert(len(rA[-1]) == len(vA[-1]))

			# This is outside the loop
			c0 += len(I)

		return (rA, cA, vA, pb)

	def __getattr__(self, name):
		if name == 'shape':
			return self.logsolidangle.shape
		raise AttributeError() #name + ': no such attribute in DustSheet.__getattr__')

	def __getstate__(self): #return self.__dict__
		D = self.__dict__.copy()
		D.pop('mp', None)
		return D
	def __setstate__(self, d): self.__dict__.update(d)

	def _getcounts(self, img, I=None):
		# This takes advantage of the coincidence that our
		# DustPhotoCal does the right thing with numpy arrays.

		if I is not None:
			# class fakebright(object):
			# 	pass
			# b = fakebright()
			sa,t,e = self.getArrays(ravel=True)
			#b.logsolidangle, b.logtemperature, b.emissivity = sa[I], t[I], e[I]
			#counts = img.getPhotoCal().brightnessToCounts(b)
			counts = img.getPhotoCal().dustParamsToCounts(sa[I], t[I], e[I])
			return counts

		counts = img.getPhotoCal().brightnessToCounts(self)
		# Thanks to NpArrayParams they get ravel()'d down to 1-d, so
		# reshape back to 2d
		counts = counts.reshape(self.shape)
		return counts

	def _computeTransformation(self, img):
		imwcs = img.getWcs()
		# Pre-compute the "grid-spread function" transformation matrix...
		H,W = self.shape
		Ngrid = W*H
		iH,iW = img.shape
		Nim = iW*iH
		Lorder = 2
		S = (Lorder * 2 + 3)

		if False:
			i0 = S/2
		else:
			print 'Image', img.name
			print 'Image PSF', img.getPsf()
			psfw = img.getPsf().getRadius()
			scale = img.getWcs().pixel_scale() / self.wcs.pixel_scale()
			print 'Image pixel scale', img.getWcs().pixel_scale()
			print 'Model pixel scale', self.wcs.pixel_scale()
			print 'PSF extent', psfw, 'pixels'
			print 'pixel scale factor', scale
			print '->', psfw * scale, 'model pixels'
			S += int(np.ceil(psfw*scale * 2.))
			print 'S=', S
			i0 = S/2

		cmock = np.zeros((S,S), np.float32)
		cmock[i0,i0] = 1.

		if True:
			spsf = img.getPsf().scale(scale)
			print 'Scaled PSF', spsf
			cmock = spsf.applyTo(cmock)
			#print 'cmock'
			#print cmock

		cwcs = Tan(self.wcs)
		cwcs.set_imagesize(S, S)
		cx0,cy0 = self.wcs.crpix[0], self.wcs.crpix[1]
		rim = np.zeros((iH,iW), np.float32)

		sumr = np.zeros((iH,iW), np.float32)

		X = {}
		for i in range(H):
			print 'Precomputing matrix for image', img.name, 'row', i
			for j in range(W):
				#print 'Precomputing matrix for grid pixel', j,i
				cwcs.set_crpix(cx0 - j + i0, cy0 - i + i0)
				rim[:,:] = 0
				##
				weighted = 1
				res = tan_wcs_resample(cwcs, imwcs.wcs, cmock, rim, weighted, Lorder)
				assert(res == 0)

				if False:
					outimg = img.getPsf().applyTo(rim).ravel()
				else:
					outimg = rim.ravel()

				I = np.flatnonzero(outimg)

				if len(I) == 0:
					continue
				if True:
					sumr.ravel()[I] += outimg[I]

				xx,yy = (I % iW), (I / iW)
				x0,y0 = xx.min(), yy.min()
				nzh,nzw = 1 + yy.max() - y0, 1 + xx.max() - x0
				NZ = ((x0, y0), (nzh, nzw))
				NZI = (xx - x0) + (yy - y0) * nzw

				X[i*W+j] = (I, outimg[I], NZ, NZI)

		if True:
			print 'sumr range', sumr.min(), sumr.max()
			sumr[sumr == 0] = 1.
			mn,mx = 0.,0.
			for (I, outim, NZ, NZI) in X.values():
				outim /= sumr.ravel()[I]
				mx = max(outim.max(), mx)
				mn = max(outim.min(), mn)
			print 'Min,Max grid-spread function:', mn,mx

		return X

	def _setTransformation(self, img, X):
		key = (img.getWcs(), img.getPsf())
		self.Tcache[key] = X

	def _getTransformation(self, img):
		imwcs = img.getWcs()
		key = (imwcs,img.getPsf())
		if not key in self.Tcache:
			X = self._computeTransformation(img)
			self.Tcache[key] = X
		else:
			X = self.Tcache[key]
		return X

	def getModelPatch(self, img):
		X = self._getTransformation(img)
		counts = self._getcounts(img)
		rim = np.zeros(img.shape)
		rim1 = rim.ravel()
		for i,c in enumerate(counts.ravel()):
			if not i in X:
				continue
			I,V,nil,nil = X[i]
			rim1[I] += V * c
		#print 'Median model patch:', np.median(rim)
		return Patch(0, 0, rim)

	def getStepSizes(self, *args, **kwargs):
		return ([0.1] * len(self.logsolidangle) +
				[0.1] * len(self.logtemperature) +
				[0.1] * len(self.emissivity))


	def getParamDerivatives(self, img):
		X = self._getTransformation(img)
		imwcs = img.getWcs()
		gridscale = self.wcs.pixel_scale()
		imscale = imwcs.wcs.pixel_scale()
		cscale = (imscale / gridscale)**2
		imshape = img.shape
		p0 = self.getParams()
		counts0 = self._getcounts(img)
		derivs = []

		ss = self.getStepSizes()
		nms = self.getParamNames()
		i0 = 0

		counts0 = counts0.ravel()
		#arrs = self.getArrays(ravel=True)
		sa,t,e = self.getArrays(ravel=True)
		photocal = img.getPhotoCal()
		for si,sub in self._enumerateActiveSubs():
			for parami,gridi in sub._indexBoth():
				if parami % 100 == 0:
					print 'Img', img.name, 'deriv', i0+parami, nms[i0+parami]
				try:
					I,V,((x0,y0),nzshape),NZI = X[gridi]
				except KeyError:
					# This model pixel doesn't touch this image.
					derivs.append(None)
					continue
				step = ss[i0+parami]
				args = [sa[gridi], t[gridi], e[gridi]]
				args[si] += step
				countsi = photocal.dustParamsToCounts(*args)
				dc = (countsi - counts0[gridi])
				dmod = np.zeros(nzshape)
				dmod.ravel()[NZI] = dc * V * (cscale / step)
				derivs.append(Patch(x0, y0, dmod))

			i0 += sub.numberOfParams()

		# This is ugly -- the dust param vectors are stored one after another, so the
		# three parameters affecting each pixel are not contiguous, and we also ignore the
		# fact that we should *know* which grid pixel is affected by each parameter!!
		# (but this might work with freezing... maybe...)
		# for i,(step,name) in enumerate(zip(self.getStepSizes(), self.getParamNames())):
		# 	print 'Img', img.name, 'deriv', i, name
		# 	oldval = self.setParam(i, p0[i] + step)
		# 	countsi = self._getcounts(img)
		# 	self.setParam(i, oldval)
		# 
		# 	dc = (countsi - counts0).ravel()
		# 	I = np.flatnonzero(dc != 0)
		# 	# I spent a long time trying to figure out why the
		# 	# PACS100 beta derivative was zero...
		# 	# (d/dX((100 um / lam0) ** X) = d/dX(1.**X) == 0...)
		# 	if len(I) == 0:
		# 		derivs.append(None)
		# 		continue
		# 	if len(I) != 1:
		# 		print 'dcounts', dc
		# 		print 'I', I
		# 	assert(len(I) == 1)
		# 	ii = I[0]
		# 
		# 	I,V = X[ii]
		# 	dc = float(dc[ii])
		# 	dmod = np.zeros(imshape)
		# 	dmod.ravel()[I] = V * (cscale / step)
		# 	derivs.append(Patch(0, 0, dmod))

		return derivs

def makeplots(tractor, step, suffix):
	print 'Making plots'
	mods = tractor.getModelImages()
	# plt.figure(figsize=(8,6))
	# for i,mod in enumerate(mods):
	# 	tim = tractor.getImage(i)
	# 	ima = dict(interpolation='nearest', origin='lower',
	# 			   vmin=tim.zr[0], vmax=tim.zr[1])
	# 	plt.clf()
	# 	plt.imshow(mod, **ima)
	# 	plt.gray()
	# 	plt.colorbar()
	# 	plt.title(tim.name)
	# 	plt.savefig('model-%i-%02i%s.png' % (i, step, suffix))
	# 
	# 	if step == 0:
	# 		plt.clf()
	# 		plt.imshow(tim.getImage(), **ima)
	# 		plt.gray()
	# 		plt.colorbar()
	# 		plt.title(tim.name)
	# 		plt.savefig('data-%i.png' % (i))
	# 
	# 	plt.clf()
	# 	# tractor.getChiImage(i), 
	# 	plt.imshow((tim.getImage() - mod) * tim.getInvError(),
	# 			   interpolation='nearest', origin='lower',
	# 			   vmin=-5, vmax=+5)
	# 	plt.gray()
	# 	plt.colorbar()
	# 	plt.title(tim.name)
	# 	plt.savefig('chi-%i-%02i%s.png' % (i, step, suffix))

	plt.figure(figsize=(16,8))
	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
	R,C = 3,len(mods)
	plt.clf()
	for i,mod in enumerate(mods):
		tim = tractor.getImage(i)

		print 'Image', tim.name, ': data median', np.median(tim.getImage()),
		print 'model median:', np.median(mod)

		ima = dict(interpolation='nearest', origin='lower',
				   vmin=tim.zr[0], vmax=tim.zr[1])

		plt.subplot(R, C, i + 1)
		plt.imshow(tim.getImage(), **ima)
		plt.xticks([])
		plt.yticks([])
		plt.gray()
		plt.colorbar()
		plt.title(tim.name)

		plt.subplot(R, C, i + 1 + C)
		plt.imshow(mod, **ima)
		plt.xticks([])
		plt.yticks([])
		plt.gray()
		plt.colorbar()
		#plt.title(tim.name)

		plt.subplot(R, C, i + 1 + 2*C)
		plt.imshow((tim.getImage() - mod) * tim.getInvError(),
				   interpolation='nearest', origin='lower',
				   vmin=-5, vmax=+5)
		plt.xticks([])
		plt.yticks([])
		plt.gray()
		plt.colorbar()
		#plt.title(tim.name)
	plt.savefig('all-%02i%s.png' % (step, suffix))

	ds = tractor.getCatalog()[0]
	logsa, logt, emis = ds.getArrays()

	plt.figure(figsize=(16,5))
	plt.clf()

	plt.subplot(1,3,1)
	plt.imshow(np.exp(logsa), interpolation='nearest', origin='lower')
	plt.hot()
	plt.colorbar()
	plt.title('Dust: solid angle (Sr?)')

	plt.subplot(1,3,2)
	plt.imshow(np.exp(logt), interpolation='nearest', origin='lower') #, vmin=0)
	plt.hot()
	plt.colorbar()
	plt.title('Dust: temperature (K)')

	plt.subplot(1,3,3)
	plt.imshow(emis, interpolation='nearest', origin='lower')
	plt.gray()
	plt.colorbar()
	plt.title('Dust: emissivity')
	plt.savefig('dust-%02i%s.png' % (step, suffix))


	# plt.figure(figsize=(8,6))
	# 
	# plt.clf()
	# plt.imshow(logsa, interpolation='nearest', origin='lower')
	# plt.gray()
	# plt.colorbar()
	# plt.title('Dust: log(solid angle)')
	# plt.savefig('logsa-%02i%s.png' % (step, suffix))
	# 
	# plt.clf()
	# plt.imshow(np.exp(logsa), interpolation='nearest', origin='lower')
	# plt.hot()
	# plt.colorbar()
	# plt.title('Dust: solid angle')
	# plt.savefig('sa-%02i%s.png' % (step, suffix))
	# 
	# plt.clf()
	# plt.imshow(logt, interpolation='nearest', origin='lower')
	# plt.gray()
	# plt.colorbar()
	# plt.title('Dust: log(temperature)')
	# plt.savefig('logt-%02i%s.png' % (step, suffix))
	# 
	# plt.clf()
	# plt.imshow(np.exp(logt), interpolation='nearest', origin='lower', vmin=0)
	# plt.hot()
	# plt.colorbar()
	# plt.title('Dust: temperature (K)')
	# plt.savefig('t-%02i%s.png' % (step, suffix))
	# 
	# plt.clf()
	# plt.imshow(emis, interpolation='nearest', origin='lower')
	# plt.gray()
	# plt.colorbar()
	# plt.title('Dust: emissivity')
	# plt.savefig('emis-%02i%s.png' % (step, suffix))

def np_err_handler(typ, flag):
	print 'Floating point error (%s), with flag %s' % (typ, flag)
	import traceback
	#print traceback.print_stack()
	tb = traceback.extract_stack()
	# omit myself
	tb = tb[:-1]
	print ''.join(traceback.format_list(tb))

def create_tractor(opt):
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
		('m31_brick15_PACS100.fits',   7.23,  7.7),
		('m31_brick15_PACS160.fits',   3.71, 12.0),
		('m31_brick15_SPIRE250.fits',  None, 18.0),
		('m31_brick15_SPIRE350.fits',  None, 25.0),
		('m31_brick15_SPIRE500.fits',  None, 37.0),
		]

	# From Groves via Rix:
	# PSF FWHMs: 6.97, 11.01, 18.01, 24.73, 35.98

	# Within the region Ive selected I found temperatures between 15
	# and 21 K, averaging 17 K, and Beta= 1.5 to 2.5 (with a larger
	# uncertainty), averaging around 2.0.

	if opt.no100:
		dataList = dataList[1:]

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
		print 'Lambda', lam
		# calibrated, yo
		assert(hdr['BUNIT'] == 'MJy/Sr')
		# "Filter" is in *microns* in the headers; convert to *m* here, to match "lam0"
		pcal = DustPhotoCal(lam * 1e-6, wcs.pixel_scale())
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

	radecbounds = []
	for tim in tims:
		H,W = tim.shape
		twcs = tim.getWcs()
		rds = []
		for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5),(0.5,0.5)]:
			rd = twcs.pixelToPosition(x,y)
			rds.append(rd)
		rds = np.array(rds)
		radecbounds.append(rds)
	rd = np.vstack(radecbounds)
	#print 'rd', rd.shape
	ramin,decmin = rd.min(axis=0)
	ramax,decmax = rd.max(axis=0)

	dr,dd = ramax-ramin, decmax-decmin
	plotrange = (ramin - 0.05*dr, ramax + 0.05*dr, decmin - 0.05*dd, decmax + 0.05*dd)

	plt.clf()
	for rds,c in zip(radecbounds, ['b','g','y',(1,0.5,0),'r']):
		plt.plot(rds[:,0], rds[:,1], '-', color=c, lw=2, alpha=0.5)
	setRadecAxes(*plotrange)
	plt.savefig('radec1%s.png' % opt.suffix)

	print 'Creating dust sheet...'
	N = opt.gridn

	# Build a WCS for the dust sheet to match the first image
	# (assuming it's square and axis-aligned)
	#wcs = tims[0].getWcs().wcs
	#r,d = wcs.radec_center()
	#H,W = tims[0].shape
	#scale = wcs.pixel_scale()
	#scale *= float(W)/max(1, N-1) / 3600.
	#c = float(N)/2. + 0.5
	#dwcs = Tan(r, d, c, c, scale, 0, 0, scale, N, N)

	# Build an axis-aligned WCS that contains all the images.

	r,d = (ramin + ramax) / 2., (decmin + decmax) / 2.
	# HACK -- ignore pole issues
	scale = max((ramax - ramin) * np.cos(np.deg2rad(d)), decmax - decmin) / float(N)
	scale *= float(N) / float(max(1, N-1))

	scale *= (1. / opt.zoom)

	cpix = float(N)/2. + 0.5
	dwcs = Tan(r, d, cpix, cpix, scale, 0, 0, scale, N, N)

	pixscale = dwcs.pixel_scale()
	logsa = np.log(1e-3)

	H,W = N,N
	logsa = np.zeros((H,W)) + logsa
	logt = np.zeros((H,W)) + np.log(17.)
	emis = np.zeros((H,W)) + 2.

	ds = DustSheet(logsa, logt, emis, dwcs)

	rds = ds.getRaDecCorners(0.5)
	plt.plot(rds[:,0], rds[:,1], 'k-', lw=1, alpha=1)
	setRadecAxes(*plotrange)
	plt.savefig('radec2%s.png' % opt.suffix)

	# plot grid of sample points.
	rds = []
	H,W = N,N
	for y in range(N):
		for x in range(N):
			r,d = dwcs.pixelxy2radec(x+1, y+1)
			rds.append((r,d))
	rds = np.array(rds)
	plt.plot(rds[:,0], rds[:,1], 'k.', lw=1, alpha=0.5)
	setRadecAxes(*plotrange)
	plt.savefig('radec3%s.png' % opt.suffix)

	#print 'DustSheet:', ds
	#print 'np', ds.numberOfParams()
	#print 'pn', ds.getParamNames()
	#print 'p', ds.getParams()

	# print 'PriorChi:', ds.getLogPriorChi()
	# ra,ca,va,pb = ds.getLogPriorChi()
	# print 'ra', ra
	# print 'ca', ca
	# print 'va', va
	# print 'pb', pb
	# for ri,ci,vi,bi in zip(ra,ca,va,pb):
	# 	print
	# 	print 'ri', ri
	# 	print 'ci', ci
	# 	print 'vi', vi
	# 	print 'bi', bi

	cat = Catalog()
	cat.append(ds)
	
	tractor = Tractor(Images(*tims), cat)
	return tractor


pipi = 0
		
		

def main():
	import optparse
	import logging
	import sys

	parser = optparse.OptionParser()
	parser.add_option('--threads', dest='threads', default=1, type=int, help='Use this many concurrent processors')
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')

	parser.add_option('--grid', '-g', dest='gridn', type=int, default=5, help='Dust parameter grid size')
	parser.add_option('--steps', '-s', dest='steps', type=int, default=10, help='# Optimization step')
	parser.add_option('--suffix', dest='suffix', default='', help='Output file suffix')

	parser.add_option('--no-100', dest='no100', action='store_true', default=False,
					  help='Omit PACS-100 data?')

	parser.add_option('--callgrind', dest='callgrind', action='store_true', default=False, help='Turn on callgrind around tractor.optimize()')

	parser.add_option('--resume', '-r', dest='resume', type=int, default=-1, help='Resume from a previous run at the given step?')

	parser.add_option('--zoom', dest='zoom', type=float, default=1, help='Scale down the model to only touch the (1/zoom x 1/zoom) central region of the images')

	opt,args = parser.parse_args()

	if opt.verbose == 0:
		lvl = logging.INFO
		log_init(2)
	else:
		lvl = logging.DEBUG
		log_init(3)
	
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	if opt.threads > 1 and False:
		global dpool
		import debugpool
		dpool = debugpool.DebugPool(opt.threads)
		Time.add_measurement(debugpool.DebugPoolMeas(dpool))
		mp = multiproc(pool=dpool)
	else:
		print 'N threads', opt.threads
		mp = multiproc(opt.threads)#, wrap_all=True)

	if opt.callgrind:
		import callgrind
	else:
		callgrind = None

	np.seterrcall(np_err_handler)
	np.seterr(all='call')
	#np.seterr(all='raise')

	if opt.resume > -1:
		pfn = 'herschel-%02i%s.pickle' % (opt.resume, opt.suffix)
		print 'Reading from', pfn
		tractor = unpickle_from_file(pfn)
		tractor.mp = mp

		ds = tractor.getCatalog()[0]
		print 'DustSheet:', ds

		# derivs = ds.getParamDerivatives(tim)
		# dim = np.zeros(tim.shape)
		# #for k,deriv in enumerate(derivs[:40]):
		# for k,deriv in enumerate(derivs[::10]):
		# 	dim[:,:] = 0
		# 	deriv.addTo(dim)
		# 	plt.clf()
		# 	plt.imshow(dim, interpolation='nearest', origin='lower')
		# 	plt.savefig('deriv-%04i.png' % k)

		#tim = tractor.getImages()[0]
		for it,tim in enumerate(tractor.getImages()):
			X = ds._getTransformation(tim)
			# #print 'X', X
			keys = X.keys()
			keys.sort()
			# for k in keys[::10]:
			# for k in keys[:40]:
			for k in keys[::202]:
				I,G,nil,nil = X[k]
				rim = np.zeros_like(tim.getImage())
				rim.ravel()[I] = G
				plt.clf()
				plt.imshow(rim, interpolation='nearest', origin='lower')
				plt.colorbar()
				plt.savefig('rim-%i-%04i.png' % (it,k))
				print 'pix', k
		sys.exit(0)

		makeplots(tractor, opt.resume, opt.suffix)
		step0 = opt.resume + 1

	else:
		step0 = 0
		tractor = create_tractor(opt)
		tractor.mp = mp

		def point_in_poly(x, y, poly):
			inside = np.zeros(np.atleast_1d(x).shape, bool)
			for i in range(len(poly)):
				j = (i-1 + len(poly)) % len(poly)
				xi,xj = poly[i,0], poly[j,0]
				yi,yj = poly[i,1], poly[j,1]
				I = np.logical_and(
					np.logical_or(np.logical_and(yi <= y, y < yj),
								  np.logical_and(yj <= y, y < yi)),
					x < (xi + ((xj - xi) * (y - yi) / (yj - yi))))
				inside[I] = np.logical_not(inside[I])
			return inside

		def point_in_poly_X(x, y, poly):

			global pipi

			a = np.zeros_like(x).astype(float)
			for i in range(len(poly)):
				dx1 = poly[i,0] - x
				dy1 = poly[i,1] - y
				dx2 = poly[(i+1) % len(poly), 0] - poly[i,0]
				dy2 = poly[(i+1) % len(poly), 1] - poly[i,1]
				angle = np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)
				angle += ( 2.*np.pi * (angle < -np.pi))
				angle += (-2.*np.pi * (angle >  np.pi))
				assert(np.all(angle <= np.pi))
				assert(np.all(angle >= -np.pi))
				a += angle

				plt.clf()
				plt.imshow(a, interpolation='nearest', origin='lower')
				plt.colorbar()
				plt.title('polygon edge %i' % i)
				plt.savefig('pip-%03i.png' % pipi)
				pipi += 1

			return (np.abs(a) > np.pi)
			
		
		# zero out invvar outside the model bounds.
		ds = tractor.getCatalog()[0]
		rd = ds.getRaDecCorners()
		for i,tim in enumerate(tractor.getImages()):
			poly = np.array([tim.getWcs().positionToPixel(RaDecPos(rdi[0], rdi[1])) for rdi in rd])
			poly = poly[:-1,:]
			print 'Model bounding box in image', tim.name, 'coordinates:'
			print poly.shape
			print poly
			H,W = tim.shape
			xx,yy = np.meshgrid(np.arange(W), np.arange(H))
			inside = point_in_poly(xx, yy, poly)
			plt.clf()
			plt.imshow(inside, interpolation='nearest', origin='lower')
			plt.savefig('inside-%i.png' % i)
			iv = tim.getInvvar()
			iv[(inside == 0)] = 0.
			tim.setInvvar(iv)

		print 'Precomputing transformations...'
		ds = tractor.getCatalog()[0]
		XX = mp.map(_map_trans, [(ds,im) for im in tractor.getImages()])
		for im,X in zip(tractor.getImages(), XX):
			ds._setTransformation(im, X)
		print 'done precomputing.'

		makeplots(tractor, 0, opt.suffix)
		pfn = 'herschel-%02i%s.pickle' % (0, opt.suffix)
		pickle_to_file(tractor, pfn)
		print 'Wrote', pfn

	for im in tractor.getImages():
		im.freezeAllBut('sky')

	for i in range(step0, opt.steps):
		if callgrind:
			callgrind.callgrind_start_instrumentation()

		tractor.optimize(damp=1., alphas=[1e-3, 1e-2, 0.1, 0.3, 1., 3., 10., 30., 100.])

		if callgrind:
			callgrind.callgrind_stop_instrumentation()

		makeplots(tractor, 1 + i, opt.suffix)
		pfn = 'herschel-%02i%s.pickle' % (1 + i, opt.suffix)
		pickle_to_file(tractor, pfn)
		print 'Wrote', pfn

def _map_trans((ds, img)):
	print 'computing transformation in PID', os.getpid()
	return ds._computeTransformation(img)

if __name__ == '__main__':
	main()
	#import cProfile
	#import sys
	#from datetime import tzinfo, timedelta, datetime
	#cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	#sys.exit(0)
