"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`sdss_galaxy.py`
================

SDSS exponential and deVaucouleurs galaxy model classes.

These models are not specific to SDSS *images*, they just use a
slightly modified definition of the exp and dev profiles from SDSS
Photo.
"""
import numpy as np

import mixture_profiles as mp
from engine import *
from utils import *

from cache import *
from astrometry.util.plotutils import PlotSequence

_galcache = Cache(maxsize=10000)
def get_galaxy_cache():
	return _galcache

def set_galaxy_cache_size(N):
	global _galcache
	_galcache.clear()
	_galcache = Cache(maxsize=N)

def disable_galaxy_cache():
	global _galcache
	_galcache = NullCache()

class GalaxyShape(ParamList):
	@staticmethod
	def getName():
		return "Galaxy Shape"
	@staticmethod
	def getNamedParams():
		# re: arcsec
		# ab: axis ratio, dimensionless, in [0,1]
		# phi: deg, "E of N", 0=direction of increasing Dec, 90=direction of increasing RA
		return dict(re=0, ab=1, phi=2)
	def __repr__(self):
		return 're=%g, ab=%g, phi=%g' % (self.re, self.ab, self.phi)
	def __str__(self):
		return '%s: re=%.2f, ab=%.2f, phi=%.1f' % (self.getName(), self.re, self.ab, self.phi)
	#def copy(self):
	#	return GalaxyShape(*self.vals)
	#def getParamNames(self):
	#	return ['re', 'ab', 'phi']

	def getStepSizes(self, *args, **kwargs):
		abstep = 0.01
		if self.ab >= (1 - abstep):
			abstep = -abstep
		return list(self._getLiquidArray([ 1., abstep, 1. ]))

	def setre(self, re):
		if re < (1./30.):
			#print 'Clamping re from', re, 'to 1/30'
			pass
		self.re = max(1./30., re)
	def setab(self, ab):
		if ab > 1.:
			#print 'Converting ab from', ab, 'to', 1./ab
			self.setab(1./ab)
			self.setphi(self.phi+90.)
		elif ab < (1./30.):
			#print 'Clamping ab from', ab, 'to 1/30'
			self.ab = 1./30
		else:
			self.ab = ab
	def setphi(self, phi):
		# limit phi to [-180,180]
		self.phi = np.fmod(phi, 360.)
		if self.phi < -180.:
			self.phi += 360.
		if self.phi > 180.:
			self.phi -= 360.

	# Note about flipping the galaxy when ab>1:
	#
	# -you might worry that the caller would choose a new ab>1 and
	#  phi, then call setab() then setphi() -- so then the phi would
	#  be reverted.
	#
	# -but in Tractor.tryParamUpdates, it only calls getParams() and
	#  setParams() to revert to original params.
	#
	# -stepping params one at a time works fine, so it's all ok.

	def setParams(self, p):
		assert(len(p) == 3)
		self.setre(p[0])
		self.setab(p[1])
		self.setphi(p[2])
	def setParam(self, i, p):
		oldval = self.vals[i]
		if i == 0:
			self.setre(p)
		elif i == 1:
			self.setab(p)
		elif i == 2:
			self.setphi(p)
		else:
			raise RuntimeError('GalaxyShape: unknown param index: ' + str(i))
		return oldval

	def getRaDecBasis(self):
		''' Returns a transformation matrix that takes vectors in r_e
		to delta-RA, delta-Dec vectors.
		'''
		# convert re, ab, phi into a transformation matrix
		phi = np.deg2rad(90 - self.phi)
		# convert re to degrees
		# HACK -- bring up to a minimum size to prevent singular matrix inversions
		re_deg = max(1./30, self.re) / 3600.
		cp = np.cos(phi)
		sp = np.sin(phi)
		# Squish, rotate, and scale into degrees.
		# G takes unit vectors (in r_e) to degrees (~intermediate world coords)
		G = re_deg * np.array([[ cp, sp * self.ab],
							   [-sp, cp * self.ab]])
		return G

	def getTensor(self, cd):
		# G takes unit vectors (in r_e) to degrees (~intermediate world coords)
		G = self.getRaDecBasis()
		# "cd" takes pixels to degrees (intermediate world coords)
		# T takes pixels to unit vectors.
		T = np.dot(np.linalg.inv(G), cd)
		return T

	def mayOverlapCircle(self, dra, ddec, radius, nre):
		# is it outside the bounding circle?
		re_deg = max(1./30, self.re) / 3600. * nre
		dd = np.hypot(dra,ddec)
		if dd > (re_deg + radius):
			#print 'outside bounding circle'
			return False
		return True
 	# is it inside the bounding circle of the squished circle?
	# if dd < (re_deg * self.ab + radius):
	# 	#print 'inside inner bounding circle'
	# 	return True
	# 	G = self.getRaDecBasis()
	# 	angles = np.linspace(0., 2.*np.pi, 36, endpoint=False)
	# 	vv = np.vstack([np.cos(angles), np.sin(angles)])
	# 	print vv.shape
	# 	dradec = np.dot(vv.T, G)
	# 	print dradec.shape
	# 	return np.min((dradec - np.array([dra,ddec]))**2 <= radius**2)

class Galaxy(MultiParams):
	def __init__(self, pos, brightness, shape):
		super(Galaxy, self).__init__(pos, brightness, shape)
		self.name = self.getName()
		self.dname = self.getDName()

	@staticmethod
	def getNamedParams():
		return dict(pos=0, brightness=1, shape=2)

	def _setRe (self, re ): self.shape.re  = re
	def _setAb (self, ab ): self.shape.ab  = ab
	def _setPhi(self, phi): self.shape.phi = phi

	# define pass-through names
	re = property(lambda x: x.shape.re, _setRe, None,
				  'galaxy effective radius')
	ab = property(lambda x: x.shape.ab, _setAb, None,
				  'galaxy axis ratio')
	phi = property(lambda x: x.shape.phi, _setPhi, None,
				   'galaxy position angle')

	def getName(self):
		return 'Galaxy'

	def getDName(self):
		'''
		Name used in labeling the derivative images d(Dname)/dx, eg
		'''
		return 'gal'
		
	def getSourceType(self):
		return self.name

	def getPosition(self):
		return self.pos
	def getShape(self):
		return self.shape
	def getBrightness(self):
		return self.brightness
	def getBrightnesses(self):
		return [self.getBrightness()]
	def setBrightness(self, brightness):
		self.brightness = brightness

	def __str__(self):
		return (self.name + ' at ' + str(self.pos)
				+ ' with ' + str(self.brightness)
				+ ' and ' + str(self.shape))
	def __repr__(self):
		return (self.name + '(pos=' + repr(self.pos) +
				', brightness=' + repr(self.brightness) +
				', re=%.1f, ab=%.2f, phi=%.1f)' % (self.re, self.ab, self.phi))

	def copy(self):
		return None

	def getProfile(self):
		return None

	def getUnitFluxModelPatch(self, img, **kwargs):
		raise RuntimeError('getUnitFluxModelPatch unimplemented in' +
						   self.getName())

	def getModelPatch(self, img, minsb=None):
		counts = img.getPhotoCal().brightnessToCounts(self.brightness)
		minval = 0.
		if minsb is not None:
			minval = minsb / counts
		p1 = self.getUnitFluxModelPatch(img, minval=minval)
		if p1 is None:
			return None
		return p1 * counts

	# returns [ Patch, Patch, ... ] of length numberOfParams().
	# Galaxy.
	def getParamDerivatives(self, img):
		pos0 = self.getPosition()
		(px0,py0) = img.getWcs().positionToPixel(pos0, self)
		counts = img.getPhotoCal().brightnessToCounts(self.brightness)
		patch0 = self.getUnitFluxModelPatch(img, px0, py0)
		if patch0 is None:
			return [None] * self.numberOfParams()
		derivs = []

		# derivatives wrt position
		psteps = pos0.getStepSizes()
		if not self.isParamFrozen('pos'):
			params = pos0.getParams()
			for i,pstep in enumerate(psteps):
				oldval = pos0.setParam(i, params[i]+pstep)
				(px,py) = img.getWcs().positionToPixel(pos0, self)
				pos0.setParam(i, oldval)
				patchx = self.getUnitFluxModelPatch(img, px, py)
				if patchx is None or patchx.getImage() is None:
					derivs.append(None)
					continue
				dx = (patchx - patch0) * (counts / pstep)
				dx.setName('d(%s)/d(pos%i)' % (self.dname, i))
				derivs.append(dx)

		# derivatives wrt brightness
		bsteps = self.brightness.getStepSizes()
		if not self.isParamFrozen('brightness'):
			params = self.brightness.getParams()
			for i,bstep in enumerate(bsteps):
				oldval = self.brightness.setParam(i, params[i] + bstep)
				countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
				self.brightness.setParam(i, oldval)
				df = patch0 * ((countsi - counts) / bstep)
				df.setName('d(%s)/d(bright%i)' % (self.dname, i))
				derivs.append(df)

		# derivatives wrt shape
		gsteps = self.shape.getStepSizes()
		if not self.isParamFrozen('shape'):
			gnames = self.shape.getParamNames()
			oldvals = self.shape.getParams()
			# print 'Galaxy.getParamDerivatives:', self.getName()
			# print '  oldvals:', oldvals
			for i,gstep in enumerate(gsteps):
				oldval = self.shape.setParam(i, oldvals[i]+gstep)
				#print '  stepped', gnames[i], 'by', gsteps[i],
				#print 'to get', self.shape
				patchx = self.getUnitFluxModelPatch(img, px0, py0)
				self.shape.setParam(i, oldval)
				#print '  reverted to', self.shape
				if patchx is None:
					print 'patchx is None:'
					print '  ', self
					print '  stepping galaxy shape', self.shape.getParamNames()[i]
					print '  stepped', gsteps[i]
					print '  to', self.shape.getParams()[i]
					derivs.append(None)

				dx = (patchx - patch0) * (counts / gstep)
				dx.setName('d(%s)/d(%s)' % (self.dname, gnames[i]))
				derivs.append(dx)
		return derivs


class CompositeGalaxy(MultiParams):
	'''
	A galaxy with Exponential and deVaucouleurs components.

	The two components share a position (ie the centers are the same),
	but have different brightnesses and shapes.
	'''
	def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev):
		MultiParams.__init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev)
		self.name = self.getName()

	def overlapsCircle(self, pos, radius):
		cosdec = np.cos(np.deg2rad(pos.dec))
		dr = (pos.ra - self.pos.ra)*cosdec
		dd = pos.dec - self.pos.dec
		return (self.shapeDev.mayOverlapCircle(dr, dd, radius, DevGalaxy.nre) or
			self.shapeExp.mayOverlapCircle(dr, dd, radius, ExpGalaxy.nre))

	@staticmethod
	def getNamedParams():
		return dict(pos=0, brightnessExp=1, shapeExp=2, brightnessDev=3, shapeDev=4)

	def getName(self):
		return 'CompositeGalaxy'

	def getPosition(self):
		return self.pos

	def __str__(self):
		return (self.name + ' at ' + str(self.pos)
				+ ' with Exp ' + str(self.brightnessExp) + ' ' + str(self.shapeExp)
				+ ' and deV ' + str(self.brightnessDev) + ' ' + str(self.shapeDev))
	def __repr__(self):
		return (self.name + '(pos=' + repr(self.pos) +
				', brightnessExp=' + repr(self.brightnessExp) +
				', shapeExp=' + repr(self.shapeExp) + 
				', brightnessDev=' + repr(self.brightnessDev) +
				', shapeDev=' + repr(self.shapeDev))
	def copy(self):
		return CompositeGalaxy(self.pos.copy(), self.brightnessExp.copy(),
							   self.shapeExp.copy(), self.brightnessDev.copy(),
							   self.shapeDev.copy())

	def getBrightness(self):
		''' This makes some assumptions about the ``Brightness`` / ``PhotoCal`` and
		should be treated as approximate.'''
		return self.brightnessExp + self.brightnessDev

	def getBrightnesses(self):
		return [self.brightnessExp, self.brightnessDev]
	
	def getModelPatch(self, img, minsb=None):
		e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
		d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
		if minsb is None:
			kw = {}
		else:
			kw = dict(minsb=minsb/2.)
		pe = e.getModelPatch(img, **kw)
		pd = d.getModelPatch(img, **kw)
		if pe is None:
			return pd
		if pd is None:
			return pe
		return pe + pd
	
	def getUnitFluxModelPatch(self, img, px=None, py=None):
		# this code is un-tested
		assert(False)
		fe = self.brightnessExp / (self.brightnessExp + self.brightnessDev)
		fd = 1. - fe
		assert(fe >= 0.)
		assert(fe <= 1.)
		e = ExpGalaxy(self.pos, fe, self.shapeExp)
		d = DevGalaxy(self.pos, fd, self.shapeDev)
		pe = e.getModelPatch(img, px, py)
		pd = d.getModelPatch(img, px, py)
		if pe is None:
			return pd
		if pd is None:
			return pe
		return pe + pd

	# MAGIC: ORDERING OF EXP AND DEV PARAMETERS
	# MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
	# CompositeGalaxy.
	def getParamDerivatives(self, img):
		#print 'CompositeGalaxy: getParamDerivatives'
		#print '  Exp brightness', self.brightnessExp, 'shape', self.shapeExp
		#print '  Dev brightness', self.brightnessDev, 'shape', self.shapeDev
		e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
		d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
		e.dname = 'comp.exp'
		d.dname = 'comp.dev'
		if self.isParamFrozen('pos'):
			e.freezeParam('pos')
			d.freezeParam('pos')
		if self.isParamFrozen('brightnessExp'):
			e.freezeParam('brightness')
		if self.isParamFrozen('shapeExp'):
			e.freezeParam('shape')
		if self.isParamFrozen('brightnessDev'):
			d.freezeParam('brightness')
		if self.isParamFrozen('shapeDev'):
			d.freezeParam('shape')

		de = e.getParamDerivatives(img)
		dd = d.getParamDerivatives(img)

		if self.isParamFrozen('pos'):
			derivs = de + dd
		else:
			derivs = []
			# "pos" is shared between the models, so add the derivs.
			npos = len(self.pos.getStepSizes())
			for i in range(npos):
				#dp = de[i] + dd[i]   -- but one or both could be None
				dp = de[i]
				if dd[i] is not None:
					if dp is None:
						dp = dd[i]
					else:
						dp += dd[i]
				if dp is not None: 
					dp.setName('d(comp)/d(pos%i)' % i)
				derivs.append(dp)
			derivs.extend(de[npos:])
			derivs.extend(dd[npos:])

		return derivs

class HoggGalaxy(Galaxy):
	#ps = PlotSequence('hg', format='%03i')

	def __init__(self, pos, brightness, *args):
		'''
		HoggGalaxy(pos, brightness, GalaxyShape)
		or
		HoggGalaxy(pos, brightness, re, ab, phi)

		re: [arcsec]
		phi: [deg]
		'''
		if len(args) == 3:
			shape = GalaxyShape(*args)
		else:
			assert(len(args) == 1)
			shape = args[0]
		super(HoggGalaxy, self).__init__(pos, brightness, shape)

	def overlapsCircle(self, pos, radius):
		cosdec = np.cos(np.deg2rad(pos.dec))
		return self.shape.mayOverlapCircle((pos.ra - self.pos.ra)*cosdec,
						   pos.dec - self.pos.dec,
						   radius, self.nre)

	def getName(self):
		return 'HoggGalaxy'

	def copy(self):
		return HoggGalaxy(self.pos.copy(), self.brightness.copy(),
						  self.shape.copy())

	def getUnitFluxModelPatch(self, img, px=None, py=None, minval=0.0):
		if px is None or py is None:
			(px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
		#
		deps = hash(('unitpatch', self.getName(), px, py, img.getWcs().hashkey(),
					 img.getPsf().hashkey(), self.shape.hashkey()))
		try:
			(cached,mv) = _galcache.get(deps)
			if mv <= minval:
				return cached
		except KeyError:
			pass
		patch = self._realGetUnitFluxModelPatch(img, px, py, minval)
		_galcache.put(deps, (patch,minval))
		return patch
		

	def _realGetUnitFluxModelPatch(self, img, px, py, minval):
		# shift and squash
		cd = img.getWcs().cdAtPixel(px, py)

		# now choose the patch size
		pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
		if self.ab <= 1:
			abscale = 1.
		else:
			abscale = self.ab
		halfsize = max(1., self.nre * self.re * max(self.ab, 1.) / 3600. / pixscale)
		#print 'halfsize', halfsize, 'pixels'
		psf = img.getPsf()
		halfsize += psf.getRadius()
		#print ' +psf -> ', halfsize, 'pixels'

		# now evaluate the mixture on the patch pixels
		(outx, inx) = get_overlapping_region(int(floor(px-halfsize)), int(ceil(px+halfsize+1)), 0, img.getWidth())
		(outy, iny) = get_overlapping_region(int(floor(py-halfsize)), int(ceil(py+halfsize+1)), 0, img.getHeight())
		if inx == [] or iny == []:
			#print 'No overlap between model and image'
			return None

		galmix = self.getProfile()
		Tinv = np.linalg.inv(self.shape.getTensor(cd))
		amix = galmix.apply_affine(np.array([px,py]), Tinv.T)

		if False:
			(eval, evec) = np.linalg.eig(amix.var[0])
			print amix.var[0]
			print 'true ab:', self.ab
			print 'eigenval-based ab:', np.sqrt(eval[1]/eval[0])
			print 'true phi:', self.phi
			print 'eigenvec-based phi:', deg2rad(np.arctan2(evec[0,1], evec[0,0])), deg2rad(np.arctan2(evec[1,0], evec[0,0]))
		amix.symmetrize()
		# now convolve with the PSF
		# We're making a strong assumption about the PSF here:
		psfmix = psf.getMixtureOfGaussians()
		psfmix.normalize()
		cmix = amix.convolve(psfmix)
		x0 = outx.start
		y0 = outy.start
		x1 = outx.stop
		y1 = outy.stop

		psfconvolvedimg = mp.mixture_to_patch(cmix, x0, x1, y0, y1, minval)

		#print 'psf sum of ampls:', np.sum(psfmix.amp)
		#print 'unconvolved mixture sum of ampls:', np.sum(amix.amp)
		#print 'convolved mixture sum of ampls:', np.sum(cmix.amp)
		#print 'psf-conv img sum:', psfconvolvedimg.sum()
		# now return a calibrated patch
		#print 'x0,y0', x0,y0
		#print 'patch shape', psfconvolvedimg.shape
		#print 'img w,h', img.getWidth(), img.getHeight()

		#print 'psf-convolved img sum:', psfconvolvedimg.sum()

		# if False:
		# 	plt.clf()
		# 	plt.imshow(psfconvolvedimg*counts,
		# 			   interpolation='nearest', origin='lower')
		# 	plt.hot()
		# 	plt.colorbar()
		# 	#HoggGalaxy.ps.savefig()

		return Patch(x0, y0, psfconvolvedimg)


class ExpGalaxy(HoggGalaxy):
	nre = 4.
	profile = mp.get_exp_mixture()
	profile.normalize()
	@staticmethod
	def getExpProfile():
		return ExpGalaxy.profile
	def __init__(self, *args, **kwargs):
		self.nre = ExpGalaxy.nre
		super(ExpGalaxy,self).__init__(*args, **kwargs)
	def getName(self):
		return 'ExpGalaxy'
	def getProfile(self):
		return ExpGalaxy.getExpProfile()
	def getShape(self):
		return self.shape
	def copy(self):
		return ExpGalaxy(self.pos.copy(), self.brightness.copy(),
						 self.shape.copy())

class DevGalaxy(HoggGalaxy):
	nre = 8.
	profile = mp.get_dev_mixture()
	profile.normalize()
	@staticmethod
	def getDevProfile():
		return DevGalaxy.profile
	def __init__(self, *args, **kwargs):
		self.nre = DevGalaxy.nre
		super(DevGalaxy,self).__init__(*args, **kwargs)
	def getName(self):
		return 'DevGalaxy'
	def getProfile(self):
		return DevGalaxy.getDevProfile()
	def copy(self):
		return DevGalaxy(self.pos.copy(), self.brightness.copy(),
						 self.shape.copy())


if __name__ == '__main__':
	from astrometry.util.plotutils import PlotSequence
	import matplotlib
	from basics import GaussianMixturePSF, PixPos, Flux, NullPhotoCal, NullWCS, ConstantSky
	from engine import Image
	matplotlib.use('Agg')
	import pylab as plt
	ps = PlotSequence('gal')
	
	# example PSF (from WISE W1 fit)
	w = np.array([ 0.77953706,  0.16022146,  0.06024237])
	mu = np.array([[-0.01826623, -0.01823262],
				   [-0.21878855, -0.0432496 ],
				   [-0.83365747, -0.13039277]])
	sigma = np.array([[[  7.72925584e-01,   5.23305564e-02],
					   [  5.23305564e-02,   8.89078473e-01]],
					   [[  9.84585869e+00,   7.79378820e-01],
					   [  7.79378820e-01,   8.84764455e+00]],
					   [[  2.02664489e+02,  -8.16667434e-01],
						[ -8.16667434e-01,   1.87881670e+02]]])
	
	psf = GaussianMixturePSF(w, mu, sigma)

	shape = GalaxyShape(10., 0.5, 30.)
	pos = PixPos(100, 50)
	bright = Flux(1000.)
	egal = ExpGalaxy(pos, bright, shape)

	pcal = NullPhotoCal()
	wcs = NullWCS()
	data = np.zeros((100, 200))
	invvar = np.zeros_like(data)
	tim = Image(data=data, invvar=invvar, psf=psf, wcs=wcs, sky=ConstantSky(0.),
				photocal=pcal)

	p0 = egal.getModelPatch(tim)
	
	p1 = egal.getModelPatch(tim, 1e-3)

	print 'p0', p0.patch.sum()
	print 'p1', p1.patch.sum()
	
	plt.clf()
	ima = dict(interpolation='nearest', origin='lower')
	plt.subplot(2,1,1)
	plt.imshow(np.log10(np.maximum(1e-16, p0.patch)), **ima)
	plt.colorbar()
	plt.subplot(2,1,2)
	plt.imshow(np.log10(np.maximum(1e-16, p1.patch)), **ima)
	plt.colorbar()
	ps.savefig()
