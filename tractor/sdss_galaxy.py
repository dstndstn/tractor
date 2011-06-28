# Copyright 2011 Dustin Lang and David W. Hogg.  All rights reserved.
import numpy as np

import mixture_profiles as mp
from engine import *

class GalaxyShape(ParamList):
	def getNamedParams(self):
		# re: arcsec
		# ab: axis ratio, dimensionless, in [0,1]
		# phi: deg, "E of N", 0=direction of increasing Dec, 90=direction of increasing RA
		return [('re', 0), ('ab', 1), ('phi', 2)]
	def hashkey(self):
		return ('GalaxyShape',) + tuple(self.vals)
	def __repr__(self):
		return 're=%g, ab=%g, phi=%g' % (self.re, self.ab, self.phi)
	def __str__(self):
		return 're=%.1f, ab=%.2f, phi=%.1f' % (self.re, self.ab, self.phi)
	def copy(self):
		return GalaxyShape(*self.vals)
	def getParamNames(self):
		return ['re', 'ab', 'phi']

	def getStepSizes(self, img):
		abstep = 0.01
		if self.ab >= (1 - abstep):
			abstep = -abstep
		return [ 1., abstep, 1. ]

	def setre(self, re):
		if re < (1./30.):
			print 'Clamping re from', re, 'to 1/30'
		self.re = max(1./30., re)
	def setab(self, ab):
		if ab > 1.:
			print 'Converting ab from', ab, 'to', 1./ab
			self.setab(1./ab)
			self.setphi(self.phi+90.)
		elif ab < (1./30.):
			print 'Clamping ab from', ab, 'to 1/30'
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
	# -but in Tractor.tryParamUpdates, it only calls getParams(),
	#  stepParams(), and setParams() to revert to original params.
	#
	# -stepping params one at a time works fine, so it's all ok.

	def setParams(self, p):
		assert(len(p) == 3)
		self.setre(p[0])
		self.setab(p[1])
		self.setphi(p[2])
	def stepParam(self, parami, delta):
		if parami == 0:
			self.setre(self.re + delta)
		elif parami == 1:
			self.setab(self.ab + delta)
		elif parami == 2:
			self.setphi(self.phi + delta)
		else:
			raise RuntimeError('GalaxyShape: unknown parami: ' + str(parami))

	def getTensor(self, cd):
		# convert re, ab, phi into a transformation matrix
		phi = np.deg2rad(90 - self.phi)
		# convert re to degrees
		re_deg = self.re / 3600.
		cp = np.cos(phi)
		sp = np.sin(phi)
		# Squish, rotate, and scale into degrees.
		# G takes unit vectors (in r_e) to degrees (~intermediate world coords)
		G = re_deg * np.array([[ cp, sp * self.ab],
							   [-sp, cp * self.ab]])
		# "cd" takes pixels to degrees (intermediate world coords)
		# T takes pixels to unit vectors.
		#print 'phi', phi, 're', re_deg
		#print 'G', G
		T = np.dot(np.linalg.inv(G), cd)
		return T

class Galaxy(MultiParams):
	def __init__(self, pos, flux, shape):
		MultiParams.__init__(self, pos, flux, shape)
		self.name = self.getName()

	def getName(self):
		return 'Galaxy'

	def getSourceType(self):
		return self.name

	def getPosition(self):
		return self.pos

	def getFlux(self):
		return self.flux

	def getNamedParams(self):
		return [('pos', 0), ('flux', 1), ('shape', 2)]

	def __getattr__(self, name):
		if name in ['re', 'ab', 'phi']:
			return getattr(self.shape, name)
		return MultiParams.__getattr__(self, name)

	def __setattr__(self, name, val):
		if name in ['re', 'ab', 'phi']:
			setattr(self.shape, name, val)
			return
		MultiParams.__setattr__(self, name, val)

	def hashkey(self):
		return (self.name, self.pos.hashkey(), self.flux.hashkey(),
				self.re, self.ab, self.phi)
	def __str__(self):
		return (self.name + ' at ' + str(self.pos)
				+ ' with ' + str(self.flux)
				+ ', re=%.1f, ab=%.2f, phi=%.1f' % (self.re, self.ab, self.phi))
	def __repr__(self):
		return (self.name + '(pos=' + repr(self.pos) +
				', flux=' + repr(self.flux) +
				', re=%.1f, ab=%.2f, phi=%.1f)' % (self.re, self.ab, self.phi))

	def copy(self):
		return None

	def getProfile(self):
		return None

	def getUnitFluxModelPatch(self, img, px=None, py=None):
		raise RuntimeError('getUnitFluxModelPatch unimplemented in' +
						   self.getName())

	def getModelPatch(self, img, px=None, py=None):
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		p1 = self.getUnitFluxModelPatch(img, px, py)
		if p1 is None:
			return None
		return p1 * counts

	# returns [ Patch, Patch, ... ] of length numberOfParams().
	# Galaxy.
	def getParamDerivatives(self, img, fluxonly=False):
		pos0 = self.getPosition()
		(px0,py0) = img.getWcs().positionToPixel(self, pos0)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		patch0 = self.getUnitFluxModelPatch(img, px0, py0)
		if patch0 is None:
			return [None] * self.numberOfParams()
		derivs = []

		# derivatives wrt position
		psteps = pos0.getStepSizes(img)
		if fluxonly:
			derivs.extend([None] * len(psteps))
		else:
			for i in range(len(psteps)):
				posx = pos0.copy()
				posx.stepParam(i, psteps[i])
				(px,py) = img.getWcs().positionToPixel(self, posx)
				patchx = self.getUnitFluxModelPatch(img, px, py)
				if patchx.getImage() is None:
					derivs.append(None)
					continue
				dx = (patchx - patch0) * (counts / psteps[i])
				dx.setName('d(gal)/d(pos%i)' % i)
				derivs.append(dx)

		# derivatives wrt flux
		fsteps = self.flux.getStepSizes(img)
		for i in range(len(fsteps)):
			fi = self.flux.copy()
			fi.stepParam(i, fsteps[i])
			countsi = img.getPhotoCal().fluxToCounts(fi)
			df = patch0 * ((countsi - counts) / fsteps[i])
			df.setName('d(gal)/d(flux%i)' % i)
			derivs.append(df)

		# derivatives wrt shape
		gsteps = self.shape.getStepSizes(img)
		gnames = self.shape.getParamNames()
		oldvals = self.shape.getParams()
		#print 'Galaxy.getParamDerivatives:', self.getName()
		#print '  oldvals:', oldvals
		if fluxonly:
			derivs.extend([None] * len(gsteps))
		else:
			for i in range(len(gsteps)):
				self.shape.stepParam(i, gsteps[i])
				#print '  stepped', gnames[i], 'by', gsteps[i],
				#print 'to get', self.shape
				patchx = self.getUnitFluxModelPatch(img, px0, py0)
				self.shape.setParams(oldvals)
				#print '  reverted to', self.shape
				if patchx is None:
					print 'patchx is None:'
					print '  ', self
					print '  stepping galaxy shape', self.shape.getParamNames()[i]
					print '  stepped', gsteps[i]
					print '  to', self.shape.getParams()[i]
					derivs.append(None)

				dx = (patchx - patch0) * (counts / gsteps[i])
				dx.setName('d(gal)/d(%s)' % (gnames[i]))
				derivs.append(dx)
		return derivs


class CompositeGalaxy(Galaxy):
	'''
	A galaxy with Exponential and deVaucouleurs components.

	The two components share a position (ie the centers are the same),
	but have different fluxes and shapes.
	'''
	def __init__(self, pos, fluxExp, shapeExp, fluxDev, shapeDev):
		MultiParams.__init__(self, pos, fluxExp, shapeExp, fluxDev, shapeDev)
		self.name = self.getName()
	def getName(self):
		return 'CompositeGalaxy'
	def getNamedParams(self):
		return [('pos', 0), ('fluxExp', 1), ('shapeExp', 2),
				('fluxDev', 3), ('shapeDev', 4),]
	def getFlux(self):
		return self.fluxExp + self.fluxDev

	def hashkey(self):
		return (self.name, self.pos.hashkey(),
				self.fluxExp.hashkey(), self.shapeExp.hashkey(),
				self.fluxDev.hashkey(), self.shapeDev.hashkey())
	def __str__(self):
		return (self.name + ' at ' + str(self.pos)
				+ ' with Exp ' + str(self.fluxExp) + ' ' + str(self.shapeExp)
				+ ' and deV ' + str(self.fluxDev) + ' ' + str(self.shapeDev))
	def __repr__(self):
		return (self.name + '(pos=' + repr(self.pos) +
				', fluxExp=' + repr(self.fluxExp) +
				', shapeExp=' + repr(self.shapeExp) + 
				', fluxDev=' + repr(self.fluxDev) +
				', shapeDev=' + repr(self.shapeDev))
	def copy(self):
		return CompositeGalaxy(self.pos, self.fluxExp, self.shapeExp,
							   self.fluxDev, self.shapeDev)
	def getModelPatch(self, img, px=None, py=None):
		e = ExpGalaxy(self.pos, self.fluxExp, self.shapeExp)
		d = DevGalaxy(self.pos, self.fluxDev, self.shapeDev)
		pe = e.getModelPatch(img, px, py)
		pd = d.getModelPatch(img, px, py)
		if pe is None:
			return pd
		if pd is None:
			return pe
		return pe + pd
	
	def getUnitFluxModelPatch(self, img, px=None, py=None):
		# this code is un-tested
		assert(False)
		fe = self.fluxExp / (self.fluxExp + self.fluxDev)
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
	def getParamDerivatives(self, img, fluxonly=False):
		print 'CompositeGalaxy: getParamDerivatives'
		print '  Exp flux', self.fluxExp, 'shape', self.shapeExp
		print '  Dev flux', self.fluxDev, 'shape', self.shapeDev
		e = ExpGalaxy(self.pos, self.fluxExp, self.shapeExp)
		d = DevGalaxy(self.pos, self.fluxDev, self.shapeDev)
		de = e.getParamDerivatives(img, fluxonly)
		dd = d.getParamDerivatives(img, fluxonly)
		npos = len(self.pos.getStepSizes(img))
		derivs = []
		for i in range(npos):
			dp = de[i] + dd[i]
			dp.setName('d(gal)/d(pos%i)' % i)
			derivs.append(dp)
		derivs.extend(de[npos:])
		derivs.extend(dd[npos:])
		return derivs

class HoggGalaxy(Galaxy):
	ps = PlotSequence('hg', format='%03i')

	def __init__(self, pos, flux, *args):
		'''
		HoggGalaxy(pos, flux, GalaxyShape)
		or
		HoggGalaxy(pos, flux, re, ab, phi)

		re: [arcsec]
		phi: [deg]
		'''
		if len(args) == 3:
			shape = GalaxyShape(*args)
		else:
			assert(len(args) == 1)
			shape = args[0]
		Galaxy.__init__(self, pos, flux, shape)

	def getName(self):
		return 'HoggGalaxy'

	def copy(self):
		return HoggGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

	def getUnitFluxModelPatch(self, img, px=None, py=None):
		if px is None or py is None:
			(px,py) = img.getWcs().positionToPixel(self, self.getPosition())
		galmix = self.getProfile()
		# shift and squash
		cd = img.getWcs().cdAtPixel(px, py)
		Tinv = np.linalg.inv(self.shape.getTensor(cd))
		try:
			amix = galmix.apply_affine(np.array([px,py]), Tinv.T)
		except:
			print 'Failed in getModelPatch of', self
			return None

		if False:
			(eval, evec) = np.linalg.eig(amix.var[0])
			print amix.var[0]
			print 'true ab:', self.ab
			print 'eigenval-based ab:', np.sqrt(eval[1]/eval[0])
			print 'true phi:', self.phi
			print 'eigenvec-based phi:', deg2rad(np.arctan2(evec[0,1], evec[0,0])), deg2rad(np.arctan2(evec[1,0], evec[0,0]))
		amix.symmetrize()
		# now convolve with the PSF
		psf = img.getPsf()
		psfmix = psf.getMixtureOfGaussians()
		psfmix.normalize()
		cmix = amix.convolve(psfmix)
		# now choose the patch size
		pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
		if self.ab <= 1:
			halfsize = max(8., 8. * (self.re / 3600.) / pixscale)
		else:
			halfsize = max(8., 8. * (self.re*self.ab / 3600.) / pixscale)
		# now evaluate the mixture on the patch pixels
		(outx, inx) = get_overlapping_region(int(floor(px-halfsize)), int(ceil(px+halfsize+1)), 0., img.getWidth())
		(outy, iny) = get_overlapping_region(int(floor(py-halfsize)), int(ceil(py+halfsize+1)), 0., img.getHeight())
		if inx == [] or iny == []:
			print 'No overlap between model and image'
			return None
		x0 = outx.start
		y0 = outy.start
		x1 = outx.stop
		y1 = outy.stop
		psfconvolvedimg = mp.mixture_to_patch(cmix, np.array([x0,y0]),
											  np.array([x1,y1]))

		#print 'psf sum of ampls:', np.sum(psfmix.amp)
		#print 'unconvolved mixture sum of ampls:', np.sum(amix.amp)
		#print 'convolved mixture sum of ampls:', np.sum(cmix.amp)
		#print 'psf-conv img sum:', psfconvolvedimg.sum()
		# now return a calibrated patch
		#print 'x0,y0', x0,y0
		#print 'patch shape', psfconvolvedimg.shape
		#print 'img w,h', img.getWidth(), img.getHeight()

		if False:
			plt.clf()
			plt.imshow(psfconvolvedimg*counts,
					   interpolation='nearest', origin='lower')
			plt.hot()
			plt.colorbar()
			HoggGalaxy.ps.savefig()

		return Patch(x0, y0, psfconvolvedimg)


class ExpGalaxy(HoggGalaxy):
	profile = mp.get_exp_mixture()
	profile.normalize()
	@staticmethod
	def getExpProfile():
		return ExpGalaxy.profile
	#def __init__(self, pos, flux, re, ab, phi):
	#	HoggGalaxy.__init__(self, pos, flux, re, ab, phi)
	def getName(self):
		return 'ExpGalaxy'
	def getProfile(self):
		return ExpGalaxy.getExpProfile()
	def copy(self):
		return ExpGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

class DevGalaxy(HoggGalaxy):
	profile = mp.get_dev_mixture()
	profile.normalize()
	@staticmethod
	def getDevProfile():
		return DevGalaxy.profile
	#def __init__(self, pos, flux, re, ab, phi):
	#	HoggGalaxy.__init__(self, pos, flux, re, ab, phi)
	def getName(self):
		return 'DevGalaxy'
	def getProfile(self):
		return DevGalaxy.getDevProfile()
	def copy(self):
		return DevGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)
