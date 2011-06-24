# Copyright 2011 Dustin Lang and David W. Hogg.  All rights reserved.
if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
from math import pi, sqrt, ceil, floor
from datetime import datetime

import pyfits
import pylab as plt
import numpy as np
import matplotlib

from engine import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.ngc2000 import ngc2000
from astrometry.util.plotutils import setRadecAxes, redgreen

from compiled_profiles import *
from galaxy_profiles import *

import mixture_profiles as mp

# might want to override this to set the step size to ~ a pixel
#class SdssRaDecPos(RaDecPos):
#	def getStepSizes(self, img):
#		return [1e-4, 1e-4]


def _check_sdss_files(sdss, run, camcol, field, bandname, filetypes,
					  retrieve=True):
	bandnum = band_index(bandname)
	for filetype in filetypes:
		fn = sdss.getFilename(filetype, run, camcol, field, bandname)
		print 'Looking for file', fn
		if not os.path.exists(fn):
			if retrieve:
				print 'Retrieving', fn
				sdss.retrieve(filetype, run, camcol, field, bandnum)
			else:
				raise os.OSError('no such file: "%s"' % fn)


def get_tractor_sources(run, camcol, field, bandname, release='DR7',
						retrieve=True, curl=False, roi=None):
	'''
	Creates tractor.Source objects corresponding to objects in the SDSS catalog
	for the given field.

	'''
	if release != 'DR7':
		raise RuntimeError('We only support DR7 currently')
	# FIXME
	rerun = 0

	sdss = DR7(curl=curl)
	bandnum = band_index(bandname)
	_check_sdss_files(sdss, run, camcol, field, bandnum,
					  ['tsObj', 'tsField'],
					  #fpC', 'tsField', 'psField', 'fpM'],
					  retrieve=retrieve)

	tsf = sdss.readTsField(run, camcol, field, rerun)


	objs = fits_table(sdss.getFilename('tsObj', run, camcol, field,
									   bandname, rerun=rerun))
	objs.indices = np.arange(len(objs))

	if roi is not None:
		x0,x1,y0,y1 = roi
		# HACK -- keep only the sources whose centers are within the ROI box.
		x = objs.colc[:,bandnum]
		y = objs.rowc[:,bandnum]
		I = ((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
		objs = objs[I]

	objs = objs[(objs.nchild == 0)]

	# NO IDEA why it is NOT necessary to get PA and adjust for it.
	# (probably that getTensor() has the phi transformation in the wrong
	# place, terrifying)
	# Since in DR7, tsObj files have phi_exp, phi_dev in image coordinates,
	# not sky coordinates.
	# Should have to Correct by finding the position angle of the field on
	# the sky.
	# cd = wcs.cdAtPixel(W/2, H/2)
	# pa = np.rad2deg(np.arctan2(cd[0,1], cd[0,0]))
	# print 'pa=', pa
	# HACK -- DR7 phi opposite to Tractor phi, apparently
	objs.phi_dev = -objs.phi_dev
	objs.phi_exp = -objs.phi_exp

	# MAGIC -- minimum size of galaxy.
	objs.r_dev = np.maximum(objs.r_dev, 1./30.)
	objs.r_exp = np.maximum(objs.r_exp, 1./30.)

	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Lgal  = (objs.prob_psf[:,bandnum] == 0)
	Ldev = Lgal * objs.fracpsf[:,bandnum]
	Lexp = Lgal * (1. - objs.fracpsf[:,bandnum])

	sources = []
	ikeep = []

	# Add stars
	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		lup = objs.psfcounts[i,bandnum]
		counts = tsf.luptitude_to_counts(lup, bandnum)
		if counts <= 0:
			print 'Skipping star with luptitude', lup, '-> counts', counts
			continue
		flux = SdssFlux(counts / SdssPhotoCal.scale)
		ps = PointSource(pos, flux)
		sources.append(ps)
		ikeep.append(i)

	# Add galaxies.
	I = np.flatnonzero(Lgal > 0)
	print len(I), 'galaxies'
	ndev, nexp, ncomp = 0, 0, 0
	for i in I:
		hasdev = (Ldev[i] > 0)
		hasexp = (Lexp[i] > 0)
		iscomp = (hasdev and hasexp)
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		# FIXME -- should we really use counts_model if it's purely one
		# or the other model type, or should we use counts_dev/exp ?
		if iscomp:
			lups = objs.counts_model[i,bandnum]
		elif hasdev:
			lups = objs.counts_dev[i,bandnum]
		elif hasexp:
			lups = objs.counts_exp[i,bandnum]
		counts = tsf.luptitude_to_counts(lups, bandnum)
		if counts <= 0:
			print 'Skipping galaxy with luptitude', lups, '-> counts', counts
			continue
											 
		if hasdev:
			dcounts = counts * Ldev[i]
			dflux = SdssFlux(dcounts / SdssPhotoCal.scale)
			re = objs.r_dev[i,bandnum]
			ab = objs.ab_dev[i,bandnum]
			phi = objs.phi_dev[i,bandnum]
			dshape = GalaxyShape(re, ab, phi)
		if hasexp:
			ecounts = counts * Lexp[i]
			eflux = SdssFlux(ecounts / SdssPhotoCal.scale)
			re = objs.r_exp[i,bandnum]
			ab = objs.ab_exp[i,bandnum]
			phi = objs.phi_exp[i,bandnum]
			eshape = GalaxyShape(re, ab, phi)

		if iscomp:
			gal = HoggCompositeGalaxy(pos, eflux, eshape, dflux, dshape)
			ncomp += 1
		elif hasdev:
			#print 'pure deV; counts_model = %g; counts_dev = %g' % (
			#	objs.counts_model[i, bandnum], objs.counts_dev[i, bandnum])
			gal = HoggDevGalaxy(pos, dflux, dshape)
			ndev += 1
		elif hasexp:
			#print 'pure exp; counts_model = %g; counts_exp = %g' % (
			#	objs.counts_model[i, bandnum], objs.counts_exp[i, bandnum])
			gal = HoggExpGalaxy(pos, eflux, eshape)
			nexp += 1
		sources.append(gal)
		ikeep.append(i)
	print 'Created', ndev, 'pure deV', nexp, 'pure exp and',
	print ncomp, 'composite galaxies'

	# if you want to cut the objs list to just the ones for which sources were created...
	ikeep = np.unique(ikeep)
	objs = objs[ikeep]

	return sources

	
def get_tractor_image(run, camcol, field, bandname, release='DR7',
					  retrieve=True, curl=False, roi=None,
					  psf='kl-gm'):
	'''
	Creates a tractor.Image given an SDSS field identifier.

	If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
	in the image, in zero-indexed pixel coordinates.  x1,y1 are
	NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.

	psf can be:
	  "dg" for double-Gaussian
	  "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

	Returns:
	  (tractor.Image, dict)

	dict contains useful details like:
	  'sky'
	  'skysig'
	'''
	# Ugly
	if release != 'DR7':
		raise RuntimeError('We only support DR7 currently')
	valid_psf = ['dg', 'kl-gm']
	if psf not in valid_psf:
		raise RuntimeError('PSF must be in ' + str(valid_psf))
	# FIXME
	rerun = 0

	bandnum = band_index(bandname)

	sdss = DR7(curl=curl)
	_check_sdss_files(sdss, run, camcol, field, bandname,
					  ['fpC', 'tsField', 'psField', 'fpM'],
					  retrieve=retrieve)
	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias
	image = fpC
	(H,W) = image.shape

	if roi is None:
		x0 = y0 = 0
	else:
		x0,x1,y0,y1 = roi

	tsf = sdss.readTsField(run, camcol, field, rerun)
	astrans = tsf.getAsTrans(bandnum)
	wcs = SdssWcs(astrans)
	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

	photocal = SdssPhotoCal()
	psfield = sdss.readPsField(run, camcol, field)
	sky = psfield.getSky(bandnum)
	skysig = sqrt(sky)
	skyobj = ConstantSky(sky)
	info = dict(sky=sky, skysig=skysig)

	fpM = sdss.readFpM(run, camcol, field, bandname)

	gain = psfield.getGain(bandnum)
	darkvar = psfield.getDarkVariance(bandnum)
	skyerr = psfield.getSkyErr(bandnum)
	invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

	if roi is not None:
		roislice = (slice(y0,y1), slice(x0,x1))
		image = image[roislice]
		invvar = invvar[roislice]

	if psf == 'dg':
		dgpsf = psfield.getDoubleGaussian(bandnum)
		print 'Creating double-Gaussian PSF approximation'
		(a,s1, b,s2) = dgpsf
		mypsf = NCircularGaussianPSF([s1, s2], [a, b])
	elif psf == 'kl-gm':
		from emfit import em_fit_2d
		from fitpsf import em_init_params
		
		# Create Gaussian mixture model PSF approximation.
		H,W = image.shape
		klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
		S = klpsf.shape[0]
		# number of Gaussian components
		K = 3
		w,mu,sig = em_init_params(K, None, None, None)
		II = klpsf.copy()
		II /= II.sum()
		# HIDEOUS HACK
		II = np.maximum(II, 0)
		print 'Multi-Gaussian PSF fit...'
		xm,ym = -(S/2), -(S/2)
		em_fit_2d(II, xm, ym, w, mu, sig)
		print 'w,mu,sig', w,mu,sig
		mypsf = GaussianMixturePSF(w, mu, sig)

	timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
					   (run, camcol, field, bandname)))
	return timg,info
		

class SdssPhotoCal(object):
	scale = 1e6
	def __init__(self, scale=None):
		if scale is None:
			scale = SdssPhotoCal.scale
		self.scale = scale
	def fluxToCounts(self, flux):
		'''
		flux: your duck-typed Flux object

		returns: float
		'''
		return flux.getValue() * self.scale
	def countsToFlux(self, counts):
		'''
		counts: float

		Returns: duck-typed Flux object
		'''
		return SdssFlux(counts / self.scale)

class SdssFlux(Flux):
	def getStepSizes(self, img):
		return [1.]
	def __str__(self):
		return 'SdssFlux: %.1f' % (self.val * SdssPhotoCal.scale)
	def __repr__(self):
		return 'SdssFlux(%.1f)' % (self.val * SdssPhotoCal.scale)
	def hashkey(self):
		return ('SdssFlux', self.val)
	def copy(self):
		return SdssFlux(self.val)

class SdssWcs(WCS):
	def __init__(self, astrans):
		self.astrans = astrans
		self.x0 = 0
		self.y0 = 0

	def setX0Y0(self, x0, y0):
		self.x0 = x0
		self.y0 = y0

	# This function is not used by the tractor, and it works in
	# *original* pixel coords (no x0,y0 offsets)
	# (x,y) to RA,Dec in deg
	def pixelToRaDec(self, x, y):
		ra,dec = self.astrans.pixel_to_radec(x, y)
		return ra,dec

	def cdAtPixel(self, x, y):
		return self.astrans.cd_at_pixel(x + self.x0, y + self.y0)

	# RA,Dec in deg to pixel x,y.
	def positionToPixel(self, src, pos):
		## FIXME -- color.
		x,y = self.astrans.radec_to_pixel(pos.ra, pos.dec)
		return x - self.x0, y - self.y0

	# (x,y) to RA,Dec in deg
	def pixelToPosition(self, src, xy):
		## FIXME -- color.
		## NOTE, "src" may be None.
		(x,y) = xy
		ra,dec = self.pixelToRaDec(x + self.x0, y + self.y0)
		return RaDecPos(ra, dec)

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
		if ab < (1./30.):
			print 'Clamping ab from', ab, 'to 1/30'
		if ab > 1.:
			print 'Clamping ab from', ab, 'to 1'
		self.ab = min(1., max(1./30., ab))
	def setphi(self, phi):
		# limit phi to [-180,180]
		#p = np.deg2rad(phi)
		#self.phi = np.rad2deg(np.arctan2(np.sin(p), np.cos(p)))
		self.phi = np.fmod(phi, 360.)
		if self.phi < -180.:
			self.phi += 360.
		if self.phi > 180.:
			self.phi -= 360.

	def setParams(self, p):
		assert(len(p) == 3)
		self.setre(p[0])
		self.setab(p[1])
		self.setphi(p[2])
	def stepParam(self, parami, delta):
		if parami == 0:
			self.setre(self.re + delta)
		if parami == 1:
			self.setab(self.ab + delta)
		if parami == 2:
			self.setphi(self.phi + delta)

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
		T = np.dot(linalg.inv(G), cd)
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

	def getGalaxyPatch(self, img, cx, cy, cd):
		# remember to include margin for psf conv
		profile = self.getProfile()
		if profile is None:
			return None
		(H,W) = img.shape
		margin = int(ceil(img.getPsf().getRadius()))
		T = self.shape.getTensor(cd)
		assert(profile.get_compiled_ab() == 1.)
		# convert re to degrees
		re_deg = self.re / 3600.
		# sqrt(abs(det(cd))) is pixel scale in deg/pix
		det = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
		pixscale = sqrt(abs(det))
		repix = re_deg / pixscale
		(prof,x0,y0) = profile.sample_transform(T, repix, self.ab,
												cx, cy, W, H, margin)
		return Patch(x0, y0, prof)

	def getUnitFluxModelPatch(self, img, px=None, py=None):
		if px is None or py is None:
			(px,py) = img.getWcs().positionToPixel(self, self.getPosition())
		cd = img.getWcs().cdAtPixel(px, py)
		patch = self.getGalaxyPatch(img, px, py, cd)
		if patch is None:
			print 'Warning, is Galaxy(subclass).getProfile() defined?'
			return Patch(0, 0, None)
		if patch.getImage() is None:
			return Patch(patch.getX0(), patch.getY0(), None)
		psf = img.getPsf()
		convimg = psf.applyTo(patch.getImage())
		#print 'PSF-convolved'
		#self.debugPatchImage(convimg)
		return Patch(patch.getX0(), patch.getY0(), convimg)

	def getModelPatch(self, img, px=None, py=None):
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		p1 = self.getUnitFluxModelPatch(img, px, py)
		if p1 is None:
			return None
		return p1 * counts

	# returns [ Patch, Patch, ... ] of length numberOfParams().
	def getParamDerivatives(self, img, fluxonly=False):
		pos0 = self.getPosition()
		(px0,py0) = img.getWcs().positionToPixel(self, pos0)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		patch0 = self.getUnitFluxModelPatch(img, px0, py0)
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
		if fluxonly:
			derivs.extend([None] * len(gsteps))
		else:
			for i in range(len(gsteps)):
				self.shape.stepParam(i, gsteps[i])
				patchx = self.getUnitFluxModelPatch(img, px0, py0)
				self.shape.setParams(oldvals)
				dx = (patchx - patch0) * (counts / gsteps[i])
				dx.setName('d(gal)/d(%s)' % (gnames[i]))
				derivs.append(dx)
		return derivs

class ExpGalaxy(Galaxy):
	profile = None
	@staticmethod
	def getExpProfile():
		if ExpGalaxy.profile is None:
			ExpGalaxy.profile = (
				CompiledProfile(modelname='exp',
								profile_func=profile_exp, re=100, nrad=4))
		return ExpGalaxy.profile
	expnum = 0

	def __init__(self, pos, flux, re, ab, phi):
		Galaxy.__init__(self, pos, flux, GalaxyShape(re, ab, phi))
		self.name = 'ExpGalaxy'
		self.num = ExpGalaxy.expnum
		ExpGalaxy.expnum += 1
		self.plotnum = 0

	def getProfile(self):
		return ExpGalaxy.getExpProfile()

	def copy(self):
		return ExpGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

	def debugPatchImage(self, img):
		if img is None:
			print 'Exp patch', img
		elif np.product(img.shape) == 0:
			print 'Patch empty:', img.shape
		else:
			print 'Patch', img.shape
			plt.clf()
			plt.imshow(img, interpolation='nearest', origin='lower')
			plt.hot()
			plt.colorbar()
			fn = 'exp-patch-%02i-%03i.png' % (self.num, self.plotnum)
			plt.savefig(fn)
			print 'saved', fn
			self.plotnum += 1

class DevGalaxy(Galaxy):
	profile = None
	@staticmethod
	def getDevProfile():
		if DevGalaxy.profile is None:
			DevGalaxy.profile = (
				CompiledProfile(modelname='dev',
								profile_func=profile_dev, re=100, nrad=8))
		return DevGalaxy.profile

	def __init__(self, pos, flux, re, ab, phi):
		Galaxy.__init__(self, pos, flux, GalaxyShape(re, ab, phi))
		self.name = 'DevGalaxy'

	def getProfile(self):
		return DevGalaxy.getDevProfile()

	def copy(self):
		return DevGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

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
	def hashkey(self):
		return (self.name, self.pos.hashkey(),
				self.fluxExp.hashkey(), self.shapeExp.hashkey(),
				self.fluxDev.hashkey(), self.shapeDev.hashkey())
	def __str__(self):
		return (self.name + ' at ' + str(self.pos)
				+ ' with Exp ' + str(self.fluxExp) + str(self.shapeExp)
				+ ' and deV ' + str(self.fluxDev) + str(self.shapeDev))
	def __repr__(self):
		return (self.name + '(pos=' + repr(self.pos) +
				', fluxExp=' + repr(self.fluxExp) +
				', shapeExp=' + repr(self.shapeExp) + 
				', fluxDev=' + repr(self.fluxDev) +
				', shapeDev=' + repr(self.shapeDev))
	def copy(self):
		return None
	def getModelPatch(self, img, px=None, py=None):
		pass
	def getParamDerivatives(self, img, fluxonly=False):
		pass

# BUG: shouldn't this function cache the HoggDevGalaxy() and HoggExpGalaxy() calls?
class HoggCompositeGalaxy(CompositeGalaxy):
	def getName(self):
		return 'HoggCompositeGalaxy'
	def copy(self):
		return HoggCompositeGalaxy(self.pos, self.fluxExp, self.shapeExp,
								   self.fluxDev, self.shapeDev)
	def getModelPatch(self, img, px=None, py=None):
		e = HoggExpGalaxy(self.pos, self.fluxExp, self.shapeExp)
		d = HoggDevGalaxy(self.pos, self.fluxDev, self.shapeDev)
		pe = e.getModelPatch(img, px, py)
		pd = d.getModelPatch(img, px, py)
		if pe is None:
			return pd
		if pd is None:
			return pe
		return pe + pd

	def getUnitFluxModelPatch(self, img, px=None, py=None):
		fe = self.fluxExp / (self.fluxExp + self.fluxDev)
		fd = 1. - fe
		assert(fe >= 0.)
		assert(fe <= 1.)
		e = HoggExpGalaxy(self.pos, fe, self.shapeExp)
		d = HoggDevGalaxy(self.pos, fd, self.shapeDev)
		pe = e.getModelPatch(img, px, py)
		pd = d.getModelPatch(img, px, py)
		if pe is None:
			return pd
		if pd is None:
			return pe
		return pe + pd

	# couldn't this function move up to CompositeGalaxy ?
	# MAGIC: ORDERING OF EXP AND DEV PARAMETERS
	# MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
	def getParamDerivatives(self, img, fluxonly=False):
		e = HoggExpGalaxy(self.pos, self.fluxExp, self.shapeExp)
		d = HoggDevGalaxy(self.pos, self.fluxDev, self.shapeDev)
		de = e.getParamDerivatives(img, fluxonly)
		dd = d.getParamDerivatives(img, fluxonly)
		npos = len(self.pos.getStepSizes(img))
		derivs = []
		for i in range(npos):
			derivs.append(de[i] + dd[i])
		derivs.extend(de[npos:])
		derivs.extend(dd[npos:])
		print 'getParamDerivatives: derivs', len(derivs)
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


class HoggExpGalaxy(HoggGalaxy):
	profile = mp.get_exp_mixture()
	profile.normalize()
	@staticmethod
	def getExpProfile():
		return HoggExpGalaxy.profile
	#def __init__(self, pos, flux, re, ab, phi):
	#	HoggGalaxy.__init__(self, pos, flux, re, ab, phi)
	def getName(self):
		return 'HoggExpGalaxy'
	def getProfile(self):
		return HoggExpGalaxy.getExpProfile()
	def copy(self):
		return HoggExpGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

class HoggDevGalaxy(HoggGalaxy):
	profile = mp.get_dev_mixture()
	profile.normalize()
	@staticmethod
	def getDevProfile():
		return HoggDevGalaxy.profile
	#def __init__(self, pos, flux, re, ab, phi):
	#	HoggGalaxy.__init__(self, pos, flux, re, ab, phi)
	def getName(self):
		return 'HoggDevGalaxy'
	def getProfile(self):
		return HoggDevGalaxy.getDevProfile()
	def copy(self):
		return HoggDevGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

class Changes(object):
	pass

class SDSSTractor(Tractor):

	def __init__(self, *args, **kwargs):
		self.debugnew = kwargs.pop('debugnew', False)
		self.debugchange = kwargs.pop('debugchange', False)

		Tractor.__init__(self, *args, **kwargs)
		self.newsource = 0
		self.changes = []
		self.changei = 0

		self.plotfns = []
		self.comments = []
		self.boxes = []

	def debugChangeSources(self, **kwargs):
		if self.debugchange:
			self.doDebugChangeSources(**kwargs)

	def doDebugChangeSources(self, step=None, src=None, newsrcs=None, alti=0,
							 dlnprob=0, **kwargs):
		if step == 'start':
			ch = self.changes = Changes()
			N = self.getNImages()
			ch.src = src
			ch.N = N
			ch.impatch = [None for i in range(N)]
			ch.mod0    = [None for i in range(N)]
			ch.mod0type = src.getSourceType()
			ch.newmods = []

			for imgi in range(N):
				img = self.getImage(imgi)
				mod = self.getModelPatch(img, src)
				ch.mod0[imgi] = mod
				print 'image', imgi, 'got model patch', mod
				if mod.getImage() is not None:
					impatch = img.getImage()[mod.getSlice(img)]
					if len(impatch.ravel()):
						ch.impatch[imgi] = impatch

		elif step in ['init', 'opt1']:
			ch = self.changes
			if newsrcs == []:
				return
			mods = []
			for imgi in range(ch.N):
				img = self.getImage(imgi)
				mod = self.getModelPatch(img, newsrcs[0])
				mods.append(mod)

			if step == 'init':
				ch.newmods.append([newsrcs[0].getSourceType(),
								   mods])
			else:
				ch.newmods[-1].extend([mods, dlnprob])

		elif step in ['switch', 'keep']:
			ch = self.changes
			M = len(ch.newmods)
			N = ch.N
			cols = M+2
			II = [i for i in range(N) if ch.impatch[i] is not None]
			rows = len(II)
			fs = 10

			imargs = {}
			plt.clf()
			# Images
			for ri,i in enumerate(II):
				img = self.getImage(i)
				sky = img.getSky().val
				skysig = sqrt(sky)
				imargs[i] = dict(vmin=-3.*skysig, vmax=10.*skysig)
				if ch.impatch[i] is None:
					continue
				plt.subplot(rows, cols, ri*cols+1)
				plotimage(ch.impatch[i] - sky, **imargs[i])
				plt.xticks([])
				plt.yticks([])
				plt.title('image %i' % i, fontsize=fs)

			# Original sources
			for ri,i in enumerate(II):
				if ch.mod0[i].getImage() is None:
					continue
				plt.subplot(rows, cols, ri*cols+2)
				plotimage(ch.mod0[i].getImage(), **imargs[i])
				plt.xticks([])
				plt.yticks([])
				plt.title('original ' + ch.mod0type, fontsize=fs)

			# New sources
			for j,newmod in enumerate(ch.newmods):
				(srctype, premods, postmods, dlnp) = newmod
				for ri,i in enumerate(II):
					if postmods[i] is None:
						continue
					plt.subplot(rows, cols, ri*cols + 3 + j)

					# HACK -- force patches to be the same size + offset...
					img = self.getImage(i)
					sl = ch.mod0[i].getSlice(img)
					#print 'slice', sl
					im = np.zeros_like(img.getImage())
					postmods[i].addTo(im)
					im = im[sl]
					if len(im.ravel()):
						plotimage(im, **imargs[i])
						plt.xticks([])
						plt.yticks([])
						plt.title(srctype + ' (dlnp=%.1f)' % dlnp, fontsize=fs)
				
			fn = 'change-%03i.png' % self.changei
			plt.savefig(fn)
			print 'Wrote', fn
			self.changei += 1
			self.plotfns.append(fn)
			if step == 'switch':
				s = '<a href="#%s">' % fn + 'accepted change</a> from ' + str(src) + '<br />to '
				if len(newsrcs) == 1:
					s += str(newsrcs[0])
				else:
					s += '[ ' + ' + '.join([str(ns) for ns in newsrcs]) + ' ]'
				self.comments.append(s)
			elif step == 'keep':
				s = '<a href="#%s">' % fn + 'rejected change</a> of ' + str(src)
				self.comments.append(s)
			#smallimg = 'border="0" width="400" height="300"'
			#s += '<a href="%s"><img src="%s" %s /></a>' % (fn, fn, smallimg)
				
				
	def debugNewSource(self, *args, **kwargs):
		if self.debugnew:
			self.doDebugNewSource(*args, **kwargs)

	def doDebugNewSource(self, *args, **kwargs):
		step = kwargs.get('type', None)
		if step in [ 'newsrc-0', 'newsrc-opt' ]:
			if step == 'newsrc-0':
				optstep = 0
				self.newsource += 1
			else:
				optstep = 1 + kwargs['step']
			src = kwargs['src']
			img = kwargs['img']

			patch = src.getModelPatch(img)
			imgpatch = img.getImage()[patch.getSlice(img)]

			plt.clf()
			plt.subplot(2,3,4)
			plotimage(imgpatch)
			cl = plt.gci().get_clim()
			plt.colorbar()
			plt.title('image patch')
			plt.subplot(2,3,5)
			plotimage(patch.getImage(), vmin=cl[0], vmax=cl[1])
			plt.colorbar()
			plt.title('new source')
			derivs = src.getParamDerivatives(img)
			assert(len(derivs) == 3)
			for i,deriv in enumerate(derivs):
				plt.subplot(2,3,i+1)
				plotimage(deriv.getImage())
				cl = plt.gci().get_clim()
				mx = max(abs(cl[0]), abs(cl[1]))
				plt.gci().set_clim(-mx, mx)
				plt.colorbar()
				plt.title(deriv.name)
			fn = 'newsource-%02i-%02i.png' % (self.newsource, optstep)
			plt.savefig(fn)
			print 'Wrote', fn
			

	def createNewSource(self, img, x, y, ht):
		wcs = img.getWcs()
		pos = wcs.pixelToPosition(None, (x,y))
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		patch = img.getPsf().getPointSourcePatch(x, y)
		ht /= patch.getImage().max()
		photocal = img.getPhotoCal()
		flux = photocal.countsToFlux(ht)
		ps = PointSource(pos, flux)
		try:
			imgi = self.images.index(img)
			patch = self.getModelPatch(img, ps)
			self.addBox(imgi, patch.getExtent())
		except:
			pass
		return ps

	def addBox(self, imgi, box):
		if len(self.boxes) == 0:
			self.boxes = [[] for i in range(self.getNImages())]
		self.boxes[imgi].append(box)

	def changeSourceTypes(self, srcs=None, **kwargs):
		if srcs is not None:
			for i,img in enumerate(self.getImages()):
				for src in srcs:
					patch = self.getModelPatch(img, src)
					self.addBox(i, patch.getExtent())
		Tractor.changeSourceTypes(self, srcs, **kwargs)


	def changeSource(self, source):
		'''
		Proposes a list of alternatives, where each alternative is a list of new
		Sources that the given Source could be changed into.
		'''
		if isinstance(source, PointSource):
			eg = ExpGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   1., 0.5, 0.)
			dg = DevGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   1., 0.5, 0.)
			#print 'Changing:'
			#print '  from ', source
			#print '  into', eg
			return [ [], [eg], [dg] ]

		elif isinstance(source, ExpGalaxy):
			dg = DevGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getFlux().copy())
			return [ [], [ps], [dg] ]

		elif isinstance(source, DevGalaxy):
			eg = ExpGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getFlux().copy())
			return [ [], [ps], [eg] ]

		else:
			print 'unknown source type for', source
			return []


def choose_field2():
	import astrometry.libkd.spherematch as spherematch

	fields = fits_table('window_flist.fits')
	print len(ngc2000), 'NGC/IC objects'
	goodngc = [n for n in ngc2000 if n.get('classification', None) == 'Gx']
	print len(goodngc), 'NGC galaxies'

	nra  = np.array([n['ra']  for n in goodngc])
	ndec = np.array([n['dec'] for n in goodngc])

	rad = 8./60.
	(I,J,dist) = spherematch.match_radec(nra, ndec, fields.ra, fields.dec, rad)

	#sdss = DR7()

	for i in np.unique(I):
		ii = (I == i)
		n = goodngc[i]
		isngc = n['is_ngc']
		num = n['id']
		if (sum(ii) > 10) & (n['dec'] > 2.0):
			print '<p>'
			print ('NGC' if isngc else 'IC'), num, 'has', sum(ii), 'fields, and is at RA = ', n['ra'], 'Dec=', n['dec']
			print '</p>'

			ff = fields[J[ii]]
			print 'rcfi = [',
			for f in ff:
				print '(', f.run, ',', f.camcol, ',', f.field, ',', f.incl, ')',
			print ']'
			for f in ff:
				print 'Incl', f.incl, '<br />'
				print ('<img src="%s" /><br /><br />' %
					   ('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpegcodec.aspx?R=%i&C=%i&F=%i&Z=25' % (f.run, f.camcol, f.field)))
				print '<br />'

	# NGC 2511 has 12 fields, and is at RA = 120.575 Dec= 9.4 

def testGalaxy():
	W,H = 200,200
	pos = PixPos(100, 100)
	flux = Flux(1000.)

	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1.
	wcs = NullWCS()
	photocal = NullPhotoCal()
	psf = NCircularGaussianPSF([1.5], [1.0])
	sky = ConstantSky(0.)
	
	img = Image(data=image, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
				photocal=photocal)
	
	eg = ExpGalaxy(pos, flux, 10., 0.5, 30.)
	patch = eg.getModelPatch(img)

	imargs1 = dict(interpolation='nearest', origin='lower')

	plt.clf()
	plt.imshow(patch.getImage(), **imargs1)
	plt.colorbar()
	plt.savefig('eg-1.png')
	
	derivs = eg.getParamDerivatives(img)
	for i,deriv in enumerate(derivs):
		plt.clf()
		plt.imshow(deriv.getImage(), **imargs1)
		plt.colorbar()
		plt.title('derivative ' + deriv.getName())
		plt.savefig('eg-deriv%i-0a.png' % i)

def plotimage(img, setcolormap=True, **kwargs):
	args = dict(interpolation='nearest', origin='lower')
	args.update(kwargs)
	plt.imshow(img, **args)
	if setcolormap:
		plt.hot()
	#plt.colorbar()

def plotfootprints(radecs, radecrange=None, catalog=None, labels=None):
	# FIXME -- RA=0 wrap-around
	for i,rd in enumerate(radecs):
		plt.plot([r for r,d in rd], [d for r,d in rd], 'b-')
		# blue dot at (0,0)
		plt.plot([rd[0][0]], [rd[0][1]], 'bo')
		# red dot at (W,0)
		plt.plot([rd[1][0]], [rd[1][1]], 'ro')
		plt.gca().add_artist(matplotlib.patches.Polygon(
			rd, ec='0.8', fc='0.8', fill=True, alpha=0.1))
		if labels is None:
			lab = '%i' % i
		else:
			lab = labels[i]
		plt.text(rd[0][0], rd[0][1], lab)

	if radecrange is None:
		radecrange = plt.axis()

	if catalog is not None:
		r,d = [],[]
		for src in catalog:
			rd = src.getPosition()
			r.append(rd.ra)
			d.append(rd.dec)
		# FIXME -- plot ellipses for galaxies?  tune-up.py has code...
		plt.plot(r, d, 'b+')
	setRadecAxes(*radecrange)
	return radecrange


def prepareTractor(initialPlots=False, useSimplexy=True, rcfcut=None):
	rcfi = [ ( 5194 , 2 , 44 , 22.500966 ), ( 4275 , 2 , 224 , 90.003437 ), ( 3638 , 2 , 209 , 90.002781 ), ( 4291 , 2 , 227 , 90.003589 ), ( 4275 , 2 , 225 , 90.003437 ), ( 5849 , 4 , 27 , 20.003216 ), ( 5803 , 5 , 41 , 19.990683 ), ( 5194 , 2 , 43 , 22.500966 ), ( 3638 , 2 , 210 , 90.002781 ), ( 5803 , 5 , 42 , 19.990683 ), ( 5925 , 5 , 30 , 19.933986 ), ( 5935 , 5 , 27 , 20.000022 ), ]			
	rcf = [(r,c,f) for r,c,f,i in rcfi if i < 85]
	print 'RCF', rcf

	sdss = DR7()

	bandname = 'i'

	if False:
		from astrometry.util import sdss_das as das
		from astrometry.util.sdss_filenames import sdss_filename
		for r,c,f in rcf:
			for filetype in ['fpC', 'fpM', 'psField', 'tsField']:
				fn = sdss_filename(filetype, r, c, f, band=bandname)
				print 'Need', fn
				#if not os.path.exists(fn):
				print 'Getting from DAS'
				das.sdss_das_get(filetype, fn, r, c, f, band=bandname)

	# we only got some of them...
	rcf = [ (5194, 2, 44), (5194, 2, 43), (5849, 4, 27), (5935, 5, 27) ]
	rcf = [rcf[0], rcf[2]]
	print 'RCF', rcf

	rois = [
		# Mostly overlapping:
		#( 0, 1000, 0, 600 ),
		#( 1000, 2000, 600, 1200 ),

		# Pick up a big galaxy
		#( 600, 1600, 0, 600 ),
		#( 1000, 2000, 600, 1200 ),

		( 800, 1600, 0, 600 ),
		( 1200, 2000, 600, 1200 ),

		# Avoid that big galaxy (that was keeping us honest)
		#( 800, 1300, 0, 500 ),
		#( 1500, 2000, 600, 1100 ),

		]
	fullsizes = []

	print 'Reading SDSS input files...'

	band = band_index(bandname)

	images = []
	zrange = []
	nziv = []

	# FIXME -- bug-bug annihilation
	rerun = 0

	simplexys = []

	if rcfcut is not None:
		rcf  = [rcf [i] for i in rcfcut]
		rois = [rois[i] for i in rcfcut]
	
	for i,(run,camcol,field) in enumerate(rcf):
		fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
		fpC = fpC.astype(float) - sdss.softbias
		image = fpC

		if useSimplexy:
			fpcfn = sdss.getFilename('fpC', run, camcol, field, bandname)
			xyfn = fpcfn.replace('.fit', '.xy')
			if not os.path.exists(xyfn):
				print 'Running image2xy...'
				cmd = 'image2xy %s -o %s' % (fpcfn, xyfn)
				print 'Command:', cmd
				os.system(cmd)
			assert(os.path.exists(xyfn))
			xy = fits_table(xyfn)
			simplexys.append(xy)

		fullsizes.append(image.shape)
	
		psfield = sdss.readPsField(run, camcol, field)
		gain = psfield.getGain(band)
		darkvar = psfield.getDarkVariance(band)
		sky = psfield.getSky(band)
		skyerr = psfield.getSkyErr(band)
		skysig = sqrt(sky)

		fpM = sdss.readFpM(run, camcol, field, bandname)

		tsfield = sdss.readTsField(run, camcol, field, rerun)

		invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

		nz = np.sum(invvar != 0)
		#print 'Non-zero invvars:', nz
		nziv.append(nz)
		
		zr = np.array([-3.,+10.]) * skysig + sky
		zrange.append(zr)

		x0,x1,y0,y1 = rois[i]

		if initialPlots:
			print 'Initial plots...'
			plt.clf()
			plotimage(image, vmin=zr[0], vmax=zr[1])
			ax = plt.axis()
			plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'b-')
			plt.axis(ax)
			plt.savefig('fullimg-%02i.png' % i)

		roislice = (slice(y0,y1), slice(x0,x1))
		image = image[roislice]
		invvar = invvar[roislice]

		if initialPlots:
			plt.clf()
			plotimage(image, vmin=zr[0], vmax=zr[1])
			plt.savefig('img-%02i.png' % i)

		dgpsf = psfield.getDoubleGaussian(band)
		print 'Creating double-Gaussian PSF approximation'
		print '  ', dgpsf
		(a,s1, b,s2) = dgpsf
		psf = NCircularGaussianPSF([s1, s2], [a, b])

		wcs = SdssWcs(tsfield.getAsTrans(bandname))
		wcs.setX0Y0(x0, y0)
		# And counts
		photocal = SdssPhotoCal(SdssPhotoCal.scale)
		skyobj = ConstantSky(sky)

		img = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
					sky=skyobj, photocal=photocal,
					name='Image%i(r/c/f=%i/%i%i)' % (i, run, camcol, field))
		images.append(img)

	print 'Creating footprint image...'
	radecs = []
	for i,img in enumerate(images):
		# Find full-size and ROI boxes
		wcs = img.getWcs()
		(H,W) = fullsizes[i]
		x0,x1,y0,y1 = rois[i]
		corners = [ (0,0), (W,0), (W,H), (0,H), (0,0) ]
		rds = [wcs.pixelToRaDec(x,y) for x,y in corners]
		radecs.append(rds)
		corners = [ (x0,y0), (x1,y0), (x1,y1), (x0,y1), (x0,y0) ]
		rds = [wcs.pixelToRaDec(x,y) for x,y in corners]
		radecs.append(rds)

	if initialPlots:
		plt.clf()
		plotfootprints(radecs, labels=['%i'%(i/2) for i in range(len(radecs))])
		plt.savefig('footprints-full.png')
	# After making the full "footprints" image, trim the list down to just the ROIs
	footradecs = radecs[1::2]

	return (images, simplexys, rois, zrange, nziv, footradecs)

def main():
	from optparse import OptionParser

	#testGalaxy()
	#choose_field2()

	parser = OptionParser()
	parser.add_option('-l', '--load', dest='loadi', type='int',
					  default=-1, help='Load catalog from step #...')
	parser.add_option('-i', '--no-initial-plots', dest='initialplots', default=True,
					  action='store_false')
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')
	opt,args = parser.parse_args()

	if opt.verbose == 0:
		lvl = logging.INFO
	else: # opt.verbose == 1:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	use_simplexy = True

	plt.figure(figsize=(10,7.5))

	(images, simplexys, rois, zrange, nziv, footradecs
	 ) = prepareTractor(opt.initialplots, use_simplexy)

	print 'Firing up tractor...'
	tractor = SDSSTractor(images, debugnew=False, debugchange=True)

	footradecrange = None

	batchsource = 10
	batchchange =  5

	np.random.seed(42)

	steps = (['plots'] +
			 ['simplesources', 'plots'] +
			 ['changebiased'] +
			 ['sky', 'plots'] +
			 ['flux', 'plots', 'opt', 'plots', 'save'] +
			 ['psfup', 'flux', 'sky', 'plots', 'save'] * 4 +
			 (['source2', 'plots', 'save'] +
			  ['changebiased', 'plots', 'save']*3 +
			  ['opt','plots', 'save'])*10 +
			 [])
	ploti = 0
	savei = 0
	stepi = 0

	# JUMP IN:
	if opt.loadi != -1:
		loadi = opt.loadi
		(savei, stepi, ploti, tractor.catalog, psfs, skys) = unpickle_from_file('catalog-%02i.pickle' % loadi)
		for i in range(tractor.getNImages()):
			tractor.getImage(i).setPsf(psfs[i])
			tractor.getImage(i).setSky(skys[i])
		print 'Starting from step', stepi
		print 'there are', len(steps), 'steps'
		print 'remaining steps:', steps[stepi:]

		# HACK
		print 'REPLACING STEPS:'
		steps = ( ['']*(stepi) +
				  [ 'changeall' ]
				  )
		print steps
		print 'there are', len(steps), 'steps'
		print 'remaining steps:', steps[stepi:]
				  
		

	chiAimargs = []

	changenext = 0

	stepi -= 1
	while True:
		stepi += 1
		if stepi >= len(steps):
			break
		step = steps[stepi]

		print
		print '-----------------------------'
		print 'Step', stepi, ':', step
		print '-----------------------------'
		print

		if step == 'changeall':
			Nsrcs = len(tractor.getCatalog())
			changenext = 0
			# add as many "changenext" steps as necessary
			Nsteps = int(ceil(Nsrcs / float(batchchange)))
			addsteps = ['changenext', 'plots', 'save'] * Nsteps
			steps = steps[:stepi+1] + addsteps + steps[stepi+1:]
			print 'modified steps array:', steps

		elif step == 'changenext':
			print 'Changing next batch of sources.'
			cat = tractor.getCatalog()
			srcis = range(changenext, len(cat))
			if len(srcis) == 0:
				continue
			srcis = srcis[:batchchange]
			srcs = [cat[i] for i in srcis]
			tractor.changeSourceTypes(srcs)
			changenext += batchchange

		elif step == 'plots':
			print 'Making plots...'
			NS = len(tractor.getCatalog())

			chis = tractor.getChiImages()
			mods = tractor.getModelImages()
			fns = []
			for i in range(len(chis)):
				chi = chis[i]
				mod = mods[i]
				img = tractor.getImage(i)
				tt = 'sources: %i, random value = %g' % (NS, np.sum(chi**2)/float(nziv[i]))
				zr = zrange[i]
				imargs = dict(interpolation='nearest', origin='lower',
							  vmin=zr[0], vmax=zr[1])

				plt.clf()
				plotimage(mod, **imargs)
				# Want x marks on the source centers?
				if False:
					ax = plt.axis()
					wcs = img.getWcs()
					x = []
					y = []
					for src in tractor.getCatalog():
						pos = src.getPosition()
						px,py = wcs.positionToPixel(src, pos)
						x.append(px)
						y.append(py)
					plt.plot(x, y, 'bo', mfc='none', mec='b')
					plt.axis(ax)

				if len(tractor.boxes) > i:
					boxes = tractor.boxes[i]
					ax = plt.axis()
					for x0,x1,y0,y1 in boxes:
						plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], 'b-')
					plt.axis(ax)

				plt.title(tt)
				fn = 'mod-%02i-%02i.png' % (ploti, i)
				plt.savefig(fn)
				print 'Wrote', fn
				fns.append(fn)

				if len(chiAimargs) <= i:
					mn,mx = (chi.min(), chi.max())
					chiAimargs.append(
						dict(interpolation='nearest', origin='lower',
							 vmin=mn, vmax=mx))
				chiAimarg = chiAimargs[i]

				plt.clf()
				plotimage(chi, **chiAimarg)
				plt.title(tt)
				fn = 'chiA-%02i-%02i.png' % (ploti, i)
				plt.savefig(fn)
				print 'Wrote', fn
				fns.append(fn)

				chiBimarg = dict(interpolation='nearest', origin='lower',
								 cmap=redgreen, vmin=-5, vmax=5)
								 #vmin=-3, vmax=10.)

				plt.clf()
				plotimage(chi, setcolormap=False, **chiBimarg)
				plt.title(tt)
				fn = 'chiB-%02i-%02i.png' % (ploti, i)
				plt.savefig(fn)
				print 'Wrote', fn
				fns.append(fn)

			tractor.boxes = []

			plt.clf()
			footradecrange = plotfootprints(footradecs, footradecrange,
											tractor.getCatalog())
			fn = 'footprints-%02i.png' % (ploti)
			plt.savefig(fn)
			print 'Wrote', fn
			footfn = fn

			html = '<html><head><title>Step %i</title></head><body>' % ploti
			html += '<h3><a href="step%02i.html">Previous</a> &nbsp;' % (ploti-1)
			html += '<a href="step%02i.html">Next</a> &nbsp;' % (ploti+1)
			html += 'Step %i' % (ploti)
			lastplot = max(0, stepi-1)
			while lastplot > 0:
				if steps[lastplot] == 'plots':
					break
				lastplot -= 1
			html += ' (%s)' % (', '.join(steps[lastplot+1:stepi]))
			t = datetime.now()
			html += ' at ' + t.isoformat()
			html += '</h3>\n'

			for txt in tractor.comments:
				html += txt + '<br />'
			tractor.comments = []

			html += 'PSF models: <ul>'
			for img in tractor.getImages():
				html += '<li>' + str(img.getPsf()) + '</li>'
			html += '</ul>\n'

			smallimg = 'border="0" width="400" height="300"'
			for i,img in enumerate(tractor.getImages()):
				imgfn = 'img-%02i.png' % i 
				# img
				html += '<br />'
				# mod, chiB
				for fn in [imgfn, fns[i*3 + 0], fns[i*3 + 2]]:
					html += '<a href="%s"><img src="%s" %s /></a>' % (fn, fn, smallimg)
				html += '\n'
			for fn in tractor.plotfns:
				html += '<br />'
				html += '<a name="%s" />' % fn
				html += '<a href="%s"><img src="%s" %s /></a>' % (fn, fn, smallimg)
			tractor.plotfns = []
			html += '<br />'
			fn = footfn
			html += '<a href="%s"><img src="%s" %s /></a>' % (fn, fn, smallimg)
			html += '</body></html>'
			write_file(html, 'step%02i.html' % ploti)
			
			ploti += 1

		elif step == 'sky':
			print 'Optimizing sky...'
			for i in range(tractor.getNImages()):
				tractor.optimizeSkyAtFixedComplexityStep(i)

		elif step == 'opt':
			print 'Optimizing catalog...'
			tractor.optimizeCatalogAtFixedComplexityStep()

		elif step == 'source':
			rtn = tractor.createSource(nbatch=batchsource)

		elif step == 'source2':
			rtn = tractor.createSource(nbatch=batchsource,
									   avoidExisting=False)

		elif step == 'simplesources':
			print "Initializing with simplexy's source lists..."
			cat = tractor.getCatalog()
			for sxy,img,roi in zip(simplexys, tractor.getImages(), rois):
				print 'Making mask image...'
				# Mask out a small region around each existing source.
				mask = np.zeros_like(img.getImage()).astype(bool)
				wcs = img.getWcs()
				for src in cat:
					(px,py) = wcs.positionToPixel(src, src.getPosition())
					r = 2
					(H,W) = img.shape
					xlo = max(px-r, 0)
					xhi = min(px+r, W)
					ylo = max(py-r, 0)
					yhi = min(py+r, H)
					mask[ylo:yhi, xlo:xhi] = True

				print 'Simplexy has', len(sxy), 'sources'
				(x0,x1,y0,y1) = roi
				I = (sxy.x >= x0) * (sxy.x <= x1) * (sxy.y >= y0) * (sxy.y <= y1)
				sxy = sxy[I]
				print 'Keeping', len(sxy), 'in bounds'
				for i in range(len(sxy)):
					# MAGIC -1: simplexy produces FITS-convention coords
					x = sxy.x[i] - x0 - 1.
					y = sxy.y[i] - y0 - 1.
					ix = int(round(x))
					iy = int(round(y))
					if mask[iy,ix]:
						#print 'Skipping masked source at', x,y
						continue
					src = tractor.createNewSource(img, x, y, sxy.flux[i])
					cat.append(src)

		elif step == 'flux':
			tractor.optimizeCatalogFluxes()

		elif step == 'psf':
			baton = (stepi,)
			tractor.optimizeAllPsfAtFixedComplexityStep()

		elif step == 'psfup':
			tractor.increaseAllPsfComplexity()

		elif step == 'change':
			tractor.changeSourceTypes()

		elif step == 'change1':
			print 'Changing one source.'
			srci = int(random.random() * len(tractor.getCatalog()))
			srcs = [tractor.getCatalog()[srci]]
			tractor.changeSourceTypes(srcs)

		elif step == 'changebiased':
			cat = tractor.getCatalog()
			chis = tractor.getChiImages()
			imgs = tractor.getImages()

			scalars = []
			for src in cat:
				scalar = 0
				for img,chi in zip(imgs,chis):
					wcs = img.getWcs()
					(px,py) = wcs.positionToPixel(src, src.getPosition())
					r = 5
					(H,W) = img.shape
					if px < -r or px > (W+r) or py < -r or py > (H+r):
						continue
					xlo = np.clip(px-r, 0, W)
					xhi = np.clip(px+r, 0, W)
					ylo = np.clip(py-r, 0, H)
					yhi = np.clip(py+r, 0, H)
					c = chi[ylo:yhi, xlo:xhi]
					# positive chi
					scalar += (c[c > 0]**2).sum()
				scalars.append(scalar)
			scalars = np.array(scalars)
			srcis = []
			while len(srcis) < batchchange and len(srcis) < len(cat):
				# draw N
				N = batchchange - len(srcis)
				X = np.random.multinomial(N, scalars/np.sum(scalars))
				# find the elements that are set
				newsrcis = np.flatnonzero(X)
				scalars[newsrcis] = 0
				srcis.extend(newsrcis)

			srcs = [cat[i] for i in srcis]
			changed = tractor.changeSourceTypes(srcs=srcs)

		elif step == 'save':

			psfs = []
			skys = []
			for i in range(tractor.getNImages()):
				psfs.append(tractor.getImage(i).getPsf())
				skys.append(tractor.getImage(i).getSky())

			pickle_to_file((savei, stepi+1, ploti, tractor.catalog,
							psfs, skys),
						   'catalog-%02i.pickle' % savei)
			savei += 1
			
		print 'Tractor cache has', len(tractor.cache), 'entries'

if __name__ == '__main__':
	main()
