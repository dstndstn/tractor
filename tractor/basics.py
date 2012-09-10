"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`basics.py`
===========

Generally useful generic implementations of things the Tractor needs:
Magnitudes, (RA,Dec) positions, FITS WCS, and so on.

"""
from math import ceil, floor, pi, sqrt, exp

from engine import *
#from ducks import *
from utils import *
import mixture_profiles as mp
import numpy as np

class Mags(ParamList):
	'''
	An implementation of `Brightness` that stores magnitudes in
	multiple bands.
	'''
	def __init__(self, **kwargs):
		'''
		Mags(r=14.3, g=15.6, order=['r','g'])

		The `order` parameter is optional; it determines the ordering
		of the bands in the parameter vector (eg, `getParams()`).
		'''
		keys = kwargs.pop('order', None)
		if keys is None:
			keys = kwargs.keys()
			keys.sort()
		assert(len(kwargs) == len(keys))
		assert(set(kwargs.keys()) == set(keys))
		vals = []
		for k in keys:
			vals.append(kwargs[k])
		super(Mags,self).__init__(*vals)
		self.order = keys
		self.addNamedParams(**dict((k,i) for i,k in enumerate(keys)))

	def getMag(self, bandname):
		''' Bandname: string
		Returns: mag in the given band.
		'''
		return getattr(self, bandname)
	def __add__(self, other):

		# mags + 0.1
		if np.isscalar(other):
			kwargs = {}
			for band in self.order:
				m1 = self.getMag(band)
				kwargs[band] = m1 + other
			return Mags(order=self.order, **kwargs)

		# ASSUME some things about calibration here...
		kwargs = {}
		for band in self.order:
			m1 = self.getMag(band)
			m2 = other.getMag(band)
			msum = -2.5 * np.log10( 10.**(-m1/2.5) + 10.**(-m2/2.5) )
			kwargs[band] = msum
		return Mags(order=self.order, **kwargs)

	def copy(self):
		return self*1.
	  
	def __mul__(self, factor):
		# Return the magnitude that corresponds to the flux rescaled by factor.
			# Flux is positive (and log(-ve) is not permitted), so we take the abs
			# of the input scale factor to prevent embarrassment.
			# Negative magnifications appear in gravitational lensing, but they just label
		# the "parity" of the source, not its brightness. So, we treat factor=-3 the 
		# same as factor=3, for example.
		kwargs = {}
		for band in self.order:
			m = self.getMag(band)
			mscaled = m - 2.5 * np.log10( np.abs(factor) )
			kwargs[band] = mscaled
		return Mags(order=self.order, **kwargs)

	def __setstate__(self, state):
		'''For pickling.'''
		self.__dict__ = state
		self.addNamedParams(**dict((k,i) for i,k in enumerate(self.order)))


class Fluxes(Mags):
	'''
	An implementation of `Brightness` that stores fluxes in multiple
	bands.
	'''
	getBand = Mags.getMag
	getFlux = Mags.getMag
	def __add__(self, other):
		kwargs = {}
		for band in self.order:
			m1 = self.getBand(band)
			m2 = other.getBand(band)
			kwargs[band] = m1 + m2
		return Fluxes(order=self.order, **kwargs)
	def __mul__(self, factor):
		raise

class FluxesPhotoCal(BaseParams):
	def __init__(self, band):
		self.band = band
		BaseParams.__init__(self)
	def copy(self):
		return FluxesPhotoCal(self.band)
	def brightnessToCounts(self, brightness):
		flux = brightness.getFlux(self.band)
		return flux
	def __str__(self):
		return 'FluxesPhotoCal(band=%s)' % (self.band)


class Mag(ScalarParam):
	'''
	An implementation of `Brightness` that stores a single magnitude.
	'''
	stepsize = 0.01
	strformat = '%.3f'

class Flux(ScalarParam):
	'''
	A `Brightness` implementation that stores raw counts.
	'''
	def __mul__(self, factor):
		new = self.copy()
		new.val *= factor
		return new
	__rmul__ = __mul__
	# enforce limit: Flux > 0
	def _set(self, val):
		if val < 0:
			#print 'Clamping Flux from', p[0], 'to zero'
			pass
		self.val = max(0., val)

class MagsPhotoCal(ParamList):
	'''
	A `PhotoCal` implementation to be used with zeropoint-calibrated `Mags`.
	'''
	def __init__(self, band, zeropoint):
		'''
		Create a new ``MagsPhotoCal`` object with a *zeropoint* in a *band*.

		The ``Mags`` objects you use must have *band* as one of their
		available bands.
		'''
		self.band = band
		# MAGIC
		self.maxmag = 50.
		ParamList.__init__(self, zeropoint)

	def copy(self):
		return MagsPhotoCal(self.band, self.zp)

	@staticmethod
	def getNamedParams():
		return dict(zp=0)

	def getStepSizes(self, *args, **kwargs):
		return [0.01]

	def brightnessToCounts(self, brightness):
		mag = brightness.getMag(self.band)
		if not np.isfinite(mag):
			return 0.
		if mag > self.maxmag:
			return 0.
		return 10.**(0.4 * (self.zp - mag))

	def countsToMag(self, counts):
		return self.zp - 2.5 * np.log10(counts)

	def __str__(self):
		return 'MagsPhotoCal(band=%s, zp=%.3f)' % (self.band, self.zp)

class NullPhotoCal(BaseParams):
	'''
	The "identity" `PhotoCal`, to be used with `Flux` -- the
	`Brightness` objects are in units of `Image` counts.
	'''
	def brightnessToCounts(self, brightness):
		return brightness.getValue()

class LinearPhotoCal(ScalarParam):
	'''
	A `PhotoCal`, to be used with `Flux` or `Fluxes` brightnesses,
	that simply scales the flux by a fixed factor; the brightness
	units are proportional to image counts.
	'''

	def __init__(self, scale, band=None):
		'''
		Creates a new LinearPhotoCal object that scales the Fluxes by
		the given factor to produce image counts.

		If 'band' is not None, will retrieve that band from a `Fluxes` object.
		'''
		super(LinearPhotoCal, self).__init__(self, scale)
		self.band = band

	def getScale(self):
		return self.val

	def brightnessToCounts(self, brightness):
		if self.band is None:
			return brightness.getValue() * self.val
		else:
			return brightness.getFlux(self.band) * self.val


class NullWCS(BaseParams):
	'''
	The "identity" WCS -- useful when you are using raw pixel
	positions rather than RA,Decs.
	'''
	def __init__(self, pixscale=1.):
		'''
		pixscale: [arcsec/pix]
		'''
		self.pixscale = pixscale
	def hashkey(self):
		return ('NullWCS',)
	def positionToPixel(self, pos, src=None):
		return pos
	def pixelToPosition(self, x, y, src=None):
		return x,y
	def cdAtPixel(self, x, y):
		return np.array([[1.,0.],[0.,1.]]) * self.pixscale / 3600.

class WcslibWcs(BaseParams):
	'''
	A WCS implementation that wraps a FITS WCS object (with a pixel
	offset), delegating to wcslib.

	FIXME: we could use the "wcssub()" functionality to handle subimages
	rather than x0,y0.

	FIXME: we could implement anwcs_copy() using wcscopy().
	
	'''
	def __init__(self, filename, hdu=0):
		self.x0 = 0.
		self.y0 = 0.
		from astrometry.util.util import anwcs
		wcs = anwcs(filename, hdu)
		self.wcs = wcs

	def copy(self):
		raise RuntimeError('unimplemented')

	def __str__(self):
		return ('WcslibWcs: x0,y0 %.3f,%.3f' % (self.x0,self.y0))

	def debug(self):
		from astrometry.util.util import anwcs_print_stdout
		print 'WcslibWcs:'
		anwcs_print_stdout(self.wcs)

	def pixel_scale(self):
		#from astrometry.util.util import anwcs_pixel_scale
		#return anwcs_pixel_scale(self.wcs)
		cd = self.cdAtPixel(self.x0, self.y0)
		return np.sqrt(np.abs(cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0])) * 3600.

	def setX0Y0(self, x0, y0):
		'''
		Sets the pixel offset to apply to pixel coordinates before putting
		them through the wrapped WCS.  Useful when using a cropped image.
		'''
		self.x0 = x0
		self.y0 = y0

	def positionToPixel(self, pos, src=None):
		ok,x,y = self.wcs.radec2pixelxy(pos.ra, pos.dec)
		# MAGIC: 1 for FITS coords.
		return x - 1. - self.x0, y - 1. - self.y0

	def pixelToPosition(self, x, y, src=None):
		# MAGIC: 1 for FITS coords.
		ra,dec = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 1. + self.y0)
		return RaDecPos(ra, dec)

	def cdAtPixel(self, x, y):
		'''
		Returns the ``CD`` matrix at the given ``x,y`` pixel position.

		(Returns the constant ``CD`` matrix elements)
		'''
		ra0,dec0 = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 1. + self.y0)
		ra1,dec1 = self.wcs.pixelxy2radec(x + 2. + self.x0, y + 1. + self.y0)
		ra2,dec2 = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 2. + self.y0)

		cosdec = np.cos(np.deg2rad(dec0))

		return np.array([[(ra1 - ra0)*cosdec, (ra2 - ra0)*cosdec],
				 [dec1 - dec0,        dec2 - dec0]])
	

class FitsWcs(ParamList):
	'''
	A WCS implementation that wraps a FITS WCS object (with a pixel
	offset).

	The WCS object must be an astrometry.util.util.Tan object, or a
	convincingly quacking duck.

	You can also give it a filename and HDU.
	'''

	@staticmethod
	def getNamedParams():
		return dict(crval1=0, crval2=1, crpix1=2, crpix2=3,
					cd1_1=4, cd1_2=5, cd2_1=6, cd2_2=7)

	def __init__(self, wcs, hdu=0):
		'''
		Creates a new ``FitsWcs`` given a :class:`~astrometry.util.util.Tan`
		object.	 To create one of these from a filename and FITS HDU extension,

		::

			fn = 'my-file.fits'
			ext = 0
			FitsWcs(Tan(fn, ext))
		'''
		if hasattr(self, 'x0'):
			print 'FitsWcs has an x0 attr:', self.x0
		self.x0 = 0
		self.y0 = 0

		if isinstance(wcs, basestring):
			from astrometry.util.util import Tan
			wcs = Tan(wcs, hdu)

		super(FitsWcs, self).__init__(self.x0, self.y0, wcs)
		# ParamList keeps its params in a list; we don't want to do that.
		del self.vals
		self.wcs = wcs

	def copy(self):
		from astrometry.util.util import Tan
		wcs = FitsWcs(Tan(self.wcs))
		wcs.setX0Y0(self.x0, self.y0)
		return wcs

	# Here we ASSUME TAN WCS!
	# oi vey...
	def _setThing(self, i, val):
		w = self.wcs
		if i in [0,1]:
			crv = w.crval
			crv[i] = val
			w.set_crval(*crv)
		elif i in [2,3]:
			crp = w.crpix
			crp[i-2] = val
			w.set_crpix(*crp)
		elif i in [4,5,6,7]:
			cd = list(w.get_cd())
			cd[i-4] = val
			w.set_cd(*cd)
		elif i == 8:
			self.x0 = val
		elif i == 9:
			self.y0 = val
		else:
			raise IndexError
	def _getThing(self, i):
		w = self.wcs
		if i in [0,1]:
			crv = w.crval
			return crv[i]
		elif i in [2,3]:
			crp = w.crpix
			return crp[i-2]
		elif i in [4,5,6,7]:
			cd = w.get_cd()
			return cd[i-4]
		elif i == 8:
			return self.x0
		elif i == 9:
			return self.y0
		else:
			raise IndexError
	def _getThings(self):
		w = self.wcs
		return w.crval + w.crpix + w.cd + [self.x0, self.y0]
	def _numberOfThings(self):
		return 10
	def getStepSizes(self, *args, **kwargs):
		pixscale = self.wcs.pixel_scale()
		# we set everything ~ 1 pixel
		dcrval = pixscale / 3600.
		# CD matrix: ~1 pixel over image size (call it 1000)
		dcd = pixscale / 3600. / 1000.
		ss = [dcrval, dcrval, 1., 1., dcd, dcd, dcd, dcd, 1., 1.]
		return list(self._getLiquidArray(ss))

	# def getParams(self):
	#	'''
	#	Returns a *copy* of the current active parameter values (list)
	#	'''
	#	return list(self._getLiquidArray(self._getThings()))

	def __str__(self):
		return ('FitsWcs: x0,y0 %.3f,%.3f, WCS ' % (self.x0,self.y0)
				+ str(self.wcs))

	def setX0Y0(self, x0, y0):
		'''
		Sets the pixel offset to apply to pixel coordinates before putting
		them through the wrapped WCS.  Useful when using a cropped image.
		'''
		self.x0 = x0
		self.y0 = y0

	def positionToPixel(self, pos, src=None):
		'''
		Converts an :class:`tractor.RaDecPos` to a pixel position.
		Returns: tuple of floats ``(x, y)``
		'''
		X = self.wcs.radec2pixelxy(pos.ra, pos.dec)
		# handle X = (ok,x,y) and X = (x,y) return values
		if len(X) == 3:
			ok,x,y = X
		else:
			assert(len(X) == 2)
			x,y = X
		# MAGIC: subtract 1 to convert from FITS to zero-indexed pixels.
		return x - 1 - self.x0, y - 1 - self.y0

	def pixelToPosition(self, x, y, src=None):
		'''
		Converts floats ``x``, ``y`` to a
		:class:`tractor.RaDecPos`.
		'''
		# MAGIC: add 1 to convert from zero-indexed to FITS pixels.
		r,d = self.wcs.pixelxy2radec(x + 1 + self.x0, y + 1 + self.y0)
		return RaDecPos(r,d)

	def cdAtPixel(self, x, y):
		'''
		Returns the ``CD`` matrix at the given ``x,y`` pixel position.

		(Returns the constant ``CD`` matrix elements)
		'''
		cd = self.wcs.get_cd()
		return np.array([[cd[0], cd[1]], [cd[2],cd[3]]])

	def pixel_scale(self):
		return self.wcs.pixel_scale()


class PixPos(ParamList):
	'''
	A Position implementation using pixel positions.
	'''
	@staticmethod
	def getNamedParams():
		return dict(x=0, y=1)
	def __str__(self):
		return 'pixel (%.2f, %.2f)' % (self.x, self.y)
	#def __repr__(self):
	#	return 'PixPos(%.4f, %.4f)' % (self.x, self.y)
	#def copy(self):
	#	return PixPos(self.x, self.y)
	#def hashkey(self):
	#	return ('PixPos', self.x, self.y)
	def getDimension(self):
		return 2
	def getStepSizes(self, *args, **kwargs):
		return [0.1, 0.1]

class RaDecPos(ParamList):
	'''
	A Position implementation using RA,Dec positions, in degrees.

	Attributes:
	  * ``.ra``
	  * ``.dec``
	'''
	@staticmethod
	def getName():
		return "RaDecPos"
	@staticmethod
	def getNamedParams():
		return dict(ra=0, dec=1)
	def __str__(self):
		return '%s: RA, Dec = (%.5f, %.5f)' % (self.getName(), self.ra, self.dec)
	#def __repr__(self):
	#	return 'RaDecPos(%.5f, %.5f)' % (self.ra, self.dec)
	#def copy(self):
	#	return RaDecPos(self.ra, self.dec)
	#def hashkey(self):
	#	return ('RaDecPos', self.ra, self.dec)
	def getDimension(self):
		return 2
	def getStepSizes(self, *args, **kwargs):
		return [1e-4, 1e-4]

	def distanceFrom(self, pos):
		from astrometry.util.starutil_numpy import degrees_between
		return degrees_between(self.ra, self.dec, pos.ra, pos.dec)

class ConstantSky(ScalarParam):
	'''
	In counts
	'''
	def getParamDerivatives(self, img):
		p = Patch(0, 0, np.ones(img.shape))
		p.setName('dsky')
		return [p]
	def addTo(self, img):
		img += self.val
	def getParamNames(self):
		return ['sky']


class PointSource(MultiParams):
	'''
	An implementation of a point source, characterized by its position
	and brightness.
	'''
	def __init__(self, pos, brightness):
		super(PointSource, self).__init__(pos, brightness)
	@staticmethod
	def getNamedParams():
		return dict(pos=0, brightness=1)

	def getSourceType(self):
		return 'PointSource'
	def getPosition(self):
		return self.pos
	def setPosition(self, position):
		self.pos = position
	def getBrightness(self):
		return self.brightness
	def setBrightness(self, brightness):
		self.brightness = brightness
	def __str__(self):
		return (self.getSourceType() + ' at ' + str(self.pos) +
				' with ' + str(self.brightness))
	def __repr__(self):
		return (self.getSourceType() + '(' + repr(self.pos) + ', ' +
				repr(self.brightness) + ')')
	#def copy(self):
	#	return PointSource(self.pos.copy(), self.brightness.copy())
	#def hashkey(self):
	#	return ('PointSource', self.pos.hashkey(), self.brightness.hashkey())

	def getModelPatch(self, img):
		(px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
		patch = img.getPsf().getPointSourcePatch(px, py)
		#print 'PointSource: PSF patch has sum', patch.getImage().sum()
		counts = img.getPhotoCal().brightnessToCounts(self.brightness)
		return patch * counts

	def getParamDerivatives(self, img):
		'''
		returns [ Patch, Patch, ... ] of length numberOfParams().
		'''
		pos0 = self.getPosition()
		(px0,py0) = img.getWcs().positionToPixel(pos0, self)
		patch0 = img.getPsf().getPointSourcePatch(px0, py0)
		counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
		derivs = []

		# Position
		if not self.isParamFrozen('pos'):
			psteps = pos0.getStepSizes(img)
			pvals = pos0.getParams()
			for i,pstep in enumerate(psteps):
				oldval = pos0.setParam(i, pvals[i] + pstep)
				(px,py) = img.getWcs().positionToPixel(pos0, self)
				patchx = img.getPsf().getPointSourcePatch(px, py)
				pos0.setParam(i, oldval)
				dx = (patchx - patch0) * (counts0 / pstep)
				dx.setName('d(ptsrc)/d(pos%i)' % i)
				derivs.append(dx)

		# Brightness
		if not self.isParamFrozen('brightness'):
			bsteps = self.brightness.getStepSizes(img)
			bvals = self.brightness.getParams()
			for i,bstep in enumerate(bsteps):
				oldval = self.brightness.setParam(i, bvals[i] + bstep)
				countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
				self.brightness.setParam(i, oldval)
				df = patch0 * ((countsi - counts0) / bstep)
				df.setName('d(ptsrc)/d(bright%i)' % i)
				derivs.append(df)
		return derivs

	def overlapsCircle(self, pos, radius):
		return self.pos.distanceFrom(pos) <= radius


class GaussianMixturePSF(BaseParams):
	'''
	A PSF model that is a mixture of general 2-D Gaussians
	(characterized by amplitude, mean, covariance)
	'''
	# Call into MOG to set params, or keep my own copy (via MultiParams)
	def __init__(self, amp, mean, var):
		self.mog = mp.MixtureOfGaussians(amp, mean, var)
		assert(self.mog.D == 2)
		self.radius = 25
		super(GaussianMixturePSF, self).__init__()

	def getMixtureOfGaussians(self):
		return self.mog
	#def proposeIncreasedComplexity(self, img):
	#	raise
	def getStepSizes(self, *args, **kwargs):
		K = self.mog.K
		# amp + mean + var
		# FIXME: -1 for normalization?
		#  : -K for variance symmetry
		return [0.01]*K + [0.01]*(K*2) + [0.1]*(K*3)

	#def isValidParamStep(self, dparam):
	#	## FIXME
	#	return True
	def applyTo(self, image):
		raise
	def getNSigma(self):
		# MAGIC -- N sigma for rendering patches
		return 5.
	def getRadius(self):
		# sqrt(det(var)) ?
		# hmm, really, max(eigenvalue)
		# well, enclosing circle of mu + Nsigma * eigs
		#K = self.mog.K
		#return self.getNSigma * np.max(self.mog.
		# HACK!
		return self.radius
	# returns a Patch object.
	def getPointSourcePatch(self, px, py):
		r = self.getRadius()
		x0,x1 = int(floor(px-r)), int(ceil(px+r))
		y0,y1 = int(floor(py-r)), int(ceil(py+r))
		grid = self.mog.evaluate_grid_dstn(x0-px, x1-px, y0-py, y1-py)
		return Patch(x0, y0, grid)

	def __str__(self):
		return (#'GaussianMixturePSF: ' + str(self.mog)
			'GaussianMixturePSF: amps=' + str(tuple(self.mog.amp.ravel())) +
			', means=' + str(tuple(self.mog.mean.ravel())) +
			', var=' + str(tuple(self.mog.var.ravel())))
	def hashkey(self):
		return ('GaussianMixturePSF',
				tuple(self.mog.amp),
				tuple(self.mog.mean.ravel()),
				tuple(self.mog.var.ravel()),)
	
	def copy(self):
		return GaussianMixturePSF(self.mog.amp.copy(),
								  self.mog.mean.copy(),
								  self.mog.var.copy())

	def numberOfParams(self):
		K = self.mog.K
		return K * (1 + 2 + 3)

	def getParamNames(self):
		K = self.mog.K
		names = ['amp%i' % i for i in range(K)]
		for i in range(K):
			names.extend(['mean%i.x'%i, 'mean%i.y'%i])
		for i in range(K):
			names.extend(['var%i.xx'%i, 'var%i.yy'%i, 'var%i.xy'%i])
		return names

	# def stepParam(self, parami, delta):
	#	K = self.mog.K
	#	if parami < K:
	#		self.mog.amp[parami] += delta
	#		return
	#	parami -= K
	#	if parami < (K*2):
	#		i,j = parami / 2, parami % 2
	#		self.mog.mean[i,j] += delta
	#		return
	#	parami -= 2*K
	#	i,j = parami / 3, parami % 3
	#	if j in [0,1]:
	#		self.mog.var[i,j,j] += deltai
	#	else:
	#		self.mog.var[i,0,1] += deltai
	#		self.mog.var[i,1,0] += deltai

	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		p = list(self.mog.amp) + list(self.mog.mean.ravel())
		for v in self.mog.var:
			p += (v[0,0], v[1,1], v[0,1])
		return p

	def setParams(self, p):
		K = self.mog.K
		self.mog.amp = np.atleast_1d(p[:K])
		pp = p[K:]
		self.mog.mean = np.atleast_2d(pp[:K*2]).reshape(K,2)
		pp = pp[K*2:]
		self.mog.var[:,0,0] = pp[::3]
		self.mog.var[:,1,1] = pp[1::3]
		self.mog.var[:,0,1] = self.mog.var[:,1,0] = pp[2::3]

	def setParam(self, i, p):
		K = self.mog.K
		if i < K:
			old = self.mog.amp[i]
			self.mog.amp[i] = p
			return old
		i -= K
		if i < K*2:
			old = self.mog.mean.ravel()[i]
			self.mog.mean.ravel()[i] = p
			return old
		i -= K*2
		j = i / 3
		k = i % 3
		if k in [0,1]:
			old = self.mog.var[j,k,k]
			self.mog.var[j,k,k] = p
			return old
		old = self.mog.var[j,0,1]
		self.mog.var[j,0,1] = p
		self.mog.var[j,1,0] = p
		return old

	@staticmethod
	def fromStamp(stamp, N=3):
		from emfit import em_fit_2d
		from fitpsf import em_init_params
		w,mu,sig = em_init_params(N, None, None, None)
		stamp = stamp.copy()
		stamp /= stamp.sum()
		stamp = np.maximum(stamp, 0)
		xm, ym = -(stamp.shape[0]/2), -(stamp.shape[1]/2)
		em_fit_2d(stamp, xm, ym, w, mu, sig)
		tpsf = GaussianMixturePSF(w, mu, sig)
		return tpsf
	
class NCircularGaussianPSF(MultiParams):
	'''
	A PSF model using N concentric, circular Gaussians.
	'''
	def __init__(self, sigmas, weights):
		'''
		Creates a new N-Gaussian (concentric, isotropic) PSF.

		sigmas: (list of floats) standard deviations of the components

		weights: (list of floats) relative weights of the components;
		given two components with weights 0.9 and 0.1, the total mass
		due to the second component will be 0.1.  These values will be
		normalized so that the total mass of the PSF is 1.0.

		eg,	  NCircularGaussianPSF([1.5, 4.0], [0.8, 0.2])
		'''
		assert(len(sigmas) == len(weights))
		super(NCircularGaussianPSF, self).__init__(ParamList(*sigmas), ParamList(*weights))

	@staticmethod
	def getNamedParams():
		return dict(sigmas=0, weights=1)

	def __str__(self):
		return ('NCircularGaussianPSF: sigmas [ ' +
				', '.join(['%.3f'%s for s in self.sigmas]) +
				' ], weights [ ' +
				', '.join(['%.3f'%w for w in self.weights]) +
				' ]')

	def __repr__(self):
		return ('NCircularGaussianPSF: sigmas [ ' +
				', '.join(['%.3f'%s for s in self.sigmas]) +
				' ], weights [ ' +
				', '.join(['%.3f'%w for w in self.weights]) +
				' ]')

	def scale(self, factor):
		''' Returns a new PSF that is *factor* times wider. '''
		return NCircularGaussianPSF(np.array(self.weights) * factor, self.weights)

	def getMixtureOfGaussians(self):
		return mp.MixtureOfGaussians(self.weights,
									 np.zeros((len(self.weights), 2)),
									 np.array(self.sigmas)**2)
		
	def proposeIncreasedComplexity(self, img):
		maxs = np.max(self.sigmas)
		# MAGIC -- make new Gaussian with variance bigger than the biggest
		# so far
		return NCircularGaussianPSF(list(self.sigmas) + [maxs + 1.],
							list(self.weights) + [0.05])

	def getStepSizes(self, *args, **kwargs):
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
		return ('NCircularGaussianPSF', tuple(self.sigmas), tuple(self.weights))
	
	def copy(self):
		return NCircularGaussianPSF(list([s for s in self.sigmas]),
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


