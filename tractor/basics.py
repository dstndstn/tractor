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

from astrometry.util.starutil_numpy import *
from astrometry.util.miscutils import *

class TAITime(ScalarParam, ArithmeticParams):
	'''
	This is TAI as used in the SDSS 'frame' headers; eg

	TAI     =        4507681767.55 / 1st row - 
	Number of seconds since Nov 17 1858

	And http://mirror.sdss3.org/datamodel/glossary.html#tai
	says:

	MJD = TAI/(24*3600)
	'''
	equinox = 53084.28 # mjd
	daysperyear = 365.25

	def __init__(self, t):
		super(TAITime, self).__init__(t)

	def toMjd(self):
		return self.getValue() / (24.*3600.)

	def getSunTheta(self):
		mjd = self.toMjd()
		th = 2. * np.pi * (mjd - TAITime.equinox) / TAITime.daysperyear
		th = np.fmod(th, 2.*np.pi)
		return th

	def toYears(self):
		return float(self.getValue() / (24.*3600. * TAITime.daysperyear))
		
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
		self.stepsizes = [0.01] * self.numberOfParams()

	def getMag(self, bandname):
		''' Bandname: string
		Returns: mag in the given band.
		'''
		return getattr(self, bandname)

	def setMag(self, bandname, mag):
		setattr(self, bandname, mag)

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
	#getBand = Mags.getMag
	#getFlux = Mags.getMag
	def __add__(self, other):
	        kwargs = {}
		for band in self.order:
			m1 = self.getBand(band)
			m2 = other.getBand(band)
			kwargs[band] = m1 + m2
		return self.__class__(order=self.order, **kwargs)
	def __mul__(self, factor):
		kwargs = {}
		for band in self.order:
			m = self.getFlux(band)
			kwargs[band] = m * factor
		return Fluxes(order=self.order, **kwargs)

	def getBand(self, *args, **kwargs):
		return super(Fluxes,self).getMag(*args,**kwargs)
	getFlux = getBand

	def setBand(self, band, val):
		self.setMag(band, val)
	setFlux = setBand	

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


class NanoMaggies(Fluxes):
	'''
	A `Brightness` implementation that stores nano-maggies (ie,
	calibrated flux units), which have the advantage of being linear
	and easily convertible to mags.
	'''
	def __repr__(self):
		return str(self)
	def __str__(self):
		s = getClassName(self) + ': '
		ss = []
		for b in self.order:
			f = self.getFlux(b)
			if f <= 0:
				ss.append('%s=(flux %.3g)' % (b,f))
			else:
				m = self.getMag(b)
				ss.append('%s=%.3g' % (b,m))
		s += ', '.join(ss)
		return s

	def getMag(self, band):
		''' Convert to mag.'''
		flux = self.getFlux(band)
		mag = -2.5 * (np.log10(flux) - 9)
		return mag

	@staticmethod
	def fromMag(mag):
		order = mag.order
		return NanoMaggies(order=order,
						   **dict([(k,NanoMaggies.magToNanomaggies(mag.getMag(k)))
								   for k in order]))

	@staticmethod
	def magToNanomaggies(mag):
		nmgy = 10. ** ((mag - 22.5) / -2.5)
		return nmgy

	@staticmethod
	def nanomaggiesToMag(nmgy):
		mag = -2.5 * (np.log10(nmgy) - 9)
		return mag

	@staticmethod
	def zeropointToScale(zp):
		'''
		Converts a traditional magnitude zeropoint to a scale factor
		by which nanomaggies should be multiplied to produce image
		counts.
		'''
		return 10.**((zp - 22.5)/2.5)


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
		super(LinearPhotoCal, self).__init__(scale)
		self.band = band

	def getScale(self):
		return self.val

	def brightnessToCounts(self, brightness):
		if self.band is None:
			counts = brightness.getValue() * self.val
		else:
			counts = brightness.getFlux(self.band) * self.val
		if counts < 0:
			#print 'Clamping counts up to zero:', counts, 'for brightness', brightness
			return 0.
		return counts
		

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

	# pickling
	def __getstate__(self):
		#print 'pickling wcslib header...'
		s = self.wcs.getHeaderString()
		#print 'pickled wcslib header: len', len(s)
		# print 'Pickling WcslibWcs: string'
		# print '------------------------------------------------'
		# print s
		# print '------------------------------------------------'
		return (self.x0, self.y0, s)

	def __setstate__(self, state):
		(x0,y0,hdrstr) = state
		self.x0 = x0
		self.y0 = y0
		from astrometry.util.util import anwcs_from_string
		# print 'Creating WcslibWcs from header string:'
		# print '------------------------------------------------'
		# print hdrstr
		# print '------------------------------------------------'
		#print 'Unpickling: wcslib header string length:', len(hdrstr)
		self.wcs = anwcs_from_string(hdrstr)
		#print 'unpickling done'
		
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


class ConstantFitsWcs(ParamList):
	'''
	A WCS implementation that wraps a FITS WCS object (with a pixel
	offset).
	'''
	def __init__(self, wcs):
		'''
		Creates a new ``ConstantFitsWcs`` given an underlying WCS object.
		'''
		self.x0 = 0
		self.y0 = 0
		super(ConstantFitsWcs, self).__init__()
		self.wcs = wcs

	def hashkey(self):
		return (self.x0, self.y0, id(self.wcs))

	def copy(self):
		return ConstantFitsWcs(self.wcs)
		
	def __str__(self):
		return ('%s: x0,y0 %.3f,%.3f, WCS ' % (getClassName(self), self.x0,self.y0)
				+ str(self.wcs))

	def getX0Y0(self):
		return self.x0,self.y0

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

	
### FIXME -- this should be called TanWcs!
class FitsWcs(ConstantFitsWcs):
	'''
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
		    from astrometry.util.util import Tan

			fn = 'my-file.fits'
			ext = 0
			FitsWcs(Tan(fn, ext))

		To create one from WCS parameters,

		    tanwcs = Tan(crval1, crval2, crpix1, crpix2,
			    cd11, cd12, cd21, cd22, imagew, imageh)
			FitsWcs(tanwcs)

		'''
		if hasattr(self, 'x0'):
			print 'FitsWcs has an x0 attr:', self.x0
		self.x0 = 0
		self.y0 = 0

		if isinstance(wcs, basestring):
			from astrometry.util.util import Tan
			wcs = Tan(wcs, hdu)

		super(FitsWcs, self).__init__(wcs)
		# ParamList keeps its params in a list; we don't want to do that.
		del self.vals

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
		return list(w.crval) + list(w.crpix) + list(w.cd) + [self.x0, self.y0]
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

class RaDecPos(ParamList, ArithmeticParams):
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
	def __init__(self, *args, **kwargs):
		self.stepsizes = [0,0]	# in case the superclass constructor cares
						# about the length?
		super(RaDecPos,self).__init__(*args,**kwargs)
		self.setStepSizes(1e-4)
	#def __repr__(self):
	#	return 'RaDecPos(%.5f, %.5f)' % (self.ra, self.dec)
	#def copy(self):
	#	return RaDecPos(self.ra, self.dec)
	#def hashkey(self):
	#	return ('RaDecPos', self.ra, self.dec)
	def getDimension(self):
		return 2
	def getStepSizes(self, *args, **kwargs):
		return self.stepsizes
	def setStepSizes(self, delta):
		self.stepsizes = [delta / np.cos(np.deg2rad(self.dec)),delta]

	def distanceFrom(self, pos):
		from astrometry.util.starutil_numpy import degrees_between
		return degrees_between(self.ra, self.dec, pos.ra, pos.dec)



	
class ConstantSky(ScalarParam):
	'''
	In counts
	'''
	def getParamDerivatives(self, tractor, img, srcs):
		p = Patch(0, 0, np.ones(img.shape))
		p.setName('dsky')
		return [p]
	def addTo(self, img, scale=1.):
		if self.val == 0:
			return
		img += (self.val * scale)
	def getParamNames(self):
		return ['sky']


class PointSource(MultiParams):
	'''
	An implementation of a point source, characterized by its position
	and brightness.

	'''
	def __init__(self, pos, br):
		'''
		PointSource(pos, brightness)
		'''
		super(PointSource, self).__init__(pos, br)
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
	def getBrightnesses(self):
		return [self.getBrightness()]
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

	def getUnitFluxModelPatch(self, img, minval=0.):
		(px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
		# print 'PointSource.getUnitFluxModelPatch: pix pos', px,py
		patch = img.getPsf().getPointSourcePatch(px, py, minval=minval)
		# print '  Patch', patch
		return patch

	def getUnitFluxModelPatches(self, *args, **kwargs):
		return [self.getUnitFluxModelPatch(*args, **kwargs)]

	def getModelPatch(self, img, minsb=0.):
		counts = img.getPhotoCal().brightnessToCounts(self.brightness)
		if counts <= 0:
			return None
		minval = minsb / counts
		upatch = self.getUnitFluxModelPatch(img, minval=minval)
		return upatch * counts

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

class Parallax(ScalarParam, ArithmeticParams):
	''' in arcesc '''
	stepsize = 1e-3
	def __str__(self):
		return 'Parallax: %.3f arcsec' % (self.getValue())

	#### FIXME -- cos(Dec)
class PMRaDec(RaDecPos):
	@staticmethod
	def getName():
		return "PMRaDec"
	def __str__(self):
		return '%s: (%.3f, %.3f) "/yr' % (self.getName(),
										  self.getRaArcsecPerYear(),
										  self.getDecArcsecPerYear())
	def __init__(self, *args, **kwargs):
		self.addParamAliases(ra=0, dec=1)
		super(PMRaDec,self).__init__(*args,**kwargs)
		self.setStepSizes(1e-6)
		
	# def setStepSizes(self, delta):
	#	self.stepsizes = [delta, delta]
		
	@staticmethod
	def getNamedParams():
		return dict(pmra=0, pmdec=1)

	def getRaArcsecPerYear(self):
		return self.pmra * 3600.
	def getDecArcsecPerYear(self):
		return self.pmdec * 3600.
	
class MovingPointSource(PointSource):
	def __init__(self, pos, brightness, pm, parallax, epoch=0.):
		# ASSUME 'pm' is the same type as 'pos'
		#assert(type(pos) == type(pm))
		# More precisely, ...
		assert(type(pos) is RaDecPos)
		assert(type(pm) is PMRaDec)
		super(PointSource, self).__init__(pos, brightness, pm,
										  Parallax(parallax))
		self.epoch = epoch

	@staticmethod
	def getNamedParams():
		return dict(pos=0, brightness=1, pm=2, parallax=3)

	def getSourceType(self):
		return 'MovingPointSource'

	def __str__(self):
		return (self.getSourceType() + ' at ' + str(self.pos) +
				' with ' + str(self.brightness) + ', pm ' + str(self.pm) +
				', parallax ' + str(self.parallax))

	def __repr__(self):
		return (self.getSourceType() + '(' + repr(self.pos) + ', ' +
				repr(self.brightness) + ', ' + repr(self.pm) + ', ' +
				repr(self.parallax) + ')')

	def getPositionAtTime(self, t):
		dt = (t - self.epoch).toYears()
		# Assume "pos" is an RaDecPos
		p = self.pos + dt * self.pm
		suntheta = t.getSunTheta()

		xyz = radectoxyz(p.ra, p.dec)
		xyz = xyz[0]
		# d(celestial coords)/d(parallax)
		# - takes numerical derivatives when it could take analytic ones
		# output is in [degrees / arcsec].	Yep.	Crazy but true.
		# HACK: fmods dRA when it should do something continuous.
		# rd2xyz(0,0) is a unit vector; 1/arcsecperrad is (a good approximation to)
		# the distance on the unit sphere spanned by an angle of 1 arcsec.
		# We take a step of that length and return the change in RA,Dec.
		# It's about 1e-5 so we don't renormalize the xyz unit vector.
		dxyz1 = radectoxyz(0., 0.) / arcsecperrad
		dxyz1 = dxyz1[0]
		# - imprecise angle of obliquity
		# - implicitly assumes circular orbit
		# output is in [degrees / arcsec].	Yep.	Crazy but true.
		dxyz2 = radectoxyz(90., axistilt) / arcsecperrad
		dxyz2 = dxyz2[0]
		xyz += self.parallax.getValue() * (dxyz1 * np.cos(suntheta) +
										   dxyz2 * np.sin(suntheta))
		r,d = xyztoradec(xyz)
		return RaDecPos(r,d)
	
	def getUnitFluxModelPatch(self, img, minval=0.):
		pos = self.getPositionAtTime(img.getTime())
		(px,py) = img.getWcs().positionToPixel(pos)
		patch = img.getPsf().getPointSourcePatch(px, py, minval=minval)
		return patch

	def getParamDerivatives(self, img):
	 	'''
		MovingPointSource derivatives.

		returns [ Patch, Patch, ... ] of length numberOfParams().
	 	'''
		t = img.getTime()
		pos0 = self.getPositionAtTime(t)
		(px0,py0) = img.getWcs().positionToPixel(pos0, self)
		patch0 = img.getPsf().getPointSourcePatch(px0, py0)
		counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
		derivs = []

		#print 'initial pixel pos', px0, py0
		
		# Position

		# FIXME -- could just compute positional derivatives once and
		# reuse them, but have to be careful about frozen-ness -- eg,
		# if RA were frozen but not Dec.
		# OR, could compute dx,dy and then use CD matrix to convert
		# dpos to derivatives.
		# pderivs = []
		# if ((not self.isParamFrozen('pos')) or
		# 	(not self.isParamFrozen('pm')) or
		# 	(not self.isParamFrozen('parallax'))):
		# 	
		# 	psteps = pos0.getStepSizes(img)
		# 	pvals = pos0.getParams()
		# 	for i,pstep in enumerate(psteps):
		# 		oldval = pos0.setParam(i, pvals[i] + pstep)
		# 		(px,py) = img.getWcs().positionToPixel(pos0, self)
		# 		patchx = img.getPsf().getPointSourcePatch(px, py)
		# 		pos0.setParam(i, oldval)
		# 		dx = (patchx - patch0) * (counts0 / pstep)
		# 		dx.setName('d(ptsrc)/d(pos%i)' % i)
		# 		pderivs.append(dx)
		# if not self.isParamFrozen('pos'):
		# 	derivs.extend(pderivs)

		def _add_posderivs(p, name):
			psteps = p.getStepSizes(img)
			pvals = p.getParams()
			for i,pstep in enumerate(psteps):
				oldval = p.setParam(i, pvals[i] + pstep)
				tpos = self.getPositionAtTime(t)
				(px,py) = img.getWcs().positionToPixel(tpos, self)
				#print 'stepping param', name, i, '-->', p, '--> pix pos', px,py
				patchx = img.getPsf().getPointSourcePatch(px, py)
				p.setParam(i, oldval)
				dx = (patchx - patch0) * (counts0 / pstep)
				dx.setName('d(ptsrc)/d(%s%i)' % (name, i))
				derivs.append(dx)

		if not self.isParamFrozen('pos'):
			_add_posderivs(self.pos, 'pos')
		
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

		if not self.isParamFrozen('pm'):
			# 	# ASSUME 'pm' is the same type as 'pos'
			# 	dt = (t - self.epoch).toYears()
			# 	for d in pderivs:
			# 		dd = d * dt
			# 		derivs.append(dd)
			_add_posderivs(self.pm, 'pm')

		if not self.isParamFrozen('parallax'):
			_add_posderivs(self.parallax, 'parallax')
				
		return derivs

class PixelizedPSF(BaseParams):
	'''
	A PSF model based on an image postage stamp, which will be
	sinc-shifted to subpixel positions.

	(Actually Lanczos shifted, with order Lorder)
	
	This will allow only Point Sources to be rendered by the Tractor!

	FIXME -- currently this class claims to have no params.
	'''
	def __init__(self, img, Lorder=3):
		self.img = img
		H,W = img.shape
		assert((H % 2) == 1)
		assert((W % 2) == 1)
		self.Lorder = Lorder
		
	def __str__(self):
		return 'PixelizedPSF'

	def hashkey(self):
		return ('PixelizedPSF', tuple(self.img.ravel()))

	def copy(self):
		return PixelizedPSF(self.img.copy())

	def getRadius(self):
		H,W = self.img.shape
		return np.hypot(H,W)/2.

	def getPointSourcePatch(self, px, py, minval=0.):
		from scipy.ndimage.filters import correlate1d
		H,W = self.img.shape
		ix = int(np.round(px))
		iy = int(np.round(py))
		dx = px - ix
		dy = py - iy
		x0 = ix - W/2
		y0 = iy - H/2
		L = self.Lorder
		Lx = lanczos_filter(L, np.arange(-L, L+1) + dx)
		Ly = lanczos_filter(L, np.arange(-L, L+1) + dy)
		sx      = correlate1d(self.img, Lx, axis=1, mode='constant')
		shifted = correlate1d(sx,       Ly, axis=0, mode='constant')
		#shifted /= (Lx.sum() * Ly.sum())
		#print 'Shifted PSF: range', shifted.min(), shifted.max()

		### ???
		#shifted = np.maximum(shifted, 0.)

		shifted /= shifted.sum()
		return Patch(x0, y0, shifted)
	
class GaussianMixturePSF(BaseParams):
	'''
	A PSF model that is a mixture of general 2-D Gaussians
	(characterized by amplitude, mean, covariance)
	'''
	# Call into MOG to set params, or keep my own copy (via MultiParams)
	def __init__(self, amp, mean, var):
		'''
		amp:  np array (size K) of Gaussian amplitudes
		mean: np array (size K,2) of means
		var:  np array (size K,2,2) of variances
		'''
		self.mog = mp.MixtureOfGaussians(amp, mean, var)
		assert(self.mog.D == 2)
		# !!
		self.radius = 25
		super(GaussianMixturePSF, self).__init__()

	def computeRadius(self):
		import numpy.linalg
		# ?
		meig = max([max(abs(numpy.linalg.eigvalsh(v)))
					for v in self.mog.var])
		#for v in self.mog.var:
		#	print 'Var', v
		#	print 'Eigs:', numpy.linalg.eigvalsh(v)
		return self.getNSigma() * np.sqrt(meig)
		
	def scaleBy(self, factor):
		# Use not advised, ever
		amp = self.mog.amp
		mean = self.mog.mean * factor
		var = self.mog.var * factor**2
		return GaussianMixturePSF(amp, mean, var)

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
	def getPointSourcePatch(self, px, py, minval=0.):
		if minval > 0.:
			r = 0.
			for v in self.mog.var:
				# overestimate
				vv = (v[0,0] + v[1,1])
				norm = 2. * np.pi * np.linalg.det(v)
				r2 = vv * -2. * np.log(minval * norm)
				if r2 > 0:
					r = max(r, np.sqrt(r2))
			rr = int(np.ceil(r))
			#print 'choosing r=', rr
			cx = int(np.round(px))
			cy = int(np.round(py))
			#x0,x1 = int(floor(px-r)), int(ceil(px+r))
			#y0,y1 = int(floor(py-r)), int(ceil(py+r))
			dx = px - cx
			dy = py - cy
			#dx = cx - px
			#dy = cy - py
			x0,y0 = cx-rr, cy-rr
			grid = self.mog.evaluate_grid_approx(-rr, rr, -rr, rr, dx, dy,  minval)

			# x1,y1 = cx+rr, cy+rr
			# XX,YY = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
			# gx = np.sum(grid * XX) / np.sum(grid)
			# gy = np.sum(grid * YY) / np.sum(grid)

			# print 'px %8.3f, py %8.3f' % (px,py)
			# print 'gx %8.3f, gy %8.3f' % (gx,gy)
			# print 'dx %8.3f, dy %8.3f' % (dx,dy)

			# r = self.getRadius()
			# x0,x1 = int(floor(px-r)), int(ceil(px+r))
			# y0,y1 = int(floor(py-r)), int(ceil(py+r))
			# grid = self.mog.evaluate_grid(x0-px, x1-px, y0-py, y1-py)
			# XX,YY = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))
			# gx = np.sum(grid * XX) / np.sum(grid)
			# gy = np.sum(grid * YY) / np.sum(grid)
			# print 'Gx %8.3f, gy %8.3f' % (gx,gy)


		else:
			r = self.getRadius()
			x0,x1 = int(floor(px-r)), int(ceil(px+r))
			y0,y1 = int(floor(py-r)), int(ceil(py+r))
			grid = self.mog.evaluate_grid(x0-px, x1-px, y0-py, y1-py)
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
		self.minradius = 1.
		
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
		ss = []
		if not self.isParamFrozen('sigmas'):
			ss.extend([0.01]*N)
		if not self.isParamFrozen('weights'):
			ss.extend([0.01]*N)
		return ss

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
		hk = ('NCircularGaussianPSF', tuple(self.sigmas), tuple(self.weights))
		#hk = ('NCircularGaussianPSF',
		#	  tuple(x for x in self.sigmas),
		#	  tuple(x for x in self.weights))
		# print 'sigmas', self.sigmas
		# print 'sigmas type', type(self.sigmas)
		# print 'sigmas[0]', self.sigmas[0]
		# print 'Hashkey', hk
		# print hash(hk)
		return hk
	
	def copy(self):
		cc = NCircularGaussianPSF(list([s for s in self.sigmas]),
								  list([w for w in self.weights]))
		#print 'NCirc copy', cc
		return cc

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
		return max(self.minradius, max(self.sigmas) * self.getNSigma())

	# returns a Patch object.
	#### FIXME -- could use mixture_profiles!!
	def getPointSourcePatch(self, px, py, minval=0.):
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


# class SubImage(Image):
# 	def __init__(self, im, roi,
# 				 skyclass=SubSky,
# 				 psfclass=SubPsf,
# 				 wcsclass=SubWcs):
# 		(x0,x1,y0,y1) = roi
# 		slc = (slice(y0,y1), slice(x0,x1))
# 		data = im.getImage[slc]
# 		invvar = im.getInvvar[slc]
# 		sky = skyclass(im.getSky(), roi)
# 		psf = psfclass(im.getPsf(), roi)
# 		wcs = wcsclass(im.getWcs(), roi)
# 		pcal = im.getPhotoCal()
# 		super(SubImage, self).__init__(data=data, invvar=invvar, psf=psf,
# 									   wcs=wcs, sky=sky, photocal=photocal,
# 									   name='sub'+im.name)
# 
# class SubSky(object):
# 	def __init__(self, sky, roi):
# 		self.sky = sky
# 		self.roi = roi
# 
# 	#def getParamDerivatives(self, img):
# 	def addTo(self, mod):

class ParamsWrapper(BaseParams):
	def __init__(self, real):
		self.real = real
	def hashkey(self):
		return self.real.hashkey()
	def getLogPrior(self):
		return self.real.getLogPrior()
	def getLogPriorDerivatives(self):
		return self.real.getLogPriorDerivatives()
	def getParams(self):
		return self.real.getParams()
	def setParams(self, x):
		return self.real.setParams(x)
	def setParam(self, i, x):
		return self.real.setParam(i, x)
	def numberOfParams(self):
		return self.real.numberOfParams()
	def getParamNames(self):
		return self.real.getParamNames()
	def getStepSizes(self, *args, **kwargs):
		return self.real.getStepSizes(*args, **kwargs)
	
class ScaledPhotoCal(ParamsWrapper):
	def __init__(self, photocal, factor):
		super(ScaledPhotoCal,self).__init__(photocal)
		self.pc = photocal
		self.factor = factor
	def hashkey(self):
		return ('ScaledPhotoCal', self.factor) + self.pc.hashkey()
	def brightnessToCounts(self, brightness):
		return self.factor * self.pc.brightnessToCounts(brightness)

class ScaledWcs(ParamsWrapper):
	def __init__(self, wcs, factor):
		super(ScaledWcs,self).__init__(photocal)
		self.factor = factor
		self.wcs = wcs

	def hashkey(self):
		return ('ScaledWcs', self.factor) + tuple(self.wcs.hashkey())

	def cdAtPixel(self, x, y):
		x,y = (x + 0.5) / self.factor - 0.5, (y + 0.5) * self.factor - 0.5
		cd = self.wcs.cdAtPixel(x, y)
		return cd / self.factor

	def positionToPixel(self, pos, src=None):
		x,y = self.wcs.positionToPixel(pos, src=src)
		# Or somethin'
		return ((x + 0.5) * self.factor - 0.5,
				(y + 0.5) * self.factor - 0.5)

class ShiftedWcs(ParamsWrapper):
	'''
	Wraps a WCS in order to use it for a subimage.
	'''
	def __init__(self, wcs, x0, y0):
		super(ShiftedWcs,self).__init__(wcs)
		self.x0 = x0
		self.y0 = y0
		self.wcs = wcs

	def hashkey(self):
		return ('ShiftedWcs', self.x0, self.y0) + tuple(self.wcs.hashkey())

	def cdAtPixel(self, x, y):
		return self.wcs.cdAtPixel(x + self.x0, y + self.y0)

	def positionToPixel(self, pos, src=None):
		x,y = self.wcs.positionToPixel(pos, src=src)
		return (x - self.x0, y - self.y0)
