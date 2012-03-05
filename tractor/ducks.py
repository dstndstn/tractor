class Params(object):
	'''
	A set of parameters that can be optimized by the Tractor.

	This is a duck-type definition.  (Plus some default convenience
	functions for subclassers.  The duck part should be pulled out.)
	FIXME
	'''
	@staticmethod
	def getClassName(self):
		name = getattr(self.__class__, 'classname', None)
		#name = getattr(__class__, 'classname', None)
		if name is not None:
			return name
		return self.__class__.__name__
		#return __class__.__name__
		#return __name__

	#def __init__(self):
	#	print 'Params __init__'
	#	super(Params,self).__init__()

	def __repr__(self):
		return self.getClassName(self) + repr(self.getParams())
	def __str__(self):
		return self.getClassName(self) + ': ' + str(self.getParams())
	def copy(self):
		return self.__class__(self.getParams())
		
	def hashkey(self):
		return (self.getClassName(self),) + tuple(self.getParams())
	def __hash__(self):
		return hash(self.hashkey())
	def __eq__(self, other):
		return hash(self) == hash(other)
	def numberOfParams(self):
		return len(self.getParams())
	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		return []
	def getStepSizes(self, *args, **kwargs):
		return []
	def setParams(self, p):
		assert(len(p) == self.numberOfParams())
		for ii,pp in enumerate(p):
			self.setParam(ii, pp)

	def setParam(self, i, p):
		'''
		Sets parameter index 'i' to new value 'p',
		Returns the old value.
		'''
		return None


class Sky(Params):
	'''
	Duck-type definition for the sky model.
	'''
	# returns [ Patch, Patch, ... ] of length numberOfParams().
	def getParamDerivatives(self, img, brightnessonly=False):
		return []
	def addTo(self, img):
		pass


class Source(Params):
	'''
	This is the duck-type definition of a Source (star, galaxy, etc)
	that the Tractor use.
	
	Must be hashable: see
	  http://docs.python.org/glossary.html#term-hashable
	'''
	def getModelPatch(self, img):
		pass
	# returns [ Patch, Patch, ... ] of length numberOfParams().
	def getParamDerivatives(self, img, brightnessonly=False):
		return []
	def getSourceType(self):
		return 'Source'

class PhotoCal(Params):
	'''
	Duck-type definition of photometric calibration.

	Converts between Brightness objects and counts in an Image.
	'''
	def brightnessToCounts(self, brightness):
		pass


class WCS(Params):
	'''
	Duck-type definition of World Coordinate System.
	
	Converts between Position objects and Image pixel coordinates.
	'''
	def positionToPixel(self, pos, src=None):
		'''
		Returns tuple (x, y) -- or any duck that supports
		iteration of two items
		'''
		return None

	def pixelToPosition(self, x, y, src=None):
		'''
		(x,y) to Position; src may be None (?)
		'''
		return None

	def cdAtPixel(self, x, y):
		'''
		(x,y) to numpy array (2,2) -- the CD matrix at pixel x,y:

		[ [ dRA/dx * cos(Dec), dRA/dy * cos(Dec) ],
		  [ dDec/dx          , dDec/dy           ] ]

		in FITS these are called:
		[ [ CD11             , CD12              ],
		  [ CD21             , CD22              ] ]

		  Note: these statements have not been verified by the FDA.
		'''
		return None


class PSF(Params):
	'''
	Duck-type definition of a Tractor PSF model.
	'''
	def applyTo(self, image):
		pass

	def getRadius(self):
		'''
		Returns the size of the support of this PSF.
		'''
		return 0

	def getPointSourcePatch(self, px, py):
		'''
		returns a Patch, a rendering of a point source at the given pixel
		coordinate.
		'''
		pass

	def proposeIncreasedComplexity(self, img):
		'''
		Returns a new PSF object that is a more complex version of self.
		'''
		return PSF()
	
	def isValidParamStep(self, dparam):
		return True
