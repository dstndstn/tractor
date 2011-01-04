

'''
Duck types:

class Position(object):
	def setElement(self, i, val):
		pass
	def getElement(self, i):
		return 1.0


source 42, adjust position elements 0 (stepsize=1e-6), 1 (stepsize=1e-6)
source 43, adjust position elements 0, 1, 2

class Catalog(object):
	[sources]

	def getSources(self, image):
		return [source]

class Band(object):
	pass

class Source(object):
	'
	Position
	Flux
	SourceType
	(sourcetype-specific parameters)
	'
	def getParameter(self, i):
		if i < self.Nparams:
			return (val, step)
		return None

	def setParameter(self, i, val):
		pass

	def isInImage(self, image):
		return True



	def getFlux(self, band):
		return 42.0



class WCS(object):
	def positionToPixel(self, position, image):
		'
		position: list of Position objects
		image: Image object

		returns: xy: a list of pixel locations -- (x,y) doubles
		:            eg, [ (1.0, 2.0), (3.0, 4.0) ]
		
		'
		return xy



class Image(object):
	-invvar
	-wcs
	-psf
	-flux calib

	
'''


class Tractor(object):

	def __init__(self, catalog, image):
		'''
		image: list of Image objects
		catalog: list of Source objects
		'''
		pass

	def optimizeAtFixedComplexityStep(self):
		'''
		-synthesize images

		-get all derivatives
		(taking numerical derivatives itself?)

		-build matrix

		-take step (try full step, back off)
		'''
		pass

	def createSources(self):
		'''
		-synthesize images
		-look for "promising" Positions with "positive" residuals
		---chi image, PSF smooth, propose positions?
		-instantiate new source (Position, flux, PSFType)
		-local optimizeAtFixedComplexity
		'''
		pass

	def modifyComplexity(self):
		'''
		-synthesize images
		-for all sources?
		---for all sourceTypes (including None)
		-----change source.sourceType -> sourceType
		-----local optimizeAtFixedComplexity
		'''
		pass
	
	def step(self):
		'''
		'''
		pass

