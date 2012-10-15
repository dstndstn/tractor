import numpy as np
import scipy.interpolate as interp
from utils import *

#class SplineSky(NamedParams):
class SplineSky(ParamList):
	def __init__(self, X, Y, bg):
		## Note -- using weights, we *could* try to make the spline
		# constructor fit for the sky residuals (averaged over the 
		# spline boxes)
		
		self.spl = interp.RectBivariateSpline(X, Y, bg.T)
		print 'fp', self.spl.fp
		print 'tck', self.spl.tck
		print 'deg', self.spl.degrees
		(tx, ty, c) = self.spl.tck
		print 'tx', tx
		print 'ty', ty
		# shape of c: (len(X) - kx(=3) - 1) * (len(Y) - ky(=3) - 1)
		print 'c', c
		print ','.join(['%g'%x for x in tx])
		super(SplineSky, self).__init__(*c)
		# override
		self.vals = c
	# def _getc(self):
	# 	tck = self.spl.tck
	# 	(tx,ty,c) = tck
	# 	return c
	# def _numberOfThings(self):
	# 	return len(self._getc())
	# def _getThing(self, i):
	# 	return self._getc()[i]
	# def _getThings(self):
	# 	return self._getc()
	# def _setThing(self, i, v):
	# 	self._getc()[i] = v

	#def getParamDerivatives(self, img):
	#	pass

	def addTo(self, mod):
		H,W = mod.shape
		X = np.arange(W)
		Y = np.arange(H)[:,np.newaxis]
		#print 'Y', Y.shape, 'Y.T', Y.T.shape
		S = self.spl(X, Y.T)
		#print 'S', S.shape
		mod += S
		
if __name__ == '__main__':
	W,H = 1024,1024
	NX,NY = 6,9
	vals = np.random.normal(size=(NY,NX), scale=100)
	XX = np.linspace(0, W, NX)
	YY = np.linspace(0, H, NY)
	ss = SplineSky(XX, YY, vals)

	print 'NP', ss.numberOfParams()
	print ss.getParamNames()
	print ss.getParams()

	X = np.zeros((H,W))
	
	ss.addTo(X)

	import matplotlib
	matplotlib.use('Agg')
	import pylab as plt

	tx,ty,c = ss.spl.tck
	
	def plotsky(X):
		plt.clf()
		plt.imshow(X, interpolation='nearest', origin='lower')
		ax = plt.axis()
		plt.colorbar()
		for x in tx:
			plt.axvline(x, '-', color='k')
		for y in ty:
			plt.axhline(x, '-', color='k')
		plt.axis(ax)
			
	plotsky(X)
	plt.savefig('sky.png')

	S = ss.getStepSizes()
	p0 = ss.getParams()
	X0 = X.copy()
	for i,step in enumerate(S):
		ss.setParam(i, p0[i]+step)
		X[:,:] = 0.
		ss.addTo(X)
		ss.setParam(i, p0[i])
		dX = (X-X0)/step

		plotsky(dX)
		plt.savefig('sky-d%02i.png' % i)
	
