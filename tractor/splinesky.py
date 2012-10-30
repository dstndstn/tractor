import numpy as np
import scipy.interpolate as interp
from utils import *

class SplineSky(ParamList):
	def __init__(self, X, Y, bg):
		## Note -- using weights, we *could* try to make the spline
		# constructor fit for the sky residuals (averaged over the 
		# spline boxes)

		self.spl = interp.RectBivariateSpline(X, Y, bg.T)
		#print 'fp', self.spl.fp
		#print 'tck', self.spl.tck
		#print 'deg', self.spl.degrees
		(tx, ty, c) = self.spl.tck
		#print 'tx', tx
		#print 'ty', ty
		# shape of c: (len(X) - kx(=3) - 1) * (len(Y) - ky(=3) - 1)
		#print 'c', c
		#print ','.join(['%g'%x for x in tx])
		super(SplineSky, self).__init__(*c)
		# override -- "c" is a list, so this should work as expected
		self.vals = c

	def addTo(self, mod):
		H,W = mod.shape
		#X = np.arange(W)
		#Y = np.arange(H)[:,np.newaxis]
		X = np.arange(W)#[:,np.newaxis]
		Y = np.arange(H)
		#print 'Y', Y.shape, 'Y.T', Y.T.shape
		#print 'X shape', X.shape, 'Y shape', Y.shape
		#S = self.spl(X, Y.T)
		S = self.spl(X, Y).T
		#print 'mod', mod.shape
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
			plt.axhline(x, color='0.5')
		for y in ty:
			plt.axvline(y, color='0.5')

		H,W = X.shape
		# Find non-zero range
		for nzx0 in range(W):
			if not np.all(X[:,nzx0] == 0):
				break
		for nzx1 in range(W-1,-1,-1):
			if not np.all(X[:,nzx1] == 0):
				break
		for nzy0 in range(H):
			if not np.all(X[nzy0,:] == 0):
				break
		for nzy1 in range(H-1,-1,-1):
			if not np.all(X[nzy1,:] == 0):
				break
		plt.axhline(nzy0, color='y')
		plt.axhline(nzy1, color='y')
		plt.axvline(nzx0, color='y')
		plt.axvline(nzx1, color='y')
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
	
