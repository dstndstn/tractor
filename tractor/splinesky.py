import numpy as np
import scipy.interpolate as interp
from utils import *

class SplineSky(ParamList):
	def __init__(self, X, Y, bg, order=3):
		'''
		X,Y: pixel positions of control points
		bg: values of spline at control points
		'''
		## Note -- using weights, we *could* try to make the spline
		# constructor fit for the sky residuals (averaged over the 
		# spline boxes)
		self.W = len(X)
		self.H = len(Y)
		self.order = order
		self.spl = interp.RectBivariateSpline(X, Y, bg.T,
											  kx=order, ky=order)
		#print 'fp', self.spl.fp
		#print 'tck', self.spl.tck
		#print 'deg', self.spl.degrees
		(tx, ty, c) = self.spl.tck
		#print 'tx', tx
		#print 'ty', ty
		# A false statement:
		#   shape of c: (len(X) + kx(=3) - 1) * (len(Y) + ky(=3) - 1)
		# It actually seems to be len(X) * len(Y), at least when order=3.
		#print 'c', c
		#print ','.join(['%g'%x for x in tx])
		super(SplineSky, self).__init__(*c)
		# override -- "c" is a list, so this should work as expected
		self.vals = c

		self.prior_smooth_sigma = None

	def setPriorSmoothness(self, sigma):
		'''
		The smoothness sigma is proportional to sky-intensity units;
		small sigma leads to smooth sky.
		'''
		self.prior_smooth_sigma = sigma

	def addTo(self, mod):
		H,W = mod.shape
		X = np.arange(W)
		Y = np.arange(H)
		#print 'Y', Y.shape, 'Y.T', Y.T.shape
		#print 'X shape', X.shape, 'Y shape', Y.shape
		#S = self.spl(X, Y.T)
		S = self.spl(X, Y).T
		#print 'mod', mod.shape
		#print 'S', S.shape
		mod += S

	def getParamGrid(self):
		arr = np.array(self.vals)
		assert(len(arr) == (self.W * self.H))
		arr = arr.reshape(self.H, -1)
		return arr

	def getLogPrior(self):
		if self.prior_smooth_sigma is None:
			return 0.
		p = self.getParamGrid()
		sig = self.prior_smooth_sigma
		lnP = (-0.5 * np.sum((p[1:,:] - p[:-1,:])**2) / (sig**2) +
			   -0.5 * np.sum((p[:,1:] - p[:,:-1])**2) / (sig**2))
		#print 'logPrior: returning', lnP
		return lnP

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

		#### FIXME -- we ignore frozenness!

		if self.prior_smooth_sigma is None:
			return None

		rA = []
		cA = []
		vA = []
		pb = []
		# columns are parameters
		# rows are prior terms

		# 
		p = self.getParamGrid()
		sig = self.prior_smooth_sigma
		#print 'Param grid', p.shape
		#print 'len', len(p)
		II = np.arange(len(p.ravel())).reshape(p.shape)
		assert(p.shape == II.shape)

		dx = p[:,1:] - p[:,:-1]
		dy = p[1:,:] - p[:-1,:]

		NX = len(dx.ravel())

		rA.append(np.arange(NX))
		cA.append(II[:, :-1].ravel())
		vA.append(np.ones(NX) / sig)
		pb.append(dx.ravel() / sig)

		rA.append(np.arange(NX))
		cA.append(II[:, 1:].ravel())
		vA.append(-np.ones(NX) / sig)

		NY = len(dy.ravel())

		rA.append(NX + np.arange(NY))
		cA.append(II[:-1, :].ravel())
		vA.append(np.ones(NY) / sig)
		pb.append(dy.ravel() / sig)

		rA.append(NX + np.arange(NY))
		cA.append(II[1:, :].ravel())
		vA.append(-np.ones(NY) / sig)

		#print 'log prior chi: returning', len(rA), 'sets of terms'

		return (rA, cA, vA, pb)

	def getParamDerivatives(self, *args):
		derivs = []
		tx,ty = self.spl.get_knots()
		print 'Knots:'
		print 'x', len(tx), tx
		print 'y', len(ty), ty
		print 'W,H', self.W, self.H
		#NX = len(tx) - self.order
		#print 'NX', NX
		for i in self.getThawedParamIndices():
			ix = i % W
			iy = i / W
			derivs.append(False)
		return derivs

		
if __name__ == '__main__':
	W,H = 1024,600
	NX,NY = 6,9
	sig = 100.
	vals = np.random.normal(size=(NY,NX), scale=sig)

	vals[1,0] = 10000.

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
	
		plt.clf()
		G = ss.getParamGrid()
		plt.imshow(G.T, interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.savefig('sky-p%02i.png' % i)

		plt.clf()
		plt.imshow(X, interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.savefig('sky-%02i.png' % i)
		
		break



	from tractor import *
	data = X.copy()
	tim = Image(data=data, invvar=np.ones_like(X) * (1./sig**2),
				psf=NCircularGaussianPSF([1.], [1.]),
				wcs=NullWCS(), sky=ss,
				photocal=NullPhotoCal(), name='sky im')
	tractor = Tractor(images=[tim])
	ss.setPriorSmoothness(sig)

	print 'Tractor', tractor
	print 'im params', tim.getParamNames()
	print 'im step sizes', tim.getStepSizes()

	print 'Initial:'
	print 'lnLikelihood', tractor.getLogLikelihood()
	print 'lnPrior', tractor.getLogPrior()
	print 'lnProb', tractor.getLogProb()

	mod = tractor.getModelImage(0)
	
	plt.clf()
	plt.imshow(mod, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('mod0.png')

	tractor.optimize()

	print 'After opt:'
	print 'lnLikelihood', tractor.getLogLikelihood()
	print 'lnPrior', tractor.getLogPrior()
	print 'lnProb', tractor.getLogProb()

	mod = tractor.getModelImage(0)
	
	plt.clf()
	plt.imshow(mod, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('mod1.png')
