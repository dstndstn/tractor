import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from math import pi, sqrt

def makevector(w, mu, sig):
	return np.append(w, np.append(mu.ravel(), sig.ravel()))

def unpackvector(T, K):
	return T[:K], T[K:K*3].reshape(K,2), T[K*3:].reshape(K,2,2)

def em_update_theta(I, theta):
	(K,N,II,xy) = I
	(w, mu, sig) = unpackvector(theta, K)
	w2,mu2,sig2 = em_update(K,N,II,xy,w,mu,sig)
	return makevector(w2,mu2,sig2)

def qn1fit(I, x0, y0, K=2, w=None, mu=None, sig=None):
	(H,W) = I.shape
	X,Y = np.meshgrid(np.arange(W)+x0, np.arange(H)+y0)
	II = I.ravel()
	II /= II.sum()
	xy = np.array(zip(X.ravel(), Y.ravel())).astype(float)
	N = len(II)

	I = (K, N, II, xy)

	def ghatfunc(theta):
		tnew = em_update_theta(I, theta)
		print 'ghat function:'
		print '  theta = ', theta
		print '  tnew  = ', tnew
		return tnew - theta

	# number of parameters
	#    w, mu, sig
	#P = (1 + 2 + 4) * K
	w,mu,sig = em_init_params(K, w, mu, sig)
	theta = makevector(w, mu, sig)
	P = len(theta)

	print 'w,mu,sig', w.shape, mu.shape, sig.shape
	print 'theta', theta.shape

	ghat = ghatfunc(theta)
	#em_update_theta(I, theta) - theta
	print 'ghat', ghat.shape
	print ghat
	A = -np.eye(P)
	print 'A', A.shape

	steps = 10
	for step in range(steps):
		deltheta = -np.dot(A, ghat)
		print 'deltheta', deltheta.shape
		print deltheta
		delghat = ghatfunc(theta + deltheta) - ghat
		print 'delghat', delghat.shape
		print delghat

		delA = (np.dot(np.dot((deltheta - np.dot(A, delghat)), deltheta.T), A)
				/ (np.dot(np.dot(deltheta.T, A), delghat)))
		print 'delA', delA.shape
		theta += deltheta
		ghat += delghat
		A += delA

def em_init_params(K, w, mu, sig):
	if w is None:
		w = np.ones(K) / float(K)
	assert(w.shape == (K,))
	w /= np.sum(w)

	if mu is None:
		mu = np.zeros((K, 2))
	assert(mu.shape == (K,2))

	if sig is None:
		sig = np.array([np.eye(2) * (i+1) for i in range(K)])
	assert(sig.shape == (K,2,2))

	return w, mu, sig

def em_update(K, N, II, xy, w, mu, sig):
	ti = np.zeros((K,N))
	for k,(wi,mi,si) in enumerate(zip(w, mu, sig)):
		ti[k,:] = wi * gauss2d(xy, mi, si)
	ti /= ti.sum(axis=0)

	for k in range(K):
		mu[k,:] = np.average(xy, weights=ti[k,:]*II, axis=0)

	for k in range(K):
		d = (xy - mu[k,:])
		wt = ti[k,:] * II
		S = np.dot(d.T * wt, d) / np.sum(wt)
		sig[k,:,:] = S

	w = np.sum(ti * II, axis=1)
	return w,mu,sig

def emfit(I, x0, y0, K=2, w=None, mu=None, sig=None):
	w,mu,sig = em_init_params(K, w, mu, sig)

	(H,W) = I.shape
	X,Y = np.meshgrid(np.arange(W)+x0, np.arange(H)+y0)

	II = I.ravel()
	II /= II.sum()
	xy = np.array(zip(X.ravel(), Y.ravel())).astype(float)
	N = len(II)

	steps = 1000
	for step in range(steps):
		print 'step', step
		print '  w=', w
		print '  mu=', mu
		print '  sig=', sig

		if step % 10 == 0:
			IM = render_image(X, Y, w, mu, sig)
			plt.clf()
			plt.imshow(IM, interpolation='nearest', origin='lower')
			a = plt.axis()
			for m in mu:
				plt.plot(m[0]-x0, m[1]-y0, 'r.')
			plt.axis(a)
			plt.savefig('fit-%02i.png' % (step/10))

		if step == steps-1:
			break

		w, mu, sig = em_update(K, N, II, xy, w, mu, sig)

	
def gauss2d(X, mu, C):
	cinv = np.linalg.inv(C)
	M = np.sum((X-mu) * np.dot(cinv, (X-mu).T).T, axis=1)
	return 1./(2.*pi*sqrt(abs(np.linalg.det(C)))) * np.exp(-0.5 * M)

def render_image(X,Y, w, mu, sig):
	I = np.zeros_like(X).astype(float)
	xy = np.array(zip(X.ravel(), Y.ravel())).astype(float)
	for m,wi,s in zip(mu, w, sig):
		I += wi * gauss2d(xy, m, s).reshape(I.shape)
	return I
	
def main():
	mu = np.array([ [1.0, 1.0], [-1.0, 2.0] ])
	w  = np.array([ 0.6, 0.4 ])
	sig = np.array([ [[1.0, 0.1],[0.1,1.0]],
					 [[1.5, -0.3],[-0.3,1.5]] ])

	x0,y0 = -10,-10
	X,Y = np.meshgrid(np.arange(x0,10), np.arange(y0,12))

	I = render_image(X, Y, w, mu, sig)

	plt.clf()
	plt.imshow(I, interpolation='nearest', origin='lower')
	a = plt.axis()
	for m in mu:
		plt.plot(m[0]-x0, m[1]-y0, 'r.')
	plt.axis(a)
	plt.savefig('psf.png')

	qn1fit(I, x0, y0, K=2)
	#emfit(I, x0, y0, K=2)
	

if __name__ == '__main__':
	main()

