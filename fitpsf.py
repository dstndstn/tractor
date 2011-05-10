import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from math import pi, sqrt

def emfit(I, x0, y0, K=2):
	mu = np.zeros((K, 2))
	#sig = np.repeat(np.eye(2)[:,:,np.newaxis], 2, axis=2)
	sig = np.array([np.eye(2) * (i+1) for i in range(K)])
	w = np.ones(K) / float(K)

	(H,W) = I.shape
	X,Y = np.meshgrid(np.arange(W)+x0, np.arange(H)+y0)

	II = I.ravel()
	II /= II.sum()
	xy = np.array(zip(X.ravel(), Y.ravel())).astype(float)
	N = len(II)

	steps = 1000
	for step in range(steps):
		#print 'w, mu, sig', w, mu, sig

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

		ti = np.zeros((K,N))
		for k,(wi,mi,si) in enumerate(zip(w, mu, sig)):
			ti[k,:] = wi * gauss2d(xy, mi, si)
		ti /= ti.sum(axis=0)

		#print 'ti', ti.shape
		#print 'mu', mu.shape
		for k in range(K):
			#mu[k,:] = np.average(xy, weights=ti[k,:], axis=0)
			mu[k,:] = np.average(xy, weights=ti[k,:]*II, axis=0)

		for k in range(K):
			#print 'xy:', xy.shape
			#print 'muk:', mu[k,:].shape
			d = (xy - mu[k,:])
			#print 'd', d.shape
			wt = ti[k,:] * II
			S = np.dot(d.T * wt, d) / np.sum(wt)
			#print 'S', S.shape
			sig[k,:,:] = S

		w = np.sum(ti * II, axis=1)


	
def gauss2d(X, mu, C):
	cinv = np.linalg.inv(C)
	M = np.sum((X-mu) * np.dot(cinv, (X-mu).T).T, axis=1)
	return 1./(2.*pi*sqrt(abs(np.linalg.det(C)))) * np.exp(-0.5 * M)

def render_image(X,Y, w, mu, sig):
	I = np.zeros_like(X).astype(float)
	xy = np.array(zip(X.ravel(), Y.ravel())).astype(float)
	#print xy.shape
	for m,wi,s in zip(mu, w, sig):
		#print 'w,mu,sig', wi,m,s
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


	emfit(I, x0, y0, K=2)
	

if __name__ == '__main__':
	main()

