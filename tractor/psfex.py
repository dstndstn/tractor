import numpy as np
import numpy.linalg
import scipy.interpolate as interp
from scipy.ndimage.interpolation import affine_transform
from astrometry.util.fits import *
from astrometry.util.plotutils import *

from tractor.basics import *
from tractor.utils import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params


class PsfEx(MultiParams):
	def __init__(self, fn, W, H, ext=1,
				 nx=11, ny=11, K=3):
		T = fits_table(fn, ext=ext)
		ims = T.psf_mask[0]
		print 'Got', ims.shape, 'PSF images'
		hdr = pyfits.open(fn)[ext].header
		# PSF distortion bases are polynomials of x,y
		assert(hdr['POLNAME1'] == 'X_IMAGE')
		assert(hdr['POLNAME2'] == 'Y_IMAGE')
		assert(hdr['POLGRP1'] == 1)
		assert(hdr['POLGRP2'] == 1)
		assert(hdr['POLNGRP' ] == 1)
		x0     = hdr.get('POLZERO1')
		xscale = hdr.get('POLSCAL1')
		y0     = hdr.get('POLZERO2')
		yscale = hdr.get('POLSCAL2')
		degree = hdr.get('POLDEG1')

		self.sampling = hdr.get('PSF_SAMP')

		# number of terms in polynomial
		ne = (degree + 1) * (degree + 2) / 2
		assert(hdr['PSFAXIS3'] == ne)
		assert(len(ims.shape) == 3)
		assert(ims.shape[0] == ne)
 		## HACK -- fit psf0 + psfi for each term i
		## (since those will probably work better as multi-Gaussians
		## than psfi alone)
		## OR instantiate PSF across the image, fit with
		## multi-gaussians, and regress the gaussian params?
		
		self.psfbases = ims
		self.x0,self.y0 = x0,y0
		self.xscale, self.yscale = xscale, yscale
		self.degree = degree
		self.W, self.H = W,H

		self.nx = nx
		self.ny = ny
		self.K = K

		# no args?
		super(PsfEx, self).__init__()

	def getPointSourcePatch(self, px, py, minval=None):
		mog = self.mogAt(px, py)
		return mog.getPointSourcePatch(px, py, minval=minval)

	def _fitParamGrid(self, nx=10, ny=10, K=3, scale=True):
		w,mu,sig = em_init_params(K, None, None, None)
		pp = []

		DX,DY = [],[]
		
		YY = np.linspace(0, self.H, ny)
		XX = np.linspace(0, self.W, nx)
		px0 = None
		for y in YY:
			pprow = []
			for ix,x in enumerate(XX):
				dx = (x - self.x0) / self.xscale
				dy = (y - self.y0) / self.yscale
				DX.append(dx)
				DY.append(dy)
				
				if ix == 0 and px0 is not None:
					w,mu,sig = px0
				im = self.instantiateAt(x, y, scale=scale)
				PS = im.shape[0]
				im /= im.sum()
				im = np.maximum(im, 0)
				xm,ym = -(PS/2), -(PS/2)
				em_fit_2d(im, xm, ym, w, mu, sig)
				#print 'w,mu,sig', w,mu,sig
				if ix == 0:
					px0 = w,mu,sig

				if not scale:
					# We didn't downsample the PSF in pixel space, so
					# scale down the MOG params.
					sfactor = self.sampling
				else:
					sfactor = 1.

				params = np.hstack((w.ravel()[:-1],
									(sfactor * mu).ravel(),
									(sfactor**2 * sig[:,0,0]).ravel(),
									(sfactor**2 * sig[:,0,1]).ravel(),
									(sfactor**2 * sig[:,1,1]).ravel())).copy()
				pprow.append(params)

				if False:
					psf = GaussianMixturePSF(w, mu, sig)
					patch = psf.getPointSourcePatch(13,13)
					pim = np.zeros_like(im)
					patch.addTo(pim)
					
					mx = max(im.max(), pim.max())
					plt.clf()
					plt.subplot(1,2,1)
					plt.imshow(im, interpolation='nearest', origin='lower',
							   vmin=0, vmax=mx)
					plt.subplot(1,2,2)
					plt.imshow(pim, interpolation='nearest', origin='lower',
							   vmin=0, vmax=mx)
					ps.savefig()
			pp.append(pprow)

		pp = np.array(pp)
		#print 'pp', pp.shape
		# ND = len(pp[:,:,0].ravel())
		# DX = np.array(DX)
		# DY = np.array(DY)
		# A = np.zeros((ND, 10))
		# A[:,0] = 1.
		# A[:,1] = DX
		# A[:,2] = DY
		# A[:,3] = DY**2
		# A[:,4] = DX*DY
		# A[:,5] = DX**2
		# A[:,6] = DY**3
		# A[:,7] = DX * DY**2
		# A[:,8] = DX**2 * DY
		# A[:,9] = DY**3

		splines = []
		N = len(pp[0][0])

		for i in range(N):
			data = pp[:,:, i]
			spl = interp.RectBivariateSpline(XX, YY, data.T)
			#print 'Building a spline on XX,YY,data', XX.shape, YY.shape, data.T.shape
			splines.append(spl)
			
			#X,res,rank,s = np.linalg.lstsq(A, data.ravel())
			##print 'Coefficients', X
			#bb = np.dot(A, X)
			##print 'reconstruction', bb
			#bb = bb.reshape(pp[:,:,i].shape)
			##print 'bb shape', bb.shape
			#
			#if False:
			#	plt.clf()
			#	cc = ['r','y','g','b','m']
			#	xx = np.linspace(0, self.W, 100)
			#	scale = (nx-1) / float(self.W-1)
			#	plt.subplot(2,2,1)
			#	ss = []
			#	for j in range(ny):
			#		plt.plot(pp[j, :, i], '-', color=cc[j%len(cc)])
			#		plt.plot(bb[j, :], '--', color=cc[j%len(cc)])
			#		plt.plot(xx * scale, spl(xx, YY[j]), ':', color=cc[j%len(cc)])
			#		s = spl(XX, YY[j])
			#		#print 'splined', s
			#		ss.append(s)
			#	#print 'ss', ss
			#	ss = np.hstack(ss).T
			#	#print 'ss', ss
			#	#print 'shape', ss.shape
			#	mn,mx = data.min(), data.max()
			#	ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
			#	plt.subplot(2,2,2)
			#	plt.imshow(data, **ima)
			#	plt.title('data')
			#	plt.subplot(2,2,3)
			#	plt.imshow(bb, **ima)
			#	plt.title('poly')
			#	plt.subplot(2,2,4)
			#	plt.imshow(ss, **ima)
			#	plt.title('spline')
			#	ps.savefig()

		self.splines = splines
		self.K = K

	def mogAt(self, x, y):
		if not hasattr(self, 'splines'):
			self._fitParamGrid(self.nx, self.ny, self.K)
		vals = [spl(x, y) for spl in self.splines]
		K = self.K
		w = np.empty(K)
		w[:-1] = vals[:K-1]
		vals = vals[K-1:]
		w[-1] = 1. - sum(w[:-1])
		mu = np.empty((K,2))
		mu.ravel()[:] = vals[:2*K]
		vals = vals[2*K:]
		sig = np.empty((K,2,2))
		sig[:,0,0] = vals[:K]
		vals = vals[K:]
		sig[:,0,1] = vals[:K]
		sig[:,1,0] = sig[:,0,1]
		vals = vals[K:]
		sig[:,1,1] = vals[:K]
		vals = vals[K:]
		return GaussianMixturePSF(w, mu, sig)
		
	def instantiateAt(self, x, y, scale=True):
		psf = np.zeros_like(self.psfbases[0])
		#print 'psf', psf.shape
		dx = (x - self.x0) / self.xscale
		dy = (y - self.y0) / self.yscale
		i = 0
		#print 'dx',dx,'dy',dy
		for d in range(self.degree + 1):
			#print 'degree', d
			for j in range(d+1):
				k = d - j
				#print 'x',j,'y',k,
				#print 'component', i
				amp = dx**j * dy**k
				#print 'amp', amp,
				# PSFEx manual pg. 111 ?
				ii = j + (self.degree+1) * k - (k * (k-1))/ 2
				#print 'ii', ii, 'vs i', i
				psf += self.psfbases[ii] * amp
				#print 'basis rms', np.sqrt(np.mean(self.psfbases[i]**2)),
				i += 1
				#print 'psf sum', psf.sum()
		#print 'min', psf.min(), 'max', psf.max()

		if scale and self.sampling != 1:
			ny,nx = psf.shape
			spsf = affine_transform(psf, [1./self.sampling]*2,
									offset=nx/2 * (self.sampling - 1.))
			return spsf
			
		return psf

if __name__ == '__main__':
	import sys
	import matplotlib
	matplotlib.use('Agg')
	import pylab as plt

	psf = PsfEx('cs82data/ptf/PTF_201112091448_i_p_scie_t032828_u010430936_f02_p002794_c08.fix.psf', 2048, 4096)

	ps = PlotSequence('ex')
	
	for scale in [False, True]:
		im = psf.instantiateAt(0.,0., scale=scale)

		ny,nx = im.shape
		XX,YY = np.meshgrid(np.arange(nx), np.arange(ny))
		print 'cx', (np.sum(im * XX) / np.sum(im))
		print 'cy', (np.sum(im * YY) / np.sum(im))
		
		plt.clf()
		mx = im.max()
		plt.imshow(im, origin='lower', interpolation='nearest',
				   vmin=-0.1*mx, vmax=mx*1.1)
		plt.hot()
		plt.colorbar()
		ps.savefig()

	print 'PSF scale', psf.sampling
	print '1./scale', 1./psf.sampling

	YY = np.linspace(0, 4096, 5)
	XX = np.linspace(0, 2048, 5)

	yims = []
	for y in YY:
		xims = []
		for x in XX:
			im = psf.instantiateAt(x, y)
			im /= im.sum()
			xims.append(im)
		xims = np.hstack(xims)
		yims.append(xims)
	yims = np.vstack(yims)
	plt.clf()
	plt.imshow(yims, origin='lower', interpolation='nearest')
	plt.gray()
	plt.hot()
	plt.title('instantiated')
	ps.savefig()
	
	for scale in [True, False]:
		print 'fitting params, scale=', scale
		psf._fitParamGrid(nx=11, ny=11, scale=scale)

		sims = []
		for y in YY:
			xims = []
			for x in XX:
				mog = psf.mogAt(x, y)
				patch = mog.getPointSourcePatch(12,12)
				pim = np.zeros((25,25))
				patch.addTo(pim)
				pim /= pim.sum()
				xims.append(pim)
			xims = np.hstack(xims)
			sims.append(xims)
		sims = np.vstack(sims)
		plt.clf()
		plt.imshow(sims, origin='lower', interpolation='nearest')
		plt.gray()
		plt.hot()
		plt.title('Spline mogs, scale=%s' % str(scale))
		ps.savefig()
	
	sys.exit(0)
	
	# import cPickle
	# print 'Pickling...'
	# SS = cPickle.dumps(psf)
	# print 'UnPickling...'
	# psf2 = cPickle.loads(SS)
	
	#psf.fitParamGrid(nx=5, ny=6)
	#psf.fitParamGrid(nx=10, ny=10)

	YY = np.linspace(0, 4096, 5)
	XX = np.linspace(0, 2048, 5)

	yims = []
	for y in YY:
		xims = []
		for x in XX:
			im = psf.instantiateAt(x, y)
			print x,y
			im /= im.sum()
			xims.append(im)
			print 'shape', xims[-1].shape
		xims = np.hstack(xims)
		print 'xims shape', xims
		yims.append(xims)
	yims = np.vstack(yims)
	print 'yims shape', yims
	plt.clf()
	plt.imshow(yims, origin='lower', interpolation='nearest')
	plt.gray()
	plt.hot()
	plt.title('PSFEx')
	ps.savefig()

	sims = []
	for y in YY:
		xims = []
		for x in XX:
			mog = psf.mogAt(x, y)
			patch = mog.getPointSourcePatch(12,12)
			pim = np.zeros((25,25))
			patch.addTo(pim)
			pim /= pim.sum()
			xims.append(pim)
		xims = np.hstack(xims)
		sims.append(xims)
	sims = np.vstack(sims)
	plt.clf()
	plt.imshow(sims, origin='lower', interpolation='nearest')
	plt.gray()
	plt.hot()
	plt.title('Spline')
	ps.savefig()


	plt.clf()
	im = yims.copy()
	im = np.sign(im) * np.sqrt(np.abs(im))
	ima = dict(origin='lower', interpolation='nearest',
			   vmin=im.min(), vmax=im.max())
	plt.imshow(im, **ima)
	plt.title('PSFEx')
	ps.savefig()

	plt.clf()
	im = sims.copy()
	im = np.sign(im) * np.sqrt(np.abs(im))
	plt.imshow(im, **ima)
	plt.title('Spline')
	ps.savefig()
	
