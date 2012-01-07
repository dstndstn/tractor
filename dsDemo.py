if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits
from astrometry.util.util import Tan
from tractor import *
from tractor import sdss as st
from tractor import sdss_galaxy as stgal
from astrometry.sdss import *


def main():

	run = 2728
	camcol = 4
	field = 236
	bandname = 'r'
	xc,yc = 1510, 1040
	ra,dec = (333.556, 0.369)
	S = 100
	roi = [xc-S, xc+S, yc-S, yc+S]

	timg,info = st.get_tractor_image(run, camcol, field, bandname, roi=roi)
	sources = st.get_tractor_sources(run, camcol, field, bandname, roi=roi)

	sdss = DR7()
	rerun = 0
	bandnum = band_index(bandname)
	tsf = sdss.readTsField(run, camcol, field, rerun)
	photocal = SdssPhotoCalMag(tsf, bandnum)
	# TEST
	counts = 100
	mag = photocal.countsToFlux(counts)
	print 'Counts', counts, '-> mag', mag
	c2 = photocal.fluxToCounts(mag)
	print '-> counts', c2

	# Replace the photocal and the Mags.
	for s in sources:
		##if hasattr(s, 'getFlux'):
		# UGH!
		x,y = timg.getWcs().positionToPixel(None, s.getPosition())
		#print 'x,y (%.1f, %.1f)' % (x,y)
		if isinstance(s, stgal.CompositeGalaxy):
			counts = timg.photocal.fluxToCounts(s.fluxExp)
			s.fluxExp = photocal.countsToFlux(counts)
			counts = timg.photocal.fluxToCounts(s.fluxDev)
			s.fluxDev = photocal.countsToFlux(counts)
			#print '  ', s.fluxExp, s.fluxDev
		else:
			#print 'Source', s, 'flux', s.getFlux()
			counts = timg.photocal.fluxToCounts(s.getFlux())
			s.setFlux(photocal.countsToFlux(counts))
			#print '  ', s.flux
		#else:
		#	print 'no getFlux() method:', s
	timg.photocal = photocal




	cffn = 'cr.fits'
	psffn = 'psfimg.fits'
	cfx,cfy = 734,4352
	S = 200
	cfroi = [cfx-S, cfx+S, cfy-S, cfy+S]

	I = pyfits.open(cffn)[1].data
	print 'Img data', I.shape
	x0,x1,y0,y1 = cfroi
	roislice = (slice(y0,y1), slice(x0,x1))
	image = I[roislice]
	I = pyfits.open(cffn)[3].data
	var = I[roislice]
	cfstd = np.sqrt(np.median(var))
	invvar = 1./var
	## FIXME -- add source photon noise, read noise
	I = pyfits.open(cffn)[2].data
	mask = I[roislice]
	invvar[mask > 0] = 0.
	del I
	del var

	psfimg = pyfits.open(psffn)[0].data
	print 'PSF image shape', psfimg.shape
	from tractor.emfit import em_fit_2d
	from tractor.fitpsf import em_init_params
	# number of Gaussian components
	S = psfimg.shape[0]
	K = 3
	w,mu,sig = em_init_params(K, None, None, None)
	II = psfimg.copy()
	II /= II.sum()
	# HACK
	II = np.maximum(II, 0)
	print 'Multi-Gaussian PSF fit...'
	xm,ym = -(S/2), -(S/2)
	em_fit_2d(II, xm, ym, w, mu, sig)
	print 'w,mu,sig', w,mu,sig
	psf = GaussianMixturePSF(w, mu, sig)

	wcs = FitsWcs(Tan(cffn, 0))
	print 'CFHT WCS', wcs

	#[
	#wcs.setX0Y0(x0+1., y0+1.)
	# From fit
	wcs.setX0Y0(535.14208988131043, 4153.665639423165)

	sky = np.median(image)
	print 'Sky', sky
	skyobj = ConstantSky(sky)

	photocal = CfhtPhotoCal(hdr=pyfits.open(cffn)[0].header)

	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal,
				   name=('CFHT'))
	
	tractor = Tractor([timg, cftimg])
	tractor.addSources(sources)

	zrs = [np.array([-2.,+6.]) * info['skysig'] + info['sky'],
		   np.array([-2.,+20.]) * cfstd + sky,]

	#np.seterr(under='print')
	np.seterr(all='warn')

	NS = 6
	for step in range(2, NS+1):
		
		for i in range(2):
			mod = tractor.getModelImage(i)
			zr = zrs[i]
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=zr[0], vmax=zr[1], cmap='gray')
			imchi = dict(interpolation='nearest', origin='lower',
						 vmin=-5., vmax=+5., cmap='gray')
			plt.clf()
			plt.imshow(mod, **ima)
			ax = plt.axis()
			for j,src in enumerate(tractor.getCatalog()):
				im = tractor.getImage(i)
				wcs = im.getWcs()
				x,y = wcs.positionToPixel(None, src.getPosition())
				plt.plot([x],[y],'r.')
				plt.text(x, y, '%i'%j)
			plt.axis(ax)
			plt.savefig('mod%i-%02i.png' % (i,step))

			if step == 0:
				data = tractor.getImage(i).getImage()
				plt.clf()
				plt.imshow(data, **ima)
				plt.savefig('data%i.png' % i)

			chi = tractor.getChiImage(i)
			plt.clf()
			plt.imshow(chi, **imchi)
			plt.savefig('chi%i-%02i.png' % (i,step))

		if step == NS:
			break

		print 'Step', step
		if step in [0, 1]:
			# fine-tune astrometry
			print 'Optimizing CFHT astrometry...'
			cfim = tractor.getImage(1)
			pa = FitsWcsShiftParams(cfim.getWcs())
			print '# params', pa.numberOfParams()
			derivs = [[] for i in range(pa.numberOfParams())]
			# getParamDerivatives
			p0 = pa.getParams()
			print 'p0:', p0
			mod0 = tractor.getModelImage(cfim)
			psteps = pa.getStepSizes()
			#plt.clf()
			#plt.imshow(mod0, **ima)
			#plt.savefig('dwcs.png')
			for i,pstep in enumerate(psteps):
				pa.stepParam(i,pstep)
				print 'pstep:', pa.getParams()
				mod = tractor.getModelImageNoCache(cfim)
				#print 'model', mod
				#plt.clf()
				#plt.imshow(mod, **ima)
				#plt.savefig('dwcs%i.png' % (i))
				pa.setParams(p0)
				print 'revert pstep:', pa.getParams()
				D = (mod - mod0) / pstep
				# --> convert to Patch
				print 'derivative:', D.min(), np.median(D), D.max()
				D = Patch(0,0,D)
				derivs[i].append((D, cfim))
			
			print 'Derivs', derivs
			X = tractor.optimize(derivs)
			print 'X:',X
			(dlogprob, alpha) = tractor.tryParamUpdates([pa], X) #, alphas)
			print 'pa after:', pa
			print pa.getParams()

		elif step in [2,3]:
			if step == 2:
				# troublesome guy...
				src = tractor.getCatalog()[6]
				print 'Troublesome source:', src
				im = tractor.getImage(1)
				derivs = src.getParamDerivatives(im)
				f1 = src.fluxExp
				f2 = src.fluxDev
				print 'Fluxes', f1, f2
				c1 = im.getPhotoCal().fluxToCounts(f1)
				c2 = im.getPhotoCal().fluxToCounts(f2)
				print 'Counts', c1, c2
				for j,d in enumerate(derivs):
					if d is None:
						print 'No derivative for param', j
					mx = max(abs(d.patch.max()), abs(d.patch.min()))
					print 'mx', mx
					plt.clf()
					plt.imshow(d.patch, interpolation='nearest',
							   origin='lower', cmap='gray',
							   vmin=-mx, vmax=mx)
					plt.title(d.name)
					plt.savefig('deriv%i.png' % j)

					mim = np.zeros_like(im.getImage())
					d.addTo(mim)
					plt.clf()
					plt.imshow(mim, interpolation='nearest',
							   origin='lower', cmap='gray',
							   vmin=-mx, vmax=mx)
					plt.title(d.name)
					plt.savefig('derivb%i.png' % j)

				derivs = src.getParamDerivatives(im, fluxonly=True)
				print 'Flux derivs', derivs
				for j,d in enumerate(derivs):
					if d is None:
						print 'No derivative for param', j
						continue
					mx = max(abs(d.patch.max()), abs(d.patch.min()))
					print 'mx', mx
					plt.clf()
					plt.imshow(d.patch, interpolation='nearest',
							   origin='lower', cmap='gray',
							   vmin=-mx, vmax=mx)
					plt.title(d.name)
					plt.savefig('derivc%i.png' % j)

					mim = np.zeros_like(im.getImage())
					d.addTo(mim)
					plt.clf()
					plt.imshow(mim, interpolation='nearest',
							   origin='lower', cmap='gray',
							   vmin=-mx, vmax=mx)
					plt.title(d.name)
					plt.savefig('derivd%i.png' % j)


			print 'Optimizing fluxes separately...'
			for j,src in enumerate(tractor.getCatalog()):
				#tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
				#							fluxonly=True)
				print 'source', j, src
				dlnp,dX,alph = tractor.optimizeCatalogFluxes(srcs=[src])
				print 'dlnp', dlnp
				print 'dX', dX
				print 'alpha', alph

			sys.exit(0)


			#print 'Optimizing sources individually...'
			#for src in tractor.getCatalog():
			#	tractor.optimizeCatalogLoop(nsteps=1, srcs=[src])


		elif step in [4]:
			print 'Optimizing fluxes jointly...'
			tractor.optimizeCatalogLoop(nsteps=1, fluxonly=True)
			#for src in tractor.getCatalog():
			#	tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
			#								fluxonly=True)

		else:
			print 'Optimizing sources jointly...'
			tractor.optimizeCatalogLoop(nsteps=1)
			
		


class FitsWcsShiftParams(ParamList):
	def __init__(self, wcs):
		super(FitsWcsShiftParams,self).__init__(wcs.x0, wcs.y0)
		self.wcs = wcs
	def getNamedParams(self):
		return [('x0',0),('y0',1)]
	def setParam(self, i, val):
		super(FitsWcsShiftParams,self).setParam(i,val)
		#print 'set wcs x0y0 to', self.vals[0], self.vals[1]
		self.wcs.setX0Y0(self.vals[0], self.vals[1])
	def stepParam(self, i, delta):
		#print 'wcs step param', i, delta
		self.setParam(i, self.vals[i]+delta)
	def getStepSizes(self, *args, **kwargs):
		#return [1.,1.]
		return [0.1, 0.1]
		#return [0.01, 0.01]

class Mag(Flux):
	'''
	An implementation of engine.Flux that actually does mags.
	UGH, bad naming!
	'''
	def hashkey(self):
		return ('Mag', self.val)
	def __repr__(self):
		return 'Mag(%g)' % self.val
	def __str__(self):
		return 'Mag: %g' % self.val
	def copy(self):
		return Mag(self.val)
	def getStepSizes(self, img, *args, **kwargs):
		return [0.01]
	# override positive flux limit from Flux
	def setParams(self, p):
		assert(len(p) == 1)
		self.val = p[0]

class CfhtPhotoCal(object):
	def __init__(self, hdr=None):
		if hdr is not None:
			self.exptime = hdr['EXPTIME']
			self.phot_c = hdr['PHOT_C']
			self.phot_k = hdr['PHOT_K']
			self.airmass = hdr['AIRMASS']
			print 'CFHT photometry:', self.exptime, self.phot_c, self.phot_k, self.airmass
		# FIXME -- NO COLOR TERMS (phot_x)!
		'''
COMMENT   Formula for Photometry, based on keywords given in this header:
COMMENT   m = -2.5*log(DN) + 2.5*log(EXPTIME)
COMMENT   M = m + PHOT_C + PHOT_K*(AIRMASS - 1) + PHOT_X*(PHOT_C1 - PHOT_C2)
'''
			
	def fluxToCounts(self, flux):
		M = flux.getValue()
		logc = (M - self.phot_c - self.phot_k * (self.airmass - 1.)) / -2.5
		return self.exptime * 10.**logc
	def countsToFlux(self, counts):
		return Mag(-2.5 * np.log10(counts / self.exptime) +
				   self.phot_c + self.phot_k * (self.airmass - 1.))
		

class SdssPhotoCalMag(object):
	def __init__(self, tsfield, band):
		'''
		band: int
		'''
		self.tsfield = tsfield
		self.band = band
	def fluxToCounts(self, flux):
		''' ugh, "flux" is a Mag '''
		return self.tsfield.mag_to_counts(flux.getValue(), self.band)

	def countsToFlux(self, counts):
		return Mag(self.tsfield.counts_to_mag(counts, self.band))




if __name__ == '__main__':
	main()
	
