import sys
import logging

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits
from astrometry.util.util import Tan
from tractor import *
from tractor import sdss as st
from tractor import cfht as cf
from tractor import sdss_galaxy as stgal
from astrometry.sdss import *


def main():

	lvl = logging.INFO
	#lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	run = 2728
	camcol = 4
	field = 236
	bandname = 'r'
	xc,yc = 1510, 1040
	ra,dec = (333.556, 0.369)
	S = 100
	roi = [xc-S, xc+S, yc-S, yc+S]

	timg,info = st.get_tractor_image(run, camcol, field, bandname, roi=roi,
									 useMags=True)
	sources = st.get_tractor_sources(run, camcol, field, bandname, roi=roi,
									 useMags=True)

	# sdss = DR7()
	# rerun = 0
	# bandnum = band_index(bandname)
	# tsf = sdss.readTsField(run, camcol, field, rerun)
	# photocal = SdssPhotoCalMag(tsf, bandnum)
	# # TEST
	# counts = 100
	# mag = photocal.countsToBrightness(counts)
	# print 'Counts', counts, '-> mag', mag
	# c2 = photocal.brightnessToCounts(mag)
	# print '-> counts', c2
	# # Replace the photocal and the Mags.
	# for s in sources:
	# 	##if hasattr(s, 'getBrightness'):
	# 	# UGH!
	# 	x,y = timg.getWcs().positionToPixel(None, s.getPosition())
	# 
	# 	print 'Converting brightness for source', s
	# 	print 'params', s.getParams()
	# 
	# 	if isinstance(s, stgal.CompositeGalaxy):
	# 		counts = timg.photocal.brightnessToCounts(s.brightnessExp)
	# 		s.brightnessExp = photocal.countsToBrightness(counts)
	# 		counts = timg.photocal.brightnessToCounts(s.brightnessDev)
	# 		s.brightnessDev = photocal.countsToBrightness(counts)
	# 		#print '  ', s.brightnessExp, s.brightnessDev
	# 	else:
	# 		#print 'Source', s, 'brightness', s.getBrightness()
	# 		counts = timg.photocal.brightnessToCounts(s.getBrightness())
	# 		s.setBrightness(photocal.countsToBrightness(counts))
	# 		#print '  ', s.brightness
	# 
	# 	print 'After converting brightness:', s
	# 	print 'params', s.getParams()
	# 	#print 'bright', s.getBrightness()
	# 	#print 'bright2', s.brightness
	# 
	# timg.photocal = photocal




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

	photocal = cf.CfhtPhotoCal(hdr=pyfits.open(cffn)[0].header)

	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal,
				   name=('CFHT'))
	
	tractor = Tractor([timg, cftimg])
	#tractor = Tractor([timg])
	tractor.addSources(sources)

	zrs = [np.array([-2.,+6.]) * info['skysig'] + info['sky'],
		   np.array([-2.,+20.]) * cfstd + sky,]

	#np.seterr(under='print')
	np.seterr(all='warn')

	NS = 9
	for step in range(2, NS+1):
		
		for i in range(len(tractor.getImages())):
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
		if step in [0, 1] and len(tractor.getImages())>1:
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
			if step == -2:
				# troublesome guy...
				src = tractor.getCatalog()[6]
				print 'Troublesome source:', src
				im = tractor.getImage(0)
				derivs = src.getParamDerivatives(im)
				f1 = src.brightnessExp
				f2 = src.brightnessDev
				print 'Brightnesses', f1, f2
				c1 = im.getPhotoCal().brightnessToCounts(f1)
				c2 = im.getPhotoCal().brightnessToCounts(f2)
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

				derivs = src.getParamDerivatives(im, brightnessonly=True)
				print 'Brightness derivs', derivs
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


			print 'Optimizing brightnesses separately...'
			for j,src in enumerate(tractor.getCatalog()):
				#tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
				#							brightnessonly=True)
				print 'source', j, src
				dlnp,dX,alph = tractor.optimizeCatalogBrightnesses(srcs=[src])
				print 'dlnp', dlnp
				print 'dX', dX
				print 'alpha', alph


			#print 'Optimizing sources individually...'
			#for src in tractor.getCatalog():
			#	tractor.optimizeCatalogLoop(nsteps=1, srcs=[src])


		elif step in [4]:
			print 'Optimizing brightnesses jointly...'
			tractor.optimizeCatalogLoop(nsteps=1, brightnessonly=True)
			#for src in tractor.getCatalog():
			#	tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
			#								brightnessonly=True)

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




if __name__ == '__main__':
	main()
	
