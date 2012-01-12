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

	# Run 4868 camcol 4 field 30 PSF FWHM 4.24519
	# Run 7164 camcol 4 field 266 PSF FWHM 6.43368
	run = 4868
	camcol = 4
	field = 31
	xc,yc = 1530, 1147
	#run = 7164
	#camcol = 4
	#field = 267
	#xc,yc = 1485,1003
	bandname = 'r'
	#xc,yc = 1510, 1040
	#ra,dec = (333.556, 0.369)
	S = 100
	roi = [xc-S, xc+S, yc-S, yc+S]
	timg2,info2 = st.get_tractor_image(run, camcol, field, bandname, roi=roi,
									   useMags=True)


	# Create CFHT tractor.Image
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

	#wcs.setX0Y0(x0+1., y0+1.)
	# From fit
	wcs.setX0Y0(535.14208988131043, 4153.665639423165)

	sky = np.median(image)
	print 'Sky', sky
	# save for later...
	cfsky = sky
	skyobj = ConstantSky(sky)

	photocal = cf.CfhtPhotoCal(hdr=pyfits.open(cffn)[0].header)

	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal,
				   name='CFHT')

	class ScaledWCS(object):
		def hashkey(self):
			return ('ScaledWcs', self.scale, self.wcs.hashkey())
		def __init__(self, scale, wcs):
			self.wcs = wcs
			self.scale = float(scale)
		def positionToPixel(self, src, pos):
			x,y = self.wcs.positionToPixel(src,pos)
			return x*self.scale, y*self.scale
		def pixelToPosition(self, src, xy):
			x,y = xy
			xy = (x/self.scale, y/self.scale)
			return self.wcs.pixelToPosition(src, xy)
		def cdAtPixel(self, x, y):
			cd = self.wcs.cdAtPixel(x/self.scale, y/self.scale)
			return cd / self.scale

	# Create fake tractor.Image
	psf = NCircularGaussianPSF([0.1], [1.])
	#sky = ConstantSky(100.)
	sky = timg.sky
	print 'SDSS Sky:', timg.sky
	photocal = timg.photocal
	fakescale = 3
	#wcs = ScaledWCS(fakescale, timg.wcs)
	#(h,w) = timg.data.shape
	#fakedata = np.zeros((h*fakescale, w*fakescale))

	#
	# 0.066
	# make-wcs.py -r 333.55503 -d 0.36438 -s 0.02 -W 1200 -H 1200 fake-wcs.fits
	# # flip parity
	# modhead fake-wcs.fits CD1_1 0
	# modhead fake-wcs.fits CD1_2 1.666666667e-5
	# modhead fake-wcs.fits CD2_1 1.666666667e-5
	# modhead fake-wcs.fits CD2_2 0
	
	wcs = FitsWcs(Tan('fake-wcs.fits', 0))
	#h,w = 600,600
	h,w = 1200,1200

	fakescale = 0.396 * w / (0.02 * 3600.)

	fakedata = np.zeros((h, w))
	print '0,0', wcs.pixelToPosition(None, (0,0))
	print 'W,0', wcs.pixelToPosition(None, (w,0))
	print '0,H', wcs.pixelToPosition(None, (0,h))
	print 'W,H', wcs.pixelToPosition(None, (w,h))
	fakewcs = wcs

	'''
	0,0 RA,Dec (333.56505, 0.37440)
	W,0 RA,Dec (333.54505, 0.37440)
	0,H RA,Dec (333.56505, 0.35440)
	W,H RA,Dec (333.54505, 0.35440)
	'''
	
	fakeimg = Image(data=fakedata, invvar=fakedata, psf=psf,
					wcs=wcs, sky=sky, photocal=photocal, name='Fake')
	del psf
	del wcs
	del photocal

	tractor = Tractor([timg, cftimg, timg2])
	tractor.addSources(sources)

	zrs = [np.array([-1.,+6.]) * info['skysig'] + info['sky'],
		   np.array([-1.,+20.]) * cfstd + cfsky,
		   np.array([-1.,+6.]) * info2['skysig'] + info2['sky'],]


	
	zrf = np.array([-1./float(fakescale**2),
					+6./float(fakescale**2)]) * info['skysig'] + info['sky']
	print 'zrf', zrf
	imfake = dict(interpolation='nearest', origin='lower',
				  vmin=zrf[0], vmax=zrf[1], cmap='gray')
	mod = tractor.getModelImage(fakeimg)

	#fig = plt.figure()
	#fig.patch.set_alpha(0.)
	plt.clf()
	plt.imshow(mod, **imfake)
	#plt.title('step %i: %s' % (step-1, action))
	#plt.colorbar()
	plt.savefig('mod-fake.png')
	print 'Wrote fake model'

	seg = sources[1]
	print 'Source:', seg
	mod = tractor.getModelImage(fakeimg, srcs=[seg])
	plt.clf()
	plt.imshow(mod, **imfake)
	#plt.title('step %i: %s' % (step-1, action))
	#plt.colorbar()
	plt.savefig('mod-fake-eg.png')
	print 'Wrote fake model'

	patch = seg.getModelPatch(fakeimg)
	px0,py0 = patch.x0, patch.y0
	print 'Patch', patch
	plt.clf()
	zrf2 = np.array([-1./float(fakescale**2),
					+6./float(fakescale**2)]) * info['skysig']

	plt.imshow(patch.patch, interpolation='nearest', origin='lower',
			   vmin=zrf2[0], vmax=zrf2[1], cmap='gray')

	ax = plt.axis()
	# 333.55503, 0.36438
	ramid,decmid = 333.55503, 0.36438
	#ravals = np.arange(333.54, 333.58, 0.01)
	#decvals = np.arange(0.35, 0.39, 0.01)
	ravals = np.arange(333.55, 333.56, 0.001)
	decvals = np.arange(0.36, 0.37, 0.001)
	# NOTE flipped [0],[1] indices... (modhead above)
	xvals = [fakewcs.positionToPixel(None, RaDecPos(ra, decmid))[1]
			 for ra in ravals]
	yvals = [fakewcs.positionToPixel(None, RaDecPos(ramid, dec))[0]
			 for dec in decvals]
	xvals = np.array(xvals) - px0
	yvals = np.array(yvals) - py0
	print 'yvals', yvals
	plt.xticks(xvals, ['%.3f'%ra for ra in ravals])
	plt.xlabel('RA (deg)')
	plt.yticks(yvals, ['%.3f'%dec for dec in decvals])
	plt.ylabel('Dec (deg)')

	plt.axis(ax)
	plt.savefig('mod-fake-eg-patch.pdf')
	plt.savefig('mod-fake-eg-patch.png')


	oldpsf = timg.psf
	timg.psf = fakeimg.psf
	
	patch = seg.getModelPatch(timg)
	plt.clf()
	#zrf2 = np.array([-1./float(fakescale**2),
	#				+6./float(fakescale**2)]) * info['skysig']
	#zr = zrs[0]
	zr = np.array([-1.,+6.]) * info['skysig'] #+ info['sky']
	plt.imshow(patch.patch, interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1], cmap='gray')
	plt.savefig('mod-sdss1-eg-patch.pdf')
	plt.savefig('mod-sdss1-eg-patch.png')

	timg.psf = oldpsf
	
	patch = seg.getModelPatch(timg)
	px0,py0 = patch.x0, patch.y0
	plt.clf()
	plt.imshow(patch.patch, interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1], cmap='gray')
	plt.savefig('mod-sdss2-eg-patch.pdf')
	plt.savefig('mod-sdss2-eg-patch.png')

	ph,pw = patch.patch.shape
	subimg = timg.getImage()[py0:py0+ph, px0:px0+pw]
	plt.clf()
	sky = info['sky']
	plt.imshow(subimg-sky, interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1], cmap='gray')
	plt.savefig('mod-sdss3-eg-patch.pdf')
	plt.savefig('mod-sdss3-eg-patch.png')


	plt.figure(figsize=(6,6))
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.94]


	derivs = seg.getParamDerivatives(fakeimg)
	for j,d in enumerate(derivs):
		if d is None:
			print 'No derivative for param', j
		mx = max(abs(d.patch.max()), abs(d.patch.min()))
		print 'mx', mx
		print 'Patch size:', d.patch.shape
		print 'patch x0,y0', d.x0, d.y0
		mim = np.zeros_like(fakeimg.getImage())
		d.addTo(mim)
		S = 25
		mim = mim[600-S:600+S, 600-S:600+S]
		plt.clf()
		plt.gca().set_position(plotpos0)
		plt.imshow(mim, #d.patch,
				   interpolation='nearest',
				   origin='lower', cmap='gray',
				   vmin=-mx/10., vmax=mx/10.)
		plt.title(d.name)
		plt.xticks([],[])
		plt.yticks([],[])
		plt.savefig('deriv-eg-%i.png' % j)

	zrf2 = np.array([-1./float(fakescale**2),
					 +20./float(fakescale**2)]) * info['skysig']

	patch = seg.getModelPatch(fakeimg)
	mim = np.zeros_like(fakeimg.getImage())
	patch.addTo(mim)
	mim = mim[600-S:600+S, 600-S:600+S]
	plt.clf()
	plt.gca().set_position(plotpos0)
	plt.imshow(mim, #d.patch,
			   interpolation='nearest',
			   origin='lower', cmap='gray',
			   vmin=zrf2[0], vmax=zrf2[1])
	plt.title('model')
	plt.xticks([],[])
	plt.yticks([],[])
	plt.savefig('deriv-eg-model.png')

	sys.exit(0)





	#np.seterr(under='print')
	np.seterr(all='warn')

	action = 'Initial'

	#plt.gca().set_position(plotpos0)

	#NS = 10
	NS = 1
	for step in range(1, NS+1):
		
		for i in range(len(tractor.getImages())):
			mod = tractor.getModelImage(i)
			zr = zrs[i]
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=zr[0], vmax=zr[1], cmap='gray')
			imchi = dict(interpolation='nearest', origin='lower',
						 vmin=-5., vmax=+5., cmap='gray')

			if step == 1:
				data = tractor.getImage(i).getImage()
				plt.clf()
				plt.gca().set_position(plotpos0)
				plt.imshow(data, **ima)
				#plt.title('step %i: %s' % (step-1, action))
				plt.title('Data')
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('data%i.png' % i)


			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(mod, **ima)
			ax = plt.axis()
			for j,src in enumerate(tractor.getCatalog()):
				im = tractor.getImage(i)
				wcs = im.getWcs()
				x,y = wcs.positionToPixel(None, src.getPosition())
				plt.plot([x],[y],'r.')
				srct = src.getSourceType()
				srct = srct[0]
				#plt.text(x, y, '%i:%s'%(j,srct))
			plt.axis(ax)
			#plt.title('step %i: %s' % (step-1, action))
			plt.title('Model')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('mod%i-%02i.png' % (i,step-1))


			chi = tractor.getChiImage(i)
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(chi, **imchi)
			#plt.title('step %i: %s' % (step-1, action))
			plt.title('Chi')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('chi%i-%02i.png' % (i,step-1))

		# troublesome guy...
		src = tractor.getCatalog()[6]
		print 'Troublesome source:', src

		if step == NS:
			break

		print
		print '---------------------------------'
		print 'Step', step
		print '---------------------------------'
		print
		if step in [1, 2] and len(tractor.getImages())>1:
			action = 'skip'
			continue
			action = 'astrometry'
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

		elif step in [3, 4, 7, 8]:
			action = 'brightness, separately'

			print 'Optimizing brightnesses separately...'
			for j,src in enumerate(tractor.getCatalog()):
				#tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
				#							brightnessonly=True)
				print 'source', j, src
				dlnp,dX,alph = tractor.optimizeCatalogBrightnesses(srcs=[src])
				print 'dlnp', dlnp
				print 'dX', dX
				print 'alpha', alph


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
					print 'Patch size:', d.patch.shape
					print 'patch x0,y0', d.x0, d.y0
					plt.clf()
					plt.imshow(d.patch, interpolation='nearest',
							   origin='lower', cmap='gray',
							   vmin=-mx/2., vmax=mx/2.)
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



		elif step in [5]:
			action = 'sources, separately'
			print 'Optimizing sources individually...'
			for src in tractor.getCatalog():
				tractor.optimizeCatalogLoop(nsteps=1, srcs=[src])

		elif step in [6]:
			action = 'brightnesses, jointly'
			print 'Optimizing brightnesses jointly...'
			tractor.optimizeCatalogLoop(nsteps=1, brightnessonly=True)
			#for src in tractor.getCatalog():
			#	tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
			#								brightnessonly=True)


		else:
			action = 'sources, jointly'
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
	
