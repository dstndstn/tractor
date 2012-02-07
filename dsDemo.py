import sys
import logging
import traceback

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits
from astrometry.util.util import Tan
from astrometry.util.file import *
from tractor import *
from tractor import sdss as st
from tractor import cfht as cf
from tractor import sdss_galaxy as stgal
from astrometry.sdss import *


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


def fakeimg_plots():
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


def get_cfht_img(ra, dec, extent):
	# Create CFHT tractor.Image
	cffn = 'cr.fits'
	psffn = 'psfimg.fits'

	wcs = FitsWcs(Tan(cffn, 0))
	print 'CFHT WCS', wcs

	x,y = wcs.positionToPixel(None, RaDecPos(ra,dec))
	print 'x,y', x,y
	cfx,cfy = x,y

	cd = wcs.cdAtPixel(x,y)
	pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
	print 'pixscale', pixscale
	S = int(extent / pixscale)
	print 'S', S

	#cfx,cfy = 734,4352
	#S = 200
	cfroi = [cfx-S, cfx+S, cfy-S, cfy+S]
	x0,x1,y0,y1 = cfroi

	wcs.setX0Y0(x0+1., y0+1.)
	# From fit
	#wcs.setX0Y0(535.14208988131043, 4153.665639423165)

	I = pyfits.open(cffn)[1].data
	print 'Img data', I.shape
	roislice = (slice(y0,y1), slice(x0,x1))
	image = I[roislice]

	sky = np.median(image)
	print 'Sky', sky
	# save for later...
	cfsky = sky
	skyobj = ConstantSky(sky)

	# Third plane in image: variance map.
	I = pyfits.open(cffn)[3].data
	var = I[roislice]
	cfstd = np.sqrt(np.median(var))

	## FIXME -- add source photon noise, read noise
	# actually the read noise will have already been measured by LSST
	phdr = pyfits.open(cffn)[0].header
	# e/ADU
	gain = phdr.get('GAIN')
	# Poisson statistics are on electrons; var = mean
	el = np.maximum(0, (image - sky) * gain)
	# var in ADU...
	srcvar = el / gain**2
	invvar = 1./(var + srcvar)
	#darkcur = phdr.get('DARKCUR')
	#readnoise = phdr.get('RDNOISE')

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

	photocal = cf.CfhtPhotoCal(hdr=phdr,
							   bandname='r')

	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal,
				   name='CFHT')
	return cftimg, cfsky, cfstd


def makeTractor(bands, ra, dec, S, RCFS):
	TI = []
	for i,(run,camcol,field) in enumerate(RCFS):

		for bandname in bands:
			# HACK - get whole-frame tractor Image to get WCS to find ROI.
			im,inf = st.get_tractor_image(run, camcol, field, bandname,
										  useMags=True, psf='dg')
			wcs = im.getWcs()
			fxc,fyc = wcs.positionToPixel(None, RaDecPos(ra,dec))
			xc,yc = [int(np.round(p)) for p in fxc,fyc]

			roi = [xc-S, xc+S, yc-S, yc+S]
			im,inf = st.get_tractor_image(run, camcol, field, bandname,
										  roi=roi,	useMags=True)
			im.dxdy = (fxc - xc, fyc - yc)
			TI.append((im,inf))

			if i == 0 and bandname == 'r':
				# im,info = TI[bands.index('r')]
				#wcs = im.getWcs()
				# this is a shifted WCS; S,S is the center.
				# rd = wcs.pixelToPosition(None, (S,S))
				# ra,dec = rd.ra,rd.dec
				# print 'RA,Dec', ra,dec
				cd = wcs.cdAtPixel(xc,yc)
				pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
				print 'pixscale', pixscale * 3600.
				extent = pixscale * S

	#for timg,info in TI:
	#	print timg.hashkey()

	sources = st.get_tractor_sources(run, camcol, field, roi=roi,
									 bands=bands)

	print 'Sources:'
	for s in sources:
		print s
	print

	print 'Tractor images:', TI

	cftimg,cfsky,cfstd = get_cfht_img(ra,dec, extent)

	# Create fake tractor.Image
	#  psf = NCircularGaussianPSF([0.1], [1.])
	#  sky = timg.sky
	#  print 'SDSS Sky:', timg.sky
	#  photocal = timg.photocal
	#  fakescale = 3
	#  #wcs = ScaledWCS(fakescale, timg.wcs)
	#  #(h,w) = timg.data.shape
	#  #fakedata = np.zeros((h*fakescale, w*fakescale))
	#  # 0.066
	#  # make-wcs.py -r 333.55503 -d 0.36438 -s 0.02 -W 1200 -H 1200 fake-wcs.fits
	#  # # flip parity
	#  # modhead fake-wcs.fits CD1_1 0
	#  # modhead fake-wcs.fits CD1_2 1.666666667e-5
	#  # modhead fake-wcs.fits CD2_1 1.666666667e-5
	#  # modhead fake-wcs.fits CD2_2 0
	#  wcs = FitsWcs(Tan('fake-wcs.fits', 0))
	#  #h,w = 600,600
	#  h,w = 1200,1200
	#  fakescale = 0.396 * w / (0.02 * 3600.)
	#  fakedata = np.zeros((h, w))
	#  # print '0,0', wcs.pixelToPosition(None, (0,0))
	#  # print 'W,0', wcs.pixelToPosition(None, (w,0))
	#  # print '0,H', wcs.pixelToPosition(None, (0,h))
	#  # print 'W,H', wcs.pixelToPosition(None, (w,h))
	#  fakewcs = wcs
	#  fakeimg = Image(data=fakedata, invvar=fakedata, psf=psf,
	#  				wcs=wcs, sky=sky, photocal=photocal, name='Fake')
	#  del psf
	#  del wcs
	#  del photocal

	tims = [cftimg] + [timg for timg,tinf in TI]
	# for tim in tims:
	# 	try:
	# 		print 'Test pickling im', tim.name
	# 		pickle_to_file(tim, 'test.pickle')
	# 	except:
	# 		print 'failed:'
	# 		traceback.print_exc()
	# 		pass
	# for src in sources:
	# 	try:
	# 		print 'Test pickling source', src
	# 		pickle_to_file(src, 'test.pickle')
	# 	except:
	# 		print 'failed:'
	# 		traceback.print_exc()
	# 		pass
	# sys.exit(0)
	
	tractor = Tractor(tims)
	tractor.addSources(sources)

	skyvals = ( [(cfstd, cfsky)] +
				[(info['skysig'], info['sky'])
				 for timg,info in TI] )
	return tractor,skyvals

def main():

	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-s', type=int, dest='step')
	opt,args = parser.parse_args()

	# cffn = 'cr.fits'
	# wcs = Tan(cffn, 0)
	# pickle_to_file(wcs, 'test.pickle')
	# wcs = FitsWcs(wcs)
	# pickle_to_file(wcs, 'test.pickle')
	# wcs2 = unpickle_from_file('test.pickle')
	# print wcs2
	# sys.exit(0)

	lvl = logging.INFO
	#lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
	np.seterr(all='warn')
	np.seterr(divide='raise')

	#bands = ['u', 'g','r','i', 'z']
	#bands = ['g','r','i']
	#bands = ['r']

	# LBL
	bands = ['r','z']


	# Run 4868 camcol 4 field 30 PSF FWHM 4.24519
	# Run 7164 camcol 4 field 266 PSF FWHM 6.43368
	#   -> PSF model sucks for this one (too broad for KL eigen size?)

	# canonical band from which to get initial position, shape
	#ra,dec = (333.556, 0.369)
	ra,dec = (333.5596, 0.3671)
	S = 80

	RCFS = [(2728, 4, 236),
			(4868, 4,  31)]
	#RCFS = [(2728, 4, 236),]

	settings = dict(bands=bands, ra=ra, dec=dec, S=S, RCFS=RCFS)

	if opt.step is not None:
		pfn = 'tractor-step%02i.pickle' % opt.step
		if os.path.exists(pfn):
			tractor,settins = unpickle_from_file(pfn)
			for k,v in settings.items():
				assert(settins[k] == v)
			settings = settins
			skyvals = settings['skyvals']

			# for src in tractor.getCatalog():
			# 	print 'Source', src
			# for img in tractor.getImages():
			# 	print 'Image', img
			# 	wcs = img.getWcs()
			# 	print 'wcs', wcs
			# 	print wcs.hashkey()
	
	else:
		tractor,skyvals = makeTractor(**settings)

		pickle_to_file(tractor, 'test.pickle')
		tr = unpickle_from_file('test.pickle')

		settings['skyvals'] = skyvals

	tims = tractor.getImages()

	# CFI = len(tims)-1
	def cfimshow(im, *args, **kwargs):
		return plt.imshow(np.rot90(im, k=1), *args, **kwargs)

	CFI = 0
	zrs = []
	for i,(st,sky) in enumerate(skyvals):
		if i == CFI:
			zrs.append(np.array([-3.,+18.]) * st + sky)
		else:
			zrs.append(np.array([-1.,+6.]) * st + sky)

	#RGBS = [((3,2,1),'SDSS r/c/f %i/%i/%i gri' % (RCFS[0])),
	#		((6,5,4),'SDSS r/c/f %i/%i/%i gri' % (RCFS[1]))]
	RGBS = []



	plt.figure(figsize=(6,6))
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.94]

	plotpos1 = [0.01, 0.01, 0.9, 0.94]

	#fakeimg_plots(fakescale, ...)

	action = 'Initial'

	steptypes = ([ 'nil', 'wcs','wcs','wcs',
				   'bright', 'bright', 'source', 'jbright',
				   'bright', 'bright', ] +
				 ['jsource', 'source']*4 +
				 ['simplify'] +
				 ['complexify'] +
				 ['nil'])
				  
	NS = len(steptypes) - 1
	#NS = 15
	#NS = 1

	step0 = 1
	if opt.step is not None:
		step0 = opt.step

	for step in range(step0, NS+1):

		stype = steptypes[step]
		print 'Step', step, ':', stype

		if step == 1:
			for j,((ri,gi,bi),rgbname) in enumerate(RGBS):
				ims = []
				for ii in ri,gi,bi:
					lo,hi = zrs[ii]
					tim = tractor.getImage(ii)
					print 'dx,dy', tim.dxdy
					im = (tim.getImage() - lo) / (hi-lo)
					ims.append(im)

				plt.clf()
				plt.gca().set_position(plotpos0)
				plt.imshow(np.clip(np.dstack(ims), 0, 1), interpolation='nearest', origin='lower')
				plt.title('Data %s' % rgbname)
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('rgbdata%02i.png' % j)
				
		
		for i in range(len(tractor.getImages())):
			mod = tractor.getModelImage(i)
			zr = zrs[i]
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=zr[0], vmax=zr[1], cmap='gray')
			imchi = dict(interpolation='nearest', origin='lower',
						 vmin=-5., vmax=+5., cmap='gray')

			if step == 1:
				tim = tractor.getImage(i)
				data = tim.getImage()
				plt.clf()
				plt.gca().set_position(plotpos0)

				if i == CFI:
					cfimshow(data, **ima)
				else:
					plt.imshow(data, **ima)
				#plt.title('step %i: %s' % (step-1, action))
				plt.title('Data %s' % tim.name)
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('data%02i.png' % i)

				plt.colorbar()
				plt.savefig('datacb%02i.png' % i)
				iv = tim.getInvError()**2
				plt.clf()
				if i == CFI:
					iv = np.rot90(iv,k=1)
				plt.imshow(iv, interpolation='nearest', origin='lower',
						   vmin=0, vmax=np.max(iv), cmap='gray')
				plt.colorbar()
				plt.title('Invvar %s' % tim.name)
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('ivar%02i.png' % i)
						   

			plt.clf()
			plt.gca().set_position(plotpos0)
			if i == CFI:
				cfimshow(mod, **ima)
			else:
				plt.imshow(mod, **ima)

			# HACK -- plot objects on first SDSS frame.
			if i == 1:
				ax = plt.axis()
				xx,yy,tt = [],[],[]
				for j,src in enumerate(tractor.getCatalog()):
					im = tractor.getImage(i)
					wcs = im.getWcs()
					x,y = wcs.positionToPixel(None, src.getPosition())
					xx.append(x)
					yy.append(y)
					srct = src.getSourceType()
					srct = srct[0]
					tt.append(srct)
				plt.plot(xx,yy, 'r.')
				plt.axis(ax)
			#plt.title('step %i: %s' % (step-1, action))
			plt.title('Model')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('mod%02i-%02i.png' % (i,step-1))
			# HACK -- add annotations
			if i == 1:
				for x,y,t in zip(xx,yy,tt):
					#plt.text(x, y, '%i:%s'%(j,srct))
					plt.text(x+1.5, y+1.5, '%s'%t, color='r')
				plt.axis(ax)
				plt.savefig('modann%02i-%02i.png' % (i,step-1))
				
			chi = tractor.getChiImage(i)
			plt.clf()
			plt.gca().set_position(plotpos0)
			if i == CFI:
				cfimshow(chi, **imchi)
			else:
				plt.imshow(chi, **imchi)
			#plt.title('step %i: %s' % (step-1, action))
			plt.title('Chi')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('chi%02i-%02i.png' % (i,step-1))

		# troublesome guy...
		#src = tractor.getCatalog()[6]
		#print 'Troublesome source:', src

		pickle_to_file((tractor,settings),
					   'tractor-step%02i.pickle' % step)

		if step == NS:
			break

		print
		print '---------------------------------'
		print 'Step', step
		print '---------------------------------'
		print
		if stype == 'wcs':
			#action = 'skip'
			#continue
			action = 'astrometry'
			# fine-tune astrometry
			print 'Optimizing CFHT astrometry...'
			cfim = tractor.getImage(CFI)
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

		elif stype == 'bright':
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



		elif stype == 'source':
			action = 'sources, separately'
			print 'Optimizing sources individually...'
			for src in tractor.getCatalog():
				tractor.optimizeCatalogLoop(nsteps=1, srcs=[src])

		elif stype == 'jbright':
			action = 'brightnesses, jointly'
			print 'Optimizing brightnesses jointly...'
			tractor.optimizeCatalogLoop(nsteps=1, brightnessonly=True)
			#for src in tractor.getCatalog():
			#	tractor.optimizeCatalogLoop(nsteps=1, srcs=[src],
			#								brightnessonly=True)


		elif stype == 'jsource':
			action = 'sources, jointly'
			print 'Optimizing sources jointly...'
			tractor.optimizeCatalogLoop(nsteps=1)

		elif stype == 'simplify':
			# Try removing each source in turn.
			for j,src in enumerate(tractor.getCatalog()):
				cat = tractor.getCatalog()
				ii = cat.index(src)
				lnp0 = tractor.getLogProb()
				p0 = cat.getAllParams()
				print 'Try removing source', src
				print 'lnp0:', lnp0
				tractor.removeSource(src)
				lnp1 = tractor.getLogProb()
				print 'dlnp1:', (lnp1 - lnp0)
				tractor.optimizeCatalogLoop(nsteps=5)
				lnp2 = tractor.getLogProb()
				print 'dlnp2:', (lnp2 - lnp0)

				plt.clf()
				plt.gca().set_position(plotpos0)
				cfim = tractor.getImages()[CFI]
				mod = tractor.getModelImage(cfim)
				zr = zrs[CFI]
				ima = dict(interpolation='nearest', origin='lower',
						   vmin=zr[0], vmax=zr[1], cmap='gray')
				cfimshow(mod, **ima)
				fn = 'mod-rem%02i-%02i-%02i.png' % (j, CFI, step)
				plt.savefig(fn)
				print 'saved', fn

				if lnp2 > lnp0:
					continue
				else:
					# reinsert
					cat.insert(ii, src)
					cat.setAllParams(p0)
					print 'Reverted'
					lnp3 = tractor.getLogProb()
					print 'lnp3', lnp3
					assert(lnp3 == lnp0)
					
			
		elif stype == 'complexify':
			for j,src in enumerate(tractor.getCatalog()):
				cat = tractor.getCatalog()
				ii = cat.index(src)
				p0 = cat.getAllParams()
				print 'Try complexifying source', src
				#if isinstance(src, PointSource):
				newsrc = None
				if (isinstance(src, stgal.ExpGalaxy) or
					isinstance(src, stgal.DevGalaxy)):
					# HACK
					faintmag = 21
					faint = Mags(**dict([(b,faintmag) for b in bands]))
					print 'Faint mag:', faint
					args = [src.pos]
					if isinstance(src, stgal.ExpGalaxy):
						args.extend([src.brightness, src.shape])
						args.extend([faint, src.shape])
					else:
						args.extend([faint, src.shape])
						args.extend([src.brightness, src.shape])
					newsrc = stgal.CompositeGalaxy(*args)
				if newsrc is None:
					continue

				lnp0 = tractor.getLogProb()
				print 'lnp0:', lnp0

				print 'Replacing', src
				print '     with', newsrc
				tractor.removeSource(src)
				tractor.addSource(newsrc)
				lnp1 = tractor.getLogProb()
				print 'dlnp1:', (lnp1 - lnp0)
				print 'Optimizing new source...'
				tractor.optimizeCatalogLoop(nsteps=5, src=[newsrc])
				lnp2 = tractor.getLogProb()
				print 'dlnp2:', (lnp2 - lnp0)
				print 'Optimizing everything...'
				tractor.optimizeCatalogLoop(nsteps=5)
				lnp3 = tractor.getLogProb()
				print 'dlnp3:', (lnp3 - lnp0)

				plt.clf()
				plt.gca().set_position(plotpos0)
				cfim = tractor.getImages()[CFI]
				mod = tractor.getModelImage(cfim)
				zr = zrs[CFI]
				ima = dict(interpolation='nearest', origin='lower',
						   vmin=zr[0], vmax=zr[1], cmap='gray')
				cfimshow(mod, **ima)
				fn = 'mod-complex%02i-%02i-%02i.png' % (j, CFI, step)
				plt.savefig(fn)
				print 'saved', fn

				if lnp3 > lnp0:
					print 'Keeping this change!'
					continue
				else:
					# reinsert
					cat.remove(newsrc)
					cat.insert(ii, src)
					cat.setAllParams(p0)
					print 'Reverted'
					lnp3 = tractor.getLogProb()
					print 'lnp3', lnp3
					assert(lnp3 == lnp0)

			
		else:
			print 'Unknown step type', stype


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
	
