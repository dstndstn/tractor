if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import sys
from math import pi

import numpy as np
import pylab as plt

from astrometry.util.sip import Tan
from astrometry.util.file import *

from sdsstractor import *

def makeTruth():
	W,H = 500,500
	ra,dec = 0,0
	width = 0.1
	
	wcs = Tan()
	wcs.crval[0] = ra
	wcs.crval[1] = dec
	wcs.crpix[0] = W/2.
	wcs.crpix[1] = H/2.
	scale = width / float(W)
	wcs.cd[0] = -scale
	wcs.cd[1] = 0
	wcs.cd[2] = 0
	wcs.cd[3] = -scale
	wcs.imagew = W
	wcs.imageh = H
	tanwcs1 = wcs
	wcs1 = FitsWcs(wcs)

	# rotate.
	W2 = int(W*np.sqrt(2.))
	H2 = W2
	rot = 30.
	cr = np.cos(np.deg2rad(rot))
	sr = np.sin(np.deg2rad(rot))
	wcs = Tan()
	wcs.crval[0] = ra
	wcs.crval[1] = dec
	wcs.crpix[0] = W2/2.
	wcs.crpix[1] = H2/2.
	wcs.cd[0] = -scale * cr
	wcs.cd[1] =  scale * sr
	wcs.cd[2] = -scale * sr
	wcs.cd[3] = -scale * cr
	wcs.imagew = W2
	wcs.imageh = H2
	wcs2 = FitsWcs(wcs)

	photocal = SdssPhotoCal(SdssPhotoCal.scale)
	psf = NCircularGaussianPSF([2.0], [1.0])
	sky = 0.
	skyobj = ConstantSky(sky)
	flux = SdssFlux(1.)

	# image 1
	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1e-4
	img1 = Image(data=image, invvar=invvar, psf=psf, wcs=wcs1,
				 sky=skyobj, photocal=photocal, name='Grid1')
	# image 2
	image = np.zeros((H2,W2))
	invvar = np.zeros_like(image) + 1e-4
	img2 = Image(data=image, invvar=invvar, psf=psf, wcs=wcs2,
				 sky=skyobj, photocal=photocal, name='Grid2')

	tractor = SDSSTractor([img1, img2])

	np.random.seed(42)

	# arcsec
	#re = 10.
	#phi = 30.

	# grid: ra -- ab
	for i,(x,a) in enumerate(zip(np.linspace(50, W-50, 10),
								np.linspace(0.1, 1, 10))):
		#       dec -- phi
		#       dec -- r_e
		for j,(y,re) in enumerate(zip(np.linspace(50, H-50, 10),
									  exp(np.linspace(np.log(1), np.log(30.), 10)))):
									  #exp(np.linspace(np.log(0.1), np.log(20.), 10)))):
			#np.linspace(0, 90, 10, endpoint=False),
			ra,dec = tanwcs1.pixelxy2radec(x, y)
			pos = RaDecPos(ra, dec)
			phi = np.random.uniform(0, 360)
			#eg = ExpGalaxy(pos, flux, re, a, p)
			#eg = HoggExpGalaxy(pos, flux, re, a, phi)
			eg = HoggDevGalaxy(pos, flux, re, a, phi)
			tractor.catalog.append(eg)

	imgs = tractor.getModelImages()
	for i,img in enumerate(imgs):
		plt.clf()
		plt.imshow(img, interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.savefig('grid%i.png' % i)


	for i,img in enumerate(imgs):
		timg = tractor.getImage(i)
		noise = (np.random.normal(size=timg.invvar.shape) *
				 np.sqrt(1. / timg.invvar))
		timg.data = img + noise

		plt.clf()
		plt.imshow(timg.getImage(),
				   interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.savefig('grid%in.png' % i)

		pyfits.writeto('grid%in.fits' % i, timg.getImage(), clobber=True)

	return tractor.images


def main():
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-v', '--verbose', dest='verbose', action='count',
					  default=0, help='Make more verbose')
	opt,args = parser.parse_args()
	print 'Opt.verbose = ', opt.verbose
	if opt.verbose == 0:
		lvl = logging.INFO
	else: # opt.verbose == 1:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s',
						stream=sys.stdout)

	set_fp_err()

	print
	print 'makeTruth...'
	print
	imgs = makeTruth()
	print
	print 'done makeTruth.'
	print
	tractor = SDSSTractor(imgs)

	xyfn = 'grid0n.xy'
	if not os.path.exists(xyfn):
		print 'Running image2xy...'
		cmd = 'image2xy %s -o %s' % (xyfn.replace('.xy', '.fits'), xyfn)
		print 'Command:', cmd
		os.system(cmd)
	assert(os.path.exists(xyfn))
	sxy = fits_table(xyfn)

	cat = tractor.getCatalog()
	for i in range(len(sxy)):
		# MAGIC -1: simplexy produces FITS-convention coords
		x = sxy.x[i] - 1.
		y = sxy.y[i] - 1.
		ix = int(round(x))
		iy = int(round(y))
		#src = tractor.createNewSource(imgs[0], x, y, sxy.flux[i])
		#src = tractor.createNewSource(imgs[0], x, y, sxy.flux[i])
		#cat.append(src)

		img = tractor.getImage(0)
		wcs = img.getWcs()
		pos = wcs.pixelToPosition(None, (ix,iy))
		photocal = img.getPhotoCal()
		flux = photocal.countsToFlux(1e6) #sxy.flux[i])

		#cat.append(ExpGalaxy(pos, flux, 1., 0.5, 0.))
		#cat.append(HoggExpGalaxy(pos, flux, 1., 0.5, 0.))
		cat.append(HoggDevGalaxy(pos, flux, 1., 0.5, 0.))


	Nsteps = 30

	for i in range(Nsteps):
		html = '<html><body>'
		html += '<p>Step %i ' % i
		if i > 0:
			html += '&nbsp;<a href="grid%02i.html">Prev</a>' % (i-1)
		html += '&nbsp;<a href="grid%02i.html">Next</a>' % (i+1)
		if i >= 5:
			html += '&nbsp;<a href="grid%02i.html">-5</a>' % (i-5)
		html += '&nbsp;<a href="grid%02i.html">+5</a>' % (i+5)
		html += '</p>'
		for j in range(2):
			html += '<img src="grid%02i-%in" width="600" />' % (i,j)
			html += '<img src="grid%i.png" width="600" />' % j
		html += '</body></html>'
		write_file(html, 'grid%02i.html' % i)

	for step in range(Nsteps):
		imgs = tractor.getModelImages()
		for i,img in enumerate(imgs):
			plt.clf()
			plt.imshow(img, interpolation='nearest', origin='lower')
			fn = 'grid%02i-%in.png' % (step, i)
			plt.colorbar()
			plt.savefig(fn)
			print 'saved', fn

		#if step == 0:
		#	tractor.changeSourceTypes()
		#else:
		print
		print 'Step', step
		print
		for src in tractor.getCatalog():
			print ' ', src
		tractor.optimizeCatalogAtFixedComplexityStep()


	sys.exit(0)

	(images, simplexys, rois, zrange, nziv, footradecs
	 ) = prepareTractor(False, False, rcfcut=[0])

	print 'Creating tractor...'
	tractor = SDSSTractor(images, debugnew=False, debugchange=True)
	opt,args = parser.parse_args()

	src = DevGalaxy(pos, flux, 0.9, 0.75, -35.7)
	src.setParams([120.60275, 9.41350, 5.2767052, 22.8, 0.81, 5.0])
	tractor.catalog.append(src)

	def makePlots(tractor, fnpat, title1='', title2=''):
		mods = tractor.getModelImages()
		imgs = tractor.getImages()
		chis = tractor.getChiImages()
		for i,(mod,img,chi) in enumerate(zip(mods,imgs,chis)):
			zr = zrange[i]
			imargs = dict(interpolation='nearest', origin='lower',
						  vmin=zr[0], vmax=zr[1])
			srcpatch = src.getModelPatch(img)
			slc = srcpatch.getSlice(img)
			plt.clf()
			plt.subplot(2,2,1)
			plotimage(img.getImage()[slc], **imargs)
			plt.title(title1)
			plt.subplot(2,2,2)
			plotimage(mod[slc], **imargs)
			plt.title(title2)
			plt.subplot(2,2,3)
			plotimage(chi[slc], interpolation='nearest', origin='lower',
					  vmin=-5, vmax=5)
			plt.title('chi')
			#plt.subplot(2,2,4)
			#plotimage(img.getInvError()[slc],
			#		  interpolation='nearest', origin='lower',
			#		  vmin=0, vmax=0.1)
			#plt.title('inv err')
			fn = fnpat % i
			plt.savefig(fn)
			print 'wrote', fn

	makePlots(tractor, 'opt-s00-%02i.png', #'pre-%02i.png',
			  title2='re %.1f, ab %.2f, phi %.1f' % (src.re, src.ab, src.phi))

	for ostep in range(10):
		print
		print 'Optimizing...'
		#alphas = [1., 0.5, 0.25, 0.1, 0.01]
		alphas=None
		ppre = src.getParams()
		lnppre = tractor.getLogProb()
		dlnp,X,alpha = tractor.optimizeCatalogAtFixedComplexityStep(alphas=alphas)
		ppost = src.getParams()
		makePlots(tractor, 'opt-s%02i-%%02i.png' % (ostep+1),
				  title1='dlnp = %.1f' % dlnp,
				  title2='re %.1f, ab %.2f, phi %.1f' % (src.re, src.ab, src.phi))

		print
		src.setParams(ppre)
		print 'Pre :', src

		src.setParams(ppost)
		print 'Post:', src


if __name__ == '__main__':
	import cProfile
	from datetime import tzinfo, timedelta, datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	sys.exit(0)
