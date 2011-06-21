if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
from math import pi, sqrt, ceil, floor
import pyfits
import pylab as plt
import numpy as np
import matplotlib

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.util import *

from tractor import *
from sdsstractor import *


def main():
	sdss = DR7()

	bandname = 'g'
	bandnum = band_index(bandname)

	# ref
	run,camcol,field = 6955,3,809
	rerun = 0

	ra,dec = 53.202125, -0.365361

	for filetype in ['fpC', 'tsObj', 'tsField', 'psField']:
		fn = sdss.getFilename(filetype, run, camcol, field, bandname)
		print 'Looking for file', fn
		if not os.path.exists(fn):
			print 'Retrieving', fn
			sdss.retrieve(filetype, run, camcol, field, bandnum)
	
	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias

	objs = fits_table(sdss.getFilename('tsObj', run, camcol, field,
									   bandname, rerun=rerun))
	print 'objs', objs

	tsf = sdss.readTsField(run, camcol, field, rerun)
	print 'tsField:', tsf
	astrans = tsf.getAsTrans(bandnum)
	#print 'astrans', astrans
	x,y = astrans.radec_to_pixel(ra, dec)
	print 'asTrans x,y', x,y

	# Render ref source into ref (sub-)image.

	psfield = sdss.readPsField(run, camcol, field)
	dgpsf = psfield.getDoubleGaussian(bandnum)
	print 'Creating double-Gaussian PSF approximation'
	print '  ', dgpsf
	(a,s1, b,s2) = dgpsf
	psf = NCircularGaussianPSF([s1, s2], [a, b])

	# half cutout size
	dpix = 150

	x0,x1 = int(x - dpix), int(x + dpix)
	y0,y1 = int(y - dpix), int(y + dpix)
	print 'x0,x1', (x0,x1), 'y0,y1', (y0,y1)

	roislice = (slice(y0,y1), slice(x0,x1))
	image = fpC[roislice]

	wcs = SdssWcs(astrans)
	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(x0+0.5, y0+0.5)

	photocal = SdssPhotoCal(SdssPhotoCal.scale)
	sky = psfield.getSky(bandnum)
	skysig = sqrt(sky)
	skyobj = ConstantSky(sky)
	zr = np.array([-3.,+10.]) * skysig + sky

	# we don't care about the invvar (for now)
	invvar = np.zeros_like(image)

	mod = np.zeros_like(image)
	H,W = mod.shape

	# Create Gaussian mixture model PSF approximation.
	klpsf = psfield.getPsfAtPoints(bandnum, x0 + W/2, y0 + H/2)
	S = klpsf.shape[0]
	# works well enough
	K = 3
	from fitpsf import em_init_params
	from emfit import em_fit_2d
	w,mu,sig = em_init_params(K, None, None, None)
	II = klpsf.copy()
	II /= II.sum()

	print 'Starting multi-Gaussian PSF fit...'
	xm,ym = -(S/2), -(S/2)
	em_fit_2d(II, xm, ym, w, mu, sig)
	print 'w,mu,sig', w,mu,sig
	psf = GaussianMixturePSF(w, mu, sig)

	timg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name='Ref (r/c/f=%i/%i%i)' % (run, camcol, field))

	tractor = SDSSTractor([timg], debugnew=False, debugchange=True)

	# Find tsObj sources within the image.
	## HACK -- I'm just grabbing things whose *centers* are within the img
	r,c = objs.rowc[:,bandnum], objs.colc[:,bandnum]
	#print 'r,c', r,c
	I = ((r > y0) * (r < y1) * (c > x0) * (c < x1))
	objs = objs[I]
	#print 'objs:', objs
	print len(objs), 'within the sub-image'

	# parent / nchild
	# ra, dec
	# #colc, rowc,
	# psfcounts
	# r_dev, ab_dev, phi_dev, counts_dev
	# r_exp, ab_exp, phi_exp, counts_exp
	# fracpsf
	# probpsf

	# Keep just the deblended children.
	objs = objs[(objs.nchild == 0)]

	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Ldev = ((objs.prob_psf[:,bandnum] == 0) * objs.fracpsf[:,bandnum])
	Lexp = ((objs.prob_psf[:,bandnum] == 0) * (1. - objs.fracpsf[:,bandnum]))
	print 'Lstar', Lstar
	print 'Ldev', Ldev
	print 'Lexp', Lexp

	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'
	print 'psf luptitudes:', objs.psfcounts[I,bandnum]
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		#counts = tsf.mag_to_counts(objs.psfcounts[i,bandnum], bandnum)
		counts = tsf.luptitude_to_counts(objs.psfcounts[i,bandnum], bandnum)
		#print 'counts:', counts
		flux = SdssFlux(counts / SdssPhotoCal.scale)
		ps = PointSource(pos, flux)
		tractor.addSource(ps)

		#x,y = wcs.positionToPixel(None, pos)
		#r,c = objs.rowc[i,bandnum],objs.colc[i,bandnum]
		#print 'x,y', x+x0, y+y0
		#print 'c,r', c, r

	I = np.flatnonzero(Ldev > 0)
	print len(I), 'deV galaxies'
	print 'luptitudes:', objs.counts_dev[I,bandnum]
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		counts = tsf.luptitude_to_counts(objs.counts_dev[i,bandnum], bandnum)
		flux = SdssFlux(counts / SdssPhotoCal.scale)
		re = objs.r_dev[i,bandnum]
		ab = objs.ab_dev[i,bandnum]
		phi = objs.phi_dev[i,bandnum]
		ps = HoggDevGalaxy(pos, flux, re, ab, phi)
		tractor.addSource(ps)

	I = np.flatnonzero(Lexp > 0)
	print len(I), 'exp galaxies'
	print 'counts:', objs.counts_exp[I,bandnum]
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		counts = tsf.luptitude_to_counts(objs.counts_exp[i,bandnum], bandnum)
		flux = SdssFlux(counts / SdssPhotoCal.scale)
		re = objs.r_exp[i,bandnum]
		ab = objs.ab_exp[i,bandnum]
		phi = objs.phi_exp[i,bandnum]
		ps = HoggExpGalaxy(pos, flux, re, ab, phi)
		tractor.addSource(ps)
	
	mods = tractor.getModelImages()
	mod = mods[0]

	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])
	imdiff = dict(interpolation='nearest', origin='lower',
				  vmin=-10, vmax=10)

	plt.clf()
	plt.imshow(image, **ima)
	plt.colorbar()
	plt.savefig('refimg.png')

	plt.clf()
	plt.imshow(mod, **ima)
	plt.colorbar()
	plt.savefig('refmod.png')

	plt.clf()
	plt.imshow(image - mod, **imdiff)
	plt.colorbar()
	plt.savefig('refdiff.png')



	imb = dict(interpolation='nearest', origin='lower',
			   vmin=np.arcsinh(zr[0]-sky), vmax=np.arcsinh(zr[1]-sky))
	plt.clf()
	plt.imshow(np.arcsinh(image - sky), **imb)
	plt.colorbar()
	plt.savefig('refimg2.png')

	plt.clf()
	plt.imshow(np.arcsinh(mod - sky), **imb)
	plt.colorbar()
	plt.savefig('refmod2.png')



if __name__ == '__main__':
	main()

