if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
from math import sqrt
import numpy as np
import pylab as plt

from astrometry.util.pyfits_utils import *
from astrometry.sdss import *

#from sdsstractor import *
#from fitpsf import em_init_params
#from emfit import em_fit_2d
from tractor import *
from tractor import sdss as st
from tractor.fitpsf import em_init_params
from tractor.emfit import em_fit_2d

def main():
	from optparse import OptionParser
	import sys

	parser = OptionParser(usage=('%prog'))

	parser.add_option('-r', '--run', dest='run', type='int')
	parser.add_option('-c', '--camcol', dest='camcol', type='int')
	parser.add_option('-f', '--field', dest='field', type='int')
	parser.add_option('-b', '--band', dest='band', help='SDSS Band (u, g, r, i, z)')
	#parser.add_option('-R', '--rerun', dest='rerun', type='int')
	parser.add_option('--curl', dest='curl', action='store_true', default=False, help='Use "curl", not "wget", to download files')

	(opt, args) = parser.parse_args()

	run = opt.run
	field = opt.field
	camcol = opt.camcol
	band = opt.band

	rerun = 0

	if run is None or field is None or camcol is None or band is None:
		parser.print_help()
		print
		print 'Must supply --run, --camcol, --field, --band'
		sys.exit(-1)

	if not band in ['u','g','r','i', 'z']:
		parser.print_help()
		print
		print 'Must supply band (u/g/r/i/z)'
		sys.exit(-1)

	bandname = band
	bandnum = band_index(bandname)

	sdss = DR7(curl=opt.curl)
	for filetype in ['fpC', 'tsObj', 'tsField', 'psField']:
		fn = sdss.getFilename(filetype, run, camcol, field, bandname)
		print 'Looking for file', fn
		if not os.path.exists(fn):
			print 'Retrieving', fn
			sdss.retrieve(filetype, run, camcol, field, bandnum)

	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias
	image = fpC

	tsf = sdss.readTsField(run, camcol, field, rerun)
	astrans = tsf.getAsTrans(bandnum)
	wcs = st.SdssWcs(astrans)
	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(0.5, 0.5)

	psfield = sdss.readPsField(run, camcol, field)
	if False:
		dgpsf = psfield.getDoubleGaussian(bandnum)
		print 'Creating double-Gaussian PSF approximation'
		(a,s1, b,s2) = dgpsf
		psf = NCircularGaussianPSF([s1, s2], [a, b])

	photocal = st.SdssPhotoCal()
	sky = psfield.getSky(bandnum)
	skysig = sqrt(sky)
	skyobj = ConstantSky(sky)
	zr = np.array([-3.,+10.]) * skysig + sky

	# we don't care about the invvar (for now)
	invvar = np.zeros_like(image)

	# Create Gaussian mixture model PSF approximation.
	H,W = image.shape
	klpsf = psfield.getPsfAtPoints(bandnum, W/2, H/2)
	S = klpsf.shape[0]
	# number of Gaussian components
	K = 3
	w,mu,sig = em_init_params(K, None, None, None)
	II = klpsf.copy()
	II /= II.sum()
	# HIDEOUS HACK
	II = np.maximum(II, 0)
	print 'Multi-Gaussian PSF fit...'
	xm,ym = -(S/2), -(S/2)
	em_fit_2d(II, xm, ym, w, mu, sig)
	print 'w,mu,sig', w,mu,sig
	psf = GaussianMixturePSF(w, mu, sig)
	
	timg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name='SDSS (r/c/f=%i/%i%i)' % (run, camcol, field))
	tractor = st.SDSSTractor([timg])

	# Select objects to keep; initialize tractor source objects for them.

	objs = fits_table(sdss.getFilename('tsObj', run, camcol, field,
									   bandname, rerun=rerun))
	# HACK
	#x = objs.colc[:,bandnum]
	#y = objs.rowc[:,bandnum]
	#objs = objs[(x > 1300)*(x < 1800)*(y < 300)]
	#objs = objs[(objs.r_exp[:,bandnum] / objs.ab_exp[:,bandnum] > 5.) *
	#			(objs.counts_exp[:,bandnum] < 21.)]

	objs = objs[(objs.nchild == 0)]
	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Ldev = ((objs.prob_psf[:,bandnum] == 0) * objs.fracpsf[:,bandnum])
	Lexp = ((objs.prob_psf[:,bandnum] == 0) * (1. - objs.fracpsf[:,bandnum]))

	# NO FUCKING IDEA why NOT necessary to get PA and adjust for it.
	# in DR7, tsObj files have phi_exp, phi_dev in image coordinates, not sky coordinates.
	# Correct by finding the position angle of the field on the sky.
	# cd = wcs.cdAtPixel(W/2, H/2)
	# pa = np.rad2deg(np.arctan2(cd[0,1], cd[0,0]))
	# print 'pa=', pa
	# pa = np.rad2deg(np.arctan2(cd[1,0], cd[0,0]))
	# print 'pa=', pa
	# pa = np.rad2deg(np.arctan2(cd[0,1], -cd[1,1]))
	# print 'pa=', pa
	# pa = np.rad2deg(np.arctan2(cd[1,0], -cd[1,1]))
	# print 'pa=', pa

	# HACK -- DR7 phi opposite to Tractor phi
	objs.phi_dev = - objs.phi_dev
	objs.phi_exp = - objs.phi_exp

	# MAGIC -- minimum size of galaxy.
	objs.r_dev = np.maximum(objs.r_dev, 1./30.)
	objs.r_exp = np.maximum(objs.r_exp, 1./30.)

	for i in np.flatnonzero(np.logical_or(Lstar>0, np.logical_or(Ldev>0, Lexp>0))):
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		x,y = wcs.positionToPixel(None, pos)
		assert(x > 0)
		assert(y > 0)
		assert(x < W)
		assert(y < H)
		
	# Add stars
	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'

	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		counts = tsf.luptitude_to_counts(objs.psfcounts[i,bandnum], bandnum)
		flux = st.SdssFlux(counts / st.SdssPhotoCal.scale)
		ps = PointSource(pos, flux)
		tractor.addSource(ps)

	# Add deV galaxies
	I = np.flatnonzero(Ldev > 0)
	print len(I), 'deV galaxies'
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		#counts = tsf.luptitude_to_counts(objs.counts_dev[i,bandnum], bandnum)
		counts = tsf.luptitude_to_counts(objs.counts_model[i,bandnum], bandnum)
		#print 'Luptitude', objs.counts_model[i,bandnum]
		#print 'Counts', counts
		counts *= Ldev[i]
		flux = st.SdssFlux(counts / st.SdssPhotoCal.scale)
		re = objs.r_dev[i,bandnum]
		ab = objs.ab_dev[i,bandnum]
		phi = objs.phi_dev[i,bandnum]
		ps = st.HoggDevGalaxy(pos, flux, re, ab, phi)
		tractor.addSource(ps)

	# Add exp galaxies
	I = np.flatnonzero(Lexp > 0)
	print len(I), 'exp galaxies'
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		#counts = tsf.luptitude_to_counts(objs.counts_exp[i,bandnum], bandnum)
		# apportion the best-fit composite model between exp and dev.
		counts = tsf.luptitude_to_counts(objs.counts_model[i,bandnum], bandnum)
		counts *= Lexp[i]
		flux = st.SdssFlux(counts / st.SdssPhotoCal.scale)
		re = objs.r_exp[i,bandnum]
		ab = objs.ab_exp[i,bandnum]
		phi = objs.phi_exp[i,bandnum]
		ps = st.HoggExpGalaxy(pos, flux, re, ab, phi)
		tractor.addSource(ps)

	mods = tractor.getModelImages()
	mod = mods[0]

	pyfits.writeto('synth-%06i-%s%i-%04i.fits' % (run, bandname, camcol, field), mod, clobber=True)

	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])
	imdiff = dict(interpolation='nearest', origin='lower',
				  vmin=-20, vmax=20)

	plt.clf()
	plt.imshow(image, **ima)
	plt.colorbar()
	plt.gray()
	plt.savefig('img.png')

	plt.clf()
	plt.imshow(mod, **ima)
	plt.colorbar()
	plt.gray()
	plt.savefig('mod.png')

	plt.clf()
	plt.imshow(image - mod, **imdiff)
	plt.colorbar()
	plt.gray()
	plt.savefig('diff.png')

	plt.clf()
	plt.imshow(image - np.median(image), **imdiff)
	plt.colorbar()
	plt.gray()
	plt.savefig('img2.png')

	plt.clf()
	plt.imshow(mod - np.median(image), **imdiff)
	plt.colorbar()
	plt.gray()
	plt.savefig('mod2.png')


if __name__ == '__main__':
	main()
