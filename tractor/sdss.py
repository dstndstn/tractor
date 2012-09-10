"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`sdss.py`
===========

SDSS-specific implementation of Tractor elements:

 - retrieves and reads SDSS images, converting them to Tractor language
 - knows about SDSS photometric and astrometric calibration conventions

"""
import os
from math import pi, sqrt, ceil, floor
from datetime import datetime

import pyfits
import pylab as plt
import numpy as np

from engine import *
from basics import *
from sdss_galaxy import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.ngc2000 import ngc2000
from astrometry.util.plotutils import setRadecAxes, redgreen

# might want to override this to set the step size to ~ a pixel
#class SdssRaDecPos(RaDecPos):
#	def getStepSizes(self, img):
#		return [1e-4, 1e-4]

def _check_sdss_files(sdss, run, camcol, field, bandname, filetypes,
					  retrieve=True):
	bandnum = band_index(bandname)
	for filetype in filetypes:
		#fn = sdss.getFilename(filetype, run, camcol, field, bandname)
		fn = sdss.getPath(filetype, run, camcol, field, bandname)
		print 'Looking for file', fn
		if not os.path.exists(fn):
			if retrieve:
				print 'Retrieving', fn
				res = sdss.retrieve(filetype, run, camcol, field, bandnum)
				if res is False:
					raise RuntimeError('No such file on SDSS DAS: %s, rcfb %i/%i/%i/%s' % (filetype, run, camcol, field, bandname))
			else:
				raise os.OSError('no such file: "%s"' % fn)

def _getBrightness(counts, tsf, bands):
	allcounts = counts
	order = []
	kwargs = {}

	BAD_MAG = 99.
	
	for i,counts in enumerate(allcounts):
		bandname = band_name(i)
		if not bandname in bands:
			continue
		if counts == 0:
			mag = BAD_MAG
		else:
			mag = tsf.counts_to_mag(counts, i)
		if not np.isfinite(mag):
			mag = BAD_MAG
		order.append(bandname)
		kwargs[bandname] = mag
		#print 'Band', bandname, 'counts', counts, 'mag', mag
	#print 'creating mags:', kwargs
	m = Mags(order=order, **kwargs)
	#print 'created', m
	return m

def get_tractor_sources(run, camcol, field, bandname='r', sdss=None, release='DR7',
						retrieve=True, curl=False, roi=None, bands=None):
	'''
	Creates tractor.Source objects corresponding to objects in the SDSS catalog
	for the given field.

	bandname: "canonical" band from which to get galaxy shapes, positions, etc

	'''
	if release != 'DR7':
		raise RuntimeError('We only support DR7 currently')
	# FIXME
	rerun = 0

	if bands is None:
		bands = band_names()

	if sdss is None:
		sdss = DR7(curl=curl)
	bandnum = band_index(bandname)
	_check_sdss_files(sdss, run, camcol, field, bandnum,
					  ['tsObj', 'tsField'],
					  #fpC', 'tsField', 'psField', 'fpM'],
					  retrieve=retrieve)

	tsf = sdss.readTsField(run, camcol, field, rerun)

	objs = fits_table(sdss.getPath('tsObj', run, camcol, field,
								   bandname, rerun=rerun))
	objs.indices = np.arange(len(objs))

	if roi is not None:
		x0,x1,y0,y1 = roi
		# HACK -- keep only the sources whose centers are within the ROI box.
		x = objs.colc[:,bandnum]
		y = objs.rowc[:,bandnum]
		I = ((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
		objs = objs[I]

	objs = objs[(objs.nchild == 0)]

	# On further reflection, we believe tsObjs are in sky coords
	# so this (below) is all kool.
	# # NO IDEA why it is NOT necessary to get PA and adjust for it.
	# # (probably that getTensor() has the phi transformation in the wrong
	# # place, terrifying)
	# # Since in DR7, tsObj files have phi_exp, phi_dev in image coordinates,
	# # not sky coordinates.
	# # Should have to Correct by finding the position angle of the field on
	# # the sky.
	# # cd = wcs.cdAtPixel(W/2, H/2)
	# # pa = np.rad2deg(np.arctan2(cd[0,1], cd[0,0]))
	# # print 'pa=', pa
	# HACK -- DR7 phi opposite to Tractor phi, apparently
	objs.phi_dev = -objs.phi_dev
	objs.phi_exp = -objs.phi_exp

	# MAGIC -- minimum size of galaxy.
	objs.r_dev = np.maximum(objs.r_dev, 1./30.)
	objs.r_exp = np.maximum(objs.r_exp, 1./30.)

	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Lgal  = (objs.prob_psf[:,bandnum] == 0)
	Ldev = Lgal * objs.fracpsf[:,bandnum]
	Lexp = Lgal * (1. - objs.fracpsf[:,bandnum])

	sources = []
	ikeep = []

	# Add stars
	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		lups = objs.psfcounts[i,:]
		counts = [tsf.luptitude_to_counts(lup,j) for j,lup in enumerate(lups)]
		bright = _getBrightness(counts, tsf, bands)
		ps = PointSource(pos, bright)
		sources.append(ps)
		ikeep.append(i)

	# Add galaxies.
	I = np.flatnonzero(Lgal > 0)
	print len(I), 'galaxies'
	ndev, nexp, ncomp = 0, 0, 0
	for i in I:
		hasdev = (Ldev[i] > 0)
		hasexp = (Lexp[i] > 0)
		iscomp = (hasdev and hasexp)
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		if iscomp:
			lups = objs.counts_model[i,:]
		elif hasdev:
			lups = objs.counts_dev[i,:]
		elif hasexp:
			lups = objs.counts_exp[i,:]
		else:
			assert(False)
		counts = [tsf.luptitude_to_counts(lup,j) for j,lup in enumerate(lups)]
		counts = np.array(counts)
		#print 'lups', lups
		#print 'counts', counts
											 
		if hasdev:
			dcounts = counts * Ldev[i]
			#print 'dcounts', dcounts
			dbright = _getBrightness(dcounts, tsf, bands)
			#print 'dbright', dbright
			re = objs.r_dev[i,bandnum]
			ab = objs.ab_dev[i,bandnum]
			phi = objs.phi_dev[i,bandnum]
			dshape = GalaxyShape(re, ab, phi)
		if hasexp:
			ecounts = counts * Lexp[i]
			#print 'ecounts', ecounts
			ebright = _getBrightness(ecounts, tsf, bands)
			#print 'ebright', ebright
			re = objs.r_exp[i,bandnum]
			ab = objs.ab_exp[i,bandnum]
			phi = objs.phi_exp[i,bandnum]
			eshape = GalaxyShape(re, ab, phi)

		if iscomp:
			gal = CompositeGalaxy(pos, ebright, eshape, dbright, dshape)
			ncomp += 1
		elif hasdev:
			gal = DevGalaxy(pos, dbright, dshape)
			ndev += 1
		elif hasexp:
			gal = ExpGalaxy(pos, ebright, eshape)
			nexp += 1
		sources.append(gal)
		ikeep.append(i)
	print 'Created', ndev, 'pure deV', nexp, 'pure exp and',
	print ncomp, 'composite galaxies'

	# if you want to cut the objs list to just the ones for which sources were created...
	ikeep = np.unique(ikeep)
	objs = objs[ikeep]

	return sources



def get_tractor_sources_dr8(run, camcol, field, bandname='r', sdss=None,
							retrieve=True, curl=False, roi=None, bands=None,
							badmag=25):
	'''
	Creates tractor.Source objects corresponding to objects in the SDSS catalog
	for the given field.

	bandname: "canonical" band from which to get galaxy shapes, positions, etc

	'''
	if bands is None:
		bands = band_names()
	if sdss is None:
		sdss = DR8(curl=curl)
	bandnum = band_index(bandname)

	bandnums = np.array([band_index(b) for b in bands])
	bandnames = bands

	fn = sdss.retrieve('photoObj', run, camcol, field)
	try:
		objs = fits_table(fn)
	except:
		# Try again in case we got a partially downloaded file
		fn = sdss.retrieve('photoObj', run, camcol, field, skipExisting=False)
		objs = fits_table(fn)

	if objs is None:
		print 'No sources in photoObj file', fn
		return []
	objs.indices = np.arange(len(objs))

	if roi is not None:
		x0,x1,y0,y1 = roi
		# HACK -- keep only the sources whose centers are within the ROI box.
		# Really we should take an object-specific margin.
		x = objs.colc[:,bandnum]
		y = objs.rowc[:,bandnum]
		I = ((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
		objs = objs[I]

	# Only deblended children.
	objs = objs[(objs.nchild == 0)]

	# FIXME -- phi_offset ?

	# DR8 and Tractor have different opinions on which way this rotation goes
	objs.phi_dev_deg *= -1.
	objs.phi_exp_deg *= -1.

	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Lgal  = (objs.prob_psf[:,bandnum] == 0)
	Ldev = Lgal * objs.fracdev[:,bandnum]
	Lexp = Lgal * (1. - objs.fracdev[:,bandnum])

	sources = []
	ikeep = []

	def lup2bright(lups, bandnames, bandnums):
		# -9999 -> badmag
		I = (lups < -100)
		l2 = np.array(lups)
		l2[I] = badmag
		mags = sdss.luptitude_to_mag(l2, bandnums)
		bright = Mags(order=bandnames, **dict(zip(bandnames,mags)))
		return bright

	def nmgy2bright(flux, bandnames):
		I = (flux > 0)
		mag = np.zeros_like(flux) + badmag
		mag[I] = sdss.nmgy_to_mag(flux[I])
		bright = Mags(order=bandnames, **dict(zip(bandnames,mag)))
		return bright
	
	# Add stars
	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		bright = lup2bright(objs.psfmag[i, bandnums], bandnames, bandnums)
		ps = PointSource(pos, bright)
		sources.append(ps)
		ikeep.append(i)

	# Add galaxies.
	I = np.flatnonzero(Lgal > 0)
	print len(I), 'galaxies'
	ndev, nexp, ncomp = 0, 0, 0
	for i in I:
		hasdev = (Ldev[i] > 0)
		hasexp = (Lexp[i] > 0)
		iscomp = (hasdev and hasexp)
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		if iscomp:
			flux = objs.cmodelflux[i,bandnums]
			dflux = flux * Ldev[i]
			dbright = nmgy2bright(dflux, bandnames)
			eflux = flux * Lexp[i]
			ebright = nmgy2bright(eflux, bandnames)
		elif hasdev:
			flux = objs.devflux[i,bandnums]
			dbright = nmgy2bright(flux, bandnames)
			#print 'Dev galaxy: flux', flux, 'bright', dbright
		elif hasexp:
			flux = objs.expflux[i,bandnums]
			ebright = nmgy2bright(flux, bandnames)
			#print 'Exp galaxy: flux', flux, 'bright', ebright
		else:
			assert(False)
											 
		if hasdev:
			re  = objs.theta_dev  [i,bandnum]
			ab  = objs.ab_dev     [i,bandnum]
			phi = objs.phi_dev_deg[i,bandnum]
			dshape = GalaxyShape(re, ab, phi)
		if hasexp:
			re  = objs.theta_exp  [i,bandnum]
			ab  = objs.ab_exp     [i,bandnum]
			phi = objs.phi_exp_deg[i,bandnum]
			eshape = GalaxyShape(re, ab, phi)

		if iscomp:
			gal = CompositeGalaxy(pos, ebright, eshape, dbright, dshape)
			ncomp += 1
		elif hasdev:
			gal = DevGalaxy(pos, dbright, dshape)
			ndev += 1
		elif hasexp:
			gal = ExpGalaxy(pos, ebright, eshape)
			nexp += 1
		sources.append(gal)
		ikeep.append(i)
	print 'Created', ndev, 'pure deV', nexp, 'pure exp and',
	print ncomp, 'composite galaxies'

	# if you want to cut the objs list to just the ones for which sources were created...
	ikeep = np.unique(ikeep)
	objs = objs[ikeep]

	return sources




	
def get_tractor_image(run, camcol, field, bandname, 
					  sdssobj=None, release='DR7',
					  retrieve=True, curl=False, roi=None,
					  psf='kl-gm', useMags=False, roiradecsize=None,
					  savepsfimg=None, zrange=[-3,10]):
	'''
	Creates a tractor.Image given an SDSS field identifier.

	If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
	in the image, in zero-indexed pixel coordinates.  x1,y1 are
	NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.

	psf can be:
	  "dg" for double-Gaussian
	  "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

	"roiradecsize" = (ra, dec, half-size in pixels) indicates that you
	want to grab a ROI around the given RA,Dec.

	Returns:
	  (tractor.Image, dict)

	dict contains useful details like:
	  'sky'
	  'skysig'
	'''
					  
	# get_tractor_sources() no longer supports !useMags, so
	assert(useMags)

	if sdssobj is None:
		# Ugly
		if release != 'DR7':
			raise RuntimeError('We only support DR7 currently')
		sdss = DR7(curl=curl)
	else:
		sdss = sdssobj

	valid_psf = ['dg', 'kl-gm']
	if psf not in valid_psf:
		raise RuntimeError('PSF must be in ' + str(valid_psf))
	# FIXME
	rerun = 0

	bandnum = band_index(bandname)

	_check_sdss_files(sdss, run, camcol, field, bandname,
					  ['fpC', 'tsField', 'psField', 'fpM'],
					  retrieve=retrieve)
	fpC = sdss.readFpC(run, camcol, field, bandname)
	hdr = fpC.getHeader()
	fpC = fpC.getImage()
	fpC = fpC.astype(np.float32) - sdss.softbias
	image = fpC
	(H,W) = image.shape

	info = dict()
	tai = hdr.get('TAI')
	stripe = hdr.get('STRIPE')
	strip = hdr.get('STRIP')
	obj = hdr.get('OBJECT')
	info.update(tai=tai, stripe=stripe, strip=strip, object=obj)

	tsf = sdss.readTsField(run, camcol, field, rerun)
	astrans = tsf.getAsTrans(bandnum)
	wcs = SdssWcs(astrans)
	#print 'Created SDSS Wcs:', wcs

	if roiradecsize is not None:
		ra,dec,S = roiradecsize
		fxc,fyc = wcs.positionToPixel(RaDecPos(ra,dec))
		print 'RA,Dec (%.3f, %.3f) -> x,y (%.2f, %.2f)' % (ra, dec, fxc, fyc)
		xc,yc = [int(np.round(p)) for p in fxc,fyc]

		roi = [np.clip(xc-S, 0, W),
			   np.clip(xc+S, 0, W),
			   np.clip(yc-S, 0, H),
			   np.clip(yc+S, 0, H)]

		if roi[0]==roi[1] or roi[2]==roi[3]:
			print "ZERO ROI?", roi
			return None,None

		print 'roi', roi

		info.update(roi=roi)
		
	if roi is not None:
		x0,x1,y0,y1 = roi
	else:
		x0 = y0 = 0

	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

	if useMags:
		photocal = SdssMagsPhotoCal(tsf, bandname)
	else:
		photocal = SdssFluxPhotoCal()
	psfield = sdss.readPsField(run, camcol, field)
	sky = psfield.getSky(bandnum)
	skysig = sqrt(sky)
	skyobj = ConstantSky(sky)
	zr = sky + np.array(zrange) * skysig
	info.update(sky=sky, skysig=skysig, zr=zr)

	fpM = sdss.readFpM(run, camcol, field, bandname)
	gain = psfield.getGain(bandnum)
	darkvar = psfield.getDarkVariance(bandnum)
	skyerr = psfield.getSkyErr(bandnum)
	invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

	if roi is not None:
		roislice = (slice(y0,y1), slice(x0,x1))
		image = image[roislice].copy()
		invvar = invvar[roislice].copy()

	if psf == 'kl-gm':
		from emfit import em_fit_2d
		from fitpsf import em_init_params
		
		# Create Gaussian mixture model PSF approximation.
		H,W = image.shape
		klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
		S = klpsf.shape[0]
		# number of Gaussian components
		K = 3
		w,mu,sig = em_init_params(K, None, None, None)
		II = klpsf.copy()
		II /= II.sum()
		# HIDEOUS HACK
		II = np.maximum(II, 0)
		#print 'Multi-Gaussian PSF fit...'
		xm,ym = -(S/2), -(S/2)
		if savepsfimg is not None:
			plt.clf()
			plt.imshow(II, interpolation='nearest', origin='lower')
			plt.title('PSF image to fit with EM')
			plt.savefig(savepsfimg)
		res = em_fit_2d(II, xm, ym, w, mu, sig)
		print 'em_fit_2d result:', res
		if res == 0:
			# print 'w,mu,sig', w,mu,sig
			mypsf = GaussianMixturePSF(w, mu, sig)
		else:
			# Failed!  Return 'dg' model instead?
			print 'PSF model fit', psf, 'failed!  Returning DG model instead'
			psf = 'dg'
	if psf == 'dg':
		dgpsf = psfield.getDoubleGaussian(bandnum)
		print 'Creating double-Gaussian PSF approximation'
		(a,s1, b,s2) = dgpsf
		mypsf = NCircularGaussianPSF([s1, s2], [a, b])

	timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
					   (run, camcol, field, bandname)))
	timg.zr = zr
	return timg,info




def get_tractor_image_dr8(run, camcol, field, bandname, sdss=None,
						  roi=None, psf='kl-gm', roiradecsize=None,
						  savepsfimg=None, curl=False,
						  zrange=[-3,10]):
	'''
	Creates a tractor.Image given an SDSS field identifier.

	If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
	in the image, in zero-indexed pixel coordinates.  x1,y1 are
	NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.

	psf can be:
	  "dg" for double-Gaussian
	  "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

	"roiradecsize" = (ra, dec, half-size in pixels) indicates that you
	want to grab a ROI around the given RA,Dec.

	Returns:
	  (tractor.Image, dict)

	dict contains useful details like:
	  'sky'
	  'skysig'
	'''
	valid_psf = ['dg', 'kl-gm']
	if psf not in valid_psf:
		raise RuntimeError('PSF must be in ' + str(valid_psf))

	if sdss is None:
		sdss = DR8(curl=curl)

	bandnum = band_index(bandname)

	for ft in ['psField', 'fpM']:
		fn = sdss.retrieve(ft, run, camcol, field, bandname)
		#print 'got', fn
	fn = sdss.retrieve('frame', run, camcol, field, bandname)
	#print 'got', fn

	# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
	frame = sdss.readFrame(run, camcol, field, bandname, filename=fn)
	image = frame.getImage().astype(np.float32)
	#print 'Image:', image.dtype
	#print 'mean:', np.mean(image.ravel())
	(H,W) = image.shape
	#print 'shape:', W,H
	
	info = dict()

	astrans = frame.getAsTrans()
	wcs = SdssWcs(astrans)
	print 'Created SDSS Wcs:', wcs
	print '(x,y) = 1,1 -> RA,Dec', wcs.pixelToPosition(1,1)

	if roiradecsize is not None:
		ra,dec,S = roiradecsize
		fxc,fyc = wcs.positionToPixel(RaDecPos(ra,dec))
		print 'ROI center RA,Dec (%.3f, %.3f) -> x,y (%.2f, %.2f)' % (ra, dec, fxc, fyc)
		xc,yc = [int(np.round(p)) for p in fxc,fyc]

		roi = [np.clip(xc-S, 0, W),
			   np.clip(xc+S, 0, W),
			   np.clip(yc-S, 0, H),
			   np.clip(yc+S, 0, H)]

		if roi[0]==roi[1] or roi[2]==roi[3]:
			print "ZERO ROI?", roi
			assert(False)
			return None,None

		print 'roi', roi
		#roi = [max(0, xc-S), min(W, xc+S), max(0, yc-S), min(H, yc+S)]
		info.update(roi=roi)
		
	if roi is not None:
		x0,x1,y0,y1 = roi
	else:
		x0 = y0 = 0
	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

	print 'Band name:', bandname
	photocal = SdssNanomaggiesPhotoCal(bandname)

	sky = 0.
	skyobj = ConstantSky(sky)

	calibvec = frame.getCalibVec()

	print 'sky', #frame.sky
	print frame.sky.shape
	print 'x', len(frame.skyxi), frame.skyxi
	print frame.skyxi.shape
	print 'y', len(frame.skyyi), frame.skyyi
	print frame.skyyi.shape

	skyim = frame.sky
	(sh,sw) = skyim.shape
	print 'Skyim shape', skyim.shape
	if sw != 256:
		skyim = skyim.T
	(sh,sw) = skyim.shape
	xi = np.round(frame.skyxi).astype(int)
	print 'xi:', xi.min(), xi.max(), 'vs [0,', sw, ']'
	yi = np.round(frame.skyyi).astype(int)
	print 'yi:', yi.min(), yi.max(), 'vs [0,', sh, ']'
	assert(all(xi >= 0) and all(xi < sw))
	assert(all(yi >= 0) and all(yi < sh))
	XI,YI = np.meshgrid(xi, yi)
	#print 'XI', XI.shape
	# Nearest-neighbour interpolation -- we just need this for approximate invvar.
	bigsky = skyim[YI,XI]
	#print 'bigsky', bigsky.shape
	assert(bigsky.shape == image.shape)

	# plt.clf()
	# plt.imshow(frame.sky, interpolation='nearest', origin='lower')
	# plt.savefig('sky.png')
	# plt.clf()
	# plt.imshow(bigsky, interpolation='nearest', origin='lower')
	# plt.savefig('bigsky.png')

	#print 'calibvec', calibvec.shape
	dn = (image / calibvec) + bigsky

	# Could get this from photoField instead
	# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html
	psfield = sdss.readPsField(run, camcol, field)
	gain = psfield.getGain(bandnum)
	darkvar = psfield.getDarkVariance(bandnum)
	dnvar = (dn / gain) + darkvar
	invvar = 1./(dnvar * calibvec**2)
	#print 'imgvar:', imgvar.shape
	invvar = invvar.astype(np.float32)
	assert(invvar.shape == image.shape)

	meansky = np.mean(frame.sky)
	meancalib = np.mean(calibvec)
	skysig = sqrt((meansky / gain) + darkvar) * meancalib
	#print 'skysig:', skysig

	info.update(sky=sky, skysig=skysig)
	zr = np.array(zrange)*skysig + sky
	info.update(zr=zr)

	# http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
	fpM = sdss.readFpM(run, camcol, field, bandname)

	if roi is not None:
		roislice = (slice(y0,y1), slice(x0,x1))
		image = image[roislice].copy()
		invvar = invvar[roislice].copy()

	for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
		fpM.setMaskedPixels(plane, invvar, 0, roi=roi)

	if psf == 'kl-gm':
		from emfit import em_fit_2d
		from fitpsf import em_init_params
		
		# Create Gaussian mixture model PSF approximation.
		H,W = image.shape
		klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
		S = klpsf.shape[0]
		# number of Gaussian components
		K = 3
		w,mu,sig = em_init_params(K, None, None, None)
		II = klpsf.copy()
		II /= II.sum()
		# HIDEOUS HACK
		II = np.maximum(II, 0)
		#print 'Multi-Gaussian PSF fit...'
		xm,ym = -(S/2), -(S/2)
		if savepsfimg is not None:
			plt.clf()
			plt.imshow(II, interpolation='nearest', origin='lower')
			plt.title('PSF image to fit with EM')
			plt.savefig(savepsfimg)
		res = em_fit_2d(II, xm, ym, w, mu, sig)
		#print 'em_fit_2d result:', res
		if res == 0:
			# print 'w,mu,sig', w,mu,sig
			mypsf = GaussianMixturePSF(w, mu, sig)
		else:
			# Failed!  Return 'dg' model instead?
			print 'PSF model fit', psf, 'failed!  Returning DG model instead'
			psf = 'dg'
	if psf == 'dg':
		dgpsf = psfield.getDoubleGaussian(bandnum)
		print 'Creating double-Gaussian PSF approximation'
		(a,s1, b,s2) = dgpsf
		mypsf = NCircularGaussianPSF([s1, s2], [a, b])

	timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
					   (run, camcol, field, bandname)))
	timg.zr = zr
	return timg,info


def get_tractor_image_dr9(*args, **kwargs):
	sdss = kwargs.get('sdss', None)
	if sdss is None:
		curl = kwargs.pop('curl', False)
		kwargs['sdss'] = DR9(curl=curl)
	#print 'Calling get_tractor_image_dr8 with sdss=', sdss
	return get_tractor_image_dr8(*args, **kwargs)

def get_tractor_sources_dr9(*args, **kwargs):
	sdss = kwargs.get('sdss', None)
	if sdss is None:
		curl = kwargs.pop('curl', False)
		kwargs['sdss'] = DR9(curl=curl)
	return get_tractor_sources_dr8(*args, **kwargs)


class SdssNanomaggiesPhotoCal(BaseParams):
	def __init__(self, bandname):
		self.bandname = bandname
	def __str__(self):
		return self.__class__.__name__
	def hashkey(self):
		return ('SdssNanomaggiesPhotoCal', self.bandname)
	def brightnessToCounts(self, brightness):
		mag = brightness.getMag(self.bandname)
		if not np.isfinite(mag):
			return 0.
		# MAGIC
		if mag > 50.:
			return 0.
		#print 'mag', mag

		if mag < -50:
			print 'Warning: mag', mag, ': clipping'
			mag = -50

		nmgy = 10. ** ((mag - 22.5) / -2.5)
		#nmgy2 = np.exp(mag * -0.9210340371976184 + 20.723265836946414)
		#print 'nmgy', nmgy
		#print 'nmgy2', nmgy2
		return nmgy

class SdssMagsPhotoCal(BaseParams):
	'''
	A photocal that uses Mags objects.
	'''
	def __init__(self, tsfield, bandname):
		self.bandname = bandname
		self.band = band_index(bandname)

		#self.tsfield = tsfield
		band = self.band
		self.exptime = tsfield.exptime
		self.aa = tsfield.aa[band]
		self.kk = tsfield.kk[band]
		self.airmass = tsfield.airmass[band]

		super(SdssMagsPhotoCal,self).__init__()

	# @staticmethod
	# def getNamedParams():
	# 	return dict(aa=0)
	# # These underscored versions are for use by NamedParams(), and ignore
	# # the active/inactive state.
	# def _setThing(self, i, val):
	# 	assert(i == 0)
	# 	self.aa = val
	# def _getThing(self, i):
	# 	assert(i == 0)
	# 	return self.aa
	# def _getThings(self):
	# 	return [self.aa]
	# def _numberOfThings(self):
	# 	return 1

	# to implement Params
	def getParams(self):
		return [self.aa]
	def getStepSizes(self, *args, **kwargs):
		return [0.01]
	def setParam(self, i, p):
		assert(i == 0)
		self.aa = p

	def getParamNames(self):
		return ['aa']

	def hashkey(self):
		return ('SdssMagsPhotoCal', self.bandname, #self.tsfield)
				self.exptime, self.aa, self.kk, self.airmass)
	
	def brightnessToCounts(self, brightness):
		mag = brightness.getMag(self.bandname)
		if not np.isfinite(mag):
			return 0.
		# MAGIC
		if mag > 50.:
			return 0.
		#return self.tsfield.mag_to_counts(mag, self.band)

		# FROM astrometry.sdss.common.TsField.mag_to_counts:
		logcounts = (-0.4 * mag + np.log10(self.exptime)
					 - 0.4*(self.aa + self.kk * self.airmass))
		rtn = 10.**logcounts
		return rtn

class SdssMagPhotoCal(object):
	'''
	A photocal that uses Mag objects.
	'''
	def __init__(self, tsfield, band):
		'''
		band: int
		'''
		self.tsfield = tsfield
		self.band = band
	def hashkey(self):
		return ('SdssMagPhotoCal', self.band, self.tsfield)
	def brightnessToCounts(self, brightness):
		return self.tsfield.mag_to_counts(brightness.getValue(), self.band)
	#def countsToBrightness(self, counts):
	#	return Mag(self.tsfield.counts_to_mag(counts, self.band))


		

class SdssFluxPhotoCal(object):
	scale = 1e6
	def __init__(self, scale=None):
		if scale is None:
			scale = SdssPhotoCal.scale
		self.scale = scale
	def brightnessToCounts(self, brightness):
		'''
		brightness: SdssFlux object
		returns: float
		'''
		return brightness.getValue() * self.scale

class SdssFlux(Flux):
	def getStepSizes(self, *args, **kwargs):
		return [1.]
	def __str__(self):
		return 'SdssFlux: %.1f' % (self.val * SdssPhotoCal.scale)
	def __repr__(self):
		return 'SdssFlux(%.1f)' % (self.val * SdssPhotoCal.scale)
	def hashkey(self):
		return ('SdssFlux', self.val)
	def copy(self):
		return SdssFlux(self.val)
	def __add__(self, other):
		assert(isinstance(other, SdssFlux))
		return SdssFlux(self.val + other.val)

class SdssWcs(ParamList):
	pnames = ['a', 'b', 'c', 'd', 'e', 'f',
			  'drow0', 'drow1', 'drow2', 'drow3',
			  'dcol0', 'dcol1', 'dcol2', 'dcol3',
			  'csrow', 'cscol', 'ccrow', 'cccol',
			  'x0', 'y0']

	@staticmethod
	def getNamedParams():
		# node and incl are properties of the survey geometry, not params.
		# riCut... not clear.
		# Note that we omit x0,y0 from this list
		return dict([(k,i) for i,k in enumerate(SdssWcs.pnames[:-2])])

	def __init__(self, astrans):
		self.x0 = 0
		self.y0 = 0
		super(SdssWcs, self).__init__(self.x0, self.y0, astrans)
		# ParamList keeps its params in a list; we don't want to do that.
		del self.vals
		self.astrans = astrans

	def _setThing(self, i, val):
		N = len(SdssWcs.pnames)
		if i == N-2:
			self.x0 = val
		elif i == N-1:
			self.y0 = val
		else:
			t = self.astrans.trans
			t[SdssWcs.pnames[i]] = val
	def _getThing(self, i):
		N = len(SdssWcs.pnames)
		if i == N-2:
			return self.x0
		elif i == N-1:
			return self.y0
		t = self.astrans.trans
		return t[SdssWcs.pnames[i]]
	def _getThings(self):
		t = self.astrans.trans
		return [t[nm] for nm in SdssWcs.pnames[:-2]] + [self.x0, self.y0]
	def _numberOfThings(self):
		return len(SdssWcs.pnames)

	def getStepSizes(self, *args, **kwargs):
		deg = 0.396 / 3600. # deg/pix
		P = 2000. # ~ image size
		# a,d: degrees
		# b,c,e,f: deg/pixels
		# drowX: 1/(pix ** (X-1)
		# dcolX: 1/(pix ** (X-1)
		# c*row,col: 1.
		ss = [ deg, deg/P, deg/P, deg, deg/P, deg/P,
			   1., 1./P, 1./P**2, 1./P**3,
			   1., 1./P, 1./P**2, 1./P**3,
			   1., 1., 1., 1.,
			   1., 1.]
		return list(self._getLiquidArray(ss))

	def setX0Y0(self, x0, y0):
		self.x0 = x0
		self.y0 = y0

	# This function is not used by the tractor, and it works in
	# *original* pixel coords (no x0,y0 offsets)
	# (x,y) to RA,Dec in deg
	def pixelToRaDec(self, x, y):
		ra,dec = self.astrans.pixel_to_radec(x, y)
		return ra,dec

	def cdAtPixel(self, x, y):
		return self.astrans.cd_at_pixel(x + self.x0, y + self.y0)

	# RA,Dec in deg to pixel x,y.
	def positionToPixel(self, pos, src=None):
		## FIXME -- color.
		x,y = self.astrans.radec_to_pixel_single(pos.ra, pos.dec)
		return x - self.x0, y - self.y0

	# (x,y) to RA,Dec in deg
	def pixelToPosition(self, x, y, src=None):
		## FIXME -- color.
		ra,dec = self.pixelToRaDec(x + self.x0, y + self.y0)
		return RaDecPos(ra, dec)

class Changes(object):
	pass

class SDSSTractor(Tractor):

	def __init__(self, *args, **kwargs):
		self.debugnew = kwargs.pop('debugnew', False)
		self.debugchange = kwargs.pop('debugchange', False)

		Tractor.__init__(self, *args, **kwargs)
		self.newsource = 0
		self.changes = []
		self.changei = 0

		self.plotfns = []
		self.comments = []
		self.boxes = []

	def debugChangeSources(self, **kwargs):
		if self.debugchange:
			self.doDebugChangeSources(**kwargs)

	def doDebugChangeSources(self, step=None, src=None, newsrcs=None, alti=0,
							 dlnprob=0, **kwargs):
		if step == 'start':
			ch = self.changes = Changes()
			N = self.getNImages()
			ch.src = src
			ch.N = N
			ch.impatch = [None for i in range(N)]
			ch.mod0    = [None for i in range(N)]
			ch.mod0type = src.getSourceType()
			ch.newmods = []

			for imgi in range(N):
				img = self.getImage(imgi)
				mod = self.getModelPatch(img, src)
				ch.mod0[imgi] = mod
				print 'image', imgi, 'got model patch', mod
				if mod.getImage() is not None:
					impatch = img.getImage()[mod.getSlice(img)]
					if len(impatch.ravel()):
						ch.impatch[imgi] = impatch

		elif step in ['init', 'opt1']:
			ch = self.changes
			if newsrcs == []:
				return
			mods = []
			for imgi in range(ch.N):
				img = self.getImage(imgi)
				mod = self.getModelPatch(img, newsrcs[0])
				mods.append(mod)

			if step == 'init':
				ch.newmods.append([newsrcs[0].getSourceType(),
								   mods])
			else:
				ch.newmods[-1].extend([mods, dlnprob])

		elif step in ['switch', 'keep']:
			ch = self.changes
			M = len(ch.newmods)
			N = ch.N
			cols = M+2
			II = [i for i in range(N) if ch.impatch[i] is not None]
			rows = len(II)
			fs = 10

			imargs = {}
			plt.clf()
			# Images
			for ri,i in enumerate(II):
				img = self.getImage(i)
				sky = img.getSky().val
				skysig = sqrt(sky)
				imargs[i] = dict(vmin=-3.*skysig, vmax=10.*skysig)
				if ch.impatch[i] is None:
					continue
				plt.subplot(rows, cols, ri*cols+1)
				plotimage(ch.impatch[i] - sky, **imargs[i])
				plt.xticks([])
				plt.yticks([])
				plt.title('image %i' % i, fontsize=fs)

			# Original sources
			for ri,i in enumerate(II):
				if ch.mod0[i].getImage() is None:
					continue
				plt.subplot(rows, cols, ri*cols+2)
				plotimage(ch.mod0[i].getImage(), **imargs[i])
				plt.xticks([])
				plt.yticks([])
				plt.title('original ' + ch.mod0type, fontsize=fs)

			# New sources
			for j,newmod in enumerate(ch.newmods):
				(srctype, premods, postmods, dlnp) = newmod
				for ri,i in enumerate(II):
					if postmods[i] is None:
						continue
					plt.subplot(rows, cols, ri*cols + 3 + j)

					# HACK -- force patches to be the same size + offset...
					img = self.getImage(i)
					sl = ch.mod0[i].getSlice(img)
					#print 'slice', sl
					im = np.zeros_like(img.getImage())
					postmods[i].addTo(im)
					im = im[sl]
					if len(im.ravel()):
						plotimage(im, **imargs[i])
						plt.xticks([])
						plt.yticks([])
						plt.title(srctype + ' (dlnp=%.1f)' % dlnp, fontsize=fs)
				
			fn = 'change-%03i.png' % self.changei
			plt.savefig(fn)
			print 'Wrote', fn
			self.changei += 1
			self.plotfns.append(fn)
			if step == 'switch':
				s = '<a href="#%s">' % fn + 'accepted change</a> from ' + str(src) + '<br />to '
				if len(newsrcs) == 1:
					s += str(newsrcs[0])
				else:
					s += '[ ' + ' + '.join([str(ns) for ns in newsrcs]) + ' ]'
				self.comments.append(s)
			elif step == 'keep':
				s = '<a href="#%s">' % fn + 'rejected change</a> of ' + str(src)
				self.comments.append(s)
			#smallimg = 'border="0" width="400" height="300"'
			#s += '<a href="%s"><img src="%s" %s /></a>' % (fn, fn, smallimg)
				
				
	def debugNewSource(self, *args, **kwargs):
		if self.debugnew:
			self.doDebugNewSource(*args, **kwargs)

	def doDebugNewSource(self, *args, **kwargs):
		step = kwargs.get('type', None)
		if step in [ 'newsrc-0', 'newsrc-opt' ]:
			if step == 'newsrc-0':
				optstep = 0
				self.newsource += 1
			else:
				optstep = 1 + kwargs['step']
			src = kwargs['src']
			img = kwargs['img']

			patch = src.getModelPatch(img)
			imgpatch = img.getImage()[patch.getSlice(img)]

			plt.clf()
			plt.subplot(2,3,4)
			plotimage(imgpatch)
			cl = plt.gci().get_clim()
			plt.colorbar()
			plt.title('image patch')
			plt.subplot(2,3,5)
			plotimage(patch.getImage(), vmin=cl[0], vmax=cl[1])
			plt.colorbar()
			plt.title('new source')
			derivs = src.getParamDerivatives(img)
			assert(len(derivs) == 3)
			for i,deriv in enumerate(derivs):
				plt.subplot(2,3,i+1)
				plotimage(deriv.getImage())
				cl = plt.gci().get_clim()
				mx = max(abs(cl[0]), abs(cl[1]))
				plt.gci().set_clim(-mx, mx)
				plt.colorbar()
				plt.title(deriv.name)
			fn = 'newsource-%02i-%02i.png' % (self.newsource, optstep)
			plt.savefig(fn)
			print 'Wrote', fn

	def createNewSource(self, img, x, y, ht):
		wcs = img.getWcs()
		pos = wcs.pixelToPosition(None, (x,y))
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		patch = img.getPsf().getPointSourcePatch(x, y)
		ht /= patch.getImage().max()
		photocal = img.getPhotoCal()
		# XXX
		flux = photocal.countsToBrightness(ht)
		ps = PointSource(pos, flux)
		try:
			imgi = self.images.index(img)
			patch = self.getModelPatch(img, ps)
			self.addBox(imgi, patch.getExtent())
		except:
			pass
		return ps

	def addBox(self, imgi, box):
		if len(self.boxes) == 0:
			self.boxes = [[] for i in range(self.getNImages())]
		self.boxes[imgi].append(box)

	def changeSourceTypes(self, srcs=None, **kwargs):
		if srcs is not None:
			for i,img in enumerate(self.getImages()):
				for src in srcs:
					patch = self.getModelPatch(img, src)
					self.addBox(i, patch.getExtent())
		Tractor.changeSourceTypes(self, srcs, **kwargs)


	def changeSource(self, source):
		'''
		Proposes a list of alternatives, where each alternative is a list of new
		Sources that the given Source could be changed into.
		'''
		if isinstance(source, PointSource):
			eg = ExpGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   1., 0.5, 0.)
			dg = DevGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   1., 0.5, 0.)
			#print 'Changing:'
			#print '  from ', source
			#print '  into', eg
			return [ [], [eg], [dg] ]

		elif isinstance(source, ExpGalaxy):
			dg = DevGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getBrightness().copy())
			return [ [], [ps], [dg] ]

		elif isinstance(source, DevGalaxy):
			eg = ExpGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getBrightness().copy())
			return [ [], [ps], [eg] ]

		else:
			print 'unknown source type for', source
			return []

