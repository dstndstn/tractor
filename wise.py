if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
import tempfile
import tractor
import pyfits
import numpy as np
from tractor import *
from tractor.sdss_galaxy import *

from matplotlib.nxutils import points_inside_poly

def read_wise_level1b(basefn, radecroi=None, filtermap={},
					  nanomaggies=False):
	intfn  = basefn + '-int-1b.fits'
	maskfn = basefn + '-msk-1b.fits'
	uncfn  = basefn + '-unc-1b.fits'

	print 'intensity image', intfn
	print 'mask image', maskfn
	print 'uncertainty image', uncfn

	P = pyfits.open(intfn)
	ihdr = P[0].header
	data = P[0].data
	print 'Read', data.shape, 'intensity'
	band = ihdr['BAND']

	P = pyfits.open(uncfn)
	uhdr = P[0].header
	unc = P[0].data
	print 'Read', unc.shape, 'uncertainty'

	P = pyfits.open(maskfn)
	mhdr = P[0].header
	mask = P[0].data
	print 'Read', mask.shape, 'mask'

	#twcs = tractor.FitsWcs(intfn)
	twcs = tractor.WcslibWcs(intfn)
	print 'WCS', twcs

	# HACK -- circular Gaussian PSF of fixed size...
	# in arcsec 
	fwhms = { 1: 6.1, 2: 6.4, 3: 6.5, 4: 12.0 }
	# -> sigma in pixels
	sig = fwhms[band] / 2.35 / twcs.pixel_scale()
	print 'PSF sigma', sig, 'pixels'
	tpsf = tractor.NCircularGaussianPSF([sig], [1.])

	if radecroi is not None:
		ralo,rahi, declo,dechi = radecroi
		xy = [twcs.positionToPixel(tractor.RaDecPos(r,d))
			  for r,d in [(ralo,declo), (ralo,dechi), (rahi,declo), (rahi,dechi)]]
		xy = np.array(xy)
		x0,x1 = xy[:,0].min(), xy[:,0].max()
		y0,y1 = xy[:,1].min(), xy[:,1].max()
		print 'RA,Dec ROI', ralo,rahi, declo,dechi, 'becomes x,y ROI', x0,x1,y0,y1

		# Clip to image size...
		H,W = data.shape
		x0 = max(0, min(x0, W-1))
		x1 = max(0, min(x1, W))
		y0 = max(0, min(y0, H-1))
		y1 = max(0, min(y1, H))
		print ' clipped to', x0,x1,y0,y1

		data = data[y0:y1, x0:x1]
		unc = unc[y0:y1, x0:x1]
		mask = mask[y0:y1, x0:x1]
		twcs.setX0Y0(x0,y0)
		print 'Cut data to', data.shape

	else:
		H,W = data.shape
		x0,x1,y0,y1 = 0,W, 0,H

	filter = 'w%i' % band
	if filtermap:
		filter = filtermap.get(filter, filter)
	zp = ihdr['MAGZP']
	if nanomaggies:
		photocal = tractor.LinearPhotoCal(tractor.NanoMaggies.zeropointToScale(zp),
										  band=filter)
	else:
		photocal = tractor.MagsPhotoCal(filter, zp)

	print 'Image median:', np.median(data)
	print 'unc median:', np.median(unc)

	sky = np.median(data)
	tsky = tractor.ConstantSky(sky)

	sigma1 = np.median(unc)
	zr = np.array([-3,10]) * sigma1 + sky

	name = 'WISE ' + ihdr['FRSETID'] + ' W%i' % band

	# Mask bits, from
	# http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4a.html#maskdef
	# 0 from static mask: excessively noisy due to high dark current alone
	# 1 from static mask: generally noisy [includes bit 0]
	# 2 from static mask: dead or very low responsivity
	# 3 from static mask: low responsivity or low dark current
	# 4 from static mask: high responsivity or high dark current
	# 5 from static mask: saturated anywhere in ramp
	# 6 from static mask: high, uncertain, or unreliable non-linearity
	# 7 from static mask: known broken hardware pixel or excessively noisy responsivity estimate [may include bit 1]
	# 8 reserved
	# 9 broken pixel or negative slope fit value (downlink value = 32767)
	# 10 saturated in sample read 1 (down-link value = 32753)
	# 11 saturated in sample read 2 (down-link value = 32754)
	# 12 saturated in sample read 3 (down-link value = 32755)
	# 13 saturated in sample read 4 (down-link value = 32756)
	# 14 saturated in sample read 5 (down-link value = 32757)
	# 15 saturated in sample read 6 (down-link value = 32758)
	# 16 saturated in sample read 7 (down-link value = 32759)
	# 17 saturated in sample read 8 (down-link value = 32760)
	# 18 saturated in sample read 9 (down-link value = 32761)
	# 19 reserved
	# 20 reserved
	# 21 new/transient bad pixel from dynamic masking
	# 22 reserved
	# 23 reserved
	# 24 reserved
	# 25 reserved
	# 26 non-linearity correction unreliable
	# 27 contains cosmic-ray or outlier that cannot be classified (from temporal outlier rejection in multi-frame pipeline)
	# 28 contains positive or negative spike-outlier
	# 29 reserved
	# 30 reserved
	# 31 not used: sign bit

	#goodmask = (mask == 0)

	goodmask = ((mask & sum([1<<bit for bit in [0,1,2,3,4,5,6,7, 9,
												10,11,12,13,14,15,16,17,18,
												21,26,27,28]])) == 0)

	invvar = np.zeros_like(data)
	invvar[goodmask] = 1./(unc[goodmask])**2
	#invvar = 1./(unc)**2

	# avoid NaNs
	data[np.logical_not(goodmask)] = sky

	tim = tractor.Image(data=data, invvar=invvar, psf=tpsf, wcs=twcs,
						sky=tsky, photocal=photocal, name=name, zr=zr)
	tim.extent = [x0,x1,y0,y1]

	# FIXME
	tim.maskplane = mask
	tim.uncplane = unc
	tim.goodmask = goodmask

	return tim


def read_wise_level3(basefn, radecroi=None, filtermap={},
					 nanomaggies=False):
	intfn = basefn + '-int-3.fits'
	uncfn = basefn + '-unc-3.fits'

	print 'intensity image', intfn
	print 'uncertainty image', uncfn

	P = pyfits.open(intfn)
	ihdr = P[0].header
	data = P[0].data
	print 'Read', data.shape, 'intensity'
	band = ihdr['BAND']

	P = pyfits.open(uncfn)
	uhdr = P[0].header
	unc = P[0].data
	print 'Read', unc.shape, 'uncertainty'

	''' cov:
	BAND	=					 1 / wavelength band number
	WAVELEN =				 3.368 / [microns] effective wavelength of band
	COADDID = '3342p000_ab41'	   / atlas-image identifier
	MAGZP	=				  20.5 / [mag] relative photometric zero point
	MEDINT	=	   4.0289044380188 / [DN] median of intensity pixels
	'''
	''' int:
	BUNIT	= 'DN	   '		   / image pixel units
	CTYPE1	= 'RA---SIN'		   / Projection type for axis 1
	CTYPE2	= 'DEC--SIN'		   / Projection type for axis 2
	CRPIX1	=		   2048.000000 / Axis 1 reference pixel at CRVAL1,CRVAL2
	CRPIX2	=		   2048.000000 / Axis 2 reference pixel at CRVAL1,CRVAL2
	CDELT1	=  -0.0003819444391411 / Axis 1 scale at CRPIX1,CRPIX2 (deg/pix)
	CDELT2	=	0.0003819444391411 / Axis 2 scale at CRPIX1,CRPIX2 (deg/pix)
	CROTA2	=			  0.000000 / Image twist: +axis2 W of N, J2000.0 (deg)
	'''
	''' unc:
	FILETYPE= '1-sigma uncertainty image' / product description
	'''

	twcs = tractor.WcslibWcs(intfn)
	print 'WCS', twcs
	#twcs.debug()
	print 'pixel scale', twcs.pixel_scale()

	# HACK -- circular Gaussian PSF of fixed size...
	# in arcsec 
	fwhms = { 1: 6.1, 2: 6.4, 3: 6.5, 4: 12.0 }
	# -> sigma in pixels
	sig = fwhms[band] / 2.35 / twcs.pixel_scale()
	print 'PSF sigma', sig, 'pixels'
	tpsf = tractor.NCircularGaussianPSF([sig], [1.])

	if radecroi is not None:
		ralo,rahi, declo,dechi = radecroi
		xy = [twcs.positionToPixel(tractor.RaDecPos(r,d))
			  for r,d in [(ralo,declo), (ralo,dechi), (rahi,declo), (rahi,dechi)]]
		xy = np.array(xy)
		x0,x1 = xy[:,0].min(), xy[:,0].max()
		y0,y1 = xy[:,1].min(), xy[:,1].max()
		print 'RA,Dec ROI', ralo,rahi, declo,dechi, 'becomes x,y ROI', x0,x1,y0,y1

		# Clip to image size...
		H,W = data.shape
		x0 = max(0, min(x0, W-1))
		x1 = max(0, min(x1, W))
		y0 = max(0, min(y0, H-1))
		y1 = max(0, min(y1, H))
		print ' clipped to', x0,x1,y0,y1

		data = data[y0:y1, x0:x1]
		unc = unc[y0:y1, x0:x1]
		twcs.setX0Y0(x0,y0)
		print 'Cut data to', data.shape

	else:
		H,W = data.shape
		x0,x1,y0,y1 = 0,W, 0,H

	filter = 'w%i' % band
	if filtermap:
		filter = filtermap.get(filter, filter)
	zp = ihdr['MAGZP']

	if nanomaggies:
		photocal = tractor.LinearPhotoCal(tractor.NanoMaggies.zeropointToScale(zp),
										  band=filter)
	else:
		photocal = tractor.MagsPhotoCal(filter, zp)

	print 'Image median:', np.median(data)
	print 'unc median:', np.median(unc)

	sky = np.median(data)
	tsky = tractor.ConstantSky(sky)

	sigma1 = np.median(unc)
	zr = np.array([-3,10]) * sigma1 + sky

	name = 'WISE ' + ihdr['COADDID'] + ' W%i' % band

	tim = tractor.Image(data=data, invvar=1./(unc**2), psf=tpsf, wcs=twcs,
						sky=tsky, photocal=photocal, name=name, zr=zr)
	tim.extent = [x0,x1,y0,y1]
	return tim

read_wise_coadd = read_wise_level3
read_wise_image = read_wise_level1b



def main():
	# from tractor.sdss_galaxy import *
	# import sys
	# 
	# ell = GalaxyShape(10., 0.5, 45)
	# print ell
	# S = 20./3600.
	# dra,ddec = np.meshgrid(np.linspace(-S, S, 200),
	#						 np.linspace(-S, S, 200))
	# overlap = []
	# for r,d in zip(dra.ravel(),ddec.ravel()):
	#	  overlap.append(ell.overlapsCircle(r,d, 0.))
	# overlap = np.array(overlap).reshape(dra.shape)
	# print 'overlap', overlap.min(), overlap.max()
	# 
	# import matplotlib
	# matplotlib.use('Agg')
	# import pylab as plt
	# plt.clf()
	# plt.imshow(overlap)
	# plt.savefig('overlap.png')
	# 
	# sys.exit(0)

	from bigboss_test import *

	#filtermap = { 'w1':'i', 'w2':'i', 'w3':'i', 'w4':'i' }
	filtermap = None
	
	import matplotlib
	matplotlib.use('Agg')

	import logging
	import sys
	#lvl = logging.INFO
	lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	from astrometry.util.util import *
	from astrometry.util.pyfits_utils import fits_table
	from astrometry.libkd.spherematch import *
	import numpy as np

	#bandnums = [1,2,3,4]
	bandnums = [1,]
	bands = ['w%i' % n for n in bandnums]

	(ra0,ra1, dec0,dec1) = radecroi

	# cfht = fits_table('/project/projectdirs/bigboss/data/cs82/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	# print 'Read', len(cfht), 'sources'
	# # Cut to ROI
	# cfht.ra  = cfht.alpha_j2000
	# cfht.dec = cfht.delta_j2000
	# cfht.cut((cfht.ra > ra0) * (cfht.ra < ra1) * (cfht.dec > dec0) * (cfht.dec < dec1))
	# print 'Cut to', len(cfht), 'objects in ROI.'
	# cfht.cut((cfht.mag_psf < 25.))
	# print 'Cut to', len(cfht), 'bright'
	#srcs = get_cfht_catalog(mags=bands, maglim=25.)

	srcs,T = get_cfht_catalog(mags=['i'] + bands, maglim=25., returnTable=True)
	print 'Got', len(srcs), 'CFHT sources'


	wise = fits_table('/project/projectdirs/bigboss/data/wise/catalogs/wisecat.fits')
	print 'Read', len(wise), 'WISE sources'
	#(ra0,ra1, dec0,dec1) = radecroi
	wise.cut((wise.ra > ra0) * (wise.ra < ra1) * (wise.dec > dec0) * (wise.dec < dec1))
	print 'Cut to', len(wise), 'objects in ROI.'

	I,J,D = match_radec(T.ra, T.dec, wise.ra, wise.dec, 1./3600.)
	print len(I), 'matches'
	print len(np.unique(I)), 'unique CFHT sources in matches'
	print len(np.unique(J)), 'unique WISE sources in matches'

	for j in np.unique(J):
		K = np.flatnonzero(J == j)
		# UGH, just assign to the nearest
		i = np.argmin(D[K])
		K = K[i]
		i = I[K]

		for band in bands:
			if isinstance(srcs[i], CompositeGalaxy):
				mag = wise.get(band+'mag')[j]
				half = mag + 0.75
				setattr(srcs[i].brightnessExp, band, half)
				setattr(srcs[i].brightnessDev, band, half)
			else:
				setattr(srcs[i].brightness, band, wise.get(band+'mag')[j])
		print 'Plugged in WISE mags for source:', srcs[i]
	
	#### Cut to just sources that had a match to the WISE catalog
	# ### uhh, why don't we just use the WISE catalog then?
	# JJ,KK = np.unique(J, return_index=True)
	# keep = Catalog()
	# for i in I[KK]:
	# keep.append(srcs[i])
	# srcs = keep
	# print 'Kept:'
	# for src in srcs:
	# print src
	

	ims = []


	basedir = '/project/projectdirs/bigboss/data/wise/level3'
	pat = '3342p000_ab41-w%i'
	for band in bandnums:
		base = pat % band
		basefn = os.path.join(basedir, base)
		im = read_wise_coadd(basefn, radecroi=radecroi, filtermap=filtermap)
		tr = tractor.Tractor(tractor.Images(im), srcs)
		make_plots('wise-%i-' % band, im, tr=tr)
		ims.append(im)


	basedir = '/project/projectdirs/bigboss/data/wise/level1b'
	pat = '04933b137-w%i'
	for band in bandnums:
		base = pat % band
		basefn = os.path.join(basedir, base)
		im = read_wise_image(basefn, radecroi=radecroi, filtermap=filtermap)
		tr = tractor.Tractor(tractor.Images(im), srcs)
		make_plots('wise-%i-' % band, im, tr=tr)
		ims.append(im)
		im.freezeAllBut('psf', 'sky')
		tr.freezeParam('catalog')

		j = 1
		while True:
			dlnp,X,alpha = tr.optimize(damp=1.)
			make_plots('wise-%i-psfsky-%i-' % (bandnum,j), im, tr=tr, plots=['model','chi'])
			j += 1

	sys.exit(0)

	tr = tractor.Tractor(tractor.Images(*ims), srcs)

	# fix all calibration
	tr.freezeParam('images')

	# freeze source positions, shapes
	tr.freezeParamsRecursive('pos', 'shape', 'shapeExp', 'shapeDev')

	# freeze all sources
	tr.catalog.freezeAllParams()
	# also freeze all bands
	tr.catalog.freezeParamsRecursive(*bands)

	for im,bandnum in zip(ims,bandnums):
		tr.setImages(tractor.Images(im))
		band = im.photocal.band
		print 'band', band
		# thaw this band
		tr.catalog.thawParamsRecursive(band)

		# sweep across the image, optimizing in circles.
		# we'll use the healpix grid for circle centers.
		# how big? in arcmin
		R = 1.
		Rpix = R / 60. / np.sqrt(np.abs(np.linalg.det(im.wcs.cdAtPixel(0,0))))
		nside = int(healpix_nside_for_side_length_arcmin(R/2.))
		print 'Nside', nside
		print 'radius in pixels:', Rpix

		# start in one corner.
		pos = im.wcs.pixelToPosition(0, 0)
		hp = radecdegtohealpix(pos.ra, pos.dec, nside)

		hpqueue = [hp]
		hpdone = []

		j = 1

		while len(hpqueue):
			hp = hpqueue.pop()
			hpdone.append(hp)
			print 'looking at healpix', hp
			ra,dec = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
			print 'RA,Dec center', ra,dec
			x,y = im.wcs.positionToPixel(tractor.RaDecPos(ra,dec))
			H,W	 = im.shape
			if x < -Rpix or y < -Rpix or x >= W+Rpix or y >= H+Rpix:
				print 'pixel', x,y, 'out of bounds'
				continue

			# add neighbours
			nn = healpix_get_neighbours(hp, nside)
			print 'healpix neighbours', nn
			for ni in nn:
				if ni in hpdone:
					continue
				if ni in hpqueue:
					continue
				hpqueue.append(ni)
				print 'enqueued neighbour', ni

			# FIXME -- add PSF-sized margin to radius
			#ra,dec = (radecroi[0]+radecroi[1])/2., (radecroi[2]+radecroi[3])/2.

			tr.catalog.thawSourcesInCircle(tractor.RaDecPos(ra, dec), R/60.)

			for step in range(10):
				print 'Optimizing:'
				for nm in tr.getParamNames():
					print nm
				(dlnp,X,alpha) = tr.optimize(damp=1.)
				print 'dlnp', dlnp
				print 'alpha', alpha
			
				if True:
					print 'plotting', j
					make_plots('wise-%i-step%03i-' % (bandnum,j), im, tr=tr, plots=['model','chi'])
					j += 1

					###################### profiling
					#if j == 10:
					#	 sys.exit(0)
					######################

				if dlnp < 1:
					break

			tr.catalog.freezeAllParams()

		# re-freeze this band
		tr.catalog.freezeParamsRecursive(band)



	# Optimize sources one at a time, and one image at a time.
	#sortmag = 'w1'
	#I = np.argsort([getattr(src.getBrightness(), sortmag) for src in srcs])
	# for j,i in enumerate(I):
	#	  srci = int(i)
	#	  # 
	#	  tr.catalog.thawParam(srci)
	#	  while True:
	#		  print 'optimizing source', j+1, 'of', len(I)
	#		  print 'Optimizing:'
	#		  for nm in tr.getParamNames():
	#			  print nm
	#		  (dlnp,X,alpha) = tr.optimize()
	#		  if dlnp < 1:
	#			  break
	# 
	#	  tr.catalog.freezeParam(srci)
	# 
	#	  #if ((j+1) % 10 == 0):
	#	  if True:
	#		  print 'plotting', j
	#		  make_plots('wise-%i-step%03i-' % (bandnum,j), im, tr=tr, plots=['model','chi'])



	#print 'Optimizing:'
	#for n in tr.getParamNames():
	#	 print n
	#tr.optimize()

	for band,im in zip([1,2,3,4], ims):
		make_plots('wise-%i-opt1-' % band, im, tr=tr, plots=['model','chi'])



def forcedphot():
	T1 = fits_table('cs82data/cas-primary-DR8.fits')
	print len(T1), 'primary'
	T1.cut(T1.nchild == 0)
	print len(T1), 'children'

	rl,rh = T1.ra.min(), T1.ra.max()
	dl,dh = T1.dec.min(), T1.dec.max()

	tims = []

	# Coadd
	basedir = os.path.join('cs82data', 'wise', 'level3')
	basefn = os.path.join(basedir, '3342p000_ab41-w1')
	tim = read_wise_coadd(basefn, radecroi=[rl,rh,dl,dh], nanomaggies=True)
	tims.append(tim)

	# Individuals
	basedir = os.path.join('cs82data', 'wise', 'level1b')
	for fn in [ '04933b137-w1', '04937b137-w1', '04941b137-w1', '04945b137-w1', '04948a112-w1',
				'04949b137-w1', '04952a112-w1', '04953b137-w1', '04956a112-w1', '04960a112-w1',
				'04964a112-w1', '04968a112-w1', '05204a106-w1' ]:
		basefn = os.path.join(basedir, '04933b137-w1')
		tim = read_wise_image(basefn, radecroi=[rl,rh,dl,dh], nanomaggies=True)
		tims.append(tim)

	# tractor.Image's setMask() does a binary dilation on bad pixels!
	for tim in tims:
		#	tim.mask = np.zeros(tim.shape, dtype=bool)
		tim.invvar = tim.origInvvar
		tim.mask = np.zeros(tim.shape, dtype=bool)
	print 'tim:', tim

	ps = PlotSequence('forced')

	plt.clf()
	plt.plot(T1.ra, T1.dec, 'r.')
	for tim in tims:
		wcs = tim.getWcs()
		H,W = tim.shape
		rr,dd = [],[]
		for x,y in zip([1,1,W,W,1], [1,H,H,1,1]):
			rd = wcs.pixelToPosition(x,y)
			rr.append(rd.ra)
			dd.append(rd.dec)
		plt.plot(rr, dd, 'k-', alpha=0.5)
	#setRadecAxes(rl,rh,dl,dh)
	ps.savefig()

	T2 = fits_table('wise-cut.fits')
	T2.w1 = T2.w1mpro
	R = 1./3600.
	I,J,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
	print len(I), 'matches'

	refband = 'r'
	#bandnum = band_index('r')

	Lstar = (T1.probpsf == 1) * 1.0
	Lgal  = (T1.probpsf == 0)
	fracdev = T1.get('fracdev_%s' % refband)
	Ldev = Lgal * fracdev
	Lexp = Lgal * (1. - fracdev)

	ndev, nexp, ncomp,nstar = 0, 0, 0, 0
	cat = Catalog()
	#for i,t in enumerate(T1):

	jmatch = np.zeros(len(T1))
	jmatch[:] = -1
	jmatch[I] = J

	for i in range(len(T1)):
		j = jmatch[i]
		if j >= 0:
			# match source: grab WISE catalog mag
			w1 = T2.w1[j]
		else:
			# unmatched: set it faint
			w1 = 18.

		bright = NanoMaggies(w1=NanoMaggies.magToNanomaggies(w1))

		pos = RaDecPos(T1.ra[i], T1.dec[i])
		if Lstar[i] > 0:
			# Star
			star = PointSource(pos, bright)
			cat.append(star)
			nstar += 1
			continue

		hasdev = (Ldev[i] > 0)
		hasexp = (Lexp[i] > 0)
		iscomp = (hasdev and hasexp)
		if iscomp:
			dbright = bright * Ldev[i]
			ebright = bright * Lexp[i]
		elif hasdev:
			dbright = bright
		elif hasexp:
			ebright = bright
		else:
			assert(False)
											 
		if hasdev:
			re  = T1.get('devrad_%s' % refband)[i]
			ab  = T1.get('devab_%s'	 % refband)[i]
			phi = T1.get('devphi_%s' % refband)[i]
			dshape = GalaxyShape(re, ab, phi)
		if hasexp:
			re  = T1.get('exprad_%s' % refband)[i]
			ab  = T1.get('expab_%s'	 % refband)[i]
			phi = T1.get('expphi_%s' % refband)[i]
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

		cat.append(gal)
	print 'Created', ndev, 'pure deV', nexp, 'pure exp and',
	print ncomp, 'composite galaxies',
	print 'and', nstar, 'stars'

	tractor = Tractor(tims, cat)

	for i,tim in enumerate(tims):
		ima = dict(interpolation='nearest', origin='lower',
				   vmin=tim.zr[0], vmax=tim.zr[1])

		mod = tractor.getModelImage(i)

		plt.clf()
		plt.imshow(mod, **ima)
		plt.gray()
		plt.title('model: %s' % tim.name)
		ps.savefig()

		plt.clf()
		plt.imshow(tim.getImage(), **ima)
		plt.gray()
		plt.title('data: %s' % tim.name)
		ps.savefig()

		plt.clf()
		plt.imshow(tim.getInvvar(), interpolation='nearest', origin='lower')
		plt.gray()
		plt.title('invvar')
		ps.savefig()

		#for tim in tims:
		wcs = tim.getWcs()
		H,W = tim.shape
		poly = []
		for r,d in zip([rl,rl,rh,rh,rl], [dl,dh,dh,dl,dl]):
			x,y = wcs.positionToPixel(RaDecPos(r,d))
			poly.append((x,y))
		xx,yy = np.meshgrid(np.arange(W), np.arange(H))
		xy = np.vstack((xx.flat, yy.flat)).T
		grid = points_inside_poly(xy, poly)
		grid = grid.reshape((H,W))

		tim.setInvvar(tim.getInvvar() * grid)

		# plt.clf()
		# plt.imshow(grid, interpolation='nearest', origin='lower')
		# plt.gray()
		# ps.savefig()

		plt.clf()
		plt.imshow(tim.getInvvar(), interpolation='nearest', origin='lower')
		plt.gray()
		plt.title('invvar')
		ps.savefig()

		if i == 1:
			plt.clf()
			plt.imshow(tim.goodmask, interpolation='nearest', origin='lower')
			plt.gray()
			plt.title('goodmask')
			ps.savefig()

			miv = (1./(tim.uncplane)**2)
			for bit in range(-1,32):
				if bit >= 0:
					miv[(tim.maskplane & (1 << bit)) != 0] = 0.
				if bit == 31:
					plt.clf()
					plt.imshow(miv, interpolation='nearest', origin='lower')
					plt.gray()
					plt.title('invvar with mask bits up to %i blanked out' % bit)
					ps.savefig()

			# for bit in range(32):
			#	plt.clf()
			#	plt.imshow(tim.maskplane & (1 << bit),
			#			   interpolation='nearest', origin='lower')
			#	plt.gray()
			#	plt.title('mask bit %i' % bit)
			#	ps.savefig()


def wisemap():
	from bigboss_test import radecroi
	from astrometry.util.util import Sip, anwcs, fits_use_error_system
	from astrometry.blind.plotstuff import *
	from astrometry.libkd.spherematch import match_radec

	fits_use_error_system()

	basedir = '/project/projectdirs/bigboss'
	wisedatadir = os.path.join(basedir, 'data', 'wise')

	(ra0,ra1, dec0,dec1) = radecroi
	ra = (ra0 + ra1) / 2.
	dec = (dec0 + dec1) / 2.
	width = 2.

	rfn = 'wise-roi.fits'
	if not os.path.exists(rfn):
		TT = []
		for part in range(1, 7):
			fn = 'index-allsky-astr-L1b-part%i.fits' % part
			catfn = os.path.join(wisedatadir, fn)
			print 'Reading', catfn
			T = fits_table(catfn)
			print 'Read', len(T)
			I,J,d = match_radec(ra, dec, T.ra, T.dec, width)
			print 'Found', len(I), 'RA,Dec matches'
			if len(I) == 0:
				del T
				continue
			T.cut(J)
			newhdr = []
			for i in range(len(T)):
				hdr = T.header[i]
				hdr = [str(s) for s in hdr]
				hdr = (['SIMPLE  =                    T',
					'BITPIX  =                    8',
					'NAXIS   =                    0',
					] + hdr +
				       ['END'])
				hdr = [x + (' ' * (80-len(x))) for x in hdr]
				hdrstr = ''.join(hdr)
				newhdr.append(hdrstr)
			T.delete_column('header')
			T.headerstr = np.array(newhdr)
			TT.append(T)

		T = merge_tables(TT)
		T.about()

		sid = np.array([np.sum([float(1 << (8*(6-i))) * ord(s[i]) for i in range(6)])
				for s in T.scan_id])
		I = np.lexsort((T.frame_num, sid))
		T.cut(I)
		T.writeto(rfn)

	T = fits_table(rfn)

	print 'Scan/Frame:'
	for s,f in zip(T.scan_id, T.frame_num):
		print '  ', s, f

	plot = Plotstuff(outformat='png', ra=ra, dec=dec, width=width, size=(800,800))
	out = plot.outline
	plot.color = 'white'
	plot.alpha = 0.1
	plot.apply_settings()

	for i in range(len(T)):
		hdrstr = T.headerstr[i]
		hdrstr = hdrstr + (' ' * (80 - (len(hdrstr)%80)))
		#print 'hdrstr:', type(hdrstr), len(hdrstr)
		#print 'XXX%sXXX' % hdrstr
		wcs = anwcs(hdrstr, -1, len(hdrstr))
		out.wcs = wcs
		out.fill = False
		plot.plot('outline')
		out.fill = True
		plot.plot('outline')

	plot.color = 'gray'
	plot.alpha = 1.0
	plot.plot_grid(1, 1, 1, 1)
	plot.write('wisemap.png')

	I,J,d = match_radec(ra, dec, T.ra, T.dec, width)
	print 'Found', len(I), 'RA,Dec matches'
	i = np.argmin(d)
	i = J[i]
	print 'ra,dec', ra,dec, 'closest', T.ra[i], T.dec[i]
	hdrstr = T.headerstr[i]
	hdrstr = hdrstr + (' ' * (80 - (len(hdrstr)%80)))
	wcs = anwcs(hdrstr, -1, len(hdrstr))
	plot.color = 'blue'
	plot.alpha = 0.2
	plot.apply_settings()
	out.wcs = wcs
	out.fill = False
	plot.plot('outline')
	out.fill = True
	plot.plot('outline')
	plot.write('wisemap2.png')
	
	

if __name__ == '__main__':
	from astrometry.util.fits import *
	from astrometry.util.plotutils import *
	from astrometry.libkd.spherematch import *
	import pylab as plt
	import sys

	wisemap()
	sys.exit(0)
	
	forcedphot()
	sys.exit(0)
	
	T1 = fits_table('cs82data/cas-primary-DR8.fits')
	print len(T1), 'SDSS'
	print '	 RA', T1.ra.min(), T1.ra.max()

	cutfn = 'wise-cut.fits'
	if not os.path.exists(cutfn):
		T2 = fits_table('wise-27-tag.fits')
		print len(T2), 'WISE'
		print '	 RA', T2.ra.min(), T2.ra.max()
		T2.cut((T2.ra  > T1.ra.min())  * (T2.ra < T1.ra.max()) *
			   (T2.dec > T1.dec.min()) * (T2.dec < T1.dec.max()))
		print 'Cut WISE to same RA,Dec region:', len(T2)
		T2.writeto('wise-cut.fits')
	else:
		T2 = fits_table(cutfn)
		print len(T2), 'WISE (cut)'

	R = 1./3600.
	I,J,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
	print len(I), 'matches'

	plt.clf()
	#loghist(T1.r[I] - T1.i[I], T1.r[I] - T2.w1mpro[J], 200, range=((0,2),(-2,5)))
	#plt.plot(T1.r[I] - T1.i[I], T1.r[I] - T2.w1mpro[J], 'r.')
	for cc,lo,hi in [('r', 8, 14), ('y', 14, 15), ('g', 15, 16), ('b', 16, 17), ('m', 17,20)]:
		w1 = T2.w1mpro[J]
		K = (w1 >= lo) * (w1 < hi)
		plt.plot(T1.r[I[K]] - T1.i[I[K]], T1.r[I[K]] - T2.w1mpro[J[K]], '.', color=cc)
		#plt.plot(T1.r[I] - T1.i[I], T1.r[I] - T2.w1mpro[J], 'r.')
	plt.xlabel('r - i')
	plt.ylabel('r - W1 3.4 micron')
	plt.axis([0,2,0,8])
	plt.savefig('wise1.png')

	plt.clf()
	plt.plot(T1.ra, T1.dec, 'bx')
	plt.plot(T2.ra, T2.dec, 'o', mec='r', mfc='none')
	plt.plot(T2.ra[J], T2.dec[J], '^', mec='g', mfc='none')
	plt.savefig('wise2.png')

	R = 2./3600.
	I2,J2,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
	print len(I), 'matches'
	plt.plot(T2.ra[J2], T2.dec[J2], '^', mec='g', mfc='none', ms=10)
	plt.savefig('wise3.png')

	plt.clf()
	plt.plot(3600.*(T1.ra [I2] - T2.ra [J2]),
			 3600.*(T1.dec[I2] - T2.dec[J2]), 'r.')
	plt.xlabel('dRA (arcsec)')
	plt.ylabel('dDec (arcsec)')
	plt.savefig('wise4.png')

	plt.clf()
	plt.subplot(2,1,1)
	plt.hist(T1.r, 100, range=(10,25), histtype='step', color='k')
	plt.hist(T1.r[I], 100, range=(10,25), histtype='step', color='r')
	plt.hist(T1.r[I2], 100, range=(10,25), histtype='step', color='m')
	plt.xlabel('r band')
	plt.axhline(0, color='k', alpha=0.5)
	plt.ylim(-5, 90)
	plt.xlim(10,25)

	plt.subplot(2,1,2)
	plt.hist(T2.w1mpro, 100, range=(10,25), histtype='step', color='k')
	plt.hist(T2.w1mpro[J], 100, range=(10,25), histtype='step', color='r')
	plt.hist(T2.w1mpro[J2], 100, range=(10,25), histtype='step', color='m')
	plt.xlabel('W1 band')
	plt.axhline(0, color='k', alpha=0.5)
	plt.ylim(-5, 60)
	plt.xlim(10,25)

	plt.savefig('wise5.png')

	sys.exit(0)

	#import cProfile
	#from datetime import tzinfo, timedelta, datetime
	#cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	main()

