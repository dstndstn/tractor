import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from glob import glob
from astrometry.util.pyfits_utils import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.file import *
from astrometry.util.util import *
from astrometry.sdss import *
from tractor import *
from tractor import cfht as cf
from tractor import sdss as st
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

def getdata():
	fn = 'cs82data/W4p1m1_i.V2.7A.swarp.cut.vig15_deV_ord2_size25.fits'
	T = fits_table(fn, hdunum=2)
	print 'Read', len(T), 'rows from', fn
	#T.about()

	r0,r1,d0,d1 = T.alpha_sky.min(), T.alpha_sky.max(), T.delta_sky.min(), T.delta_sky.max()
	print 'Range', r0,r1,d0,d1
	plt.clf()
	plt.plot(T.alpha_sky, T.delta_sky, 'r.')
	plt.xlabel('alpha_sky')
	plt.ylabel('delta_sky')

	rcf = radec_to_sdss_rcf((r0+r1)/2., (d0+d1)/2., radius=60., tablefn='s82fields.fits')
	print 'SDSS fields nearby:', len(rcf)

	rr = [ra  for r,c,f,ra,dec in rcf]
	dd = [dec for r,c,f,ra,dec in rcf]
	plt.plot(rr, dd, 'k.')
	plt.savefig('rd.png')

	RA,DEC = 334.4, 0.3

	rcf = radec_to_sdss_rcf(RA,DEC, radius=10., tablefn='s82fields.fits',
							 contains=True)
	print 'SDSS fields nearby:', len(rcf)
	rcf = [(r,c,f,ra,dec) for r,c,f,ra,dec in rcf if r != 206]
	print 'Filtering out run 206:', len(rcf)

	sdss = DR7()
	sdss.setBasedir('cs82data')
	for r,c,f,ra,dec in rcf:
		for band in 'ugriz':
			print 'Retrieving', r,c,f,band
			st.get_tractor_image(r, c, f, band, psf='dg', useMags=True, sdssobj=sdss)

	plt.clf()
	plt.plot(T.alpha_sky, T.delta_sky, 'r.')
	plt.xlabel('alpha_sky')
	plt.ylabel('delta_sky')
	rr = [ra  for r,c,f,ra,dec in rcf]
	dd = [dec for r,c,f,ra,dec in rcf]
	plt.plot(rr, dd, 'k.')
	#RR,DD = [],[]
	keepWCS = []
	for j,fn in enumerate(glob('cs82data/86*p.fits')):
	# for j,fn in enumerate(glob('cs82data/86*p-21.fits')):
		for i in range(36):
		# for i in [-1]:
			wcs = anwcs(fn, i+1)
			#WCS.append(wcs)
			#print 'image size', wcs.imagew, wcs.imageh
			rr,dd = [],[]
			W,H = wcs.imagew, wcs.imageh
			for x,y in [(1,1),(W,1),(W,H),(1,H),(1,1)]:
				r,d = wcs.pixelxy2radec(x,y)
				rr.append(r)
				dd.append(d)
			rc,dc = wcs.pixelxy2radec(W/2,H/2)
			tc = '%i:%i' % (j,i)
			if wcs.is_inside(RA,DEC):
				keepWCS.append(wcs)
				print 'Keeping', tc
			#RR.append(rr)
			#DD.append(dd)
			plt.plot(rr,dd, 'b-')
			plt.text(rc,dc,tc, color='b')
	#plt.plot(np.array(RR).T, np.array(DD).T, 'b-')
	plt.savefig('rd2.png')


def get_cfht_image(fn, psffn, pixscale):
	wcs = Tan(fn, 0)
	x,y = wcs.radec2pixelxy(RA,DEC)
	print 'x,y', x,y
	S = int(sz / pixscale) / 2
	print '(half) S', S
	cfx,cfy = int(x),int(y)
	cfroi = [cfx-S, cfx+S, cfy-S, cfy+S]
	#cfroi = [cfx-S*2, cfx+S*2, cfy-S, cfy+S]
	x0,x1,y0,y1 = cfroi

	#wcs = FitsWcs(wcs)
	#wcs = RotatedFitsWcs(wcs)
	#wcs.setX0Y0(x0+1., y0+1.)

	P = pyfits.open(fn)
	I = P[1].data
	print 'Img data', I.shape
	roislice = (slice(y0,y1), slice(x0,x1))
	image = I[roislice]
	sky = np.median(image)
	print 'Sky', sky
	# save for later...
	cfsky = sky
	skyobj = ConstantSky(sky)
	# Third plane in image: variance map.
	I = P[3].data
	var = I[roislice]
	cfstd = np.sqrt(np.median(var))

	# Add source noise...
	phdr = P[0].header
	# e/ADU
	gain = phdr.get('GAIN')
	# Poisson statistics are on electrons; var = mean
	el = np.maximum(0, (image - sky) * gain)
	# var in ADU...
	srcvar = el / gain**2
	invvar = 1./(var + srcvar)

	# Apply mask
	I = P[2].data
	mask = I[roislice]
	invvar[mask > 0] = 0.
	del I
	del var

	psfimg = pyfits.open(psffn)[0].data
	print 'PSF image shape', psfimg.shape
	# number of Gaussian components
	PS = psfimg.shape[0]
	K = 3
	w,mu,sig = em_init_params(K, None, None, None)
	II = psfimg.copy()
	II /= II.sum()
	# HACK
	II = np.maximum(II, 0)
	print 'Multi-Gaussian PSF fit...'
	xm,ym = -(PS/2), -(PS/2)
	em_fit_2d(II, xm, ym, w, mu, sig)
	print 'w,mu,sig', w,mu,sig
	psf = GaussianMixturePSF(w, mu, sig)

	photocal = cf.CfhtPhotoCal(hdr=phdr, bandname='i')

	filename = phdr['FILENAME'].strip()


	(H,W) = image.shape
	print 'Image shape', W, H
	print 'x0,y0', x0,y0
	print 'Original WCS:', wcs

	rdcorners = [wcs.pixelxy2radec(x+x0,y+y0) for x,y in [(1,1),(W,1),(W,H),(1,H)]]
	print 'Original RA,Dec corners:', rdcorners

	x,y = wcs.radec2pixelxy(RA,DEC)
	print 'x,y', x,y
	wcs = crop_wcs(wcs, x0, y0, W, H)
	print 'Cropped WCS:', wcs
	x,y = wcs.radec2pixelxy(RA,DEC)
	print 'x,y', x,y
	rdcorners = [wcs.pixelxy2radec(x,y) for x,y in [(1,1),(W,1),(W,H),(1,H)]]
	print 'cropped RA,Dec corners:', rdcorners


	#wcs = rot90_wcs(wcs, W)
	wcs = rot90_wcs(wcs, W, H)
	#wcs = rot90_wcs(wcs, W)
	#wcs = rot90_wcs(wcs, W)
	print 'Rotated WCS:', wcs
	x,y = wcs.radec2pixelxy(RA,DEC)
	print 'x,y', x,y

	#rdcorners = [wcs.pixelxy2radec(x,y) for x,y in [(1,1),(W,1),(W,H),(1,H)]]
	rdcorners = [wcs.pixelxy2radec(x,y) for x,y in [(1,1),(H,1),(H,W),(1,W)]]
	print 'rotated RA,Dec corners:', rdcorners

	wcs = FitsWcs(wcs)
	wcs.setX0Y0(1., 1.)
	#wcs.setX0Y0(x0+1., y0+1.)

	image = np.rot90(image, k=1)
	invvar = np.rot90(invvar, k=1)
	#wcs.W = W
							   
	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal, name='CFHT %s' % filename)
	return cftimg, cfsky, cfstd


# The CFHT images are rotated wrt to SDSS.
#def cfimshow(im, *args, **kwargs):
#	return plt.imshow(np.rot90(im, k=1), *args, **kwargs)

def crop_wcs(wcs, x0, y0, W, H):
	out = Tan()
	out.set_crval(wcs.crval[0], wcs.crval[1])
	out.set_crpix(wcs.crpix[0] - x0, wcs.crpix[1] - y0)
	cd = wcs.get_cd()
	out.set_cd(*cd)
	out.imagew = W
	out.imageh = H
	return out

def rot90_wcs(wcs, W, H):
	#
	out = Tan()
	out.set_crval(wcs.crval[0], wcs.crval[1])
	#out.set_crpix(W+1 - wcs.crpix[1], wcs.crpix[0])
	out.set_crpix(wcs.crpix[1], W+1 - wcs.crpix[0])
	cd = wcs.get_cd()
	out.set_cd(cd[1], -cd[0], cd[3], -cd[2])
	out.imagew = wcs.imageh
	out.imageh = wcs.imagew

	# one direction:
	#out.set_crpix(H+1 - wcs.crpix[1], wcs.crpix[0])
	#out.set_cd(-cd[1], cd[0], -cd[3], cd[2])

	return out


def get_tractor(RA, DEC, sz):
	tractor = Tractor([])

	skies = []
	pixscale = 0.187
	cffns = glob('cs82data/86*p-21-cr.fits')
	print 'CFHT images:', cffns
	for fn in cffns[:1]:
		psffn = fn.replace('-cr', '-psf')
		cfimg,cfsky,cfstd = get_cfht_image(fn, psffn, pixscale)
		#cfimg.imshow = cfimshow
		tractor.addImage(cfimg)
		skies.append((cfsky, cfstd))

	pixscale = 0.396
	S = int(sz / pixscale)
	print 'SDSS size:', S
	rcf = radec_to_sdss_rcf(RA,DEC, radius=10., tablefn='s82fields.fits', contains=True)
	print 'SDSS fields nearby:', len(rcf)
	rcf = [(r,c,f,ra,dec) for r,c,f,ra,dec in rcf if r != 206]
	print 'Filtering out run 206:', len(rcf)

	#rcf = rcf[:16]
	rcf = rcf[:1]
	sdss = DR7()
	sdss.setBasedir('cs82data')
	for r,c,f,ra,dec in rcf:
		for band in 'i':
			print 'Retrieving', r,c,f,band
			im,info = st.get_tractor_image(r, c, f, band, psf='kl-gm', useMags=True,
										   sdssobj=sdss, roiradecsize=(RA,DEC,S/2))
			#im.imshow = plt.imshow
			tractor.addImage(im)
			skies.append((info['sky'], info['skysig']))

	zrs = [np.array([-1.,+6.]) * std + sky for sky,std in skies]
	return tractor,zrs
	
if __name__ == '__main__':
	#getdata()

	RA,DEC = 334.4, 0.3
	sz = 2.*60. # arcsec

	fn = 'cs82data/W4p1m1_i.V2.7A.swarp.cut.vig15_deV_ord2_size25.fits'
	T = fits_table(fn, hdunum=2)
	print 'Read', len(T), 'rows from', fn
	T.ra  = T.alpha_sky
	T.dec = T.delta_sky

	# approx...
	S = sz / 3600.
	ra0 ,ra1  = RA-S/2.,  RA+S/2.
	dec0,dec1 = DEC-S/2., DEC+S/2.

	T = T[(T.ra > ra0) * (T.ra < ra1) * (T.dec > dec0) * (T.dec < dec1)]
	print 'Cut to', len(T), 'objects nearby.'


	pfn = 'tractor.pickle'
	if os.path.exists(pfn):
		print 'Reading pickle', pfn
		tractor,zrs = unpickle_from_file(pfn)
	else:
		tractor,zrs = get_tractor(RA,DEC,sz)
		pickle_to_file((tractor,zrs), pfn)
	

	tim = tractor.getImage(0)
	wcs = tim.getWcs()
	x,y = wcs.positionToPixel(None, RaDecPos(RA,DEC))
	print 'x,y', x,y

	plt.figure(figsize=(6,6))
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.94]

	for step in range(100):


		for i in range(len(tractor.getImages())):
			zr = zrs[i]
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=zr[0], vmax=zr[1], cmap='gray')
			imchi = dict(interpolation='nearest', origin='lower',
						 vmin=-5., vmax=+5., cmap='gray')
			if step == 0:
				tim = tractor.getImage(i)
				data = tim.getImage()

				plt.clf()
				plt.gca().set_position(plotpos0)
				#tim.imshow(data, **ima)
				plt.imshow(data, **ima)
				plt.title('Data %s' % tim.name)
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('data%02i.png' % i)

				if i == 0:
					ax = plt.axis()
					xy = np.array([tim.getWcs().positionToPixel(None, RaDecPos(r,d))
								   for r,d in zip(T.ra, T.dec)])
					print 'xy', xy
					plt.plot(xy[:,0], xy[:,1], 'r.')
					plt.axis(ax)
					plt.savefig('data%02i-ann.png' % i)
					
