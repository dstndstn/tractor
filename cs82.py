#
# http://vn90.phas.ubc.ca/CS82/CS82_data_products/singleframe_V2.7/W4p1m1/i/single_V2.7A/
#
import os
import math
import time
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import multiprocessing
from glob import glob
from astrometry.util.pyfits_utils import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.multiproc import *
from astrometry.util.file import *
from astrometry.util.plotutils import ArcsinhNormalize
from astrometry.util.util import *
from astrometry.sdss import *
from tractor import *
from tractor import cfht as cf
from tractor import sdss as st
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params
import emcee

#from tractor.mpcache import createCache

def print_frozen(tractor):
	for nm,meliq,liq in tractor.getParamStateRecursive():
		if liq:
			print ' ',
		else:
			print 'F',
		if meliq:
			print ' ',
		else:
			print 'F',
		print nm

def get_cfht_image(fn, psffn, pixscale, RA, DEC, sz, bandname=None,
				   filtermap=None, rotate=True):
	if filtermap is None:
		filtermap = {'i.MP9701': 'i'}
	wcs = Tan(fn, 0)
	x,y = wcs.radec2pixelxy(RA,DEC)
	x -= 1
	y -= 1
	print 'x,y', x,y
	S = int(sz / pixscale) / 2
	print '(half) S', S
	cfx,cfy = int(np.round(x)),int(np.round(y))

	P = pyfits.open(fn)
	I = P[1].data
	print 'Img data', I.shape
	H,W = I.shape

	cfroi = [np.clip(cfx-S, 0, W),
			 np.clip(cfx+S, 0, W),
			 np.clip(cfy-S, 0, H),
			 np.clip(cfy+S, 0, H)]
	x0,x1,y0,y1 = cfroi

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
	# MP_BAD  =                    0
	# MP_SAT  =                    1
	# MP_INTRP=                    2
	# MP_CR   =                    3
	# MP_EDGE =                    4
	# HIERARCH MP_DETECTED =       5
	# HIERARCH MP_DETECTED_NEGATIVE = 6
	I = P[2].data.astype(np.uint16)
	#print 'I:', I
	#print I.dtype
	mask = I[roislice]
	#print 'Mask:', mask
	hdr = P[2].header
	badbits = [hdr.get('MP_%s' % nm) for nm in ['BAD', 'SAT', 'INTRP', 'CR']]
	print 'Bad bits:', badbits
	badmask = sum([1 << bit for bit in badbits])
	#print 'Bad mask:', badmask
	#print 'Mask dtype', mask.dtype
	invvar[(mask & int(badmask)) > 0] = 0.
	del I
	del var

	psfimg = pyfits.open(psffn)[0].data
	print 'PSF image shape', psfimg.shape
	# number of Gaussian components
	K = 3
	PS = psfimg.shape[0]
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

	if bandname is None:
		# try looking up in filtermap.
		filt = phdr['FILTER']
		if filt in filtermap:
			print 'Mapping filter', filt, 'to', filtermap[filt]
			bandname = filtermap[filt]
		else:
			print 'No mapping found for filter', filt
			bandname = flit

	photocal = cf.CfhtPhotoCal(hdr=phdr, bandname=bandname)

	filename = phdr['FILENAME'].strip()

	(H,W) = image.shape
	print 'Image shape', W, H
	print 'x0,y0', x0,y0
	print 'Original WCS:', wcs
	rdcorners = [wcs.pixelxy2radec(x+x0,y+y0) for x,y in [(1,1),(W,1),(W,H),(1,H)]]
	print 'Original RA,Dec corners:', rdcorners
	wcs = crop_wcs(wcs, x0, y0, W, H)
	print 'Cropped WCS:', wcs
	rdcorners = [wcs.pixelxy2radec(x,y) for x,y in [(1,1),(W,1),(W,H),(1,H)]]
	print 'cropped RA,Dec corners:', rdcorners
	if rotate:
		wcs = rot90_wcs(wcs, W, H)
		print 'Rotated WCS:', wcs
		rdcorners = [wcs.pixelxy2radec(x,y) for x,y in [(1,1),(H,1),(H,W),(1,W)]]
		print 'rotated RA,Dec corners:', rdcorners

		print 'rotating images...'
		image = np.rot90(image, k=1)
		invvar = np.rot90(invvar, k=1)

	wcs = FitsWcs(wcs)

	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal, name='CFHT %s' % filename)
	return cftimg, cfsky, cfstd

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
	out = Tan()
	out.set_crval(wcs.crval[0], wcs.crval[1])
	out.set_crpix(wcs.crpix[1], W+1 - wcs.crpix[0])
	cd = wcs.get_cd()
	out.set_cd(cd[1], -cd[0], cd[3], -cd[2])
	out.imagew = wcs.imageh
	out.imageh = wcs.imagew
	# opposite direction:
	#out.set_crpix(H+1 - wcs.crpix[1], wcs.crpix[0])
	#out.set_cd(-cd[1], cd[0], -cd[3], cd[2])
	return out


def _mapf_sdss_im((r, c, f, band, sdss, sdss_psf, cut_sdss, RA, DEC, S, objname,
				   nanomaggies)):
	print 'Retrieving', r,c,f,band
	kwargs = {}
	if cut_sdss:
		kwargs.update(roiradecsize=(RA,DEC,S/2))
	try:
		im,info = st.get_tractor_image(r, c, f, band, sdssobj=sdss,
									   psf=sdss_psf, nanomaggies=nanomaggies,
									   **kwargs)
	except:
		import traceback
		print 'Exception in get_tractor_image():'
		traceback.print_exc()
		print 'Failed to get R,C,F,band', r,c,f,band
		return None,None
		#raise
	if im is None:
		return None,None

	if objname is not None:
		print 'info', info
		obj = info['object']
		print 'Header object: "%s"' % obj
		if obj != objname:
			print 'Skipping obj !=', objname
			return None,None
		
	print 'Image size', im.getWidth(), im.getHeight()
	if im.getWidth() == 0 or im.getHeight() == 0:
		return None,None
	im.rcf = (r,c,f)
	return im,(info['sky'], info['skysig'])

def get_tractor(RA, DEC, sz, cffns, mp, filtermap=None, sdssbands=None,
				just_rcf=False,
				sdss_psf='kl-gm', cut_sdss=True,
				good_sdss_only=False, sdss_object=None,
				rotate_cfht=True,
				nanomaggies=False,
				nimages=None):
	if sdssbands is None:
		sdssbands = ['u','g','r','i','z']
	tractor = Tractor()

	skies = []
	ims = []
	pixscale = 0.187
	print 'CFHT images:', cffns
	for fn in cffns:
		psffn = fn.replace('-cr', '-psf')
		cfimg,cfsky,cfstd = get_cfht_image(fn, psffn, pixscale, RA, DEC, sz,
										   filtermap=filtermap, rotate=rotate_cfht)
		ims.append(cfimg)
		skies.append((cfsky, cfstd))
		if nanomaggies:
			# unimplemented
			assert(False)

	pixscale = 0.396
	S = int(sz / pixscale)
	print 'SDSS size:', S, 'pixels'
	# Find all SDSS images that could overlap the RA,Dec +- S/2,S/2 box
	R = np.sqrt(2.*(S/2.)**2 + (2048/2.)**2 + (1489/2.)**2) * pixscale / 60.
	print 'Search radius:', R, 'arcmin'
	rcf = radec_to_sdss_rcf(RA,DEC, radius=R, tablefn='s82fields.fits')
	print 'SDSS fields nearby:', len(rcf)
	rcf = [(r,c,f,ra,dec) for r,c,f,ra,dec in rcf if r != 206]
	print 'Filtering out run 206:', len(rcf)

	if just_rcf:
		return rcf
	sdss = DR7(basedir='cs82data/dr7')
	#sdss = DR9(basedir='cs82data/dr9')

	if good_sdss_only:
		W = fits_table('window_flist-DR8-S82.fits')
		print 'Building fidmap...'
		fidmap = dict(zip(W.run * 10000 + W.camcol * 1000 + W.field, W.score))
		print 'finding scores...'
		scores = []
		noscores = []
		rcfscore = {}
		for r,c,f,nil,nil in rcf:
			print 'RCF', r,c,f
			fid = r*10000 + c*1000 + f
			score = fidmap.get(fid, None)
			if score is None:
				print 'No entry'
				noscores.append((r,c,f))
				continue
			print 'score', score
			scores.append(score)
			rcfscore[(r,c,f)] = score
		print 'No scores:', noscores
		#plt.clf()
		#plt.hist(scores, 20)
		#plt.savefig('scores.png')
		print len(scores), 'scores'
		scores = np.array(scores)
		print sum(scores > 0.5), '> 0.5'

	args = []
	for r,c,f,ra,dec in rcf:
		if good_sdss_only:
			score = rcfscore.get((r,c,f), 0.)
			if score < 0.5:
				print 'Skipping,', r,c,f
				continue
		for band in sdssbands:
			args.append((r, c, f, band, sdss, sdss_psf, cut_sdss, RA, DEC, S, sdss_object, nanomaggies))

	# Just do a subset of the fields?
	if nimages:
		args = args[:nimages]

	print 'Getting', len(args), 'SDSS images...'
	X = mp.map(_mapf_sdss_im, args)
	print 'Got', len(X), 'SDSS images.'
	for im,sky in X:
		if im is None:
			continue
		ims.append(im)
		skies.append(sky)
	print 'Kept', len(X), 'SDSS images.'

	tractor.setImages(Images(*ims))
	return tractor,skies

def mysavefig(fn):
	plt.savefig(fn)
	print 'Wrote', fn

def get_cf_sources2(RA, DEC, sz, maglim=25, mags=['u','g','r','i','z'],
					nanomaggies=False):
	Tcomb = fits_table('cs82data/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	#Tcomb.about()

	plt.clf()
	plt.plot(Tcomb.mag_spheroid, Tcomb.mag_disk, 'r.')
	plt.axhline(26)
	plt.axvline(26)
	plt.axhline(27)
	plt.axvline(27)
	plt.savefig('magmag.png')

	plt.clf()
	I = (Tcomb.chi2_psf < 1e7)
	plt.loglog(Tcomb.chi2_psf[I], Tcomb.chi2_model[I], 'r.')
	plt.xlabel('chi2 psf')
	plt.ylabel('chi2 model')
	ax = plt.axis()
	plt.plot([ax[0],ax[1]], [ax[0],ax[1]], 'k-')
	plt.axis(ax)
	plt.savefig('psfgal.png')

	# approx...
	S = sz / 3600.
	ra0 ,ra1  = RA-S/2.,  RA+S/2.
	dec0,dec1 = DEC-S/2., DEC+S/2.

	T = Tcomb
	print 'Read', len(T), 'sources'
	T.ra  = T.alpha_j2000
	T.dec = T.delta_j2000
	T = T[(T.ra > ra0) * (T.ra < ra1) * (T.dec > dec0) * (T.dec < dec1)]
	print 'Cut to', len(T), 'objects nearby.'

	srcs = Catalog()
	for t in T:

		if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
			#print 'PSF'
			themag = t.mag_psf
			m = Mags(order=mags, **dict([(k, themag) for k in mags]))
			if nanomaggies:
				m = NanoMaggies.fromMag(m)
			srcs.append(PointSource(RaDecPos(t.alpha_j2000, t.delta_j2000), m))
			continue

		if t.mag_disk > maglim and t.mag_spheroid > maglim:
			#print 'Faint'
			continue

		# deV: spheroid
		# exp: disk

		themag = t.mag_spheroid
		m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))
		themag = t.mag_disk
		m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))

		if nanomaggies:
			m_dev = NanoMaggies.fromMag(m_dev)
			m_exp = NanoMaggies.fromMag(m_exp)

		# SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

		shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600., t.disk_aspect_world,
								t.disk_theta_world + 90.)
		shape_dev = GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
								t.spheroid_theta_world + 90.)
		pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

		if t.mag_disk > maglim and t.mag_spheroid <= maglim:
			srcs.append(DevGalaxy(pos, m_dev, shape_dev))
			continue
		if t.mag_disk <= maglim and t.mag_spheroid > maglim:
			srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
			continue

		srcs.append(CompositeGalaxy(pos, m_exp, shape_exp, m_dev, shape_dev))

	print 'Sources:', len(srcs)
	return srcs

def get_cf_sources3(RA, DEC, sz, magcut=100, mags=['u','g','r','i','z']):
	Tcomb = fits_table('cs82data/cs82_morphology_may2012.fits')
	Tcomb.about()

	plt.clf()
	plt.plot(Tcomb.mag_spheroid, Tcomb.mag_disk, 'r.')
	plt.axhline(26)
	plt.axvline(26)
	plt.axhline(27)
	plt.axvline(27)
	plt.savefig('magmag.png')

	plt.clf()
	I = (Tcomb.chi2_psf < 1e7)
	plt.loglog(Tcomb.chi2_psf[I], Tcomb.chi2_model[I], 'r.')
	plt.xlabel('chi2 psf')
	plt.ylabel('chi2 model')
	ax = plt.axis()
	plt.plot([ax[0],ax[1]], [ax[0],ax[1]], 'k-')
	plt.axis(ax)
	plt.savefig('psfgal.png')

	# approx...
	S = sz / 3600.
	ra0 ,ra1  = RA-S/2.,  RA+S/2.
	dec0,dec1 = DEC-S/2., DEC+S/2.

	T = Tcomb
	print 'Read', len(T), 'sources'
	T.ra  = T.alpha_j2000
	T.dec = T.delta_j2000
	T = T[(T.ra > ra0) * (T.ra < ra1) * (T.dec > dec0) * (T.dec < dec1)]
	print 'Cut to', len(T), 'objects nearby.'

	maglim = 27.
	
	srcs = []
	for t in T:

		if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
			#print 'PSF'
			themag = t.mag_psf
			m = Mags(order=mags, **dict([(k, themag) for k in mags]))
			srcs.append(PointSource(RaDecPos(t.alpha_j2000, t.delta_j2000), m))
			continue

		if t.mag_disk > maglim and t.mag_spheroid > maglim:
			#print 'Faint'
			continue

		themag = t.mag_spheroid
		m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))
		themag = t.mag_disk
		m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))

		# SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

		shape_dev = GalaxyShape(t.disk_scale_world * 1.68 * 3600., t.disk_aspect_world,
								t.disk_theta_world + 90.)
		shape_exp = GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
								t.spheroid_theta_world + 90.)
		pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

		if t.mag_disk > maglim and t.mag_spheroid <= maglim:
			# exp
			#print 'Exp'
			srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
			continue
		if t.mag_disk <= maglim and t.mag_spheroid > maglim:
			# deV
			#print 'deV'
			srcs.append(DevGalaxy(pos, m_dev, shape_dev))
			continue

		# exp + deV
		#print 'comp'
		srcs.append(CompositeGalaxy(pos, m_exp, shape_exp, m_dev, shape_dev))
	print 'Sources:', len(srcs)
	#for src in srcs:
	#	print '  ', src
	return srcs




def tweak_wcs((tractor, im)):
	#print 'Tractor', tractor
	#print 'Image', im
	tractor.images = Images(im)
	print 'tweak_wcs: fitting params:', tractor.getParamNames()
	for step in range(10):
		print 'Run optimization step', step
		t0 = Time()
		dlnp,X,alpha = tractor.optimize(alphas=[0.5, 1., 2., 4.])
		t_opt = (Time() - t0)
		print 'alpha', alpha
		print 'Optimization took', t_opt, 'sec'
		lnp0 = tractor.getLogProb()
		print 'Lnprob', lnp0
		if dlnp == 0:
			break
	return im.getParams()

def plot1((tractor, i, zr, plotnames, step, pp, ibest, tsuf, colorbar, fmt)):
	#plt.figure(figsize=(6,6))
	plt.figure(figsize=(10,10))
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.94]

	print 'zr = ', zr

	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1], cmap='gray')
	imchi = dict(interpolation='nearest', origin='lower',
				 vmin=-5., vmax=+5., cmap='gray')
	imchi2 = dict(interpolation='nearest', origin='lower',
				  vmin=-50., vmax=+50., cmap='gray')
	tim = tractor.getImage(i)

	data = tim.getImage()
	q0,q1,q2,q3,q4 = np.percentile(data.ravel(), [0, 25, 50, 75, 100])
	print 'Data quartiles:', q0, q1, q2, q3, q4

	ima.update(norm=ArcsinhNormalize(mean=q2, std=(q3-q1)/2., vmin=zr[0], vmax=zr[1]),
			   vmin=None, vmax=None,    nonl=True)

	#plt.clf()
	#plt.hist(data.ravel(), bins=100, log=True)
	#plt.savefig('data-hist-%02i.png' % step)

	if 'data' in plotnames:
		data = tim.getImage()
		plt.clf()
		plt.gca().set_position(plotpos0)
		myimshow(data, **ima)
		tt = 'Data %s' % tim.name
		if tsuf is not None:
			tt += tsuf
		plt.title(tt)
		#plt.xticks([],[])
		#plt.yticks([],[])
		if colorbar:
			plt.colorbar()
		mysavefig('data-%02i' % i + fmt)

	if 'dataann' in plotnames and i == 0:
		ax = plt.axis()
		xy = np.array([tim.getWcs().positionToPixel(s.getPosition())
					   for s in tractor.catalog])
		plt.plot(xy[:,0], xy[:,1], 'r+')
		plt.axis(ax)
		mysavefig(('data-%02i-ann'+fmt) % i)

	if ('modbest' in plotnames or 'chibest' in plotnames or
		'modnoise' in plotnames or 'chinoise' in plotnames):
		pbest = pp[ibest,:]
		tractor.setParams(pp[ibest,:])

		if 'modnoise' in plotnames or 'chinoise' in plotnames:
			ierr = tim.getInvError()
			noiseim = np.random.normal(size=ierr.shape)
			I = (ierr > 0)
			noiseim[I] *= 1./ierr[I]
			noiseim[np.logical_not(I)] = 0.

		if 'modbest' in plotnames or 'modnoise' in plotname:
			mod = tractor.getModelImage(i)

		if 'modbest' in plotnames:
			#plt.clf()
			#plt.hist(mod.ravel(), bins=100, log=True)
			#plt.savefig(('mod-hist-%02i'+fmt) % step)

			plt.clf()
			plt.gca().set_position(plotpos0)
			myimshow(mod, **ima)
			tt = 'Model %s' % tim.name
			if tsuf is not None:
				tt += tsuf
			plt.title(tt)

			#plt.xticks([],[])
			#plt.yticks([],[])
			if colorbar:
				plt.colorbar()

			mysavefig(('modbest-%02i-%02i'+fmt) % (i,step))

		if 'modnoise' in plotnames:
			plt.clf()
			plt.gca().set_position(plotpos0)
			myimshow(mod + noiseim, **ima)
			tt = 'Model+noise %s' % tim.name
			if tsuf is not None:
				tt += tsuf
			plt.title(tt)
			if colorbar:
				plt.colorbar()
			mysavefig(('modnoise-%02i-%02i'+fmt) % (i,step))

		if 'chibest' in plotnames:
			chi = tractor.getChiImage(i)
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(chi, **imchi)
			tt = 'Chi (best) %s' % tim.name
			if tsuf is not None:
				tt += tsuf
			plt.title(tt)
			plt.xticks([],[])
			plt.yticks([],[])
			if colorbar:
				plt.colorbar()
			mysavefig(('chibest-%02i-%02i'+fmt) % (i,step))

			# plt.clf()
			# plt.gca().set_position(plotpos0)
			# plt.imshow(chi, **imchi2)
			# plt.title(tt)
			# plt.xticks([],[])
			# plt.yticks([],[])
			# plt.colorbar()
			# mysavefig('chibest2-%02i-%02i'+fmt % (i,step))

		if 'chinoise' in plotnames:
			chi = (data - (mod + noiseim)) * tim.getInvError()
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(chi, **imchi)
			tt = 'Chi+noise %s' % tim.name
			if tsuf is not None:
				tt += tsuf
			plt.title(tt)
			plt.xticks([],[])
			plt.yticks([],[])
			if colorbar:
				plt.colorbar()
			mysavefig(('chinoise-%02i-%02i'+fmt) % (i,step))

	if 'modsum' in plotnames or 'chisum' in plotnames:
		modsum = None
		chisum = None
		if pp is None:
			pp = np.array([tractor.getParams()])
		nw = len(pp)
		print 'modsum/chisum plots for', nw, 'walkers'
		for k in xrange(nw):
			tractor.setParams(pp[k,:])
			mod = tractor.getModelImage(i)
			chi = tractor.getChiImage(i)
			if k == 0:
				modsum = mod
				chisum = chi
			else:
				modsum += mod
				chisum += chi

		if 'modsum' in plotnames:
			plt.clf()
			plt.gca().set_position(plotpos0)
			myimshow(modsum/float(nw), **ima)
			tt = 'Model (sum) %s' % tim.name
			if tsuf is not None:
				tt += tsuf
			plt.title(tt)
			plt.xticks([],[])
			plt.yticks([],[])
			if colorbar:
				plt.colorbar()
			mysavefig(('modsum-%02i-%02i'+fmt) % (i,step))
		if 'chisum' in plotnames:
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(chisum/float(nw), **imchi)
			tt = 'Chi (sum) %s' % tim.name
			if tsuf is not None:
				tt += tsuf
			plt.title(tt)
			plt.xticks([],[])
			plt.yticks([],[])
			plt.colorbar()
			mysavefig('chisum-%02i-%02i'+fmt % (i,step))
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(chisum/float(nw), **imchi2)
			plt.title(tt)
			plt.xticks([],[])
			plt.yticks([],[])
			if colorbar:
				plt.colorbar()
			mysavefig(('chisum2-%02i-%02i'+fmt) % (i,step))

def plots(tractor, plotnames, step, pp=None, mp=None, ibest=None, imis=None, alllnp=None,
		  tsuf=None, colorbar=True, format='.png'):
	print 'plots...'
	if 'lnps' in plotnames:
		plotnames.remove('lnps')
		plt.figure(figsize=(6,6))
		plt.clf()
		plotpos0 = [0.15, 0.15, 0.84, 0.80]
		plt.gca().set_position(plotpos0)
		for s,lnps in enumerate(alllnp):
			plt.plot(np.zeros_like(lnps)+s, lnps, 'r.')
		plt.savefig('lnps-%02i.png' % step)

	args = []
	if imis is None:
		imis = range(len(tractor.getImages()))
	NI = len(tractor.getImages())
	for i in imis:
		if i >= NI:
			print 'Skipping plot of image', i, 'with N images', NI
			continue
		zr = tractor.getImage(i).zr
		args.append((tractor, i, zr, plotnames, step, pp, ibest, tsuf, colorbar, format))
	if mp is None:
		map(plot1, args)
	else:
		mp.map(plot1, args)
	print 'plots done'


def nlmap(X):
	S = 0.01
	return np.arcsinh(X * S)/S
def myimshow(x, *args, **kwargs):
	if kwargs.get('nonl', False):
		kwargs = kwargs.copy()
		kwargs.pop('nonl')
		return plt.imshow(x, *args, **kwargs)
	mykwargs = kwargs.copy()
	if 'vmin' in kwargs:
		mykwargs['vmin'] = nlmap(kwargs['vmin'])
	if 'vmax' in kwargs:
		mykwargs['vmax'] = nlmap(kwargs['vmax'])
	return plt.imshow(nlmap(x), *args, **mykwargs)

def getlnp((tractor, i, par0, step)):
	tractor.setParam(i, par0+step)
	lnp = tractor.getLogProb()
	tractor.setParam(i, par0)
	return lnp


dpool = None
def pool_stats():
	if dpool is None:
		return
	print 'Total pool CPU time:', dpool.get_worker_cpu()

def cut_bright(cat, magcut=24, mag='i'):
	brightcat = Catalog()
	I = []
	mags = []
	for i,src in enumerate(cat):
		#m = getattr(src.getBrightness(), mag)
		m = src.getBrightness().getMag(mag)
		if m < magcut:
			#brightcat.append(src)
			I.append(i)
			mags.append(m)
	J = np.argsort(mags)
	I = np.array(I)
	I = I[J]
	for i in I:
		brightcat.append(cat[i])
	return brightcat, I


def get_wise_coadd_images(RA, DEC, radius, bandnums = [1,2,3,4],
						  nanomaggies=False):
	from wise import read_wise_coadd
	bands = ['w%i' % n for n in bandnums]
	basedir = 'cs82data/wise/level3/'
	pat = '3342p000_ab41-w%i'
	#pat = '04933b137-w%i'
	filtermap = None

	radius /= 3600.
	# HACK - no cos(dec)
	radecbox = [RA-radius, RA+radius, DEC-radius, DEC+radius]
	
	ims = []
	for band in bandnums:
		base = pat % band
		basefn = os.path.join(basedir, base)
		im = read_wise_coadd(basefn, radecroi=radecbox, filtermap=filtermap,
							 nanomaggies=nanomaggies)
		ims.append(im)
	return ims

def get_cfht_coadd_image(RA, DEC, S, bandname=None, filtermap=None,
						 doplots=False, psfK=3, nanomaggies=False):
	if filtermap is None:
		filtermap = {'i.MP9701': 'i'}
	fn = 'cs82data/W4p1m1_i.V2.7A.swarp.cut.fits'
	wcs = Tan(fn, 0, 1)
	P = pyfits.open(fn)
	image = P[0].data
	phdr = P[0].header
	print 'Image', image.shape
	(H,W) = image.shape
	OH,OW = H,W

	#x,y = np.array([1,W,W,1,1]), np.array([1,1,H,H,1])
	#rco,dco = wcs.pixelxy2radec(x, y)

	# The coadd image has my ROI roughly in the middle.
	# Pixel 1,1 is the high-RA, low-Dec corner.
	x,y = wcs.radec2pixelxy(RA, DEC)
	x -= 1
	y -= 1
	print 'Center pix:', x,y
	xc,yc = int(x), int(y)
	image = image[yc-S: yc+S, xc-S: xc+S]
	image = image.copy()
	print 'Subimage:', image.shape
	twcs = FitsWcs(wcs)
	twcs.setX0Y0(xc-S, yc-S)
	xs,ys = twcs.positionToPixel(RaDecPos(RA, DEC))
	print 'Subimage center pix:', xs,ys
	rd = twcs.pixelToPosition(xs, ys)
	print 'RA,DEC vs RaDec', RA,DEC, rd

	if bandname is None:
		# try looking up in filtermap.
		filt = phdr['FILTER']
		if filt in filtermap:
			print 'Mapping filter', filt, 'to', filtermap[filt]
			bandname = filtermap[filt]
		else:
			print 'No mapping found for filter', filt
			bandname = filt

	zp = float(phdr['MAGZP'])
	print 'Zeropoint', zp
	if nanomaggies:
		photocal = LinearPhotoCal(NanoMaggies.zeropointToScale(zp), band=bandname)
	else:
		photocal = MagsPhotoCal(bandname, zp)
	print photocal

	fn = 'cs82data/W4p1m1_i.V2.7A.swarp.cut.weight.fits'
	P = pyfits.open(fn)
	weight = P[0].data
	weight = weight[yc-S:yc+S, xc-S:xc+S].copy()
	print 'Weight', weight.shape
	print 'Median', np.median(weight.ravel())
	invvar = weight

	fn = 'cs82data/W4p1m1_i.V2.7A.swarp.cut.flag.fits'
	P = pyfits.open(fn)
	flags = P[0].data
	flags = flags[yc-S:yc+S, xc-S:xc+S].copy()
	print 'Flags', flags.shape
	del P
	invvar[flags == 1] = 0.

	fn = 'cs82data/snap_W4p1m1_i.V2.7A.swarp.cut.fits'
	psfim = pyfits.open(fn)[0].data
	H,W = psfim.shape
	N = 9
	assert(((H % N) == 0) and ((W % N) == 0))
	# Select which of the NxN PSF images applies to our cutout.
	ix = int(N * float(xc) / OW)
	iy = int(N * float(yc) / OH)
	print 'PSF image number', ix,iy
	PW,PH = W/N, H/N
	print 'PSF image shape', PW,PH

	psfim = psfim[iy*PH: (iy+1)*PH, ix*PW: (ix+1)*PW]
	print 'my PSF image shape', PW,PH
	psfim = np.maximum(psfim, 0)
	psfim /= np.sum(psfim)

	K = psfK
	w,mu,sig = em_init_params(K, None, None, None)
	xm,ym = -(PW/2), -(PH/2)
	em_fit_2d(psfim, xm, ym, w, mu, sig)
	tpsf = GaussianMixturePSF(w, mu, sig)

	tsky = ConstantSky(0.)

	obj = phdr['OBJECT'].strip()

	tim = Image(data=image, invvar=invvar, psf=tpsf, wcs=twcs, photocal=photocal,
				sky=tsky, name='CFHT coadd %s %s' % (obj, bandname))

	# set "zr" for plots
	sig = 1./np.median(tim.inverr)
	tim.zr = np.array([-1., +20.]) * sig

	if not doplots:
		return tim

	psfimpatch = Patch(-(PW/2), -(PH/2), psfim)
	# number of Gaussian components
	for K in range(1, 4):
		w,mu,sig = em_init_params(K, None, None, None)
		xm,ym = -(PW/2), -(PH/2)
		em_fit_2d(psfim, xm, ym, w, mu, sig)
		#print 'w,mu,sig', w,mu,sig
		psf = GaussianMixturePSF(w, mu, sig)
		patch = psf.getPointSourcePatch(0, 0)

		plt.clf()
		plt.subplot(1,2,1)
		plt.imshow(patch.getImage(), interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.subplot(1,2,2)
		plt.imshow((patch - psfimpatch).getImage(), interpolation='nearest', origin='lower')
		plt.colorbar()
		plt.savefig('copsf-%i.png' % K)

	plt.clf()
	plt.imshow(psfim, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('copsf.png')

	print 'image min', image.min()
	plt.clf()
	plt.imshow(image, interpolation='nearest', origin='lower',
			   vmin=0, vmax=10.)
	plt.colorbar()
	plt.savefig('coim.png')
	plt.clf()
	plt.imshow(image, interpolation='nearest', origin='lower',
			   vmin=0, vmax=3.)
	plt.colorbar()
	plt.savefig('coim2.png')
	plt.clf()
	plt.imshow(image, interpolation='nearest', origin='lower',
			   vmin=0, vmax=1.)
	plt.colorbar()
	plt.savefig('coim3.png')
	plt.clf()
	plt.imshow(image, interpolation='nearest', origin='lower',
			   vmin=0, vmax=0.3)
	plt.colorbar()
	plt.savefig('coim4.png')

	plt.clf()
	plt.imshow(image * np.sqrt(invvar), interpolation='nearest', origin='lower',
			   vmin=-3, vmax=10.)
	plt.colorbar()
	plt.savefig('cochi.png')

	plt.clf()
	plt.imshow(weight, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('cowt.png')

	plt.clf()
	plt.imshow(invvar, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('coiv.png')

	plt.clf()
	plt.imshow(flags, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('cofl.png')

	return tim



def optsources_searchlight(tractor, im, step0,
						   npix = 100,
						   doplots=True, plotsa={},
						   mindlnp=1e-3):
	step = step0
	alllnp = []

	# # sweep across the image, optimizing in circles.
	# # we'll use the healpix grid for circle centers.
	# # how big? in arcmin
	# R = 0.25
	# Rpix = R / 60. / np.sqrt(np.abs(np.linalg.det(im.wcs.cdAtPixel(0,0))))
	# nside = int(healpix_nside_for_side_length_arcmin(R/2.))
	# if nside > 13377:
	# 	print 'Clamping Nside from', nside, 'to 13376'
	# 	nside = 13376
	# print 'Nside', nside
	# #print 'radius in pixels:', Rpix
	# # start in one corner.
	# pos = im.wcs.pixelToPosition(0, 0)
	# hp = radecdegtohealpix(pos.ra, pos.dec, nside)
	# hpqueue = [hp]
	# hpdone = []

	H,W = im.shape
	nx,ny = int(np.ceil(W / float(npix))), int(np.ceil(H / float(npix)))
	XX = np.linspace(npix/2, W-npix/2, nx)
	YY = np.linspace(npix/2, H-npix/2, ny)
	dx,dy = XX[1]-XX[0], YY[1]-YY[0]
	print 'Optimizing on a grid of', len(XX), 'x', len(YY), 'cells'
	print 'Cell sizes', dx,'x',dy, 'pixels'
	XY = zip(XX,YY)
	R = np.sqrt(2.)*npix/2. * np.sqrt(np.abs(np.linalg.det(im.wcs.cdAtPixel(0,0))))
	print 'Radius', R, 'deg'

	# while len(hpqueue):
	# 	hp = hpqueue.pop()
	# 	hpdone.append(hp)
	# 	print 'looking at healpix', hp
	# 	ra,dec = healpix_to_radecdeg(hp, nside, 0.5, 0.5)
	# 	print 'RA,Dec center', ra,dec
	# 	x,y = im.wcs.positionToPixel(RaDecPos(ra,dec))
	# 	H,W	 = im.shape
	# 	if x < -Rpix or y < -Rpix or x >= W+Rpix or y >= H+Rpix:
	# 		print 'pixel', x,y, 'out of bounds'
	# 		continue
	# 
	# 	# add neighbours
	# 	nn = healpix_get_neighbours(hp, nside)
	# 	print 'healpix neighbours', nn
	# 	for ni in nn:
	# 		if ni in hpdone:
	# 			continue
	# 		if ni in hpqueue:
	# 			continue
	# 		hpqueue.append(ni)
	# 		print 'enqueued neighbour', ni
	# 
	# 	# FIXME -- add PSF-sized margin to radius
	# 	#ra,dec = (radecroi[0]+radecroi[1])/2., (radecroi[2]+radecroi[3])/2.
	for xi,yi in XY:
		print 'x,y', xi,yi
		rd = im.wcs.pixelToPosition(xi, yi)
		ra,dec = rd.ra, rd.dec
		print 'ra,dec', ra,dec

		print 'All sources:', len(tractor.catalog)
		tractor.catalog.freezeAllParams()
		tractor.catalog.thawSourcesInCircle(RaDecPos(ra, dec), R/60.)

		plt.clf()
		xx,yy = [],[]
		for src in tractor.catalog:
			x,y = im.wcs.positionToPixel(src.getPosition())
			xx.append(x)
			yy.append(y)
		plt.plot(xx, yy, 'k.', alpha=0.5)
		xx,yy = [],[]
		for src in tractor.catalog.getThawedSources():
			x,y = im.wcs.positionToPixel(src.getPosition())
			xx.append(x)
			yy.append(y)
		plt.plot(xx, yy, 'r.')
		plt.plot([xi],[yi], 'ro')
		for xx,yy in XY:
			plt.axhline(yy, color='k', alpha=0.5)
			plt.axvline(xx, color='k', alpha=0.5)
		plt.axis([0,W,0,H])
		fn = 'circle-%04i.png' % step
		plt.savefig(fn)
		print 'Saved', fn

		for ss in range(10):
			print 'Optimizing:', len(tractor.getParamNames())
			for nm in tractor.getParamNames()[:10]:
				print nm
			print '...'
			(dlnp,X,alpha) = tractor.optimize() #damp=1.)
			print 'dlnp', dlnp
			print 'alpha', alpha
			lnp0 = tractor.getLogProb()
			alllnp.append([lnp0])

			if doplots:
				pp = np.array([tractor.getParams()])
				ibest = 0
				plots(tractor,
					  ['modbest', 'chibest', 'lnps'],
					  step, pp=pp, ibest=ibest, alllnp=alllnp, **plotsa)
			step += 1
			if alpha == 0 or dlnp < mindlnp:
				break

		tractor.catalog.freezeAllParams()
	return step, alllnp
	

def optsourcestogether(tractor, step0, doplots=True, plotsa={}, mindlnp=1e-3):
	step = step0
	alllnp = []
	while True:
		print 'Run optimization step', step
		t0 = Time()
		dlnp,X,alpha = tractor.optimize(alphas=[0.01, 0.125, 0.25, 0.5, 1., 2., 4.])
		t_opt = (Time() - t0)
		#print 'alpha', alpha
		print 'Optimization took', t_opt, 'sec'
		assert(all(np.isfinite(X)))
		lnp0 = tractor.getLogProb()
		print 'Lnprob', lnp0
		alllnp.append([lnp0])
		if doplots:
			pp = np.array([tractor.getParams()])
			ibest = 0
			plots(tractor,
				  ['modbest', 'chibest', 'lnps'],
				  step, pp=pp, ibest=ibest, alllnp=alllnp, **plotsa)
		step += 1
		if alpha == 0 or dlnp < mindlnp:
			break
	return step, alllnp

def optsourcesseparate(tractor, step0, plotmod=10, plotsa={}, doplots=True, sortmag='i2',
					   mindlnp=1e-3):
	step = step0 - 1
	tractor.catalog.freezeAllParams()
	I = np.argsort([getattr(src.getBrightness(), sortmag) for src in tractor.catalog])

	allsources = tractor.catalog

	pool_stats()

	alllnp = []
	for j,srci in enumerate(I):
		srci = int(srci)
		print 'source', j, 'of', len(I), ', srci', srci, ':', tractor.catalog[srci]
		tractor.catalog.thawParam(srci)
		print tractor.numberOfParams(), 'active parameters'
		for nm in tractor.getParamNames():
			print '  ', nm
		print 'Source:', tractor.catalog[srci]

		# Here we subtract the other models from each image.
		# We could instead keep a table of model patches x images,
		# but this is not the bottleneck at the moment...
		tt0 = Time()
		others = tractor.catalog[0:srci] + tractor.catalog[srci+1:]
		origims = []
		for im in tractor.images:
			origims.append(im.data)
			sub = im.data.copy()
			sub -= tractor.getModelImage(im, others, sky=False)
			im.data = sub
		tractor.catalog = Catalog(allsources[srci])
		tpre = Time()-tt0
		print 'Removing other sources:', tpre
		src = allsources[srci]

		tt0 = Time()
		p0 = tractor.catalog.getParams()
		while True:
			step += 1
			print 'Run optimization step', step
			t0 = Time()
			dlnp,X,alpha = tractor.optimize(alphas=[0.01, 0.125, 0.25, 0.5, 1., 2., 4.])
			t_opt = (Time() - t0)
			print 'Optimization took', t_opt, 'sec'
			print src
			lnp0 = tractor.getLogProb()
			print 'Lnprob', lnp0
			alllnp.append([lnp0])
			if doplots and step % plotmod == 0:
				print 'Plots...'
				pp = np.array([tractor.getParams()])
				ibest = 0
				plots(tractor,
					  ['modbest', 'chibest', 'lnps'],
					  step, pp=pp, ibest=ibest, alllnp=alllnp, **plotsa)
				print 'Done plots.'
			if alpha == 0 or dlnp < mindlnp:
				break
		print 'removing other sources:', Time()-tt0

		tractor.catalog = allsources
		for im,dat in zip(tractor.images, origims):
			im.data = dat

		tractor.catalog.freezeParam(srci)

		pool_stats()

	return step, alllnp






# Read image files and catalogs, make Tractor object.
def stage00(mp=None, plotsa=None, RA=None, DEC=None, sz=None,
			doplots=True, **kwargs):
	filtermap = {'i.MP9701': 'i2'}
	sdssbands = ['u','g','r','i','z']

	opt = kwargs.pop('opt')
	if opt.bands:
		sdssbands = list(opt.bands)
	if opt.nimages:
		nimages = opt.nimages
	else:
		nimages = None
		
	# nanomaggies?
	donm = True

	print 'Grabbing WISE images'
	wiseims = get_wise_coadd_images(RA, DEC, sz/2., nanomaggies=donm)

	import cPickle
	print 'Pickling WISE images...'
	SS = cPickle.dumps(wiseims)

	print 'UnPickling WISE images...'
	wims = cPickle.loads(SS)

	print wims

	srcs = get_cf_sources2(RA, DEC, sz, mags=['i2'] + sdssbands,
						   nanomaggies=donm)
	#srcs = get_cf_sources2(RA, DEC, sz, mags=sdssbands + ['i2'])
	#srcs = get_cf_sources3(RA, DEC, sz, mags=sdssbands + ['i2'])

	#cffns = glob('cs82data/86*p-21-cr.fits')
	cffns = []
	tractor,skies = get_tractor(RA,DEC,sz, cffns, mp, filtermap=filtermap, sdssbands=sdssbands,
								cut_sdss=True, sdss_psf='dg', good_sdss_only=True,
								sdss_object = '82 N', rotate_cfht=False,
								nanomaggies=donm,
		nimages=nimages)

	for im in reversed(wiseims):
		tractor.images.prepend(im)
	
	pixscale = 0.187
	S = int(1.01 * sz / pixscale) / 2
	print 'Grabbing S =', S, 'subregion of CFHT coadd'
	coim = get_cfht_coadd_image(RA, DEC, S, filtermap=filtermap, doplots=False,
								nanomaggies=donm)
	tractor.images.prepend(coim)

	tractor.catalog = srcs

	tims = tractor.getImages()
	print 'Plotting outlines of', len(tims), 'images'
	plt.clf()
	for tim in tims:
		H,W = tim.shape
		twcs = tim.getWcs()
		rr,dd = [],[]
		for x,y in [(1,1),(W,1),(W,H),(1,H),(1,1)]:
			rd = twcs.pixelToPosition(x,y)
			rr.append(rd.ra)
			dd.append(rd.dec)
		if tim.name.startswith('SDSS'):
			sty = dict(color='b', lw=2, alpha=0.2, zorder=10)
		elif tim.name.startswith('WISE'):
			sty = dict(color='g', lw=2, alpha=0.8, zorder=10)
		else:
			sty = dict(color='r', lw=2, alpha=0.8, zorder=11)
		plt.plot(rr, dd, '-', **sty)
	plt.savefig('radec.png')

	if doplots:
		print 'Data plots...'
		plots(tractor, ['data'], 0, **plotsa)

	# clear unused image planes.
	for im in tractor.getImages():
		im.invvar = None
		im.starMask = None
		im.origInvvar = None
		im.mask = None

	return dict(tractor=tractor)


#def stage101(tractor=None, mp=None, **kwargs):
	

def stage100(tractor=None, mp=None, **kwargs):
	print 'Tractor cache is', tractor.cache

	allsources = tractor.getCatalog()
	allimages = tractor.getImages()

	# maglim = 24.
	#########
	#maglim = 22.
	maglim = 21.
	brightcat,Ibright = cut_bright(allsources, magcut=maglim, mag='i2')
	tractor.setCatalog(brightcat)

	print 'Cut to', len(brightcat), 'sources'

	sbands = ['u','g','r','i','z']
	allbands = ['i2'] + sbands

	# Set all bands = i2, and save those params.
	for src in brightcat:
		for b in sbands:
			br = src.getBrightness()
			setattr(br, b, br.i2)

	tractor.thawParamsRecursive('*')
	tractor.freezeParam('images')
	tractor.catalog.freezeParamsRecursive('pos', 'shape', 'shapeExp', 'shapeDev')

	print 'params0:'
	for nm in tractor.getParamNames()[:10]:
		print '  ', nm
	print '...'
	params0 = tractor.getParams()

	####

	if False:
		plotims = [0,1,2,3,4,5]
		plotsa = dict(mp=mp, imis=plotims)
		plots(tractor, ['data'], 0, **plotsa)

	cat2,I2 = cut_bright(allsources, magcut=24, mag='i2')
	tractor.setCatalog(cat2)
	plotims = [0]
	plotsa = dict(mp=mp, imis=plotims, colorbar=False, format='.pdf')
	plots(tractor, ['modbest', 'chibest'], 0, pp=np.array([tractor.getParams()]),
		  ibest=0, tsuf=': init', **plotsa)

	tractor.setCatalog(brightcat)

	pfn = 's2-006.pickle'
	(ap,i2,cat) = unpickle_from_file(pfn)

	plotsa = dict(mp=mp, colorbar=False, format='.pdf')

	for imi,im in enumerate(allimages[:6]):
		tractor.setImages(Images(im))

		#if im.name.startswith('SDSS'):
		#	band = im.photocal.bandname
		#else:
		#	band = im.photocal.band

		# LinearPhotoCal -- nanomaggies
		band = im.photocal.band

		print im
		print 'Band', band

		(ii,bb,pp) = ap[imi]
		assert(ii == imi)
		assert(bb == band)

		step = 1000 + imi*3

		# Reset params; need to thaw first though!
		tractor.catalog.thawParamsRecursive(*allbands)
		tractor.setParams(params0)

		plots(tractor, ['modbest', 'chibest'], step, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': init', **plotsa)

		# Thaw just this image's band
		tractor.catalog.freezeParamsRecursive(*allbands)
		tractor.catalog.thawParamsRecursive(band)

		assert(len(pp) == len(tractor.getParams()))
		tractor.setParams(pp)

		plots(tractor, ['modbest', 'chibest'], step+2, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': opt', **plotsa)


	sys.exit(0)
	####




def stage01(tractor=None, mp=None, **kwargs):

	#
	# Forced photometry on individual images (SDSS and CFHT coadd) using unoptimized catalog.
	#

	#cache = createCache(maxsize=10000)
	#print 'Using multiprocessing cache', cache
	#tractor.cache = cache
	#tractor.pickleCache = True

	print 'Tractor cache is', tractor.cache

	allsources = tractor.getCatalog()
	allimages = tractor.getImages()

	# maglim = 24.
	#########
	#maglim = 22.
	maglim = 21.
	brightcat,Ibright = cut_bright(allsources, magcut=maglim, mag='i2')
	tractor.setCatalog(brightcat)

	print 'Cut to', len(brightcat), 'sources'

	sbands = ['u','g','r','i','z']
	allbands = ['i2'] + sbands

	# Set all bands = i2...
	for src in brightcat:
		for b in sbands:
			br = src.getBrightness()
			setattr(br, b, br.i2)

	tractor.thawParamsRecursive('*')
	tractor.freezeParam('images')
	tractor.catalog.freezeParamsRecursive('pos', 'shape', 'shapeExp', 'shapeDev')

	# ... and save those params.
	print 'params0:'
	for nm in tractor.getParamNames()[:10]:
		print '  ', nm
	params0 = tractor.getParams()

	allp = []

	for imi,im in enumerate(allimages):
		print 'Fitting image', imi, 'of', len(allimages)
		print im.name
		tractor.setImages(Images(im))

		#if im.name.startswith('SDSS'):
		#	band = im.photocal.bandname
		#else:
		#	band = im.photocal.band

		print im
		band = im.photocal.band
		print 'Band', band

		###### !!!
		
		#if band != 'r':
		#	continue

		### Plot CMD results so far...
		i2mags = np.array([src.getBrightness().i2 for src in tractor.catalog])
		# allmags = []
		# print 'i2 mags', i2mags
		# for ii,bb,pa in allp:
		# 	print 'pa', pa
		# 
		# 	# Thaw just this image's band
		# 	tractor.catalog.freezeParamsRecursive(*allbands)
		# 	tractor.catalog.thawParamsRecursive(bb)
		# 	tractor.catalog.setParams(pa)
		# 
		# 	mags = [src.getBrightness().getMag(bb) for src in tractor.catalog]
		# 	print 'mags', mags
		# 	print len(mags)
		# 
		# 	assert(len(mags) == len(i2mags))
		# 	allmags.append(mags)
		# 
		# allmags = np.array(allmags)
		# print 'i2 mags shape', i2mags.shape
		# print 'allmags shape', allmags.shape
		# plt.figure(figsize=(6,6))
		# plt.clf()
		# #plotpos0 = [0.15, 0.15, 0.84, 0.80]
		# #plt.gca().set_position(plotpos0)
		# for i2,rr in zip(i2mags, allmags.T):
		# 	ii2 = i2.repeat(len(rr))
		# 	plt.plot(rr - ii2, ii2, 'b.')
		# 	mr = np.mean(rr)
		# 	sr = np.std(rr)
		# 	plt.plot([(mr-sr) - i2, (mr+sr) - i2], [i2,i2], 'b-', lw=3, alpha=0.25)
		# print 'Axis', plt.axis()
		# plt.axis([-3, 3, 23, 15])
		# plt.xlabel('SDSS r - CFHT i (mag)')
		# plt.ylabel('CFHT i (mag)')
		# plt.yticks([16,17,18,19,20])
		# plt.savefig('cmd-%03i.png' % imi)

		#pfn = 's1-%03i.pickle' % imi

		pfn = 's2-%03i.pickle' % imi
		pickle_to_file((allp, i2mags, tractor.catalog), pfn)
		print 'saved pickle', pfn

		# Reset params; need to thaw first though!
		tractor.catalog.thawParamsRecursive(*allbands)
		tractor.setParams(params0)

		# Thaw just this image's band
		tractor.catalog.freezeParamsRecursive(*allbands)
		tractor.catalog.thawParamsRecursive(band)

		print 'Tractor:', tractor
		print 'Active params:', len(tractor.getParamNames())
		for nm in tractor.getParamNames()[:10]:
			print '  ', nm
		print '...'

		plotims = [0,]
		plotsa = dict(imis=plotims, mp=mp)
		 
		step = 1000 + imi*3
		plots(tractor, ['modbest', 'chibest'], step, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': '+im.name+' init', **plotsa)

		optargs = dict(doplots=False, mindlnp=0.1)

		#if imi != 0:
		if True:
			optsourcestogether(tractor, step, **optargs)
			plots(tractor, ['modbest', 'chibest'], step+1,
				  pp=np.array([tractor.getParams()]),
				  ibest=0, tsuf=': '+im.name+' joint', **plotsa)

			#optsources_searchlight(tractor, im, step)

		optsourcesseparate(tractor, step, **optargs)
		# AFTER THIS CALL, ALL CATALOG PARAMS ARE FROZEN!

		tractor.catalog.thawParamsRecursive('*')
		tractor.catalog.freezeParamsRecursive('pos', 'shape', 'shapeExp', 'shapeDev')
		tractor.catalog.freezeParamsRecursive(*allbands)
		tractor.catalog.thawParamsRecursive(band)

		plots(tractor, ['modbest', 'chibest'], step+2, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': '+im.name+' indiv', **plotsa)

		p = tractor.getParams()
		print 'params', p
		#print 'Saving params:'
		#for nm,val in zip(tractor.getParamNames(), p):
		#print '  ', nm, '=', val
		allp.append((imi, band, p))

		print 'Cache stats:'
		print 'tractor:'
		tractor.cache.printStats()
		print 'galaxies:'
		get_galaxy_cache().printStats()

	tractor.setImages(allimages)
	tractor.setCatalog(allsources)

	return dict(tractor=tractor, allp=allp, Ibright=Ibright, params0=params0)



def runstage(stage, force=[], threads=1, doplots=True, opt=None):
	if threads > 1:
		if False:
			global dpool
			import debugpool
			dpool = debugpool.DebugPool(threads)
			Time.add_measurement(debugpool.DebugPoolMeas(dpool))
			mp = multiproc(pool=dpool)
		else:
			mp = multiproc(pool=multiprocessing.Pool(threads))
			
	else:
		mp = multiproc(threads)

	print 'Runstage', stage

	pfn = 'tractor%02i.pickle' % stage
	if os.path.exists(pfn):
		if stage in force:
			print 'Ignoring pickle', pfn, 'and forcing stage', stage
		else:
			print 'Reading pickle', pfn

			#multiprocessing.managers.RebuildProxy = fakeRebuildProxy
			R = unpickle_from_file(pfn)
			#multiprocessing.managers.RebuildProxy = realRebuildProxy

			return R

	if stage > 0:

		prereqs = {
			100: 0,
			
			}

		# Get prereq: from dict, or stage-1
		
		prereq = prereqs.get(stage, stage-1)

		P = runstage(prereq, force=force, threads=threads, doplots=doplots, opt=opt)

	else:
		P = {}
	print 'Running stage', stage
	F = eval('stage%02i' % stage)

	P.update(mp=mp, doplots=doplots)
	if 'tractor' in P:
		tractor = P['tractor']
		tractor.mp = mp
		tractor.modtype = np.float32

	#P.update(RA = 334.4, DEC = 0.3, sz = 2.*60.) # arcsec
	#P.update(RA = 334.4, DEC = 0.3, sz = 15.*60.) # arcsec

	# "sz" is the square side length of the ROI, in arcsec
	# This is a full SDSS frame (2048 x 2048)
	P.update(RA = 334.32, DEC = 0.315, sz = 0.24 * 3600.)

	#P.update(RA = 334.32, DEC = 0.315, sz = 0.12 * 3600.)

	#plotims = [0,1,2,3, 7,8,9]
	# CFHT, W[1234], ugriz
	plotims = [0, 1,2,3,4, 5,6,7,8,9]
	plotsa = dict(imis=plotims, mp=mp)
	P.update(plotsa=plotsa)
	P.update(opt=opt)
	
	R = F(**P)
	print 'Stage', stage, 'finished'

	if 'tractor' in R:
		try:
			tractor = R['tractor']
			tractor.pickleCache = False
		except:
			pass
	print 'Saving pickle', pfn
	pickle_to_file(R, pfn)
	print 'Saved', pfn
	return R


def kicktires():
	Tcomb = fits_table('cs82data/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	Tcomb.about()

	I = np.flatnonzero(Tcomb.chi2_psf < Tcomb.chi2_model)
	print len(I), 'point sources'
	plt.clf()
	plt.hist(Tcomb[I].mag_psf, 100, range=(15,25))
	plt.savefig('psf-mag.png')

	I = np.flatnonzero(Tcomb.chi2_psf > Tcomb.chi2_model)
	print len(I), 'galaxies'

	plt.clf()
	ha=dict(histtype='step', range=(15,40), bins=100)
	nil,nil,p1 = plt.hist(Tcomb[I].mag_spheroid, color='r', **ha)
	nil,nil,p2 = plt.hist(Tcomb[I].mag_disk, color='b', **ha)
	plt.xlabel('Mag')
	plt.legend((p1[0],p2[0]), ('spheroid', 'disk'))
	plt.savefig('gal-mag.png')

	plt.clf()
	ha=dict(histtype='step', range=(-7, -2), bins=100)
	nil,nil,p1 = plt.hist(np.log10(Tcomb[I].spheroid_reff_world), color='r', **ha)
	nil,nil,p2 = plt.hist(np.log10(Tcomb[I].disk_scale_world), color='b', **ha)
	plt.xlabel('log_10 Scale')
	plt.legend((p1[0],p2[0]), ('spheroid', 'disk'))
	plt.savefig('gal-scale.png')

	plt.clf()
	ha=dict(histtype='step', range=(0, 1), bins=100)
	nil,nil,p1 = plt.hist(Tcomb[I].spheroid_aspect_world, color='r', **ha)
	nil,nil,p2 = plt.hist(Tcomb[I].disk_aspect_world, color='b', **ha)
	plt.xlabel('Aspect ratio')
	plt.legend((p1[0],p2[0]), ('spheroid', 'disk'))
	plt.savefig('gal-aspect.png')

	T = fits_table('cs82data/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	T.about()
	maglim = 23
	psf = (T.chi2_psf < T.chi2_model) * (T.mag_psf <= maglim)
	faint = (T.mag_disk > maglim) * (T.mag_spheroid > maglim)
	Tgal = T[np.logical_not(psf) * np.logical_not(faint)]
	Texp = Tgal[Tgal.mag_disk <= maglim]
	Tdev = Tgal[Tgal.mag_spheroid <= maglim]
	plt.clf()
	plt.hist(Texp.disk_scale_world, 100)
	plt.xlabel('disk_scale')
	plt.savefig('1.png')
	plt.clf()
	plt.hist(Texp.disk_aspect_world, 100)
	plt.xlabel('disk_aspect_world')
	plt.savefig('2.png')
	plt.clf()
	plt.hist(Texp.disk_theta_world, 100)
	plt.xlabel('disk_theta_world')
	plt.savefig('3.png')
	plt.clf()
	plt.hist(Texp.mag_disk, 100)
	plt.xlabel('mag_disk')
	plt.savefig('4.png')
	plt.clf()
	plt.hist(Tdev.spheroid_reff_world, 100)
	plt.xlabel('spheroid_reff')
	plt.savefig('5.png')
	plt.clf()
	plt.hist(Tdev.spheroid_aspect_world, 100)
	plt.xlabel('spheroid_aspect_world')
	plt.savefig('6.png')
	plt.clf()
	plt.hist(Tdev.spheroid_theta_world, 100)
	plt.xlabel('spheroid_theta_world')
	plt.savefig('7.png')
	plt.clf()
	plt.hist(Tdev.mag_spheroid, 100)
	plt.xlabel('mag_spheroid')
	plt.savefig('8.png')




def checkstarmags():
	T = fits_table('cs82data/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	T.ra  = T.alpha_j2000
	T.dec = T.delta_j2000

	print 'RA', T.ra.min(), T.ra.max()
	print 'Dec', T.dec.min(), T.dec.max()

	'''select probpsf, psfmag_i, devmag_i, expmag_i, cmodelmag_i, fracDev_i, ra, dec into mydb.MyTable from photoprimary
	where
	  ra between 333.77 and 334.77
	    and dec between -0.14 and 0.91'''
	S = fits_table('MyTable_dstn.fit')

	from astrometry.libkd.spherematch import match_radec
	#from astrometry.util.plotutils import *

	I,J,d = match_radec(T.ra, T.dec, S.ra, S.dec, 1./3600.)

	#plt.clf()
	#plothist(T.ra[I] - S.ra[J], T.dec[I] - S.dec[J])
	#plt.savefig('draddec.png')

	TT = T[I]
	SS = S[J]

	print len(TT), 'matches'

	plt.clf()
	#plothist(TT.mag_psf, SS.psfmag_i)
	plt.plot(TT.mag_psf, SS.psfmag_i, 'r.')

	K = (SS.probpsf == 1)
	plt.plot(TT.mag_psf[K], SS.psfmag_i[K], 'b.')

	plt.plot([17,24],[17,24], 'k-')
	plt.axis([17, 24, 17, 24])
	plt.xlabel('CS82 mag_psf')
	plt.ylabel('SDSS psfmag_i')
	plt.savefig('psfmag.png')


	plt.clf()
	plt.plot(SS.psfmag_i, TT.mag_psf - SS.psfmag_i, 'r.')

	K = (SS.probpsf == 1)
	plt.plot(SS.psfmag_i[K], (TT.mag_psf - SS.psfmag_i)[K], 'b.')

	plt.axhline(0, color='k')
	plt.axis([17, 24, -1, 1])
	plt.xlabel('SDSS psfmag_i')
	plt.ylabel('CS82 mag_psf - SDSS psfmag_i')
	plt.savefig('psfdmag.png')

	#TT.mag_total = -2.5 * np.log10(10.**(0.4 * -TT.mag_spheroid) + 10.**(0.4 * -TT.mag_disk))

	plt.clf()
	plt.plot(TT.mag_model, SS.cmodelmag_i, 'r.')
	K = (SS.probpsf == 0)
	plt.plot(TT.mag_model[K], SS.cmodelmag_i[K], 'g.')
	plt.plot([17,24],[17,24], 'k-')
	plt.axis([17, 24, 17, 24])
	plt.xlabel('CS82 mag_model')
	plt.ylabel('SDSS cmodelmag_i')
	plt.savefig('modmag.png')
	

	
def main():
	#kicktires()
	#checkstarmags()
	#sys.exit(0)

	import optparse
	parser = optparse.OptionParser()
	parser.add_option('--threads', dest='threads', default=16, type=int, help='Use this many concurrent processors')
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')
	parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
					  help="Force re-running the given stage(s) -- don't read from pickle.")
	parser.add_option('-s', '--stage', dest='stage', default=4, type=int,
					  help="Run up to the given stage")
	parser.add_option('-P', '--no-plots', dest='plots', action='store_false', default=True, help='No plots')

	parser.add_option('--nimages', dest='nimages', type=int, help='Number of SDSS images to include')

	parser.add_option('--bands', dest='bands', type=str, help='SDSS bands')
	
	opt,args = parser.parse_args()

	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	# RA = 334.32
	# DEC = 0.315
	# # Cut to a ~ 1k x 1k subimage
	# S = 500
	# get_cfht_coadd_image(RA, DEC, S, doplots=True)

	runstage(opt.stage, opt.force, opt.threads, doplots=opt.plots, opt=opt)

if __name__ == '__main__':
	main()
	sys.exit(0)
