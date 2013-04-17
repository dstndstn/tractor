if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
import logging
import tempfile
import tractor
import pyfits
import pylab as plt
import numpy as np
import sys
from glob import glob
from scipy.ndimage.measurements import label,find_objects
from collections import Counter

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import match_radec
from astrometry.util.util import * #Sip, anwcs, Tan
from astrometry.blind.plotstuff import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *

#from astrometry.util.sdss_radec_to_rcf import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params
from tractor.ttime import *

import wise

def get_l1b_file(basedir, scanid, frame, band):
	assert(band == 1)
	scangrp = scanid[-2:]
	return os.path.join(basedir, 'wise1', '4band_p1bm_frm', scangrp, scanid,
						'%03i' % frame, '%s%03i-w1-int-1b.fits' % (scanid, frame))


def coadd():
	if False:
		ps.skipto(100)
	
		coadd = np.zeros((S,S))
		coaddw = np.zeros((S,S))
		conn  = np.zeros((S,S))
		connw = np.zeros((S,S))
	
		resam  = np.zeros((S,S))
		resamw = np.zeros((S,S))
		resamnn  = np.zeros((S,S))
		resamnnw = np.zeros((S,S))
	
		ii = []
		for i,(sid,fnum) in enumerate(zip(T.scan_id, T.frame_num)):
	
			if i == 5:
				break
	
			print 'scan,frame', sid, fnum
			fn = get_l1b_file(sid, fnum, band)
			print '-->', fn
			assert(os.path.exists(fn))
	
			tim = wise.read_wise_level1b(fn.replace('-int-1b.fits',''),
										 nanomaggies=True, mask_gz=True, unc_gz=True,
										 sipwcs=True)
			awcs = anwcs_new_sip(tim.wcs.wcs)
			sky = np.median(tim.getImage())
	
			im = (tim.getImage() - sky).astype(np.float32)
			L = 3
			Yo,Xo,Yi,Xi,rims = resample_with_wcs(cowcs, awcs, [im], L)
			if Yo is None:
				continue
	
			sys.exit(0)
		
			ii.append(i)
	
			w = np.median(tim.getInvvar())
	
			coadd [Yo,Xo] += rims[0] * w
			coaddw[Yo,Xo] += w
			conn  [Yo,Xo] += im[Yi,Xi]
			connw [Yo,Xo] += 1
	
			resam   [:,:] = 0
			resamw  [:,:] = 0
			resamnn [:,:] = 0
			resamnnw[:,:] = 0
			resam   [Yo,Xo] = rims[0] * w
			resamw  [Yo,Xo] = w
			resamnn [Yo,Xo] = im[Yi,Xi]
			resamnnw[Yo,Xo] = 1
			
			pyfits.writeto('resam-nn-%02i.fits' % i,    resamnn,  clobber=True)
			pyfits.writeto('resam-nn-w-%02i.fits' % i,  resamnnw, clobber=True)
			pyfits.writeto('resam-L-acc-%02i.fits' % i, resam,    clobber=True)
			pyfits.writeto('resam-L-w-%02i.fits' % i,   resamw,   clobber=True)
	
			# plt.clf()
			# plt.imshow(np.log10(np.maximum(tim.getInvvar(), w/100.)),
			# 		   interpolation='nearest', origin='lower')
			# plt.gray()
			# plt.colorbar()
			# plt.title('Weight map (log10)')
			# ps.savefig()
	
			snn = conn / np.maximum(1., connw)
			s = coadd / np.maximum(w, coaddw)
	
			ok = np.flatnonzero(connw > 0)
			pl,ph = [np.percentile(snn.flat[ok], p) for p in [10,98]]
			print 'plo,phi', pl,ph
	
			plt.clf()
			plt.imshow(snn, interpolation='nearest', origin='lower',
					   vmin=pl, vmax=ph)
			plt.gray()
			plt.colorbar()
			plt.title('Coadd (nn) of %i WISE frames' % (i+1))
			ps.savefig()
	
			plt.clf()
			plt.imshow(s, interpolation='nearest', origin='lower',
					   vmin=pl, vmax=ph)
			plt.gray()
			plt.colorbar()
			plt.title('Coadd (L) of %i WISE frames' % (i+1))
			ps.savefig()
	
			#plt.clf()
			#plt.hist(snap.ravel(), 100, range=(pl,ph))
			#ps.savefig()
	
		pyfits.writeto('coadd-nn.fits',    conn, clobber=True)
		pyfits.writeto('coadd-nn-w.fits',  connw, clobber=True)
		pyfits.writeto('coadd-L-acc.fits', coadd, clobber=True)
		pyfits.writeto('coadd-L-w.fits',   coaddw, clobber=True)
	
		sys.exit(0)
	
		co = coadd_new_from_wcs(cowcs);
		coadd_set_lanczos(co, 3);
	
		for i,(sid,fnum) in enumerate(zip(T.scan_id, T.frame_num)):
			print 'scan,frame', sid, fnum
			fn = T.filename[i]
	
			tim = wise.read_wise_level1b(fn.replace('-int-1b.fits',''),
										 nanomaggies=True, mask_gz=True, unc_gz=True,
										 sipwcs=True)
			awcs = anwcs_new_sip(tim.wcs.wcs)
			sky = np.median(tim.getImage())
			
			coadd_add_numpy(co, (tim.getImage() - sky).astype(np.float32),
							tim.getInvvar().astype(np.float32), 1., awcs)
	
			snap = coadd_get_snapshot_numpy(co, -100.)
			print 'Snapshot:', snap.min(), snap.max(), np.median(snap)
	
			ok = np.flatnonzero(snap > -100)
			pl,ph = [np.percentile(snap.flat[ok], p) for p in [10,98]]
			print 'plo,phi', pl,ph
	
			plt.clf()
			plt.imshow(snap, interpolation='nearest', origin='lower',
					   vmin=pl, vmax=ph)
			plt.gray()
			plt.colorbar()
			plt.title('Coadd of %i WISE frames' % (i+1))
			ps.savefig()
	
			plt.clf()
			plt.hist(snap.ravel(), 100, range=(pl,ph))
			ps.savefig()
	
		coadd_free(co)
	


def main(opt):
	#ralo = 36
	#rahi = 42
	#declo = -1.25
	#dechi = 1.25
	#width = 7
	  
	ralo = 37.5
	rahi = 41.5
	declo = -1.5
	dechi = 2.5
	width = 2.5

	rl,rh = 39,40
	dl,dh = 0,1
	roipoly = np.array([(rl,dl),(rl,dh),(rh,dh),(rh,dl)])

	ra  = (ralo  + rahi ) / 2.
	dec = (declo + dechi) / 2.

	ps = PlotSequence('wise', format='%03i')
	bandnum = 1
	band = 'w%i' % bandnum
	plt.figure(figsize=(12,12))

	#basedir = '/project/projectdirs/bigboss'
	#wisedatadir = os.path.join(basedir, 'data', 'wise')

	wisedatadirs = ['/clusterfs/riemann/raid007/bosswork/boss/wise_level1b',
					'/clusterfs/riemann/raid000/bosswork/boss/wise1ext']

	wisecatdir = '/home/boss/products/NULL/wise/trunk/fits/'

	ofn = 'wise-images-overlapping.fits'

	if os.path.exists(ofn):
		print 'File exists:', ofn
		T = fits_table(ofn)
		print 'Found', len(T), 'images overlapping'

		print 'Reading WCS headers...'
		wcses = []
		T.filename = [fn.strip() for fn in T.filename]
		for fn in T.filename:
			wcs = anwcs(fn, 0)
			wcses.append(wcs)

	else:
		TT = []
		for d in wisedatadirs:
			ifn = os.path.join(d, 'WISE-index-L1b.fits') #'index-allsky-astr-L1b.fits')
			T = fits_table(ifn, columns=['ra','dec','scan_id','frame_num'])
			print 'Read', len(T), 'from WISE index', ifn
			I = np.flatnonzero((T.ra > ralo) * (T.ra < rahi) * (T.dec > declo) * (T.dec < dechi))
			print len(I), 'overlap RA,Dec box'
			T.cut(I)

			fns = []
			for sid,fnum in zip(T.scan_id, T.frame_num):
				print 'scan,frame', sid, fnum
				fn = get_l1b_file(d, sid, fnum, bandnum)
				print '-->', fn
				assert(os.path.exists(fn))
				fns.append(fn)
			T.filename = np.array(fns)
			TT.append(T)
		T = merge_tables(TT)

		if False:
			T = fits_table(ifn, rows=I)
			print 'Read', len(T), 'rows'
			newhdr = []

		wcses = []
		corners = []
		ii = []
		for i in range(len(T)):

			# hdr = T.header[i]
			# hdr = [str(s) for s in hdr]
			# hdr = (['SIMPLE  =                    T',
			# 		'BITPIX  =                    8',
			# 		'NAXIS   =                    0',
			# 		] + hdr +
			# 	   ['END'])
			# hdr = [x + (' ' * (80-len(x))) for x in hdr]
			# hdrstr = ''.join(hdr)
			# newhdr.append(hdrstr)
			#print hdrstr
			#wcs = anwcs(hdrstr)

			wcs = anwcs(T.filename[i], 0)
			W,H = wcs.get_width(), wcs.get_height()
			rd = []
			for x,y in [(1,1),(1,H),(W,H),(W,1)]:
				rd.append(wcs.pixelxy2radec(x,y))
			rd = np.array(rd)

			if polygons_intersect(roipoly, rd):
				wcses.append(wcs)
				corners.append(rd)
				ii.append(i)

		print 'Found', len(wcses), 'overlapping'
		I = np.array(ii)
		T.cut(I)

		outlines = corners
		corners = np.vstack(corners)
		print 'Corners', corners.shape

		nin = sum([1 if point_in_poly(ra,dec,ol) else 0 for ol in outlines])
		print 'Number of images containing RA,Dec,', ra,dec, 'is', nin

		r0,r1 = corners[:,0].min(), corners[:,0].max()
		d0,d1 = corners[:,1].min(), corners[:,1].max()
		print 'RA,Dec extent', r0,r1, d0,d1

		T.writeto(ofn)
		print 'Wrote', ofn


	print 'Plotting map...'
	plot = Plotstuff(outformat='png', ra=ra, dec=dec, width=width, size=(800,800))
	out = plot.outline
	plot.color = 'white'
	plot.alpha = 0.07
	plot.apply_settings()

	for wcs in wcses:
		out.wcs = wcs
		out.fill = False
		plot.plot('outline')
		out.fill = True
		plot.plot('outline')

	# MAGIC 2.75: approximate pixel scale, "/pix
	S = int(3600. / 2.75)
	print 'Coadd size', S
	cowcs = anwcs_create_box(ra, dec, 1., S, S)

	plot.color = 'gray'
	plot.alpha = 1.0
	plot.lw = 1
	plot.plot_grid(1, 1, 1, 1)

	plot.color = 'red'
	plot.lw = 3
	plot.alpha = 0.75
	out.wcs = cowcs
	out.fill = False
	plot.plot('outline')

	if opt.sources:
		rd = plot.radec
		plot_radec_set_filename(rd, opt.sources)
		plot.plot('radec')

	pfn = ps.getnext()
	plot.write(pfn)
	print 'Wrote', pfn

	# Re-sort by distance to RA,Dec center...
	I = np.argsort(np.hypot(T.ra - ra, T.dec - dec))
	T.cut(I)

	# Video!
	if False:
		plot.color = 'black'
		plot.plot('fill')
		plot.color = 'white'
		plot.op = CAIRO_OPERATOR_ADD
		plot.apply_settings()
		img = plot.image
		img.image_low = 0.
		img.image_high = 1e3
		img.resample = 1
		for sid,fnum in zip(T.scan_id[I], T.frame_num[I]):
			print 'scan,frame', sid, fnum
			fn = get_l1b_file(sid, fnum, bandnum)
			print '-->', fn
			assert(os.path.exists(fn))
			#I = pyfits.open(fn)[0].data
			#print 'img min,max,median', I.min(), I.max(), np.median(I.ravel())
			img.set_wcs_file(fn, 0)
			img.set_file(fn)
			plot.plot('image')
			pfn = ps.getnext()
			plot.write(pfn)
			print 'Wrote', pfn


	if opt.sources:

		S = fits_table(opt.sources)
		print 'Read', len(S), 'sources from', opt.sources
		cat = get_tractor_sources_dr9(None, None, None, bandname='r',
									  objs=S, bands=[], nanomaggies=True,
									  extrabands=[band])
		print 'Got', len(cat), 'tractor sources'
		cat = Catalog(*cat)
		print cat
		for src in cat:
			print '  ', src

		# ??
		#WW = S
		WW = tabledata()
		
	else:
		wfn = 'wise-sources-nearby.fits'
		if os.path.exists(wfn):
			print 'Reading existing file', wfn
			W = fits_table(wfn)
			print 'Got', len(W), 'with range RA', W.ra.min(), W.ra.max(), ', Dec', W.dec.min(), W.dec.max()
		else:
			# Range of WISE slices (inclusive) containing this Dec range.
			ws0, ws1 = 26,27
			WW = []
			for w in range(ws0, ws1+1):
				fn = os.path.join(wisecatdir, 'wise-allsky-cat-part%02i-radec.fits' % w)
				print 'Searching for sources in', fn
				W = fits_table(fn)
				I = np.flatnonzero((W.ra >= r0) * (W.ra <= r1) * (W.dec >= d0) * (W.dec <= d1))
				fn = os.path.join(wisecatdir, 'wise-allsky-cat-part%02i.fits' % w)
				print 'Reading', len(I), 'rows from', fn
				W = fits_table(fn, rows=I)
				print 'Cut to', len(W), 'sources in range'
				WW.append(W)
			W = merge_tables(WW)
			del WW
			print 'Total of', len(W)
			W.writeto(wfn)
			print 'wrote', wfn
	
		# DEBUG
		W.cut((W.ra >= rl) * (W.ra <= rh) * (W.dec >= dl) * (W.dec <= dh))
		print 'Cut to', len(W), 'in the central region'
	
		print 'Creating', len(W), 'Tractor sources'
		cat = Catalog()
		for i in range(len(W)):
			w1 = W.w1mpro[i]
			nm = NanoMaggies.magToNanomaggies(w1)
			cat.append(PointSource(RaDecPos(W.ra[i], W.dec[i]), NanoMaggies(w1=nm)))

		WW = W

	cat.freezeParamsRecursive('*')
	cat.thawPathsTo(band)

	cat0 = cat.getParams()
	br0 = [src.getBrightness().copy() for src in cat]
	nm0 = np.array([b.getBand(band) for b in br0])

	WW.nm0 = nm0

	w1psf = wise.get_psf_model(bandnum)

	# Create fake image in the "coadd" footprint in order to find overlapping
	# sources.
	H,W = int(cowcs.imageh), int(cowcs.imagew)
	# MAGIC -- sigma a bit smaller than typical images (4.0-ish)
	sig = 3.5
	faketim = Image(data=np.zeros((H,W), np.float32),
					invvar=np.zeros((H,W), np.float32) + (1./sig**2),
					psf=w1psf, wcs=ConstantFitsWcs(cowcs), sky=ConstantSky(0.),
					photocal=LinearPhotoCal(1., band=band),
					name='fake')
	minsb = 0.1 * sig

	print 'Finding overlapping sources...'
	t0 = Time()
	tractor = Tractor([faketim], cat)
	groups,L = tractor.getOverlappingSources(0, minsb=minsb)
	print 'Overlapping sources took', Time()-t0
	print 'Got', len(groups), 'groups of sources'
	nl = L.max()
	gslices = find_objects(L, nl)

	# Find sources touching each group's (rectangular) ROI
	tgroups = {}
	for i,gslice in enumerate(gslices):
		gl = i+1
		tg = np.unique(L[gslice])
		tsrcs = []
		for g in tg:
			if not g in [gl,0]:
				if g in groups:
					tsrcs.extend(groups[g])
		tgroups[gl] = tsrcs

	print 'Group size histogram:'
	ng = Counter()
	for g in groups.values():
		ng[len(g)] += 1
	kk = ng.keys()
	kk.sort()
	for k in kk:
		print '  ', k, 'sources:', ng[k], 'groups'

	nms = []
	tims = []
	allrois = {}
	badrois = {}


	for imi,fn in enumerate(T.filename):
		print 'File', fn

		#if imi == 20:
		#	break

		tim = wise.read_wise_level1b(fn.replace('-int-1b.fits',''),
									 nanomaggies=True, mask_gz=True, unc_gz=True,
									 sipwcs=True)
		tim.psf = w1psf
		tims.append(tim)

		ie = tim.getInvError()
		nz = np.flatnonzero(ie)
		meanerr = 1. / np.median(ie.flat[nz])
		sig = meanerr
		print 'Sigma', sig
		tim.sig = sig

		H,W = tim.shape
		nin = 0
		for src in cat:
			x,y = tim.getWcs().positionToPixel(src.getPosition())
			if x >= 0 and y >= 0 and x < W and y < H:
				nin += 1
		print 'Number of sources inside image:', nin

		#imslices,cat1 = _fit_image(tim, cat, gslices, groups)
		#_fit_image(tim, subcat, gslice)
		
		tractor = Tractor([tim], cat)
		tractor.freezeParam('images')
		### ??
		cat.setParams(cat0)

		pgroups = 0
		pobjs = 0

		for gi in range(len(gslices)):
			gl = gi
			# note, gslices is zero-indexed
			gslice = gslices[gl]
			gl += 1
			if not gl in groups:
				print 'Group', gl, 'not in groups array; skipping'
				continue
			gsrcs = groups[gl]
			tsrcs = tgroups[gl]

			# print 'Group number', (gi+1), 'of', len(Gorder), ', id', gl, ': sources', gsrcs
			# print 'sources in groups touching slice:', tsrcs

			# Convert from 'canonical' ROI to this image.
			yl,yh = gslice[0].start, gslice[0].stop
			xl,xh = gslice[1].start, gslice[1].stop
			x0,y0 = W-1,H-1
			x1,y1 = 0,0
			for x,y in [(xl,yl),(xh-1,yl),(xh-1,yh-1),(xl,yh-1)]:
				r,d = cowcs.pixelxy2radec(x+1, y+1)
				x,y = tim.getWcs().positionToPixel(RaDecPos(r,d))
				x = int(np.round(x))
				y = int(np.round(y))

				x = np.clip(x, 0, W-1)
				y = np.clip(y, 0, H-1)
				x0 = min(x0, x)
				y0 = min(y0, y)
				x1 = max(x1, x)
				y1 = max(y1, y)
			if x1 == x0 or y1 == y0:
				print 'Gslice', gslice, 'is completely outside this image'
				continue
			
			gslice = (slice(y0,y1+1), slice(x0, x1+1))

			if np.all(tim.getInvError()[gslice] == 0):
				print 'This whole object group has invvar = 0.'

				if not gl in badrois:
					badrois[gl] = {}
				badrois[gl][imi] = gslice

				continue

			if not gl in allrois:
				allrois[gl] = {}
			allrois[gl][imi] = gslice

			if not opt.individual:
				continue

			fullcat = tractor.catalog
			subcat = Catalog(*[fullcat[i] for i in gsrcs + tsrcs])
			for i in range(len(tsrcs)):
				subcat.freezeParam(len(gsrcs) + i)
			tractor.catalog = subcat

			print len(gsrcs), 'sources unfrozen; total', len(subcat)

			pgroups += 1
			pobjs += len(gsrcs)
			
			t0 = Time()
			tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
											   rois=[gslice])
			print 'optimize_forced_photometry took', Time()-t0

			tractor.catalog = fullcat

		print 'Photometered', pgroups, 'groups containing', pobjs, 'objects'

		if False:
			mod = tractor.getModelImage(0, minsb=minsb)
			noise = np.random.normal(size=mod.shape)
			noise[tim.getInvError() == 0] = 0.
			nz = (tim.getInvError() > 0)
			noise[nz] *= (1./tim.getInvError()[nz])
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=tim.zr[0], vmax=tim.zr[1])
			imchi = dict(interpolation='nearest', origin='lower',
					   vmin=-5, vmax=5)
			plt.clf()
			plt.subplot(2,2,1)
			plt.imshow(tim.getImage(), **ima)
			plt.gray()
			plt.subplot(2,2,2)
			plt.imshow(mod, **ima)
			plt.gray()
			plt.subplot(2,2,3)
			plt.imshow((tim.getImage() - mod) * tim.getInvError(), **imchi)
			plt.gray()
			plt.subplot(2,2,4)
			plt.imshow(mod + noise, **ima)
			plt.gray()
			plt.suptitle('W1, scan %s, frame %i' % (sid, fnum))
			ps.savefig()

		cat.thawPathsTo(band)
		cat1 = cat.getParams()
		br1 = [src.getBrightness().copy() for src in cat]
		nm1 = np.array([b.getBand(band) for b in br1])
		nms.append(nm1)

		WW.nms = np.array(nms).T
		# print 'nm0', WW.nm0.shape
		# print 'nms', WW.nms.shape
		fn = opt.output % imi
		WW.writeto(fn)
		print 'Wrote', fn

		if False:
			print 'Plotting results...'
			plt.clf()
			for nm in nms:
				I = np.flatnonzero(nm != nm0)
				plt.loglog(nm0[I], np.maximum(1e-6, nm[I] / nm0[I]), 'b.', alpha=0.5)
			if False:
				nmx = np.array(nms)
				mn = []
				st = []
				ii = []
				for i,nm in enumerate(nm0):
					I = np.flatnonzero(nmx[:,i] != nm)
					if len(I) == 0:
						continue
					ii.append(i)
					mn.append(np.mean(nmx[I,i]))
					st.append(np.std (nmx[I,i]))
				I = np.array(ii)
				mn = np.array(mn)
				st = np.array(st)
				plt.loglog([nm0[I],nm0[I]], [np.maximum(1e-6, (mn-st) / nm0[I]),
											 np.maximum(1e-6, (mn+st) / nm0[I])], 'b-', alpha=0.5)
			plt.axhline(1., color='k', lw=2, alpha=0.5)
			plt.xlabel('WISE brightness (nanomaggies)')
			plt.ylabel('Tractor-measured brightness (nanomaggies)')
			plt.ylim(0.1, 10.)
			ps.savefig()


	# Simultaneous photometry

	#tractor = Tractor(tims, cat)
	#tractor.freezeParam('images')
	cat.setParams(cat0)

	for gi in range(len(gslices)):
		gl = gi
		# note, gslices is zero-indexed
		gslice = gslices[gl]
		gl += 1
		if not gl in groups:
			print 'Group', gl, 'not in groups array; skipping'
			continue
		gsrcs = groups[gl]
		tsrcs = tgroups[gl]

		if (not gl in allrois) and (not gl in badrois):
			print 'Group', gl, 'does not touch any images?'
			continue

		mytims = []
		rois = []
		if gl in allrois:
			for imi,roi in allrois[gl].items():
				mytims.append(tims[imi])
				rois.append(roi)

		mybadtims = []
		mybadrois = []
		if gl in badrois:
			for imi,roi in badrois[gl].items():
				mybadtims.append(tims[imi])
				mybadrois.append(roi)

		print 'Group', gl, 'touches', len(mytims), 'images and', len(mybadtims), 'bad ones'

		if len(mytims):
			fullcat = cat
			subcat = Catalog(*[fullcat[i] for i in gsrcs + tsrcs])
			for i in range(len(tsrcs)):
				subcat.freezeParam(len(gsrcs) + i)

			tractor = Tractor(mytims, subcat)
			tractor.freezeParam('images')

			print len(gsrcs), 'sources unfrozen; total', len(subcat)

			t0 = Time()
			ims0,ims1 = tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
														   rois=rois)
			print 'optimize_forced_photometry took', Time()-t0

		N = len(mytims)
		C = int(np.ceil(np.sqrt(N)))
		R = int(np.ceil(N / float(C)))
		plt.clf()
		#for i,(tim,roi) in enumerate(zip(mytims, rois)):
		for i,(tim,im) in enumerate(zip(mytims, ims0)):
			(img, mod, chi, roi) = im
			plt.subplot(R,C, i+1)
			plt.imshow(tim.getImage()[roi], interpolation='nearest', origin='lower',
					   vmin=tim.zr[0], vmax=tim.zr[1])
			plt.gray()
		plt.suptitle('Data')
		ps.savefig()

		#print 'ims0:', ims0
		#print 'ims1:', ims1

		plt.clf()
		for i,(tim,im) in enumerate(zip(mytims, ims0)):
			(img, mod, chi, roi) = im
			plt.subplot(R,C, i+1)
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin=tim.zr[0], vmax=tim.zr[1])
			plt.gray()
		plt.suptitle('Initial model')
		ps.savefig()

		imchi = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)

		plt.clf()
		for i,(tim,im) in enumerate(zip(mytims, ims0)):
			(img, mod, chi, roi) = im
			plt.subplot(R,C, i+1)
			plt.imshow(chi, **imchi)
			plt.gray()
		plt.suptitle('Initial chi')
		ps.savefig()

		if ims1 is not None:
			plt.clf()
			for i,(tim,im) in enumerate(zip(mytims, ims1)):
				(img, mod, chi, roi) = im
				plt.subplot(R,C, i+1)
				plt.imshow(mod, interpolation='nearest', origin='lower',
						   vmin=tim.zr[0], vmax=tim.zr[1])
				plt.gray()
			plt.suptitle('Final model')
			ps.savefig()

			plt.clf()
			for i,(tim,im) in enumerate(zip(mytims, ims1)):
				(img, mod, chi, roi) = im
				plt.subplot(R,C, i+1)
				plt.imshow(chi, **imchi)
				plt.gray()
				plt.suptitle('Final chi')
			ps.savefig()


		N = len(mybadtims)
		if N:
			C = int(np.ceil(np.sqrt(N)))
			R = int(np.ceil(N / float(C)))
			plt.clf()
			for i,(tim,roi) in enumerate(zip(mybadtims, mybadrois)):
				plt.subplot(R,C, i+1)
				plt.imshow(tim.getImage()[roi], interpolation='nearest', origin='lower',
						   vmin=tim.zr[0], vmax=tim.zr[1])
				plt.gray()
			plt.suptitle('Data in bad regions')
			ps.savefig()

			plt.clf()
			for i,(tim,roi) in enumerate(zip(mybadtims, mybadrois)):
				plt.subplot(R,C, i+1)
				plt.imshow(tim.getInvError()[roi], interpolation='nearest', origin='lower')
				plt.gray()
			plt.suptitle('Inverr in bad regions')
			ps.savefig()


		if False:
			print 'Optimizing:'
			cat.thawPathsTo('ra','dec')
			p0 = cat.getParams()
			cat.printThawedParams()
			while True:
				dlnp,X,alpha = tractor.optimize()
				print 'dlnp', dlnp
				print 'alpha', alpha
				if dlnp < 0.1:
					break
			p1 = cat.getParams()

			print 'Param changes:'
			for nm,pp0,pp1 in zip(cat.getParamNames(), p0, p1):
				print '  ', nm, pp0, 'to', pp1, '; delta', pp1-pp0

			cat.freezeParamsRecursive('ra', 'dec')
					   



	cat.thawPathsTo(band)
	cat1 = cat.getParams()
	br1 = [src.getBrightness().copy() for src in cat]
	nm1 = np.array([b.getBand(band) for b in br1])
	WW.nmall = nm1

	fn = opt.output % 999 #'measurements-all.fits'
	WW.writeto(fn)
	print 'Wrote', fn

	


if __name__ == '__main__':
	import cProfile
	from datetime import datetime

	import optparse
	parser = optparse.OptionParser('%prog [options]')
	parser.add_option('-v', dest='verbose', action='store_true')

	parser.add_option('-s', dest='sources',
					  help='Input SDSS source list')
	parser.add_option('-i', dest='individual', action='store_true',
					  help='Fit individual images?')
	parser.add_option('-o', dest='output', default='measurements-%03i.fits',
					  help='Filename pattern for outputs; default %default')


	parser.add_option('-p', dest='plots', action='store_true',
					  help='Make result plots?')
	parser.add_option('-r', dest='result',
					  help='result file to compare', default='measurements-257.fits')
	parser.add_option('-m', dest='match', action='store_true',
					  help='do RA,Dec match to compare results; else assume 1-to-1')
	
	opt,args = parser.parse_args()

	if opt.verbose:
		lvl = logging.DEBUG
	else:
		lvl = logging.INFO
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	if not opt.plots:
		profn = 'prof-%s.dat' % (datetime.now().isoformat())
		cProfile.run('main(opt)', profn)
		print 'Wrote profile to', profn
		#main(opt)
		sys.exit(0)

	ps = PlotSequence('compare')


	if False:
		psf = pyfits.open('wise-psf-w1-500-500.fits')[0].data
		print 'PSF image shape', psf.shape
		H,W = psf.shape
		X,Y = np.meshgrid(np.arange(W), np.arange(H))
		mx = np.sum(X * psf) / np.sum(psf)
		my = np.sum(Y * psf) / np.sum(psf)
		print 'First moments:', mx, my

		#S = fits_table('stripe82-19objs.fits', hdu=2)
		S = fits_table('wise-sources-nearby.fits')
		print 'Got', len(S)
		S.cut((S.w1mpro >= 14) * (S.w1mpro < 15))
		print 'Cut to', len(S)

		T = fits_table('wise-images-overlapping.fits')
		T.filename = [fn.strip() for fn in T.filename]
		for fn in T.filename:
			im = pyfits.open(fn)[0].data
			wcs = anwcs(fn, 0)
			#print 'Got WCS', wcs.getHeaderString()
			anwcs_print_stdout(wcs)

			H,W = im.shape
			m = 5

			X,Y = np.meshgrid(np.linspace(0, W, 30), np.linspace(0, H, 30))
			X = X.ravel()
			Y = Y.ravel()
			X2 = []
			Y2 = []
			for x,y in zip(X,Y):
				r,d = wcs.pixelxy2radec(x,y)
				ok,x2,y2 = wcs.radec2pixelxy(r,d)
				X2.append(x2)
				Y2.append(y2)
			X2 = np.array(X2)
			Y2 = np.array(Y2)
			print 'Round-trip error on x,y:', np.std(X2-X), np.std(Y2-Y)

			plt.clf()
			plt.plot(np.vstack((X, X + (X2 - X)*100.)),
					 np.vstack((Y, Y + (Y2 - Y)*100.)), 'b-')
			plt.plot(X, Y, 'b.')
			plt.axis('scaled')
			plt.title("WISE WCS round-trip (X,Y -> RA,Dec -> X',Y') residuals")
			plt.axis([-50, 1066, -50, 1066])
			ps.savefig()

			plt.clf()
			n,b,p1 = plt.hist(X - X2, 100, range=(-1,1), histtype='step', color='b')
			n,b,p2 = plt.hist(Y - Y2, 100, range=(-1,1), histtype='step', color='r')
			plt.legend((p1,p2), ('dx','dy'))
			plt.title('WISE WCS round-trip residuals')
			ps.savefig()


			sip = Sip(fn, 0)
			print 'Read', sip

			X3 = []
			Y3 = []
			for x,y in zip(X,Y):
				r,d = sip.pixelxy2radec(x,y)
				x2,y2 = sip.radec2pixelxy(r,d)
				X3.append(x2)
				Y3.append(y2)
			X3 = np.array(X3)
			Y3 = np.array(Y3)
			print 'Round-trip error:', np.std(X3-X), np.std(Y3-Y)

			sip_compute_inverse_polynomials(sip, 30, 30, 0, 0, 0, 0)
			print 'After computing inverse polynomials:', sip

			X4 = []
			Y4 = []
			for x,y in zip(X,Y):
				r,d = sip.pixelxy2radec(x,y)
				x2,y2 = sip.radec2pixelxy(r,d)
				X4.append(x2)
				Y4.append(y2)
			X4 = np.array(X4)
			Y4 = np.array(Y4)
			print 'Round-trip error on x,y with 4th-order:', np.std(X4-X), np.std(Y4-Y)


			plt.clf()
			plt.plot(np.vstack((X, X + (X4 - X)*1e6)),
					 np.vstack((Y, Y + (Y4 - Y)*1e6)), 'b-')
			plt.plot(X, Y, 'b.')
			plt.axis('scaled')
			plt.title("Refit WISE WCS round-trip (X,Y -> RA,Dec -> X',Y') residuals")
			plt.axis([-50, 1066, -50, 1066])
			ps.savefig()

			plt.clf()
			n,b,p1 = plt.hist(X - X4, 100, range=(-1e-3,1e-3), histtype='step', color='b')
			n,b,p2 = plt.hist(Y - Y4, 100, range=(-1e-3,1e-3), histtype='step', color='r')
			plt.legend((p1,p2), ('dx','dy'))
			plt.title('Refit WISE WCS round-trip residuals')
			ps.savefig()


			# sip.ap_order = 5
			# sip.bp_order = 5
			# sip_compute_inverse_polynomials(sip, 30, 30, 0, 0, 0, 0)
			# print 'After computing 5th-order  inverse polynomials:', sip
			# X5 = []
			# Y5 = []
			# for x,y in zip(X,Y):
			# 	r,d = sip.pixelxy2radec(x,y)
			# 	x2,y2 = sip.radec2pixelxy(r,d)
			# 	X5.append(x2)
			# 	Y5.append(y2)
			# X5 = np.array(X5)
			# Y5 = np.array(Y5)
			# print 'Round-trip error on x,y with 5th-order:', np.std(X5-X), np.std(Y5-Y)




			xx,yy = [],[]
			for r,d in zip(S.ra, S.dec):
				ok,x,y = wcs.radec2pixelxy(r, d)
				#if x >= 0 and y >= 0 and x < H and y < W:
				if x >= m and y >= m and y < H-m and x < W-m:
					xx.append(x-1)
					yy.append(y-1)

			plt.clf()
			for i,(x,y) in enumerate(zip(xx,yy)):
				if i == 16:
					break
				ix = int(np.round(x))
				iy = int(np.round(y))
				plt.subplot(4,4,i+1)
				plt.imshow(im[iy-m:iy+m+1, ix-m:ix+m+1], interpolation='nearest', origin='lower',
						   vmin=15., vmax=75.)
				plt.gray()
				ax = plt.axis()
				plt.plot(x - (ix-m), y - (iy - m), 'r+', mec='r', mfc='none', mew=2)
				# ms=15, mew=2, alpha=0.6)
				plt.axis(ax)

			tt = fn.split('/')[-1].replace('-int-1b.fits', '')
			plt.suptitle('WISE ' + tt)
			ps.savefig()



			xx,yy = [],[]
			for r,d in zip(S.ra, S.dec):
				x,y = sip.radec2pixelxy(r, d)
				#if x >= 0 and y >= 0 and x < H and y < W:
				if x >= m and y >= m and y < H-m and x < W-m:
					xx.append(x-1)
					yy.append(y-1)

			plt.clf()
			for i,(x,y) in enumerate(zip(xx,yy)):
				if i == 16:
					break
				ix = int(np.round(x))
				iy = int(np.round(y))
				plt.subplot(4,4,i+1)
				plt.imshow(im[iy-m:iy+m+1, ix-m:ix+m+1], interpolation='nearest', origin='lower',
						   vmin=15., vmax=75.)
				plt.gray()
				ax = plt.axis()
				plt.plot(x - (ix-m), y - (iy - m), 'r+', mec='r', mfc='none', mew=2)
				# ms=15, mew=2, alpha=0.6)
				plt.axis(ax)

			tt = fn.split('/')[-1].replace('-int-1b.fits', '')
			plt.suptitle('WISE ' + tt + ' (re-fit)')
			ps.savefig()


			break

	T = fits_table('stripe82-19objs.fits', hdu=2)
	print 'Reading results file', opt.result
	W = fits_table(opt.result)

	print 'nm', W.nms.shape

	print 'Plotting measurements...'
	plt.clf()
	nm0 = W.nm0
	R,C = W.nms.shape
	for j in range(C):
		nm = W.nms[:,j]
		I = np.flatnonzero(nm != nm0)
		plt.loglog(nm0[I], np.maximum(1e-6, nm[I] / nm0[I]), 'b.', alpha=0.01)
	if False:
		nmx = W.nms.T
		mn = []
		st = []
		ii = []
		for i,nm in enumerate(nm0):
			I = np.flatnonzero(nmx[:,i] != nm)
			if len(I) == 0:
				continue
			ii.append(i)
			mn.append(np.mean(nmx[I,i]))
			st.append(np.std (nmx[I,i]))
		I = np.array(ii)
		mn = np.array(mn)
		st = np.array(st)
		plt.loglog([nm0[I],nm0[I]], [np.maximum(1e-6, (mn-st) / nm0[I]),
									 np.maximum(1e-6, (mn+st) / nm0[I])], 'b-', alpha=0.5)
	plt.axhline(1., color='k', lw=2, alpha=0.5)
	plt.xlabel('WISE brightness (nanomaggies)')
	plt.ylabel('Tractor-measured brightness (nanomaggies)')
	plt.ylim(0.1, 10.)
	ps.savefig()

	if opt.match:
		R = 4./3600.
		I,J,d = match_radec(T.ra, T.dec, W.ra, W.dec, R, nearest=True)
		print 'Matched', len(I)
		T.cut(I)
		W.cut(J)
	else:
		assert(len(T) == len(W))

	plt.clf()
	p1 = plt.loglog(T.wiseflux[:,0], W.nm0, 'r.', zorder=30)
	if 'nmall' in W.get_columns():
		p2 = plt.loglog(T.wiseflux[:,0], W.nmall, 'mx', zorder=30)
	R,C = W.nms.shape
	mns = []
	sts = []
	mx = []
	for j in range(R):
		nm = W.nms[j,:]
		I = np.flatnonzero(nm != W.nm0[j])
		print 'Measured flux', j, 'in', len(I), 'images'
		if len(I) == 0:
			continue
		mns.append(np.mean(nm[I]))
		sts.append(np.std(nm[I]))
		mx.append(T.wiseflux[j,0])
		p3 = plt.loglog(T.wiseflux[j,0] + np.zeros(len(I)), nm[I], 'b.', alpha=0.5, zorder=25)
	if len(mx):
		p4 = plt.errorbar(mx, mns, yerr=sts, fmt='o', mec='b', mfc='none')

	ax = plt.axis()
	lo,hi = min(ax[0],ax[2]), max(ax[1],ax[3])
	plt.plot([lo,hi], [lo,hi], 'k-', lw=3, alpha=0.3)
	plt.axis(ax)

	plt.xlabel("Schlegel's measurements (nanomaggies)")
	plt.ylabel("My measurements (nanomaggies)")

	ps.savefig()


	plt.clf()
	rband = W.nm0
	if 'nmall' in W.get_columns():
		simul = W.nmall
		I = np.flatnonzero(simul != rband)
		xx = T.wiseflux[I,0]
		simul = simul[I]
		# p1 = plt.loglog(xx, W.nm0   / xx, 'r.', zorder=30)
		p2 = plt.loglog(xx, simul / xx, 'b.', zorder=30)
		sig = 1./np.sqrt(T.wiseflux_ivar[I,0])
		p5 = plt.loglog([xx-sig, xx+sig], [simul / xx]*2, 'b-', zorder=29)

	R,C = W.nms.shape
	mns = []
	sts = []
	mx = []
	for j in range(R):
		nm = W.nms[j,:]
		I = np.flatnonzero(nm != W.nm0[j])
		print 'Measured flux', j, 'in', len(I), 'images'
		if len(I) == 0:
			continue
		mns.append(np.mean(nm[I]))
		sts.append(np.std(nm[I]))
		mx.append(T.wiseflux[j,0])
		xx = T.wiseflux[j,0] + np.zeros(len(I))
		p3 = plt.loglog(xx, nm[I] / xx, 'b.', alpha=0.5, zorder=25)
	if len(mx):
		mns = np.array(mns)
		mx = np.array(mx)
		sts = np.array(sts)
		p4 = plt.errorbar(mx, mns / mx, yerr = sts / mx, fmt='o', mec='b', mfc='none')

	plt.axhline(1., color='k', lw=3, alpha=0.3)
	plt.ylim(0.1, 10.)
	plt.xlabel("Schlegel's measurements (nanomaggies)")
	plt.ylabel("My measurements / Schlegel's")
	ps.savefig()


	
