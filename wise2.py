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

from astrometry.util.sdss_radec_to_rcf import *

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


def main():
	#ralo = 36
	#rahi = 42
	#declo = -1.25
	#dechi = 1.25
	#width = 7

	#lvl = logging.INFO
	lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
	  
	ralo = 37.5 #38.25
	rahi = 41.5 #40.75
	declo = -1.5 #-0.75
	dechi = 2.5 #1.75
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
	plot.alpha = 0.05
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

	pfn = ps.getnext()
	plot.write(pfn)
	print 'Wrote', pfn

	# Re-sort by distance to RA,Dec center...
	I = np.argsort(np.hypot(T.ra - ra, T.dec - dec))
	T.cut(I)

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

	cat.freezeParamsRecursive('*')
	cat.thawPathsTo(band)

	cat0 = cat.getParams()
	br0 = [src.getBrightness().copy() for src in cat]
	nm0 = np.array([b.getBand(band) for b in br0])

	WW = W
	#X = tabledata()
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

	for imi,fn in enumerate(T.filename):
		print 'File', fn

		tim = wise.read_wise_level1b(fn.replace('-int-1b.fits',''),
									 nanomaggies=True, mask_gz=True, unc_gz=True,
									 sipwcs=True)
		tim.psf = w1psf

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
			ims0,ims1 = tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
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
		fn = 'measurements-%02i.fits' % imi
		WW.writeto(fn)
		print 'Wrote', fn

		print 'Plotting results...'
		plt.clf()
		print 'Plotting measurements...'
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


if __name__ == '__main__':
	import cProfile
	from datetime import datetime

	#cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	#sys.exit(0)
	
	T = fits_table('stripe82-19objs.fits', hdu=2)
	W = fits_table('measurements-257.fits')
	#W = fits_table('measurements-07.fits')

	print 'nm', W.nms.shape

	R,C = W.nms.shape
	#W.nms = W.nms.ravel().reshape(C,R).T   #(R,C)  # (C,R).T
	print 'nm', W.nms.shape

	plt.clf()
	print 'Plotting measurements...'

	nm0 = W.nm0
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
	plt.savefig('comp1.png')



	R = 4./3600.
	I,J,d = match_radec(T.ra, T.dec, W.ra, W.dec, R, nearest=True)
	print 'Matched', len(I)
	T.cut(I)
	W.cut(J)

	plt.clf()
	plt.loglog(T.wiseflux[:,0], W.nm0, 'r.', zorder=30)
	print 'nm', W.nms.shape
	R,C = W.nms.shape
	for j in range(R):
		nm = W.nms[j,:]
		print 'nm', nm.shape
		print 'nm0', W.nm0.shape

		print 'nm', nm
		print 'nm0', W.nm0[j]

		I = np.flatnonzero(nm != W.nm0[j])
		print 'Measured flux', j, 'in', len(I), 'images'
		plt.loglog(T.wiseflux[j,0] + np.zeros(len(I)), nm[I], 'b.', alpha=0.05, zorder=25)
	plt.savefig('comp.png')

	
