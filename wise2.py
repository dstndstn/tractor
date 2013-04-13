if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
import tempfile
import tractor
import pyfits
import pylab as plt
import numpy as np
import sys
from glob import glob

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

import wise

def get_l1b_file(scanid, frame, band):
	assert(band == 1)
	scangrp = scanid[-2:]
	return os.path.join('/clusterfs/riemann/raid007/bosswork/boss/wise_level1b/wise1/4band_p1bm_frm',
						scangrp, scanid, '%03i' % frame, '%s%03i-w1-int-1b.fits' % (scanid, frame))


def main():
	#ralo = 36
	#rahi = 42
	#declo = -1.25
	#dechi = 1.25
	#width = 7

	ralo = 37.5 #38.25
	rahi = 41.5 #40.75
	declo = -1.5 #-0.75
	dechi = 2.5 #1.75
	width = 2.5

	rl,rh = 39,40
	dl,dh = 0,1

	ra  = (ralo  + rahi ) / 2.
	dec = (declo + dechi) / 2.

	basedir = '/project/projectdirs/bigboss'
	wisedatadir = os.path.join(basedir, 'data', 'wise')

	wisedatadir = '/clusterfs/riemann/raid007/bosswork/boss/wise_level1b'

	wisecatdir = '/home/boss/products/NULL/wise/trunk/fits/'
	
	ifn = os.path.join(wisedatadir, 'index-allsky-astr-L1b.fits')
	T = fits_table(ifn, columns=['ra','dec','scan_id','frame_num'])
				   
	print 'Read', len(T), 'from WISE index'

	I = np.flatnonzero((T.ra > ralo) * (T.ra < rahi) * (T.dec > declo) * (T.dec < dechi))
	print len(I), 'overlap RA,Dec box'

	T = fits_table(ifn, rows=I)
	print 'Read', len(T), 'rows'

	print 'Header:', type(T.header), T.header.shape

	roipoly = np.array([(rl,dl),(rl,dh),(rh,dh),(rh,dl)])
	
	newhdr = []
	wcses = []
	corners = []

	ii = []
	
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

		#print hdrstr

		wcs = anwcs(hdrstr)
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

	plot = Plotstuff(outformat='png', ra=ra, dec=dec, width=width, size=(800,800))
	out = plot.outline
	plot.color = 'white'
	plot.alpha = 0.1
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

	# DEBUG
	cowcs = anwcs_create_box(ra, dec, 1., 2000, S)
	#cowcs = anwcs_create_box(ra, dec, 1., S, S)

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

	plot.write('wisemap.png')


	ps = PlotSequence('wise', format='%03i')
	band = 1

	plt.figure(figsize=(12,12))


	# Re-sort by distance to RA,Dec center...
	I = np.argsort(np.hypot(T.ra - ra, T.dec - dec))
	T.cut(I)

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
		fn = get_l1b_file(sid, fnum, band)
		print '-->', fn
		assert(os.path.exists(fn))

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
			fn = get_l1b_file(sid, fnum, band)
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


	print 'Creating', len(W), 'Tractor sources'
	cat = Catalog()
	for i in range(len(W)):
		w1 = W.w1mpro[i]
		nm = NanoMaggies.magToNanomaggies(w1)
		cat.append(PointSource(RaDecPos(W.ra[i], W.dec[i]), NanoMaggies(w1=nm)))

	w1psf = wise.get_psf_model(band)


	for sid,fnum in zip(T.scan_id, T.frame_num):
		print 'scan,frame', sid, fnum
		band = 1
		fn = get_l1b_file(sid, fnum, band)
		print '-->', fn
		assert(os.path.exists(fn))

		tim = wise.read_wise_level1b(fn.replace('-int-1b.fits',''),
									 nanomaggies=True, mask_gz=True, unc_gz=True,
									 sipwcs=True)
		tim.psf = w1psf

		meanerr = 1. / np.median(tim.getInvError())

		tractor = Tractor([tim], cat)
		mod = tractor.getModelImage(0, minsb= 0.1 * meanerr)

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


if __name__ == '__main__':
	import cProfile
	from datetime import datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	
