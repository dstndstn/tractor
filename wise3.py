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
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import match_radec, cluster_radec
from astrometry.util.util import * #Sip, anwcs, Tan
from astrometry.blind.plotstuff import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *
from astrometry.util.stages import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params
from tractor.ttime import *

import wise

def get_l1b_file(basedir, scanid, frame, band):
	scangrp = scanid[-2:]
	return os.path.join(basedir, 'wise%i' % band, '4band_p1bm_frm', scangrp, scanid,
						'%03i' % frame, '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))



#
#   stage0 --
#    - find overlapping images
#    - find SDSS sources nearby
#    - create Tractor objects for each cluster
#
#   stage1 --
#    - run forced photometry, make plots
#
#   stage2 --
#    - make comparison plots
#



def stage0(opt=None, ps=None, **kwa):
	bandnum = 1
	band = 'w%i' % bandnum

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

		wcses = []
		corners = []
		ii = []
		for i in range(len(T)):
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

		nin = sum([1 if point_in_poly(ra,dec,ol) else 0 for ol in outlines])
		print 'Number of images containing RA,Dec,', ra,dec, 'is', nin

		r0,r1 = corners[:,0].min(), corners[:,0].max()
		d0,d1 = corners[:,1].min(), corners[:,1].max()
		print 'RA,Dec extent', r0,r1, d0,d1

		T.writeto(ofn)
		print 'Wrote', ofn


	# Look at a radius this big, in arcsec, around each source position.
	# 15" = about 6 WISE pixels
	Wrad = opt.wrad / 3600.

	# Look for SDSS objects within this radius; Wrad + a margin
	if opt.srad == 0.:
		Srad = Wrad + 5./3600.
	else:
		Srad = opt.srad / 3600.

	S = fits_table(opt.sources)
	print 'Read', len(S), 'sources from', opt.sources

	groups,singles = cluster_radec(S.ra, S.dec, Wrad, singles=True)
	print 'Source clusters:', groups
	print 'Singletons:', singles

	tractors = []

	sdss = DR9(basedir='data-dr9')
	sband = 'r'

	for i in singles:
		r,d = S.ra[i],S.dec[i]
		print 'Source', i, 'at', r,d
		fn = sdss.retrieve('photoObj', S.run[i], S.camcol[i], S.field[i], band=sband)
		print 'Reading', fn
		oo = fits_table(fn)
		print 'Got', len(oo)
		cat1,obj1,I = get_tractor_sources_dr9(None, None, None, bandname=sband,
											  objs=oo, radecrad=(r,d,Srad), bands=[],
											  nanomaggies=True, extrabands=[band],
											  fixedComposites=True,
											  getobjs=True, getobjinds=True)
		print 'Got', len(cat1), 'SDSS sources nearby'

		# Find images that overlap?

		ims = []
		for j,wcs in enumerate(wcses):

			print 'Filename', T.filename[j]
			ok,x,y = wcs.radec2pixelxy(r,d)
			print 'WCS', j, '-> x,y:', x,y

			if not anwcs_radec_is_inside_image(wcs, r, d):
				continue

			tim = wise.read_wise_level1b(
				T.filename[j].replace('-int-1b.fits',''),
				nanomaggies=True, mask_gz=True, unc_gz=True,
				sipwcs=True, constantInvvar=True, radecrad=(r,d,Wrad))
			ims.append(tim)
		print 'Found', len(ims), 'images containing this source'

		tr = Tractor(ims, cat1)
		tractors.append(tr)
		

	if len(groups):
		# TODO!
		assert(False)

	return dict(tractors=tractors, sources=S, bandnum=bandnum, band=band,
				opt0=opt)



def _plot_grid(ims, kwas):
	N = len(ims)
	C = int(np.ceil(np.sqrt(N)))
	R = int(np.ceil(N / float(C)))
	plt.clf()
	for i,(im,kwa) in enumerate(zip(ims, kwas)):
		plt.subplot(R,C, i+1)
		#print 'plotting grid cell', i, 'img shape', im.shape
		plt.imshow(im, **kwa)
		plt.gray()
		plt.xticks([]); plt.yticks([])
	return R,C

def _plot_grid2(ims, cat, tims, kwas, ptype='mod'):
	xys = []
	stamps = []
	for (img,mod,ie,chi,roi),tim in zip(ims, tims):
		if ptype == 'mod':
			stamps.append(mod)
		elif ptype == 'chi':
			stamps.append(chi)
		wcs = tim.getWcs()
		if roi is None:
			y0,x0 = 0,0
		else:
			y0,x0 = roi[0].start, roi[1].start
		xy = []
		for src in cat:
			xi,yi = wcs.positionToPixel(src.getPosition())
			xy.append((xi - x0, yi - y0))
		xys.append(xy)
		#print 'X,Y source positions in stamp of shape', stamps[-1].shape
		#print '  ', xy
	R,C = _plot_grid(stamps, kwas)
	for i,xy in enumerate(xys):
		if len(xy) == 0:
			continue
		plt.subplot(R, C, i+1)
		ax = plt.axis()
		xy = np.array(xy)
		plt.plot(xy[:,0], xy[:,1], 'r+', lw=2)
		plt.axis(ax)


def _stage1fit((tractor, ti, minsb, ocat, minFlux)):
	tims = tractor.images

	print 'Optimize_forced_photometry:'
	tractor.printThawedParams()

	## ASSUME LinearPhotoCal here -- convert minFlux to nmgy
	if minFlux is not None:
		minFlux = -np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val for tim in tims])
		print 'minFlux:', minFlux, 'nmgy'
		
	ims0,ims1 = tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
												   sky=True, minFlux=minFlux)

	print 'After optimize_forced_photometry 1:'
	tractor.printThawedParams()

	# HACK!
	# Re-run to ensure we minimized chisq...
	tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
									   sky=True, minFlux=minFlux)
	print 'After optimize_forced_photometry 2:'
	tractor.printThawedParams()

	p1 = tractor.getParams()

	ims3 = None
	if ocat:
		cat = tractor.catalog
		tractor.catalog = ocat
		tractor.freezeParam('images')
		nil,nil,ims3 = tractor.optimize_forced_photometry(minsb=minsb,
														  justims0=True)
		tractor.catalog = cat

	return p1,ims0,ims1,ims3

def stage1(opt=None, ps=None, tractors=None, band=None, bandnum=None, **kwa):

	minsb = opt.minsb

	ocat = None
	if opt.osources:
		O = fits_table(opt.osources)
		ocat = Catalog()
		print 'Other catalog:'
		for i in range(len(O)):
			w1 = O.wiseflux[i, 0]
			s = PointSource(RaDecPos(O.ra[i], O.dec[i]), NanoMaggies(w1=w1))
			ocat.append(s)
		print ocat
		ocat.freezeParamsRecursive('*')
		ocat.thawPathsTo(band)

	#w1psf = wise.get_psf_model(bandnum, opt.pixpsf)

	args = []

	print 'Got', len(tractors), 'tractors'
	for ti,tractor in enumerate(tractors):
		print '  ', tractor

		tims = tractor.images
		cat = tractor.catalog

		for tim in tims:
			x0,y0 = tim.getWcs().getX0Y0()
			h,w = tim.shape
			#print 'Image bounds:', x0,y0, '+', w,h
			#tim.psf = w1psf
			tim.psf = wise.get_psf_model(bandnum, opt.pixpsf, xy=(x0+w/2, y0+h/2),
										 positive=False)

		for tim in tims:
			if opt.constInvvar:
				tim.setInvvar(tim.cinvvar)
			else:
				tim.setInvvar(tim.vinvvar)
										 
		if opt.ptsrc:
			print 'Converting all sources to PointSources'
			pcat = Catalog()
			for src in cat:
				pt = PointSource(src.getPosition(), src.getBrightness())
				pcat.append(pt)
			print 'PointSource catalog:', pcat
			cat = pcat
			tractor.catalog = cat

		# #tractor.freezeParam('images')
		# tims.freezeParamsRecursive('*')
		# tims.thawAllParams()
		# for tim in tims:
		# 	tim.thawParam('sky')
		# 	# FIXME -- ConstantSky is a ScalarParams, with no thawAllRecursive() call.
		# 	#tim.getSky().thawAllRecursive()

		tims.freezeParamsRecursive('*')
		tims.thawPathsTo('sky')

		cat.freezeParamsRecursive('*')
		cat.thawPathsTo(band)

		args.append((tractor, ti, minsb, ocat, opt.minflux))

	res = mp.map(_stage1fit, args)

	for ti,((p1,ims0,ims1,ims3),tractor) in enumerate(zip(res, tractors)):
		tims = tractor.images
		cat = tractor.catalog

		tims.freezeParamsRecursive('*')
		tims.thawPathsTo('sky')

		tractor.setParams(p1)

		imas = [dict(interpolation='nearest', origin='lower',
					 vmin=tim.zr[0], vmax=tim.zr[1])
				for tim in tims]
		imchi = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)
		imchis = [imchi] * len(tims)

		tt = 'source %i' % ti

		_plot_grid([img for (img, mod, chi, roi) in ims0], imas)
		plt.suptitle('Data: ' + tt)
		ps.savefig()

		if ims1 is not None:
			_plot_grid2(ims1, cat, tims, imas)
			plt.suptitle('Forced-phot model: ' + tt)
			ps.savefig()

			_plot_grid2(ims1, cat, tims, imchis, ptype='chi')
			plt.suptitle('Forced-phot chi: ' + tt)
			ps.savefig()

		if opt.osources:
			_plot_grid2(ims3, ocat, tims, imas)
			plt.suptitle("Schlegel's model: " + tt)
			ps.savefig()

			#_plot_grid2(ims3, ocat, tims, imchis, ptype='chi')
			#plt.suptitle("Schlegel's chi: " + tt)
			#ps.savefig()

	return dict(opt1=opt)

def stage2(opt=None, ps=None, tractors=None, band=None, **kwa):

	assert(opt.osources)
	O = fits_table(opt.osources)

	W = fits_table('wise-sources-nearby.fits', columns=['ra','dec','w1mpro'])
	print 'Read', len(W), 'WISE sources nearby'

	zpoff = 0.2520
	fscale = 10. ** (zpoff / 2.5)
	print 'Flux scale', fscale
		
	nms = []
	rr,dd = [],[]
	print 'Got', len(tractors), 'tractors'
	for ti,tractor in enumerate(tractors):
		#print '  ', tractor
		cat = tractor.catalog
		nm = np.array([src.getBrightness().getBand(band) for src in cat])
		#print 'My fluxes:', nm
		nm *= fscale
		#print 'Scaled:', nm
		nms.append(nm)
		rr.append(np.array([src.getPosition().ra  for src in cat]))
		dd.append(np.array([src.getPosition().dec for src in cat]))

		if ti == 0:
			for r,d,f in zip(rr[-1],dd[-1],nm):
				print 'Flux at RA=%16.6f DEC=%16.7f = %15.5f' % (r,d,f)
			for tim in tractor.images:
				print 'Sky in %s = %15f' % (tim.name, tim.getSky().val / tim.getPhotoCal().val * fscale)

			for tim in tractor.images:
				I = np.flatnonzero(tim.maskplane & sum([1 << b for b in 8,19,20,22,23,24,25,29,30,31]))
				print 'Mask plane has', len(I), 'pixels with reserved bits set'
				if len(I):
					for b in [8,19,20,22,23,24,25,29,30,31]:
						I = np.flatnonzero(tim.maskplane & (1 << b))
						if len(I):
							print '  ', len(I), 'pixels have bit', b, 'set'



			djs_srcs = np.array([
				(39.277403,  0.67364460,   979.52605 ,        18.776073),
				(39.277793,  0.67658440,   1509.4260 ,        19.783079),
				(39.272547,  0.67414207,   12.962583 ,        16.930656),
				(39.282410,  0.67476651,   272.42451 ,        17.146239),
				(39.282606,  0.67672310,   15.386274 ,        16.532081),
				(39.272069,  0.66805896,   19.962492 ,        16.689978),
				(39.270089,  0.67791569,   20.861612 ,        16.527072),
				(39.269417,  0.66896378,   21.421076 ,        17.288091),
				(39.286591,  0.67643173,   31.950005 ,        16.576358),
				(39.277809,  0.66381082,  -54.976775 ,        16.332439),
				(39.266579,  0.66889227,   759.26272 ,        60.572543),
				])

			djs_skies = [
				('01154a128-w1-int-1b',   113.89485),
				('01158a128-w1-int-1b',   119.40917),
				('01158a129-w1-int-1b',   121.52171),
				('01162a128-w1-int-1b',   119.59323),
				('01165a104-w1-int-1b',   124.99107),
				('01166b128-w1-int-1b',   123.17599),
				('01169a105-w1-int-1b',   132.69438),
				('01170a128-w1-int-1b',   156.90454),
				('01173a104-w1-int-1b',   150.64179),
				('01177a104-w1-int-1b',   129.21403),
				('01181a104-w1-int-1b',   126.20651),
				('01181a105-w1-int-1b',   133.51273),
				('01226b131-w1-int-1b',   167.09509),
				('01229a107-w1-int-1b',   168.54541),
				('01233a107-w1-int-1b',   170.66543),
				('01237a106-w1-int-1b',   171.99296),
				('01241a106-w1-int-1b',   176.90949),
				('06849b175-w1-int-1b',   116.29544),
				('06853b175-w1-int-1b',   130.32421),
				('06857b175-w1-int-1b',   128.34128),
				('06860a150-w1-int-1b',   139.65082),
				('06861b175-w1-int-1b',   121.10086),
				('06864a150-w1-int-1b',   128.39598),
				('06865b175-w1-int-1b',   128.92038),
				('06868a150-w1-int-1b',   132.56495),
				('06869b175-w1-int-1b',   130.77367),
				('06872a150-w1-int-1b',   125.48015),
				('06876a150-w1-int-1b',   130.32865),
				('06880a150-w1-int-1b',   127.63005),
				('07008a147-w1-int-1b',   140.53451),
				('12214a131-w1-int-1b',   95.726806),
				('12218a131-w1-int-1b',   85.909363),
				('12222a130-w1-int-1b',   92.525802),
				('12225a107-w1-int-1b',   89.126797),
				('12226a131-w1-int-1b',   91.563038),
				('12229a106-w1-int-1b',   95.609643),
				('12230a131-w1-int-1b',   91.266718),
				('12233a107-w1-int-1b',   93.705763),
				('12234a130-w1-int-1b',   92.043619),
				('12237a107-w1-int-1b',   93.719181),
				('12241a106-w1-int-1b',   90.128343),
				('12245a107-w1-int-1b',   92.570261),
				]

			ra = djs_srcs[:,0]
			dec = djs_srcs[:,1]
			r = rr[-1]
			d = dd[-1]

			I,J,d = match_radec(ra, dec, r, d, 1./3600.)
			plt.clf()
			plt.plot(djs_srcs[I, 2], nm[J], 'b.')
			plt.xlabel('DJS flux (nmgy)')
			plt.ylabel('DL flux (nmgy)')
			plt.title('src 0 drill-down')
			ax = plt.axis()
			lo,hi = min(ax[0],ax[2]), max(ax[1],ax[3])
			plt.plot([lo,hi], [lo,hi], 'k-', lw=3, alpha=0.3)
			plt.axis([lo,hi,lo,hi])
			ps.savefig()

			tims = tractor.images
			
			mynames = [tim.name.replace('WISE ','').replace(' W1','')
					   for tim in tims]
			mysky = [tim.getSky().val / tim.getPhotoCal().val * fscale
					 for tim in tims]
			djsnames = [nm.replace('-w1-int-1b','') for nm,s in djs_skies]
			djssky = np.array([s for nm,s in djs_skies])
			djsskies = dict(zip(djsnames, djssky))

			plt.clf()
			for nm,s in zip(mynames, mysky):
				if not nm in djsskies:
					print 'Not found in DJS sky:', nm
					continue
				plt.plot(djsskies[nm], s, 'b.')
			plt.xlabel('DJS sky flux (nmgy/pix)')
			plt.ylabel('DL sky flux (nmgy/pix)')
			plt.title('src 0 drill-down: sky')
			ax = plt.axis()
			lo,hi = min(ax[0],ax[2]), max(ax[1],ax[3])
			plt.plot([lo,hi], [lo,hi], 'k-', lw=3, alpha=0.3)
			plt.axis([lo,hi,lo,hi])
			ps.savefig()


					

	print 'My flux measurements:'
	for nm in nms:
		print nm
	

	X = O.wiseflux[:,0]
	DX = 1./np.sqrt(O.wiseflux_ivar[:,0])




	plt.clf()
	for ti,(nm,r,d) in enumerate(zip(nms,rr,dd)):
		x = X[ti]
		xx = [x]*len(nm)
		p1 = plt.loglog(xx, nm, 'b.', zorder=32)

		plt.plot([x,x], [nm[nm>0].min(), nm.max()], 'b--', alpha=0.25, zorder=25)

		R = 4./3600.
		I,J,d = match_radec(O.ra[ti], O.dec[ti], r, d, R)
		p2 = plt.loglog([x]*len(J), nm[J], 'bo', zorder=28)

	I,J,d = match_radec(O.ra, O.dec, W.ra, W.dec, R)
	wf = NanoMaggies.magToNanomaggies(W.w1mpro[J]) * fscale
	p3 = plt.loglog(X[I], wf, 'rx', mew=1.5, ms=6, zorder=30)
	#p3 = plt.loglog(X[I], wf, 'r.', ms=8, zorder=30)

	nil,nil,p4 = plt.errorbar(X, X, yerr=DX, fmt=None, color='k', alpha=0.5, ecolor='0.5',
							  lw=2, capsize=10)

	#plt.loglog(X, X/fscale, 'k-', alpha=0.1)
	
	ax = plt.axis()
	lo,hi = min(ax[0],ax[2]), max(ax[1],ax[3])
	plt.plot([lo,hi], [lo,hi], 'k-', lw=3, alpha=0.3)

	J = np.argsort(X)
	for j,i in enumerate(J):
		#for i,x in enumerate(X):
		x = X[i]
		if x > 0:
			y = ax[2]*(3 if ((j%2)==0) else 5)
			plt.text(x, y, '%i' % i, color='k', fontsize=8, ha='center')
			plt.plot([x,x], [x*0.1, y*1.1], 'k-', alpha=0.1)
	plt.axis(ax)

	plt.xlabel("Schlegel's measurements (nanomaggies)")
	plt.ylabel("My measurements (nanomaggies)")

	plt.legend((p1, p2, p3, p4), ('Mine (all)', 'Mine (nearest)', 'WISE', 'Schlegel'),
			   loc='upper left')

	ps.savefig()




	## Plot relative to Schlegel's measurements = 1

	plt.clf()
	for ti,(nm,r,d) in enumerate(zip(nms,rr,dd)):
		x = X[ti]
		xx = np.array([x]*len(nm))
		# All sources
		p1 = plt.loglog(xx, nm / xx, 'b.', zorder=32)
		# Line connecting my sources
		plt.plot([x,x], [nm[nm>0].min()/x, nm.max()/x], 'b--', alpha=0.25, zorder=25)
		# My sources within R
		R = 4./3600.
		I,J,d = match_radec(O.ra[ti], O.dec[ti], r, d, R)
		xx = np.array([x]*len(J))
		p2 = plt.loglog(xx, nm[J] / xx, 'bo', zorder=28)
	# WISE sources
	I,J,d = match_radec(O.ra, O.dec, W.ra, W.dec, R)
	wf = NanoMaggies.magToNanomaggies(W.w1mpro[J]) * fscale
	p3 = plt.loglog(X[I], wf /X[I], 'rx', mew=1.5, ms=6, zorder=30)
	# Schlegel errorbars
	nil,nil,p4 = plt.errorbar(X, np.ones_like(X), yerr=DX/X, fmt=None, color='k',
							  alpha=0.5, ecolor='0.5', lw=2, capsize=10)
	ax = plt.axis()
	# Horizontal line
	lo,hi = min(ax[0],ax[2]), max(ax[1],ax[3])
	plt.plot([lo,hi], [1., 1.], 'k-', lw=2, alpha=0.3)

	# Label sources
	J = np.argsort(X)
	for j,i in enumerate(J):
		x = X[i]
		if x > 0:
			y = ax[2]*(3 if ((j%2)==0) else 5)
			plt.text(x, y, '%i' % i, color='k', fontsize=8, ha='center')
			plt.plot([x,x], [0.1, 1.], 'k-', alpha=0.1)
	plt.axis(ax)

	plt.xlabel("Schlegel's measurements (nanomaggies)")
	plt.ylabel("My measurements / Schlegel's")
	plt.legend((p1, p2, p3, p4), ('Mine (all)', 'Mine (nearest)', 'WISE', 'Schlegel'),
			   loc='upper left')
	ps.savefig()

	# Label again
	for j,i in enumerate(J):
		x = X[i]
		if x > 0:
			y = (0.55 if ((j%2)==0) else 0.6)
			plt.text(x, y, '%i' % i, color='k', fontsize=8, ha='center')
			plt.plot([x,x], [y, 1.], 'k-', alpha=0.1)
	plt.axis([ax[0],ax[1], 0.5, 2.0])
	ps.savefig()



	return dict(opt2=opt)




# Individual-image fits.

def stage3(opt=None, ps=None, tractors=None, band=None, **kwa):
	minsb = opt.minsb
	minFlux = opt.minflux

	for ti,tractor in enumerate(tractors):
		print '  ', tractor
		tims = tractor.images
		cat = tractor.catalog

		if ti != 10:
			continue
		for tim in tims:
			print 'scale', tim.getPhotoCal().val

		tractor.thawParam('images')

		cat.freezeParamsRecursive('*')
		cat.thawPathsTo(band)

		names = []
		skies = []
		ras = []
		decs = []
		fluxes = []

		for ii,tim in enumerate(tims):
			tractor.images = Images(tim)
			tim.freezeParamsRecursive('*')
			tim.thawPathsTo('sky')

			ims0,ims1 = tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
														   sky=True, minFlux=minFlux)

			#tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
			#								   sky=True, minFlux=minFlux)
			#nm = np.array([src.getBrightness().getBand(band) for src in cat])
			#fluxes.append(nm)

			zpoff = 0.2520
			fscale = 10. ** (zpoff / 2.5)

			print 'Fscale', fscale

			print
			print 'Image', ii, tim.name
			x0,y0 = tim.getWcs().getX0Y0()
			h,w = tim.shape
			print 'Image bounds:', x0,y0, '+', w,h
			sky = (tim.getSky().val / tim.getPhotoCal().val * fscale)
			print 'Sky:', sky

			rdf = []
			for si,src in enumerate(cat):
				pos = src.getPosition()
				f = src.getBrightness().getBand(band)
				print 'RA,Dec (%10.6f, %10.6f), Flux %12.6f' % (pos.ra, pos.dec, f * fscale)
				rdf.append((pos.ra,pos.dec,f*fscale))

				sk = tim.getSky().val
				plt.clf()
				mod = tractor.getModelImage(tim, [src], minsb=minsb, sky=False)
				plt.subplot(1,2,1)
				plt.imshow(mod, interpolation='nearest', origin='lower',
						   vmin=tim.zr[0]-sk, vmax=tim.zr[1]-sk)
				plt.subplot(1,2,2)
				plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
						   vmin=tim.zr[0], vmax=tim.zr[1])
				ps.savefig()

				umodp = src.getUnitFluxModelPatch(tim)
				umod = np.zeros_like(mod)
				umodp.addTo(umod)
				
				pyfits.writeto('source10-im%02i-mod%02i.fits' % (ii,si), mod, clobber=True)
				pyfits.writeto('source10-im%02i-umod%02i.fits' % (ii,si), umod, clobber=True)

			pyfits.writeto('source10-im%02i-data.fits' % ii, tim.getImage(), clobber=True)
			pyfits.writeto('source10-im%02i-invvar.fits' % ii, tim.getInvvar(), clobber=True)

			rdf = np.array(rdf)
			ras.append(rdf[:,0])
			decs.append(rdf[:,1])
			fluxes.append(rdf[:,2])
			names.append(tim.name)
			skies.append(sky)
			
		T = tabledata()
		T.name = np.array(names)
		T.sky = np.array(skies)
		T.ras = np.array(ras)
		T.decs = np.array(decs)
		T.fluxes = np.array(fluxes)
		T.writeto('source10.fits')

		fluxes = np.array(fluxes)
		nims,nsrcs = fluxes.shape

		plt.clf()
		for i in range(nsrcs):
			f = fluxes[:,i]
			I = np.flatnonzero((f > 1) * (f < 1e4))
			plt.semilogy(I, f[I], '.-')
			#plt.semilogy(fluxes[:,i], 'b.-')
		plt.xlabel('measurements in individual images')
		plt.ylim(1., 10000.)
		ps.savefig()





def stage100(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             wisedatadirs=None,
             **kwa):
	bandnum = opt.bandnum
	band = 'w%i' % bandnum

	print 'RA,Dec range', ralo, rahi, declo, dechi

	roipoly = np.array([(ralo,declo),(ralo,dechi),(rahi,dechi),(rahi,declo)])

	TT = []
	for d,tag in wisedatadirs:
		ifn = os.path.join(d, 'WISE-index-L1b.fits')
		T = fits_table(ifn, columns=['ra','dec','scan_id','frame_num'])
		print 'Read', len(T), 'from WISE index', ifn

		# Add a margin around the CRVAL so we catch all fields that touch the RA,Dec box.
		margin = (1016. * 2.75 * np.sqrt(2.) / 3600.) / 2.
		cosdec = np.cos(np.deg2rad((declo + dechi) / 2.))
		print 'Margin:', margin, 'degrees'

		r0 = ralo - margin/cosdec
		r1 = rahi + margin/cosdec
		d0 = declo - margin
		d1 = dechi + margin

		I = np.flatnonzero((T.ra > r0) * (T.ra < r1) * (T.dec > d0) * (T.dec < d1))
		print len(I), 'overlap RA,Dec box'
		T.cut(I)
		T.tag = [tag] * len(T)

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

	wcses = []
	corners = []
	ii = []
	for i in range(len(T)):
		wcs = Sip(T.filename[i], 0)
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

	if ps:
		r0,r1 = corners[:,0].min(), corners[:,0].max()
		d0,d1 = corners[:,1].min(), corners[:,1].max()
		print 'RA,Dec extent', r0,r1, d0,d1

		plot = Plotstuff(outformat='png', ra=(r0+r1)/2., dec=(d0+d1)/2., width=d1-d0, size=(800,800))
		out = plot.outline
		plot.color = 'white'
		plot.alpha = 0.07
		plot.apply_settings()
		for wcs in wcses:
			out.wcs = anwcs_new_sip(wcs)
			out.fill = False
			plot.plot('outline')
			out.fill = True
			plot.plot('outline')
		plot.color = 'gray'
		plot.alpha = 1.0
		plot.lw = 1
		plot.plot_grid(1, 1, 1, 1)
		pfn = ps.getnext()
		plot.write(pfn)
		print 'Wrote', pfn

	return dict(opt100=opt, rd=(ralo,rahi,declo,dechi), T=T, outlines=outlines,
				wcses=wcses, bandnum=bandnum, band=band)


def stage101(opt=None, ps=None, T=None, outlines=None, wcses=None, rd=None, **kwa):
	r0,r1,d0,d1 = rd

	xyrois = []
	subwcses = []
	tims = []

	# Margin 1: grab WISE images that extend outside the RA,Dec box.

	margin1 = 10./3600.

	cosdec = np.cos(np.deg2rad((d0+d1)/2.))
	
	rm0 = r0 - margin1/cosdec
	rm1 = r1 + margin1/cosdec
	dm0 = d0 - margin1
	dm1 = d1 + margin1
	
	# Find the pixel ROI in each image containing the RA,Dec ROI.
	for i,(Ti,wcs) in enumerate(zip(T,wcses)):
		xy = []
		for r,d in [(rm0,dm0),(rm0,dm1),(rm1,dm1),(rm1,dm0)]:
			ok,x,y = wcs.radec2pixelxy(r,d)
			xy.append((x,y))
		xy = np.array(xy)
		x0,y0 = xy.min(axis=0)
		x1,y1 = xy.max(axis=0)
		W,H = int(wcs.get_width()), int(wcs.get_height())
		x0 = np.clip(int(np.floor(x0)), 0, W-1)
		y0 = np.clip(int(np.floor(y0)), 0, H-1)
		x1 = np.clip(int(np.ceil (x1)), 0, W-1)
		y1 = np.clip(int(np.ceil (y1)), 0, H-1)
		assert(x0 <= x1)
		assert(y0 <= y1)
		x1 += 1
		y1 += 1
		
		xyrois.append([x0,x1,y0,y1])

		tim = wise.read_wise_level1b(Ti.filename.replace('-int-1b.fits',''),
									 nanomaggies=True, mask_gz=True, unc_gz=True,
									 sipwcs=True, constantInvvar=True,
									 roi=[x0,x1,y0,y1])
		print 'Read', tim

		# Mask pixels outside the RA,Dec ROI
		if x0 > 0 or y0 > 0 or x1 < W-1 or y1 < H-1:
			print 'Image was clipped -- masking pixels outside ROI'
			h,w = tim.shape
			print 'Clipped size:', w,'x',h
			wcs = tim.getWcs()
			x0,y0 = wcs.getX0Y0()
			XX,YY = np.meshgrid(np.arange(x0, x0+w), np.arange(y0, y0+h))
			# approximate, but *way* faster than doing full WCS per pixel!
			J = point_in_poly(XX.ravel(), YY.ravel(), xy)
			K = J.reshape(XX.shape)
			iv = tim.getInvvar()
			tim.setInvvar(iv * K)
			tim.rdmask = K

		tims.append(tim)

	return dict(opt101=opt, tims=tims, margin1=margin1)

def stage102(opt=None, ps=None, T=None, outlines=None, wcses=None, rd=None,
			 tims=None, band=None, margin1=None, **kwa):
	r0,r1,d0,d1 = rd

	# Read SDSS sources in range.

	S = fits_table(opt.sources, columns=['ra','dec'])
	print 'Read', len(S), 'sources from', opt.sources

	margin2 = margin1 * 2.
	cosdec = np.cos(np.deg2rad((d0+d1)/2.))
	mr = margin2 / cosdec
	md = margin2
	I = np.flatnonzero((S.ra  > (r0-mr)) * (S.ra  < (r1+mr)) *
					   (S.dec > (d0-md)) * (S.dec < (d1+md)))
	print 'Reading', len(I), 'in range'

	S = fits_table(opt.sources, rows=I,
				   column_map=dict(r_dev='theta_dev',
								   r_exp='theta_exp',
								   fracpsf='fracdev'))
	S.row = I
	S.inblock = ((S.ra  >= r0) * (S.ra  < r1) *
				 (S.dec >= d0) * (S.dec < d1))
	S.cmodelflux = S.modelflux

	sband = 'r'

	## NOTE, this method CUTS the "S" arg
	cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
								  objs=S, bands=[], nanomaggies=True, extrabands=[band],
								  fixedComposites=True, forcePointSources=opt.ptsrc)
	print 'Created', len(cat), 'Tractor sources'
	assert(len(cat) == len(S))

	tractor = Tractor(tims, cat)
	cat = tractor.catalog

	## Give each source a minimum brightness
	minbright = 250.
	cat.freezeParamsRecursive('*')
	cat.thawPathsTo(band)
	p0 = cat.getParams()
	cat.setParams(np.maximum(minbright, p0))

	return dict(opt102=opt, tractor=tractor, S=S, margin2=margin2)


psfcache = {}

def stage103(opt=None, ps=None, tractor=None, band=None, bandnum=None, **kwa):
	tims = tractor.images
	tractor.freezeParamsRecursive('*')
	tractor.thawPathsTo(band)
	tractor.thawPathsTo('sky')
	for tim in tims:
		x0,y0 = tim.getWcs().getX0Y0()
		h,w = tim.shape
		tim.psf = wise.get_psf_model(bandnum, opt.pixpsf, xy=(x0+w/2, y0+h/2),
									 positive=False, cache=psfcache)
	return dict(opt103=opt)

def stage104(opt=None, ps=None, tractor=None, band=None, bandnum=None, T=None,
			 S=None, ri=None, di=None,
			 **kwa):
	tims = tractor.images

	minFlux = opt.minflux
	if minFlux is not None:
		minFlux = np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val for tim in tims])
		print 'minFlux:', minFlux, 'nmgy'

	t0 = Time()
	ims0,ims1,IV,fs = tractor.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
														 sky=True, minFlux=minFlux,
														 fitstats=True,
														 variance=True)
	print 'Forced phot took', Time()-t0

	cat = tractor.catalog
	assert(len(cat) == len(S))
	assert(len(cat) == len(IV))

	# The parameters are stored in the order: sky, then fluxes
	#print 'Variance vector:', len(V)
	#print '# images:', len(tims)
	#print '# sources:', len(cat)

	R = tabledata()
	R.ra  = np.array([src.getPosition().ra  for src in cat])
	R.dec = np.array([src.getPosition().dec for src in cat])
	R.set(band, np.array([src.getBrightness().getBand(band) for src in cat]))
	R.set(band + '_ivar', IV)
	R.row = S.row
	R.inblock = S.inblock.astype(np.uint8)

	imstats = tabledata()

	if fs is not None:
		for k in ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
			R.set(k, getattr(fs, k))

		imstats = tabledata()
		for k in ['imchisq', 'imnpix', 'sky']:
			X = getattr(fs, k)
			imstats.set(k, X)
			#print 'image stats', k, '=', X
		imstats.scan_id = T.scan_id
		imstats.frame_num = T.frame_num

	return dict(R=R, imstats=imstats, ims0=ims0, ims1=ims1)


def stage105_OLD(R=None, imstats=None, opt=None, **kwa):
	fn = '%s-w%i-forced.fits' % (opt.name, opt.bandnum)
	R.writeto(fn)
	print 'Wrote', fn
	fn = '%s-w%i-imstats.fits' % (opt.name, opt.bandnum)
	imstats.writeto(fn)
	print 'Wrote', fn



def stage205(opt=None, ps=None, tractor=None, band=None, bandnum=None, T=None,
			 S=None, ri=None, di=None,
			 ims0=None, ims1=None,
			 ttsuf='', pcat=[], addSky=False,
			 **kwa):

	if ps is not None:
		ptims = tractor.images

		imas = [dict(interpolation='nearest', origin='lower',
					 vmin=tim.zr[0], vmax=tim.zr[1]) for tim in ptims]
		imchis = [dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)]*len(ptims)

		tt = 'Block ' + str(ri) + ', ' + str(di) + ttsuf

		if addSky:
			# in engine.py, we subtracted the sky when computing per-image
			for tim,(img,mod,ie,chi,roi) in zip(ptims, ims1):
				tim.getSky().addTo(mod)
	
		_plot_grid([img for (img, mod, ie, chi, roi) in ims0], imas)
		plt.suptitle('Data: ' + tt)
		ps.savefig()

		if ims1 is not None:
			_plot_grid2(ims1, pcat, ptims, imas)
			plt.suptitle('Forced-phot model: ' + tt)
			ps.savefig()

			_plot_grid2(ims1, pcat, ptims, imchis, ptype='chi')
			plt.suptitle('Forced-phot chi: ' + tt)
			ps.savefig()


def stage204(opt=None, ps=None, tractor=None, band=None, bandnum=None, **kwa):
	tims = tractor.images

	tractor.freezeParam('images')

	minFlux = opt.minflux
	if minFlux is not None:
		minFlux = np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val for tim in tims])
		print 'minFlux:', minFlux, 'nmgy'

	t0 = Time()
	ims0,ims1 = tractor.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
												   sky=False, minFlux=minFlux)
	print 'Forced phot took', Time()-t0


def stage304(opt=None, ps=None, tractor=None, band=None, bandnum=None, rd=None, **kwa):

	r0,r1,d0,d1 = rd
	cat = tractor.catalog

	ra  = np.array([src.getPosition().ra  for src in tractor.catalog])
	dec = np.array([src.getPosition().dec for src in tractor.catalog])

	# W = fits_table('/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part45-radec.fits')
	# print 'Read', len(W), 'WISE'
	# W.cut((W.ra > r0) * (W.ra < r1) * (W.dec > d0) * (W.dec < d1))
	# print 'Cut to', len(W), 'WISE'
	# I,J,d = match_radec(ra, dec, W.ra, W.dec, 4./3600.)
	# print len(I), 'matches to WISE sources'
	
	print 'Clustering with radius', opt.wrad, 'arcsec'
	Wrad = opt.wrad / 3600.
	groups,singles = cluster_radec(ra, dec, Wrad, singles=True)
	#print 'Source clusters:', groups
	#print 'Singletons:', singles
	print 'Source clusters:', len(groups)
	print 'Singletons:', len(singles)
	
	print 'Group size histogram:'
	ng = Counter()
	for g in groups:
		ng[len(g)] += 1
	kk = ng.keys()
	kk.sort()
	for k in kk:
		print '  ', k, 'sources:', ng[k], 'groups'

	tims = tractor.images
	tractor.freezeParam('images')

	minFlux = opt.minflux
	if minFlux is not None:
		minFlux = np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val for tim in tims])
		print 'minFlux:', minFlux, 'nmgy'

	dpix = opt.wrad / 2.75

	sgroups = [[i] for i in singles]

	NG = len(sgroups) + len(groups)
	
	for gi,X in enumerate(sgroups + groups):

		print 'Group', gi, 'of', NG, 'groups;', len(X), 'sources'

		mysrcs = [cat[i] for i in X]
		mytims = []
		myrois = []
		for tim in tims:
			wcs = tim.getWcs()
			xy = []
			for src in mysrcs:
				xy.append(wcs.positionToPixel(src.getPosition()))
			xy = np.array(xy)
			xi,yi = xy[:,0], xy[:,1]
			H,W = tim.shape
			x0 = np.clip(int(np.floor(xi.min() - dpix)), 0, W-1)
			y0 = np.clip(int(np.floor(yi.min() - dpix)), 0, H-1)
			x1 = np.clip(int(np.ceil (xi.max() + dpix)), 0, W-1)
			y1 = np.clip(int(np.ceil (yi.max() + dpix)), 0, H-1)
			if x0 == x1 or y0 == y1:
				continue
			#myrois.append([x0,x1,y0,y1])
			myrois.append((slice(y0,y1+1), slice(x0,x1+1)))
			mytims.append(tim)

		# FIXME -- Find sources nearby!
		
		subtr = Tractor(mytims, mysrcs)
		subtr.freezeParamsRecursive('*')
		subtr.thawPathsTo(band)

		t0 = Time()
		ims0,ims1 = subtr.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
													 sky=False, minFlux=minFlux,
													 rois=myrois)
		print 'Forced phot took', Time()-t0


def stage305(opt=None, ps=None, tractor=None, band=None, bandnum=None, rd=None, **kwa):
	r0,r1,d0,d1 = rd
	cat = tractor.catalog
	tims = tractor.images

	cat.freezeParamsRecursive('*')
	tractor.thawPathsTo('sky')

	for ti,tim in enumerate(tims):
		print 'Image', ti, 'of', len(tims)

		subtr = Tractor([tim], cat)
		t0 = Time()
		ims0,ims1 = subtr.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
													 sky=True, minFlux=None)
		print 'Optimizing sky took', Time()-t0


def stage306(opt=None, ps=None, tractor=None, band=None, bandnum=None, rd=None, **kwa):
	r0,r1,d0,d1 = rd
	cat = tractor.catalog
	tims = tractor.images

	cat.freezeParamsRecursive('*')
	cat.thawPathsTo(band)
	tractor.thawPathsTo('sky')

	p0 = tractor.getParams()

	perimparams = []
	perimfit = []
	perimsky = []

	cat0 = cat.getParams()

	for ti,tim in enumerate(tims):
		print 'Image', ti, 'of', len(tims)

		cat.setParams(cat0)

		wcs = tim.getWcs()
		H,W = tim.shape
		margin = 5.
		srcs, fsrcs = [],[]
		ii = []
		for i,src in enumerate(cat):
			x,y = wcs.positionToPixel(src.getPosition())
			if x > 0 and x < W and y > 0 and y < H:
				srcs.append(src)
				ii.append(i)
			elif x > -margin and x < (W+margin) and y > -margin and y < (H+margin):
				fsrcs.append(src)
		print len(srcs), 'in bounds plus', len(fsrcs), 'nearby'
		subcat = Catalog(*(srcs + fsrcs))
		print len(subcat), 'in subcat'
		for i in range(len(fsrcs)):
			subcat.freezeParam(len(srcs) + i)

		### We should do something about sources that live in regions of ~ 0 invvar!


		subtr = Tractor([tim], subcat)
		t0 = Time()
		ims0,ims1 = subtr.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
													 sky=True, minFlux=None)
		print 'Forced phot took', Time()-t0

		continue

		imas = [dict(interpolation='nearest', origin='lower',
					 vmin=tim.zr[0], vmax=tim.zr[1])]
		imchis = [dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)]
		tt = tim.name

		pcat = []
		ptims = [tim]

		_plot_grid([img for (img, mod, ie, chi, roi) in ims0], imas)
		plt.suptitle('Data: ' + tt)
		ps.savefig()

		_plot_grid2(ims0, pcat, ptims, imas)
		plt.suptitle('Initial model: ' + tt)
		ps.savefig()

		_plot_grid2(ims0, pcat, ptims, imchis, ptype='chi')
		plt.suptitle('Initial chi: ' + tt)
		ps.savefig()

		if ims1 is not None:
			_plot_grid2(ims1, pcat, ptims, imas)
			plt.suptitle('Forced-phot model: ' + tt)
			ps.savefig()

			_plot_grid2(ims1, pcat, ptims, imchis, ptype='chi')
			plt.suptitle('Forced-phot chi: ' + tt)
			ps.savefig()

		perimparams.append(cat.getParams())
		perimfit.append(ii)
		perimsky.append(tim.getParams())

	return dict(perimflux=perimparams,
				periminds=perimfit,
				perimsky=perimsky)
				


def main():

	#plt.figure(figsize=(12,12))
	#plt.figure(figsize=(10,10))
	plt.figure(figsize=(8,8))


	import optparse
	parser = optparse.OptionParser('%prog [options]')
	parser.add_option('-v', dest='verbose', action='store_true')

	parser.add_option('--ri', dest='ri', type=int,
					  default=0, help='RA slice')
	parser.add_option('--di', dest='di', type=int,
					  default=0, help='Dec slice')

	parser.add_option('-S', '--stage', dest='stage', type=int,
					  default=0, help='Run to stage...')
	parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
					  help="Force re-running the given stage(s) -- don't read from pickle.")

	parser.add_option('--nw', dest='write', action='store_false', default=True,
					  help='Do not write stage pickle files')

	parser.add_option('-w', dest='bandnum', type=int, default=1,
					  help='WISE band (default %default)')

	parser.add_option('--ppat', dest='picklepat', default=None,
					  help='Stage pickle pattern')

	parser.add_option('--threads', dest='threads', type=int, help='Multiproc')

	parser.add_option('--osources', dest='osources',
					  help='File containing competing measurements to produce a model image for')

	parser.add_option('-s', dest='sources',
					  help='Input SDSS source list')
	parser.add_option('-i', dest='individual', action='store_true',
					  help='Fit individual images?')

	parser.add_option('-n', dest='name', default='wise',
					  help='Base filename for outputs (plots, stage pickles)')

	parser.add_option('-P', dest='ps', default=None,
					  help='Filename pattern for plots')

	
	parser.add_option('-M', dest='plotmask', action='store_true',
					  help='Plot mask plane bits?')

	parser.add_option('--ptsrc', dest='ptsrc', action='store_true',
					  help='Set all sources to point sources')
	parser.add_option('--pixpsf', dest='pixpsf', action='store_true',
					  help='Use pixelized PSF -- use with --ptsrc')

	parser.add_option('--nonconst-invvar', dest='constInvvar', action='store_false',
					  default=True, help='Do not set the invvar constant')

	parser.add_option('--wrad', dest='wrad', default=15., type=float,
					  help='WISE radius: look at a box this big in arcsec around the source position')
	parser.add_option('--srad', dest='srad', default=0., type=float,
					  help='SDSS radius: grab SDSS sources within this radius in arcsec.  Default: --wrad + 5')

	parser.add_option('--minsb', dest='minsb', type=float, default=0.05,
					  help='Minimum surface-brightness approximation, default %default')

	parser.add_option('--minflux', dest='minflux', type=str, default="-5",
					  help='Minimum flux a source is allowed to have, in sigma; default %default; "none" for no limit')

	parser.add_option('-p', dest='plots', action='store_true',
					  help='Make result plots?')
	parser.add_option('-r', dest='result',
					  help='result file to compare', default='measurements-257.fits')
	parser.add_option('-m', dest='match', action='store_true',
					  help='do RA,Dec match to compare results; else assume 1-to-1')
	parser.add_option('-N', dest='nearest', action='store_true', default=False,
					  help='Match nearest, or all?')

	opt,args = parser.parse_args()

	if opt.verbose:
		lvl = logging.DEBUG
	else:
		lvl = logging.INFO
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	if opt.picklepat is None:
		opt.picklepat = opt.name + '-stage%0i.pickle'
	if opt.ps is None:
		opt.ps = opt.name

	if opt.threads:
		mp = multiproc(opt.threads)
	else:
		mp = multiproc(1)

	if opt.minflux in ['none','None']:
		opt.minflux = None
	else:
		opt.minflux = float(opt.minflux)

	# W3 area
	r0,r1 = 210.593,  219.132
	d0,d1 =  51.1822,  54.1822
	dd = np.linspace(d0, d1, 51)
	rr = np.linspace(r0, r1, 91)

	ri = opt.ri
	di = opt.di
	rlo,rhi = rr[ri],rr[ri+1]
	dlo,dhi = dd[di],dd[di+1]

    basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise1test'
    wisedatadirs = [(os.path.join(basedir, 'allsky'), 'cryo'),
                    (os.path.join(basedir, 'prelim_postcryo'), 'post-cryo'),]
    
	runtostage(opt.stage, opt, mp, rlo,rhi,dlo,dhi,
               wisedatadirs=wisedatadirs)



def runtostage(stage, opt, mp, rlo,rhi,dlo,dhi, **kwa):

	class MyCaller(CallGlobal):
		def getkwargs(self, stage, **kwargs):
			kwa = self.kwargs.copy()
			kwa.update(kwargs)
			kwa.update(ps = PlotSequence(opt.ps + '-s%i' % stage, format='%03i'))
			return kwa

	prereqs = { 100: None,
				204: 103,
				205: 104,
				304: 103,
				}

	runner = MyCaller('stage%i', globals(), opt=opt, mp=mp,
					  declo=dlo, dechi=dhi, ralo=rlo, rahi=rhi,
					  ri=opt.ri, di=opt.di, **kwa)

	R = runstage(stage, opt.picklepat, runner, force=opt.force, prereqs=prereqs,
				 write=opt.write)
	return R

if __name__ == '__main__':
	main()
	
