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
	assert(band == 1)
	scangrp = scanid[-2:]
	return os.path.join(basedir, 'wise1', '4band_p1bm_frm', scangrp, scanid,
						'%03i' % frame, '%s%03i-w1-int-1b.fits' % (scanid, frame))




def stage0(opt=None, ps=None):
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
	Wrad = 15. / 3600.

	# Look for SDSS objects within this radius; Wrad + a margin
	Srad = Wrad + 5./3600.


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

	return dict(tractors=tractors, sources=S, bandnum=bandnum, band=band)



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
	for (img,mod,chi,roi),tim in zip(ims, tims):
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
		plt.subplot(R, C, i+1)
		ax = plt.axis()
		xy = np.array(xy)
		plt.plot(xy[:,0], xy[:,1], 'r+', lw=2)
		plt.axis(ax)



def stage1(opt=None, ps=None, tractors=None, band=None, **kwa):

	#minsb = 0.25
	minsb = 0.05

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

	print 'Got', len(tractors), 'tractors'
	for ti,tractor in enumerate(tractors):
		print '  ', tractor

		tractor.freezeParam('images')
		tims =tractor.images
		cat = tractor.catalog
		cat.freezeParamsRecursive('*')
		cat.thawPathsTo(band)

		ims0,ims1 = tractor.optimize_forced_photometry(minsb=minsb, mindlnp=1.)

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
			tractor.catalog = ocat
			nil,nil,ims3 = tractor.optimize_forced_photometry(minsb=minsb,
															  justims0=True)
			tractor.catalog = cat

			_plot_grid2(ims3, ocat, tims, imas)
			plt.suptitle("Schlegel's model: " + tt)
			ps.savefig()

			_plot_grid2(ims3, ocat, tims, imchis, ptype='chi')
			plt.suptitle("Schlegel's chi: " + tt)
			ps.savefig()


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
	wf = NanoMaggies.magToNanomaggies(W.w1mpro[J])
	p3 = plt.loglog(X[I], wf, 'rx', mew=1.5, ms=6, zorder=30)
	#p3 = plt.loglog(X[I], wf, 'r.', ms=8, zorder=30)

	nil,nil,p4 = plt.errorbar(X, X, yerr=DX, fmt=None, color='k', alpha=0.5, ecolor='0.5',
							  lw=2, capsize=10)

	plt.loglog(X, X/fscale, 'k-', alpha=0.1)
	
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



if __name__ == '__main__':

	#plt.figure(figsize=(12,12))
	#plt.figure(figsize=(10,10))
	plt.figure(figsize=(8,8))


	import optparse
	parser = optparse.OptionParser('%prog [options]')
	parser.add_option('-v', dest='verbose', action='store_true')

	parser.add_option('--stage', dest='stage', type=int,
					  default=0, help='Run to stage...')
	parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
					  help="Force re-running the given stage(s) -- don't read from pickle.")
	parser.add_option('--ppat', dest='picklepat', default='stage%i.pickle',
					  help='Stage pickle pattern')

	parser.add_option('--threads', dest='threads', type=int, help='Multiproc')

	parser.add_option('--osources', dest='osources',
					  help='File containing competing measurements to produce a model image for')

	parser.add_option('-s', dest='sources',
					  help='Input SDSS source list')
	parser.add_option('-i', dest='individual', action='store_true',
					  help='Fit individual images?')
	parser.add_option('-o', dest='output', default='measurements-%03i.fits',
					  help='Filename pattern for outputs; default %default')
	parser.add_option('-P', dest='ps', default='wise',
					  help='Filename pattern for plots; default %default')
	parser.add_option('-M', dest='plotmask', action='store_true',
					  help='Plot mask plane bits?')
	parser.add_option('-O', dest='opt', action='store_true',
					  help='Optimize RA,Dec too (not just forced photom)?')
	parser.add_option('-C', dest='cache',
					  help='Cache file after individual-epoch measurements')
	parser.add_option('--ptsrc', dest='ptsrc', action='store_true',
					  help='Set all sources to point sources')
	parser.add_option('--pixpsf', dest='pixpsf', action='store_true',
					  help='Use pixelized PSF -- use with --ptsrc')

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

	ps = PlotSequence(opt.ps, format='%03i')

	runner = CallGlobal('stage%i', globals(), opt=opt, ps=ps)

	runstage(opt.stage, opt.picklepat, runner, force=opt.force)

