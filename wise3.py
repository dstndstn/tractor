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




def stage0(opt=None):
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


def stage1(opt=None, tractors=None, **kwa):
	print 'Got', len(tractors), 'tractors'
	for tractor in tractors:
		print '  ', tractor
	

	




if __name__ == '__main__':

	#plt.figure(figsize=(12,12))
	plt.figure(figsize=(10,10))


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

	#print 'Locals:', locals().keys()
	#print 'Globals:', globals().keys()


	# class Caller(CallGlobal):
	# 	def getfunc(self, stage):
	# 		func = self.pat % stage
	# 		func = eval(func)
	# 		return func
	#runner = Caller('stage%i', (), opt=opt)
	#runner = CallGlobal('stage%i', (), opt=opt)

	runner = CallGlobal('stage%i', globals(), opt=opt)

	runstage(opt.stage, opt.picklepat, runner, force=opt.force)

