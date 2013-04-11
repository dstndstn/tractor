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
from astrometry.util.util import Sip, anwcs, Tan
from astrometry.blind.plotstuff import *

from astrometry.util.sdss_radec_to_rcf import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

from wise import *

def get_l1b_file(scanid, frame, band):
	assert(band == 1)
	scangrp = scanid[-2:]
	return os.path.join('/clusterfs/riemann/raid007/bosswork/boss/wise_level1b/wise1/4band_p1bm_frm',
						scangrp, scanid, '%03i' % frame, '%s%03i-w1-int-1b.fits' % (scanid, frame))


if __name__ == '__main__':

	#ralo = 36
	#rahi = 42
	#declo = -1.25
	#dechi = 1.25
	#width = 7

	ralo = 38.25
	rahi = 40.75
	declo = -0.75
	dechi = 1.75
	width = 2.5

	rl,rh = 39,40
	dl,dh = 0,1

	ra  = (ralo  + rahi ) / 2.
	dec = (declo + dechi) / 2.

	basedir = '/project/projectdirs/bigboss'
	wisedatadir = os.path.join(basedir, 'data', 'wise')

	wisedatadir = '/clusterfs/riemann/raid007/bosswork/boss/wise_level1b'
	
	ifn = os.path.join(wisedatadir, 'index-allsky-astr-L1b.fits')
	T = fits_table(ifn, columns=['ra','dec','scan_id','frame_num'])
				   
	print 'Read', len(T), 'from WISE index'

	print 'RA', T.ra.min(), T.ra.max()
	print 'Dec', T.dec.min(), T.dec.max()

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

		wcs = getWcsFromHeaderString(hdrstr)
		W,H = wcs.get_width(), wcs.get_height()
		rd = []
		for x,y in [(1,1),(1,H),(W,H),(W,1)]:
			rd.append(wcs.pixelxy2radec(x,y))
		rd = np.array(rd)
		corners.append(rd)
		
		if polygons_intersect(roipoly, rd):
			wcses.append(wcs)
			ii.append(i)

	print 'Found', len(wcses), 'overlapping'
	I = np.array(ii)
	T.cut(I)

	corners = np.vstack(corners)
	print 'Corners', corners.shape

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

	plot.color = 'gray'
	plot.alpha = 1.0
	plot.plot_grid(1, 1, 1, 1)
	plot.write('wisemap.png')

	plot.color = 'black'
	plot.plot('fill')
	plot.color = 'white'
	plot.op = CAIRO_OPERATOR_ADD
	plot.apply_settings()
	img = plot.image
	img.image_low = 0.
	img.image_high = 1e3
	img.resample = 1

	ps = PlotSequence('wise', format='%03i')

	for sid,fnum in zip(T.scan_id[I], T.frame_num[I]):
		print 'scan,frame', sid, fnum
		band = 1
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
