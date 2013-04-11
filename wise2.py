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


if __name__ == '__main__':

	#ralo = 36
	#rahi = 42
	#declo = -1.25
	#dechi = 1.25
	#width = 7

	ralo = 38.5
	rahi = 40.5
	declo = -0.5
	dechi = 1.5
	width = 2.5

	rl,rh = 39,40
	dl,dh = 0,1

	ra  = (ralo  + rahi ) / 2.
	dec = (declo + dechi) / 2.

	basedir = '/project/projectdirs/bigboss'
	wisedatadir = os.path.join(basedir, 'data', 'wise')
	
	ifn = os.path.join(wisedatadir, 'index-allsky-astr-L1b.fits')
	T = fits_table(ifn, columns=['ra','dec','scan_id','frame_num'])
				   
	print 'Read', len(T), 'from WISE index'

	print 'RA', T.ra.min(), T.ra.max()
	print 'Dec', T.dec.min(), T.dec.max()

	I = np.flatnonzero((T.ra > ralo) * (T.ra < rahi) * (T.dec > declo) * (T.dec < dechi))
	print len(I), 'overlap RA,Dec box'

	for sid,fnum in zip(T.scan_id[I], T.frame_num[I]):
		print 'scan,frame', sid, fnum

	T = fits_table(ifn, rows=I)
	print 'Read', len(T), 'rows'

	print 'Header:', type(T.header), T.header.shape

	roipoly = np.array([(rl,dl),(rl,dh),(rh,dh),(rh,dl)])
	
	newhdr = []
	wcses = []
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

		print hdrstr

		wcs = getWcsFromHeaderString(hdrstr)

		W,H = 1016,1016

		rd = []
		for x,y in [(1,1),(1,H),(W,H),(W,1)]:
			rd.append(wcs.pixelxy2radec(x,y))
		rd = np.array(rd)

		if polygons_intersect(roipoly, rd):
			wcses.append(wcs)


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
