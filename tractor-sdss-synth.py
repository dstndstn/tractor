if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
from math import sqrt
import numpy as np
import pylab as plt

from astrometry.util.pyfits_utils import *
from astrometry.sdss import *

from tractor import *
from tractor import sdss as st

# Assumes one image.
def save(idstr, tractor, zr):
	mod = tractor.getModelImages()[0]

	synthfn = 'synth-%s.fits' % idstr
	print 'Writing synthetic image to', synthfn
	pyfits.writeto(synthfn, mod, clobber=True)

	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])
	data = tractor.getImage(0).getImage()

	def savepng(pre, img, **kwargs):
		fn = '%s-%s.png' % (pre, idstr)
		print 'Saving', fn
		plt.clf()
		plt.imshow(img, **kwargs)
		plt.colorbar()
		plt.gray()
		plt.savefig(fn)

	sky = np.median(mod)
	savepng('data', data - sky, **ima)
	savepng('model', mod - sky, **ima)
	savepng('diff', data - mod, **ima)

def main():
	from optparse import OptionParser
	import sys

	parser = OptionParser(usage=('%prog'))
	parser.add_option('-r', '--run', dest='run', type='int')
	parser.add_option('-c', '--camcol', dest='camcol', type='int')
	parser.add_option('-f', '--field', dest='field', type='int')
	parser.add_option('-b', '--band', dest='band', help='SDSS Band (u, g, r, i, z)')
	parser.add_option('--curl', dest='curl', action='store_true', default=False, help='Use "curl", not "wget", to download files')
	parser.add_option('--ntune', dest='ntune', type='int', default=0, help='Improve synthetic image over DR7 by locally optimizing likelihood for nsteps iterations')
	parser.add_option('--roi', dest='roi', type=int, nargs=4, help='Select an x0,x1,y0,y1 subset of the image')
	parser.add_option('--prefix', dest='prefix', help='Set output filename prefix; default is the SDSS  RRRRRR-BC-FFFF string (run, band, camcol, field)')
	(opt, args) = parser.parse_args()

	run = opt.run
	field = opt.field
	camcol = opt.camcol
	band = opt.band
	rerun = 0
	if run is None or field is None or camcol is None or band is None:
		parser.print_help()
		print 'Must supply --run, --camcol, --field, --band'
		sys.exit(-1)
	if not band in ['u','g','r','i', 'z']:
		parser.print_help()
		print
		print 'Must supply band (u/g/r/i/z)'
		sys.exit(-1)
	bandname = band
	bandnum = band_index(bandname)
	prefix = opt.prefix
	if prefix is None:
		prefix = '%06i-%s%i-%04i' % (run, bandname, camcol, field)

	timg,info = st.get_tractor_image(run, camcol, field, bandname,
					 curl=opt.curl, roi=opt.roi)
	sources = st.get_tractor_sources(run, camcol, field, bandname,
					 curl=opt.curl, roi=opt.roi)

	### DEBUG
	wcs = timg.getWcs()
	for i,s in enumerate(sources):
		x,y = wcs.positionToPixel(s, s.getPosition())
		print i, ('(%.1f, %.1f): ' % (x,y)), s

	tractor = st.SDSSTractor([timg])
	tractor.addSources(sources)

	zr = np.array([-5.,+5.]) * info['skysig']
	save(prefix, tractor, zr)

	for i in range(opt.ntune):
		tractor.optimizeCatalogLoop(nsteps=1)
		save('tune-%d-' % (i+1) + prefix, tractor, zr)

if __name__ == '__main__':
	import cProfile
	import sys
	from datetime import tzinfo, timedelta, datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	sys.exit(0)
	#main()
