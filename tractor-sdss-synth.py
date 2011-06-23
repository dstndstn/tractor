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

def main():
	from optparse import OptionParser
	import sys

	parser = OptionParser(usage=('%prog'))
	parser.add_option('-r', '--run', dest='run', type='int')
	parser.add_option('-c', '--camcol', dest='camcol', type='int')
	parser.add_option('-f', '--field', dest='field', type='int')
	parser.add_option('-b', '--band', dest='band', help='SDSS Band (u, g, r, i, z)')
	parser.add_option('--curl', dest='curl', action='store_true', default=False, help='Use "curl", not "wget", to download files')
	parser.add_option('--tune', dest='tune', action='store_true', default=False, help='Improve synthetic image over DR7 by locally optimizing likelihood')
	parser.add_option('--roi', dest='roi', type=int, nargs=4, help='Select an x0,x1,y0,y1 subset of the image')
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
	if opt.tune:
		tractor.optimizeCatalogLoop(nsteps=2)

	mods = tractor.getModelImages()
	mod = mods[0]

	synthfn = 'synth-%06i-%s%i-%04i.fits' % (run, bandname, camcol, field)
	print 'Writing synthetic image to', synthfn
	pyfits.writeto(synthfn, mod, clobber=True)

	zr = np.array([-3.,+10.]) * info['skysig'] + info['sky']
	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])
	imdiff = dict(interpolation='nearest', origin='lower',
				  vmin=-20, vmax=20)
	image = timg.getImage()

	plt.clf()
	plt.imshow(image, **ima)
	plt.colorbar()
	plt.gray()
	plt.savefig('img.png')

	plt.clf()
	plt.imshow(mod, **ima)
	plt.colorbar()
	plt.gray()
	plt.savefig('mod.png')

	plt.clf()
	plt.imshow(image - mod, **imdiff)
	plt.colorbar()
	plt.gray()
	plt.savefig('diff.png')

	plt.clf()
	plt.imshow(image - np.median(image), **imdiff)
	plt.colorbar()
	plt.gray()
	plt.savefig('img2.png')

	plt.clf()
	plt.imshow(mod - np.median(image), **imdiff)
	plt.colorbar()
	plt.gray()
	plt.savefig('mod2.png')


if __name__ == '__main__':
	import cProfile
	import sys
	from datetime import tzinfo, timedelta, datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	sys.exit(0)
	#main()
