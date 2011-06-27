if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
import logging
import numpy as np
import pylab as plt

import pyfits

from astrometry.util.file import *

from tractor import *
from tractor import sdss as st

# Assumes one image.
def save(idstr, tractor, zr):
	mod = tractor.getModelImages()[0]

	synthfn = 'synth-%s.fits' % idstr
	print 'Writing synthetic image to', synthfn
	pyfits.writeto(synthfn, mod, clobber=True)

	pfn = 'tractor-%s.pickle' % idstr
	print 'Saving state to', pfn
	pickle_to_file(tractor, pfn)

	timg = tractor.getImage(0)
	data = timg.getImage()
	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])

	def savepng(pre, img, title=None, **kwargs):
		fn = '%s-%s.png' % (pre, idstr)
		print 'Saving', fn
		plt.clf()
		plt.imshow(img, **kwargs)
		if title is not None:
			plt.title(title)
		plt.colorbar()
		plt.gray()
		plt.savefig(fn)

	sky = np.median(mod)
	savepng('data', data - sky, title='Data '+timg.name, **ima)
	savepng('model', mod - sky, title='Model', **ima)
	savepng('diff', data - mod, title='Data - Model', **ima)

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
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')
	opt,args = parser.parse_args()

	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

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
	prefix = opt.prefix
	if prefix is None:
		prefix = '%06i-%s%i-%04i' % (run, bandname, camcol, field)

	timg,info = st.get_tractor_image(run, camcol, field, bandname,
					 curl=opt.curl, roi=opt.roi)
	sources = st.get_tractor_sources(run, camcol, field, bandname,
					 curl=opt.curl, roi=opt.roi)

	photocal = timg.getPhotoCal()
	for source in sources:
		print 'source', source
		print 'flux', source.getFlux()
		print 'counts', photocal.fluxToCounts(source.getFlux())
		assert(photocal.fluxToCounts(source.getFlux()) > 0.)

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

	makeflipbook(opt, prefix)
	print
	print 'Created flip-book flip-%s.pdf' % prefix



def makeflipbook(opt, prefix):
	# Create a tex flip-book of the plots
	tex = r'''
	\documentclass[compress]{beamer}
	\usepackage{helvet}
	\newcommand{\plot}[1]{\includegraphics[width=0.5\textwidth]{#1}}
	\begin{document}
	'''
	if opt.ntune:
		tex += r'''\part{Tuning steps}\frame{\partpage}''' + '\n'
	page = r'''
	\begin{frame}{%s}
	\plot{data-%s}
	\plot{model-%s} \\
	\plot{diff-%s}
	\end{frame}'''
	tex += page % (('Initial model',) + (prefix,)*3)
	for i in range(opt.ntune):
		tex += page % (('Tuning step %i' % (i+1),) +
					   ('tune-%d-' % (i+1) + prefix,)*3)
	if opt.ntune:
		# Finish with a 'blink'
		tex += r'''\part{Before-n-after}\frame{\partpage}''' + '\n'
		tex += (r'''
		\begin{frame}{Data}
		\plot{data-%s}
		\plot{data-%s} \\
		\plot{diff-%s}
		\plot{diff-%s}
		\end{frame}
		\begin{frame}{Before (left); After (right)}
		\plot{model-%s}
		\plot{model-%s} \\
		\plot{diff-%s}
		\plot{diff-%s}
		\end{frame}
		''' % ((prefix,)*2 +
			   (prefix, 'tune-%d-' % (opt.ntune) + prefix)*3))
	tex += r'\end{document}' + '\n'
	fn = 'flip-' + prefix + '.tex'
	print 'Writing', fn
	open(fn, 'wb').write(tex)
	os.system("pdflatex '%s'" % fn)

if __name__ == '__main__':
	import cProfile
	import sys
	from datetime import tzinfo, timedelta, datetime
	cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
	sys.exit(0)
	#main()
