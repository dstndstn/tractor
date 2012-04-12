import sys
import math
import logging
import multiprocessing
from astrometry.util.ngc2000 import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.multiproc import *
from astrometry.sdss.dr7 import *
from tractor import *
from tractor import sdss as st


def main():
	lvl = logging.DEBUG
	logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
	import optparse
	parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')
	parser.add_option('--threads', dest='threads', default=16, type=int, help='Use this many concurrent processors')
	parser.add_option('--force-radius', dest='radius', default=None, type=float, help='Force using this radius (in arcmin)')
	opt,args = parser.parse_args()
	if len(args) != 1:
		parser.print_help()
		sys.exit(-1)

	ngcnum = int(args[0])
	if not ngcnum:
		print 'Failed to parse NGC number "%s"' % args[0]
		sys.exit(-1)

	ngc = get_ngc(ngcnum)
	if ngc is None:
		print 'Failed to find NGC object in ngc2000 list'
		sys.exit(-1)

	mp = multiproc(nthreads = opt.threads)

	ra,dec,radius = ngc.ra, ngc.dec, ngc.size / 2. / 60.
	print 'Found NGC', ngcnum, 'at RA,Dec', ngc.ra, ngc.dec, 'radius', ngc.size/2, 'arcmin'

	if opt.radius is not None:
		print 'Forcing radius of', opt.radius, 'arcmin'
		radius = opt.radius / 60.

	bands=['r']#,'g']#,'u','i','z']
	canonband = 'r'
	pixscale = 0.396
	pixr = radius * 3600. / pixscale
	rerun = 0
	H,W = 1489,2048

	TI = []
	sources = []
	sdss = DR7()
	RCF = radec_to_sdss_rcf(ra, dec, radius=math.hypot(radius*60., 13./2.))
	print 'Run,camcol,field list:', RCF
	for run,camcol,field,rr,dd in RCF:
		fn = sdss.getPath('tsField', run, camcol, field)
		if not os.path.exists(fn):
			sdss.retrieve('tsField', run, camcol, field)
		tsf = sdss.readTsField(run, camcol, field, rerun)
		astrans = tsf.getAsTrans(canonband)
		x,y = astrans.radec_to_pixel(ra, dec)
		print 'x,y', x,y
		roi = [np.clip(x-pixr, 0, W), np.clip(x+pixr, 0, W),
			   np.clip(y-pixr, 0, H), np.clip(y+pixr, 0, H)]
		if roi[0] == roi[1] or roi[2] == roi[3]:
			print 'Skipping', run, camcol, field
			continue

		TI.extend([st.get_tractor_image(run, camcol, field, band, useMags=True,
										roiradecsize=(ra,dec,pixr), sdssobj=sdss)
				   for band in bands])
		sources.extend(st.get_tractor_sources(run, camcol, field, bandname=canonband,
											  bands=bands, roi=roi))
		# (could also get 'roi' from the st.get_tractor_image info dict)


	tractor = Tractor([tim for tim,tinf in TI], sources, mp)

	for im in tractor.getImages():
		im.freezeParams('wcs', 'photocal', 'psf', 'sky')

	#tractor.freezeParam('catalog')

	for i in range(5):
		if True or (i % 5 == 0):
			print 'Thawing sky...'
			tractor.images.thawParamsRecursive('sky')

		print 'Iteration', i
		nms = tractor.getParamNames()
		print len(nms), 'parameters:'
		for nm in nms:
			print '	 ', nm
		#tractor.optimizeCatalogLoop(nsteps=1)
		tractor.opt2()

		#tractor.cache.about()
		tractor.cache.printStats()
		#tractor.clearCache()

		if True or (i % 5 == 0):
			print 'Freezing sky...'
			tractor.images.freezeParamsRecursive('sky')

if __name__ == '__main__':
	main()
	
