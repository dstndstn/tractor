import sys
import math
from astrometry.util.ngc2000 import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.sdss.dr7 import *
from tractor import *
from tractor import sdss as st


def main():
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
	import optparse
	parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')
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
	ra,dec,radius = ngc.ra, ngc.dec, ngc.size / 2. / 60.
	print 'Found NGC', ngcnum, 'at RA,Dec', ngc.ra, ngc.dec, 'radius', ngc.size/2, 'arcmin'

	bands=['r','g','u','i','z']
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


	tractor = Tractor([tim for tim,tinf in TI], sources)

	for i in range(5):
		tractor.optimizeCatalogLoop(nsteps=1)


if __name__ == '__main__':
	main()
	
