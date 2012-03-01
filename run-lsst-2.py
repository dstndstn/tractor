import os
import math
import sys

import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.pipe.tasks as pipeTasks
import lsst.pipe.tasks.calibrate
#import lsst.pipe.base as pipeBase

if __name__ == '__main__':
	from optparse import OptionParser

	parser = OptionParser(usage='%prog <input-image> <output-base>')
	parser.add_option('--ra', dest='ra', type=float, help='RA at which to instantiate PSF')
	parser.add_option('--dec', dest='dec', type=float, help='Dec at which to instantiate PSF')
	opt,args = parser.parse_args()
	if len(args) != 2:
		parser.print_help()
		sys.exit(-1)

	infn = args[0]
	outbase = args[1]

	crfn = '%s-cr.fits' % outbase
	maskfn = '%s-mask.fits' % outbase
	psffn = '%s-psf.fits' % outbase

	exposure = afwImage.ExposureF(infn)
	print 'Read', exposure
	W,H = exposure.getWidth(), exposure.getHeight()

	wcs = exposure.getWcs()
	if opt.ra is not None and opt.dec is not None:
		x,y = wcs.skyToPixel(opt.ra * afwGeom.degrees, opt.dec * afwGeom.degrees)
		print 'Instantiating PSF at x,y', x,y
	else:
		x,y = W/2, H/2

	# Plug in a reasonable variance plane.
	mi = exposure.getMaskedImage()
	bg = afwMath.makeStatistics(mi, afwMath.MEDIAN).getValue()
	print 'bg', bg

	varval = afwMath.makeStatistics(mi, afwMath.VARIANCE).getValue()
	print 'variance', varval, 'std', math.sqrt(varval)
	varval = afwMath.makeStatistics(mi, afwMath.VARIANCECLIP).getValue()
	print 'clipped variance', varval, 'std', math.sqrt(varval)
	var = exposure.getMaskedImage().getVariance()
	var.set(varval)

	var = exposure.getMaskedImage().getVariance()
	print 'Variance:', var.get(0,0)

	calconf = pipeTasks.calibrate.CalibrateConfig()
	calconf.doAstrometry = False
	calconf.doZeropoint = False
	calconf.doApCorr = False

	cr = calconf.repair.cosmicray
	cr.minSigma = 10.
	cr.min_DN = 500.
	cr.niteration = 3
	cr.nCrPixelMax = 1000000

	#calconf.fwhm = 1.0
	#calconf.size = 25
	#calconf.model = 'DoubleGaussian'
	calconf.thresholdValue = 50.

	calconf.photometry.thresholdValue = 50.

	print 'Calibration config:', calconf

	cal = pipeTasks.calibrate.CalibrateTask(config=calconf)
	result = cal.run(exposure)

	exposure = result.exposure
	psf = result.psf

	print 'Exposure', exposure
	mi = exposure.getMaskedImage()
	mask = mi.getMask()
	mask.writeFits(maskfn)
	exposure.writeFits(crfn)

	print 'PSF', psf
	psfimg = psf.computeImage(afwGeom.Point2D(x,y))
	psfimg.writeFits(psffn)
