import os
import math

import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg
import lsst.pex.policy as pexPolicy
import lsst.meas.utils.sourceDetection as muDetection
import lsst.meas.utils.sourceMeasurement as muMeasurement


def getFakePsf(pixscale):
	#fwhmarcsec = 0.7 #1.0 #0.5
	fwhmarcsec = 1.0
	fwhm = fwhmarcsec / wcs.pixelScale().asArcseconds()
	print 'fwhm', fwhm
	psfsize = 25
	model = 'DoubleGaussian'
	sig = fwhm/(2.*math.sqrt(2.*math.log(2.)))
	print 'sigma', sig
	psf = afwDet.createPsf(model, psfsize, psfsize, sig, 0., 0.)
	print 'psf', psf
	return psf

def cr(crfn):
	exposure = afwImage.ExposureF('850994p-21.fits')
	print 'exposure', exposure
	print 'w,h', exposure.getWidth(), exposure.getHeight()
	W,H = exposure.getWidth(), exposure.getHeight()

	#var = exposure.getMaskedImage().getVariance()
	#print 'Variance', var.get(0,0)

	wcs = exposure.getWcs()
	print 'wcs', wcs
	pixscale = wcs.pixelScale().asArcseconds()
	psf = getFakePsf(pixscale)

	# CRs
	mask = exposure.getMaskedImage().getMask()
	crBit = mask.getMaskPlane("CR")
	mask.clearMaskPlane(crBit)
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

	keepCRs = False
	policy = pexPolicy.Policy()
	# policy.add('minSigma', 6.)
	# policy.add('min_DN', 150.)
	# policy.add('cond3_fac', 2.5)
	# policy.add('cond3_fac2', 0.6)
	# policy.add('niteration', 3)
	# policy.add('nCrPixelMax', 200000)

	policy.add('minSigma', 10.)
	policy.add('min_DN', 500.)
	policy.add('cond3_fac', 2.5)
	policy.add('cond3_fac2', 0.6)
	policy.add('niteration', 1)
	policy.add('nCrPixelMax', 100000)

	#psfimg = psf.computeImage(afwGeom.Point2D(W/2., H/2.))
	#psfimg.writeFits('psf.fits')

	print 'Finding cosmics...'
	crs = measAlg.findCosmicRays(mi, psf, bg, policy, keepCRs)
	print 'got', len(crs), 'cosmic rays',

	mask = mi.getMask()
	crBit = mask.getPlaneBitMask("CR")
	afwDet.setMaskFromFootprintList(mask, crs, crBit)
	mask.writeFits('mask.fits')
	exposure.writeFits(crfn)





if __name__ == '__main__':
	crfn = 'cr.fits'
	if not os.path.exists(crfn):
		cr(crfn)
	print 'Reading', crfn

	exposure = afwImage.ExposureF(crfn)
	print 'Read', exposure
	W,H = exposure.getWidth(), exposure.getHeight()

	var = exposure.getMaskedImage().getVariance()
	print 'Variance:', var.get(0,0)

	wcs = exposure.getWcs()
	pixscale = wcs.pixelScale().asArcseconds()
	psf = getFakePsf(pixscale)

	mi = exposure.getMaskedImage()
	bg = afwMath.makeStatistics(mi, afwMath.MEDIAN).getValue()
	print 'bg', bg

	print 'Before subtracting bg:', mi.getImage().get(W/2,H/2)
	img = mi.getImage()
	img -= bg
	print 'After subtracting bg:', mi.getImage().get(W/2,H/2)

	# phot.py
	#footprintSet = self.detect(exposure, psf)
	policy = pexPolicy.Policy()
	policy.add('minPixels', 1)
	policy.add('nGrow', 1)
	#policy.add('thresholdValue', 5.)
	policy.add('thresholdValue', 50.)
	policy.add('thresholdType', 'stdev')
	policy.add('thresholdPolarity', 'positive')
	posSources, negSources = muDetection.detectSources(exposure, psf, policy)
	numPos = len(posSources.getFootprints()) if posSources is not None else 0
	numNeg = len(negSources.getFootprints()) if negSources is not None else 0
	print 'Detected', numPos, 'pos and', numNeg, 'neg'
	footprintSet = posSources
	del negSources

	for fp in footprintSet.getFootprints():
		print 'footprint', fp
		print 'npix', fp.getNpix()

	# phot.py
	#bg, exposure = self.background(exposure); del bg
	policy = pexPolicy.Policy()
	policy.add('statisticsproperty', 'MEANCLIP')
	policy.add('undersamplestyle', 'THROW_EXCEPTION')
	policy.add('binsize', 256)
	policy.add('algorithm', 'NATURAL_SPLINE')
	subtract = True
	bg, subtracted = muDetection.estimateBackground(exposure, policy, subtract=subtract)
	print 'bg', bg
	exposure = subtracted

	# phot.py
	#sources = self.measure(exposure, footprintSet, psf, apcorr=apcorr, wcs=wcs)
	footprints = []
	num = len(footprintSet.getFootprints())
	print 'Measuring', num, 'sources'
	footprints.append([footprintSet.getFootprints(), False])
	policy = pexPolicy.Policy()
	sources = muMeasurement.sourceMeasurement(exposure, psf, footprints, policy)
	muMeasurement.computeSkyCoords(wcs, sources)

	# calibrate.py
	#psf, cellSet = self.psf(exposure, sources)
	print 'Finding PSF...'

	selName = 'secondMomentStarSelector'
	selPolicy = pexPolicy.Policy()
	selPolicy.add('fluxLim', 200.)
	selPolicy.add('fluxMax', 0.)
	selPolicy.add('clumpNSigma', 1.5)
	selPolicy.add('borderWidth', 0)
	selPolicy.add('kernelSize', 21)

	algName = 'pcaPsfDeterminer'
	algPolicy = pexPolicy.Policy()
	algPolicy.add('nonLinearSpatialFit', False)
	algPolicy.add('sizeCellX', 512)
	algPolicy.add('sizeCellY', 512)
	algPolicy.add('borderWidth', 0)
	algPolicy.add('nIterForPsf', 10)
	algPolicy.add('constantWeight', True)
	algPolicy.add('lambda', 0.5)
	algPolicy.add('reducedChi2ForPsfCandidates', 2.)
	algPolicy.add('nEigenComponents', 4)
	algPolicy.add('kernelSize', 5)
	algPolicy.add('kernelSizeMin', 13)
	algPolicy.add('kernelSizeMax', 45)
	algPolicy.add('spatialOrder', 2)
	algPolicy.add('nStarPerCell', 0)
	algPolicy.add('nStarPerCellSpatialFit', 10)
	algPolicy.add('tolerance', 1e-2)
	algPolicy.add('spatialReject', 3.)
	
	starSelector = measAlg.makeStarSelector(selName, selPolicy)
	psfCandidateList = starSelector.selectStars(exposure, sources)

	psfDeterminer = measAlg.makePsfDeterminer(algName, algPolicy)
	psf, cellSet = psfDeterminer.determinePsf(exposure, psfCandidateList)

	# The PSF candidates contain a copy of the source, and so we need to explicitly propagate new flags
	for cand in psfCandidateList:
		cand = measAlg.cast_PsfCandidateF(cand)
		src = cand.getSource()
		if src.getFlagForDetection() & measAlg.Flags.PSFSTAR:
			ident = src.getId()
			src = sources[ident]
			assert src.getId() == ident
			src.setFlagForDetection(src.getFlagForDetection() | measAlg.Flags.PSFSTAR)
	exposure.setPsf(psf)

	print 'Got PSF', psf

	# Target cluster
	x,y = 726., 4355.
	psfimg = psf.computeImage(afwGeom.Point2D(x,y))
	psfimg.writeFits('psfimg.fits')

	

	
