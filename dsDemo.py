if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits
from astrometry.util.util import Tan
from tractor import *
from tractor import sdss as st
from tractor import sdss_galaxy as stgal
from astrometry.sdss import *


def main():

	run = 2728
	camcol = 4
	field = 236
	bandname = 'r'
	xc,yc = 1510, 1040
	ra,dec = (333.556, 0.369)
	S = 100
	roi = [xc-S, xc+S, yc-S, yc+S]

	timg,info = st.get_tractor_image(run, camcol, field, bandname, roi=roi)
	sources = st.get_tractor_sources(run, camcol, field, bandname, roi=roi)

	sdss = DR7()
	rerun = 0
	bandnum = band_index(bandname)
	tsf = sdss.readTsField(run, camcol, field, rerun)
	photocal = SdssPhotoCalMag(tsf, bandnum)
	# TEST
	counts = 100
	mag = photocal.countsToFlux(counts)
	print 'Counts', counts, '-> mag', mag
	c2 = photocal.fluxToCounts(mag)
	print '-> counts', c2

	# Replace the photocal and the Mags.
	for s in sources:
		##if hasattr(s, 'getFlux'):
		# UGH!
		x,y = timg.getWcs().positionToPixel(None, s.getPosition())
		print 'x,y (%.1f, %.1f)' % (x,y)
		if isinstance(s, stgal.CompositeGalaxy):
			counts = timg.photocal.fluxToCounts(s.fluxExp)
			s.fluxExp = photocal.countsToFlux(counts)
			counts = timg.photocal.fluxToCounts(s.fluxDev)
			s.fluxDev = photocal.countsToFlux(counts)
			print '  ', s.fluxExp, s.fluxDev
		else:
			#print 'Source', s, 'flux', s.getFlux()
			counts = timg.photocal.fluxToCounts(s.getFlux())
			s.setFlux(photocal.countsToFlux(counts))
			print '  ', s.flux
		#else:
		#	print 'no getFlux() method:', s
	timg.photocal = photocal




	cffn = 'cr.fits'
	psffn = 'psfimg.fits'
	cfx,cfy = 734,4352
	S = 200
	cfroi = [cfx-S, cfx+S, cfy-S, cfy+S]

	I = pyfits.open(cffn)[1].data
	print 'Img data', I.shape
	x0,x1,y0,y1 = cfroi
	roislice = (slice(y0,y1), slice(x0,x1))
	image = I[roislice]
	I = pyfits.open(cffn)[3].data
	var = I[roislice]
	cfstd = np.sqrt(np.median(var))
	invvar = 1./var
	## FIXME -- add source photon noise, read noise
	I = pyfits.open(cffn)[2].data
	mask = I[roislice]
	invvar[mask > 0] = 0.
	del I
	del var

	psfimg = pyfits.open(psffn)[0].data
	print 'PSF image shape', psfimg.shape
	from tractor.emfit import em_fit_2d
	from tractor.fitpsf import em_init_params
	# number of Gaussian components
	S = psfimg.shape[0]
	K = 3
	w,mu,sig = em_init_params(K, None, None, None)
	II = psfimg.copy()
	II /= II.sum()
	# HACK
	II = np.maximum(II, 0)
	print 'Multi-Gaussian PSF fit...'
	xm,ym = -(S/2), -(S/2)
	em_fit_2d(II, xm, ym, w, mu, sig)
	print 'w,mu,sig', w,mu,sig
	psf = GaussianMixturePSF(w, mu, sig)

	wcs = FitsWcs(Tan(cffn, 0))
	print 'CFHT WCS', wcs
	wcs.setX0Y0(x0+1., y0+1.)

	sky = np.median(image)
	print 'Sky', sky
	skyobj = ConstantSky(sky)

	photocal = CfhtPhotoCal(hdr=pyfits.open(cffn)[0].header)

	cftimg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				   sky=skyobj, photocal=photocal,
				   name=('CFHT'))
	
	tractor = Tractor([timg, cftimg])
	tractor.addSources(sources)

	zrs = [np.array([-5.,+5.]) * info['skysig'] + info['sky'],
		   np.array([-5.,+5.]) * cfstd + sky,]

	for i in range(2):
		mod = tractor.getModelImages()[i]
		zr = zrs[i]
		ima = dict(interpolation='nearest', origin='lower',
				   vmin=zr[0], vmax=zr[1])
		plt.clf()
		plt.imshow(mod, **ima)
		plt.savefig('mod%i.png' % i)

		data = tractor.getImage(i).getImage()
		plt.clf()
		plt.imshow(data, **ima)
		plt.savefig('data%i.png' % i)



class Mag(Flux):
	'''
	An implementation of engine.Flux that actually does mags.
	UGH, bad naming!
	'''
	def hashkey(self):
		return ('Mag', self.val)
	def __repr__(self):
		return 'Mag(%g)' % self.val
	def __str__(self):
		return 'Mag: %g' % self.val
	def copy(self):
		return Mag(self.val)
	def getStepSizes(self, img, *args, **kwargs):
		return [0.01]
	# override positive flux limit from Flux
	def setParams(self, p):
		assert(len(p) == 1)
		self.val = p[0]

class CfhtPhotoCal(object):
	def __init__(self, hdr=None):
		if hdr is not None:
			self.exptime = hdr['EXPTIME']
			self.phot_c = hdr['PHOT_C']
			self.phot_k = hdr['PHOT_K']
			self.airmass = hdr['AIRMASS']
			print 'CFHT photometry:', self.exptime, self.phot_c, self.phot_k, self.airmass
		# FIXME -- NO COLOR TERMS (phot_x)!
		'''
COMMENT   Formula for Photometry, based on keywords given in this header:
COMMENT   m = -2.5*log(DN) + 2.5*log(EXPTIME)
COMMENT   M = m + PHOT_C + PHOT_K*(AIRMASS - 1) + PHOT_X*(PHOT_C1 - PHOT_C2)
'''
			
	def fluxToCounts(self, flux):
		M = flux.getValue()
		logc = (M - self.phot_c - self.phot_k * (self.airmass - 1.)) / -2.5
		return self.exptime * 10.**logc
	def countsToFlux(self, counts):
		return Mag(-2.5 * np.log10(counts / self.exptime) +
				   self.phot_c + self.phot_k * (self.airmass - 1.))
		

class SdssPhotoCalMag(object):
	def __init__(self, tsfield, band):
		'''
		band: int
		'''
		self.tsfield = tsfield
		self.band = band
	def fluxToCounts(self, flux):
		''' ugh, "flux" is a Mag '''
		return self.tsfield.mag_to_counts(flux.getValue(), self.band)

	def countsToFlux(self, counts):
		return Mag(self.tsfield.counts_to_mag(counts, self.band))




if __name__ == '__main__':
	main()
	
