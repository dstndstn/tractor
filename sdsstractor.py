if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import pyfits
import pylab as plt
import numpy as np
from math import pi, sqrt

from tractor import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.ngc2000 import ngc2000

class SDSSTractor(Tractor):

	def createNewSource(self, img, x, y, ht):
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		#patch = img.getPsf().getPointSourcePatch(x, y)
		#print 'psf patch:', patch.shape
		#print 'psf patch: max', patch.max(), 'sum', patch.sum()
		#print 'new source peak height:', ht, '-> flux', ht/patch.max()
		#ht /= patch.max()
		return PointSource(PixPos([x,y]), Flux(ht))








def choose_field():
	# Nice cluster IC21:
	#  http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpegcodec.aspx?R=3437&C=3&F=392&Z=50
	# 4 Big ones!  NGC 192, 196
	# http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpegcodec.aspx?R=1755&C=6&F=442&Z=50
	# NGC 307 at RA,Dec (14.15, -1.76667) Run 1755 Camcol 1 Field 471
	# 3 nice smooth ones -- NGC 426 at RA,Dec (18.225000000000001, -0.283333) Run 125 Camcol 3 Field 196 
	# Variety -- NGC 560 at RA,Dec (21.850000000000001, -1.9166700000000001) Run 1755 Camcol 1 Field 522
	# nice (but bright star) -- IC 232 at RA,Dec (37.799999999999997, 1.25) Run 4157 Camcol 6 Field 128
	# Crowded field! NGC 6959 at RA,Dec (311.77499999999998, 0.45000000000000001) Run 3360 Camcol 5 Field 39
	# medium spiral -- NGC 6964 at RA,Dec (311.85000000000002, 0.29999999999999999) Run 4184 Camcol 4 Field 68
	# two blended together -- NGC 7783 at RA,Dec (358.55000000000001, 0.38333299999999998) Run 94 Camcol 4 Field 158
	#### 3 nice smooth ones -- NGC 426 at RA,Dec (18.225000000000001, -0.283333) Run 125 Camcol 3 Field 196 
	# 94/1r/33

	s82 = fits_table('/Users/dstn/deblend/s82fields.fits')
	for n in ngc2000:
		#print n
		ra,dec = n['ra'], n['dec']
		if abs(dec) > 2.:
			continue
		if 60 < ra < 300:
			continue
		if not 'classification' in n:
			continue
		clas = n['classification']
		if clas != 'Gx':
			continue
		isngc = n['is_ngc']
		num = n['id']
		print 'NGC' if isngc else 'IC', num, 'at RA,Dec', (ra,dec)
		dst = ((s82.ra - ra)**2 + (s82.dec - dec)**2)
		I = np.argmin(dst)
		f = s82[I]
		print 'Run', f.run, 'Camcol', f.camcol, 'Field', f.field
		print 'dist', np.sqrt(dst[I])
		print ('<img src="%s" /><br /><br />' %
			   ('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpegcodec.aspx?R=%i&C=%i&F=%i&Z=25' % (f.run, f.camcol, f.field)))
		


def main():
	# image
	# invvar
	# sky
	# psf

	sdss = DR7()

	# choose_field()

	run = 125
	camcol = 3
	field = 196
	bandname = 'i'

	#x0,x1,y0,y1 = 1000,1250, 400,650
	#x0,x1,y0,y1 = 250,500, 1150,1400
	#x0,x1,y0,y1 = 200,500, 1100,1400
	# A sparse field with a small galaxy
	# (gets fit by only one star)
	x0,x1,y0,y1 = 0,300, 200,500
	# A sparse field with no galaxies
	#x0,x1,y0,y1 = 500,800, 400,700

	band = band_index(bandname)

	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias
	image = fpC
	
	psfield = sdss.readPsField(run, camcol, field)
	gain = psfield.getGain(band)
	darkvar = psfield.getDarkVariance(band)
	sky = psfield.getSky(band)
	skyerr = psfield.getSkyErr(band)
	skysig = sqrt(sky)

	fpM = sdss.readFpM(run, camcol, field, bandname)

	invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

	zrange = np.array([-3.,+10.]) * skysig + sky

	plt.clf()
	plt.imshow(image, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1])
	plt.hot()
	plt.colorbar()
	ax = plt.axis()
	plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'b-')
	plt.axis(ax)
	plt.savefig('fullimg.png')

	roi = (slice(y0,y1), slice(x0,x1))
	image = image[roi]
	invvar = invvar[roi]

	dgpsf = psfield.getDoubleGaussian(band)
	print 'Creating double-Gaussian PSF approximation'
	print '  ', dgpsf

	(a,s1, b,s2) = dgpsf
	psf = NGaussianPSF([s1, s2], [a, b])

	# We'll start by working in pixel coords
	wcs = NullWCS()
	# And counts
	photocal = NullPhotoCal()

	data = Image(data=image, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
				 photocal=photocal)
	
	tractor = SDSSTractor([data])

	
	plt.clf()
	plt.imshow(image, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1])
	plt.hot()
	plt.colorbar()
	plt.savefig('img.png')

	plt.clf()
	plt.imshow((image - sky) * np.sqrt(invvar),
			   interpolation='nearest', origin='lower')
	plt.hot()
	plt.colorbar()
	plt.savefig('chi.png')

	plt.clf()
	plt.imshow((image - sky) * np.sqrt(invvar),
			   interpolation='nearest', origin='lower',
			   vmin=-3, vmax=10.)
	plt.hot()
	plt.colorbar()
	plt.savefig('chi2.png')


	Nsrc = 10
	#steps = (['plots'] + ['source']*Nsrc + ['plots'] + ['psf'] + ['plots'] + ['psf2'])*3 + ['plots'] + ['break']

	#steps = (['plots'] + (['source']*5 + ['plots'])*Nsrc + ['psf'] + ['plots'])

	steps = (['plots'] + (['source']*5 + ['plots'])*Nsrc)
	print 'steps:', steps

	chiArange = None

	ploti = 0

	for i,step in enumerate(steps):

		if step == 'plots':
			print 'Making plots...'
			NS = len(tractor.getCatalog())

			chis = tractor.getChiImages()
			chi = chis[0]

			tt = 'sources: %i, chi^2 = %g' % (NS, np.sum(chi**2))

			mods = tractor.getModelImages()
			mod = mods[0]

			plt.clf()
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin=zrange[0], vmax=zrange[1])
			plt.hot()
			plt.colorbar()
			ax = plt.axis()
			img = tractor.getImage(0)
			wcs = img.getWcs()
			x = []
			y = []
			for src in tractor.getCatalog():
				pos = src.getPosition()
				px,py = wcs.positionToPixel(pos)
				x.append(px)
				y.append(py)
			plt.plot(x, y, 'b+')
			plt.axis(ax)
			plt.title(tt)
			plt.savefig('mod-%02i.png' % ploti)

			if chiArange is None:
				chiArange = (chi.min(), chi.max())

			plt.clf()
			plt.imshow(chi, interpolation='nearest', origin='lower',
					   vmin=chiArange[0], vmax=chiArange[1])
			plt.hot()
			plt.colorbar()
			plt.title(tt)
			plt.savefig('chiA-%02i.png' % ploti)

			plt.clf()
			plt.imshow(chi, interpolation='nearest', origin='lower',
					   vmin=-3, vmax=10.)
			plt.hot()
			plt.colorbar()
			plt.title(tt)
			plt.savefig('chiB-%02i.png' % ploti)

			ploti += 1

		elif step == 'source':
			print
			print 'Before createSource, catalog is:',
			tractor.getCatalog().printLong()
			print
			rtn = tractor.createSource()
			print
			print 'After  createSource, catalog is:',
			tractor.getCatalog().printLong()
			print

			if False:
				(sm,tryxy) = rtn[0]
				plt.clf()
				plt.imshow(sm, interpolation='nearest', origin='lower')
				plt.hot()
				plt.colorbar()
				ax = plt.axis()
				plt.plot([x for x,y in tryxy], [y for x,y in tryxy], 'b+')
				plt.axis(ax)
				plt.savefig('create-%02i.png' % i)

		elif step == 'psf':
			baton = (i,)
			tractor.optimizeAllPsfAtFixedComplexityStep()
			#derivCallback=(psfDerivCallback, baton))

		elif step == 'psf2':
			tractor.increaseAllPsfComplexity()

		print 'Tractor cache has', len(tractor.cache), 'entries'


if __name__ == '__main__':
	main()
