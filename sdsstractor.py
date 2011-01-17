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
from astrometry.util.file import *
from astrometry.util.ngc2000 import ngc2000

from compiled_profiles import *
from galaxy_profiles import *

class Galaxy(Source):
	def __init__(self, pos, flux, re, ab, phi):
		self.name = 'Galaxy'
		self.pos = pos
		self.flux = flux
		self.re = re
		self.ab = ab
		self.phi = phi

	def hashkey(self):
		return (self.name, self.pos.hashkey(), self.flux.hashkey(),
				self.re, self.ab, self.phi)
	def __eq__(self, other):
		return hash(self) == hash(other)

	def __str__(self):
		return (self.name + ' at ' + str(self.pos)
				+ ' with ' + str(self.flux)
				+ ', re=%.1f, ab=%.2f, phi=%.1f' % (self.re, self.ab, self.phi))
	def __repr__(self):
		return (self.name + '(pos=' + repr(self.pos) +
				', flux=' + repr(self.flux) +
				', re=%.1f, ab=%.2f, phi=%.1f)' % (self.re, self.ab, self.phi))

	def copy(self):
		return None

	def getGalaxyPatch(self, img, cx, cy):
		# remember to include margin for psf conv
		return None

	def getPosition(self):
		return self.pos

	def getModelPatch(self, img, px=None, py=None):
		if px is None or py is None:
			(px,py) = img.getWcs().positionToPixel(self.getPosition())
		patch = self.getGalaxyPatch(img, px, py)
		if patch.getImage() is None:
			return Patch(patch.getX0(), patch.getY0(), None)
		psf = img.getPsf()
		convinv = psf.applyTo(patch.getImage())
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		return Patch(patch.getX0(), patch.getY0(), convinv * counts)

	def numberOfGalaxyParams(self):
		return 3

	#def getGalaxyParamDerivatives(self, img):
	#	return [ None, None, None ]

	def getGalaxyParamStepSizes(self, img):
		return [ self.re * 0.1, max(0.01, self.ab * 0.1), 1. ]

	def getGalaxyParamNames(self):
		return ['re', 'ab', 'phi']

	def stepGalaxyParam(self, i, dg):
		assert(i >= 0)
		assert(i < 3)
		if i == 0:
			self.re += dg
		elif i == 1:
			self.ab += dg
		elif i == 2:
			self.phi += dg

	def stepGalaxyParams(self, dg):
		assert(len(dg) == 3)
		self.re  += dg[0]
		self.ab  += dg[1]
		self.phi += dg[2]

	def getGalaxyParams(self):
		return [ self.re, self.ab, self.phi ]

	def setGalaxyParams(self, g):
		assert(len(g) == 3)
		self.re  = g[0]
		self.ab  = g[1]
		self.phi = g[2]

	# [pos], [flux], re, ab, phi
	def numberOfFitParams(self):
		return (self.pos.getDimension() + self.flux.numberOfFitParams() +
				self.numberOfGalaxyParams())

	# returns [ Patch, Patch, ... ] of length numberOfFitParams().
	def getFitParamDerivatives(self, img):
		pos0 = self.getPosition()
		psteps = pos0.getFitStepSizes(img)

		(px0,py0) = img.getWcs().positionToPixel(pos0)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		patch0 = self.getModelPatch(img, px0, py0)

		derivs = []
		for i in range(len(psteps)):
			posx = pos0.copy()
			posx.stepParam(i, psteps[i])
			(px,py) = img.getWcs().positionToPixel(posx)
			patchx = self.getModelPatch(img, px, py)
			dx = (patchx - patch0) * (1. / psteps[i])
			dx.setName('d(src)/d(pos%i)' % i)
			derivs.append(dx)

		fsteps = self.flux.getFitStepSizes(img)
		for i in range(len(fsteps)):
			fi = self.flux.copy()
			fi.stepParam(i, fsteps[i])
			countsi = img.getPhotoCal().fluxToCounts(fi)
			df = patch0 * ((countsi - counts) / counts / fsteps[i])
			df.setName('d(src)/d(flux%i)' % i)
			derivs.append(df)

		gsteps = self.getGalaxyParamStepSizes(img)
		gnames = self.getGalaxyParamNames()
		for i in range(len(gsteps)):
			galx = self.copy()
			galx.stepGalaxyParam(i, gsteps[i])
			patchx = galx.getModelPatch(img, px0, py0)
			dx = (patchx - patch0) * (1. / gsteps[i])
			dx.setName('d(src)/d(%s)' % (gnames[i]))
			derivs.append(dx)

		return derivs

	# update parameters in this direction
	def stepParams(self, dparams):
		pos = self.getPosition()
		np = pos.getDimension()
		nf = self.flux.numberOfFitParams()
		ng = self.numberOfGalaxyParams()
		assert(len(dparams) == (np + nf + ng))
		dp = dparams[:np]
		pos.stepParams(dp)
		df = dparams[np:np+nf]
		self.flux.stepParams(df)
		dg = dparams[np+nf:]
		self.stepGalaxyParams(dg)

	def getParams(self):
		pp = self.getPosition().getParams()
		pf = self.flux.getParams()
		np = self.getPosition().getDimension()
		assert(len(pp) == np)
		pg = self.getGalaxyParams()
		p = pp + pf + pg
		# ensure that "+" means "append"...
		assert(len(p) == (len(pp) + len(pf) + len(pg)))
		return p

	def setParams(self, p):
		pos = self.getPosition()
		np = pos.getDimension()
		nf = self.flux.numberOfFitParams()
		ng = self.numberOfGalaxyParams()
		assert(len(p) == (np + nf + ng))
		pp = p[:np]
		pos.setParams(pp)
		pf = p[np:np+nf]
		self.flux.setParams(pf)
		pg = p[np+nf:]
		self.setGalaxyParams(pg)


class ExpGalaxy(Galaxy):

	profile = None
	@staticmethod
	def getProfile():
		if ExpGalaxy.profile is None:
			ExpGalaxy.profile = CompiledProfile(modelname='exp',
												profile_func=profile_exp, re=100, nrad=4)
		return ExpGalaxy.profile
	#pdev = CompiledProfile(modelname='deV', profile_func=profile_dev, re=100, nrad=8)

	def __init__(self, pos, flux, re, ab, phi):
		Galaxy.__init__(self, pos, flux, re, ab, phi)
		self.name = 'ExpGalaxy'

	def copy(self):
		return ExpGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

	def getGalaxyPatch(self, img, cx, cy):
		# remember to include margin for psf conv
		profile = ExpGalaxy.getProfile()
		re = max(1./30, self.re)
		(H,W) = img.shape
		# MAGIC
		margin = 5.
		(prof,x0,y0) = profile.sample(re, self.ab, self.phi, cx, cy, W, H, margin)
		return Patch(x0, y0, prof)


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

	def changeSource(self, source):
		'''
		Proposes a list of alternatives, where each is a lists of new
		Sources that the given Source could be changed into.
		'''
		if isinstance(source, PointSource):
			eg = ExpGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   5., 0.3, 120.)
			print 'Changing:'
			print '  from ', source
			print '  into', eg
			return [ [eg] ]

		elif isinstance(source, ExpGalaxy):
			return []
		else:
			print 'unknown source type for', source
			return []




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
		

def testGalaxy():
	W,H = 200,200
	pos = PixPos([100, 100])
	flux = Flux(1000.)

	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1.
	wcs = NullWCS()
	photocal = NullPhotoCal()
	psf = NGaussianPSF([1.5], [1.0])
	sky = 0.
	
	img = Image(data=image, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
				photocal=photocal)
	
	eg = ExpGalaxy(pos, flux, 10., 0.5, 30.)
	patch = eg.getModelPatch(img)

	imargs1 = dict(interpolation='nearest', origin='lower')

	plt.clf()
	plt.imshow(patch.getImage(), **imargs1)
	plt.colorbar()
	plt.savefig('eg-1.png')
	
	derivs = eg.getFitParamDerivatives(img)
	for i,deriv in enumerate(derivs):
		plt.clf()
		plt.imshow(deriv.getImage(), **imargs1)
		plt.colorbar()
		plt.title('derivative ' + deriv.getName())
		plt.savefig('eg-deriv%i-0a.png' % i)
		

def main():
	#testGalaxy()
	#return

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
	# Smallish galaxy (gets fit by ~10 stars)
	x0,x1,y0,y1 = 200,500, 1100,1400

	# A sparse field with a small galaxy
	# (gets fit by only one star)
	# (needs about 20 srcs)
	#x0,x1,y0,y1 = 0,300, 200,500

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

	nziv = np.sum(invvar != 0)
	print 'Non-zero invvars:', nziv

	Nsrc = 10
	#steps = (['plots'] + ['source']*Nsrc + ['plots'] + ['psf'] + ['plots'] + ['psf2'])*3 + ['plots'] + ['break']

	#steps = (['plots'] + (['source']*5 + ['plots'])*Nsrc + ['psf'] + ['plots'])

	#steps = (['plots'] + (['source']*5 + ['plots'])*Nsrc)
	#steps = (['plots'] + ['source']*5 + ['plots'] + ['change', 'plots'])
	#steps = (['source']*5 + ['change', 'plots'])

	steps = (['plots'] + (['source']*5 + ['plots', 'save', 'change',
										  'plots', 'save'])*10)
	print 'steps:', steps

	chiArange = None

	ploti = 0
	savei = 0
	stepi = 0

	# JUMP IN:
	if True:
		loadi = 6
		(savei, step, ploti, tractor.catalog) = unpickle_from_file('catalog-%02i.pickle' % loadi)


	#for i,step in enumerate(steps):
	for stepi,step in zip(range(stepi, len(steps)), steps[stepi:]):

		if step == 'plots':
			print 'Making plots...'
			NS = len(tractor.getCatalog())

			chis = tractor.getChiImages()
			chi = chis[0]

			tt = 'sources: %i, chi^2/pix = %g' % (NS, np.sum(chi**2)/float(nziv))

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
				plt.savefig('create-%02i.png' % stepi)

		elif step == 'psf':
			baton = (stepi,)
			tractor.optimizeAllPsfAtFixedComplexityStep()
			#derivCallback=(psfDerivCallback, baton))

		elif step == 'psf2':
			tractor.increaseAllPsfComplexity()

		elif step == 'change':

			def altCallback(tractor, src, newsrcs, srci, alti):
				#global ploti
				print 'alt callback'
				plt.clf()
				plt.imshow(tractor.getModelImages()[0],
						   interpolation='nearest', origin='lower',
						   vmin=zrange[0], vmax=zrange[1])
				plt.hot()
				plt.colorbar()
				plt.savefig('alt-%i-%i-%i.png' % (ploti, srci, alti))

			#tractor.changeSourceTypes(altCallback=altCallback)
			tractor.changeSourceTypes()

		elif step == 'save':

			pickle_to_file((savei, stepi, ploti, tractor.catalog),
						   'catalog-%02i.pickle' % savei)

			savei += 1
			
		print 'Tractor cache has', len(tractor.cache), 'entries'


if __name__ == '__main__':
	main()
