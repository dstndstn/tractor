if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

from math import pi, sqrt

import pyfits
import pylab as plt
import numpy as np
import matplotlib

from tractor import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.ngc2000 import ngc2000

from compiled_profiles import *
from galaxy_profiles import *

class SdssWcs(WCS):
	def __init__(self, astrans):
		self.astrans = astrans

	# RA,Dec in deg to pixel x,y.
	def positionToPixel(self, src, pos):
		## FIXME -- color.
		x,y = self.astrans.radec_to_pixel(pos.ra, pos.dec)
		#print 'astrans: ra,dec', (pos.ra, pos.dec), '--> x,y', (x,y)
		#print 'HIDEOUS WORKAROUND'
		x,y = x[0],y[0]
		return x,y

	def pixelToRaDec(self, x, y):
		ra,dec = self.astrans.pixel_to_radec(x, y)
		#print 'astrans: x,y', (x,y), '--> ra,dec', (ra,dec)
		#print 'HIDEOUS WORKAROUND'
		ra,dec = ra[0], dec[0]
		return ra,dec
	
	# (x,y) to RA,Dec in deg
	def pixelToPosition(self, src, xy):
		## FIXME -- color.
		## NOTE, "src" may be None.
		(x,y) = xy
		ra,dec = self.pixelToRaDec(x, y)
		return RaDecPos(ra, dec)


class GalaxyShape(ParamList):
	def getNamedParams(self):
		return [('re', 0), ('ab', 1), ('phi', 2)]
	def hashkey(self):
		return ('GalaxyShape',) + tuple(self.vals)
	def __repr__(self):
		return 're=%g, ab=%g, phi=%g' % (self.re, self.ab, self.phi)
	def __str__(self):
		return 're=%.1f, ab=%.2f, phi=%.1f' % (self.re, self.ab, self.phi)
	def copy(self):
		return GalaxyShape(*self.vals)
	def getParamNames(self):
		return ['re', 'ab', 'phi']

	def getStepSizes(self, img):
		abstep = 0.01
		if self.ab >= (1 - abstep):
			abstep = -abstep
		return [ 1., abstep, 1. ]
	

class Galaxy(MultiParams):
	def __init__(self, pos, flux, shape):
		self.name = 'Galaxy'
		MultiParams.__init__(self, pos, flux, shape)

	def getPosition(self):
		return self.pos

	def getFlux(self):
		return self.flux

	def getNamedParams(self):
		return [('pos', 0), ('flux', 1), ('shape', 2)]

	def __getattr__(self, name):
		if name in ['re', 'ab', 'phi']:
			return getattr(self.shape, name)
		return MultiParams.__getattr__(self, name)

	def hashkey(self):
		return (self.name, self.pos.hashkey(), self.flux.hashkey(),
				self.re, self.ab, self.phi)
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

	def getProfile(self):
		return None

	def getGalaxyPatch(self, img, cx, cy):
		# remember to include margin for psf conv
		profile = self.getProfile()
		if profile is None:
			return None
		re = max(1./30, self.re)
		(H,W) = img.shape
		margin = int(ceil(img.getPsf().getRadius()))
		(prof,x0,y0) = profile.sample(re, self.ab, self.phi, cx, cy, W, H, margin)
		return Patch(x0, y0, prof)

	def getModelPatch(self, img, px=None, py=None):
		if px is None or py is None:
			(px,py) = img.getWcs().positionToPixel(self, self.getPosition())
		patch = self.getGalaxyPatch(img, px, py)
		if patch is None:
			print 'Warning, is Galaxy(subclass).getProfile() defined?'
			return Patch(0, 0, None)
		if patch.getImage() is None:
			return Patch(patch.getX0(), patch.getY0(), None)
		psf = img.getPsf()
		convimg = psf.applyTo(patch.getImage())
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		#print 'PSF-convolved'
		#self.debugPatchImage(convimg)
		return Patch(patch.getX0(), patch.getY0(), convimg * counts)

	# returns [ Patch, Patch, ... ] of length numberOfParams().
	def getParamDerivatives(self, img):
		pos0 = self.getPosition()
		(px0,py0) = img.getWcs().positionToPixel(self, pos0)
		counts = img.getPhotoCal().fluxToCounts(self.flux)
		patch0 = self.getModelPatch(img, px0, py0)
		derivs = []
		psteps = pos0.getStepSizes(img)
		for i in range(len(psteps)):
			posx = pos0.copy()
			posx.stepParam(i, psteps[i])
			(px,py) = img.getWcs().positionToPixel(self, posx)
			patchx = self.getModelPatch(img, px, py)
			dx = (patchx - patch0) * (1. / psteps[i])
			dx.setName('d(gal)/d(pos%i)' % i)
			derivs.append(dx)
		fsteps = self.flux.getStepSizes(img)
		for i in range(len(fsteps)):
			fi = self.flux.copy()
			fi.stepParam(i, fsteps[i])
			countsi = img.getPhotoCal().fluxToCounts(fi)
			df = patch0 * ((countsi - counts) / counts / fsteps[i])
			df.setName('d(gal)/d(flux%i)' % i)
			derivs.append(df)
		gsteps = self.shape.getStepSizes(img)
		gnames = self.shape.getParamNames()
		oldvals = self.shape.getParams()
		for i in range(len(gsteps)):
			self.shape.stepParam(i, gsteps[i])
			patchx = self.getModelPatch(img, px0, py0)
			self.shape.setParams(oldvals)
			dx = (patchx - patch0) * (1. / gsteps[i])
			dx.setName('d(gal)/d(%s)' % (gnames[i]))
			derivs.append(dx)

		return derivs


class ExpGalaxy(Galaxy):
	profile = None
	@staticmethod
	def getExpProfile():
		if ExpGalaxy.profile is None:
			ExpGalaxy.profile = (
				CompiledProfile(modelname='exp',
								profile_func=profile_exp, re=100, nrad=4))
		return ExpGalaxy.profile

	expnum = 0

	def __init__(self, pos, flux, re, ab, phi):
		Galaxy.__init__(self, pos, flux, GalaxyShape(re, ab, phi))
		self.name = 'ExpGalaxy'
		self.num = ExpGalaxy.expnum
		ExpGalaxy.expnum += 1
		self.plotnum = 0

	def getProfile(self):
		return ExpGalaxy.getExpProfile()

	def copy(self):
		return ExpGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)

	def debugPatchImage(self, img):
		if img is None:
			print 'Exp patch', img
		elif np.product(img.shape) == 0:
			print 'Patch empty:', img.shape
		else:
			print 'Patch', img.shape
			plt.clf()
			plt.imshow(img, interpolation='nearest', origin='lower')
			plt.hot()
			plt.colorbar()
			fn = 'exp-patch-%02i-%03i.png' % (self.num, self.plotnum)
			plt.savefig(fn)
			print 'saved', fn
			self.plotnum += 1

class DevGalaxy(Galaxy):
	profile = None
	@staticmethod
	def getDevProfile():
		if DevGalaxy.profile is None:
			DevGalaxy.profile = (
				CompiledProfile(modelname='exp',
								profile_func=profile_dev, re=100, nrad=8))
		return DevGalaxy.profile

	def __init__(self, pos, flux, re, ab, phi):
		Galaxy.__init__(self, pos, flux, GalaxyShape(re, ab, phi))
		self.name = 'DevGalaxy'

	def getProfile(self):
		return DevGalaxy.getDevProfile()

	def copy(self):
		return DevGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)



class SDSSTractor(Tractor):

	def createNewSource(self, img, x, y, ht):
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		#patch = img.getPsf().getPointSourcePatch(x, y)
		#print 'psf patch:', patch.shape
		#print 'psf patch: max', patch.max(), 'sum', patch.sum()
		#print 'new source peak height:', ht, '-> flux', ht/patch.max()
		#ht /= patch.max()

		wcs = img.getWcs()
		pos = wcs.pixelToPosition(None, (x,y))

		return PointSource(pos, Flux(ht))

	def changeSource(self, source):
		'''
		Proposes a list of alternatives, where each alternative is a list of new
		Sources that the given Source could be changed into.
		'''
		if isinstance(source, PointSource):
			eg = ExpGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   1., 0.5, 0.)
			dg = DevGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   1., 0.5, 0.)
			#print 'Changing:'
			#print '  from ', source
			#print '  into', eg
			return [ [], [eg], [dg] ]

		elif isinstance(source, ExpGalaxy):
			dg = DevGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getFlux().copy())
			return [ [], [ps], [dg] ]

		elif isinstance(source, DevGalaxy):
			eg = ExpGalaxy(source.getPosition().copy(), source.getFlux().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getFlux().copy())
			return [ [], [ps], [eg] ]

		else:
			print 'unknown source type for', source
			return []


def choose_field2():
	import astrometry.libkd.spherematch as spherematch

	fields = fits_table('window_flist.fits')
	print len(ngc2000), 'NGC/IC objects'
	goodngc = [n for n in ngc2000 if n.get('classification', None) == 'Gx']
	print len(goodngc), 'NGC galaxies'

	nra  = np.array([n['ra']  for n in goodngc])
	ndec = np.array([n['dec'] for n in goodngc])

	rad = 8./60.
	(I,J,dist) = spherematch.match_radec(nra, ndec, fields.ra, fields.dec, rad)

	#sdss = DR7()

	for i in np.unique(I):
		ii = (I == i)
		n = goodngc[i]
		isngc = n['is_ngc']
		num = n['id']
		if (sum(ii) > 10) & (n['dec'] > 2.0):
			print '<p>'
			print ('NGC' if isngc else 'IC'), num, 'has', sum(ii), 'fields, and is at RA = ', n['ra'], 'Dec=', n['dec']
			print '</p>'

			ff = fields[J[ii]]
			print 'rcfi = [',
			for f in ff:
				print '(', f.run, ',', f.camcol, ',', f.field, ',', f.incl, ')',
			print ']'
			for f in ff:
				print 'Incl', f.incl, '<br />'
				print ('<img src="%s" /><br /><br />' %
					   ('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpegcodec.aspx?R=%i&C=%i&F=%i&Z=25' % (f.run, f.camcol, f.field)))
				print '<br />'

	# NGC 2511 has 12 fields, and is at RA = 120.575 Dec= 9.4 

def testGalaxy():
	W,H = 200,200
	pos = PixPos(100, 100)
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
	
	derivs = eg.getParamDerivatives(img)
	for i,deriv in enumerate(derivs):
		plt.clf()
		plt.imshow(deriv.getImage(), **imargs1)
		plt.colorbar()
		plt.title('derivative ' + deriv.getName())
		plt.savefig('eg-deriv%i-0a.png' % i)
		

def main():
	from optparse import OptionParser

	#testGalaxy()
	#choose_field2()

	parser = OptionParser()
	parser.add_option('-l', '--load', dest='loadi', type='int',
					  default=-1, help='Load catalog from step #...')
	parser.add_option('-v', '--verbose', dest='verbose', action='count',
					  help='Make more verbose')
	opt,args = parser.parse_args()

	if opt.verbose == 0:
		lvl = logging.INFO
	else: # opt.verbose == 1:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl)

	rcfi = [ ( 5194 , 2 , 44 , 22.500966 ), ( 4275 , 2 , 224 , 90.003437 ), ( 3638 , 2 , 209 , 90.002781 ), ( 4291 , 2 , 227 , 90.003589 ), ( 4275 , 2 , 225 , 90.003437 ), ( 5849 , 4 , 27 , 20.003216 ), ( 5803 , 5 , 41 , 19.990683 ), ( 5194 , 2 , 43 , 22.500966 ), ( 3638 , 2 , 210 , 90.002781 ), ( 5803 , 5 , 42 , 19.990683 ), ( 5925 , 5 , 30 , 19.933986 ), ( 5935 , 5 , 27 , 20.000022 ), ]			

	rcf = [(r,c,f) for r,c,f,i in rcfi if i < 85]
	print 'RCF', rcf

	sdss = DR7()

	bandname = 'i'

	if False:
		from astrometry.util import sdss_das as das
		from astrometry.util.sdss_filenames import sdss_filename
		for r,c,f in rcf:
			for filetype in ['fpC', 'fpM', 'psField', 'tsField']:
				fn = sdss_filename(filetype, r, c, f, band=bandname)
				print 'Need', fn
				#if not os.path.exists(fn):
				print 'Getting from DAS'
				das.sdss_das_get(filetype, fn, r, c, f, band=bandname)

	# we only got some of them...
	rcf = [ (5194, 2, 44), (5194, 2, 43), (5849, 4, 27), (5935, 5, 27) ]
	rcf = [rcf[0], rcf[2]]
	print 'RCF', rcf

	print 'Reading SDSS input files...'

	band = band_index(bandname)

	images = []
	zrange = []
	nziv = []

	# FIXME -- bug-bug annihilation
	rerun = 0
	
	for i,(run,camcol,field) in enumerate(rcf):
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

		tsfield = sdss.readTsField(run, camcol, field, rerun)

		invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

		nz = np.sum(invvar != 0)
		print 'Non-zero invvars:', nz
		nziv.append(nz)
		
		zr = np.array([-3.,+10.]) * skysig + sky
		zrange.append(zr)

		print 'Initial plots...'
		plt.clf()
		plt.imshow(image, interpolation='nearest', origin='lower',
				   vmin=zr[0], vmax=zr[1])
		plt.hot()
		plt.colorbar()
		ax = plt.axis()
		plt.axis(ax)
		#plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'b-')
		plt.savefig('fullimg-%02i.png' % i)

		#roi = (slice(y0,y1), slice(x0,x1))
		#image = image[roi]
		#invvar = invvar[roi]

		dgpsf = psfield.getDoubleGaussian(band)
		print 'Creating double-Gaussian PSF approximation'
		print '  ', dgpsf
		(a,s1, b,s2) = dgpsf
		psf = NGaussianPSF([s1, s2], [a, b])

		# We'll start by working in pixel coords
		wcs = SdssWcs(tsfield.getAsTrans(bandname))
		# And counts
		photocal = NullPhotoCal()

		img = Image(data=image, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
					photocal=photocal,
					name='Image%i(r/c/f=%i/%i%i)' % (i, run, camcol, field))
		images.append(img)

	print 'Creating footprint image...'
	radecs = []
	for img in images:
		wcs = img.getWcs()
		(H,W) = img.shape
		corners = [ (0,0), (W,0), (W,H), (0,H), (0,0) ]
		rds = []
		for x,y in corners:
			rds.append(wcs.pixelToRaDec(x,y))
		radecs.append(rds)
	plt.clf()
	# FIXME -- RA=0 wrap-around
	for rd in radecs:
		plt.plot([r for r,d in rd], [d for r,d in rd], 'b-')
		plt.gca().add_artist(matplotlib.patches.Polygon(rd, ec='b', fill=True, alpha=0.3))
	plt.xlabel('RA (deg)')
	ax = plt.axis()
	plt.xlim(ax[1], ax[0])
	plt.ylabel('Dec (deg)')
	plt.savefig('footprints.png')

	
	print 'Firing up tractor...'
	tractor = SDSSTractor(images)

	print 'Start: catalog is', tractor.catalog
	
	#steps = ['source']*5 + ['plots', 'change', 'plots']
	steps = ['plots'] + ['source', 'plots']*5 + ['change', 'plots']
	ploti = 0
	savei = 0
	stepi = 0

	chiAimargs = []

	for stepi,step in zip(range(stepi, len(steps)), steps[stepi:]):

		if step == 'plots':
			print 'Making plots...'
			NS = len(tractor.getCatalog())

			chis = tractor.getChiImages()
			mods = tractor.getModelImages()

			for i in range(len(chis)):
				chi = chis[i]
				mod = mods[i]
				img = tractor.getImage(i)
				tt = 'sources: %i, chi^2/pix = %g' % (NS, np.sum(chi**2)/float(nziv[i]))
				zr = zrange[i]
				imargs = dict(interpolation='nearest', origin='lower',
							  vmin=zr[0], vmax=zr[1])

				plt.clf()
				plt.imshow(mod, **imargs)
				plt.hot()
				plt.colorbar()
				ax = plt.axis()
				wcs = img.getWcs()
				x = []
				y = []
				for src in tractor.getCatalog():
					pos = src.getPosition()
					px,py = wcs.positionToPixel(src, pos)
					x.append(px)
					y.append(py)
				plt.plot(x, y, 'bo', mfc='none', mec='b')
				plt.axis(ax)
				plt.title(tt)
				fn = 'mod-%02i-%02i.png' % (ploti, i)
				plt.savefig(fn)
				print 'Wrote', fn

				if len(chiAimargs) <= i:
					mn,mx = (chi.min(), chi.max())
					chiAimargs.append(
						dict(interpolation='nearest', origin='lower',
							 vmin=mn, vmax=mx))
				chiAimarg = chiAimargs[i]

				plt.clf()
				plt.imshow(chi, **chiAimarg)
				plt.hot()
				plt.colorbar()
				plt.title(tt)
				fn = 'chiA-%02i-%02i.png' % (ploti, i)
				plt.savefig(fn)
				print 'Wrote', fn

				chiBimarg = dict(interpolation='nearest', origin='lower',
								 vmin=-3, vmax=10.)

				plt.clf()
				plt.imshow(chi, **chiBimarg)
				plt.hot()
				plt.colorbar()
				plt.title(tt)
				fn = 'chiB-%02i-%02i.png' % (ploti, i)
				plt.savefig(fn)
				print 'Wrote', fn

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

			pickle_to_file((savei, stepi+1, ploti, tractor.catalog),
						   'catalog-%02i.pickle' % savei)
			savei += 1
			
		print 'Tractor cache has', len(tractor.cache), 'entries'

if __name__ == '__main__':
	main()
