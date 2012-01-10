# Copyright 2011 Dustin Lang and David W. Hogg.  All rights reserved.
import os
from math import pi, sqrt, ceil, floor
from datetime import datetime

import pyfits
import pylab as plt
import numpy as np
import matplotlib

from engine import *
from sdss_galaxy import *

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.ngc2000 import ngc2000
from astrometry.util.plotutils import setRadecAxes, redgreen

# might want to override this to set the step size to ~ a pixel
#class SdssRaDecPos(RaDecPos):
#	def getStepSizes(self, img):
#		return [1e-4, 1e-4]

def _check_sdss_files(sdss, run, camcol, field, bandname, filetypes,
					  retrieve=True):
	bandnum = band_index(bandname)
	for filetype in filetypes:
		fn = sdss.getFilename(filetype, run, camcol, field, bandname)
		print 'Looking for file', fn
		if not os.path.exists(fn):
			if retrieve:
				print 'Retrieving', fn
				sdss.retrieve(filetype, run, camcol, field, bandnum)
			else:
				raise os.OSError('no such file: "%s"' % fn)


def get_tractor_sources(run, camcol, field, bandname, release='DR7',
						retrieve=True, curl=False, roi=None):
	'''
	Creates tractor.Source objects corresponding to objects in the SDSS catalog
	for the given field.

	'''
	if release != 'DR7':
		raise RuntimeError('We only support DR7 currently')
	# FIXME
	rerun = 0

	sdss = DR7(curl=curl)
	bandnum = band_index(bandname)
	_check_sdss_files(sdss, run, camcol, field, bandnum,
					  ['tsObj', 'tsField'],
					  #fpC', 'tsField', 'psField', 'fpM'],
					  retrieve=retrieve)

	tsf = sdss.readTsField(run, camcol, field, rerun)


	objs = fits_table(sdss.getFilename('tsObj', run, camcol, field,
									   bandname, rerun=rerun))
	objs.indices = np.arange(len(objs))

	if roi is not None:
		x0,x1,y0,y1 = roi
		# HACK -- keep only the sources whose centers are within the ROI box.
		x = objs.colc[:,bandnum]
		y = objs.rowc[:,bandnum]
		I = ((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
		objs = objs[I]

	objs = objs[(objs.nchild == 0)]

	# NO IDEA why it is NOT necessary to get PA and adjust for it.
	# (probably that getTensor() has the phi transformation in the wrong
	# place, terrifying)
	# Since in DR7, tsObj files have phi_exp, phi_dev in image coordinates,
	# not sky coordinates.
	# Should have to Correct by finding the position angle of the field on
	# the sky.
	# cd = wcs.cdAtPixel(W/2, H/2)
	# pa = np.rad2deg(np.arctan2(cd[0,1], cd[0,0]))
	# print 'pa=', pa
	# HACK -- DR7 phi opposite to Tractor phi, apparently
	objs.phi_dev = -objs.phi_dev
	objs.phi_exp = -objs.phi_exp

	# MAGIC -- minimum size of galaxy.
	objs.r_dev = np.maximum(objs.r_dev, 1./30.)
	objs.r_exp = np.maximum(objs.r_exp, 1./30.)

	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Lgal  = (objs.prob_psf[:,bandnum] == 0)
	Ldev = Lgal * objs.fracpsf[:,bandnum]
	Lexp = Lgal * (1. - objs.fracpsf[:,bandnum])

	sources = []
	ikeep = []

	# Add stars
	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'
	for i in I:
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		lup = objs.psfcounts[i,bandnum]
		counts = tsf.luptitude_to_counts(lup, bandnum)
		if counts <= 0:
			print 'Skipping star with luptitude', lup, '-> counts', counts
			continue
		flux = SdssFlux(counts / SdssPhotoCal.scale)
		ps = PointSource(pos, flux)
		sources.append(ps)
		ikeep.append(i)

	# Add galaxies.
	I = np.flatnonzero(Lgal > 0)
	print len(I), 'galaxies'
	ndev, nexp, ncomp = 0, 0, 0
	for i in I:
		hasdev = (Ldev[i] > 0)
		hasexp = (Lexp[i] > 0)
		iscomp = (hasdev and hasexp)
		pos = RaDecPos(objs.ra[i], objs.dec[i])
		if iscomp:
			lups = objs.counts_model[i,bandnum]
		elif hasdev:
			lups = objs.counts_dev[i,bandnum]
		elif hasexp:
			lups = objs.counts_exp[i,bandnum]
		counts = tsf.luptitude_to_counts(lups, bandnum)
		if counts <= 0:
			print 'Skipping galaxy with luptitude', lups, '-> counts', counts
			continue
											 
		if hasdev:
			dcounts = counts * Ldev[i]
			dflux = SdssFlux(dcounts / SdssPhotoCal.scale)
			re = objs.r_dev[i,bandnum]
			ab = objs.ab_dev[i,bandnum]
			phi = objs.phi_dev[i,bandnum]
			dshape = GalaxyShape(re, ab, phi)
		if hasexp:
			ecounts = counts * Lexp[i]
			eflux = SdssFlux(ecounts / SdssPhotoCal.scale)
			re = objs.r_exp[i,bandnum]
			ab = objs.ab_exp[i,bandnum]
			phi = objs.phi_exp[i,bandnum]
			eshape = GalaxyShape(re, ab, phi)

		if iscomp:
			gal = CompositeGalaxy(pos, eflux, eshape, dflux, dshape)
			ncomp += 1
		elif hasdev:
			#print 'pure deV; counts_model = %g; counts_dev = %g' % (
			#	objs.counts_model[i, bandnum], objs.counts_dev[i, bandnum])
			gal = DevGalaxy(pos, dflux, dshape)
			ndev += 1
		elif hasexp:
			#print 'pure exp; counts_model = %g; counts_exp = %g' % (
			#	objs.counts_model[i, bandnum], objs.counts_exp[i, bandnum])
			gal = ExpGalaxy(pos, eflux, eshape)
			nexp += 1
		sources.append(gal)
		ikeep.append(i)
	print 'Created', ndev, 'pure deV', nexp, 'pure exp and',
	print ncomp, 'composite galaxies'

	# if you want to cut the objs list to just the ones for which sources were created...
	ikeep = np.unique(ikeep)
	objs = objs[ikeep]

	return sources

	
def get_tractor_image(run, camcol, field, bandname, release='DR7',
					  retrieve=True, curl=False, roi=None,
					  psf='kl-gm'):
	'''
	Creates a tractor.Image given an SDSS field identifier.

	If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
	in the image, in zero-indexed pixel coordinates.  x1,y1 are
	NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.

	psf can be:
	  "dg" for double-Gaussian
	  "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

	Returns:
	  (tractor.Image, dict)

	dict contains useful details like:
	  'sky'
	  'skysig'
	'''
	# Ugly
	if release != 'DR7':
		raise RuntimeError('We only support DR7 currently')
	valid_psf = ['dg', 'kl-gm']
	if psf not in valid_psf:
		raise RuntimeError('PSF must be in ' + str(valid_psf))
	# FIXME
	rerun = 0

	bandnum = band_index(bandname)

	sdss = DR7(curl=curl)
	_check_sdss_files(sdss, run, camcol, field, bandname,
					  ['fpC', 'tsField', 'psField', 'fpM'],
					  retrieve=retrieve)
	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias
	image = fpC
	(H,W) = image.shape

	if roi is None:
		x0 = y0 = 0
	else:
		x0,x1,y0,y1 = roi

	tsf = sdss.readTsField(run, camcol, field, rerun)
	astrans = tsf.getAsTrans(bandnum)
	wcs = SdssWcs(astrans)
	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

	photocal = SdssPhotoCal()
	psfield = sdss.readPsField(run, camcol, field)
	sky = psfield.getSky(bandnum)
	skysig = sqrt(sky)
	skyobj = ConstantSky(sky)
	info = dict(sky=sky, skysig=skysig)

	fpM = sdss.readFpM(run, camcol, field, bandname)

	gain = psfield.getGain(bandnum)
	darkvar = psfield.getDarkVariance(bandnum)
	skyerr = psfield.getSkyErr(bandnum)
	invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

	if roi is not None:
		roislice = (slice(y0,y1), slice(x0,x1))
		image = image[roislice]
		invvar = invvar[roislice]

	if psf == 'dg':
		dgpsf = psfield.getDoubleGaussian(bandnum)
		print 'Creating double-Gaussian PSF approximation'
		(a,s1, b,s2) = dgpsf
		mypsf = NCircularGaussianPSF([s1, s2], [a, b])
	elif psf == 'kl-gm':
		from emfit import em_fit_2d
		from fitpsf import em_init_params
		
		# Create Gaussian mixture model PSF approximation.
		H,W = image.shape
		klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
		S = klpsf.shape[0]
		# number of Gaussian components
		K = 3
		w,mu,sig = em_init_params(K, None, None, None)
		II = klpsf.copy()
		II /= II.sum()
		# HIDEOUS HACK
		II = np.maximum(II, 0)
		print 'Multi-Gaussian PSF fit...'
		xm,ym = -(S/2), -(S/2)
		em_fit_2d(II, xm, ym, w, mu, sig)
		print 'w,mu,sig', w,mu,sig
		mypsf = GaussianMixturePSF(w, mu, sig)

	timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
					   (run, camcol, field, bandname)))
	return timg,info
		

class SdssPhotoCal(object):
	scale = 1e6
	def __init__(self, scale=None):
		if scale is None:
			scale = SdssPhotoCal.scale
		self.scale = scale
	def brightnessToCounts(self, brightness):
		'''
		brightness: SdssFlux object
		returns: float
		'''
		return brightness.getValue() * self.scale
	def countsToBrightness(self, counts):
		'''
		counts: float
		Returns: duck-typed Brightness object (SdssFlux)
		'''
		return SdssFlux(counts / self.scale)

class SdssFlux(Flux):
	def getStepSizes(self, img):
		return [1.]
	def __str__(self):
		return 'SdssFlux: %.1f' % (self.val * SdssPhotoCal.scale)
	def __repr__(self):
		return 'SdssFlux(%.1f)' % (self.val * SdssPhotoCal.scale)
	def hashkey(self):
		return ('SdssFlux', self.val)
	def copy(self):
		return SdssFlux(self.val)
	def __add__(self, other):
		assert(isinstance(other, SdssFlux))
		return SdssFlux(self.val + other.val)

class SdssWcs(WCS):
	def __init__(self, astrans):
		self.astrans = astrans
		self.x0 = 0
		self.y0 = 0

	def hashkey(self):
		return ('SdssWcs', self.x0, self.y0, self.astrans)

	def setX0Y0(self, x0, y0):
		self.x0 = x0
		self.y0 = y0

	# This function is not used by the tractor, and it works in
	# *original* pixel coords (no x0,y0 offsets)
	# (x,y) to RA,Dec in deg
	def pixelToRaDec(self, x, y):
		ra,dec = self.astrans.pixel_to_radec(x, y)
		return ra,dec

	def cdAtPixel(self, x, y):
		return self.astrans.cd_at_pixel(x + self.x0, y + self.y0)

	# RA,Dec in deg to pixel x,y.
	def positionToPixel(self, src, pos):
		## FIXME -- color.
		x,y = self.astrans.radec_to_pixel(pos.ra, pos.dec)
		return x - self.x0, y - self.y0

	# (x,y) to RA,Dec in deg
	def pixelToPosition(self, src, xy):
		## FIXME -- color.
		## NOTE, "src" may be None.
		(x,y) = xy
		ra,dec = self.pixelToRaDec(x + self.x0, y + self.y0)
		return RaDecPos(ra, dec)

class Changes(object):
	pass

class SDSSTractor(Tractor):

	def __init__(self, *args, **kwargs):
		self.debugnew = kwargs.pop('debugnew', False)
		self.debugchange = kwargs.pop('debugchange', False)

		Tractor.__init__(self, *args, **kwargs)
		self.newsource = 0
		self.changes = []
		self.changei = 0

		self.plotfns = []
		self.comments = []
		self.boxes = []

	def debugChangeSources(self, **kwargs):
		if self.debugchange:
			self.doDebugChangeSources(**kwargs)

	def doDebugChangeSources(self, step=None, src=None, newsrcs=None, alti=0,
							 dlnprob=0, **kwargs):
		if step == 'start':
			ch = self.changes = Changes()
			N = self.getNImages()
			ch.src = src
			ch.N = N
			ch.impatch = [None for i in range(N)]
			ch.mod0    = [None for i in range(N)]
			ch.mod0type = src.getSourceType()
			ch.newmods = []

			for imgi in range(N):
				img = self.getImage(imgi)
				mod = self.getModelPatch(img, src)
				ch.mod0[imgi] = mod
				print 'image', imgi, 'got model patch', mod
				if mod.getImage() is not None:
					impatch = img.getImage()[mod.getSlice(img)]
					if len(impatch.ravel()):
						ch.impatch[imgi] = impatch

		elif step in ['init', 'opt1']:
			ch = self.changes
			if newsrcs == []:
				return
			mods = []
			for imgi in range(ch.N):
				img = self.getImage(imgi)
				mod = self.getModelPatch(img, newsrcs[0])
				mods.append(mod)

			if step == 'init':
				ch.newmods.append([newsrcs[0].getSourceType(),
								   mods])
			else:
				ch.newmods[-1].extend([mods, dlnprob])

		elif step in ['switch', 'keep']:
			ch = self.changes
			M = len(ch.newmods)
			N = ch.N
			cols = M+2
			II = [i for i in range(N) if ch.impatch[i] is not None]
			rows = len(II)
			fs = 10

			imargs = {}
			plt.clf()
			# Images
			for ri,i in enumerate(II):
				img = self.getImage(i)
				sky = img.getSky().val
				skysig = sqrt(sky)
				imargs[i] = dict(vmin=-3.*skysig, vmax=10.*skysig)
				if ch.impatch[i] is None:
					continue
				plt.subplot(rows, cols, ri*cols+1)
				plotimage(ch.impatch[i] - sky, **imargs[i])
				plt.xticks([])
				plt.yticks([])
				plt.title('image %i' % i, fontsize=fs)

			# Original sources
			for ri,i in enumerate(II):
				if ch.mod0[i].getImage() is None:
					continue
				plt.subplot(rows, cols, ri*cols+2)
				plotimage(ch.mod0[i].getImage(), **imargs[i])
				plt.xticks([])
				plt.yticks([])
				plt.title('original ' + ch.mod0type, fontsize=fs)

			# New sources
			for j,newmod in enumerate(ch.newmods):
				(srctype, premods, postmods, dlnp) = newmod
				for ri,i in enumerate(II):
					if postmods[i] is None:
						continue
					plt.subplot(rows, cols, ri*cols + 3 + j)

					# HACK -- force patches to be the same size + offset...
					img = self.getImage(i)
					sl = ch.mod0[i].getSlice(img)
					#print 'slice', sl
					im = np.zeros_like(img.getImage())
					postmods[i].addTo(im)
					im = im[sl]
					if len(im.ravel()):
						plotimage(im, **imargs[i])
						plt.xticks([])
						plt.yticks([])
						plt.title(srctype + ' (dlnp=%.1f)' % dlnp, fontsize=fs)
				
			fn = 'change-%03i.png' % self.changei
			plt.savefig(fn)
			print 'Wrote', fn
			self.changei += 1
			self.plotfns.append(fn)
			if step == 'switch':
				s = '<a href="#%s">' % fn + 'accepted change</a> from ' + str(src) + '<br />to '
				if len(newsrcs) == 1:
					s += str(newsrcs[0])
				else:
					s += '[ ' + ' + '.join([str(ns) for ns in newsrcs]) + ' ]'
				self.comments.append(s)
			elif step == 'keep':
				s = '<a href="#%s">' % fn + 'rejected change</a> of ' + str(src)
				self.comments.append(s)
			#smallimg = 'border="0" width="400" height="300"'
			#s += '<a href="%s"><img src="%s" %s /></a>' % (fn, fn, smallimg)
				
				
	def debugNewSource(self, *args, **kwargs):
		if self.debugnew:
			self.doDebugNewSource(*args, **kwargs)

	def doDebugNewSource(self, *args, **kwargs):
		step = kwargs.get('type', None)
		if step in [ 'newsrc-0', 'newsrc-opt' ]:
			if step == 'newsrc-0':
				optstep = 0
				self.newsource += 1
			else:
				optstep = 1 + kwargs['step']
			src = kwargs['src']
			img = kwargs['img']

			patch = src.getModelPatch(img)
			imgpatch = img.getImage()[patch.getSlice(img)]

			plt.clf()
			plt.subplot(2,3,4)
			plotimage(imgpatch)
			cl = plt.gci().get_clim()
			plt.colorbar()
			plt.title('image patch')
			plt.subplot(2,3,5)
			plotimage(patch.getImage(), vmin=cl[0], vmax=cl[1])
			plt.colorbar()
			plt.title('new source')
			derivs = src.getParamDerivatives(img)
			assert(len(derivs) == 3)
			for i,deriv in enumerate(derivs):
				plt.subplot(2,3,i+1)
				plotimage(deriv.getImage())
				cl = plt.gci().get_clim()
				mx = max(abs(cl[0]), abs(cl[1]))
				plt.gci().set_clim(-mx, mx)
				plt.colorbar()
				plt.title(deriv.name)
			fn = 'newsource-%02i-%02i.png' % (self.newsource, optstep)
			plt.savefig(fn)
			print 'Wrote', fn

	def createNewSource(self, img, x, y, ht):
		wcs = img.getWcs()
		pos = wcs.pixelToPosition(None, (x,y))
		# "ht" is the peak height (difference between image and model)
		# convert to total flux by normalizing by my patch's peak pixel value.
		patch = img.getPsf().getPointSourcePatch(x, y)
		ht /= patch.getImage().max()
		photocal = img.getPhotoCal()
		flux = photocal.countsToBrightness(ht)
		ps = PointSource(pos, flux)
		try:
			imgi = self.images.index(img)
			patch = self.getModelPatch(img, ps)
			self.addBox(imgi, patch.getExtent())
		except:
			pass
		return ps

	def addBox(self, imgi, box):
		if len(self.boxes) == 0:
			self.boxes = [[] for i in range(self.getNImages())]
		self.boxes[imgi].append(box)

	def changeSourceTypes(self, srcs=None, **kwargs):
		if srcs is not None:
			for i,img in enumerate(self.getImages()):
				for src in srcs:
					patch = self.getModelPatch(img, src)
					self.addBox(i, patch.getExtent())
		Tractor.changeSourceTypes(self, srcs, **kwargs)


	def changeSource(self, source):
		'''
		Proposes a list of alternatives, where each alternative is a list of new
		Sources that the given Source could be changed into.
		'''
		if isinstance(source, PointSource):
			eg = ExpGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   1., 0.5, 0.)
			dg = DevGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   1., 0.5, 0.)
			#print 'Changing:'
			#print '  from ', source
			#print '  into', eg
			return [ [], [eg], [dg] ]

		elif isinstance(source, ExpGalaxy):
			dg = DevGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getBrightness().copy())
			return [ [], [ps], [dg] ]

		elif isinstance(source, DevGalaxy):
			eg = ExpGalaxy(source.getPosition().copy(), source.getBrightness().copy(),
						   source.re, source.ab, source.phi)
			ps = PointSource(source.getPosition().copy(), source.getBrightness().copy())
			return [ [], [ps], [eg] ]

		else:
			print 'unknown source type for', source
			return []

