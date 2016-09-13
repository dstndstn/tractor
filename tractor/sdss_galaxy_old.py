# Copyright 2011 Dustin Lang and David W. Hogg.  All rights reserved.
from __future__ import print_function

'''
This is an old, flawed, attempt by dstn to model the galaxy
appearance above the atmosphere (by sampling rectangles in a precompiled
profile using the integral image trick) then convolve.

Here for historical interest, and because dstn doesn't *really* believe in
svn rm'ing junk :)
'''

from compiled_profiles import *
from galaxy_profiles import *


'''
These are methods that were removed from Galaxy, etc, along with these
old galaxy models.
'''
class Galaxy(MultiParams):
	def getGalaxyPatch(self, img, cx, cy, cd):
		'''
		Returns
		'''
		# remember to include margin for psf conv
		profile = self.getProfile()
		if profile is None:
			return None
		(H,W) = img.shape
		margin = int(ceil(img.getPsf().getRadius()))
		T = self.shape.getTensor(cd)
		assert(profile.get_compiled_ab() == 1.)
		# convert re to degrees
		re_deg = self.re / 3600.
		# sqrt(abs(det(cd))) is pixel scale in deg/pix
		det = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
		pixscale = sqrt(abs(det))
		repix = re_deg / pixscale
		(prof,x0,y0) = profile.sample_transform(T, repix, self.ab,
												cx, cy, W, H, margin)
		return Patch(x0, y0, prof)

	def getUnitFluxModelPatch(self, img, px=None, py=None):
		if px is None or py is None:
			(px,py) = img.getWcs().positionToPixel(self, self.getPosition())
		cd = img.getWcs().cdAtPixel(px, py)
		patch = self.getGalaxyPatch(img, px, py, cd)
		if patch is None:
			print('Warning, is Galaxy(subclass).getProfile() defined?')
			return Patch(0, 0, None)
		if patch.getImage() is None:
			return Patch(patch.getX0(), patch.getY0(), None)
		psf = img.getPsf()
		convimg = psf.applyTo(patch.getImage())
		#print 'PSF-convolved'
		#self.debugPatchImage(convimg)
		return Patch(patch.getX0(), patch.getY0(), convimg)

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
			print('Exp patch', img)
		elif np.product(img.shape) == 0:
			print('Patch empty:', img.shape)
		else:
			print('Patch', img.shape)
			plt.clf()
			plt.imshow(img, interpolation='nearest', origin='lower')
			plt.hot()
			plt.colorbar()
			fn = 'exp-patch-%02i-%03i.png' % (self.num, self.plotnum)
			plt.savefig(fn)
			print('saved', fn)
			self.plotnum += 1

class DevGalaxy(Galaxy):
	profile = None
	@staticmethod
	def getDevProfile():
		if DevGalaxy.profile is None:
			DevGalaxy.profile = (
				CompiledProfile(modelname='dev',
								profile_func=profile_dev, re=100, nrad=8))
		return DevGalaxy.profile

	def __init__(self, pos, flux, re, ab, phi):
		Galaxy.__init__(self, pos, flux, GalaxyShape(re, ab, phi))
		self.name = 'DevGalaxy'

	def getProfile(self):
		return DevGalaxy.getDevProfile()

	def copy(self):
		return DevGalaxy(self.pos, self.flux, self.re, self.ab, self.phi)
