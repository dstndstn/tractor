from tractor import MultiParams, PointSource, ParamList, RaDecPos, Mags
from tractor.sdss_galaxy import DevGalaxy, GalaxyShape

class LensedQuasar(MultiParams):
	@staticmethod
	def getNamedParams():
		return dict(light=0, mass=1, quasar=2, magfudge=3)

	def getModelPatch(self, img):
		# We start by rendering the visible lens galaxy.
		patch = self.light.getModelPatch(img)

		# We will use the lens model to predict the quasar's image positions.
		positions,mags = self.mass.getLensedImages(self.light.position, self.quasar)
		# 'positions' should be a list of RaDecPos objects
		# 'mags' should be a list of Mags objects

		for pos,mag,fudge in zip(positions, mags, self.magfudge):
			# For each image of the quasar, we will create a PointSource
			ps = PointSource(pos, mag + fudge)
			# ... and add it to the patch.
			patch += ps.getModelPatch(img)

		return patch

	def getParamDerivatives(img):
		pass

class LensingMass(ParamList):
	@staticmethod
	def getNamedParams():
		return dict(mass=0, radius=1)

	def getStepSizes(self):
		'''We are using units of solar masses and arcsec'''
		return [1e12, 0.1]

	def getLensedImages(self, mypos, quasar):
		pass

class Quasar(ParamList):
	pass

class MagFudge(ParamList):
	pass


pos = RaDecPos(234.5, 17.9)
bright = Mags(r=17.4, g=18.9, order=['g','r'])
# re [arcsec], ab ratio, phi [deg]
shape = GalaxyShape(2., 0.5, 48.)
light = DevGalaxy(pos, bright, shape)

mass = LensingMass(1e14, 0.1)

quasar = Quasar()

fudge = MagFudge(0., 0., 0., 0.)

lq = LensedQuasar(light, mass, quasar, fudge)

print 'LensedQuasar params:'
for nm,val in zip(lq.getParamNames(), lq.getParams()):
	print '  ', nm, '=', val


