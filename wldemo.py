import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from tractor import *
from tractor.sdss_galaxy import *

class WeakLensWcs(ParamsWrapper):
	def __init__(self, wcs):
		super(WeakLensWcs,self).__init__(wcs)

	def positionToPixel(self, pos, src=None):
		p2 = self.positionToLensedPosition(pos, src=src)
		return self.real.positionToPixel(p2, src=src)

	def cdAtPixel(self, x, y):
		cd = self.real.cdAtPixel(x,y)
		J = self.lensJacobianAtPixel(x,y)
		#print 'cd', cd
		#print 'J', J
		Jcd = np.dot(J, cd)
		return Jcd

	def positionToLensedPosition(self, pos, src=None):
		return pos
	def lensJacobianAtPixel(self, x, y):
		return np.eye(2)

class PointSourceWeakLensWcs(WeakLensWcs):
	def __init__(self, wcs, pos, mass):
		super(PointSourceWeakLensWcs, self).__init__(wcs)
		self.pos = pos
		self.mass = mass

	def hashkey(self):
		return ('PointSourceWeakLensWcs', self.pos.hashkey(),
				self.mass, self.real.hashkey())
		
	def positionToLensedPosition(self, pos, src=None):
		rscale = np.cos(np.deg2rad(self.pos.dec))
		dr = (self.pos.ra - pos.ra) * rscale
		dd = (self.pos.dec - pos.dec)
		defl = self.mass / np.hypot(dr,dd)
		return RaDecPos(pos.ra - defl * dr / rscale,
						pos.dec - defl * dd)

	def lensJacobianAtPixel(self, x, y):
		pos = self.real.pixelToPosition(x, y)
		J = self.lensJacobianAtPosition(pos)
		return J

	def lensJacobianAtPosition(self, pos):
		K = 0.
		rscale = np.cos(np.deg2rad(self.pos.dec))
		dr = (self.pos.ra - pos.ra) * rscale
		dd = (self.pos.dec - pos.dec)
		gamma = self.mass / (dr**2 + dd**2)
		phi = np.arctan2(dd, dr)
		g1 = gamma * np.cos(2.*phi)
		g2 = gamma * np.sin(2.*phi)
		return np.array([[1. + K + g1, g2], [g2, 1. + K - g1]])


if __name__ == '__main__':

	from astrometry.util.util import Tan
	from astrometry.util.plotutils import *

	ps = PlotSequence('wl')
	
	W,H = 500,500
	data = np.zeros((H,W))
	iv = np.ones((H,W))
	psf = NCircularGaussianPSF([1.], [1.])

	ra,dec = 90.,0.
	pscale = 1. / 3600.
	wcs = FitsWcs(Tan(ra, dec, (1.+W)/2., (1.+H)/2.,
					  pscale, 0., 0., pscale, W, H))
	lwcs = PointSourceWeakLensWcs(wcs, RaDecPos(ra, dec), 1e-2)
	photocal = NullPhotoCal()
	sky = ConstantSky(0.)
	
	tim = Image(data=data, invvar=iv, psf=psf, wcs=lwcs,
				sky=sky, photocal=photocal, name='im')

	srcs = []
	NX,NY = 21,21
	xx = np.linspace(0, W, NX)
	yy = np.linspace(0, H, NY)
	xx += (xx[1]-xx[0])/2.
	yy += (yy[1]-yy[0])/2.
	xx,yy = np.meshgrid(xx,yy)
	for x,y in zip(xx.ravel(),yy.ravel()):
		pos = wcs.pixelToPosition(x,y)
		br = Flux(1000.)
		# re, ab, phi
		shape = GalaxyShape(10., 1., 0.)
		gal = DevGalaxy(pos, br, shape)
		srcs.append(gal)

	tractor = Tractor([tim], srcs)

	#for mass in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]:
	for mass in [0] + list(10.**np.linspace(-5, -3, 11)):
		lwcs.mass = mass
		mod = tractor.getModelImage(0)
		imc = dict(interpolation='nearest', origin='lower',
				   norm=ArcsinhNormalize(mean=0., std=1.),
				   vmin=-1., vmax=100.)
		
		plt.clf()
		plt.imshow(mod, **imc)
		plt.title('mass=%g' % mass)
		ps.savefig()
	
