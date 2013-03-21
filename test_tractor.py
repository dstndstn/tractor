import matplotlib
matplotlib.use('Agg')
import pylab as plt

import unittest

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *

class TractorTest(unittest.TestCase):
	def test_pixpsf(self):
		tim,tinf = get_tractor_image_dr8(94, 2, 520, 'i', psf='kl-pix',
										 roi=[500,600,500,600], nanomaggies=True)
		psf = tim.getPsf()
		print 'PSF', psf

		for i,(dx,dy) in enumerate([
				(0.,0.), (0.2,0.), (0.4,0), (0.6,0),
				(0., -0.2), (0., -0.4), (0., -0.6)]):
			px,py = 50.+dx, 50.+dy
			patch = psf.getPointSourcePatch(px, py)
			print 'Patch size:', patch.shape
			print 'x0,y0', patch.x0, patch.y0
			H,W = patch.shape
			XX,YY = np.meshgrid(np.arange(W), np.arange(H))
			im = patch.getImage()
			cx = patch.x0 + (XX * im).sum() / im.sum()
			cy = patch.y0 + (YY * im).sum() / im.sum()
			print 'cx,cy', cx,cy
			print 'px,py', px,py

			self.assertLess(np.abs(cx - px), 0.1)
			self.assertLess(np.abs(cy - py), 0.1)
			
			plt.clf()
			plt.imshow(patch.getImage(), interpolation='nearest', origin='lower')
			plt.title('dx,dy %f, %f' % (dx,dy))
			plt.savefig('pixpsf-%i.png' % i)
		


	def test_expgal(self):
		ra,dec = 123., 45.
		pos = RaDecPos(ra, dec)
		flux = 6.
		sflux = SdssFlux(flux)
		re,ab,phi = 7, 0.8, 9
		shape = GalaxyShape(re, ab, phi)
		gal = ExpGalaxy(pos, sflux, shape)
		# harsh
		self.assertEqual(str(gal),
						 'ExpGalaxy at RaDecPos: RA, Dec = (123.00000, 45.00000) with SdssFlux: 6000000.0 and Galaxy Shape: re=7.00, ab=0.80, phi=9.0')
		self.assertEqual(str(gal.shape), str(shape))
		self.assertEqual(shape.re, re)
		self.assertEqual(shape.ab, ab)
		self.assertEqual(shape.phi, phi)
		self.assertEqual(gal.re, re)
		self.assertEqual(gal.ab, ab)
		self.assertEqual(gal.phi, phi)
		self.assertEqual(gal.getParams(), [ra, dec, flux, re, ab, phi])
		self.assertEqual(shape.getParams(), [re, ab, phi])

		re2 = 7.7
		gal.re = re2
		print gal
		self.assertEqual(shape.re, re2)
		self.assertEqual(gal.re, re2)
		print gal.subs
		print shape.vals
		self.assertEqual(gal.getParams(), [ra, dec, flux, re2, ab, phi])
		self.assertEqual(shape.getParams(), [re2, ab, phi])

		re3 = 7.77
		gal.shape = GalaxyShape(re3, ab, phi)
		self.assertEqual(gal.re, re3)

		# However:
		self.assertNotEqual(gal.shape, shape)
		self.assertNotEqual(shape.re, re3)
		self.assertEqual(shape.re, re2)


if __name__ == '__main__':
	unittest.main()
		
	#import matplotlib
	#matplotlib.use('Agg')
	#import pylab as plt
		
