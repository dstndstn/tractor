import unittest

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *

class TractorTest(unittest.TestCase):
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
						 'ExpGalaxy at RA,Dec (123.00000, 45.00000) with SdssFlux: 6000000.0, re=7.0, ab=0.80, phi=9.0')
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

if __name__ == '__main__':
	unittest.main()
		
