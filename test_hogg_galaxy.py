import sys
from math import pi

import numpy as np
import pylab as plt

# FIXME -- use better swig'd version
from astrometry.util.sip import Tan

from sdsstractor import *

def main():
	#W,H = 500,500
	W,H = 100,100
	ra,dec = 0,0
	width = 0.1 # deg

	wcs = Tan()
	wcs.crval[0] = ra
	wcs.crval[1] = dec
	wcs.crpix[0] = W/2.
	wcs.crpix[1] = H/2.
	scale = width / float(W)
	wcs.cd[0] = -scale
	wcs.cd[1] = 0
	wcs.cd[2] = 0
	wcs.cd[3] = -scale
	wcs.imagew = W
	wcs.imageh = H
	tanwcs1 = wcs
	wcs1 = FitsWcs(wcs)

	photocal = SdssPhotoCal(SdssPhotoCal.scale)
	psf = NGaussianPSF([2.0], [1.0])
	sky = 0.
	skyobj = ConstantSky(sky)
	flux = SdssFlux(1.)

	# image 1
	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1e-4
	img1 = Image(data=image, invvar=invvar, psf=psf, wcs=wcs1,
				 sky=skyobj, photocal=photocal, name='Grid1')

	# arcsec
	re = 10.
	ab = 0.5
	phi = 30.

	x,y = W/2,H*0.66
	ra,dec = tanwcs1.pixelxy2radec(x, y)
	pos = RaDecPos(ra, dec)
	eg1 = HoggExpGalaxy(pos, flux, re, ab, phi)
	patch1 = eg1.getModelPatch(img1)

	x,y = W/2,H*0.33
	ra,dec = tanwcs1.pixelxy2radec(x, y)
	pos = RaDecPos(ra, dec)
	eg2 = ExpGalaxy(pos, flux, re, ab, phi)
	patch2 = eg2.getModelPatch(img1)

	model = np.zeros_like(image)
	patch1.addTo(model)
	patch2.addTo(model)

	plt.clf()
	plt.imshow(model, interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.savefig('hg.png')

if __name__ == '__main__':
	main()
	sys.exit(0)
