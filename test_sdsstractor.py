
import sys
from math import pi

import numpy as np
import pylab as plt

from astrometry.util.sip import Tan

from sdsstractor import *


class FitsWcs(object):
	def __init__(self, wcs):
		self.wcs = wcs

	def positionToPixel(self, src, pos):
		x,y = self.wcs.radec2pixelxy(pos.ra, pos.dec)
		return x,y

	def pixelToPosition(self, src, xy):
		(x,y) = xy
		r,d = self.wcs.pixelxy2radec(x, y)
		return RaDecPos(r,d)

	def cdAtPixel(self, x, y):
		cd = self.wcs.cd
		return np.array([[cd[0], cd[1]], [cd[2],cd[3]]])

if __name__ == '__main__':

	angles = np.linspace(0, 2.*pi, 360)
	x,y = np.cos(angles), np.sin(angles)

	re = 3600./2. # arcsec
	ab = 0.5
	phi = 30. # deg

	abfactor = ab
	re_deg = re / 3600.
	phi = np.deg2rad(phi)

	# units of degrees
	G = np.array([[ re_deg * np.cos(phi), re_deg * np.sin(phi) ],
				  [ re_deg * abfactor * -np.sin(phi), re_deg * abfactor * np.cos(phi) ]])

	R = np.array([[ np.cos(phi),  np.sin(phi) ],
				  [-np.sin(phi),  np.cos(phi) ]])
	S = re_deg * np.array([[ 1., 0 ],
						   [ 0, abfactor ]])

	cp = np.cos(phi)
	sp = np.sin(phi)
	GG = re_deg * np.array([[ cp, sp * abfactor],
							[-sp, cp * abfactor]])

	print 'R', R
	print 'S', S

	RS = np.dot(R, S)

	print 'RS', RS

	print 'G', G
	#G = RS
	G = GG
	rd = np.dot(G, np.vstack((x,y)))
	print 'rd', rd.shape

	r = rd[0,:]
	d = rd[1,:]
	plt.clf()
	plt.plot(r, d, 'b-')
	plt.axis('equal')
	plt.savefig('g.png')


	width = (2./7.2) # in deg
	W,H = 500,500
	scale = width / float(W)
	cd = np.array([[-scale, 0],[0,-scale]])

	cdi = linalg.inv(cd)

	pg = np.dot(cdi, G)

	pxy = np.dot(pg, np.vstack((x,y)))
	px = pxy[0,:]
	py = pxy[1,:]
	plt.clf()
	plt.plot(px, py, 'b-')
	plt.axis('equal')
	plt.savefig('g2.png')
	

	T = np.dot(linalg.inv(G), cd)

	XX,YY = np.meshgrid(np.arange(-1000,1200, 200),
						np.arange( -600, 800, 200))
	XX = XX.ravel()
	YY = YY.ravel()
	XY = vstack((XX,YY))
	Tij = np.dot(T, XY)

	print 'Tij', Tij.shape
	for i in range(len(XX)):
		plt.text(XX[i], YY[i], '(%.1f,%.1f)' % (Tij[0,i], Tij[1,i]),
				 fontsize=8, ha='center', va='center')
	plt.savefig('g3.png')


	#sys.exit(0)

	ra,dec = 1.,45.
	width = (2./7.2) 	# in deg
	W,H = 500,500

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

	wcs = FitsWcs(wcs)

	pos = RaDecPos(ra, dec)
	flux = SdssFlux(1e4)

	# arcsec
	repix = 25.
	re = 3600. * scale * repix
	ab = 0.5
	phi = 30.0
	
	eg = ExpGalaxy(pos, flux, re, ab, phi)

	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1.
	photocal = SdssPhotoCal(SdssPhotoCal.scale)
	psf = NGaussianPSF([1.5], [1.0])
	sky = 0.
	img = Image(data=image, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
				photocal=photocal)

	patch = eg.getModelPatch(img)

	imargs1 = dict(interpolation='nearest', origin='lower')

	plt.clf()
	plt.imshow(patch.getImage(), **imargs1)
	plt.colorbar()
	plt.savefig('eg-1.png')
	
