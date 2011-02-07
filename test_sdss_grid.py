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


def makeTruth():
	W,H = 500,500
	ra,dec = 0,0
	width = 0.1
	
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
	wcs1 = FitsWcs(wcs)

	# rotate.
	W2 = int(W*np.sqrt(2.))
	H2 = W2
	rot = 30.
	cr = np.cos(np.deg2rad(rot))
	sr = np.sin(np.deg2rad(rot))
	wcs = Tan()
	wcs.crval[0] = ra
	wcs.crval[1] = dec
	wcs.crpix[0] = W2/2.
	wcs.crpix[1] = H2/2.
	wcs.cd[0] = -scale * cr
	wcs.cd[1] =  scale * sr
	wcs.cd[2] = -scale * sr
	wcs.cd[3] = -scale * cr
	wcs.imagew = W2
	wcs.imageh = H2
	wcs2 = FitsWcs(wcs)

	# image 1
	image = np.zeros((H,W))
	invvar = np.zeros_like(image) + 1.
	photocal = SdssPhotoCal(SdssPhotoCal.scale)
	psf = NGaussianPSF([1.5], [1.0])
	sky = 0.

	

	eg = ExpGalaxy(pos, flux, re, ab, phi)

	#img1 = Image(data=image1, invvar=invvar1, psf=psf1, wcs=wcs1, sky=sky1,
	#			photocal=photocal1)
	




def main():
	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option('-v', '--verbose', dest='verbose', action='count',
					  default=0, help='Make more verbose')
	opt,args = parser.parse_args()
	print 'Opt.verbose = ', opt.verbose
	if opt.verbose == 0:
		lvl = logging.INFO
	else: # opt.verbose == 1:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	(images, simplexys, rois, zrange, nziv, footradecs
	 ) = prepareTractor(False, False, rcfcut=[0])

	print 'Creating tractor...'
	tractor = SDSSTractor(images, debugnew=False, debugchange=True)
	opt,args = parser.parse_args()

	src = DevGalaxy(pos, flux, 0.9, 0.75, -35.7)
	src.setParams([120.60275, 9.41350, 5.2767052, 22.8, 0.81, 5.0])
	tractor.catalog.append(src)

	def makePlots(tractor, fnpat, title1='', title2=''):
		mods = tractor.getModelImages()
		imgs = tractor.getImages()
		chis = tractor.getChiImages()
		for i,(mod,img,chi) in enumerate(zip(mods,imgs,chis)):
			zr = zrange[i]
			imargs = dict(interpolation='nearest', origin='lower',
						  vmin=zr[0], vmax=zr[1])
			srcpatch = src.getModelPatch(img)
			slc = srcpatch.getSlice(img)
			plt.clf()
			plt.subplot(2,2,1)
			plotimage(img.getImage()[slc], **imargs)
			plt.title(title1)
			plt.subplot(2,2,2)
			plotimage(mod[slc], **imargs)
			plt.title(title2)
			plt.subplot(2,2,3)
			plotimage(chi[slc], interpolation='nearest', origin='lower',
					  vmin=-5, vmax=5)
			plt.title('chi')
			#plt.subplot(2,2,4)
			#plotimage(img.getInvError()[slc],
			#		  interpolation='nearest', origin='lower',
			#		  vmin=0, vmax=0.1)
			#plt.title('inv err')
			fn = fnpat % i
			plt.savefig(fn)
			print 'wrote', fn

	makePlots(tractor, 'opt-s00-%02i.png', #'pre-%02i.png',
			  title2='re %.1f, ab %.2f, phi %.1f' % (src.re, src.ab, src.phi))

	for ostep in range(10):
		print
		print 'Optimizing...'
		#alphas = [1., 0.5, 0.25, 0.1, 0.01]
		alphas=None
		ppre = src.getParams()
		lnppre = tractor.getLogProb()
		dlnp,X,alpha = tractor.optimizeCatalogAtFixedComplexityStep(alphas=alphas)
		ppost = src.getParams()
		makePlots(tractor, 'opt-s%02i-%%02i.png' % (ostep+1),
				  title1='dlnp = %.1f' % dlnp,
				  title2='re %.1f, ab %.2f, phi %.1f' % (src.re, src.ab, src.phi))

		print
		src.setParams(ppre)
		print 'Pre :', src

		src.setParams(ppost)
		print 'Post:', src


if __name__ == '__main__':
	main()
	sys.exit(0)
