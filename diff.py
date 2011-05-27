

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
from math import pi, sqrt, ceil, floor
import pyfits
import pylab as plt
import numpy as np
import matplotlib

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.util import *

from tractor import *
from sdsstractor import *


def main():
	sdss = DR7()

	bandname = 'g'
	bandnum = band_index(bandname)

	# ref
	run,camcol,field = 6955,3,809
	rerun = 0

	ra,dec = 53.202125, -0.365361
	
	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias

	objs = fits_table(sdss.getFilename('tsObj', run, camcol, field,
									   bandname, rerun=rerun))
	print 'objs', objs

	tsf = sdss.readTsField(run, camcol, field, rerun)
	print 'tsField:', tsf
	astrans = tsf.getAsTrans(bandnum)
	#print 'astrans', astrans
	x,y = astrans.radec_to_pixel(ra, dec)
	print 'asTrans x,y', x,y

	### debug weird asTrans
	wcs = Tan(sdss.getFilename('fpC', run, camcol, field, bandname,
							   rerun=rerun))
	x1,y1 = wcs.radec2pixelxy(ra,dec)
	print 'WCS x,y', x1,y1


	# Render ref source into ref (sub-)image.

	psfield = sdss.readPsField(run, camcol, field)
	dgpsf = psfield.getDoubleGaussian(bandnum)
	print 'Creating double-Gaussian PSF approximation'
	print '  ', dgpsf
	(a,s1, b,s2) = dgpsf
	psf = NGaussianPSF([s1, s2], [a, b])

	# half cutout size
	dpix = 150

	x0,x1 = int(x - dpix), int(x + dpix)
	y0,y1 = int(y - dpix), int(y + dpix)
	print 'x0,x1', (x0,x1), 'y0,y1', (y0,y1)

	roislice = (slice(y0,y1), slice(x0,x1))
	image = fpC[roislice]

	wcs = SdssWcs(astrans)
	wcs.setX0Y0(x0, y0)

	photocal = SdssPhotoCal(SdssPhotoCal.scale)
	sky = psfield.getSky(bandnum)
	skysig = sqrt(sky)
	skyobj = ConstantSky(sky)
	zr = np.array([-3.,+10.]) * skysig + sky

	# we don't care about the invvar (for now)
	invvar = np.zeros_like(image)

	timg = Image(data=image, invvar=invvar, psf=psf, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name='Ref (r/c/f=%i/%i%i)' % (run, camcol, field))

	tractor = SDSSTractor([timg], debugnew=False, debugchange=True)

	# Find tsObj sources within the image.
	## HACK -- I'm just grabbing things whose *centers* are within the img
	r,c = objs.rowc[:,bandnum], objs.colc[:,bandnum]
	#print 'r,c', r,c
	I = ((r > y0) * (r < y1) * (c > x0) * (c < x1))
	objs = objs[I]
	#print 'objs:', objs
	print len(objs), 'within the sub-image'

	# parent / nchild
	# ra, dec
	# #colc, rowc,
	# psfcounts
	# r_dev, ab_dev, phi_dev, counts_dev
	# r_exp, ab_exp, phi_exp, counts_exp
	# fracpsf
	# probpsf

	# Keep just the deblended children.
	objs = objs[(objs.nchild == 0)]

	Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
	Ldev = ((objs.prob_psf[:,bandnum] == 0) * objs.fracpsf[:,bandnum])
	Lexp = ((objs.prob_psf[:,bandnum] == 0) * (1. - objs.fracpsf[:,bandnum]))

	print 'Lstar', Lstar
	print 'Ldev', Ldev
	print 'Lexp', Lexp

	I = np.flatnonzero(Lstar > 0)
	print len(I), 'stars'
	print 'psf counts:', objs.psfcounts[I,bandnum]
	for i in I:


		pos = RaDecPos(objs.ra[i], objs.dec[i])
		#flux = SdssFlux(objs.psfcounts[i,bandnum] / SdssPhotoCal.scale)
		ps = PointSource(pos, flux)
		tractor.addSource(ps)
	
	
	mods = tractor.getModelImages()
	mod = mods[0]

	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])

	plt.clf()
	plt.imshow(image, **ima)
	plt.savefig('refimg.png')

	plt.clf()
	plt.imshow(mod, **ima)
	plt.savefig('refmod.png')


if __name__ == '__main__':
	main()

