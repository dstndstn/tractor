
from tractor import Tractor

import numpy as np
from math import pi


class HSTTractor(Tractor):
	pass



def measure_sky_variance(img):
	x0,y0 = 5,7
	d = img[y0:,x0:] - img[:-y0,:-x0]
	mad = abs(d.ravel())
	I = np.argsort(mad)
	mad = mad[I[len(I)/2]]
	print 'median abs diff:', mad
	sigmasq = mad**2 * pi / 4.
	print 'sigma', np.sqrt(sigmasq)
	return sigmasq

if __name__ == '__main__':
	import pyfits

	# Dealing with cosmic rays is a PITA so use DRZ for now...
	#P = pyfits.open('jbf108bzq_flt.fits')
	#img = P[1].data
	#err = P[2].data
	#dq  = P[3].data

	P = pyfits.open('jbf108020_drz.fits')
	img = P[1].data
	wht = P[2].data
	ctx = P[3].data

	cut = [slice(1000,1300), slice(2300,2600)]
	img = img[cut]
	wht = wht[cut]

	skyvar = measure_sky_variance(img)
	invvar = wht / np.median(wht) / skyvar

	import matplotlib
	matplotlib.use('Agg')
	import pylab as pl

	skymed = np.median(img.ravel())
	skysig = np.sqrt(skyvar)
	zrange = np.array([-3.,+10.]) * skysig + skymed

	# add in source noise to variance map
	# problem for the reader:  why *divide* by wht?
	srcvar = np.maximum(0, (img - skymed) / np.maximum(wht, np.median(wht)*1e-6))
	invvar = invvar / (1.0 + invvar * srcvar)

	pl.clf()
	pl.hist(img.ravel(), bins=np.linspace(zrange[0], zrange[1], 100))
	pl.savefig('hist.png')

	pl.clf()
	pl.imshow(img, interpolation='nearest', origin='lower',
			  vmin=zrange[0], vmax=zrange[1]) #vmin=50, vmax=500)
	pl.hot()
	pl.colorbar()
	pl.savefig('img.png')
	
	pl.clf()
	pl.imshow(invvar, interpolation='nearest', origin='lower',
			  vmin=0., vmax=2./(skysig**2))
	pl.hot()
	pl.colorbar()
	pl.savefig('invvar.png')
	
	pl.clf()
	pl.imshow((img-skymed) * np.sqrt(invvar), interpolation='nearest', origin='lower',
			  vmin=-3, vmax=10.)
	pl.hot()
	pl.colorbar()
	pl.savefig('chi.png')
