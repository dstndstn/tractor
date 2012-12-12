if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
	import pylab as plt

import pyfits

from tractor import *
from tractor.psfex import *

from astrometry.util.fits import *
# for Tan wcs, etc
#from astrometry.util.util import *


if __name__ == '__main__':

	imgfn = 'dec028475.fits.fz'
	imgext = 1
	psffn = 'dec028475.psf'
	psfext = imgext
	catfn = 'sextractor-028475.fits'
	
	P = pyfits.open(imgfn)
	img = P[imgext].data
	H,W = img.shape

	band = P[0].get('FILTER')
	band = band.split()[0]
	print 'band', i

	pixscale = P[0].get('PIXSCAL1')
	# it would be evil to make PIXSCAL1 != PIXSCAL2...

	name = P[0].get('FILENAME').replace('.fits','')
	
	# FIXME!
	invvar = np.ones_like(img)

	psf = PsfEx(psffn, W, H, ext=psfext)

	# FIXME?
	sky = ConstantSky(0.)

	# Work in raw counts?
	photocal = NullPhotoCal()

	#scale = NanoMaggies.zeropointToScale(zp)
	#photocal = LinearPhotoCal(scale, band=)

	# WCS -- on work in pixels?
	wcs = NullWCS(pixscale=pixscale)

	# Haha, this will never work!
	#wcs = FitsWCS(Tan(imgfn, imgext))
	
	# build tractor Image class
	tim = Image(data=img, invvar=invvar, psf=psf, wcs=wcs, sky=sky,
				photocal=photocal, name=name)

	# build tractor catalog
	T = fits_table(catfn)

	cat = Catalog()
	#for i in range(len(T)):

	# from cs82.py:
	#T = fits_table('cs82data/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	# T.ra  = T.alpha_j2000
	# T.dec = T.delta_j2000
	# T = T[(T.ra > ra0) * (T.ra < ra1) * (T.dec > dec0) * (T.dec < dec1)]
	# print 'Cut to', len(T), 'objects nearby.'
	# for t in T:
	# 	if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
	# 		#print 'PSF'
	# 		themag = t.mag_psf
	# 		m = Mags(order=mags, **dict([(k, themag) for k in mags]))
	# 		if nanomaggies:
	# 			m = NanoMaggies.fromMag(m)
	# 		srcs.append(PointSource(RaDecPos(t.alpha_j2000, t.delta_j2000), m))
	# 		continue
	# 	if t.mag_disk > maglim and t.mag_spheroid > maglim:
	# 		#print 'Faint'
	# 		continue
	# 	# deV: spheroid
	# 	# exp: disk
	# 	themag = t.mag_spheroid
	# 	m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))
	# 	themag = t.mag_disk
	# 	m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))
	# 	if nanomaggies:
	# 		m_dev = NanoMaggies.fromMag(m_dev)
	# 		m_exp = NanoMaggies.fromMag(m_exp)
	# 	# SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE
	# 	shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600., t.disk_aspect_world,
	# 							t.disk_theta_world + 90.)
	# 	shape_dev = GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
	# 							t.spheroid_theta_world + 90.)
	# 	pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)
	# 	if t.mag_disk > maglim and t.mag_spheroid <= maglim:
	# 		srcs.append(DevGalaxy(pos, m_dev, shape_dev))
	# 		continue
	# 	if t.mag_disk <= maglim and t.mag_spheroid > maglim:
	# 		srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
	# 		continue
	# 	srcs.append(CompositeGalaxy(pos, m_exp, shape_exp, m_dev, shape_dev))

	tractor = Tractor([tim], cat)
	
