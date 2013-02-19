import os
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import multiprocessing
from glob import glob
from astrometry.util.pyfits_utils import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.multiproc import *
from astrometry.util.file import *
from astrometry.util.plotutils import ArcsinhNormalize
from astrometry.util.util import *
from astrometry.sdss import *
from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params


def get_cs82_sources(T, maglim=25, mags=['u','g','r','i','z']):
	srcs = Catalog()
	isrcs = []
	for i,t in enumerate(T):
		if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
			#print 'PSF'
			themag = t.mag_psf
			m = Mags(order=mags, **dict([(k, themag) for k in mags]))
			m = NanoMaggies.fromMag(m)
			srcs.append(PointSource(RaDecPos(t.ra, t.dec), m))
			isrcs.append(i)
			continue

		if t.mag_disk > maglim and t.mag_spheroid > maglim:
			#print 'Faint'
			continue

		# deV: spheroid
		# exp: disk

		themag = t.mag_spheroid
		m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))
		themag = t.mag_disk
		m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))

		m_dev = NanoMaggies.fromMag(m_dev)
		m_exp = NanoMaggies.fromMag(m_exp)

		# SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

		shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600., t.disk_aspect_world,
								t.disk_theta_world + 90.)
		shape_dev = GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
								t.spheroid_theta_world + 90.)
		pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

		isrcs.append(i)
		if t.mag_disk > maglim and t.mag_spheroid <= maglim:
			srcs.append(DevGalaxy(pos, m_dev, shape_dev))
			continue
		if t.mag_disk <= maglim and t.mag_spheroid > maglim:
			srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
			continue
		srcs.append(CompositeGalaxy(pos, m_exp, shape_exp, m_dev, shape_dev))

	print 'Sources:', len(srcs)
	return srcs, np.array(isrcs)



def main():
	T = fits_table('cs82data/S82p18p_y.V2.7A.swarp.cut.deVexp.fit',
				   hdu=2, column_map={'ALPHA_J2000':'ra',
									  'DELTA_J2000':'dec'})
	print 'RA', T.ra.min(), T.ra.max()
	print 'Dec', T.dec.min(), T.dec.max()

	F = fits_table('window_flist-DR9.fits')

	# For Stripe 82, mu-nu is aligned with RA,Dec.
	rd = []
	#for mu0,mu1,nu0,nu1,node,incl in zip(F.mu_start, F.mu_end, F.nu_start, F.nu_end,
	#									 F.node, F.incl):
	#	rd.append([munu_to_radec_deg(mu, nu, node, incl)
	#			   for mu,nu in [(mu0,nu0),(mu1,nu1)]])
	rd.append(munu_to_radec_deg(F.mu_start, F.nu_start, F.node, F.incl))
	rd.append(munu_to_radec_deg(F.mu_end,   F.nu_end,   F.node, F.incl))
	rd = np.array(rd)
	print rd.shape
	# F.ra0 = np.min(rd[:,:,0], axis=1)
	# F.ra1 = np.max(rd[:,:,0], axis=1)
	# F.dec0 = np.min(rd[:,:,1], axis=1)
	# F.dec1 = np.max(rd[:,:,1], axis=1)
	F.ra0  = np.min(rd[:,0,:], axis=0)
	F.ra1  = np.max(rd[:,0,:], axis=0)
	F.dec0 = np.min(rd[:,1,:], axis=0)
	F.dec1 = np.max(rd[:,1,:], axis=0)

	I = np.flatnonzero((F.ra0 <= T.ra.max()) *
					   (F.ra1 >= T.ra.min()) *
					   (F.dec0 <= T.dec.max()) *
					   (F.dec1 >= T.dec.min()))
	print 'Overlapping fields:', len(I)

	I = np.flatnonzero((F.ra0 >= T.ra.min()) *
					   (F.ra1 <= T.ra.max()) *
					   (F.dec0 >= T.dec.min()) *
					   (F.dec1 <= T.dec.max()))
	print 'Enclosed fields:', len(I)

	sdss = DR9(basedir='cs82data/dr9')

	maglim = 24

	#cat = get_cs82_sources(T)

	ps = PlotSequence('cs82b')

	for r,c,f,r0,r1,d0,d1 in zip(F.run[I], F.camcol[I], F.field[I],
								 F.ra0[I], F.ra1[I], F.dec0[I], F.dec1[I]):
		#for band in 'ugriz':

		margin = 10. / 3600.
		Ti = T[(T.ra  + margin >= r0) * (T.ra  - margin <= r1) *
			   (T.dec + margin >= d0) * (T.dec - margin <= d1)]
		print len(Ti), 'CS82 sources in range'

		cat,icat = get_cs82_sources(Ti, maglim=maglim)
		print 'Got', len(cat), 'sources'

		for band in 'i':
			tim,inf = get_tractor_image_dr9(r, c, f, band, sdss=sdss,
											nanomaggies=True, zrange=[-2,3])
			tr = Tractor([tim], cat)
			print tr

			tr.freezeParam('images')
			cat = tr.catalog
			cat.freezeParamsRecursive('*')
			cat.thawPathsTo(band)

			cat0 = cat.getParams()

			sig = 1./np.median(tim.getInvError())
			print 'Image sigma:', sig

			#minsb = 0.001 * sig
			#minsb = 1. * sig
			minsb = 0.1 * sig

			img = tim.getImage()
			mod = tr.getModelImage(0, minsb=minsb)

			groups,L = tr.getOverlappingSources(0, minsb=minsb)
			print 'Got', len(groups), 'groups of sources'

			zr = tim.zr
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=zr[0], vmax=zr[1], cmap='gray')
			imbin = dict(interpolation='nearest', origin='lower',
						 vmin=0, vmax=1, cmap='gray')
			imx = dict(interpolation='nearest', origin='lower')

			plt.clf()
			plt.imshow(img, **ima)
			plt.title('Image %s' % tim.name)
			ps.savefig()

			# plt.clf()
			# plt.imshow(L, **imx)
			# plt.title('Labels %s' % tim.name)
			# ps.savefig()

			plt.clf()
			plt.imshow(mod, **ima)
			plt.title('Mod %s -- 0' % tim.name)
			ps.savefig()

			# plt.clf()
			# plt.imshow(np.log10(np.maximum(minsb*0.1, mod)), **imx)
			# plt.jet()
			# plt.colorbar()
			# plt.title('Mod %s -- 0' % tim.name)
			# ps.savefig()
			# 
			# plt.clf()
			# plt.imshow(np.log10(np.maximum(minsb*0.01, mod)), vmin=-4, vmax=-2, **imx)
			# plt.jet()
			# plt.colorbar()
			# plt.title('Mod %s -- 0' % tim.name)
			# ps.savefig()
			# 
			# 
			# plt.clf()
			# plt.imshow(mod > 0, **imbin)
			# plt.title('Mod > 0 %s -- 0' % tim.name)
			# ps.savefig()
			# 
			# plt.clf()
			# plt.imshow(mod > minsb, **imbin)
			# plt.title('Mod > minsb %s -- 0' % tim.name)
			# ps.savefig()
			# 
			# plt.clf()
			# plt.imshow(mod > (0.1*minsb), **imbin)
			# plt.title('Mod > 0.1 minsb %s -- 0' % tim.name)
			# ps.savefig()
			# 
			# plt.clf()
			# plt.imshow(mod > (0.01 * minsb), **imbin)
			# plt.title('Mod > 0.01 minsb %s -- 0' % tim.name)
			# ps.savefig()
			# 
			# 
			# plt.clf()
			# plt.hist(mod.ravel(), range=(0, 0.01), bins=100)
			# plt.title('Mod %s -- 0' % tim.name)
			# ps.savefig()

			# for src in cat:
			# 	mod = tr.getModelPatch(tim, src)
			# 	mod.trimToNonZero()
			# 	ext = mod.getExtent()
			# 	print 'ext', ext
			# 	[x0,x1,y0,y1] = ext
			# 	if ((x1-x0) * (y1-y0) > 1000):
			# 		plt.clf()
			# 		plt.subplot(1,2,1)
			# 		plt.imshow(mod.patch, extent=ext, **ima)
			# 		plt.subplot(1,2,2)
			# 		plt.imshow(mod.patch > 0, extent=ext, **imbin)

			for gi,(gl,gsrcs) in enumerate(groups.items()):
				print 'Group', gl, ': sources', gsrcs
				tr.catalog.freezeAllBut(*gsrcs)
				#srcs = []
				#for i in gsrcs:
				#	srcs.append(tr.catalog[i])

				print 'Thawed params:'
				for nm in tr.getParamNames():
					print '  ', nm

				while True:
					dlnp,X,alpha = tr.optimize()

					pc = tim.getPhotoCal()
					for i in gsrcs:
						bb = tr.catalog[i].getBrightnesses()
						counts = 0.
						for b in bb:
							c = pc.brightnessToCounts(b)
							if c <= 0:
								print 'Clamping brightness up to zero for', tr.catalog[i]
								b.setBand(band, 0.)
							else:
								counts += c
						if counts == 0.:
							print 'Freezing zero-flux source', i, tr.catalog[i]
							tr.catalog.freezeParam(i)
							gsrcs.remove(i)
							
					print 'delta-logprob', dlnp
					if dlnp < 1.:
						break

				mod = tr.getModelImage(0, minsb=minsb)
				plt.clf()
				plt.imshow(mod, **ima)
				plt.title('Mod %s -- group %i' % (tim.name, gi+1))
				ps.savefig()

		cat.thawPathsTo(band)
		cat1 = cat.getParams()

		plt.clf()
		plt.plot(cat0, cat1, 'r.')
		plt.xlabel('i-band (CS82)')
		plt.ylabel('i-band (%s)' % tim.name)
		plt.axis([25,8, 25,8])
		ps.savefig()

		break

if __name__ == '__main__':
	import cProfile
	from datetime import datetime
	cProfile.run('main()', 'prof-cs82b-%s.dat' % (datetime.now().isoformat()))
	#main()
	


