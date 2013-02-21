import os
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import multiprocessing
from glob import glob
from scipy.ndimage.measurements import label,find_objects

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

import cProfile
from datetime import datetime


def get_cs82_sources(T, maglim=25, mags=['u','g','r','i','z']):
	srcs = Catalog()
	isrcs = []
	for i,t in enumerate(T):
		if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
			#print 'PSF'
			themag = t.mag_psf
			#m = Mags(order=mags, **dict([(k, themag) for k in mags]))
			#m = NanoMaggies.fromMag(m)
			nm = NanoMaggies.magToNanomaggies(themag)
			m = NanoMaggies(order=mags, **dict([(k, nm) for k in mags]))
			srcs.append(PointSource(RaDecPos(t.ra, t.dec), m))
			isrcs.append(i)
			continue

		if t.mag_disk > maglim and t.mag_spheroid > maglim:
			#print 'Faint'
			continue

		# deV: spheroid
		# exp: disk

		themag = t.mag_spheroid
		#m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))
		#m_dev = NanoMaggies.fromMag(m_dev)
		nm = NanoMaggies.magToNanomaggies(themag)
		m_dev = NanoMaggies(order=mags, **dict([(k, nm) for k in mags]))

		themag = t.mag_disk
		#m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))
		#m_exp = NanoMaggies.fromMag(m_exp)
		nm = NanoMaggies.magToNanomaggies(themag)
		m_exp = NanoMaggies(order=mags, **dict([(k, nm) for k in mags]))

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


def runone(tr, ps, band):
	from tractor.ttime import Time

	tr.freezeParam('images')
	cat = tr.catalog
	tim = tr.getImages()[0]
	
	cat.freezeParamsRecursive('*')
	cat.thawPathsTo(band)

	cat0 = cat.getParams()

	sig = 1./np.median(tim.getInvError())
	print 'Image sigma:', sig

	#minsb = 0.001 * sig
	#minsb = 1. * sig
	minsb = 0.1 * sig

	img = tim.getImage()
	print 'Getting initial model...'
	t0 = Time()
	mod = tr.getModelImage(0, minsb=minsb)
	print 'initial model took', Time()-t0

	print 'Finding overlapping sources...'
	t0 = Time()
	groups,L = tr.getOverlappingSources(0, minsb=minsb)
	print 'Overlapping sources took', Time()-t0
	print 'Got', len(groups), 'groups of sources'

	zr = tim.zr
	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1], cmap='gray')
	imchi = dict(interpolation='nearest', origin='lower',
				 vmin=-5, vmax=5, cmap='gray')
	imbin = dict(interpolation='nearest', origin='lower',
				 vmin=0, vmax=1, cmap='gray')
	imx = dict(interpolation='nearest', origin='lower')

	plt.clf()
	plt.imshow(img, **ima)
	plt.title('Image %s' % tim.name)
	ps.savefig()

	plt.clf()
	plt.imshow(mod, **ima)
	plt.title('Mod %s -- 0' % tim.name)
	ps.savefig()

	# Sort the groups by the chi-squared values they contain
	print 'Getting chi image...'
	t0 = Time()
	chi = tr.getChiImage(0, minsb=minsb)
	print 'Chi image took', Time()-t0

	nl = L.max()
	gslices = find_objects(L, nl)
	chisq = []
	for i,gs in enumerate(gslices):
		c = np.sum(chi[L == (i+1)]**2)
		chisq.append(c)
	Gorder = np.argsort(-np.array(chisq))

	for gi,gl in enumerate(Gorder):
		gl += 1
		if not gl in groups:
			print 'Group', gl, 'not in groups array; skipping'
			continue
		gslice = gslices[gl]
		gsrcs = groups[gl]
		print 'Group number', (gi+1), 'of', len(Gorder), ', id', gl, ': sources', gsrcs

		#print 'slice', gslice
		tgroups = np.unique(L[gslice])
		#print 'groups touching slice:', tgroups
		tsrcs = []
		for g in tgroups:
			if not g in [gl,0]:
				tsrcs.extend(groups[g])
		print 'sources in groups touching slice:', tsrcs

		fullcat = tr.catalog
		subcat = Catalog(*[fullcat[i] for i in gsrcs + tsrcs])
		for i in range(len(tsrcs)):
			subcat.freezeParam(len(gsrcs) + i)
		tr.catalog = subcat

		print 'Thawed params:'
		tr.printThawedParams()

		t0 = Time()
		ims0,ims1 = tr.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
												  rois=[gslice])
		print 'optimize_forced_photometry took', Time()-t0

		print 'After params:'
		tr.printThawedParams()

		tr.catalog = fullcat

		# t0 = Time()
		# mod = tr.getModelImage(0, minsb=minsb)
		# print 'Getting model for plot took', Time()-t0
		# plt.clf()
		# plt.imshow(mod, **ima)
		# plt.title('Mod %s -- group %i' % (tim.name, gi+1))
		# ps.savefig()

		if gi < 50:
			(im,mod0,chi0,roi0) = ims0[0]
			if ims1 is not None:
				(im,mod1,chi1,roi1) = ims1[0]
	
			plt.clf()
			plt.subplot(2,3,1)
			print 'gslice', gslice
			sy,sx = gslice
			print 'sy', sy
			print 'sx', sx
			x0,x1,y0,y1 = [sx.start, sx.stop, sy.start, sy.stop]
			print 'ext', [x0,x1,y0,y1]
			margin = 25
			H,W = img.shape
			ext = [max(0, x0-margin), min(W-1, x1+margin),
				   max(0, y0-margin), min(H-1, y1+margin)]
			plt.imshow(img, extent=ext, **ima)
			ax = plt.axis()
			plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'r-')
			plt.axis(ax)
			plt.subplot(2,3,2)
			plt.imshow(mod0, **ima)
			plt.xticks([]); plt.yticks([])
			plt.subplot(2,3,3)
			plt.imshow(chi0, **imchi)
			plt.xticks([]); plt.yticks([])
			plt.subplot(2,3,4)
			plt.imshow(im, **ima)
			if ims1 is not None:
				plt.subplot(2,3,5)
				plt.imshow(mod1, **ima)
				plt.xticks([]); plt.yticks([])
				plt.subplot(2,3,6)
				plt.imshow(chi1, **imchi)
				plt.xticks([]); plt.yticks([])
			ps.savefig()
	
		#### Plot before-n-after for the ROI
		#### Plot chi image
		#### Modify invvar code to *not* include image Poisson term.

		#if gi == 10:
		#	break

	cat.thawPathsTo(band)
	cat1 = cat.getParams()

	T = tabledata()
	T.cs82_imag = cat0
	T.sdss_imag = cat1
	T.writeto('mags.fits')

	plt.clf()
	plt.plot(cat0, cat1, 'r.')
	plt.xlabel('i-band (CS82)')
	plt.ylabel('i-band (%s)' % tim.name)
	plt.axis([25,8, 25,8])
	ps.savefig()
	


def main():
	cs82field = 'S82p18p'

	T = fits_table('cs82data/%s_y.V2.7A.swarp.cut.deVexp.fit' % cs82field,
				   hdu=2, column_map={'ALPHA_J2000':'ra',
									  'DELTA_J2000':'dec'})
	print 'RA', T.ra.min(), T.ra.max()
	print 'Dec', T.dec.min(), T.dec.max()

	fn = 'sdssfield-%s.fits' % cs82field
	if os.path.exists(fn):
		print 'Reading', fn
		F = fits_table(fn)
	else:
		F = fits_table('window_flist-DR9.fits')
		# For Stripe 82, mu-nu is aligned with RA,Dec.
		rd = []
		rd.append(munu_to_radec_deg(F.mu_start, F.nu_start, F.node, F.incl))
		rd.append(munu_to_radec_deg(F.mu_end,   F.nu_end,   F.node, F.incl))
		rd = np.array(rd)
		F.ra0  = np.min(rd[:,0,:], axis=0)
		F.ra1  = np.max(rd[:,0,:], axis=0)
		F.dec0 = np.min(rd[:,1,:], axis=0)
		F.dec1 = np.max(rd[:,1,:], axis=0)
		##
		I = np.flatnonzero((F.ra0 <= T.ra.max()) *
						   (F.ra1 >= T.ra.min()) *
						   (F.dec0 <= T.dec.max()) *
						   (F.dec1 >= T.dec.min()))
		print 'Overlapping fields:', len(I)
		##
		I = np.flatnonzero((F.ra0 >= T.ra.min()) *
						   (F.ra1 <= T.ra.max()) *
						   (F.dec0 >= T.dec.min()) *
						   (F.dec1 <= T.dec.max()))
		print 'Enclosed fields:', len(I)
		F.cut(I)
		F.writeto(fn)
		print 'Wrote', fn



	sdss = DR9(basedir='cs82data/dr9')

	maglim = 24

	#cat = get_cs82_sources(T)

	ps = PlotSequence('cs82b')

	for r,c,f,r0,r1,d0,d1 in zip(F.run, F.camcol, F.field,
								 F.ra0, F.ra1, F.dec0, F.dec1):
		#for band in 'ugriz':

		margin = 10. / 3600.
		print 'Cutting to sources in range of image.'
		Ti = T[(T.ra  + margin >= r0) * (T.ra  - margin <= r1) *
			   (T.dec + margin >= d0) * (T.dec - margin <= d1)]
		print len(Ti), 'CS82 sources in range'

		print 'Creating Tractor sources...'
		cat,icat = get_cs82_sources(Ti, maglim=maglim)
		print 'Got', len(cat), 'sources'

		for band in 'i':
			tim,inf = get_tractor_image_dr9(r, c, f, band, sdss=sdss,
											nanomaggies=True, zrange=[-2,5],
											invvarIgnoresSourceFlux=True)
			tr = Tractor([tim], cat)
			print tr

			fn = 'prof-cs82b-%s.dat' % (datetime.now().isoformat())
			locs = dict(tr=tr, ps=ps, band=band)
			cProfile.runctx('runone(tr,ps,band)', globals(), locs, fn)


		break

if __name__ == '__main__':
	#cProfile.run('main()', 'prof-cs82b-%s.dat' % (datetime.now().isoformat()))
	main()

