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
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.sdss import *
from astrometry.libkd.spherematch import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params
from tractor.ttime import *

import cProfile
from datetime import datetime


def get_cs82_sources(T, maglim=25, mags=['u','g','r','i','z']):
	srcs = Catalog()
	isrcs = []
	for i,t in enumerate(T):
		if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
			#print 'PSF'
			themag = t.mag_psf
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

		dmag = t.mag_spheroid
		emag = t.mag_disk

		# SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

		if dmag <= maglim:
			shape_dev = GalaxyShape(t.spheroid_reff_world * 3600.,
									t.spheroid_aspect_world,
									t.spheroid_theta_world + 90.)

		if emag <= maglim:
			shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600.,
									t.disk_aspect_world,
									t.disk_theta_world + 90.)

		pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

		isrcs.append(i)
		if emag > maglim and dmag <= maglim:
			nm = NanoMaggies.magToNanomaggies(dmag)
			m_dev = NanoMaggies(order=mags, **dict([(k, nm) for k in mags]))
			srcs.append(DevGalaxy(pos, m_dev, shape_dev))
			continue
		if emag <= maglim and dmag > maglim:
			nm = NanoMaggies.magToNanomaggies(emag)
			m_exp = NanoMaggies(order=mags, **dict([(k, nm) for k in mags]))
			srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
			continue

		#srcs.append(CompositeGalaxy(pos, m_exp, shape_exp, m_dev, shape_dev))

		nmd = NanoMaggies.magToNanomaggies(dmag)
		nme = NanoMaggies.magToNanomaggies(emag)
		nm = nmd + nme
		fdev = (nmd / nm)
		m = NanoMaggies(order=mags, **dict([(k, nm) for k in mags]))
		srcs.append(FixedCompositeGalaxy(pos, m, fdev, shape_exp, shape_dev))

	#print 'Sources:', len(srcs)
	return srcs, np.array(isrcs)


#def runfield((r, c, f, band, basename, r0,r1,d0,d1, T, opt)):
def runfield((r, c, f, band, basename, r0,r1,d0,d1, opt, cat,icat)):
	print 'Runfield:', r,c,f,band
	sdss = DR9(basedir='cs82data/dr9')

	tim,inf = get_tractor_image_dr9(r, c, f, band, sdss=sdss,
									nanomaggies=True, zrange=[-2,5],
									invvarIgnoresSourceFlux=True)
	(H,W) = tim.shape
	tim.wcs.setConstantCd(W/2., H/2.)
	tr = Tractor([tim], cat)
	#print tr

	#iv = tim.invvar
	#print 'invvar range:', iv[iv>0].min(), iv[iv>0].max()
	#basename = 'cs82-%s-r%04ic%if%04ib%s' % (cs82field, r, c, f, band)
	#fn = 'prof-cs82b-%s.dat' % (datetime.now().isoformat())
	#locs = dict(tr=tr, ps=ps, band=band, opt=opt, basename=basename)
	#cProfile.runctx('runone(tr,ps,band,opt,basename)', globals(), locs, fn)
	#runone(tr, ps, band, opt, basename)
	#args.append((tr, ps, band, opt, basename))
	#def runone(tr, ps, band, opt):

	t00 = Time()

	ps = PlotSequence(basename)

	tr.freezeParam('images')
	cat = tr.catalog
	tim = tr.getImages()[0]
	
	cat.freezeParamsRecursive('*')
	cat.thawPathsTo(band)

	cat0 = cat.getParams()
	br0 = [src.getBrightness().copy() for src in cat]
	nm0 = np.array([b.getBand(band) for b in br0])

	sig = 1./np.median(tim.getInvError())
	#print 'Image sigma:', sig

	#minsb = 0.001 * sig
	#minsb = 1. * sig
	minsb = 0.1 * sig

	img = tim.getImage()

	#print 'Finding overlapping sources...'
	#t0 = Time()
	groups,L = tr.getOverlappingSources(0, minsb=minsb)
	#print 'Overlapping sources took', Time()-t0
	#print 'Got', len(groups), 'groups of sources'

	zr = tim.zr
	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1], cmap='gray')
	imchi = dict(interpolation='nearest', origin='lower',
				 vmin=-5, vmax=5, cmap='gray')
	imbin = dict(interpolation='nearest', origin='lower',
				 vmin=0, vmax=1, cmap='gray')
	imx = dict(interpolation='nearest', origin='lower')

	if opt.plots:
		print 'Getting initial model...'
		t0 = Time()
		mod = tr.getModelImage(0, minsb=minsb)
		print 'initial model took', Time()-t0

		plt.clf()
		plt.imshow(img, **ima)
		plt.title('Image %s' % tim.name)
		ps.savefig()

		plt.clf()
		plt.imshow(mod, **ima)
		plt.title('Mod %s -- 0' % tim.name)
		ps.savefig()

		plt.clf()
		plt.imshow(tim.getInvError(), vmin=0, **imx)
		plt.colorbar()
		plt.title('InvError %s -- 0' % tim.name)
		ps.savefig()

	# Sort the groups by the chi-squared values they contain
	#print 'Getting chi image...'
	#t0 = Time()
	chi = tr.getChiImage(0, minsb=minsb)
	#print 'Chi image took', Time()-t0

	#t0 = Time()
	nl = L.max()
	gslices = find_objects(L, nl)
	#print 'find_objects took', Time()-t0
	#t0 = Time()
	chisq = []
	for i,gs in enumerate(gslices):
		subL = L[gs]
		subchi = chi[gs]
		c = np.sum(subchi[subL == (i+1)]**2)
		chisq.append(c)
	Gorder = np.argsort(-np.array(chisq))
	#print 'Sorting objects took', Time()-t0

	for gi,gl in enumerate(Gorder):
		#print
		# note, gslices is zero-indexed
		gslice = gslices[gl]
		gl += 1
		if not gl in groups:
			print 'Group', gl, 'not in groups array; skipping'
			continue
		gsrcs = groups[gl]
		#print 'Group number', (gi+1), 'of', len(Gorder), ', id', gl, ': sources', gsrcs

		tgroups = np.unique(L[gslice])
		tsrcs = []
		for g in tgroups:
			if not g in [gl,0]:
				if g in groups:
					tsrcs.extend(groups[g])
		#print 'sources in groups touching slice:', tsrcs

		fullcat = tr.catalog
		subcat = Catalog(*[fullcat[i] for i in gsrcs + tsrcs])
		for i in range(len(tsrcs)):
			subcat.freezeParam(len(gsrcs) + i)
		tr.catalog = subcat

		#print 'Thawed params:'
		#tr.printThawedParams()

		#t0 = Time()
		ims0,ims1 = tr.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
															rois=[gslice])
		#print 'optimize_forced_photometry took', Time()-t0

		#print 'After params:'
		#tr.printThawedParams()

		tr.catalog = fullcat

		if opt.plots and gi % 10 == 0 and gi < 250:
			(im,mod0,chi0,roi0) = ims0[0]
			if ims1 is not None:
				(im,mod1,chi1,roi1) = ims1[0]

			gx,gy = [],[]
			wcs = tim.getWcs()
			for src in gsrcs:
				x,y = wcs.positionToPixel(tr.catalog[src].getPosition())
				gx.append(x)
				gy.append(y)
			tx,ty = [],[]
			for src in tsrcs:
				x,y = wcs.positionToPixel(tr.catalog[src].getPosition())
				tx.append(x)
				ty.append(y)

			plt.clf()
			plt.subplot(2,3,1)
			sy,sx = gslice
			x0,x1,y0,y1 = [sx.start, sx.stop, sy.start, sy.stop]
			margin = 25
			H,W = img.shape
			ext = [max(0, x0-margin), min(W-1, x1+margin),
				   max(0, y0-margin), min(H-1, y1+margin)]
			plt.imshow(img[ext[2]:ext[3]+1, ext[0]:ext[1]+1], extent=ext, **ima)
			ax = plt.axis()
			plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'r-')
			plt.plot(gx, gy, 'rx')
			plt.plot(tx, ty, 'gx')
			plt.axis(ax)
			plt.title('data (context)')
			plt.subplot(2,3,2)
			plt.imshow(mod0, **ima)
			plt.title('cs82 model')
			plt.xticks([]); plt.yticks([])
			plt.subplot(2,3,3)
			plt.imshow(chi0, **imchi)
			plt.title('cs82 chi')
			plt.xticks([]); plt.yticks([])
			plt.subplot(2,3,4)
			plt.imshow(im, **ima)
			plt.title('data')
			plt.xticks([]); plt.yticks([])
			if ims1 is not None:
				plt.subplot(2,3,5)
				plt.imshow(mod1, **ima)
				plt.title('fit model')
				plt.xticks([]); plt.yticks([])
				plt.subplot(2,3,6)
				plt.imshow(chi1, **imchi)
				plt.title('fit chi')
				plt.xticks([]); plt.yticks([])
			ps.savefig()

	cat.thawPathsTo(band)
	cat1 = cat.getParams()
	br1 = [src.getBrightness().copy() for src in cat]
	nm1 = np.array([b.getBand(band) for b in br1])

	print 'Runfield:', r,c,f,band, 'took', Time()-t00

	mags0 = NanoMaggies.nanomaggiesToMag(nm0)
	mags1 = NanoMaggies.nanomaggiesToMag(nm1)

	print 'mags0:', mags0
	print 'mags1:', mags1

	plt.clf()
	I = (mags0 == mags1)
	plt.plot(mags0[I], mags1[I], 'r.', alpha=0.5)
	I = (mags0 != mags1)
	plt.plot(mags0[I], mags1[I], 'b.', alpha=0.5)
	plt.xlabel('CS82 i-band (mag)')
	plt.ylabel('SDSS %s-band (mag)' % band)
	plt.title('Forced photometry of %s' % tim.name)
	plt.axis([25,8, 25,8])
	ps.savefig()

	plt.clf()
	I = (mags0 == mags1)
	plt.plot(mags1[I]-mags0[I], mags0[I], 'r.', alpha=0.5)
	I = (mags0 != mags1)
	plt.plot(mags1[I]-mags0[I], mags0[I], 'b.', alpha=0.5)
	plt.xlabel('SDSS %s-band - CS82 i-band (mag)' % band)
	plt.ylabel('CS82 i-band (mag)')
	plt.title('Forced photometry of %s' % tim.name)
	plt.axis([-8,8, 25,8])
	ps.savefig()

	T = tabledata()
	T.cs82_mag_i = mags0
	T.set('sdss_mag_%s' % band, mags1)
	T.cs82_index = icat
	fn = 'mags-%s.fits' % basename
	print 'Writing:'
	T.about()
	T.writeto(fn)
	print 'Wrote', fn


def getTables(cs82field, enclosed=True, extra_cols=[]):
	T = fits_table('cs82data/%s_y.V2.7A.swarp.cut.deVexp.fit' % cs82field,
				   hdu=2, column_map={'ALPHA_J2000':'ra',
									  'DELTA_J2000':'dec'},
				   columns=[x.upper() for x in
							['ALPHA_J2000', 'DELTA_J2000',
							'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
							 'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
							 'disk_theta_world', 'spheroid_reff_world',
							 'spheroid_aspect_world', 'spheroid_theta_world',
							 'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
	ra0,ra1 = T.ra.min(), T.ra.max()
	dec0,dec1 = T.dec.min(), T.dec.max()
	print 'RA', ra0,ra1
	print 'Dec', dec0,dec1
	T.index = np.arange(len(T))

	# ASSUME no RA wrap-around in the catalog
	trad = 0.5 * np.hypot(ra1 - ra0, dec1 - dec0)
	tcen = radectoxyz((ra1+ra0)*0.5, (dec1+dec0)*0.5)

	frad = 0.5 * np.hypot(13., 9.) / 60.

	fn = 'sdssfield-%s.fits' % cs82field
	if os.path.exists(fn):
		print 'Reading', fn
		F = fits_table(fn)
	else:
		F = fits_table('window_flist-DR9.fits')

		# These runs don't appear in DAS
		#F.cut((F.run != 4322) * (F.run != 4240) * (F.run != 4266))
		F.cut(F.rerun != "157")

		# For Stripe 82, mu-nu is aligned with RA,Dec.
		rd = []
		rd.append(munu_to_radec_deg(F.mu_start, F.nu_start, F.node, F.incl))
		rd.append(munu_to_radec_deg(F.mu_end,   F.nu_end,   F.node, F.incl))
		rd = np.array(rd)
		F.ra0  = np.min(rd[:,0,:], axis=0)
		F.ra1  = np.max(rd[:,0,:], axis=0)
		F.dec0 = np.min(rd[:,1,:], axis=0)
		F.dec1 = np.max(rd[:,1,:], axis=0)

		I = np.flatnonzero((F.ra0 <= T.ra.max()) *
						   (F.ra1 >= T.ra.min()) *
						   (F.dec0 <= T.dec.max()) *
						   (F.dec1 >= T.dec.min()))
		print 'Possibly overlapping fields:', len(I)
		F.cut(I)

		# When will I ever learn not to cut on RA boxes when there is wrap-around?
		xyz = radectoxyz(F.ra, F.dec)
		r2 = np.sum((xyz - tcen)**2, axis=1)
		I = np.flatnonzero(r2 < deg2distsq(trad + frad))
		print 'Possibly overlapping fields:', len(I)
		F.cut(I)

		F.enclosed = ((F.ra0 >= T.ra.min()) *
					  (F.ra1 <= T.ra.max()) *
					  (F.dec0 >= T.dec.min()) *
					  (F.dec1 <= T.dec.max()))
		
		# Sort by distance from the center of the field.
		ra  = (T.ra.min()  + T.ra.max() ) / 2.
		dec = (T.dec.min() + T.dec.max()) / 2.
		I = np.argsort( ((F.ra0  + F.ra1 )/2. - ra )**2 +
						((F.dec0 + F.dec1)/2. - dec)**2 )
		F.cut(I)

		F.writeto(fn)
		print 'Wrote', fn

	if enclosed:
		F.cut(F.enclosed)
		print 'Enclosed fields:', len(F)
		
	return T,F

def makecmd(opt, cs82field):
	T,F = getTables(cs82field)
	urun = np.unique(F.run)
	print len(urun), 'unique runs'

	rmap = dict([(r,i) for i,r in enumerate(urun)])

	for band in 'ugriz':
		T.set('all_%s' % band, np.zeros((len(T), len(urun)), np.float32))

	for r,c,f,r0,r1,d0,d1 in zip(F.run, F.camcol, F.field,
								 F.ra0, F.ra1, F.dec0, F.dec1):
		ri = rmap[r]
		print 'Run', r, '->', ri
		for band in 'ugriz':
			basename = 'cs82-%s-%04i-%i-%04i-%s' % (cs82field, r, c, f, band)
			fn = 'mags-%s.fits' % basename
			if not os.path.exists(fn):
				print 'No such file', fn, '-- skipping'
				continue
			print 'Reading', fn
			M = fits_table(fn)
			print 'Got', len(M)
			m = M.get('sdss_mag_%s' % band)
			m[np.logical_not(np.isfinite(m))] = 0.
			I = M.cs82_index
			T.get('all_%s' % band)[I, ri] = m

	T.mag = np.zeros_like(T.mag_psf)
	T.mag[T.chi2_psf < T.chi2_model] = T.mag_psf
	J = T.chi2_psf >= T.chi2_model
	T.mag[J] = NanoMaggies.nanomaggiesToMag(
		NanoMaggies.magToNanomaggies(T.mag_disk[J]) +
		NanoMaggies.magToNanomaggies(T.mag_spheroid[J]))

	for band in 'ugriz':
		smag = T.get('all_%s' % band)
		
		mn = []
		st = []
		cc = []
		ss = []
		V = []
		for i in range(len(T)):
			J = np.flatnonzero(smag[i,:])
			if len(J) == 0:
				continue
			V.append(i)
			s = smag[i,J]
			ss.append(s)
			c = np.zeros_like(s) + T.mag[i]
			cc.append(c)
			mn.append(np.mean(s))
			st.append(np.std(s))
		V = np.array(V)
		Ti = T[V]
		J = (Ti.chi2_psf < Ti.chi2_model)
		plt.clf()
		plt.plot(np.hstack([c for c,s,psf in zip(cc,ss,J) if psf]),
				 np.hstack([s for c,s,psf in zip(cc,ss,J) if psf]), 'b.', alpha=0.1, zorder=20)
		plt.plot(np.hstack([c for c,s,psf in zip(cc,ss,J) if not psf]),
				 np.hstack([s for c,s,psf in zip(cc,ss,J) if not psf]), 'b.', alpha=0.1, zorder=20)
		mn = np.array(mn)
		st = np.array(st)
		plt.errorbar(Ti.mag[J], mn[J], yerr=st[J], fmt='.', color='r', alpha=0.5, zorder=25)
		K = np.logical_not(J)
		plt.errorbar(Ti.mag[K], mn[K], yerr=st[K], fmt='.', color='b', alpha=0.5, zorder=25)
		plt.xlim(26, 8)
		plt.ylim(26, 8)
		plt.savefig('mm-%s.png' % band)

		plt.clf()
		plt.plot(np.hstack([c - s for c,s,psf in zip(cc,ss,J) if psf]),
				 np.hstack([c for c,s,psf in zip(cc,ss,J) if psf]), 'b.', alpha=0.1, zorder=20)
		plt.errorbar(Ti.mag[J] - mn[J], Ti.mag[J], xerr=st[J], fmt='.', color='b', alpha=0.5, zorder=25)
		plt.ylim(26, 8)
		plt.xlim(-10,10)
		plt.xlabel('CFHT i - SDSS %s (mag)' % band)
		plt.ylabel('CFHT i (mag)')
		plt.savefig('cm-stars-%s.png' % band)
		
		plt.clf()
		plt.plot(np.hstack([c - s for c,s,psf in zip(cc,ss,J) if not psf]),
				 np.hstack([c for c,s,psf in zip(cc,ss,J) if not psf]), 'r.', alpha=0.1, zorder=20)
		plt.errorbar(Ti.mag[K] - mn[K], Ti.mag[K], xerr=st[K], fmt='.', color='r', alpha=0.5, zorder=25)
		plt.ylim(26, 8)
		plt.xlim(-10,10)
		plt.xlabel('CFHT i - SDSS %s (mag)' % band)
		plt.ylabel('CFHT i (mag)')
		plt.savefig('cm-gals-%s.png' % band)

		plt.clf()
		plt.plot(np.hstack([c - s for c,s,psf in zip(cc,ss,J) if not psf]),
				 np.hstack([c for c,s,psf in zip(cc,ss,J) if not psf]), 'r.', alpha=0.1, zorder=20)
		plt.ylim(26, 8)
		plt.xlim(-10,10)
		plt.xlabel('CFHT i - SDSS %s (mag)' % band)
		plt.ylabel('CFHT i (mag)')
		plt.savefig('cm-gals-%s-1.png' % band)

		plt.clf()
		plt.errorbar(Ti.mag[K] - mn[K], Ti.mag[K], xerr=st[K], fmt='.', color='r', alpha=0.5, zorder=25)
		plt.ylim(26, 8)
		plt.xlim(-10,10)
		plt.xlabel('CFHT i - SDSS %s (mag)' % band)
		plt.ylabel('CFHT i (mag)')
		plt.savefig('cm-gals-%s-2.png' % band)

		plt.clf()
		loghist(Ti.mag[K] - mn[K], Ti.mag[K], bins=200, range=((-10,10),(8,26)))
		plt.ylim(26, 8)
		plt.savefig('cm-gals-%s-3.png' % band)

		plt.clf()
		loghist(Ti.mag[J] - mn[J], Ti.mag[J], bins=200, range=((-10,10),(8,26)))
		plt.ylim(26, 8)
		plt.savefig('cm-stars-%s-3.png' % band)


		xl,xh = -4,4
		yl,yh = 16,24

		G = np.flatnonzero(K)
		
		eflux = NanoMaggies.magToNanomaggies(Ti.mag_disk)
		dflux = NanoMaggies.magToNanomaggies(Ti.mag_spheroid)
		D = np.flatnonzero(K * (dflux > eflux * 10.))
		E = np.flatnonzero(K * (eflux > dflux * 10.))
		C = np.flatnonzero(K * (dflux <= eflux * 10.) * (eflux <= dflux * 10.))
		
		for j,(I,txt) in enumerate([ (G, 'galaxies'), (D, 'deV galaxies'), (E, 'exp galaxies'), (C, 'comp galaxies') ]):
			plt.clf()
			loghist(Ti.mag[I] - mn[I], Ti.mag[I], bins=200, range=((xl,xh),(yl,yh)))
			plt.ylim(yh,yl)
			plt.xlabel('CFHT i - SDSS %s (mag)' % band)
			plt.ylabel('CFHT i (mag)')
			plt.title('%s forced photometry (%i %s)' % (cs82field, len(I), txt))
			plt.savefig('cm-gals-%s-%i.png' % (band, 4+j))


def main(opt, cs82field):
	T,F = getTables(cs82field)
	mp = multiproc(opt.threads)

	results = []
	args = []
	alldone = False
	for r,c,f,r0,r1,d0,d1 in zip(F.run, F.camcol, F.field,
								 F.ra0, F.ra1, F.dec0, F.dec1):

		dobands = []
		for band in 'ugriz':
			basename = 'cs82-%s-%04i-%i-%04i-%s' % (cs82field, r, c, f, band)
			outfn = 'mags-%s.fits' % basename
			if opt.skipExisting and os.path.exists(outfn):
				print 'File', outfn, 'exists, skipping.'
				continue
			dobands.append(band)

		if len(dobands) == 0:
			continue

		margin = 10. / 3600.
		print 'Cutting to sources in range of image.'
		Ti = T[(T.ra  + margin >= r0) * (T.ra  - margin <= r1) *
			   (T.dec + margin >= d0) * (T.dec - margin <= d1)]
		print len(Ti), 'CS82 sources in range'
		print 'Creating Tractor sources...'
		maglim = 24
		cat,icat = get_cs82_sources(Ti, maglim=maglim)
		print 'Got', len(cat), 'sources'

		realinds = Ti.index[icat]

		for band in dobands:
			basename = 'cs82-%s-%04i-%i-%04i-%s' % (cs82field, r, c, f, band)
			res = mp.apply(runfield, ((r,c,f,band,basename, r0,r1,d0,d1, opt, cat,realinds),))
			results.append(res)
			if opt.nfields and len(results) >= opt.nfields:
				alldone = True
				break

		if alldone:
			break

	print len(results), 'async jobs'
	for r in results:
		print '  waiting for', r
		if r is None:
			continue
		r.wait()
	print 'Done!'
	


def simulfit(opt, cs82field):
	T,F = getTables(cs82field, enclosed=False)

	# We probably have to work in Dec slices to keep the memory reasonable
	dec0 = T.dec.min()
	dec1 = T.dec.max()
	print 'Dec range:', dec0, dec1
	#nslices = 4
	#ddec = (dec1 - dec0) / nslices
	#print 'ddec:', ddec

	sdss = DR9(basedir='cs82data/dr9')

	### HACK -- ignore 0/360 issues
	ra0 = T.ra.min()
	ra1 = T.ra.max()
	print 'RA range:', ra0, ra1
	assert(ra1 - ra0 < 2.)

	decs = np.linspace(dec0, dec1, 5)
	ras  = np.linspace(ra0,  ra1, 5)

	print 'Score range:', F.score.min(), F.score.max()
	print 'Before score cut:', len(F)
	F.cut(F.score > 0.5)
	print 'Cut on score:', len(F)

	ps = PlotSequence('simul')

	for decslice,(dlo,dhi) in enumerate(zip(decs, decs[1:])):
		print 'Dec slice:', dlo, dhi
		for raslice,(rlo,rhi) in enumerate(zip(ras, ras[1:])):
			print 'RA slice:', rlo, rhi

			# in deg
			margin = 15. / 3600.
			Ti = T[((T.dec + margin) >= dlo) * ((T.dec - margin) <= dhi) *
				   ((T.ra  + margin) >= rlo) * ((T.ra  - margin) <= rhi)]
			Ti.marginal = np.logical_not((Ti.dec >= dlo) * (Ti.dec <= dhi) *
										 (Ti.ra  >= rlo) * (Ti.ra  <= rhi))
			print len(Ti), 'sources in RA,Dec slice'
			print len(np.flatnonzero(Ti.marginal)), 'are in the margins'

			Fi = F[np.logical_not(np.logical_or(F.dec0 > dhi, F.dec1 < dlo)) *
				   np.logical_not(np.logical_or(F.ra0  > rhi, F.ra1  < rlo))]
			print len(Fi), 'fields in RA,Dec slice'

			band = 'i'

			pixscale = 0.396 / 3600.
			W = int((rhi - rlo) / pixscale)
			H = int((dhi - dlo) / pixscale)
			print 'Fake tim:', W, H
			wcs = FitsWcs(Tan((rlo+rhi)/2., (dlo+dhi)/2., (W+1)/2., (H+1)/2.,
							  0., pixscale, pixscale, 0, W, H))
							  #-pixscale, 0, 0, pixscale, W, H))
			fwhm = 1.5
			psig = fwhm / 0.396 / 2.35
			print 'PSF sigma:', psig
			psf = GaussianMixturePSF(np.array([1.]), np.array([[0.,0.]]),
									 np.array([[[psig**2, 0],[0,psig**2]]]))
			ie = 25.
			faketim = Image(data=np.zeros((H,W), np.float32),
							invvar=np.zeros((H,W), np.float32) + (ie**2),
							psf=psf, wcs=wcs, sky=ConstantSky(0.),
							photocal=LinearPhotoCal(1., band=band),
							name='fake')

			sig = (1./ie)
			minsb = 0.1 * sig

			print 'Creating Tractor sources...'
			maglim = 24
			cat,icat = get_cs82_sources(Ti, maglim=maglim)
			print 'Got', len(cat), 'sources'
			cat.freezeParamsRecursive('*')
			cat.thawPathsTo(band)

			#cat0 = cat.getParams()
			br0 = [src.getBrightness().copy() for src in cat]
			nm0 = np.array([b.getBand(band) for b in br0])

			print 'Finding overlapping sources...'
			tr = Tractor([faketim], cat)
			groups,L = tr.getOverlappingSources(0, minsb=minsb)
			print 'Got', len(groups), 'groups of sources'
			nl = L.max()
			gslices = find_objects(L, nl)
			# find_objects returns a (zero-indexed) list corresponding to the
			# (one-indexed) objects; make a map parallel to "groups" instead
			gslices = dict([(i+1, gs) for i,gs in enumerate(gslices)])

			# Order by model flux in group
			gflux = []
			for i in range(nl+1):
				if not i in groups:
					gflux.append(0.)
					continue
				gsrcs = groups[i]
				f = sum([cat[i].getBrightness().getBand(band) for i in gsrcs])
				gflux.append(f)
			Gorder = np.argsort(-np.array(gflux))

			tims = []
			npix = 0
			ie = []
			for i,(r,c,f) in enumerate(zip(Fi.run, Fi.camcol, Fi.field)):
				print 'Reading', (i+1), 'of', len(Fi), ':', r,c,f,band
				tim,inf = get_tractor_image_dr9(r, c, f, band, sdss=sdss,
												nanomaggies=True, zrange=[-2,5],
												roiradecbox=[rlo,rhi,dlo,dhi],
												invvarIgnoresSourceFlux=True)
				if tim is None:
					continue
			
				(H,W) = tim.shape
				tim.wcs.setConstantCd(W/2., H/2.)
				#print 'CD matrix:', tim.wcs.constant_cd

				del tim.origInvvar
				del tim.starMask
				del tim.mask
				# needed for optimize_forced_photometry with rois
				#del tim.invvar

				e = np.median(tim.inverr)
				ie.append(e)

				tims.append(tim)
				npix += (H*W)
				print 'got', (H*W), 'pixels, total', npix
			
			print 'Read', len(tims), 'images'
			print 'total of', npix, 'pixels'
			memusage()

			ie = max(ie)
			print 'max inverr:', ie

			marginal = Ti.marginal[icat]
			assert(len(icat) == len(cat))
			assert(len(marginal) == len(cat))

			for gi,gl in enumerate(Gorder):
				print
				if not gl in groups:
					print 'Group', gl, 'not in groups array; skipping'
					continue
				gslice = gslices[gl]
				gsrcs = groups[gl]
				print 'Group number', (gi+1), 'of', len(Gorder), ', id', gl, ': sources', gsrcs
				tgroups = np.unique(L[gslice])
				tsrcs = []
				for g in tgroups:
					if not g in [gl,0]:
						if g in groups:
							tsrcs.extend(groups[g])

				# make a copy
				gsrcs = [i for i in gsrcs]
				rm = []
				for i in gsrcs:
					if marginal[i]:
						tsrcs.append(i)
						rm.append(i)
				for i in rm:
					gsrcs.remove(i)
				if len(gsrcs) == 0:
					print 'All sources are in the margin region'
					continue

				#print 'sources in groups touching slice:', tsrcs

				# Naively convert a slice in the "fake" image into a slice
				# in each tim.
				# This will need some work for datasets with more complicated geometry!
				sy,sx = gslice
				x0,x1,y0,y1 = [sx.start, sx.stop, sy.start, sy.stop]
				rd = []
				for x,y in [(x0,y0),(x0,y1),(x1,y0),(x1,y1)]:
					p = faketim.wcs.pixelToPosition(x, y)
					rd.append((p.ra, p.dec))
				rd = np.array(rd)
				r0,r1 = rd[:,0].min(), rd[:,0].max()
				d0,d1 = rd[:,1].min(), rd[:,1].max()

				mytims = []
				myrois = []
				for tim in tims:
					xy = []
					for r,d in [(r0,d0),(r1,d0),(r0,d1),(r1,d1)]:
						xy.append(tim.wcs.positionToPixel(RaDecPos(r,d)))
					xy = np.array(xy)
					xy = np.round(xy).astype(int)
					x0 = xy[:,0].min()
					x1 = xy[:,0].max()
					y0 = xy[:,1].min()
					y1 = xy[:,1].max()
					H,W = tim.shape
					roi = [np.clip(x0,   0, W),
						   np.clip(x1+1, 0, W),
						   np.clip(y0,   0, H),
						   np.clip(y1+1, 0, H)]
					if roi[0] == roi[1] or roi[2] == roi[3]:
						#print 'Empty roi'
						continue
					print 'Keeping image (%i x %i) with roi' % (W,H), roi, 'size %i x %i' % (roi[1]-roi[0], roi[3]-roi[2])
					mytims.append(tim)
					myrois.append((slice(y0,y1), slice(x0,x1)))

				if len(mytims) == 0:
					continue

				subcat = Catalog(*[cat[i] for i in gsrcs + tsrcs])
				for i in range(len(tsrcs)):
					subcat.freezeParam(len(gsrcs) + i)

				tr = Tractor(mytims, subcat)
				tr.freezeParam('images')
				print tr
				ims0,ims1 = tr.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
														  rois=myrois)

				#print 'ims0,ims1'
				#print ims0
				#print ims1

				if ims1 is None:
					continue

				if opt.plots:
					print 'ims0,ims1', len(ims0), len(ims1)
					n = len(ims0)
					imchi = dict(interpolation='nearest', origin='lower',
								 vmin=-5, vmax=5, cmap='gray')
					plt.figure(figsize=(20,5))
					plt.clf()
					plt.subplots_adjust(hspace=0.01, wspace=0.01,
										left=0.1, right=0.96,
										bottom=0.1, top=0.90)
					for i,((data, mod0, chi0, roi),(data, mod1, chi1, roi)) in enumerate(zip(ims0,ims1)):

						tim = mytims[i]
						zr = tim.zr
						ima = dict(interpolation='nearest', origin='lower',
								   vmin=zr[0], vmax=zr[1], cmap='gray')

						plt.subplot(5, n, i+1)
						plt.imshow(data, **ima)
						plt.xticks([]); plt.yticks([])
						plt.subplot(5, n, i+1+n)
						plt.imshow(mod0, **ima)
						plt.xticks([]); plt.yticks([])
						plt.subplot(5, n, i+1+2*n)
						plt.imshow(mod1, **ima)
						plt.xticks([]); plt.yticks([])
						plt.subplot(5, n, i+1+3*n)
						plt.imshow(chi0, **imchi)
						plt.xticks([]); plt.yticks([])
						plt.subplot(5, n, i+1+4*n)
						plt.imshow(chi1, **imchi)
						plt.xticks([]); plt.yticks([])
					ps.savefig()
				

			#cat1 = cat.getParams()
			br1 = [src.getBrightness().copy() for src in cat]
			nm1 = np.array([b.getBand(band) for b in br1])

			mags0 = NanoMaggies.nanomaggiesToMag(nm0)
			mags1 = NanoMaggies.nanomaggiesToMag(nm1)

			M = tabledata()
			M.cs82_mag_i = mags0
			M.cs82_nmag_i = nm0
			M.set('sdss_mag_%s' % band, mags1)
			M.set('sdss_nmag_%s' % band, nm1)
			M.cs82_index = Ti.index[icat]
			M.cs82_marginal = Ti.marginal[icat]
			fn = 'smags-%s-%s-%i-%i.fits' % (cs82field, band, raslice, decslice)
			M.writeto(fn)
			print 'Wrote', fn


def xmatch(opt, cs82field):
	'''
	select run,camcol,field, RA,Dec,probpsf, nchild,
	psfmag_u,psfmag_g,psfmag_r,psfmag_i,psfmag_z,
	modelmag_u, modelmag_g, modelmag_r, modelmag_i, modelmag_z
	into mydb.MyTable_2 from PhotoObjAll
	where ra between 15.77 and 16.77
	and dec between -0.11 and 0.95
	-> cas-DR7-S82-p18p.fits
	'''
	S = fits_table('cs82data/cas-DR7-S82-p18p.fits')
	T,F = getTables(cs82field, extra_cols=['mag_model'])

	print 'Got', len(S), 'CAS sources'
	print 'unique runs:', np.unique(S.run)
	print 'Got', len(T), 'CS82 sources'

	I = np.logical_or(S.run == 106, S.run == 206)
	S.cut(I)
	print 'Cut to', len(S), 'Annis coadd sources'

	R = 1. / 3600.
	I,J,d = match_radec(S.ra, S.dec, T.ra, T.dec, R)
	print 'Found', len(I), 'matches'

	Si = S[I]
	Ti = T[J]

	xl,xh = -4,4
	yl,yh = 16,24
	for band in 'ugriz':
		plt.clf()
		loghist(Ti.mag_model - Si.get('modelmag_%s' % band), Ti.mag_model,
				bins=200, range=((xl,xh),(yl,yh)))
		plt.ylim(yh,yl)
		plt.xlabel('CFHT i - SDSS %s (mag)' % band)
		plt.ylabel('CFHT i (mag)')
		plt.title('DR7 CAS Annis coadd sources')
		plt.savefig('xm-%s.png' % band)
		
if __name__ == '__main__':
	import optparse
	parser = optparse.OptionParser('%prog [options]')
	parser.add_option('--no-plots', dest='plots', default=True, action='store_false',
					  help='Do not produce plots')
	parser.add_option('--threads', dest='threads', type=int, default=1,
					  help='Multiprocessing?')
	parser.add_option('-s', '--skip', dest='skipExisting', action='store_true',
					  help='Skip fields whose outputs already exist?')
	parser.add_option('-n', dest='nfields', type=int, default=None,
					  help='Run at most this number of fields')

	parser.add_option('-S', dest='simul', action='store_true',
					  help='Simultaneous fit?')

	parser.add_option('-X', dest='xmatch', action='store_true',
					  help='Cross-match to SDSS Annis coadd')

	parser.add_option('--cmd', action='store_true',
					  help='Produce summary CMD plots')

	opt,args = parser.parse_args()
	#cProfile.run('main()', 'prof-cs82b-%s.dat' % (datetime.now().isoformat()))

	cs82field = 'S82p18p'

	if opt.xmatch:
		xmatch(opt, cs82field)
		sys.exit(0)

	if opt.cmd:
		makecmd(opt, cs82field)
		sys.exit(0)

	if opt.simul:
		simulfit(opt, cs82field)
		sys.exit(0)

	main(opt, cs82field)

