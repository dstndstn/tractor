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

	from tractor.ttime import Time

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

def main(opt):
	cs82field = 'S82p18p'

	T = fits_table('cs82data/%s_y.V2.7A.swarp.cut.deVexp.fit' % cs82field,
				   hdu=2, column_map={'ALPHA_J2000':'ra',
									  'DELTA_J2000':'dec'},
				   columns=[x.upper() for x in
							['ALPHA_J2000', 'DELTA_J2000',
							'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
							 'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
							 'disk_theta_world', 'spheroid_reff_world',
							 'spheroid_aspect_world', 'spheroid_theta_world',
							 'alphamodel_j2000', 'deltamodel_j2000']])
	print 'RA', T.ra.min(), T.ra.max()
	print 'Dec', T.dec.min(), T.dec.max()
	T.index = np.arange(len(T))

	fn = 'sdssfield-%s.fits' % cs82field
	if os.path.exists(fn):
		print 'Reading', fn
		F = fits_table(fn)
	else:
		F = fits_table('window_flist-DR9.fits')

		# These run doesn't appear in DAS
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

	# Sort by distance from the center of the field.
	ra  = (T.ra.min()  + T.ra.max() ) / 2.
	dec = (T.dec.min() + T.dec.max()) / 2.
	I = np.argsort( ((F.ra0  + F.ra1 )/2. - ra )**2 +
					((F.dec0 + F.dec1)/2. - dec)**2 )
	F.cut(I)



	if opt.cmd:

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


			xl,xh = -5,3
			yl,yh = 16,24
			
			plt.clf()
			loghist(Ti.mag[K] - mn[K], Ti.mag[K], bins=200, range=((xl,xh),(yl,yh)))
			plt.ylim(yh,yl)
			plt.xlabel('CFHT i - SDSS %s (mag)' % band)
			plt.ylabel('CFHT i (mag)')
			plt.savefig('cm-gals-%s-4.png' % band)

			plt.clf()
			loghist(Ti.mag[J] - mn[J], Ti.mag[J], bins=200, range=((xl,xh),(yl,yh)))
			plt.ylim(yh,yl)
			plt.xlabel('CFHT i - SDSS %s (mag)' % band)
			plt.ylabel('CFHT i (mag)')
			plt.savefig('cm-stars-%s-4.png' % band)


				
		return



		
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

	parser.add_option('--cmd', action='store_true',
					  help='Produce summary CMD plots')

	opt,args = parser.parse_args()
	#cProfile.run('main()', 'prof-cs82b-%s.dat' % (datetime.now().isoformat()))
	main(opt)

