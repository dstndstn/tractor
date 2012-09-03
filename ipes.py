
"""
We want to show some Tractor vs SDSS improvements using only SDSS
data.  Some validation can be done using the stripe overlaps -- the
"ipes".

Things to show / tests to run include:

-weird galaxy parameter distributions go away.	Don't even need ipes
 for this, just need a swath of imaging.

 -> galstats.py
 to select a galaxy sample
 -> ipes.py : refit_galaxies() -> mye4.fits

-our error estimates are better.  Do this by cross-matching objects in
 the ipes and comparing their measurements.	 Compare the variance of
 the two SDSS measurements to the SDSS error estimates.	 *This much is
 just evaluating SDSS results and doesn't even involve the Tractor!*
 We should find that the SDSS error estimates tend to
 over/underestimate the sample variance of the measurements.

-...then, look at the Tractor error estimates, and the two samples, on
 single fields fit separately.	We should find that the error
 estimates and the sample variance coincide.

-...then, look at the Tractor error estimates fitting the ipes
 simultaneously.  We should find that the error estimates tighten up.

-we can detect objects in the ipes that are not in the SDSS catalog.
 This requires either building a (multi-image) residual map [the
 correct approach], or guess-n-testing 2.5-sigma residuals in the
 single-image residuals [the pragmatic approach].  For main-survey
 ipes, I think I'm for the latter, since we can plug the fact that
 there's a better way to do it; and for two images the difference just
 isn't that much; and it cuts the dependency on the detection paper /
 technology.

-we do better on hard deblends / crowded regions / clusters.  I took a
 look at the Perseus cluster and the SDSS catalog is a mess, and we
 can tune it up without much trouble.  That's not the best example,
 since RHL will complain that Photo was built to perform well on the
 whole sky, so it's not surprising that it breaks down in peculiar
 places like rich clusters and Messier objects.	 That's fair, so we
 could instead look at less-extreme clusters that are still of
 significant interest and where Photo's difficulties really matter.
 This would be an opportunity to show the power of simultaneously
 fitting sky + galaxies, though that does involve writing a flexible
 sky model.	 That's required for projects with Rachel anyway so I'm
 not bothered.	For this paper, I think I want to just show the
 before-n-after pictures, rather than do any sort of numerical
 characterization of what's going on.  Do we want to show any sampling
 results?  If so, this would be the place since we could show
 covariances between shape measurements of neighbouring galaxies,
 which I think is something nobody else is even trying.




"""
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import logging

from astrometry.util.fits import *
from astrometry.blind.plotstuff import *
from astrometry.util.c import *
from astrometry.util.multiproc import multiproc
from astrometry.util.plotutils import *

from astrometry.sdss import DR9

from astrometry.libkd.spherematch import *

from tractor.sdss import *
from tractor import *


def main():

	lvl = logging.INFO
	#lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	#plot_ipes()
	#sys.exit(0)
	#refit_galaxies()

	ipe_errors()

def ipe_errors():
	#T = fits_table('ipe1_dstn.fit')
	#T.cut((T.ra < 249.85) * (T.dec < 17.65))
	#print 'Cut to', len(T)

	#T = fits_table('ipe2_dstn_1.fit')
	T = fits_table('ipe3_dstn_2.fit')
	print len(T), 'objects'

	print 'Runs', np.unique(T.run)
	print 'Camcols', np.unique(T.camcol)
	print 'Fields', np.unique(T.field)

	T1 = T[T.run == 5183]
	T2 = T[T.run == 5224]

	plt.clf()
	plt.plot(T1.ra, T1.dec, 'r.', alpha=0.1)
	plt.plot(T2.ra, T2.dec, 'bx', alpha=0.1)
	plt.savefig('ipe1.png')

	for T in [T1,T2]:
		# self-matches:
		print 'T:', len(T)
		R = 0.5 / 3600.
		I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, R, notself=True)
		print len(I), 'matches'
		K = (I < J)
		I = I[K]
		J = J[K]
		print len(I), 'symmetric'

		print sum(T.field[I] == T.field[J]), 'are in the same field'

		#plt.clf()
		#plt.plot(T.rowc[I], T.colc[I], 'r.')
		#plt.plot(T.rowc[J], T.colc[J], 'b.')
		#plt.savefig('ipe2.png')

		keep = np.ones(len(T), bool)
		keep[I[T.field[I] != T.field[J]]] = False

		T.cut(keep)
		print 'Cut to', len(T), 'with no matches in other fields'


	R = 1./3600.

	I,J,d = match_radec(T1.ra, T1.dec, T2.ra, T2.dec, R)
	print len(I), 'matches'

	dra = (T1.ra[I] - T2.ra[J])*np.cos(np.deg2rad(T1.dec[I])) * 3600.
	ddec = (T1.dec[I] - T2.dec[J]) * 3600.

	plt.clf()
	loghist(dra, ddec, 100, range=((-1,1),(-1,1)))
	plt.savefig('ipe4.png')

	X1 = np.vstack((T1.ra * np.cos(np.deg2rad(T1.dec)), T1.dec)).T
	X2 = np.vstack((T2.ra * np.cos(np.deg2rad(T2.dec)), T2.dec)).T

	I,d = nearest(X1, X2, R)
	#print 'nearest I', I
	J = np.arange(len(X2))
	K = (I >= 0)
	I = I[K]
	J = J[K]
	#print 'I', I
	#print 'J', J
	print 'Nearest-neighbour matches:', len(I)

	print 'All T1', T1.rowc.min(), T1.rowc.max(), T1.colc.min(), T1.colc.max()
	print 'Matched T1', T1.rowc[I].min(), T1.rowc[I].max(), T1.colc[I].min(), T1.colc[I].max()

	print 'All T2', T2.rowc.min(), T2.rowc.max(), T2.colc.min(), T2.colc.max()
	print 'Matched T2', T2.rowc[J].min(), T2.rowc[J].max(), T2.colc[J].min(), T2.colc[J].max()

	dra = (T1.ra[I] - T2.ra[J])*np.cos(np.deg2rad(T1.dec[I])) * 3600.
	ddec = (T1.dec[I] - T2.dec[J]) * 3600.

	plt.clf()
	loghist(dra, ddec, 100, range=((-1,1),(-1,1)))
	plt.savefig('ipe3.png')
	
	# Errors are in arcsec.
	rerr1 = T1.raerr[I]
	derr1 = T1.decerr[I]
	rerr2 = T2.raerr[J]
	derr2 = T2.decerr[J]

	hi = 6.
	S = 1./np.sqrt(2.)
	plt.clf()
	n,b,p = plt.hist(S * np.hypot(dra, ddec) / np.hypot(rerr1, derr1), 100, range=(0, hi), histtype='step', color='r')
	plt.hist(S * np.hypot(dra, ddec) / np.hypot(rerr2, derr2), 100, range=(0, hi), histtype='step', color='b')
	xx = np.linspace(0, hi, 500)
	from scipy.stats import chi
	yy = chi.pdf(xx, 2)
	plt.plot(xx, yy * len(dra) * (b[1]-b[0]), 'k-')
	plt.xlim(0,hi)
	plt.xlabel('N sigma of RA,Dec repeat observations')
	plt.ylabel('Number of sources')
	plt.savefig('ipe5.png')

	for c,cerr,nn in [('psfmag_r', 'psfmagerr_r', 6)]:
		plt.clf()
		n,b,p = plt.hist(S * np.abs(T1.get(c)[I] - T2.get(c)[J]) / T1.get(cerr)[I], 100, range=(0,hi), histtype='step', color='r')
		plt.xlabel('N sigma of ' + c)
		xx = np.linspace(0, hi, 500)
		yy = 2./np.sqrt(2.*np.pi)*np.exp(-0.5 * xx**2)
		print 'yy', sum(yy)
		plt.plot(xx, yy * len(I) * (b[1]-b[0]), 'k-')
		plt.savefig('ipe%i.png' % nn)

	# Pairs where both are galaxies
	K = ((T1.type[I] == 3) * (T2.type[J] == 3))
	G1 = T1[I[K]]
	G2 = T2[J[K]]
	print len(G1), 'galaxies'
	
	print (np.sum((T1.type[I] == 3) * (T2.type[J] != 3)) +
		   np.sum((T1.type[I] != 3) * (T2.type[J] == 3))), 'galaxy type-mismatches'
	
	for c,cerr,nn in [('modelmag_r', 'modelmagerr_r', 7),]:
		plt.clf()
		n,b,p = plt.hist(S * np.abs(G1.get(c) - G2.get(c)) / G1.get(cerr), 100, range=(0,hi),
						 histtype='step', color='r')
		plt.xlabel('N sigma of ' + c)
		yy = np.exp(-0.5 * b**2)
		yy *= sum(n) / np.sum(yy)
		plt.plot(b, yy, 'k-')
		plt.savefig('ipe%i.png' % nn)

	plt.clf()
	loghist(G1.fracdev_r, G2.fracdev_r, 100)
	plt.xlabel('G1 fracdev_r')
	plt.ylabel('G2 fracdev_r')
	plt.savefig('ipe8.png')

	S = 1.
	for t in ['exp', 'dev']:
		if t == 'exp':
			I = (G1.fracdev_r < 0.5) * (G2.fracdev_r < 0.5)
			print sum(I), 'of', len(G1), 'both have fracdev_r < 0.5'
		else:
			I = (G1.fracdev_r >= 0.5) * (G2.fracdev_r >= 0.5)
			print sum(I), 'of', len(G1), 'both have fracdev_r >= 0.5'
		H1 = G1[I]
		H2 = G2[I]
			
		for c,cerr,nn in [('%smag_r', '%smagerr_r', 9),
						  ('%sab_r',  '%saberr_r', 10),
						  ('%srad_r', '%sraderr_r', 11),
						  ]:
			cc = c % t
			ccerr = cerr % t
			dval = np.abs(H1.get(cc) - H2.get(cc))
			derr = H1.get(ccerr)
			plt.clf()
			n,b,p = plt.hist(S * dval / derr, 100, range=(0,hi),
							 histtype='step', color='r')
			plt.xlabel('N sigma of ' + cc)
			#yy = np.exp(-0.5 * b**2)
			#yy *= sum(n) / np.sum(yy)
			#plt.plot(b, yy, 'k-')
			plt.savefig('ipe%i%s.png' % (nn,t))


	return






	
	I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, 0.5/3600., notself=True)
	print len(I), 'matches'

	plt.clf()
	loghist((T.ra[I] - T.ra[J])*np.cos(np.deg2rad(T.dec[I])) * 3600.,
			(T.dec[I] - T.dec[J])*3600., 100, range=((-1,1),(-1,1)))
	plt.savefig('ipe4.png')


	K = (I < J)
	I = I[K]
	J = J[K]
	d = d[K]
	print 'Cut to', len(I), 'symmetric'
	
	plt.clf()
	plt.plot(T.ra, T.dec, 'r.')
	plt.plot(T.ra[I], T.dec[I], 'bo', mec='b', mfc='none')
	plt.savefig('ipe2.png')

	dra,ddec = [],[]
	raerr,decerr = [],[]
	RC = T.run * 10 + T.camcol
	RCF = T.run * 10 * 1000 + T.camcol * 1000 + T.field
	for i in np.unique(I):
		K = (I == i)
		JJ = J[K]
		print
		print 'Source', i, 'has', len(JJ), 'matches'
		print '  ', np.sum(RC[JJ] == RC[i]), 'in the same run/camcol'
		print '  ', np.sum(RCF[JJ] == RCF[i]), 'in the same run/camcol/field'
		orc = (RC[JJ] != RC[i])
		print '  ', np.sum(orc), 'are in other run/camcols'
		print '  ', len(np.unique(RC[JJ][orc])), 'unique other run/camcols'
		print '  ', len(np.unique(RCF[JJ][orc])), 'unique other run/camcols/fields'
		print '  other sources:', JJ
		dra.extend((T.ra[JJ] - T.ra[i]) * np.cos(np.deg2rad(T.dec[i])))
		ddec.extend(T.dec[JJ] - T.dec[i])
		raerr.extend ([T.raerr [i]] * len(JJ))
		decerr.extend([T.decerr[i]] * len(JJ))

	dra,ddec = np.array(dra), np.array(ddec)
	raerr,decerr = np.array(raerr), np.array(decerr)

	plt.clf()
	plt.hist(np.hypot(dra,ddec) / np.hypot(raerr,decerr), 100)
	plt.savefig('ipe3.png')


def refit_galaxies():
	import optparse
	parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')
	parser.add_option('--threads', dest='threads', type=int, default=1,
					  help='use multiprocessing')
	opt,args = parser.parse_args()
	mp = multiproc(nthreads=opt.threads)

	'''
	select fracdev_i,exprad_i,expab_i,expmag_i,expphi_i,run,camcol,field,ra,dec,flags
	into mydb.exp5 from Galaxy where
	exprad_i > 3
	and expmag_i < 20
	and clean=1 and probpsf=0
	and fracdev_i < 0.5
	
	select * from mydb.exp5 into mydb.exp5b
	where (flags & 0x10000000) = 0x10000000
	and (flags & 0x3002000a0020008) = 0
	'''

	#T = fits_table('exp4_dstn.fit')
	T = fits_table('exp5b_dstn.fit')

	sdss = DR9(basedir='paper0-data-dr9')

	print 'basedir', sdss.basedir
	print 'dasurl', sdss.dasurl

	bandname = 'i'
	# ROI radius in pixels
	S = 100

	rlo,rhi = 4.1, 4.4
	Ti = T[(T.exprad_i > rlo) * (T.exprad_i < rhi)]
	Ti = Ti[Ti.expmag_i < 19]
	I = np.argsort(Ti.expmag_i)

	###
	#I = I[:4]
	#I = I[7:8]

	Ti = Ti[I]

	print 'Cut to', len(Ti), 'galaxies in radius and mag cuts'

	for prefix in ['my_', 'init_', 'sw_']:
		for c in ['exprad_i', 'expab_i', 'expphi_i', 'expmag_i', 'ra', 'dec']:
			Ti.set(prefix + c, np.zeros_like(Ti.get(c)))
		for c in ['devrad_i', 'devab_i', 'devphi_i', 'devmag_i']:
			Ti.set(prefix + c, np.zeros_like(Ti.get(c.replace('dev','exp'))))
		Ti.set(prefix + 'type', np.chararray(len(Ti), 1))
	Ti.set('sw_dlnp', np.zeros(len(Ti), np.float32))

	args = []

	for gali in range(len(Ti)):
		ti = Ti[gali]
		#pickle_to_file(ti, '/tmp/%04i.pickle' % gali)
		#ti.about()
		args.append((ti, bandname, S, sdss, gali))

	#tinew = mp.map(_refit_gal, args)

	# Run in blocks.
	tinew = []
	B = 0
	while len(args):

		N = 100
		#N = 4
		B += N
		thisargs = args[:N]
		args = args[N:]
		#print 'thisargs:', thisargs
		thisres = mp.map(_refit_gal, thisargs)
		#print 'thisres:', thisres
		tinew.extend(thisres)
		#print 'tinew:', tinew

		for gali in range(min(len(Ti), len(tinew))):
			tin = tinew[gali]
			if tin is None:
				print 'Skipping', gali
				continue
			Ti[gali] = tin
		#Ti.about()

		Ti.writeto('mye4-%06i.fits' % B)
	Ti.writeto('mye4.fits')


def _refit_gal((ti, bandname, S, sdss, gali)):
	try:
		return _real_refit_gal((ti, bandname, S, sdss, gali))
	except:
		import traceback
		traceback.print_exc()
		return None


def _real_refit_gal((ti, bandname, S, sdss, gali)):
	im,info = get_tractor_image_dr9(ti.run, ti.camcol, ti.field, bandname,
									roiradecsize=(ti.ra, ti.dec, S),
									sdss=sdss)
	roi = info['roi']
	cat = get_tractor_sources_dr9(ti.run, ti.camcol, ti.field, bandname,
								  sdss=sdss, roi=roi, bands=[bandname])

	tractor = Tractor(Images(im), cat)
	print 'Tractor', tractor

	ima = dict(interpolation='nearest', origin='lower')
	zr = im.zr
	ima.update(vmin=zr[0], vmax=zr[1])
	ima.update(extent=roi)
	imchi = ima.copy()
	imchi.update(vmin=-5, vmax=5)

	mod0 = tractor.getModelImage(0)
	chi0 = tractor.getChiImage(0)

	tractor.freezeParam('images')
	p0 = tractor.getParams()


	# Find the galaxy in question
	im = tractor.getImage(0)
	wcs = im.getWcs()
	dd = 1e6
	ii = None
	xc,yc = wcs.positionToPixel(RaDecPos(ti.ra, ti.dec))
	for i,src in enumerate(tractor.catalog):
		pos = src.getPosition()
		x,y = wcs.positionToPixel(pos)
		d = np.hypot(x-xc, y-yc)
		if d < dd:
			ii = i
			dd = d
	assert(ii is not None)
	print 'Closest to image center:', tractor.catalog[ii]
	gal = tractor.catalog[ii]

	gal0 = gal.copy()

	set_table_from_galaxy(ti, gal0, 'init_')

	while True:
		dlnp,X,alpha = tractor.optimize(damp=1e-3)
		print 'dlnp', dlnp
		print 'alpha', alpha
		if dlnp < 1:
			# p0 = np.array(tractor.getParams())
			# dp = np.array(X)				
			# tractor.setParams(p0 + dp)
			# modi = tractor.getModelImage(0)
			# tractor.setParams(p0)
			# plt.clf()
			# plt.imshow(modi, interpolation='nearest', origin='lower')
			# plt.savefig('badstep-%06i.png' % gali)
			# print 'Attempted parameter changes:'
			# for x,nm in zip(X, tractor.getParamNames()):
			#	print '	 ', nm, '  ', x
			break

	# Find bright sources and unfreeze them.
	tractor.catalog.freezeAllParams()
	for i,src in enumerate(tractor.catalog):
		if src.getBrightness().i < 19.:
			tractor.catalog.thawParam(i)

	print 'Fitting bright sources:'
	for nm in tractor.getParamNames():
		print '	 ', nm

	while True:
		dlnp,X,alpha = tractor.optimize(damp=1e-3)
		print 'dlnp', dlnp
		print 'alpha', alpha
		if dlnp < 1:
			break

	print 'Fitting the key galaxy:'
	tractor.catalog.freezeAllBut(ii)
	while True:
		dlnp,X,alpha = tractor.optimize(damp=1e-3)
		print 'dlnp', dlnp
		print 'alpha', alpha
		if dlnp < 1:
			break

	tractor.catalog.thawAllParams()

	p1 = tractor.getParams()
	mod1 = tractor.getModelImage(0)
	chi1 = tractor.getChiImage(0)
	lnp1 = tractor.getLogProb()
	gal1 = gal.copy()

	set_table_from_galaxy(ti, gal1, 'my_')

	mod2 = chi2 = gal2 = None
	dlnp2 = None
		
	if True:
		# Try making model-switching changes to the galaxy...
		tractor.catalog.freezeAllBut(ii)
		#print 'Catalog length (with all but one frozen):', len(tractor.catalog)
		gal = tractor.catalog[ii]
		print 'Galaxy', gal

		if isinstance(gal, DevGalaxy) or isinstance(gal, ExpGalaxy):
			print 'Single-component.  Try Composite...'
			m = gal.brightness
			# Give the new component 1% of the flux...
			m1 = m + 0.01
			m2 = m + 5.
			print 'Mag 1', m1
			print 'Mag 2', m2

			s1 = gal.shape.copy()
			s2 = gal.shape.copy()
			print 'Galaxy shape 1', s1
			print 'Galaxy shape 2', s2

			if isinstance(gal, DevGalaxy):
				comp = CompositeGalaxy(gal.pos, m2, gal.shape.copy(),
									   m1, gal.shape.copy())
			else:
				comp = CompositeGalaxy(gal.pos, m1, gal.shape.copy(),
									   m2, gal.shape.copy())

			tractor.catalog[ii] = comp

			print 'Trying composite', comp

			lnp2 = tractor.getLogProb()
			print 'Single->comp Initial dlnp:', lnp2 - lnp1

			print 'Fitting:'
			for nm in tractor.getParamNames():
				print '	 ', nm

			while True:
				dlnp,X,alpha = tractor.optimize(damp=1e-3)
				print 'Single->comp'
				print 'dlnp', dlnp
				print 'alpha', alpha
				if dlnp < 0.1:
					break

			lnp2 = tractor.getLogProb()
			print 'Single->comp Final dlnp:', lnp2 - lnp1

			tractor.catalog.thawAllParams()

			p2 = tractor.getParams()
			mod2 = tractor.getModelImage(0)
			chi2 = tractor.getChiImage(0)
			lnp2 = tractor.getLogProb()
			gal2 = comp.copy()

			print 'tractor.catalog[ii]:', tractor.catalog[ii]
			print 'comp:', comp.copy()

			print 'Reverting'
			tractor.catalog[ii] = gal


		elif isinstance(gal, CompositeGalaxy):
			print 'Composite.  Flux ratio:'
			photocal = im.getPhotoCal()
			ce = photocal.brightnessToCounts(gal.brightnessExp)
			cd = photocal.brightnessToCounts(gal.brightnessDev)
			print ce / (ce + cd), 'exp'

			frac = ce / (ce + cd)

			#if frac < 0.1:
			if frac < 0.5:
				print 'Trying pure Dev'
				newgal = DevGalaxy(gal.pos, gal.getBrightness(), gal.shapeDev)
			#elif frac > 0.9:
			elif frac >= 0.5:
				print 'Trying pure Exp'
				newgal = ExpGalaxy(gal.pos, gal.getBrightness(), gal.shapeExp)
			else:
				newgal = None
			if newgal is not None:
				print newgal
				tractor.catalog[ii] = newgal
				print tractor.catalog[ii]
				lnp2 = tractor.getLogProb()
				print 'Comp->single: Initial dlnp:', lnp2 - lnp1

				print 'Fitting:'
				for nm in tractor.getParamNames():
					print '	 ', nm

				while True:
					dlnp,X,alpha = tractor.optimize(damp=1e-3)
					print 'comp->single'
					print 'dlnp', dlnp
					print 'alpha', alpha
					if dlnp < 0.1:
						break

				lnp2 = tractor.getLogProb()
				print 'comp->single Final dlnp:', lnp2 - lnp1

			tractor.catalog.thawAllParams()
			p2 = tractor.getParams()
			mod2 = tractor.getModelImage(0)
			chi2 = tractor.getChiImage(0)
			lnp2 = tractor.getLogProb()
			#gal2 = tractor.catalog[ii].copy()
			gal2 = newgal.copy()

			print 'tractor.catalog[ii]:', tractor.catalog[ii]
			print 'newgal:', newgal.copy()

			print 'Reverting'
			tractor.catalog[ii] = gal

		else:
			print 'Hmmm?  Unknown source type', gal

	if gal2 is not None:
		set_table_from_galaxy(ti, gal2, 'sw_')
		ti.sw_dlnp = lnp2 - lnp1

	R,C = 3,3
	plt.figure(figsize=(8,8))
	plt.clf()
	plt.suptitle(im.name)
	plt.subplot(R,C,1)

	plt.imshow(im.getImage(), **ima)
	plt.gray()

	plt.subplot(R,C,2)

	plt.imshow(mod0, **ima)
	plt.gray()

	tractor.setParams(p0)
	plot_ellipses(im, tractor.catalog)
	tractor.setParams(p1)

	plt.subplot(R,C,3)

	plt.imshow(chi0, **imchi)
	plt.gray()

	plt.subplot(R,C,5)

	plt.imshow(mod1, **ima)
	plt.gray()
	plot_ellipses(im, tractor.catalog)

	plt.subplot(R,C,6)

	plt.imshow(chi1, **imchi)
	plt.gray()

	if mod2 is not None:
		plt.subplot(R,C,8)
		plt.imshow(mod2, **ima)
		plt.gray()

		tractor.catalog[ii] = gal2
		tractor.setParams(p2)
		plot_ellipses(im, tractor.catalog)
		tractor.catalog[ii] = gal1
		tractor.setParams(p1)

		plt.subplot(R,C,9)

		plt.imshow(chi2, **imchi)
		plt.gray()

	plt.savefig('trgal-%06i.png' % gali)

	return ti
	

def set_table_from_galaxy(ti, gal, prefix):
	ti.set(prefix + 'ra',  gal.pos.ra)
	ti.set(prefix + 'dec', gal.pos.dec)
	if isinstance(gal, ExpGalaxy):
		ti.set(prefix + 'type', 'E')
		for c in ['devrad_i', 'devab_i', 'devphi_i', 'devmag_i']:
			ti.set(c, np.nan)
		ti.set(prefix + 'exprad_i', gal.shape.re)
		ti.set(prefix + 'expphi_i', gal.shape.phi)
		ti.set(prefix + 'expab_i', gal.shape.ab)
		ti.set(prefix + 'expmag_i', gal.brightness.i)
	elif isinstance(gal, DevGalaxy):
		ti.set(prefix + 'type', 'D')
		for c in ['exprad_i', 'expab_i', 'expphi_i', 'expmag_i']:
			ti.set(prefix + c, np.nan)
		ti.set(prefix + 'devrad_i', gal.shape.re)
		ti.set(prefix + 'devphi_i', gal.shape.phi)
		ti.set(prefix + 'devab_i', gal.shape.ab)
		ti.set(prefix + 'devmag_i', gal.brightness.i)
	elif isinstance(gal, CompositeGalaxy):
		ti.set(prefix + 'type', 'C')
		ti.set(prefix + 'exprad_i', gal.shapeExp.re)
		ti.set(prefix + 'expphi_i', gal.shapeExp.phi)
		ti.set(prefix + 'expab_i', gal.shapeExp.ab)
		ti.set(prefix + 'expmag_i', gal.brightnessExp.i)
		ti.set(prefix + 'devrad_i', gal.shapeDev.re)
		ti.set(prefix + 'devphi_i', gal.shapeDev.phi)
		ti.set(prefix + 'devab_i', gal.shapeDev.ab)
		ti.set(prefix + 'devmag_i', gal.brightnessDev.i)


def plot_ellipses(im, cat):
	wcs = im.getWcs()
	x0,y0 = wcs.x0, wcs.y0
	#xc,yc = wcs.positionToPixel(RaDecPos(ti.ra, ti.dec))
	H,W = im.getImage().shape
	xc,yc = W/2., H/2.
	cd = wcs.cdAtPixel(xc,yc)
	ax = plt.axis()
	for src in cat:
		pos = src.getPosition()
		x,y = wcs.positionToPixel(pos)
		x += x0
		y += y0
		gals = []
		if isinstance(src, PointSource):
			plt.plot(x, y, 'g+')
			continue
		elif isinstance(src, ExpGalaxy):
			gals.append((True, src.shape, 'r', {}))
		elif isinstance(src, DevGalaxy):
			gals.append((False, src.shape, 'b', {}))
		elif isinstance(src, CompositeGalaxy):
			gals.append((True,	src.shapeExp, 'm', dict(lw=2, alpha=0.5)))
			gals.append((False, src.shapeDev, 'c', {}))
		else:
			print 'Unknown source type:', src
			continue

		theta = np.linspace(0, 2*np.pi, 90)
		ux,uy = np.cos(theta), np.sin(theta)
		u = np.vstack((ux,uy)).T

		for isexp,shape,c,kwa in gals:
			T = np.linalg.inv(shape.getTensor(cd))
			#print 'T shape', T.shape
			#print 'u shape', u.shape
			dx,dy = np.dot(T, u.T)
			#if isexp:
			#	c = 'm'
			#else:
			#	c = 'c'
			#print 'x,y', x, y
			#print 'dx range', dx.min(), dx.max()
			#print 'dy range', dy.min(), dy.max()
			plt.plot(x + dx, y + dy, '-', color=c, **kwa)
			plt.plot([x], [y], '+', color=c)
			
	plt.axis(ax)
	


def plot_ipes():
	T = fits_table('dr9fields.fits')

	def get_rrdd(T):
		dx,dy = 2048./2, 1489./2
		RR,DD = [],[]
		for sx,sy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
			r = T.ra  + sx*dx*T.cd11 + sy*dy*T.cd12
			d = T.dec + sx*dx*T.cd21 + sy*dy*T.cd22
			RR.append(r)
			DD.append(d)
		RR.append(RR[0])
		DD.append(DD[0])
		RR = np.vstack(RR)
		DD = np.vstack(DD)
		#print RR.shape
		return RR,DD

	
	if False:
		RR,DD = get_rrdd(T)
	
		W,H = 1000,500
		p = Plotstuff(outformat=PLOTSTUFF_FORMAT_PNG, size=(W,H))
		p.wcs = anwcs_create_allsky_hammer_aitoff(0., 0., W, H)
		p.color = 'verydarkblue'
		p.plot('fill')
		p.color = 'white'
		p.alpha = 0.25
		p.apply_settings()
		for rr,dd in zip(RR.T, DD.T):
			if not (np.all((rr - 180) > 0) or np.all((rr - 180) < 0)):
				print 'Skipping boundary-spanning', rr,dd
				continue
			p.move_to_radec(rr[0], dd[0])
			for r,d in zip(rr[1:],dd[1:]):
				p.line_to_radec(r,d)
			p.fill()
		p.color = 'gray'
		p.plot_grid(30, 30, 30, 30)
		p.write('rd1.png')

	Ti = T[(T.ra > 190) * (T.ra < 200) * (T.dec > 23) * (T.dec < 24)]
	print 'Runs', np.unique(Ti.run)
	print 'Camcols', np.unique(Ti.camcol)
	print 'Stripes', np.unique(Ti.stripe)
	print 'Strips', np.unique(Ti.strip)

	Ti = T[(T.ra > 194) * (T.ra < 195) * (T.dec > 23.3) * (T.dec < 23.7)]
	print 'Runs', np.unique(Ti.run)
	print 'Camcols', np.unique(Ti.camcol)
	print 'Stripes', np.unique(Ti.stripe)
	print 'Strips', np.unique(Ti.strip)

	
	#if True:
	for r,d,w,g,fn,lab in [#(230,30,120,10,'rd2.png', False),
		#(255,15,30,5,'rd3.png', False),
		#(250,20,10,1,'rd4.png', False),
		#(250,18,3,1, 'rd5.png', False),
		#(250,17.8,1,0.5, 'rd6.png', False),
		(195, 25, 5, 1, 'rd7.png', False),
		(194.5, 23.5, 1, 0.1, 'rd8.png', True),
		(195, 25, 20, 1, 'rd9.png', False),
		]:
	
		W,H = 1000,1000
		p = Plotstuff(outformat=PLOTSTUFF_FORMAT_PNG, size=(W,H),
					  rdw=(r,d,w))
		rmn,rmx,dmn,dmx = anwcs_get_radec_bounds(p.wcs, 100)
		print 'Bounds', rmn,rmx,dmn,dmx
	
		Ti = T[((T.ramin < rmx) * (T.ramax > rmn) *
				(T.decmin < dmx) * (T.decmax > dmn))]
		RR,DD = get_rrdd(Ti)
	
		p.color = 'verydarkblue'
		p.plot('fill')
		p.color = 'white'
		p.alpha = 0.25
		#p.op = CAIRO_OPERATOR_ADD
		p.apply_settings()
		for rr,dd in zip(RR.T, DD.T):
			if not (np.all((rr - 180) > 0) or np.all((rr - 180) < 0)):
				#print 'Skipping boundary-spanning', rr,dd
				continue
			p.move_to_radec(rr[0], dd[0])
			for r,d in zip(rr[1:],dd[1:]):
				p.line_to_radec(r,d)
			p.fill_preserve()
			p.stroke()
		p.color = 'gray'
		p.plot_grid(g,g,g,g)

		if lab:
			p.color = 'white'
			p.apply_settings()
			for i in range(len(Ti)):
				p.text_radec(Ti.ra[i], Ti.dec[i],
							 '%i/%i/%i' % (Ti.run[i], Ti.camcol[i], Ti.field[i]))

		p.write(fn)
	
   
if __name__ == '__main__':
	main()
	
