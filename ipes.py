
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

from tractor.utils import PlotSequence

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

	#p1 = Patch(10, 15, np.zeros((100,200)))
	#p2 = Patch(20, 20, np.zeros((20,25)))
	#p1.hasNonzeroOverlapWith(p2)
	#p2.hasNonzeroOverlapWith(p1)
	#sys.exit(0)

	#plot_ipes()
	#refit_galaxies_1()
	#ipe_errors()

	#refit_galaxies_2()
	my_ipe_errors()


def my_ipe_errors():
	ps = PlotSequence('myipe')
	T = fits_table('my-ipes.fits')
	print 'Got', len(T), 'galaxies'

	# Photo errors are in arcsec.
	T.raerr /= 3600.
	T.decerr /= 3600.

	# The galaxies here are in consecutive pairs.
	T1 = T[::2]
	T2 = T[1::2]
	print len(T1), 'pairs'

	plt.clf()
	plt.plot(T1.raerr * 3600., np.abs((T1.ra - T2.ra) / T1.raerr), 'r.')
	plt.plot(T2.raerr * 3600., np.abs((T1.ra - T2.ra) / T2.raerr), 'm.')
	plt.plot(T1.my_ra_err * 3600., np.abs(T1.my_ra - T2.my_ra) / T1.my_ra_err, 'b.')
	plt.plot(T2.my_ra_err * 3600., np.abs(T1.my_ra - T2.my_ra) / T2.my_ra_err, 'c.')
	plt.gca().set_xscale('log')
	plt.gca().set_yscale('log')
	ps.savefig()

	plt.clf()
	#mn = np.min(T1.raerr.min(), T1.my_ra_err.min())*3600.
	#mx = np.max(T1.raerr.max(), T1.my_ra_err.max())*3600.
	plt.subplot(2,1,1)
	plt.loglog(T1.raerr * 3600., np.abs((T1.ra - T2.ra) / T1.raerr), 'r.', alpha=0.5)
	plt.plot(T2.raerr * 3600., np.abs((T1.ra - T2.ra) / T2.raerr), 'r.', alpha=0.5)
	ax1 = plt.axis()

	x = np.append(T1.raerr, T2.raerr)*3600.
	y = np.append(np.abs(T1.ra - T2.ra) / T1.raerr, np.abs(T1.ra - T2.ra) / T2.raerr)
	x1,y1 = x,y
	I = np.argsort(x)
	S = np.linspace(0, len(x), 11).astype(int)
	xm,ym,ys = [],[],[]
	for ilo,ihi in zip(S[:-1], S[1:]):
		J = I[ilo:ihi]
		xm.append(np.mean(x[J]))
		ym.append(np.median(y[J]))
		ys.append(np.abs(np.percentile(y[J], 75) - np.percentile(y[J], 25)))
	p,c,b = plt.errorbar(xm, ym, yerr=ys, color='k', fmt='o')
	for pp in [p]+list(c)+list(b):
		pp.set_zorder(20)
	#for ci in c: ci.set_zorder(20)
	#for bi in b: bi.set_zorder(20)
	xm1,ym1,ys1 = xm,ym,ys

	#plt.xlim(mn,mx)
	plt.subplot(2,1,2)
	plt.loglog(T1.my_ra_err * 3600., np.abs(T1.my_ra - T2.my_ra) / T1.my_ra_err, 'b.', alpha=0.5)
	plt.plot(T2.my_ra_err * 3600., np.abs(T1.my_ra - T2.my_ra) / T2.my_ra_err, 'b.', alpha=0.5)

	x = np.append(T1.my_ra_err, T2.my_ra_err)*3600.
	y = np.append(np.abs(T1.my_ra - T2.my_ra) / T1.my_ra_err, np.abs(T1.my_ra - T2.my_ra) / T2.my_ra_err)
	x2,y2 = x,y
	I = np.argsort(x)
	S = np.linspace(0, len(x), 11).astype(int)
	xm,ym,ys = [],[],[]
	for ilo,ihi in zip(S[:-1], S[1:]):
		J = I[ilo:ihi]
		xm.append(np.mean(x[J]))
		ym.append(np.median(y[J]))
		ys.append(np.abs(np.percentile(y[J], 75) - np.percentile(y[J], 25)))
	p,c,b = plt.errorbar(xm, ym, yerr=ys, color='k', fmt='o')#, zorder=20, barsabove=True)
	for pp in [p]+list(c)+list(b):
		pp.set_zorder(20)

	ax2 = plt.axis()
	#print 'ax1', ax1
	#print 'ax2', ax2
	ax = [min(ax1[0],ax2[0]), max(ax1[1],ax2[1]), min(ax1[2],ax2[2]), max(ax1[3],ax2[3])]
	#print 'ax', ax
	plt.axis(ax)
	plt.gca().set_yscale('symlog', linthreshy=1e-3)
	plt.axhline(1, color='k', alpha=0.5, lw=2)
	#plt.gca().set_xscale('log')
	#plt.gca().set_yscale('log')
	plt.subplot(2,1,1)
	plt.axis(ax)
	plt.gca().set_yscale('symlog', linthreshy=1e-3)
	plt.axhline(1, color='k', alpha=0.5, lw=2)
	#plt.xlim(mn,mx)
	#plt.gca().set_xscale('log')
	#plt.gca().set_yscale('log')
	ps.savefig()

	for sp in [1,2]:
		plt.subplot(2,1,sp)
		#plt.axis([1e-2, 1, 1e-3, 10])
		plt.axis([1e-2, 1, 1e-2, 1e2])
		plt.ylabel('Inter-ipe difference / error')
	plt.xlabel('RA error (arcsec)')
	ps.savefig()

	# plt.clf()
	# plt.loglog(x1, y1, 'r.')
	# plt.loglog(x2, y2, 'r.')
	# plt.axis([1e-3, 1e1, 1e-5, 1e3]) 
	# ps.savefig()

	plt.clf()
	plt.hist(y1, 100, range=(0, 6), histtype='step', color='r')
	plt.hist(y2, 100, range=(0, 6), histtype='step', color='b')
	#plt.xlabel('N sigma of inter-ipe difference')
	plt.xlabel('Inter-ipe difference / error')
	plt.title('RA')
	ps.savefig()


def ipe_errors():
	#T = fits_table('ipe1_dstn.fit')
	#T.cut((T.ra < 249.85) * (T.dec < 17.65))
	#print 'Cut to', len(T)

	ps = PlotSequence('ipe')

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
	ps.savefig()

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

	#plt.clf()
	#loghist(dra, ddec, 100, range=((-1,1),(-1,1)))
	#ps.savefig()

	print 'Range of Decs:', T1.dec.min(), T1.dec.max(), T2.dec.min(), T2.dec.max()
	ras = np.cos(np.deg2rad(np.append(T1.dec, T2.dec)))
	print 'Range of RAscales:', ras.min(), ras.max()

	rascale = np.mean(ras)

	X1 = np.vstack((T1.ra * rascale, T1.dec)).T
	X2 = np.vstack((T2.ra * rascale, T2.dec)).T
	inds,d = nearest(X1, X2, R)
	J = np.flatnonzero(inds > -1)
	I = inds[J]
	print 'Nearest-neighbour matches:', len(I)
	d = np.sqrt(d[J])
	print 'd', d.shape
	print 'I,J', len(I), len(J)
	print 'd max', np.max(d), 'min', np.min(d)
	print 'R', R
	assert(np.all(d <= R))
	assert(np.all(J >= 0))
	assert(np.all(I >= 0))

	dx = X1[I] - X2[J]
	print 'dx', dx.shape
	dr = np.hypot(dx[:,0], dx[:,1])
	print 'dr', dr.shape
	assert(np.all(dr <= R))

	dra  = (T1.ra [I] - T2.ra [J]) * rascale * 3600.
	ddec = (T1.dec[I] - T2.dec[J]) * 3600.

	plt.clf()
	loghist(dra, ddec, 100, range=((-1,1),(-1,1)))
	ps.savefig()

	M1 = T1[I]
	M2 = T2[J]

	#print 'All T1', T1.rowc.min(), T1.rowc.max(), T1.colc.min(), T1.colc.max()
	#print 'Matched T1', T1.rowc[I].min(), T1.rowc[I].max(), T1.colc[I].min(), T1.colc[I].max()
	#print 'All T2', T2.rowc.min(), T2.rowc.max(), T2.colc.min(), T2.colc.max()
	#print 'Matched T2', T2.rowc[J].min(), T2.rowc[J].max(), T2.colc[J].min(), T2.colc[J].max()

	# Errors are in arcsec.
	rerr1 = M1.raerr
	derr1 = M1.decerr
	rerr2 = M2.raerr
	derr2 = M2.decerr

	hi = 6.
	dscale = 1./np.sqrt(2.)
	plt.clf()
	n,b,p = plt.hist(dscale * np.hypot(dra, ddec) / np.hypot(rerr1, derr1), 100,
					 range=(0, hi), histtype='step', color='r')
	plt.hist(dscale * np.hypot(dra, ddec) / np.hypot(rerr2, derr2), 100,
			 range=(0, hi), histtype='step', color='b')
	xx = np.linspace(0, hi, 500)
	from scipy.stats import chi
	yy = chi.pdf(xx, 2)
	plt.plot(xx, yy * len(dra) * (b[1]-b[0]), 'k-')
	plt.xlim(0,hi)
	plt.xlabel('N sigma of RA,Dec repeat observations')
	plt.ylabel('Number of sources')
	ps.savefig()

	#loghist(np.hypot(dra, ddec), np.sqrt(np.hypot(rerr1, derr1) * np.hypot(rerr2, derr2)), 100,
	#		clamp=((0,1),(0,1)))
	loghist(np.hypot(dra, ddec), (np.hypot(rerr1, derr1) + np.hypot(rerr2, derr2)) / 2., 100, range=((0,1),(0,1)), clamp=True)
	plt.xlabel('Inter-ipe difference: RA,Dec (arcsec)')
	plt.ylabel('Photo errors: RA,Dec (arcsec)')
	ps.savefig()

	loghist(np.log10(np.hypot(dra, ddec)), np.log10((np.hypot(rerr1, derr1) + np.hypot(rerr2, derr2)) / 2.),
			100, range=((-3,0),(-3,0)), clamp=True)
	plt.xlabel('Inter-ipe difference: log RA,Dec (arcsec)')
	plt.ylabel('Photo errors: log RA,Dec (arcsec)')
	ps.savefig()

	plt.clf()
	n,b,p = plt.hist(dscale * np.abs(M1.psfmag_r - M2.psfmag_r) / M1.psfmagerr_r, 100, range=(0,hi), histtype='step', color='r')
	plt.xlabel('N sigma of psfmag_r')
	xx = np.linspace(0, hi, 500)
	yy = 2./np.sqrt(2.*np.pi)*np.exp(-0.5 * xx**2)
	print 'yy', sum(yy)
	plt.plot(xx, yy * len(M1) * (b[1]-b[0]), 'k-')
	ps.savefig()

	# Galaxy-star matches
	K1 = (M1.type == 3) * (M2.type == 6)
	K2 = (M1.type == 6) * (M2.type == 3)
	G = merge_tables((M1[K1], M2[K2]))
	S = merge_tables((M2[K1], M1[K2]))
	print 'G types:', np.unique(G.type)
	print 'S types:', np.unique(S.type)
	mhi,mlo = 24,10
	K = ((G.modelmag_r < mhi) * (S.psfmag_r < mhi) *
		 (G.modelmag_r > mlo) * (S.psfmag_r > mlo))
	print 'Star/gal mismatches with good mags:', np.sum(K)

	# gm = G.modelmag_r.copy()
	# gm[np.logical_or(gm > mhi, gm < mlo)] = 25.
	# sm = S.psfmag_r.copy()
	# sm[np.logical_or(sm > mhi, sm < mlo)] = 25.
	# 
	# #loghist(G.modelmag_r[K], S.psfmag_r[K], 100)
	# loghist(gm, sm, 100)

	loghist(G.modelmag_r, S.psfmag_r, clamp=((mlo,mhi),(mlo,mhi)),
			clamp_to=((mlo-1,mhi+1),(mlo-1,mhi+1)))
	ax = plt.axis()
	plt.axhline(mhi, color='b')
	plt.axvline(mhi, color='b')
	plt.plot(*([  [min(ax[0],ax[2]), max(ax[1],ax[3])] ]*2) + ['b-',])
	plt.axis(ax)
	plt.xlabel('Galaxy modelmag_r')
	plt.ylabel('Star psfmag_r')
	plt.title('Star/Galaxy ipe mismatches')
	ps.savefig()

	K = ((G.modelmag_r < mhi) * (G.modelmag_r > mlo))
	plt.clf()
	KK = (G.fracdev_r < 0.5)
	kwargs = dict(bins=100, range=(np.log10(0.01), np.log10(30.)), histtype='step')
	plt.hist(np.log10(G.exprad_r[K * KK]), color='r', **kwargs)
	KK = (G.fracdev_r >= 0.5)
	plt.hist(np.log10(G.devrad_r[K * KK]), color='b', **kwargs)
	plt.xlabel('*rad_r (arcsec)')
	loc,lab = plt.xticks()
	plt.xticks(loc, ['%g' % (10.**x) for x in loc])
	plt.title('Star/Galaxy ipe mismatches')
	ps.savefig()



	# Pairs where both are galaxies
	K = ((M1.type == 3) * (M2.type == 3))
	G1 = M1[K]
	G2 = M2[K]
	print len(G1), 'pairs where both are galaxies'
	
	#for 
	plt.clf()
	c,cerr = 'modelmag_r', 'modelmagerr_r'
	n,b,p = plt.hist(dscale * np.abs(G1.get(c) - G2.get(c)) / G1.get(cerr), 100, range=(0,hi),
					 histtype='step', color='r')
	plt.xlabel('N sigma of ' + c)
	yy = np.exp(-0.5 * b**2)
	yy *= sum(n) / np.sum(yy)
	plt.plot(b, yy, 'k-')
	ps.savefig()

	loghist(np.abs(G1.get(c) - G2.get(c)), G1.get(cerr), 100, range=((0,2),(0,2)), clamp=True)
	plt.xlabel('Inter-ipe difference: ' + c)
	plt.ylabel('Photo error: ' + cerr)
	ps.savefig()

	loghist(np.log10(np.abs(G1.get(c) - G2.get(c))), np.log10(G1.get(cerr)), 100, range=((-3,1),(-3,1)), clamp=True)
	plt.xlabel('Inter-ipe difference: ' + c)
	plt.ylabel('Photo error: ' + cerr)
	ps.savefig()

	plt.clf()
	loghist(G1.fracdev_r, G2.fracdev_r, 100, range=((0,1),(0,1)), clamp=True)
	plt.xlabel('G1 fracdev_r')
	plt.ylabel('G2 fracdev_r')
	ps.savefig()

	dscale = 1.

	I = (G1.fracdev_r < 0.5) * (G2.fracdev_r < 0.5)
	print sum(I), 'of', len(G1), 'both have fracdev_r < 0.5'
	E1 = G1[I]
	E2 = G2[I]

	I = (G1.fracdev_r >= 0.5) * (G2.fracdev_r >= 0.5)
	print sum(I), 'of', len(G1), 'both have fracdev_r >= 0.5'
	D1 = G1[I]
	D2 = G2[I]

	for t,H1,H2 in [('exp',E1,E2),('dev',D1,D2)]:

		c,cerr = '%smag_r'%t, '%smagerr_r'%t
		dval = np.abs(H1.get(c) - H2.get(c))
		derr = H1.get(cerr)
		rng = ((0,1),(0,1))

		loghist(dval, derr, 100, range=rng, clamp=True)
		plt.xlabel('Inter-ipe difference: ' + c)
		plt.ylabel('Photo error: ' + cerr)
		ps.savefig()

		loghist(np.log10(dval), np.log10(derr), 100, range=((-3,0),(-3,0)), clamp=True)
		plt.xlabel('Inter-ipe difference: log ' + c)
		plt.ylabel('Photo error: log ' + cerr)
		ps.savefig()

		c,cerr = '%sab_r'%t,  '%saberr_r'%t
		dval = np.abs(H1.get(c) - H2.get(c))
		derr = H1.get(cerr)
		rng = ((0,1),(0,1))

		loghist(dval, derr, 100, range=rng, clamp=True)
		plt.xlabel('Inter-ipe difference: ' + c)
		plt.ylabel('Photo error: ' + cerr)
		ps.savefig()

		loghist(np.log10(dval), np.log10(derr), 100, range=((-3,0),(-3,0)), clamp=True)
		plt.xlabel('Inter-ipe difference: log ' + c)
		plt.ylabel('Photo error: log ' + cerr)
		ps.savefig()

		c,cerr = '%srad_r'%t, '%sraderr_r'%t
		dval = np.abs(H1.get(c) - H2.get(c))
		derr = H1.get(cerr)
		rng = ((0,30),(0,30))

		loghist(dval, derr, 100, range=rng, clamp=True)
		plt.xlabel('Inter-ipe difference: ' + c)
		plt.ylabel('Photo error: ' + cerr)
		ps.savefig()

		loghist(np.log10(dval), np.log10(derr), 100, range=((-2,2),(-2,2)), clamp=True)
		plt.xlabel('Inter-ipe difference: log ' + c)
		plt.ylabel('Photo error: log ' + cerr)
		ps.savefig()


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



def refit_galaxies_2():
	import optparse
	parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')
	parser.add_option('--threads', dest='threads', type=int, default=1,
					  help='use multiprocessing')
	opt,args = parser.parse_args()
	mp = multiproc(nthreads=opt.threads)

	band = 'r'

	T = fits_table('ipe3_dstn_2.fit')
	print len(T), 'objects'

	binned1 = 268435456
	blended = 8
	bad = 216207968633487360
	bright = 2

	T.cut((T.flags & (bad | blended)) == 0)
	print 'Removing bad or blended:', len(T)
	T.cut((T.flags & bright) == 0)
	print 'Removing bright:', len(T)
	T.cut((T.flags & binned1) > 0)
	print 'Binned1:', len(T)

	T1 = T[T.run == 5183]
	T2 = T[T.run == 5224]

	rascale = np.mean(np.cos(np.deg2rad(np.append(T1.dec, T2.dec))))
	X1 = np.vstack((T1.ra * rascale, T1.dec)).T
	X2 = np.vstack((T2.ra * rascale, T2.dec)).T
	R = 1. / 3600.
	inds,d = nearest(X1, X2, R)
	J = np.flatnonzero(inds > -1)
	I = inds[J]
	print len(J), 'matches'
	print len(np.unique(I)), 'unique objs in target'

	from collections import Counter
	tally = Counter(I)
	# subtract 1
	for k in tally: tally[k] -= 1
	# remove zero and negative entries
	tally += Counter()
	# Now "tally" just contains keys (from I) with >= 2 counts
	print 'multiple matches:', len(tally), ':', tally.keys()
	multi = set(tally.keys())
	K = np.array([k for k,ii in enumerate(I) if not ii in multi])
	I = I[K]
	J = J[K]
	print 'Kept', len(I), 'non-multi-matched pairs'
	print len(np.unique(J)), 'unique J'
	print len(np.unique(I)), 'unique I'
	M1 = T1[I]
	M2 = T2[J]

	# select both-galaxy subset.
	K = ((M1.type == 3) * (M2.type == 3))
	M1 = M1[K]
	M2 = M2[K]
	print 'Both galaxies:', len(M1)

	# sort by mag, moving -9999 to the end.
	mag = M1.get('modelmag_' + band).copy()
	mag[mag < 0] = 9999.
	# avoid bright guys too...
	mag[mag < 16] = 9000.
	I = np.argsort(mag)
	M1 = M1[I]
	M2 = M2[I]

	# interleave them
	N = len(M1)
	for c in M1.get_columns():
		X1 = M1.get(c)
		X2 = M2.get(c)
		#print 'column', c, 'shape', X1.shape
		## can't handle arrays here
		assert(len(X1.shape) == 1)
		XX = np.zeros(N * 2, dtype=X1.dtype)
		XX[0::2] = X1
		XX[1::2] = X2
		M1.set(c, XX)
	MM = M1
	# We have to set the length manually.
	MM._length = 2*N
	print 'Length of MM:', len(MM)
	del M1
	del M2

	#####
	#MM = MM[:2]
		
	refit_galaxies(MM, intermediate_fn='my-ipes-%06i.fits', mp=mp,
				   modswitch=False, errors=True, band=band)
	MM.writeto('my-ipes.fits')

def refit_galaxies_1():
	import optparse
	parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')
	parser.add_option('--threads', dest='threads', type=int, default=None,
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
	Ti = fits_table('exp5b_dstn.fit')

	rlo,rhi = 4.1, 4.4
	Ti = T[(T.exprad_i > rlo) * (T.exprad_i < rhi)]
	Ti = Ti[Ti.expmag_i < 19]
	I = np.argsort(Ti.expmag_i)

	###
	#I = I[:4]
	#I = I[7:8]

	Ti = Ti[I]

	print 'Cut to', len(Ti), 'galaxies in radius and mag cuts'

	refit_galaxies(Ti, intermediate_fn='mye4-%06i.fits', mp=mp,
				   modswitch=True)
	Ti.writeto('mye4.fits')

def refit_galaxies(T, band='i', S=100,
				   intermediate_fn='refit-%06i.fits', mp=None,
				   modswitch=False, errors=False):
	if mp is None:
		mp = multiproc()
	sdss = DR9(basedir='paper0-data-dr9')
	print 'basedir', sdss.basedir
	print 'dasurl', sdss.dasurl

	N = len(T)
	ps = [('my_',''), ('init_',''),]
	if modswitch:
		ps.append(('sw_',''))
	if errors:
		ps.append(('my_','_err'))

	for prefix,suffix in ps:
		for c in ['exprad_', 'expab_', 'expphi_', 'expmag_',
				  'devrad_', 'devab_', 'devphi_', 'devmag_']:
			T.set(prefix + c + band + suffix, np.zeros(N, dtype=np.float32))

		for c in ['ra', 'dec']:
			if len(suffix):
				dt = np.float32
			else:
				dt = np.float64
			T.set(prefix + c + suffix, np.zeros(N, dtype=dt))

		# assume suffix implies _error; omit prefix_type_err field
		if len(suffix):
			continue
		T.set(prefix + 'type', np.chararray(len(T), 1))

	if modswitch:
		T.set('sw_dlnp', np.zeros(len(T), np.float32))

	args = []

	for gali in range(len(T)):
		ti = T[gali]
		args.append((ti, band, S, sdss, gali, modswitch, errors))

	# Run in blocks.
	tinew = []
	B = 0
	while len(args):

		N = 100
		#N = 4
		B += N
		# Pop N args off the front of the list
		thisargs = args[:N]
		args = args[N:]

		# Run on those args
		thisres = mp.map(_refit_gal, thisargs)

		#tinew.extend(thisres)
		#print 'tinew:', tinew

		for resi,argi in zip(thisres, thisargs):
			###
			gali = argi[4]
			###
			if resi is None:
				print 'Result', gali, 'is None'
				continue
			print 'Saving result', gali
			T[gali] = resi

		#for gali in range(min(len(T), len(tinew))):
		#	tin = tinew[gali]
		#	if tin is None:
		#		print 'Skipping', gali
		#		continue
		#	T[gali] = tin
		#Ti.about()

		if intermediate_fn:
			T.writeto(intermediate_fn % B)

def _refit_gal(*args):
	try:
		return _real_refit_gal(*args)
	except:
		import traceback
		traceback.print_exc()
		return None

def _real_refit_gal((ti, band, S, sdss, gali,
					 modswitch, errors)):
	im,info = get_tractor_image_dr9(ti.run, ti.camcol, ti.field, band,
									roiradecsize=(ti.ra, ti.dec, S),
									sdss=sdss)
	roi = info['roi']
	cat = get_tractor_sources_dr9(ti.run, ti.camcol, ti.field, band,
								  sdss=sdss, roi=roi, bands=[band])

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
	set_table_from_galaxy(ti, gal0, 'init_', band=band)

	print 'Fitting the target galaxy:'
	tractor.catalog.freezeAllBut(ii)
	while True:
		dlnp,X,alpha = tractor.optimize(damp=1e-3)
		print 'dlnp', dlnp
		print 'alpha', alpha
		if dlnp < 1:
			break

	print 'Fitting all sources that overlap the target galaxy:'
	thaw = []
	galpatch = tractor.getModelPatch(im, gal)
	galpatch.trimToNonZero()
	for j,src in enumerate(tractor.catalog):
		if src is gal:
			continue
		patch = tractor.getModelPatch(im, src)
		if patch is None:
			continue
		patch.trimToNonZero()
		if galpatch.hasNonzeroOverlapWith(patch):
			thaw.append(j)

	tractor.catalog.freezeAllBut(*thaw)

	print len(tractor.getCatalog()), 'sources in the region'
	print len(thaw), 'overlap the target galaxy'

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
	#tractor.catalog.freezeAllParams()
	#for i,src in enumerate(tractor.catalog):
	#if src.getBrightness().i < 19.:
	#		tractor.catalog.thawParam(i)
	#print 'Fitting bright sources:'
	#for nm in tractor.getParamNames():
	#	print '	 ', nm
	#while True:
	#	dlnp,X,alpha = tractor.optimize(damp=1e-3)
	#	print 'dlnp', dlnp
	#	print 'alpha', alpha
	#	if dlnp < 1:
	#		break

	if errors:
		print 'Computing errors on the target galaxy:'
		tractor.catalog.freezeAllBut(ii)
		sigs = tractor.computeParameterErrors()
		nms = gal.getParamNames()

	tractor.catalog.thawAllParams()
	p1 = tractor.getParams()
	mod1 = tractor.getModelImage(0)
	chi1 = tractor.getChiImage(0)
	lnp1 = tractor.getLogProb()
	gal1 = gal.copy()

	err = None
	if errors:
		err = dict(zip(nms, sigs))
	set_table_from_galaxy(ti, gal1, 'my_', errors=err, band=band)

	mod2 = chi2 = gal2 = None
	dlnp2 = None
	if modswitch:
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
			#gal2 = tractor.catalog[0].copy()
			gal2 = newgal.copy()

			print 'tractor.catalog[ii]:', tractor.catalog[ii]
			print 'newgal:', newgal.copy()

			print 'Reverting'
			tractor.catalog[ii] = gal

		else:
			print 'Hmmm?  Unknown source type', gal

	if gal2 is not None:
		set_table_from_galaxy(ti, gal2, 'sw_', band=band)
		ti.sw_dlnp = lnp2 - lnp1

	if mod2 is not None:
		R,C = 3,3
		plt.figure(figsize=(8,8))
	else:
		R,C = 2,3
		plt.figure(figsize=(8,6))

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
	

def set_table_from_galaxy(ti, gal, prefix, band='i', errors=None):
	fields = []
	fields.append(('ra', 'pos.ra'))
	fields.append(('dec', 'pos.dec'))
	if isinstance(gal, ExpGalaxy):
		ti.set(prefix + 'type', 'E')
		for c in [f + band for f in ['devrad_', 'devab_', 'devphi_', 'devmag_']]:
			ti.set(prefix + c, np.nan)
		fields.append(('exprad_' + band, 'shape.re'))
		fields.append(('expphi_' + band, 'shape.phi'))
		fields.append(('expab_'  + band, 'shape.ab'))
		fields.append(('expmag_' + band, 'brightness.' + band))
	elif isinstance(gal, DevGalaxy):
		ti.set(prefix + 'type', 'D')
		for c in [f + band for f in ['exprad_', 'expab_', 'expphi_', 'expmag_']]:
			ti.set(prefix + c, np.nan)
		fields.append(('devrad_' + band, 'shape.re'))
		fields.append(('devphi_' + band, 'shape.phi'))
		fields.append(('devab_'  + band, 'shape.ab'))
		fields.append(('devmag_' + band, 'brightness.' + band))
	elif isinstance(gal, CompositeGalaxy):
		ti.set(prefix + 'type', 'C')
		fields.append(('exprad_' + band, 'shapeExp.re'))
		fields.append(('expphi_' + band, 'shapeExp.phi'))
		fields.append(('expab_'  + band, 'shapeExp.ab'))
		fields.append(('expmag_' + band, 'brightnessExp.' + band))
		fields.append(('devrad_' + band, 'shapeDev.re'))
		fields.append(('devphi_' + band, 'shapeDev.phi'))
		fields.append(('devab_'  + band, 'shapeDev.ab'))
		fields.append(('devmag_' + band, 'brightnessDev.' + band))

	if errors:
		print 'Errors:', errors

	for tnm,gnm in fields:
		val = gal
		for t in gnm.split('.'):
			val = getattr(val, t)
		#val = getattr(gal, gnm)
		print 'galaxy', gnm, '->', val
		ti.set(prefix + tnm, val)
		if errors:
			err = errors.get(gnm, None)
			print 'Error', err
			if err is not None:
				ti.set(prefix + tnm + '_err', err)


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
	
