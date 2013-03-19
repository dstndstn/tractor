'''
Date: Wed, 23 Jan 2013 00:38:33 -0300
From: Jacqueline Faherty <jfaherty17@gmail.com>

NAME RA DEC PMra (mas/yr) PMdec (mas/yr) Pi (mas-expected)
SDSS0330-0025 52.646659 -0.42659916 414+/-30 -355+/-51 40+/-10
SDSS2057-0050 314.48301 -0.83521203 -37.2+/-20 -30+/-27 33+/-10
'''

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.sdss import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import Tan
from astrometry.libkd.spherematch import *

from tractor import *
from tractor.sdss import *

import emcee

def test_moving_source():
	times = [TAITime(4412918525.29), TAITime(4417916046.19),
			 TAITime(4418002191.5), TAITime(4507780010.53),
			 TAITime(4512184688.24), TAITime(4513044948.53),
			 TAITime(4517263915.33), TAITime(4517610657.07),
			 TAITime(4540534186.72), TAITime(4543374645.95),
			 TAITime(4571631988.29), TAITime(4573701132.96),
			 TAITime(4602564263.1), TAITime(4602741388.32),
			 TAITime(4604985333.12)]
	t0 = min(times)

	t1 = max(times)
	tt = np.linspace(t0.getValue(), t1.getValue(), 300)
	TT = [TAITime(x) for x in tt]
	
	nm = NanoMaggies.magToNanomaggies(20.)
	pm = RaDecPos(1./3600., 0.)
	mps = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(z=nm), pm, 0.,
							epoch=t0)

	pm = RaDecPos(0., -1./3600.)
	mps2 = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(z=nm), pm, 0.,
							 epoch=t0)

	pm = RaDecPos(0., 0.)
	mps3 = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(z=nm), pm, 1.,
							 epoch=t0)

	pm = RaDecPos(0., 1./3600.)
	mps4 = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(z=nm), pm, 1.,
							 epoch=t0)
	
	plt.clf()
	pp = [mps.getPositionAtTime(t) for t in times]
	plt.plot([p.ra for p in pp], [p.dec for p in pp], 'r.')
	pp = [mps2.getPositionAtTime(t) for t in times]
	plt.plot([p.ra for p in pp], [p.dec for p in pp], 'b.')
	pp = [mps3.getPositionAtTime(t) for t in times]
	#for p,t in zip(pp,times):
	#	print 't', t, '-->', p
	plt.plot([p.ra for p in pp], [p.dec for p in pp], 'g.')
	pp = [mps4.getPositionAtTime(t) for t in TT]
	#for p,t in zip(pp,TT):
	#		print 't', t, '-->', p
	plt.plot([p.ra for p in pp], [p.dec for p in pp], '-', color='0.5', alpha=0.5)
	pp = [mps4.getPositionAtTime(t) for t in times]
	plt.plot([p.ra for p in pp], [p.dec for p in pp], 'm.')
	ps.savefig()


def plot_chain(fn, ps, ra, dec, band, stari):
	X = unpickle_from_file(fn)
	alllnp = np.array(X['alllnp'])
	allp = np.array(X['allp'])
	tr = X['tr']
	
	print 'allp shape', allp.shape
	print 'all lnp shape', alllnp.shape

	# number of steps, number of walkers, number of params
	(N, W, P) = allp.shape
	
	plt.clf()
	for i in range(W):
		plt.plot(alllnp[:,i], 'k.-', alpha=0.2)
	plt.xlabel('emcee step')
	plt.ylabel('lnprob')
	plt.title('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra, dec, band))
	ps.savefig()

	mps = tr.getCatalog()[0]
	nm = mps.getParamNames()
	assert(len(nm) == P)
	nm = [n.split('.')[-1] for n in nm]
	
	for i in range(P):
		
		pp = allp[:,:,i]
		units = ''
		xt = None
		xtl = None
		if i == 0:
			# ra
			pp = (pp - ra) * 3600.
			units = '(arcsec - nominal)'
		elif i == 1:
			# dec
			pp = (pp - dec) * 3600.
			units = '(arcsec - nominal)'
		elif i == 2:
			# z
			pp = NanoMaggies.nanomaggiesToMag(pp)
			units = '(mag)'
			#xt = np.arange(18.02, 18.06, 0.01)
			xt = np.arange(np.floor(pp.min() * 100)/100.,
						   np.ceil(pp.max() * 100)/100., 0.01)
			#18.02, 18.06, 0.01)
			xtl = ['%0.2f' % x for x in xt]
		elif i in [3, 4]:
			# pmra, pmdec
			pp = pp * 3600. * 1000.
			units = '(mas/yr)'
		elif i == 5:
			# parallax
			pp = pp * 1000.
			units = '(mas)'

		plt.clf()
		for j in range(W):
			plt.plot(pp[:,j], 'k.-', alpha=0.2)
		plt.xlabel('emcee step')
		plt.ylabel('%s %s' % (nm[i], units))
		if xt is not None:
			plt.yticks(xt, xtl)
		plt.title('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra, dec, band))
		ps.savefig()

		plt.clf()
		plt.hist(pp[-500:,:].ravel(), 100)
		plt.xlabel('%s %s' % (nm[i], units))
		if xt is not None:
			plt.xticks(xt, xtl)
		plt.title('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra, dec, band))
		ps.savefig()
		

	tr.setParams(allp[-1, 0, :])
	plot_images(tr, ptype='img')
	plt.suptitle('Data - RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()
	plot_images(tr, ptype='mod')
	plt.suptitle('Model - RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()
	plot_images(tr, ptype='mod+noise')
	plt.suptitle('Model - RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()
	plot_images(tr, ptype='chi')
	plt.suptitle('Chi - RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()

	plt.clf()
	k = 1
	pp = allp[-500:,:,:].reshape((-1, P))
	print 'pp shape', pp.shape
	for i in range(P):
		for j in range(P):
			plt.subplot(P, P, k)
			if i == j:
				plt.hist(pp[:,i], 20)
			else:
				#plt.plot(pp[:,j], pp[:,i], 'b.')
				plothist(pp[:,j], pp[:,i], nbins=50, doclf=False, docolorbar=False, dohot=False,
						 imshowargs=dict(cmap=antigray))
			if k % P == 1:
				#print 'k', k, 'i', i, 'j', j, '-> ylabel', nm[i]
				plt.ylabel(nm[i])

			if (k-1) / P == (P - 1):
				plt.xlabel(nm[j])
				#print 'k', k, 'i', i, 'j', j, '-> xlabel', nm[j]
				#else:
			plt.xticks([])
			plt.yticks([])
			k += 1

	plt.suptitle('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()

	

def plot_images(tr, ptype='mod'):
	tims = tr.getImages()
	t0 = min([tim.time for tim in tims])
	N = len(tims)
	cols = int(np.ceil(np.sqrt(N)))
	rows = int(np.ceil(N / float(cols)))
	plt.clf()
	for j,tim in enumerate(tims):
		zr = tim.zr
		plt.subplot(rows, cols, j+1)
		if ptype == 'mod':
			mod = tr.getModelImage(tim)
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
		elif ptype == 'mod+noise':
			mod = tr.getModelImage(tim)
			ie = tim.getInvError()
			ie[ie == 0] = np.mean(ie[ie > 0])
			mod += np.random.normal(size=mod.shape) / ie
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
		elif ptype == 'img':
			plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
		elif ptype == 'chi':
			chi = tr.getChiImage(0)
			plt.imshow(chi, interpolation='nearest', origin='lower',
					   vmin = -5, vmax=5)
		else:
			assert(False)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		plt.title('%.2f yr' % ((tim.time - t0).toYears()), fontsize=8)
	
		

if __name__ == '__main__':
	plot_chain('parallax-star0-z-step0900.pickle', PlotSequence('chain-0z'),
			   52.646659, -0.42659916, 'z', 0)
	plot_chain('parallax-star1-z-step0900.pickle', PlotSequence('chain-1z'),
			   314.48301, -0.83521203, 'z', 1)
	sys.exit(0)


	
window_flist_fn = 'window_flist-DR9.fits'
T = fits_table(window_flist_fn)
print 'Read', len(T), 'S82 fields'
T.cut(T.score >= 0.5)
print 'Cut to', len(T), 'photometric'

sdss = DR9(basedir='data-dr9')

ps = PlotSequence('parallax')

for stari,(ra,dec, pmra,pmdec,parallax) in enumerate(
		[(52.646659, -0.42659916, 414., -355., 40.),
		 (314.48301, -0.83521203, -37., -30., 33.),]):
	I = np.flatnonzero(np.hypot(T.ra - ra, T.dec - dec) < 13./(2.*60.))
	print 'Got', len(I), 'fields possibly in range'

	print 'Runs:', np.unique(T.run[I])

	S = 15

	#for band in 'ugriz':
	for band in 'z':

		tims = []
		for ii in I:
			t = T[ii]
			tim,tinf = get_tractor_image_dr8(
				t.run, t.camcol, t.field, band,
				sdss=sdss, roiradecsize=(ra,dec,S),
				nanomaggies=True, invvarIgnoresSourceFlux=True)
			#zrange=[-2,5])
			if tim is None:
				continue
			#tim.tai = tinf['tai']
			tims.append(tim)

		# print 'times', [tim.time for tim in tims]
		t0 = min([tim.time for tim in tims])
		print 't0:', t0
		t1 = max([tim.time for tim in tims])
		print 't1:', t1

		tmid = (t0 + t1)/2.
		print 'tmid:', tmid


		nm = NanoMaggies.magToNanomaggies(20.)
		pm = PMRaDec(pmra / (1000.*3600.), pmdec / (1000.*3600.))
		#pm.setStepSizes(1e-6)
		print 'Proper motion:', pm
		mps = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(**{band:nm}),
								pm, parallax / 1000., epoch=tmid)
		print 'source:', mps
		

		
		N = len(I)
		cols = int(np.ceil(np.sqrt(N)))
		rows = int(np.ceil(N / float(cols)))

		plt.clf()
		for j,tim in enumerate(tims):
			zr = tim.zr
			plt.subplot(rows, cols, j+1)
			plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
			plt.gray()
			plt.xticks([]); plt.yticks([])
			plt.title('%.2f yr' % ((tim.time - t0).toYears()))
		plt.suptitle('%s band' % band)
		ps.savefig()

		tr = Tractor(tims, [mps])
		print 'Tractor:', tr
		tr.freezeParam('images')
		
		DD = [mps.getParamDerivatives(tim) for tim in tims]
		for i,nm in enumerate(mps.getParamNames()):
			plt.clf()
			for j,tim in enumerate(tims):
				D = DD[j][i]
				dd = np.zeros_like(tim.getImage())
				D.addTo(dd)
				plt.subplot(rows, cols, j+1)
				plt.imshow(dd, interpolation='nearest', origin='lower')
				plt.gray()
				plt.xticks([]); plt.yticks([])
				plt.title('%.2f yr' % ((tim.time - t0).toYears()))
			plt.suptitle('%s band: derivs for param: %s' % (band, nm))
			ps.savefig()

		plt.clf()
		for j,tim in enumerate(tims):
			zr = tim.zr
			plt.subplot(rows, cols, j+1)
			mod = tr.getModelImage(tim)
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
			plt.gray()
			plt.xticks([]); plt.yticks([])
			plt.title('%.2f yr' % ((tim.time - t0).toYears()))
		plt.suptitle('%s band: initial model' % band)
		ps.savefig()
			
		while True:
			dlnp,X,alpha = tr.optimize()
			print 'Stepping by', alpha, 'for dlnp', dlnp
			if alpha == 0:
				break
			if dlnp < 1e-3:
				break

		print 'opt source:', mps
		
		plt.clf()
		for j,tim in enumerate(tims):
			zr = tim.zr
			plt.subplot(rows, cols, j+1)
			mod = tr.getModelImage(tim)
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
			plt.gray()
			plt.xticks([]); plt.yticks([])
			plt.title('%.2f yr' % ((tim.time - t0).toYears()))
		plt.suptitle('%s band: final model' % band)
		ps.savefig()
		

		p0 = np.array(tr.getParams())
		ndim = len(p0)
		nw = 50
		print 'ndim', ndim
		print 'nw', nw

		sampler = emcee.EnsembleSampler(nw, ndim, tr, threads=8)
		
		steps = np.array(tr.getStepSizes())
		colscales = tr.getParameterScales()

		pp0 = np.vstack([p0 + 1e-4 * steps / colscales *
						 np.random.normal(size=len(steps))
						 for i in range(nw)])
		
		alllnp = []
		allp = []

		lnp = None
		pp = pp0
		rstate = None
		for step in range(1000):
			print 'Taking emcee step', step
			pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
			#print 'lnprobs:', lnp

			if (step+0) % 10 == 0:
				mods = [np.zeros_like(tim.getImage()) for tim in tims]
				for k,(p,x) in enumerate(zip(lnp,pp)):
					tr.setParams(x)
					
					for j,tim in enumerate(tims):
						mods[j] += tr.getModelImage(tim)

				for mod in mods:
					mod /= len(lnp)
						
				plt.clf()
				for j,tim in enumerate(tims):
					plt.subplot(rows, cols, j+1)
					zr = tim.zr
					plt.imshow(mods[j], interpolation='nearest', origin='lower',
							   vmin = zr[0], vmax = zr[1])
					plt.gray()
					plt.xticks([]); plt.yticks([])
					plt.title('%.2f yr' % ((tim.time - t0).toYears()))
				plt.suptitle('%s band: sample %i, sum' % (band, step))
				ps.savefig()

				print 'pp', pp.shape
				
				plt.clf()
				nparams = ndim
				nm = [n.split('.')[-1] for n in mps.getParamNames()]
				k = 1
				for i in range(nparams):
					for j in range(nparams):
						plt.subplot(nparams, nparams, k)
						if i == j:
							plt.hist(pp[:,i], 20)
						else:
							plt.plot(pp[:,j], pp[:,i], 'b.')

						if k % nparams == 1:
							#print 'k', k, 'i', i, 'j', j, '-> ylabel', nm[i]
							plt.ylabel(nm[i])

						if (k-1) / nparams == (nparams - 1):
							plt.xlabel(nm[j])
							#print 'k', k, 'i', i, 'j', j, '-> xlabel', nm[j]
							#else:
						plt.xticks([])
						plt.yticks([])
						k += 1

				plt.suptitle('%s band: sample %i' % (band, step))
				ps.savefig()

				
			print 'Max lnprob:', max(lnp)
			print 'Std in lnprobs:', np.std(lnp)

			alllnp.append(lnp.copy())
			allp.append(pp.copy())

			if (step+0) % 100 == 0:
				pickle_to_file(dict(alllnp=alllnp, allp=allp, tr=tr),
							   'parallax-star%i-%s-step%04i.pickle' % (stari, band, step))
			

			
