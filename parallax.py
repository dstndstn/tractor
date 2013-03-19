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
from astrometry.sdss import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import Tan
from astrometry.libkd.spherematch import *

from tractor import *
from tractor.sdss import *

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

	
window_flist_fn = 'window_flist-DR9.fits'
T = fits_table(window_flist_fn)
print 'Read', len(T), 'S82 fields'
T.cut(T.score >= 0.5)
print 'Cut to', len(T), 'photometric'

sdss = DR9(basedir='data-dr9')

ps = PlotSequence('parallax')

for i,(ra,dec) in enumerate([(52.646659, -0.42659916),
							 (314.48301, -0.83521203),]):
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

		nm = NanoMaggies.magToNanomaggies(20.)
		pm = RaDecPos(0., 0.)
		pm.setStepSizes(1e-6)
		mps = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(**{band:nm}),
								pm, 0.)
		print 'source:', mps

		tr = Tractor(tims, [mps])
		print 'Tractor:', tr
		tr.freezeParam('images')
		
		DD = [mps.getParamDerivatives(tim) for tim in tims]
		for i,nm in enumerate(mps.getParamNames()):
			plt.clf()
			for j,tim in enumerate(tims):
				D = DD[j][i]
				plt.subplot(rows, cols, j+1)
				plt.imshow(D.getPatch(), interpolation='nearest', origin='lower')
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
		
