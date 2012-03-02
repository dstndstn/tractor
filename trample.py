import sys
import logging
import traceback

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits
import emcee
from astrometry.util.util import Tan
from astrometry.util.file import *
from tractor import *
from tractor import sdss as st
from tractor import cfht as cf
from tractor import sdss_galaxy as stgal
from astrometry.sdss import *

def main():
	lvl = logging.INFO
	#lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
	np.seterr(all='warn')
	np.seterr(divide='raise')

	bands = ['r']
	RCFS = [(2728, 4, 236),]
	ra,dec = (333.5596, 0.3671)
	S = 40


	TI = []
	for i,(run,camcol,field) in enumerate(RCFS):
		for bandname in bands:
			im,inf = st.get_tractor_image(run, camcol, field, bandname,
										  useMags=True,
										  roiradecsize=(ra,dec,S))
			#im.dxdy = (fxc - xc, fyc - yc)
			TI.append((im,inf))
	tims = [im for im,inf in TI]
	skyvals = [(info['skysig'], info['sky']) for timg,info in TI]
	zrs = [np.array([-1.,+6.]) * std + sky for std,sky in skyvals]

	# Grab sources from the LAST RCF
	roi = inf['roi']
	sources = st.get_tractor_sources(run, camcol, field, roi=roi,
									 bands=bands)
	
	tractor = Tractor(tims)
	tractor.addSources(sources)

	lnp0 = tractor.getLogProb()

	nthreads = 16
	p0 = np.array(tractor.catalog.getParams())
	ndim = len(p0)
	nw = 2*ndim
	print 'ndim', ndim

	sampler = emcee.EnsembleSampler(nw, ndim, tractor,
									threads=nthreads)
									
	im0 = tractor.getImages()[0]
	steps = np.hstack([src.getStepSizes(im0) for src in tractor.catalog])
	#print 'steps', steps
	print 'p0', p0
	pp0 = np.vstack([p0 + 1e-2 * steps * np.random.normal(size=len(steps))
					 for i in range(nw)])

	plt.figure(figsize=(6,6))
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.94]

	alllnp = []
	allp = []

	lnp = None #lnp0
	pp = pp0
	rstate = None
	for step in range(100):
		print 'Taking step', step
		print 'pp shape', pp.shape
		pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
		print 'lnprobs:', lnp
		print 'pp shape', pp.shape
		# nw,ndim

		alllnp.append(lnp.copy())
		allp.append(pp.copy())
		pickle_to_file((allp, alllnp), 'trample-%03i.pickle' % step)
		
		plt.clf()
		plt.plot(alllnp, 'k', alpha=0.5)
		plt.axhline(lnp0, color='r', lw=2, alpha=0.5)
		mx = np.max([p.max() for p in alllnp])
		plt.ylim(mx-100, mx)
		plt.xlim(0, len(alllnp)-1)
		plt.savefig('lnp.png')

		for i in range(len(tractor.getImages())):
			zr = zrs[i]
			ima = dict(interpolation='nearest', origin='lower',
					   vmin=zr[0], vmax=zr[1], cmap='gray')
			imchi = dict(interpolation='nearest', origin='lower',
						 vmin=-5., vmax=+5., cmap='gray')
			if step == 0:
				tim = tractor.getImage(i)
				data = tim.getImage()

				plt.clf()
				plt.gca().set_position(plotpos0)
				plt.imshow(data, **ima)
				plt.title('Data %s' % tim.name)
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('data%02i.png' % i)

			modsum = None
			#KK = 
			for k in xrange(nw):
				tractor.setAllSourceParams(pp[k,:])
				mod = tractor.getModelImage(i)
				if k == 0:
					modsum = mod
				else:
					modsum += mod

				if step%10 == 0:
					plt.clf()
					plt.gca().set_position(plotpos0)
					plt.imshow(mod, **ima)
					plt.title('Model')
					plt.xticks([],[])
					plt.yticks([],[])
					plt.savefig('mod%02i-%02i-%03i.png' % (i,step,k))

					chi = tractor.getChiImage(i)
					plt.clf()
					plt.gca().set_position(plotpos0)
					plt.imshow(chi, **imchi)
					plt.title('Chi')
					plt.xticks([],[])
					plt.yticks([],[])
					plt.savefig('chi%02i-%02i-%03i.png' % (i,step,k))

			ibest = np.argmax(lnp)
			print 'ibest', ibest
			tractor.setAllSourceParams(pp[ibest,:])
			mod = tractor.getModelImage(i)
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(mod, **ima)
			plt.title('Best Model')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('modbest%02i-%02i.png' % (i,step))

			chi = tractor.getChiImage(i)
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(chi, **imchi)
			plt.title('Best Chi')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('chibest%02i-%02i.png' % (i,step))
			

			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(modsum/float(nw), **ima)
			plt.title('Model')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('modsum%02i-%02i.png' % (i,step))



if __name__ == '__main__':
	main()
