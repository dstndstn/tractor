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
	from optparse import OptionParser
	parser = OptionParser(usage=('%prog'))
	parser.add_option('--steps', dest='steps', type=int, help='Number of steps',
					  default=100)
	parser.add_option('--step0', dest='step0', type=int, help='Starting step',
					  default=0)
	parser.add_option('--nomodplot', dest='modplot', action='store_false', default=True)
	opt,args = parser.parse_args()
	


	lvl = logging.INFO
	#lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
	np.seterr(all='warn')
	np.seterr(divide='raise')

	bands = ['r']
	#RCFS = [(2728, 4, 236),]
	#ra,dec = (333.5596, 0.3671)
	#S = 40
	RCFS = [(756, 3, 243),]
	ra,dec = (152.208922638573, -0.202430118237826)
	S = 18


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

	# DevGalaxy
	s0 = sources[0]
	s0.re = 0.01
	s0.brightness.r = 22.
	
	tractor = Tractor(tims)
	tractor.addSources(sources)

	lnp0 = tractor.getLogProb()

	print 'Catalog:'
	print tractor.catalog
	print tractor.catalog.getParams()

	nthreads = 16
	p0 = np.array(tractor.catalog.getParams())
	ndim = len(p0)
	nw = max(50, 2*ndim)
	print 'ndim', ndim
	print 'nw', nw

	sampler = emcee.EnsembleSampler(nw, ndim, tractor,
									threads=nthreads)
									
	im0 = tractor.getImages()[0]
	steps = np.hstack([src.getStepSizes(im0) for src in tractor.catalog])
	#print 'steps', steps
	print 'p0', p0
	pp0 = np.vstack([p0 + 1e-2 * steps * np.random.normal(size=len(steps))
					 for i in range(nw)])

	plt.figure(1, figsize=(6,6))
	#plt.figure(2, figsize=(18,18))
	plt.figure(2, figsize=(12,12))
	plt.figure(3, figsize=(4,4))

	plt.figure(1)
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.94]

	alllnp = []
	allp = []

	lnp = None #lnp0
	pp = pp0
	rstate = None
	for step in range(opt.step0, opt.steps):

		pfn = 'trample-%03i.pickle' % step
		if os.path.exists(pfn):
			print 'Unpickling', pfn
			allp,alllnp = unpickle_from_file(pfn)
			lnp = alllnp[-1]
			pp  = allp[-1]
		else:
			print 'Taking step', step
			print 'pp shape', pp.shape
			pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
			print 'lnprobs:', lnp

			for p,x in zip(lnp,pp):
				tractor.setAllSourceParams(x)
				print 'lnp', p
				print tractor.catalog

			#print 'pp shape', pp.shape
			# nw,ndim
			alllnp.append(lnp.copy())
			allp.append(pp.copy())
			pickle_to_file((allp, alllnp), pfn)
		
		plt.clf()
		plt.plot(alllnp, 'k', alpha=0.5)
		plt.axhline(lnp0, color='r', lw=2, alpha=0.5)
		mx = np.max([p.max() for p in alllnp])
		plt.ylim(mx-300, mx+10)
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

				if step%50 == 0 and opt.modplot:
					print 'mod,chi'
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

			if step % 10 == 0:
				print 'modbest, chibest'
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

			if step % 1 == 0:
				print 'modsum'
				plt.clf()
				plt.gca().set_position(plotpos0)
				plt.imshow(modsum/float(nw), **ima)
				plt.title('Model')
				plt.xticks([],[])
				plt.yticks([],[])
				plt.savefig('modsum%02i-%02i.png' % (i,step))


			if step % 1 == 0 and False:
				# Plot models, exaggerating the variance
				# pp shape is (nw,ndim)
				print 'exag'
				p0 = pp.mean(axis=0)
				print 'p0 shape', p0
				for i in range(nw):
					delta = pp[i,:] - p0
					exagg = 10.
					pnew = pp[i,:] + (exagg-1.)*delta
					tractor.setAllSourceParams(pnew)
					modex = tractor.getModelImage(0)
					
					plt.clf()
					plt.gca().set_position(plotpos0)
					plt.imshow(modex, **ima)
					plt.title('Model (exaggerated x %g)' % exagg)
					plt.xticks([],[])
					plt.yticks([],[])
					plt.savefig('modex%02i-%02i.png' % (i,step))



		if step % 50 == 0:
			pnames = tractor.catalog[0].getParamNames()
			print 'Params', pnames
			NP = len(pnames)

			# select objid, ra,raerr,dec,decerr,devmag_r,devmagerr_r,devrad_r,devraderr_r,devphi_r,devab_r,devaberr_r from mydb.mytable_2 where run=756 and camcol=3 and field=243
			#objid,ra,raerr,dec,decerr,devmag_r,devmagerr_r,devrad_r,devraderr_r,devphi_r,devab_r,devaberr_r
			#1237648721218568377,152.208922638573,0.00391166133082899,-0.202430118237826,0.00423603626875907,17.59624,0.005895343,1.081388,0.0371073,26.40831,0.435521,0.01508615


			# DR8
			sdssvals = [ 152.208922638573,-0.202430118237826,17.59624,1.081388,0.435521, 26.40831 * -1. ]
			sdsserrs = [ 0.00391166133082899, 0.00423603626875907,0.005895343,0.0371073, 0.01508615, 0. ]
			# DR7; phi*-1; raErr, decErr are in arcsec, and raErr is err in ra*cos(dec)
			sdssvals7 = [ 152.20891415,-0.20242878,17.60266,1.044212,0.4157184,26.3763*-1.,]
			sdsserrs7 = [ 0.0443575305964374 / 3600., 0.041734802988402 / 3600.,0.006056787,0.03344231,0.0143005, 0.]


			tractor.setAllSourceParams(sdssvals7)
			sdssmod = tractor.getModelImage(0)
			sdsschi = tractor.getChiImage(0)
					
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(sdssmod, **ima)
			plt.title('SDSS Model')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('sdssmod.png')
			plt.clf()
			plt.gca().set_position(plotpos0)
			plt.imshow(sdsschi, **imchi)
			plt.title('SDSS Model Chi')
			plt.xticks([],[])
			plt.yticks([],[])
			plt.savefig('sdsschi.png')


			ra0, dec0 = sdssvals7[:2]
			#152.209, -0.202
			pnames = [#'RA - %.3f (arcsec)' % np.abs(ra0),
					  #'Dec + %.3f (arcsec)' % np.abs(dec0),
				'RA - SDSS (milli-arcsec)',
				'Dec - SDSS (milli-arcsec)',
				'r (mag)',
				'effective radius (arcsec)', 'axis ratio',
				'position angle (deg)']

			nsteps = 20
			ppp = allp[-nsteps:]

			tmn = {}
			tstd = {}
			paris = {}
			
			for i in range(NP):
				pari = np.hstack([p[:,i] for p in ppp])

				mn,std,c =(sdssvals7[i],sdsserrs7[i],'m')
				# in [#(sdssvals[i],sdsserrs[i],'r'),

				if i == 0:
					pari -= ra0
					mn -= ra0
					pari *= 3600. * 1000.
					mn *= 3600. * 1000.
					std *= 3600. * 1000.
				elif i == 1:
					pari -= dec0
					mn -= dec0
					pari *= 3600. * 1000.
					mn *= 3600. * 1000.
					std *= 3600. * 1000.

				paris[i] = pari
				tmn[i] = np.mean(pari)
				tstd[i] = np.std(pari)

				plt.figure(3)
				plt.clf()
				xlo, xhi = min(pari.min(), mn-std), max(pari.max(), mn+std)
				n,b,p1 = plt.hist(pari, 25, ec='none', alpha=0.5)
				plt.hist(pari, 25, histtype='step', ec='b', lw=2)
				xx = np.linspace(xlo, xhi, 100)
				ax = plt.axis()
				dx = b[1]-b[0]
				#yy = np.exp(-0.5 * (xx - mn)**2 / std**2)
				#yy /= sum(yy)
				#yy *= sum(n)*dx / (xx[1]-xx[0])
				#print 'yy', yy
				#plt.plot(xx, yy, 'b-', lw=3, alpha=0.8)
				print 'mn', mn, 'std', std
				#plt.axvline(mn, color='b', lw=3, alpha=0.5)

				#for mn,std,c in [#(sdssvals[i],sdsserrs[i],'r'),
				#				 (sdssvals7[i],sdsserrs7[i],'m')]:
				if std != 0:
					yy = np.exp(-0.5 * (xx - mn)**2 / std**2)
					yy /= sum(yy)
					yy *= sum(n)*dx / (xx[1]-xx[0])
					plt.plot(xx, yy, '-', color=c, lw=3, alpha=0.8)
				print 'sdss', mn, std
				p2 = plt.axvline(mn, color=c, lw=3, alpha=0.8)

				#plt.axis(ax)

				#plt.xlim(min(xx.min(), mn-std), max(xx.max(), mn+std))
				plt.xlim(xlo,xhi)
				plt.xlabel(pnames[i])
				plt.legend((p1[0],p2), ('Tractor', 'SDSS DR7'))
				fn = 'modparam%02i-%02i.png' % (step,i)
				print 'save', fn
				plt.savefig(fn)


			plt.figure(2)
			plt.clf()
			plt.subplots_adjust(left=0.01, right=0.98, bottom=0.03, top=0.98,
								wspace=0.1, hspace=0.1)
			for i in range(NP):
				pari = paris[i]
				for j in range(i, NP):
					parj = paris[j]
					plt.subplot(NP, NP, (i*NP)+j+1)
					mn,std = tmn[i],tstd[i]
					irng = (mn-3*std, mn+3*std)
					mn,std = tmn[j],tstd[j]
					jrng = (mn-3*std, mn+3*std)
					if i == j:
						plt.hist(pari, 25, ec='none', alpha=0.5, range=irng)
						plt.hist(pari, 25, histtype='step', ec='b', lw=1, range=irng)
						plt.xlabel(pnames[i])
					else:
						plt.plot(pari, parj, 'b.', alpha=0.1)
						plt.ylim(*jrng)
						#plt.ylabel(pnames[j])
					#plt.xlabel(pnames[i])
					plt.xticks([])
					plt.yticks([])
					plt.xlim(*irng)
			plt.savefig('modhist%02i.png' % step)
			plt.savefig('modhist%02i.pdf' % step)
			plt.figure(1)

if __name__ == '__main__':
	main()
