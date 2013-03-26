'''
Date: Wed, 23 Jan 2013 00:38:33 -0300
From: Jacqueline Faherty <jfaherty17@gmail.com>

NAME RA DEC PMra (mas/yr) PMdec (mas/yr) Pi (mas-expected)
SDSS0330-0025 52.646659 -0.42659916 414+/-30 -355+/-51 40+/-10
SDSS2057-0050 314.48301 -0.83521203 -37.2+/-20 -30+/-27 33+/-10
'''

'''
WISE query:
at
http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd
on WISE All-Sky Single Exposure (L1b) Source Table
'''

'''
UKIDSS query:
http://surveys.roe.ac.uk:8080/wsa/SQL_form.jsp

select d.ra,d.dec,d.multiframeid,d.filterid,d.xerr,d.yerr,
m.frametype,m.mjdobs
from lasDetection as d
   JOIN multiframe as m on d.multiframeid = m.multiframeid
where ra between 52 and 53 and dec between -1 and 0
and 
abs(ra - 52.646659) + abs(dec + 0.42659916) < 0.003
'''


import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('font', serif='computer modern roman')
matplotlib.rc('font', **{'sans-serif': 'computer modern sans serif'})
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.gator import *
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


def plot_chain(fn, ps, ra, dec, band, stari, tag, opt):
	X = unpickle_from_file(fn)
	alllnp = np.array(X['alllnp'])
	allp = np.array(X['allp'])
	tr = X['tr']
	
	print 'allp shape', allp.shape
	print 'all lnp shape', alllnp.shape

	# number of steps, number of walkers, number of params
	(N, W, P) = allp.shape

	mps = tr.getCatalog()[0]
	if opt.cat:
		# unwrap
		mps = mps.mps
		
	nm = mps.getParamNames()
	assert(len(nm) == P)
	nm = [n.split('.')[-1] for n in nm]

	tims = tr.getImages()
	times = [tim.time for tim in tims]
	t0 = min(times)
	t1 = max(times)

	print 'SDSS parallax angles:'
	for t in times:
		print '  ', np.fmod(360. + np.rad2deg(t.getSunTheta()), 360.)
		
	if opt.wise:
		wise = gator2fits('wise_allsky.wise_allsky_4band_p1bs_psd29506.tbl')
		print 'Read WISE table:'
		wise.about()
		wise.writeto('wise.fits')

		wisetimes = [TAITime(w.mjd * 24. * 3600.) for w in wise]
		t0 = min(t0, min(wisetimes))
		t1 = max(t1, max(wisetimes))

	if opt.ukidss:
		ukidss = fits_table('ukidss.fits')
		utimes = [TAITime(u.mjdobs * 24. * 3600.) for u in ukidss]
		t0 = min(t0, min(utimes))
		t1 = max(t1, max(utimes))

		print 'UKIDSS parallax angles:'
		for t in utimes:
			print '  ', np.rad2deg(t.getSunTheta())

	if opt.twomass:
		twomass = gator2fits('fp_2mass.fp_psc12438.tbl')
		twomass.about()
		ttimes = [TAITime(jdtomjd(t.jdate) * 24. * 3600.) for t in twomass]
		t0 = min(t0, min(ttimes))
		t1 = max(t1, max(ttimes))

			
	print 'Times', t0, t1
	sixmonths = (3600. * 24. * 180)
	t0 = t0 - sixmonths
	t1 = t1 + sixmonths
	print 'Times', t0, t1
	TT = [TAITime(x) for x in np.linspace(t0.getValue(), t1.getValue(), 300)]

	sdss = DR9(basedir='data-dr9')
	bandnum = band_index(band)

	pfn = 'pobjs-%s.pickle' % tag
	if os.path.exists(pfn):
		print 'Reading', pfn
		pobjs = unpickle_from_file(pfn)
	else:
		pobjs = []
		for i,tim in enumerate(tims):
			r,c,f = tim.rcf
			S = 15
			rad = S * np.sqrt(2.) * 0.396 / 3600.
			cat,objs,objI = get_tractor_sources_dr9(
				r,c,f, bandname=band,
				sdss=sdss, radecrad=(ra,dec,rad),
				getobjs=True, getobjinds=True)
			print 'Got', len(cat), 'sources;', len(objI)
			if len(cat) == 0:
				continue
			if len(cat) > 1:
				print 'Found too many objects!'
				continue
			assert(len(objI) == len(cat))
			objs.cut(objI)
			obj = objs[0]
	
			# get the AsTrans
			sdss.retrieve('frame', r, c, f, band)
			frame = sdss.readFrame(r, c, f, band)
			ast = frame.getAsTrans()
			# r-i color
			color = obj.psfmag[band_index('r')] - obj.psfmag[band_index('i')]
			print 'color', color
			x,y = obj.colc[bandnum], obj.rowc[bandnum]
			r0,d0 = ast.pixel_to_radec(x, y, color=color)
			#pos = RaDecPos(ra, dec)
			pobjs.append((i, obj, r0,d0))
			
		pickle_to_file(pobjs, pfn)
	
	# Plot samples in RA,Dec coords.
	plt.clf()
	for i in range(10):
		tr.setParams(allp[-1, i, :])
		pp = [mps.getPositionAtTime(t) for t in TT]
		rr,dd = np.array([p.ra for p in pp]), np.array([p.dec for p in pp])
		plt.plot((rr-ra)*3600., (dd-dec)*3600., '-', color='k', alpha=0.2, zorder=10)
		pp = [mps.getPositionAtTime(t) for t in times]
		rr,dd = np.array([p.ra for p in pp]), np.array([p.dec for p in pp])
		plt.plot((rr-ra)*3600., (dd-dec)*3600., 'b.', zorder=20) #'o', mec='k', mfc='b')

	#plt.plot([(o.ra - ra)*3600. for i,o,r,d in pobjs],
	#		 [(o.dec - dec)*3600. for i,o,r,d in pobjs], 'r.')
	#plt.plot([(r - ra)*3600. for i,o,r,d in pobjs],
	#		 [(d - dec)*3600. for i,o,r,d in pobjs], 'o', mec='g', mfc='none', ms=10, lw=2, zorder=15)

	angle = np.linspace(0, 2.*np.pi, 30)
	for i,obj,r,d in pobjs:
		xe,ye = obj.colcerr[bandnum], obj.rowcerr[bandnum]
		#print 'Pixel error:', (xe+ye)/2.

		dradec = ((xe + ye)/2.) * 0.396 / 3600.
		#r = r + np.sin(angle) * dradec
		#d = d + np.cos(angle) * dradec
		#plt.plot((r-ra)*3600., (d-dec)*3600., 'r-', lw=2, alpha=0.5, zorder=30)

		# 3-sigma
		dradec *= 3
		
		plt.plot((np.array([r + dradec, r - dradec]) - ra)*3600.,
				 (np.array([d, d]) - dec)*3600.,
				 'r-', lw=2, alpha=0.5, zorder=30)
		plt.plot((np.array([r, r]) - ra)*3600.,
				 (np.array([d + dradec, d - dradec]) - dec)*3600.,
				 'r-', lw=2, alpha=0.5, zorder=30)
		plt.plot((r - ra)*3600.,
				 (d - dec)*3600.,
				 'r.', zorder=30)


	if opt.wise:
		for w in wise:
			dr = w.sigra / 3600.
			dd = w.sigdec / 3600.
			nsig = 1.
			plt.plot((np.array([[w.ra  + dr*nsig, w.ra  - dr*nsig],[w.ra,w.ra]]).T - ra )*3600.,
					 (np.array([[w.dec, w.dec], [w.dec + dd*nsig, w.dec - dd*nsig]]).T - dec)*3600.,
					 'm-', lw=2, alpha=0.5, zorder=30)
	if opt.ukidss:
		for u in ukidss:
			# They give RA,Dec in RADIANS.  Really...
			u.ra  *= 180./np.pi
			u.dec *= 180./np.pi
			# MAGIC 0.4 arcsec/pix
			dr = u.xerr * 0.4 / 3600.
			dd = u.yerr * 0.4 / 3600.
			nsig = 3.
			plt.plot((np.array([[u.ra  + dr*nsig, u.ra  - dr*nsig],[u.ra,u.ra]]).T - ra )*3600.,
					 (np.array([[u.dec, u.dec], [u.dec + dd*nsig, u.dec - dd*nsig]]).T - dec)*3600.,
					 'g-', lw=2, alpha=0.5, zorder=30)

	if opt.twomass:
		for t in twomass:
			dr = dd = t.err_maj / 3600.
			nsig = 1.
			plt.plot((np.array([[t.ra  + dr*nsig, t.ra  - dr*nsig],[t.ra,t.ra]]).T - ra )*3600.,
					 (np.array([[t.dec, t.dec], [t.dec + dd*nsig, t.dec - dd*nsig]]).T - dec)*3600.,
					 'c-', lw=2, alpha=0.5, zorder=30)
			
	plt.xlabel('RA - nominal (arcsec)')
	plt.ylabel('Dec - nominal (arcsec)')
	ps.savefig()

	# Plot the trajectories on a little RA,Dec zoomin of each object.
	N = len(tims)
	cols = int(np.ceil(np.sqrt(N)))
	rows = int(np.ceil(N / float(cols)))

	for k in [1]: #range(2):
		plt.clf()
		for i,obj,r0,d0 in pobjs:
			tim = tims[i]
			plt.subplot(rows, cols, i+1)
			plt.plot(r0, d0, 'bo')
	
			xe,ye = obj.colcerr[bandnum], obj.rowcerr[bandnum]
			print 'Pixel error:', (xe+ye)/2.
			dradec = ((xe + ye)/2.) * 0.396 / 3600.

			angle = np.linspace(0, 2.*np.pi, 360)
			plt.plot(r0 + np.sin(angle)*dradec,
					 d0 + np.cos(angle)*dradec, 'b-', lw=2, alpha=0.5)
			plt.plot(r0 + np.sin(angle)*dradec * 2,
					 d0 + np.cos(angle)*dradec * 2, 'b-', lw=2, alpha=0.5)

			for i in range(10):
				tr.setParams(allp[-1, i, :])
				pp = [mps.getPositionAtTime(t) for t in TT]
				rr,dd = np.array([p.ra for p in pp]), np.array([p.dec for p in pp])
				plt.plot(rr, dd, '-', color='k', alpha=0.2)

				pp = mps.getPositionAtTime(tim.time)
				plt.plot(pp.ra, pp.dec, 'b.')
				
			R = 0.396 / 3600.
			if k == 1:
				R *= 5 * (xe + ye) / 2.
			plt.axis([r0-R, r0+R, d0-R, d0+R])
			plt.xticks([]); plt.yticks([])
			
		ps.savefig()
		
		
	
	PA = plot_images(tr, ptype='img')

	if not opt.cat:
		rows = PA['rows']
		cols = PA['cols']
		for j,tim in enumerate(tims):
			plt.subplot(rows, cols, j+1)
			for i in range(10):
				tr.setParams(allp[-1, i, :])
				pos = mps.getPositionAtTime(tim.time)
				x,y = tim.getWcs().positionToPixel(pos)
				plt.plot(x, y, 'r.')
	plt.suptitle('RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()
	
	plt.clf()
	for i in range(W):
		plt.plot(alllnp[:,i], 'k.-', alpha=0.2)
	plt.xlabel('emcee step')
	plt.ylabel('lnprob')
	plt.title('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra, dec, band))
	ps.savefig()

	nm.append('lnp')
	
	for i in range(P+1):

		print 'param', i, ':', nm[i]
		
		if i < P:
			pp = allp[:,:,i]
		else:
			pp = alllnp
		units = ''
		xt = None
		xtl = None
		mfmt = '%g'
		sfmt = '%g'
		if nm[i] == 'ra':
			pp = (pp - ra) * 3600.
			units = '(arcsec - nominal)'
			mfmt = sfmt = '%.3f'
		elif nm[i] == 'dec':
			pp = (pp - dec) * 3600.
			units = '(arcsec - nominal)'
			mfmt = sfmt = '%.3f'
		elif nm[i] == 'z':
			pp = NanoMaggies.nanomaggiesToMag(pp)
			units = '(mag)'
			xt = np.arange(np.floor(pp.min() * 100)/100.,
						   np.ceil(pp.max() * 100)/100., 0.01)
			xtl = ['%0.2f' % x for x in xt]
			mfmt = sfmt = '%.3f'
		elif nm[i] in ['pmra', 'pmdec']:
			# pmra, pmdec
			pp = pp * 3600. * 1000.
			units = '(mas/yr)'
			mfmt = sfmt = '%.1f'
		elif nm[i].lower() == 'parallax':
			# parallax
			pp = pp * 1000.
			units = '(mas)'
			mfmt = sfmt = '%.0f'

		plt.clf()
		for j in range(W):
			plt.plot(pp[:,j], 'k.-', alpha=0.2)
		plt.xlabel('emcee step')
		plt.ylabel('%s %s' % (nm[i], units))
		if xt is not None:
			plt.yticks(xt, xtl)
		plt.title('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra, dec, band))
		ps.savefig()

		p = pp[-500:,:].ravel()
		mn = np.mean(p)
		st = np.std(p)
			
		plt.clf()
		n,b,p = plt.hist(p, 100, histtype='step', color='b')
		ax = plt.axis()
		X = np.linspace(ax[0], ax[1], 100)
		Y = np.exp(-0.5 * (X-mn)**2/(st**2))
		Y /= sum(Y)
		Y *= sum(n) * (b[1]-b[0]) / (X[1]-X[0])
		plt.plot(X, Y, 'b-', lw=3, alpha=0.5)
		plt.xlabel('%s %s' % (nm[i], units))
		if xt is not None:
			plt.xticks(xt, xtl)
		plt.title('RA,Dec=(%.3f,%.3f), %s band' % (ra, dec, band))
		plt.ylim(min(ax[2],0), 1.08 * max(max(n), max(Y)))
		plt.xlim(ax[0],ax[1])
		ax = plt.axis()
		plt.text(ax[0] + 0.1 * (ax[1]-ax[0]), ax[2]+0.95*(ax[3]-ax[2]),
				 (r'%%s = $%s \pm %s$ %%s' % (mfmt,sfmt)) % (nm[i], mn, st, units),
				 bbox=dict(fc='w', alpha=0.5, ec='None'))
		plt.axis(ax)
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
				plothist(pp[:,j], pp[:,i], nbins=50, doclf=False, docolorbar=False, dohot=False,
						 imshowargs=dict(cmap=antigray))
			if k % P == 1:
				plt.ylabel(nm[i])

			if (k-1) / P == (P - 1):
				plt.xlabel(nm[j])
			plt.xticks([])
			plt.yticks([])
			k += 1

	plt.suptitle('Source at RA,Dec=(%.3f,%.3f), %s band' % (ra,dec,band))
	ps.savefig()


	
	

def plot_images(tr, ptype='mod', mods=None):
	tims = tr.getImages()
	t0 = min([tim.time for tim in tims])
	N = len(tims)
	cols = int(np.ceil(np.sqrt(N)))
	rows = int(np.ceil(N / float(cols)))
	plt.clf()
	for j,tim in enumerate(tims):
		zr = tim.zr
		plt.subplot(rows, cols, j+1)

		if ptype in ['mod', 'mod+noise', 'chi']:
			if mods is None:
				mod = tr.getModelImage(tim)
			else:
				mod = mods[j]

		if ptype == 'mod':
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
		elif ptype == 'mod+noise':
			ie = tim.getInvError()
			ie[ie == 0] = np.mean(ie[ie > 0])
			mod = mod + np.random.normal(size=mod.shape) / ie
			plt.imshow(mod, interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
		elif ptype == 'img':
			plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
					   vmin = zr[0], vmax = zr[1])
		elif ptype == 'chi':
			#chi = tr.getChiImage(j)
			chi = (tim.getImage() - mod) * tim.getInvError()
			plt.imshow(chi, interpolation='nearest', origin='lower',
					   vmin = -5, vmax=5)
		else:
			assert(False)
		plt.gray()
		plt.xticks([]); plt.yticks([])

		if True:
			plt.title('%.1f yr' % ((tim.time - t0).toYears()), fontsize=8)
			#plt.title('%.1f yr (sc %.2f)' % ((tim.time - t0).toYears(), tim.score),
			#		  fontsize=8)
		else:
			r,c,f = tim.rcf
			plt.title('%.1f (%.2f) %i-%i-%i' %
					  ((tim.time - t0).toYears(), tim.score, r,c,f),
					  fontsize=8)

	return dict(rows=rows, cols=cols)


class SourceInfo(object):
	def __init__(self, **kwargs):
		for k,v in kwargs.items():
			setattr(self, k, v)

def getSource(i):
	srcs = []
	for stari,(ra,dec, pmra,pmdec,parallax) in enumerate(
		[(52.646659, -0.42659916, 414., -355., 40.),
		(314.48301, -0.83521203, -37., -30., 33.),]):
		srcs.append(SourceInfo(ra=ra, dec=dec, pmra=pmra, pmdec=pmdec,
							   parallax=parallax, srci=stari))
	return srcs[i]






class CatalogFakeSource(ParamsWrapper):
	def __init__(self, mps):
		super(CatalogFakeSource, self).__init__(mps)
		self.mps = mps

	def getModelPatch(self, img, **kwargs):
		pos = self.mps.getPositionAtTime(img.time)
		patch = np.array([[pos.ra - img.ra0, pos.dec - img.dec0]])
		return Patch(0, 0, patch)

	def getParamDerivatives(self, img):
		'''
		returns [ Patch, Patch, ... ] of length numberOfParams().
		'''
		derivs = []
		p0 = self.getModelPatch(img)
		steps = self.mps.getStepSizes(img)
		vals = self.mps.getParams()
		for i,step in enumerate(steps):
			oldval = self.mps.setParam(i, vals[i] + step)
			p = self.getModelPatch(img)
			self.mps.setParam(i, oldval)
			derivs.append((p - p0) / step)
		return derivs
	
class CatalogFakeImage(BaseParams):
	def __init__(self, ra, dec, dra, ddec, time, rcfb):
		self.ra0 = ra
		self.dec0 = dec
		self.sky = ConstantSky(0.)
		self.time = time
		#self.data = np.array([[ra, dec]])
		self.data = np.array([[0., 0.]])
		self.inverr = np.array([[1./dra, 1./ddec]])
		#self.invvar = np.array([[1./dra**2, 1./ddec**2]])
		if rcfb:
			self.name = 'CatalogFakeImage: %i/%i/%i/%s' % rcfb
		#self.zr = (-180, 180)
		self.zr = (-1./3600., 1./3600.)

	def hashkey(self): return (tuple(self.data.ravel()),
							   tuple(self.inverr.ravel()),
							   self.time.hashkey())
		
	def getImage(self): return self.data

	def getTime(self): return self.time

	def getShape(self): return self.data.shape

	shape = property(getShape, None, None, 'shape of this image: (H,W)')
		
	def getInvError(self): return self.inverr

	# uh, you mean 2?
	def numberOfPixels(self):
		H,W = self.getShape()
		return H*W


def run_star(src, band, tag, opt):
	window_flist_fn = 'window_flist-DR9.fits'
	T = fits_table(window_flist_fn)
	print 'Read', len(T), 'S82 fields'
	T.cut(T.score >= 0.5)
	print 'Cut to', len(T), 'photometric'

	sdss = DR9(basedir='data-dr9')

	stari = src.srci
	
	ps = PlotSequence('parallax-%s' % (tag))

	if opt.wise:
		wise = gator2fits('wise_allsky.wise_allsky_4band_p1bs_psd29506.tbl')
		print 'Read WISE table:'
		wise.about()
		wise.writeto('wise.fits')
	ra = src.ra
	dec = src.dec
	pmra = src.pmra
	pmdec = src.pmdec
	parallax = src.parallax
	bandnum = band_index(band)
	
	if opt.freeze_parallax:
		print 'Freezing parallax to zero.'
		parallax = 0.
	
	I = np.flatnonzero(np.hypot(T.ra - ra, T.dec - dec) < np.hypot(13.,9.)/(2.*60.))
	print 'Got', len(I), 'fields possibly in range'
	print 'Runs:', np.unique(T.run[I])
	S = 15

	# We use this to estimate the source position within each image
	# when deciding whether to keep or not.
	tmid = TAITime(4.55759e+09)

	nm = NanoMaggies.magToNanomaggies(20.)
	pm = PMRaDec(pmra / (1000.*3600.), pmdec / (1000.*3600.))
	mps = MovingPointSource(RaDecPos(ra, dec), NanoMaggies(**{band:nm}),
							pm, parallax / 1000., epoch=tmid)

	psfmod = 'kl-gm'
	if opt.pixpsf:
		psfmod = 'kl-pix'

	tims = []
	
	for ii in I:
		t = T[ii]
		print
		print 'R/C/F', t.run, t.camcol, t.field

		if opt.cat:
			# search radius for catalog sources, in deg
			rad = S * np.sqrt(2.) * 0.396 / 3600.
			cat,objs,objI = get_tractor_sources_dr9(
				t.run, t.camcol, t.field, bandname=band,
				sdss=sdss, radecrad=(ra,dec,rad),
				getobjs=True, getobjinds=True)
			print 'Got', len(cat), 'sources;', len(objI)
			if len(cat) == 0:
				continue
			if len(cat) > 1:
				print 'Found too many objects!'
				continue
			assert(len(objI) == len(cat))
			objs.cut(objI)
			obj = objs[0]
			print 'Score', obj.score
			if obj.score < 0.5:
				continue
			y = obj.rowc[bandnum]
			if y < 64. or y > 1425.:
				print 'Duplicate field -- y=%f' % y
				continue

			src = cat[0]


			# sdss.retrieve('frame', t.run, t.camcol, t.field, 'r')
			# frame = sdss.readFrame(t.run, t.camcol, t.field, 'r')
			# ast = frame.getAsTrans()
			# wcs = SdssWcs(ast)
			# x,y = obj.colc[2], obj.rowc[2]
			# pos = wcs.pixelToPosition(x, y)
			# print 'r', 'pixel position ->', pos.ra, pos.dec
			# print '          vs RA,Dec', obj.ra, obj.dec

			# get the AsTrans
			sdss.retrieve('frame', t.run, t.camcol, t.field, band)
			frame = sdss.readFrame(t.run, t.camcol, t.field, band)
			ast = frame.getAsTrans()

			# r-i color
			color = obj.psfmag[band_index('r')] - obj.psfmag[band_index('i')]
			print 'color', color
			
			x,y = obj.colc[bandnum], obj.rowc[bandnum]

			ra,dec = ast.pixel_to_radec(x, y, color=color)
			pos = RaDecPos(ra, dec)
			
			print band, 'pixel position ->', pos.ra, pos.dec
			print '          vs RA,Dec', obj.ra, obj.dec

			wcs = SdssWcs(ast)
			p2 = wcs.pixelToPosition(x, y)
			print '          vs no-color WCS RA,Dec', p2.ra, p2.dec

			
			xe,ye = obj.colcerr[bandnum], obj.rowcerr[bandnum]
			print 'xerr,yerr', xe,ye
			# isotropize, convert to ra,dec deg
			dradec = ((xe + ye)/2.) * 0.396 / 3600.

			# {RA,Dec}{,err} use the r-band values, which are much worse for these objects!
			# obj.ra, obj.dec, obj.raerr / 3600., obj.decerr / 3600.,

			time = TAITime(obj.tai[bandnum])
			tim = CatalogFakeImage(pos.ra, pos.dec, dradec, dradec,
								   time, (obj.run, obj.camcol, obj.field, band))
			tim.rcf = (t.run, t.camcol, t.field)
			tim.score = t.score
			tims.append(tim)
			continue

		tim,tinf = get_tractor_image_dr9(
			t.run, t.camcol, t.field, band,
			sdss=sdss, roiradecsize=(ra,dec,S),
			nanomaggies=True, invvarIgnoresSourceFlux=True,
			psf=psfmod)
		if tim is None:
			continue
		tim.score = t.score

		pos = mps.getPositionAtTime(tim.time)
		x,y = tim.getWcs().positionToPixel(pos)
		#print 'x,y', x,y
		H,W = tim.shape
		#print 'W,H', W,H
		#print 'roi', tinf['roi']
		if x < 0 or y < 0 or x >= W or y >= H:
			print 'Skipping -- OOB'
			continue

		roi = tinf['roi']
		ymid = (roi[2]+roi[3]) / 2.
		if ymid < 64. or ymid > 1425.:
			print 'Duplicate field -- ymid=%f' % ymid
			continue
			
		tim.rcf = (t.run, t.camcol, t.field)
		tims.append(tim)



	# if opt.wise:
	# 	for iw,w in enumerate(wise):
	# 
	# 		time = TAITime(w.mjd * 24. * 3600.)
	# 		tim = CatalogFakeImage(w.ra, w.dec,
	# 							   w.sigra / 3600., w.sigdec / 3600.,
	# 		 					   time, None)
	# 		tim.name = 'WISE %i' % iw
	# 		tims.append(tim)

	times = [tim.time for tim in tims]
	t0 = min(times)
	print 't0:', t0
	t1 = max(times)
	print 't1:', t1
	tmid = (t0 + t1)/2.
	print 'tmid:', tmid

	# Re-create this object now that we have the real "tmid".
	pos = RaDecPos(ra, dec)
	pos.setStepSizes(1e-6)
	mps = MovingPointSource(pos, NanoMaggies(**{band:nm}),
							pm, parallax / 1000., epoch=tmid)
	src = mps

	if opt.cat:
		src = CatalogFakeSource(mps)

	tr = Tractor(tims, [src])
	tr.freezeParam('images')
	print 'Tractor:', tr

	if opt.cat:
		tr.modtype = np.float64
		mps.freezeParam('brightness')
		
	if opt.freeze_parallax:
		mps.freezeParam('parallax')
	print 'MPS args:'
	mps.printThawedParams()
	
	PA = plot_images(tr, ptype='img')
	plt.suptitle('%s band: data' % (band))
	ps.savefig()

	rows = PA['rows']
	cols = PA['cols']

	# Plot derivatives
	#if not opt.cat:
	if True:
		DD = [src.getParamDerivatives(tim) for tim in tims]
		for i,nm in enumerate(src.getParamNames()):
			plt.clf()
			for j,tim in enumerate(tims):
				D = DD[j][i]
				dd = np.zeros_like(tim.getImage())
				D.addTo(dd)
				mx = max(dd.max(), np.abs(dd.min()))
				plt.subplot(rows, cols, j+1)
				plt.imshow(dd, interpolation='nearest', origin='lower',
						   vmin=-mx, vmax=mx)
				plt.gray()
				plt.xticks([]); plt.yticks([])
				plt.title('%.2f yr' % ((tim.time - t0).toYears()))
				plt.suptitle('%s band: derivs for param: %s' % (band, nm))
			ps.savefig()

	plot_images(tr, ptype='mod')
	plt.suptitle('%s band: initial model' % band)
	ps.savefig()

	plot_images(tr, ptype='chi')
	plt.suptitle('%s band: initial chi' % band)
	ps.savefig()


	mps.freezeParams('parallax', 'pm')
	while True:
		dlnp,X,alpha = tr.optimize()
		print 'Stepping by', alpha, 'for dlnp', dlnp
		if alpha == 0:
			break
		if dlnp < 1e-3:
			break
	mps.thawParams('parallax', 'pm')
	print 'opt source 1:', src
	while True:
		dlnp,X,alpha = tr.optimize()
		print 'Stepping by', alpha, 'for dlnp', dlnp
		if alpha == 0:
			break
		if dlnp < 1e-3:
			break

	print 'opt source:', src

	plot_images(tr, ptype='mod')
	plt.suptitle('%s band: opt model' % band)
	ps.savefig()

	plot_images(tr, ptype='chi')
	plt.suptitle('%s band: opt chi' % band)
	ps.savefig()

	p0 = np.array(tr.getParams())
	ndim = len(p0)
	nw = 50

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
	for step in range(1001):
		print 'Taking emcee step', step
		pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
		#print 'lnprobs:', lnp

		if step and (step % 100 == 0):
			mods = [np.zeros_like(tim.getImage()) for tim in tims]
			for k,(p,x) in enumerate(zip(lnp,pp)):
				tr.setParams(x)
				for j,tim in enumerate(tims):
					mods[j] += tr.getModelImage(tim)
			for mod in mods:
				mod /= len(lnp)

			plot_images(tr, ptype='mod', mods=mods)
			plt.suptitle('%s band: sample %i, sum, model' % (band, step))
			ps.savefig()
			
			plot_images(tr, ptype='chi', mods=mods)
			plt.suptitle('%s band: sample %i, sum, chi' % (band, step))
			ps.savefig()
				
			#print 'pp', pp.shape
			
			plt.clf()
			nparams = ndim
			nm = [n.split('.')[-1] for n in mps.getParamNames()]
			k = 1
			for i in range(nparams):
				for j in range(nparams):
					plt.subplot(nparams, nparams, k)
					if i == j:
						plt.hist(pp[:,i], 20)
						print nm[i], ': mean', np.mean(pp[:,i]), 'std', np.std(pp[:,i])
					else:
						plt.plot(pp[:,j], pp[:,i], 'b.')

					if k % nparams == 1:
						plt.ylabel(nm[i])
					if (k-1) / nparams == (nparams - 1):
						plt.xlabel(nm[j])
					plt.xticks([])
					plt.yticks([])
					k += 1
			plt.suptitle('%s band: sample %i' % (band, step))
			ps.savefig()

			plt.clf()
			for i in range(nparams + 1):
				if i == 0:
					p = np.array(alllnp)
					ll = 'lnp'
				else:
					p = np.array(allp)[:,:,i-1]
					ll = nm[i-1]
				plt.subplot(nparams+1, 1, i+1)
				N,W = p.shape
				for j in range(W):
					plt.plot(p[:,j], 'k-', alpha=0.2)
				plt.xlabel(ll)
			ps.savefig()
			
		print 'Max lnprob:', max(lnp)
		print 'Std in lnprobs:', np.std(lnp)

		alllnp.append(lnp.copy())
		allp.append(pp.copy())

		if (step+0) % 100 == 0:
			pickle_to_file(dict(alllnp=alllnp, allp=allp, tr=tr),
						   'parallax-star%s-step%04i.pickle' % (tag, step))
		
	pickle_to_file(dict(alllnp=alllnp, allp=allp, tr=tr),
				   'parallax-star%s-end.pickle' % (tag))


def compare1(src, tag1, tag2, pfn1, pfn2, opt, band):
	'''
	Compare parallax (1) vs no-parallax (2).
	'''
	c1 = 'b'
	c2 = 'r'	

	X1 = unpickle_from_file(pfn1)
	alllnp1 = np.array(X1['alllnp'])
	allp1 = np.array(X1['allp'])
	tr1 = X1['tr']

	X2 = unpickle_from_file(pfn2)
	alllnp2 = np.array(X2['alllnp'])
	allp2 = np.array(X2['allp'])
	tr2 = X2['tr']

	ps = PlotSequence('compare1-%s-%s' % (tag1, tag2))
	
	print 'allp shape', allp1.shape
	print 'all lnp shape', alllnp1.shape
	
	plt.clf()
	n,b,p1 = plt.hist(alllnp1[-500:, :].ravel(), 100, histtype='step', color=c1)
	n,b,p2 = plt.hist(alllnp2[-500:, :].ravel(), 100, histtype='step', color=c2)
	plt.legend((p1[0], p2[0]), ('With parallax', 'No parallax'),
			   loc='upper center')
	plt.xlabel('log-prob of samples')
	plt.ylabel('number of samples')
	plt.title('Parallax vs No-Parallax comparison -- Source %i, %s band' % (src.srci, band))
	ps.savefig()

	# Plot samples in RA,Dec coords.
	plt.clf()
	for tr, allp, cc in [(tr1, allp1, c1), (tr2, allp2, c2)]:
		tims = tr.getImages()
		mps = tr.getCatalog()[0]
		ra,dec = src.ra, src.dec
		times = [tim.time for tim in tims]
		t0 = min(times)
		t1 = max(times)
		print 'Times', t0, t1
		sixmonths = (3600. * 24. * 180)
		t0 = t0 - sixmonths
		t1 = t1 + sixmonths
		print 'Times', t0, t1
		print 't0', t0.getValue(), t0.getParams()
		TT = [TAITime(x) for x in np.linspace(t0.getValue(), t1.getValue(), 300)]
		for i in range(10):
			tr.setParams(allp[-1, i, :])
			pp = [mps.getPositionAtTime(t) for t in TT]
			rr,dd = np.array([p.ra for p in pp]), np.array([p.dec for p in pp])
			plt.plot((rr-ra)*3600., (dd-dec)*3600., '-', color='k', alpha=0.2, zorder=20)
			pp = [mps.getPositionAtTime(t) for t in times]
			rr,dd = np.array([p.ra for p in pp]), np.array([p.dec for p in pp])
			plt.plot((rr-ra)*3600., (dd-dec)*3600., '.', color=cc, alpha=0.2, zorder=30) #'o', ms=4, mec=cc, mfc='none', alpha=0.5, zorder=30)
	plt.xlabel('RA - nominal (arcsec)')
	plt.ylabel('Dec - nominal (arcsec)')
	ps.savefig()

	i1 = np.argmax(alllnp1.ravel())
	i2 = np.argmax(alllnp2.ravel())
	print 'max lnp1', alllnp1.flat[i1]
	print 'max lnp2', alllnp2.flat[i2]
	(s1,w1) = np.unravel_index(i1, (alllnp1.shape))
	(s2,w2) = np.unravel_index(i2, (alllnp2.shape))
	print 's1,w1', s1,w1
	print 's2,w2', s2,w2

	tr1.setParams(allp1[s1, w1, :])
	tr2.setParams(allp2[s2, w2, :])

	plot_images(tr1, ptype='img')
	plt.suptitle('Data')
	ps.savefig()

	plot_images(tr1, ptype='mod+noise')
	plt.suptitle('With Parallax')
	ps.savefig()

	plot_images(tr2, ptype='mod+noise')
	plt.suptitle('Without Parallax')
	ps.savefig()

	plot_images(tr1, ptype='chi')
	plt.suptitle('With Parallax')
	ps.savefig()

	plot_images(tr2, ptype='chi')
	plt.suptitle('Without Parallax')
	ps.savefig()


	
	
	
if __name__ == '__main__':
	from optparse import OptionParser
	import sys
	parser = OptionParser(usage='%prog [options]')
	parser.add_option('-p', '--plots', dest='plots', action='store_true',
					  help='Make summary plots?')
	parser.add_option('-s', '--source', dest='sources', action='append', type=int,
					  help='Operate on the given (set of) sources')
	parser.add_option('-P', '-F', dest='freeze_parallax', action='store_true',
					  default=False, help='Freeze the parallax at zero?')
	parser.add_option('-b', dest='band', type=str, default='z', help='Band (default %default)')
	parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
					  help='Make more verbose')

	parser.add_option('--cat', dest='cat', action='store_true',
					  help='Use photoObj catalogs rather than images?')

	parser.add_option('--wise', dest='wise', action='store_true',
					  help='Include WISE data')
	parser.add_option('--ukidss', dest='ukidss', action='store_true',
					  help='Include UKIDSS data')
	parser.add_option('--twomass', dest='twomass', action='store_true',
					  help='Include 2MASS data')
	
	parser.add_option('--pixpsf', dest='pixpsf', action='store_true',
					  help='Use pixelized KL PSF model')
	
	parser.add_option('--c1', dest='compare1', action='store_true',
					  help='Compare parallax vs no-parallax results')
	
	opt,args = parser.parse_args()
	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	srcs = [getSource(i) for i in opt.sources]

	def get_tag(src, band, opt):
		stari = src.srci
		tag = '%i-%s' % (stari, band)
		if opt.pixpsf:
			tag += '-pix'
		if opt.cat:
			tag += '-cat'
		# if opt.wise:
		# tag += '-wise'
		if opt.freeze_parallax:
			tag += '-p0'
		return tag

	ppat = 'parallax-star%s-end.pickle'

	if opt.compare1:
		for src in srcs:
			opt.freeze_parallax = False
			tag1 = get_tag(src, opt.band, opt)
			opt.freeze_parallax = True
			tag2 = get_tag(src, opt.band, opt)
			compare1(src, tag1, tag2, ppat % tag1, ppat % tag2, opt, opt.band)
		sys.exit(0)

	plt.figure(figsize=(10,10))
	#plt.subplots_adjust(left=0.1, right=0.95, hspace=0.25, bottom=)
	
	if opt.plots:
		band = opt.band
		for src in srcs:
			tag = get_tag(src, band, opt)
			plot_chain(ppat % tag,
					   PlotSequence('chain-%s' % tag),
					   src.ra, src.dec, band, src.srci, tag, opt)
		sys.exit(0)

	for src in srcs:
		tag = get_tag(src, opt.band, opt)
		run_star(src, opt.band, tag, opt)
		

	
