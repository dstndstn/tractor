from __future__ import print_function

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import os
import logging
import tempfile
import tractor
import pyfits
import pylab as plt
import numpy as np
import sys
import re
from glob import glob

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import match_radec, cluster_radec
from astrometry.util.util import *
from astrometry.blind.plotstuff import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *
from astrometry.util.stages import *
from astrometry.util.ttime import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

import wise

def grenumerate(it):
	n = len(it)
	lastgrass = -1
	for i,x in enumerate(it):
		gr = int(80 * i / n)
		if gr != lastgrass:
			print('.', end='')
			lastgrass = gr
		yield i,x
	print()

rex = re.compile(r'-r(?P<ri>\d\d)-d(?P<di>\d\d)-')
def get_ridi(fn):
	m = rex.search(fn)
	if m is None:
		raise RuntimeError('regex on filename did not match')
	ri = int(m.group('ri'), 10)
	di = int(m.group('di'), 10)
	return ri,di

## HACK!
NDEC = 50
r0,r1 = 210.593,  219.132
d0,d1 =  51.1822,  54.1822
NRA = 90
dd = np.linspace(d0, d1, NDEC + 1)
rr = np.linspace(r0, r1, NRA  + 1)

def merge_results(S, basefn, outfn):
	W = []
	for band in [1,2]:
		bname = 'w%i' % band
		fns = glob('%s-r??-d??-w%i.fits' % (basefn, band))
		TT = []
		print('Reading', len(fns), 'W%i results' % band)
		for i,fn in grenumerate(fns):
			try:
				T = fits_table(fn)
			except:
				print('WARNING: failed to read', fn)
				continue
			
			assert(hasattr(T, 'inblock'))

			if hasattr(T, bname + '_var'):
				T.delete_column(bname + '_var')
				T.set(bname + '_ivar', np.zeros(len(T))-1.)
		
			TT.append(T)

		T = merge_tables(TT)
		print('Read total of', len(T))
	
		T.cut(np.flatnonzero(T.inblock))
		print('Cut to', len(T), 'in block')
	
		W.append(T)
	
	W1,W2 = W
	
	NS = len(S)
	SW = tabledata()
	# X = np.zeros(NS)
	# X[W1.row] = W1.ra
	# X[W2.row] = W2.ra
	# SW.ra2 = X
	# X = np.zeros(NS)
	# X[W1.row] = W1.dec
	# X[W2.row] = W2.dec
	# SW.dec2 = X
	SW.ra  = S.ra
	SW.dec = S.dec
	for b,W in [('w1',W1),('w2',W2)]:
		for k in ['prochi2','pronpix', 'profracflux','proflux','npix']:
			Y = W.get(k)
			X = np.zeros(NS, Y.dtype)
			X[W.row] = Y
			SW.set(b+'_'+k, X)

		for k in [b, b+'_ivar']:
			X = np.zeros(NS)
			X[W.row] = W.get(k)
			SW.set(k, X)

	SW.writeto(outfn)
	return SW


def cut_wise_cat():
	TT = []
	for s in [45,46]:
		
		fn = '/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part%02i-radec.fits' % s
		print('Reading', fn)
		T = fits_table(fn)
		I = np.flatnonzero((T.ra > r0) * (T.ra < r1) * (T.dec > d0) * (T.dec < d1))

		fn = '/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part%02i.fits' % s
		print('Reading', fn)
		T = fits_table(fn, rows=I, columns=['ra','dec','cntr',
											'w1mpro', 'w1sigmpro', 'w1mag', 'w1sigm', 'w2mpro', 'w2sigmpro', 'w2mag', 'w2sigm'])

		# fn = '/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part%02i.fits' % s
		# print('Reading', fn)
		# T = fits_table(fn, columns=['ra','dec','cntr',
		# 							'w1mpro', 'w1mag', 'w2mpro', 'w2mag'])
		# T.cut((T.ra > r0) * (T.ra < r1) * (T.dec > d0) * (T.dec < d1))
		print('Cut to', len(T))
		TT.append(T)
	W = merge_tables(TT)
	return W

ps = PlotSequence('wisecheck', suffixes=['png','eps','pdf'])
#ps = PlotSequence('wisecheck', suffixes=['png'])



def imgstats():
	# Some plots looking at image statistics.
	
	for band in [1,2]:

		#fns = glob('%s-r??-d??-w%i-imstats.fits' % (basefn, band))

		oldbasefn = 'ebossw3-v2'

		fns = []
		for ri in range(90):
			for di in range(50):
				fn = '%s-r%02i-d%02i-w%i-imstats.fits' % (basefn, ri, di, band)
				if os.path.exists(fn):
					fns.append(fn)
					continue
				fn = '%s-r%02i-d%02i-w%i-imstats.fits' % (oldbasefn, ri, di, band)
				if os.path.exists(fn):
					fns.append(fn)
					continue

		TT = []
	
		# sum by block
		bchisq, bnpix = [],[]
		# sum by scan+frame
		fchisq, fnpix = {},{}

		NR,ND = 90,50
		bchisq_map = np.zeros((ND,NR))
		bnpix_map  = np.zeros((ND,NR))

		print('Reading', len(fns), 'W%i image stats' % band)
	
		for i,fn in grenumerate(fns):
			T = fits_table(fn)
			#print fn, '-->', len(T)
			ri,di = get_ridi(fn)
	
			# Make maps of the sky estimates (by frame) in different blocks?
	
			bchisq.append(T.imchisq.sum())
			bnpix.append(T.imnpix.sum())

			bchisq_map[di,ri] += T.imchisq.sum()
			bnpix_map [di,ri] += T.imnpix.sum()

			for s,f,c,n in zip(T.scan_id, T.frame_num, T.imchisq, T.imnpix):
				key = '%s %i' % (s,f)
				if not key in fchisq:
					fchisq[key] = 0.
					fnpix[key] = 0.
				fchisq[key] += c
				fnpix[key] += n
	
			TT.append(T)
	
		bnpix = np.array(bnpix)
		bchisq = np.array(bchisq)
			
		T = merge_tables(TT)
		print('Read total of', len(T), 'stats')

		worst = np.argsort(-bchisq_map.ravel())
		wdi,wri = np.unravel_index(worst, bchisq_map.shape)

		print('<p><hr><p>')
		print('Worst chisq blocks:')
		print('<ul>')
		for r,d in zip(wri.ravel(),wdi.ravel())[:10]:
			print('<li>ri % 2i   di % 2i  : %g' % (r,d, bchisq_map[d,r]))
			url = 'http://skyserver.sdss3.org/public/en/tools/chart/navi.asp?ra=%f&dec=%f&scale=2' % (rr[r] + (rr[1]-rr[0])/2., dd[d] + (dd[1]-dd[0])/2.)
			print('  <a href="%s">nav</a>' % (url))
			K = np.flatnonzero((S.ra > rr[r]) * (S.ra < rr[r+1]) * (S.dec > dd[d]) * (S.dec < dd[d+1]))
			RCF = np.unique(zip(S.run[K], S.camcol[K], S.field[K]))
			for run,c,f in RCF:
				url = 'http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpegcodec.aspx?R=%i&C=%i&F=%i&Z=50' % (run,c,f)
				print('    <a href="%s">%i/%i/%i</a>' % (url, run,c,f))
			print('</li>')
		print('</ul>')
		print()

		plt.clf()
		plothist(S.ra, S.dec, 200)
		setRadecAxes(r0,r1,d0,d1)
		plt.title('SDSS sources')
		ps.savefig()

		K = np.flatnonzero(W.w1 > 1.)
		plt.clf()
		plothist(W.ra[K], W.dec[K], 200)
		setRadecAxes(r0,r1,d0,d1)
		plt.title('W1 measurements')
		ps.savefig()

		plt.clf()
		plt.imshow(bchisq_map, interpolation='nearest', origin='lower',
				   extent=[r0,r1,d0,d1])
		plt.title('Total chi-squared, W%i' % band)
		plt.colorbar()
		setRadecAxes(r0,r1,d0,d1)
		ps.savefig()

		plt.clf()
		plt.imshow(bchisq_map / np.maximum(bnpix_map, 1), interpolation='nearest', origin='lower',
				   extent=[r0,r1,d0,d1])
		plt.title('Total chi-squared / npix, W%i' % band)
		plt.colorbar()
		setRadecAxes(r0,r1,d0,d1)
		ps.savefig()

		plt.clf()
		plt.imshow(np.log10(bchisq_map), interpolation='nearest', origin='lower',
				   extent=[r0,r1,d0,d1])
		plt.title('Total log chi-squared, W%i' % band)
		plt.colorbar()
		setRadecAxes(r0,r1,d0,d1)
		ps.savefig()

		plt.clf()
		plt.imshow(np.log10(bchisq_map / np.maximum(bnpix_map, 1)),
				   interpolation='nearest', origin='lower',
				   extent=[r0,r1,d0,d1])
		plt.title('Total chi-squared / npix, W%i' % band)
		plt.colorbar()
		setRadecAxes(r0,r1,d0,d1)
		ps.savefig()

	
		# plt.clf()
		# plt.loglog(T.imnpix, np.maximum(0.1, T.imchisq), 'b.', alpha=0.1)
		# plt.xlabel('Image number of pixels')
		# plt.ylabel('Image chi-squared')
		# plt.title('W%i' % band)
		# ps.savefig()
	
		ok = np.flatnonzero(T.imnpix > 0)
		loghist(np.log10(T.imnpix[ok]), np.log10(np.maximum(0.1, T.imchisq[ok])), 200)
		ax = plt.axis()
		mn,mx = min(ax[0],ax[2]), max(ax[1],ax[3])
		plt.plot([mn,mx],[mn,mx],'b-')
		plt.axis(ax)
		plt.xlabel('log Image number of pixels')
		plt.ylabel('log Image chi-squared')
		plt.title('W%i' % band)
		ps.savefig()
		
		plt.clf()
		plt.loglog(bnpix, bchisq, 'b.', alpha=0.5)
		ax = plt.axis()
		mn,mx = min(ax[0],ax[2]), max(ax[1],ax[3])
		plt.plot([mn,mx],[mn,mx],'b-')
		plt.axis(ax)
		# plt.semilogy(bnpix, bchisq, 'b.', alpha=0.5)
		plt.xlabel('Block number of pixels')
		plt.ylabel('Block chi-squared')
		xt = [150,200,250]
		plt.xticks([x*1000 for x in xt], ['%i k' % x for x in xt])
		plt.xlim(bnpix.min() * 0.9, bnpix.max() * 1.1)
		print('npix range', bnpix.min(), bnpix.max())
		plt.title('W%i' % band)
		ps.savefig()
	
		k = fchisq.keys()
		k.sort()
		fc = []
		fn = []
		for kk in k:
			fc.append(fchisq[kk])
			fn.append(fnpix[kk])
		fc = np.array(fc)
		fn = np.array(fn)
		
		# plt.clf()
		# plt.loglog(np.maximum(1e2, fn), np.maximum(1e2, fc), 'b.', alpha=0.5)
		# ax = plt.axis()
		# mn,mx = min(ax[0],ax[2]), max(ax[1],ax[3])
		# plt.plot([mn,mx],[mn,mx],'b-')
		# plt.axis(ax)
		# plt.xlabel('Frame number of pixels')
		# plt.ylabel('Frame chi-squared')
		# plt.title('W%i' % band)
		# ps.savefig()
	
		loghist(np.log10(np.maximum(1e2, fn)), np.log10(np.maximum(1e2, fc)), 200,
				range=((2,6.5),(2,np.log10(fc.max()*1.2))))
		ax = plt.axis()
		print('ax:', ax)
		mn,mx = min(ax[0],ax[2]), max(ax[1],ax[3])
		plt.plot([mn,mx],[mn,mx],'b-')
		plt.axis(ax)
		plt.xlabel('log Frame number of pixels')
		plt.ylabel('log Frame chi-squared')
		plt.title('W%i' % band)
		ps.savefig()
	

def fluxtomag(nmgy):
	return -2.5 * (np.log10(np.maximum(1e-3, nmgy)) - 9.)

def dfluxtodmag(nmgy, dnmgy):
	return np.abs(-2.5 * (np.log10(1. + dnmgy / nmgy)))


#psfplots()
#print('<html>')


# duck-type command-line options
class myopts(object):
	pass

def qsocuts(SW):
	in1 = ( ((SW.gpsf - SW.ipsf) < 1.5) *
			(SW.optpsf > 17.) *
			(SW.optpsf < 22.) *
			((SW.optmod - SW.wise) > ((SW.gpsf - SW.ipsf) + 3)) *
			np.logical_or(SW.ispsf, (SW.optpsf - SW.optmod) < 0.1) )
	I = np.flatnonzero(in1)
	print('Selected', len(I))

	in2 = in1 * (SW.w1mag < 25.) * (SW.w2mag < 25.)
	I2 = np.flatnonzero(in2)
	print('With w1,w2 < 25:', len(I2))

	SW.w1rchi2 = SW.w1_prochi2 / SW.w1_pronpix

	in3 = in2 * (SW.w1rchi2 < 10.)
	I3 = np.flatnonzero(in3)
	print('And chi2/pix < 10:', len(I3))

	I = I2



	# Check against WISE catalog
	#wfn = 'w3-wise.fits'
	#WC = fits_table(wfn)

	# SDSS matched to WISE catalog
	swfn = 'eboss-w3-wise-cat-dr9.fits'
	#SWC = fits_table(swfn)


	S = fits_table('objs-eboss-w3-dr9.fits')
	print('Read', len(S), 'SDSS objects')
	wfn = 'w3-wise.fits'
	WC = fits_table(wfn)
	print('Read', len(WC), 'WISE catalog objects')

	R = 4./3600.
	I,J,d = match_radec(S.ra, S.dec, WC.ra, WC.dec, R, nearest=True)
	print(len(I), 'matches of SDSS to WISE catalog')
	SWC = S[I]
	for k in WC.columns():
		if k in ['ra','dec']:
			outkey = k+'_wise'
		else:
			outkey = k
		X = WC.get(k)
		SWC.set(outkey, X[J])
	SWC.writeto(swfn)

	SWC.gpsf = fluxtomag(SWC.psfflux[:,1])
	SWC.rpsf = fluxtomag(SWC.psfflux[:,2])
	SWC.ipsf = fluxtomag(SWC.psfflux[:,3])
	SWC.ispsf = (SWC.objc_type == 6)
	SWC.isgal = (SWC.objc_type == 3)

	SWC.optpsf = fluxtomag((SWC.psfflux[:,1] * 0.8 +
							SWC.psfflux[:,2] * 0.6 +
							SWC.psfflux[:,3] * 1.0) / 2.4)
	SWC.optmod = fluxtomag((SWC.modelflux[:,1] * 0.8 +
							SWC.modelflux[:,2] * 0.6 +
							SWC.modelflux[:,3] * 1.0) / 2.4)
	SWC.wise = fluxtomag((SWC.w1mpro * 1.0 +
						  SWC.w2mpro * 0.5) / 1.5)


	in1 = ( ((SWC.gpsf - SWC.ipsf) < 1.5) *
			(SWC.optpsf > 17.) *
			(SWC.optpsf < 22.) *
			((SWC.optpsf - SWC.wise) > ((SWC.gpsf - SWC.ipsf) + 3)) *
			np.logical_or(SWC.ispsf, (SWC.optpsf - SWC.optmod) < 0.1) )
	I = np.flatnonzero(in1)
	print('Selected', len(I), 'from WISE catalog')






	sys.exit(0)




	worstI = I[np.argsort(-SW.w1rchi2[I])]
	print('Worst:')

	mp = multiproc(1)

	for wi,i in enumerate(worstI):
		print('  %8.3f, %8.3f,  chi2/npix %g' % (SW.ra[i], SW.dec[i], SW.w1rchi2[i]))

		ra, dec = SW.ra[i], SW.dec[i]
		ri = int((ra  - r0) / (rr[1]-rr[0]))
		di = int((dec - d0) / (dd[1]-dd[0]))
		print('  ri %i, di %i' % (ri,di))
		
		if SW.w1rchi2[i] > 25:
			continue

		opt = myopts()
		wbasefn = 'eboss-w3-worst%04i-r%02i-d%02i' % (wi, ri,di)
		opt.picklepat = '%s-stage%%0i.pickle' % wbasefn
		opt.ps = wbasefn
		opt.minflux = None
		opt.bandnum = 1
		opt.osources = None
		opt.sources = 'objs-eboss-w3-dr9.fits'
		opt.ptsrc = False
		opt.pixpsf = False
		# opt.minsb = 0.05
		opt.minsb = 0.005
		opt.write = True
		opt.force = [205]
		opt.ri = ri
		opt.di = di

		import wise3
		import tractor
		pcat = []
		pcat.append(tractor.PointSource(RaDecPos(ra, dec), None))

		R = wise3.runtostage(205, opt, mp, rr[ri],rr[ri+1],dd[di],dd[di+1],
							 ttsuf='chi2/npix %g' % SW.w1rchi2[i],
							 pcat=pcat, addSky=True)


		### Try to use the stored solved values -- actually doesn't make things
		### much faster.
		# R = wise3.runtostage(103, opt, mp, rr[ri],rr[ri+1],dd[di],dd[di+1],
		# 					 ttsuf='chi2/npix %g' % SW.w1rchi2[i],
		# 					 pcat=pcat)
		# t = R['tractor']
		# T = fits_table('ebossw3-v4-r%02i-d%02i-w1.fits' % (ri, di))
		# assert(len(t.catalog) == len(T))
		# for i,src in enumerate(t.catalog):
		# 	print('Source', src)
		# 	assert(src.getPosition().ra  == T.ra [i])
		# 	assert(src.getPosition().dec == T.dec[i])
		# t.catalog.freezeParamsRecursive('*')
		# t.catalog.thawPathsTo('w1')
		# assert(len(t.catalog.getParams()) == len(T))
		# t.catalog.setParams(T.w1)
		# R2 = wise3.stage204(opt=opt, mp=mp, ri=opt.ri, di=opt.di, **R)
		# R2['ims1'] = R2['ims0']
		# ps = PlotSequence(wbasefn)
		# R3 = wise3.stage205(opt=opt, mp=mp, ri=opt.ri, di=opt.di, ps=ps, **R2)


	
	plt.clf()
	plt.hist(SW.w1mag, 100, range=(10,30), histtype='step', color='b', log=True)
	plt.hist(SW.w1mag[I], 100, range=(10,30), histtype='step', color='r', log=True)
	plt.xlabel('W1 mag')
	ylo,yhi = plt.ylim()
	plt.ylim(0.3, yhi)
	ps.savefig()

	plt.clf()
	plt.hist(SW.w2mag, 100, range=(10,30), histtype='step', color='b', log=True)
	plt.hist(SW.w2mag[I], 100, range=(10,30), histtype='step', color='r', log=True)
	plt.xlabel('W2 mag')
	ylo,yhi = plt.ylim()
	plt.ylim(0.3, yhi)
	ps.savefig()
	
	plt.clf()
	plt.hist(np.log10(SW.w1_prochi2 / SW.w1_pronpix), 100, range=(0,3),
			 log=True, histtype='step', color='b')
	plt.hist(np.log10(SW.w1_prochi2[I] / SW.w1_pronpix[I]), 100, range=(0,3),
			 log=True, histtype='step', color='r')
	plt.xlabel('log chi2/npix')
	ylo,yhi = plt.ylim()
	plt.ylim(0.3, yhi)
	ps.savefig()





basefn = 'v4/ebossw3-v4'
fn = 'eboss-w3-v4-wise-dr9.fits'

# basefn = 'ebossw3-v5'
# fn = 'eboss-w3-v5-wise-dr9.fits'



matchedfn = 'sw.fits'
# if os.path.exists(matchedfn):
# 	SW = fits_table(matchedfn)
# 	qsocuts(SW)
# 	sys.exit(0)




def coadd_plots():

	basefn = 'ebossw3-tst'

	#basefn = 'v4/eboss-w3-v4'


	ra,dec = 215.095, 52.72
	ri = np.flatnonzero(rr < ra)[-1]
	di = np.flatnonzero(dd < dec)[-1]

	print('ri', ri, 'di', di)
	print('ra', rr[ri],rr[ri+1], 'dec', dd[di], dd[di+1])

	#ri = 25
	#di = 0

	pfn = '%s-r%02i-d%02i-w1.pickle' % (basefn, ri, di)
	P = unpickle_from_file(pfn)
	print('Got:', P.keys())
	
	#r0,r1,d0,d1 = P['rd']
	res1 = P['res1']
	S = P['S']
	cat = P['cat']
	for tim,mod,roi in res1:
		tim.inverr = np.sqrt(tim.invvar)

	for src in cat:
		print('Source', src)
	
	dra  = rr[1]-rr[0]
	ddec = dd[1]-dd[0]
	cosdec = np.cos(np.deg2rad((d0 + d1)/2.))
	
	rl,rh,dl,dh = rr[ri], rr[ri+1], dd[di], dd[di+1]
	ra =  (rl+rh)/2.
	dec = (dl+dh)/2.

	print('RA range for coadd block:', rl, rh)
	print('Dec', dl,dh)
	
	class TractorToFitsWcs(object):
		'''
		The Tractor WCS code operates in pixel coords; wrap to FITS coords
		'''
		def __init__(self, wcs):
			self.wcs = wcs
		def radec2pixelxy(self, ra, dec):
			ra  = np.atleast_1d(ra)
			dec = np.atleast_1d(dec)
			x = np.zeros_like(ra)
			y = np.zeros_like(dec)
			for i,(r,d) in enumerate(zip(ra,dec)):
				xi,yi = self.wcs.positionToPixel(RaDecPos(r,d))
				x[i] = xi
				y[i] = yi
			#x,y = self.wcs.positionToPixel(RaDecPos(ra,dec))
			return True, x+1., y+1.
	
		def pixelxy2radec(self, x, y):
			x = np.atleast_1d(x)
			y = np.atleast_1d(y)
			ra  = np.zeros_like(x)
			dec = np.zeros_like(y)
			for i,(xi,yi) in enumerate(zip(x,y)):
				pos = self.wcs.pixelToPosition(xi-1., yi-1.)
				ra[i] = pos.ra
				dec[i] = pos.dec
			return ra,dec
			#return pos.ra, pos.dec


	# if True:
	SW, SH = int(dra * cosdec * 3600. / 2.75), int(ddec * 3600. / 2.75)
	print('Coadd size', SW, SH)
	cowcs = anwcs_create_box(ra, dec, dra * cosdec, SW, SH)
	cowcs = anwcs_get_sip(cowcs)

	xx,yy = [],[]
	for src in cat:
		pos = src.getPosition()
		ok,x,y = cowcs.radec2pixelxy(pos.ra, pos.dec)
		xx.append(x - 1.)
		yy.append(y - 1.)

	wfn = 'w3-wise.fits'
	WC = fits_table(wfn)
	catx,caty = [],[]
	for r,d in zip(WC.ra, WC.dec):
		ok,x,y = cowcs.radec2pixelxy(r,d)
		catx.append(x-1)
		caty.append(y-1)


	ima = dict(interpolation='nearest', origin='lower',
			   vmin=-10, vmax=50)
	wima = ima

	pfn = 'coadd-wise.pickle'
	if os.path.exists(pfn):
		conn,connw = unpickle_from_file(pfn)
	else:
		conn  = np.zeros((SH,SW))
		connw = np.zeros((SH,SW))

		CO1 = []
		for i,(tim,mod,roi) in enumerate(res1):
			print('Resampling WISE img', i)
			# plt.clf()
			# plt.imshow(tim.getImage(), **ima)
			# plt.gray()
			# ax = plt.axis()
			# xi,yi = [],[]
			# for src in cat:
			# 	pos = src.getPosition()
			# 	x,y = tim.getWcs().positionToPixel(pos)
			# 	xi.append(x)
			# 	yi.append(y)
			# plt.plot(xi,yi, 'r.')
			# plt.axis(ax)
			# ps.savefig()
	
			wcs = TractorToFitsWcs(tim.getWcs())
			wcs.imageh, wcs.imagew = tim.shape
		
	   		# x0,y0 = tim.getWcs().getX0Y0()
			# H,W = tim.shape
			# wcs = wcs.get_subimage(x0, x0+W-1, y0, y0+H-1)
		
			Yo,Xo,Yi,Xi,ims = resample_with_wcs(cowcs, wcs, [], 0, spline=False)
			if Xi is None:
				continue
			sky = tim.getSky().val
			ie = tim.getInvError()
			pix = tim.getImage()
			conn  [Yo,Xo] += ((pix[Yi,Xi] - sky) * (ie[Yi,Xi] > 0))
			connw [Yo,Xo] += (1 * (ie[Yi,Xi] > 0))
		
			co1  = np.zeros((SH,SW))
			co1[Yo,Xo] = (pix[Yi,Xi] - sky)
			CO1.append(co1)
	
			# plt.clf()
			# plt.imshow(co1, **ima)
			# plt.xticks([]), plt.yticks([])
			# plt.gray()
			# ax = plt.axis()
			# plt.plot(xx, yy, 'r.')
			# plt.axis(ax)
			# ps.savefig()


		pickle_to_file((conn,connw), pfn)

	
	plt.clf()
	plt.imshow(conn / np.maximum(1, connw), **ima)
	plt.xticks([]), plt.yticks([])
	plt.gray()
	ps.savefig()

	ax = plt.axis()
	plt.plot(xx, yy, 'r.')

	plt.plot(catx, caty, 'gx')

	plt.axis(ax)
	ps.savefig()

	
	# N = len(CO1)
	# C = int(np.ceil(np.sqrt(N)))
	# R = int(np.ceil(N / float(C)))
	# plt.clf()
	# for i,co1 in enumerate(CO1):
	# 	plt.subplot(R,C, i+1)
	# 	plt.imshow(co1, **ima)
	# 	plt.xticks([]), plt.yticks([])
	# 	plt.gray()
	# ps.savefig()


	# for i,co1 in enumerate(CO1):
	# 	plt.clf()
	# 	plt.imshow(co1, **ima)
	# 	plt.xticks([]), plt.yticks([])
	# 	plt.gray()
	# 
	# 	ax = plt.axis()
	# 	plt.plot(xx, yy, 'r.')
	# 	plt.axis(ax)
	# 	ps.savefig()


	pix = 0.396
	SW, SH = int(dra * cosdec * 3600. / pix), int(ddec * 3600. / pix)
	print('SDSS coadd size', SW, SH)
	cowcs = anwcs_create_box(ra, dec, dra * cosdec, SW, SH)
	cowcs = anwcs_get_sip(cowcs)

	pfn = 'coadd-sdss.pickle'
	if os.path.exists(pfn):
		conn,connw,zr = unpickle_from_file(pfn)
	else:
		conn  = np.zeros((SH,SW))
		connw = np.zeros((SH,SW))

		zr = None
		CO1 = []
		for r,c,f in np.unique(zip(S.run, S.camcol, S.field)):
			print('Run, camcol, field', r,c,f)
			tim,tinf = get_tractor_image_dr9(r, c, f, 'r', psf='dg')
			print('got', tim)
			if zr is None:
				zr = tim.zr
			wcs = TractorToFitsWcs(tim.getWcs())
			wcs.imageh, wcs.imagew = tim.shape
			Yo,Xo,Yi,Xi,ims = resample_with_wcs(cowcs, wcs, [], 0, spline=False)
			if Xi is None:
				continue
			pix = tim.getImage()
			conn  [Yo,Xo] += pix[Yi,Xi]
			connw [Yo,Xo] += 1
		
			co1  = np.zeros((SH,SW))
			co1[Yo,Xo] = pix[Yi,Xi]
			CO1.append(co1)

		pickle_to_file((conn,connw,zr), pfn)

	plt.clf()
	ima = dict(interpolation='nearest', origin='lower',
			   vmin=zr[0], vmax=zr[1])
	plt.imshow(conn / np.maximum(1, connw), **ima)
	plt.xticks([]), plt.yticks([])
	plt.gray()
	ps.savefig()
	
	# N = len(CO1)
	# C = int(np.ceil(np.sqrt(N)))
	# R = int(np.ceil(N / float(C)))
	# plt.clf()
	# for i,co1 in enumerate(CO1):
	# 	plt.subplot(R,C, i+1)
	# 	plt.imshow(co1, **ima)
	# 	plt.xticks([]), plt.yticks([])
	# 	plt.gray()
	# ps.savefig()


	# label brightest sources
	I = np.argsort(-S.psfflux[:,2])
	for j,i in enumerate(I[:20]):
		r,d = S.ra[i], S.dec[i]
		ok,x,y = cowcs.radec2pixelxy(r,d)
		plt.text(x, y, '%i' % j, color='r')
	ps.savefig()


	if False:
		# Number 7 - nice galaxy
		# Numbers 4,16,19 - blob
		jj = [4,16,19]
		srr = [S.ra [I[j]]  for j in jj]
		sdd = [S.dec[I[j]]  for j in jj]
		print('rr', srr)
		print('dd', sdd)
	
		sz = int(0.8 * (max(sdd) - min(sdd)) / (0.396/3600.))
		print('sz', sz)
		#ra = np.mean(srr)
		#dec = np.mean(sdd)
		ra  = (max(srr) + min(srr)) / 2.
		dec = (max(sdd) + min(sdd)) / 2.
	
		print('RA,Dec', ra,dec)
	
		#ii = I[4]
		#ra,dec = S.ra[ii],S.dec[ii]
		#sz = 15

	sz = int(0.01 * (3600./0.396))
	print('sz', sz)

	for r,c,f in np.unique(zip(S.run, S.camcol, S.field)):
		print('Run, camcol, field', r,c,f)

	r,c,f = np.unique(zip(S.run, S.camcol, S.field)) [1]
	tim,tinf = get_tractor_image_dr9(r, c, f, 'r', #psf='dg',
									 roiradecsize=(ra,dec,sz))

	roi = tinf['roi']
	ssrcs = get_tractor_sources_dr9(r, c, f, 'r', roi=roi, nanomaggies=True)

	tr = Tractor([tim], ssrcs)
	mod = tr.getModelImage(0)

	xx,yy = [],[]
	for src in ssrcs:
		x,y = tim.getWcs().positionToPixel(src.getPosition())
		xx.append(x)
		yy.append(y)

	print('Initial models:')
	for src in tr.catalog:
		print('  ', src)
	for src in tr.catalog:
		print('  RA,Dec %.8f, %.8f' % (src.getPosition().ra, src.getPosition().dec))

		x,y = tim.getWcs().positionToPixel(src.getPosition(), color=0.5)
		print('  x,y %.6f, %.6f  w/color' % (x,y))
		x,y = tim.getWcs().positionToPixel(src.getPosition(), color=0.)
		print('  x,y %.6f, %.6f' % (x,y))

	plt.figure(figsize=(2,2))
	plt.clf()
	plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

	# plt.subplot(2,2,1)
	plt.imshow(tim.getImage(), **ima)
	plt.xticks([]), plt.yticks([])
	ps.savefig()

	# plt.subplot(2,2,2)
	plt.imshow(mod, **ima)
	ps.savefig()

	# plt.subplot(2,2,3)
	plt.imshow((tim.getImage() - mod) * tim.getInvError(), interpolation='nearest',
	 		   origin='lower', vmin=-5, vmax=5)
	ax = plt.axis()
	plt.plot(xx, yy, 'rx')
	plt.axis(ax)
	ps.savefig()

	plt.clim(vmin=-3, vmax=3)
	ps.savefig()

	tr.freezeAllRecursive()
	tr.thawPathsTo('r')
	tr.thawPathsTo('ra')
	tr.thawPathsTo('dec')
	while True:
		dlnp,X,a = tr.optimize()
		print('dlnp', dlnp)
		if dlnp < 1.:
			break

	print('Final models:')
	for src in tr.catalog:
		print('  ', src)
	for src in tr.catalog:
		print('  RA,Dec %.8f, %.8f' % (src.getPosition().ra, src.getPosition().dec))

		x,y = tim.getWcs().positionToPixel(src.getPosition(), color=0.5)
		print('  x,y %.6f, %.6f  w/color' % (x,y))
		x,y = tim.getWcs().positionToPixel(src.getPosition(), color=0.)
		print('  x,y %.6f, %.6f' % (x,y))

	mod = tr.getModelImage(0)

	#plt.clf()
	#plt.subplot(2,2,1)
	#plt.imshow(tim.getImage(), **ima)
	#plt.subplot(2,2,2)

	plt.clf()
	plt.imshow(mod, **ima)
	plt.xticks([]), plt.yticks([])
	ps.savefig()
	
	#plt.subplot(2,2,3)
	#plt.imshow((tim.getImage() - mod) * tim.getInvError(), interpolation='nearest',
	#		   origin='lower', vmin=-5, vmax=5)
	#ps.savefig()

	plt.imshow((tim.getImage() - mod) * tim.getInvError(), interpolation='nearest',
	 		   origin='lower', vmin=-5, vmax=5)
	ax = plt.axis()
	plt.plot(xx, yy, 'rx')
	plt.axis(ax)
	ps.savefig()

	plt.clim(vmin=-3, vmax=3)
	ps.savefig()




	SH,SW = tim.shape

	cowcs = TractorToFitsWcs(tim.getWcs())
	cowcs.imageh,cowcs.imagew = tim.shape

	pfn = 'coadd-wise-2.pickle'
	if os.path.exists(pfn):
		conn,connw, comonn, comonnw, CO1 = unpickle_from_file(pfn)
	else:
		conn  = np.zeros((SH,SW))
		connw = np.zeros((SH,SW))
		comonn  = np.zeros((SH,SW))
		comonnw = np.zeros((SH,SW))

		CO1 = []
		for i,(tim,mod,roi) in enumerate(res1):
			wcs = TractorToFitsWcs(tim.getWcs())
			wcs.imageh, wcs.imagew = tim.shape
			Yo,Xo,Yi,Xi,ims = resample_with_wcs(cowcs, wcs, [], 0, spline=False)
			if Xi is None:
				continue
			sky = tim.getSky().val
			ie = tim.getInvError()
			pix = tim.getImage()
			conn  [Yo,Xo] += ((pix[Yi,Xi] - sky) * (ie[Yi,Xi] > 0))
			connw [Yo,Xo] += (1 * (ie[Yi,Xi] > 0))

			comonn  [Yo,Xo] += ((mod[Yi,Xi] - sky) * (ie[Yi,Xi] > 0))
			comonnw [Yo,Xo] += (1 * (ie[Yi,Xi] > 0))

			co1  = np.zeros((SH,SW))
			co1w  = np.zeros((SH,SW))
			mo1  = np.zeros((SH,SW))
			mo1w  = np.zeros((SH,SW))
			co1 [Yo,Xo] += ((pix[Yi,Xi] - sky) * (ie[Yi,Xi] > 0))
			co1w[Yo,Xo] += (1 * (ie[Yi,Xi] > 0))
			mo1 [Yo,Xo] += ((mod[Yi,Xi] - sky) * (ie[Yi,Xi] > 0))
			mo1w[Yo,Xo] += (1 * (ie[Yi,Xi] > 0))
			CO1.append((co1,co1w, mo1,mo1w))

		pickle_to_file((conn,connw, comonn, comonnw, CO1), pfn)


	wima = dict(interpolation='nearest', origin='lower',
			 #  vmin=-10, vmax=50)
				vmin=0, vmax=20)

	print(len(CO1), 'WISE images overlap')

	for p in [10, 25, 50, 75, 90, 95, 99]:
		print('Percentiles: ', p, np.percentile(conn/np.maximum(1,connw), p))

	plt.clf()
	plt.imshow(conn / np.maximum(1, connw), **wima)
	plt.xticks([]), plt.yticks([])
	plt.gray()
	ps.savefig()

	plt.clf()
	plt.imshow(comonn / np.maximum(1, comonnw), **wima)
	plt.xticks([]), plt.yticks([])
	plt.gray()
	ps.savefig()

	#for (co1, co1w, mo1, mo1w) in CO1[:10]:
	(co1, co1w, mo1, mo1w) = CO1[2]
	if True:
		plt.clf()
		plt.imshow(co1, **wima)
		plt.xticks([]), plt.yticks([])
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(mo1, **wima)
		plt.xticks([]), plt.yticks([])
		plt.gray()
		ps.savefig()





#coadd_plots()
#sys.exit(0)




S = fits_table('objs-eboss-w3-dr9.fits')
print('Read', len(S))

if not os.path.exists(fn):
	W = merge_results(S, basefn, fn)
else:
	print('Reading existing', fn)
	W = fits_table(fn)


#imgstats()
#print('</html>')
#sys.exit(0)

# from x import *
# plt.clf()
# plt.plot(var1, var2, 'r.')
# ps.savefig()

# T = fits_table('ebossw3-v2-w1-forced.fits')
# plt.clf()
# #plt.plot(T.w1, T.w1_var, 'r.')
# plt.loglog(T.w1, 1./np.sqrt(T.w1_var), 'r.')
# plt.xlabel('w1 flux')
# plt.ylabel('w1 flux sigma')
# ps.savefig()
# sys.exit(0)

W.w1mag = fluxtomag(W.w1)
W.w2mag = fluxtomag(W.w2)
print('w1mag', W.w1mag.min(), W.w1mag.max())
print('w2mag', W.w2mag.min(), W.w2mag.max())

W.w1magerr = dfluxtodmag(W.w1, 1./np.sqrt(W.w1_ivar))
W.w2magerr = dfluxtodmag(W.w2, 1./np.sqrt(W.w2_ivar))


S.gpsf = fluxtomag(S.psfflux[:,1])
S.rpsf = fluxtomag(S.psfflux[:,2])
S.ipsf = fluxtomag(S.psfflux[:,3])

S.gmod = fluxtomag(S.modelflux[:,1])
S.rmod = fluxtomag(S.modelflux[:,2])
S.imod = fluxtomag(S.modelflux[:,3])

S.ispsf = (S.objc_type == 6)
S.isgal = (S.objc_type == 3)

print('g', S.gpsf.min(), S.gpsf.max())
print('r', S.rpsf.min(), S.rpsf.max())
print('i', S.ipsf.min(), S.ipsf.max())

wfn = 'w3-wise.fits'
if not os.path.exists(wfn):
	WC = cut_wise_cat()
	WC.writeto(wfn)
else:
	print('Reading', wfn)
	WC = fits_table(wfn)






# Some summary plots


I = np.flatnonzero((W.ra > r0) * (W.ra < r1) * (W.dec > d0) * (W.dec < d1))
print(len(I), 'of', len(W), 'W are in the RA,Dec box')
W.cut(I)

# H,xe,ye = np.histogram2d(W.ra, W.dec, 100, range=((r0,r1),(d0,d1)))
# H = H.ravel()
# print(sum(H>0), 'of', len(H), 'bins of W are non-zero')
# frac = float(sum(H>0)) / len(H)
# 
# H,xe,ye = np.histogram2d(WC.ra, WC.dec, 100, range=((r0,r1),(d0,d1)))
# H = H.ravel()
# print(sum(H>0), 'of', len(H), 'bins of WC are non-zero')
# frac2 = float(sum(H>0)) / len(H)
frac = 1.

# I = np.flatnonzero((WC.ra > r0) * (WC.ra < r1) * (WC.dec > d0) * (WC.dec < d1))
# print(len(I), 'of', len(WC), 'WC are in the RA,Dec box')

# plt.clf()
# plothist(W.ra, W.dec, 100, range=((r0,r1),(d0,d1)))
# ps.savefig()
# 
# plt.clf()
# plothist(WC.ra, WC.dec, 100, range=((r0,r1),(d0,d1)))
# ps.savefig()



#xlo,xhi = 7,26
xlo,xhi = 10,25



R = 4.0
I,J,d = match_radec(W.ra, W.dec, WC.ra, WC.dec, R, nearest=True)

loghist(W.w1mag[I], WC.w1mpro[J], 200, range=((xlo,xhi),(xlo,xhi)))
plt.plot([xlo,xhi],[xlo,xhi], '-', color='1')
plt.xlabel('Tractor w1mag')
plt.ylabel('Catalog w1mpro')
plt.axis([xlo,xhi,xlo,xhi])
ps.savefig()


lo,hi = 0,0.25
loghist(W.w1magerr[I], WC.w1sigmpro[J], 200, range=((lo,hi),(lo,hi)))
plt.plot([lo,hi],[lo,hi], '-', color='1')
plt.xlabel('Tractor w1mag err')
plt.ylabel('Catalog w1sigmpro')
plt.axis([lo,hi,lo,hi])
ps.savefig()
print('error hist')

m = WC[J].w1mpro
K = np.flatnonzero((m > 16) * (m < 17))

lo,hi = 0,0.25
loghist(W.w1magerr[I[K]], WC.w1sigmpro[J[K]], 200, range=((lo,hi),(lo,hi)))
for d in [0., 0.01, 0.02, 0.03]:
	plt.plot([lo,hi],[lo+d,hi+d], '-', color='1')
plt.xlabel('Tractor w1mag err')
plt.ylabel('Catalog w1sigmpro')
plt.axis([lo,hi,lo,hi])
plt.title('Mag 16-17')
ps.savefig()
print('error hist')


K = np.flatnonzero((m > 16.5) * (m < 17.5))

lo,hi = 0,0.25
loghist(W.w1magerr[I[K]], WC.w1sigmpro[J[K]], 200, range=((lo,hi),(lo,hi)))
for d in [0., 0.01, 0.02, 0.03]:
	plt.plot([lo,hi],[lo+d,hi+d], '-', color='1')
plt.xlabel('Tractor w1mag err')
plt.ylabel('Catalog w1sigmpro')
plt.axis([lo,hi,lo,hi])
plt.title('Mag 16.5-17.5')
ps.savefig()
print('error hist')


#sys.exit(0)



plt.clf()

#catsty = dict(linestyle='dashed', lw=2)
#catsty = dict(lw=2, alpha=0.5)

#catsty1 = dict(color='c', lw=3, alpha=0.3)
#catsty2 = dict(color='m', lw=3, alpha=0.3)

catsty1 = dict(color=(0.8,0.8,1.0), lw=3)
catsty2 = dict(color=(1.0,0.8,0.8), lw=3)

tsty1 = dict(color='b', lw=1, alpha=1.)
tsty2 = dict(color='r', lw=1, alpha=1.)



plt.clf()
plt.hist(WC.w1sigmpro[J[K]], 100, range=(lo,hi), histtype='step', **catsty1)
plt.hist(W.w1magerr[I[K]],   100, range=(lo,hi), histtype='step', **tsty1)
plt.xlabel('w1 mag errors')
ps.savefig()




n1,b,p1 = plt.hist(W.w1mag,   100, log=True, histtype='step', range=(xlo,xhi), **tsty1)
n2,b,p2 = plt.hist(W.w2mag,   100, log=True, histtype='step', range=(xlo,xhi), **tsty2)
n3,b,p3 = plt.hist(WC.w1mpro, 100, log=True, histtype='step', range=(xlo,xhi), **catsty1)
n4,b,p4 = plt.hist(WC.w2mpro, 100, log=True, histtype='step', range=(xlo,xhi), **catsty2)

# Just for the legend...
bx = (b[1:] + b[:-1]) / 2.
p1 = plt.plot(bx[:1],n1[:1], '-', **tsty1)
p2 = plt.plot(bx[:1],n1[:1], '-', **tsty2)
p3 = plt.plot(bx[:1],n1[:1], '-', **catsty1)
p4 = plt.plot(bx[:1],n1[:1], '-', **catsty2)

plt.xlabel('WISE mag')
plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'))
plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
plt.ylabel('Number of sources')
plt.title('W3 area: WISE forced photometry mag distribution')
plt.xlim(xlo, xhi)
ps.savefig()

bx = (b[1:] + b[:-1]) / 2.

def bxy(b, y):
	return np.repeat(b, 2)[1:-1], np.repeat(y, 2)

if False:
	plt.clf()
	bx,by = bxy(b, n1)
	p1 = plt.semilogy(bx, by / frac, 'b-')
	bx,by = bxy(b, n1)
	p1 = plt.semilogy(bx, by, 'b-')
	bx,by = bxy(b, n2)
	p2 = plt.semilogy(bx, by / frac, 'r-')
	bx,by = bxy(b, n3)
	p3 = plt.semilogy(bx, by, 'c-', **catsty)
	bx,by = bxy(b, n4)
	p4 = plt.semilogy(bx, by, 'm-', **catsty)
	plt.xlabel('WISE mag')
	plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'))
	plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
	plt.xlim(xlo, xhi)
	plt.ylabel('Number of sources')
	plt.title('W3 area: WISE forced photometry mag distribution')
	ps.savefig()
	

	plt.clf()
	bx,by = bxy(b, n1)
	p1 = plt.semilogy(bx, by / frac, 'b-')
	#bx,by = bxy(b, n2)
	#p2 = plt.semilogy(bx, by / frac, 'r-')
	bx,by = bxy(b, n3)
	p3 = plt.semilogy(bx, by, 'c-', **catsty)
	#bx,by = bxy(b, n4)
	#p4 = plt.semilogy(bx, by, 'm-', **catsty)
	#plt.xlabel('WISE mag')
	#plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'))
	plt.legend((p1,p3),('W1 (Tractor)', 'W1 (WISE cat)'), loc='lower right')
	plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
	plt.xlim(xlo, xhi)
	plt.ylabel('Number of sources')
	plt.title('W3 area: WISE forced photometry mag distribution')
	ps.savefig()
	
	#plt.xlim(14,15)
	#ps.savefig()
	



I = np.argsort(WC.w1mpro)
m1 = WC.w1mpro[I[10000]]
print('10k-th w1mpro:', m1)
I = np.argsort(W.w1mag)
m2 = W.w1mag[I[10000]]
print('10k-th w1mag:', m2)
dm = m1-m2

plt.clf()
n1,b,p1 = plt.hist(W.w1mag + dm, 100, log=True, histtype='step', range=(xlo,xhi), **tsty1)
n2,b,p2 = plt.hist(W.w2mag + dm, 100, log=True, histtype='step', range=(xlo,xhi), **tsty2)
n3,b,p3 = plt.hist(WC.w1mpro,    100, log=True, histtype='step', range=(xlo,xhi), **catsty1)
n4,b,p4 = plt.hist(WC.w2mpro,    100, log=True, histtype='step', range=(xlo,xhi), **catsty2)

n3 = np.maximum(n3, 0.1)
n4 = np.maximum(n4, 0.1)

plt.clf()
bx,by = bxy(b, n1)
p1 = plt.semilogy(bx, by, '-', **tsty1)
bx,by = bxy(b, n2)
p2 = plt.semilogy(bx, by, '-', **tsty2)
bx,by = bxy(b, n3)
p3 = plt.semilogy(bx, by, '-', **catsty1)
bx,by = bxy(b, n4)
p4 = plt.semilogy(bx, by, '-', **catsty2)
plt.xlabel('WISE mag')
plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'), loc='lower right')
plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
plt.xlim(xlo, xhi)
plt.ylabel('Number of sources')
plt.title('W3 area: WISE forced photometry mag distribution')
ps.savefig()



plt.clf()
bx,by = bxy(b, n3)
p3 = plt.semilogy(bx, by, '-', **catsty1)
bx,by = bxy(b, n1)
p1 = plt.semilogy(bx, by, '-', **tsty1)
plt.xlabel('WISE mag')
plt.legend((p1,p3),('W1 (Tractor)', 'W1 (WISE cat)'), loc='lower right')
plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
plt.xlim(xlo, xhi)
plt.ylabel('Number of sources')
plt.title('W3 area: WISE forced photometry mag distribution')
ps.savefig()

if False:
	## FINER BINS
	bins = 200
	
	plt.clf()
	n1,b,p1 = plt.hist(W.w1mag +dm,   bins, log=True, histtype='step', color='b', range=(xlo,xhi), **tsty)
	n2,b,p2 = plt.hist(W.w2mag +dm,   bins, log=True, histtype='step', color='r', range=(xlo,xhi), **tsty)
	n3,b,p3 = plt.hist(WC.w1mpro, bins, log=True, histtype='step', color='c', range=(xlo,xhi), **catsty)
	n4,b,p4 = plt.hist(WC.w2mpro, bins, log=True, histtype='step', color='m', range=(xlo,xhi), **catsty)
	
	n3 = np.maximum(n3, 0.1)
	n4 = np.maximum(n4, 0.1)
	
	def bxy(b, y):
		return (b[:-1]+b[1:])/2., y
	
	plt.clf()
	bx,by = bxy(b, n1)
	p1 = plt.semilogy(bx, by, 'b-')
	bx,by = bxy(b, n2)
	p2 = plt.semilogy(bx, by, 'r-')
	bx,by = bxy(b, n3)
	p3 = plt.semilogy(bx, by, 'c-', **catsty)
	bx,by = bxy(b, n4)
	p4 = plt.semilogy(bx, by, 'm-', **catsty)
	plt.xlabel('WISE mag')
	plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'), loc='lower right')
	#plt.legend((p1,p3),('W1 (Tractor)', 'W1 (WISE cat)'), loc='lower right')
	plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
	plt.xlim(xlo, xhi)
	plt.ylabel('Number of sources')
	plt.title('W3 area: WISE forced photometry mag distribution')
	ps.savefig()
	
	
	
	plt.clf()
	bx,by = bxy(b, n1)
	p1 = plt.semilogy(bx, by, 'b-')
	#bx,by = bxy(b, n2)
	#p2 = plt.semilogy(bx, by, 'r-')
	bx,by = bxy(b, n3)
	p3 = plt.semilogy(bx, by, 'c-', **catsty)
	#bx,by = bxy(b, n4)
	#p4 = plt.semilogy(bx, by, 'm-', **catsty)
	plt.xlabel('WISE mag')
	#plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'))
	plt.legend((p1,p3),('W1 (Tractor)', 'W1 (WISE cat)'), loc='lower right')
	plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
	plt.xlim(xlo, xhi)
	plt.ylabel('Number of sources')
	plt.title('W3 area: WISE forced photometry mag distribution')
	ps.savefig()






sys.exit(0)









swfn = 'eboss-w3-wise-cat-dr9.fits'
if not os.path.exists(swfn):
	R = 4./3600.
	I,J,d = match_radec(S.ra, S.dec, WC.ra, WC.dec, R, nearest=True)
	print(len(I), 'matches of SDSS to WISE catalog')
	SWC = tabledata()
	SWC.ra = S.ra
	SWC.dec = S.dec
	for k in WC.columns():
		if k in ['ra','dec']:
			outkey = k+'_wise'
		else:
			outkey = k
		X = WC.get(k)
		Y = np.zeros(len(S), X.dtype)
		Y[I] = X[J]
		SWC.set(outkey, Y)
	SWC.writeto(swfn)
else:
	SWC = fits_table(swfn)



def psfplots():
	tpsf = wise.get_psf_model(1, pixpsf=True)
	
	psfp = tpsf.getPointSourcePatch(0, 0)
	psf = psfp.patch
	
	psf /= psf.sum()
	
	plt.clf()
	plt.imshow(np.log10(np.maximum(1e-5, psf)), interpolation='nearest', origin='lower')
	plt.colorbar()
	ps.savefig()
	
	h,w = psf.shape
	cx,cy = w/2, h/2
	
	X,Y = np.meshgrid(np.arange(w), np.arange(h))
	R = np.sqrt((X - cx)**2 + (Y - cy)**2)
	plt.clf()
	plt.semilogy(R.ravel(), psf.ravel(), 'b.')
	plt.xlabel('Radius (pixels)')
	plt.ylabel('PSF value')
	plt.ylim(1e-8, 1.)
	ps.savefig()
	
	plt.clf()
	plt.loglog(R.ravel(), psf.ravel(), 'b.')
	plt.xlabel('Radius (pixels)')
	plt.ylabel('PSF value')
	plt.ylim(1e-8, 1.)
	ps.savefig()
	
	print('PSF norm:', np.sqrt(np.sum(np.maximum(0, psf)**2)))
	print('PSF max:', psf.max())


def tractor_vs_cat():
	# Tractor/SDSS vs WISE/SDSS comparisons

	R = 4./3600.
	# NOTE, this matches all W entries (ie, same RA,Dec as SDSS), not just
	# the ones with photometry.
	I,J,d = match_radec(W.ra, W.dec, WC.ra, WC.dec, R, nearest=True)
	print(len(I), 'matches to WISE catalog')

	I2 = np.flatnonzero(np.logical_or(W.w1 > 0, W.w2 > 0))
	
	T = S[I2]
	T.add_columns_from(W[I2])
	
	C = WC[J]
	C.add_columns_from(S[I])
	
	print(len(T), 'Tractor-SDSS matches')
	print(len(C), 'WISE-SDSS matches')
	
	I = np.flatnonzero((T.gpsf != T.ipsf) * (T.w1mag < 25))

	wcuts = [17,18,19,20,21]
	
	plothist(T.gpsf[I] - T.ipsf[I], T.imod[I] - T.w1mag[I], 200, range=((-1,5),(0,7)))
	plt.xlabel('g - i (psf)')
	plt.ylabel('i - W1 (model)')
	plt.title('Tractor photometry')
	ps.savefig()

	for wcut in wcuts:
		J = I[(T.w1mag[I] < wcut)]
		plothist(T.gpsf[J] - T.ipsf[J], T.imod[J] - T.w1mag[J], 200, range=((-1,5),(0,7)))
		plt.xlabel('g - i (psf)')
		plt.ylabel('i - W1 (model)')
		plt.title('Tractor photometry: W1 < %i' % wcut)
		ps.savefig()

	
	plothist(C.gpsf - C.ipsf, C.ipsf - C.w1mag, 200, range=((-1,5),(0,7)))
	plt.xlabel('g - i (psf)')
	plt.ylabel('i - W1 (psf)')
	plt.title('WISE catalog photometry')
	ps.savefig()
	
	
	IT = np.logical_and(T.w1mag < 25, T.w2mag < 25)
	
	for l1,x1,l2,x2 in [
		#('g (psf)', 'gpsf')*2,
		('r (psf)', 'rpsf')*2,
		('W1', 'w1mag', 'W1', 'w1mpro'),
		('W2', 'w2mag', 'W2', 'w2mpro')]:
	
		if l1.startswith('W'):
			rng = ((10,20),(-2,2))
		else:
			rng = ((15,25),(-2,2))
	
		plothist(T.get(x1)[IT], (T.w1mag - T.w2mag)[IT], 200,
				 range=rng)
		plt.xlabel(l1)
		plt.ylabel('W1 - W2')
		plt.title('Tractor photometry')
		ps.savefig()

		for wcut in wcuts:
			J = IT[(T.w1mag[IT] < wcut)]
			plothist(T.get(x1)[J], (T.w1mag - T.w2mag)[J], 200,
					 range=rng)
			plt.xlabel(l1)
			plt.ylabel('W1 - W2')
			plt.title('Tractor photometry: W1 < %i' % wcut)
			ps.savefig()
		
		plothist(C.get(x2), C.w1mpro - C.w2mpro, 200, range=rng)
		plt.xlabel(l2)
		plt.ylabel('W1 - W2')
		plt.title('WISE catalog photometry')
		ps.savefig()

 

#tractor_vs_cat()
#sys.exit(0)



I = np.flatnonzero((W.w1 + W.w2) > 0.)

SW = S[I]
SW.add_columns_from(W[I])
print(len(SW), 'rows with WISE measurements')

SW.optpsf = fluxtomag((SW.psfflux[:,1] * 0.8 +
					   SW.psfflux[:,2] * 0.6 +
					   SW.psfflux[:,3] * 1.0) / 2.4)
SW.optmod = fluxtomag((SW.modelflux[:,1] * 0.8 +
					   SW.modelflux[:,2] * 0.6 +
					   SW.modelflux[:,3] * 1.0) / 2.4)
SW.wise = fluxtomag((SW.w1 * 1.0 +
					 SW.w2 * 0.5) / 1.5)
SW.writeto(matchedfn)


print('optpsf:', SW.optpsf.min(), SW.optpsf.max())
print('optmod:', SW.optmod.min(), SW.optmod.max())
print('wise:', SW.wise.min(), SW.wise.max())

# I = np.flatnonzero( ((SW.gpsf - SW.ipsf) < 1.5) )
# print(len(I), 'pass g-i cut')
# I = np.flatnonzero( ((SW.gpsf - SW.ipsf) < 1.5) *
# 					(SW.optpsf > 17.) *
# 					(SW.optpsf < 22.))
# print(len(I), 'pass g-i and 17-22 cut')
# I = np.flatnonzero( ((SW.gpsf - SW.ipsf) < 1.5) *
# 					(SW.optpsf > 17.) *
# 					(SW.optpsf < 22.) *
# 					((SW.optmod - SW.wise) > ((SW.gpsf - SW.ipsf) + 3)))
# print(len(I), 'pass g-i and 17-22 and opt-wise color cut')
# I = np.flatnonzero( ((SW.gpsf - SW.ipsf) < 1.5) *
# 					(SW.optpsf > 17.) *
# 					(SW.optpsf < 22.) *
# 					((SW.optmod - SW.wise) > ((SW.gpsf - SW.ipsf) + 3)) *
# 					np.logical_or(SW.ispsf, (SW.optpsf - SW.optmod) < 0.1) )
# print(len(I), 'pass g-i and 17-22 and opt-wise color and PSF cut')



#qsocuts(SW)
#sys.exit(0)






# Tractor -- WISE catalog comparison

R = 4./3600.
# NOTE, this matches all W entries (ie, same RA,Dec as SDSS), not just
# the ones with photometry.
I,J,d = match_radec(W.ra, W.dec, WC.ra, WC.dec, R, nearest=True)
print(len(I), 'matches to WISE catalog')
plt.clf()
lo,hi = 8,20

K = S.ispsf[I]
L = S.isgal[I]
for wb in ['w1','w2']:
	for wcol in ['mpro', 'mag']:
		for txt,cI,cJ in [('all',I,J), ('psf',I[K],J[K]), ('gal',I[L],J[L])]:
			Wb = wb.upper()
			loghist(WC.get(wb + wcol)[cJ], W.get(wb + 'mag')[cI], 200, range=((lo,hi),(lo,hi)))
			plt.xlabel('WISE catalog %s mag (%s%s)' % (Wb, wb, wcol))
			plt.ylabel('Tractor %s mag' % Wb)
			plt.plot([lo,hi],[lo,hi],'b--')
			plt.axis([hi,lo,hi,lo])
			plt.title('Tractor vs WISE catalog: ' + txt)
			ps.savefig()


# Some summary plots


# plt.clf()
# plt.hist(np.log10(np.maximum(1e-3, S.modelflux[:,2])), 100)
# plt.xlabel('log modelflux r-band (nmgy)')
# ps.savefig()

plt.clf()
n1,b,p1 = plt.hist(S.rmod, 100, log=True, range=(10,30), histtype='step', color='r')
n2,b,p2 = plt.hist(S.rpsf, 100, log=True, range=(10,30), histtype='step', color='b')
plt.xlabel('r (mag)')
plt.legend((p1,p2),('Model flux', 'PSF flux'))
plt.ylim(0.1, max(n1.max(), n2.max())*1.2)
ps.savefig()

## Spatial variation in errors from micro-steradian blocks?

# position within block in RA
rblock = np.fmod(W.ra - r0, rr[1]-rr[0])
for bname in ['w1','w2']:
	wf = W.get(bname)
	I = np.flatnonzero(wf != 0)
	err = 1./np.sqrt(np.maximum(1e-16, W.get(bname + '_ivar')[I]))

	loghist(rblock[I], np.log10(np.clip(err, 1., 1e3)), 200)
	plt.xlabel('dRA within block')
	plt.ylabel(bname + ' flux error')
	ps.savefig()



for bname in ['w1','w2']:
	rf = S.modelflux[:,2]
	wf = W.get(bname)
	I = np.flatnonzero(wf != 0)
	rf = rf[I]
	wf = wf[I]
	wmag = W.get(bname + 'mag')[I]
	wferr = 1./np.sqrt(np.maximum(1e-16, W.get(bname + '_ivar')[I]))

	print('Got', len(wf), 'non-zero', bname, 'measurements')

	# plt.clf()
	# plt.hist(-2.5*(np.log10(np.maximum(1e-3, wf))-9), 100, log=True)
	# plt.xlabel(bname + ' (mag)')
	# ps.savefig()

	loghist(np.log10(np.maximum(1e-3, wf)), np.log10(np.clip(wferr, 1., 1e3)), 100)
	plt.xlabel('log ' + bname + ' flux')
	plt.ylabel('log ' + bname + ' flux err')
	ps.savefig()

	plothist(W.ra[I], W.dec[I], 200, range=((r0,r1),(d0,d1)))
	setRadecAxes(r0,r1,d0,d1)
	plt.title(bname + ' measurements')
	ps.savefig()

	#print('Unique objc_types:', np.unique(S.objc_type))
	if sum(S.ispsf[I]) and sum(np.logical_not(S.ispsf[I])):
		plt.clf()
		n1,b,p1 = plt.hist(wmag[S.ispsf[I]], 100, log=True, histtype='step', color='r')
		n2,b,p2 = plt.hist(wmag[np.logical_not(S.ispsf[I])], 100, log=True,
						   histtype='step', color='b')
		plt.ylim(1, max(max(n1),max(n2))*1.2)
		plt.xlabel(bname + ' (mag)')
		plt.legend((p1,p2), ('Point srcs', 'Extended'), loc='upper left')
	ps.savefig()

	loghist(np.log10(np.maximum(1e-3, rf)), np.log10(np.maximum(1e-3, wf)), 200)
	plt.xlabel('log r flux (nmgy)')
	plt.ylabel('log ' + bname + ' flux (nmgy)')
	ps.savefig()

	ok = np.flatnonzero((rf > 0) * (wf > 0))
	rmag = -2.5 * (np.log10(rf[ok]) - 9)
	wmag = -2.5 * (np.log10(wf[ok]) - 9)

	lo,hi = 10,25
	loghist(rmag, wmag, 200, range=((lo,hi),(lo,hi)))
	plt.xlabel('r mag')
	plt.ylabel(bname + ' mag')
	plt.xlim(hi,lo)
	plt.ylim(hi,lo)
	ps.savefig()
	
	# plothist(wmag-rmag, rmag, 200)
	# plt.ylabel('r (mag)')
	# plt.xlabel('%s - r (mag)' % bname)
	# ylo,yhi = plt.ylim()
	# plt.ylim(yhi,ylo)
	# ps.savefig()
	
	plothist(wmag-rmag, rmag, 200, range=((-15,5),(15,25)))
	plt.ylabel('r (mag)')
	plt.xlabel('%s - r (mag)' % bname)
	ylo,yhi = plt.ylim()
	plt.ylim(yhi,ylo)
	ps.savefig()
	
	loghist(wmag-rmag, rmag, 200, range=((-15,5),(15,25)))
	plt.ylabel('r (mag)')
	plt.xlabel('%s - r (mag)' % bname)
	ylo,yhi = plt.ylim()
	plt.ylim(yhi,ylo)
	ps.savefig()








I = np.flatnonzero((W.w1 > 0) * (W.w2 > 0))
print('Found', len(I), 'with w1 and w2 pos')

rf = S.modelflux[:,2]
I = np.flatnonzero((W.w1 > 0) * (W.w2 > 0) * (rf > 0))

rf = rf[I]
w1 = W.w1[I]
w2 = W.w2[I]

rmag  = -2.5 * (np.log10(rf) - 9)
w1mag = -2.5 * (np.log10(w1) - 9)
w2mag = -2.5 * (np.log10(w2) - 9)

print(len(w1), 'with W1,W2,r')
J = np.flatnonzero((w1mag - w2mag > 0.8) * (rmag < 22) * (w1mag < 20) * (w2mag < 20))
print(len(J), 'with w1-w2 > 0.8, r<22, w1,w2 < 20')

K = np.flatnonzero((w1mag - w2mag > 0.8) * (rmag < 22) * (w1mag < 20) * (w2mag < 20) * (S.objc_type[I] == 6))
print(len(K), 'also OBJTYPE=6')

loghist(w1mag - w2mag, rmag, 200, range=((-15,15),(15,25)))
plt.ylabel('r (mag)')
plt.xlabel('W1 - W2 (mag)')
#ax = plt.axis()
#plt.axis(ax)
plt.axvline(0.8, color=(0,0.5,1))
ylo,yhi = plt.ylim()
plt.ylim(yhi,ylo)
ps.savefig()

ok = np.flatnonzero((S.rpsf > 17) * (S.rpsf < 22) *
					(W.w1mag < 20) * (W.w2mag < 20))
print(len(ok), 'are in basic cut')

dw = (W.w1mag - W.w2mag)[ok]
print('dw range', dw.min(), dw.max())
print('r range', S.rpsf[ok].min(), S.rpsf[ok].max())


loghist((W.w1mag - W.w2mag)[ok], S.rpsf[ok], 200, range=((-15,15),(15,25)))
plt.ylabel('r (mag)')
plt.xlabel('W1 - W2 (mag)')
plt.axvline(0.8, color=(0,0.5,1))
ylo,yhi = plt.ylim()
plt.ylim(yhi,ylo)
plt.title('17 < r < 22, w1,w2 < 20')
ps.savefig()

ok = np.flatnonzero((S.rpsf < 22) * (S.rpsf > 17) *
					(W.w1mag < 20) * (W.w2mag < 20) *
					(S.objc_type == 6))
loghist((W.w1mag - W.w2mag)[ok], S.rpsf[ok], 200, range=((-15,15),(15,25)))
plt.ylabel('r (mag)')
plt.xlabel('W1 - W2 (mag)')
ylo,yhi = plt.ylim()
plt.axvline(0.8, color=(0,0.5,1))
plt.ylim(yhi,ylo)
plt.title('r < 22, w1,w2 < 20, OBJTYPE=6')
ps.savefig()

inbox1 = ((S.rpsf < 22) * (S.rpsf > 17) *
		  (W.w1mag < 20) * (W.w2mag < 20) *
		  (S.objc_type == 6) *
		  (W.w1mag - W.w2mag > 0.8))
box1 = np.flatnonzero(inbox1)
print(len(box1), 'in box1')

inbox2 = ((S.rpsf < 22) * (S.rpsf > 17) *
		  (W.w1mag < 20) *
		  (S.objc_type == 6) *
		  ((S.rpsf - W.w1mag) - 2. > 1.5 * (S.gpsf - S.ipsf)) *
		  ((S.gpsf - S.ipsf) < 1.))
box2 = np.flatnonzero(inbox2)
print(len(box2), 'in box2')
		 
sb1 = set(box1)
sb2 = set(box2)

#print(len(np.flatnonzero(inbox1 * inbox2)), 'in both')
#print(len(np.flatnonzero(np.logical_or(inbox1, inbox2))), 'in either')


either = sb1.union(sb2)
print(len(either), 'in the union')

both = sb1.intersection(sb2)
print(len(both), 'in the intersection')



