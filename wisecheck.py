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
from astrometry.util.util import * #Sip, anwcs, Tan
from astrometry.blind.plotstuff import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *
from astrometry.util.stages import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params
from tractor.ttime import *


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
											'w1mpro', 'w1mag', 'w2mpro', 'w2mag'])

		# fn = '/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part%02i.fits' % s
		# print('Reading', fn)
		# T = fits_table(fn, columns=['ra','dec','cntr',
		# 							'w1mpro', 'w1mag', 'w2mpro', 'w2mag'])
		# T.cut((T.ra > r0) * (T.ra < r1) * (T.dec > d0) * (T.dec < d1))
		print('Cut to', len(T))
		TT.append(T)
	W = merge_tables(TT)
	return W

ps = PlotSequence('wisecheck')



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
		basefn = 'eboss-w3-worst%04i-r%02i-d%02i' % (wi, ri,di)
		opt.picklepat = '%s-stage%%0i.pickle' % basefn
		opt.ps = basefn
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
		# ps = PlotSequence(basefn)
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




# T = tabledata()
# T.even = (np.arange(10) % 2 == 0)
# T.about()
# T.writeto('even.fits')
# T2 = fits_table('even.fits')
# T2.about()
# print(T2.even)
# print(T2.even.astype(np.uint8))
# assert(np.all(T2.even == T.even))

matchedfn = 'sw.fits'
if os.path.exists(matchedfn):
	SW = fits_table(matchedfn)
	qsocuts(SW)
	sys.exit(0)


basefn = 'ebossw3-v4'
fn = 'eboss-w3-v4-wise-dr9.fits'

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


import wise


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

SW.writeto(matchedfn)

qsocuts(SW)

sys.exit(0)



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



plt.clf()
n1,b,p1 = plt.hist(W.w1mag, 100, log=True, histtype='step', color='b', range=(5,29))
n2,b,p2 = plt.hist(W.w2mag, 100, log=True, histtype='step', color='r', range=(5,29))
n3,b,p3 = plt.hist(WC.w1mpro, 100, log=True, histtype='step', color='c', range=(5,29))
n4,b,p4 = plt.hist(WC.w2mpro, 100, log=True, histtype='step', color='m', range=(5,29))
plt.xlabel('WISE mag')
plt.legend((p1,p2,p3,p4),('W1 (Tractor)','W2 (Tractor)', 'W1 (WISE cat)', 'W2 (WISE cat)'))
plt.ylim(1, max(n1.max(), n2.max(), n3.max(), n4.max())*1.2)
plt.ylabel('Number of sources')
plt.title('W3 area: WISE forced photometry mag distribution')
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



