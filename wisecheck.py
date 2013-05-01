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

def merge_results(S, outfn):
	## HACK!
	NDEC = 50
	r0,r1 = 210.593,  219.132
	d0,d1 =  51.1822,  54.1822
	dd = np.linspace(d0, d1, NDEC + 1)
	rr = np.linspace(r0, r1, 91)
	W = []
	for band in [1,2]:
		bname = 'w%i' % band
		fns = glob('ebossw3-r??-d??-w%i.fits' % band)
		TT = []
		rex = re.compile(r'-r(?P<ri>\d\d)-d(?P<di>\d\d)-')
		print('Reading', len(fns), 'W%i results' % band)
		for i,fn in grenumerate(fns):
			T = fits_table(fn)
			# inside/outside the block that was being fit?
			m = rex.search(fn)
			if m is None:
				raise RuntimeError('regex on filename did not match')
			ri = int(m.group('ri'), 10)
			di = int(m.group('di'), 10)
			rlo,rhi = rr[ri], rr[ri+1]
			dlo,dhi = dd[di], dd[di+1]
			T.inblock = (T.ra >= rlo) * (T.ra < rhi) * (T.dec >= dlo) * (T.dec < dhi)
		
			TT.append(T)
		T = merge_tables(TT)
		print('Read total of', len(T))
	
		T.cut(T.inblock)
		print('Cut to', len(T), 'in block')
	
		W.append(T)
	
		## I messed up the indexing... spherematch to the rescue.
		R = 0.001 / 3600.
		I,J,d = match_radec(T.ra, T.dec, S.ra, S.dec, R)
		print(len(I), 'matches of', len(T))
		T.row[:] = -1	
		T.row[I] = J
	
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
	X = np.zeros(NS)
	X[W1.row] = W1.w1
	SW.w1 = X
	X = np.zeros(NS)
	X[W2.row] = W2.w2
	SW.w2 = X
	SW.writeto(outfn)
	return SW


S = fits_table('objs-eboss-w3-dr9.fits')
print('Read', len(S))

fn = 'eboss-w3-wise-dr9.fits'
if not os.path.exists(fn):
	W = merge_results(S, fn)
else:
	W = fits_table(fn)

ps = PlotSequence('wisecheck')



import wise
tpsf = wise.get_psf_model(1, pixpsf=True)

psfp = tpsf.getPointSourcePatch(0, 0)
psf = psfp.patch

psf /= psf.sum()

plt.clf()
plt.imshow(np.log10(np.maximum(1e-5, psf)), interpolation='nearest', origin='lower')
ps.savefig()

print('PSF norm:', np.sqrt(np.sum(np.maximum(0, psf)**2)))
print('PSF max:', psf.max())

sys.exit(0)


# Some summary plots

# plt.clf()
# plt.hist(np.log10(np.maximum(1e-3, S.modelflux[:,2])), 100)
# plt.xlabel('log modelflux r-band (nmgy)')
# ps.savefig()

plt.clf()
plt.hist(-2.5*(np.log10(np.maximum(1e-3, S.modelflux[:,2]))-9), 100, log=True,
		 range=(10,30))
plt.xlabel('modelflux r (mag)')
ps.savefig()

plt.clf()
plt.hist(-2.5*(np.log10(np.maximum(1e-3, S.psfflux[:,2]))-9), 100, log=True,
		 range=(10,30))
plt.xlabel('psfflux r (mag)')
ps.savefig()

for bname in ['w1','w2']:
	rf = S.modelflux[:,2]
	wf = W.get(bname)
	I = np.flatnonzero(wf != 0)
	rf = rf[I]
	wf = wf[I]

	plt.clf()
	plt.hist(-2.5*(np.log10(np.maximum(1e-3, wf))-9), 100, log=True)
	plt.xlabel(bname + ' (mag)')
	ps.savefig()

	print('Unique objc_types:', np.unique(S.objc_type))

	S.ispsf = (S.objc_type == 6)

	plt.clf()
	n1,b,p1 = plt.hist(-2.5*(np.log10(np.maximum(1e-3, wf[S.ispsf[I]]))-9), 100, log=True,
					  histtype='step', color='r')
	n2,b,p2 = plt.hist(-2.5*(np.log10(np.maximum(1e-3, wf[np.logical_not(S.ispsf[I])]))-9), 100, log=True,
					  histtype='step', color='b')
	plt.ylim(1, max(max(n1),max(n2))*1.2)
	plt.xlabel(bname + ' (mag)')
	plt.legend((p1,p2), ('Point srcs', 'Extended'), loc='upper left')
	ps.savefig()

	#loghist(rf, wf, 200)
	loghist(np.log10(np.maximum(1e-3, rf)), np.log10(np.maximum(1e-3, wf)), 200)
	plt.xlabel('log r flux (nmgy)')
	plt.ylabel('log ' + bname + ' flux (nmgy)')
	ps.savefig()

	ok = np.flatnonzero((rf > 0) * (wf > 0))
	rmag = -2.5 * (np.log10(rf[ok]) - 9)
	wmag = -2.5 * (np.log10(wf[ok]) - 9)
	
	plothist(wmag-rmag, rmag, 200)
	plt.ylabel('r (mag)')
	plt.xlabel('%s - r (mag)' % bname)
	ylo,yhi = plt.ylim()
	plt.ylim(yhi,ylo)
	ps.savefig()
	
	plothist(wmag-rmag, rmag, 200, range=((-15,10),(15,25)))
	plt.ylabel('r (mag)')
	plt.xlabel('%s - r (mag)' % bname)
	ylo,yhi = plt.ylim()
	plt.ylim(yhi,ylo)
	ps.savefig()
	
	loghist(wmag-rmag, rmag, 200, range=((-15,10),(15,25)))
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
ylo,yhi = plt.ylim()
plt.ylim(yhi,ylo)
ps.savefig()

ok = np.flatnonzero((rmag < 22) * (w1mag < 20) * (w2mag < 20))
loghist((w1mag - w2mag)[ok], rmag[ok], 200, range=((-15,15),(15,25)))
plt.ylabel('r (mag)')
plt.xlabel('W1 - W2 (mag)')
ylo,yhi = plt.ylim()
plt.ylim(yhi,ylo)
plt.title('r < 22, w1,w2 < 20')
ps.savefig()

ok = np.flatnonzero((rmag < 22) * (w1mag < 20) * (w2mag < 20) * (S.objc_type[I] == 6))
loghist((w1mag - w2mag)[ok], rmag[ok], 200, range=((-15,15),(15,25)))
plt.ylabel('r (mag)')
plt.xlabel('W1 - W2 (mag)')
ylo,yhi = plt.ylim()
plt.ylim(yhi,ylo)
plt.title('r < 22, w1,w2 < 20, OBJTYPE=6')
ps.savefig()

sys.exit(0)

# Some plots looking at image statistics.

for band in [1,2]:
	fns = glob('ebossw3-r??-d??-w%i-imstats.fits' % band)
	TT = []

	# sum by block
	bchisq, bnpix = [],[]
	# sum by scan+frame
	fchisq, fnpix = {},{}

	print('Reading', len(fns), 'W%i image stats' % band)

	for i,fn in grenumerate(fns):
		T = fits_table(fn)
		#print fn, '-->', len(T)

		bchisq.append(T.imchisq.sum())
		bnpix.append(T.imnpix.sum())

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

