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

ps = PlotSequence('wisecheck')

S = fits_table('objs-eboss-w3-dr9.fits')
print('Read', len(S))

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

		m = rex.search(fn)
		if m is None:
			raise RuntimeError('regex on filename did not match')
		ri = int(m.group('ri'), 10)
		di = int(m.group('di'), 10)
		#print('ri', ri, 'di', di)
		
		rlo,rhi = rr[ri], rr[ri+1]
		dlo,dhi = dd[di], dd[di+1]
		T.inblock = (T.ra >= rlo) * (T.ra < rhi) * (T.dec >= dlo) * (T.dec < dhi)
	
		TT.append(T)
	T = merge_tables(TT)
	print('Read total of', len(T))

	T.cut(T.inblock)
	print('Cut to', len(T), 'in block')

	W.append(T)

	R = 0.001 / 3600.
	I,J,d = match_radec(T.ra, T.dec, S.ra, S.dec, R)
	print(len(I), 'matches of', len(T))
	T.row[:] = -1	
	T.row[I] = J

	# SS = S[T.row]
	# rf = SS.modelflux[:,2]
	# wf = T.get(bname)
	# 
	# loghist(rf, wf, 200)
	# plt.xlabel('r flux (nmgy)')
	# plt.ylabel(bname + ' flux (nmgy)')
	# ps.savefig()
	# 
	# ok = np.flatnonzero((rf > 0) * (wf > 0))
	# rmag = -2.5 * (np.log10(rf[ok]) - 9)
	# wmag = -2.5 * (np.log10(wf[ok]) - 9)
	# 
	# plothist(wmag-rmag, rmag, 200)
	# plt.ylabel('r (mag)')
	# plt.xlabel('%s - r (mag)' % bname)
	# ylo,yhi = plt.ylim()
	# plt.ylim(yhi,ylo)
	# ps.savefig()
	# 
	# plothist(wmag-rmag, rmag, 200, range=((-15,10),(15,25)))
	# plt.ylabel('r (mag)')
	# plt.xlabel('%s - r (mag)' % bname)
	# ylo,yhi = plt.ylim()
	# plt.ylim(yhi,ylo)
	# ps.savefig()
	# 
	# loghist(wmag-rmag, rmag, 200, range=((-15,10),(15,25)))
	# plt.ylabel('r (mag)')
	# plt.xlabel('%s - r (mag)' % bname)
	# ylo,yhi = plt.ylim()
	# plt.ylim(yhi,ylo)
	# ps.savefig()


W1,W2 = W

NS = len(S)
SW = tabledata()

X = np.zeros(NS)
X[W1.row] = W1.ra
X[W2.row] = W2.ra
SW.ra2 = X
X = np.zeros(NS)
X[W1.row] = W1.dec
X[W2.row] = W2.dec
SW.dec2 = X

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
SW.writeto('eboss-w3-wise-dr9.fits')

SS = S
for bname in ['w1','w2']:
	rf = SS.modelflux[:,2]
	wf = SW.get(bname)
	I = np.flatnonzero(wf != 0)
	rf = rf[I]
	wf = wf[I]
	
	
	loghist(rf, wf, 200)
	plt.xlabel('r flux (nmgy)')
	plt.ylabel(bname + ' flux (nmgy)')
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

I = np.flatnonzero((SW.w1 > 0) * (SW.w2 > 0))
print('Found', len(I), 'with w1 and w2 pos')

rf = S.modelflux[:,2]
I = np.flatnonzero((SW.w1 > 0) * (SW.w2 > 0) * (rf > 0))

rf = rf[I]
w1 = SW.w1[I]
w2 = SW.w2[I]

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


# 
# 
# hasw1 = np.empty(len(S), int)
# hasw1[:] = -1
# hasw1[W1.row] = np.arange(len(W1))
# 
# hasw2 = np.empty(len(S), int)
# hasw2[:] = -1
# hasw2[W2.row] = np.arange(len(W2))
# 
# hasboth = ((hasw1 >= 0) * (hasw2 >= 0))
# I1 = hasw1[hasboth]
# I2 = hasw2[hasboth]
# print(len(I1), len(I2), 'with both w1 and w2')
# 
# W1 = W1[I1]
# W2 = W2[I2]
# 
# assert(np.all(W1.row == W2.row))
# SS = S[W1.row]
# rf = SS.modelflux[:,2]
# wf1 = W1.w1
# wf2 = W2.w2
# 
# pos = np.flatnonzero((wf1 > 0) * (wf2 > 0))
# print(len(pos), 'with positive flux')
# 
# ok = np.flatnonzero((rf > 0) * (wf1 > 0) * (wf2 > 0))
# rmag = -2.5 * (np.log10(rf[ok]) - 9)
# w1mag = -2.5 * (np.log10(wf1[ok]) - 9)
# w2mag = -2.5 * (np.log10(wf2[ok]) - 9)
# 
# loghist(w1mag - w2mag, rmag, 200, range=((-15,15),(15,25)))
# plt.ylabel('r (mag)')
# plt.xlabel('W1 - W2 (mag)')
# ylo,yhi = plt.ylim()
# plt.ylim(yhi,ylo)
# ps.savefig()




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

