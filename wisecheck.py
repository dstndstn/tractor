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

S = fits_table('objs-eboss-w3-dr9.fits')
print('Read', len(S))

#fn = 'eboss-w3-wise-dr9.fits'
#basefn = 'ebossw3'

#fn = 'eboss-w3-v2-wise-dr9.fits'
#basefn = 'ebossw3-v2'

fn = 'eboss-w3-v3-wise-dr9.fits'
basefn = 'ebossw3-v3'

if not os.path.exists(fn):
	W = merge_results(S, basefn, fn)
else:
	print('Reading existing', fn)
	W = fits_table(fn)

W.w1mag = -2.5*(np.log10(np.maximum(1e-3, W.w1))-9)
W.w2mag = -2.5*(np.log10(np.maximum(1e-3, W.w2))-9)

print('w1mag', W.w1mag.min(), W.w1mag.max())
print('w2mag', W.w2mag.min(), W.w2mag.max())

S.gpsf = -2.5*(np.log10(np.maximum(1e-3, S.psfflux[:,1]))-9)
S.rpsf = -2.5*(np.log10(np.maximum(1e-3, S.psfflux[:,2]))-9)
S.ipsf = -2.5*(np.log10(np.maximum(1e-3, S.psfflux[:,3]))-9)

S.gmod = -2.5*(np.log10(np.maximum(1e-3, S.modelflux[:,1]))-9)
S.rmod = -2.5*(np.log10(np.maximum(1e-3, S.modelflux[:,2]))-9)
S.imod = -2.5*(np.log10(np.maximum(1e-3, S.modelflux[:,3]))-9)

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
	S.ispsf = (S.objc_type == 6)
	S.isgal = (S.objc_type == 3)

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


# Tractor/SDSS vs WISE/SDSS comparisons

I2 = np.flatnonzero(np.logical_or(W.w1 > 0, W.w2 > 0))

T = S[I2]
T.add_columns_from(W[I2])

C = WC[J]
C.add_columns_from(S[I])

print(len(T), 'Tractor-SDSS matches')
print(len(C), 'WISE-SDSS matches')

I = np.flatnonzero((T.gpsf != T.ipsf) * (T.w1mag < 25))

plothist(T.gpsf[I] - T.ipsf[I], T.imod[I] - T.w1mag[I], 200, range=((-1,5),(0,7)))
plt.xlabel('g - i (psf)')
plt.ylabel('i - W1 (model)')
plt.title('Tractor photometry')
ps.savefig()

plothist(C.gpsf - C.ipsf, C.ipsf - C.w1mag, 200, range=((-1,5),(0,7)))
plt.xlabel('g - i (psf)')
plt.ylabel('i - W1 (psf)')
plt.title('WISE catalog photometry')
ps.savefig()


IT = np.logical_and(T.w1mag < 25, T.w2mag < 25)

for l1,x1,l2,x2 in [('g (psf)', 'gpsf')*2, ('r (psf)', 'rpsf')*2,
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
	
	plothist(C.get(x2), C.w1mpro - C.w2mpro, 200, range=rng)
	plt.xlabel(l2)
	plt.ylabel('W1 - W2')
	plt.title('WISE catalog photometry')
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



 




# Some plots looking at image statistics.

for band in [1,2]:
	fns = glob('%s-r??-d??-w%i-imstats.fits' % (basefn, band))
	TT = []

	# sum by block
	bchisq, bnpix = [],[]
	# sum by scan+frame
	fchisq, fnpix = {},{}

	print('Reading', len(fns), 'W%i image stats' % band)

	for i,fn in grenumerate(fns):
		T = fits_table(fn)
		#print fn, '-->', len(T)

		# Make maps of the sky estimates (by frame) in different blocks?
		#ri,di = get_ridi(fn)

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

