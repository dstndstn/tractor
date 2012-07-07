import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *

def plot_cmd(allmags, i2mags, band):
	print 'i2 mags shape', i2mags.shape
	print 'allmags shape', allmags.shape

	plt.figure(figsize=(6,6))
	plt.clf()
	#plotpos0 = [0.15, 0.15, 0.84, 0.80]
	#plt.gca().set_position(plotpos0)
	xx,yy,xerr = [],[],[]
	xx2,xerr2 = [],[]
	for i2,rr in zip(i2mags, allmags):
		print 'rr', rr

		# When the source is off the image, the optimizer doesn't change anything and
		# we end up with r = i2
		I = (rr != i2)
		rr = rr[I]
		ii2 = i2.repeat(len(rr))

		rr = np.minimum(rr, 25.)
		
		#plt.plot(rr - ii2, ii2, 'o', mfc='b', mec='none', ms=5, alpha=0.5)
		mr = np.mean(rr)
		sr = np.std(rr)
		#plt.plot([(mr-sr) - i2, (mr+sr) - i2], [i2,i2], 'b-', lw=3, alpha=0.25)

		medr = np.median(rr)
		iqr = (1./0.6745) * 0.5 * (np.percentile(rr, 75) - np.percentile(rr, 25))
		#plt.plot([(medr - iqr) - i2, (medr + iqr) - i2], [i2,i2], 'g-', lw=3, alpha=0.25)

		xx.append(mr - i2)
		yy.append(i2)
		xerr.append(sr)

		xx2.append(medr - i2)
		xerr2.append(iqr)

	yy2 = np.array(yy)
	xx2 = np.array(xx2)
	xerr2 = np.array(xerr2)
	I = (xerr2 < 1)
	xx2 = xx2[I]
	yy2 = yy2[I]
	xerr2 = xerr2[I]

	plt.clf()
	plt.plot(xx2, yy2, 'o', mfc='b', mec='none', mew=0, ms=5, alpha=0.8)
	plt.plot([xx2-xerr2, xx2+xerr2], [yy2,yy2], '-', color='b', lw=2, mew='none', alpha=0.5)
	plt.axis([-3, 3, 21.5, 15.5])
	plt.xlabel('SDSS %s - CFHT i (mag)' % band)
	plt.ylabel('CFHT i (mag)')
	plt.yticks(range(16, 21 + 1))
	plt.title('CS82 test patch: SDSS--CFHT CMD')
	plt.savefig('cmd-%s.png' % band)
	plt.savefig('cmd-%s.pdf' % band)


if __name__ == '__main__':
	(allp, i2mags, cat) = unpickle_from_file('s2-260.pickle')

	allbands = ['i2','u','g','r','i','z']

	#print 'i2 mags', i2mags
	allmags = dict([(b, []) for b in allbands])
	for ii,bb,pa in allp:
		#print 'pa', pa
		# Thaw just this image's band
		cat.freezeParamsRecursive(*allbands)
		cat.thawParamsRecursive(bb)
		cat.setParams(pa)
		mags = [src.getBrightness().getMag(bb) for src in cat]
		#print 'mags', mags
		#print len(mags)
		assert(len(mags) == len(i2mags))
		allmags[bb].append(mags)

	for bb in allbands:
		print 'Band', bb, 'shape', np.array(allmags[bb]).shape
		#allmags[bb] = np.array(allmags[bb])
		allmags = np.array(allmags[bb])
		if bb != 'i2':
			allmags = allmags.T
		plot_cmd(allmags, i2mags, bb)




def s1():
	(allp, i2mags, cat) = unpickle_from_file('s1-258.pickle')
	plt.figure(figsize=(6,6))

	T=fits_table('ri.fits')
	plt.clf()
	plt.plot(T.r - T.i, T.i, 'r,', mfc='r', mec='none', alpha=0.5)
	plt.xlabel('SDSS r - SDSS i (mag)')
	plt.ylabel('SDSS i (mag)')
	plt.title('SDSS galaxies')
	plt.axis([-3, 3, 21.5, 15.5])
	#plt.axis([-3, 3, 22, 16])
	plt.savefig('gal-ri.pdf')

	T=fits_table('star-ri.fits')
	plt.clf()
	plt.plot(T.r - T.i, T.i, 'r,', mfc='r', mec='none', alpha=0.5)
	plt.xlabel('SDSS r - SDSS i (mag)')
	plt.ylabel('SDSS i (mag)')
	plt.title('SDSS stars')
	plt.axis([-3, 3, 21.5, 15.5])
	#plt.axis([-3, 3, 22, 16])
	plt.savefig('star-ri.pdf')

	(allp, i2mags, cat) = unpickle_from_file('s1-258.pickle')

	allbands = ['i2','u','g','r','i','z']

	#print 'i2 mags', i2mags
	allmags = []
	for ii,bb,pa in allp:
		#print 'pa', pa
		# Thaw just this image's band
		cat.freezeParamsRecursive(*allbands)
		cat.thawParamsRecursive(bb)
		cat.setParams(pa)
		mags = [src.getBrightness().getMag(bb) for src in cat]
		#print 'mags', mags
		#print len(mags)
		assert(len(mags) == len(i2mags))
		allmags.append(mags)

	allmags = np.array(allmags)
	print 'i2 mags shape', i2mags.shape
	print 'allmags shape', allmags.shape

	plt.figure(figsize=(6,6))
	plt.clf()
	#plotpos0 = [0.15, 0.15, 0.84, 0.80]
	#plt.gca().set_position(plotpos0)
	xx,yy,xerr = [],[],[]
	xx2,xerr2 = [],[]
	for i2,rr in zip(i2mags, allmags.T):
		print 'rr', rr

		# When the source is off the image, the optimizer doesn't change anything and
		# we end up with r = i2
		I = (rr != i2)
		rr = rr[I]
		ii2 = i2.repeat(len(rr))

		rr = np.minimum(rr, 25.)
		
		#plt.plot(rr - ii2, ii2, 'b+', mfc='b', mec='b', ms=5)
		#plt.plot(rr - ii2, ii2, 'bo', mfc='none', mec='b', ms=5)
		#plt.plot(rr - ii2, ii2, '.', mfc='b', mec='none', ms=15)
		plt.plot(rr - ii2, ii2, 'o', mfc='b', mec='none', ms=5, alpha=0.5)
		#plt.plot([rr - ii2]*2, [ii2 - 0.02, ii2 + 0.02], 'b-', lw=2, alpha=0.5)
		mr = np.mean(rr)
		sr = np.std(rr)
		plt.plot([(mr-sr) - i2, (mr+sr) - i2], [i2,i2], 'b-', lw=3, alpha=0.25)

		medr = np.median(rr)
		iqr = (1./0.6745) * 0.5 * (np.percentile(rr, 75) - np.percentile(rr, 25))
		plt.plot([(medr - iqr) - i2, (medr + iqr) - i2], [i2,i2], 'g-', lw=3, alpha=0.25)

		xx.append(mr - i2)
		yy.append(i2)
		xerr.append(sr)

		xx2.append(medr - i2)
		xerr2.append(iqr)

	print 'Axis', plt.axis()
	plt.axis([-3, 3, 21, 15])
	plt.xlabel('SDSS r - CFHT i (mag)')
	plt.ylabel('CFHT i (mag)')
	plt.yticks(range(15, 21 + 1))
	plt.savefig('cmd.png')


	yy2 = np.array(yy)

	xx = np.array(xx)
	yy = np.array(yy)
	xerr = np.array(xerr)
	I = (xerr < 1)
	xx = xx[I]
	yy = yy[I]
	xerr = xerr[I]

	xx2 = np.array(xx2)
	xerr2 = np.array(xerr2)
	I = (xerr2 < 1)
	xx2 = xx2[I]
	yy2 = yy2[I]
	xerr2 = xerr2[I]

	plt.clf()
	plt.errorbar(xx, yy, xerr=xerr, fmt=None, linewidth=2, alpha=0.5)
	plt.errorbar(xx2, yy2, xerr=xerr2, fmt=None, color='g', linewidth=2, alpha=0.5)
	plt.axis([-3, 3, 21, 15])
	plt.xlabel('SDSS r - CFHT i (mag)')
	plt.ylabel('CFHT i (mag)')
	plt.yticks(range(15, 21 + 1))
	plt.savefig('cmd2.png')


	plt.clf()
	plt.plot(xx, yy, 'o', mfc='b', mec='none', mew=0, ms=5, alpha=0.8)
	plt.plot([xx-xerr, xx+xerr], [yy,yy], '-', color='b', lw=2, mew='none', alpha=0.5)
	plt.axis([-3, 3, 21, 15])
	plt.xlabel('SDSS r - CFHT i (mag)')
	plt.ylabel('CFHT i (mag)')
	plt.yticks(range(15, 21 + 1))
	plt.title('CS82 test patch: SDSS--CFHT CMD')
	plt.savefig('cmd3.png')

	plt.clf()
	plt.plot(xx2, yy2, 'o', mfc='b', mec='none', mew=0, ms=5, alpha=0.8)
	plt.plot([xx2-xerr2, xx2+xerr2], [yy2,yy2], '-', color='b', lw=2, mew='none', alpha=0.5)
	plt.axis([-3, 3, 21.5, 15.5])
	plt.xlabel('SDSS r - CFHT i (mag)')
	plt.ylabel('CFHT i (mag)')
	plt.yticks(range(16, 21 + 1))
	plt.title('CS82 test patch: SDSS--CFHT CMD')
	plt.savefig('cmd4.png')
	plt.savefig('cmd4.pdf')


