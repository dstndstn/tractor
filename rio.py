import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.libkd.spherematch import *

def plot_cmd(allmags, i2mags, band, catflags, classstar):
	print 'i2 mags shape', i2mags.shape
	print 'allmags shape', allmags.shape
	print 'catflags shape', catflags.shape

	plt.figure(figsize=(6,6))
	plt.clf()
	#plotpos0 = [0.15, 0.15, 0.84, 0.80]
	#plt.gca().set_position(plotpos0)
	xx,yy,xerr = [],[],[]
	xx2,xerr2 = [],[]
	for i2,rr in zip(i2mags, allmags):
		#print 'rr', rr

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

	flag = catflags[I]
	cstar = classstar[I]

	plt.clf()
	#plt.plot(xx2, yy2, 'o', mfc='b', mec='none', mew=0, ms=5, alpha=0.8)

	LL = []
	for F,c in [((flag > 0), '0.5'), ((flag == 0) * (cstar < 0.5), 'b'),
				((flag == 0) * (cstar >= 0.5), 'g')]:
		p1 = plt.plot(xx2[F], yy2[F], 'o', mfc=c, mec='none', mew=0, ms=5, alpha=0.8)
		LL.append(p1[0])
		plt.plot([xx2[F]-xerr2[F], xx2[F]+xerr2[F]], [yy2[F],yy2[F]], '-',
				 color=c, lw=2, mew='none', alpha=0.5)

	#plt.axis([-3, 3, 21.5, 15.5])
	plt.legend(LL, ('flagged', 'galaxy', 'star'))
	plt.ylim(21.5, 15.5)
	cl,ch = { 'u': (-3,6), 'g': (-1,5), 'r': (-2,3), 'i': (-2,2), 'z': (-2,1) }[band]
	plt.xticks(range(cl,ch+1))
	plt.xlim(cl,ch)
	plt.xlabel('SDSS %s - CFHT i (mag)' % band)
	plt.ylabel('CFHT i (mag)')
	plt.yticks(range(16, 21 + 1))
	plt.title('CS82 test patch: SDSS--CFHT CMD')
	plt.savefig('cmd-%s.png' % band)
	plt.savefig('cmd-%s.pdf' % band)

	I = np.flatnonzero((flag == 0) * (xx2 < -1))
	print 'Not-flagged and c < -1:', I
	return I


if __name__ == '__main__':
	(allp, i2mags, cat) = unpickle_from_file('s2-260.pickle')

	#allbands = ['i2','u','g','r','i','z']
	allbands = ['i2','u','g','r','i','z']

	T = fits_table('cs82data/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
	#RA = 334.32
	#DEC = 0.315
	#sz = 0.12 * 3600.
	#S = sz / 3600.
	#ra0 ,ra1  = RA-S/2.,  RA+S/2.
	#dec0,dec1 = DEC-S/2., DEC+S/2.
	print 'Read', len(T), 'sources'
	T.ra  = T.alpha_j2000
	T.dec = T.delta_j2000
	
	sra  = np.array([src.getPosition().ra for src in cat])
	sdec = np.array([src.getPosition().dec for src in cat])
	ra0,ra1 = sra.min(), sra.max()
	dec0,dec1 = sdec.min(), sdec.max()

	T = T[(T.ra >= ra0) * (T.ra <= ra1) * (T.dec >= dec0) * (T.dec <= dec1)]
	print 'ra', ra0, ra1, 'dec', dec0, dec1
	print 'Cut to', len(T), 'objects nearby.'


	#print 'RA', sra.min(), sra.max()
	#print 'Dec', sdec.min(), sdec.max()


	I1,I2,D = match_radec(sra, sdec, T.ra, T.dec, 0.5/3600.)
	print 'Matched', len(I1), 'of', len(cat)
	print 'D', D
	print len(np.unique(I1)), 'unique cat'
	print len(np.unique(I2)), 'unique T'
	catflags = np.zeros(len(cat), int)
	for i1,i2 in zip(I1,I2):
		catflags[i1] |= T.flags[i2]
	print 'Set', np.sum(catflags), 'catalog flags'
	classstar = np.zeros(len(cat))
	for i1,i2 in zip(I1,I2):
		classstar[i1] = T.class_star[i2]

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

		

	print 'allmags:', allmags.keys()
	for bb in allbands:
		m = np.array(allmags[bb])
		print 'Band', bb, 'shape', m.shape
		#allmags[bb] = np.array(allmags[bb])
		if bb == 'i2':
			continue
		#if bb != 'i2':
		m = m.T
		I = plot_cmd(m, i2mags, bb, catflags, classstar)
		if bb == 'i':
			outliers = I

	from cs82 import *
	RA = 334.32
	DEC = 0.315
	sz = 0.12 * 3600.
	pixscale = 0.187
	S = int(1.01 * sz / pixscale) / 2
	filtermap = {'i.MP9701': 'i2'}
	coim = get_cfht_coadd_image(RA, DEC, S, filtermap=filtermap)

	rr,dd = [],[]
	xx,yy = [],[]
	for i in outliers:
		print 'Outlier source', cat[i]
		rr.append(cat[i].getPosition().ra)
		dd.append(cat[i].getPosition().dec)
		x,y = coim.getWcs().positionToPixel(cat[i].getPosition())
		xx.append(x)
		yy.append(y)
	plt.clf()
	plt.imshow(coim.getImage(), interpolation='nearest', origin='lower',
			   vmin=coim.zr[0], vmax=coim.zr[1])
	plt.gray()
	ax = plt.axis()
	#plt.plot(rr, dd, 'r+', ms=10)
	plt.plot(xx, yy, 'o', ms=25, mec='r', lw=2, alpha=0.5)
	plt.axis(ax)
	plt.savefig('outliers.png')



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


