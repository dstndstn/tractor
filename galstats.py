import os
import sys
import matplotlib
matplotlib.use('Agg')
from astrometry.util.pyfits_utils import *
from astrometry.sdss import *
import pylab as plt
import numpy as np

def plots1():

	"""
	select fracdev_r,exprad_r,expab_r,expmag_r,expphi_r,run,camcol,field into mydb.exp3818
	from Galaxy where run=3818
	and clean=1 and probpsf=0
	and (flags & (dbo.fPhotoFlags('BINNED1'))) = (dbo.fPhotoFlags('BINNED1'))
	and (flags & (dbo.fPhotoFlags('INTERP') | dbo.fPhotoFlags('BLENDED') |
    dbo.fPhotoFlags('BINNED2') | dbo.fPhotoFlags('MOVED') | dbo.fPhotoFlags('DEBLENDED_AT_EDGE') |
    dbo.fPhotoFlags('MAYBE_CR') | dbo.fPhotoFlags('MAYBE_EGHOST'))) = 0
    and fracdev_r < 0.5


	select fracdev_r,exprad_r,expab_r,expmag_r,expphi_r,run,camcol,field,ra,dec into mydb.exp3818
	from Galaxy where
	run=3818
	and camcol=2
	and clean=1 and probpsf=0
	and (flags & (dbo.fPhotoFlags('BINNED1'))) = (dbo.fPhotoFlags('BINNED1'))
	and (flags & (dbo.fPhotoFlags('INTERP') | dbo.fPhotoFlags('BLENDED') |
    dbo.fPhotoFlags('BINNED2') | dbo.fPhotoFlags('MOVED') | dbo.fPhotoFlags('DEBLENDED_AT_EDGE') |
    dbo.fPhotoFlags('MAYBE_CR') | dbo.fPhotoFlags('MAYBE_EGHOST'))) = 0
	and fracdev_r < 0.5
	and exprad_r between 0.4 and 0.55
	and expmag_r between 21 and 22
	=> exp3818b.fits
	
	"""
	
	T=fits_table('exp3818_dstn.fit')

	plt.clf()
	n,b,p = plt.hist(np.log10(T.exprad_r), 100, range=(np.log10(0.1), np.log10(10.)))
	plt.xlabel('log r_e')
	plt.ylabel('number of galaxies')
	plt.title('Run 3818: total %i' % sum(n))
	plt.savefig('rad1.png')

	plt.clf()
	plt.hist(T.exprad_r, 100, range=(0.4, 0.55))
	plt.xlim(0.4, 0.55)
	plt.savefig('rad2.png')

	plt.clf()
	H,xe,ye = np.histogram2d(T.exprad_r, T.expmag_r, 200, range=((0.4,0.55),(20,22)))
	plt.imshow(H.T, extent=(xe[0],xe[-1],ye[0],ye[-1]), aspect='auto', origin='lower', interpolation='nearest')
	plt.colorbar()
	plt.savefig('rad-mag.png')

	T2 = T[(T.expmag_r > 21) * (T.expmag_r < 22)]
	print len(T2), 'in mag 21-22 range'
	T2 = T2[(T2.exprad_r >= 0.4) * (T2.exprad_r <= 0.55)]
	print len(T2), 'in radius 0.4-0.55 range'

	#for cc in np.unique(T2.camcol):
	#	n = sum(T2.camcol == cc)
	#	print n, 'in camcol', cc

	T2 = T2[T2.camcol == 2]
	print len(T2), 'in camcol 2'
	print 'Fields:', min(T2.field), max(T2.field)
	T2.about()
	plt.clf()
	H,xe,ye = np.histogram2d(T2.exprad_r, T2.expmag_r, 200, range=((0.4,0.55),(21,22)))
	plt.imshow(H.T, extent=(xe[0],xe[-1],ye[0],ye[-1]), aspect='auto', origin='lower', interpolation='nearest')
	plt.colorbar()
	plt.savefig('rad-mag2.png')

	plt.clf()
	n,b,p = plt.hist(T2.exprad_r, 100, range=(0.4, 0.55))
	print sum(n), 'counts plotted'
	plt.xlim(0.4, 0.55)
	plt.title('Run 3818, Camcol 2: total %i' % sum(n))
	plt.xlabel('exp r_e (arcsec)')
	plt.ylabel('number of galaxies')
	plt.savefig('rad3.png')

	plt.clf()
	plt.hist(T2.exprad_r, 300, range=(0.4, 0.55))
	plt.xlim(0.4, 0.55)
	plt.savefig('rad4.png')

	T2 = T2[(T2.field >= 100) * (T2.field < 200)]
	print len(T2), 'in fields [100, 200)'

	plt.clf()
	n,b,p = plt.hist(T2.exprad_r, 100, range=(0.4, 0.55))
	plt.xlim(0.4, 0.55)
	plt.title('Run 3818, Camcol 2, Fields 100-200: total %i' % sum(n))
	plt.xlabel('exp r_e (arcsec)')
	plt.ylabel('number of galaxies')
	plt.savefig('rad5.png')

def plots2():
	T = fits_table('exp3818i_dstn.fit')

	plt.clf()
	H,xe,ye = np.histogram2d(T.exprad_i, T.expmag_i, 200, range=((0.1,10),(15,22)))
	plt.imshow(H.T, extent=(xe[0],xe[-1],ye[0],ye[-1]), aspect='auto', origin='lower', interpolation='nearest')
	plt.colorbar()
	plt.savefig('radi-mag1.png')

	plt.clf()
	n,b,p = plt.hist(np.log10(T.exprad_i), 100, range=(np.log10(0.1), np.log10(10.)))
	plt.xlabel('log r_e')
	plt.ylabel('number of galaxies')
	plt.title('Run 3818, i-band, total %i' % sum(n))
	plt.savefig('radi1.png')

	plt.clf()
	n,b,p = plt.hist(T.exprad_i, 100, range=(1.3, 1.5))
	plt.xlabel('r_e (arcsec)')
	plt.ylabel('number of galaxies')
	plt.title('Run 3818, i-band, total %i' % sum(n))
	plt.xlim(1.3, 1.5)
	plt.savefig('radi2.png')

	T2 = T[(T.exprad_i >= 1.3) * (T.exprad_i <= 1.5)]
	print len(T2), 'in radius range'
	T2 = T2[T2.expmag_i < 21]
	print len(T2), 'in mag range'

	T2.band = np.array(['i']*len(T2))

	T2.writeto('exp3818c.fits')

	plt.clf()
	n,b,p = plt.hist(T2.exprad_i, 100, range=(1.3, 1.5))
	plt.xlabel('r_e (arcsec)')
	plt.ylabel('number of galaxies')
	plt.title('Run 3818, i-band, mag < 21, total %i' % sum(n))
	plt.xlim(1.3, 1.5)
	plt.savefig('radi3.png')

	plt.clf()
	H,xe,ye = np.histogram2d(T2.exprad_i, T2.expmag_i, 200, range=((1.3,1.5),(15,22)))
	plt.imshow(H.T, extent=(xe[0],xe[-1],ye[0],ye[-1]), aspect='auto', origin='lower', interpolation='nearest')
	plt.colorbar()
	plt.savefig('radi-mag2.png')

	

if __name__ == '__main__':
	#plots1()
	#plots2()
	#sys.exit(0)
	
	from tractor.sdss import *
	from tractor import *

	#T = fits_table('exp3818b.fits')
	T = fits_table('exp3818c.fits')
	print len(T), 'galaxies'

	sdss = DR7()
	dn = 'paper0-data'
	if not os.path.exists(dn):
		os.mkdir(dn)
	sdss.setBasedir(dn)

	#band = 'r'

	# radius in pixels of ROI; max r_e = 0.55 arcsec, 8 r_e
	#sz = 0.55 * 8 / 0.396
	sz = 1.5 * 8 / 0.396
	tractors = []
	for i,(run,camcol,field,ra,dec) in enumerate(zip(T.run, T.camcol, T.field, T.ra, T.dec)):
		if hasattr(T, 'band'):
			band = T.band[i]
		else:
			band = 'r'
		print
		print 'Galaxy', (i+1), 'of', len(T)
		im,info = get_tractor_image(run, camcol, field, band, sdssobj=sdss, useMags=True,
									roiradecsize=(ra,dec,sz))
		sky,skysig = info['sky'],info['skysig']
		pa = dict(origin='lower', interpolation='nearest', vmin=sky-3*skysig, vmax=sky+10*skysig)
		plt.clf()
		plt.imshow(im.data, **pa)
		plt.savefig('gs-data-%03i.png' % i)

		roi = info['roi']
		srcs = get_tractor_sources(run, camcol, field, band, bands=[band], sdss=sdss, roi=roi)
		tractor = Tractor([im], srcs)
		synth = tractor.getModelImage(im)
		plt.clf()
		plt.imshow(synth, **pa)
		plt.savefig('gs-synth-%03i.png' % i)
		tractors.append(tractor)
