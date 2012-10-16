# http://vizier.u-strasbg.fr/cgi-bin/VizieR-2?-source=VII/110A

# Full RAJ2000 DEJ2000 ACO BMtype  Count  z  Rich  	Dclass  m10
# 1689	197.89	-1.37	1689	II-III:	228 	0.1810	4	6	17.6
# 2219	250.09	+46.69	2219	III	159 	 	3	6	17.4
# 2261	260.62	+32.15	2261	 	128 	 	2	6	17.4

# Abell 2219
# (250.08, 46.71)
# 1453 5 57 (dist: 5.29006 arcmin)
# -> ~ x 200-600, y 0-400
# python tractor-sdss-synth.py -r 1453 -c 5 -f 57 -b r --dr9 --roi 200 600 0 400

# Abell 2261
# python tractor-sdss-synth.py -r 2207 -c 5 -f 162 -b r --dr9

# Abell 1689
# python tractor-sdss-synth.py -r 1140 -c 6 -f 300 -b r --dr9 --roi 0 1000 600 1400
# -> WHOA, Photo is messed up there!

# Richest cluster (=5)
# Abell 665
# (127.69, 65.88)

# Brightest cluster (in SDSS footprint)
# Abell 426
# (incl NGC 1270)
# python tractor-sdss-synth.py -r 3628 -c 1 -f 103 -b i --dr9
# --> Photo's models are too bright!

# Next,
# Abell 1656
# RCF [(5115, 5, 150, 194.92231764240464, 27.884313738504037), (5087, 6, 274, 194.91114434965587, 28.095153527922157)]
# 	  aco (<type 'numpy.int16'>) 1656 dtype int16
# 	  bmtype (<type 'numpy.string_'>) II dtype |S2
# 	  count (<type 'numpy.int16'>) 106 dtype int16
# 	  dclass (<type 'numpy.uint8'>) 1 dtype uint8
# 	  dec (<type 'numpy.float64'>) 27.9807003863 dtype float64
# 	  m10 (<type 'numpy.float32'>) 13.5 dtype float32
# 	  ra (<type 'numpy.float64'>) 194.953047094 dtype float64
# 	  rich (<type 'numpy.uint8'>) 2 dtype uint8
# 	  z (<type 'numpy.float32'>) 0.0232 dtype float32
#
# python tractor-sdss-synth.py -r 5115 -c 5 -f 150 -b i --dr9
# python tractor-sdss-synth.py -r 5115 -c 5 -f 151 -b i --dr9
##### ^^^^ #### nice
# python tractor-sdss-synth.py -r 5115 -c 5 -f 151 -b i --dr9 --roi 1048 2048 0 1000

# Richness:
#   Group 0: 30-49 galaxies
#   Group 1: 50-79 galaxies
#   Group 2: 80-129 galaxies
#   Group 3: 130-199 galaxies
#   Group 4: 200-299 galaxies
#   Group 5: more than 299 galaxies

# Distance:  Abell divided the clusters into seven "distance groups" according to the magnitudes of their tenth brightest members:
#   Group 1: mag 13.3-14.0
#   Group 2: mag 14.1-14.8
#   Group 3: mag 14.9-15.6
#   Group 4: mag 15.7-16.4
#   Group 5: mag 16.5-17.2
#   Group 6: mag 17.3-18.0
#   Group 7: mag > 18.0

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

from astrometry.util.fits import *
from astrometry.util.sdss_radec_to_rcf import *
from tractor.utils import *
from tractor import sdss as st
from tractor import *

from astrometry.sdss import *


def test1():
	ps = PlotSequence('abell')

	run, camcol, field = 5115, 5, 151
	band = 'i'
	bands = [band]
	roi = (1048,2048, 0,1000)

	tim,tinf = st.get_tractor_image_dr9(run, camcol, field, band,
										roi=roi, nanomaggies=True)
	srcs = st.get_tractor_sources_dr9(run, camcol, field, band,
		roi=roi, nanomaggies=True,
		bands=bands)
	tractor = Tractor([tim], srcs)

	print tractor

	sdss = DR9()
	fn = sdss.retrieve('frame', run, camcol, field, band)
	frame = sdss.readFrame(run, camcol, field, band, filename=fn)

	sky = st.get_sky_dr9(frame)

	print tinf
	
	plt.clf()
	plt.imshow(sky, interpolation='nearest', origin='lower')
	plt.colorbar()
	ps.savefig()

	x0,x1,y0,y1 = roi
	roislice = (slice(y0,y1), slice(x0,x1))

	sky = sky[roislice]

	z0,z1 = tim.zr
	
	ima = dict(interpolation='nearest', origin='lower',
			   extent=roi)

	mn = (sky.min() + sky.max()) / 2.
	d = (z1 - z0)
	
	plt.clf()
	plt.imshow(sky, vmin=mn - d/2, vmax=mn + d/2, **ima)
	plt.colorbar()
	ps.savefig()
	
	imb = ima.copy()
	imb.update(vmin=tim.zr[0], vmax=tim.zr[1])

	plt.clf()
	plt.imshow(tim.getImage(), **imb)
	plt.colorbar()
	ps.savefig()

	mod = tractor.getModelImage(0)
	plt.clf()
	plt.imshow(mod, **imb)
	plt.colorbar()
	ps.savefig()

	tractor.freezeParam('images')

	j=0
	while True:
		print '-------------------------------------'
		print 'Optimizing all step', j
		print '-------------------------------------'
		dlnp,X,alpha = tractor.optimize()
		print 'delta-logprob', dlnp
		for src in tractor.getCatalog():
			for b in src.getBrightnesses():
				f = b.getFlux(band)
				if f < 0:
					print 'Clamping flux', f, 'up to zero'
					b.setFlux(band, 0.)
		if dlnp < 1:
			break
		j += 1

		mod = tractor.getModelImage(0)
		plt.clf()
		plt.imshow(mod, **imb)
		plt.colorbar()
		ps.savefig()
	
	mags = []
	for src in tractor.getCatalog():
		mags.append(src.getBrightness().getMag(band))
	I = np.argsort(mags)
	for i in I:
		tractor.catalog.freezeAllBut(i)
		j = 1
		while True:
			print '-------------------------------------'
			print 'Optimizing source', i, 'step', j
			print '-------------------------------------'
			print tractor.catalog[i]
			dlnp,X,alpha = tractor.optimize()
			print 'delta-logprob', dlnp

			for b in tractor.catalog[i].getBrightnesses():
				f = b.getFlux(band)
				if f < 0:
					print 'Clamping flux', f, 'up to zero'
					b.setFlux(band, 0.)

			print tractor.catalog[i]
			print
			if dlnp < 1:
				break
			j += 1

			mod = tractor.getModelImage(0)
			plt.clf()
			plt.imshow(mod, **imb)
			plt.colorbar()
			ps.savefig()
		tractor.catalog.thawAllParams()

		
	mod = tractor.getModelImage(0)
	plt.clf()
	plt.imshow(mod, **imb)
	plt.colorbar()
	ps.savefig()
	

	
	
if __name__ == '__main__':
	import pylab as plt
	import numpy as np

	#find()
	test1()

	
def find():
	cmap = {'_RAJ2000':'ra', '_DEJ2000':'dec', 'ACOS':'aco'}
	T1 = fits_table('abell.fits', column_map=cmap)
	#T1.about()
	T2 = fits_table('abell2.fits', column_map=cmap)
	#T2.about()
	T3 = fits_table('abell3.fits', column_map=cmap)
	#T3.about()
	#T3.rename('acos', 'aco')
	T = merge_tables([T1, T2, T3])
	T.about()
	#T.rename('_raj2000', 'ra')
	#T.rename('_dec2000', 'dec')

	ps = PlotSequence('abell')
	
	plt.clf()
	plt.hist(T.rich, bins=np.arange(-0.5,max(T.rich)+0.5))
	plt.xlabel('Richness')
	ps.savefig()

	T5 = T[T.rich == 5]
	print 'Richness 5:', len(T5)
	T5[0].about()

	#plt.clf()
	#plt.hist(T.dclass
	I = np.argsort(T.m10)
	Tm = T[I]
	print Tm.m10[:20]
	urls = []
	for Ti in Tm[:20]:
		print 'ACO', Ti.aco
		rcf = radec_to_sdss_rcf(Ti.ra, Ti.dec, contains=True,
								tablefn='dr9fields.fits')
		if len(rcf) == 0:
			continue
		print 'RCF', rcf
		Ti.about()
		run,camcol,field,nil,nil = rcf[0]
		
		getim = st.get_tractor_image_dr9
		getsrc = st.get_tractor_sources_dr9
		bandname = 'i'
		tim,tinf = getim(run, camcol, field, bandname)
		sources = getsrc(run, camcol, field, bandname)
		tractor = Tractor([tim], sources)
		mod = tractor.getModelImage(0)

		urls.append('http://skyserver.sdss3.org/dr8/en/tools/chart/navi.asp?ra=%f&dec=%f' % (Ti.ra, Ti.dec))
		
		plt.clf()
		plt.imshow(mod, interpolation='nearest', origin='lower',
				   vmin=tim.zr[0], vmax=tim.zr[1])
		plt.gray()
		ps.savefig()
		plt.clf()
		plt.imshow(tim.getImage(),
				   interpolation='nearest', origin='lower',
				   vmin=tim.zr[0], vmax=tim.zr[1])
		plt.gray()
		ps.savefig()

	print '\n'.join(urls)
