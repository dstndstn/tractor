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
import numpy as np
import pylab as plt
import scipy.interpolate
import os

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.plotutils import ArcsinhNormalize, plothist, antigray
from astrometry.util.starutil_numpy import *
import astrometry.libkd.spherematch as sm
from astrometry.sdss import *

from astrometry.util.stages import *

from tractor.utils import *
from tractor import sdss as st
from tractor import *
from tractor.sdss_galaxy import *
from tractor.splinesky import SplineSky

def get_dm_table():
	from astrometry.util import casjobs
	import os
	casjobs.setup_cookies()
	cas = casjobs.get_known_servers()['dr9']
	username = os.environ['SDSS_CAS_USER']
	password = os.environ['SDSS_CAS_PASS']
	cas.login(username, password)
	sqls = []
	t = []
	for i,z in enumerate(np.linspace(0., 1., 1001)):
		#t.append('dbo.fCosmoDistanceModulus(%.4f, 0.3, 0.7, DEFAULT, DEFAULT, 0.7) as DM%04i' % (z,i))
		t.append('dbo.fCosmoDl(%.4f, 0.3, 0.7, DEFAULT, DEFAULT, 0.7) as DL%04i' % (z,i))

		if i % 500 == 0 and len(t)>1:
			sqls.append(t)
			t = []
	print 't=',t
	if len(t):
		sqls.append(t)

	dbnames = []
	for j,t in enumerate(sqls):
		dbname = 'dlz%i' % j
		sql = 'select into mydb.%s\n' % dbname + ',\n'.join(t) + '\n'
		jid = cas.submit_query(sql)
		print 'Submitted job id', jid
		while True:
			jobstatus = cas.get_job_status(jid)
			print 'Job id', jid, 'is', jobstatus
			if jobstatus in ['Finished', 'Failed', 'Cancelled']:
				break
			print 'Sleeping...'
			time.sleep(5)
		dbnames.append(dbname)
	dodelete = True
	tabfns = [nm + '.fits' for nm in dbnames]
	cas.output_and_download(dbnames, tabfns, dodelete)

	DL = []
	Z = []
	for tabfn in tabfns:
		print 'Reading', tabfn
		T = fits_table(tabfn)
		for c in T.get_columns():
			print 'col', c
			if c.startswith('dl') and len(c) == 6:
				z = float(c[2] + '.' + c[3:])
				Z.append(z)
				dl = T.get(c)[0]
				DL.append(dl)
	T = tabledata()
	T.dl = DL
	T.z = Z
	T.writeto('dlz.fits')
	plt.clf()
	plt.plot(Z, DL, 'k-')
	plt.savefig('dlz.png')

class LuminosityDistance(object):
	def __init__(self):
		T = fits_table('dlz.fits')
		self.spline = scipy.interpolate.InterpolatedUnivariateSpline(T.z, T.dl)
	def __call__(self, z):
		return self.spline(z)

def compile_nyu_vagc(bandnum):
	# Compile NYU-VAGC values
	print 'Compiling NYU-VAGC catalog...'
	nyudir = 'nyu-vagc-2'
	Tcat = fits_table(os.path.join(nyudir, 'object_catalog.fits'))
	I = np.flatnonzero((Tcat.sdss_imaging_position != -1) * (Tcat.sdss_spectro_position != -1))
	print 'Found', len(I), 'rows with spectra and imaging'
	del Tcat

	print 'Reading imaging...'
	Tim = fits_table(os.path.join(nyudir, 'object_sdss_imaging.fits'),
					 rows=I, columns=['run','camcol','field','id',
									  'devflux'])
	print Tim
	Tim.about()
	print 'Reading spec...'
	Tspec = fits_table(os.path.join(nyudir, 'object_sdss_spectro.fits'),
					   rows=I, columns=['vdisp','z', 'sn_median', 'zwarning', 'objtype',
										'class', 'subclass'],
						column_map={'class':'clazz'})
	print Tspec
	Tspec.about()
	print 'Reading kcorrect...'
	Tk = fits_table(os.path.join(nyudir, 'kcorrect', 'kcorrect.nearest.model.z0.00.fits'),
					rows=I, columns=['absmag','kcorrect','z'])
	print Tk
	Tk.about()

	Tnyu = tabledata()
	Tnyu.sigma = Tspec.vdisp

	Tnyu.objtype = Tspec.objtype
	Tnyu.clazz = Tspec.clazz
	Tnyu.subclass = Tspec.subclass
	Tnyu.sn_median = Tspec.sn_median
	Tnyu.zwarning = Tspec.zwarning
	
	#Tnyu.z = Tspec.z
	Tnyu.z = Tk.z
	# abs mag
	Tnyu.M = Tk.absmag[:,bandnum]
	# K-correction
	Tnyu.Kz = Tk.kcorrect[:,bandnum]

	Tnyu.devflux = Tim.devflux[:,bandnum]

	for b in ['g','r','i','z']:
		i = band_index(b)
		Tnyu.set('devflux_'+b, Tim.devflux[:,i])
	
	Tnyu.rdev = np.zeros(len(Tnyu))
	Tnyu.ab   = np.zeros(len(Tnyu))

	bands = ['g','r','i','z']
	binds = [band_index(b) for b in bands]
	for b in bands:
		Tnyu.set('rdev_'+b, np.zeros(len(Tnyu)))
	
	Tnyu.r90i = np.zeros(len(Tnyu))
	Tnyu.r50i = np.zeros(len(Tnyu))
	Tnyu.exp_lnl = np.zeros(len(Tnyu))
	Tnyu.dev_lnl = np.zeros(len(Tnyu))
	Tnyu.prob_psf = np.zeros(len(Tnyu))
	Tnyu.fracpsf = np.zeros(len(Tnyu))
	
	iband = band_index('i')

	#gband = band_index('g')
	#rband = band_index('r')
	#zband = band_index('z')
	
	for run,camcol in np.unique(zip(Tim.run, Tim.camcol)):
		print 'Run', run, 'camcol', camcol
		J = np.flatnonzero((Tim.run == run) * (Tim.camcol == camcol))
		Tc = fits_table(os.path.join(nyudir, 'sdss', 'parameters',
									 'calibObj-%06i-%i.fits' % (run, camcol)),
									 columns=['r_dev','ab_dev','field','id',
											  'exp_lnl', 'dev_lnl', 'fracpsf', 'prob_psf',
											  'petror50', 'petror90'])
		for j in J:
			K = np.flatnonzero((Tc.field == Tim.field[j]) * (Tc.id == Tim.id[j]))
			assert(len(K) == 1)
			K = K[0]
			Tnyu.rdev[j] = Tc.r_dev [K, bandnum]
			Tnyu.ab[j]   = Tc.ab_dev[K, bandnum]

			for b,i in zip(bands,binds):
				Tnyu.get('rdev_'+b)[j] = Tc.r_dev[K, i]
			#Tnyu.rdev_r[j] = Tc.r_dev [K, rband]
			#Tnyu.rdev_z[j] = Tc.r_dev [K, zband]
			
			Tnyu.r90i[j] = Tc.petror90[K, iband]
			Tnyu.r50i[j] = Tc.petror50[K, iband]
			Tnyu.exp_lnl[j] = Tc.exp_lnl[K, bandnum]
			Tnyu.dev_lnl[j] = Tc.dev_lnl[K, bandnum]
			Tnyu.prob_psf[j] = Tc.prob_psf[K, bandnum]
			Tnyu.fracpsf[j] = Tc.fracpsf[K, bandnum]
			
	return Tnyu
	
	
def fp():
	band = 'i'
	ps = PlotSequence('fp')
	
	run, camcol, field = 5115, 5, 151
	bands = [band]
	roi = (1048,2048, 0,1000)
	tim,tinf = st.get_tractor_image_dr9(run, camcol, field, band,
										roi=roi, nanomaggies=True)
	ima = dict(interpolation='nearest', origin='lower',
			   extent=roi)
	zr2 = tinf['sky'] + tinf['skysig'] * np.array([-3, 100])
	#imb = ima.copy()
	#imb.update(vmin=tim.zr[0], vmax=tim.zr[1])
	imc = ima.copy()
	imc.update(norm=ArcsinhNormalize(mean=tinf['sky'], 
									 std=tinf['skysig']),
				vmin=zr2[0], vmax=zr2[1])


	

	# Match spectra with Abell catalog
	T = fits_table('a1656-spectro.fits', column_map={'class':'clazz'})

	IJ = []
	cats = []
	for step in [0, 12]:
		tractor = unpickle_from_file('clustersky-%02i.pickle' % step)
		cat = tractor.getCatalog()
		cat = [src for src in cat if src.getBrightness().getMag(band) < 20]
		cats.append(cat)
		rd = [src.getPosition() for src in cat]
		ra  = np.array([p.ra  for p in rd])
		dec = np.array([p.dec for p in rd])
		rad = 1./3600.
		I,J,d = sm.match_radec(T.ra, T.dec, ra, dec, rad,
							   nearest=True)
		print len(I), 'matches on RA,Dec'
		#print I, J
		IJ.append((I,J))

	(I1,J1),(I2,J2) = IJ
	assert(np.all(I1 == I2))
	cat1 = [cats[0][j] for j in J1]
	cat2 = [cats[1][j] for j in J2]
	m1 = np.array([src.getBrightness().getMag(band) for src in cat1])
	m2 = np.array([src.getBrightness().getMag(band) for src in cat2])

	plt.clf()
	plt.plot(m1, m2, 'k.')
	plt.xlabel('SDSS i mag')
	plt.ylabel('Tractor i mag')
	ax = plt.axis()
	mn,mx = min(ax[0],ax[2]), max(ax[1],ax[3])
	plt.plot([mn,mx],[mn,mx], 'k-', alpha=0.5)
	plt.axis(ax)
	ps.savefig()

	plt.clf()
	plt.plot(m1, m2-m1, 'k.')
	plt.xlabel('SDSS i mag')
	plt.ylabel('(Tractor - SDSS) i mag')
	plt.axhline(0, color='k', alpha=0.5)
	ps.savefig()

	sdss = DR9()
	p = sdss.readPhotoObj(run, camcol, field)
	print 'PhotoObj:', p
	objs = p.getTable()
	# from tractor.sdss.get_tractor_sources
	x0,x1,y0,y1 = roi
	bandnum = band_index(band)
	x = objs.colc[:,bandnum]
	y = objs.rowc[:,bandnum]
	I = ((x >= x0) * (x < x1) * (y >= y0) * (y < y1))
	objs = objs[I]
	# Only deblended children.
	objs = objs[(objs.nchild == 0)]
	#objs.about()
	I3,J3,d = sm.match_radec(T.ra, T.dec, objs.ra, objs.dec, rad,
							 nearest=True)
	print 'Got', len(I3), 'matches to photoObj'
	cat3 = Catalog()
	for j in J3:
		phi = -objs.phi_dev_deg[j, bandnum]
		ab  =  objs.     ab_dev[j, bandnum]
		re  =  objs.  theta_dev[j, bandnum]
		flux =     objs.devflux[j, bandnum]
		dshape = GalaxyShape(re, ab, phi)
		dbright = NanoMaggies(**{band: flux})
		pos = RaDecPos(objs.ra[j], objs.dec[j])
		gal = DevGalaxy(pos, dbright, dshape)
		cat3.append(gal)

	# T.about()
	print 'Spectro types:', T.sourcetype[I1]
	print 'Spectro classes:', T.clazz[I1]
	print 'Spectro subclasses:', T.subclass[I1]

	DL = LuminosityDistance()

	log = np.log10
	

	cutfn = 'nyu-vagc-cut.fits'
	if not os.path.exists(cutfn):
		fn = 'nyu-vagc.fits'
		if not os.path.exists(fn):
			Tnyu = compile_nyu_vagc(bandnum)
			Tnyu.writeto(fn)
		else:
			Tnyu = fits_table(fn, lower=False)

		#Tnyu.cut((Tnyu.z > 0) * (Tnyu.z <= 1.))
		# Bernardi-like cuts
		print 'Got', len(Tnyu), 'NYU-VAGC objects'
		Tnyu.cut((Tnyu.z > 0) * (Tnyu.z <= 0.3))
		print 'Cut on redshift:', len(Tnyu)
		Tnyu.cut(Tnyu.clazz == 'GALAXY')
		print 'Cut on GALAXY:', len(Tnyu)
		print 'subclasses:', np.unique(Tnyu.subclass)
		#Tnyu.cut(Tnyu.subclass == '')
		#print 'Cut on plain GALAXY:', len(Tnyu)
		Tnyu.cut(Tnyu.r90i / Tnyu.r50i > 2.5)
		print 'Cut on concentration index:', len(Tnyu)
		Tnyu.cut(Tnyu.dev_lnl > Tnyu.exp_lnl)
		print 'Cut on deV lnl:', len(Tnyu)
		Tnyu.cut(Tnyu.sn_median > 10.)
		print 'Cut on SN:', len(Tnyu)
		Tnyu.writeto(cutfn)
	else:
		Tnyu = fits_table(cutfn, lower=False)

	h70 = 0.7

	# -> arcsec
	pixscale = 0.396

	Tnyu.rdev *= pixscale
	Tnyu.rdev_g *= pixscale
	Tnyu.rdev_r *= pixscale
	Tnyu.rdev_i *= pixscale
	Tnyu.rdev_z *= pixscale

	### HACK -- unexplained scale difference between us and Bernardi.
	#FUDGE = 1. / 0.85 / 1.07
	FUDGE = 1.1
	Tnyu.rdev *= FUDGE
	Tnyu.rdev_g *= FUDGE
	Tnyu.rdev_r *= FUDGE
	Tnyu.rdev_i *= FUDGE
	Tnyu.rdev_z *= FUDGE
	
	Tnyu.DLz = DL(Tnyu.z)
	Tnyu.DMz = 5. * log(Tnyu.DLz * 1e6 / 10.)
	Tnyu.DAz = Tnyu.DLz / ((1.+Tnyu.z)**2)

	#Tnyu.r0 = Tnyu.rdev * Tnyu.ab
	#Tnyu.r0 = Tnyu.rdev * np.sqrt(Tnyu.ab)
	#Tnyu.r0 = Tnyu.rdev
	Tnyu.r0a = Tnyu.rdev * Tnyu.ab
	Tnyu.r0b = Tnyu.rdev * np.sqrt(Tnyu.ab)
	Tnyu.r0c = Tnyu.rdev
	Tnyu.R0a = arcsec2rad(Tnyu.r0a) * Tnyu.DAz
	Tnyu.R0b = arcsec2rad(Tnyu.r0b) * Tnyu.DAz
	Tnyu.R0c = arcsec2rad(Tnyu.r0c) * Tnyu.DAz
	Tnyu.r0 = Tnyu.r0b
	Tnyu.R0 = Tnyu.R0b

	#Tnyu.R0 = arcsec2rad(Tnyu.r0) * Tnyu.DAz
	# hack
	#Tnyu.mdev = Tnyu.M + Tnyu.DMz + Tnyu.Kz

	# absmag from Kcorrect file
	Tnyu.M2 = Tnyu.M
	
	Tnyu.mdev = 22.5 - 2.5*log(Tnyu.devflux)
	Tnyu.M = Tnyu.mdev - Tnyu.DMz - Tnyu.Kz

	print 'M2 - M:', np.mean(Tnyu.M2 - Tnyu.M)
	#MFUDGE = np.mean(Tnyu.M2 - Tnyu.M)

	#MFUDGE = 5.*log(h70)
	#print 'ADDING MAG FUDGE OF', MFUDGE
	#Tnyu.mdev += MFUDGE
	#Tnyu.M = Tnyu.mdev - Tnyu.DMz - Tnyu.Kz
	#print 'M2 - M now:', np.mean(Tnyu.M2 - Tnyu.M)

	Tnyu.mu0 = (Tnyu.mdev + 2.5 * log(2. * np.pi * Tnyu.r0**2)
				- Tnyu.Kz - 10.* log(1. + Tnyu.z))
	
	plt.clf()
	plt.hist(Tnyu.z, 100, range=(0,0.3))
	plt.xlabel('z')
	ps.savefig()

	plt.clf()
	plt.hist(Tnyu.Kz, 100)
	plt.xlabel('K-correction')
	ps.savefig()

	# Build a spline approximation to the median K-correction in the NYU-VAGC
	edges = np.arange(0, 0.3, 0.01)
	xx,yy = [],[]
	for e0,e1 in zip(edges[:-1], edges[1:]):
		I = np.flatnonzero((Tnyu.z >= e0) * (Tnyu.z < e1))
		if len(I) == 0:
			continue
		mid = (e0 + e1) / 2.
		kmed = np.median(Tnyu.Kz[I])
		xx.append(mid)
		yy.append(kmed)
	NYUK = scipy.interpolate.InterpolatedUnivariateSpline(xx,yy)
		
	plt.clf()
	pha = dict(docolorbar=False,
			   dohot=False, imshowargs=dict(cmap=antigray))
	plothist(Tnyu.z, Tnyu.Kz, range=((0,0.3),(-0.1,0.4)), **pha)
	zz = np.linspace(0, 0.3, 300)
	plt.plot(zz, NYUK(zz), 'r-')
	plt.xlabel('z')
	plt.ylabel('K-correction')
	plt.axis([0, 0.3, -0.1, 0.4])
	ps.savefig()

	plt.clf()
	plt.subplot(2,2,1)
	plt.hist(Tnyu.rdev_g, 50, range=(0, 10))
	plt.xlabel('g (mean: %g)' % np.mean(Tnyu.rdev_g))
	plt.subplot(2,2,2)
	plt.hist(Tnyu.rdev_r, 50, range=(0, 10))
	plt.xlabel('r (mean: %g)' % np.mean(Tnyu.rdev_r))
	plt.subplot(2,2,3)
	plt.hist(Tnyu.rdev, 50, range=(0, 10))
	plt.xlabel('%s (mean: %g)' % (band, np.mean(Tnyu.rdev)))
	plt.subplot(2,2,4)
	plt.hist(Tnyu.rdev_z, 50, range=(0, 10))
	plt.xlabel('z (mean: %g)' % np.mean(Tnyu.rdev_z))
	ps.savefig()

	bands = ['g','r','i','z']
	plt.clf()
	for i,b in enumerate(bands):
		plt.subplot(2,2,i+1)
		plothist(Tnyu.z, 22.5 - 2.5*log(Tnyu.get('devflux_'+b)),
				 range=((0,0.3),(13,20)), doclf=False, **pha)
		plt.xlabel('redshift')
		plt.ylabel('mdev ' + b)
	ps.savefig()

	pha = dict(docolorbar=False, doclf=False,
			   dohot=False, imshowargs=dict(cmap=antigray))
	plt.clf()
	plt.suptitle('Bernardi paper 1 fig 1')
	plt.subplots_adjust(left=0.2, right=0.8, wspace=0, hspace=0)
	plt.subplot(3,2,1)
	plothist(Tnyu.ab, Tnyu.rdev_r, range=((0,1),(-1, 11)), **pha)
	plt.ylabel('r_dev (arcsec)')
	plt.xticks([])
	
	R0r = arcsec2rad(Tnyu.rdev_r * np.sqrt(Tnyu.ab)) * Tnyu.DAz
	R0r *= (1e3 * h70)
	I = (R0r > 0)
	plt.subplot(3,2,2)
	plt.yticks([])
	plt.twinx()
	plothist(Tnyu.ab[I], log(R0r[I]), range=((0,1),(-0.7, 1.7)), **pha)
	plt.ylabel('log R_0 [kpc/h70]')
	plt.xticks([])
	
	plt.subplot(3,2,3)
	plothist(Tnyu.ab, log(Tnyu.sigma), range=((0,1),(1.8,2.7)), **pha)
	plt.ylabel('log sigma')
	plt.xticks([])

	plt.subplot(3,2,4)
	plt.yticks([])
	plt.twinx()
	plothist(Tnyu.ab, Tnyu.z, range=((0,1),(-0.025,0.375)), **pha)
	plt.ylabel('z')
	plt.xticks([])
	
	plt.subplot(3,2,6)
	plt.yticks([])
	plt.twinx()
	plt.hist(Tnyu.ab, 50, range=(0,1))
	plt.xlabel('ab')
	ps.savefig()

	# revert to defaults
	plt.subplots_adjust(left=0.125, right=0.9, wspace=0.2, hspace=0.2)
	
	
	# plt.clf()
	# plt.plot(Tnyu.ab, Tnyu.rdev, 'k,')
	# plt.xlabel('ab')
	# plt.ylabel('r_dev (arcsec)')
	# plt.ylim(-1, 11)
	# ps.savefig()
	
	# plt.clf()
	# I = (Tnyu.R0 > 0)
	# plothist(Tnyu.ab[I], log(Tnyu.R0[I] * 1e3),
	# 		 range=((0,1),(-0.7, 1.7)), **pha)
	# #plothist(Tnyu.ab[I], log(Tnyu.R0[I] * 1e3), **pha)
	# plt.xlabel('ab')
	# plt.ylabel('log R_0')
	# ps.savefig()

	if True:
		plt.clf()
		for i,(nm,R0) in enumerate([
				('r_dev * ab', Tnyu.R0a),
				('r_dev * sqrt(ab)', Tnyu.R0b),
				('r_dev', Tnyu.R0c)]):
			plt.subplot(2,2, i+1)
			I = (R0 > 0)
			plothist(Tnyu.ab[I], log(R0[I] * 1e3 / h70),
					 range=((0,1),(-0.7, 1.7)), **pha)
			plt.xlabel('ab')
			plt.ylabel('log R_0 [kpc/h70]')
			plt.title('r0 = %s' % nm)
		ps.savefig()

	if False:
		plt.clf()
		zz = np.linspace(0., 1., 1000)
		DLz = DL(zz)
		plt.plot(zz, DLz, 'k-')
		plt.ylabel('DL [Mpc]')
		plt.xlabel('z')
		ps.savefig()

		plt.clf()
		DMz = 5. * log(DLz * 1e6 / 10.)
		plt.plot(zz, DMz, 'k-')
		plt.ylabel('DM [mag]')
		plt.xlabel('z')
		ps.savefig()
			
		plt.clf()
		DAz = DLz / ((1. + zz)**2)
		plt.plot(zz, DAz, 'k-')
		plt.ylabel('DA [Mpc/radian ?]')
		plt.xlabel('z')
		ps.savefig()



	TT = []
	for I,cat,nm in [(I1,cat1,'SDSS'),(I3,cat3,'SDSS-forced'),(I2,cat2,'Tractor')]:
		m0 = np.array([src.getBrightness().getMag(band) for src in cat])
		wcs = tim.getWcs()
		x0,y0 = wcs.x0,wcs.y0
		xy0 = np.array([wcs.positionToPixel(src.getPosition())
						for src in cat])
		
		origI = I
		keep = []
		keepI = []
		for i,src in zip(I,cat):
			if type(src) is CompositeGalaxy:
				devflux = src.brightnessDev.getFlux(band)
				expflux = src.brightnessExp.getFlux(band)
				df = devflux / (devflux + expflux)
				print '  dev fraction', df
				if df > 0.8:
					keep.append(src)
					keepI.append(i)
			if type(src) in [DevGalaxy]:
				keep.append(src)
				keepI.append(i)
		I,cat = keepI,keep
		m1 = np.array([src.getBrightness().getMag(band) for src in cat])
		xy1 = np.array([wcs.positionToPixel(src.getPosition())
						for src in cat])

		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		ax = plt.axis()
		plt.gray()
		plt.colorbar()
		p1 = plt.plot(xy0[:,0]+x0, xy0[:,1]+y0, 'o', mec='m', mfc='none',
					  ms=8, mew=1)
		p2 = plt.plot(xy1[:,0]+x0, xy1[:,1]+y0, 'o', mec='r', mfc='none',
					  ms=10, mew=2)
		plt.legend((p1[0],p2[0]), ('Spectro sources', 'deV profiles'))
		for (x,y),i in zip(xy0, origI):
			plt.text(x+x0, y+y0 + 20,
					 '%.03f' % T.z[i], color='r', size=8,
				ha='center', va='bottom')
		plt.axis(ax)
		plt.title(nm)
		ps.savefig()

		## Compile measurements into tables.
		vals = []
		for i,src in zip(I,cat):
			if type(src) is CompositeGalaxy:
				shape = src.shapeDev
				# this is total exp + dev
				mag = src.getBrightness()
			elif type(src) is DevGalaxy:
				shape = src.shape
				mag = src.getBrightness()
			# major axis [arcsec]
			rdev = shape.re
			# b/a axis ratio
			ab = shape.ab
			# deV mag [mag]
			mdev = mag.getMag(band)
			# redshift
			z = T.z[i]
			# FIXME!!!  K-correction(z)
			#Kz = 0.
			Kz = NYUK(z)

			# velocity dispersion - sigma [km/s]
			sigma = T.veldisp[i]
			vals.append((rdev, ab, mdev, z, Kz, sigma))
		vals = np.array(vals)
			
		Ti = tabledata()
		for j,col in enumerate(['rdev','ab','mdev','z','Kz', 'sigma']):
			Ti.set(col, vals[:,j])

		# Derived values

		# Bernardi paper 1, pg 1823, first paragraph.
		# This seems nutty as squirrel poo to me...
		#Ti.r0 = Ti.rdev * Ti.ab
		Ti.r0 = Ti.rdev * np.sqrt(Ti.ab)

		# FIXME -- r0 to R0 correction via a poorly-described
		# correction similar to K-correction to size in a given
		# band; claimed 4-10% correction in radius.
			
		# "effective surface brightness" [mag/arcsec**2]
		Ti.mu0 = (Ti.mdev + 2.5 * log(2. * np.pi * Ti.r0**2) - Ti.Kz
				  - 10.* log(1. + Ti.z))

		# "mean surface brightness within effective radius R0"
		Ti.I0 = 10.**(Ti.mu0 / -2.5)

		# Luminosity distance(z) [megaparsecs]
		Ti.DLz = DL(Ti.z)
		Ti.DMz = 5. * log(Ti.DLz * 1e6 / 10.)

		# absolute mag
		Ti.M = Ti.mdev - Ti.DMz - Ti.Kz

		# R0: want "proper size" aka "angular diameter distance"
		#  D_L = [10. pc] * 10. ** [DM / 5.]
		#  D_A = D_L / (1+z)**2
		# where DM is the distance modulus,
		#       D_L is the luminosity distance, and
		#       D_A is the angular diameter distance.
		Ti.DAz = Ti.DLz / ((1.+Ti.z)**2)
		# R0 [Mpc]
		Ti.R0 = arcsec2rad(Ti.r0) * Ti.DAz
		TT.append(Ti)

		
	# FP
	for Ti,nm in zip(TT + [Tnyu],
					 ['SDSS', 'SDSS-forced', 'Tractor'] + ['NYU-VAGC']):
		MM = Ti.M
		SS = Ti.sigma
		RR = Ti.R0
		MU = Ti.mu0
		
		def plot_nyu(x, y, xr, yr):
			plothist(x, y, range=((xr,yr)), docolorbar=False,
					 dohot=False, imshowargs=dict(cmap=antigray))
		
		def plot_sdss(x, y, xr, yr):
			pa = dict(color='k', marker='.', linestyle='None')
			plt.plot(x, y, **pa)
			plt.xlim(xr)
			plt.ylim(yr)

		la = dict(color='k', alpha=0.25)
		ta = dict(color='r', lw=2,)
		fa = dict(color='b', lw=2,)
		ga = dict(color='g', lw=2,)
			
		if nm.startswith('NYU'):
			plot_pts = plot_nyu
		else:
			plot_pts = plot_sdss
			#ta = dict(color='k', lw=2, alpha=0.5)
			#fa = dict(color='b', lw=2, alpha=0.5)
			#ga = dict(color='g', lw=2, alpha=0.5)

		plt.clf()
		xl,xh = [1.25, 3.75]
		yl,yh = [-0.5, 2.0]
		plot_pts(log(SS) + 0.2*(MU - 19.61),
				 log(RR * 1e3 / h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 2.) * (1.52) + 0.2, **ta)
		plt.xlabel('log(sigma) + 0.2 (mu_0 - 19.61)')
		plt.ylabel('log(R_0) [kpc / h]')
		#plt.axhline(0.2, **la)
		#plt.axhline(1.7, **la)
		#plt.axvline(2., **la)
		#plt.axvline(3., **la)
		plt.ylim(yl,yh)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 3 fig 1: %s' % nm)
		ps.savefig()

		if not nm.startswith('NYU'):
			continue
			
		plt.clf()
		yl,yh = [-24,-17]
		#xl,xh = [1.2, 2.6]
		xl,xh = [1.8, 2.8]

		xx = log(SS)
		yy = MM - 5.*log(h70)
		I = (np.isfinite(xx) * np.isfinite(yy))
		xx = xx[I]
		yy = yy[I]
		mx = np.mean(xx)
		my = np.mean(yy)
		A = np.zeros((len(xx),2))
		A[:,0] = 1.
		A[:,1] = xx-mx
		b,res,rank,s = np.linalg.lstsq(A, yy-my)
		print 'b', b
		m = b[1]
		b = b[0]
		A[:,1] = yy-my
		b2,res,rank,s = np.linalg.lstsq(A, xx-mx)
		print 'b2', b2
		m2 = b2[1]
		b2 = b2[0]
		
		plot_pts(log(SS), MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		Y = np.array([yl,yh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 2.) * -3.95*2.5 + -19.5, **ta)
		plt.plot(X, (X - 2.) * -3.95 + -19.5, **ta)
		plt.plot(X, (X-mx)*m + b + my, **fa)
		plt.plot((Y-my)*m2 + b2 + mx, Y, **ga)
		plt.xlabel('log(sigma) [km/s]')
		plt.ylabel('M - 5 log(h) [mag]')
		plt.axvline(2., **la)
		plt.axvline(2.4, **la)
		plt.axhline(-19.5, **la)
		plt.axvline(-23.5, **la)
		plt.ylim(yh,yl)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 2 fig 4: %s' % nm)
		ps.savefig()

		if hasattr(Ti, 'M2'):
			plt.clf()
			plot_pts(log(SS), Ti.M2 - 5.*log(h70), (xl,xh), (yl,yh))
			X = np.array([xl,xh])
			# eyeballed Bernardi relation for i-band
			plt.plot(X, (X - 2.) * -3.95*2.5 + -19.5, **ta)
			#plt.plot(X, (X - 2.) * -3.95 + -19.5, **ta)
			plt.xlabel('log(sigma) [km/s]')
			plt.ylabel('M - 5 log(h) [mag]')
			plt.ylim(yh,yl)
			plt.xlim(xl,xh)
			plt.title('Bernardi paper 2 fig 4: %s' % nm)
			ps.savefig()

		# plt.clf()
		# yl,yh = min(LL),max(LL)
		# plot_pts(log(SS), LL, (xl,xh), (yl,yh))
		# plt.xlabel('log(sigma) [km/s]')
		# plt.ylabel('LL')
		# ps.savefig()
		
		plt.clf()
		xl,xh = [-1., 1.5]
		plot_pts(log(RR * 1e3 / h70), MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 0.) * -1.59*2.5 + -19.5, **ta)
		plt.xlabel('log(R_0) [kpc / h]')
		plt.ylabel('M - 5 log(h) [mag]')
		plt.ylim(yh,yl)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 2 fig 5: %s' % nm)
		ps.savefig()

		plt.clf()
		xl,xh = [2., 7.]
		plot_pts(log(RR * 1e3 * SS**2), MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 4.) * -0.88*2.5 + -19.5, **ta)
		plt.xlabel('log(R_0 sigma^2) [kpc (km/s)^2]')
		plt.ylabel('M - 5 log(h) [mag]')
		plt.ylim(yh,yl)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 2 fig 7: %s' % nm)
		ps.savefig()

		plt.clf()
		xl,xh = [1., 6.]
		plot_pts(log((SS / (RR * 1e3))**2), MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 4.) * 1.34*2.5 + -19.5, **ta)
		plt.xlabel('log((sigma / R_0)^2) [((km/s) / kpc)^2]')
		plt.ylabel('M - 5 log(h) [mag]')
		plt.ylim(yh,yl)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 2 fig 8: %s' % nm)
		ps.savefig()

		plt.clf()
		xl,xh = [15., 24.]
		plot_pts(MU, MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		#### CHECK
		plt.plot(X, (X - 19.) * -(3.91/2.5) + -20., **ta)
		plt.xlabel('mu_0 [mag/arcsec^2]')
		plt.ylabel('M - 5 log(h) [mag]')
		plt.ylim(yh,yl)
		plt.xlim(xh,xl)
		plt.title('Bernardi paper 2 fig 9: %s' % nm)
		ps.savefig()


		plt.clf()
		xl,xh = [15., 24.]
		yl,yh = [-1, 1.5]
		plot_pts(MU, log(RR * 1e3 / h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 18.) * (0.76/2.5) + 0., **ta)
		plt.xlabel('mu_0 [mag/arcsec^2]')
		plt.ylabel('log(R_0) [kpc / h]')
		plt.ylim(yl,yh)
		plt.xlim(xh,xl)
		plt.title('Bernardi paper 2 fig 10: %s' % nm)
		ps.savefig()


		
			
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

	mags = [src.getBrightness().getMag(band) for src in srcs]
	I = np.argsort(mags)
	print 'Brightest sources:'
	for i in I[:10]:
		print '  ', srcs[i]
	
	ima = dict(interpolation='nearest', origin='lower',
			   extent=roi)
	zr2 = tinf['sky'] + tinf['skysig'] * np.array([-3, 100])
	imb = ima.copy()
	imb.update(vmin=tim.zr[0], vmax=tim.zr[1])
	imc = ima.copy()
	#imc.update(vmin=zr2[0], vmax=zr2[1])
	imc.update(norm=ArcsinhNormalize(mean=tinf['sky'],
									 std=tinf['skysig']),
				vmin=zr2[0], vmax=zr2[1])
	plt.clf()
	plt.imshow(tim.getImage(), **imb)
	plt.gray()
	plt.colorbar()
	ps.savefig()

	T = fits_table('a1656-spectro.fits')
	wcs = tim.getWcs()
	x0,y0 = wcs.x0,wcs.y0
	print 'x0,y0', x0,y0
	xy = np.array([wcs.positionToPixel(RaDecPos(r,d))
				   for r,d in zip(T.ra, T.dec)])
	sxy = np.array([wcs.positionToPixel(src.getPosition())
					for src in srcs])
	sxy2 = sxy[I[:20]]
	sa = dict(mec='r', mfc='None', ms=8, mew=1, alpha=0.5)
	pa = dict(mec='b', mfc='None', ms=6, mew=1, alpha=0.5)
	
	ax = plt.axis()
	plt.plot(xy[:,0]+x0, xy[:,1]+y0, 'o', **sa)
	plt.plot(sxy2[:,0]+x0, sxy2[:,1]+y0, 's', **pa)
	plt.axis(ax)
	ps.savefig()
	
	plt.clf()
	plt.imshow(tim.getImage(), **imc)
	plt.colorbar()
	ps.savefig()

	ax = plt.axis()
	plt.plot(xy[:,0]+x0, xy[:,1]+y0, 'o', **sa)
	plt.plot(sxy2[:,0]+x0, sxy2[:,1]+y0, 's', **pa)
	for (x,y),z in zip(xy, T.z):
		plt.text(x+x0, y+y0, '%.3f'%z)

	plt.axis(ax)
	ps.savefig()

	tractor = Tractor([tim], srcs)

	pnum = 00
	pickle_to_file(tractor, 'clustersky-%02i.pickle' % pnum)
	pnum += 1
	
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
	mn = (sky.min() + sky.max()) / 2.
	d = (z1 - z0)
	
	plt.clf()
	plt.imshow(sky, vmin=mn - d/2, vmax=mn + d/2, **ima)
	plt.colorbar()
	ps.savefig()
	
	imchi1 = ima.copy()
	imchi1.update(vmin=-5, vmax=5)
	imchi2 = ima.copy()
	imchi2.update(vmin=-50, vmax=50)
	
	def plotmod():
		mod = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)
		plt.clf()
		plt.imshow(mod, **imb)
		plt.gray()
		plt.colorbar()
		ps.savefig()
		plt.clf()
		plt.imshow(mod, **imc)
		plt.gray()
		plt.colorbar()
		ps.savefig()
		plt.clf()
		plt.imshow(chi, **imchi1)
		plt.gray()
		plt.colorbar()
		ps.savefig()
		plt.clf()
		plt.imshow(chi, **imchi2)
		plt.gray()
		plt.colorbar()
		ps.savefig()

	plotmod()
		
	tractor.freezeParam('images')

	tractor.catalog.freezeAllRecursive()
	tractor.catalog.thawPathsTo(band)
	print 'Params:'
	for nm in tractor.getParamNames():
		print nm

	j=0
	while True:
		print '-------------------------------------'
		print 'Optimizing flux step', j
		print '-------------------------------------'
		dlnp,X,alpha = tractor.optimize()
		print 'delta-logprob', dlnp
		nup = 0
		for src in tractor.getCatalog():
			for b in src.getBrightnesses():
				f = b.getFlux(band)
				if f < 0:
					#print 'Clamping flux', f, 'up to zero'
					nup += 1
					b.setFlux(band, 0.)
		print 'Clamped', nup, 'fluxes up to zero'
		if dlnp < 1:
			break
		j += 1
		plotmod()
		
		pickle_to_file(tractor, 'clustersky-%02i.pickle' % pnum)
		print 'Saved pickle', pnum
		pnum += 1
		
	
	# for src in tractor.getCatalog():
	# 	for b in src.getBrightnesses():
	# 		f = b.getFlux(band)
	# 		if f <= 0:
	# 			print 'src', src
	# find_clusters()

	tractor.catalog.thawAllRecursive()
	
	j=0
	while True:
		print '-------------------------------------'
		print 'Optimizing all, step', j
		print '-------------------------------------'
		dlnp,X,alpha = tractor.optimize()
		print 'delta-logprob', dlnp
		nup = 0
		for src in tractor.getCatalog():
			for b in src.getBrightnesses():
				f = b.getFlux(band)
				if f < 0:
					#print 'Clamping flux', f, 'up to zero'
					nup += 1
					b.setFlux(band, 0.)
		print 'Clamped', nup, 'fluxes up to zero'
		if dlnp < 1:
			break
		j += 1

		plotmod()

		pickle_to_file(tractor, 'clustersky-%02i.pickle' % pnum)
		print 'Saved pickle', pnum
		pnum += 1


		
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
					#print 'Clamping flux', f, 'up to zero'
					b.setFlux(band, 0.)

			print tractor.catalog[i]
			print
			if dlnp < 1:
				break
			j += 1

		plotmod()
		tractor.catalog.thawAllParams()

	pickle_to_file(tractor, 'clustersky-%02i.pickle' % pnum)
	print 'Saved pickle', pnum
	pnum += 1
		
def test2():
	band = 'i'
	ps = PlotSequence('abell')
	
	ps.skipto(56)
	pnum = 13
	tractor = unpickle_from_file('clustersky-12.pickle')

	print tractor
	tim = tractor.getImage(0)
	rng = tim.zr[1]-tim.zr[0]
	# Magic!
	zr2 = (tim.zr[0], tim.zr[0] + 103./13.)

	ima = dict(interpolation='nearest', origin='lower')
	imb = ima.copy()
	imb.update(vmin=tim.zr[0], vmax=tim.zr[1])
	imc = ima.copy()
	imc.update(vmin=zr2[0], vmax=zr2[1])
	imchi1 = ima.copy()
	imchi1.update(vmin=-5, vmax=5)
	imchi2 = ima.copy()
	imchi2.update(vmin=-50, vmax=50)
	
	def plotmod():
		mod = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)
		plt.clf()
		plt.imshow(mod, **imb)
		plt.gray()
		plt.colorbar()
		ps.savefig()
		plt.clf()
		plt.imshow(mod, **imc)
		plt.gray()
		plt.colorbar()
		ps.savefig()
		plt.clf()
		plt.imshow(chi, **imchi1)
		plt.gray()
		plt.colorbar()
		ps.savefig()
		plt.clf()
		plt.imshow(chi, **imchi2)
		plt.gray()
		plt.colorbar()
		ps.savefig()

	from tractor.splinesky import SplineSky

	H,W = tim.shape
	NX,NY = 10,10
	vals = np.zeros((NY,NX))
	XX = np.linspace(0, W, NX)
	YY = np.linspace(0, H, NY)
	tim.sky = SplineSky(XX, YY, vals)

	tractor.thawAllRecursive()
	tractor.images[0].freezeAllBut('sky')
	tractor.catalog.freezeAllRecursive()
	tractor.catalog.thawPathsTo(band)

	def plotsky():
		skyim = np.zeros(tim.shape)
		tim.sky.addTo(skyim)
		plt.clf()
		plt.imshow(skyim, **ima)
		plt.gray()
		plt.colorbar()
		plt.title('Spline sky model')
		ps.savefig()
		

	j=0
	while True:
		print '-------------------------------------'
		print 'Optimizing fluxes + sky step', j
		print '-------------------------------------'
		dlnp,X,alpha = tractor.optimize()
		print 'delta-logprob', dlnp
		nup = 0
		for src in tractor.getCatalog():
			for b in src.getBrightnesses():
				f = b.getFlux(band)
				if f < 0:
					#print 'Clamping flux', f, 'up to zero'
					nup += 1
					b.setFlux(band, 0.)
		print 'Clamped', nup, 'fluxes up to zero'
		if dlnp < 1:
			break
		j += 1
		plotmod()
		plotsky()
		
		pickle_to_file(tractor, 'clustersky-%02i.pickle' % pnum)
		print 'Saved pickle', pnum
		pnum += 1

		print 'Sky:', tim.sky
	

	tractor.catalog.thawAllRecursive()

	j=0
	while True:
		print '-------------------------------------'
		print 'Optimizing all sources + sky step', j
		print '-------------------------------------'
		dlnp,X,alpha = tractor.optimize()
		print 'delta-logprob', dlnp
		nup = 0
		for src in tractor.getCatalog():
			for b in src.getBrightnesses():
				f = b.getFlux(band)
				if f < 0:
					#print 'Clamping flux', f, 'up to zero'
					nup += 1
					b.setFlux(band, 0.)
		print 'Clamped', nup, 'fluxes up to zero'
		if dlnp < 1:
			break
		j += 1
		plotmod()
		plotsky()
		
		pickle_to_file(tractor, 'clustersky-%02i.pickle' % pnum)
		print 'Saved pickle', pnum
		pnum += 1

		print 'Sky:', tim.sky


	
	
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

	ps = PlotSequence('abell-b')

	LookupRcf = RaDecToRcf(tablefn='dr9fields.fits')

	for anum in [2151]:
		I = np.flatnonzero(T.aco == anum)
		print 'Abell', anum, ': found', len(I)
		Ti = T[I[0]]
		Ti.about()

		rcf = LookupRcf(Ti.ra, Ti.dec, contains=True)
		if len(rcf) == 0:
			print '-> Not in SDSS'
			continue
		print 'RCF', rcf
		for r,c,f,ra,dec in rcf:
			print 'http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpegcodec.aspx?R=%i&C=%i&F=%i&Z=50' % (r,c,f)
		
	return

	
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



def find_clusters(tractor, tim):
	# Find connected clusters of sources
	# [ (mask patch, [src,src]), ... ]
	dtype = np.int
	clusters = []
	for i,src in enumerate(tractor.getCatalog()):
		print 'Clustering source', i
		p = tractor.getModelPatch(tim, src)
		nz = p.getNonZeroMask()
		nz.patch = nz.patch.astype(dtype)
		#print '  nz vals:', np.unique(nz.patch)
		found = []
		for j,(mask, srcs) in enumerate(clusters):
			if not mask.hasNonzeroOverlapWith(nz):
				continue
			print 'Overlaps cluster', j
			found.append(j)
			#print '  Nonzero mask pixels:', len(np.flatnonzero(mask.patch))
			mask.set(mask.performArithmetic(nz, '__iadd__', otype=dtype))
			#print '  Nonzero mask pixels:', len(np.flatnonzero(mask.patch))
			mask.trimToNonZero()
			#print '  Nonzero mask pixels:', len(np.flatnonzero(mask.patch))
			print '  mask type', mask.patch.dtype
			srcs.append(src)
				
		if len(found) == 0:
			print 'Creating new cluster', len(clusters)
			clusters.append((nz, [src]))

		elif len(found) > 1:
			print 'Merging clusters', found
			m0,srcs0 = clusters[found[0]]
			for j in found[1:]:
				mi,srcsi = clusters[j]
				m0.set(m0.performArithmetic(mi, '__iadd__', otype=dtype))
				srcs0.extend(srcsi)
			for j in reversed(found[1:]):
				del clusters[j]
			print 'Now have', len(clusters), 'clusters'
			
	print 'Found', len(clusters), 'clusters'
	for i,(mask,srcs) in enumerate(clusters):
		n = len(np.flatnonzero(mask.patch))
		print 'Cluster', i, 'has', len(srcs), 'sources and', n, 'pixels'
		if n == 0:
			continue
		plt.clf()
		plt.imshow(np.sqrt(mask.patch),
				   interpolation='nearest', origin='lower',
				   extent=mask.getExtent(), vmin=0, vmax=sqrt(max(1, mask.patch.max())))
		ax = plt.axis()
		plt.gray()
		xy = np.array([tim.getWcs().positionToPixel(src.getPosition())
					   for src in srcs])
		plt.plot(xy[:,0], xy[:,1], 'r+')
		plt.axis(ax)
		plt.colorbar()
		ps.savefig()


def runlots(stage, N, force=[]):
	cmap = {'_RAJ2000':'ra', '_DEJ2000':'dec', 'ACOS':'aco'}
	T1 = fits_table('abell.fits', column_map=cmap)
	T2 = fits_table('abell2.fits', column_map=cmap)
	T3 = fits_table('abell3.fits', column_map=cmap)
	T = merge_tables([T1, T2, T3])

	# T = fits_table('maxbcg.fits')
	# I = np.argsort(-T.ngals)
	# T = T[I]
	# print T.ngals[:10]

	I = np.argsort(T.m10)
	T = T[I]
	RR = []

	LookupRcf = RaDecToRcf(tablefn='dr9fields.fits')

	for ai in range(len(T)):
		Ti = T[ai]
		print 'Abell', Ti.aco, 'with m10', Ti.m10
		# Totally arbitrary radius in arcmin
		R = 5.
		RS = np.hypot(R, np.hypot(13., 9.)/2.)
		rcf = LookupRcf(Ti.ra, Ti.dec, contains=True, radius=RS)
		if len(rcf) == 0:
			continue
		print 'RCF', rcf

		for run,camcol,field,nil,nil in rcf:
			print 'RCF', run, camcol, field
			#for bandname in ['g','r','i']:
			for bandname in ['i']:
				print 'Band', bandname

				ppat = 'clusky-a%04i-r%04i-c%i-f%04i-%s-s%%02i.pickle' % (Ti.aco, run, camcol, field, bandname)

				r = RunAbell(run, camcol, field, bandname,
							 Ti.ra, Ti.dec, R, Ti.aco)
				r.abell = Ti
				r.pat = ppat
				r.force = force
				res = r.runstage(0)
				if res is None:
					continue
				RR.append(r)
				if len(RR) >= N:
					break
			if len(RR) >= N:
				break

		if len(RR) >= N:
			break

	for R in RR:
		R.runstage(stage)

	#from astrometry.util.multiproc import multiproc
	#mp = multiproc(8)
	#mp.map(_run, [(R,stage) for R in RR])

def _run((R, stage)):
    return R(stage)


class SubImage(Image):
	def __init__(self, im, roi):
		#skyclass=SubSky,
		#psfclass=SubPsf,
		#wcsclass=SubWcs):
		(x0,x1,y0,y1) = roi
		slc = (slice(y0,y1), slice(x0,x1))
		data = im.getImage()[slc]
		invvar = im.getInvvar()[slc]
		sky = im.getSky()
		psf = im.getPsf()
		#wcs = wcsclass(im.getWcs(), roi)
		pcal = im.getPhotoCal()
		wcs = ShiftedWcs(im.getWcs(), x0, y0)
		#print 'PSF:', im.getPsf()
		#print 'Sky:', im.getSky()
		super(SubImage, self).__init__(data=data, invvar=invvar, psf=psf,
									   wcs=wcs, sky=sky, photocal=pcal,
									   name='sub'+im.name)

		
class RunAbell(object):
	def __init__(self, run, camcol, field, bandname,
				 ra, dec, R, aco):
		self.run = run
		self.camcol = camcol
		self.field = field
		self.bandname = bandname
		self.ra = ra
		self.dec = dec
		self.R = R
		self.aco = aco
		self.prereqs = { 103: 2, 203: 2 }
		#self.S = S
	def __call__(self, stage, **kwargs):
		kwargs.update(band=self.bandname, run=self.run,
					  camcol=self.camcol, field=self.field,
					  ra=self.ra, dec=self.dec)
		func = getattr(self.__class__, 'stage%i' % stage)
		return func(self, **kwargs)

	def runstage(self, stage, **kwargs):
		res = runstage(stage, self.pat, self, prereqs=self.prereqs,
					   force=self.force, **kwargs)
		return res

	def optloop(self, tractor):
		band = self.bandname
		j=0
		while True:
			print '-------------------------------------'
			print 'Optimizing: step', j
			print '-------------------------------------'
		   	dlnp,X,alpha = tractor.optimize() #priors=False)
			print 'delta-logprob', dlnp
			nup = 0
			for src in tractor.getCatalog():
				for b in src.getBrightnesses():
					f = b.getFlux(band)
					if f < 0:
						nup += 1
						b.setFlux(band, 0.)
			print 'Clamped', nup, 'fluxes up to zero'
			if dlnp < 1:
				break
			j += 1
		
	def stage0(self, run=None, camcol=None, field=None,
			   band=None, ra=None, dec=None, **kwargs):
		#
		S = (self.R * 60.) / 0.396
		getim = st.get_tractor_image_dr9
		getsrc = st.get_tractor_sources_dr9
		tim,tinf = getim(run, camcol, field, band,
						 roiradecsize=(ra, dec, S), nanomaggies=True)
		if tim.shape == (0,0):
			print 'Tim shape', tim.shape
			return None
		roi = tinf.get('roi', None)
		#print 'Stage 0: roi', roi
		sources = getsrc(run, camcol, field, band,
						 roi=roi, nanomaggies=True, bands=[band])
		tractor = Tractor([tim], sources)
		return dict(tractor=tractor,
					roi=roi, tinf=tinf)

	def stage1(self, tractor=None, band=None, **kwargs):
		# Opt fluxes only
		tractor.freezeParam('images')
		tractor.catalog.freezeAllRecursive()
		tractor.catalog.thawPathsTo(band)
		self.optloop(tractor)
		#self.plotmod(tractor)
		tractor.catalog.thawAllRecursive()
		return dict(tractor=tractor)

	def stage2(self, tractor=None, band=None, **kwargs):
		# Remove sources with 0 flux ?
		for src in tractor.getCatalog():
			if src.getBrightness().getFlux(band) <= 0:
				tractor.getCatalog().remove(src)
		return dict(tractor=tractor)

	def stage3(self, tractor=None, **kwargs):
		self.optloop(tractor)
		return dict(tractor=tractor)

	def stage203(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None,
				 **kwargs):
		print 'Stage203: kwargs', kwargs
		S = (self.R * 60.) / 0.396
		getim = st.get_tractor_image_dr9
		getsrc = st.get_tractor_sources_dr9
		tim,tinf = getim(run, camcol, field, band,
						 roiradecsize=(self.ra, self.dec, S), nanomaggies=True)
		print 'tinf', tinf
		tim = tractor.getImage(0)
		tim.origInvvar = None
		tim.starMask = None
		return dict(tinf=tinf, roi=tinf['roi'])
		
	def stage204(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None, roi=None, tinf=None,
				 **kwargs):
		fn = 'a%04i-spectro.fits' % self.aco
		if not os.path.exists(fn):
			sql = ' '.join(['select ra,dec,sourceType,z,zerr,',
							'class as clazz, subclass, velDisp,',
							'velDispErr',
							'from SpecObj where',
							'ra between %f and %f and',
							'dec between %f and %f']) % (ra-1, ra+1, dec-1, dec+1)
			from astrometry.util import casjobs
			casjobs.setup_cookies()
			cas = casjobs.get_known_servers()['dr9']
			username = os.environ['SDSS_CAS_USER']
			password = os.environ['SDSS_CAS_PASS']
			cas.login(username, password)
			cas.sql_to_fits(sql, fn, dbcontext='DR9')
			print 'Saved', fn

		T = fits_table(fn)
		print 'Read', len(T), 'spectro targets'
		T.about()
		T.cut(T.clazz == 'GALAXY')
		print 'Cut to', len(T), 'galaxies'

		cat = tractor.getCatalog()
		for i,src in enumerate(cat):
			src.ind = i

		rd = [src.getPosition() for src in cat]
		ra  = np.array([p.ra  for p in rd])
		dec = np.array([p.dec for p in rd])
		rad = 1./3600.
		I,J,d = sm.match_radec(T.ra, T.dec, ra, dec, rad,
							   nearest=True)
		print len(I), 'matches on RA,Dec'
		T.cut(I)
		specI = J

		# Optimize SDSS deblend families for spectro objects
		sdss = DR9()
		fn = sdss.retrieve('photoObj', run, camcol, field)
		objs = fits_table(fn)
		
		kids = objs[objs.nchild == 0]
		allkids = kids
		I,J,d = sm.match_radec(
			np.array([src.getPosition().ra  for src in cat]),
			np.array([src.getPosition().dec for src in cat]),
			kids.ra, kids.dec, 1./3600., nearest=True)
		print len(cat), 'tractor sources'
		print len(kids), 'SDSS kids'
		print len(I), 'matched'
		#kids = kids[J]

		for src in cat:
			src.parent = -2
		for i,j in zip(I,J):
			cat[i].parent = kids[j].parent

		P = np.array([src.parent for src in cat])
		specP = P[specI]
		print 'spec parents:', specP
		sgis = []
		gis = []
		specgroups = []
		groups = []
		for i,p in zip(specI, specP):
			if p == -1:
				sgis.append([i])
				groups.append([cat[i]])
				specgroups.append([cat[i]])
		print len(groups), 'unblended'
		for p in np.unique(specP):
			if p in [-1, -2]:
				continue
			print 'parent', p
			K = np.flatnonzero(p == specP)
			print 'specs', K
			specgroups.append([cat[i] for i in specI[K]])
			sgis.append(specI[K])
			K = np.flatnonzero(p == P)
			print len(K), 'with parent', p
			groups.append([cat[i] for i in K])
			gis.append([i for i in K if not i in sgis[-1]])
		print 'Groups:', len(groups)

		ps = PlotSequence(self.pat.replace('-s%02i.pickle', ''))
		ima = dict(interpolation='nearest', origin='lower',
				   extent=roi)
		zr2 = tinf['sky'] + tinf['skysig'] * np.array([-3, 100])
		imc = ima.copy()
		imc.update(norm=ArcsinhNormalize(mean=tinf['sky'], std=tinf['skysig']),
				   vmin=zr2[0], vmax=zr2[1])
		
		tim = tractor.getImage(0)
		mod = tractor.getModelImage(0)

		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(mod, **imc)
		plt.gray()
		ps.savefig()
		
		# Find the bbox of the sources in this group
		for i,(sgroup,group) in enumerate(zip(specgroups, groups)):
			bbox = tractor.getBbox(tim, group)
			
			plt.clf()
			plt.imshow(mod, **imc)
			plt.gray()
			ax = plt.axis()
			wcs = tim.getWcs()
			ix0,iy0 = wcs.x0,wcs.y0
			for src in group:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x+ix0],[y+iy0], 'o', mec='y', mfc='none',
						 mew=1.5, ms=10, alpha=0.5)
			for src in sgroup:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x+ix0],[y+iy0], 'o', mec='r', mfc='none',
						 mew=1.5, ms=8, alpha=0.5)
			x0,x1,y0,y1 = bbox
			plt.plot([x+ix0 for x in [x0,x0,x1,x1,x0]],
					 [y+iy0 for y in [y0,y1,y1,y0,y0]], 'r-')
			plt.axis(ax)
			plt.title('Group %i' % (i+1))
			ps.savefig()

			sigma = 1./np.sqrt(np.median(tim.getInvvar()))
			modi = tractor.getModelImage(tim, group, sky=False)

			# norm = ArcsinhNormalize(mean=tinf['sky'],
			# 						std=tinf['skysig'])
			# norm.vmin = zr2[0]
			# norm.vmax = zr2[1]
			# rgb = norm(modi)
			# rgb = np.clip(rgb, 0, 1)
			# print 'rgb', rgb
			# print rgb.shape
			# print rgb.min(), rgb.max()
			#r2 = rgb[:,:,np.newaxis].repeat(3,axis=2)

			# plt.clf()
			# #r2[:,:,2] = np.clip(rgb + 0.2*(modi >= sigma), 0, 1)
			# #plt.imshow(r2, **ima)
			# plt.imshow(modi >= sigma, **ima)
			# plt.title('1 sigma')
			# ps.savefig()
			# 
			# plt.clf()
			# #r2[:,:,2] = np.clip(rgb + 0.2*(modi >= 0.1*sigma), 0, 1)
			# #plt.imshow(r2, **ima)
			# plt.imshow(modi >= 0.1*sigma, **ima)
			# plt.title('0.1 sigma')
			# ps.savefig()
			# 
			# plt.clf()
			# plt.imshow(modi >= 0.01*sigma, **ima)
			# plt.title('0.01 sigma')
			# ps.savefig()

			thresh = 0.1*sigma
			mask = (modi >= thresh)

			mpatch = Patch(0, 0, mask)
			mpatch.trimToNonZero()
			print 'Mask:', mpatch
			
			over = []
			for src in cat:
				if src in group:
					continue
				p = tractor.getModelPatch(tim, src)
				if p is None:
					continue
				ns = np.sum(p.patch) / sigma
				# otherwise we corrupt the cache...!
				p = p.copy()
				p.patch = (p.patch >= thresh)
				if mpatch.hasNonzeroOverlapWith(p):
					#print 'Patch:', ns, 'sigma'
					over.append(src)
					
			for src in over:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x+ix0],[y+iy0], 'o', mec='g', mfc='none',
						 mew=1.5, ms=6, alpha=0.5)
			ps.savefig()

			##### HACK!
			if i != 1:
				continue
			
			plt.clf()
			plt.imshow(((modi >= 0.01*sigma).astype(int) +
						(modi >= 0.1*sigma).astype(int) +
						(modi >= sigma).astype(int)), **ima)
			plt.title('1 / 0.1 / 0.01 sigma')
			ps.savefig()

			subcat = Catalog(*(group + over))
			subcat.freezeAllParams()
			for i in range(len(group)):
				subcat.thawParam(i)
			tractor.setCatalog(subcat)

			# Zero out chi contribution outside masked area
			ie = tim.getInvError()
			orig_ie = ie.copy()
			ie[mask == 0] = 0.


			return dict(cat=cat, subcat=subcat,
						sgroup=sgroup, group=group, over=over,
						orig_ie=orig_ie,
						mpatch=mpatch, imc=imc, ps=ps,
						tractor=tractor)

	def stage205(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None, roi=None, tinf=None,
				 subcat=None, sgroup=None, group=None, over=None,
				 orig_ie=None, mpatch=None, imc=None, ps=None,
				 **kwargs):
		ima = dict(interpolation='nearest', origin='lower',
				   extent=roi)
		tim = tractor.getImage(0)
		wcs = tim.getWcs()
		ix0,iy0 = wcs.x0,wcs.y0

		slc = mpatch.getSlice()
		subext = mpatch.getExtent()
		#print 'Subimage extent:', subext
		subext = [subext[0]+ix0, subext[1]+ix0,
				  subext[2]+iy0, subext[3]+iy0]
		#print 'Subimage extent:', subext
		imsub = imc.copy()
		imsub.update(extent=subext)
		imchi1 = ima.copy()
		imchi1.update(vmin=-5, vmax=5, extent=subext)

		modj = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(modj, **imc)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(tim.getImage()[slc], **imsub)
		ax = plt.axis()
		for src in sgroup:
			x,y = wcs.positionToPixel(src.getPosition())
			#print 'sgroup', src, 'x,y', x+ix0, y+iy0
			plt.plot([x+ix0],[y+iy0], 'o', mec='r', mfc='none',
					 mew=1.5, ms=8, alpha=0.5)
		plt.axis(ax)
		plt.gray()
		ps.savefig()
		
		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(chi[slc], **imchi1)
		plt.gray()
		ps.savefig()

		print 'Sub tractor: params'
		for nm in tractor.getParamNames():
			print '  ', nm
		
		self.optloop(tractor)

		modj = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(chi[slc], **imchi1)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(modj, **imc)
		plt.gray()
		ps.savefig()

		return dict(imsub=imsub, imchi1=imchi1)


	def stage206(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None, roi=None, tinf=None,
				 subcat=None, sgroup=None, group=None, over=None,
				 orig_ie=None, mpatch=None, imc=None, ps=None,
				 imsub=None, imchi1=None,
				 **kwargs):

		tim = tractor.getImage(0)
		wcs = tim.getWcs()
		ix0,iy0 = wcs.x0,wcs.y0
		slc = mpatch.getSlice()

		# Model-switching the spectro targets
		origsrcs = sgroup
		newsrcs = []
		for src in sgroup:
			if isinstance(src, DevGalaxy):
				mag = src.getBrightness().getMag(band)
				# Give it 1% of the flux
				#en = NanoMaggies.magToNanomaggies(mag + 5.)
				#dn = NanoMaggies.magToNanomaggies(mag + 0.01)
				# Give it 10% of the flux
				en = NanoMaggies.magToNanomaggies(mag + 2.5)
				dn = NanoMaggies.magToNanomaggies(mag + 0.1)
				ebr = NanoMaggies(**{band:en})
				dbr = NanoMaggies(**{band:dn})
				newgal = CompositeGalaxy(
					src.pos.copy(), ebr, src.getShape().copy(),
					dbr, src.getShape().copy())
				newsrcs.append(newgal)
			else:
				newsrcs.append(src)

		sgi = []
		for src,newsrc in zip(sgroup,newsrcs):
			if newsrc is None:
				continue
			i = subcat.index(src)
			sgi.append(i)
			if newsrc == src:
				continue
			subcat[i] = newsrc
			print 'Switching source', i, 'from:'
			print ' from  ', src
			print ' to    ', newsrc
		tractor.catalog.freezeAllBut(*sgi)

		modj = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(chi[slc], **imchi1)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ps.savefig()

		print 'Model-switching: opt params'
		for nm in tractor.getParamNames():
			print '  ', nm
		
		self.optloop(tractor)

		modj = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(chi[slc], **imchi1)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ps.savefig()

		tractor.catalog.thawAllParams()
		print 'Model-switching: opt all'
		for nm in tractor.getParamNames():
			print '  ', nm

		modj = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(chi[slc], **imchi1)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ps.savefig()

		return dict(old_sgroup=sgroup, new_sgroup=newsrcs,
					sgi=sgi)
	
	def stage207(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None, roi=None, tinf=None,
				 subcat=None, sgroup=None, group=None, over=None,
				 orig_ie=None, mpatch=None, imc=None, ps=None,
				 old_sgroup=None, new_sgroup=None, sgi=None,
				 cat=None,
				 **kwargs):
		ima = dict(interpolation='nearest', origin='lower',
				   extent=roi)
		tim = tractor.getImage(0)
		wcs = tim.getWcs()
		ix0,iy0 = wcs.x0,wcs.y0
		slc = mpatch.getSlice()

		print 'Mpatch', mpatch
		print 'Slice', slc
		subext = mpatch.getExtent()
		print 'Extent', subext
		subext = [subext[0]+ix0, subext[1]+ix0,
				  subext[2]+iy0, subext[3]+iy0]
		print 'Subimage extent:', subext
		imsub = imc.copy()
		imsub.update(extent=subext)
		imchi1 = ima.copy()
		imchi1.update(vmin=-5, vmax=5, extent=subext)

		srcs = group + over
		I = np.argsort([src.getBrightness().getMag(band) for src in srcs])
		modj = tractor.getModelImage(0)
		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ax = plt.axis()
		print 'Brightest sources:'
		for j,i in enumerate(I[:10]):
			src = srcs[i]
			print src
			x,y = wcs.positionToPixel(src.getPosition())
			plt.plot([x+ix0],[y+iy0], 'o', mec='r', mfc='none',
					 mew=1.5, ms=8, alpha=0.5)
			plt.text(x+ix0, y+iy0, '%i'%j, color='r')
		plt.axis(ax)
		ps.savefig()

		# blank = [0]
		# ie = tim.getInvError()
		# for j in blank:
		# 	src = srcs[I[j]]
		# 	x,y = wcs.positionToPixel(src.getPosition())
		# 	rr = 25
		# 	H,W = ie.shape
		# 	ie[max(0, y-rr): min(H, y+rr+1),
		# 	   max(0, x-rr): min(W, x+rr+1)] = 0
		
		# Add spline sky
		modj = tractor.getModelImage(0)
		submod = modj[slc]
		H,W = submod.shape
		print 'Submod shape:', submod.shape
		G = 100.
		NX,NY = 2 + int(np.ceil(W/G)), 2 + int(np.ceil(H/G))
		print 'NX,NY', NX,NY
		vals = np.zeros((NY,NX))
		print 'vals shape', vals.shape
		XX = np.linspace(0, W, NX)
		YY = np.linspace(0, H, NY)
		ssky = SplineSky(XX, YY, vals)
		###
		sigma = 1./np.median(orig_ie)
		print 'Sigma', sigma
		ssky.setPriorSmoothness(sigma * 0.1)

		subsky = SubSky(ssky, slc)		

		tim.sky = subsky
		tractor.thawParam('images')
		tim.freezeAllBut('sky')
		tractor.catalog.freezeAllBut(*sgi)

		print 'Supposed to be thawing sources', sgi

		print 'Spline sky: opt'
		for nm in tractor.getParamNames():
			print '  ', nm

		print 'Initial:'
		print 'lnLikelihood', tractor.getLogLikelihood()
		print 'lnPrior', tractor.getLogPrior()
		print 'lnProb', tractor.getLogProb()

		self.optloop(tractor)

		print 'After opt:'
		print 'lnLikelihood', tractor.getLogLikelihood()
		print 'lnPrior', tractor.getLogPrior()
		print 'lnProb', tractor.getLogProb()

		modj = tractor.getModelImage(0)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(chi[slc], **imchi1)
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(modj[slc], **imsub)
		plt.gray()
		ps.savefig()

		skyim = np.zeros_like(modj[slc])
		ssky.addTo(skyim)
		plt.clf()
		plt.imshow(skyim, **imsub)
		plt.gray()
		plt.title('Spline sky model')
		ps.savefig()
		
		# Revert
		#tim.inverr = orig_ie
		#tractor.setCatalog(cat)

		return dict()
			
		
		
	def stage103(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 tinf=None, roi=None,
				 **kwargs):

		print 'tinf', tinf
		S = (self.R * 60.) / 0.396
		getim = st.get_tractor_image_dr9
		getsrc = st.get_tractor_sources_dr9
		tim,tinf = getim(run, camcol, field, band,
						 roiradecsize=(self.ra, self.dec, S), nanomaggies=True)
		print 'tinf', tinf
		

		ps = PlotSequence(self.pat.replace('-s%02i.pickle', ''))

		ima = dict(interpolation='nearest', origin='lower',
				   extent=roi)
		zr2 = tinf['sky'] + tinf['skysig'] * np.array([-3, 100])
		imc = ima.copy()
		imc.update(norm=ArcsinhNormalize(mean=tinf['sky'], std=tinf['skysig']),
				   vmin=zr2[0], vmax=zr2[1])

		# Optimize SDSS deblend families
		sdss = DR9()
		fn = sdss.retrieve('photoObj', run, camcol, field)
		objs = fits_table(fn)
		#print 'SDSS objects:'
		#objs.about()

		cat = tractor.getCatalog()
		# match tractor sources and SDSS children
		kids = objs[objs.nchild == 0]
		I,J,d = sm.match_radec(np.array([src.getPosition().ra  for src in cat]),
							   np.array([src.getPosition().dec for src in cat]),
							   kids.ra, kids.dec, 1./3600., nearest=True)
		print len(cat), 'tractor sources'
		print len(kids), 'SDSS kids'
		print len(I), 'matched'

		kids = kids[J]
		tractor.catalog.freezeAllParams()
		tim = tractor.getImage(0)
		tim.origInvvar = None
		tim.starMask = None
		#tim.invvar = None
		#tim.invvar = (tim.inverr)**2
		psf = tim.getPsf()
		psf.radius = min(25, int(np.ceil(psf.computeRadius())))
		print 'PSF radius:', psf.radius

		mod = tractor.getModelImage(0)
		plt.clf()
		plt.imshow(mod, **imc)
		plt.gray()
		#plt.colorbar()
		ps.savefig()

		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		plt.gray()
		ps.savefig()
		
		for p in np.unique(kids.parent):
			if p == -1:
				continue
			K = np.flatnonzero(kids.parent == p)
			print len(K), 'with parent', p

			srcs = []
			nzsum = None
			for i in I[K]:
				cat.thawParam(i)

				# find bbox
				src = cat[i]
				p = tractor.getModelPatch(tim, src)
				if p is None:
					continue
				nz = p.getNonZeroMask()
				dtype = np.int
				nz.patch = nz.patch.astype(dtype)
				if nzsum is None:
					nzsum = nz
				else:
					nzsum += nz

				ie = tim.getInvError()
				#ie = ie[p.getSlice(ie)]
				#p2 = p * ie
				p2 = np.zeros_like(ie)
				p.addTo(p2)
				effect = np.sum(p2)
				#effect = np.sum(p2.patch)
				print 'Source:', src
				print 'Total chi contribution:', effect, 'sigma'

				srcs.append(src)
			nzsum.trimToNonZero()
			roi = nzsum.getExtent()

			Nin = len(srcs)
			
			# find other sources that overlap the ROI.
			for src in cat:
				if src in srcs:
					continue
				p = tractor.getModelPatch(tim, src)
				if p is None:
					continue
				if p.overlapsBbox(roi):
					srcs.append(src)
			print 'Found', len(srcs), 'total sources overlapping the bbox'

			ax = plt.axis()
			for i,src in enumerate(srcs):
				x,y = tim.getWcs().positionToPixel(src.getPosition())
				if i < Nin:
					cc = 'r'
				else:
					cc = 'g'
				plt.plot([x],[y], 'x', color=cc)

				p = tractor.getModelPatch(tim, src)
				x0,x1,y0,y1 = p.getExtent()
				plt.plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0], '-', color=cc, alpha=0.5)

			x0,x1,y0,y1 = roi
			plt.plot([x0,x0,x1,x1,x0],[y0,y1,y1,y0,y0], 'r-')
			plt.axis(ax)
			ps.savefig()

			subimg = SubImage(tim, roi)
			subcat = Catalog(*srcs)
			subcat.freezeAllParams()
			for i in range(Nin):
				#print 'Thawing subcat param', i
				subcat.thawParam(i)
			subtractor = Tractor(Images(subimg), subcat)
			subtractor.freezeParam('images')

			print 'Subtractor: params'
			for nm in subtractor.getParamNames():
				print '  ', nm

			#print 'Subimage shape', subimg.shape
			#print 'Subimage image shape', subimg.getImage().shape
			#print 'ROI', roi
			submod = subtractor.getModelImage(0)
			#print 'Submod', submod.shape
			imsub = imc.copy()
			imsub.update(extent=roi)

			plt.clf()
			plt.imshow(submod, **imsub)
			plt.gray()
			ps.savefig()

			plt.clf()
			plt.imshow(subimg.getImage(), **imsub)
			plt.gray()
			ps.savefig()

			self.optloop(subtractor)

			submod = subtractor.getModelImage(0)
			plt.clf()
			plt.imshow(submod, **imsub)
			plt.gray()
			ps.savefig()

			for i in I[K]:
				cat.freezeParam(i)

			mod = tractor.getModelImage(0)
			plt.clf()
			plt.imshow(mod, **imc)
			plt.gray()
			ps.savefig()

		return dict(tractor=tractor, tinf=tinf)

	def stage4(self, tractor=None, band=None, **kwargs):
		# Opt sources individually, brightest first
		mags = []
		for src in tractor.getCatalog():
			mags.append(src.getBrightness().getMag(band))
		I = np.argsort(mags)
		for i in I:
			tractor.catalog.freezeAllBut(i)
			self.optloop(tractor)
			print tractor.catalog[i]
		#plotmod()
		tractor.catalog.thawAllParams()
		return dict(tractor=tractor)

	def stage5(self, tractor=None, band=None, **kwargs):
		tim = tractor.getImages()[0]
		H,W = tim.shape
		NX,NY = [int(np.ceil(x / 100) + 1) for x in [W,H]]
		vals = np.zeros((NY,NX))
		XX = np.linspace(0, W, NX)
		YY = np.linspace(0, H, NY)
		tim.sky = SplineSky(XX, YY, vals)
		tractor.thawAllRecursive()
		tractor.images[0].freezeAllBut('sky')
		tractor.catalog.freezeAllRecursive()
		tractor.catalog.thawPathsTo(band)
		self.optloop(tractor)
		tractor.catalog.thawAllRecursive()
		return dict(tractor=tractor)

	def stage6(self, tractor=None, band=None, **kwargs):
		self.optloop(tractor)
		return dict(tractor=tractor)

	#def stage8(self, tractor=None, **kwargs):

class SubSky(ParamsWrapper):
	def __init__(self, real, slc):
		super(SubSky, self).__init__(real)
		self.real = real
		self.slc = slc
	def getParamDerivatives(self, img):
		return self.real.getParamDerivatives(img)
	def addTo(self, mod):
		self.real.addTo(mod[self.slc])

	

if __name__ == '__main__':
	import logging
	from optparse import OptionParser
	import sys
	parser = OptionParser(usage=('%prog'))
	parser.add_option('-v', '--verbose', dest='verbose', action='count',
					  default=0, help='Make more verbose')
	parser.add_option('-s', '--stage', dest='stage', type=int,
					  default=6, help='Run to stage...')
	parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
					  help="Force re-running the given stage(s) -- don't read from pickle.")
	parser.add_option('-n', dest='N', type=int,
					  default=20, help='Run this # of fields')
	opt,args = parser.parse_args()
	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	#find()
	#fp()
	#get_dm_table()
	#join()
	runlots(stage=opt.stage, N=opt.N, force=opt.force)
	sys.exit(0)
	test1()

	if False:
		run, camcol, field = 5115, 5, 151
		band = 'i'
		sdss = DR9()
		fn = sdss.retrieve('idR', run, camcol, field, band)
		print 'Got', fn
		P = pyfits.open(fn)[0]
		from astrometry.util.fix_sdss_idr import fix_sdss_idr
		P = fix_sdss_idr(P)
		D = P.data
		plt.clf()
		plt.imshow(D, interpolation='nearest', origin='lower',
				   vmin=2070, vmax=2120)
		plt.gray()
		plt.colorbar()
		plt.savefig('idr.png')
	
	test2()

