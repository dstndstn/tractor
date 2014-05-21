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
	matplotlib.rc('text', usetex=True)
import numpy as np
import pylab as plt
import scipy.interpolate
import scipy.spatial
import os

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.sdss.fields import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
import astrometry.libkd.spherematch as sm
from astrometry.sdss import *

from astrometry.util.stages import *

from tractor.utils import *
from tractor import sdss as st
from tractor import *
from tractor.sdss_galaxy import *
from tractor.splinesky import SplineSky
from tractor.sdss import SdssPointSource, SdssBrightPSF

def global_init(*args, **kwargs):
	#plt.figure(figsize=(12,6))
	H = 4
	hfrac = 0.94
	W = (941 - 100.) / (2000 - 1311.) * H * hfrac
	print 'W,H', W,H
	plt.figure(figsize=(W,H))
	plt.clf()
	b = 0.
	plt.subplots_adjust(hspace=0.01, wspace=0.01,
						left=0.005, right=0.995,
						bottom=b, top=b + hfrac)

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

def get_spectro_table(ra, dec, aco):
	fn = 'a%04i-spectro.fits' % aco
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
		try:
			cas.sql_to_fits(sql, fn, dbcontext='DR9')
			print 'Saved', fn
		except:
			print 'Failed to execute SQL query on CasJobs'
			return None
	return fn


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
	T.writeto('abell-all.fits')
	
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


def runlots(stage, N, force=[], threads=None):
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

		#####
		#if not Ti.aco in [2147, 1656]:
		if not Ti.aco in [1656]:
			continue

		blacklist = [2666, # no SDSS spectro coverage
					 2634,
					 592,
					 2626,
					 179,
					 2572,
					 2256,
					 195,
					 ]
		if Ti.aco in blacklist:
			print 'Skipping blacklisted Abell', Ti.aco
			continue

		# Totally arbitrary radius in arcmin
		R = 5.
		RS = np.hypot(R, np.hypot(13., 9.)/2.)
		rcf = LookupRcf(Ti.ra, Ti.dec, radius=RS) #contains=True, 
		if len(rcf) == 0:
			continue
		print 'RCF', rcf

		ofn = 'overview-n%04i-a%04i.png' % (ai, Ti.aco)
		if not os.path.exists(ofn):
			# Overview plot
			fn = get_spectro_table(Ti.ra, Ti.dec, Ti.aco)
			if fn is None:
				continue
			Tspec = fits_table(fn)
			sdss = DR9()
			corners = []
			for run,camcol,field,nil,nil in rcf:
				bandname = 'i'
				fn = sdss.retrieve('frame', run, camcol, field, bandname)
				frame = sdss.readFrame(run, camcol, field, bandname,
									   filename=fn)
				astrans = frame.getAsTrans()
				W,H = 2048,1489
				rds = [astrans.pixel_to_radec(x,y)
					   for x,y in [(0,0),(W,0),(W,H),(0,H),(0,0)]]
				corners.append(rds)
			plt.clf()
			for (r,c,f,nil,nil),rds in zip(rcf,corners):
				rds = np.array(rds)
				plt.plot(rds[:,0], rds[:,1], 'k-')
				plt.text(rds[0,0], rds[0,1], '%i/%i/%i' % (r,c,f),
						 bbox=dict(facecolor='1', alpha=0.8))
			plt.plot(Tspec.ra, Tspec.dec, 'o', mec='r', mfc='none',
					 mew=1.5, ms=5, alpha=0.5)
			plt.plot([Ti.ra], [Ti.dec], 'k+', ms=30, mew=2)
			plt.xlabel('RA (deg)')
			plt.ylabel('Dec (deg)')
			R = 20./60.
			plt.axis('equal')
			plt.axis([Ti.ra-R, Ti.ra+R, Ti.dec-R, Ti.dec+R])
			plt.title('Abell %i (m10=%.1f) at z=%.3f' % (Ti.aco, Ti.m10, Ti.z))
			plt.savefig(ofn)


		for run,camcol,field,nil,nil in rcf:
			print 'RCF', run, camcol, field

			###
			#if not (run in [4678, 5237, 
			if not (run == 5115 and camcol == 5 and field == 150):
				continue

			#for bandname in ['g','r','i']:
			for bandname in ['i']:
				print 'Band', bandname

				ppat = 'clusky-a%04i-r%04i-c%i-f%04i-%s-s%%02i.pickle' % (Ti.aco, run, camcol, field, bandname)

				r = RunAbell(run, camcol, field, bandname,
							 Ti.ra, Ti.dec, R, Ti.aco)
				r.abell = Ti
				r.pat = ppat
				r.force = force
				#res = r.runstage(0)
				#if res is None:
				#	continue
				RR.append(r)
				print 'Got r/c/f', len(RR)
				if len(RR) >= N:
					break
			if len(RR) >= N:
				break
		if len(RR) >= N:
			break

	if stage == -1:
		return

	from astrometry.util.multiproc import multiproc
	if threads is None:
		threads = 1

	mpa = dict(init=global_init, initargs=())

	mp = multiproc(threads, **mpa)

	# Run stage 0
	print 'Running stage 0 on all...'
	s = 0
	res = mp.map(_run, [(R,s) for R in RR])
	RR = [r for r,result in zip(RR,res) if result is not None]

	print 'Running stage', stage, 'on all...'
	s = stage
	mp.map(_run, [(R,s) for R in RR])


	#for R in RR:
	#	R.runstage(stage)
	#from astrometry.util.multiproc import multiproc
	#mp = multiproc(8)
	#mp.map(_run, [(R,stage) for R in RR])

def _run((R, stage)):
	return R.runstage(stage)


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


class Family(object):
	pass

class RunOneGroup(object):
	def __init__(self, pat, fam, force=[], **kwa):
		self.pat = pat
		self.fam = fam
		self.force = force
		kwa.update(ps = PlotSequence(pat.replace('.pickle', ''), ''))
		self.globals = kwa
		self.prereqs = { }
	def __call__(self, stage, **kwargs):
		kwargs.update(self.globals)
		func = getattr(self.__class__, 'stage%i' % stage)
		return func(self, **kwargs)
	def runstage(self, stage, **kwargs):
		res = runstage(stage, self.pat, self, prereqs=self.prereqs,
					   force=self.force, **kwargs)
		return res

	def stage0(self, **kwargs):
		#return self.globals
		return dict(fam = self.fam)
	
	def stage1(self, fam=None, ps=None, **kwargs):
		return dict()


dpool = None


def draw_gaussian(mu, sigma, size):
	sigma = np.abs(sigma)
	if sigma == 0:
		return np.zeros(size) + mu
	return np.random.normal(loc=mu, scale=sigma, size=size)

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
		#self.prereqs = { 103: 2, 203: 2 }
		self.prereqs = { 103: 2, 204: 0, 1000: 0,  1004:1002,
						 1008:1006,
						 1009:1006,

						 1020:1015,
						 }
		#self.S = S
	def __call__(self, stage, **kwargs):
		kwargs.update(band=self.bandname, run=self.run,
					  camcol=self.camcol, field=self.field,
					  ra=self.ra, dec=self.dec, stage=stage)
		func = getattr(self.__class__, 'stage%i' % stage)
		return func(self, **kwargs)

	def runstage(self, stage, **kwargs):
		res = runstage(stage, self.pat, self, prereqs=self.prereqs,
					   force=self.force, **kwargs)
		return res

	def optloop(self, tractor, **optargs):
		band = self.bandname
		j=0
		while True:
			print '-------------------------------------'
			print 'Optimizing: step', j
			print '-------------------------------------'
		   	dlnp,X,alpha = tractor.optimize(**optargs)
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

	def stage1000(self, run=None, camcol=None, field=None,
				  band=None, ra=None, dec=None, **kwargs):
		tim,tinf = st.get_tractor_image_dr9(run, camcol, field, band,
											nanomaggies=True, psf='dg')
		if tim is None:
			return None
		srcs,objs,objI = st.get_tractor_sources_dr9(run, camcol, field, band,
													bands=[band], nanomaggies=True,
													getobjs=True, getobjinds=True)
		ps = PlotSequence(self.pat.replace('-s%02i.pickle', '') + '-g2')

		#sdss = DR9()
		#fn = sdss.retrieve('photoObj', run, camcol, field)
		#objs = fits_table(fn)

		# top-level deblend families
		dt = objs.parent.dtype
		idToIndex = np.zeros(max(objs.id)+1, dt) - 1
		idToIndex[objs.id.astype(dt)] = np.arange(len(objs)).astype(dt)
		#print 'idToIndex:', idToIndex
		# objs.family is the *index* (not id) of the top-level
		# deblend ancestor for each object.
		objs.family = np.zeros_like(objs.parent) - 1
		#print 'objs.family:', objs.family
		I = np.flatnonzero(objs.parent > -1)
		#print 'parents:', objs.parent[I].astype(dt)
		#print 'ids', idToIndex[objs.parent[I].astype(dt)]
		objs.family[I] = idToIndex[objs.parent[I].astype(dt)].astype(dt)

		while True:
			I = np.flatnonzero(objs.family > -1)
			# current ancestors
			A = objs[ objs.family[I] ]
			# ancestors that still have parents
			J = np.flatnonzero(A.family > -1)
			if len(J) == 0:
				break
			print 'Updating', len(J), 'ancestors'
			# update family pointers
			objs.family[I[J]] = A.family[J]

		kids = objs[objs.nchild == 0]

		abell = self.abell
		print abell.about()

		fn = get_spectro_table(abell.ra, abell.dec, self.aco)
		if fn is None:
			return dict()
		print 'Looking for', fn
		#fn = 'a%04i-spectro.fits' % self.aco
		Tspec = fits_table(fn)
		print len(Tspec), 'SDSS spectra'
		Tspec.cut(Tspec.clazz == 'GALAXY')
		print len(Tspec), 'after cut on GALAXY'
		Tspec.cut((Tspec.z > 0) * (Tspec.z <= 0.3))
		print len(Tspec), 'after cut on z'

		ima = dict(interpolation='nearest', origin='lower')
		zr2 = tinf['sky'] + tinf['skysig'] * np.array([-3, 100])
		imc = ima.copy()
		imc.update(norm=ArcsinhNormalize(mean=tinf['sky'], std=tinf['skysig']),
				   vmin=zr2[0], vmax=zr2[1])

		I,J,d = sm.match_radec(Tspec.ra, Tspec.dec, kids.ra, kids.dec,
							   1./3600., nearest=True)
		print len(Tspec), 'SDSS spectra'
		print len(kids), 'SDSS kids'
		print len(I), 'matched'

		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		plt.gray()
		ax = plt.axis()
		wcs = tim.getWcs()

		for i,j in zip(I,J):
			x,y = wcs.positionToPixel(RaDecPos(Tspec.ra[i], Tspec.dec[i]))
			#print 'spec z', Tspec.z[i], 'abell z', abell.z
			#print 'diff', abell.z - Tspec[i]
			#print 'abs', np.abs(abell.z - Tspec[i])
			#print 'in:', (np.abs(abell.z - Tspec[i]) <= 0.02)
			if np.abs(abell.z - Tspec.z[i]) <= 0.02:
				cc = 'r'
			else:
				cc = (0,0.5,1)
			#print 'color', cc
			plt.text(x, y+25, '%.3f' % Tspec.z[i], ha='center',
					 va='bottom', color=cc)
			plt.plot([x],[y], 'o', mec=cc, mfc='none',
					 mew=1.5, ms=8, alpha=0.5)

		x,y = wcs.positionToPixel(RaDecPos(abell.ra, abell.dec))
		plt.plot([x],[y], 'r+', ms=30, mew=3, alpha=0.5)
		plt.text(x, y+25, '%.3f' % abell.z, color='r')
		plt.title('Abell %i (m10=%.1f) at z=%.3f (pix %i,%i)' % (abell.aco, abell.m10, abell.z, int(x), int(y)))
													  
		xy = np.array([wcs.positionToPixel(RaDecPos(r,d))
					   for r,d in zip(kids.ra, kids.dec)])
		kids.x = xy[:,0]
		kids.y = xy[:,1]
		kids.xy = xy

		for f in np.unique(kids.family):
			if f == -1:
				continue
			I = np.flatnonzero(kids.family == f)
			if len(I) == 1:
				continue
			if len(I) > 2:
				D = scipy.spatial.Delaunay(kids.xy[I])
				#print 'Convex hull:', D.convex_hull
				hull = [(I[i],I[j]) for i,j in D.convex_hull]
			else:
				hull = [I]
			for i,j in hull:
				plt.plot([kids.x[i],kids.x[j]],
						 [kids.y[i],kids.y[j]], 'r-',
						 lw=2, alpha=0.25)
			
		plt.axis(ax)
		ps.savefig()

		tractor = Tractor([tim], srcs)
		tim.origInvvar = None
		tim.starMask = None

		return dict(tractor=tractor, objs=objs, objI=objI,
					tinf=tinf, Tspec=Tspec, ima=ima, imc=imc)


	def stage1001(self, run=None, camcol=None, field=None,
				  band=None, ra=None, dec=None,
				  tractor=None, objs=None, objI=None,
				  tinf=None, Tspec=None, ima=None, imc=None,
				  **kwargs):
		tobjs = objs[objI]
		kids = tobjs
		cat = tractor.getCatalog()
		print 'Tractor catalog', len(cat), 'objs', tobjs

		for src,i in zip(cat,objI):
			src.sdssobj = objs[i]

		# Select sources and deblend families of spectro targets...

		I,J,d = sm.match_radec(Tspec.ra, Tspec.dec, kids.ra, kids.dec,
							   1./3600.) #, nearest=True)

		print len(Tspec), 'SDSS spectra'
		print len(kids), 'SDSS kids'
		print len(I), 'matched'

		for i,j in zip(I,J):
			K = np.flatnonzero(i == I)
			if len(K) > 1:
				print len(K), 'matches for this spectro object'
				#continue
			K = np.flatnonzero(j == J)
			if len(K) > 1:
				print len(K), 'matches for this photo object'
				# Shouldn't happen due to fiber collisions, unless multiple plates
				# per field
				assert(False)
				#continue
			#spec = Tspec[i]
			cat[j].spec = Tspec[i]
			cat[j].speci = i

		donefams = []

		ps = PlotSequence(self.pat.replace('-s%02i.pickle', '') + '-groups')

		tim = tractor.getImage(0)
		mod = tractor.getModelImage(0)

		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)
		chi = tractor.getChiImage(0)

		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		plt.xticks([]); plt.yticks([])
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(mod, **imc)
		plt.xticks([]); plt.yticks([])
		plt.gray()
		ps.savefig()

		noise = np.random.normal(size=mod.shape)
		I = (tim.getInvvar() == 0)
		noise[I] = 0.
		I = np.logical_not(I)
		noise[I] *= 1./(tim.getInvError()[I])
		plt.clf()
		plt.imshow(mod + noise, **imc)
		plt.xticks([]); plt.yticks([])
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(chi, **imchi)
		plt.xticks([]); plt.yticks([])
		plt.gray()
		ps.savefig()

		plt.clf()
		plt.imshow(chi, cmap=redgreen, **imchi)
		plt.xticks([]); plt.yticks([])
		ps.savefig()

		plt.clf()
		plt.imshow(chi, cmap=bluegrayred, **imchi)
		plt.xticks([]); plt.yticks([])
		ps.savefig()

		#ax = plt.axis()
		wcs = tim.getWcs()
		sigma = 1./np.sqrt(np.median(tim.getInvvar()))

		fams = []

		for i,j in zip(I,J):
			src = cat[j]
			kid = kids[j]
			if kid.family == -1:
				fam = np.array([j])
			else:
				if kid.family in donefams:
					continue
				donefams.append(kid.family)
				fam = np.flatnonzero(kids.family == kid.family)
			print len(fam), 'sources in deblend family'

			family = Family()
			fams.append(family)
			# Photo family ID
			family.family = kid.family

			plt.clf()
			plt.imshow(tim.getImage(), **imc)
			#plt.xticks([]); plt.yticks([])
			plt.gray()
			ax = plt.axis()

			srcs = [cat[k] for k in fam]
			bbox = tractor.getBbox(tim, srcs)
			specsrcs = []
			for src in srcs:
				x,y = wcs.positionToPixel(src.getPosition())
				kwa = dict(mec='y', mfc='none', mew=1.5, ms=8, alpha=0.5)
				if hasattr(src, 'spec'):
					kwa.update(mec='r', ms=10)
					specsrcs.append(src)
				plt.plot([x], [y], 'o', **kwa)

			family.srcs = srcs
			#family.bbox = bbox
			family.specsrcs = specsrcs

			modi = tractor.getModelImage(tim, srcs, sky=False)

			thresh = 0.1*sigma
			mask = (modi >= thresh)
			mpatch = Patch(0, 0, mask)
			mpatch.trimToNonZero()
			#print 'Mask:', mpatch
			ext = mpatch.getExtent()

			family.ext = ext
			family.mpatch = mpatch

			over = []
			for src in cat:
				if src in srcs:
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
				plt.plot([x],[y], 'o', mec='g', mfc='none',
						 mew=1.5, ms=6, alpha=0.5)

			family.over = over

			plt.axis(ax)
			ps.savefig()

			# find unique spectro sources, and disentangle multiple
			# matches.
			specI = [src.speci for src in specsrcs]
			for speci in np.unique(specI):
				I = np.flatnonzero(specI == speci)
				N = len(I) + 3
				C = int(np.ceil(np.sqrt(N)))
				R = int(np.ceil(float(N) / C))

				plt.clf()
				plt.subplot(R,C,1)
				plt.imshow(tim.getImage(), **imc)
				plt.xticks([]); plt.yticks([])
				plt.gray()
				plt.axis(ext)

				plt.subplot(R,C,2)
				plt.imshow(tim.getImage(), **imc)
				plt.xticks([]); plt.yticks([])
				plt.gray()

				for src in srcs:
					x,y = wcs.positionToPixel(src.getPosition())
					kwa = dict(mec='y', mfc='none', mew=1.5, ms=8, alpha=0.5)
					if hasattr(src, 'spec'):
						kwa.update(mec='r', ms=10)
						if src.speci == speci:
							kwa.update(mec='m', ms=12)
					plt.plot([x], [y], 'o', **kwa)
				plt.axis(ext)

				modj = tractor.getModelImage(tim, over+srcs, sky=False)
				plt.subplot(R,C, 3)
				plt.imshow(modj, **imc)
				plt.xticks([]); plt.yticks([])
				plt.axis(ext)

				if len(I) > 1:
					for j,i in enumerate(I):
						src = specsrcs[i]
						print 'Source', src
						sdss = src.sdssobj
						flags  = sdss.objc_flags
						flags2 = sdss.objc_flags2
						print 'Bits set:'
						for bit,nm,desc in photo_flags1_info:
							if (1 << bit) & flags:
								print '  ', nm
						for bit,nm,desc in photo_flags2_info:
							if (1 << bit) & flags2:
								print '  ', nm

				Ts = Tspec[speci]
				sx,sy = wcs.positionToPixel(RaDecPos(Ts.ra, Ts.dec))
				for j,i in enumerate(I):
					src = specsrcs[i]

					modj = tractor.getModelImage(tim, [src], sky=False)
					plt.subplot(R,C, 4+j)
					plt.imshow(modj, **imc)
					plt.xticks([]); plt.yticks([])
					x,y = wcs.positionToPixel(src.getPosition())
					plt.plot([x], [y], 'r+', ms=8)
					plt.plot([sx], [sy], 'ro', mec='r', mfc='none', ms=8)
					plt.title('SDSS id %i' % src.sdssobj.id)

					plt.axis(ext)

				ps.savefig()

		return dict(fams=fams)

	def stage1002(self, fams=None, **kwargs):
		R = []
		for i,fam in enumerate(fams):
			pat = self.pat.replace('s%02i.pickle', 'g%02i-s%%02i.pickle' % i)
			ro = RunOneGroup(pat, fam=fam, parent=self, **kwargs)
			res = ro.runstage(1)
			R.append(res)
		newfams = [r['fam'] for r in R]
		return dict(fams=newfams)

	def stage1003(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None,
				  **kwargs):
		bandnum = band_index(band)
		pspat = self.pat.replace('-s%02i.pickle', '')
		print 'plotsequence pattern:', pspat
		ps = PlotSequence(pspat)

		tim = tractor.getImage(0)
		wcs = tim.getWcs()

		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		ax = plt.axis()
		plt.gray()
		for fam in fams:
			print 'Fam has', len(fam.specsrcs), 'spec', len(fam.srcs), 'srcs', len(fam.over), 'overlaps'
			for src in fam.over:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x],[y], 'o', mec='g', mfc='none',
						 mew=1.5, ms=6, alpha=0.5)
			for src in fam.srcs:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x],[y], 'o', mec='y', mfc='none',
						 mew=1.5, ms=8, alpha=0.5)
			for src in fam.specsrcs:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x],[y], 'o', mec='r', mfc='none',
						 mew=1.5, ms=10, alpha=0.5)
		plt.axis(ax)
		ps.savefig()

		for i,fam in enumerate(fams):
			plt.clf()
			plt.imshow(tim.getImage(), **imc)
			ax = plt.axis()
			plt.gray()
			print 'Fam has', len(fam.specsrcs), 'spec', len(fam.srcs), 'srcs', len(fam.over), 'overlaps'
			for src in fam.over:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x],[y], 'o', mec='g', mfc='none',
						 mew=1.5, ms=6, alpha=0.5)
			for src in fam.srcs:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x],[y], 'o', mec='y', mfc='none',
						 mew=1.5, ms=8, alpha=0.5)
			for src in fam.specsrcs:
				x,y = wcs.positionToPixel(src.getPosition())
				plt.plot([x],[y], 'o', mec='r', mfc='none',
						 mew=1.5, ms=10, alpha=0.5)
			plt.axis(ax)
			plt.title('Family %i' % i)
			ps.savefig()


		DLfunc = LuminosityDistance()
		log = np.log10
		h70 = 0.7

		def FP(sigma, mdev, rdev, ab, z, kz):
			r0 = rdev * np.sqrt(ab)
			mu0 = mdev + 2.5*log(2.*np.pi * r0**2) - kz - 10.*log(1.+z)
			DL = DLfunc(z)
			DA = DL / ((1.+z)**2)
			R0 = arcsec2rad(r0) * DA
			fpx = log(sigma) + 0.2 * (mu0 - 19.61)
			fpy = log(R0 * 1e3 / h70)
			if fpx.shape == fpy.shape:
				return fpx, fpy
			B = np.broadcast(fpx, fpy)
			xy = np.array([[x,y] for x,y in B])
			x,y = xy[:,0], xy[:,1]
			return x,y
					
		
		N = 100
		FPXY = []
		FPXY2 = []

		FPXY3 = []
		FPXY4 = []
		FPXY5 = []
		FPXY6 = []
		FPXY7 = []

		#vals = []
		for fam in fams:
			for src in fam.specsrcs:
				sdss = src.sdssobj
				spec = src.spec
				# [arcsec]
				re  = sdss.theta_dev[bandnum]
				dre = sdss.theta_deverr[bandnum]
				ab  = sdss.ab_dev[bandnum]
				dab = sdss.ab_deverr[bandnum]
				# [mag]
				mag  = sdss.devmag[bandnum]
				dmag = sdss.devmagerr[bandnum]
				# redshift
				z = spec.z
				dz = spec.zerr
				# HACK!! K-correction
				k = 0.
				dk = 0.
				# velocity dispersion - sigma [km/s]
				s = spec.veldisp
				ds = spec.veldisperr
				#vals.append((re, ab, mag, z, s))
				fpxy = FP(
					draw_gaussian(s,   ds,   N),
					draw_gaussian(mag, dmag, N),
					draw_gaussian(re,  dre,  N),
					draw_gaussian(ab,  dab,  N),
					draw_gaussian(z,   dz,   N),
					draw_gaussian(k,   dk,   N))
				FPXY.append(fpxy)
				FPXY2.append(FP(
					draw_gaussian(s,   0.,   N),
					draw_gaussian(mag, dmag, N),
					draw_gaussian(re,  dre,  N),
					draw_gaussian(ab,  dab,  N),
					draw_gaussian(z,   dz,   N),
					draw_gaussian(k,   dk,   N)))

				FPXY3.append(FP(draw_gaussian(s,   ds,   N),
								mag, re, ab, z, k))
				FPXY4.append(FP(s, draw_gaussian(mag, dmag, N),
								re, ab, z, k))
				FPXY5.append(FP(s, mag, draw_gaussian(re,  dre,  N),
								ab, z, k))
				FPXY6.append(FP(s, mag, re, draw_gaussian(ab,  dab,  N),
								z, k))
				FPXY7.append(FP(s, mag, re, ab,
								draw_gaussian(z,   dz,   N), k))


		for FPXY,tt in [(FPXY,''), (FPXY2, r': sigma\_err = 0'),
						(FPXY3, r': sigma error only'),
						(FPXY4, r': mag error only'),
						(FPXY5, r': r\_e error only'),
						(FPXY6, r': ab error only'),
						(FPXY7, r': z error only'),
						]:
			plt.clf()
			plt.gca().set_position([0.17, 0.1, 0.81, 0.80])
			for X,Y in FPXY:
				plt.plot(X,Y, 'k.', alpha=0.1)
			for X,Y in FPXY:
				I = (np.isfinite(X) * np.isfinite(Y))
				X = X[I]
				Y = Y[I]
				mx = np.mean(X)
				my = np.mean(Y)
				plt.plot(mx,my, 'ro',
						 # ms=4, alpha=0.5)
						 ms=4, mew=1.5, mec='r', mfc='none', alpha=0.5)
						 
				#dd = np.vstack((X-mx, Y-my))
				#print 'dd', dd.shape
				xx = np.mean((X-mx)**2)
				xy = np.mean((X-mx)*(Y-my))
				yy = np.mean((Y-my)**2)
				C = np.array([[xx,xy],[xy,yy]])
				s,v = np.linalg.eigh(C)
				print 'evals', s
				print 'evecs', v
				[mini,maxi] = np.argsort(s)
				a = v[maxi,:]
				angle = np.rad2deg(np.arctan2(a[1],a[0]))
				from matplotlib.patches import Ellipse
				gca().add_artist(Ellipse([mx,my], np.sqrt(s[maxi]), np.sqrt(s[mini]),
										 angle, ec='m', fc='none', lw=1.5))
										 

			xl,xh = [0., 3.5]
			yl,yh = [-1, 2.0]
			X = np.array([xl,xh])
			ta = dict(color='r', lw=2,)
			# eyeballed Bernardi relation for i-band
			plt.plot(X, (X - 2.) * (1.52) + 0.2, **ta)
			plt.xlabel('log(sigma) + 0.2 ($\mu_0$ - 19.61)')
			plt.ylabel('log($R_0$) [kpc / h]')
			plt.ylim(yl,yh)
			plt.xlim(xl,xh)
			plt.title('FP from SDSS error estimates' + tt)
			ps.savefig()

		# vals = np.array(vals)
		# plt.clf()
		# for i,(s,v) in enumerate(zip(['re','ab','mag','z','s'], vals.T)):
		# 	plt.subplot(2,3, i+1)
		# 	plt.hist(v, 25)
		# 	plt.title(s)
		# ps.savefig()

		return dict(ps=ps)


	def stage1004(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None, **kwargs):
		bandnum = band_index(band)
		pspat = self.pat.replace('-s%02i.pickle', '')
		print 'plotsequence pattern:', pspat
		ps = PlotSequence(pspat, format='%03i')

		plt.figure(figsize=(5,4))
		plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.92)

		tim = tractor.getImage(0)
		wcs = tim.getWcs()

		# the big one
		
		fam = fams[22]
		# fam.srcs
		# fam.specsrcs
		# family.ext = ext
		# family.mpatch = mpatch

		mpatch = fam.mpatch
		slc = mpatch.getSlice()
		print 'Slice', slc

		fam.mpatch = Patch(mpatch.x0, 100,
						   mpatch.patch[100-mpatch.y0:, :2000 - mpatch.x0])
		mpatch = fam.mpatch
		slc = mpatch.getSlice()
		print 'Slice', slc

		fam.ext = list(fam.ext)
		# Trim ugly edges
		fam.ext[1] = 2000
		fam.ext[2] = 100
		rext = fam.ext[2:] + fam.ext[:2]

		print 'mpatch extent:', mpatch.getExtent()
		print 'fam ext', fam.ext




		print 'Spectroscopic sources:'
		for src in fam.specsrcs:
			print '  ', src
		#I = np.argsort([src.getBrightness().getMag(band) for src in fam.srcs])
		#print 'Brightest sources:'
		#for i in I[:10]:


		plt.clf()
		plt.imshow(tim.getImage(), **imc)
		plt.gray()
		print 'Fam has', len(fam.specsrcs), 'spec', len(fam.srcs), 'srcs', len(fam.over), 'overlaps'
		for src in fam.over:
			x,y = wcs.positionToPixel(src.getPosition())
			plt.plot([x],[y], 'o', mec='g', mfc='none',
					 mew=1.5, ms=6, alpha=0.5)
		for src in fam.srcs:
			x,y = wcs.positionToPixel(src.getPosition())
			plt.plot([x],[y], 'o', mec='y', mfc='none',
					 mew=1.5, ms=8, alpha=0.5)
		for src in fam.specsrcs:
			x,y = wcs.positionToPixel(src.getPosition())
			plt.plot([x],[y], 'o', mec='r', mfc='none',
					 mew=1.5, ms=10, alpha=0.5)
			if isinstance(src, DevGalaxy):
				plt.text(x, y, 'D', color='r')
			elif isinstance(src, ExpGalaxy):
				plt.text(x, y, 'E', color='r')
			elif isinstance(src, CompositeGalaxy):
				plt.text(x, y, 'C', color='r')
			elif isinstance(src, PointSource):
				plt.text(x, y, 'P', color='r')
		plt.axis(fam.ext)
		ps.savefig()

		plt.clf()
		#plt.imshow(tim.getImage(), **imc)
		plt.imshow(tim.getImage().T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		#plt.axis(fam.ext)
		plt.axis(rext)
		plt.title('Data')
		ps.savefig()

		inbox = []
		for src in tractor.getCatalog():
			if src in fam.srcs or src in fam.over:
				continue
			x,y = wcs.positionToPixel(src.getPosition())
			if (x >= fam.ext[0] and x <= fam.ext[1] and
				y >= fam.ext[2] and y <= fam.ext[3]):
				inbox.append(src)

		noise = np.random.normal(size=tim.shape)
		I = (tim.getInvvar() == 0)
		noise[I] = 0.
		I = np.logical_not(I)
		noise[I] *= 1./(tim.getInvError()[I])

		ima = dict(interpolation='nearest', origin='lower')
		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)
		imchi2 = ima.copy()
		imchi2.update(vmin=-50, vmax=50)

		def plotem(srcs=None):
			self.plots(tractor, imc, imchi, rext, ps, srcs=srcs, noise=noise,
					   imchi2=imchi2)

		plotem(srcs=fam.srcs + fam.over + inbox)


		ix0,iy0 = wcs.x0,wcs.y0
		subext = mpatch.getExtent()
		print 'Extent', subext
		subext = [subext[0]+ix0, subext[1]+ix0,
				  subext[2]+iy0, subext[3]+iy0]
		print 'Subimage extent:', subext

		#imsub = imc.copy()
		#imsub.update(extent=subext)

		# Add spline sky
		mod = tractor.getModelImage(tim)
		submod = mod[slc]
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

		### Zero out the invvar except for our ROI
		ie = tim.getInvError()
		orig_ie = ie.copy()
		sigma = 1./np.median(orig_ie)
		print 'Sigma', sigma
		ssky.setPriorSmoothness(sigma * 0.1)

		sub_ie = np.zeros_like(ie)
		sub_ie[slc] = ie[slc]

		sub_cat = Catalog()
		for src in fam.srcs + fam.over + inbox:
			mod = tractor.getModelImage(tim, [src])
			print 'Source', src
			print '  brightness:', src.getBrightness()
			ms = mod.max()/sigma
			print '  max model pixel:', ms, 'sigma'
			if ms < 3.:
				continue
			ms = (mod * sub_ie).max()
			print '  max model pixel * sub_ie:', ms
			if ms < 3.:
				continue
			sub_cat.append(src)

		tim.setInvvar(sub_ie**2)
		tractor.setCatalog(sub_cat)
		
		#plt.clf()
		#plt.imshow(tim.getInvError())
		#ps.savefig()

		subsky = SubSky(ssky, slc)		
		tim.sky = subsky

		def plotsky():
			skyim = np.zeros_like(tim.getImage()[slc])
			ssky.addTo(skyim)
			plt.clf()
			# plt.imshow(skyim, **imsub)
			plt.imshow(skyim.T, **imc)
			plt.gray()
			plt.title('Spline sky model')
			# plt.axis(fam.ext)
			# plt.axis(rext)
			plt.xticks([]); plt.yticks([])
			ps.savefig()

		print 'After cutting to sub_cat, sub_ie, etc.'
		plotem()



		print 'Initial:'
		print 'lnLikelihood', tractor.getLogLikelihood()
		print 'lnPrior', tractor.getLogPrior()
		print 'lnProb', tractor.getLogProb()

		tractor.freezeParam('images')
		tractor.catalog.freezeAllRecursive()
		tractor.catalog.thawPathsTo(band)
		#tractor.catalog.freezeAllBut(*fam.srcs)
		tractor.catalog.thawAllParams()

		print 'Fluxes: opt'
		for nm in tractor.getParamNames():
			print '  ', nm

		self.optloop(tractor, scale_columns=False)
		tractor.catalog.thawAllRecursive()

		plotem()

		print 'After optimizing fluxes:'
		print 'lnLikelihood', tractor.getLogLikelihood()
		print 'lnPrior', tractor.getLogPrior()
		print 'lnProb', tractor.getLogProb()



		subsubcat = Catalog()
		cat = tractor.getCatalog()
		mod0 = tractor.getModelImage(tim, cat)
		iv = tim.getInvvar()
		data = tim.getImage()
		lnp0 = -0.5 * np.sum((data - mod0)**2 * iv)
		print 'lnp0:', lnp0
		for i,src in enumerate(cat):
			print 'Source', src
			print '  brightness:', src.getBrightness()
			mod = tractor.getModelImage(tim, cat[:i] + cat[i+1:])
			#lnp1 = tractor.getLogLikelihood()
			lnp = -0.5 * np.sum((data - mod)**2 * iv)
			dlnp = lnp - lnp0
			print '  removing it: lnp:', lnp
			print '  dlnp:', dlnp
			print '  delta-mod:', np.abs((mod - mod0) * tim.getInvError()).sum(), 'sigma'
			if dlnp < -1:
				subsubcat.append(src)
			else:
				print '  dropping it'

		tractor.setCatalog(subsubcat)

		print 'After dropping small dlnp:'
		plotem()

		I = np.argsort([src.getBrightness().getMag(band)
						for src in tractor.getCatalog()])
		#for i,src in enumerate(tractor.getCatalog()):
		for i in I:
			src = tractor.getCatalog()[i]
			mod = tractor.getModelImage(tim, [src])
		
			# plt.clf()
			# plt.imshow(mod.T, **imc)
			# plt.gray()
			# plt.xticks([]); plt.yticks([])
			# x,y = wcs.positionToPixel(src.getPosition())
			# plt.plot([y],[x],'ro', ms=10, mec='r', mfc='none')
			# plt.axis(rext)
			# plt.title('Model for source %i' % i)
			# ps.savefig()
			print 'Source', i, ':', src
			ms = (mod * sub_ie).max()
			print '  max model pixel * sub_ie:', ms

		# HACK!!  Remove ones that offend me
		# dropsrcs = [tractor.getCatalog()[i] for i in [50,53]]
		# for src in dropsrcs:
		# 	tractor.getCatalog().remove(src)
		#print 'After removing offensive ones:'
		#plotem()

		# This isn't necessary -- it's linear, after all; I got confused
		# because I was thawing all params, not just fluxes.
		# while True:
		# 	gotone = False
		# 	for i,src in enumerate(tractor.getCatalog()):
		# 		print 'Optimizing source', i
		# 		tractor.catalog.freezeAllBut(i)
		# 		src.freezeAllParams()
		# 		src.thawPathsTo(band)
		# 		for nm in tractor.getParamNames():
		# 			print '  ', nm
		# 		dlnp,X,alpha = tractor.optimize()
		# 		print 'delta-logprob', dlnp
		# 		if dlnp > 1:
		# 			gotone = True
		# 	if not gotone:
		# 		break
		# 
		# 	plotem()

		return dict(ps=ps)

	def plots(self, tractor, imc, imchi, rext, ps, srcs=None, noise=None, imchi2=None):
		tim = tractor.getImage(0)
		mod = tractor.getModelImage(tim, srcs=srcs)

		plt.clf()
		plt.imshow(mod.T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		plt.axis(rext)
		plt.title('Model')
		ps.savefig()

		if noise is not None:
			plt.clf()
			plt.imshow((mod + noise).T, **imc)
			plt.xticks([]); plt.yticks([])
			plt.gray()
			plt.axis(rext)
			plt.title('Model')
			ps.savefig()

		chi = (tim.getImage() - mod) * tim.getInvError()
		plt.clf()
		plt.imshow(chi.T, **imchi)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		plt.axis(rext)
		plt.title('Chi')
		ps.savefig()

		if imchi2 is not None:
			plt.clf()
			plt.imshow(chi.T, **imchi2)
			plt.gray()
			plt.xticks([]); plt.yticks([])
			plt.axis(rext)
			plt.title('Chi (+-50)')
			ps.savefig()


	def stage1005(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None, **kwargs):
		bandnum = band_index(band)

		fam = fams[22]
		tim = tractor.getImage(0)

		# tractor.catalog.freezeAllRecursive()
		# tractor.catalog.thawPathsTo(band)
		# tractor.catalog.thawAllParams()
		# print 'Fluxes: opt'
		# for nm in tractor.getParamNames():
		# 	print '  ', nm
		# self.optloop(tractor)
		# tractor.catalog.thawAllRecursive()
		# 
		# plotem()
		# 
		# print 'Fluxes 2:'
		# print 'lnLikelihood', tractor.getLogLikelihood()
		# print 'lnPrior', tractor.getLogPrior()
		# print 'lnProb', tractor.getLogProb()

		mpatch = fam.mpatch
		slc = mpatch.getSlice()
		ima = dict(interpolation='nearest', origin='lower')
		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)
		imchi2 = ima.copy()
		imchi2.update(vmin=-50, vmax=50)

		noise = np.random.normal(size=tim.shape)
		I = (tim.getInvvar() == 0)
		noise[I] = 0.
		I = np.logical_not(I)
		noise[I] *= 1./(tim.getInvError()[I])

		# ssky is the SplineSky (not the SubSky)
		ssky = tim.getSky().real
		rext = fam.ext[2:] + fam.ext[:2]
		#print 'Ssky:', ssky
		#print '.real', ssky.real
		#print '.slc', ssky.slc

		def plotem(srcs=None):
			self.plots(tractor, imc, imchi, rext, ps, srcs=srcs, noise=noise,
					   imchi2=imchi2)
		def plotsky():
			skyim = np.zeros_like(tim.getImage()[slc])
			#print 'Slice', slc
			#print 'Skyim', skyim.shape
			ssky.addTo(skyim)
			plt.clf()
			plt.imshow(skyim.T, **imc)
			plt.gray()
			plt.title('Spline sky model')
			# plt.axis(rext)
			plt.xticks([]); plt.yticks([])
			ps.savefig()


		plotsky()
		plotem()

		# wcs = tim.getWcs()
		# mod = tractor.getModelImage(tim)
		# plt.clf()
		# plt.imshow(mod.T, **imc)
		# plt.gray()
		# plt.xticks([]); plt.yticks([])
		# for src in tractor.getCatalog():
		# 	x,y = wcs.positionToPixel(src.getPosition())
		# 	plt.plot([y],[x],'ro', ms=10, mec='r', mfc='none')
		# for src in fam.specsrcs:
		# 	x,y = wcs.positionToPixel(src.getPosition())
		# 	plt.plot([y],[x],'ro', ms=8, mec=(0,1,0), mfc='none')
		# plt.axis(rext)
		# plt.title('Model')
		# ps.savefig()


		tractor.thawParam('images')
		tim.freezeAllBut('sky')
		tractor.catalog.thawAllRecursive()
		tractor.catalog.freezeAllParams()
		keptspecs = []
		for src in fam.specsrcs:
			if src in tractor.catalog:
				keptspecs.append(src)
				tractor.catalog.thawParam(src)
				src.freezeAllParams()
				src.thawPathsTo(band)
			else:
				print 'Missing src:', src
				wcs = tim.getWcs()
				x,y = wcs.positionToPixel(src.getPosition())
				print 'x,y', x,y
		# Some of them were removed??
		#tractor.catalog.freezeAllBut(*fam.specsrcs)

		print 'Spline sky: opt'
		for nm in tractor.getParamNames():
			print '  ', nm

		j=0
		while True:
			print '-------------------------------------'
			print 'Optimizing: step', j
			print '-------------------------------------'
		   	dlnp,X,alpha = tractor.optimize()
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

			print 'After opt:'
			print 'lnLikelihood', tractor.getLogLikelihood()
			print 'lnPrior', tractor.getLogPrior()
			print 'lnProb', tractor.getLogProb()

			plotem()
			plotsky()
			
		print 'After opt:'
		print 'lnLikelihood', tractor.getLogLikelihood()
		print 'lnPrior', tractor.getLogPrior()
		print 'lnProb', tractor.getLogProb()

		plotem()
		plotsky()

		### Thaw all the spectro sources

		for src in keptspecs:
			src.thawAllParams()

		print 'Spline sky: opt'
		for nm in tractor.getParamNames():
			print '  ', nm

		j=0
		while True:
			print '-------------------------------------'
			print 'Optimizing: step', j
			print '-------------------------------------'
		   	dlnp,X,alpha = tractor.optimize()
			print 'delta-logprob', dlnp
			nup = 0
			for src in tractor.getCatalog():
				for b in src.getBrightnesses():
					f = b.getFlux(band)
					if f < 0:
						nup += 1
						b.setFlux(band, 0.)
			print 'Clamped', nup, 'fluxes up to zero'
			j += 1
			print 'After opt:'
			print 'lnLikelihood', tractor.getLogLikelihood()
			print 'lnPrior', tractor.getLogPrior()
			print 'lnProb', tractor.getLogProb()
			plotem()
			plotsky()
			if dlnp < 1:
				break


	def stage1006(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None, run=None, camcol=None, field=None, **kwargs):
		bandnum = band_index(band)
		fam = fams[22]
		tim = tractor.getImage(0)
		mpatch = fam.mpatch
		slc = mpatch.getSlice()
		ima = dict(interpolation='nearest', origin='lower')
		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)
		imchi2 = ima.copy()
		imchi2.update(vmin=-50, vmax=50)

		noise = np.random.normal(size=tim.shape)
		I = (tim.getInvvar() == 0)
		noise[I] = 0.
		I = np.logical_not(I)
		noise[I] *= 1./(tim.getInvError()[I])

		# ssky is the SplineSky (not the SubSky)
		ssky = tim.getSky().real
		rext = fam.ext[2:] + fam.ext[:2]

		def plotem(srcs=None):
			self.plots(tractor, imc, imchi, rext, ps, srcs=srcs, noise=noise,
					   imchi2=imchi2)
		def plotsky():
			skyim = np.zeros_like(tim.getImage()[slc])
			ssky.addTo(skyim)
			plt.clf()
			plt.imshow(skyim.T, **imc)
			plt.gray()
			plt.title('Spline sky model')
			plt.xticks([]); plt.yticks([])
			ps.savefig()

		wcs = tim.getWcs()
		mod = tractor.getModelImage(tim)
		plt.clf()
		plt.imshow(mod.T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		for i,src in enumerate(tractor.getCatalog()):
			x,y = wcs.positionToPixel(src.getPosition())
			c = 'r'
			if src in fam.specsrcs:
				c = (0,1,0)
			elif src in fam.srcs:
				c = (0,1,1)
			plt.plot([y],[x],'o', ms=10, mec=c, mfc='none')
			if isinstance(src, PointSource):
				st = 'P'
			elif isinstance(src, DevGalaxy):
				st = 'D'
			elif isinstance(src, ExpGalaxy):
				st = 'E'
			elif isinstance(src, CompositeGalaxy):
				st = 'C'
			plt.text(y+5, x+5, '%i' % i + st, color='w', fontsize=8)
			print i, 'mag', src.getBrightness().getMag(band)
		plt.axis(rext)
		plt.title('Model')
		ps.savefig()


		newcat = Catalog()
		bright = 19.
		bthresh = NanoMaggies.magToNanomaggies(bright - 1.)
		for src in tractor.getCatalog():
			if ((not isinstance(src, PointSource)) or
				(not src.getBrightness().getMag(band) < bright)):
				newcat.append(src)
				continue
			ss = SdssPointSource(src.getPosition(), src.getBrightness(), thresh=bthresh)
			newcat.append(ss)
			# FIXME -- replace entry in fam.srcs, fam.specsrcs
			if src in fam.srcs:
				i = fam.srcs.index(src)
				fam.srcs[i] = ss
			if src in fam.specsrcs:
				i = fam.specsrcs.index(src)
				fam.specsrcs[i] = ss

		tractor.setCatalog(newcat)

		print 'Wrapping PSF in SdssBrightPSF'
		sdss = DR9()
		psfield = sdss.readPsField(run, camcol, field)
		(a1,s1, a2,s2, a3,sigmap,beta) = psfield.getPowerLaw(bandnum)
		mypsf = SdssBrightPSF(tim.getPsf(), a1,s1,a2,s2,a3,sigmap,beta)
		print 'PSF:', mypsf
		tim.setPsf(mypsf)

		mod = tractor.getModelImage(tim)
		plt.clf()
		plt.imshow(mod.T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		for i,src in enumerate(tractor.getCatalog()):
			x,y = wcs.positionToPixel(src.getPosition())
			c = 'r'
			if src in fam.specsrcs:
				c = (0,1,0)
			elif src in fam.srcs:
				c = (0,1,1)
			plt.plot([y],[x],'o', ms=10, mec=c, mfc='none')
			if isinstance(src, SdssPointSource):
				st = 'B'
			elif isinstance(src, PointSource):
				st = 'P'
			elif isinstance(src, DevGalaxy):
				st = 'D'
			elif isinstance(src, ExpGalaxy):
				st = 'E'
			elif isinstance(src, CompositeGalaxy):
				st = 'C'
			plt.text(y+5, x+5, '%i' % i + st, color='w', fontsize=8)
			#print i, 'mag', src.getBrightness().getMag(band)
		plt.axis(rext)
		plt.title('Model')
		ps.savefig()

		plotem()

		I = np.argsort([src.getBrightness().getMag(band)
						for src in tractor.getCatalog()])
		tractor.freezeParam('images')

		for srcset in [ fam.specsrcs, fam.srcs, tractor.getCatalog() ]:
			while True:
				activeset = srcset
				# Keep resetting the active set and checking them all until we
				# find that none of them move.
				firsttime = True
				print 'Resetting active set to', len(activeset), 'sources'
				while True:
					# Narrow down to just the sources that are actually changing
					print
					print 'Looping through', len(activeset), 'active sources'
					moved = []
					for j,i in enumerate(I):
						src = tractor.getCatalog()[i]
						if not src in activeset:
							continue
						tractor.catalog.freezeAllBut(i)
						src.thawAllRecursive()
						print
						print 'Optimizing source', i, '(%i of %i)' % (j, len(I))
						for nm in tractor.getParamNames():
							print '  ', nm
						dlnp,X,alpha = tractor.optimize()
						print 'delta-logprob', dlnp
						if dlnp > 1:
							moved.append(src)
					if len(moved) == 0:
						break
					firsttime = False
					activeset = moved
					plotem()
				if firsttime:
					break
		# for srcset in [ fam.specsrcs, fam.srcs, tractor.getCatalog() ]:
		# 	while True:
		# 		gotone = False
		# 		for j,i in enumerate(I):
		# 			src = tractor.getCatalog()[i]
		# 			if not src in srcset:
		# 				continue
		# 			tractor.catalog.freezeAllBut(i)
		# 			src.thawAllRecursive()
		# 			print
		# 			print 'Optimizing source', i, '(%i of %i)' % (j, len(I))
		# 			for nm in tractor.getParamNames():
		# 				print '  ', nm
		# 			dlnp,X,alpha = tractor.optimize()
		# 			print 'delta-logprob', dlnp
		# 			if dlnp > 1:
		# 				gotone = True
		# 		if not gotone:
		# 			break
		# 		plotem()


	def stage1007(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None, run=None, camcol=None, field=None, **kwargs):
		bandnum = band_index(band)
		fam = fams[22]
		tim = tractor.getImage(0)
		mpatch = fam.mpatch
		slc = mpatch.getSlice()
		ima = dict(interpolation='nearest', origin='lower')
		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)
		imchi2 = ima.copy()
		imchi2.update(vmin=-50, vmax=50)

		noise = np.random.normal(size=tim.shape)
		I = (tim.getInvvar() == 0)
		noise[I] = 0.
		I = np.logical_not(I)
		noise[I] *= 1./(tim.getInvError()[I])

		# ssky is the SplineSky (not the SubSky)
		ssky = tim.getSky().real
		rext = fam.ext[2:] + fam.ext[:2]

		def plotem(srcs=None):
			self.plots(tractor, imc, imchi, rext, ps, srcs=srcs, noise=noise,
					   imchi2=imchi2)
		def plotsky():
			skyim = np.zeros_like(tim.getImage()[slc])
			ssky.addTo(skyim)
			plt.clf()
			plt.imshow(skyim.T, **imc)
			plt.gray()
			plt.title('Spline sky model')
			plt.xticks([]); plt.yticks([])
			ps.savefig()


		tractor.thawParam('images')
		I = np.argsort([src.getBrightness().getMag(band)
						for src in tractor.getCatalog()])
		for srcset in [ fam.specsrcs ]: #, fam.srcs, tractor.getCatalog() ]:
			while True:
				activeset = srcset
				# Keep resetting the active set and checking them all until we
				# find that none of them move.
				firsttime = True
				print 'Resetting active set to', len(activeset), 'sources'
				while True:
					# Narrow down to just the sources that are actually changing
					print
					print 'Looping through', len(activeset), 'active sources'
					moved = []
					for j,i in enumerate(I):
						src = tractor.getCatalog()[i]
						if not src in activeset:
							continue
						tractor.catalog.freezeAllBut(i)
						src.thawAllRecursive()
						print
						print 'Optimizing source', i, '(%i of %i)' % (j, len(I))
						for nm in tractor.getParamNames():
							print '  ', nm
						dlnp,X,alpha = tractor.optimize()
						print 'delta-logprob', dlnp
						if dlnp > 1:
							moved.append(src)
					if len(moved) == 0:
						break
					firsttime = False
					activeset = moved

					plotem()
					plotsky()

				if firsttime:
					break


	def stage1008(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None, run=None, camcol=None, field=None, **kwargs):
		ps = PlotSequence('clustersky', suffix='pdf')

		bandnum = band_index(band)
		fam = fams[22]
		tim = tractor.getImage(0)
		mpatch = fam.mpatch
		slc = mpatch.getSlice()
		ima = dict(interpolation='nearest', origin='lower')
		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)
		imchi2 = ima.copy()
		imchi2.update(vmin=-50, vmax=50)

		print 'kwargs:', kwargs.keys()
		print 'kwargs', kwargs

		noise = np.random.normal(size=tim.shape)
		I = (tim.getInvvar() == 0)
		noise[I] = 0.
		I = np.logical_not(I)
		noise[I] *= 1./(tim.getInvError()[I])

		# ssky is the SplineSky (not the SubSky)
		ssky = tim.getSky().real
		rext = fam.ext[2:] + fam.ext[:2]

		# def plotem(srcs=None):
		# 	self.plots(tractor, imc, imchi, rext, ps, srcs=srcs, noise=noise,
		# 			   imchi2=imchi2)
		# def plotsky():
		# 	skyim = np.zeros_like(tim.getImage()[slc])
		# 	ssky.addTo(skyim)
		# 	plt.clf()
		# 	plt.imshow(skyim.T, **imc)
		# 	plt.gray()
		# 	plt.title('Spline sky model')
		# 	plt.xticks([]); plt.yticks([])
		# 	ps.savefig()

		
		print 'rext', rext

		P = unpickle_from_file('clusky-a1656-r5115-c5-f0150-i-s1001.pickle')
		t0 = P['tractor']
		ss0 = P['fams'][22].specsrcs

		print 'Optimize spectro sources:'
		for src in fam.specsrcs:
			print '  ', src

		print 'Original spectro sources:'
		for src in ss0:
			print '  ', src

		#rd0 = [src.getPosition() for src in t0.getCatalog()]
		#ra0  = np.array([rd.ra for rd in rd0])
		#dec0 = np.array([rd.dec for rd in rd0])

		plt.clf()
		plt.imshow(tim.getImage().T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		plt.axis(rext)
		plt.title('Data')
		ps.savefig()

		mod0 = t0.getModelImage(0)
		plt.clf()
		plt.imshow((mod0+noise).T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		plt.axis(rext)
		plt.title('SDSS Model')
		ps.savefig()

		mod = tractor.getModelImage(tim)
		plt.clf()
		plt.imshow((mod+noise).T, **imc)
		plt.gray()
		plt.xticks([]); plt.yticks([])
		plt.axis(rext)
		plt.title('Optimized Model')
		ps.savefig()

		skyim = np.zeros_like(tim.getImage()[slc])
		ssky.addTo(skyim)
		plt.clf()
		plt.imshow(skyim.T, **imc)
		plt.gray()
		plt.title('Sky model')
		plt.xticks([]); plt.yticks([])
		ps.savefig()

		print 'skyim', skyim.min(), skyim.max()

		print 'imc', imc
		imsky = imc.copy()
		imsky.update(vmax=skyim.max())

		plt.clf()
		plt.imshow(skyim.T, **imsky)
		plt.gray()
		plt.title('Sky model')
		plt.xticks([]); plt.yticks([])
		ps.savefig()

		imsky2 = ima.copy()
		imsky2.update(vmin=0, vmax=skyim.max())

		plt.clf()
		plt.imshow(skyim.T, **imsky2)
		plt.gray()
		plt.title('Sky model')
		plt.xticks([]); plt.yticks([])
		ps.savefig()


	def stage1009(self, fams=None, band=None, ps=None, tractor=None,
				  imc=None, run=None, camcol=None, field=None, **kwargs):
		ps = PlotSequence('clusky1009')

		fam = fams[22]
		rext = fam.ext[2:] + fam.ext[:2]

		tractor.thawParam('images')
		tractor.catalog.freezeAllParams()
		for src in fam.specsrcs:
			if src in tractor.catalog:
				tractor.catalog.thawParam(src)
				src.thawAllRecursive()

		import emcee
		lnp0 = tractor.getLogProb()
		print 'Tractor: active params'
		for nm in tractor.getParamNames():
			print '  ', nm

		p0 = np.array(tractor.getParams())
		ndim = len(p0)
		nw = 2*ndim
		print 'ndim', ndim
		print 'nw', nw

		sampler = emcee.EnsembleSampler(nw, ndim, tractor,
										threads=8)

		steps = np.array(tractor.getStepSizes())
		print 'steps: len', len(steps)

		# Scale the step sizes by the size of their derivatives.
		cs = tractor.getParameterScales()

		pp0 = np.vstack([p0 + 1e-4 * steps / cs *
						 np.random.normal(size=len(steps))
						 for i in range(nw)])
		alllnp = []
		allp = []

		lnp = None
		pp = pp0
		rstate = None
		for step in range(100):
			print 'Taking emcee step', step
			pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
			#print 'lnprobs:', lnp

			if (step+0) % 10 == 0:
				for k,(p,x) in enumerate(zip(lnp,pp)):
					tractor.setParams(x)

					#chi = tractor.getChiImage(0)
					modj = tractor.getModelImage(0)

					plt.clf()
					plt.imshow(modj.T, **imc)
					plt.gray()
					plt.xticks([]); plt.yticks([])
					plt.axis(rext)
					plt.title('Sampled Model: step %i' % step)
					ps.savefig()

					if k == 4:
						break

			print 'Max lnprob:', max(lnp)
			print 'Std in lnprobs:', np.std(lnp)

			alllnp.append(lnp.copy())
			allp.append(pp.copy())

			plt.clf()
			plt.gca().set_position([0.1, 0.1, 0.88, 0.80])
			plt.plot(np.array(alllnp) - np.array(alllnp).max(), 'k-', alpha=0.1)
			ps.savefig()

		return dict(alllnp=alllnp, allp=allp)


	def stage101x(self, stage=None, fams=None, band=None, ps=None, tractor=None,
				  imc=None, run=None, camcol=None, field=None,
				  alllnp=None, allp=None, **kwargs):
		ps = PlotSequence('clusky%04i' % stage)

		fam = fams[22]
		rext = fam.ext[2:] + fam.ext[:2]

		# alllnp = np.array(alllnp)
		# allp = np.array(allp)
		# print 'all lnp', alllnp.shape
		# print 'all p', allp.shape
		# plt.clf()
		# plt.plot(alllnp, 'k-', alpha=0.1)
		# ps.savefig()
		# print 'tractor params:', tractor.numberOfParams()
		# for nm in tractor.getParamNames():
		# 	print '  ', nm

		import emcee
		# allp shape (100, 422, 211) == nsteps, nw, ndim

		step0, nw, ndim = np.array(allp).shape

		# DEBUG
		# global dpool
		# import debugpool
		# threads = 8
		# dpool = debugpool.DebugPool(threads)
		# Time.add_measurement(debugpool.DebugPoolMeas(dpool))

		sampler = emcee.EnsembleSampler(nw, ndim, tractor) #, pool=dpool)

		pp0 = allp[-1]

		lnp = None
		pp = pp0
		rstate = None
		for step in range(step0, step0 + 100):
			print 'Taking emcee step', step
			t0 = Time()
			pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
			print 'Emcee step took:'
			print Time() - t0
			if (step+0) % 10 == 0:
				for k,(p,x) in enumerate(zip(lnp,pp)):
					tractor.setParams(x)

					#chi = tractor.getChiImage(0)
					modj = tractor.getModelImage(0)

					plt.clf()
					plt.imshow(modj.T, **imc)
					plt.gray()
					plt.xticks([]); plt.yticks([])
					plt.axis(rext)
					plt.title('Sampled Model: step %i' % step)
					ps.savefig()

					if k == 4:
						break

				plt.clf()
				plt.gca().set_position([0.1, 0.1, 0.88, 0.80])
				plt.plot(np.array(alllnp) - np.array(alllnp).max(), 'k-', alpha=0.1)
				ps.savefig()

			print 'Max lnprob:', max(lnp)
			print 'Std in lnprobs:', np.std(lnp)

			alllnp.append(lnp.copy())
			allp.append(pp.copy())

		return dict(alllnp=alllnp, allp=allp)

	# again again
	stage1010 = stage101x
	stage1011 = stage101x
	stage1012 = stage101x
	stage1013 = stage101x
	stage1014 = stage101x
	stage1015 = stage101x
	stage1016 = stage101x
	stage1017 = stage101x
	stage1018 = stage101x

	def stage1020(self, allp=None, tractor=None, **kwa):
		allp = np.array(allp)
		allp = allp[-100::10, :, :]
		steps, nw, ndim = allp.shape

		ps = PlotSequence('corr')

		print 'Shape', allp.shape
		allp = allp.reshape((-1, ndim))
		print 'Reshaped to', allp.shape
		ns,ndim = allp.shape

		print 'Params:'
		for nm in tractor.getParamNames():
			print '  ', nm

		names = tractor.getParamNames()
		# Look for the largest correlation coefficients
		#ia = [i for i,nm in enumerate(names) if 'sky' in name]
		#ib = [i for i,nm in enumerate(names) if 'shape' in name or 'brightness' in name]

		sigmas = np.std(allp, axis=0)
		means = np.mean(allp, axis=0)
		print 'sigmas', sigmas.shape

		aa = (allp - means[np.newaxis,:]) / sigmas[np.newaxis,:]

		cov = np.dot(aa.T, aa)
		print 'cov', cov.shape

		for i,j in zip(*np.unravel_index(np.argsort(-np.abs(cov.flat)), cov.shape)):
			if j >= i:
				continue
			print 'Cov:', i, j, cov[i,j], names[i], names[j]

			plt.clf()
			plt.gca().set_position([0.17, 0.1, 0.81, 0.80])
			plt.subplot(2,1,1)
			plt.plot(allp[:,i] - means[i], allp[:,j] - means[j], 'r.', alpha=0.25)
			plt.xlabel(names[i] + '+ %g' % means[i])
			plt.ylabel(names[j] + '+ %g' % means[j])
			plt.subplot(2,1,2)
			plt.plot((allp[:,i] - means[i]) / sigmas[i], 'r-')
			plt.plot((allp[:,j] - means[j]) / sigmas[j], 'b-')
			ps.savefig()

		#cc = []
		#ij = []
		#for i in range(ns):
		#		for j in range(i+1, ns):
		#		aa = 

		
		
		
	def stage0(self, run=None, camcol=None, field=None,
			   band=None, ra=None, dec=None, **kwargs):
		#
		S = (self.R * 60.) / 0.396
		getim = st.get_tractor_image_dr9
		getsrc = st.get_tractor_sources_dr9
		tim,tinf = getim(run, camcol, field, band,
						 roiradecsize=(ra, dec, S), nanomaggies=True)
		if tim is None:
			return None
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

		fn = get_spectro_table(ra, dec, self.aco)
		print 'Looking for', fn
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

		#ps = PlotSequence(self.pat.replace('-s%02i.pickle', ''))
		ps = PlotSequence(self.pat.replace('-s%02i.pickle', '') + '-groups')


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
			#if i != 1:
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
			

	def stage208(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None, roi=None, tinf=None,
				 subcat=None, sgroup=None, group=None, over=None,
				 orig_ie=None, mpatch=None, imc=None, ps=None,
				 old_sgroup=None, new_sgroup=None, sgi=None,
				 cat=None,
				 **kwargs):
		tim = tractor.getImage(0)
		wcs = tim.getWcs()
		ix0,iy0 = wcs.x0,wcs.y0
		slc = mpatch.getSlice()
		subext = mpatch.getExtent()
		print 'Extent', subext
		subext = [subext[0]+ix0, subext[1]+ix0,
				  subext[2]+iy0, subext[3]+iy0]
		imsub = imc.copy()
		imsub.update(extent=subext)

		import emcee
		lnp0 = tractor.getLogProb()
		print 'Tractor: active params'
		for nm in tractor.getParamNames():
			print '  ', nm

		p0 = np.array(tractor.getParams())
		ndim = len(p0)
		#nw = max(50, 2*ndim)
		nw = 2*ndim
		print 'ndim', ndim
		print 'nw', nw

		sampler = emcee.EnsembleSampler(nw, ndim, tractor,
										threads=8)

		steps = np.array(tractor.getStepSizes())
		print 'steps: len', len(steps)
		pp0 = np.vstack([p0 + 1e-2 * steps *
						 np.random.normal(size=len(steps))
						 for i in range(nw)])

		alllnp = []
		allp = []

		lnp = None
		pp = pp0
		rstate = None
		for step in range(100):
			print 'Taking emcee step', step
			pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
			print 'lnprobs:', lnp

			if (step+0) % 10 == 0:
				for k,(p,x) in enumerate(zip(lnp,pp)):
					tractor.setParams(x)

					#chi = tractor.getChiImage(0)
					modj = tractor.getModelImage(0)
					plt.clf()
					plt.imshow(modj[slc], **imsub)
					plt.gray()
					plt.title('Step %i, walker %i: lnp %f' % (step, k, p))
					ps.savefig()

					if k == 5:
						break

			alllnp.append(lnp.copy())
			allp.append(pp.copy())

		return dict(alllnp=alllnp, allp=allp)


	def stage209(self, tractor=None, band=None,
				 run=None, camcol=None, field=None,
				 ra=None, dec=None, roi=None, tinf=None,
				 subcat=None, sgroup=None, group=None, over=None,
				 orig_ie=None, mpatch=None, imc=None, ps=None,
				 old_sgroup=None, new_sgroup=None, sgi=None,
				 cat=None,
				 alllnp=None, allp=None,
				 **kwargs):

		alllnp = np.array(alllnp)
		allp = np.array(allp)
		print 'alllnp shape', alllnp.shape
		print 'allp shape', allp.shape
		# steps, walkers, params

		print 'sgi', sgi

		burn = 50

		plt.clf()
		plt.plot(alllnp, 'k.')
		plt.xlabel('step')
		plt.ylabel('lnp')
		ps.savefig()

		DL = LuminosityDistance()
		log = np.log10
		h70 = 0.7

		print 'Tractor: active params'
		for nm in tractor.getParamNames():
			print '  ', nm

		# for i,nm in enumerate(tractor.getParamNames()):
		# 	if not nm.startswith('catalog.source'):
		# 		continue
		# 	P = allp[burn:, :, i].ravel()
		# 	nm = nm.replace('catalog.','')
		# 	plt.clf()
		# 	plt.hist(P, 100)
		# 	plt.xlabel(nm)
		# 	ps.savefig()

		pnames = tractor.getParamNames()

		# Deja vu...
		fn = get_spectro_table(ra, dec, self.aco)
		print 'Looking for', fn
		T = fits_table(fn)

		svals = []

		for si in sgi:
			src = tractor.getCatalog()[si]
			print 'Source', src

			rd = src.getPosition()
			ra,dec = rd.ra, rd.dec
			rad = 1./3600.
			I,J,d = sm.match_radec(T.ra, T.dec, ra, dec, rad,
								   nearest=True)
			print len(I), 'matches on RA,Dec'
			Ti = T[I]
			print 'Matched spectro'
			Ti.about()
			Ti = Ti[0]

			samp = []
			for pp in allp[-10:]:
				samp.extend(pp)
			print len(samp), 'samples'

		   	# redshift
	   		z = Ti.z
			# HACK!! K-correction
			Kz = 0.
			#Kz = NYUK(z)
			# velocity dispersion - sigma [km/s]
			sigma = Ti.veldisp

			vals = []
			for p in samp:
				tractor.setParams(p)

				if type(src) is CompositeGalaxy:
					# pnm = 'catalog.source%i.shapeDev.re' % si
					# ire = pnames.index(pnm)
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
				vals.append((rdev, ab, mdev, z, Kz, sigma))

			Ti = tabledata()
			vals = np.array(vals)
			for j,col in enumerate(['rdev','ab','mdev','z','Kz', 'sigma']):
				Ti.set(col, vals[:,j])

			Ti.r0 = Ti.rdev * np.sqrt(Ti.ab)
			Ti.mu0 = (Ti.mdev + 2.5 * log(2. * np.pi * Ti.r0**2) - Ti.Kz
					  - 10.* log(1. + Ti.z))
			Ti.I0 = 10.**(Ti.mu0 / -2.5)
			Ti.DLz = DL(Ti.z)
			Ti.DMz = 5. * log(Ti.DLz * 1e6 / 10.)
			Ti.M = Ti.mdev - Ti.DMz - Ti.Kz
			Ti.DAz = Ti.DLz / ((1.+Ti.z)**2)
			Ti.R0 = arcsec2rad(Ti.r0) * Ti.DAz

			for c in ['rdev', 'r0', 'mdev', 'mu0', 'I0', 'M', 'R0']:
				plt.clf()
				plt.hist(Ti.get(c), 100)
				plt.xlabel(c)
				ps.savefig()

			MM = Ti.M
			SS = Ti.sigma
			RR = Ti.R0
			MU = Ti.mu0

			print 'SS', SS
			xx = log(SS) + 0.2*(MU - 19.61)
			plt.clf()
			plt.hist(xx, 100)
			plt.xlabel('FP x-axis')
			ps.savefig()

			yy = log(RR * 1e3 / h70)
			plt.clf()
			plt.hist(yy, 100)
			plt.xlabel('FP y-axis')
			ps.savefig()

			def plot_pts(x, y, xr, yr):
				pa = dict(color='k', marker='.', linestyle='None')
				plt.plot(x, y, **pa)
				#plt.xlim(xr)
				#plt.ylim(yr)
			la = dict(color='k', alpha=0.25)
			ta = dict(color='r', lw=2,)
			fa = dict(color='b', lw=2,)
			ga = dict(color='g', lw=2,)

			#print 'MM', MM.shape
			print 'SS', SS.shape
			print 'RR', RR.shape
			print 'MU', MU.shape

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
			#plt.ylim(yl,yh)
			#plt.xlim(xl,xh)
			plt.title('FP')
			ps.savefig()


			svals.append(Ti)

		#T = tabledata()
		#T.
		#stepslice = slice(-10,None)


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

class SubSky(ParamsWrapper):
	def __init__(self, real, slc):
		super(SubSky, self).__init__(real)
		self.real = real
		self.slc = slc
	def getParamDerivatives(self, *args):
		derivs = self.real.getParamDerivatives(*args)
		for d in derivs:
			if d in [False, None]:
				continue
			# Shift Patch
			assert(isinstance(d, Patch))
			sly,slx = self.slc
			d.x0 += self.slx.start
			d.y0 += self.sly.start
		return derivs

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
	parser.add_option('--threads', dest='threads', type=int,
					  help='Run multithreaded with this # of threads')

	#parser.add_option('-na', dest='N', type=int,
	#				  default=20, help='Run this # of Abells')
	#parser.add_option('-a', type=int, dest='abells', action='append',
	#				  default=[], help='Run only these Abell clusters')
	#parser.add_option('-r', type=int, dest='runs', action='append',
	#				  default=

	opt,args = parser.parse_args()
	if opt.verbose == 0:
		lvl = logging.INFO
	else:
		lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	find()
	#fp()
	#get_dm_table()
	#join()
	runlots(stage=opt.stage, N=opt.N, force=opt.force, threads=opt.threads)
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

