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

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.plotutils import ArcsinhNormalize, plothist
from astrometry.util.starutil_numpy import *
import astrometry.libkd.spherematch as sm
from astrometry.sdss import *

from tractor.utils import *
from tractor import sdss as st
from tractor import *
from tractor.sdss_galaxy import *

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

def compile_nyu_vagc():
	# Compile NYU-VAGC values
	print 'Compiling NYU-VAGC catalog...'
	nyudir = 'nyu-vagc-2'
	Tcat = fits_table(os.path.join(nyudir, 'object_catalog.fits'))
	#Tcat.about()
	I = np.flatnonzero((Tcat.sdss_imaging_position != -1) * (Tcat.sdss_spectro_position != -1))
	print 'Found', len(I), 'rows with spectra and imaging'
	#Tcat.cut(I)
	del Tcat

	print 'Reading imaging...'
	Tim = fits_table(os.path.join(nyudir, 'object_sdss_imaging.fits'),
					 rows=I, columns=['run','camcol','field','id'])
	print Tim
	Tim.about()
	print 'Reading spec...'
	Tspec = fits_table(os.path.join(nyudir, 'object_sdss_spectro.fits'),
					   rows=I, columns=['vdisp','z'])
	print Tspec
	Tspec.about()
	print 'Reading kcorrect...'
	Tk = fits_table(os.path.join(nyudir, 'kcorrect', 'kcorrect.nearest.model.z0.00.fits'),
					rows=I, columns=['absmag','kcorrect','z'])
	print Tk
	Tk.about()
	#Tim.cut(I)
	#Tspec.cut(I)
	#Tk.cut(I)

	Tnyu = tabledata()
	Tnyu.sigma = Tspec.vdisp
	#Tnyu.z = Tspec.z
	Tnyu.z = Tk.z
	# abs mag
	Tnyu.M = Tk.absmag[:,bandnum]
	# K-correction
	Tnyu.Kz = Tk.kcorrect[:,bandnum]
	
	Tnyu.rdev = np.zeros(len(Tnyu))
	Tnyu.ab   = np.zeros(len(Tnyu))
	
	for run,camcol in np.unique(zip(Tim.run, Tim.camcol)):
		print 'Run', run, 'camcol', camcol
		J = np.flatnonzero((Tim.run == run) * (Tim.camcol == camcol))
		Tc = fits_table(os.path.join(nyudir, 'sdss', 'parameters',
									 'calibObj-%06i-%i.fits' % (run, camcol)),
									 columns=['r_dev','ab_dev','field','id'])
		for j in J:
			K = np.flatnonzero((Tc.field == Tim.field[j]) * (Tc.id == Tim.id[j]))
			assert(len(K) == 1)
			K = K[0]
			Tnyu.rdev[j] = Tc.r_dev [K, bandnum]
			Tnyu.ab[j]   = Tc.ab_dev[K, bandnum]

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
			Kz = 0.
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
		Ti.r0 = Ti.rdev * Ti.ab

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

	fn = 'nyu-vagc.fits'
	if not os.path.exists(fn):
		Tnyu = compile_nyu_vagc()
		Tnyu.writeto('nyu-vagc.fits')
	else:
		Tnyu = fits_table(fn, lower=False)
		
	# -> arcsec
	Tnyu.rdev *= 0.396
	Tnyu.r0 = Tnyu.rdev * Tnyu.ab
	Tnyu.DLz = DL(Tnyu.z)
	Tnyu.DMz = 5. * log(Tnyu.DLz * 1e6 / 10.)
	Tnyu.DAz = Tnyu.DLz / ((1.+Tnyu.z)**2)
	Tnyu.R0 = arcsec2rad(Tnyu.r0) * Tnyu.DAz
	# hack
	Tnyu.mdev = Tnyu.M + Tnyu.DMz + Tnyu.Kz
	Tnyu.mu0 = (Tnyu.mdev + 2.5 * log(2. * np.pi * Tnyu.r0**2)
				- Tnyu.Kz - 10.* log(1. + Tnyu.z))
	
	# FP
	for Ti,nm in zip(TT + [Tnyu],
					 ['SDSS', 'SDSS-forced', 'Tractor'] + ['NYU-VAGC']):
		MM = Ti.M
		SS = Ti.sigma
		RR = Ti.R0
		MU = Ti.mu0
		
		h70 = 0.7

		def plot_nyu(x, y, xr, yr):
			plothist(x, y, range=((xr,yr)), docolorbar=False,
					 dohot=False)
			plt.gray()
		
		def plot_sdss(x, y, xr, yr):
			pa = dict(color='k', marker='.', linestyle='None')
			plt.plot(x, y, **pa)
			plt.xlim(xr)
			plt.ylim(yr)

		if nm.startswith('NYU'):
			plot_pts = plot_nyu
			ta = dict(color='r', lw=2,)
		else:
			plot_pts = plot_sdss
			ta = dict(color='k', lw=2, alpha=0.5)
		
		plt.clf()
		yl,yh = [-24,-17]
		xl,xh = [1.2, 2.6]
		plot_pts(log(SS), MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 2.) * -3.95 + -19.5, **ta)
		plt.xlabel('log(sigma) [km/s]')
		plt.ylabel('M - 5 log(h) [mag]')
		plt.ylim(yh,yl)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 2 fig 4: %s' % nm)
		ps.savefig()

		plt.clf()
		xl,xh = [-1., 1.5]
		plot_pts(log(RR * 1e3 / h70), MM - 5.*log(h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 0.) * -1.59 + -19.5, **ta)
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
		plt.plot(X, (X - 4.) * -0.88 + -19.5, **ta)
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
		plt.plot(X, (X - 4.) * 1.34 + -19.5, **ta)
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

		plt.clf()
		xl,xh = [-0.5, 3.5]
		yl,yh = [-1, 1.5]
		plot_pts(log(SS) + 0.2*(MU - 19.61)*log(SS) + 0.2*(MU - 19.24),
				 log(RR * 1e3 / h70), (xl,xh), (yl,yh))
		X = np.array([xl,xh])
		# eyeballed Bernardi relation for i-band
		plt.plot(X, (X - 2.) * (1.52) + 0.2, **ta)
		plt.xlabel('log(sigma) + 0.2 (mu_0 - 19.61) log(sigma) + 0.2 (mu_0 - 19.24)')
		plt.ylabel('log(R_0) [kpc / h]')
		plt.ylim(yl,yh)
		plt.xlim(xl,xh)
		plt.title('Bernardi paper 3 fig 1: %s' % nm)
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
					print 'Clamping flux', f, 'up to zero'
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

	for anum in [2151]:
		I = np.flatnonzero(T.aco == anum)
		print 'Abell', anum, ': found', len(I)
		Ti = T[I[0]]
		Ti.about()

		rcf = radec_to_sdss_rcf(Ti.ra, Ti.dec, contains=True,
								tablefn='dr9fields.fits')
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

	
if __name__ == '__main__':
	#find()
	fp()
	#get_dm_table()
	#join()
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

