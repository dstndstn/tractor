	# imstd = estimate_noise(image)
	# print 'estimated std:', imstd
	# # Turns out that a ~= 1
	# # From SExtractor manual: MAP_WEIGHT propto 1/var;
	# # scale to variance units by calibrating to the estimated image variance.
	# a = np.median(weight) * imstd**2
	# print 'a', a
	# #invvar = weight / a

def estimate_noise(im, S=5):
	# From "dsigma.c" by Mike Blanton via Astrometry.net
	(H,W) = im.shape
	xi = np.arange(0, W, S)
	yi = np.arange(0, H, S)
	I,J = np.meshgrid(xi, yi)
	D = np.abs((im[J[:-1, :-1], I[:-1, :-1]] - im[J[1:, 1:], I[1:, 1:]]).ravel())
	D.sort()
	N = len(D)
	print 'estimating noise at', N, 'samples'
	nsig = 0.7
	s = 0.
	while s == 0.:
		k = int(math.floor(N * math.erf(nsig / math.sqrt(2.))))
		if k >= N:
			raise 'Failed to estmate noise in image'
		s = D[k] / (nsig * math.sqrt(2.))
		nsig += 0.1
	return s


def getdata():
	fn = 'cs82data/W4p1m1_i.V2.7A.swarp.cut.vig15_deV_ord2_size25.fits'
	T = fits_table(fn, hdunum=2)
	print 'Read', len(T), 'rows from', fn
	#T.about()

	r0,r1,d0,d1 = T.alpha_sky.min(), T.alpha_sky.max(), T.delta_sky.min(), T.delta_sky.max()
	print 'Range', r0,r1,d0,d1
	plt.clf()
	plt.plot(T.alpha_sky, T.delta_sky, 'r.')
	plt.xlabel('alpha_sky')
	plt.ylabel('delta_sky')

	rcf = radec_to_sdss_rcf((r0+r1)/2., (d0+d1)/2., radius=60., tablefn='s82fields.fits')
	print 'SDSS fields nearby:', len(rcf)

	rr = [ra  for r,c,f,ra,dec in rcf]
	dd = [dec for r,c,f,ra,dec in rcf]
	plt.plot(rr, dd, 'k.')
	plt.savefig('rd.png')

	RA,DEC = 334.4, 0.3

	rcf = radec_to_sdss_rcf(RA,DEC, radius=10., tablefn='s82fields.fits',
							 contains=True)
	print 'SDSS fields nearby:', len(rcf)
	rcf = [(r,c,f,ra,dec) for r,c,f,ra,dec in rcf if r != 206]
	print 'Filtering out run 206:', len(rcf)

	sdss = DR7()
	sdss.setBasedir('cs82data')
	for r,c,f,ra,dec in rcf:
		for band in 'ugriz':
			print 'Retrieving', r,c,f,band
			st.get_tractor_image(r, c, f, band, psf='dg', useMags=True, sdssobj=sdss)

	plt.clf()
	plt.plot(T.alpha_sky, T.delta_sky, 'r.')
	plt.xlabel('alpha_sky')
	plt.ylabel('delta_sky')
	rr = [ra  for r,c,f,ra,dec in rcf]
	dd = [dec for r,c,f,ra,dec in rcf]
	plt.plot(rr, dd, 'k.')
	#RR,DD = [],[]
	keepWCS = []
	for j,fn in enumerate(glob('cs82data/86*p.fits')):
	# for j,fn in enumerate(glob('cs82data/86*p-21.fits')):
		for i in range(36):
		# for i in [-1]:
			wcs = anwcs(fn, i+1)
			#WCS.append(wcs)
			#print 'image size', wcs.imagew, wcs.imageh
			rr,dd = [],[]
			W,H = wcs.imagew, wcs.imageh
			for x,y in [(1,1),(W,1),(W,H),(1,H),(1,1)]:
				r,d = wcs.pixelxy2radec(x,y)
				rr.append(r)
				dd.append(d)
			rc,dc = wcs.pixelxy2radec(W/2,H/2)
			tc = '%i:%i' % (j,i)
			if wcs.is_inside(RA,DEC):
				keepWCS.append(wcs)
				print 'Keeping', tc
			#RR.append(rr)
			#DD.append(dd)
			plt.plot(rr,dd, 'b-')
			plt.text(rc,dc,tc, color='b')
	#plt.plot(np.array(RR).T, np.array(DD).T, 'b-')
	plt.savefig('rd2.png')

def read_cf_catalogs(RA, DEC, sz):
	fn = 'cs82data/v1/W4p1m1_i.V2.7A.swarp.cut.vig15_deV_ord2_size25.fits'
	T = fits_table(fn, hdunum=2)
	print 'Read', len(T), 'rows from', fn
	T.ra  = T.alpha_sky
	T.dec = T.delta_sky

	fn = 'cs82data/v1/W4p1m1_i.V2.7A.swarp.cut.vig15_exp_ord2_size25.fit'
	T2 = fits_table(fn, hdunum=2)
	print 'Read', len(T2), 'rows from', fn
	T2.ra  = T2.alpha_sky
	T2.dec = T2.delta_sky

	# approx...
	S = sz / 3600.
	ra0 ,ra1  = RA-S/2.,  RA+S/2.
	dec0,dec1 = DEC-S/2., DEC+S/2.

	if False:
		T = T[np.logical_or(T.mag_model < 50, T.mag_psf < 50)]
		Tstar = T[np.logical_and(T.chi2_psf < T.chi2_model, T.mag_psf < 50)]
		Tgal = T[np.logical_and(T.chi2_model < T.chi2_psf, T.mag_model < 50)]
		# 'mag_psf', 'chi2_psf',
		for i,c in enumerate(['mag_model', 'chi2_model', 
							  'spheroid_reff_world', 'spheroid_aspect_world',
							  'spheroid_theta_world']):
			plt.clf()
			plt.hist(Tgal.get(c), 100)
			plt.xlabel(c)
			mysavefig('hist%i.png' % i)
		sys.exit(0)

		plt.clf()
		plt.semilogx(T.chi2_psf, T.chi2_psf - T.chi2_model, 'r.')
		plt.ylim(-100, 100)
		plt.xlabel('chi2_psf')
		plt.ylabel('chi2_psf - chi2_model')
		mysavefig('chi.png')

	for c in ['disk_scale_world', 'disk_aspect_world', 'disk_theta_world']:
		T.set(c, T2.get(c))
	T.ra_disk  = T2.alphamodel_sky
	T.dec_disk = T2.deltamodel_sky
	T.mag_disk = T2.mag_model
	T.chi2_disk = T2.chi2_model
	T.ra_sph  = T2.alphamodel_sky
	T.dec_sph = T2.deltamodel_sky
	T.mag_sph = T.mag_model
	T.chi2_sph = T.chi2_model

	T = T[(T.ra > ra0) * (T.ra < ra1) * (T.dec > dec0) * (T.dec < dec1)]
	print 'Cut to', len(T), 'objects nearby.'

	T.chi2_gal = np.minimum(T.chi2_disk, T.chi2_sph)
	T.mag_gal = np.where(T.chi2_disk < T.chi2_sph, T.mag_disk, T.mag_sph)

	Tstar = T[np.logical_and(T.chi2_psf < T.chi2_gal, T.mag_psf < 50)]
	Tgal = T[np.logical_and(T.chi2_gal < T.chi2_psf, T.mag_gal < 50)]
	print len(Tstar), 'stars'
	print len(Tgal), 'galaxies'
	Tdisk = Tgal[Tgal.chi2_disk < Tgal.chi2_sph]
	Tsph  = Tgal[Tgal.chi2_sph  <= Tgal.chi2_disk]
	print len(Tdisk), 'disk'
	print len(Tsph), 'spheroid'

	return Tstar, Tdisk, Tsph


def get_cf_sources(Tstar, Tdisk, Tsph, magcut=100, mags=['u','g','r','i','z']):
	srcs = []
	for t in Tdisk:
		# xmodel_world == alphamodel_sky
		if t.mag_disk > magcut:
			#print 'Skipping source with mag=', t.mag_disk
			continue
		#origwcs = Tan(cffns[0],0)
		#x,y = origwcs.radec2pixelxy(t.alphamodel_sky, t.deltamodel_sky)
		#print 'WCS x,y', x,y
		#print '    x,y', t.xmodel_image, t.ymodel_image
		#print '    del', t.xmodel_image - x, t.ymodel_image - y
		#print '    x,y', t.x_image, t.y_image
		m = Mags(order=mags, **dict([(k, t.mag_disk) for k in mags]))
		src = DevGalaxy(RaDecPos(t.ra_disk, t.dec_disk), m,
						GalaxyShape(t.disk_scale_world * 3600., t.disk_aspect_world,
									t.disk_theta_world + 90.))
		#print 'Adding source', src
		srcs.append(src)
	for t in Tsph:
		if t.mag_sph > magcut:
			#print 'Skipping source with mag=', t.mag_sph
			continue
		m = Mags(order=mags, **dict([(k, t.mag_sph) for k in mags]))
		src = ExpGalaxy(RaDecPos(t.ra_sph, t.dec_sph), m,
						GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
									t.spheroid_theta_world + 90.))
		#print 'Adding source', src
		srcs.append(src)
	assert(len(Tstar) == 0)
	return srcs



def stage01TEST(tractor=None, mp=None, **kwargs):
	allsources = tractor.getCatalog()
	allimages = tractor.getImages()

	maglim = 24.

	brightcat,Ibright = cut_bright(allsources, magcut=maglim, mag='i2')
	tractor.setCatalog(brightcat)
	print ' Cut to:', tractor
	print len(tractor.getParams()), 'params'

	# TRY drilling down to a much smaller region to check out galaxy scale parameters
	tim = allimages[0]
	im = tim.getImage()
	x0,x1,y0,y1 = (200, 400, 800, 1000)
	#x0,x1,y0,y1 = (250, 350, 900, 1000)
	subim = im[y0:y1, x0:x1].copy()
	suberr = tim.getInvError()[y0:y1, x0:x1].copy()
	subwcs = tim.getWcs().copy()
	print 'x0,y0', subwcs.x0, subwcs.y0
	subwcs.setX0Y0(subwcs.x0 + x0, subwcs.y0 + y0)
	print 'subwcs:', subwcs
	pc = tim.getPhotoCal().copy()
	psf = tim.getPsf().copy()
	sky = tim.getSky().copy()
	subtim = Image(data=subim, invvar=suberr**2, wcs=subwcs,
				   photocal=pc, psf=psf, sky=sky, name=tim.name)
	subtim.zr = tim.zr

	tractor.setImages(Images(subtim))

	# TRY cutting to sources in the RA,Dec box.
	#im = allimages[0]
	im = subtim

	W,H = im.getWidth(), im.getHeight()
	wcs = im.getWcs()
	r,d = [],[]
	for x,y in zip([1,W,W,1,1], [1,1,H,H,1]):
		rd = wcs.pixelToPosition(x, y)
		r.append(rd.ra)
		d.append(rd.dec)
		print 'pix', x,y, 'radec', rd
	ramin = np.min(r)
	ramax = np.max(r)
	decmin = np.min(d)
	decmax = np.max(d)
	print 'RA,Dec range', ramin, ramax, decmin, decmax
	cutrd = Catalog()
	Icut = []
	for i in xrange(len(brightcat)):
		src = brightcat[i]
		pos = src.getPosition()
		if pos.ra > ramin and pos.ra < ramax and pos.dec > decmin and pos.dec < decmax:
			cutrd.append(src)
			Icut.append(Ibright[i])
		else:
			pass

	tractor.setCatalog(cutrd)
	print ' Cut on RA,Dec box to:', tractor
	print len(tractor.getParams()), 'params'

	mags = ['i2']

	cats = [cutrd]
	for i,(fn,gal) in enumerate([('cs82data/W4p1m1_i.V2.7A.swarp.cut.deV.fit', 'deV'),
								 ('cs82data/W4p1m1_i.V2.7A.swarp.cut.exp.fit', 'exp'),]):
		T = fits_table(fn, hdunum=2)
		T.ra  = T.alpha_j2000
		T.dec = T.delta_j2000
		print 'Read', len(T), gal, 'galaxies'
		T.cut((T.ra > ramin) * (T.ra < ramax) * (T.dec > decmin) * (T.dec < decmax))
		print 'Cut to', len(T), 'in RA,Dec box'

		srcs = Catalog()
		for t in T:
			if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
				themag = t.mag_psf
				m = Mags(order=mags, **dict([(k, themag) for k in mags]))
				srcs.append(PointSource(RaDecPos(t.alpha_j2000, t.delta_j2000), m))
				continue

			
			# deV: spheroid
			# exp: disk

			if gal == 'deV' and t.mag_spheroid > maglim:
				continue
			if gal == 'exp' and t.mag_disk > maglim:
				continue

			pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)
			if gal == 'deV':
				shape_dev = GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
										t.spheroid_theta_world + 90.)
				themag = t.mag_spheroid
				m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))
				srcs.append(DevGalaxy(pos, m_dev, shape_dev))
			else:
				shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600., t.disk_aspect_world,
										t.disk_theta_world + 90.)
				themag = t.mag_disk
				m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))
				srcs.append(ExpGalaxy(pos, m_exp, shape_exp))

		cats.append(srcs)


	plotims = [0,]
	plotsa = dict(imis=plotims, mp=mp)
	plots(tractor, ['data', 'dataann'], 0,
		  ibest=0, pp=np.array([tractor.getParams()]), alllnp=[0.], **plotsa)

	tractor.freezeParam('images')

	for i,(srcs,gal) in enumerate(zip(cats, ['deV+exp', 'deV', 'exp'])):
		tractor.setCatalog(srcs)
		tractor.catalog.thawParamsRecursive('*')

		plotims = [0,]
		plotsa = dict(imis=plotims, mp=mp, tsuf=' '+gal)
		plots(tractor, ['modbest', 'chibest', 'modnoise', 'chinoise'], 2*i + 0,
			  ibest=0, pp=np.array([tractor.getParams()]), alllnp=[0.], **plotsa)

		for j in xrange(5):
			step, alllnp = optsourcestogether(tractor, 0, doplots=False)
			step, alllnp2 = optsourcesseparate(tractor, step, 10, plotsa, doplots=False)
			
		plots(tractor, ['modbest', 'chibest', 'modnoise', 'chinoise',], 2*i + 1,
			  ibest=0, pp=np.array([tractor.getParams()]), alllnp=[0.], **plotsa)





def stage01OLD(tractor=None, mp=None, step0=0, thaw_wcs=['crval1','crval2'],
			#thaw_sdss=['a','d'],
			thaw_sdss=[],
			RA=None, DEC=None, sz=None,
			**kwargs):
	print 'tractor', tractor
	tractor.mp = mp

	S=500
	filtermap = {'i.MP9701': 'i2'}
	coim = get_cfht_coadd_image(RA, DEC, S, filtermap=filtermap)

	newims = Images(*([coim] + [im for im in tractor.getImages() if im.name.startswith('SDSS')]))
	tractor.setImages(newims)
	print 'tractor:', tractor

	# We only use the 'inverr' element, so don't also store 'invvar'.
	for im in tractor.getImages():
		if hasattr(im, 'invvar'):
			del im.invvar

	# Make RA,Dec overview plot
	plt.clf()
	#plt.plot(rco, dco, 'k-', lw=1, alpha=0.8)
	#plt.plot(rco[0], dco[0], 'ko')
	if False:
		cffns = glob('cs82data/86*p-21-cr.fits')
		for fn in cffns:
			wcs = Tan(fn, 0)
			#W,H = wcs.imagew, wcs.imageh
			P = pyfits.open(fn)
			image = P[1].data
			(H,W) = image.shape
			x,y = np.array([1,W,W,1,1]), np.array([1,1,H,H,1])
			print 'x,y', x,y
			r,d = wcs.pixelxy2radec(x, y)
			plt.plot(r, d, 'k-', lw=1, alpha=0.8)

	for im in tractor.getImages():
		wcs = im.getWcs()
		W,H = im.getWidth(), im.getHeight()
		r,d = [],[]
		for x,y in zip([1,W,W,1,1], [1,1,H,H,1]):
			rd = wcs.pixelToPosition(x, y)
			r.append(rd.ra)
			d.append(rd.dec)
		if im.name.startswith('SDSS'):
			band = im.getPhotoCal().bandname
			cmap = dict(u='b', g='g', r='r', i='m', z=(0.6,0.,1.))
			cc = cmap[band]
			#aa = 0.1
			aa = 0.06
		else:
			print 'CFHT WCS x0,y0', wcs.x0, wcs.y0
			print 'image W,H', W,H
			cc = 'k'
			aa = 0.5
		plt.plot(r, d, '-', color=cc, alpha=aa, lw=1)
		#plt.gca().add_artist(matplotlib.patches.Polygon(np.vstack((r,d)).T, ec='none', fc=cc,
		#alpha=aa/4.))

	ax = plt.axis()

	r,d = [],[]
	for src in tractor.getCatalog():
		pos = src.getPosition()
		r.append(pos.ra)
		d.append(pos.dec)
	plt.plot(r, d, 'k,', alpha=0.3)

	plt.axis(ax)
	plt.xlim(ax[1],ax[0])
	plt.axis('equal')
	plt.xlabel('RA (deg)')
	plt.ylabel('Dec (deg)')
	plt.savefig('outlines.png')

	if True:
		from astrometry.libkd.spherematch import match_radec
		r,d = np.array(r),np.array(d)
		for dr in [7,8]:
			T = fits_table('cs82data/cas-primary-DR%i.fits' % dr)
			I,J,nil = match_radec(r, d, T.ra, T.dec, 1./3600.)

			plt.clf()
			H,xe,ye = np.histogram2d((r[I]-T.ra[J])*3600., (d[I]-T.dec[J])*3600., bins=100)
			plt.imshow(H.T, extent=(min(xe), max(xe), min(ye), max(ye)),
					   aspect='auto', interpolation='nearest', origin='lower')
			plt.hot()
			plt.title('CFHT - SDSS DR%i source positions' % dr)
			plt.xlabel('dRA (arcsec)')
			plt.ylabel('dDec (arcsec)')
			plt.savefig('dradec-%i.png' % dr)

	return dict(tractor=tractor)


def stage02(tractor=None, mp=None, Ibright=[], allp=[], params0=None, **kwargs):

	print 'Tractor sources:', len(tractor.getCatalog())
	print 'Ibright:', Ibright
	print 'Ibright len', len(Ibright)
	#print 'allp:', allp
	#print 'params0:', params0

	bright = Catalog()
	for i in Ibright:
		bright.append(tractor.getCatalog()[i])
	tractor.setCatalog(bright)

	(imi, band, i2fit) = allp[0]

	print 'imi', imi
	print 'band', band
	print 'i2fit', i2fit
	print len(i2fit)
	
	assert(band == 'i2')
	assert(len(i2fit) == tractor.getCatalog().numberOfParams())

	# ugriz + i2
	assert(len(params0) == 6 * len(i2fit))
	
	i2cat = params0[0::6]

	cat = tractor.getCatalog()
	cat.freezeParamsRecursive('*')

	for src in cat:
		src.getModelPatch(tractor.getImage(0))


	#print 'Freeze state:'
	#print_frozen(cat)

	#print 'All frozen:'
	#for nm in cat.getParamNames():
	#	print '  ', nm
	cat.thawPathsTo('i2')

	#print 'Thawed:'
	#for nm in cat.getParamNames():
	#	print '  ', nm

	cat.setParams(i2cat)
	catsum = np.array([src.getBrightness().i2 for src in cat])

	cat.setParams(i2fit)
	fitsum = np.array([src.getBrightness().i2 for src in cat])

	star = np.array([isinstance(src, PointSource) for src in cat])

	plt.clf()
	p1 = plt.plot((fitsum - catsum)[star], catsum[star], 'b.')
	p2 = plt.plot((fitsum - catsum)[np.logical_not(star)], catsum[np.logical_not(star)], 'g.')
	plt.xlabel('Fit i2 - Catalog i2 (mag)')
	plt.ylabel('Catalog i2 (mag)')
	plt.ylim(21, 16)
	plt.xlim(-1, 1)
	plt.legend((p1[0],p2[0]), ('Stars', 'Galaxies'))
	plt.savefig('i2.png')

	return dict(tractor=tractor)

	# set dummy "invvar"s
	for im in tractor.getImages():
		im.invvar = None

	print 'Plots...'
	plots(tractor, ['data', 'modbest', 'chibest'], 0, ibest=0, pp=np.array([tractor.getParams()]),
		  alllnp=[0.], imis=[0,1,2,3,4,5])

	return dict(tractor=tractor)


def stage03OLD(tractor=None, mp=None, **kwargs):
	tractor.freezeParam('images')
	tractor.catalog.thawParamsRecursive('*')

	params0 = np.array(tractor.getParams()).copy()
	allsources = tractor.getCatalog()
	#brightcat,Ibright = cut_bright(allsources, magcut=23)
	#brightcat,Ibright = cut_bright(allsources, magcut=24)
	brightcat,Ibright = cut_bright(allsources, magcut=23.8)
	#brightcat,Ibright = cut_bright(allsources, magcut=27)
	tractor.setCatalog(brightcat)
	bparams0 = np.array(tractor.getParams()).copy()
	allimages = tractor.getImages()
	tractor.setImages(Images(allimages[0]))
	print ' Cut to:', tractor
	print len(tractor.getParams()), 'params'

	# TRY drilling down to a much smaller region to check out galaxy scale parameters
	tim = allimages[0]
	im = tim.getImage()
	x0,x1,y0,y1 = (200, 400, 800, 1000)
	#x0,x1,y0,y1 = (250, 350, 900, 1000)
	subim = im[y0:y1, x0:x1].copy()
	suberr = tim.getInvError()[y0:y1, x0:x1].copy()
	subwcs = tim.getWcs().copy()
	print 'x0,y0', subwcs.x0, subwcs.y0
	subwcs.setX0Y0(subwcs.x0 + x0, subwcs.y0 + y0)
	print 'subwcs:', subwcs
	pc = tim.getPhotoCal().copy()
	psf = tim.getPsf().copy()
	sky = tim.getSky().copy()
	subtim = Image(data=subim, invvar=suberr**2, wcs=subwcs,
				   photocal=pc, psf=psf, sky=sky, name=tim.name)
	subtim.zr = tim.zr

	tractor.setImages(Images(subtim))
	
	# TRY cutting to sources in the RA,Dec box.
	#im = allimages[0]
	im = subtim

	W,H = im.getWidth(), im.getHeight()
	wcs = im.getWcs()
	r,d = [],[]
	for x,y in zip([1,W,W,1,1], [1,1,H,H,1]):
		rd = wcs.pixelToPosition(x, y)
		r.append(rd.ra)
		d.append(rd.dec)
		print 'pix', x,y, 'radec', rd
	ramin = np.min(r)
	ramax = np.max(r)
	decmin = np.min(d)
	decmax = np.max(d)
	print 'RA,Dec range', ramin, ramax, decmin, decmax
	cutrd = Catalog()
	Icut = []
	for i in xrange(len(brightcat)):
		src = brightcat[i]
		#print 'Source:', src
		pos = src.getPosition()
		#print '  pos', pos
		if pos.ra > ramin and pos.ra < ramax and pos.dec > decmin and pos.dec < decmax:
			#print '  -> Keep'
			cutrd.append(src)
			Icut.append(Ibright[i])
		else:
			#print '  -> cut.'
			pass
	print 'Keeping sources:'
	for src in cutrd:
		print '  ', src
		x,y = wcs.positionToPixel(src.getPosition())
		print '  --> (%.1f, %.1f)' % (x,y)

	tractor.setCatalog(cutrd)
	print ' Cut on RA,Dec box to:', tractor
	print len(tractor.getParams()), 'params'

	cparams0 = np.array(tractor.getParams()).copy()
	p0 = cparams0

	tractor.catalog.thawParamsRecursive('*')
	bparams1 = np.array(tractor.getParams()).copy()
	tractor.setCatalog(allsources)
	tractor.setImages(allimages)
	params1 = np.array(tractor.getParams()).copy()
	#print 'dparams for bright sources:', bparams1 - bparams0
	#print 'dparams for all sources:', params1 - params0
	return dict(tractor=tractor, alllnp3=alllnp, Ibright3=Ibright)





def stage01_OLD(tractor=None, mp=None, step0=0, thaw_wcs=['crval1','crval2'],
			#thaw_sdss=['a','d'],
			thaw_sdss=[],
			RA=None, DEC=None, sz=None,
			**kwargs):
	# For the initial WCS alignment, cut to brightish sources...
	allsources = tractor.getCatalog()
	brightcat,nil = cut_bright(allsources)
	tractor.setCatalog(brightcat)
	print 'Cut to', len(brightcat), 'bright sources', brightcat.numberOfParams(), 'params'
	allims = tractor.getImages()
	fitims = []
	for im in allims:
		im.freezeParams('photocal', 'psf', 'sky')
		if hasattr(im.wcs, 'crval1'):
			# FitsWcs:
			im.wcs.freezeAllBut(*thaw_wcs)
		elif len(thaw_sdss):
			# SdssWcs: equivalent is 'a','d'
			im.wcs.freezeAllBut(*thaw_sdss)
		else:
			continue
		fitims.append(im)
	print len(tractor.getImages()), 'images', tractor.images.numberOfParams(), 'image params'
	tractor.freezeParam('catalog')

	lnp0 = tractor.getLogProb()
	print 'Lnprob', lnp0
	plots(tractor, ['modsum', 'chisum'], step0 + 0, **plotsa)

	#wcs0 = tractor.getParams()
	wcs0 = np.hstack([im.getParams() for im in fitims])
	print 'Orig WCS:', wcs0
	# We will tweak the WCS parameters one image at a time...
	tractor.images = Images()
	# Do the work:
	wcs1 = mp.map(tweak_wcs, [(tractor,im) for im in fitims], wrap=True)
	# Reset the images
	tractor.setImages(allims)
	# Save the new WCS params!
	for im,p in zip(fitims,wcs1):
		im.setParams(p)
	lnp1 = tractor.getLogProb()
	print 'Before lnprob', lnp0
	print 'After  lnprob', lnp1
	wcs1 = np.hstack(wcs1)
	print 'Orig WCS:', wcs0
	print 'Opt  WCS:', wcs1
	print 'dWCS (arcsec):', (wcs1 - wcs0) * 3600.
	plots(tractor, ['modsum', 'chisum'], step0 + 1, None, **plotsa)

	# Re-freeze WCS
	for im in tractor.getImages():
		im.wcs.thawAllParams()
		im.freezeParam('wcs')
	tractor.unfreezeParam('catalog')

	tractor.setCatalog(allsources)
	return dict(tractor=tractor)

# Also fit the WCS rotation matrix (CD) terms.
def stage02_OLD(tractor=None, mp=None, **kwargs):
	for i,im in enumerate(tractor.getImages()):
		#print 'Tractor image i:', im.name
		#print 'Params:', im.getParamNames()
		im.unfreezeAllParams()
		#print 'Params:', im.getParamNames()
	stage01(tractor=tractor, mp=mp, step0=2,
			thaw_wcs=['crval1', 'crval2', 'cd1_1', 'cd1_2', 'cd2_1', 'cd2_2'],
			#thaw_sdss=['a','d','b','c','e','f'],
			**kwargs)
	return dict(tractor=tractor)



# stage 3 replacement: "quick" fit of bright sources to one CFHT image
def stage03_OLD(tractor=None, mp=None, steps=None, **kwargs):
	print 'Tractor:', tractor
	tractor.mp = mp
	tractor.freezeParam('images')
	tractor.catalog.thawParamsRecursive('*')
	params0 = np.array(tractor.getParams()).copy()
	allsources = tractor.getCatalog()
	brightcat,Ibright = cut_bright(allsources, magcut=23)
	tractor.setCatalog(brightcat)
	bparams0 = np.array(tractor.getParams()).copy()
	allimages = tractor.getImages()
	tractor.setImages(Images(allimages[0]))
	print ' Cut to:', tractor

	plotims = [0,]
	plotsa = dict(imis=plotims, mp=mp)

	step,alllnp = optsourcestogether(tractor, 0)

	step, alllnp2 = optsourcesseparate(tractor, step, 10, plotsa)
	alllnp += alllnp2

	tractor.catalog.thawParamsRecursive('*')
	bparams1 = np.array(tractor.getParams()).copy()
	tractor.setCatalog(allsources)
	tractor.setImages(allimages)
	params1 = np.array(tractor.getParams()).copy()
	#print 'dparams for bright sources:', bparams1 - bparams0
	#print 'dparams for all sources:', params1 - params0
	return dict(tractor=tractor, alllnp3=alllnp, Ibright3=Ibright)


def stage04(tractor=None, mp=None, steps=None, alllnp3=None, Ibright3=None,
			**kwargs):
	print 'Tractor:', tractor
	tractor.mp = mp
	tractor.freezeParam('images')
	tractor.catalog.thawParamsRecursive('*')
	params0 = np.array(tractor.getParams()).copy()
	allsources = tractor.getCatalog()
	allimages = tractor.getImages()

	tractor.setImages(Images(allimages[0]))
	print ' Cut to:', tractor
	plotims = [0,]
	plotsa = dict(imis=plotims, mp=mp)

	step = 200-1
	step, alllnp = optsourcestogether(tractor, step)

	step, alllnp2 = optsourcesseparate(tractor, step, 10, plotsa)
	alllnp += alllnp2

	tractor.catalog.thawParamsRecursive('*')
	tractor.setImages(allimages)
	params1 = np.array(tractor.getParams()).copy()
	return dict(tractor=tractor, alllnp3=alllnp3, Ibright3=Ibright3,
				alllnp4=alllnp)

def stage05(tractor=None, mp=None, steps=None,
			**kwargs):
	print 'Tractor:', tractor
	tractor.mp = mp
	tractor.freezeParam('images')
	tractor.catalog.thawParamsRecursive('*')
	tractor.catalog.freezeParamsRecursive('g', 'r', 'i')

	allimages = tractor.getImages()
	tractor.setImages(Images(*[im for im in allimages if im.name.startswith('CFHT')]))
	print ' Cut to:', tractor
	plotims = [0,]
	plotsa = dict(imis=plotims, mp=mp)

	step = 5000
	step, alllnp = optsourcestogether(tractor, step)
	step, alllnp2 = optsourcesseparate(tractor, step, 10, plotsa)
	alllnp += alllnp2

	##
	tractor.setImages(allimages)

	tractor.catalog.thawParamsRecursive('*')
	return dict(tractor=tractor, alllnp5=alllnp)

def stage06(tractor=None, mp=None, steps=None,
			**kwargs):
	print 'Tractor:', tractor
	cache = createCache(maxsize=10000)
	print 'Using multiprocessing cache', cache
	tractor.cache = cache
	tractor.pickleCache = True

	# oops, forgot to reset the images in an earlier version of stage05...
	pfn = 'tractor%02i.pickle' % 4
	print 'Reading pickle', pfn
	R = unpickle_from_file(pfn)
	allimages = R['tractor'].getImages()
	del R

	tractor.mp = mp
	tractor.freezeParam('images')
	tractor.catalog.thawParamsRecursive('*')
	tractor.catalog.freezeParamsRecursive('pos', 'i2', 'shape')

	# tractor.catalog.source46.shape
	#                         .brightness.i2
	#                         .pos

	sdssimages = [im for im in allimages if im.name.startswith('SDSS')]
	#tractor.setImages(Images(*sdssimages))

	allsources = tractor.getCatalog()
	brightcat,Ibright = cut_bright(allsources, magcut=23, mag='i2')
	tractor.setCatalog(brightcat)

	plotims = [0,]
	plotsa = dict(imis=plotims, mp=mp)

	print 'Tractor:', tractor

	allp = []

	sbands = ['g','r','i']

	# Set all bands = i2, and save those params.
	for src in brightcat:
		for b in sbands:
			br = src.getBrightness()
			setattr(br, b, br.i2)
	# this is a no-op; we did a thawRecursive above.
	#tractor.catalog.thawParamsRecursive(*sbands)
	params0 = tractor.getParams()

	#frozen0 = tractor.getFreezeState()
	#allparams0 = tractor.getParamsEvenFrozenOnes()

	for imi,im in enumerate(sdssimages):
		print 'Fitting image', imi, 'of', len(sdssimages)
		tractor.setImages(Images(im))
		band = im.photocal.bandname
		print im
		print 'Band', band

		print 'Current param names:'
		for nm in tractor.getParamNames():
			print '  ', nm
		print len(tractor.getParamNames()), 'params, p0', len(params0)

		# Reset params; need to thaw first though!
		tractor.catalog.thawParamsRecursive(*sbands)
		tractor.setParams(params0)

		#tractor.catalog.setFreezeState(frozen0)
		#tractor.catalog.setParams(params0)

		tractor.catalog.freezeParamsRecursive(*sbands)
		tractor.catalog.thawParamsRecursive(band)

		print 'Tractor:', tractor
		print 'Active params:'
		for nm in tractor.getParamNames():
			print '  ', nm

		step = 7000 + imi*3
		plots(tractor, ['modbest', 'chibest'], step, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': '+im.name+' init', **plotsa)
		optsourcestogether(tractor, step, doplots=False)
		plots(tractor, ['modbest', 'chibest'], step+1, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': '+im.name+' joint', **plotsa)
		optsourcesseparate(tractor, step, doplots=False)
		#print 'Tractor params:', tractor.getParams()
		tractor.catalog.thawAllParams()
		#print 'Tractor params:', tractor.getParams()
		plots(tractor, ['modbest', 'chibest'], step+2, pp=np.array([tractor.getParams()]),
			  ibest=0, tsuf=': '+im.name+' indiv', **plotsa)

		p = tractor.getParams()
		print 'Saving params', p
		allp.append((band, p))

		print 'Cache stats'
		cache.printStats()

	tractor.setImages(allimages)
	tractor.setCatalog(allsources)

	return dict(tractor=tractor, allp=allp, Ibright=Ibright)

def stage07(tractor=None, allp=None, Ibright=None, mp=None, **kwargs):
	print 'Tractor:', tractor

	sbands = ['g','r','i']

	#rcf = get_tractor(RA, DEC, sz, [], just_rcf=True)
	tr1,nil = get_tractor(RA, DEC, sz, [], sdss_psf='dg', sdssbands=sbands)
	rcf = []
	for im in tr1.getImages():
		rcf.append(im.rcf + (None,None))
	print 'RCF', rcf
	print len(rcf)
	print 'Kept', len(rcf), 'of', len(tr1.getImages()), 'images'
	# W = fits_table('window_flist-dr8.fits')
	W = fits_table('window_flist-DR8-S82.fits')
	scores = []
	noscores = []
	rcfscore = {}
	for r,c,f,nil,nil in rcf:
		print 'RCF', r,c,f
		w = W[(W.run == r) * (W.camcol == c) * (W.field == f)]
		print w
		if len(w) == 0:
			print 'No entry'
			noscores.append((r,c,f))
			continue
		score = w.score[0]
		print 'score', score
		scores.append(score)
		rcfscore[(r,c,f)] = score
	print 'No scores:', noscores
	plt.clf()
	plt.hist(scores, 20)
	plt.savefig('scores.png')
	print len(scores), 'scores'
	scores = np.array(scores)
	print sum(scores > 0.5), '> 0.5'

	allsources = tractor.getCatalog()
	bright = Catalog()
	for i in Ibright:
		bright.append(allsources[i])
	tractor.setCatalog(bright)

	allimages = tractor.getImages()
	sdssimages = [im for im in allimages if im.name.startswith('SDSS')]
	tractor.setImages(Images(*sdssimages))

	# Add "rcf" and "score" fields to the SDSS images...
	# first build name->rcf map
	namercf = {}
	print len(tr1.getImages()), 'tr1 images'
	print len(tractor.getImages()), 'tractor images'
	for im in tr1.getImages():
		print im.name, '->', im.rcf
		namercf[im.name] = im.rcf
	for im in tractor.getImages():
		print im.name
		rcf = namercf[im.name]
		print '->', rcf
		im.rcf = rcf
		im.score = rcfscore.get(rcf, None)

	allmags = {}
	for band,p in allp:
		if not band in allmags:
			allmags[band] = [p]
		else:
			allmags[band].append(p)
	for band,vals in allmags.items():
		vals = np.array(vals)
		allmags[band] = vals
		nimg,nsrcs = vals.shape

	for i in range(nsrcs):
		print 'source', i
		print tractor.getCatalog()[i]
		plt.clf()
		cmap = dict(r='r', g='g', i='m')
		pp = {}
		for band,vals in allmags.items():
			# hist(histtype='step') breaks when no vals are within the range.
			X = vals[:,i]
			mn,mx = 15,25
			keep = ((X >= mn) * (X <= mx))
			if sum(keep) == 0:
				print 'skipping', band
				continue
			n,b,p = plt.hist(vals[:,i], 100, range=(mn,mx), color=cmap[band], histtype='step', alpha=0.7)
			pp[band] = p
		pp['i2'] = plt.axvline(bright[i].brightness.i2, color=(0.5,0,0.5), lw=2, alpha=0.5)
		order = ['g','r','i','i2']
		plt.legend([pp[k] for k in order if k in pp], [k for k in order if k in pp], 'upper right')
		fn = 'mags-%02i.png' % i
		plt.ylim(0, 75)
		plt.xlim(mn,mx)
		plt.savefig(fn)
		print 'Wrote', fn

	goodimages = []
	Iimages = []
	for i,im in enumerate(allimages):
		if im.name.startswith('SDSS') and im.score is not None and im.score >= 0.5:
			goodimages.append(im)
			Iimages.append(i)
	Iimages = np.array(Iimages)
	tractor.setImages(Images(*goodimages))

	cache = createCache(maxsize=10000)
	print 'Using multiprocessing cache', cache
	tractor.cache = cache
	tractor.pickleCache = True
	tractor.freezeParam('images')
	tractor.catalog.thawParamsRecursive('*')
	tractor.catalog.freezeParamsRecursive('pos', 'i2', 'shape')

	plotims = [0,]
	plotsa = dict(imis=plotims, mp=mp)

	print 'Tractor:', tractor

	# Set all bands = i2, and save those params.
	for src in bright:
		br = src.getBrightness()
		m = min(br.i2, 28)
		for b in sbands:
			setattr(br, b, m)
	tractor.catalog.thawParamsRecursive(*sbands)
	params0 = tractor.getParams()

	print 'Tractor:', tractor
	print 'Active params:'
	for nm in tractor.getParamNames():
		print '  ', nm

	step = 8000
	plots(tractor, ['modbest', 'chibest'], step, pp=np.array([tractor.getParams()]),
		  ibest=0, tsuf=': '+im.name+' init', **plotsa)
	optsourcestogether(tractor, step, doplots=False)
	plots(tractor, ['modbest', 'chibest'], step+1, pp=np.array([tractor.getParams()]),
		  ibest=0, tsuf=': '+im.name+' joint', **plotsa)
	optsourcesseparate(tractor, step, doplots=False)
	tractor.catalog.thawAllParams()
	plots(tractor, ['modbest', 'chibest'], step+2, pp=np.array([tractor.getParams()]),
		  ibest=0, tsuf=': '+im.name+' indiv', **plotsa)

	tractor.setImages(allimages)
	tractor.setCatalog(allsources)

	return dict(tractor=tractor, Ibright=Ibright, Iimages=Iimages)

def stage08(tractor=None, Ibright=None, Iimages=None, mp=None, **kwargs):
	print 'Tractor:', tractor
	sbands = ['g','r','i']

	# Grab individual-image fits from stage06
	pfn = 'tractor%02i.pickle' % 6
	print 'Reading pickle', pfn
	R = unpickle_from_file(pfn)
	allp = R['allp']
	del R

	allsources = tractor.getCatalog()
	bright = Catalog(*[allsources[i] for i in Ibright])

	allimages = tractor.getImages()
	# = 4 + all SDSS
	print 'All images:', len(allimages)

	# 78 = 26 * 3 good-scoring images
	# index is in 'allimages'.
	print 'Iimages:', len(Iimages)
	# 219 = 73 * 3 all SDSS images
	print 'allp:', len(allp)

	# HACK -- assume CFHT images are at the front
	Ncfht = len(allimages) - len(allp)

	allmags = dict([(b,[]) for b in sbands])
	goodmags = dict([(b,[]) for b in sbands])
	for i,(band,p) in enumerate(allp):
		allmags[band].append(p)
		if (Ncfht + i) in Iimages:
			goodmags[band].append(p)
	for band,vals in allmags.items():
		vals = np.array(vals)
		allmags[band] = vals
		nimg,nsrcs = vals.shape
		goodmags[band] = np.array(goodmags[band])
	print 'nsrcs', nsrcs

	for i in range(nsrcs):
		##### 
		continue
		print 'source', i
		#print tractor.getCatalog()[i]
		plt.clf()
		#cmap = dict(r='r', g='g', i='m')
		cmap = dict(r=(1,0,0), g=(0,0.7,0), i=(0.8,0,0.8))
		pp = {}
		for band,vals in allmags.items():
			# hist(histtype='step') breaks when no vals are within the range.
			X = vals[:,i]
			mn,mx = 15,25
			keep = ((X >= mn) * (X <= mx))
			if sum(keep) == 0:
				print 'skipping', band
				continue
			cc = cmap[band]
			s = 0.2
			cc = [c*s+(1.-s)*0.5 for c in cc]
			n,b,p = plt.hist(X, 100, range=(mn,mx), color=cc, histtype='step', alpha=0.5)#, ls='dashed')
			#pp[band] = p
		for band,vals in goodmags.items():
			# hist(histtype='step') breaks when no vals are within the range.
			X = vals[:,i]
			mn,mx = 15,25
			keep = ((X >= mn) * (X <= mx))
			if sum(keep) == 0:
				print 'skipping', band
				continue
			n,b,p = plt.hist(X, 100, range=(mn,mx), color=cmap[band], histtype='step', alpha=0.7)
			pp[band] = p
			plt.axvline(bright[i].brightness.getMag(band), color=cmap[band], lw=2, alpha=0.5)
			me,std = np.mean(X), np.std(X)
			xx = np.linspace(mn,mx, 500)
			yy = 1./(np.sqrt(2.*np.pi)*std) * np.exp(-0.5*(xx-me)**2/std**2)
			yy *= sum(n)*(b[1]-b[0])
			plt.plot(xx, yy, '-', color=cmap[band])

		pp['i2'] = plt.axvline(bright[i].brightness.i2, color=(0.5,0,0.5), lw=2, alpha=0.5)
		order = ['g','r','i','i2']
		plt.legend([pp[k] for k in order if k in pp], [k for k in order if k in pp], 'upper right')
		fn = 'mags-%02i.png' % i
		plt.ylim(0, 60)
		plt.xlim(mn,mx)
		plt.savefig(fn)
		print 'Wrote', fn


	goodimages = Images(*[allimages[i] for i in Iimages])
	tractor.setImages(goodimages)
	tractor.setCatalog(bright)
	print 'MCMC with tractor:', tractor

	# DEBUG
	#goodimages = Images(*[allimages[i] for i in Iimages[:4]])
	#tractor.setImages(goodimages)
	#tractor.setCatalog(Catalog(*bright[:5]))
	#print 'MCMC with tractor:', tractor

	# cache = createCache(maxsize=10000)
	# print 'Using multiprocessing cache', cache
	# tractor.cache = Cache(maxsize=100)
	# tractor.pickleCache = True
	# Ugh!
	# No cache:
	# MCMC took 307.5 wall, 3657.9 s worker CPU, pickled 64/64 objs, 30922.6/0.008 MB
	# With cache:
	# MCMC took 354.3 wall, 4568.5 s worker CPU, pickled 64/64 objs, 30922.6/0.008 MB
	# but:
	# Cache stats
	# Cache has 480882 items
	# Total of 0 cache hits and 613682 misses
	# so maybe something is bogus...

	p0 = np.array(tractor.getParams())
	print 'Tractor params:'
	for i,nm in enumerate(tractor.getParamNames()):
		print '  ', nm, '=', p0[i]

	ndim = len(p0)
	# DEBUG
	#nw = 100
	nw = 5
	sampler = emcee.EnsembleSampler(nw, ndim, tractor, pool = mp.pool,
									live_dangerously=True)

	psteps = np.zeros_like(p0) + 0.001
	pp = emcee.EnsembleSampler.sampleBall(p0, psteps, nw)
	# Put one walker at the nominal position.
	pp[0,:] = p0

	rstate = None
	lnp = None
	smags = []
	for step in range(1, 201):
		smags.append(pp.copy())
		print 'Run MCMC step', step
		kwargs = dict(storechain=False)
		t0 = Time()
		pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate, **kwargs)
		print 'Running acceptance fraction: emcee'
		print 'after', sampler.iterations, 'iterations'
		print 'mean', np.mean(sampler.acceptance_fraction)
		t_mcmc = (Time() - t0)
		print 'Best lnprob:', np.max(lnp)
		print 'dlnprobs:', ', '.join(['%.1f' % d for d in lnp - np.max(lnp)])
		print 'MCMC took', t_mcmc, 'sec'

		#print 'Cache stats'
		#cache.printStats()

		print 'SDSS galaxy cache:'
		print get_galaxy_cache()
		

		break

	tractor.setParams(p0)
	tractor.setCatalog(allsources)
	tractor.setImages(allimages)
	return dict(tractor=tractor, Ibright=Ibright, Iimages=Iimages,
				allp=allp, smags=smags, allmags=allmags, goodmags=goodmags)



def stage06old():
	p0 = np.array(tractor.getParams())
	pnames = np.array(tractor.getParamNames())

	ndim = len(p0)
	nw = 32

	sampler = emcee.EnsembleSampler(nw, ndim, tractor, pool = mp.pool,
									live_dangerously=True)
	mhsampler = emcee.EnsembleSampler(nw, ndim, tractor, pool = mp.pool,
									  live_dangerously=True)

	# Scale step sizes until we get small lnp changes
	stepscale = 1e-7
	psteps = stepscale * np.array(tractor.getStepSizes())
	while True:
		pp = emcee.EnsembleSampler.sampleBall(p0, psteps, nw)
		# Put one walker at the nominal position.
		pp[0,:] = p0
		lnp = np.array(mp.map(tractor, pp))
		dlnp = lnp - np.max(lnp)
		print 'dlnp min', np.min(dlnp), 'median', np.median(dlnp)
		if np.median(dlnp) > -10:
			break
		psteps *= 0.1
		stepscale *= 0.1

	stepscales = np.zeros_like(psteps) + stepscale

	rstate = None
	alllnp = []
	allp = []
	for step in range(1, 201):
		allp.append(pp.copy())
		alllnp.append(lnp.copy())
		if step % 10 == 0:
			ibest = np.argmax(lnp)
			plots(tractor, #['modsum', 'chisum', 'modbest', 'chibest', 'lnps'],
				  ['modbest', 'chibest', 'lnps'],
				  step, pp=pp, ibest=ibest, alllnp=alllnp, **plotsa)
		print 'Run MCMC step', step
		kwargs = dict(storechain=False)
		# Alternate 5 steps of stretch move, 5 steps of MH.
		t0 = Time()
		if step % 10 in [4, 9]:
			# Resample walkers that are doing badly
			bestlnp = np.max(lnp)
			dlnp = lnp - bestlnp
			cut = -20
			I = np.flatnonzero(dlnp < cut)
			print len(I), 'walkers have lnprob more than', -cut, 'worse than the best'
			if len(I):
				ok = np.flatnonzero(dlnp >= cut)
				print 'Resampling from', len(ok), 'good walkers'
				# Sample another walker
				J = ok[np.random.randint(len(ok), size=len(I))]
				lnp0 = lnp[I]
				lnp1 = lnp[J]
				print 'J', J.shape, J
				#print 'lnp0', lnp0.shape, lnp0
				#print 'lnp1', lnp1.shape, lnp1
				#print 'pp[J,:]', pp[J,:].shape
				#print 'psteps', psteps.shape
				#print 'rand', np.random.normal(size=(len(J),len(psteps))).shape
				ppnew = pp[J,:] + psteps * np.random.normal(size=(len(J),len(psteps)))
				#print 'ppnew', ppnew.shape
				lnp2 = np.array(mp.map(tractor, ppnew))
				#print 'lnp2', lnp2.shape, lnp2
				print 'dlnps', ', '.join(['%.1f' % d for d in lnp2 - np.max(lnp)])
				# M-H acceptance rule (from original position, not resampled)
				acc = emcee.EnsembleSampler.mh_accept(lnp0, lnp2)
				dlnp = lnp2[acc] - lnp[I[acc]]
				lnp[I[acc]] = lnp2[acc]
				pp[I[acc],:] = ppnew[acc,:]
				print 'Accepted', sum(acc), 'resamplings'
				print '  with dlnp mean', np.mean(dlnp), 'median', np.median(dlnp)
				# FIXME: Should record the acceptance rate of this...

		elif step % 10 >= 5:
			print 'Using MH proposal'
			kwargs['mh_proposal'] = emcee.MH_proposal_axisaligned(psteps)
			pp,lnp,rstate = mhsampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate, **kwargs)
			print 'Running acceptance fraction: MH'#, mhsampler.acceptance_fraction
			print 'after', mhsampler.iterations, 'iterations'
			print 'mean', np.mean(mhsampler.acceptance_fraction)
		else:
			pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate, **kwargs)
			print 'Running acceptance fraction: emcee'#, sampler.acceptance_fraction
			print 'after', sampler.iterations, 'iterations'
			print 'mean', np.mean(sampler.acceptance_fraction)
		t_mcmc = (Time() - t0)
		print 'Best lnprob:', np.max(lnp)
		print 'dlnprobs:', ', '.join(['%.1f' % d for d in lnp - np.max(lnp)])
		print 'MCMC took', t_mcmc, 'sec'

		# Tweak step sizes...
		print 'Walker stdevs / psteps:'
		st = np.std(pp, axis=0)
		f = st / np.abs(psteps)
		print '  median', np.median(f)
		print '  range', np.min(f), np.max(f)
		if step % 10 == 9:
			# After this batch of MH updates, tweak step sizes.
			acc = np.mean(mhsampler.acceptance_fraction)
			tweak = 2.
			if acc < 0.33:
				psteps /= tweak
				stepscales /= tweak
				print 'Acceptance rate too low: decreasing step sizes'
			elif acc > 0.66:
				psteps *= tweak
				stepscales *= tweak
				print 'Acceptance rate too high: increasing step sizes'
			print 'log-mean step scales', np.exp(np.mean(np.log(stepscales)))

		# # Note that this is per-parameter.
		# mx = 1.2
		# tweak = np.clip(f, 1./mx, mx)
		# psteps *= tweak
		# print 'After tweaking:'
		# f = st / np.abs(psteps)
		# print '  median', np.median(f)
		# print '  range', np.min(f), np.max(f)
	tractor.setCatalog(allsources)
	tractor.setImages(allimages)
	print 'Checking all-image, all-source logprob...'
	params1 = tractor.getParams()
	print 'Getting initial logprob...'
	tractor.setParams(params0)
	alllnp0 = tractor.getLogProb()
	print 'Initial log-prob (all images, all sources)', alllnp0
	print 'Getting final logprob...'
	tractor.setParams(params1)
	alllnp1 = tractor.getLogProb()
	print 'Initial log-prob (all images, all sources)', alllnp0
	print 'Final   log-prob (all images, all sources)', alllnp1
	return dict(tractor=tractor, allp3=allp, pp3=pp, psteps3=psteps, Ibright3=Ibright)




# One time I forgot to reset the mpcache before pickling a Tractor... upon unpickling it
# fails here... so hack it up.
realRebuildProxy = multiprocessing.managers.RebuildProxy
def fakeRebuildProxy(*args, **kwargs):
	try:
		return realRebuildProxy(*args, **kwargs)
	except Exception as e:
		print 'real RebuildProxy failed:', e
		return None
