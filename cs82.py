import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import logging
from glob import glob

from astrometry.util.fits import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.ttime import *
from astrometry.sdss import *
from astrometry.libkd.spherematch import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

def get_cs82_sources(T, maglim=25, bands=['u','g','r','i','z']):
	srcs = Catalog()
	isrcs = []
	for i,t in enumerate(T):
		if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
			#print 'PSF'
			themag = t.mag_psf
			nm = NanoMaggies.magToNanomaggies(themag)
			m = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
			srcs.append(PointSource(RaDecPos(t.ra, t.dec), m))
			isrcs.append(i)
			continue

		if t.mag_disk > maglim and t.mag_spheroid > maglim:
			#print 'Faint'
			continue

		# deV: spheroid
		# exp: disk

		dmag = t.mag_spheroid
		emag = t.mag_disk

		# SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

		if dmag <= maglim:
			shape_dev = GalaxyShape(t.spheroid_reff_world * 3600.,
									t.spheroid_aspect_world,
									t.spheroid_theta_world + 90.)

		if emag <= maglim:
			shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600.,
									t.disk_aspect_world,
									t.disk_theta_world + 90.)

		pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

		isrcs.append(i)
		if emag > maglim and dmag <= maglim:
			nm = NanoMaggies.magToNanomaggies(dmag)
			m_dev = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
			srcs.append(DevGalaxy(pos, m_dev, shape_dev))
			continue
		if emag <= maglim and dmag > maglim:
			nm = NanoMaggies.magToNanomaggies(emag)
			m_exp = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
			srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
			continue

        # print 'Composite'
        nmd = NanoMaggies.magToNanomaggies(dmag)
		nme = NanoMaggies.magToNanomaggies(emag)
		nm = nmd + nme
		fdev = (nmd / nm)
		m = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
		srcs.append(FixedCompositeGalaxy(pos, m, fdev, shape_exp, shape_dev))

	#print 'Sources:', len(srcs)
	return srcs, np.array(isrcs)


def getTables(cs82field, enclosed=True, extra_cols=[]):
	T = fits_table('cs82data/%s_y.V2.7A.swarp.cut.deVexp.fit' % cs82field,
				   hdu=2, column_map={'ALPHA_J2000':'ra',
									  'DELTA_J2000':'dec'},
				   columns=[x.upper() for x in
							['ALPHA_J2000', 'DELTA_J2000',
							'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
							 'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
							 'disk_theta_world', 'spheroid_reff_world',
							 'spheroid_aspect_world', 'spheroid_theta_world',
							 'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
	ra0,ra1 = T.ra.min(), T.ra.max()
	dec0,dec1 = T.dec.min(), T.dec.max()
	print 'RA', ra0,ra1
	print 'Dec', dec0,dec1
	T.index = np.arange(len(T))

	# ASSUME no RA wrap-around in the catalog
	trad = 0.5 * np.hypot(ra1 - ra0, dec1 - dec0)
	tcen = radectoxyz((ra1+ra0)*0.5, (dec1+dec0)*0.5)

	frad = 0.5 * np.hypot(13., 9.) / 60.

	fn = 'sdssfield-%s.fits' % cs82field
	if os.path.exists(fn):
		print 'Reading', fn
		F = fits_table(fn)
	else:
		F = fits_table('window_flist-DR9.fits')

		# These runs don't appear in DAS
		#F.cut((F.run != 4322) * (F.run != 4240) * (F.run != 4266))
		F.cut(F.rerun != "157")

		# For Stripe 82, mu-nu is aligned with RA,Dec.
		rd = []
		rd.append(munu_to_radec_deg(F.mu_start, F.nu_start, F.node, F.incl))
		rd.append(munu_to_radec_deg(F.mu_end,   F.nu_end,   F.node, F.incl))
		rd = np.array(rd)
		F.ra0  = np.min(rd[:,0,:], axis=0)
		F.ra1  = np.max(rd[:,0,:], axis=0)
		F.dec0 = np.min(rd[:,1,:], axis=0)
		F.dec1 = np.max(rd[:,1,:], axis=0)

		I = np.flatnonzero((F.ra0 <= T.ra.max()) *
						   (F.ra1 >= T.ra.min()) *
						   (F.dec0 <= T.dec.max()) *
						   (F.dec1 >= T.dec.min()))
		print 'Possibly overlapping fields:', len(I)
		F.cut(I)

		# When will I ever learn not to cut on RA boxes when there is wrap-around?
		xyz = radectoxyz(F.ra, F.dec)
		r2 = np.sum((xyz - tcen)**2, axis=1)
		I = np.flatnonzero(r2 < deg2distsq(trad + frad))
		print 'Possibly overlapping fields:', len(I)
		F.cut(I)

		F.enclosed = ((F.ra0 >= T.ra.min()) *
					  (F.ra1 <= T.ra.max()) *
					  (F.dec0 >= T.dec.min()) *
					  (F.dec1 <= T.dec.max()))
		
		# Sort by distance from the center of the field.
		ra  = (T.ra.min()  + T.ra.max() ) / 2.
		dec = (T.dec.min() + T.dec.max()) / 2.
		I = np.argsort( ((F.ra0  + F.ra1 )/2. - ra )**2 +
						((F.dec0 + F.dec1)/2. - dec)**2 )
		F.cut(I)

		F.writeto(fn)
		print 'Wrote', fn

	if enclosed:
		F.cut(F.enclosed)
		print 'Enclosed fields:', len(F)
		
	return T,F


def main(opt, cs82field):
    bands = opt.bands

    T,F = getTables(cs82field, enclosed=False)

	# We probably have to work in Dec slices to keep the memory reasonable
	dec0 = T.dec.min()
	dec1 = T.dec.max()
	print 'Dec range:', dec0, dec1
	#nslices = 4
	#ddec = (dec1 - dec0) / nslices
	#print 'ddec:', ddec

	sdss = DR9(basedir='cs82data/dr9')

	### HACK -- ignore 0/360 issues
	ra0 = T.ra.min()
	ra1 = T.ra.max()
	print 'RA range:', ra0, ra1
	assert(ra1 - ra0 < 2.)

	decs = np.linspace(dec0, dec1, 5)
	ras  = np.linspace(ra0,  ra1, 5)

	print 'Score range:', F.score.min(), F.score.max()
	print 'Before score cut:', len(F)
	F.cut(F.score > 0.5)
	print 'Cut on score:', len(F)
    
	for decslice,(dlo,dhi) in enumerate(zip(decs, decs[1:])):
		print 'Dec slice:', dlo, dhi
		for raslice,(rlo,rhi) in enumerate(zip(ras, ras[1:])):
			print 'RA slice:', rlo, rhi

			# in deg
			margin = 15. / 3600.
			Ti = T[((T.dec + margin) >= dlo) * ((T.dec - margin) <= dhi) *
				   ((T.ra  + margin) >= rlo) * ((T.ra  - margin) <= rhi)]
			Ti.marginal = np.logical_not((Ti.dec >= dlo) * (Ti.dec <= dhi) *
										 (Ti.ra  >= rlo) * (Ti.ra  <= rhi))
			print len(Ti), 'sources in RA,Dec slice'
			print len(np.flatnonzero(Ti.marginal)), 'are in the margins'

			Fi = F[np.logical_not(np.logical_or(F.dec0 > dhi, F.dec1 < dlo)) *
				   np.logical_not(np.logical_or(F.ra0  > rhi, F.ra1  < rlo))]
			print len(Fi), 'fields in RA,Dec slice'


			print 'Creating Tractor sources...'
			maglim = 24
			cat,icat = get_cs82_sources(Ti, maglim=maglim, bands=bands)
			print 'Got', len(cat), 'sources'

            # FIXME -- initialize fluxes by matching to SDSS sources?
            # FIXME -- freeze marginal sources!
            
            for band in bands:
                cat.freezeParamsRecursive('*')
                cat.thawPathsTo(band)

                tims = []
                sigs = []
                for i,(r,c,f) in enumerate(zip(Fi.run, Fi.camcol, Fi.field)):
                    print 'Reading', (i+1), 'of', len(Fi), ':', r,c,f,band
                    tim,inf = get_tractor_image_dr9(r, c, f, band, sdss=sdss,
                                                    nanomaggies=True, zrange=[-2,5],
                                                    roiradecbox=[rlo,rhi,dlo,dhi],
                                                    invvarIgnoresSourceFlux=True)
                    if tim is None:
                        continue
                    (H,W) = tim.shape
                    tim.wcs.setConstantCd(W/2., H/2.)
                    del tim.origInvvar
                    del tim.starMask
                    del tim.mask
                    # needed for optimize_forced_photometry with rois
                    #del tim.invvar
                    tims.append(tim)
                    npix += (H*W)
                    print 'got', (H*W), 'pixels, total', npix

                    sigs.append(1./np.sqrt(np.median(tim.invvar)))


                print 'Read', len(tims), 'images'
                print 'total of', npix, 'pixels'

                #minsig = getattr(opt, 'minsig%i' % band)
                #minsb = sig1 * minsig
                sig1 = np.median(sigs)
                minsig = 0.1
                minsb= minsig * sig1
                print 'Sigma1:', sig1, 'minsig', minsig, 'minsb', minsb
                
                tr = Tractor(tims, cat)
                tr.freezeParam('images')
                sz = 8
                phot = tr.optimize_forced_photometry(
                    minsb=minsb, mindlnp=1.,
                    fitstats=True, variance=True,
                    shared_params=False, use_ceres=True,
                    BW=sz, BH=sz)

                print 'Forced phot finished'
                

                

if __name__ == '__main__':
	import optparse
	parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-b', dest='bands', type=str, default='ugriz',
                      help='SDSS bands (default %default)')
	opt,args = parser.parse_args()

	cs82field = 'S82p18p'
	main(opt, cs82field)
