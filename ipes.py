
"""
We want to show some Tractor vs SDSS improvements using only SDSS
data.  Some validation can be done using the stripe overlaps -- the
"ipes".

Things to show / tests to run include:

-weird galaxy parameter distributions go away.  Don't even need ipes
 for this, just need a swath of imaging.

 -> galstats.py
 to select a galaxy sample

-our error estimates are better.  Do this by cross-matching objects in
 the ipes and comparing their measurements.  Compare the variance of
 the two SDSS measurements to the SDSS error estimates.  *This much is
 just evaluating SDSS results and doesn't even involve the Tractor!*
 We should find that the SDSS error estimates tend to
 over/underestimate the sample variance of the measurements.

-...then, look at the Tractor error estimates, and the two samples, on
 single fields fit separately.  We should find that the error
 estimates and the sample variance coincide.

-...then, look at the Tractor error estimates fitting the ipes
 simultaneously.  We should find that the error estimates tighten up.

-we can detect objects in the ipes that are not in the SDSS catalog.
 This requires either building a (multi-image) residual map [the
 correct approach], or guess-n-testing 2.5-sigma residuals in the
 single-image residuals [the pragmatic approach].  For main-survey
 ipes, I think I'm for the latter, since we can plug the fact that
 there's a better way to do it; and for two images the difference just
 isn't that much; and it cuts the dependency on the detection paper /
 technology.

-we do better on hard deblends / crowded regions / clusters.  I took a
 look at the Perseus cluster and the SDSS catalog is a mess, and we
 can tune it up without much trouble.  That's not the best example,
 since RHL will complain that Photo was built to perform well on the
 whole sky, so it's not surprising that it breaks down in peculiar
 places like rich clusters and Messier objects.  That's fair, so we
 could instead look at less-extreme clusters that are still of
 significant interest and where Photo's difficulties really matter.
 This would be an opportunity to show the power of simultaneously
 fitting sky + galaxies, though that does involve writing a flexible
 sky model.  That's required for projects with Rachel anyway so I'm
 not bothered.  For this paper, I think I want to just show the
 before-n-after pictures, rather than do any sort of numerical
 characterization of what's going on.  Do we want to show any sampling
 results?  If so, this would be the place since we could show
 covariances between shape measurements of neighbouring galaxies,
 which I think is something nobody else is even trying.




"""
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import logging

from astrometry.util.fits import *
from astrometry.blind.plotstuff import *
from astrometry.util.c import *

from astrometry.sdss import DR9

from tractor.sdss import *
from tractor import *


def main():

	#lvl = logging.INFO
	lvl = logging.DEBUG
	logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

	#plot_ipes()
	refit_galaxies()

def refit_galaxies():
	T = fits_table('exp4_dstn.fit')

	sdss = DR9(basedir='paper0-data-dr9')

	print 'basedir', sdss.basedir
	print 'dasurl', sdss.dasurl

	bandname = 'i'
	# ROI radius in pixels
	S = 100

	rlo,rhi = 4.1, 4.4
	Ti = T[(T.exprad_i > rlo) * (T.exprad_i < rhi)]
	Ti = Ti[Ti.expmag_i < 19]
	I = np.argsort(Ti.expmag_i)
	Ti = Ti[I]

	print 'Cut to', len(Ti), 'galaxies in radius and mag cuts'


	for gali in range(len(Ti)):
		ti = Ti[gali]

		print 'psField:', sdss.retrieve('psField', ti.run, ti.camcol, ti.field,
										bandname)

		im,info = get_tractor_image_dr9(ti.run, ti.camcol, ti.field, bandname,
										roiradecsize=(ti.ra, ti.dec, S),
										sdss=sdss)

		roi = info['roi']

		cat = get_tractor_sources_dr9(ti.run, ti.camcol, ti.field, bandname,
									  sdss=sdss, roi=roi, bands=[bandname])

		tractor = Tractor(Images(im), cat)
		print 'Tractor', tractor

		ima = dict(interpolation='nearest', origin='lower')
		zr = im.zr
		ima.update(vmin=zr[0], vmax=zr[1])
		ima.update(extent=roi)

		imchi = ima.copy()
		imchi.update(vmin=-5, vmax=5)

		mod0 = tractor.getModelImage(0)
		chi0 = tractor.getChiImage(0)

		tractor.freezeParam('images')
		p0 = tractor.getParams()

		while True:
			dlnp,X,alpha = tractor.optimize(damp=1e-3)
			print 'dlnp', dlnp
			print 'alpha', alpha
			if dlnp < 1:
				# p0 = np.array(tractor.getParams())
				# dp = np.array(X)				
				# tractor.setParams(p0 + dp)
				# modi = tractor.getModelImage(0)
				# tractor.setParams(p0)
				# plt.clf()
				# plt.imshow(modi, interpolation='nearest', origin='lower')
				# plt.savefig('badstep-%06i.png' % gali)
				# print 'Attempted parameter changes:'
				# for x,nm in zip(X, tractor.getParamNames()):
				# 	print '  ', nm, '  ', x
				break

		# Find bright sources and unfreeze them.
		tractor.catalog.freezeAllParams()
		for i,src in enumerate(tractor.catalog):
			if src.getBrightness().i < 19.:
				tractor.catalog.thawParam(i)

		print 'Fitting bright sources:'
		for nm in tractor.getParamNames():
			print '  ', nm

		while True:
			dlnp,X,alpha = tractor.optimize(damp=1e-3)
			print 'dlnp', dlnp
			print 'alpha', alpha
			if dlnp < 1:
				break

		tractor.catalog.thawAllParams()

		p1 = tractor.getParams()
		mod1 = tractor.getModelImage(0)
		chi1 = tractor.getChiImage(0)

		R,C = 2,3
		plt.clf()
		plt.suptitle(im.name)
		plt.subplot(R,C,1)

		plt.imshow(im.getImage(), **ima)
		plt.gray()

		plt.subplot(R,C,2)

		plt.imshow(mod0, **ima)
		plt.gray()

		im = tractor.getImage(0)
		tractor.setParams(p0)
		plot_ellipses(im, tractor.catalog)
		tractor.setParams(p1)

		plt.subplot(R,C,3)

		plt.imshow(chi0, **imchi)
		plt.gray()

		plt.subplot(R,C,5)

		plt.imshow(mod1, **ima)
		plt.gray()
		plot_ellipses(im, tractor.catalog)

		plt.subplot(R,C,6)

		plt.imshow(chi1, **imchi)
		plt.gray()

		plt.savefig('trgal-%06i.png' % gali)



def plot_ellipses(im, cat):
	wcs = im.getWcs()
	x0,y0 = wcs.x0, wcs.y0
	#xc,yc = wcs.positionToPixel(RaDecPos(ti.ra, ti.dec))
	H,W = im.getImage().shape
	xc,yc = W/2., H/2.
	cd = wcs.cdAtPixel(xc,yc)
	ax = plt.axis()
	for src in cat:
		pos = src.getPosition()
		x,y = wcs.positionToPixel(pos)
		x += x0
		y += y0
		gals = []
		if isinstance(src, PointSource):
			plt.plot(x, y, 'g+')
			continue
		elif isinstance(src, ExpGalaxy):
			gals.append((True, src.shape))
		elif isinstance(src, DevGalaxy):
			gals.append((False, src.shape))
		elif isinstance(src, CompositeGalaxy):
			gals.append((True,  src.shapeExp))
			gals.append((False, src.shapeDev))
		else:
			print 'Unknown source type:', src
			continue

		theta = np.linspace(0, 2*np.pi, 90)
		ux,uy = np.cos(theta), np.sin(theta)
		u = np.vstack((ux,uy)).T

		for isexp,shape in gals:
			T = np.linalg.inv(shape.getTensor(cd))
			#print 'T shape', T.shape
			#print 'u shape', u.shape
			dx,dy = np.dot(T, u.T)
			if isexp:
				c = 'm'
			else:
				c = 'c'
			#print 'x,y', x, y
			#print 'dx range', dx.min(), dx.max()
			#print 'dy range', dy.min(), dy.max()
			plt.plot(x + dx, y + dy, '-', color=c)
			plt.plot([x], [y], '+', color=c)
			
	plt.axis(ax)
	


def plot_ipes():
	T = fits_table('dr9fields.fits')

	def get_rrdd(T):
		dx,dy = 2048./2, 1489./2
		RR,DD = [],[]
		for sx,sy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
			r = T.ra  + sx*dx*T.cd11 + sy*dy*T.cd12
			d = T.dec + sx*dx*T.cd21 + sy*dy*T.cd22
			RR.append(r)
			DD.append(d)
		RR.append(RR[0])
		DD.append(DD[0])
		RR = np.vstack(RR)
		DD = np.vstack(DD)
		#print RR.shape
		return RR,DD

	
	if True:
		RR,DD = get_rrdd(T)
	
		W,H = 1000,500
		p = Plotstuff(outformat=PLOTSTUFF_FORMAT_PNG, size=(W,H))
		p.wcs = anwcs_create_allsky_hammer_aitoff(0., 0., W, H)
		p.color = 'verydarkblue'
		p.plot('fill')
		p.color = 'white'
		p.alpha = 0.25
		p.apply_settings()
		for rr,dd in zip(RR.T, DD.T):
			if not (np.all((rr - 180) > 0) or np.all((rr - 180) < 0)):
				print 'Skipping boundary-spanning', rr,dd
				continue
			p.move_to_radec(rr[0], dd[0])
			for r,d in zip(rr[1:],dd[1:]):
				p.line_to_radec(r,d)
			p.fill()
		p.color = 'gray'
		p.plot_grid(30, 30, 30, 30)
		p.write('rd1.png')
	
	#if True:
	for r,d,w,g,fn in [#(230,30,120,10,'rd2.png'),
		(255,15,30,5,'rd3.png'),
		(250,20,10,1,'rd4.png'),
		(250,18,3,1, 'rd5.png'),
		(250,17.8,1,0.5, 'rd6.png')]:
	
		W,H = 1000,1000
		p = Plotstuff(outformat=PLOTSTUFF_FORMAT_PNG, size=(W,H),
					  rdw=(r,d,w))
		rmn,rmx,dmn,dmx = anwcs_get_radec_bounds(p.wcs, 100)
		print 'Bounds', rmn,rmx,dmn,dmx
	
		T.cut((T.ramin < rmx) * (T.ramax > rmn) *
			  (T.decmin < dmx) * (T.decmax > dmn))
		RR,DD = get_rrdd(T)
	
		p.color = 'verydarkblue'
		p.plot('fill')
		p.color = 'white'
		p.alpha = 0.25
		#p.op = CAIRO_OPERATOR_ADD
		p.apply_settings()
		for rr,dd in zip(RR.T, DD.T):
			if not (np.all((rr - 180) > 0) or np.all((rr - 180) < 0)):
				#print 'Skipping boundary-spanning', rr,dd
				continue
			p.move_to_radec(rr[0], dd[0])
			for r,d in zip(rr[1:],dd[1:]):
				p.line_to_radec(r,d)
			p.fill_preserve()
			p.stroke()
		p.color = 'gray'
		p.plot_grid(g,g,g,g)
		p.write(fn)
	
   
if __name__ == '__main__':
	main()
	
