
"""
We want to show some Tractor vs SDSS improvements using only SDSS
data.  Some validation can be done using the stripe overlaps -- the
"ipes".

Things to show / tests to run include:

-weird galaxy parameter distributions go away.  Don't even need ipes
 for this, just need a swath of imaging.

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

from astrometry.util.fits import *


T = fits_table('dr9fields.fits')

print 'cd11', T.cd11.min(), T.cd11.max()
print 'cd12', T.cd12.min(), T.cd12.max()
print 'cd21', T.cd21.min(), T.cd21.max()
print 'cd22', T.cd22.min(), T.cd22.max()

def get_rrdd(T):
	dx,dy = 2048./2, 1489./2
	RR,DD = [],[]
	for sx,sy in [(-1,-1), (1,-1), (1,1), (-1,1)]:
		r,d = T.ra + sx*dx*T.cd11 + sy*dy*T.cd12, T.dec + sx*dx*T.cd21 + sy*dy*T.cd22
		RR.append(r)
		DD.append(d)
	RR.append(RR[0])
	DD.append(DD[0])
	RR = np.vstack(RR)
	DD = np.vstack(DD)
	print RR.shape
	return RR,DD

from astrometry.blind.plotstuff import *
from astrometry.util.c import *

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



# plt.clf()
# plt.plot(RR, DD, 'r-')
# plt.xlabel('RA (deg)')
# plt.xlim(360, 0)
# plt.ylabel('Dec (deg)')
# plt.ylim(-30, 90)
# plt.savefig('rd.pdf')

   
