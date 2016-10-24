from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
from astrometry.libkd.spherematch import *
from astrometry.util.util import *
from astrometry.util.miscutils import *
from astrometry.sdss import *

ps = PlotSequence('w4')

fn = 'wisew4-infootprint.fits'

if os.path.exists(fn):
    print('Reading', fn)
    T = fits_table(fn)
    ps.skip(2)

else:
    T = fits_table('wisew4_nomatch.fits')
    print('Read', len(T), 'unmatched W4 detections')
    
    plt.clf()
    loghist(T.ra, T.dec, 200)
    plt.title('W4 detections, no SDSS match: %i' % (len(T)))
    ps.savefig()
    
    W = fits_table('window_flist.fits', columns=['ra','dec','rerun',
                                                 'mu_start', 'mu_end', 'nu_start', 'nu_end',
                                                 'node', 'incl'])
    print('Read', len(W), 'fields')
    W.cut(W.rerun == '301')
    print('Cut to', len(W), 'rerun 301')
    
    #infootprint = np.zeros(len(T), bool)
    #I,J,d = match_radec(T.ra, T.dec, W.ra, W.dec, np.hypot(13., 9.)/2./60.)
    #infootprint[I] = True
    
    #inds = match_radec(T.ra, T.dec, W.ra, W.dec, np.hypot(13., 9.)/2./60.)
    #infootprint = np.array([ii is not None for ii in inds])
    
    infootprint = np.zeros(len(T), bool)
    inds = match_radec(W.ra, W.dec, T.ra, T.dec, np.hypot(13., 9.)/2./60.,
                       indexlist=True)
    for ii in inds:
        if ii is None:
            continue
        infootprint[np.array(ii)] = True
    
    T.cut(infootprint)
    print('Cut to', len(T), 'W4 detections in SDSS footprint')
    
    plt.clf()
    loghist(T.ra, T.dec, 200)
    plt.title('W4 detections, no SDSS match, rough footprint %i' % (len(T)))
    ps.savefig()
    
    inds = match_radec(W.ra, W.dec, T.ra, T.dec, np.hypot(13., 9.)/2./60.,
                       indexlist=True)
    
    infootprint = np.zeros(len(T), bool)
    for i,jj in enumerate(inds):
        if jj is None:
            continue
        # field i -- check which of WISE sources jj are within.
        # fake WCS for intermediate world coords
        if i % 1000 == 0:
            print('.', end=' ')
        wi = W[i]
        fakewcs = Tan(wi.ra, wi.dec, 0., 0., 1e-3, 0., 0., 1e3, 1000., 1000.)
        rd = np.array([munu_to_radec_deg(mu, nu, wi.node, wi.incl)
                       for mu,nu in [(wi.mu_start, wi.nu_start),
                                     (wi.mu_start, wi.nu_end),
                                     (wi.mu_end,   wi.nu_end),
                                     (wi.mu_end,   wi.nu_start)]])
        ok,uu,vv = fakewcs.radec2iwc(rd[:,0], rd[:,1])
        poly = np.array(zip(uu,vv))
    
        jj = np.array(jj)
        rr,dd = T.ra[jj], T.dec[jj]
        ok,uu,vv = fakewcs.radec2iwc(rr,dd)
        inside = point_in_poly(uu, vv, poly)
        #print sum(inside), 'of', len(inside), 'are inside'
        infootprint[jj[inside]] = True
    print()
    T.cut(infootprint)
    print('Cut to', len(T), 'W4 detections actually inside SDSS fields')

    T.writeto(fn)
    print('Saved', fn)

plt.clf()
loghist(T.ra, T.dec, 200)
plt.title('W4 detections, no SDSS match, fine footprint %i' % (len(T)))
ps.savefig()

descr = 'W4 detections, no SDSS match'

l,b = radectolb(T.ra, T.dec)
bcut = 20.
descr = descr + ', |b|>%.0f' % bcut
T.cut(np.abs(b) > bcut)
plt.clf()
loghist(T.ra, T.dec, 200)
plt.title(descr + ': %i' % (len(T)))
ps.savefig()

l,b = radectolb(T.ra, T.dec)
gdist = np.hypot(l, b)
descr = descr + ', ||GC||>30'
T.cut(gdist > 30.)
plt.clf()
loghist(T.ra, T.dec, 200)
plt.title(descr + ': %i' % (len(T)))
ps.savefig()

u = np.linspace(0, 360)
v = np.zeros_like(u)
r,d = ecliptictoradec(u, v)
ax = plt.axis()
plt.plot(r, d, 'b-')
plt.axis(ax)
ps.savefig()

u,v = radectoecliptic(T.ra, T.dec)
T.cut(np.abs(v) > 5)
plt.clf()
loghist(T.ra, T.dec, 200)
descr = descr + ', |elat|>5'
plt.title(descr + ': %i' % (len(T)))
ps.savefig()

T.cut(T.ra < 195.)
print('Cut to', len(T), 'in RA box')

T.writeto('w4targets.fits')


# T.cut((T.ra > 125) * (T.ra < 225) * (T.dec > 0) * (T.dec < 60))
# plt.clf()
# loghist(T.ra, T.dec, 200)
# plt.title(descr + ': %i' % (len(T)))
# ps.savefig()


