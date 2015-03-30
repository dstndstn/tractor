import matplotlib
matplotlib.use('Agg')
import pylab as plt

from tpv import tpv
from astrometry.util.fits import *
from astrometry.util.util import *
from astropy.io import fits

import os

hdrfn = os.path.join(os.path.dirname(__file__), 'pvhdr.fits')

# read header
hdulist = fits.open(hdrfn)
head = hdulist[0].header

#T = fits_table('stxy.fits')
print 'TPV.transform'
rr,dd = tpv.transform(np.array([1.]), np.array([1.]), head)
print 'TPV RA,Dec', rr,dd
print

print 'Running wcs-pv2sip...'
print
H,W = 4094, 2046
wfn = 'pvsip.wcs'
os.unlink(wfn)
cmd = ('wcs-pv2sip -v -v -S -o 5 -e %i -W %i -H %i -X %i -Y %i %s %s' %
       (0, W, H, W, H, hdrfn, wfn))
print cmd
rtn = os.system(cmd)
assert(rtn == 0)

pvsip = Sip(wfn)

sip_compute_inverse_polynomials(pvsip, 100, 100, 1, W, 1, H)

r,d = pvsip.pixelxy2radec(1., 1.)
print 'PV-SIP RA,Dec', r,d
print 'vs TPV RA,Dec', rr,dd

ok,x,y = pvsip.radec2pixelxy(rr,dd)
print 'PV-SIP RA,Dec', rr,dd, '-> x,y', x,y

#import sys
#sys.exit(0)

xx = np.linspace(1., W, 100)
yy = np.linspace(1., H, 100)
X = np.linspace(1., W, 20)
Y = np.linspace(1., H, 20)
z = np.zeros_like(xx)

plt.clf()
y2 = yy
for x in X:
    x2 = x + z
    r,d = tpv.transform(x2, y2, head)
    r2,d2 = pvsip.pixelxy2radec(x2, y2)
    plt.plot(y2, (r2 - r)*3600., 'r-')
    plt.plot(y2, (d2 - d)*3600., 'b-')

x2 = xx
for y in Y:
    y2 = y + z
    r,d = tpv.transform(x2, y2, head)
    r2,d2 = pvsip.pixelxy2radec(x2, y2)
    plt.plot(x2, (r2 - r)*3600., 'r-')
    plt.plot(x2, (d2 - d)*3600., 'b-')

plt.xlabel('pixel coord')
plt.ylabel('dRA or dDec (arcsec)')
plt.savefig('2.png')
print 'saved plot 2'







plt.clf()
y2 = yy
for x in X:
    x2 = x + z
    r,d = tpv.transform(x2, y2, head)
    ok,x3,y3 = pvsip.radec2pixelxy(r, d)
    plt.plot(y2, (x3 - x2), 'r-')
    plt.plot(y2, (y3 - y2), 'b-')

x2 = xx
for y in Y:
    y2 = y + z
    r,d = tpv.transform(x2, y2, head)
    ok,x3,y3 = pvsip.radec2pixelxy(r, d)
    plt.plot(x2, (x3 - x2), 'r-')
    plt.plot(x2, (y3 - y2), 'b-')

plt.xlabel('pixel coord')
plt.ylabel('dPixel')
plt.savefig('3.png')
print 'saved plot 3'


##### this is just checking the SIP inversion

plt.clf()
y2 = yy
for x in X:
    x2 = x + z
    r,d = pvsip.pixelxy2radec(x2, y2)
    ok,x3,y3 = pvsip.radec2pixelxy(r, d)
    plt.plot(y2, (x3 - x2)*3600., 'r-')
    plt.plot(y2, (y3 - y2)*3600., 'b-')

x2 = xx
for y in Y:
    y2 = y + z
    r,d = pvsip.pixelxy2radec(x2, y2)
    ok,x3,y3 = pvsip.radec2pixelxy(r, d)
    plt.plot(x2, (x3 - x2)*3600., 'r-')
    plt.plot(x2, (y3 - y2)*3600., 'b-')

plt.xlabel('pixel coord')
plt.ylabel('dPixel')
plt.savefig('4.png')
print 'saved plot 4'


    
