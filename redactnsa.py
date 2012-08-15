import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits as pyf
from general import general
from halflight import halflight


from astrometry.libkd.spherematch import match_radec

data = pyf.open("nsa-short.fits.gz")[1].data
a=data.field('RA')
b=data.field('DEC')
y = data.field('SERSICFLUX')
z = data.field('SERSIC_TH50')
n=data.field('SERSIC_N')
p50=data.field('PETROTH50')
p90=data.field('PETROTH90')
e=data.field('NSAID')
good=np.array([True for x in data.field('RA')])
indx1=np.where(y[:,5] <= 0)
good[indx1]=False
indx2=np.where(y[:,3] <= 0)
good[indx2]=False
indx3=np.where(z > 158)
good[indx3]=False
indx4=np.where(z < 120)
good[indx4]=False

gra=a[good]
gdec = b[good]
grad = z[good]
g = e[good]
print len(g)

newGood=np.array([True for x in g])


rc3 = pyf.open('rc3limited.fits')

for entry in rc3[1].data:
    radius = ((10**entry['LOG_D25'])/10.)/2. #In arc-minutes
    radius /= 60. #In degrees
    m1,m2,d12 = match_radec(gra,gdec,entry['RA'],entry['DEC'],radius)
    if len(m1) !=0:
        print m1
    newGood[m1]=False

gra=gra[newGood]
g=g[newGood]
gdec=gdec[newGood]
grad = grad[newGood]
print g
print gra[0]
print gdec[0]
print len(g)


#general("NSA_ID_%d" % g[0],gra[0],gdec[0],25./60.,itune1=6,itune2=6,nocache=True)
#halflight("NSA_ID_%d" % g[0])
