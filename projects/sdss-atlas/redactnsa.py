from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits as pyf
from general import general
from halflight import halflight
import urllib


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
w=data.field('IAUNAME')
good=np.array([True for x in data.field('RA')])
indx1=np.where(y[:,5] <= 0)
good[indx1]=False
indx2=np.where(y[:,3] <= 0)
good[indx2]=False
indx3=np.where(z > 60)
good[indx3]=False
indx4=np.where(z < 40)
good[indx4]=False

gra=a[good]
gdec = b[good]
grad = z[good]
g = e[good]
newGood=np.array([True for x in g])

def getImage(x):
    #nocom = [t for t in xrange(len(e[good])) if g[good][t]==x]
    mask = e==x
    t = data[mask][0]
    iau = str(t['IAUNAME'])
    print(iau)
    if t['DEC'] > 0: 
        url='http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/%sh/p%02d/%s/%s.jpg' %(iau[1:3],((int(iau[11:13]))/2)*2,iau,iau)
    else: 
        url='http://sdss.physics.nyu.edu/mblanton/v0/detect/v0_1/%sh/m%02d/%s/%s.jpg' %(iau[1:3],((int(iau[11:13]))/2)*2,iau,iau)

    return url

rc3 = pyf.open('allrc3.fits')

for entry in rc3[1].data:
    radius = ((10**entry['LOG_D25'])/10.)/2. #In arc-minutes
    radius /= 60. #In degrees
    m1,m2,d12 = match_radec(gra,gdec,entry['RA'],entry['DEC'],radius)
    newGood[m1]=False

print(len(g[newGood]))

gra=gra[newGood]
g=g[newGood] #list of all nsaids that should now be checked 
gdec=gdec[newGood]
grad = grad[newGood]
i = 0
print(len(g))

fekta = open("objs2_ekta.txt",'w')
fmykytyn = open("objs2_mykytyn.txt",'w')
fhogg = open("objs2_hogg.txt",'w')
for obj in g:
    if (i==0):
        url = getImage(obj)
        urllib.urlretrieve(url, 'nsahogg2/%s.jpg' %(obj))
        fhogg.write("%s\n" % obj)
    elif (i==1):
        url = getImage(obj)
        urllib.urlretrieve(url, 'nsamykytyn2/%s.jpg' %(obj))
        fmykytyn.write("%s\n" % obj)
    else:
        url = getImage(obj)
        urllib.urlretrieve(url, 'nsaekta2/%s.jpg' %(obj))
        fekta.write("%s\n" % obj)
    i = (i+1) % 3


#general("NSA_ID_%d" % g[0],gra[0],gdec[0],25./60.,itune1=6,itune2=6,nocache=True)
#halflight("NSA_ID_%d" % g[0])
