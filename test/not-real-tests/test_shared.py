from __future__ import print_function
import numpy as np
from tractor import *

psf = NCircularGaussianPSF([1.0],[1.0])
sky = ConstantSky(1.)

tims = []
for i in range(2):
    H,W = 10,10
    tims.append(Image(data = np.zeros((H,W)) + 2*i,
                      invvar = np.ones((H,W)),
                      psf=psf,
                      wcs = NullWCS(),
                      sky = sky,
                      photocal = NullPhotoCal()))

ims = Images(*tims)
print('Tims:')
for nm,p in zip(ims.getParamNames(), ims.getParams()):
    print('  ', nm, p)

cat = [PointSource(PixPos(5,5), Flux(1.))]

tractor = Tractor(ims, cat)

allderivs = tractor.getDerivs()

print()
print('With shared params:')
X1,V1 = tractor.getUpdateDirection(allderivs, variance=True)
print('X:', X1)
print('V:', V1)

print()
print('Without shared params:')
X2,V2 = tractor.getUpdateDirection(allderivs, variance=True,
                                   shared_params=False)
print('X2:', X2)
print('V2:', V2)


p0 = tractor.getParams()

dlnp1,X1,alpha1 = tractor.optimize()
p1 = tractor.getParams()

tractor.setParams(p0)

dlnp2,X2,alpha2 = tractor.optimize(shared_params=False)
p2 = tractor.getParams()

print('Optimize:')
print('dlnp1', dlnp1)
print('dlnp2', dlnp2)
print('X1', X1)
print('X2', X2)

print('params:')
for nm,x0,x1,x2 in zip(tractor.getParamNames(), p0, p1, p2):
    print('  ', nm, 'init', x0, 'opt1', x1, 'opt2', x2)
