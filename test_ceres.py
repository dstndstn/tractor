import sys
import numpy as np
import logging
lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

from tractor import *

from ceres import *

# H,W = 100,100
# NS = 10
H,W = 5,5
NS = 2

f = 100.
img = np.zeros((H,W))
tim1 = Image(data=img, invvar=np.ones_like(img),
             psf=NCircularGaussianPSF([1.],[1.]),
             wcs=NullWCS(),
             photocal=LinearPhotoCal(1.),
             sky=ConstantSky(0.))
srcs = []

# A source off the image, to test non-packed columns
# srcs.append(PointSource(PixPos(-10,-10), Flux(f)))

srcs.extend([PointSource(PixPos(np.random.randint(W), np.random.randint(H)),
                         Flux(f)) for x in range(NS) ])


tractor = Tractor([tim1], srcs)
tractor.freezeParam('images')
for src in srcs:
    src.freezeParam('pos')
#tractor.freezeParamsRecursive('*')

mod = tractor.getModelImage(0)
mod += np.random.normal(size=mod.shape)

tim1.data = mod




derivs = []
for i,src in enumerate(srcs):
    patches = src.getUnitFluxModelPatches(tim1)
    assert(len(patches) == 1)
    patch = patches[0]
    derivs.append((i, patch.x0, patch.y0, patch.patch.astype(np.float64)))

iv = np.ones_like(mod)
data = (0, 0, mod.astype(np.float64), iv.astype(np.float64))
blocks = [ (data, derivs) ]

fluxes = np.zeros(len(srcs)) + 10.

print 'Ceres forced phot:'
x = ceres_forced_phot(blocks, fluxes)
print 'got', x

