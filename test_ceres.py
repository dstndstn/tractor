import sys
import numpy as np
import logging
lvl = logging.INFO
#lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

from tractor import *

from ceres import *

# H,W = 100,100
# NS = 10
#H,W = 13,25
H,W = 25,25
NS = 25

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

iv = np.ones_like(mod)

tim1.data = mod

H,W = mod.shape

BH,BW = 5,5

nbw = int(np.ceil(W / float(BW)))
nbh = int(np.ceil(H / float(BH)))

blocks = []
for iy in range(nbh):
    for ix in range(nbw):
        x0 = ix * BW
        y0 = iy * BH
        slc = slice(y0, min(y0+BH, H)), slice(x0, min(x0+BW, W))
        data = (x0, y0, mod[slc].astype(np.float64),
                np.sqrt(iv[slc]).astype(np.float64))
        blocks.append((data, []))

# derivs = []
for i,src in enumerate(srcs):
    patches = src.getUnitFluxModelPatches(tim1)
    assert(len(patches) == 1)
    patch = patches[0]
    patch.clipTo(W,H)

    ph,pw = patch.shape
    bx0 = np.clip(int(np.floor( patch.x0       / float(BW))), 0, nbw-1)
    bx1 = np.clip(int(np.ceil ((patch.x0 + pw) / float(BW))), 0, nbw-1)
    by0 = np.clip(int(np.floor( patch.y0       / float(BH))), 0, nbh-1)
    by1 = np.clip(int(np.ceil ((patch.y0 + ph) / float(BH))), 0, nbh-1)

    print 'Patch touches x blocks [%i,%i], y blocks [%i,%i]' % (bx0, bx1, by0, by1)

    for by in range(by0, by1+1):
        for bx in range(bx0, bx1+1):
            bi = by * nbw + bx
            deriv = (i, patch.x0, patch.y0, patch.patch.astype(np.float64))
            blocks[bi][1].append(deriv)

fluxes = np.zeros(len(srcs)) + 10.

print 'Ceres forced phot:'
x = ceres_forced_phot(blocks, fluxes)
print 'got', x

print 'Fluxes:', fluxes

tractor.setParams(np.zeros(tractor.numberOfParams()) + 10.)
X2 = tractor.optimize_forced_photometry()
print 'optimize_forced_photometry() fluxes:', tractor.getParams()

