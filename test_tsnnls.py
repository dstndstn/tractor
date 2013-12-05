import sys
import numpy as np
import logging
lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

from tractor import *

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

#kwa = dict(shared_params=False, scale_columns=False)
kwa = dict(shared_params=True, scale_columns=True)

tractor.setParams(np.array([1.] * tractor.numberOfParams()))
tractor.printThawedParams()

allderivs = tractor.getDerivs()
X1 = tractor.getUpdateDirection(allderivs, use_tsnnls=True, **kwa)
print 'tsnnls:', X1

X2 = tractor.getUpdateDirection(allderivs, **kwa)
print 'lsqr:', X2

kwa = dict(shared_params=False)

cat = tractor.getCatalog()

print

tractor.setCatalog(cat.copy())
for src in tractor.getCatalog():
    src.freezeParam('pos')
tractor.printThawedParams()

X2 = tractor.optimize_forced_photometry(use_tsnnls=False, **kwa)
print 'Got normal:' #, X2
for src in tractor.getCatalog():
    print src.getBrightness()


tractor.setCatalog(cat.copy())
for src in tractor.getCatalog():
    src.freezeParam('pos')

X1 = tractor.optimize_forced_photometry(use_tsnnls=True, **kwa)
print 'Got TSNNLS:' #, X1
for src in tractor.getCatalog():
    print src.getBrightness()






# allderivs = [
#     [],
#     [],
#     [ (Patch(6, 0, np.array([[22.],])), tim1) ],
#     [ (Patch(7, 0, np.array([[33.],])), tim1) ],
#     [ (Patch(6, 0, np.array([[44.],])), tim1) ],
# ]
# 
# chi = np.ones_like(img) * 10.
# chi = np.cumsum(chi)
# 
# X = tractor.getUpdateDirection(allderivs, use_tsnnls=True,
#                                chiImages=[chi], scale_columns=False,
#                                shared_params=False)
# print 'Got:', X




