import sys
import numpy as np
import logging
lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

from tractor import *

img = np.zeros((1,10))
tim1 = Image(data=img, invvar=np.ones_like(img),
             psf=NCircularGaussianPSF([1.],[1.]),
             wcs=NullWCS(),
             photocal=NullPhotoCal(),
             sky=ConstantSky(0.))

srcs = [ PointSource(PixPos(x,0), Flux(1.)) for x in range(5) ]
for src in srcs:
    src.freezeParam('pos')

tractor = Tractor([tim1], srcs)
tractor.freezeParam('images')

allderivs = [
    [],
    [],
    [ (Patch(6, 0, np.array([[22.],])), tim1) ],
    [ (Patch(7, 0, np.array([[33.],])), tim1) ],
    [ (Patch(6, 0, np.array([[44.],])), tim1) ],
]

chi = np.ones_like(img) * 10.
chi = np.cumsum(chi)

X = tractor.getUpdateDirection(allderivs, use_tsnnls=True,
                               chiImages=[chi], scale_columns=False,
                               shared_params=False)
print 'Got:', X




