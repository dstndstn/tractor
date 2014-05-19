
import numpy as np

from tractor import *

H,W = 10,10
img = np.zeros((H,W), np.float32)
sig1 = 0.1

tim = Image(data=img, invvar=np.zeros_like(img) + 1./sig1**2,
            psf=NCircularGaussianPSF([1.], [1.]),
            wcs=NullWCS(), photocal=LinearPhotoCal(1.),
            sky=ConstantSky(0.),
            name='Test', domask=False)

src = PointSource(PixPos(W/2, H/2), Flux(100.))

tractor = Tractor([tim], [src])
mod = tractor.getModelImage(0)

tim.data = mod + np.random.normal(scale=sig1, size=mod.shape)
src.brightness = Flux(10.)
src.pos = PixPos(W/2 - 1, H/2 - 1)

print 'Thawed param:'
tractor.printThawedParams()
print 'Params:', tractor.getParams()
lnp0 = tractor.getLogProb()
print 'Logprob:', lnp0

print 'Testing _getOneImageDerivs...'
tractor._getOneImageDerivs(0)

print 'Calling _ceres_opt...'
tractor._ceres_opt()

print '_ceres_opt finished'
print 'Params:', tractor.getParams()
lnp1 = tractor.getLogProb()
print 'Logprob:', lnp1
