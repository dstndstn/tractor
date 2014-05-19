if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

from tractor import *

from astrometry.util.plotutils import *

ps = PlotSequence('ctest')

class MyTractor(Tractor):
    def setParams(self, p):
        print 'MyTractor.setParams', p
        super(MyTractor, self).setParams(p)

        tim = self.getImage(0)
        data = tim.getImage()
        mod = self.getModelImage(0)
        mx = max(data.max(), mod.max())
        mn = min(data.min(), mod.min())
        ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(mod, **ima)
        plt.subplot(2,2,2)
        plt.imshow(data, **ima)
        plt.subplot(2,2,4)
        plt.imshow(-(data - mod)*tim.getInvError(), interpolation='nearest',
                   origin='lower', vmin=-5, vmax=5, cmap='RdBu')
        ps.savefig()



H,W = 10,10
img = np.zeros((H,W), np.float32)
sig1 = 0.1

tim = Image(data=img, invvar=np.zeros_like(img) + 1./sig1**2,
            psf=NCircularGaussianPSF([1.], [1.]),
            wcs=NullWCS(), photocal=LinearPhotoCal(1.),
            sky=ConstantSky(0.),
            name='Test', domask=False)

src = PointSource(PixPos(W/2, H/2), Flux(100.))

tractor = MyTractor([tim], [src])
mod = tractor.getModelImage(0)

tim.data = mod + np.random.normal(scale=sig1, size=mod.shape)
src.brightness = Flux(10.)
src.pos = PixPos(W/2 - 1, H/2 - 1)

print 'Thawed param:'
tractor.printThawedParams()
print 'Params:', tractor.getParams()
lnp0 = tractor.getLogProb()
print 'Logprob:', lnp0

#print 'Testing _getOneImageDerivs...'
#tractor._getOneImageDerivs(0)

print 'Calling _ceres_opt...'
tractor._ceres_opt()

print '_ceres_opt finished'
print 'Params:', tractor.getParams()
lnp1 = tractor.getLogProb()
print 'Logprob:', lnp1
