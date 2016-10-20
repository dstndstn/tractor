from __future__ import print_function

from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

if False:
    class MyParam(ParamList):
    
        def __init__(self, *args, **kwargs):
            super(MyParam, self).__init__(*args, **kwargs)
            self.priors = GaussianPriors(self)
            self.priors.add('a', 1., 0.1)
            self.priors.add('c', 40., 10.)
            self.priors.add('b', 0., 1.)
    
        @staticmethod
        def getNamedParams():
            return dict(a=0, b=1, c=2)
    
        def getLogPriorDerivatives(self):
            return self.priors.getDerivs()
    
        def getLogPrior(self):
            return self.priors.getLogPrior()
    
    p = MyParam(1.3, 4, 70)
    print(p)
    
    print(p.getLogPriorDerivatives())
    print(p.getLogPrior())
    
    p.freezeParam('a')
    print(p.getLogPriorDerivatives())
    print(p.getLogPrior())
    
    p.freezeParam('b')
    print(p.getLogPriorDerivatives())
    print(p.getLogPrior())
    
    p.freezeParam('c')
    print(p.getLogPriorDerivatives())
    print(p.getLogPrior())
    
    p.thawParam('a')
    p.thawParam('b')
    print(p.getLogPriorDerivatives())
    print(p.getLogPrior())
    
    tractor = Tractor([], [p])
    dlnp,X,alpha = tractor.optimize()
    print(p)
    print('dlnp', dlnp)
    print('X', X)
    print('alpha', alpha)
    print(p.getLogPrior())
    
    p.thawParam('c')
    dlnp,X,alpha = tractor.optimize()
    print(p)
    print('dlnp', dlnp)
    print('X', X)
    print('alpha', alpha)
    print(p.getLogPrior())

g = ExpGalaxy(PixPos(4., 7.), Flux(100.), EllipseESoft(0., 0.5, 0.5))
g.shape.addGaussianPrior('e1', 0., 0.25)
g.shape.addGaussianPrior('e2', 0., 0.25)

print('priors:', g.shape.gpriors)
print('G:', g)
print('G prior:', g.getLogPrior())
print('G prior derivs:', g.getLogPriorDerivatives())

tractor = Tractor([], [g])
dlnp,X,alpha = tractor.optimize()
print(g)
print('dlnp', dlnp)
print('X', X)
print('alpha', alpha)
print(g)
print('Log prior:', g.getLogPrior())

print('Step sizes:', zip(g.getParamNames(), g.getStepSizes()))
g.shape.freezeParam('e1')
g.stepsizes = [1e-3]*3
print('Step sizes:', zip(g.getParamNames(), g.getStepSizes()))

rd = RaDecPos(13.5, 88.0)
print('Step sizes:', zip(rd.getParamNames(), rd.getStepSizes()))

psf = NCircularGaussianPSF([0.5, 0.5], [1., 2.])
print('PSF step sizes:', zip(psf.getParamNames(), psf.getStepSizes()))
psf.sigmas.stepsizes = [1e-3]*2
print('PSF step sizes:', zip(psf.getParamNames(), psf.getStepSizes()))
psf.freezeParam('sigmas')
print('PSF step sizes:', zip(psf.getParamNames(), psf.getStepSizes()))
psf.freezeParam('weights')
print('PSF step sizes:', zip(psf.getParamNames(), psf.getStepSizes()))
psf.thawAllParams()
print('PSF step sizes:', zip(psf.getParamNames(), psf.getStepSizes()))
