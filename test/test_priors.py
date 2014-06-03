
from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

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
print p

print p.getLogPriorDerivatives()
print p.getLogPrior()

p.freezeParam('a')
print p.getLogPriorDerivatives()
print p.getLogPrior()

p.freezeParam('b')
print p.getLogPriorDerivatives()
print p.getLogPrior()

p.freezeParam('c')
print p.getLogPriorDerivatives()
print p.getLogPrior()

p.thawParam('a')
p.thawParam('b')
print p.getLogPriorDerivatives()
print p.getLogPrior()

tractor = Tractor([], [p])
dlnp,X,alpha = tractor.optimize()
print p
print 'dlnp', dlnp
print 'X', X
print 'alpha', alpha
print p.getLogPrior()

p.thawParam('c')
dlnp,X,alpha = tractor.optimize()
print p
print 'dlnp', dlnp
print 'X', X
print 'alpha', alpha
print p.getLogPrior()


class GaussianPriorsMixin(object):
    def __init__(self, *args, **kwargs):
        super(GaussianPriorsMixin, self).__init__(*args, **kwargs)
        self.gpriors = GaussianPriors(self)

    def addGaussianPrior(self, name, mu, sigma):
        self.gpriors.add(name, mu, sigma)

    def getLogPriorDerivatives(self):
        '''
        You might want to override like this:

        X = self.getGaussianLogPriorDerivatives()
        Y = << other log prior derivatives >>
        return [x+y for x,y in zip(X,Y)]
        '''
        return self.getGaussianLogPriorDerivatives()

    def getGaussianLogPriorDerivatives(self):
        return self.gpriors.getDerivs()

    def getLogPrior(self):
        return self.getGaussianLogPrior()

    def getGaussianLogPrior(self):
        return self.gpriors.getLogPrior()

class MyEESoft(GaussianPriorsMixin, EllipseESoft):
    pass

#g = MyExp(PixPos(4., 7.), Flux(100.), EllipseESoft(0., 0.5, 0.5))
g = ExpGalaxy(PixPos(4., 7.), Flux(100.), MyEESoft(0., 0.5, 0.5))

g.shape.addGaussianPrior('e1', 0., 0.25)
g.shape.addGaussianPrior('e2', 0., 0.25)

print 'G:', g
print 'G prior:', g.getLogPrior()
print 'G prior derivs:', g.getLogPriorDerivatives()


tractor = Tractor([], [g])
dlnp,X,alpha = tractor.optimize()
print g
print 'dlnp', dlnp
print 'X', X
print 'alpha', alpha
print g
print 'Log prior:', g.getLogPrior()

