
from tractor import *

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
