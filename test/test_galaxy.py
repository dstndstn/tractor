import numpy as np

from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

pos = RaDecPos(100., 10.)
bright = NanoMaggies(r=42., g=17.)
shape = EllipseESoft(-1, 0., 0.1)

gal = DevGalaxy(pos, bright, shape)
p0 = gal.getParams()

gal2 = gal.copy()

gal.freezeParam('brightness')
pf = gal.getParams()

p2 = gal2.getParams()

assert(np.all(p0 == p2))
assert(len(p0) == 7)
assert(len(pf) == 5)

gal.getPosition().setParams([101., 11.])
gal.getBrightness().setFlux('r', 19.)
gal.getShape().setParams([-3, 0.2, 0.3])
p1 = gal.getParams()

# setting "gal" params has no effect on "gal2"
assert(np.all(p2 == gal2.getParams()))

print gal
print p1

print gal2
print p2

