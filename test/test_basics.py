from tractor import GaussianMixturePSF
import numpy as np

var = np.zeros((2,2,2))
var[0,0,0] = var[0,1,1] = 2.
var[1,0,0] = var[1,1,1] = 4.
g = GaussianMixturePSF([0.7, 0.3], np.zeros((2,2)), var)
c = g.copy()

print g
print c

print g.getParams()
print c.getParams()

g.setParam(0, 999.)
g.setParam(2, 9999.)
g.setParam(6, 99999.)
print g
print c


print g.hashkey()

