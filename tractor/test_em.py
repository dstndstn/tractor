from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from tractor.emfit import em_fit_1d_samples
import numpy as np
from tractor.fitpsf import em_init_params

mus   = np.array([1.77, 3.7] )
stds = np.array([3.9, 1.]   )
amps  = np.array([ 0.8, 0.2 ])

N = 1000

#X = np.array(N)

a = np.random.uniform(size=N)
assert(len(amps) == 2)
z = (a > amps[0]) * 1
print('z', np.unique(z))
print('n = 0:', np.sum(z == 0))
print('n = 1:', np.sum(z == 1))
X = np.random.normal(size=N) * stds[z] + mus[z]

K = 2
ww = np.ones(K)
mm = np.zeros(K)
vv = np.array([1., 2.])

for i in range(3):
    r = em_fit_1d_samples(X, ww, mm, vv)
    print('result:', r)
    print('fit / true:')
    print('A', ww, amps)
    print('mu', mm, mus)
    print('std', np.sqrt(vv), stds)

plt.clf()
n,b,p = plt.hist(X, 50, histtype='step', color='b')
B = (b[1]-b[0])
lo,hi = plt.xlim()
xx = np.linspace(lo, hi, 500)
gtrue = [a * N*B / (np.sqrt(2.*np.pi) * s) * 
         np.exp(-0.5 * (xx-m)**2/s**2)
         for (a,m,s) in zip(amps, mus, stds)]
plt.plot(xx, gtrue[0]+gtrue[1], 'b-', lw=2, alpha=0.5)
plt.plot(xx, gtrue[0], 'b-', lw=2, alpha=0.5)
plt.plot(xx, gtrue[1], 'b-', lw=2, alpha=0.5)

gfit = [a * N*B / (np.sqrt(2.*np.pi) * s) * 
        np.exp(-0.5 * (xx-m)**2/s**2)
        for (a,m,s) in zip(ww, mm, np.sqrt(vv))]
plt.plot(xx, gfit[0]+gfit[1], 'r-', lw=2, alpha=0.5)
plt.plot(xx, gfit[0], 'r-', lw=2, alpha=0.5)
plt.plot(xx, gfit[1], 'r-', lw=2, alpha=0.5)

plt.savefig('testem.png')
