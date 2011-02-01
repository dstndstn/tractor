# make mixture-of-Gaussian galaxy profiles

import matplotlib
matplotlib.use('Agg')
from math import pi as pi
import pylab as plt
import numpy as np
import scipy.optimize as op

# magic number
maxradius = 8.

# note wacky normalization because this is for 2-d Gaussians
# (but only ever called in 1-d).  Wacky!
def not_normal(x, m, V):
    return 1. / (2. * pi * V) * np.exp(-0.5 * x**2 / V)

def mixture_of_not_normals(x, pp):
    K = len(pp)/2
    y = 0.
    for k in range(K):
        y += pp[k] * not_normal(x, 0., pp[k+K])
    return y

def badness_of_fit_exp(pars):
    pp = np.exp(pars)
    x = np.arange(0., maxradius, 0.01)
    return np.sum((x * mixture_of_not_normals(x, pp)
                   - x * np.exp(-x))**2)/len(x)

def optimize_mixture(K):
    # first guess
    amps = np.log(np.random.uniform(size=(K)))
    vars = np.log((8.0*np.random.uniform(size=(K)))**2)
    pars = np.append(amps,vars)
    # optimize
    func = badness_of_fit_exp
    print func(pars)
    print np.exp(pars)
    newpars = op.fmin_cg(func, pars)
    print func(newpars)
    print np.exp(newpars)
    return (func(newpars), np.exp(newpars))

def plot_mixture(pars, fn):
    x1 = np.arange(0., maxradius, 0.001)
    y1 = np.exp(-x1)
    x2 = np.arange(0., maxradius+2., 0.001)
    y2 = mixture_of_not_normals(x2, pars)
    plt.clf()
    plt.plot(x1, y1, 'k-')
    plt.plot(x2, y2, 'k-', lw=4, alpha=0.5)
    plt.xlim(np.min(x2), np.max(x2))
    plt.ylim(-0.1*np.max(y1), 1.1*np.max(y1))
    plt.title('K = %d / badness = %e' %
              (len(pars)/2, badness_of_fit_exp(np.log(pars))))
    plt.savefig(fn)

if __name__ == '__main__':
    large = 100.
    ntry = 10
    for K in range(3,11):
        badness = large
        for t in range(ntry):
            (b, p) = optimize_mixture(K)
            if b < badness:
                pars = p
                badness = b
        indx = np.argsort(pars[K:K+K])
        indx = np.append(indx,K+indx)
        pars = pars[indx]
        print '----- best at %i is %f' % (K, b)
        print pars
        plot_mixture(pars, 'K%02d.png' % K)
