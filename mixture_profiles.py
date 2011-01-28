# make mixture-of-Gaussian galaxy profiles

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

def hogg_exp(x):
    y = np.zeros_like(x)
    I = x > -20.0
    y[I] = np.exp(x[I])
    return y

# your documentation IS the code
def hogg_lsqr_mix(x_k_list, x, y):
    A = np.zeros((len(x),K))
    for k in range(K):
        # use numerically stable (?) hogg exps:
        A[:,k] = x * hogg_exp(-0.5 * x * x / x_k_list[k]**2)
    ATA = np.dot(np.transpose(A), A)
    ATb = np.dot(np.transpose(A), y)
    ATAinv = np.linalg.inv(ATA)
    X = np.dot(ATAinv, ATb)
    for k in range(K):
        # switch back to REAL (not hogg) exps:
        A[:,k] = x * np.exp(-0.5 * x * x / x_k_list[k]**2)
    return (X, np.dot(A, X))

# this code is a HACK because it FIXES the x_k
def test_exp(K):
    # MAGIC numbers 20 and 0.001
    x = np.arange(0.,20.,0.001)
    yexp = x * np.exp(-x)
    # MAGIC numbers 5 and 2:
    x_k_list = 5. * ((np.arange(K) + 0.5) / K)**2
    A_k_list, ymix = hogg_lsqr_mix(x_k_list, x, yexp)
    plt.clf()
    plt.plot(x,yexp,'k-')
    plt.plot(x,ymix,'r-')
    plt.xlabel('$r$ with $r_e = 1$')
    plt.xlim(0., 10.)
    plt.ylabel('$r\,\exp(-r)$')
    plt.ylim(-0.1 * max(yexp), 1.1 * max(yexp))
    plt.title('black: truth / red: %d-Gaussian approximation' % K)
    plt.savefig('test_exp_%02d.png' % K)
    return

# this code is a HACK because it FIXES the x_k
def test_dev(K):
    # MAGIC numbers 20 and 0.001
    x = (np.arange(0.,20.,0.001))**4
    ydev = np.exp(-(x**0.25))
    # MAGIC numbers 5 and 2:
    x_k_list = (5. * ((np.arange(K) + 0.5) / K)**2)**4
    A_k_list, ymix = hogg_lsqr_mix(x_k_list, x, ydev)
    plt.clf()
    plt.plot(x,ydev,'k-')
    plt.plot(x,ymix,'r-')
    plt.xlabel('$r$ with $r_e = 1$')
    plt.xlim(0., 1000.)
    plt.ylabel('$r\,\exp(-r^{1/4})$')
    plt.ylim(-0.1 * max(ydev), 1.1 * max(ydev))
    plt.title('black: truth / red: %d-Gaussian approximation' % K)
    plt.savefig('test_dev_%02d.png' % K)
    plt.xlim(0., 10.)
    plt.savefig('test_dev_zoom_%02d.png' % K)
    return

if __name__ == '__main__':
    for K in range(5,20):
        test_exp(K)
        test_dev(K)
