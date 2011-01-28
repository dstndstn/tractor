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

# this code is a HACK because it FIXES the x_k
def test_exp(K):
    x = np.arange(0.,20.,0.001)
    yexp = x * np.exp(-x)
    # MAGIC numbers 5 and 2:
    x_k_list = 5. * ((np.arange(K) + 0.5) / K)**2
    A = np.zeros((len(x),K))
    for k in range(K):
        A[:,k] = x * hogg_exp(-0.5 * x * x / x_k_list[k]**2)
    ATA = np.dot(np.transpose(A), A)
    ATb = np.dot(np.transpose(A), yexp)
    ATAinv = np.linalg.inv(ATA)
    A_k_list = np.dot(ATAinv, ATb)
    print x_k_list
    print A_k_list
    plt.clf()
    plt.plot(x,yexp,'k-')
    for k in range(K):
        A[:,k] = x * np.exp(-0.5 * x * x / x_k_list[k]**2)
    ymix = np.dot(A, A_k_list)
    print ymix.size
    plt.plot(x,ymix,'r-')
    plt.xlabel('$r$ with $r_e = 1$')
    plt.xlim(0., 10.)
    plt.ylabel('$r\,\exp(-r)$')
    plt.ylim(-0.1 * max(yexp), 1.1 * max(yexp))
    plt.title('black: truth / red: %d-Gaussian approximation' % K)
    plt.savefig('test_exp_%d.png' % K)
    return

if __name__ == '__main__':
    for K in range(3,20):
        test_exp(K)
