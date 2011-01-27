# make mixture-of-Gaussian galaxy profiles

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

def test_exp():
    plt.clf()
    x = np.arange(0.,10.,0.001)
    yexp = np.exp(-x)
    plt.plot(x,np.log(yexp),'r-')
    K = 50
    Delta = 0.1
    ymix = np.zeros_like(yexp)
    x_k_list = (np.arange(K) + 1.) * Delta
    A_k_list = np.exp(-x_k_list) * Delta
    for k in np.arange(K):
        ymix += A_k_list[k] * np.exp(-0.5 * x * x / x_k_list[k]**2)
    plt.plot(x,np.log(ymix),'k-')
    plt.savefig('test_exp.png')
    return

if __name__ == '__main__':
    test_exp()
