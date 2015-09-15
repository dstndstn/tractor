import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
from astrometry.util.plotutils import *

if __name__ == '__main__':

    import fastexp

    print 'fastexp:', dir(fastexp)

    xx = np.linspace(-10., 0., 1000)

    e = np.exp(xx)
    fe = np.array([fastexp.my_fastexp(x) for x in xx])

    ps = PlotSequence('fastexp')
    
    plt.clf()
    plt.plot(xx, e, 'b-', alpha=0.25, lw=3)
    plt.plot(xx, fe, 'r-')
    ps.savefig()

    plt.yscale('log')
    ps.savefig()
    
    plt.clf()
    plt.axhline(0., color='b', alpha=0.25, lw=3)
    v = fe / e - 1.
    mx = max(np.abs(v))
    plt.plot(xx, v, 'r-')
    plt.ylim(-1.1*mx, 1.1*mx)
    ps.savefig()
    
