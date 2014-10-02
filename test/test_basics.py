import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.plotutils import *

from tractor import GaussianMixturePSF, GaussianMixtureEllipsePSF
from tractor.ellipses import *
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

e = GaussianMixtureEllipsePSF([0.7, 0.3], np.array([[0.1,0.],[0.,0.2]]),
                              [EllipseESoft(0., 0., 0.),
                               EllipseESoft(0.1, 0.1, -0.1)])
print e
p = e.getParams()
print 'params', p
e.setParams(p)
print e

n1,n2 = 7,7
E1list = np.linspace(-1.2, 1.2, n1)
E2list = np.linspace(-1.2, 1.2, n2)
E1,E2 = np.meshgrid(E1list, E2list)

angle = np.linspace(0., 2.*np.pi, 100)
xx,yy = np.sin(angle), np.cos(angle)
#xy = np.vstack((xx,yy)) * 3600.
xy = np.vstack((xx,yy)) * 3600 * 0.01

ps = PlotSequence('gell')

for logre,cc in zip([1., 2., 3.], 'rgb'):
    plt.clf()
    for e1,e2 in zip(E1.ravel(), E2.ravel()):
        e = EllipseESoft(logre, e1, e2)
        print e
        T = e.getRaDecBasis()
        txy = np.dot(T, xy)
        plt.plot(e1 + txy[0,:], e2 + txy[1,:], '-', color=cc, alpha=0.5)
    plt.xlabel('ee1')
    plt.ylabel('ee2')
    plt.axis('scaled')
    plt.title('EllipseESoft')
    ps.savefig()

    plt.clf()
    rows = []
    for e2 in E2list:
        row = []
        for e1 in E1list:
            e = EllipseESoft(logre, e1, e2)
            psf = GaussianMixtureEllipsePSF([1.], [0.,0.], [e])
            patch = psf.getPointSourcePatch(0., 0., extent=[-10,10,-10,10])
            patch = patch.patch
            patch /= patch.max()
            patch = np.log10(patch)
            row.append(patch)
        row = np.hstack(row)
        rows.append(row)
    rows = np.vstack(rows)
    dimshow(rows, vmin=-3, vmax=0, extent=[-0.5,n1-0.5, -0.5, n2-0.5],
            cmap='jet')
    #extent=[-n1/2., n1/2., -n2/2., n2/2.])
    cc = 'k'
    ax = plt.axis()
    for y,e2 in enumerate(E2list):
        for x,e1 in enumerate(E1list):
            e = EllipseESoft(logre, e1, e2)
            T = e.getRaDecBasis()
            txy = np.dot(T, xy)
            S = 5
            plt.plot(x + S*txy[0,:], y + S*txy[1,:], '-', color=cc, alpha=0.5)
    plt.axis(ax)
    ps.savefig()
    
