import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from tractor import *
from tractor.galaxy import *
from astrometry.util.util import Tan
from astrometry.util.plotutils import *

ps = PlotSequence('test-ps')

W,H = 100,100
pixscale = 0.4/3600.
rot = np.deg2rad(10.)

for cd in [
        (-pixscale*np.cos(rot), pixscale*np.sin(rot),
         pixscale*np.sin(rot),  pixscale*np.cos(rot),
         ),
         (-2.11357712641E-07,
          7.32269335496E-05 ,
          -7.32016721769E-05,
          -1.88067009846E-07),]:

    wcs = Tan(*[0., 0., W/2., H/2.] + list(cd) + [float(W), float(H)])

    ptsrc = PointSource(RaDecPos(0., 0.), Flux(100.))

    psf=GaussianMixturePSF(np.array([0.8, 0.2]),
                           np.zeros((2,2)),
                           np.array([[[6.,0.],[0.,6.]], [[18.,0.],[0.,18.]]]))

    tim = Image(data=np.zeros((H,W), np.float32), wcs=ConstantFitsWcs(wcs),
                psf=psf)
    tim.modelMinval = 1e-8

    ax = [0, W, 0, H]
    derivs = ptsrc.getParamDerivatives(tim, fastPosDerivs=False)
    print 'Derivs:', derivs
    rows,cols = 2,2
    plt.clf()
    for i,deriv in enumerate(derivs):
        plt.subplot(rows,cols,i+1)
        dimshow(deriv.patch, extent=deriv.getExtent())
        plt.axis(ax)
        plt.title('Orig ' + deriv.name)
        plt.colorbar()
    ps.savefig()

    derivs = ptsrc.getParamDerivatives(tim)
    print 'Derivs:', derivs
    plt.clf()
    for i,deriv in enumerate(derivs):
        plt.subplot(rows,cols,i+1)
        dimshow(deriv.patch, extent=deriv.getExtent())
        plt.axis(ax)
        plt.title('Fast ' + deriv.name)
        plt.colorbar()
    ps.savefig()


src = FixedCompositeGalaxy(RaDecPos(0., 0.), Flux(100.),
                             0.5, EllipseESoft(1., 0., 0.2),
                             EllipseESoft(1., 0.2, 0.))
derivs = src.getParamDerivatives(tim)
print 'Derivs:', derivs
cols = int(np.ceil(np.sqrt(len(derivs))))
rows = int(np.ceil(float(len(derivs)) / cols))
plt.clf()
for i,deriv in enumerate(derivs):
    plt.subplot(rows,cols,i+1)
    dimshow(deriv.patch, extent=deriv.getExtent())
    plt.axis(ax)
    plt.title(deriv.name, fontsize=8)
    #plt.colorbar()
    plt.xticks([]); plt.yticks([])    
ps.savefig()
    
