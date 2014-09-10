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


fsrc = FixedCompositeGalaxy(RaDecPos(0., 0.), Flux(100.), 0.25,
                            EllipseESoft(1., 0., 0.2),
                            EllipseESoft(1., 0., -0.2))

    #src = ExpGalaxy(RaDecPos(0., 0.), Flux(100.),
    #            EllipseESoft(1., 0., 0.2))
    #src = DevGalaxy(RaDecPos(0., 0.), Flux(100.),
    #            EllipseESoft(1., 0., 0.2))

derivs = fsrc.getParamDerivatives(tim)
e = ExpGalaxy(fsrc.pos, fsrc.brightness, fsrc.shapeExp)
d = DevGalaxy(fsrc.pos, fsrc.brightness, fsrc.shapeDev)
f = fsrc.fracDev.getClippedValue()
de = e.getParamDerivatives(tim)
dd = d.getParamDerivatives(tim)

for deriv in de:
    if deriv is not None:
        deriv *= (1.-f)
for deriv in dd:
    if deriv is not None:
        deriv *= f

plt.clf()
mx = np.max(np.abs(derivs[0].patch))
plt.subplot(2,3,1)
dimshow(derivs[0].patch, vmin=-mx, vmax=mx)
plt.title('FixedComp')
plt.subplot(2,3,4)
dimshow(de[0].patch, vmin=-mx, vmax=mx)
plt.title('exp')
plt.subplot(2,3,5)
dimshow(dd[0].patch, vmin=-mx, vmax=mx)
plt.title('deV')

plt.subplot(2,3,2)
#dimshow((dd[0] * f + de[0] * (1.-f)).patch, vmin=-mx, vmax=mx)
dimshow((dd[0] + de[0]).patch, vmin=-mx, vmax=mx)
plt.title('sum')

plt.subplot(2,3,3)
ss = fsrc.getStepSizes()
p0 = fsrc.getParams()
patch0 = fsrc.getModelPatch(tim)
i=0
s = ss[i]
oldval = fsrc.setParam(i, p0[i]+s)
patchx = fsrc.getModelPatch(tim)
fsrc.setParam(i, p0[i])
dp = (patchx - patch0) / s
dimshow(dp.patch, vmin=-mx, vmax=mx)
plt.title('step')

ps.savefig()

    
for src in [fsrc, e, d]:

    derivs = src.getParamDerivatives(tim)
    print 'Derivs:', derivs
    cols = int(np.ceil(np.sqrt(len(derivs))))
    rows = int(np.ceil(float(len(derivs)) / cols))
    plt.clf()
    maxes = []
    for i,deriv in enumerate(derivs):
        plt.subplot(rows,cols,i+1)
        mx = max(np.abs(deriv.patch.min()), deriv.patch.max())
        dimshow(deriv.patch, extent=deriv.getExtent(), vmin=-mx, vmax=mx)
        maxes.append(mx)
        plt.axis(ax)
        plt.title(deriv.name, fontsize=8)
        #plt.colorbar()
        plt.xticks([]); plt.yticks([])    
    plt.suptitle('getParamDerivatives')
    ps.savefig()
    
    patch0 = src.getModelPatch(tim)
    print 'Patch sum:', patch0.patch.sum()    

    p0 = src.getParams()
    ss = src.getStepSizes()
    names = src.getParamNames()
    plt.clf()
    for i,(s,name) in enumerate(zip(ss, names)):
        plt.subplot(rows,cols,i+1)
    
        oldval = src.setParam(i, p0[i]+s)
        patchx = src.getModelPatch(tim)
        src.setParam(i, p0[i])
    
        dp = (patchx - patch0) / s
        
        dimshow(dp.patch, extent=dp.getExtent(), vmin=-maxes[i], vmax=maxes[i])
        plt.axis(ax)
        plt.title(name, fontsize=8)
        #plt.colorbar()
        plt.xticks([]); plt.yticks([])    
    plt.suptitle('Stepping parameters')
    ps.savefig()

