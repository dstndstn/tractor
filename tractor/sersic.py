'''
This file is part of the Tractor project.
Copyright 2014, Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`sersic.py`
===========

General Sersic galaxy model.
'''
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline

import mixture_profiles as mp
from engine import *
from utils import *
from cache import *
from sdss_galaxy import *

class SersicMixture(object):
    singleton = None
    @staticmethod
    def getProfile(sindex):
        if SersicMixture.singleton is None:
            SersicMixture.singleton = SersicMixture()
        return SersicMixture.singleton._getProfile(sindex)

    
    def __init__(self):

        # python -c "from astrometry.util.file import *; p=unpickle_from_file('ser2_K08_MR08.pickle'); print repr(p[:8]); print repr(p[8:])"

        # Have I mentioned lately how much I love my job?

        self.fits = [
        # exp
        (1.,
         np.array([  7.73835603e-04,   1.01672452e-02,   7.31297606e-02,
                     3.71875005e-01,   1.39727069e+00,   3.56054423e+00,
                     4.74340409e+00,   1.78731853e+00]),
         np.array([  5.72481639e-04,   4.21236311e-03,   1.84425003e-02,
                     6.29785639e-02,   1.84402973e-01,   4.85424877e-01,
                     1.18547337e+00,   2.79872887e+00]),
                     ),
        # ser2
        (2.,
         np.array([  2.83066070e-03,   2.98109751e-02,   1.70462302e-01,
                     6.72109095e-01,   1.94637497e+00,   4.07818245e+00,
                     5.58981857e+00,   3.64571339e+00]),
         np.array([  6.41326241e-05,   6.98618884e-04,   4.28218364e-03,
                     2.02745634e-02,   8.36658982e-02,   3.24006007e-01,
                     1.26549998e+00,   5.68924078e+00]),
                     ),
        # ser3
        (3.,
         np.array([  2.97478081e-03,   2.98325539e-02,   1.62926966e-01,
                     6.21897569e-01,   1.79221947e+00,   3.93826776e+00,
                     6.27309371e+00,   6.02826557e+00]),
         np.array([  8.02949133e-06,   1.17776376e-04,   9.29524545e-04,
                     5.60991573e-03,   2.96692431e-02,   1.50068210e-01,
                     7.96528251e-01,   5.36403456e+00]),
                     ),
        # dev
        (4.,
         np.array([  2.62202676e-03,   2.50014044e-02,   1.34130119e-01,
                     5.13259912e-01,   1.52004848e+00,   3.56204592e+00,
                     6.44844889e+00,   8.10104944e+00]),
         np.array([  1.26864655e-06,   2.25833632e-05,   2.13622743e-04,
                     1.54481548e-03,   9.85336661e-03,   6.10053309e-02,
                     4.08099539e-01,   3.70794983e+00]),
                     ),
        # ser5
        (5.,
         np.array([  5.30069413e-03,   3.33623146e-02,   1.49418074e-01,
                     5.16448916e-01,   1.45115226e+00,   3.36990018e+00,
                     6.37772131e+00,   9.24101590e+00]),
         np.array([  7.41574279e-07,   1.05154771e-05,   9.35192405e-05,
                     6.88777943e-04,   4.67219862e-03,   3.17741406e-02,
                     2.41556167e-01,   2.63694124e+00]),
                     ),
        ]

        N = None
        for index, amps, vars in self.fits:
            if N is None:
                N = len(amps)
            assert(len(amps) == N)
            assert(len(vars) == N)
        inds = [index for index,amps,vars in self.fits]
        # bbox=[0.5, 5.5], k=3
        self.amps = [
            InterpolatedUnivariateSpline(
                inds, [amps[i] for index,amps,vars in self.fits])
            for i in range(N)]
        self.vars = [
            InterpolatedUnivariateSpline(
                inds, [vars[i] for index,amps,vars in self.fits])
            for i in range(N)]
            
    def _getProfile(self, sindex):
        amps = np.array([f(sindex) for f in self.amps])
        vars = np.array([f(sindex) for f in self.vars])
        amps /= amps.sum()
        return mp.MixtureOfGaussians(amps, np.zeros((len(amps),2)), vars)

class SersicIndex(ScalarParam):
    stepsize = 0.01

class SersicGalaxy(HoggGalaxy):
    nre = 8.

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2, sersicindex=3)

    def __init__(self, pos, brightness, shape, sersicindex, **kwargs):
        #super(SersicMultiParams.__init__(self, pos, brightness, shape, sersicindex)
        #self.name = self.getName()
        self.nre = SersicGalaxy.nre
        super(SersicGalaxy, self).__init__(pos, brightness, shape, sersicindex)
        #**kwargs)
        #self.sersicindex = sersicindex

    def __str__(self):
        return (super(SersicGalaxy, self).__str__() +
                ', Sersic index %.2f' % self.sersicindex.val)
        
    def getName(self):
        return 'SersicGalaxy'

    def getProfile(self):
        return SersicMixture.getProfile(self.sersicindex.val)

    def copy(self):
        return SersicGalaxy(self.pos.copy(), self.brightness.copy(),
                            self.shape.copy(), self.sersicindex.copy())
    
    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(), px, py,
                     img.getWcs().hashkey(),
                     img.getPsf().hashkey(),
                     self.shape.hashkey(),
                     self.sersicindex.hashkey()))

    def getParamDerivatives(self, img):
        # superclass produces derivatives wrt pos, brightness, and shape.
        derivs = super(SersicGalaxy, self).getParamDerivatives(img)

        pos0 = self.getPosition()
        (px0,py0) = img.getWcs().positionToPixel(pos0, self)
        patch0 = self.getUnitFluxModelPatch(img, px0, py0)
        if patch0 is None:
            derivs.append(None)
            return derivs
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)

        # derivatives wrt Sersic index
        isteps = self.sersicindex.getStepSizes()
        if not self.isParamFrozen('sersicindex'):
            inames = self.sersicindex.getParamNames()
            oldvals = self.sersicindex.getParams()
            for i,istep in enumerate(isteps):
                oldval = self.sersicindex.setParam(i, oldvals[i]+istep)
                patchx = self.getUnitFluxModelPatch(img, px0, py0)
                self.sersicindex.setParam(i, oldval)
                if patchx is None:
                    print 'patchx is None:'
                    print '  ', self
                    print '  stepping galaxy sersicindex', self.sersicindex.getParamNames()[i]
                    print '  stepped', isteps[i]
                    print '  to', self.sersicindex.getParams()[i]
                    derivs.append(None)

                dx = (patchx - patch0) * (counts / istep)
                dx.setName('d(%s)/d(%s)' % (self.dname, inames[i]))
                derivs.append(dx)
        return derivs


if __name__ == '__main__':
    from basics import *
    from ellipses import *
    
    s = SersicGalaxy(PixPos(100., 100.),
                     Flux(1000.),
                     EllipseE(2., 0.5, 0.5),
                     SersicIndex(2.5))
    print s
    print s.getProfile()

    s.sersicindex.setValue(4.0)
    print s.getProfile()

    d = DevGalaxy(s.pos, s.brightness, s.shape)
    print d
    print d.getProfile()
    
    # Extrapolation!
    # s.sersicindex.setValue(0.5)
    # print s.getProfile()
    

    from astrometry.util.plotutils import PlotSequence
    import pylab as plt
    ps = PlotSequence('ser')
    
    # example PSF (from WISE W1 fit)
    w = np.array([ 0.77953706,  0.16022146,  0.06024237])
    mu = np.array([[-0.01826623, -0.01823262],
                   [-0.21878855, -0.0432496 ],
                   [-0.83365747, -0.13039277]])
    sigma = np.array([[[  7.72925584e-01,   5.23305564e-02],
                       [  5.23305564e-02,   8.89078473e-01]],
                       [[  9.84585869e+00,   7.79378820e-01],
                       [  7.79378820e-01,   8.84764455e+00]],
                       [[  2.02664489e+02,  -8.16667434e-01],
                        [ -8.16667434e-01,   1.87881670e+02]]])
    
    psf = GaussianMixturePSF(w, mu, sigma)
    
    data = np.zeros((200, 200))
    invvar = np.zeros_like(data)
    tim = Image(data=data, invvar=invvar, psf=psf,
                wcs=NullWCS(), sky=ConstantSky(0.),
                photocal=NullPhotoCal())

    tractor = Tractor([tim], [s])

    nn = np.linspace(0.5, 5.5, 12)
    cols = int(np.ceil(np.sqrt(len(nn))))
    rows = int(np.ceil(len(nn) / float(cols)))
    plt.clf()
    for i,n in enumerate(nn):
        s.sersicindex.setValue(n)
        mod = tractor.getModelImage(0)

        plt.subplot(rows, cols, i+1)
        plt.imshow(np.log10(np.maximum(1e-16, mod)), interpolation='nearest',
                   origin='lower')
        plt.title('index %.2f' % n)
    ps.savefig()
