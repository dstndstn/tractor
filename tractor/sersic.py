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
        # (0.5,
        # np.array([  0.00000000e+000,   0.00000000e+000,   5.97360116e+198,
        #          7.26746001e-037,   2.50004003e-119,   9.77713758e-002,
        #          3.76242606e+000,   5.20454258e+000]),
        # np.array([  4.39317232e-43,   5.60638305e-26,   2.08879632e-25,
        #          1.26626995e-17,   1.58106523e-17,   7.01549406e-01,
        #          7.21125242e-01,   7.21890993e-01]),
        # ),
        # (0.55,
        # np.array([  1.59481046e+307,   3.15487712e+307,   2.45652327e-004,
        #          4.36909452e-003,   4.71731489e-002,   4.70269591e-001,
        #          3.71062814e+000,   5.15450190e+000]),
        # np.array([  3.52830040e-68,   2.06589509e-25,   6.35140546e-03,
        #          3.39547886e-02,   1.16386327e-01,   3.11207647e-01,
        #          6.23025538e-01,   8.88126344e-01]),
        # ),
        (0.6,
        np.array([  2.35059121e-05,   4.13721322e-04,   3.92293893e-03,
                 2.85625019e-02,   1.89838613e-01,   1.20615614e+00,
                 4.74797981e+00,   3.52402557e+00]),
        np.array([  9.56466036e-04,   5.63033141e-03,   2.09789252e-02,
                 6.26359534e-02,   1.62128157e-01,   3.69124775e-01,
                 6.99199094e-01,   1.06945187e+00]),
        ),
        (0.65,
        np.array([  6.33289982e-05,   9.92144846e-04,   8.80546187e-03,
                 6.04526939e-02,   3.64161094e-01,   1.84433400e+00,
                 5.01041449e+00,   2.71713117e+00]),
        np.array([  1.02431077e-03,   6.00267283e-03,   2.24606615e-02,
                 6.75504786e-02,   1.75591563e-01,   3.99764693e-01,
                 7.73156172e-01,   1.26419221e+00]),
        ),
        (0.7,
        np.array([  1.39910412e-04,   2.11974313e-03,   1.77871639e-02,
                 1.13073467e-01,   5.99838314e-01,   2.43606518e+00,
                 4.97726634e+00,   2.15764611e+00]),
        np.array([  1.07167590e-03,   6.54686686e-03,   2.48658528e-02,
                 7.49393553e-02,   1.93700754e-01,   4.38556714e-01,
                 8.61967334e-01,   1.48450726e+00]),
        ),
        (0.8,
        np.array([  3.11928667e-04,   4.47378538e-03,   3.54873170e-02,
                 2.07033725e-01,   9.45282820e-01,   3.03897766e+00,
                 4.83305346e+00,   1.81226322e+00]),
        np.array([  8.90900573e-04,   5.83282884e-03,   2.33187424e-02,
                 7.33352158e-02,   1.97225551e-01,   4.68406904e-01,
                 9.93007283e-01,   1.91959493e+00]),
        ),
        (0.9,
        np.array([  5.26094326e-04,   7.19992667e-03,   5.42573298e-02,
                    2.93808638e-01,   1.20034838e+00,   3.35614909e+00,
                    4.75813890e+00,   1.75240066e+00]),
        np.array([  7.14984597e-04,   4.97740520e-03,   2.08638701e-02,
                 6.84402817e-02,   1.92119676e-01,   4.80831073e-01,
                 1.09767934e+00,   2.35783460e+00]),
        ),
        # exp
        (1.,
         np.array([  7.73835603e-04,   1.01672452e-02,   7.31297606e-02,
                     3.71875005e-01,   1.39727069e+00,   3.56054423e+00,
                     4.74340409e+00,   1.78731853e+00]),
         np.array([  5.72481639e-04,   4.21236311e-03,   1.84425003e-02,
                     6.29785639e-02,   1.84402973e-01,   4.85424877e-01,
                     1.18547337e+00,   2.79872887e+00]),
                     ),
        (1.25,
         np.array([  1.43424042e-03,   1.73362596e-02,   1.13799622e-01,
                     5.17202414e-01,   1.70456683e+00,   3.84122107e+00,
                     4.87413759e+00,   2.08569105e+00]),
         np.array([  3.26997106e-04,   2.70835745e-03,   1.30785763e-02,
                     4.90588258e-02,   1.58683880e-01,   4.68953025e-01,
                     1.32631667e+00,   3.83737061e+00]),
         ),
         (1.5,
          np.array([  2.03745495e-03,   2.31813045e-02,   1.42838322e-01,
                      6.05393876e-01,   1.85993681e+00,   3.98203612e+00,
                      5.10207126e+00,   2.53254513e+00]),
          np.array([  1.88236828e-04,   1.72537665e-03,   9.09041026e-03,
                      3.71208318e-02,   1.31303364e-01,   4.29173028e-01,
                      1.37227840e+00,   4.70057547e+00]),
          ),
         (1.75,
          np.array([  2.50657937e-03,   2.72749636e-02,   1.60825323e-01,
                      6.52207158e-01,   1.92821692e+00,   4.05148405e+00,
                      5.35173671e+00,   3.06654746e+00]),
          np.array([  1.09326774e-04,   1.09659966e-03,   6.25155085e-03,
                      2.75753740e-02,   1.05729535e-01,   3.77827360e-01,
                      1.34325363e+00,   5.31805274e+00]),
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
        (2.25,
         np.array([  3.02233733e-03,   3.10959566e-02,   1.74091827e-01,
                  6.74457937e-01,   1.93387183e+00,   4.07555480e+00,
                  5.80412767e+00,   4.24327026e+00]),
         np.array([  3.79516055e-05,   4.46695835e-04,   2.92969367e-03,
                  1.48143362e-02,   6.54274109e-02,   2.72741926e-01,
                  1.16012436e+00,   5.84499592e+00]),
         ),
        (2.5,
         np.array([  3.09907888e-03,   3.13969645e-02,   1.73360850e-01,
                  6.64847427e-01,   1.90082698e+00,   4.04984377e+00,
                  5.99057823e+00,   4.84416683e+00]),
         np.array([  2.25913531e-05,   2.86414090e-04,   2.00271733e-03,
                  1.07730420e-02,   5.06946307e-02,   2.26291195e-01,
                  1.04135407e+00,   5.82166367e+00]),
         ),
        (2.75,
         np.array([  3.07759263e-03,   3.09199432e-02,   1.69375193e-01,
                  6.46610533e-01,   1.85258212e+00,   4.00373109e+00,
                  6.14743945e+00,   5.44062854e+00]),
         np.array([  1.34771532e-05,   1.83790379e-04,   1.36657861e-03,
                  7.79600019e-03,   3.89487163e-02,   1.85392485e-01,
                  9.18220664e-01,   5.65190045e+00]),
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
        (3.25,
         np.array([  2.81333543e-03,   2.83103276e-02,   1.54743106e-01,
                  5.92538218e-01,   1.72231584e+00,   3.85446072e+00,
                  6.36549870e+00,   6.60246632e+00]),
         np.array([  4.77515101e-06,   7.53310436e-05,   6.30003331e-04,
                  4.01365507e-03,   2.24120138e-02,   1.20086835e-01,
                  6.80450508e-01,   4.98555042e+00]),
         ),
        (3.5,
         np.array([  2.63493918e-03,   2.66202873e-02,   1.45833127e-01,
                  5.61055473e-01,   1.64694115e+00,   3.75564199e+00,
                  6.42306039e+00,   7.15406756e+00]),
         np.array([  2.86364388e-06,   4.83717889e-05,   4.27246310e-04,
                  2.86453738e-03,   1.68362578e-02,   9.52427526e-02,
                  5.73853421e-01,   4.54960434e+00]),
         ),
        (3.75,
         np.array([  2.52556233e-03,   2.52687568e-02,   1.38061528e-01,
                  5.32259513e-01,   1.57489025e+00,   3.65196012e+00,
                  6.44759766e+00,   7.66322744e+00]),
         np.array([  1.79898320e-06,   3.19025602e-05,   2.94738112e-04,
                  2.06601434e-03,   1.27125806e-02,   7.55475779e-02,
                  4.81498066e-01,   4.10421637e+00]),
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
        (4.25,
         np.array([  2.98703553e-03,   2.60418901e-02,   1.34745429e-01,
                  5.05981783e-01,   1.48704427e+00,   3.49526076e+00,
                  6.43784889e+00,   8.46064115e+00]),
         np.array([  1.02024747e-06,   1.74340853e-05,   1.64846771e-04,
                  1.21125378e-03,   7.91888730e-03,   5.06072396e-02,
                  3.52330049e-01,   3.38157214e+00]),
         ),
        (4.5,
         np.array([  3.57010614e-03,   2.79496099e-02,   1.38169983e-01,
                  5.05879847e-01,   1.46787842e+00,   3.44443589e+00,
                  6.42125506e+00,   8.76168208e+00]),
         np.array([  8.86446183e-07,   1.42626489e-05,   1.32908651e-04,
                  9.82479942e-04,   6.53278969e-03,   4.28068927e-02,
                  3.08213788e-01,   3.10322461e+00]),
         ),
        (4.75,
         np.array([  4.34147576e-03,   3.04293019e-02,   1.43230140e-01,
                  5.09832167e-01,   1.45679015e+00,   3.40356818e+00,
                  6.40074908e+00,   9.01902624e+00]),
         np.array([  8.01531774e-07,   1.20948120e-05,   1.10300128e-04,
                  8.15434233e-04,   5.48651484e-03,   3.66906220e-02,
                  2.71953278e-01,   2.85731362e+00]),
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
        (5.25,
        np.array([  6.45944550e-03,   3.67009077e-02,   1.56495371e-01,
                 5.25048515e-01,   1.44962975e+00,   3.34201845e+00,
                 6.35327017e+00,   9.43317911e+00]),
        np.array([  6.96302951e-07,   9.31687929e-06,   8.06697436e-05,
                 5.90325057e-04,   4.02564583e-03,   2.77601343e-02,
                 2.15789342e-01,   2.43845348e+00])
        ),
        (5.5,
         np.array([  7.83422239e-03,   4.04238492e-02,   1.64329516e-01,
                  5.35236245e-01,   1.45142179e+00,   3.31906077e+00,
                  6.32826172e+00,   9.59975321e+00]),
         np.array([  6.60557943e-07,   8.38015660e-06,   7.05996176e-05,
                  5.12344075e-04,   3.50453676e-03,   2.44453624e-02,
                  1.93782688e-01,   2.25936724e+00]),
         ),
        (5.75,
        np.array([  9.44354234e-03,   4.45212136e-02,   1.72835877e-01,
                 5.46749762e-01,   1.45597815e+00,   3.30040905e+00,
                 6.30333260e+00,   9.74419729e+00]),
        np.array([  6.31427920e-07,   7.63131191e-06,   6.25591461e-05,
                 4.49619447e-04,   3.07929986e-03,   2.16823076e-02,
                 1.74874928e-01,   2.09764087e+00]),
        ),
        (6.,
         np.array([ 0.0113067 ,  0.04898785,  0.18195408,  0.55939775,  1.46288372,
                 3.28556791,  6.27896305,  9.86946446]),
         np.array([  6.07125356e-07,   7.02153046e-06,   5.60375312e-05,
                  3.98494081e-04,   2.72853912e-03,   1.93601976e-02,
                  1.58544866e-01,   1.95149972e+00]),
        ),
        (6.25,
        np.array([ 0.01344308,  0.05382052,  0.19163668,  0.57302986,  1.47180585,
                3.2741163 ,  6.25548875,  9.97808294]),
        np.array([  5.86478729e-07,   6.51723629e-06,   5.06751401e-05,
                 3.56331345e-04,   2.43639735e-03,   1.73940780e-02,
                 1.44372912e-01,   1.81933298e+00]),
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
    from astrometry.util.plotutils import PlotSequence
    import pylab as plt

    mix = SersicMixture()
    plt.clf()
    for (n, amps, vars) in mix.fits:
        plt.loglog(vars, amps, 'b.-')
        plt.text(vars[0], amps[0], '%.2f' % n, ha='right', va='top')
    plt.xlabel('Variance')
    plt.ylabel('Amplitude')
    plt.savefig('serfits.png')
    
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
