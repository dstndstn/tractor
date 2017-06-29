from __future__ import print_function
from tractor.galaxy import HoggGalaxy
from tractor.utils import MogParams, ParamList
from tractor.mixture_profiles import MixtureOfGaussians
import numpy as np

#################### First way -- expose the mixture-of-Gaussians directly,
#################### allowing them to be fit in general.

class MogGalaxy(HoggGalaxy):
    '''
    A galaxy model that directly exposes the Mixture of Gaussians components.
    '''
    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, mog=2,
                    # Alias '.shape' to '.mog' to use Galaxy derivatives code
                    shape=2)
    def getProfile(self):
        return self.mog.mog

    def getRadius(self):
        return 5. * np.sqrt(np.max(self.mog.mog.var))*3600.

    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        cd = img.getWcs().cdAtPixel(px, py)
        Tinv = np.linalg.inv(cd)
        galmix = self.getProfile()
        amix = galmix.apply_affine(np.array([px,py]), Tinv.T)
        amix.symmetrize()
        return amix
    
class MyMogParams(MogParams):
    def getStepSizes(self):
        '''Set step sizes when taking derivatives of the parameters of the mixture of Gaussians.'''
        K = self.mog.K
        vv = (self.mog.var[:,0,0] + self.mog.var[:,1,1]) / 2.
        ss = [0.01]*K + [0.01]*K*2 + list((0.01 * vv).repeat(3))
        return list(self._getLiquidArray(ss))

#################### (end of first way)

#################### Second way -- fit the radial profile as MoG, but keep
#################### the elliptical 2-d shape

class EllipticalMogGalaxy(HoggGalaxy):
    '''
    A galaxy model that is still based on an elliptical radial profile
    but allows the radial profile to be fit as a Mixture of Gaussians.
    '''
    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2, profile=3)

    nre = 5.

    def getName(self):
        return 'EllipticalMogGalaxy'

    def getProfile(self):
        return self.profile.getMog()

    def getParamDerivatives(self, img, modelMask=None):
        derivs = super(EllipticalMogGalaxy, self).getParamDerivatives(img, modelMask=modelMask)

        pos0 = self.getPosition()
        (px0,py0) = img.getWcs().positionToPixel(pos0, self)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        patch0 = self.getUnitFluxModelPatch(img, px0, py0,
                                            modelMask=modelMask)
        if patch0 is None:
            return [None] * self.numberOfParams()

        # derivatives wrt MoG componets... this is boilerplate-ish
        psteps = self.profile.getStepSizes()
        if not self.isParamFrozen('profile'):
            pnames = self.profile.getParamNames()
            oldvals = self.profile.getParams()
            if counts == 0:
                derivs.extend([None] * len(oldvals))
                psteps = []
            for i,pstep in enumerate(psteps):
                oldval = self.profile.setParam(i, oldvals[i]+pstep)
                patchx = self.getUnitFluxModelPatch(
                    img, px0, py0, modelMask=modelMask)
                self.profile.setParam(i, oldval)
                if patchx is None:
                    continue
                dx = (patchx - patch0) * (counts / pstep)
                dx.setName('d(%s)/d(%s)' % (self.dname, pnames[i]))
                derivs.append(dx)
        return derivs

class MogProfile(ParamList):
    def __init__(self, *args):
        K = len(args) / 2
        ## HACK -- internally, keep stddevs rather than variances (to avoid negatives?)
        # OR work in log-variances?
        args = np.array(args)
        #args[K:] = np.sqrt(args[K:])
        args[K:] = np.log10(args[K:])
        super(MogProfile, self).__init__(*args)
        self.K = self.numberOfParams() / 2
        self._set_param_names(self.K)

    def getMog(self):
        p = self.getAllParams()
        K = len(p) / 2
        assert(K == self.K)
        amps = np.array(p[:K])
        # ??
        amps /= np.sum(amps)
        # log-variance
        var = 10.**np.array(p[K:])

        if hasattr(self, 'mog'):
            assert(self.mog.K == K)
            self.mog.amp[:] = amps
            self.mog.var[:,0,0] = self.mog.var[:,1,1] = var
        else:
            vv = np.zeros((K,2,2))
            vv[:,0,0] = vv[:,1,1] = var
            self.mog = MixtureOfGaussians(amps, np.zeros((K,2)), vv)
        return self.mog

    def _set_param_names(self, K):
        names = {}
        for k in range(K):
            names['amp%i' % k] = k
            #names['var%i' % k] = k+K
            names['logvar%i' % k] = k+K
        self.addNamedParams(**names)

    def getStepSizes(self):
        '''Set step sizes when taking derivatives of the parameters of the mixture of Gaussians.'''
        return [0.01]*self.K*2

#################### (end of second way)

    
if __name__ == '__main__':
    h,w = 100,100
    from tractor.galaxy import ExpGalaxy
    from tractor import Image, GaussianMixturePSF, LinearPhotoCal
    from tractor import PixPos, Flux, EllipseE, Tractor, ModelMask
    import pylab as plt

    # Create a Tractor Image that works in pixel space (WCS not specified).
    tim = Image(data=np.zeros((h,w)), inverr=np.ones((h,w)),
                psf=GaussianMixturePSF(1., 0., 0., 3., 3., 0.),
                photocal=LinearPhotoCal(1.))

    # Create a plain Exp galaxy to generate a postage stamp that we'll try to fit with
    # the MogGalaxy model.
    gal = ExpGalaxy(PixPos(w//2, h//2), Flux(1000.),
                    EllipseE(10., 0.5, 0.3))

    # Get the model
    tractor = Tractor([tim], [gal])
    mod = tractor.getModelImage(0)

    #mog = gal._getAffineProfile(tim, w//2, h//2)
    #print('Exp galaxy profile:', str(mog))

    # Plot the model
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.savefig('mod.png')

    # Set the tractor Image to the Exp model postage stamp -- this is what we'll try to fit.
    tim.data = mod

    # Initialize the MoG components
    amp = np.array([0.4, 0.3, 0.3])
    mean = np.zeros((3,2))
    var = np.array([
        [0.01,0., 0., 0.01],
        [0.1, 0., 0., 0.1 ],
        [1.,  0., 0., 1.0 ],
        ])
    var *= 64.
    var = var.reshape((-1,2,2))
    # ~ arcsec -> degrees
    var /= 3600.**2

    if False:
        # Create the MoG galaxy object
        moggal = MogGalaxy(gal.pos.copy(), gal.brightness.copy(),
                           MyMogParams(amp, mean, var))
        # Freeze the MoG means -- The overall mean is degenerate with
        # galaxy position.
        K = moggal.mog.mog.K
        for i in range(K):
            moggal.mog.freezeParam('meanx%i' % i)
            moggal.mog.freezeParam('meany%i' % i)

        # Freeze the galaxy brightness -- otherwise it's degenerate with MoG amplitudes.
        moggal.freezeParam('brightness')

    else:

        # The "MogProfile" call here takes a list of amplitudes (they
        # get normalized), followed by a list of variances.

        moggal = EllipticalMogGalaxy(gal.pos.copy(), gal.brightness.copy(),
                                     EllipseE(1., 0., 0.),
                                     MogProfile(1.0, 1.0, 1.0,
                                                1.0, 4.0, 9.0))

    # Create a Tractor object that will fit the "moggal" given its appearance in "tim".
    tractor = Tractor([tim], [moggal])

    # Initial model:
    mod = tractor.getModelImage(0)

    #mog = moggal._getAffineProfile(tim, w//2, h//2)
    #print('MoG galaxy profile:', str(mog))

    # Plot initial model.
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.savefig('mod2.png')

    # Don't fit any of the image calibration params
    tractor.freezeParam('images')

    # Plot the parameter derivatives
    derivs = moggal.getParamDerivatives(tim, modelMask=ModelMask(0,0,w,h))
    for i,p in enumerate(moggal.getParamNames()):
        print('Param', p, 'derivative:', derivs[i])
        if derivs[i] is None:
            continue
        plt.clf()
        plt.imshow(derivs[i].patch, interpolation='nearest', origin='lower')
        plt.title('MoG galaxy derivative for parameter %s' % p)
        plt.savefig('deriv-%02i.png' % i)

    # import sys
    # import logging
    # lvl = logging.DEBUG
    # logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    # Optimize the model.
    for step in range(50):
        print('Tractor params:')
        tractor.printThawedParams()
        dlnp,X,alpha = tractor.optimize(damp=1.)
        print('dlnp', dlnp)
        print('galaxy:', moggal)
        #print('Mog', moggal.mog.getParams())
        if dlnp == 0:
            break

        # Plot the model as we're optimizing...
        mod = tractor.getModelImage(0)
        chi = (tim.getImage() - mod) * tim.getInvError()
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(mod, interpolation='nearest', origin='lower')
        plt.title('Model')
        plt.subplot(1,2,2)
        mx = np.abs(chi).max()
        plt.imshow(chi, interpolation='nearest', origin='lower',
                   vmin=-mx, vmax=mx)
        plt.colorbar()
        plt.title('Chi residuals')
        plt.suptitle('MoG model after optimization step %i' % step)
        plt.savefig('mod-o%02i.png' % step)
        
