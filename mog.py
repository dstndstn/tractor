from __future__ import print_function
from tractor.galaxy import HoggGalaxy
from tractor.utils import MogParams
import numpy as np

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
        return 5. * np.sqrt(np.max(self.mog.mog.var))

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

if __name__ == '__main__':
    h,w = 100,100
    from tractor.galaxy import ExpGalaxy
    from tractor import Image, GaussianMixturePSF, LinearPhotoCal
    from tractor import PixPos, Flux, EllipseE, Tractor
    import pylab as plt

    # Create a Tractor Image that works in pixel space (WCS not specified).
    tim = Image(data=np.zeros((h,w)), inverr=np.ones((h,w)),
                psf=GaussianMixturePSF(1., 0., 0., 3., 3., 0.),
                photocal=LinearPhotoCal(1.))

    # Create a plain Exp galaxy to generate a postage stamp that we'll try to fit with
    # the MogGalaxy model.
    gal = ExpGalaxy(PixPos(w//2, h//2), Flux(1000.),
                    EllipseE(5., 0.5, 0.3))

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
    var *= 16.
    var = var.reshape((-1,2,2))
    # ~ arcsec -> degrees
    var /= 3600.**2

    # Create the MoG galaxy object
    moggal = MogGalaxy(gal.pos.copy(), gal.brightness.copy(),
                       MyMogParams(amp, mean, var))
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

    # Freeze the MoG means -- The overall mean is degenerate with
    # galaxy position.
    K = moggal.mog.mog.K
    for i in range(K):
        moggal.mog.freezeParam('meanx%i' % i)
        moggal.mog.freezeParam('meany%i' % i)
    # Freeze the galaxy brightness -- otherwise it's degenerate with MoG amplitudes.
    moggal.freezeParam('brightness')
    
    # Optimize the model.
    for step in range(20):
        print('Tractor params:')
        tractor.printThawedParams()
        dlnp,X,alpha = tractor.optimize()
        print('dlnp', dlnp)
        print('galaxy:', moggal)
        #print('Mog', moggal.mog.getParams())
        if dlnp == 0:
            break

        # Plot the model as we're optimizing...
        mod = tractor.getModelImage(0)
        plt.clf()
        plt.imshow(mod, interpolation='nearest', origin='lower')
        plt.savefig('mod-o%02i.png' % step)
        
