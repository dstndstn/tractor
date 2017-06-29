from __future__ import print_function
from tractor.galaxy import HoggGalaxy
from tractor.utils import MogParams
import numpy as np

class MyMogParams(MogParams):
    def getStepSizes(self):
        '''Set step sizes when taking derivatives of the parameters of the mixture of Gaussians.'''
        K = self.mog.K
        vv = (self.mog.var[:,0,0] + self.mog.var[:,1,1]) / 2.
        ss = [0.01]*K + [0.01]*K*2 + list((0.01 * vv).repeat(3))
        return list(self._getLiquidArray(ss))

class MogGalaxy(HoggGalaxy):
    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, mog=2,
                    # Alias '.shape' to '.mog' to use Galaxy derivatives code
                    shape=2)

    # def __str__(self):
    #     return (self.name + ' at ' + str(self.pos)
    #             + ' with ' + str(self.brightness)
    #             + ' and ' + str(self.mog))
    # def __repr__(self):
    #     return (self.name + '(pos=' + repr(self.pos) +
    #             ', brightness=' + repr(self.brightness) +
    #              ', mog=' + repr(self.mog) + ')')
    
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
    



if __name__ == '__main__':
    h,w = 100,100
    from tractor.galaxy import ExpGalaxy
    from tractor import Image, GaussianMixturePSF, LinearPhotoCal
    from tractor import PixPos, Flux, EllipseE, Tractor
    import pylab as plt
    
    tim = Image(data=np.zeros((h,w)), inverr=np.ones((h,w)),
                psf=GaussianMixturePSF(1., 0., 0., 3., 3., 0.),
                photocal=LinearPhotoCal(1.))

    gal = ExpGalaxy(PixPos(w//2, h//2), Flux(1000.),
                    EllipseE(5., 0.5, 0.3))

    tractor = Tractor([tim], [gal])

    mod = tractor.getModelImage(0)

    mog = gal._getAffineProfile(tim, w//2, h//2)
    #print('Exp galaxy profile:', str(mog))

    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.savefig('mod.png')
    

    tim.data = mod

    amp = np.array([0.4, 0.3, 0.3])
    mean = np.zeros((3,2))
    var = np.array([
        [0.01,0., 0., 0.01],
        [0.1, 0., 0., 0.1 ],
        [1.,  0., 0., 1.0 ],
        ])
    var *= 16.

    var = var.reshape((-1,2,2))
    var /= 3600.**2
    #print('Var shape', var.shape)
    
    moggal = MogGalaxy(gal.pos.copy(), gal.brightness.copy(),
                       MyMogParams(amp, mean, var))
    # MogParams(0.4, 0.3, 0.3,
    #           0., 0., 0., 0., 0., 0.,
    #           0.01, 0.01,0.,
    #           0.1, 0.1,0.,
    #           1., 1., 0.))
    tractor = Tractor([tim], [moggal])

    mod = tractor.getModelImage(0)

    mog = moggal._getAffineProfile(tim, w//2, h//2)
    print('MoG galaxy profile:', str(mog))
    
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.savefig('mod2.png')
    
    tractor.freezeParam('images')

    #moggal.freezeParam('pos')
    K = moggal.mog.mog.K
    # The overall mean is degenerate with galaxy position
    for i in range(K):
        moggal.mog.freezeParam('meanx%i' % i)
        moggal.mog.freezeParam('meany%i' % i)
    # otherwise degenerate with MoG amplitudes
    moggal.freezeParam('brightness')
    
    for step in range(20):
        print('Tractor params:')
        tractor.printThawedParams()
        dlnp,X,alpha = tractor.optimize()
        print('dlnp', dlnp)
        print('galaxy:', moggal)
        #print('Mog', moggal.mog.getParams())
        if dlnp == 0:
            break

        mod = tractor.getModelImage(0)
        plt.clf()
        plt.imshow(mod, interpolation='nearest', origin='lower')
        plt.savefig('mod-o%02i.png' % step)
        
