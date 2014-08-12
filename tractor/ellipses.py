if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    from astrometry.util.plotutils import *
    from tractor.galaxy import *

import numpy as np

from tractor import *

class EllipseE(ParamList):
    '''
    Ellipse parameterization with r, e1, e2.
    '''
    @staticmethod
    def getName():
        return "EllipseE"

    def __init__(self, *args, **kwargs):
        super(EllipseE, self).__init__(*args, **kwargs)
        self.stepsizes = [0.01]*3
    
    @staticmethod
    def getNamedParams():
        # re: effective radius in arcsec
        # e1: e cos 2 theta, dimensionless
        # e2: e sin 2 theta, dimensionless
        return dict(re=0, e1=1, e2=2)

    @staticmethod
    def fromEllipseESoft(esoft, maxe=1.0):
        re = esoft.re
        e = min(maxe, esoft.e)
        theta = esoft.theta
        e1 = e * np.cos(2. * theta)
        e2 = e * np.sin(2. * theta)
        return EllipseE(re, e1, e2)

    @staticmethod
    def fromRAbPhi(r, ba, phi):
        ab = 1./ba
        e = (ab - 1) / (ab + 1)
        angle = np.deg2rad(2.*(-phi))
        e1 = e * np.cos(angle)
        e2 = e * np.sin(angle)
        return EllipseE(r, e1, e2)

    @property
    def e(self):
        return np.hypot(self.e1, self.e2)

    @property
    def theta(self):
        '''
        Returns position angle in *radians*
        '''
        return np.arctan2(self.e2, self.e1) / 2.

    def __repr__(self):
        return 're=%g, e1=%g, e2=%g' % (self.re, self.e1, self.e2)
    def __str__(self):
        return self.getName() + ': ' + repr(self)

    # def getAllStepSizes(self, *args, **kwargs):
    #     # re
    #     # e1,e2: step toward e=0
    #     ss = [ 0.01,
    #            0.01 if self.e1 <= 0 else -0.01,
    #            0.01 if self.e2 <= 0 else -0.01 ]
    #     return ss

    def isLegal(self):
        return ((self.e1**2 + self.e2**2) < 1.) and (self.re >= 0.)

    def getRaDecBasis(self):
        ''' Returns a transformation matrix that takes vectors in r_e
        to delta-RA, delta-Dec vectors.
        '''
        theta = self.theta
        ct = np.cos(theta)
        st = np.sin(theta)

        # Using this (untested) function could eliminate the arctan2/cos/sin above.
        # Faster?  Maybe.
        # def halfangletrig(ecos2theta,esin2theta):
        #     e =np.hypot(ecos2theta, esin2theta)
        #     costheta = np.sqrt(0.5 * (1. + ecostwotheta / e))
        #     if esin2theta < 0.:
        #         costheta *= -1.
        #     sintheta = np.sqrt(0.5 * (1. - ecostwotheta / e))
        #     return costheta, sintheta

        e = self.e
        maxab = 1000.
        if e >= 1.:
            ab = maxab
        else:
            ab = min(maxab, (1.+e)/(1.-e))
        r_deg = self.re / 3600.
        
        # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
        G = r_deg * np.array([[ ct / ab, st],
                              [-st / ab, ct]])
        return G

    def getTensor(self, cd):
        # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
        G = self.getRaDecBasis()
        # "cd" takes pixels to degrees (intermediate world coords)
        # T takes pixels to unit vectors.
        T = np.dot(np.linalg.inv(G), cd)
        return T

class EllipseESoft(EllipseE):
    '''
    This is an alternate implementation of the ellipse describing a
    galaxy shape, and can be used as a drop-in replacement of the
    "GalaxyShape" class used in the tractor.galaxy ExpGalaxy and
    DevGalaxy classes.

    The parameters are a tweak on the usual ellipticity parameters
    e1,e2, plus log(effective radius).  The tweak is that we map 'e'
    through a sigmoid-like 1-exp(-|ee|) function so that there are no
    forbidden regions in the parameter space.

    In this class, we use "ee" to indicate the "softened" parameters
    (before they have gone through the sigmoid to bring them into
    |e|<1, and "e" to indicate the usual, unsoftened versions.
    '''
    @staticmethod
    def getName():
        return "EllipseESoft"

    @staticmethod
    def getNamedParams():
        # log r: log of effective radius in arcsec
        # ee1: e cos 2 theta, dimensionless
        # ee2: e sin 2 theta, dimensionless
        return dict(logre=0, ee1=1, ee2=2)

    @staticmethod
    def fromRAbPhi(r, ba, phi):
        ab = 1./ba
        e = (ab - 1) / (ab + 1)
        ee = -np.log(1 - e)
        angle = np.deg2rad(2.*(-phi))
        ee1 = ee * np.cos(angle)
        ee2 = ee * np.sin(angle)
        return EllipseESoft(np.log(r), ee1, ee2)

    #def getAllStepSizes(self, *args, **kwargs):
    #    return [0.01] * 3

    def __repr__(self):
        return 'log r_e=%g, ee1=%g, ee2=%g' % (self.logre, self.ee1, self.ee2)

    @property
    def re(self):
        #return np.exp(self.logre)
        # HACK - limits shouldn't be HERE...
        return np.exp(np.clip(self.logre, -100, 100))

    @property
    def e(self):
        '''
        Returns the "usual" ellipticity e in [0,1]
        '''
        ee = np.hypot(self.ee1, self.ee2)
        return 1. - np.exp(-ee)

    @property
    def softe(self):
        '''
        Returns the "softened" ellipticity ee in [0, inf]
        '''
        #return np.hypot(self.ee1, self.ee2)
        # HACK - limits shouldn't be HERE...
        return np.clip(np.hypot(self.ee1, self.ee2), 0, 100.)

    @property
    def theta(self):
        '''
        Returns position angle in *radians*
        '''
        return np.arctan2(self.ee2, self.ee1) / 2.
    
    # Have to override this because all parameter values are legal,
    # unlike the superclass.
    def isLegal(self):
        return True
    
if __name__ == '__main__':
    ps = PlotSequence('ell')

    #r,ab,phi = 1., 0.5, 55.
    for r,ab,phi in [(1., 1.0, 0.),
                     (1., 0.5, 0.),
                     (2., 0.25, 0.),
                     (2., 0.5, 90.),
                     (2., 0.5, 180.),
                     (2., 0.5, 45.),
                     (2., 0.5, 30.),
                     (2., 0.1, -30.),
                     ]:
        ell = GalaxyShape(r, ab, phi)
        print 'ell:', ell
        ebasis = ell.getRaDecBasis()
        print 'basis:', ebasis
        
        esoft = EllipseESoft.fromRAbPhi(r, ab, phi)
        print 'soft:', esoft
        sbasis = esoft.getRaDecBasis()
        print 'basis:', sbasis

        enorm = EllipseE.fromRAbPhi(r, ab, phi)
        print 'e normal:', enorm
        nbasis = enorm.getRaDecBasis()
        print 'basis:', nbasis

        angle = np.linspace(0., 2.*np.pi, 100)
        xx,yy = np.sin(angle), np.cos(angle)
        xy = np.vstack((xx,yy)) * 3600.

        plt.clf()
        txy = np.dot(ebasis, xy)
        plt.plot(txy[0,:], txy[1,:], 'r-', alpha=0.25, lw=4)
        txy = np.dot(sbasis, xy)
        plt.plot(txy[0,:], txy[1,:], 'b-', alpha=0.5, lw=2)
        txy = np.dot(nbasis, xy)
        plt.plot(txy[0,:], txy[1,:], 'g-', alpha=0.8)
        plt.axis('equal')
        ps.savefig()

    import sys
    sys.exit(0)
    
    ps = PlotSequence('ell')
    
    angle = np.linspace(0., 2.*np.pi, 20)
    xx,yy = np.sin(angle), np.cos(angle)
    xy = np.vstack((xx,yy))
    #print 'xy', xy.shape

    n1,n2 = 7,7
    E1,E2 = np.meshgrid(np.linspace(-1.2, 1.2, n2), np.linspace(-1.2, 1.2, n2))
    
    plt.clf()
    for logre,cc in zip([4,5,6], 'rgb'):
        for e1,e2 in zip(E1.ravel(), E2.ravel()):
            e = EllipseESoft(logre, e1, e2)
            print e

            #ec = e.copy()
            #print 'Copy:', ec

            T = e.getRaDecBasis()
            #print 'T', T
            txy = np.dot(T, xy)
            #print 'txy', txy.shape
            plt.plot(e1 + txy[0,:], e2 + txy[1,:], '-', color=cc, alpha=0.5)
    plt.xlabel('"e1"')
    plt.ylabel('"e2"')
    plt.axis('scaled')
    plt.title('EllipseESoft')
    ps.savefig()


    plt.clf()
    for re,cc in zip([np.exp(4.), np.exp(5.), np.exp(6.)], 'rgb'):
        for e1,e2 in zip(E1.ravel(), E2.ravel()):
            e = EllipseE(re, e1, e2)
            print e
            T = e.getRaDecBasis()
            #print 'T', T
            txy = np.dot(T, xy)
            #print 'txy', txy.shape
            plt.plot(e1 + txy[0,:], e2 + txy[1,:], '-', color=cc, alpha=0.5)
    plt.xlabel('e1')
    plt.ylabel('e2')
    plt.axis('scaled')
    plt.title('EllipseE')
    ps.savefig()





    W,H = 500,500
    img = np.zeros((H,W), np.float32)
    sig1 = 1.
    pixscale = 1.
    psf = NCircularGaussianPSF([1.5], [1.])
    tim = Image(data=img, invvar=np.ones_like(img) * (1./sig1**2),
                psf=psf, wcs=NullWCS(pixscale=pixscale), sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1.),
                domask=False, zr=[-2.*sig1, 3.*sig1])

    cat = []
    logre = 3.
    x = np.linspace(0, W, n1, endpoint=False)
    x += (x[1]-x[0])/2.
    y = np.linspace(0, H, n2, endpoint=False)
    y += (y[1]-y[0])/2.
    xx,yy = np.meshgrid(x, y)
    for e1,e2,x,y in zip(E1.ravel(), E2.ravel(), xx.ravel(), yy.ravel()):
        e = EllipseESoft(logre, e1, e2)
        gal = ExpGalaxy(PixPos(x, y), Flux(500.*sig1), e)
        # FIXME -- if 'halfsize' is not set, checks e.ab, e.re, etc.
        gal.halfsize = int(np.ceil(gal.nre * np.exp(logre) / (pixscale/3600.)))
        print 'Galaxy', gal
        cat.append(gal)

        # theta = np.arctan2(e2, e1) / 2.
        # e = np.sqrt(e1**2 + e2**2)
        # e = 1. - np.exp(-e)
        # ab = (1.+e)/(1.-e)
        # r = np.exp(logre)
        # gal = ExpGalaxy(PixPos(x, y), Flux(50.*sig1), GalaxyShape(r,ab,theta))
        # gal.halfsize = 20.
        # cat2.append(gal)
        # 
        # px,py = tim.wcs.positionToPixel(gal.pos)
        # print 'px,py', px,py

    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-1*sig1, vmax=3*sig1)
        
    tractor = Tractor([tim], cat)
    mod = tractor.getModelImage(0)
    plt.clf()
    plt.imshow(mod, **ima)
    ps.savefig()

