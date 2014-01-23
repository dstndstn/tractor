if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    from astrometry.util.plotutils import *
    from tractor.sdss_galaxy import *

import numpy as np

from tractor import *
    
class EllipseE(ParamList):
    '''
    This is an alternate implementation of the ellipse describing a
    galaxy shape, and can be used as a drop-in replacement of the
    "GalaxyShape" class used in the tractor.sdss_galaxy ExpGalaxy and
    DevGalaxy classes.

    The parameters are a tweak on the usual ellipticity parameters
    e1,e2, plus log(effective radius).  The tweak is that we map 'e'
    through a sigmoid-like 1-exp(-e) function so that there are no
    forbidden regions in the parameter space.
    '''
    @staticmethod
    def getName():
        return "EllipseE"

    @staticmethod
    def getNamedParams():
        # log r: log of effective radius in arcsec
        # e1: e cos 2 theta, dimensionless
        # e2: e sin 2 theta, dimensionless
        return dict(logr=0, e1=1, e2=2)

    def __repr__(self):
        return 'log r=%g, e1=%g, e2=%g' % (self.logr, self.e1, self.e2)
    def __str__(self):
        return self.getName() + ': ' + repr(self)

    def getRaDecBasis(self):
        ''' Returns a transformation matrix that takes vectors in r_e
        to delta-RA, delta-Dec vectors.
        '''
        e1 = self.e1
        e2 = self.e2

        # FIXME -- some trig here... or redefine e1,e2 to be e sin/cos theta?
        theta = np.arctan2(e2, e1) / 2.
        ct = np.cos(theta)
        st = np.sin(theta)
        
        e = np.sqrt(e1**2 + e2**2)
        e = 1. - np.exp(-e)
        ab = (1.+e)/(1.-e)
        r_deg = np.exp(self.logr) / 3600.
        
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




if __name__ == '__main__':
    ps = PlotSequence('ell')
    
    angle = np.linspace(0., 2.*np.pi, 20)
    xx,yy = np.sin(angle), np.cos(angle)
    xy = np.vstack((xx,yy))
    #print 'xy', xy.shape

    n1,n2 = 7,7
    E1,E2 = np.meshgrid(np.linspace(-1.2, 1.2, n2), np.linspace(-1.2, 1.2, n2))
    
    plt.clf()
    for logr,cc in zip([4,5,6], 'rgb'):
        for e1,e2 in zip(E1.ravel(), E2.ravel()):
            e = EllipseE(logr, e1, e2)
            print e
            T = e.getRaDecBasis()
            #print 'T', T
            txy = np.dot(T, xy)
            #print 'txy', txy.shape
            plt.plot(e1 + txy[0,:], e2 + txy[1,:], '-', color=cc, alpha=0.5)
    plt.xlabel('"e1"')
    plt.ylabel('"e2"')
    plt.axis('scaled')
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
    logr = 3.
    x = np.linspace(0, W, n1, endpoint=False)
    x += (x[1]-x[0])/2.
    y = np.linspace(0, H, n2, endpoint=False)
    y += (y[1]-y[0])/2.
    xx,yy = np.meshgrid(x, y)
    for e1,e2,x,y in zip(E1.ravel(), E2.ravel(), xx.ravel(), yy.ravel()):
        e = EllipseE(logr, e1, e2)
        gal = ExpGalaxy(PixPos(x, y), Flux(500.*sig1), e)
        # FIXME -- if 'halfsize' is not set, checks e.ab, e.re, etc.
        gal.halfsize = int(np.ceil(gal.nre * np.exp(logr) / (pixscale/3600.)))
        print 'Galaxy', gal
        cat.append(gal)

        # theta = np.arctan2(e2, e1) / 2.
        # e = np.sqrt(e1**2 + e2**2)
        # e = 1. - np.exp(-e)
        # ab = (1.+e)/(1.-e)
        # r = np.exp(logr)
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

