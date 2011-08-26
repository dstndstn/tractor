# ============================================================================

# Copyright 2011 David W. Hogg (NYU) and Phil Marshall (Oxford).
# All rights reserved (for now).

# to-do
# -----
# - write and test lens equation solver: 1  image done, awaiting more
# - write down priors on magnification / mag bias

import numpy as np
import markovpy as dfm
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':18})
    rc('text', usetex=True)
    import pylab as plt
import matplotlib.nxutils as nx

# ============================================================================

# cheap and nasty singular isothermal sphere model
# initialize with
# - position        : lens position (same units as einsteinradius)
# - einsteinradius  : Einstein radius (some angular units)
# - gammacos2phi    : external shear amplitude times cos(2 phi)
# - gammasin2phi    : external shear amplitude times sin(2 phi)
class GravitationalLens:

    def __init__(self, position, einsteinradius, gammacos2phi, gammasin2phi):
        self.position = np.array(position)
        self.einsteinradius = einsteinradius
        self.gammacos2phi = gammacos2phi
        self.gammasin2phi = gammasin2phi
        self.gamma = np.sqrt(self.gammacos2phi**2 + self.gammasin2phi**2)
        self.phi = 0.5 * np.arctan2(self.gammasin2phi, self.gammacos2phi)
        self.name = 'SIS + external shear'
        return None

# ----------------------------------------------------------------------------

    def __str__(self):
        return '%s (%s %f %f %f)' % (self.name, self.position, self.einsteinradius, self.gammacos2phi, self.gammasin2phi)
    
# ----------------------------------------------------------------------------
    
    def eigs(self):
   	return (np.array([np.cos(self.phi), np.sin(self.phi)]), 
                np.array([-np.sin(self.phi), np.cos(self.phi)]))
    
    def cross(self):
        eig1, eig2 = self.eigs()
        foo = np.array([ eig1 / (1. - self.gamma),
                         eig2 / (1. + self.gamma),
                         -eig1 / (1. - self.gamma),
                         -eig2 / (1. + self.gamma)])
        return self.position + self.einsteinradius * foo
    
# ----------------------------------------------------------------------------

    # magic number 100 (sets precision)
    def critical_curve(self):
        eig1, eig2 = self.eigs()
        npts = 100
        dphi = np.linspace(0.0, 2.0*np.pi, num=npts, endpoint=True)
        cosphi = np.cos(dphi)
        sinphi = np.sin(dphi)
        costwophi = np.cos(2.0*dphi)
        r = self.einsteinradius*(1 - self.gamma*costwophi)/(1.0-self.gamma*self.gamma)
        critcurve = np.outer(np.ones(npts), self.position)
        critcurve += np.outer(r*cosphi, eig1)
        critcurve += np.outer(r*sinphi, eig2)
        return critcurve

# ----------------------------------------------------------------------------

    # magic number 100
    # note computational overkill; it's a fucking circle.
    def radial_caustic(self):
        eig1, eig2 = self.eigs()
        npts = 100
        dphi = np.linspace(0.0, 2.0*np.pi, num=npts, endpoint=True)
        cosphi = np.cos(dphi)
        sinphi = np.sin(dphi)
        caustic = np.outer(np.ones(npts), self.position)
        caustic += np.outer(self.einsteinradius * cosphi, eig1)
        caustic += np.outer(self.einsteinradius * sinphi, eig2)
        return caustic

# ----------------------------------------------------------------------------

    # magic number 100
    def tangential_caustic(self):
        eig1, eig2 = self.eigs()
        npts = 100
        dphi = np.linspace(0.0, 2.0*np.pi, num=npts, endpoint=True)
        sincubedphi = np.sin(dphi)**3
        coscubedphi = np.cos(dphi)**3
        betaplus = 2.0*self.einsteinradius*self.gamma/(1.0+self.gamma)
        betaminus = 2.0*self.einsteinradius*self.gamma/(1.0-self.gamma)
        caustic = np.outer(np.ones(npts), self.position)
        caustic += np.outer(-betaplus*coscubedphi, eig1)
        caustic += np.outer(betaminus*sincubedphi, eig2)
        return caustic

# ----------------------------------------------------------------------------

    # return the number of images you SEE
    # return 1, 2, 3, or 4
    def number_of_images(self, sourceposition):
        if nx.points_inside_poly(np.atleast_2d(sourceposition), self.radial_caustic())[0]:
            if nx.points_inside_poly(np.atleast_2d(sourceposition), self.tangential_caustic())[0]:
                return 4
            return 2
        if nx.points_inside_poly(np.atleast_2d(sourceposition), self.tangential_caustic())[0]:
            return 3
        return 1

# ----------------------------------------------------------------------------

    # only partially written
    def guess_imagepositions(self, sourceposition):
        N = self.number_of_images(sourceposition)
        assert(N > 0 and N < 5)
        if N == 1:
            ipos = sourceposition.copy()
        if N == 2:
            ipos = np.zeros(2, 2)
        if N == 3:
            ipos = np.zeros(3, 2)
        if N == 4:
            ipos = np.zeros(4, 2)
        return ipos

# ----------------------------------------------------------------------------

    # input: a single source position shape (2) and a first guess at N image positions shape (N, 2)
    # output: N image positions shape (N, 2)
    # MAGIC NUMBER: 1e-10
    def refine_imagepositions(self, sourceposition, guessedimagepositions):
        ipos = np.atleast_2d(guessedimagepositions)
        N = len(ipos)
        dspos = np.outer(np.ones(N), sourceposition) - self.sourcepositions(ipos)
        i = 0
        iposlist = [ipos.copy(), ]
        while np.sum(dspos**2) > (N * 1e-10):
            i += 1
            dipos = np.array([np.dot(tens, dsp) for tens, dsp in zip(self.magnificationtensors(ipos), dspos)])
            ipos += dipos
            iposlist.append(ipos.copy())
            dspos = np.outer(np.ones(N), sourceposition) - self.sourcepositions(ipos)
        return iposlist

# ----------------------------------------------------------------------------

    # read the source code
    def imagepositions(self, sourceposition):
        assert(sourceposition.shape == (2, ))
        ipos = self.guess_imagepositions(sourceposition)
        iposlist = self.refine_imagepositions(sourceposition, ipos)
        return iposlist[-1]

# ----------------------------------------------------------------------------

    # input: image positions shape (N, 2)
    # output: source position
    # note outer, sqrt, sum craziness
    def sourcepositions(self, imagepositions):
        ipos = np.atleast_2d(imagepositions)
        dpos = ipos - np.outer(np.ones(len(ipos)), self.position)
        r = np.outer(np.sqrt(np.sum(dpos * dpos, axis=1)), [1,1])
        spos = 1. * ipos
        spos -= self.einsteinradius * dpos / r
        spos[:,0] -= self.gammacos2phi * dpos[:,0]
        spos[:,0] -= self.gammasin2phi * dpos[:,1]
        spos[:,1] += self.gammacos2phi * dpos[:,1]
        spos[:,1] -= self.gammasin2phi * dpos[:,0]
        return spos

 # ----------------------------------------------------------------------------

    # output shape (N, 2, 2)
    def inversemagnificationtensors(self, imagepositions):
        ipos = np.atleast_2d(imagepositions)
        mag = np.zeros((len(ipos), 2, 2))
        mag[:,0,0] = 1.
        mag[:,1,1] = 1.
        dpos = ipos - self.position
        rcubed = np.sum(dpos * dpos, axis=1)**1.5
        if np.min(rcubed) <= 0.:
            print ipos
            print self.position
            print mag
            print self
        assert(np.min(rcubed) > 0.)
        mag[:,0,0] -= self.einsteinradius * dpos[:,1] * dpos[:,1] / rcubed
        mag[:,0,1] += self.einsteinradius * dpos[:,1] * dpos[:,0] / rcubed
        mag[:,1,0] += self.einsteinradius * dpos[:,0] * dpos[:,1] / rcubed
        mag[:,1,1] -= self.einsteinradius * dpos[:,0] * dpos[:,0] / rcubed
        mag[:,0,0] -= self.gammacos2phi
        mag[:,0,1] -= self.gammasin2phi
        mag[:,1,0] -= self.gammasin2phi
        mag[:,1,1] += self.gammacos2phi
        return mag

 # ----------------------------------------------------------------------------

    def magnificationtensors(self, imagepositions):
        return np.array([np.linalg.inv(t) for t in self.inversemagnificationtensors(imagepositions)])

# ----------------------------------------------------------------------------

    # APPROXIMATION: Not yet marginalizing over true source position, true source flux
    # look for "WRONG" in code
    # Note magnification sign insanity
    def ln_prior(self, imagepositions, imagefluxes, positionvariance, fluxvariance, paritycheck=True):
        def ln_Gaussian_1d_zeromean(x, var):
            return -0.5 * np.log(2. * np.pi * var) - 0.5 * x**2 / var
        assert(len(imagepositions) == 4)
        sourcepositions = self.sourcepositions(imagepositions)
        meansourceposition = np.mean(sourcepositions, axis=0) # WRONG
        magtensors = self.magnificationtensors(imagepositions)
        mags = np.array(map(np.linalg.det, magtensors))
        if paritycheck:
            if mags[0] <= 0.:
                return -np.Inf
            if mags[1] >= 0.:
                return -np.Inf
            if mags[2] <= 0.:
                return -np.Inf
            if mags[3] >= 0.:
                return -np.Inf
        dimagepositions = np.array([np.dot(tens, (spos - meansourceposition)) for tens, spos in zip(magtensors, sourcepositions)])
        return np.sum(ln_Gaussian_1d_zeromean(dimagepositions, positionvariance))

# ----------------------------------------------------------------------------

    def sample_prior(self, imagepositions, imagefluxes, positionvariance, fluxvariance, nlink):
        def lnp(pars):
            return self.ln_prior(np.reshape(pars[:8], (4, 2)), None, positionvariance, None)
        pars = np.ravel(imagepositions)
        nwalkers = 100
        ndim     = len(pars)
        initial_position = [pars + 0.01 * np.random.normal(size=ndim) for i in xrange(nwalkers)]
        sampler = dfm.EnsembleSampler(nwalkers, ndim, lnp)
        pos, prob, state = sampler.run_mcmc(initial_position, None, nlink)
        print 'Mean acceptance fraction: ',np.mean(sampler.acceptance_fraction())
        return (sampler.get_chain(), sampler.get_lnprobability())

# ----------------------------------------------------------------------------

    def plot(self, sourcepositions=None, imagepositions=None):
        causticlw = 0.5
        c = self.tangential_caustic().T
        plt.plot(c[0], c[1], 'k', lw=causticlw)
        c = self.radial_caustic().T
        plt.plot(c[0], c[1], 'k', lw=causticlw)
        c = self.critical_curve().T
        plt.plot(c[0], c[1], 'k', lw=2.)
        if sourcepositions is not None:
            spos = np.atleast_2d(sourcepositions)
            plt.scatter(spos[:,0], spos[:,1], c='k', marker='x')
        if imagepositions is not None:
            ipos = np.atleast_2d(imagepositions)
            print sis.magnificationtensors(ipos)
            mags = np.array(map(np.linalg.det, sis.magnificationtensors(ipos)))
            print mags
            print np.linalg.det(sis.magnificationtensors(ipos)[0])
            s = 10. * np.abs(mags)
            I = mags < 0
            if np.sum(I) > 0:
                plt.scatter(ipos[I,0], ipos[I,1], s=s[I], c='k', marker='o', facecolor='none')
            I = mags > 0
            if np.sum(I) > 0:
                plt.scatter(ipos[I,0], ipos[I,1], s=s[I], c='k', marker='s', facecolor='none')
        plt.xlabel('x (arcsec)')
        plt.ylabel('y (arcsec)')
        plt.axes().set_aspect('equal')
        return None

# ============================================================================
# Command line test: sample from prior and plot.

if __name__ == '__main__':
        
    lenspos = [0.5, 0.75]
    b = 1.3 # arcsec
    gamma = 0.2
    phi = 0.2 # rad
    sis = GravitationalLens(lenspos, b, gamma * np.cos(2. * phi), gamma * np.sin(2. * phi))
    spos = np.array([3.5, 0.])
    plt.clf()
    sis.plot()
    plt.savefig('foo.png')
    plt.clf()
    ipos = sis.guess_imagepositions(spos)
    iposlist = sis.refine_imagepositions(spos, ipos)
    a, b, two = np.array(iposlist).shape
    iposlist = np.reshape(iposlist, (a * b, 2))
    print iposlist
    sis.plot(sourcepositions=spos, imagepositions=iposlist)
    plt.savefig('bar.png')

    sys.exit(0)
    caustic = sis.tangential_caustic()
    critcurve = sis.critical_curve()
    posvar = 1.e-3

    imagefluxes = np.array([10., 10., 10., 10.])
    magtensors = sis.magnificationtensors(imagepositions)
    magscalars = np.array([np.linalg.det(ten) for ten in magtensors])
    chain, lnp = sis.sample_prior(imagepositions, None, posvar, None, 10)
    
    plt.clf()
    for i in range(5):
        plt.plot(lnp[i,:])
    plt.xlabel('link number')
    plt.ylabel('ln prior probability')
    plt.savefig('lnp.png')
    for i in range(5):
        if i == 0:
            ipos = imagepositions
            print sis.ln_prior(imagepositions, None, posvar, None)
        else:
            ipos = np.reshape(chain[i,:,-1], (4,2))
        spos = sis.sourcepositions(ipos)
        plt.clf()
        plt.subplot(111, aspect='equal')
        mags = np.abs(map(np.linalg.det, sis.magnificationtensors(ipos)))
        plt.plot(caustic[:,0], caustic[:,1], 'g')
        plt.plot(critcurve[:,0], critcurve[:,1], 'b')
        plt.scatter(ipos[:,0], ipos[:,1], s=mags, c='k', marker='o')
        plt.scatter(spos[:,0], spos[:,1], c='r', marker='o')
        xt = 3 * np.random.uniform(size=(3000,2))
        color = ['m', 'r', 'k', 'g', 'c', 'b']
        colors = [color[sis.number_of_images(x)] for x in xt]
        plt.scatter(xt[:,0], xt[:,1], c=colors, marker='o', alpha=0.5)
        plt.xlim(-2, 3.)
        plt.ylim(-2, 3.)
        plt.xlabel('x (arcsec)')
        plt.ylabel('y (arcsec)')
        plt.savefig('sky-%02d.png' % i)

# ============================================================================

