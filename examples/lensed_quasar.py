# ============================================================================

# Copyright 2011 David W. Hogg (NYU) and Phil Marshall (Oxford).
# All rights reserved (for now).

# to-do
# -----
# - write and test lens equation solver: 1  image done, awaiting more
# - write down priors on magnification / mag bias

import numpy as np
# import markovpy as dfm
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
        assert(self.gamma > 0.)
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
    # note computational overkill; it's a circle.
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

    # input: a single source position, shape (2)
    # output: a first guess at N image positions, shape (N, 2)
    def guess_image_positions(self, sourceposition):
        N = self.number_of_images(sourceposition)
        assert(N > 0 and N < 5)
        if N == 1:
            ipos = sourceposition.copy()
        if N == 2:
            dpos = sourceposition.copy() - self.position
            # radial vector, length = thetaE
            blah = self.einsteinradius * dpos / np.sqrt(np.dot(dpos, dpos))
            ipos = np.array([sourceposition.copy() + blah, sourceposition.copy() - blah])
        if N == 3:
            dpos = sourceposition.copy() - self.position
            # tangential vector, length = 5*thetaE to cope with very 
            # wide separation arcs...
            blah = 5.0*self.einsteinradius * dpos / np.sqrt(np.dot(dpos, dpos))
            rot = np.array([[0,1],[-1,0]])
            wffl = np.dot(rot,blah)
            ipos = np.array([sourceposition.copy() - wffl, 
                             sourceposition.copy(), 
                             sourceposition.copy() + wffl])
            # print "guessed image position:",ipos
        if N == 4:
            ipos = self.cross()
        return ipos

# ----------------------------------------------------------------------------

    # input: a single source position shape (2) and a first guess at N image positions shape (N, 2)
    # output: N image positions shape (N, 2)
    # MAGIC NUMBER: tol
    def refine_image_positions(self, sourceposition, guessedimagepositions, tol=1.e-20):
        ipos = np.atleast_2d(guessedimagepositions)
        parities = self.parities(ipos)
        N = len(ipos)
        dspos = np.outer(np.ones(N), sourceposition) - self.source_positions(ipos)
        i = 0
        while np.sum(dspos**2) > tol:
            i += 1
            dipos = np.array([np.dot(tens, dsp) for tens, dsp in zip(self.magnification_tensors(ipos), dspos)])
            ipos = ipos + dipos
            dspos = np.outer(np.ones(N), sourceposition) - self.source_positions(ipos)
            # print "refine_image_positions:",ipos,self.parities(ipos)
                            
        return ipos

# ----------------------------------------------------------------------------

    # solve for image positions given source position.
    # input: source position, shape (2,)
    # output: image positions, shape (N,2)
    def image_positions(self, sourceposition):
        assert(sourceposition.shape == (2, ))
        
        ipos = self.guess_image_positions(sourceposition)
        N = self.number_of_images(sourceposition)
        initialparities = self.parities(ipos)
                
        ipos = self.refine_image_positions(sourceposition, ipos)
        
        fail = False
        if (N == 2) or (N == 4):
            if (np.sum(self.parities(ipos)) != 0):
                print 'image_positions: WARNING: parities wrong, some images have either merged or collapsed:'
                print 'image positions:',ipos
                print 'image magnifications:',self.magnifications(ipos)
                fail = True
        elif (N == 3):
            if (np.sum(self.parities(ipos) - initialparities) != 0):
                print 'image_positions: WARNING: parities wrong, some images have merged or collapsed:'
                print 'image positions:',ipos
                print 'image magnifications:',self.magnifications(ipos)
                fail = True
        # Try refining again from a different guess?
        return ipos,fail

# ----------------------------------------------------------------------------

    # NB: MUST BE SYNCHRONIZED WITH DEFLECTIONS() AND INVERSE_MAGNIFICATION_TENSORS()
    def potentials(self, imagepositions):
        ipos = np.atleast_2d(imagepositions)
        dpos = ipos - np.outer(np.ones(len(ipos)), self.position)
        r = np.sqrt(np.sum(dpos * dpos, axis=1))
        cosphi = dpos[:,0] / r
        sinphi = dpos[:,1] / r
        cos2phi = cosphi * cosphi - sinphi * sinphi
        sin2phi = 2. * cosphi * sinphi
        psis = self.einsteinradius * r
        psis += 0.5 * r * r * self.gammacos2phi * cos2phi
        psis += 0.5 * r * r * self.gammacos2phi * sin2phi
        return psis

    # input: image-plane positions shape (N, 2)
    # output: potential values at those positions shape (N, )
    def time_delays(self, sourceposition, imagepositions):
        N = len(imagepositions)
        dpos = np.outer(np.ones(N), sourceposition) - imagepositions
        return 0.5 * np.sum(dpos**2, axis=1) - self.potentials(imagepositions)

# ----------------------------------------------------------------------------

    # NB: MUST BE SYNCHRONIZED WITH POTENTIALS() AND INVERSE_MAGNIFICATION_TENSORS()
    def deflections(self, imagepositions):
        ipos = np.atleast_2d(imagepositions)
        dpos = ipos - np.outer(np.ones(len(ipos)), self.position)
        r = np.outer(np.sqrt(np.sum(dpos * dpos, axis=1)), [1,1])
        alphas = self.einsteinradius * dpos / r
        alphas[:,0] += self.gammacos2phi * dpos[:,0]
        alphas[:,0] += self.gammasin2phi * dpos[:,1]
        alphas[:,1] -= self.gammacos2phi * dpos[:,1]
        alphas[:,1] += self.gammasin2phi * dpos[:,0]
        return alphas

    # check synchronization between potential and deflection
    # NOT YET WRITTEN
    def test_deflections(self, imagepositions):
        alphas = self.deflections(self, imagepositions)
        psis = self.potentials(self, imagepositions)
        alphas2 = np.zeros_like(alphas)
        dx = 0.001 * self.einsteinradius
        return None

    # input: image positions shape (N, 2)
    # output: source position
    # note outer, sqrt, sum craziness
    def source_positions(self, imagepositions):
        return imagepositions - self.deflections(imagepositions)

 # ----------------------------------------------------------------------------

    # output shape (N, 2, 2)
    # NB: MUST BE SYNCHRONIZED WITH POTENTIALS() AND DEFLECTIONS()
    def inverse_magnification_tensors(self, imagepositions):
        ipos = np.atleast_2d(imagepositions)
        mag = np.zeros((len(ipos), 2, 2))
        mag[:,0,0] = 1.
        mag[:,1,1] = 1.
        # print "inverse_magnification_tensors: ipos = ",ipos
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

    def magnification_tensors(self, imagepositions):
        return np.array([np.linalg.inv(t) for t in self.inverse_magnification_tensors(imagepositions)])

    # crazy if you run this and magnificationtensors in the same code; think caching
    def magnifications(self, imagepositions):
        return 1. / np.array(map(np.linalg.det, self.inverse_magnification_tensors(imagepositions)))

    # crazy if you run this and magnificationtensors in the same code; think caching
    def parities(self, imagepositions):
        return np.sign(self.magnifications(imagepositions))

# ----------------------------------------------------------------------------

    # APPROXIMATION: Not yet marginalizing over true source position, true source flux
    # look for "WRONG" in code
    # Note magnification sign insanity
    def ln_prior(self, imagepositions, imagefluxes, positionvariance, fluxvariance, paritycheck=True):
        def ln_Gaussian_1d_zeromean(x, var):
            return -0.5 * np.log(2. * np.pi * var) - 0.5 * x**2 / var
        assert(len(imagepositions) == 4)
        sourcepositions = self.source_positions(imagepositions)
        meansourceposition = np.mean(sourcepositions, axis=0) # WRONG
        magtensors = self.magnification_tensors(imagepositions)
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

    # if timedelay then plot time-delay surface
    # several MAGIC NUMBERS
    def plot(self, sourcepositions=None, imagepositions=None, timedelay=False):
        causticlw = 0.5
        c = self.tangential_caustic().T
        plt.plot(c[0], c[1], 'k', lw=causticlw)
        c = self.radial_caustic().T
        plt.plot(c[0], c[1], 'k', lw=causticlw)
        c = self.critical_curve().T
        plt.plot(c[0], c[1], 'k', lw=2.)
        if sourcepositions is not None:
            spos = np.atleast_2d(sourcepositions)
            # print 'plot: plotting spos:', spos
            plt.scatter(spos[:,0], spos[:,1], c='k', marker='x', lw=causticlw)
        if imagepositions is not None:
            ipos = np.atleast_2d(imagepositions)
            mags = np.array(map(np.linalg.det, self.magnification_tensors(ipos)))
            s = 20. * np.sqrt(np.abs(mags))
            I = mags < 0
            if np.sum(I) > 0:
                plt.scatter(ipos[I,0], ipos[I,1], s=s[I], c='k', marker='o', facecolor='none')
            I = mags > 0
            if np.sum(I) > 0:
                plt.scatter(ipos[I,0], ipos[I,1], s=s[I], c='k', marker='s', facecolor='none')
        plt.xlabel('x (arcsec)')
        plt.ylabel('y (arcsec)')
        plt.axes().set_aspect('equal')
        if timedelay:
            print 'plotting time delay for', spos[0]
            dts = self.time_delays(spos[0], ipos)
            ta = np.min(dts) - self.einsteinradius**2
            tb = np.max(dts) + self.einsteinradius**2
            tcontours = np.arange(ta, tb, 0.01 * self.einsteinradius**2)
            xa, xb = plt.xlim()
            ya, yb = plt.ylim()
            xg, yg = np.meshgrid(np.arange(xa, xb, 0.01 * self.einsteinradius),
                                np.arange(ya, yb, 0.01 * self.einsteinradius))
            ig = np.array(zip(np.ravel(xg), np.ravel(yg)))
            tg = np.reshape(self.time_delays(spos[0], ig), xg.shape)
            plt.contour(xg, yg, tg, tcontours, alpha=0.5)
        return None

# ============================================================================
# Command line test: sample from prior and plot.

if __name__ == '__main__':
        
    # Initialise lens:
    
    # test = 'high_mag'
    test = 'mergers'

    # config = 'major_cusp'
    config = 'minor_cusp'
    # config = 'naked_cusp'
    print "Requested image configuration is",config

    lenspos = [0.5, 0.75]
    b = 1.3 # arcsec
    if config == 'naked_cusp':
      gamma = 0.5
    else:
      gamma = 0.2
    phi = 0.2 # rad
    sis = GravitationalLens(lenspos, b, gamma*np.cos(2.0*phi), gamma*np.sin(2.0*phi))

    plt.clf()
    sis.plot()
    foofile = 'foo.png'
    plt.savefig(foofile)
    print "Lens outline plotted in",foofile
    
    nsample = 100
    print "Drawing",nsample,"sample image positions..."
    ipos = np.zeros((nsample, 2))

    if config == 'minor_cusp':
      ipos[:,0] = np.random.uniform(1.5, 3.0, size=nsample)
      ipos[:,1] = np.random.uniform(0.5, 1.5, size=nsample)
    elif config == 'major_cusp':
      ipos[:,0] = np.random.uniform(0.5, 1.0, size=nsample)
      ipos[:,1] = np.random.uniform(-0.6, -1.0, size=nsample)
    elif config == 'naked_cusp':
      ipos[:,0] = np.random.uniform(0.5, 1.0, size=nsample)
      ipos[:,1] = np.random.uniform(-0.7, -2.0, size=nsample)
    else:
      ipos[:,0] = np.random.uniform(-2.0, 3.0, size=nsample)
      ipos[:,1] = np.random.uniform(-2.0, 3.0, size=nsample)
  
    if test == 'high_mag':
      # High mag test - find high magnification image, its source position, and 
      # then solve for the  image positions.
      mags = sis.magnifications(ipos)
      print "Highest mag sample has mag =",mags[np.argmax(mags)]
      print "Image position is",ipos[np.argmax(mags)]
      spos = sis.source_positions(ipos[np.argmax(mags)])[0]
      fail = True

    elif test == 'mergers':
      # Mergers test - find an image position that does not solve properly, by
      # testing target and solved image numbers:
      spoz = sis.source_positions(ipos)
      nimz = np.ones(nsample)
      for i in range(nsample):
        nimz[i] = sis.number_of_images(spoz[i])
      if config == 'naked_cusp':
        target = 3
      else:
        target = 4
      index = np.where(nimz == target)
      ipoz = ipos[index]
      spoz = spoz[index]
      fail = False
      total = len(spoz)
      count = 0
      for spos in spoz:
        ipos,fail = sis.image_positions(spos)
        count += 1
        if fail:
          print "Image solve failure (after ",int(count/(0.01*total)),"% of samples)"
          break

    if fail:
      print "Source position is",spos
      nim = sis.number_of_images(spos)
      print "No. of expected images:",nim
      ipos,dummy = sis.image_positions(spos)
      print "Solved image positions:",ipos
      print "Image magnifications:",sis.magnifications(ipos)
      spos2 = sis.source_positions(ipos)
      print "Corresponding source positions:",spos2

      plt.clf()
      sis.plot(sourcepositions=np.append(np.atleast_2d(spos), spos2, axis=0), imagepositions=ipos, timedelay=True)
      barfile = 'bar.png'
      plt.savefig(barfile)
      print "Images and source(s) plotted in",barfile
   
    else:
      print "All tests passed OK"
     
# if False:
#     caustic = sis.tangential_caustic()
#     critcurve = sis.critical_curve()
#     posvar = 1.e-3
# 
#     imagefluxes = np.array([10., 10., 10., 10.])
#     magtensors = sis.magnificationtensors(imagepositions)
#     magscalars = np.array([np.linalg.det(ten) for ten in magtensors])
#     chain, lnp = sis.sample_prior(imagepositions, None, posvar, None, 10)
#     
#     plt.clf()
#     for i in range(5):
#         plt.plot(lnp[i,:])
#     plt.xlabel('link number')
#     plt.ylabel('ln prior probability')
#     plt.savefig('lnp.png')
#     for i in range(5):
#         if i == 0:
#             ipos = imagepositions
#             print sis.ln_prior(imagepositions, None, posvar, None)
#         else:
#             ipos = np.reshape(chain[i,:,-1], (4,2))
#         spos = sis.sourcepositions(ipos)
#         plt.clf()
#         plt.subplot(111, aspect='equal')
#         mags = np.abs(map(np.linalg.det, sis.magnificationtensors(ipos)))
#         plt.plot(caustic[:,0], caustic[:,1], 'g')
#         plt.plot(critcurve[:,0], critcurve[:,1], 'b')
#         plt.scatter(ipos[:,0], ipos[:,1], s=mags, c='k', marker='o')
#         plt.scatter(spos[:,0], spos[:,1], c='r', marker='o')
#         xt = 3 * np.random.uniform(size=(3000,2))
#         color = ['m', 'r', 'k', 'g', 'c', 'b']
#         colors = [color[sis.number_of_images(x)] for x in xt]
#         plt.scatter(xt[:,0], xt[:,1], c=colors, marker='o', alpha=0.5)
#         plt.xlim(-2, 3.)
#         plt.ylim(-2, 3.)
#         plt.xlabel('x (arcsec)')
#         plt.ylabel('y (arcsec)')
#         plt.savefig('sky-%02d.png' % i)
# 
# ============================================================================
