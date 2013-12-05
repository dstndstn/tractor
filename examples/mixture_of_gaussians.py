# Copyright 2011 David W. Hogg (NYU) and Phillip J. Marshall (Oxford).
# All rights reserved.

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import numpy as np
import pyfits as pf
import scipy.optimize as op
import pylab as plt

class gaussian_2d():

    # pre-compute lots, because we *assume* that any instantiated gaussian will get evaluated
    def __init__(self, mean, var):
        self.mean = np.array(mean)
        assert(self.mean.shape == (2, ))
        self.var = np.array(var)
        assert(self.var.shape == (2, 2))
        assert(self.var[0,1] == self.var[1,0])
        self.det = self.var[0, 0] * self.var[1,1] - self.var[0,1] * self.var[1,0]
        self.invdet = 1. / self.det
        self.invvar = np.zeros((2,2))
        self.invvar[0,0] = self.invdet * self.var[1,1]
        self.invvar[0,1] = -1. * self.invdet * self.var[0,1]
        self.invvar[1,0] = -1. * self.invdet * self.var[1,0]
        self.invvar[1,1] = self.invdet * self.var[0,0]
        self.norm = np.sqrt(self.invdet) / (2. * np.pi)
        return None

    def evaluate(self, x, y):
        dx = x - self.mean[0]
        dy = y - self.mean[1]
        return self.norm * np.exp(-0.5 * (
                dx * dx * self.invvar[0,0] +
                dx * dy * self.invvar[0,1] +
                dy * dx * self.invvar[1,0] +
                dy * dy * self.invvar[1,1]))

    def plot(self):
        plt.plot(self.mean[1], self.mean[0], 'cx')
        return None

class mixture_of_gaussians():

    def __init__(self, amps, gaussians):
        self.amps = np.array(amps)
        self.K = len(self.amps)
        self.gaussians = np.array(gaussians)
        assert(len(gaussians) == self.K)
        return None

    def __getitem__(self, k):
        return self.amps[k], self.gaussians[k]

    def copy(self):
        return mixture_of_gaussians(self.amps, self.gaussians)

    def evaluate(self, x, y):
        return np.sum([a * g.evaluate(x, y) for a, g in self], axis=0)

    def synthesize(self, data):
        return self.evaluate(data.x, data.y)

    def chi(self, data):
        return (data.data - self.synthesize(data)) * data.invvar

    def chisq(self, data):
        return np.sum(self.chi(data)**2)

    def plot(self):
        for a, g in self:
            g.plot()
        return None

    def pars(self):
        pars = np.zeros((self.K * 7, ))
        i = 0
        for a, g in self:
            j = i + 1
            pars[i:j] = a
            i = j
            j = i + 2
            pars[i:j] = g.mean
            i = j
            j = i + 4
            pars[i:j] = np.reshape(g.invvar, (4, ))
            i = j
        return pars

    def set_parameters_from_pars(self, pars):
        i = 0
        for k, (a, g) in enumerate(self):
            j = i + 1
            self.amps[k] = pars[i]
            i = j
            j = i + 2
            g.mean[:] = pars[i:j]
            i = j
            j = i + 4
            g.invvar[:,:] = np.reshape(pars[i:j], (2, 2))
            i = j
        return None

    # op.leastsq CRUSHES op.fmin, duh!
    def optimize_leastsq(self, data):
        def chivec(pars):
            self.set_parameters_from_pars(pars)
            c = self.chi(data).flatten()
            print np.sum(c*c)
            return c
        firstpars = self.pars()
        bestpars, foo = op.leastsq(chivec, firstpars, maxfev=1000000, ftol=1.e-7)
        self.set_parameters_from_pars(bestpars)
        return None

    def optimize_fmin(self, data):
        def cost(pars):
            self.set_parameters_from_pars(pars)
            c = self.chisq(data)
            print c
            return c
        firstpars = self.pars()
        op.fmin(cost, firstpars, maxfun=np.Inf, maxiter=np.Inf, ftol=1.e-5)
        self.set_parameters_from_pars(bestpars)
        return None

    optimize = optimize_leastsq

class panstarrs_data():

    def __init__(self, datafn, varfn):
        self.data = pf.open(datafn)[0].data
        self.var = pf.open(varfn)[0].data
        self.invvar = 1. / self.var
        self.invvar[np.isnan(self.var)] = 0.
        self.invvar[self.var <= 0.] = 0.
        self.data[np.isnan(self.data)] = 0.
        self.data[self.invvar == 0.] = 0.
        nx, ny = self.data.shape
        self.x, self.y = np.mgrid[0:nx,0:ny]
        return None

if __name__ == '__main__':

    d = panstarrs_data('H1413+117_10x10arcsec_55377.34051_z_sci.fits',
                       'H1413+117_10x10arcsec_55377.34051_z_var.fits')
    imshow_opts = dict(interpolation='nearest', origin='lower')
    plt.gray()
    plt.clf()
    plt.imshow(d.data, **imshow_opts)
    plt.savefig('data.png')
    plt.clf()
    plt.imshow(d.invvar, **imshow_opts)
    plt.savefig('invvar.png')
    for K in range(1,5):
        g2 = gaussian_2d([23., 28.], [[3., 0.], [0., 3.]])
        g1 = gaussian_2d([23., 23.], [[3., 0.], [0., 3.]])
        g3 = gaussian_2d([28., 28.], [[3., 0.], [0., 3.]])
        g4 = gaussian_2d([28., 23.], [[3., 0.], [0., 3.]])
        amps = [10000., 10000., 10000., 10000.]
        gaussians = [g1, g2, g3, g4]
        mix = mixture_of_gaussians(amps[0:K], gaussians[0:K])
        mix.optimize(d)
        plt.clf()
        plt.imshow(mix.synthesize(d), **imshow_opts)
        plt.savefig('synth-%d.png' % K)
        plt.clf()
        plt.imshow(mix.chi(d), **imshow_opts)
        plt.savefig('chi-%d.png' % K)
