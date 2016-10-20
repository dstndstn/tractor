# Copyright 2011 David W. Hogg (NYU).  All rights reserved.

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import matplotlib.cm as cm
import numpy as np
from mixture_profiles import MixtureOfGaussians
import scipy.optimize as op
import astrometry.util.file as an
import os

# note MAGIC NUMBERS
def lnLikelihood(image, flux, x, y, psf, invvar):
    star = MixtureOfGaussians(np.array([flux, ]), np.array([x, y]), np.array([0., ])) # delta-function
    cstar = star.convolve(psf)
    modelimage = cstar.evaluate_grid(-5, 7, -5, 7)
    return -0.5 * np.sum(invvar * (image - modelimage)**2)

def cost(pars, notpars):
    flux, x, y = pars
    image, psf, invvar = notpars
    return -1. * lnLikelihood(image, flux, x, y, psf, invvar)

def main():
    # make a simple psf
    amp = np.array([0.9, 0.1])
    mean = np.random.uniform(size=(2, 2))
    var = np.array([2.0, 20.])
    psf = MixtureOfGaussians(amp, mean, var)
    psf.normalize()
    trueimage = psf.evaluate_grid(-5, 7, -5, 7)
    plt.clf()
    plt.imshow(trueimage, interpolation='nearest', cmap='gray')
    plt.savefig('truth.png')

    # do some simple test fitting to check likelihood code
    invvar = 0.01 * np.ones_like(trueimage)
    flux = 1.0
    y = 0.5
    xlist = np.arange(-1.0, 1.0, 0.01)
    lnLlist = np.array([lnLikelihood(trueimage, flux, x, y, psf, invvar) for x in xlist])
    plt.clf()
    plt.plot(xlist, lnLlist, 'k-', alpha=0.5)
    plt.xlabel('model star position')
    plt.ylabel('$\ln$ likelihood')
    plt.savefig('lnL.png')

    # try an optimization
    if False:
        pars = np.array([0.9, -1.0, 1.0])
        notpars = (trueimage, psf, invvar)
        bestpars = op.fmin(cost, pars, args=(notpars, ))
        print(bestpars)

    # now loop over noisiness
    minx = -3.5
    maxx = -1.5
    picklefn = 'fluxes.pickle'
    if not os.path.exists(picklefn):
        logvarlist = np.random.uniform(minx, maxx, size=(3000))
        fluxlist = np.zeros_like(logvarlist)
        for i, logvar in enumerate(logvarlist):
            noise = np.random.normal(size=trueimage.shape)
            var = 10.**logvar
            image = trueimage + var * noise
            invvar = np.ones_like(image) / var / var
            pars = np.array([1.0, 0.0, 0.0])
            notpars = (image, psf, invvar)
            bestpars = op.fmin(cost, pars, args=(notpars, ))
            fluxlist[i] = bestpars[0]
            print(i, logvar, bestpars)
        an.pickle_to_file((logvarlist, fluxlist), picklefn)
    (logvarlist, fluxlist) = an.unpickle_from_file(picklefn)

    # make plots
    plt.clf()
    plt.axhline(1.1, color='k', alpha=0.25)
    plt.axhline(1.0, color='k', alpha=0.25)
    plt.axhline(0.9, color='k', alpha=0.25)
    plt.plot(logvarlist, fluxlist, 'k.', alpha=0.5)
    plt.xlabel('$\log_{10}$ pixel noise variance')
    plt.ylabel('inferred flux')
    plt.xlim(minx, maxx)
    plt.ylim(0.8, 1.2)
    plt.savefig('flux.png')

    medbins = np.arange(minx, maxx+0.0001, 0.25)
    for i in range(len(medbins)-1):
        a, b = medbins[[i, i+1]]
        inside = np.sort(fluxlist[(logvarlist > a) * (logvarlist < b)])
        nin = len(inside)
        plt.plot([a, b], [inside[0.025* nin], inside[0.025* nin]], 'k-', alpha=0.5)
        plt.plot([a, b], [inside[0.16 * nin], inside[0.16 * nin]], 'k-', alpha=0.5)
        plt.plot([a, b], [inside[0.50 * nin], inside[0.50 * nin]], 'k-', alpha=0.5)
        plt.plot([a, b], [inside[0.84 * nin], inside[0.84 * nin]], 'k-', alpha=0.5)
        plt.plot([a, b], [inside[0.975* nin], inside[0.975* nin]], 'k-', alpha=0.5)
    plt.savefig('fluxquant.png')

if __name__ == '__main__':
    main()
