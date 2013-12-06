if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

# for PlotSequence
from astrometry.util.plotutils import *

from tractor import *
from tractor.sdss_galaxy import *

import emcee

if __name__ == '__main__':
    # Great3 ground-based images have this pixel scale.
    pixscale = 0.2
    # Great3 postage stamp size
    SS = 48


    ps = PlotSequence('great')

    # which input data?
    #branch = 'great3/multiepoch/ground/constant'
    #food = 'dimsum'
    branch = 'great3/control/ground/constant'
    food = 'vanilla'
    subfield = 0
    epoch = 0

    tag = '%03i-%i' % (subfield, epoch)
    
    # paths
    imgfn  = os.path.join(branch, 'image-%s.fits' % tag)
    starfn = os.path.join(branch, 'starfield_image-%s.fits' % tag)

    img = fitsio.read(imgfn)
    print 'Image size', img.shape
    stars = fitsio.read(starfn)
    print 'Starfield size', stars.shape

    # first star is "centered" in the 48x48 subimage (not centered on a pixel)
    star = stars[:SS,:SS]
    print 'Star shape', star.shape
    
    # estimate noise in image via Blanton's difference between 5-pixel
    # offset pixel pairs for a subset of pixels; median abs diff.
    diffs = img[:-5:10,:-5:10] - img[5::10,5::10]
    mad = np.median(np.abs(diffs).ravel())
    # convert to Gaussian -- sqrt(2) because our diffs are the differences of
    # deviations of two pixels.
    sig1 = 1.4826 * mad / np.sqrt(2.)
    print 'MAD', mad, '-> sigma', sig1

    # histogram pixel values and noise estimate.
    plt.clf()
    n,b,p = plt.hist(img.ravel(), 100, range=(-0.1, 0.1),
                     histtype='step', color='r')
    xx = np.linspace(-0.1, 0.1, 500)
    plt.plot(xx, max(n) * np.exp(-xx**2 / (2.*sig1**2)), 'k-')
    plt.xlim(-0.1, 0.1)
    plt.title('Pixel histogram for image')
    ps.savefig()

    # histogram the total fluxes per postage stamp
    fluxes = []
    for stampy in range(100):
        for stampx in range(100):
            fluxes.append(np.sum(img[stampy*SS:(stampy+1)*SS, stampx*SS:(stampx+1)*SS]))
    plt.clf()
    plt.hist(fluxes, 50)
    plt.title('Sum of flux per postage stamp')
    ps.savefig()

    # Grab one postage stamp
    stampx,stampy = 0,0

    img = img[stampy*SS:(stampy+1)*SS, stampx*SS:(stampx+1)*SS]

    # image plotting args
    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-2*sig1, vmax=3*sig1)
    plt.clf()
    plt.imshow(img, **ima)
    plt.title('Image data')
    ps.savefig()
    
    # Create tractor PSF model for the starfield image
    star /= star.sum()
    # do an Expectation Maximization fit of the postage stamp
    psf = GaussianMixturePSF.fromStamp(star)
    # the first star is centered "between" pixels -- shift it.
    psf.shiftBy(0.5, 0.5)

    # Render the model PSF to check that it looks okay
    psfmodel = psf.getPointSourcePatch(0., 0., radius=24)
    print 'PSF model', psfmodel.shape
    psfmodel /= psfmodel.patch.sum()

    # Plot star postage stamp and model
    plt.clf()
    plt.subplot(1,2,1)
    psfima = dict(interpolation='nearest', origin='lower',
                  cmap='gray', vmin=-6, vmax=0)
    plt.imshow(np.log10(star), **psfima)
    plt.title('PSF stamp')
    plt.subplot(1,2,2)
    plt.imshow(np.log10(psfmodel.patch), **psfima)
    plt.title('PSF model')
    ps.savefig()

    # create tractor Image object.
    tim = Image(data=img, invvar=np.ones_like(img) * (1./sig1**2),
                psf=psf, wcs=NullWCS(pixscale=pixscale), sky=ConstantSky(0.),
                photocal=LinearPhotoCal(1.),
                name='%s %s %i,%i' % (food, tag, stampx, stampy),
                domask=False, zr=[-2.*sig1, 3.*sig1])

    # Create an initial galaxy model object.
    re, ab, phi = 1., 0.5, 0.
    gal = ExpGalaxy(PixPos(SS/2-1., SS/2-1.), Flux(1.), re, ab, phi)
    print 'Initial', gal

    # Create Tractor object from list of images and list of sources
    tractor = Tractor([tim], [gal])

    # Plot initial model image
    mod = tractor.getModelImage(0)
    noise = np.random.normal(size=img.shape) * sig1
    imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
                 vmin=-3, vmax=3)
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(img, **ima)
    plt.title('Image')
    plt.subplot(2,2,2)
    plt.imshow(mod, **ima)
    plt.title('Initial model')
    plt.subplot(2,2,3)
    plt.imshow(mod+noise, **ima)
    plt.title('Model + noise')
    plt.subplot(2,2,4)
    # show mod - img to match "red-to-blue" colormap.
    plt.imshow(-(img - mod)/sig1, **imchi)
    plt.title('Chi')
    ps.savefig()

    print 'All params:'
    print tractor.printThawedParams()
    print
        
    # Freeze all the image calibration parameters.
    tractor.freezeParam('images')

    # Freeze the galaxy position for the initial optimization
    #gal.freezeParam('pos')
    
    # Do a few rounds of optimization (each .optimize() is a single
    # linearized least squares step.
    print 'Optimizing params:'
    print tractor.printThawedParams()
    for i in range(10):
        dlnp,X,alpha = tractor.optimize()
        print 'dlnp', dlnp
        print 'alpha', alpha
    print 'Optimized:', gal

    # Plot the optimized model
    mod = tractor.getModelImage(0)
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(img, **ima)
    plt.title('Image')
    plt.subplot(2,2,2)
    plt.imshow(mod, **ima)
    plt.title('Opt Model')
    plt.subplot(2,2,3)
    plt.imshow(mod+noise, **ima)
    plt.title('Model + noise')
    plt.subplot(2,2,4)
    # show mod - img to match "red-to-blue" colormap.
    plt.imshow(-(img - mod)/sig1, **imchi)
    plt.title('Chi')
    ps.savefig()

    # Now thaw the positions and sample...
    gal.thawParam('pos')

    # Initial parameter vector:
    p0 = np.array(tractor.getParams())
    ndim = len(p0)
    # number of walkers
    nw = max(50, 2*ndim)
    print 'ndim', ndim
    print 'nw', nw
    nthreads = 1

    # Create emcee sampler
    sampler = emcee.EnsembleSampler(nw, ndim, tractor, threads=nthreads)

    # Jitter the walker parameter values according to their
    # (hard-coded) step sizes.
    steps = np.array(tractor.getStepSizes())
    # Initial parameters for walkers
    pp0 = np.vstack([p0 + 1e-2 * steps * np.random.normal(size=len(steps))
                     for i in range(nw)])

    alllnp = []
    allp = []
    
    lnp = None
    pp = pp0
    rstate = None
    for step in range(200):
        print 'Taking step', step
        #print 'pp shape', pp.shape
        pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        print 'Max lnprob:', np.max(lnp)
        #print 'lnprobs:', lnp
        # store all the params
        alllnp.append(lnp.copy())
        allp.append(pp.copy())

    # Plot logprobs
    plt.clf()
    plt.plot(alllnp, 'k', alpha=0.5)
    mx = np.max([p.max() for p in alllnp])
    plt.ylim(mx-20, mx+5)
    plt.title('logprob')
    ps.savefig()

    # Plot parameter distributions
    allp = np.array(allp)
    print 'All params:', allp.shape
    for i,nm in enumerate(tractor.getParamNames()):
        pp = allp[:,:,i].ravel()
        lo,hi = [np.percentile(pp,x) for x in [5,95]]
        mid = (lo + hi)/2.
        lo = mid + (lo-mid)*2
        hi = mid + (hi-mid)*2
        plt.clf()
        plt.subplot(2,1,1)
        plt.hist(pp, 50, range=(lo,hi))
        plt.xlim(lo,hi)
        plt.subplot(2,1,2)
        plt.plot(allp[:,:,i], 'k-', alpha=0.5)
        plt.xlabel('emcee step')
        plt.ylim(lo,hi)
        plt.suptitle(nm)
        ps.savefig()
