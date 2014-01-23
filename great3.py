if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

# for PlotSequence
from astrometry.util.plotutils import *
from astrometry.util.file import *
from astrometry.util.fits import *

from tractor import *
from tractor.sdss_galaxy import *

import emcee

if __name__ == '__main__':
    import sys
    import logging
    #lvl = logging.INFO
    lvl = logging.WARN
    #    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    gfn = 'galparams-100.fits'
    if os.path.exists(gfn):
        ps = PlotSequence('gals')
        T = fits_table(gfn)
        print 'Read', len(T), 'from', gfn
        plt.figure(figsize=(12,8))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.92,
                            wspace=0.25, hspace=0.25)
        rows,cols = 2,3
        plt.clf()
        plt.suptitle('Galaxy properties from "vanilla" branch: 10,000 galaxies')
        plt.subplot(rows,cols, 1)
        plt.hist(T.re, 50, histtype='step', color='b')
        plt.xlim(0, 1.4)
        plt.xlabel('r_e (arcsec)')
        plt.yticks([])
        
        plt.subplot(rows,cols, 2)
        plt.hist(T.flux, 50, histtype='step', color='b')
        plt.xlabel('flux')
        plt.xlim(0, 200)
        plt.yticks([])

        plt.subplot(rows,cols, 3)
        plt.hist(T.e1, 50, histtype='step', color='r')
        plt.hist(T.e2, 50, histtype='step', color='b')
        plt.xlabel('e1, e2')
        plt.yticks([])

        plt.subplot(rows,cols, 4)
        plt.hist(np.hypot(T.e1, T.e2), 50, histtype='step', color='b')
        plt.xlabel('e')
        plt.yticks([])

        plt.subplot(rows,cols, 5)
        theta = np.rad2deg(np.arctan2(T.e2, T.e1)) / 2.
        plt.hist(theta, 50, histtype='step', color='b', range=(-90,90))
        plt.xlabel('theta (deg)')
        plt.xlim(-90, 90)
        plt.yticks([])

        ps.savefig()
        sys.exit(0)
        
    # Great3 ground-based images have this pixel scale.
    pixscale = 0.2
    # Great3 postage stamp size
    SS = 48

    ps = PlotSequence('great')
    ps.suffixes = ['png']#, 'pdf']
    
    # which input data?
    #branch = 'great3/multiepoch/ground/constant'
    #food = 'dimsum'
    branch = 'great3/control/ground/constant'
    food = 'vanilla'
    subfield = 1
    epoch = 0

    tag = '%03i-%i' % (subfield, epoch)
    
    # paths
    imgfn  = os.path.join(branch, 'image-%s.fits' % tag)
    starfn = os.path.join(branch, 'starfield_image-%s.fits' % tag)

    #imgfn = 'demo2.fits'
    #starfn = 'demo2_epsf.fits'

    fns = [imgfn, starfn]
    if not all([os.path.exists(x) for x in fns]):
        try:
            os.makedirs(branch)
        except:
            pass
        for fn in fns:
            if not os.path.exists(fn):
                os.system('wget -O %s http://broiler.astrometry.net/~dstn/%s' % (fn, fn))

    img = fitsio.read(imgfn).astype(np.float32)
    print 'Image size', img.shape
    stars = fitsio.read(starfn).astype(np.float32)
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

    plt.figure(figsize=(5,5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.05, top=0.92,
                        wspace=0.25, hspace=0.25)
    
    # histogram pixel values and noise estimate.
    plt.clf()
    lo,hi = -5.*sig1, 5.*sig1
    n,b,p = plt.hist(img.ravel(), 100, range=(lo, hi),
                     histtype='step', color='r')
    xx = np.linspace(lo, hi, 500)
    plt.plot(xx, max(n) * np.exp(-xx**2 / (2.*sig1**2)), 'k-')
    plt.xlim(lo, hi)
    plt.title('Pixel histogram for image')
    ps.savefig()

    # histogram the total fluxes per postage stamp
    fluxes = []
    for stampy in range(100):
        for stampx in range(100):
            fluxes.append(np.sum(img[stampy*SS:(stampy+1)*SS,
                                     stampx*SS:(stampx+1)*SS]))
    plt.clf()
    plt.hist(fluxes, 50)
    plt.title('Sum of flux per postage stamp')
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
    plt.subplot(2,2,1)
    psfima = dict(interpolation='nearest', origin='lower',
                  cmap='gray', vmin=-6, vmax=0)
    plt.imshow(np.log10(star), **psfima)
    plt.title('log(PSF stamp)')
    plt.subplot(2,2,2)
    plt.imshow(np.log10(psfmodel.patch), **psfima)
    plt.title('log(PSF model)')

    plt.subplot(2,2,3)
    sh,sw = star.shape
    pimg = np.zeros_like(star)
    m2 = psf.getPointSourcePatch(sw/2.-0.5, sh/2.-0.5, radius=24)
    m2.addTo(pimg)
    dpsf = pimg - star
    mx = np.abs(dpsf).max()
    plt.imshow(dpsf, interpolation='nearest', origin='lower',
               vmin=-mx, vmax=mx, cmap='RdBu')
    plt.title('Model - Stamp')
    plt.colorbar()
    ps.savefig()

    # image plotting args
    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-2*sig1, vmax=3*sig1)

    dosample = False

    optparams = []
    
    for stamp in range(100*100):
        print 'Postage stamp', stamp
        #if stamp > 100:
        #    break
        # Grab one postage stamp
        stampx,stampy = stamp % 100, stamp / 100
        subimg = img[stampy*SS:(stampy+1)*SS, stampx*SS:(stampx+1)*SS]

        plots = stamp < 5
        
        # if plots:
        #     plt.clf()
        #     plt.imshow(subimg, **ima)
        #     plt.title('Image data')
        #     ps.savefig()

        # create tractor Image object.
        tim = Image(data=subimg, invvar=np.ones_like(subimg)*(1./sig1**2),
                    psf=psf, wcs=NullWCS(pixscale=pixscale),
                    sky=ConstantSky(0.),
                    photocal=LinearPhotoCal(1.),
                    name='%s %s %i,%i' % (food, tag, stampx, stampy),
                    domask=False, zr=[-2.*sig1, 3.*sig1])

        # Create an initial galaxy model object.
        e = EllipseE(0., 0., 0.)
        # rough flux estimate (sky = 0)
        flux = np.sum(subimg)
        #print 'Flux:', flux
        flux = Flux(flux)
        flux.stepsize = sig1
        gal = ExpGalaxy(PixPos(SS/2-1., SS/2-1.), flux, e)
        #print 'Initial', gal

        # Create Tractor object from list of images and list of sources
        tractor = Tractor([tim], [gal])

        if plots:
            # Plot initial model image
            mod = tractor.getModelImage(0)
            noise = np.random.normal(size=subimg.shape) * sig1
            imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
                         vmin=-3, vmax=3)
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(subimg, **ima)
            plt.title('Image')
            plt.xticks([]); plt.yticks([])
            plt.subplot(2,2,2)
            plt.imshow(mod, **ima)
            plt.title('Initial model')
            plt.xticks([]); plt.yticks([])
            plt.subplot(2,2,3)
            plt.imshow(mod+noise, **ima)
            plt.title('Model + noise')
            plt.xticks([]); plt.yticks([])
            plt.subplot(2,2,4)
            # show mod - img to match "red-to-blue" colormap.
            plt.imshow(-(subimg - mod)/sig1, **imchi)
            plt.xticks([]); plt.yticks([])
            plt.title('Chi')
            ps.savefig()

        # print 'All params:'
        # print tractor.printThawedParams()
        # print
        
        # Freeze all the image calibration parameters.
        tractor.freezeParam('images')

        # Freeze the galaxy position for the initial optimization
        #gal.freezeParam('pos')
    
        # Do a few rounds of optimization (each .optimize() is a single
        # linearized least squares step.
        # print 'Optimizing params:'
        # print tractor.printThawedParams()
        for i in range(10):
            #print 'Optimization step', i
            dlnp,X,alpha = tractor.optimize(shared_params=False)
            #print 'dlnp', dlnp
            #print 'alpha', alpha
            #print 'Optimized:', gal
            if dlnp < 0.1:
                break
            
        if plots:
            # Plot the optimized model
            mod = tractor.getModelImage(0)
            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(subimg, **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('Image')
            plt.subplot(2,2,2)
            plt.imshow(mod, **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('Opt Model')
            plt.subplot(2,2,3)
            plt.imshow(mod+noise, **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('Model + noise')
            plt.subplot(2,2,4)
            # show mod - img to match "red-to-blue" colormap.
            plt.imshow(-(subimg - mod)/sig1, **imchi)
            plt.xticks([]); plt.yticks([])
            plt.title('Chi')
            ps.savefig()

        optparams.append(gal.getParams() + [tractor.getLogProb()])


        if stamp % 100 == 99:
        
            op = np.array(optparams)
            print 'optparams:', op.shape
            pnames = gal.getParamNames()
            print 'Param names:', pnames

            re = np.exp(op[:,3])
            fakee1 = op[:,4]
            fakee2 = op[:,5]
            theta = np.arctan2(fakee2, fakee1) / 2.
            e = np.sqrt(fakee1**2 + fakee2**2)
            e = 1. - np.exp(-e)
            e1 = e * np.cos(2.*theta)
            e2 = e * np.sin(2.*theta)
            
            T = fits_table()
            T.fakee1 = fakee1
            T.fakee2 = fakee2
            T.e1 = e1
            T.e2 = e2
            T.re = re
            T.x = op[:,0]
            T.y = op[:,1]
            T.flux = op[:,2]
            T.logprob = op[:,6]
            
            T.writeto('galparams-%03i.fits' % ((stamp+1)/100))
            
            plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.92,
                                wspace=0.25, hspace=0.25)

            plt.clf()
            plt.subplot(2,2,1)
            plt.hist(re, 50, histtype='step')
            plt.xlabel('re (arcsec)')

            plt.subplot(2,2,2)
            plt.hist(e1, 50, histtype='step')
            plt.xlabel('e1')

            plt.subplot(2,2,3)
            plt.hist(e2, 50, histtype='step')
            plt.xlabel('e2')
            
            plt.subplot(2,2,4)
            plt.plot(e1, e2, 'b.', alpha=0.5)
            plt.xlabel('e1')
            plt.ylabel('e2')
            ps.savefig()
        

        
        if not dosample:
            continue

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
        pp0 = np.vstack([p0 + 1e-1 * steps * np.random.normal(size=len(steps))
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
        burn = 50
        print 'All params:', allp.shape
        for i,nm in enumerate(tractor.getParamNames()):
            pp = allp[:,:,i].ravel()
            lo,hi = [np.percentile(pp,x) for x in [5,95]]
            mid = (lo + hi)/2.
            lo = mid + (lo-mid)*2
            hi = mid + (hi-mid)*2
            plt.clf()
            plt.subplot(2,1,1)
            plt.hist(allp[burn:,:,i].ravel(), 50, range=(lo,hi))
            plt.xlim(lo,hi)
            plt.subplot(2,1,2)
            plt.plot(allp[:,:,i], 'k-', alpha=0.5)
            plt.xlabel('emcee step')
            plt.ylim(lo,hi)
            plt.suptitle(nm)
            ps.savefig()
    
        # Plot a sampling of ellipse parameters
        ellp = allp[-1, :, -3:]
        print 'ellp:', ellp.shape
        E = EllipseE(0.,0.,0.)
        angle = np.linspace(0., 2.*np.pi, 100)
        xx,yy = np.sin(angle), np.cos(angle)
        xy = np.vstack((xx,yy)) * 3600.
        plt.clf()
        for ell in ellp:
            E.setParams(ell)
            T = E.getRaDecBasis()
            txy = np.dot(T, xy)
            plt.plot(txy[0,:], txy[1,:], '-', color='b', alpha=0.1)
        plt.title('sample of galaxy ellipses')
        plt.xlabel('dx (arcsec)')
        plt.ylabel('dy (arcsec)')
        mx = np.max(np.abs(plt.axis()))
        plt.axis([-mx,mx,-mx,mx])
        plt.axis('scaled')
        ps.savefig()
            
        # # Plot (some) parameter pairs
        # re = allp[burn:,:,3].ravel()
        # ab = allp[burn:,:,4].ravel()
        # rerange = 1.95, 2.3
        # abrange = 0.55, 0.75
        # 
        # reticks = [2.0, 2.1, 2.2, 2.3]
        # abticks = [0.6, 0.7]
        # 
        # plt.clf()
        # plt.subplot(2,2,1)
        # plt.hist(re, 25, range=rerange)
        # plt.xlabel('r_e')
        # plt.xlim(rerange)
        # plt.xticks(reticks)
        # plt.yticks([])
        # 
        # plt.subplot(2,2,2)
        # plt.plot(ab, re, 'b.', alpha=0.2)
        # plt.ylim(rerange)
        # plt.ylabel('r_e')
        # plt.yticks(reticks)
        # plt.xlim(abrange)
        # plt.xlabel('a/b')
        # plt.xticks(abticks)
        # 
        # plt.subplot(2,2,4)
        # plt.hist(ab, 25, range=abrange)
        # plt.xlim(abrange)
        # plt.xlabel('a/b')
        # plt.xticks(abticks)
        # plt.yticks([])
        # 
        # ps.savefig()

    
        
