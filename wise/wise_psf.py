if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import sys
import os

import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *

import tractor
import fitsio
from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

from tractor.psfex import *


class WisePSF(VaryingGaussianPSF):
    def __init__(self, band, savedfn=None, ngrid=11, bright=False):
        '''
        band: integer 1-4
        '''
        assert(band in [1,2,3,4])
        S = 1016
        # W4 images are binned on-board 2x2
        if band == 4:
            S /= 2
        self.band = band
        self.ngrid = ngrid
        self.bright = bright
        
        super(WisePSF, self).__init__(S, S, nx=self.ngrid, ny=self.ngrid)

        if savedfn:
            T = fits_table(savedfn)
            pp = T.data
            (NP,NY,NX) = pp.shape
            pp2 = np.zeros((NY,NX,NP))
            for i in range(NP):
                pp2[:,:,i] = pp[i,:,:]
            XX = np.linspace(0, S-1, NX)
            YY = np.linspace(0, S-1, NY)
            self.fitSavedData(pp2, XX, YY)

    def instantiateAt(self, x, y):
        '''
        This is used during fitting.  When used afterwards, you just
        want to use the getPointSourcePatch() and similar methods
        defined in the parent class.
        '''
        # clip to nearest grid point...
        dx = (self.W - 1) / float(self.ngrid - 1)
        gx = dx * int(np.round(x / dx))
        gy = dx * int(np.round(y / dx))

        btag = '-bright' if self.bright else ''
        fn = 'wise-psf/wise-psf-w%i-%.1f-%.1f%s.fits' % (self.band, gx, gy, btag)
        if not os.path.exists(fn):
            '''
            module load idl
            module load idlutils
            export WISE_DATA=$(pwd)/wise-psf/etc
            '''
            fullfn = os.path.abspath(fn)
            btag = ', BRIGHT=1' if self.bright else ''
            idlcmd = ("mwrfits, wise_psf_cutout(%.1f, %.1f, band=%i, allsky=1%s), '%s'" % 
                      (gx, gy, self.band, btag, fullfn))
            print 'IDL command:', idlcmd
            idl = os.path.join(os.environ['IDL_DIR'], 'bin', 'idl')
            cmd = 'cd wise-psf/pro; echo "%s" | %s' % (idlcmd, idl)
            print 'Command:', cmd
            os.system(cmd)

        print 'Reading', fn
        psf = pyfits.open(fn)[0].data
        return psf




def create_average_psf_model(bright=False):
    import fitsio

    btag = '-bright' if bright else ''

    for band in [1,2,3,4]:

        H,W = 1016,1016
        nx,ny = 11,11
        if band == 4:
            H /= 2
            W /= 2
        YY = np.linspace(0, H, ny)
        XX = np.linspace(0, W, nx)


        psfsum = 0.
        for y in YY:
            for x in XX:
                # clip to nearest grid point...
                dx = (W - 1) / float(nx - 1)
                dy = (H - 1) / float(ny - 1)
                gx = dx * int(np.round(x / dx))
                gy = dy * int(np.round(y / dy))

                fn = 'wise-psf/wise-psf-w%i-%.1f-%.1f%s.fits' % (band, gx, gy, btag)
                print 'Reading', fn
                I = fitsio.read(fn)
                psfsum = psfsum + I
        psfsum /= psfsum.sum()
        fitsio.write('wise-psf-avg-pix%s.fits' % btag, psfsum, clobber=(band == 1))

        psf = GaussianMixturePSF.fromStamp(psfsum)
        fn = 'wise-psf-avg.fits'
        T = fits_table()
        T.amp = psf.mog.amp
        T.mean = psf.mog.mean
        T.var = psf.mog.var
        append = (band > 1)
        T.writeto(fn, append=append)


def create_wise_psf_models(bright, K=3):
    tag = ''
    if bright:
        tag += '-bright'
    if K != 3:
        tag += '-K%i' % K
        
    for band in [1,2,3,4]:
        pfn = 'w%i%s.pickle' % (band, tag)
        if os.path.exists(pfn):
            print 'Reading', pfn
            w = unpickle_from_file(pfn)
        else:
            w = WisePSF(band)
            w.bright = bright
            w.K = K
            w.savesplinedata = True
            w.ensureFit()
            pickle_to_file(w, pfn)

        print 'Fit data:', w.splinedata
        T = tabledata()
        (pp,xx,yy) = w.splinedata
        (NY,NX,NP) = pp.shape
        pp2 = np.zeros((NP,NY,NX))
        for i in range(NP):
            pp2[i,:,:] = pp[:,:,i]

        T.data = pp2
        T.writeto('w%ipsffit%s.fits' % (band, tag))


if __name__ == '__main__':

    #create_average_psf_model()
    #create_average_psf_model(bright=True)

    from astrometry.util.util import *

    band = 4
    pix = fitsio.read('wise-psf-avg-pix.fits', ext=band-1)
    fit = fits_table('wise-psf-avg.fits', hdu=band)

    scale = 1.

    psf = GaussianMixturePSF(fit.amp, fit.mean * scale, fit.var * scale**2)

    h,w = pix.shape
    # Render the model PSF to check that it looks okay
    psfmodel = psf.getPointSourcePatch(0., 0., radius=h/2)

    slc = slice(h/2-8, h/2+8), slice(w/2-8, w/2+8)

    opix = pix
    pix /= pix.sum()
    pix = pix[slc]

    mod = psfmodel.patch
    mod /= mod.sum()
    mod = mod[slc]

    mx = mod.max()

    def plot_psf(img, mod):
        mx = max(img.max(), mod.max()) * 1.05
        plt.clf()
        plt.subplot(2,2,1)
        plt.imshow(img, interpolation='nearest', origin='lower',
                   vmin=0, vmax=mx)
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(mod, interpolation='nearest', origin='lower',
                   vmin=0, vmax=mx)
        plt.colorbar()
        plt.subplot(2,2,3)
        plt.imshow(img - mod, interpolation='nearest', origin='lower',
                   vmin=-0.1*mx, vmax=0.1*mx)
    
    ps = PlotSequence('psf')

    plot_psf(pix, mod)
    ps.savefig()

    # Lanczos sub-sample
    sh,sw = opix.shape
    scale = 2
    # xx,yy = np.meshgrid(np.linspace(-0.5, sw-0.5, scale*sw),
    #                     np.linspace(-0.5, sh-0.5, scale*sh))
    xx,yy = np.meshgrid(np.arange(0, sw, 1./scale)[:-1],
                        np.arange(0, sh, 1./scale)[:-1])
    lh,lw = xx.shape
    xx = xx.ravel()
    yy = yy.ravel()
    ix = np.round(xx).astype(np.int32)
    iy = np.round(yy).astype(np.int32)
    dx = (xx - ix).astype(np.float32)
    dy = (yy - iy).astype(np.float32)
    RR = [np.zeros(lh*lw, np.float32)]
    LL = [opix]
    lanczos3_interpolate(ix, iy, dx, dy, RR, LL)
    lpix = RR[0].reshape((lh,lw))
    #lh,lw = lpix.shape
    print 'new size', lh,lw
    print 'vs', lpix.shape
    
    slc = slice(lh/2-16, lh/2+16), slice(lw/2-16, lw/2+16)

    print 'lpix sum', lpix.sum()
    lpix = lpix / lpix.sum()
    lpix = lpix[slc]

    scale = 2.
    psf = GaussianMixturePSF(fit.amp, fit.mean * scale, fit.var * scale**2)
    # Render the model PSF to check that it looks okay
    psfmodel = psf.getPointSourcePatch(0., 0., radius=lh/2)

    mod = psfmodel.patch
    mod /= mod.sum()
    mod = mod[slc]
    mx = mod.max()

    plot_psf(lpix, mod)
    ps.savefig()

    psfx = GaussianMixturePSF.fromStamp(lpix, P0=(fit.amp, fit.mean*scale, fit.var*scale**2))
    psfmodel = psfx.getPointSourcePatch(0., 0., radius=lh/2)
    mod = psfmodel.patch
    mod /= mod.sum()
    mod = mod[slc]
    mx = mod.max()

    plot_psf(lpix, mod)
    ps.savefig()
    
    print 'Fit PSF params:', psfx

    h,w = lpix.shape

    class MyGaussianMixturePSF(GaussianMixturePSF):
        def getLogPrior(self):
            if np.any(self.mog.amp < 0.):
                return -np.inf
            for k in range(self.mog.K):
                if np.linalg.det(self.mog.var[k]) <= 0:
                    return -np.inf
            return 0

    mypsf = MyGaussianMixturePSF(psfx.mog.amp, psf.mog.mean, psf.mog.var)
        
    tim = Image(data=lpix, invvar=1e6 * np.ones_like(lpix),
                wcs=NullWCS(), photocal=LinearPhotoCal(1.),
                psf=mypsf, sky=ConstantSky(0.))

    src = PointSource(PixPos(w/2., h/2.), Flux(1.))

    tractor = Tractor([tim], [src])

    mx = lpix.max() * 1.1
    print 'mx:', mx
    
    mod = tractor.getModelImage(0)

    print 'lpix sum', lpix.sum()
    print 'mod sum', mod.sum()
    
    plot_psf(lpix, mod)
    ps.savefig()

    print 'All Params:'
    tractor.printThawedParams()
    
    tim.freezeAllBut('psf')
    #tractor.freezeParam('catalog')
    #tractor.thawPathsTo('brightness')
    src.freezeAllBut('brightness')

    tractor.freezeParam('images')
    tractor.optimize_forced_photometry()
    tractor.thawParam('images')
    
    print 'Params:'
    tractor.printThawedParams()

    for i in range(10):
        dlnp,X,alpha = tractor.optimize()
        print 'dlnp', dlnp
        print 'alpha', alpha
        print 'Sum of PSF amps:', mypsf.mog.amp.sum(), np.sum(np.abs(mypsf.mog.amp))
        
    print 'lpix sum', lpix.sum()
    print 'mod sum', mod.sum()
        
    mod = tractor.getModelImage(0)

    plot_psf(lpix, mod)
    ps.savefig()

    if False:
        # Try concentric gaussian PSF
        sigmas = []
        for k in range(mypsf.mog.K):
            v = mypsf.mog.var[k,:,:]
            sigmas.append(np.sqrt(np.sqrt(v[0,0] * v[1,1])))
        gpsf = NCircularGaussianPSF(sigmas, mypsf.mog.amp)
    
        tim.psf = gpsf
    
        print 'Params:'
        tractor.printThawedParams()
    
        for i in range(10):
            dlnp,X,alpha = tractor.optimize()
            print 'dlnp', dlnp
            print 'alpha', alpha
            print 'PSF', tim.psf
    
        mod = tractor.getModelImage(0)
    
        plot_psf(lpix, mod)
        plt.suptitle('Concentric MOG')
        ps.savefig()

        tim.psf = mypsf

    tractor.freezeParam('catalog')
    var = tractor.optimize(variance=True, just_variance=True)

    print 'Initializing sampler at:'
    tractor.printThawedParams()
    print 'Stddevs', np.sqrt(var)
    
    import emcee

    def sampleBall(p0, stdev, nw):
        '''
        Produce a ball of walkers around an initial parameter value 'p0'
        with axis-aligned standard deviation 'stdev', for 'nw' walkers.
        '''
        assert(len(p0) == len(stdev))
        return np.vstack([p0 + stdev * np.random.normal(size=len(p0))
                          for i in range(nw)])    

    nw = 100
    p0 = tractor.getParams()
    ndim = len(p0)
    
    sampler = emcee.EnsembleSampler(nw, ndim, tractor)

    var *= 1e-10
    
    # Calculate initial lnp, and ensure that all are finite.
    pp = sampleBall(p0, np.sqrt(var), nw)
    todo = None
    while True:
        if todo is None:
            lnp,nil = sampler._get_lnprob(pos=pp)
        else:
            lnp[todo],nil = sampler._get_lnprob(pos=pp[todo,:])
        todo = np.flatnonzero(np.logical_not(np.isfinite(lnp)))
        if len(todo) == 0:
            break
        print 'Re-drawing', len(todo), 'initial parameters'
        pp[todo,:] = sampleBall(p0, 0.5 * np.sqrt(var), len(todo))
    lnp0 = lnp

    bestlnp = -1e100
    bestp = None
    
    alllnp = []
    allp = []
    lnp = None
    rstate = None
    for step in range(1000):
        print 'Taking step', step
        pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        imax = np.argmax(lnp)
        print 'Max lnp', lnp[imax]
        if lnp[imax] > bestlnp:
            bestlnp = lnp[imax]
            bestp = pp[imax,:]
            print 'New best params', bestp

        alllnp.append(lnp.copy())
        allp.append(pp.copy())

    allp = np.array(allp)
    alllnp = np.array(alllnp)
        
    print 'Best params', bestp
    tractor.setParams(bestp)

    mod = tractor.getModelImage(0)

    plot_psf(lpix, mod)
    plt.suptitle('Best sample')
    ps.savefig()

    h,w = lpix.shape
    xx,yy = np.meshgrid(np.arange(w), np.arange(h))

    print 'Pix mean', np.sum(xx * lpix)/np.sum(lpix), np.sum(yy * lpix)/np.sum(lpix)
    print 'Mod mean', np.sum(xx * mod)/np.sum(mod), np.sum(yy * mod)/np.sum(mod)


    amps = mypsf.mog.amp.copy()

    for i,a in enumerate(amps):
        mypsf.mog.amp[:] = 0
        mypsf.mog.amp[i] = a

        mod = tractor.getModelImage(0)
        plot_psf(lpix, mod)
        plt.suptitle('Component %i' % i)
        ps.savefig()

        mypsf.mog.amp[:] = amps

    # Plot logprobs
    plt.clf()
    plt.plot(alllnp, 'k', alpha=0.5)
    mx = np.max([p.max() for p in alllnp])
    plt.ylim(mx-20, mx+5)
    plt.title('logprob')
    ps.savefig()

    # Plot parameter distributions
    burn = 0
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
        
    import triangle
    burn = 0
    nkeep = allp.shape[0] - burn
    X = allp[burn:, :,:].reshape((nkeep * nw, ndim))
    plt.clf()
    triangle.corner(X, labels=tractor.getParamNames(), plot_contours=False)
    ps.savefig()
    
    sys.exit(0)

    #create_wise_psf_models(True)
    create_wise_psf_models(True, K=4)
    sys.exit(0)
    


    # How to load 'em...
    # w = WisePSF(1, savedfn='w1psffit.fits')
    # print 'Instantiate...'
    # im = w.getPointSourcePatch(50., 50.)
    # print im.shape
    # plt.clf()
    # plt.imshow(im.patch, interpolation='nearest', origin='lower')
    # plt.savefig('w1.png')
    # sys.exit(0)



    w = WisePSF(1, savedfn='w1psffit.fits')
    #w.radius = 100
    pix = w.instantiateAt(50., 50.)
    r = (pix.shape[0]-1)/2
    mod = w.getPointSourcePatch(50., 50., radius=r)
    print 'mod', mod.shape
    print 'pix', pix.shape
    w.bright = True
    bpix = w.instantiateAt(50., 50.)
    print 'bpix:', bpix.shape

    s1 = mod.shape[0]
    s2 = bpix.shape[1]
    #bpix = bpix[(s2-s1)/2:, (s2-s1)/2:]
    #bpix = bpix[:s1, :s1]

    mod = mod.patch
    mod /= mod.sum()
    pix /= pix.sum()
    bpix /= bpix.sum()
    mx = max(mod.max(), pix.max())

    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.subplot(2,3,2)
    plt.imshow(pix, interpolation='nearest', origin='lower')
    plt.subplot(2,3,3)
    plt.imshow(bpix, interpolation='nearest', origin='lower')
    plt.subplot(2,3,4)
    ima = dict(interpolation='nearest', origin='lower',
               vmin=-8, vmax=0)
    plt.imshow(np.log10(np.maximum(1e-16, mod/mx)), **ima)
    plt.subplot(2,3,5)
    plt.imshow(np.log10(np.maximum(1e-16, pix/mx)), **ima)
    plt.subplot(2,3,6)
    plt.imshow(np.log10(np.maximum(1e-16, bpix/mx)), **ima)
    # plt.imshow(np.log10(np.abs((bpix - pix)/mx)), interpolation='nearest',
    #            origin='lower', vmin=-8, vmax=0)
    plt.savefig('w1.png')
    sys.exit(0)


    
