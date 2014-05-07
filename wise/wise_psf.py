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

    ps = PlotSequence('psf')
    plt.clf()
    plt.imshow(pix, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
    ps.savefig()
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
    ps.savefig()
    plt.clf()
    plt.imshow(pix-mod, interpolation='nearest', origin='lower', vmin=-0.1*mx, vmax=0.1*mx)
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


    plt.clf()
    plt.imshow(lpix, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
    ps.savefig()
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
    ps.savefig()
    plt.clf()
    plt.imshow(lpix-mod, interpolation='nearest', origin='lower', vmin=-0.1*mx, vmax=0.1*mx)
    ps.savefig()
                                                                                                                

    psfx = GaussianMixturePSF.fromStamp(lpix, P0=(fit.amp, fit.mean*scale, fit.var*scale**2))
    psfmodel = psfx.getPointSourcePatch(0., 0., radius=lh/2)
    mod = psfmodel.patch
    mod /= mod.sum()
    mod = mod[slc]
    mx = mod.max()

    plt.clf()
    plt.imshow(lpix, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
    ps.savefig()
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
    ps.savefig()
    plt.clf()
    plt.imshow(lpix-mod, interpolation='nearest', origin='lower', vmin=-0.1*mx, vmax=0.1*mx)
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


    
