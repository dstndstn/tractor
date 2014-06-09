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



def _allwise_psf_models():
    global plotslice
    # AllWISE PSF models
    for band in [1,2,3,4]:
        print
        print 'W%i' % band
        print

        pix = reduce(np.add,
                     [fitsio.read('wise-w%i-psf-wpro-09x09-%02ix%02i.fits' %
                                  (band, 1+(i/9), 1+(i%9)))
                                  for i in range(9*9)])
        print 'Pix', pix.shape
        h,w = pix.shape
        print 'Pix sum', pix.sum()
        print 'Pix max', pix.max()
        pix /= pix.sum()
        
        psf = GaussianMixturePSF.fromStamp(pix)
        print 'Fit PSF', psf
        psfmodel = psf.getPointSourcePatch(0., 0., radius=h/2)
        mod = psfmodel.patch
        print 'Orig mod sum', mod.sum()

        S = h/2
        plotss = 30
        plotslice = slice(S-plotss, -(S-plotss)), slice(S-plotss, -(S-plotss))
        
        _plot_psf(pix, mod, psf)
        plt.suptitle('W%i: from stamp' % band)
        ps.savefig()

        # Try concentric gaussian PSF
        sh,sw = pix.shape
        sigmas = []
        for k in range(psf.mog.K):
            v = psf.mog.var[k,:,:]
            sigmas.append(np.sqrt(np.sqrt(v[0,0] * v[1,1])))

        
        print 'Initializing concentric Gaussian model with sigmas', sigmas
        print 'And weights', psf.mog.amp

        # (Initial) W1:
        # sig [7.1729391870564569, 24.098952864976805, 108.78869923786333]
        # weights [ 0.80355109  0.15602835  0.04195982]
        # sigmas [ 6.928, 17.091, 45.745 ], weights [ 0.760, 0.154, 0.073 ]
        
        # W2:
        # sig [8.2356371659198189, 25.741694812001221, 110.17431488810806]
        # weights [ 0.80636191  0.1563742   0.03926182]
        # sigmas [ 7.177, 10.636, 30.247 ], weights [ 0.500, 0.341, 0.134 ]

        # W3:
        # sig [6.860124763919889, 19.160849966251824, 134.20812055907825]
        # weights [ 0.28227047  0.55080156  0.16706357]
        # sigmas [ 6.850, 18.922, 109.121 ], weights [ 0.276, 0.554, 0.148 ]

        # W4:
        # [4.6423676603462001, 5.31542132669962, 18.375512539373585]
        # weights [ 0.10764341  0.26915295  0.62320374]
        # (Opt)
        # sigmas [ 6.293, 6.314, 20.932 ], weights [ 0.129, 0.309, 0.598 ]

        weights = psf.mog.amp

        # HACK
        if band == 4:
            sigmas =  [ 2., 6., 25. ]
            weights = [ 0.5, 0.25, 0.25 ]
        else:
            sigmas =  [ 8., 24., 100. ]
            weights = [ 0.5, 0.25, 0.25 ]
        
        gpsf = NCircularGaussianPSF(sigmas, weights)
        gpsf.radius = sh/2
        mypsf = gpsf
        tim = Image(data=pix, invvar=1e6 * np.ones_like(pix), psf=mypsf)
        cx,cy = sw/2, sh/2
        src = PointSource(PixPos(cx, cy), Flux(1.0))
        
        tractor = Tractor([tim], [src])
        
        tim.freezeAllBut('psf')
        #tractor.freezeParam('catalog')
        src.freezeAllBut('pos')

        print 'Optimizing Params:'
        tractor.printThawedParams()
    
        for i in range(100):
            dlnp,X,alpha = tractor.optimize(damp=0.1)
            print 'dlnp', dlnp
            print 'PSF', mypsf
            if dlnp < 0.001:
                break

        print 'PSF3(opt): flux', src.brightness
        print 'PSF amps:', np.sum(mypsf.amp)
        print 'PSF amps * Source brightness:', src.brightness.getValue() * np.sum(mypsf.amp)
        print 'pix sum:', pix.sum()

        print 'Optimize source:', src
        print 'Optimized PSF:', mypsf
        
        mod = tractor.getModelImage(0)
        print 'Mod sum:', mod.sum()
        print 'Mod min:', mod.min()
        print 'Mod max:', mod.max()

        _plot_psf(pix, mod, mypsf, flux=src.brightness.getValue())
        plt.suptitle('W%i psf3 (opt): %g = %s, resid %.3f' %
                     (band, np.sum(mypsf.amp), ','.join(['%.3f'%a for a in mypsf.amp]), np.sum(pix-mod)))
        ps.savefig()


        # These postage stamps are subsampled at 8x the 2.75"/pix,
        # except for the W4 one, which is at 4x that.
        scale = 8
        if band == 4:
            scale = 4
        
        T = fits_table()
        T.amp  = mypsf.mog.amp
        T.mean = mypsf.mog.mean / scale
        T.var  = mypsf.mog.var / scale**2
        T.writeto('psf-allwise-con3-w%i.fits' % band)
        T.writeto('psf-allwise-con3.fits', append=(band != 1))

        
        mypsf.weights.setParams(np.array(mypsf.weights.getParams()) /
                                sum(mypsf.weights.getParams()))
        print 'Normalized PSF weights:', mypsf
        mod = tractor.getModelImage(0)
        print 'Mod sum:', mod.sum()
        _plot_psf(pix, mod, mypsf, flux=src.brightness.getValue())
        plt.suptitle('W%i psf3 (opt): %g = %s, resid %.3f' %
                     (band, np.sum(mypsf.amp), ','.join(['%.3f'%a for a in mypsf.amp]), np.sum(pix-mod)))
        ps.savefig()

def _meisner_psf_models():
    global plotslice
    # Meisner's PSF models
    #for band in [4]:
    for band in [1,2,3,4]:
        print
        print 'W%i' % band
        print
        
        #pix = fitsio.read('wise-psf-avg-pix.fits', ext=band-1)
        pix = fitsio.read('wise-psf-avg-pix-bright.fits', ext=band-1)
        fit = fits_table('wise-psf-avg.fits', hdu=band)
        #fit = fits_table('psf-allwise-con3.fits', hdu=band)
    
        scale = 1.
    
        print 'Pix shape', pix.shape
        h,w = pix.shape

        xx,yy = np.meshgrid(np.arange(w), np.arange(h))
        cx,cy = np.sum(xx*pix)/np.sum(pix), np.sum(yy*pix)/np.sum(pix)
        print 'Centroid:', cx,cy

        #S = 100
        S = h/2
        slc = slice(h/2-S, h/2+S+1), slice(w/2-S, w/2+S+1)

        plotss = 30
        plotslice = slice(S-plotss, -(S-plotss)), slice(S-plotss, -(S-plotss))
        
        opix = pix
        print 'Original pixels sum:', opix.sum()
        pix /= pix.sum()
        pix = pix[slc]
        print 'Sliced pix sum', pix.sum()

        psf = GaussianMixturePSF(fit.amp, fit.mean * scale, fit.var * scale**2)
        psfmodel = psf.getPointSourcePatch(0., 0., radius=h/2)
        mod = psfmodel.patch
        print 'Orig mod sum', mod.sum()
        mod = mod[slc]
        print 'Sliced mod sum', mod.sum()
        print 'Amps:', np.sum(psf.mog.amp)
        
        _plot_psf(pix, mod, psf)
        plt.suptitle('W%i: Orig' % band)
        ps.savefig()
                
        # Lanczos sub-sample
        if band == 4:
            scale = 2
            sh,sw = opix.shape
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
            print 'new size', lh,lw
            print 'vs', lpix.shape

            pix = lpix
            pix /= (scale**2)
            h,w = pix.shape

            #S = 140
            slc = slice(h/2-S, h/2+S+1), slice(w/2-S, w/2+S+1)
            print 'Resampled pix sum', pix.sum()
            pix = pix[slc]
            print 'sliced pix sum', pix.sum()

            psf = GaussianMixturePSF(fit.amp, fit.mean * scale, fit.var * scale**2)
            psfmodel = psf.getPointSourcePatch(0., 0., radius=h/2)
            mod = psfmodel.patch
            print 'Scaled mod sum', mod.sum()
            mod = mod[slc]
            print 'Sliced mod sum', mod.sum()

            _plot_psf(pix, mod, psf)
            plt.suptitle('W%i: Scaled' % band)
            ps.savefig()

            plotslice = slice(S-plotss,-(S-plotss)),slice(S-plotss,-(S-plotss))

        psfx = GaussianMixturePSF.fromStamp(pix, P0=(fit.amp, fit.mean*scale, fit.var*scale**2))
        psfmodel = psfx.getPointSourcePatch(0., 0., radius=h/2)
        mod = psfmodel.patch
        print 'From stamp: mod sum', mod.sum()
        mod = mod[slc]
        print 'Sliced mod sum:', mod.sum()
        _plot_psf(pix, mod, psfx)
        plt.suptitle('W%i: fromStamp: %g = %s, res %.3f' %
                     (band, np.sum(psfx.mog.amp), ','.join(['%.3f'%a for a in psfx.mog.amp]),
                      np.sum(pix - mod)))
        ps.savefig()
        
        print 'Stamp-Fit PSF params:', psfx
    
        # class MyGaussianMixturePSF(GaussianMixturePSF):
        #     def getLogPrior(self):
        #         if np.any(self.mog.amp < 0.):
        #             return -np.inf
        #         for k in range(self.mog.K):
        #             if np.linalg.det(self.mog.var[k]) <= 0:
        #                 return -np.inf
        #         return 0
        # 
        #     @property
        #     def amp(self):
        #         return self.mog.amp
        #     
        # mypsf = MyGaussianMixturePSF(psfx.mog.amp, psfx.mog.mean, psfx.mog.var)
        # mypsf.radius = sh/2

        sh,sw = pix.shape

        # Try concentric gaussian PSF
        sigmas = []
        for k in range(psfx.mog.K):
            v = psfx.mog.var[k,:,:]
            sigmas.append(np.sqrt(np.sqrt(np.abs(v[0,0] * v[1,1]))))
        print 'Initializing concentric Gaussian PSF with sigmas', sigmas
        gpsf = NCircularGaussianPSF(sigmas, psfx.mog.amp)
        gpsf.radius = sh/2
        mypsf = gpsf
        
        tim = Image(data=pix, invvar=1e6 * np.ones_like(pix), psf=mypsf)

        # xx,yy = np.meshgrid(np.arange(sw), np.arange(sh))
        # cx,cy = np.sum(xx*pix)/np.sum(pix), np.sum(yy*pix)/np.sum(pix)
        # print 'Centroid:', cx,cy
        # print 'Pix midpoint:', sw/2, sh/2
        cx,cy = sw/2, sh/2
        src = PointSource(PixPos(cx, cy), Flux(1.0))
        
        tractor = Tractor([tim], [src])
        
        tim.freezeAllBut('psf')
        #tractor.freezeParam('catalog')
        src.freezeAllBut('pos')
        
        # src.freezeAllBut('brightness')
        # tractor.freezeParam('images')
        # tractor.optimize_forced_photometry()
        # tractor.thawParam('images')
        # print 'Source flux after forced-photom fit:', src.getBrightness()
    
        print 'Optimizing Params:'
        tractor.printThawedParams()
    
        for i in range(100):
            dlnp,X,alpha = tractor.optimize(damp=0.1)
            print 'dlnp', dlnp
            if dlnp < 0.001:
                break

        print 'PSF3(opt): flux', src.brightness
        print 'PSF amps:', np.sum(mypsf.amp)
        print 'PSF amps * Source brightness:', src.brightness.getValue() * np.sum(mypsf.amp)
        print 'pix sum:', pix.sum()

        print 'Optimize source:', src
        print 'Optimized PSF:', mypsf
        
        mod = tractor.getModelImage(0)
        print 'Mod sum:', mod.sum()
        _plot_psf(pix, mod, mypsf, flux=src.brightness.getValue())
        plt.suptitle('W%i psf3 (opt): %g = %s, resid %.3f' %
                     (band, np.sum(mypsf.amp), ','.join(['%.3f'%a for a in mypsf.amp]), np.sum(pix-mod)))
        ps.savefig()

        # Write before normalizing!
        T = fits_table()
        T.amp  = mypsf.mog.amp
        T.mean = mypsf.mog.mean
        T.var  = mypsf.mog.var
        T.writeto('psf3-w%i.fits' % band)
        T.writeto('psf3.fits', append=(band != 1))
        
        mypsf.weights.setParams(np.array(mypsf.weights.getParams()) /
                                sum(mypsf.weights.getParams()))
        print 'Normalized PSF weights:', mypsf
        mod = tractor.getModelImage(0)
        print 'Mod sum:', mod.sum()
        _plot_psf(pix, mod, mypsf, flux=src.brightness.getValue())
        plt.suptitle('W%i psf3 (opt): %g = %s, resid %.3f' %
                     (band, np.sum(mypsf.amp), ','.join(['%.3f'%a for a in mypsf.amp]), np.sum(pix-mod)))
        ps.savefig()
        


if __name__ == '__main__':

    import logging
    lev = level=logging.DEBUG
    lev = level=logging.INFO
    lev = level=logging.WARN
    logging.basicConfig(format="%(message)s", level=lev,
                        stream=sys.stdout)

    #create_average_psf_model()
    #create_average_psf_model(bright=True)
    plotslice = None
    
    def _plot_psf(img, mod, psf, flux=1.):
        if plotslice is not None:
            img = img[plotslice]
            mod = mod[plotslice]
            
        mx = img.max() * 1.1
        logmx = np.log10(mx)
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=0, vmax=mx)
        imlog = dict(interpolation='nearest', origin='lower',
                     vmin=logmx-4, vmax=logmx)
        imdiff = dict(interpolation='nearest', origin='lower',
                      vmin=-0.05*mx, vmax=0.05*mx, cmap='RdBu')
        plt.clf()
        plt.subplot(3,3,1)
        plt.imshow(img, **ima)
        plt.xticks([]); plt.yticks([])

        plt.subplot(3,3,4)
        plt.imshow(np.log10(img), **imlog)
        plt.xticks([]); plt.yticks([])

        plt.subplot(3,3,2)
        plt.imshow(mod, **ima)
        plt.xticks([]); plt.yticks([])

        plt.subplot(3,3,5)
        plt.imshow(np.log10(mod), **imlog)
        plt.xticks([]); plt.yticks([])

        dd = 10

        plt.subplot(3,3,9)
        plt.imshow(img - mod, **imdiff)
        plt.xticks([]); plt.yticks([])
        h,w = img.shape
        plt.axis([w/2-dd, w/2+dd, h/2-dd, h/2+dd])
        
        if psf is not None:
            for sp in [3,6,9]:
                plt.subplot(3,3,sp)
                if sp == 3:
                    plt.imshow(mod, **ima)
                    cc = 'w'
                    aa = 1.
                else:
                    #plt.imshow(np.log10(mod), **imlog)
                    plt.imshow(img - mod, **imdiff)
                    cc = 'g'
                    aa = 1.#0.5
                plt.xticks([]); plt.yticks([])

                ax = plt.axis()
                #if isinstance(psf, GaussianMixturePSF):
                vv = psf.mog.var
                mu = psf.mog.mean
                K = psf.mog.K
                # elif isinstance(psf, NCircularGaussianPSF):
                #     K = len(psf.sigmas.getParams())
                #     vv = np.zeros((K,2,2))
                #     vv[:,0,0] = vv[:,1,1] = psf.sigmas.getParams()**2
                #     mu = np.zeros((K,2))
                    
                h,w = mod.shape
                for k in range(K):
                    v = vv[k,:,:]
                    u,s,v = np.linalg.svd(v)
                    angle = np.linspace(0., 2.*np.pi, 200)
                    u1 = u[0,:]
                    u2 = u[1,:]
                    s1,s2 = np.sqrt(s)
                    xy = (u1[np.newaxis,:] * s1 * np.cos(angle)[:,np.newaxis] +
                          u2[np.newaxis,:] * s2 * np.sin(angle)[:,np.newaxis])
                    plt.plot(xy[:,0]+w/2+mu[k,0], xy[:,1]+h/2+mu[k,1], '-',
                             color=cc, alpha=aa)
                plt.axis(ax)

        
        shift = max(img.max(), mod.max())*1.05
        
        maxpix = np.argmax(img)
        y,x = np.unravel_index(maxpix, img.shape)
        if h >= (dd*2+1):
            slc = slice(h/2-dd, h/2+dd+1)
            x0 = h/2-dd
        else:
            slc = slice(0, h)
            x0 = 0
        plt.subplot(3,3,7)
        plt.plot(img[y,slc], 'r-')
        plt.plot(mod[y,slc], 'b-', alpha=0.5, lw=3)
        plt.axhline(shift, color='k')
        plt.plot(img[slc,x]+shift, 'r-')
        plt.plot(mod[slc,x]+shift, 'b-', alpha=0.5, lw=3)
        # if psf is not None:
        #     ax = plt.axis()
        #     xx = np.linspace(ax[0], ax[1], 100)
        #     modx = np.zeros_like(xx)
        #     mody = np.zeros_like(xx)
        #     for k in range(K):
        #         v = vv[k,:,:]
        #         vx = v[0,0]
        #         mux = psf.mog.mean[k,0]
        #         vy = v[1,1]
        #         muy = psf.mog.mean[k,1]
        #         a = psf.mog.amp[k]
        #         a *= flux
        #         compx = a/(2.*np.pi*vx) * np.exp(-0.5* (xx - (x-x0+mux))**2 / vx)
        #         compy = a/(2.*np.pi*vy) * np.exp(-0.5* (xx - (y-x0+muy))**2 / vy)
        #         modx += compx
        #         mody += compy
        #         plt.plot(xx, compx, 'b-')
        #         plt.plot(xx, compy + shift, 'b-')
        #     plt.plot(xx, modx, 'b-')
        #     plt.plot(xx, mody + shift, 'b-')
        #     plt.axis(ax)
        plt.xticks([]); plt.yticks([])
        plt.ylim(0, 2.*shift)

        dshift = 0.05*shift
        plt.subplot(3,3,8)
        plt.axhline(dshift, color='k')
        plt.plot(mod[y,slc] - img[y,slc] + dshift, 'b-', alpha=0.5, lw=3)
        plt.axhline(dshift*3, color='k')
        plt.plot(mod[slc,x] - img[slc,x] + dshift*3, 'b-', alpha=0.5, lw=3)
        plt.xticks([]); plt.yticks([])
        plt.ylim(0, 4*dshift)
        return
        
    
    from astrometry.util.util import *

    ps = PlotSequence('psf')

    _meisner_psf_models()
        
    sys.exit(0)
        
    psf2 = GaussianMixturePSF(mypsf.mog.amp[:2]/mypsf.mog.amp[:2].sum(),
                              mypsf.mog.mean[:2,:], mypsf.mog.var[:2,:,:])
    psf2.radius = sh/2
    tim.psf = psf2
    mod = tractor.getModelImage(0)
    _plot_psf(lpix, mod, psf2, flux=src.brightness.getValue())
    plt.suptitle('psf2 (init)')
    ps.savefig()

    src.freezeAllBut('brightness')
    tractor.freezeParam('catalog')
    for i in range(100):

        print 'Optimizing PSF:'
        tractor.printThawedParams()
        
        dlnp,X,alpha = tractor.optimize(damp=1.)

        tractor.freezeParam('images')
        tractor.thawParam('catalog')

        print 'Optimizing flux:'
        tractor.printThawedParams()

        tractor.optimize_forced_photometry()
        tractor.thawParam('images')
        tractor.freezeParam('catalog')
        
    mod = tractor.getModelImage(0)
    _plot_psf(lpix, mod, psf2, flux=src.brightness.getValue())
    plt.suptitle('psf2 (opt)')
    ps.savefig()

    print 'amps', psf2.mog.amp
    
    T = fits_table()
    T.amp  = psf2.mog.amp / psf2.mog.amp.sum()
    cent = np.sum(T.amp[:,np.newaxis] * psf2.mog.mean, axis=0)
    T.mean = psf2.mog.mean - cent[np.newaxis,:]
    T.var  = psf2.mog.var
    T.writeto('mypsf2.fits')

    sys.exit(0)

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
    
        _plot_psf(lpix, mod)
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

    _plot_psf(lpix, mod)
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
        _plot_psf(lpix, mod)
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


    
