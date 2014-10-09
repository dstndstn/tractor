import numpy as np

from .basics import *
from .utils import *
from .emfit import em_fit_2d
from .fitpsf import em_init_params
from . import mixture_profiles as mp
from . import ducks

from astrometry.util.fits import *

class VaryingGaussianPSF(MultiParams, ducks.ImageCalibration):
    '''
    A mixture-of-Gaussians (MoG) PSF with spatial variation,
    represented as a spline(x,y) in each of the MoG parameters.

    This is a base class -- subclassers must implement "instantiateAt"
    '''
    def __init__(self, W, H, nx=11, ny=11, K=3, psfClass=GaussianMixturePSF):
        '''
        W,H: image size (for the image where this PSF lives)
        nx,ny: number of sample points for the spline fit
        K: number of components in the Gaussian mixture
        '''
        ## HACK -- no args?
        super(VaryingGaussianPSF, self).__init__()
        self.psfclass = psfClass
        self.splines = None
        self.W = W
        self.H = H
        self.K = K
        self.nx = nx
        self.ny = ny
        self.savesplinedata = False

    def getRadius(self):
        if hasattr(self, 'radius'):
            return self.radius
        # FIXME uhhh...?
        return 25.

    def fitSavedData(self, pp, XX, YY):
        import scipy.interpolate
        # One spline per parameter
        splines = []
        N = len(pp[0][0])
        for i in range(N):
            data = pp[:,:, i]
            spl = scipy.interpolate.RectBivariateSpline(XX, YY, data.T)
            splines.append(spl)
        self.splines = splines

    def ensureFit(self):
        '''
        This class does lazy evaluation of the spatial-variation fit.
        This methods does the evaluation if necessary.
        '''
        if self.splines is None:
            self._fitParamGrid()

    def getPointSourcePatch(self, px, py, **kwargs):
        psf = self.psfAt(px, py)
        return psf.getPointSourcePatch(px, py, **kwargs)

    def psfAt(self, x, y):
        '''
        Returns a PSF model at the given pixel position (x, y)
        '''
        params = self.psfParamsAt(x, y)
        return self.psfclass(*params)

    mogAt = psfAt
    
    def psfParamsAt(self, x, y):
        '''
        Return PSF model parameters at the given pixel position x,y.
        '''
        self.ensureFit()
        vals = np.zeros(len(self.splines))
        for i,spl in enumerate(self.splines):
            vals[i] = spl(x, y, grid=False)
        #vals = [spl(x, y) for spl in self.splines]
        return vals
        
        # K = self.K
        # w = np.empty(K)
        # # w[:-1] = vals[:K-1]
        # # vals = vals[K-1:]
        # # w[-1] = 1. - sum(w[:-1])
        # w[:] = vals[:K]
        # vals = vals[K:]
        # 
        # mu = np.empty((K,2))
        # mu.flat[:] = vals[:2*K]
        # vals = vals[2*K:]
        # var = np.empty((K,2,2))
        # var[:,0,0] = vals[:K]
        # vals = vals[K:]
        # var[:,1,0] = var[:,0,1] = vals[:K]
        # vals = vals[K:]
        # var[:,1,1] = vals[:K]
        # vals = vals[K:]
        # return w, mu, var

    def _fitParamGrid(self, fitfunc=None, **kwargs):
        # all MoG fit parameters (we need to make them shaped (ny,nx)
        # for spline fitting)
        pp = []
        # x,y coords at which we will evaluate the PSF.
        YY = np.linspace(0, self.H, self.ny)
        XX = np.linspace(0, self.W, self.nx)
        # fit params at start of this row
        px0 = None
        for y in YY:
            pprow = []
            # We start each row with the MoG fit parameters of the
            # start of the previous row (to try to make the fit
            # more continuous)
            p0 = px0
            for ix,x in enumerate(XX):
                im = self.instantiateAt(x, y)

                if fitfunc is not None:
                    f = fitfunc
                else:
                    f = self.psfclass.fromStamp
                psf = f(im, N=self.K, P0=p0, **kwargs)
                
                # gpsf = GaussianMixturePSF.fromStamp(im, N=self.K, P0=p0)
                # #v3=True, approx=1e-6)
                # print 'Fit PSF at', x,y
                # if ix == 0:
                #     px0 = (w,mu,var)
                # w,mu,var = gpsf.get_wmuvar()
                #epsf = GaussianMixtureEllipsePSF.fromStamp(im, N=self.K, P0=p0)
                p0 = psf.getParams()
                if ix == 0:
                    px0 = p0

                #psf = epsf.toMog()
                #print 'MoG psf:', psf
                #w,mu,var = psf.get_wmuvar()
                # params = np.hstack((w.ravel(),
                #                     mu.ravel(),
                #                     var[:,0,0].ravel(),
                #                     var[:,0,1].ravel(),
                #                     var[:,1,1].ravel())).copy()
                params = np.array(psf.getAllParams())
                pprow.append(params)
            pp.append(pprow)
        pp = np.array(pp)

        self.fitSavedData(pp, XX, YY)
        if self.savesplinedata:
            self.splinedata = (pp, XX, YY)

class PsfEx(VaryingGaussianPSF):
    def __init__(self, fn, W, H, ext=1,
                 scale=True,
                 nx=11, ny=11, K=3,
                 psfClass=GaussianMixturePSF):
        '''
        scale (boolean): resample the eigen-PSFs (True), or scale the
              fit parameters (False)?  
        '''
        from astrometry.util.fits import fits_table

        # See psfAt(): this needs updating & testing.
        assert(scale)
        
        if fn is not None:
            T = fits_table(fn, ext=ext)
            ims = T.psf_mask[0]
            print 'Got', ims.shape, 'PSF images'
            hdr = T.get_header()
            # PSF distortion bases are polynomials of x,y
            assert(hdr['POLNAME1'].strip() == 'X_IMAGE')
            assert(hdr['POLNAME2'].strip() == 'Y_IMAGE')
            assert(hdr['POLGRP1'] == 1)
            assert(hdr['POLGRP2'] == 1)
            assert(hdr['POLNGRP' ] == 1)
            x0     = hdr.get('POLZERO1')
            xscale = hdr.get('POLSCAL1')
            y0     = hdr.get('POLZERO2')
            yscale = hdr.get('POLSCAL2')
            degree = hdr.get('POLDEG1')
            self.sampling = hdr.get('PSF_SAMP')
            # number of terms in polynomial
            ne = (degree + 1) * (degree + 2) / 2
            assert(hdr['PSFAXIS3'] == ne)
            assert(len(ims.shape) == 3)
            assert(ims.shape[0] == ne)
            ## HACK -- fit psf0 + psfi for each term i
            ## (since those will probably work better as multi-Gaussians
            ## than psfi alone)
            ## OR instantiate PSF across the image, fit with
            ## multi-gaussians, and regress the gaussian params?
            self.psfbases = ims
            self.xscale, self.yscale = xscale, yscale
            self.degree = degree
            bh,bw = self.psfbases[0].shape
            self.radius = (bh+1)/2.
            self.x0,self.y0 = x0,y0

        self.scale = scale
        super(PsfEx, self).__init__(W, H, nx, ny, K, psfClass=psfClass)

    # def getMixtureOfGaussians(self, mean=None):
    #     if mean is not None:
    #         x = mean[0]
    #         y = mean[1]
    #     else:
    #         x = y = 0.
    #     w,mu,var = self.scaledMogParamsAt(x, y)
    #     return mp.MixtureOfGaussians(w, mu, var)
    # 
    # def mogAt(self, x, y):
    #     w,mu,var = self.scaledMogParamsAt(x, y)
    #     mog = GaussianMixturePSF(w, mu, var)
    #     mog.radius = self.radius
    #     return mog

    # def getPointSourcePatch(self, px, py, **kwargs):
    #     self.ensureFit()
    #     vals = np.zeros(len(self.splines))
    #     for i,spl in enumerate(self.splines):
    #         vals[i] = spl(px, py, grid=False)
    #     #vals = np.hstack([spl(px, py, grid=False) for spl in self.splines])
    #     #print 'vals:', vals
    #     psf = GaussianMixtureEllipsePSF(*vals)
    #     return psf.getPointSourcePatch(px, py, **kwargs)
    #     #mog = self.mogAt(px, py)
    #     #return mog.getPointSourcePatch(px, py, **kwargs)
    # 
    # def getPsfAt(self, x, y):
    #     self.ensureFit()
    #     vals = np.zeros(len(self.splines))
    #     for i,spl in enumerate(self.splines):
    #         vals[i] = spl(x, y, grid=False)
    #     psf = GaussianMixtureEllipsePSF(*vals)
    #     return psf
    # def scaledMogParamsAt(self, x, y):
    #     w,mu,var = self.mogParamsAt(x, y)
    #     if not self.scale:
    #         # We didn't downsample the PSF in pixel space, so
    #         # scale down the MOG params.
    #         sfactor = self.sampling
    #         mu  *= sfactor
    #         var *= sfactor**2
    #     return w, mu, var
    
    # def psfAt(self, x, y):
    #     psf = super(PsfEx, self).psfAt(x, y)
    #     if not self.scale:
    #         psf.scale(self.sampling)
    #     return psf
    
    def instantiateAt(self, x, y, nativeScale=False):
        from scipy.ndimage.interpolation import affine_transform
        psf = np.zeros_like(self.psfbases[0])
        #print 'psf', psf.shape
        dx = (x - self.x0) / self.xscale
        dy = (y - self.y0) / self.yscale
        i = 0
        #print 'dx',dx,'dy',dy
        for d in range(self.degree + 1):
            #print 'degree', d
            for j in range(d+1):
                k = d - j
                #print 'x',j,'y',k,
                #print 'component', i
                amp = dx**j * dy**k
                #print 'amp', amp,
                # PSFEx manual pg. 111 ?
                ii = j + (self.degree+1) * k - (k * (k-1))/ 2
                #print 'ii', ii, 'vs i', i
                psf += self.psfbases[ii] * amp
                #print 'basis rms', np.sqrt(np.mean(self.psfbases[i]**2)),
                i += 1
                #print 'psf sum', psf.sum()
        #print 'min', psf.min(), 'max', psf.max()

        if (self.scale or nativeScale) and self.sampling != 1:
            ny,nx = psf.shape
            spsf = affine_transform(psf, [1./self.sampling]*2,
                                    offset=nx/2 * (self.sampling - 1.))
            return spsf
            
        return psf

    @staticmethod
    def fromFits(fn):
        import fitsio
        hdr = fitsio.read_header(fn, ext=1)
        T = fits_table(fn)
        assert(len(T) == 1)
        #for col in T.get_columns():
        #    if col.strip() != col:
        #        T.rename_column(col, col.strip())
        T = T[0]
        t = hdr['PSFEX_T'].strip()
        print 'Type:', t
        assert(t == 'tractor.psfex.PsfEx')
        psft = hdr['PSF_TYPE']
        knowntypes = dict([(typestring(x), x)
                           for x in [GaussianMixturePSF,
                                     GaussianMixtureEllipsePSF]])
        psft = knowntypes[psft]
        print 'PSF type:', psft

        nx = hdr['PSF_NX']
        ny = hdr['PSF_NY']
        w = hdr['PSF_W']
        h = hdr['PSF_H']
        k = hdr['PSF_K']

        psfex = PsfEx(None, w, h, nx=nx, ny=ny, K=k, psfClass=psft)

        nargs = hdr['PSF_NA']
        #psf = psft(*np.zeros(nargs))
        #print 'PSF:', psf

        pp = np.zeros((ny,nx,nargs))
        
        for i in range(nargs):
            nm = hdr['PSF_A%i' % i].strip()
            print 'param name', nm
            pi = T.get(nm)
            assert(pi.shape == (ny,nx))
            pp[:,:,i] = pi

        psfex.splinedata = (pp, T.xx, T.yy)
        return psfex
    

        
    def toFits(self, fn, data=None, hdr=None):
        if hdr is None:
            import fitsio
            hdr = fitsio.FITSHDR()

        hdr.add_record(dict(name='PSFEX_T', value=typestring(type(self)),
                            comment='PsfEx type'))
        hdr.add_record(dict(name='PSF_TYPE',
                            value=typestring(self.psfclass),
                            comment='PsfEx PSF type'))
        hdr.add_record(dict(name='PSF_W', value=self.W,
                            comment='Image width'))
        hdr.add_record(dict(name='PSF_H', value=self.H,
                            comment='Image height'))
        #hdr.add_record(dict(name='PSF_SCALING',
        hdr.add_record(dict(name='PSF_K', value=self.K,
                            comment='Number of PSF components'))
        hdr.add_record(dict(name='PSF_NX', value=self.nx,
                            comment='Number of X grid points'))
        hdr.add_record(dict(name='PSF_NY', value=self.ny,
                            comment='Number of Y grid points'))

        if data is None:
            data = self.splinedata
        (pp,XX,YY) = data
        ny,nx,nparams = pp.shape
        assert(ny == self.ny)
        assert(nx == self.nx)

        X = self.psfclass(*pp[0,0])
        names = X.getParamNames()
        
        hdr.add_record(dict(name='PSF_NA', value=len(names),
                            comment='PSF number of params'))
        for i,nm in enumerate(names):
            hdr.add_record(dict(name='PSF_A%i' % i, value=nm,
                                comment='PSF param name'))

        T = fits_table()
        T.xx = XX.reshape((1, len(XX)))
        T.yy = YY.reshape((1, len(YY)))
        for i,nm in enumerate(names):
            T.set(nm, pp[:,:,i].reshape((1,ny,nx)))
        T.writeto(fn, header=hdr)
        

def typestring(t):
    t = repr(t).replace("<class '", '').replace("'>", "")
    return t
        
    
class CachingPsfEx(PsfEx):
    @staticmethod
    def fromPsfEx(psfex):
        c = CachingPsfEx(None, psfex.W, psfex.H, nx=psfex.nx, ny=psfex.ny,
                         scale=psfex.scale, K=psfex.K)
        c.sampling = psfex.sampling
        c.xscale = psfex.xscale
        c.yscale = psfex.yscale
        c.degree = psfex.degree
        c.radius = psfex.radius
        c.x0 = psfex.x0
        c.y0 = psfex.y0
        c.splinedata = psfex.splinedata
        return c

    def __init__(self, *args, **kwargs):
        from tractor.cache import Cache

        super(CachingPsfEx, self).__init__(*args, **kwargs)
        self.cache = Cache(maxsize=100)
        # round pixel coordinates to the nearest...
        self.rounding = 10
        
    def psfAt(self, x, y):
        key = (int(x)/self.rounding, int(y)/self.rounding)
        mog = self.cache.get(key, None)
        if mog is not None:
            return mog
        mog = super(CachingPsfEx, self).mogAt(x, y)
        self.cache.put(key, mog)
        return mog

    
# class PixelizedPsfEx(PsfEx):
#     def getPointSourcePatch(self, px, py, minval=0., extent=None):
#         pix = self.instantiateAt(px, py, nativeScale=True)
#         pixpsf = PixelizedPSF(pix).getPointSourcePatch(px, py)
#         #sz = pix.shape[0]
#         #return Patch(pix, -sz/2, -sz/2)

if __name__ == '__main__':
    from astrometry.util.plotutils import *
    import sys
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    psf = PsfEx('cs82data/ptf/PTF_201112091448_i_p_scie_t032828_u010430936_f02_p002794_c08.fix.psf', 2048, 4096)

    ps = PlotSequence('ex')
    
    for scale in [False, True]:
        # reset fit
        psf.scale = scale
        im = psf.instantiateAt(0.,0.)

        ny,nx = im.shape
        XX,YY = np.meshgrid(np.arange(nx), np.arange(ny))
        print 'cx', (np.sum(im * XX) / np.sum(im))
        print 'cy', (np.sum(im * YY) / np.sum(im))
        
        plt.clf()
        mx = im.max()
        plt.imshow(im, origin='lower', interpolation='nearest',
                   vmin=-0.1*mx, vmax=mx*1.1)
        plt.hot()
        plt.colorbar()
        ps.savefig()

    print 'PSF scale', psf.sampling
    print '1./scale', 1./psf.sampling

    YY = np.linspace(0, 4096, 5)
    XX = np.linspace(0, 2048, 5)

    yims = []
    for y in YY:
        xims = []
        for x in XX:
            im = psf.instantiateAt(x, y)
            im /= im.sum()
            xims.append(im)
        xims = np.hstack(xims)
        yims.append(xims)
    yims = np.vstack(yims)
    plt.clf()
    plt.imshow(yims, origin='lower', interpolation='nearest')
    plt.gray()
    plt.hot()
    plt.title('instantiated')
    ps.savefig()
    
    for scale in [True, False]:
        print 'fitting params, scale=', scale
        psf.scale = scale
        psf.splines = None
        psf.ensureFit()

        sims = []
        for y in YY:
            xims = []
            for x in XX:
                mog = psf.mogAt(x, y)
                patch = mog.getPointSourcePatch(12,12)
                pim = np.zeros((25,25))
                patch.addTo(pim)
                pim /= pim.sum()
                xims.append(pim)
            xims = np.hstack(xims)
            sims.append(xims)
        sims = np.vstack(sims)
        plt.clf()
        plt.imshow(sims, origin='lower', interpolation='nearest')
        plt.gray()
        plt.hot()
        plt.title('Spline mogs, scale=%s' % str(scale))
        ps.savefig()
    
    sys.exit(0)
    
    # import cPickle
    # print 'Pickling...'
    # SS = cPickle.dumps(psf)
    # print 'UnPickling...'
    # psf2 = cPickle.loads(SS)
    
    #psf.fitParamGrid(nx=5, ny=6)
    #psf.fitParamGrid(nx=10, ny=10)

    YY = np.linspace(0, 4096, 5)
    XX = np.linspace(0, 2048, 5)

    yims = []
    for y in YY:
        xims = []
        for x in XX:
            im = psf.instantiateAt(x, y)
            print x,y
            im /= im.sum()
            xims.append(im)
            print 'shape', xims[-1].shape
        xims = np.hstack(xims)
        print 'xims shape', xims
        yims.append(xims)
    yims = np.vstack(yims)
    print 'yims shape', yims
    plt.clf()
    plt.imshow(yims, origin='lower', interpolation='nearest')
    plt.gray()
    plt.hot()
    plt.title('PSFEx')
    ps.savefig()

    sims = []
    for y in YY:
        xims = []
        for x in XX:
            mog = psf.mogAt(x, y)
            patch = mog.getPointSourcePatch(12,12)
            pim = np.zeros((25,25))
            patch.addTo(pim)
            pim /= pim.sum()
            xims.append(pim)
        xims = np.hstack(xims)
        sims.append(xims)
    sims = np.vstack(sims)
    plt.clf()
    plt.imshow(sims, origin='lower', interpolation='nearest')
    plt.gray()
    plt.hot()
    plt.title('Spline')
    ps.savefig()


    plt.clf()
    im = yims.copy()
    im = np.sign(im) * np.sqrt(np.abs(im))
    ima = dict(origin='lower', interpolation='nearest',
               vmin=im.min(), vmax=im.max())
    plt.imshow(im, **ima)
    plt.title('PSFEx')
    ps.savefig()

    plt.clf()
    im = sims.copy()
    im = np.sign(im) * np.sqrt(np.abs(im))
    plt.imshow(im, **ima)
    plt.title('Spline')
    ps.savefig()
    
