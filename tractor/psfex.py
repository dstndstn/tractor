import numpy as np
import numpy.linalg
#import scipy.interpolate as interp
import scipy.interpolate
from scipy.ndimage.interpolation import affine_transform
from astrometry.util.fits import *
from astrometry.util.plotutils import *

from tractor.basics import *
from tractor.utils import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

class VaryingGaussianPSF(MultiParams):
    '''
    A mixture-of-Gaussians (MoG) PSF with spatial variation,
    represented as a spline(x,y) in each of the MoG parameters.

    This is a base class -- subclassers must implement "instantiateAt"
    '''
    def __init__(self, W, H, nx=11, ny=11, K=3):
        '''
        W,H: image size (for the image where this PSF lives)
        nx,ny: number of sample points for the spline fit
        K: number of components in the Gaussian mixture
        '''
        ## HACK -- no args?
        super(VaryingGaussianPSF, self).__init__()
        self.splines = None
        self.W = W
        self.H = H
        self.K = K
        self.nx = nx
        self.ny = ny

        self.savesplinedata = False

    def fitSavedData(self, pp, XX, YY):
        # One spline per parameter
        splines = []
        N = len(pp[0][0])
        for i in range(N):
            data = pp[:,:, i]
            spl = scipy.interpolate.RectBivariateSpline(XX, YY, data.T)
            #print 'Building a spline on XX,YY,data', XX.shape, YY.shape, data.T.shape
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
        mog = self.mogAt(px, py)
        return mog.getPointSourcePatch(px, py, **kwargs)

    def mogAt(self, x, y):
        '''
        Returns a Mixture-of-Gaussians representation of this PSF at the given
        pixel position x,y
        '''
        w,mu,sig = self.mogParamsAt(x, y)
        return GaussianMixturePSF(w, mu, sig)

    def mogParamsAt(self, x, y):
        '''
        Return Mixture-of-Gaussian parameters for this PSF at the
        given pixel position x,y.
        '''
        self.ensureFit()
        vals = [spl(x, y) for spl in self.splines]
        K = self.K
        w = np.empty(K)
        w[:-1] = vals[:K-1]
        vals = vals[K-1:]
        w[-1] = 1. - sum(w[:-1])
        mu = np.empty((K,2))
        mu.ravel()[:] = vals[:2*K]
        vals = vals[2*K:]
        sig = np.empty((K,2,2))
        sig[:,0,0] = vals[:K]
        vals = vals[K:]
        sig[:,0,1] = vals[:K]
        sig[:,1,0] = sig[:,0,1]
        vals = vals[K:]
        sig[:,1,1] = vals[:K]
        vals = vals[K:]
        return w, mu, sig

    def _fitParamGrid(self):
        # number of MoG mixture components
        K = self.K
        w,mu,sig = em_init_params(K, None, None, None)
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
            for ix,x in enumerate(XX):
                # We start each row with the MoG fit parameters of the start of the
                # previous row (to try to make the fit more continuous)
                if ix == 0 and px0 is not None:
                    w,mu,sig = px0
                im = self.instantiateAt(x, y)
                PS = im.shape[0]
                im /= im.sum()
                im = np.maximum(im, 0)
                xm,ym = -(PS/2), -(PS/2)
                em_fit_2d(im, xm, ym, w, mu, sig)
                #print 'w,mu,sig', w,mu,sig
                if ix == 0:
                    px0 = w,mu,sig

                params = np.hstack((w.ravel()[:-1],
                                    mu.ravel(),
                                    sig[:,0,0].ravel(),
                                    sig[:,0,1].ravel(),
                                    sig[:,1,1].ravel())).copy()
                pprow.append(params)
            pp.append(pprow)
        pp = np.array(pp)

        self.fitSavedData(pp, XX, YY)
        if self.savesplinedata:
            self.splinedata = (pp, XX, YY)

class PsfEx(VaryingGaussianPSF):
    def __init__(self, fn, W, H, ext=1,
                 scale=True,
                 nx=11, ny=11, K=3):
        '''
        scale (boolean): resample the eigen-PSFs (True), or scale the
              fit parameters (False)?  
        '''
        T = fits_table(fn, ext=ext)
        ims = T.psf_mask[0]
        print 'Got', ims.shape, 'PSF images'
        hdr = pyfits.open(fn)[ext].header
        # PSF distortion bases are polynomials of x,y
        assert(hdr['POLNAME1'] == 'X_IMAGE')
        assert(hdr['POLNAME2'] == 'Y_IMAGE')
        assert(hdr['POLGRP1'] == 1)
        assert(hdr['POLGRP2'] == 1)
        assert(hdr['POLNGRP' ] == 1)
        x0     = hdr.get('POLZERO1')
        xscale = hdr.get('POLSCAL1')
        y0     = hdr.get('POLZERO2')
        yscale = hdr.get('POLSCAL2')
        degree = hdr.get('POLDEG1')

        self.sampling = hdr.get('PSF_SAMP')
        self.scale = scale

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
        self.x0,self.y0 = x0,y0
        self.xscale, self.yscale = xscale, yscale
        self.degree = degree

        super(PsfEx, self).__init__(W, H, nx, ny, K)

    def mogAt(self, x, y):
        w,mu,sig = self.mogParamsAt(x, y)
        if self.scale:
            # We didn't downsample the PSF in pixel space, so
            # scale down the MOG params.
            sfactor = self.sampling
            mu  *= sfactor
            sig *= sfactor**2
        return GaussianMixturePSF(w, mu, sig)

    def instantiateAt(self, x, y):
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

        if self.scale and self.sampling != 1:
            ny,nx = psf.shape
            spsf = affine_transform(psf, [1./self.sampling]*2,
                                    offset=nx/2 * (self.sampling - 1.))
            return spsf
            
        return psf

if __name__ == '__main__':
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
    
