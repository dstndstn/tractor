from __future__ import print_function
import numpy as np
from astrometry.util.miscutils import lanczos_filter

from .image import Image
from .pointsource import PointSource
from .wcs import PixPos
from .brightness import Flux
from .engine import Tractor
from .patch import Patch
from .utils import BaseParams, ParamList, MultiParams
from . import mixture_profiles as mp
from . import ducks

# class VaryingPsfMixin(ducks.ImageCalibration):
#     def getShifted(self, x0, y0):
#         return ShiftedPsf(self, x0, y0)
# 
# class ConstantPsfMixin(ducks.ImageCalibration):
#     def getShifted(self, x0, y0):
#         return self

class HybridPSF(object):
    pass

class PixelizedPSF(BaseParams, ducks.ImageCalibration):
    '''
    A PSF model based on an image postage stamp, which will be
    sinc-shifted to subpixel positions.

    Galaxies will be rendered using FFT convolution.

    FIXME -- currently this class claims to have no params.
    '''

    def __init__(self, img, Lorder=3):
        '''
        Creates a new PixelizedPSF object from the given *img* (numpy
        array) image of the PSF. 

        - *img* must be an ODD size.
        - *Lorder* is the order of the Lanczos interpolant used for
           shifting the image to subpixel positions.
        '''
        self.img = img
        H,W = img.shape
        assert((H % 2) == 1)
        assert((W % 2) == 1)
        self.radius = np.hypot(H/2., W/2.)
        self.H, self.W = H,W
        self.Lorder = Lorder
        self.fftcache = {}
        
    def __str__(self):
        return 'PixelizedPSF'

    def clear_cache(self):
        self.fftcache = {}

    @property
    def shape(self):
        return (self.H, self.W)
    
    def hashkey(self):
        return ('PixelizedPSF', tuple(self.img.ravel()))

    def copy(self):
        return PixelizedPSF(self.img.copy())

    def getShifted(self, x0, y0):
        # not spatially varying
        return self

    def constantPsfAt(self, x, y):
        # not spatially varying
        return self
    
    def getRadius(self):
        return self.radius

    def getImage(self, px, py):
        return self.img

    def getPointSourcePatch(self, px, py, minval=0., modelMask=None, **kwargs):
        from scipy.ndimage.filters import correlate1d
        from astrometry.util.miscutils import get_overlapping_region

        img = self.getImage(px, py)

        H,W = img.shape
        ix = int(np.round(px))
        iy = int(np.round(py))
        dx = px - ix
        dy = py - iy
        x0 = ix - W/2
        y0 = iy - H/2

        if modelMask is not None:
            mh,mw = modelMask.shape
            mx0,my0 = modelMask.x0, modelMask.y0

            # print 'PixelizedPSF + modelMask'
            # print 'mx0,my0', mx0,my0, '+ mw,mh', mw,mh
            # print 'PSF image x0,y0', x0,y0, '+ W,H', W,H

            if (mx0 >= x0 + W or
                my0 >= y0 + H or
                mx0 + mw <= x0 or
                my0 + mh <= y0):
                # No overlap
                return None
            # Otherwise, we'll just produce the Lanczos-shifted PSF
            # image as usual, and then copy it into the modelMask
            # space.

        L = self.Lorder
        Lx = lanczos_filter(L, np.arange(-L, L+1) + dx)
        Ly = lanczos_filter(L, np.arange(-L, L+1) + dy)
        # Normalize the Lanczos interpolants (preserve flux)
        Lx /= Lx.sum()
        Ly /= Ly.sum()
        sx      = correlate1d(img, Lx, axis=1, mode='constant')
        shifted = correlate1d(sx,  Ly, axis=0, mode='constant')
        if modelMask is None:
            return Patch(x0, y0, shifted)

        # Pad or clip to modelMask size
        mm = np.zeros((mh,mw), shifted.dtype)
        yo = y0 - my0
        yi = 0
        ny = min(y0+H, my0+mh) - max(y0, my0)
        if yo < 0:
            yi = -yo
            yo = 0
        xo = x0 - mx0
        xi = 0
        nx = min(x0+W, mx0+mw) - max(x0, mx0)
        if xo < 0:
            xi = -xo
            xo = 0
        mm[yo:yo+ny, xo:xo+nx] = shifted[yi:yi+ny, xi:xi+nx]
        return Patch(mx0, my0, mm)

    def getFourierTransformSize(self, radius):
        # Next power-of-two size
        sz = 2**int(np.ceil(np.log2(radius*2.)))
        return sz

    def _padInImage(self, H,W, img=None):
        '''
        Embeds this PSF image into a larger or smaller image of shape H,W.

        Return (img, cx, cy), where *cx*,*cy* are the coordinates of the PSF
        center in *img*.
        '''
        if img is None:
            img = self.img
        ph,pw = img.shape
        subimg = img

        if H >= ph:
            y0 = (H - ph) / 2
            cy = y0 + ph/2
        else:
            y0 = 0
            cut = (ph - H) / 2
            subimg = subimg[cut:cut+H, :]
            cy = ph/2 - cut

        if W >= pw:
            x0 = (W - pw) / 2
            cx = x0 + pw/2
        else:
            x0 = 0
            cut = (pw - W) / 2
            subimg = subimg[:, cut:cut+W]
            cx = pw/2 - cut
        sh,sw = subimg.shape

        pad = np.zeros((H, W), img.dtype)
        pad[y0:y0+sh, x0:x0+sw] = subimg
        return pad, cx, cy
        
    def getFourierTransform(self, px, py, radius):
        '''
        Returns the Fourier Transform of this PSF, with the
        next-power-of-2 size up from *radius*.

        Returns: (FFT, (x0, y0), (imh,imw), (v,w))

        *FFT*: numpy array, the FFT
        *xc*: float, pixel location of the PSF /center/ in the PSF subimage
        *yc*:    ditto
        *imh,imw*: ints, shape of the padded PSF image
        *v,w*: v=np.fft.rfftfreq(imw), w=np.fft.fftfreq(imh)
        
        '''
        sz = self.getFourierTransformSize(radius)
        # print 'PixelizedPSF FFT size', sz
        if sz in self.fftcache:
            return self.fftcache[sz]

        pad,cx,cy = self._padInImage(sz,sz)
        ## cx,cy: coordinate of the PSF center in *pad*
        P = np.fft.rfft2(pad)
        pH,pW = pad.shape
        v = np.fft.rfftfreq(pW)
        w = np.fft.fftfreq(pH)
        rtn = P, (cx, cy), (pH,pW), (v,w)
        self.fftcache[sz] = rtn
        return rtn

class GaussianMixturePSF(ParamList, ducks.ImageCalibration):
    '''
    A PSF model that is a mixture of general 2-D Gaussians
    (characterized by amplitude, mean, covariance)
    '''
    def __init__(self, *args):
        '''
        GaussianMixturePSF(amp, mean, var)

        or

        GaussianMixturePSF(a0,a1,a2, mx0,my0,mx1,my1,mx2,my2,
                           vxx0,vyy0,vxy0, vxx1,vyy1,vxy1, vxx2,vyy2,vxy2)
        
        amp:  np array (size K) of Gaussian amplitudes
        mean: np array (size K,2) of means
        var:  np array (size K,2,2) of variances
        '''
        if len(args) == 3:
            amp, mean, var = args
        else:
            assert(len(args) % 6 == 0)
            K = len(args) / 6
            amp  = np.array(args[:K])
            mean = np.array(args[K:3*K]).reshape((K,2))
            args = args[3*K:]
            var  = np.zeros((K,2,2))
            var[:,0,0] = args[::3]
            var[:,1,1] = args[1::3]
            var[:,0,1] = var[:,1,0] = args[2::3]

        self.mog = mp.MixtureOfGaussians(amp, mean, var)
        assert(self.mog.D == 2)
        self.radius = 25
        super(GaussianMixturePSF, self).__init__()

        del self.vals
        
        K = self.mog.K
        self.stepsizes = [0.01]*K + [0.01]*(K*2) + [0.1]*(K*3)
        self._set_param_names(K)

    def getShifted(self, x0, y0):
        # not spatially varying
        return self
    def constantPsfAt(self, x, y):
        return self
    
    def _set_param_names(self, K):
        # ordering: A0, A1, ... Ak, mux0, muy0, mux1, muy1, mux2, muy2, ...
        #   var0xx,var0yy,var0xy, var1xx, var1yy, var1xy
        names = {}
        for k in range(K):
            names['amp%i'%k] = k
            names['meanx%i'%k] = K+(k*2)
            names['meany%i'%k] = K+(k*2)+1
            names['varxx%i'%k] = K*3 + (k*3)
            names['varyy%i'%k] = K*3 + (k*3)+1
            names['varxy%i'%k] = K*3 + (k*3)+2
        # print 'Setting param names:', names
        self.addNamedParams(**names)

    def mogAt(self, x, y):
        return self
        
    def get_wmuvar(self):
        return (self.mog.amp, self.mog.mean, self.mog.var)
        
    @classmethod
    def fromFitsHeader(clazz, hdr, prefix=''):
        params = []
        for i in range(100):
            k = prefix + 'P%i' % i
            print('Key', k)
            if not k in hdr:
                break
            params.append(hdr.get(k))
        print('PSF Params:', params)
        if len(params) == 0 or (len(params) % 6 != 0):
            raise RuntimeError('Failed to create %s from FITS header: expected '
                               'factor of 6 parameters, got %i' % 
                               (str(clazz), len(params)))
        K = len(params) / 6
        psf = clazz(np.zeros(K), np.zeros((K,2)), np.zeros((K,2,2)))
        psf.setParams(params)
        return psf
    
    def getMixtureOfGaussians(self, px=None, py=None):
        return self.mog

    def applyTo(self, image):
        raise
    
    def scaleBy(self, factor):
        # Use not advised, ever
        amp = self.mog.amp
        mean = self.mog.mean * factor
        var = self.mog.var * factor**2
        return GaussianMixturePSF(amp, mean, var)

    def shiftBy(self, dx, dy):
        self.mog.mean[:,0] += dx
        self.mog.mean[:,1] += dy
    
    def computeRadius(self):
        import numpy.linalg
        # ?
        meig = max([max(abs(numpy.linalg.eigvalsh(v)))
                    for v in self.mog.var])
        return self.getNSigma() * np.sqrt(meig)
        
    def getNSigma(self):
        # MAGIC -- N sigma for rendering patches
        return 5.

    def getRadius(self):
        return self.radius

    # returns a Patch object.
    def getPointSourcePatch(self, px, py, minval=0., radius=None,
                            derivs=False, minradius=None, modelMask=None,
                            clipExtent=None,
                            **kwargs):
        # clipExtent: [xl,xh, yl,yh] to avoid rendering a model that extends
        # outside the image bounds.
        # 
        if modelMask is not None:
            if modelMask.mask is None:
                return self.mog.evaluate_grid(modelMask.x0, modelMask.x1,
                                              modelMask.y0, modelMask.y1,
                                              px, py)
            else:
                return self.mog.evaluate_grid_masked(modelMask.x0, modelMask.y0,
                                                     modelMask.mask, px, py,
                                                     derivs=derivs, **kwargs)

        def get_extent(px, py, rr, clipExtent):
            x0 = int(np.floor(px - rr))
            x1 = int(np.ceil (px + rr)) + 1
            y0 = int(np.floor(py - rr))
            y1 = int(np.ceil (py + rr)) + 1
            if clipExtent is not None:
                [xl,xh,yl,yh] = clipExtent
                # clip
                x0 = max(x0, xl)
                x1 = min(x1, xh)
                y0 = max(y0, yl)
                y1 = min(y1, yh)
            return x0,x1,y0,y1

        # Yuck!

        if minval is None:
            minval = 0.
        if minval > 0. or minradius is not None:
            if radius is not None:
                rr = radius
            elif self.radius is not None:
                rr = self.radius
            else:
                r = 0.
                for v in self.mog.var:
                    # overestimate
                    vv = (v[0,0] + v[1,1])
                    norm = 2. * np.pi * np.linalg.det(v)
                    r2 = vv * -2. * np.log(minval * norm)
                    if r2 > 0:
                        r = max(r, np.sqrt(r2))
                rr = int(np.ceil(r))

            x0,x1,y0,y1 = get_extent(px, py, rr, clipExtent)
            if x0 >= x1 or y0 >= y1:
                return None

            kwa = {}
            if minradius is not None:
                kwa['minradius'] = minradius
            
            return self.mog.evaluate_grid_approx3(
                x0, x1, y0, y1, px, py, minval, derivs=derivs, **kwa)

        if radius is None:
            r = self.getRadius()
        else:
            r = radius

        x0,x1,y0,y1 = get_extent(px, py, r, clipExtent)
        if x0 >= x1 or y0 >= y1:
            return None
        return self.mog.evaluate_grid(x0, x1, y0, y1, px, py)

    def __str__(self):
        return (
            'GaussianMixturePSF: amps=' + str(tuple(self.mog.amp.ravel())) +
            ', means=' + str(tuple(self.mog.mean.ravel())) +
            ', var=' + str(tuple(self.mog.var.ravel())))

    def _numberOfThings(self):
        K = self.mog.K
        return K * (1 + 2 + 3)
    def _getThings(self):
        p = list(self.mog.amp) + list(self.mog.mean.ravel())
        for v in self.mog.var:
            p += (v[0,0], v[1,1], v[0,1])
        return p
    def _getThing(self, i):
        return self._getThings()[i]
    def _setThings(self, p):
        K = self.mog.K
        self.mog.amp = np.atleast_1d(p[:K])
        pp = p[K:]
        self.mog.mean = np.atleast_2d(pp[:K*2]).reshape(K,2)
        pp = pp[K*2:]
        self.mog.var[:,0,0] = pp[::3]
        self.mog.var[:,1,1] = pp[1::3]
        self.mog.var[:,0,1] = self.mog.var[:,1,0] = pp[2::3]
    def _setThing(self, i, p):
        K = self.mog.K
        if i < K:
            old = self.mog.amp[i]
            self.mog.amp[i] = p
            return old
        i -= K
        if i < K*2:
            old = self.mog.mean.ravel()[i]
            self.mog.mean.ravel()[i] = p
            return old
        i -= K*2
        j = i / 3
        k = i % 3
        if k in [0,1]:
            old = self.mog.var[j,k,k]
            self.mog.var[j,k,k] = p
            return old
        old = self.mog.var[j,0,1]
        self.mog.var[j,0,1] = p
        self.mog.var[j,1,0] = p
        return old

    @staticmethod
    def fromStamp(stamp, N=3, P0=None, xy0=None, alpha=0.,
                  emsteps=1000, v2=False, approx=1e-30,
                  v3=False):
        '''
        optional P0 = (w,mu,var): initial parameter guess.

        w has shape (N,)
        mu has shape (N,2)
        var (variance) has shape (N,2,2)

        optional xy0 = int x0,y0 origin of stamp.
        '''
        from emfit import em_fit_2d, em_fit_2d_reg
        from fitpsf import em_init_params
        if P0 is not None:
            w,mu,var = P0
        else:
            w,mu,var = em_init_params(N, None, None, None)
        stamp = stamp.copy()
        
        if xy0 is None:
            xm, ym = -(stamp.shape[1]/2), -(stamp.shape[0]/2)
        else:
            xm, ym = xy0

        if v3:
            tpsf = GaussianMixturePSF(w, mu, var)
            tim = Image(data=stamp, invvar=1e6*np.ones_like(stamp),
                        psf=tpsf)
            h,w = tim.shape
            src = PointSource(PixPos(w/2, h/2), Flux(1.))
            tr = Tractor([tim],[src])
            tr.freezeParam('catalog')
            tim.freezeAllBut('psf')
            tim.modelMinval = approx
            for step in range(20):
                dlnp,X,alpha = tr.optimize(shared_params=False)
                print('dlnp', dlnp)
                if dlnp < 1e-6:
                    break
            return tpsf
                
        elif v2:
            from emfit import em_fit_2d_reg2
            print('stamp sum:', np.sum(stamp))
            #stamp *= 1000.
            ok,skyamp = em_fit_2d_reg2(stamp, xm, ym, w, mu, var, alpha,
                                       emsteps, approx)
            #print 'sky amp:', skyamp
            #print 'w sum:', sum(w)
            tpsf = GaussianMixturePSF(w, mu, var)
            return tpsf,skyamp
        else:

            stamp /= stamp.sum()
            stamp = np.maximum(stamp, 0)

            em_fit_2d_reg(stamp, xm, ym, w, mu, var, alpha, emsteps)

        tpsf = GaussianMixturePSF(w, mu, var)
        return tpsf

class HybridPixelizedPSF(HybridPSF):
    '''
    This class wraps a PixelizedPSF model, adding a Gaussian approximation
    model.
    '''
    def __init__(self, pix, gauss=None, N=2):
        '''
        Create a new hybrid PSF model using the given PixelizedPSF
        model *pix* and Gaussian approximation *gauss*.
        
        If *gauss* is *None*, a *GaussianMixturePSF* model will be fit
        to the PixelizedPSF image using *N* Gaussian components.
        '''
        super(HybridPixelizedPSF, self).__init__()
        self.pix = pix
        if gauss is None:
            gauss = GaussianMixturePSF.fromStamp(pix.getImage(0.,0.), N=N)
        self.gauss = gauss

    def __str__(self):
        return ('HybridPixelizedPSF: Gaussian sigma %.2f, Pix %s' %
                (np.sqrt(self.gauss.mog.var[0,0,0]), str(self.pix)))
        
    def getMixtureOfGaussians(self, **kwargs):
        return self.gauss.getMixtureOfGaussians(**kwargs)

    def getShifted(self, dx, dy):
        pix = self.pix.getShifted(dx, dy)
        return HybridPixelizedPSF(pix, self.gauss)
    
    def constantPsfAt(self, x, y):
        pix = self.pix.constantPsfAt(x, y)
        return HybridPixelizedPSF(pix, self.gauss)

    def __getattr__(self, name):
        '''Delegate to my pixelized PSF model.'''
        return getattr(self.pix, name)

    def __setattr__(self, name, val):
        '''Delegate to my pixelized PSF model.'''
        if name in ['gauss', 'pix']:
            return object.__setattr__(self, name, val)
        setattr(self.__dict__['pix'], name, val)

    # for pickling:
    def __getstate__(self):
        return (self.gauss, self.pix)
     
    def __setstate__(self, state):
        self.gauss, self.pix = state

class GaussianMixtureEllipsePSF(GaussianMixturePSF):
    '''
    A variant of GaussianMixturePSF that uses EllipseESoft to describe
    the covariance ellipse.
    '''

    def __init__(self, *args):
        '''
        args = (amp, mean, ell)

        or

        args = (a0,a1,..., mx0,my0,mx1,my1,..., logr0,ee1-0,ee2-0,logr1,ee1-2,...)

        
        amp:  np array (size K) of Gaussian amplitudes
        mean: np array (size K,2) of means
        ell:  list (length K) of EllipseESoft objects
        '''
        if len(args) == 3:
            amp, mean, ell = args
        else:
            from .ellipses import EllipseESoft
            assert(len(args) % 6 == 0)
            K = len(args) / 6
            amp  = np.array(args[:K])
            mean = np.array(args[K:3*K]).reshape((K,2))
            args = args[3*K:]
            ell = [EllipseESoft(*args[3*k: 3*(k+1)]) for k in range(K)]

        K = len(amp)
        var = np.zeros((K,2,2))
        for k in range(K):
            var[k,:,:] = self.ellipseToVariance(ell[k])
        self.ellipses = [e.copy() for e in ell]
        super(GaussianMixtureEllipsePSF, self).__init__(amp, mean, var)
        self.stepsizes = [0.001]*K + [0.001]*(K*2) + [0.001]*(K*3)

    def ellipseToVariance(self, ell):
        return ell.getCovariance()
        
    def getShifted(self, x0, y0):
        # not spatially varying
        return self
    
    def _set_param_names(self, K):
        names = {}
        for k in range(K):
            names['amp%i'%k] = k
            names['meanx%i'%k] = K+(k*2)
            names['meany%i'%k] = K+(k*2)+1
            names['logr%i'%k] = K*3 + (k*3)
            names['ee1-%i'%k] = K*3 + (k*3)+1
            names['ee2-%i'%k] = K*3 + (k*3)+2
        self.addNamedParams(**names)

    def toMog(self):
        return GaussianMixturePSF(self.mog.amp, self.mog.mean, self.mog.var)

    def mogAt(self, x, y):
        return self.toMog()

    def constantPsfAt(self, x, y):
        return self.mogAt(x, y)
        
    def __str__(self):
        return (
            'GaussianMixtureEllipsePSF: amps=' +
            '['+', '.join(['%.3f' % a for a in self.mog.amp.ravel()]) + ']' +
            ', means=[' + ', '.join([
                '(%.3f, %.3f)' % (x,y) for x,y in self.mog.mean]) + ']' +
            ', ellipses=' + ', '.join(str(e) for e in self.ellipses) +
            ', var=' + str(tuple(self.mog.var.ravel())))
    
    def _getThings(self):
        p = list(self.mog.amp) + list(self.mog.mean.ravel())
        for e in self.ellipses:
            p += e.getAllParams()
        return p
    def _setThings(self, p):
        K = self.mog.K
        self.mog.amp = np.atleast_1d(p[:K])
        pp = p[K:]
        self.mog.mean = np.atleast_2d(pp[:K*2]).reshape(K,2)
        pp = pp[K*2:]
        for i,e in enumerate(self.ellipses):
            e.setAllParams(pp[:3])
            pp = pp[3:]
            self.mog.var[i,:,:] = self.ellipseToVariance(e)
    def _setThing(self, i, p):
        ## hack
        things = self._getThings()
        old = things[i]
        things[i] = p
        self._setThings(things)
        return old

    @staticmethod
    def fromStamp(stamp, N=3, P0=None, approx=1e-6, damp=0.):
        '''
        optional P0 = (list of floats): initial parameter guess.

        (parameters of a GaussianMixtureEllipsePSF)
        '''
        from .ellipses import EllipseESoft
        w = np.ones(N) / float(N)
        mu = np.zeros((N,2))
        ell = [EllipseESoft(np.log(2*r), 0., 0.) for r in range(1, N+1)]
        psf = GaussianMixtureEllipsePSF(w, mu, ell)
        if P0 is not None:
            psf.setAllParams(P0)
        tim = Image(data=stamp, invvar=1e6*np.ones_like(stamp), psf=psf)
        H,W = stamp.shape
        src = PointSource(PixPos(W/2, H/2), Flux(1.))
        tr = Tractor([tim],[src])
        tr.freezeParam('catalog')
        tim.freezeAllBut('psf')
        #print 'Fitting:'
        #tr.printThawedParams()
        tim.modelMinval = approx
        alphas = [0.1, 0.3, 1.0]
        for step in range(50):
            dlnp,X,alpha = tr.optimize(shared_params=False, alphas=alphas,
                                       damp=damp)
            #print 'dlnp', dlnp, 'alpha', alpha
            #print 'X', X
            if dlnp < 1e-6:
                break
            #print 'psf', psf
        return psf
    
class NCircularGaussianPSF(MultiParams, ducks.ImageCalibration):
    '''
    A PSF model using N concentric, circular Gaussians.
    '''
    def __init__(self, sigmas, weights):
        '''
        Creates a new N-Gaussian (concentric, isotropic) PSF.

        sigmas: (list of floats) standard deviations of the components

        weights: (list of floats) relative weights of the components;
        given two components with weights 0.9 and 0.1, the total mass
        due to the second component will be 0.1.  These values will be
        normalized so that the total mass of the PSF is 1.0.

        eg,   NCircularGaussianPSF([1.5, 4.0], [0.8, 0.2])
        '''
        assert(len(sigmas) == len(weights))
        psigmas = ParamList(*sigmas)
        psigmas.stepsizes = [0.01] * len(sigmas)
        pweights = ParamList(*weights)
        pweights.stepsizes = [0.01] * len(weights)
        super(NCircularGaussianPSF, self).__init__(psigmas, pweights)
        self.minradius = 1.

    def getShifted(self, x0, y0):
        # not spatially varying
        return self

    def constantPsfAt(self, x, y):
        return self
    
    @property
    def amp(self):
        return self.weights

    @property
    def mog(self):
        return self.getMixtureOfGaussians()
    
    @staticmethod
    def getNamedParams():
        return dict(sigmas=0, weights=1)

    def __str__(self):
        return ('NCircularGaussianPSF: sigmas [ ' +
                ', '.join(['%.3f'%s for s in self.sigmas]) +
                ' ], weights [ ' +
                ', '.join(['%.3f'%w for w in self.weights]) +
                ' ]')

    def __repr__(self):
        return ('NCircularGaussianPSF: sigmas [ ' +
                ', '.join(['%.3f'%s for s in self.sigmas]) +
                ' ], weights [ ' +
                ', '.join(['%.3f'%w for w in self.weights]) +
                ' ]')

    # Get the real underlying ones without paying attention to frozen state
    @property
    def mysigmas(self):
        return self.sigmas.vals
    @property
    def myweights(self):
        return self.weights.vals

    def scale(self, factor):
        ''' Returns a new PSF that is *factor* times wider. '''
        return NCircularGaussianPSF(np.array(self.mysigmas) * factor,
                                    self.myweights)

    def getMixtureOfGaussians(self, px=None, py=None):
        K = len(self.myweights)
        amps = np.array(self.myweights)
        means = np.zeros((K,2))
        vars = np.zeros((K,2,2))
        for k in range(K):
            vars[k,0,0] = vars[k,1,1] = self.mysigmas[k]**2
        return mp.MixtureOfGaussians(amps, means, vars)
        
    def hashkey(self):
        hk = ('NCircularGaussianPSF', tuple(self.sigmas), tuple(self.weights))
        return hk
    
    def copy(self):
        return NCircularGaussianPSF(list([s for s in self.sigmas]),
                                    list([w for w in self.weights]))

    def applyTo(self, image):
        from scipy.ndimage.filters import gaussian_filter
        # gaussian_filter normalizes the Gaussian; the output has ~ the
        # same sum as the input.
        res = np.zeros_like(image)
        for s,w in zip(self.sigmas, self.weights):
            res += w * gaussian_filter(image, s)
        res /= sum(self.weights)
        return res

    def getNSigma(self):
        # HACK - MAGIC -- N sigma for rendering patches
        return 5.

    def getRadius(self):
        if hasattr(self, 'radius'):
            return self.radius
        return max(self.minradius, max(self.mysigmas) * self.getNSigma())

    # returns a Patch object.
    def getPointSourcePatch(self, px, py, minval=0., radius=None,
                            modelMask=None, **kwargs):
        if modelMask is not None:
            x0,x1,y0,y1 = modelMask.extent
        else:
            ix = int(round(px))
            iy = int(round(py))
            if radius is None:
                rad = int(np.ceil(self.getRadius()))
            else:
                rad = radius
            x0 = ix - rad
            x1 = ix + rad + 1
            y0 = iy - rad
            y1 = iy + rad + 1
        mix = self.getMixtureOfGaussians()
        mix.mean[:,0] += px
        mix.mean[:,1] += py
        return mp.mixture_to_patch(mix, x0, x1, y0, y1, minval=minval,
                                   exactExtent=(modelMask is not None))
