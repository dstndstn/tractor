from __future__ import print_function
from __future__ import division

import sys
import functools

import numpy as np

from tractor.image import Image
from tractor.pointsource import PointSource
from tractor.wcs import PixPos
from tractor.brightness import Flux
from tractor.engine import Tractor
from tractor.patch import Patch
from tractor.utils import BaseParams, ParamList, MultiParams, MogParams
from tractor import mixture_profiles as mp
from tractor import ducks

if sys.version_info[0] == 2:
    # Py2
    def round(x):
        import __builtin__
        return int(__builtin__.round(float(x)))

mp_fourier = -1
def lanczos_shift_image(img, dx, dy, inplace=False, force_python=False):
    global mp_fourier
    if mp_fourier == -1:
        try:
            from tractor import mp_fourier
        except:
            print('tractor.psf: failed to import C version of mp_fourier library.  Falling back to python version.')
            mp_fourier = None

    H,W = img.shape
    if (mp_fourier is None or force_python or W <= 8 or H <= 8
        or H > work_corr7f.shape[0] or W > work_corr7f.shape[1]):
        # fallback to python:
        from scipy.ndimage import correlate1d
        from astrometry.util.miscutils import lanczos_filter
        L = 3
        Lx = lanczos_filter(L, np.arange(-L, L+1) + dx)
        Ly = lanczos_filter(L, np.arange(-L, L+1) + dy)
        # Normalize the Lanczos interpolants (preserve flux)
        Lx /= Lx.sum()
        Ly /= Ly.sum()
        sx     = correlate1d(img, Lx, axis=1, mode='constant')
        outimg = correlate1d(sx,  Ly, axis=0, mode='constant')
        return outimg

    outimg = np.empty(img.shape, np.float32)
    mp_fourier.lanczos_shift_3f(img.astype(np.float32), outimg, dx, dy,
                                work_corr7f)
    # yuck!  (don't change this without ensuring the "restrict"
    # keyword still applies in lanczos_shift_3f!)
    if inplace:
        img[:,:] = outimg
    return outimg

def lanczos_shift_image_batch_gpu(imgs, dxs, dys):
    """Translated from lanczos_shift_image python version to GPU using cupy
        and helper functions from tractor.miscutils"""
    import cupy as cp
    from tractor.miscutils import gpu_lanczos_filter,batch_correlate1d_gpu
    L = 3
    nimg = dxs.size 
    lr = cp.tile(cp.arange(-L, L+1), (nimg, 1))
    Lx = gpu_lanczos_filter(L, lr+dxs.reshape((nimg,1)))
    Ly = gpu_lanczos_filter(L, lr+dys.reshape((nimg,1)))
    # Normalize the Lanczos interpolants (preserve flux)
    Lx /= Lx.sum(1).reshape((nimg,1))
    Ly /= Ly.sum(1).reshape((nimg,1))
    sx = batch_correlate1d_gpu(imgs, Lx, axis=2, mode='constant')
    outimg = batch_correlate1d_gpu(sx, Ly, axis=1, mode='constant')
    return outimg

# GLOBAL scratch array for lanczos_shift_image!
work_corr7f = np.zeros((4096, 4096), np.float32)
work_corr7f = np.require(work_corr7f, requirements=['A'])

class HybridPSF(object):
    pass


class PixelizedPSF(BaseParams, ducks.ImageCalibration):
    '''
    A PSF model based on an image postage stamp, which will be
    sinc-shifted to subpixel positions.

    Galaxies will be rendered using FFT convolution.

    Also handles the case where the PSF model is sampled at a
    different pixel spacing than the native pixel, eg, an oversampled
    model to be used when the image itself is undersampled.

    FIXME -- currently this class claims to have no params.
    '''

    def __init__(self, img, sampling=1., Lorder=3):
        '''
        Creates a new PixelizedPSF object from the given *img* (numpy
        array) image of the PSF.

        - *img* must be an ODD size.
        - *Lorder* is the order of the Lanczos interpolant used for
           shifting the image to subpixel positions.
        '''
        # ensure float32 and align
        img = img.astype(np.float32)
        self.img = np.require(img, requirements=['A'])
        H,W = img.shape
        assert((H % 2) == 1)
        assert((W % 2) == 1)
        self.radius = np.hypot(H / 2., W / 2.)
        self.H, self.W = H, W
        self.Lorder = Lorder
        self.fftcache = {}
        self.sampling = sampling
        if sampling != 1.:
            # The size of PSF image we will return.
            self.nativeW = int(np.ceil(self.W * self.sampling))
            self.nativeH = int(np.ceil(self.H * self.sampling))

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
        return self.__class__(self.img.copy())

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

    def getPointSourcePatch(self, px, py, minval=0., modelMask=None,
                            radius=None, **kwargs):
        if self.sampling != 1.:
            return self._getOversampledPointSourcePatch(px, py, minval=minval,
                                                        modelMask=modelMask,
                                                        radius=radius, **kwargs)

        from astrometry.util.miscutils import get_overlapping_region

        # get PSF image at desired pixel location
        img = self.getImage(px, py)

        if radius is not None:
            R = int(np.ceil(radius))
            H,W = img.shape
            cx = W//2
            cy = H//2
            img = img[max(cy-R, 0) : min(cy+R+1,H-1),
                      max(cx-R, 0) : min(cx+R+1,W-1)]
            
        H, W = img.shape
        # float() required because builtin round(np.float64(11.0)) returns 11.0 !!
        ix = round(float(px))
        iy = round(float(py))
        dx = px - ix
        dy = py - iy
        x0 = ix - W // 2
        y0 = iy - H // 2

        if modelMask is None:
            outimg = lanczos_shift_image(img, dx, dy)
            return Patch(x0, y0, outimg)

        mh, mw = modelMask.shape
        mx0, my0 = modelMask.x0, modelMask.y0

        # print 'PixelizedPSF + modelMask'
        # print 'mx0,my0', mx0,my0, '+ mw,mh', mw,mh
        # print 'PSF image x0,y0', x0,y0, '+ W,H', W,H

        if ((mx0 >= x0 + W) or (mx0 + mw <= x0) or
            (my0 >= y0 + H) or (my0 + mh <= y0)):
            # No overlap
            return None
        # Otherwise, we'll just produce the Lanczos-shifted PSF
        # image as usual, and then copy it into the modelMask
        # space.
        L = 3
        padding = L
        # Create a modelMask + padding sized stamp and insert PSF image into it
        mm = np.zeros((mh+2*padding, mw+2*padding), np.float32)
        yi,yo = get_overlapping_region(my0-y0-padding, my0-y0+mh-1+padding, 0, H-1)
        xi,xo = get_overlapping_region(mx0-x0-padding, mx0-x0+mw-1+padding, 0, W-1)
        mm[yo,xo] = img[yi,xi]
        mm = lanczos_shift_image(mm, dx, dy)
        mm = mm[padding:-padding, padding:-padding]
        assert(np.all(np.isfinite(mm)))

        return Patch(mx0, my0, mm)

    def getFourierTransformSize(self, radius):
        # Next power-of-two size
        sz = 2**int(np.ceil(np.log2(radius * 2.)))
        return sz

    def _padInImage(self, H, W, img=None):
        '''
        Embeds this PSF image into a larger or smaller image of shape H,W.

        Return (img, cx, cy), where *cx*,*cy* are the coordinates of the PSF
        center in *img*.
        '''
        if img is None:
            img = self.img
        ph, pw = img.shape
        subimg = img

        if H >= ph:
            y0 = (H - ph) // 2
            cy = y0 + ph // 2
        else:
            y0 = 0
            cut = (ph - H) // 2
            subimg = subimg[cut:cut + H, :]
            cy = ph // 2 - cut

        if W >= pw:
            x0 = (W - pw) // 2
            cx = x0 + pw // 2
        else:
            x0 = 0
            cut = (pw - W) // 2
            subimg = subimg[:, cut:cut + W]
            cx = pw // 2 - cut
        sh, sw = subimg.shape

        pad = np.zeros((H, W), img.dtype)
        pad[y0:y0 + sh, x0:x0 + sw] = subimg
        return pad, cx, cy

    def getFourierTransform(self, px, py, radius):
        '''
        Returns the Fourier Transform of this PSF, with the
        next-power-of-2 size up from *radius*.

        Returns: (FFT, (xc, yc), (imh,imw), (v,w))

        *FFT*: numpy array, the FFT
        *xc*: float, pixel location of the PSF /center/ in the PSF subimage
        *yc*:    ditto
        *imh,imw*: ints, shape of the padded PSF image
        *v,w*: v=np.fft.rfftfreq(imw), w=np.fft.fftfreq(imh)

        '''
        if self.sampling != 1.:
            return self._getOversampledFourierTransform(px, py, radius)

        sz = self.getFourierTransformSize(radius)
        # print 'PixelizedPSF FFT size', sz
        if sz in self.fftcache:
            return self.fftcache[sz]

        pad, cx, cy = self._padInImage(sz, sz)
        # cx,cy: coordinate of the PSF center in *pad*
        P = np.fft.rfft2(pad)
        P = P.astype(np.complex64)
        pH, pW = pad.shape
        v = np.fft.rfftfreq(pW)
        w = np.fft.fftfreq(pH)
        rtn = P, (cx, cy), (pH, pW), (v, w)
        self.fftcache[sz] = rtn
        return rtn

    # The following routines are used when sampling != 1.0

    def _sampleImage(self, img, dx, dy,
                     xlo=None, ylo=None, width=None, height=None):
        from astrometry.util.util import lanczos3_interpolate_grid
        if img is None:
            img = self.img
        if xlo is None:
            xlo = -(self.nativeW//2)
        if ylo is None:
            ylo = -(self.nativeH//2)
        if width is None:
            width = self.nativeW
        if height is None:
            height = self.nativeH

        H,W = img.shape
        cx = W//2
        cy = H//2

        xstep = 1./self.sampling
        ystep = 1./self.sampling
        xstart = (cx - dx/self.sampling) + xlo * xstep
        ystart = (cy - dy/self.sampling) + ylo * ystep

        native_img = np.zeros((height, width), np.float32)
        lanczos3_interpolate_grid(xstart, xstep, ystart, ystep,
                                  native_img, img)
        return xlo, ylo, native_img

    def _getOversampledPointSourcePatch(self, px, py, minval=0., modelMask=None,
                                        radius=None, **kwargs):
        # get PSF image at desired pixel location
        img = self.getImage(px, py)

        ix = round(float(px))
        iy = round(float(py))
        dx = px - ix
        dy = py - iy

        if modelMask is not None:
            mh, mw = modelMask.shape
            mx0, my0 = modelMask.x0, modelMask.y0
            xl,yl,native_img = self._sampleImage(img, dx, dy, xlo=mx0-ix, ylo=my0-iy,
                                                 width=mw, height=mh)
            return Patch(xl+ix, yl+iy, native_img)

        xl,yl,img = self._sampleImage(img, dx, dy)
        x0 = ix + xl
        y0 = iy + yl

        if radius is not None:
            R = int(np.ceil(radius))
            H,W = img.shape
            cx = W//2
            cy = H//2
            xlo = max(cx-R, 0)
            ylo = max(cy-R, 0)
            img = img[ylo : min(cy+R+1,H-1),
                      xlo : min(cx+R+1,W-1)]
            x0 += xlo
            y0 += ylo

        return Patch(x0, y0, img)

    def _getOversampledFourierTransform(self, px, py, radius):
        sz = self.getFourierTransformSize(radius)
        key = (sz, px, py)
        if key in self.fftcache:
            return self.fftcache[key]
        # shift by fractional pixel
        dx = px - int(px)
        dy = py - int(py)
        _, _, img = self._sampleImage(None, dx, dy)
        pad, cx, cy = self._padInImage(sz, sz, img=img)
        cx += dx
        cy += dy
        # cx,cy: coordinate of the PSF center in *pad*
        P = np.fft.rfft2(pad)
        P = P.astype(np.complex64)
        pH, pW = pad.shape
        v = np.fft.rfftfreq(pW)
        w = np.fft.fftfreq(pH)
        rtn = P, (cx, cy), (pH, pW), (v, w)
        self.fftcache[key] = rtn
        return rtn

class GaussianMixturePSF(MogParams, ducks.ImageCalibration):
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
        super(GaussianMixturePSF, self).__init__(*args)
        assert(self.mog.D == 2)
        self.radius = 25
        K = self.mog.K
        self.stepsizes = [0.01] * K + [0.01] * (K * 2) + [0.1] * (K * 3)

    def getShifted(self, x0, y0):
        # not spatially varying
        return self

    def constantPsfAt(self, x, y):
        return self

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
        K = len(params) // 6
        psf = clazz(np.zeros(K), np.zeros((K, 2)), np.zeros((K, 2, 2)))
        psf.setParams(params)
        return psf

    def getMixtureOfGaussians(self, px=None, py=None):
        return self.mog

    def applyTo(self, image):
        raise RuntimeError('Not implemented')

    def scaleBy(self, factor):
        # Use not advised, ever
        amp = self.mog.amp
        mean = self.mog.mean * factor
        var = self.mog.var * factor**2
        return GaussianMixturePSF(amp, mean, var)

    def shiftBy(self, dx, dy):
        self.mog.mean[:, 0] += dx
        self.mog.mean[:, 1] += dy

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
            x1 = int(np.ceil(px + rr)) + 1
            y0 = int(np.floor(py - rr))
            y1 = int(np.ceil(py + rr)) + 1
            if clipExtent is not None:
                [xl, xh, yl, yh] = clipExtent
                # clip
                x0 = max(x0, xl)
                x1 = min(x1, xh)
                y0 = max(y0, yl)
                y1 = min(y1, yh)
            return x0, x1, y0, y1

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
                    vv = (v[0, 0] + v[1, 1])
                    norm = 2. * np.pi * np.linalg.det(v)
                    r2 = vv * -2. * np.log(minval * norm)
                    if r2 > 0:
                        r = max(r, np.sqrt(r2))
                rr = int(np.ceil(r))

            x0, x1, y0, y1 = get_extent(px, py, rr, clipExtent)
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

        x0, x1, y0, y1 = get_extent(px, py, r, clipExtent)
        if x0 >= x1 or y0 >= y1:
            return None
        return self.mog.evaluate_grid(x0, x1, y0, y1, px, py)

    def __str__(self):
        return (
            'GaussianMixturePSF: amps=' + str(tuple(self.mog.amp.ravel())) +
            ', means=' + str(tuple(self.mog.mean.ravel())) +
            ', var=' + str(tuple(self.mog.var.ravel())))

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
        from tractor.emfit import em_fit_2d_reg
        from tractor.fitpsf import em_init_params
        if P0 is not None:
            w, mu, var = P0
        else:
            w, mu, var = em_init_params(N, None, None, None)
        stamp = stamp.copy()

        if xy0 is None:
            xm, ym = -(stamp.shape[1] // 2), -(stamp.shape[0] // 2)
        else:
            xm, ym = xy0

        if v3:
            tpsf = GaussianMixturePSF(w, mu, var)
            tim = Image(data=stamp, invvar=1e6 * np.ones_like(stamp),
                        psf=tpsf)
            h, w = tim.shape
            src = PointSource(PixPos(w // 2, h // 2), Flux(1.))
            tr = Tractor([tim], [src])
            tr.freezeParam('catalog')
            tim.freezeAllBut('psf')
            tim.modelMinval = approx
            for step in range(20):
                dlnp, X, alpha = tr.optimize(shared_params=False)
                print('dlnp', dlnp)
                if dlnp < 1e-6:
                    break
            return tpsf

        elif v2:
            from tractor.emfit import em_fit_2d_reg2
            print('stamp sum:', np.sum(stamp))
            #stamp *= 1000.
            ok, skyamp = em_fit_2d_reg2(stamp, xm, ym, w, mu, var, alpha,
                                        emsteps, approx)
            # print 'sky amp:', skyamp
            # print 'w sum:', sum(w)
            tpsf = GaussianMixturePSF(w, mu, var)
            return tpsf, skyamp
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

    def __init__(self, pix, gauss=None, N=2, cx=0., cy=0.):
        '''
        Create a new hybrid PSF model using the given PixelizedPSF
        model *pix* and Gaussian approximation *gauss*.

        If *gauss* is *None*, a *GaussianMixturePSF* model will be fit
        to the PixelizedPSF image using *N* Gaussian components.
        '''
        super(HybridPixelizedPSF, self).__init__()
        self.pix = pix
        if gauss is None:
            gauss = GaussianMixturePSF.fromStamp(pix.getImage(cx, cy), N=N)
        #print('Fit Gaussian PSF model:', gauss)
        self.gauss = gauss

    def __str__(self):
        return ('HybridPixelizedPSF: Gaussian sigma %.2f, Pix %s' %
                (np.sqrt(self.gauss.mog.var[0, 0, 0]), str(self.pix)))

    def copy(self):
        s = self.__class__(self.pix.copy(), self.gauss.copy())
        return s
        
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
            from tractor.ellipses import EllipseESoft
            assert(len(args) % 6 == 0)
            K = len(args) // 6
            amp = np.array(args[:K])
            mean = np.array(args[K:3 * K]).reshape((K, 2))
            args = args[3 * K:]
            ell = [EllipseESoft(*args[3 * k: 3 * (k + 1)]) for k in range(K)]

        K = len(amp)
        var = np.zeros((K, 2, 2))
        for k in range(K):
            var[k, :, :] = self.ellipseToVariance(ell[k])
        self.ellipses = [e.copy() for e in ell]
        super(GaussianMixtureEllipsePSF, self).__init__(amp, mean, var)
        self.stepsizes = [0.001] * K + [0.001] * (K * 2) + [0.001] * (K * 3)

    def ellipseToVariance(self, ell):
        return ell.getCovariance()

    def getShifted(self, x0, y0):
        # not spatially varying
        return self

    def _set_param_names(self, K):
        names = {}
        for k in range(K):
            names['amp%i' % k] = k
            names['meanx%i' % k] = K + (k * 2)
            names['meany%i' % k] = K + (k * 2) + 1
            names['logr%i' % k] = K * 3 + (k * 3)
            names['ee1-%i' % k] = K * 3 + (k * 3) + 1
            names['ee2-%i' % k] = K * 3 + (k * 3) + 2
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
            '[' + ', '.join(['%.3f' % a for a in self.mog.amp.ravel()]) + ']' +
            ', means=[' + ', '.join([
                '(%.3f, %.3f)' % (x, y) for x, y in self.mog.mean]) + ']' +
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
        self.mog.mean = np.atleast_2d(pp[:K * 2]).reshape(K, 2)
        pp = pp[K * 2:]
        for i, e in enumerate(self.ellipses):
            e.setAllParams(pp[:3])
            pp = pp[3:]
            self.mog.var[i, :, :] = self.ellipseToVariance(e)

    def _setThing(self, i, p):
        # hack
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
        from tractor.ellipses import EllipseESoft
        w = np.ones(N) / float(N)
        mu = np.zeros((N, 2))
        ell = [EllipseESoft(np.log(2 * r), 0., 0.) for r in range(1, N + 1)]
        psf = GaussianMixtureEllipsePSF(w, mu, ell)
        if P0 is not None:
            psf.setAllParams(P0)
        tim = Image(data=stamp, invvar=1e6 * np.ones_like(stamp), psf=psf)
        H, W = stamp.shape
        src = PointSource(PixPos(W // 2, H // 2), Flux(1.))
        tr = Tractor([tim], [src])
        tr.freezeParam('catalog')
        tim.freezeAllBut('psf')
        print('Fitting:')
        tr.printThawedParams()
        tim.modelMinval = approx
        alphas = [0.1, 0.3, 1.0]
        for step in range(50):
            dlnp, X, alpha = tr.optimize(shared_params=False, alphas=alphas,
                                         damp=damp)
            # print 'dlnp', dlnp, 'alpha', alpha
            # print 'X', X
            if dlnp < 1e-6:
                break
            print('psf', psf)
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
                ', '.join(['%.3f' % s for s in self.sigmas]) +
                ' ], weights [ ' +
                ', '.join(['%.3f' % w for w in self.weights]) +
                ' ]')

    def __repr__(self):
        return ('NCircularGaussianPSF: sigmas [ ' +
                ', '.join(['%.3f' % s for s in self.sigmas]) +
                ' ], weights [ ' +
                ', '.join(['%.3f' % w for w in self.weights]) +
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
        amps = tuple(self.myweights)
        sigs = tuple(self.mysigmas)
        return getCircularMog(amps, sigs)

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
        for s, w in zip(self.sigmas, self.weights):
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
            x0, x1, y0, y1 = modelMask.extent
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
        # UGH - API question -- is the getMixtureOfGaussians() result read-only?
        mix = self.getMixtureOfGaussians()
        oldmean = mix.mean
        mix.mean = mix.mean.copy()
        mix.mean[:, 0] += px
        mix.mean[:, 1] += py
        p = mp.mixture_to_patch(mix, x0, x1, y0, y1, minval=minval,
                                exactExtent=(modelMask is not None))
        mix.mean = oldmean
        return p

def getCircularMog(amps, sigmas):
    K = len(amps)
    amps = np.array(amps).astype(np.float32)
    means = np.zeros((K, 2), np.float32)
    vars = np.zeros((K, 2, 2), np.float32)
    for k in range(K):
        vars[k, 0, 0] = vars[k, 1, 1] = sigmas[k]**2
    return mp.MixtureOfGaussians(amps, means, vars, quick=True)

# lru_cache is new in python 3.2
try:
    getCircularMog = functools.lru_cache(maxsize=16)(getCircularMog)
except:
    pass
