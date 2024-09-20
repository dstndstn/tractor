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
import cupy as cp

if sys.version_info[0] == 2:
    # Py2
    def round(x):
        import __builtin__
        return int(__builtin__.round(float(x)))

def lanczos_shift_image_batch_gpu(imgs, dxs, dys):
    """Translated from lanczos_shift_image python version to GPU using cupy
        and helper functions from tractor.miscutils"""
    import cupy as cp
    from tractor.miscutils import gpu_lanczos_filter,batch_correlate1d_gpu
    do_reshape = False
    if len(imgs.shape) == 4:
        do_reshape = True
        oldshape = imgs.shape
        imgs = imgs.reshape((oldshape[0]*oldshape[1], oldshape[2], oldshape[3]))
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
    if (do_reshape):
        outimg = outimg.reshape(oldshape)
    return outimg

class BatchHybridPSF(object):
    pass


class BatchPixelizedPSF(BaseParams, ducks.ImageCalibration):
    '''
    A PSF model based on N x image postage stamp, which will be
    sinc-shifted to subpixel positions.

    Galaxies will be rendered using FFT convolution.

    Also handles the case where the PSF model is sampled at a
    different pixel spacing than the native pixel, eg, an oversampled
    model to be used when the image itself is undersampled.

    FIXME -- currently this class claims to have no params.
    '''

    def __init__(self, psfs):
        '''
        Creates a new PixelizedPSF object from the given *img* (numpy
        array) image of the PSF.

        - *psfs* is a list of PixelizedPSF objects - img, Lorder, and sampling
            are taken from the individual psfs.
        - *img* must be an ODD size. 3D array (N x H x W)
        - *Lorder* is the order of the Lanczos interpolant used for
           shifting the image to subpixel positions.
        '''
        # ensure float32 and align
        N = len(psfs)
        iH = np.zeros(N, dtype=np.int32) #individual height
        iW = np.zeros(N, dtype=np.int32) #individual width

        #Find max w, h
        for i, psf in enumerate(psfs):
            iH[i], iW[i] = psf.img.shape
        H = np.max(iH)
        W = np.max(iW)
        img = np.zeros((N, H, W), dtype=np.float32)

        #Now loop over psfs and copy data into one 3-d zero-padded array
        for i, psf in enumerate(psfs):
            img[i,:iH[i],:iW[i]] = psf.img
        img = cp.asarray(img)
        #self.img = cp.require(img, requirements=['A'])
        self.img = img
        assert((H % 2) == 1)
        assert((W % 2) == 1)
        self.radius = cp.hypot(H / 2., W / 2.)
        self.N, self.H, self.W = N, H, W
        self.iH, self.iW = iH, iW #keep copies of original W and H
        self.Lorder = psfs[0].Lorder
        self.fftcache = {}
        self.sampling = psfs[0].sampling
        if self.sampling != 1.:
            # The size of PSF image we will return.
            self.nativeW = int(np.ceil(self.W * self.sampling))
            self.nativeH = int(np.ceil(self.H * self.sampling))

        from tractor.psf import HybridPSF
        from tractor.batch_mixture_profiles import BatchMixtureOfGaussians
        psfmogs = []
        maxK = 0
        for i,psf in enumerate(psfs):
            assert(isinstance(psf, HybridPSF))
            psfmog = psf.getMixtureOfGaussians()
            psfmogs.append(psfmog)
            maxK = max(maxK, psfmog.K)
        amps = np.zeros((N, maxK))
        means = np.zeros((N, maxK, 2))
        varrs = np.zeros((N, maxK, 2, 2))
        for i,psfmog in enumerate(psfmogs):
            amps[i, :psfmog.K] = psfmog.amp
            means[i, :psfmog.K, :] = psfmog.mean
            varrs[i, :psfmog.K, :, :] = psfmog.var
        amps = cp.asarray(amps)
        means = cp.asarray(means)
        varrs = cp.asarray(varrs)
        self.psf_mogs = BatchMixtureOfGaussians(amps, means, varrs, quick=True)


    def __str__(self):
        return 'BatchPixelizedPSF'

    def clear_cache(self):
        self.fftcache = {}

    @property
    def shape(self):
        return (self.H, self.W)

    def hashkey(self):
        return ('BatchPixelizedPSF', tuple(self.img.ravel()))

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

    def getFourierTransformSizeBatchGPU(self, radius):
        # Next power-of-two size
        sz = 2**int(np.ceil(np.log2(radius.max() * 2.)))
        return sz

    def _padInImageBatchGPU(self, H, W, img=None):
        '''
        Embeds this PSF image into a larger or smaller image of shape H,W.

        Return (img, cx, cy), where *cx*,*cy* are the coordinates of the PSF
        center in *img*.
        '''
        if img is None:
            img = self.img
        N, ph, pw = img.shape
        subimg = img

        if H >= ph:
            y0 = (H - ph) // 2
            cy = y0 + ph // 2
        else:
            y0 = 0
            cut = (ph - H) // 2
            subimg = subimg[:, cut:cut + H, :]
            cy = ph // 2 - cut

        if W >= pw:
            x0 = (W - pw) // 2
            cx = x0 + pw // 2
        else:
            x0 = 0
            cut = (pw - W) // 2
            subimg = subimg[:, :, cut:cut + W]
            cx = pw // 2 - cut
        N, sh, sw = subimg.shape

        pad = cp.zeros((N, H, W), img.dtype)
        pad[:, y0:y0 + sh, x0:x0 + sw] = subimg
        return pad, cx, cy

    def getFourierTransformBatchGPU(self, px, py, radius):
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
        import cupy as cp
        px = cp.asarray(px)
        py = cp.asarray(py)
        radius = cp.asarray(radius)
        if self.sampling != 1.:
            return self._getOversampledFourierTransformBatchGPU(px, py, radius)
    
        sz = self.getFourierTransformSizeBatchGPU(radius)
        # print 'PixelizedPSF FFT size', sz
        if sz in self.fftcache:
            return self.fftcache[sz]
        
        pad, cx, cy = self._padInImageBatchGPU(sz, sz)
        # cx,cy: coordinate of the PSF center in *pad*
        P = cp.fft.rfft2(pad)
        P = P.astype(cp.complex64)
        nimages, pH, pW = pad.shape
        v = cp.fft.rfftfreq(pW)
        w = cp.fft.fftfreq(pH)
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

    def _getOversampledFourierTransformBatchGPU(self, px, py, radius):
        sz = self.getFourierTransformSizeBatchGPU(radius)
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
        P = cp.fft.rfft2(pad)
        P = P.astype(cp.complex64)
        nimages, pH, pW = pad.shape
        v = cp.fft.rfftfreq(pW)
        w = cp.fft.fftfreq(pH)
        rtn = P, (cx, cy), (pH, pW), (v, w)
        self.fftcache[key] = rtn
        return rtn

