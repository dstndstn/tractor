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
    del lr
    Lx /= Lx.sum(1).reshape((nimg,1))
    Ly /= Ly.sum(1).reshape((nimg,1))
    sx = batch_correlate1d_gpu(imgs, Lx, axis=2, mode='constant')
    del Lx
    outimg = batch_correlate1d_gpu(sx, Ly, axis=1, mode='constant')
    del Ly
    del sx
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
        self.ex_indices = np.zeros(N, dtype=bool)
        self.psfexs = []
        self.normalized = False
        for i,psf in enumerate(psfs):
            if str(psf.pix) == 'NormalizedPixelizedPsfEx' or str(psf.pix) == 'NormalizedPixelizedPsf':
                self.normalized = True
            if hasattr(psf, 'psfex'):
                self.psfexs.append(psf.psfex)
                self.ex_indices[i] = True
        #iH = np.zeros(N, dtype=np.int32) #individual height
        #iW = np.zeros(N, dtype=np.int32) #individual width

        #Find max w, h
        iH,iW = np.array([psf.img.shape for psf in psfs]).T
        #for i, psf in enumerate(psfs):
        #    iH[i], iW[i] = psf.img.shape
        #H = np.max(iH)
        #W = np.max(iW)
        H = iH.max()
        W = iW.max()
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

    def _padInImageBatchGPUBases(self, H, W, img=None):
        '''
        Embeds this PSF image into a larger or smaller image of shape H,W.

        Return (img, cx, cy), where *cx*,*cy* are the coordinates of the PSF
        center in *img*.
        '''
        if img is None:
            img = self.img
        N, Nb, ph, pw = img.shape
        subimg = img

        if H >= ph:
            y0 = (H - ph) // 2
            cy = y0 + ph // 2
        else:
            y0 = 0
            cut = (ph - H) // 2
            subimg = subimg[:, :, cut:cut + H, :]
            cy = ph // 2 - cut

        if W >= pw:
            x0 = (W - pw) // 2
            cx = x0 + pw // 2
        else:
            x0 = 0
            cut = (pw - W) // 2
            subimg = subimg[:, :, :, cut:cut + W]
            cx = pw // 2 - cut
        N, Nb, sh, sw = subimg.shape

        pad = np.zeros((N, Nb, H, W), img.dtype)
        pad[:, :, y0:y0 + sh, x0:x0 + sw] = subimg
        pad = cp.asarray(pad)
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
        if len(self.psfexs) == self.N:
            #All are psfEx
            if self.normalized:
                return self.getFourierTransformExNormalizedBatchGPU(px, py, radius)
            return self.getFourierTransformExBatchGPU(px, py, radius)
        import cupy as cp
        px = cp.asarray(px)
        py = cp.asarray(py)
        radius = cp.asarray(radius)
        if self.sampling != 1.:
            return self._getOversampledFourierTransformBatchGPU(px, py, radius)
    
        sz = self.getFourierTransformSizeBatchGPU(radius)
        if sz in self.fftcache:
            #print ("CACHE1")
            return self.fftcache[sz]
        
        pad, cx, cy = self._padInImageBatchGPU(sz, sz)
        #print ("PAD", pad.shape)
        # cx,cy: coordinate of the PSF center in *pad*
        P = cp.fft.rfft2(pad)
        P = P.astype(cp.complex64)
        nimages, pH, pW = pad.shape
        v = cp.fft.rfftfreq(pW)
        w = cp.fft.fftfreq(pH)
        if (len(self.psfexs) > 0):
            #Only some are psfEx - others have been calculated above
            if self.normalized:
                sumfft, (cx, cy), shape, (v, w) = self.getFourierTransformExNormalizedBatchGPU(px, py, radius)
            else:
                sumfft, (cx, cy), shape, (v, w) = self.getFourierTransformExBatchGPU(px, py, radius)
            P[self.ex_indices,:,:] = sumfft
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

    def getFourierTransformExNormalizedBatchGPU(self, px, py, radius):
        fft, (cx,cy), shape, (v,w) = self.getFourierTransformExBatchGPU(px, py, radius)
        n = cp.abs(fft[:,0,0])
        mask = n != 0
        fft[mask,:,:] /= n[mask,None,None]
        return fft, (cx,cy), shape, (v,w)

    def getFourierTransformExBatchGPU(self, px, py, radius):
        import cupy as cp
        px = cp.asarray(px[self.ex_indices])
        py = cp.asarray(py[self.ex_indices])
        radius = cp.asarray(radius[self.ex_indices])
        
        if self.sampling != 1.:
            # The method below assumes that the eigenPSF bases can be
            # Fourier-transformed and combined; that doesn't work for
            # oversampled models.  Fall back (which does mean that
            # we're evaluating the PSF model image every time and FFT'ing).
            return self._getOversampledFourierTransformBatchGPU(px, py, radius)

        sz = self.getFourierTransformSizeBatchGPU(radius)

        N = len(self.psfexs)
        Nb = 0
        h = 0
        w = 0
        for pex in self.psfexs:
            bshape = pex.bases().shape
            Nb = max(Nb, bshape[0])
            h = max(h, bshape[1])
            w = max(w, bshape[2])
        bases = np.zeros((N, Nb, h, w), dtype=np.float32)
        for i, pex in enumerate(self.psfexs):
            (iNb, ih, iw) = pex.bases().shape
            bases[i,:iNb, :ih, :iw] = pex.bases()

        if sz in self.fftcache:
            fftbases, cx, cy, shape, v, w = self.fftcache[sz]
            #print ("CACHE2")
        else:
            N, Nb, pH, pW = bases.shape
            #bases = 4d image N x NB x H x W
            pad, cx, cy = self._padInImageBatchGPUBases(sz, sz, img=bases)
            # cx,cy: coordinate of the PSF center in *pad*
            P = cp.fft.rfft2(pad)
            P = P.astype(cp.complex64)
            nimages, Nb, pH, pW = pad.shape
            v = cp.fft.rfftfreq(pW)
            w = cp.fft.fftfreq(pH)
            fftbases = P
            self.fftcache[sz] = (fftbases, cx, cy, pad.shape, v, w)
        # Now sum the bases by the polynomial coefficients
        sumfft = cp.zeros(fftbases[:,0,:,:].shape, fftbases.dtype)
        shape = pad[0,0].shape
        for i, psfex in enumerate(self.psfexs):
            for amp, base in zip(psfex.polynomials(px[i], py[i]), fftbases[i]):
                #print('getFourierTransform: px,py', (px,py), 'amp', amp)
                sumfft[i] += amp * base
                #print('sum of inverse Fourier transform of component:', cp.sum(cp.fft.irfft2(amp * base, s=shape)))
            #print('sum of inverse Fourier transform of PSF:', np.sum(np.fft.irfft2(sumfft, s=shape)))
        #cp.savetxt("gsumfft.txt", sumfft.ravel())
        return sumfft, (cx, cy), shape, (v, w)

    def getPointSourcePatch(self, px, py, minval=0., modelMask=None,
                            radius=None, **kwargs):
        #print ("getPointSourcePatch1", self)
        if self.sampling != 1.:
            return self._getOversampledPointSourcePatch(px, py, minval=minval,
                                                        modelMask=modelMask,
                                                        radius=radius, **kwargs)
            
        import cupy as cp
        px = cp.asarray(px)
        py = cp.asarray(py)
        radius = cp.asarray(radius)

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
