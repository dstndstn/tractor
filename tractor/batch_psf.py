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
import time
tbs = np.zeros(9)

if sys.version_info[0] == 2:
    # Py2
    def round(x):
        import __builtin__
        return int(__builtin__.round(float(x)))

def lanczos_shift_image(img, dx, dy, inplace=False, force_python=False):
    H,W = img.shape
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

def plot_comparison(data_gpu, data_cpu, title="Comparison"):
    """
    Plots GPU data, CPU data, and the Difference side-by-side.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Plot GPU Version
    im0 = axes[0].imshow(data_gpu, origin='lower', cmap='viridis')
    axes[0].set_title(f'GPU {title}')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Plot CPU Version
    im1 = axes[1].imshow(data_cpu, origin='lower', cmap='viridis')
    axes[1].set_title(f'CPU {title}')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Plot Difference (Residuals)
    # Using 'seismic' or 'RdBu' helps visualize positive/negative offsets
    diff = data_gpu - data_cpu
    im2 = axes[2].imshow(diff, origin='lower', cmap='seismic')
    axes[2].set_title('Difference (GPU - CPU)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def plot_one(data, title="Plot", cmap='viridis'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))

    # origin='lower' puts (0,0) at bottom-left
    # interpolation='nearest' prevents blurring of pixel boundaries
    im = plt.imshow(data, origin='lower', cmap=cmap, interpolation='nearest')

    plt.title(title)
    plt.xlabel("X [pixels]")
    plt.ylabel("Y [pixels]")

    # Add a colorbar that scales to the image height
    plt.colorbar(im, label='Value')

    plt.tight_layout()
    plt.show()

#def lanczos_shift_image_batch_gpu(imgs, dxs, dys):
#    """Translated from lanczos_shift_image python version to GPU using cupy
#        and helper functions from tractor.miscutils"""
"""
    import cupy as cp
    from tractor.miscutils import gpu_lanczos_filter,batch_correlate1d_gpu
    do_reshape = False
    if len(imgs.shape) == 4:
        do_reshape = True
        oldshape = imgs.shape
        imgs = imgs.reshape((oldshape[0]*oldshape[1], oldshape[2], oldshape[3]))
    L = 3
    nimg = dxs.size 
    print (f'{nimg=}')
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
"""

def lanczos_shift_image_3d_gpu(imgs, dxs, dys):
    """Translated from lanczos_shift_image python version to GPU using cupy
        and helper functions from tractor.miscutils"""
    import cupy as cp
    from tractor.miscutils import gpu_lanczos_filter,batch_correlate1d_gpu_fast, batch_correlate1d_cpu_fast
    from tractor.miscutils import batch_correlate1d_gpu, batch_correlate1d_cpu 
    t = time.time()
    L = 3
    # Ensure dxs/dys are 1D for the tile operation
    dxs = cp.atleast_1d(dxs).ravel()
    dys = cp.atleast_1d(dys).ravel()
    nimg = dxs.size

    # 1. Coordinate Grid
    lr = cp.arange(-L, L+1).astype(cp.float32)
    # Use broadcasting instead of tile for efficiency
    # Note the MINUS sign: often needed if correlating to "pull" the image
    grid_x = lr[cp.newaxis, :] + dxs[:, cp.newaxis]
    grid_y = lr[cp.newaxis, :] + dys[:, cp.newaxis]

    # 2. Generate Filters
    Lx = gpu_lanczos_filter(L, grid_x)
    Ly = gpu_lanczos_filter(L, grid_y)
    tbs[3] += time.time()-t
    t = time.time()

    # 3. Robust Normalization (avoid division by zero/epsilon)
    Lx /= (cp.sum(Lx, axis=1, keepdims=True) + 1e-12)
    Ly /= (cp.sum(Ly, axis=1, keepdims=True) + 1e-12)
    tbs[4] += time.time()-t

    """
    cimgs = imgs.get()
    clx = Lx.get()
    cly = Ly.get()

    t = time.time()
    sx = batch_correlate1d_gpu(imgs, Lx, axis=2, mode='constant')
    outimg1 = batch_correlate1d_gpu(sx, Ly, axis=1, mode='constant')
    tbs[5] += time.time()-t

    t = time.time()
    try:
        sx = batch_correlate1d_cpu(cimgs, clx, axis=2, mode='constant')
        outimg2 = batch_correlate1d_cpu(sx, cly, axis=1, mode='constant')
    except Exception as ex:
        print ("Exception "+str(ex))
    tbs[6] += time.time()-t

    t = time.time()
    sx = batch_correlate1d_cpu_fast(cimgs, clx, axis=2, mode='constant')
    outimg3 = batch_correlate1d_cpu_fast(sx, cly, axis=1, mode='constant')
    tbs[7] += time.time()-t
    """

    # 4. Apply 1D Separable Correlations
    # axis=2 is Width (X), axis=1 is Height (Y)
    t = time.time()
    sx = batch_correlate1d_gpu_fast(imgs, Lx, axis=2, mode='constant')
    outimg = batch_correlate1d_gpu_fast(sx, Ly, axis=1, mode='constant')

    tbs[8] += time.time()-t

    return outimg

def lanczos_shift_image_batch_gpu(imgs, dxs, dys, chunk_size=None):
    import cupy as cp
    from tractor.miscutils import gpu_lanczos_filter, batch_correlate1d_gpu

    # Handle 4D (N, Nd, H, W) -> 3D (Z, H, W)
    do_reshape = False
    if len(imgs.shape) == 4:
        do_reshape = True
        oldshape = imgs.shape
        imgs = imgs.reshape((-1, oldshape[2], oldshape[3]))
        dxs = dxs.ravel()
        dys = dys.ravel()

    nimg = dxs.size

    # Smart chunking: If chunk_size isn't provided,
    # let's assume a safe default like 50 images for A100.
    if chunk_size is None:
        chunk_size = 50

    outimg = cp.empty_like(imgs)
    L = 3

    for i in range(0, nimg, chunk_size):
        print (f'Using chunk {i=} of size {chunk_size=}')
        end = min(i + chunk_size, nimg)

        # Slicing doesn't copy data, it's a view
        img_chunk = imgs[i:end]

        # Generate filters for this chunk
        lr = cp.tile(cp.arange(-L, L+1), (end-i, 1))
        Lx = gpu_lanczos_filter(L, lr + dxs[i:end, cp.newaxis])
        Ly = gpu_lanczos_filter(L, lr + dys[i:end, cp.newaxis])
        del lr

        # Normalize
        Lx /= Lx.sum(1)[:, cp.newaxis]
        Ly /= Ly.sum(1)[:, cp.newaxis]

        # First correlation (Horizontal)
        sx = batch_correlate1d_gpu(img_chunk, Lx, axis=2, mode='constant')
        del Lx

        # Second correlation (Vertical)
        outimg[i:end] = batch_correlate1d_gpu(sx, Ly, axis=1, mode='constant')

        del Ly, sx
        # Forces the mempool to release blocks back to the GPU
        cp.get_default_memory_pool().free_all_blocks()

    if do_reshape:
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

    def __init__(self, psfs, isPointSource=False):
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
            if hasattr(psf, 'pix') and (str(psf.pix) == 'NormalizedPixelizedPsfEx' or str(psf.pix) == 'NormalizedPixelizedPsf'):
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
            #print ("BATCH1", i, psf.img.max(), np.where(psf.img == psf.img.max()))
        img = cp.asarray(img)
        #print ("H,W", H, W)
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

        if isPointSource:
            print ("Point source")
            return
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
        #print (f'{N=} {Nb=} {H=} {W=}, img.dtype')
        #print ("PAD shape", pad.shape, pad.size)
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
        del pad
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


    def getBatchPointSourcePatch(self, pxs, pys, ux0, uy0, u_width, u_height):
            """
            Projects and shifts a point source into a 3D global canvas.
            - pxs, pys: CuPy arrays of shape (N,) for sub-pixel positions.
            - ux0, uy0: The top-left corner of the global canvas in pixel coordinates.
            - u_width, u_height: Dimensions of the global canvas.
            """
            import cupy as cp

            """
            N = self.N
            L = self.Lorder
            padding = L

            # 1. Allocate a padded 3D canvas on the GPU
            ch, cw = u_height + 2 * padding, u_width + 2 * padding
            canvas = cp.zeros((N, ch, cw), dtype=cp.float32)
            canvas_x_start = ux0 - padding
            canvas_y_start = uy0 - padding
            print (f'{pxs=} {pys=} {ch=} {cw=}')

            # 2. Integer shifts for placement and sub-pixel shifts for Lanczos
            ixs = cp.round(pxs).astype(cp.int32)
            iys = cp.round(pys).astype(cp.int32)
            dxs = pxs - ixs
            dys = pys - iys

            # 3. Insert PSF stamps into the canvas at integer positions
            psf_h, psf_w = self.H, self.W
            for i in range(N):
                ix, iy = int(ixs[i]), int(iys[i])
                # Integer origin of PSF stamp in global coordinates
                #x0, y0 = ix - psf_w // 2, iy - psf_h // 2
                #x0, y0 = ix - u_width // 2, iy - u_height // 2
                x0 = ix
                y0 = iy

                # Find the intersection of the PSF stamp and the Padded Canvas
                # Both are in the same global coordinate system here
                inter_x_lo = max(x0, canvas_x_start)
                inter_x_hi = min(x0 + psf_w, canvas_x_start + cw)
                inter_y_lo = max(y0, canvas_y_start)
                inter_y_hi = min(y0 + psf_h, canvas_y_start + ch)

                ## Intersection between PSF stamp and the padded Canvas
                #y_lo, y_hi = max(y0, uy0 - padding), min(y0 + psf_h, uy0 + u_height + padding)
                #x_lo, x_hi = max(x0, ux0 - padding), min(x0 + psf_w, ux0 + u_width + padding)
                #print (f'GPU {i=} {ix=} {iy=} {x0=} {y0=} {psf_w=} {psf_h=} {ux0=} {uy0=} {padding=} {y_lo=} {y_hi=} {x_lo=} {x_hi=}')
                print (f'GPU {i=} {ix=} {iy=} {x0=} {y0=} {psf_w=} {psf_h=} {ux0=} {uy0=} {padding=} {inter_y_lo=} {inter_y_hi=} {inter_x_lo=} {inter_x_hi=}')
                if inter_y_hi > inter_y_lo and inter_x_hi > inter_x_lo:
                    # 1. Local indices within the PSF stamp [0:63]
                    ly0, ly1 = inter_y_lo - y0, inter_y_hi - y0
                    lx0, lx1 = inter_x_lo - x0, inter_x_hi - x0

                    # 2. Local indices within the Padded Canvas
                    cy0, cy1 = inter_y_lo - canvas_y_start, inter_y_hi - canvas_y_start
                    cx0, cx1 = inter_x_lo - canvas_x_start, inter_x_hi - canvas_x_start
                    canvas[i, cy0:cy1, cx0:cx1] = self.img[i, ly0:ly1, lx0:lx1]
                    print(f'GPU {cx0=} {cx1=} {cy0=} {cy1=} {lx0=} {lx1=} {ly0=} {ly1=}')

            """
            """
                if y_hi > y_lo and x_hi > x_lo:
                    # Relative indices for source PSF and target canvas
                    ly0, ly1 = y_lo - y0, y_hi - y0
                    lx0, lx1 = x_lo - x0, x_hi - x0
                    cy0, cy1 = y_lo - (uy0 - padding), y_hi - (uy0 - padding)
                    cx0, cx1 = x_lo - (ux0 - padding), x_hi - (ux0 - padding)

                    canvas[i, cy0:cy1, cx0:cx1] = self.img[i, ly0:ly1, lx0:lx1]
            """
            """

            # 4. Perform the 3D Batch Lanczos shift
            shifted = lanczos_shift_image_batch_gpu(canvas, dxs, dys)

            # 5. Crop back to the unpadded global canvas
            return shifted[:, padding:-padding, padding:-padding]
            """

            #N = self.N
            N = len(pxs)
            print (f'{N=} {self.img.shape=}')
            L = self.Lorder
            padding = L

            # 1. Allocate a padded 3D canvas on the GPU
            ch, cw = u_height + 2 * padding, u_width + 2 * padding
            canvas = cp.zeros((N, ch, cw), dtype=cp.float32)
            psf_h, psf_w = self.H, self.W

            # 2. Establish the coordinate system
            # The CPU evaluates models at (px - x0, py - y0).
            # To match, we must project our PSF into the patch using these coordinates.
            # Calculate x0, y0 per star exactly as the CPU/Logs do
            ixs_glob = cp.round(pxs).astype(cp.int32)
            iys_glob = cp.round(pys).astype(cp.int32)
            dxs = pxs - ixs_glob
            dys = pys - iys_glob
            x0s = ixs_glob - psf_w // 2
            y0s = iys_glob - psf_h // 2


            """
            # These are the coordinates in the "Patch Frame" where the star is ~31.5
            pxs_patch = pxs - x0s
            pys_patch = pys - y0s

            # Integer placement and fractional Lanczos shift
            ixs_patch = cp.round(pxs_patch).astype(cp.int32)
            iys_patch = cp.round(pys_patch).astype(cp.int32)
            dxs = pxs_patch - ixs_patch
            dys = pys_patch - iys_patch
            print (f'{pxs=} {pys=} {ch=} {cw=}')
            """

            # However, our target canvas is still defined by (u_width, u_height).
            # We need to account for the fact that the star at px_patch needs to be 
            # placed in the canvas.
            
            #print ("CANVAS", canvas.shape)
            t = time.time()
            for i in range(N):
                #print ("BATCH IMG", i, self.img[i].max(), self.img[i].shape, cp.where(self.img[i] == self.img[i].max()))
                # 1. Coordinate of the PSF's (0,0) pixel in the x0-relative system
                # For a 63x63 PSF and ix=31, this is 0.
                ix, iy = int(ixs_glob[i]), int(iys_glob[i])
                x0_psf = ix - psf_w // 2
                y0_psf = iy - psf_h // 2

                # 2. Coordinate of our Canvas (0,0) pixel in the x0-relative system
                # ux0 is the modelmask offset. padding is the Lanczos margin.
                x0_canvas = ux0 - padding
                y0_canvas = uy0 - padding

                # 3. Calculate the overlap using the get_overlapping_region logic
                # We want to place the PSF (img) into the Canvas (canvas)
                # The CPU logic: yi, yo = get_overlapping_region(target_start, target_end, img_min, img_max)

                # In CPU terms: xlo = x0_canvas - x0_psf
                # This tells us where the canvas starts relative to the PSF's 0,0
                rel_x = x0_canvas - x0_psf
                rel_y = y0_canvas - y0_psf

                # Calculate slices for the PSF (Source) and Canvas (Destination)
                # We use the clamped intersection logic
                src_x0 = max(0, rel_x)
                src_y0 = max(0, rel_y)

                dst_x0 = max(0, -rel_x)
                dst_y0 = max(0, -rel_y)

                # Determine width/height of the overlap
                overlap_w = min(psf_w, rel_x + cw) - src_x0
                overlap_h = min(psf_h, rel_y + ch) - src_y0
                #print (f'GPU {i=} {ix=} {iy=} {x0_psf=} {y0_psf=} {x0_canvas=} {y0_canvas=} {psf_w=} {psf_h=} {ux0=} {uy0=} {padding=} {rel_x=} {rel_y=} {src_x0=} {src_y0=} {dst_x0=} {dst_y0=}') 

                if overlap_w > 0 and overlap_h > 0:
                    lx0, lx1 = src_x0, src_x0 + overlap_w
                    ly0, ly1 = src_y0, src_y0 + overlap_h

                    cx0, cx1 = dst_x0, dst_x0 + overlap_w
                    cy0, cy1 = dst_y0, dst_y0 + overlap_h

                    canvas[i, cy0:cy1, cx0:cx1] = self.img[i, ly0:ly1, lx0:lx1]
                    #print(f'GPU {cx0=} {cx1=} {cy0=} {cy1=} {lx0=} {lx1=} {ly0=} {ly1=}')
                    #bx = cp.where(self.img[i] == self.img[i].max())
                    #print (f'{bx=} {self.img[i].shape=}')
                #if i < 2:
                #    plot_one(canvas[i].get(), "Canvas "+str(i))
                #    plot_one(self.img[i].get(), "Img "+str(i))

            #print (f'{dxs=} {dys=}')
            tbs[0] += time.time()-t
            t = time.time()
            # 4. Perform the 3D Batch Lanczos shift
            shifted = lanczos_shift_image_3d_gpu(canvas, dxs, dys)
            tbs[1] += time.time()-t
            tbs[2] += 1
            #shifted = lanczos_shift_image_batch_gpu(canvas, dxs, dys)
            #cdx = dxs.get()
            #cdy = dys.get()
            #for j in range(2):
                #sx = lanczos_shift_image(canvas[j].get(), cdx[j], cdy[j])
                #plot_one(sx[padding:-padding, padding:-padding], "C SHIFT "+str(j))
                #plot_one(shifted[j, padding:-padding, padding:-padding].get(), "GPU "+str(j))
                #plot_comparison(shifted[j].get(), sx, "Lanczos comp")

            # 5. Crop back to the unpadded global canvas
            print (f'{tbs=}')
            from tractor.psf import print_tcps
            print_tcps()
            return shifted[:, padding:-padding, padding:-padding]

    """
    # In batch_psf.py (class BatchPixelizedPSF)
    def getBatchPointSourcePatch(self, pxs, pys, counts, modelMasks=None):
        '''
        pxs, pys, counts: 1D arrays of length N_images
        modelMasks: list of N ModelMask objects (or None)
        '''
        import cupy as cp
        from tractor.psf import lanczos_shift_image_batch_gpu
        from astrometry.util.miscutils import get_overlapping_region

        N = self.N # Number of images
        H, W = self.H, self.W # Max PSF dimensions
        
        # Calculate fractional shifts
        ixs = cp.round(pxs).astype(cp.int32)
        iys = cp.round(pys).astype(cp.int32)
        dxs = pxs - ixs
        dys = pys - iys
        
        L = 3 # Lanczos order
        padding = L

        if modelMasks is not None:
            # Assume point source masks are the same size for batching (typical)
            mh, mw = modelMasks[0].shape
            # Create a 3D canvas (N, H_mask + 2L, W_mask + 2L)
            mm_stack = cp.zeros((N, mh + 2*padding, mw + 2*padding), dtype=cp.float32)
            
            for i in range(N):
                m = modelMasks[i]
                # PSF origin in global image
                x0 = ixs[i] - W // 2
                y0 = iys[i] - H // 2
                
                # Use overlap logic to copy the unshifted PSF into the mask canvas
                yi, yo = get_overlapping_region(m.y0 - y0 - padding, m.y0 - y0 + mh - 1 + padding, 0, H - 1)
                xi, xo = get_overlapping_region(m.x0 - x0 - padding, m.x0 - x0 + mw - 1 + padding, 0, W - 1)
                
                if len(yi) > 0 and len(xi) > 0:
                    mm_stack[i, yo, xo] = self.img[i, yi, xi]
            
            # Perform 3D Lanczos shift on the entire stack in one go
            shifted = lanczos_shift_image_batch_gpu(mm_stack, dxs, dys)
            
            # Crop back and apply flux counts
            shifted = shifted[:, padding:-padding, padding:-padding]
            shifted *= counts[:, cp.newaxis, cp.newaxis]
            
            return [Patch(modelMasks[i].x0, modelMasks[i].y0, shifted[i]) for i in range(N)]
        else:
            # No masks: Shift the full 3D padded PSF stack (self.img)
            shifted = lanczos_shift_image_batch_gpu(self.img, dxs, dys)
            shifted *= counts[:, cp.newaxis, cp.newaxis]
            
            # Origins
            x0s = ixs - W // 2
            y0s = iys - H // 2
            return [Patch(x0s[i], y0s[i], shifted[i]) for i in range(N)]
    """

    """
    def getBatchPointSourcePatch(self, pxs, pys, counts, modelMask=None, radius=None):
        import cupy as cp
        from astrometry.util.miscutils import get_overlapping_region

        n_cand = len(pxs)
        L = 3 # Lanczos radius
        padding = L

        # 1. Determine integer and fractional shifts
        ixs = cp.round(cp.asarray(pxs)).astype(cp.int32)
        iys = cp.round(cp.asarray(pys)).astype(cp.int32)
        dxs = cp.asarray(pxs) - ixs
        dys = cp.asarray(pys) - iys

        # 2. Get PSF template
        img = self.getImage(pxs[0], pys[0]) # (H_psf, W_psf)
        ph, pw = img.shape

        if modelMask is None:
            # Default behavior: patch is just the size of the PSF
            # (Similar to your existing logic)
            psf_stack = cp.tile(cp.asarray(img), (n_cand, 1, 1))
            shifted = lanczos_shift_image_batch_gpu(psf_stack, dxs, dys)
            counts_gpu = cp.asarray(counts)[:, cp.newaxis, cp.newaxis]
            return shifted * counts_gpu, ixs - pw//2, iys - ph//2

        # 3. ModelMask Logic
        mh, mw = modelMask.shape
        mx0, my0 = modelMask.x0, modelMask.y0

        # Create the 3D "canvas" with padding for Lanczos
        # Shape: (N_candidates, mh + 2*L, mw + 2*L)
        canvas = cp.zeros((n_cand, mh + 2*padding, mw + 2*padding), dtype=cp.float32)

        for i in range(n_cand):
            # Calculate where the PSF template sits relative to the ModelMask
            # x0, y0 is the bottom-left of the PSF template in image coords
            x0 = ixs[i] - pw // 2
            y0 = iys[i] - ph // 2

            # Determine overlap between PSF template and (ModelMask + Padding)
            # This mirrors the get_overlapping_region logic in psf.py
            yi, yo = get_overlapping_region(my0 - y0 - padding,
                                            my0 - y0 + mh - 1 + padding,
                                            0, ph - 1)
            xi, xo = get_overlapping_region(mx0 - x0 - padding,
                                            mx0 - x0 + mw - 1 + padding,
                                            0, pw - 1)

            if len(yi) > 0 and len(xi) > 0:
                canvas[i, yo, xo] = cp.asarray(img[yi, xi])

        # 4. Batch Shift the entire canvas
        # The padding ensures the shift doesn't "lose" pixels at the mask edges
        shifted_canvas = lanczos_shift_image_batch_gpu(canvas, dxs, dys)

        # 5. Crop back to the ModelMask size and scale by flux
        final_patches = shifted_canvas[:, padding:-padding, padding:-padding]
        counts_gpu = cp.asarray(counts)[:, cp.newaxis, cp.newaxis]

        return final_patches * counts_gpu, mx0, my0
    """


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
