import time

import numpy as np
#from cupy_wrapper import cp
#import cupy as cp

from tractor.smarter_dense_optimizer import SmarterDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, ConstantSky, PointSource
from tractor.brightness import LinearPhotoCal

'''
TO-DO:
-- check in oneblob.py - are we passing NormalizedPixelizedPsfEx (VARYING) PSF objects
   in at some point?  They should all get turned into constant PSFs.
-- check PSF sampling != 1.0

'''

class Duck(object):
    pass

class GpuFriendlyOptimizer(SmarterDenseOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_super = 0.
        self.n_super = 0

    def getLinearUpdateDirection(self, tr, priors=True, get_icov=False, **kwargs):
        if not (tr.isParamFrozen('images') and
            (len(tr.catalog) == 1) and
            isinstance(tr.catalog[0], (ProfileGalaxy, PointSource))):
            return super().getLinearUpdateDirection(tr, priors=priors, get_icov=get_icov,
                                                    **kwargs)
        assert(get_icov == False)
        return self.one_source_update(tr, priors=priors, **kwargs)

    # def per_image_updates(self, tr, **kwargs):
    #     if not (tr.isParamFrozen('images') and
    #         (len(tr.catalog) == 1) and
    #         isinstance(tr.catalog[0], ProfileGalaxy)):
    #         t0 = time.time()
    #         R = super().per_image_updates(tr, **kwargs)
    #         dt = time.time() - t0
    #         self.t_super += dt
    #         self.n_super += 1
    #         return R
    #     return self.one_galaxy_update(tr, **kwargs)

    def one_source_update(self, tr, **kwargs):
        #return super().all_image_updates(tr, **kwargs)
        return super().getLinearUpdateDirection(tr, **kwargs)

class GpuOptimizer(GpuFriendlyOptimizer):

    def __init__(self, cp, *args, **kwargs):
        self.cp = cp
        super().__init__(*args, **kwargs)

    def one_source_update(self, tr, **kwargs):
        return self.gpu_one_source_update(tr, **kwargs)

        # free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        # # predict memory required
        # totalpix = sum([np.prod(tim.shape) for tim in tr.images])
        # nd = tr.numberOfParams()+2
        # kmax = 9
        # # Double size of 16 bit (complex 128) array x npix x
        # # n derivs x kmax.  5D array in batch_mixture_profiles.py
        # est_mem = totalpix * nd * kmax * 16
        # 
        # if free_mem < est_mem:
        #     print (f"Warning: Estimated memory {est_mem} is greater than free memory {free_mem}; Running CPU mode instead!")
        #     R_gpuv = super().one_galaxy_update(tr, **kwargs)
        #     return R_gpuv
        # 
        # try:
        #     t0 = time.time()
        #     R_gpuv = self.gpu_one_source_update(tr, **kwargs)
        #     #mempool = cp.get_default_memory_pool()
        #     #mempool.free_all_blocks()
        #     dt = time.time() - t0
        #     return R_gpuv
        # except AssertionError:
        #     import traceback
        #     print ("AssertionError in VECTORIZED GPU code:")
        #     traceback.print_exc()
        # except cp.cuda.memory.OutOfMemoryError:
        #     free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        #     mempool = cp.get_default_memory_pool()
        #     used_bytes = mempool.used_bytes()
        #     tot_bytes = mempool.total_bytes()
        #     print ('Out of Memory for source: ', tr.catalog[0])
        #     print (f'OOM Device {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=}')
        #     mempool.free_all_blocks()
        # except Exception as ex:
        #     import traceback
        #     print('Exception in GPU Vectorized code:')
        #     traceback.print_exc()
        #     raise(ex)
        # 
        # # Fallback to CPU version
        # print('Falling back to CPU code...')
        # t0 = time.time()
        # R = super().one_source_update(tr, **kwargs)
        # dt = time.time() - t0
        # return R

    def gpu_one_source_update(self, tr, priors=True, get_A=False, **kwargs):
        cp = self.cp

        t0 = time.time()
        # Assume single source
        assert(len(tr.catalog) == 1)
        Nimages = len(tr.images)

        # Assume galaxy or point source
        src = tr.catalog[0]
        is_galaxy = isinstance(src, ProfileGalaxy)
        is_psf = isinstance(src, PointSource)
        assert(isinstance(src, (ProfileGalaxy, PointSource)))

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        #masks = [tr._getModelMaskByIdx(i, src) for i in range(len(tr.images))]
        masks = [tr._getModelMaskFor(tim, src) for tim in tr.images]
        if any(m is None for m in masks):
            raise RuntimeError('One or more modelMasks is None in GPU code')
        assert(all([m is not None for m in masks]))
        extents = [mm.extent for mm in masks]

        # Pixel positions
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tr.images]
        px, py = np.array(pxy, dtype=np.float32).T
        # Round source pixel position to nearest integer
        ipx = px.round().astype(np.int32)
        ipy = py.round().astype(np.int32)

        # WCS inv(CD) matrix
        img_cdi = [tim.getWcs().cdInverseAtPixel(x,y) for tim,x,y in zip(tr.images, px, py)]
        # Current counts
        img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                      for tim in tr.images]
        src_bands = src.getBrightness().getParamNames()

        ## FIXME -- this should be based on *SOURCE* properties as well, no?
        ## FIXME -- used to have PSF size included in this mix...
        x0, x1, y0, y1 = np.asarray(extents).T
        halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py]))
        #psfH//2, psfW//2]))

        # Pre-process image: PSF, padded pix, etc
        img_params = self.gpu_setup_image_params(tr.images, halfsize, extents, ipx, ipy)

        if is_galaxy:
            # Get galaxy profiles for shape derivatives.
            amixes = [[('current', src._getShearedProfile(tim,x,y), 0.)] +
                    src.getDerivativeShearedProfiles(tim,x,y)
                    for tim,x,y in zip(tr.images, px, py)]
            Nprofiles = max([len(d) for d in amixes])
            mogs = [[m for _,m,_ in amix_img] for amix_img in amixes]

        # subpixel portion: shifted via Lanczos interpolation
        mux = px - ipx
        muy = py - ipy
        assert(np.abs(mux).max() <= 0.5)
        assert(np.abs(muy).max() <= 0.5)

        if is_galaxy:
            # Render galaxy profiles for shape derivatives
            G = self.gpu_get_unitflux_galaxy_profiles(mogs, img_params, mux, muy)
        else:
            G = self.gpu_get_unitflux_psf(img_params, mux, muy)
        # import pylab as plt
        # cG = G.get()
        # plt.clf()
        # k = 1
        # for i in range(Nimages):
        #     for j in range(Nprofiles):
        #         plt.subplot(Nimages, Nprofiles, k)
        #         k += 1
        #         if j == 0:
        #             plt.imshow(cG[i, j, :, :], interpolation='nearest', origin='lower')
        #         else:
        #             plt.imshow(cG[i, j, :, :] - cG[i, 0, :, :], interpolation='nearest', origin='lower')
        # plt.savefig('g.png')

        # Now we have computed the galaxy profiles for the finite-differences steps.
        # Turn these into derivatives and solve the update matrix.
        # - compute per-image updates and commbine? (ie, factored)
        # - build one big matrix?

        pH, pW = img_params.pH, img_params.pW
        assert(img_params.padpix.shape == (Nimages, pH, pW))
        assert(img_params.padie.shape  == (Nimages, pH, pW))

        # We have either RaDecPos or GaiaPositions for the pos -- 2 or 0 params.
        Npos = 0
        if src.isParamThawed('pos'):
            Npos = src.pos.numberOfParams()
        #print('Source position parameters:', Npos)
        assert(Npos in [0, 2])

        # List of the band for each tim
        tim_bands = [tim.getPhotoCal().band for tim in tr.images]
        # The unique set of bands in all tims.  This is the number of flux parameters we
        # will fit for in the A matrix, in this order.
        tim_ubands = list(np.unique(tim_bands))
        # List of the index into tim_ubands for each tim
        # This is what we'll use to build the A matrix
        tim_band_index = [tim_ubands.index(b) for b in tim_bands]
        #print('source has bands:', src_bands)
        #print('tims have bands:', tim_bands)
        #print('Unique set of tim bands:', tim_ubands)
        Nbands = len(tim_ubands)

        if is_galaxy:
            # Nprofiles - 1: profile 0 is the current galaxy shape, the rest are shape/sersic derivs.
            Nshapes = (Nprofiles - 1)
        else:
            Nshapes = 0
        Nderivs = Nshapes + Npos + Nbands

        # We'll need both directions of the mapping between
        # source parameter index and index in our A matrix
        # due to "tims" having only a subset of the bands in the source's params.
        # List of (source parameter index, A matrix index)
        param_indices = []
        if Npos:
            param_indices.append((0,0))
            param_indices.append((1,1))
        for i,band in enumerate(src_bands):
            if band in tim_ubands:
                param_indices.append((Npos + i, Npos + tim_ubands.index(band)))
        for i in range(Nshapes):
            param_indices.append((Npos + len(src_bands) + i, Npos + Nbands + i))
        #print('Parameter indices:', param_indices)
        src_to_fit_param = dict(param_indices)
        fit_to_src_param = dict([(f,s) for s,f in param_indices])

        # We can produce the residual map from the nominal galaxy profile model
        # (scaled by flux) and the image pix.
        assert(all([isinstance(tim.getPhotoCal(), LinearPhotoCal) for tim in tr.images]))
        fluxes = cp.array(img_counts)
        padie = img_params.padie
        padpix = img_params.padpix
        nu = cp.newaxis

        # import pylab as plt
        # chi_pix = (padpix - fluxes[:, nu, nu] * G[:, 0, :, :]) * padie
        # cChi = chi_pix.get()
        # cG = G.get()
        # cpix = padpix.get()
        # cflux = fluxes.get()
        # print('Fluxes:', cflux)
        # plt.clf()
        # k = 1
        # for i in range(Nimages):
        #     plt.subplot(Nimages, 3, 3*i+1)
        #     mod = cflux[i, nu, nu] * cG[i, 0, :, :]
        #     plt.imshow(mod, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.subplot(Nimages, 3, 3*i+2)
        #     img = cpix[i,:,:]
        #     plt.imshow(img, interpolation='nearest', origin='lower')
        #     plt.colorbar()
        #     plt.subplot(Nimages, 3, 3*i+3)
        #     plt.imshow(cChi[i, :, :], interpolation='nearest', origin='lower')
        # plt.savefig('gchi.png')

        Npix_total = Nimages * pH * pW

        Npriors = 0
        if priors:
            priorVals = tr.getLogPriorDerivatives()
            if priorVals is not None:
                rA, cA, vA, pb, mub = priorVals
                Npriors = max(Npriors, max([1+max(r) for r in rA]))

        Nrows = Npix_total + Npriors
        Ncols = Nderivs
        A = cp.zeros((Nrows, Ncols), cp.float32)
        B = cp.zeros(Nrows, cp.float32)

        if Npos:
            # Spatial derivatives:
            dx = cp.empty((Nimages, pH, pW), np.float32)
            dy = cp.empty((Nimages, pH, pW), np.float32)
            # zero out the edge pixels
            dx[:,:, 0] = 0
            dx[:,:,-1] = 0
            dx[:, 0,:] = 0
            dx[:,-1,:] = 0
            dy[:,:, 0] = 0
            dy[:,:,-1] = 0
            dy[:, 0,:] = 0
            dy[:,-1,:] = 0
            # We leave a one-pixel margin in the spatial derivatives
            # in both axes, because we're going to turn these dx,dy
            # derivatives into RA,Dec derivatives, and having one of
            # the terms artificially zero is confusing.
            dx[:, 1:-1, 1:-1] = (G[:, 0, 1:-1, 2:  ] - G[:, 0, 1:-1,  :-2]) / 2.
            dy[:, 1:-1, 1:-1] = (G[:, 0, 2:  , 1:-1] - G[:, 0,  :-2, 1:-1]) / 2.

            cdi = cp.array(img_cdi)
            assert(cdi.shape == (Nimages,2,2))

            A[:Npix_total, 0] = -(fluxes[:, nu, nu] *
                                  (dx * cdi[:, 0, 0][:, nu, nu] +
                                   dy * cdi[:, 1, 0][:, nu, nu]) * padie).flat
            A[:Npix_total, 1] = -(fluxes[:, nu, nu] *
                                  (dx * cdi[:, 0, 1][:, nu, nu] +
                                   dy * cdi[:, 1, 1][:, nu, nu]) * padie).flat

        # Flux derivatives.
        for i in range(Nimages):
            # Image i fills in the column corresponding to its flux
            # and a block of rows corresponding to its pixels.
            col = tim_band_index[i] + Npos
            A[i * pH*pW: (i+1) * pH*pW, col] = (G[i, 0, :, :] * padie[i, :, :]).flat
        # We *could* form the *col* into an array and do this as a
        # one-liner; not sure that would be faster.

        if is_galaxy:
            # Shape derivatives.
            # This gets the axis ordering wrong in the reshape -- need a swapaxes or something...
            #A[:Npix_total, Npos + Nbands:] = ((G[:, 1:, :, :] - G[:, 0, :, :][:, nu, :, :]) *
            #   padie[:, nu, :, :]).reshape((-1, Nshapes))
            stepsizes = np.empty((Nimages, Nshapes), np.float32)
            for i_img, amix_img in enumerate(amixes):
                for i_shape,(_,_,step) in enumerate(amix_img[1:]):
                    stepsizes[i_img, i_shape] = step
            steps = cp.array(stepsizes)
            del stepsizes
            for i in range(Nshapes):
                A[:Npix_total, Npos + Nbands + i] = (fluxes[:, nu, nu] *
                                                     (G[:, i+1, :, :] - G[:, 0, :, :]) /
                                                     steps[:, i, nu, nu] *
                                                     padie).flat

        B[:Npix_total] = ((padpix - fluxes[:, nu, nu] * G[:, 0, :, :]) * padie).flat

        if Npriors > 0:
            rA, cA, vA, pb, mub = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                for rij,vij,bij in zip(ri, vi, bi):
                    # Map the source's parameter index to our local fitting parameter index
                    ci = src_to_fit_param[ci]
                    A[Npix_total + rij, ci] = vij
                    B[Npix_total + rij] += bij
            del priorVals, rA, cA, vA, pb, mub

        # plt.clf()
        # plt.imshow(A.get() != 0, interpolation='nearest', origin='lower',
        #            vmin=0, vmax=1, cmap='gray', aspect='auto')
        # plt.savefig('a.png')
        # 
        # plt.clf()
        # cB = B.get()
        # mx = np.abs(cB).max()
        # plt.imshow(cB[:,nu], interpolation='nearest', origin='lower', aspect='auto',
        #            cmap='RdBu', vmin=-mx, vmax=mx)
        # plt.savefig('b.png')

        # Precondition A: column scales
        colscales = cp.sqrt((A**2).sum(axis=0))
        A /= colscales[nu, :]
        #print('Colscales:', colscales.get())

        # plt.clf()
        # cA = A.get()
        # mx = np.abs(cA).max()
        # plt.imshow(cA, interpolation='nearest', origin='lower', aspect='auto',
        #            cmap='RdBu', vmin=-mx, vmax=mx)
        # plt.savefig('a2.png')

        X,_,_,_ = cp.linalg.lstsq(A, B)
        #print('(scaled) X:', X.get())
        X /= colscales
        #print('X:', X.get())

        # Parameter indices in src of our vector X:
        X = X.get()
        I = np.array([fit_to_src_param[i] for i in range(len(X))])
        #print('Source parameter indices:', I)
        sX = np.zeros(src.numberOfParams(), np.float32)
        sX[I] = X

        if get_A:
            sA = np.zeros((Nrows, src.numberOfParams()), np.float32)
            sA[:,I] = A.get()
            s_scales = np.zeros(src.numberOfParams(), np.float32)
            s_scales[I] = colscales.get()
            return sA, B.get(), sX, s_scales, pH,pW

        return sX

    def gpu_setup_image_params(self, tims, halfsize, extents, ipx, ipy):
        cp = self.cp
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tims]))
        # Assume sky levels are zero.
        assert(all([tim.getSky().getValue() == 0 for tim in tims]))

        psfs = [tim.getPsf() for tim in tims]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        # Assume ConstantSky models, grab constant sky levels
        # NOTE - instead of building this list and passing it around in ImageDerivs, etc,
        # we could perhaps just subtract it off img_pix at the start...
        #img_sky = [tim.getSky().getConstant() for tim in tr.images]
        #print('sky', img_sky)
        #assert(np.all([s == 0 for s in img_sky]))

        img_params = Duck()
        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        P, (cx, cy), (pH, pW), (v, w), psf_mogs, psf_pad = self.get_vectorized_psfs(psfs, halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        Nimages = len(tims)
        assert(P.shape == (Nimages,len(w),len(v)))

        # Embed pix and ie in images the same size as pW,pH.
        # FIXME -- should be able to cache this; rationalize pixel transfer to GPU.
        padpix = cp.zeros((Nimages, pH,pW), cp.float32)
        padie  = cp.zeros((Nimages, pH,pW), cp.float32)
        for i,(tim,(x0,x1,y0,y1)) in enumerate(zip(tims, extents)):
            # FIXME -- put the whole images on the GPU??
            #pix = tim.getGpuImage()
            #ie  = tim.getGpuInvError()
            pix = tim.getImage()
            ie = tim.getInvError()

            dx = cx - ipx[i]
            dy = cy - ipy[i]
            p = pix[y0:y1, x0:x1]
            padpix[i, y0+dy : y1+dy, x0+dx : x1+dx] = p #cp.array(p)
            p =  ie[y0:y1, x0:x1]
            padie [i, y0+dy : y1+dy, x0+dx : x1+dx] = p # cp.array(p)

        # GPU arrays:
        img_params.psf_pad = psf_pad
        img_params.P = P
        img_params.v = v
        img_params.w = w
        img_params.padpix = padpix
        img_params.padie  = padie
        # numpy arrays:
        img_params.psf_mogs = psf_mogs
        # scalars:
        img_params.pH = pH
        img_params.pW = pW
        img_params.cx = cx
        img_params.cy = cy
        return img_params

    def gpu_get_unitflux_psf(self, img_params, mux, muy):
        cp = self.cp
        ## FIXME -- could create a new lanczos method rather than first copying
        # and then running the in-place.
        padpsf = img_params.psf_pad
        nimg,h,w = padpsf.shape
        G = cp.empty((nimg, 1, h, w), cp.float32)
        G[:, 0, :, :] = padpsf
        self.lanczos_shift_images_inplace_gpu(G, mux, muy)
        return G

    def gpu_get_unitflux_galaxy_profiles(self, mogs, img_params, mux, muy):
        cp = self.cp
        # mogs[image][profile] = mog
        Nimages = len(mogs)
        Nprofiles = max([len(d) for d in mogs])
        Kmax = 0
        for img_mogs in mogs:
            for mog in img_mogs:
                Kmax = max(Kmax, len(mog.amp))

        mix_vars = np.zeros((Nimages, Nprofiles, Kmax, 3), np.float32)
        mix_amps = np.zeros((Nimages, Nprofiles, Kmax), np.float32)
        for i,img_mogs in enumerate(mogs):
            for j,mog in enumerate(img_mogs):
                K = len(mog.amp)
                # called "a,b,d" elsewhere in the code
                mix_vars[i, j, :K, 0] = mog.var[:, 0, 0]
                mix_vars[i, j, :K, 1] = mog.var[:, 0, 1]
                mix_vars[i, j, :K, 2] = mog.var[:, 1, 1]
                mix_amps[i, j, :K] = mog.amp

        nsigma1 = 3.
        nsigma2 = 4.

        # Nimages x Nprofiles x Nmog
        sz = img_params.pW/2
        vv = mix_vars[:,:,:,0] + mix_vars[:,:,:,2]
        IM = (sz**2 < (nsigma2**2 * vv))
        IF = (sz**2 > (nsigma1**2 * vv))
        ramp = np.any((IM*IF))

        mogweights = np.ones(vv.shape, dtype=cp.float32)
        fftweights = np.ones(vv.shape, dtype=cp.float32)
        if ramp:
            # ramp
            ns = sz / np.maximum(1e-6, np.sqrt(vv))
            mogweights = np.minimum(1., (nsigma2 - ns) / (nsigma2 - nsigma1))*IM
            fftweights = np.minimum(1., (ns - nsigma1) / (nsigma2 - nsigma1))*IF
            assert(np.all(mogweights[IM] > 0.))
            assert(np.all(mogweights[IM] <= 1.))
            assert(np.all(fftweights[IF] > 0.))
            assert(np.all(fftweights[IF] <= 1.))

        # Assume the MoG components are sort by increasing variance; terms we'll evaluate
        # with the FFT will be at the front, and MoG at the back.
        Nmog = IM.sum(axis=2).max(axis=(0,1))
        Nfft = IF.sum(axis=2).max(axis=(0,1))

        mog_mix_vars = mix_vars[:, :, -Nmog:, :]
        mog_mix_amps = (mix_amps * mogweights)[:, :, -Nmog:]

        fft_mix_vars = mix_vars[:, :, :Nfft:, :]
        fft_mix_amps = (mix_amps * fftweights)[:, :, :Nfft]

        #g_fft_mix_vars = cp.array(fft_mix_vars)
        #a = g_fft_mix_vars[:, :, :, 0]
        #b = 2. * g_fft_mix_vars[:, :, :, 1]
        #d = g_fft_mix_vars[:, :, :, 2]

        g_fft_mix_amps = cp.array(fft_mix_amps)
        a = cp.array(fft_mix_vars[:, :, :, 0])
        b = cp.array(2. * fft_mix_vars[:, :, :, 1])
        d = cp.array(fft_mix_vars[:, :, :, 2])

        v, w = img_params.v, img_params.w
        Nv = len(v)
        Nw = len(w)

        nu = cp.newaxis

        Fsum = cp.zeros((Nimages, Nprofiles, Nw, Nv), cp.float32)
        for k in range(Nfft):
            Fsum += (g_fft_mix_amps[:, :, k, nu, nu] *
                     cp.exp(-2. * cp.pi**2 *
                            (a[:, :, k, nu, nu] *  v[nu, nu, nu, :]**2 +
                             b[:, :, k, nu, nu] * (v[nu, nu, nu, :] *
                                                   w[nu, nu, :, nu]) +
                             d[:, :, k, nu, nu] *  w[nu, nu, :, nu]**2)))
        del a,b,d
        del g_fft_mix_amps
        G = cp.fft.irfft2(Fsum * img_params.P[:, nu, :, :])
        del Fsum
        pH, pW = img_params.pH, img_params.pW
        if Nmog > 0:
            ## FIXME -- trim these arrays to just non-zero weighted elements???
            # (ie, the non-padded rectangles of the images)
            psf_amps, psf_vars = img_params.psf_mogs
            _,Npsfmog = psf_amps.shape
            assert(G.shape == (Nimages,Nprofiles,pH,pW))
            assert(mog_mix_amps.shape == (Nimages, Nprofiles, Nmog))
            assert(mog_mix_vars.shape == (Nimages, Nprofiles, Nmog, 3))
            assert(psf_amps.shape == (Nimages, Npsfmog))
            assert(psf_vars.shape == (Nimages, Npsfmog, 3))

            # Convolve!
            Ncmog = Nmog * Npsfmog
            conv_amps = (mog_mix_amps[:, :, :, nu]    * psf_amps[:, nu, nu, :]).reshape((Nimages, Nprofiles, Ncmog))
            conv_vars = (mog_mix_vars[:, :, :, nu, :] + psf_vars[:, nu, nu, :, :]).reshape((Nimages, Nprofiles, Ncmog, 3))

            # variance terms: (00, 01, 11) covariance matrix elements
            det = (conv_vars[:,:,:,0] * conv_vars[:,:,:,2] - conv_vars[:,:,:,1]**2)
            iv0 = conv_vars[:,:,:,2] / det
            iv1 = -2. * conv_vars[:,:,:,1] / det
            iv2 = conv_vars[:,:,:,0] / det
            scale = conv_amps / (2. * np.pi * np.sqrt(det))
            assert(iv0.shape == (Nimages, Nprofiles, Ncmog))

            iv0 = cp.array(iv0, dtype=cp.float32)
            iv1 = cp.array(iv1, dtype=cp.float32)
            iv2 = cp.array(iv2, dtype=cp.float32)
            scale = cp.array(scale, dtype=cp.float32)

            # We're going to do the sub-pixel shift with Lanczos, so the
            # x position and mean are both integers.

            cx, cy = img_params.cx, img_params.cy
            # FIXME -- is it faster for these to be int, or float?
            xx = cp.arange(0-cx, pW-cx, dtype=cp.int32)
            yy = cp.arange(0-cy, pH-cy, dtype=cp.int32)
            # The distsq array is going to be nimages x nderivs x nmog x ny x nx
            distsq = -0.5 * (iv0[:,:,:,nu,nu] *  xx[nu,nu,nu,nu,:]**2 +
                             iv1[:,:,:,nu,nu] * (xx[nu,nu,nu,nu,:] *
                                                 yy[nu,nu,nu,:,nu]) +
                             iv2[:,:,:,nu,nu] *  yy[nu,nu,nu,:,nu]**2)
            del xx, yy, iv0, iv1, iv2
            distsq = cp.exp(distsq)
            assert(distsq.shape == (Nimages, Nprofiles, Ncmog, pH, pW))
            assert(scale.shape == (Nimages, Nprofiles, Ncmog))
            # Sum over the MoG components
            G_mog = cp.sum(distsq * scale[..., nu, nu], axis=2)
            del distsq
            assert(G_mog.shape == G.shape)
            G += G_mog
            del G_mog
            # FIXME -- avoid instantiating G_mog at all, just += into G
        assert(G.shape == (Nimages,Nprofiles,pH,pW))

        # FIXME -- check that this all remains float32 through the computations
        self.lanczos_shift_images_inplace_gpu(G, mux, muy)

        return G

    def get_vectorized_psfs(self, psfs, halfsize):
        cp = self.cp

        psfmogs = []
        maxK = 0
        for i,psf in enumerate(psfs):
            assert(isinstance(psf, HybridPSF))
            psfmog = psf.getMixtureOfGaussians()
            psfmogs.append(psfmog)
            maxK = max(maxK, psfmog.K)
        N = len(psfs)
        # We're going to assert zero mean here, and flatten the variance
        amps = np.zeros((N, maxK))
        #means = np.zeros((N, maxK, 2))
        varrs = np.zeros((N, maxK, 3))
        for i,psfmog in enumerate(psfmogs):
            amps [i, :psfmog.K] = psfmog.amp
            assert(np.all(psfmog.mean == 0))
            #means[i, :psfmog.K, :] = psfmog.mean
            varrs[i, :psfmog.K, 0] = psfmog.var[:, 0, 0]
            varrs[i, :psfmog.K, 1] = psfmog.var[:, 0, 1]
            varrs[i, :psfmog.K, 2] = psfmog.var[:, 1, 1]
        psf_mogs = amps,varrs

        imsize = psfs[0].img.shape
        for psf in psfs:
            #print('PSF:', psf)
            #print('sampling:', psf.sampling)
            assert(psf.sampling == 1.)
            #print('pixelized size', psf.img.shape)

        sz = 2**int(np.ceil(np.log2(halfsize * 2.)))
        W = H = sz
        pad = cp.zeros((N, H, W), cp.float32)
        cx = W//2
        cy = H//2
        for i,psf in enumerate(psfs):
            psfimg = psf.img
            ph,pw = psfimg.shape
            # We assume the center of the PSF image is at:
            pcy,pcx = ph//2, pw//2
            # And it must end up at cx,cy in the padded image.
            if pcx >= cx:
                # Trimming the PSF image
                out_x0 = 0
                in_x0 = pcx - cx
            else:
                # Padding the PSF image
                in_x0 = 0
                out_x0 = cx - pcx
            nx = min(pw, W)

            if pcy >= cy:
                # Trimming the PSF image
                out_y0 = 0
                in_y0 = pcy - cy
            else:
                # Padding the PSF image
                in_y0 = 0
                out_y0 = cy - pcy
            ny = min(ph, H)

            pad[i, out_y0 : out_y0 + ny, out_x0 : out_x0 + nx] = cp.array(
                psfimg[in_y0 : in_y0 + ny, in_x0 : in_x0 + nx])
        P = cp.fft.rfft2(pad)
        v = cp.fft.rfftfreq(W)
        w = cp.fft.fftfreq(H)
        # FIXME -- ??
        v = v.astype(cp.float32)
        w = w.astype(cp.float32)
        return P, (cx, cy), (H, W), (v, w), psf_mogs, pad

    def lanczos_shift_images_inplace_gpu(self, G, x, y, work=None):
        cp = self.cp
        '''
        Only vectorized in the specific way we need:
        G images:
        (Nimages x Nmodels x H x W)
        x, y:
        each of length (Nimages,)
        work:
        same shape as G; pre-allocated work array.
        '''
        assert(len(G.shape) == 4)
        if work is not None:
            assert(work.shape == G.shape)
        assert(len(x) == len(y))
        Nim, Nmod, H, W = G.shape
        assert(len(x) == Nim)
        # Create Lanczos filter arrays, shape (Nim, 7)
        fx = np.arange(-3, +4)[np.newaxis, :] + x[:, np.newaxis]
        fy = np.arange(-3, +4)[np.newaxis, :] + y[:, np.newaxis]
        fx = lanczos_filter(3, fx)
        fy = lanczos_filter(3, fy)
        self.correlate7_2d_inplace_gpu(G, fx, fy, work=work)
        del work

    def correlate7_2d_inplace_gpu(self, G, fx, fy, work=None):
        cp = self.cp
        if work is None:
            work = cp.empty_like(G)
        else:
            ## FIXME - only really need work array to be larger; use a view
            assert(work.shape == G.shape)

        fx = cp.array(fx)
        fy = cp.array(fy)

        assert(len(G.shape) == 4)
        Nim,Nmod,H,W = G.shape
        assert(len(fx.shape) == 2)
        assert(fx.shape == fy.shape)
        Nim2,K = fx.shape
        assert(Nim2 == Nim)
        assert(K == 7)

        # Apply X filter

        na = cp.newaxis

        # Special handling - left edge.
        work[:, :, :, 0] = cp.sum(G[:, :, :, :4] * fx[:, na, na, 3:], axis=-1)
        work[:, :, :, 1] = cp.sum(G[:, :, :, :5] * fx[:, na, na, 2:], axis=-1)
        work[:, :, :, 2] = cp.sum(G[:, :, :, :6] * fx[:, na, na, 1:], axis=-1)

        # Special handling - right edge.
        work[:, :, :, -1] = cp.sum(G[:, :, :, -4:] * fx[:, na, na, :4], axis=-1)
        work[:, :, :, -2] = cp.sum(G[:, :, :, -5:] * fx[:, na, na, :5], axis=-1)
        work[:, :, :, -3] = cp.sum(G[:, :, :, -6:] * fx[:, na, na, :6], axis=-1)

        # Middle
        work[:, :, :, 3:-3]  = G[:, :, :,  :-6] * fx[:, na, na, na, 0]
        work[:, :, :, 3:-3] += G[:, :, :, 1:-5] * fx[:, na, na, na, 1]
        work[:, :, :, 3:-3] += G[:, :, :, 2:-4] * fx[:, na, na, na, 2]
        work[:, :, :, 3:-3] += G[:, :, :, 3:-3] * fx[:, na, na, na, 3]
        work[:, :, :, 3:-3] += G[:, :, :, 4:-2] * fx[:, na, na, na, 4]
        work[:, :, :, 3:-3] += G[:, :, :, 5:-1] * fx[:, na, na, na, 5]
        work[:, :, :, 3:-3] += G[:, :, :, 6:  ] * fx[:, na, na, na, 6]

        # Apply Y filter

        # Special handling - bottom edge.
        G[:, :, 0, :] = cp.sum(work[:, :, :4, :] * fy[:, na, 3:, na], axis=-2)
        G[:, :, 1, :] = cp.sum(work[:, :, :5, :] * fy[:, na, 2:, na], axis=-2)
        G[:, :, 2, :] = cp.sum(work[:, :, :6, :] * fy[:, na, 1:, na], axis=-2)

        # Special handling - top edge.
        G[:, :, -1, :] = cp.sum(work[:, :, -4:, :] * fy[:, na, :4, na], axis=-2)
        G[:, :, -2, :] = cp.sum(work[:, :, -5:, :] * fy[:, na, :5, na], axis=-2)
        G[:, :, -3, :] = cp.sum(work[:, :, -6:, :] * fy[:, na, :6, na], axis=-2)

        # Middle
        G[:, :, 3:-3, :]  = work[:, :,  :-6, :] * fy[:, na, na, 0, na]
        G[:, :, 3:-3, :] += work[:, :, 1:-5, :] * fy[:, na, na, 1, na]
        G[:, :, 3:-3, :] += work[:, :, 2:-4, :] * fy[:, na, na, 2, na]
        G[:, :, 3:-3, :] += work[:, :, 3:-3, :] * fy[:, na, na, 3, na]
        G[:, :, 3:-3, :] += work[:, :, 4:-2, :] * fy[:, na, na, 4, na]
        G[:, :, 3:-3, :] += work[:, :, 5:-1, :] * fy[:, na, na, 5, na]
        G[:, :, 3:-3, :] += work[:, :, 6:  , :] * fy[:, na, na, 6, na]

# eg, lanczos_filter(3, -0.3 + np.arange(-3, +4))
def lanczos_filter(order, x):
    x = np.atleast_1d(x)
    nz = np.flatnonzero(np.logical_and(x != 0., np.logical_and(x < order, x > -order)))
    out = np.zeros(x.shape, dtype=np.float32)
    pinz = np.pi * x.flat[nz]
    out.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    out /= np.sum(out, axis=-1)[..., np.newaxis]
    return out

if __name__ == '__main__':
    from tractor.galaxy import ExpGalaxy
    from tractor.ellipses import EllipseE, EllipseESoft
    from tractor.basics import PixPos, Flux, ConstantSky, PointSource
    from tractor.basics import RaDecPos
    from tractor.wcs import ConstantFitsWcs
    from tractor.psfex import PixelizedPsfEx
    from tractor.psfex import NormalizedPixelizedPsfEx
    from tractor.psf import HybridPixelizedPSF, NCircularGaussianPSF
    from tractor import Image, NullWCS, Tractor
    from tractor.utils import _GaussianPriors
    from tractor import NanoMaggies, LinearPhotoCal
    from tractor.patch import ModelMask
    from tractor import ParamList
    import os
    import pylab as plt
    from astrometry.util.util import Tan

    from cupy_wrapper import cp

    def difference(x1, x2):
        return np.sum(np.abs(x1 - x2) / np.maximum(1e-16, (np.abs(x1) + np.abs(x2)) / 2.))

    def compare(meth1, meth2, vec1, vec2, icov):
        m = max(len(meth1), len(meth2))
        for meth,vec in [(meth1,vec1), (meth2,vec2)]:
            print(meth + ' '*(m-len(meth)) + ': [' +
                  ', '.join(['%12.5f' % v for v in vec]) + ' ]')
        print('Fractional difference (%s - %s): %.4g' % (meth1,meth2, difference(vec1, vec2)))
        chisq = (vec1 - vec2).T @ (icov @ (vec1 - vec2))
        print('Chi difference: %.4g' % np.sqrt(chisq))

    h,w = 100,200
    arcsec = 1./3600.
    ra_cen = 1.
    dec_cen = 2.

    # From legacypipe, a simplified EllipseESoft object with priors on the ellipticities.
    class EllipseWithPriors(EllipseESoft):
        ellipticityStd = 0.25
        ellipsePriors = None
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.ellipsePriors is None:
                ellipsePriors = _GaussianPriors(None)
                ellipsePriors.add('ee1', 0., self.ellipticityStd,
                                param=EllipseESoft(1.,0.,0.))
                ellipsePriors.add('ee2', 0., self.ellipticityStd,
                                param=EllipseESoft(1.,0.,0.))
                self.__class__.ellipsePriors = ellipsePriors
            self.gpriors = self.ellipsePriors
        @classmethod
        def getName(cls):
            return "EllipseWithPriors(%g)" % cls.ellipticityStd

    # From legacypipe, a Position class with no parameters.
    # Gaia measures positions better than we will, we assume, so the
    # GaiaPosition class pretends that it does not have any parameters
    # that can be optimized; therefore they stay fixed.
    class GaiaPosition(ParamList):
        def __init__(self, ra, dec, ref_epoch, pmra, pmdec, parallax):
            '''
            Units:
            - matches Gaia DR1
            - pmra,pmdec are in mas/yr.  pmra is in angular speed (ie, has a cos(dec) factor)
            - parallax is in mas.
            - ref_epoch: year (eg 2015.5)
            '''
            self.ra = ra
            self.dec = dec
            self.ref_epoch = float(ref_epoch)
            self.pmra = pmra
            self.pmdec = pmdec
            self.parallax = parallax
            super(GaiaPosition, self).__init__()
            self.cached_positions = {}
        def copy(self):
            return GaiaPosition(self.ra, self.dec, self.ref_epoch, self.pmra, self.pmdec,
                                self.parallax)
        def getPositionAtTime(self, mjd):
            from tractor import RaDecPos
            try:
                return self.cached_positions[mjd]
            except KeyError:
                # not cached
                pass
            if self.pmra == 0. and self.pmdec == 0. and self.parallax == 0.:
                pos = RaDecPos(self.ra, self.dec)
                self.cached_positions[mjd] = pos
                return pos
            ra,dec = radec_at_mjd(self.ra, self.dec, self.ref_epoch,
                                  self.pmra, self.pmdec, self.parallax, mjd)
            pos = RaDecPos(ra, dec)
            self.cached_positions[mjd] = pos
            return pos
        @staticmethod
        def getName():
            return 'GaiaPosition'
        def __str__(self):
            return ('%s: RA, Dec = (%.5f, %.5f), pm (%.1f, %.1f), parallax %.3f' %
                    (self.getName(), self.ra, self.dec, self.pmra, self.pmdec, self.parallax))
        def __getstate__(self):
            '''
            For pickling: omit cached positions
            '''
            d = self.__dict__.copy()
            d['cached_positions'] = dict()
            return d

    brightness = NanoMaggies(g=1000., r=2000., z=500.)

    shape = EllipseWithPriors(np.log(5.), 0.1, 0.4)
    pos = RaDecPos(ra_cen - 25.*arcsec, dec_cen)
    gpos = GaiaPosition(ra_cen - 25.*arcsec, dec_cen, 2016.0, 0., 0., 0.)
    #gal = ExpGalaxy(pos, brightness, shape)
    #param_scales = [1./3600, 1./3600., 100., 100., 100., 1., 0.1, 0.1]
    ptsrc = PointSource(pos, brightness)
    param_scales = [1./3600, 1./3600, 100., 100., 100]
    #gal = ExpGalaxy(gpos, brightness, shape)
    #param_scales = [100., 100., 100., 1., 0.1, 0.1]
    #cat = [gal]
    cat = [ptsrc]

    psf = NormalizedPixelizedPsfEx(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        'test',
                                        'psfex-decam-00392360-S31.fits'))
    psf = HybridPixelizedPSF(psf,
                             gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))
    psf = psf.constantPsfAt(w/2, h/2)
    print('const psf', psf)

    pixscale1 = 0.5*arcsec

    wcs1 = Tan(ra_cen + 0.4*arcsec, dec_cen - 0.3*arcsec, w/2+0.5, h/2+0.5,
               -pixscale1, 0., 0., pixscale1, float(w), float(h))
    sig1 = 1.0
    sig2 = 1.0
    tim1 = Image(np.zeros((h,w), np.float32),
                 inverr=np.ones((h,w), np.float32) / sig1,
                 psf=psf, sky=ConstantSky(0.),
                 wcs=ConstantFitsWcs(wcs1),
                 photocal=LinearPhotoCal(1.0, band='g'),
                )
    tr = Tractor([tim1], cat)
    mod = tr.getModelImage(0)
    noisy1 = mod + np.random.normal(scale=sig1, size=(h,w))
    tim1.data = noisy1

    pixscale2 = pixscale1
    rot = np.deg2rad(10.)
    wcs2 = Tan(ra_cen + 1.2*arcsec, dec_cen + 0.1*arcsec, w/2+0.5, h/2+0.5,
               -pixscale2 * np.cos(rot), -pixscale2 * np.sin(rot),
               -pixscale2 * np.sin(rot),  pixscale2 * np.cos(rot), float(w), float(h))

    h2,w2 = 105, 205
    psf = NormalizedPixelizedPsfEx(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        'test',
                                        'c4d_140717_065122_ooi_i_ls11-S2-se.psf'))
    psf = HybridPixelizedPSF(psf,
                             gauss=NCircularGaussianPSF([psf.fwhm / 2.35, psf.fwhm / 2.35 / 2],
                                                        [0.9, 0.1]))
    psf = psf.constantPsfAt(w/2, h/2)
    tim2 = Image(np.zeros((h2,w2), np.float32),
                 inverr=np.ones((h2,w2), np.float32) / sig2,
                 psf=psf, sky=ConstantSky(0.),
                 wcs=ConstantFitsWcs(wcs2),
                 photocal=LinearPhotoCal(1.0, band='r'),
                )
    tr = Tractor([tim1, tim2], cat)
    tr.freezeParam('images')

    mod2 = tr.getModelImage(tim2)
    noisy2 = mod2 + np.random.normal(scale=sig2, size=(h2,w2))
    tim2.data = noisy2

    true_params = np.array(tr.getParams())

    ## FIXME -- move initial params away from truth!

    p0 = tr.getParams() + np.random.normal(size=len(param_scales)) * np.array(param_scales)
    tr.setParams(p0)

    print('Opt', tr.optimizer)

    optargs = dict(shared_params=False, priors=True)

    orig_opt = tr.optimizer

    up0 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('LSQR Update:', up0)

    tr.setParams(p0)

    src = cat[0]
    tr.setModelMasks([{src: ModelMask(110, 10, 80, 80),}
                      for tim in tr.images])

    up0m = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('LSQR Update w/ modelMasks:', up0m)
    up0 = up0m

    sm_opt = SmarterDenseOptimizer()
    tr.optimizer = sm_opt

    allderivs = tr.getDerivs()
    up1,ic,colmap = tr.optimizer.getUpdateDirection(tr, allderivs, get_cov=True, **optargs)
    print('colmap', colmap)
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Smarter update:', up1)
    n = len(up1)
    ic1 = np.zeros((n,n,), np.float32)
    for j,i in enumerate(colmap):
        ic1[i,colmap] = ic[j,:]
    ic = ic1

    tr.setParams(p0)

    from cupy_wrapper import cp

    gpu_opt = GpuOptimizer(cp)
    tr.optimizer = gpu_opt
    up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('GPU Update:', up2)

    tr.setParams(p0)

    print('Fractional difference (LSQR-SM):', difference(up0, up1))
    compare('LSQR', 'SM',  up0, up1, ic)

    print('Fractional difference (LSQR-GPU):', difference(up0, up2))
    print('Fractional difference (SM-GPU):', difference(up1, up2))

    compare('LSQR', 'GPU', up0, up2, ic)
    compare('SM',   'GPU', up1, up2, ic)

    tr.setParams(p0)

    mod1 = tr.getModelImage(0)
    mod2 = tr.getModelImage(1)
    mn,mx = np.percentile(tim1.getImage().ravel(), [2,98])
    ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
    chima = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=+5)
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(mod1, **ima)
    plt.subplot(2,3,4)
    plt.imshow(mod2, **ima)
    plt.subplot(2,3,2)
    plt.imshow(tim1.getImage(), **ima)
    plt.subplot(2,3,5)
    plt.imshow(tim2.getImage(), **ima)
    plt.subplot(2,3,3)
    plt.imshow((tim1.getImage() - mod1) * tim1.getInvError(), **chima)
    plt.subplot(2,3,6)
    plt.imshow((tim2.getImage() - mod2) * tim2.getInvError(), **chima)
    plt.savefig('before.png')

    tr.optimize_loop(**optargs)

    print('After optimization:', tr.catalog[0])

    mod1 = tr.getModelImage(0)
    mod2 = tr.getModelImage(1)
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(mod1, **ima)
    plt.subplot(2,3,4)
    plt.imshow(mod2, **ima)
    plt.subplot(2,3,2)
    plt.imshow(tim1.getImage(), **ima)
    plt.subplot(2,3,5)
    plt.imshow(tim2.getImage(), **ima)
    plt.subplot(2,3,3)
    plt.imshow((tim1.getImage() - mod1) * tim1.getInvError(), **chima)
    plt.subplot(2,3,6)
    plt.imshow((tim2.getImage() - mod2) * tim2.getInvError(), **chima)
    plt.savefig('after.png')
