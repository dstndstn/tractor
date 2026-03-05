import time

import numpy as np

from tractor.smarter_dense_optimizer import SmarterDenseOptimizer
from tractor import ProfileGalaxy, HybridPSF, ConstantSky, PointSource, Patch
from tractor.brightness import LinearPhotoCal
from tractor.basics import ConstantSurfaceBrightness

'''
TO-DO:
-- check in oneblob.py - are we passing NormalizedPixelizedPsfEx (VARYING) PSF objects
   in at some point?  They should all get turned into constant PSFs.
-- check PSF sampling != 1.0

'''
import logging
logger = logging.getLogger('tractor.gpu_optimizer')
def logverb(*args):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(' '.join(map(str, args)))
debug = logverb
def logmsg(*args):
    logger.info(' '.join(map(str, args)))
info = logmsg
def isverbose():
    return logger.isEnabledFor(logging.DEBUG)

class Duck(object):
    pass

class GpuFriendlyOptimizer(SmarterDenseOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_super = 0.
        self.n_super = 0

    def getLinearUpdateDirection(self, tr, priors=True, get_icov=False, **kwargs):
        if not (tr.isParamFrozen('images') and
                (((len(tr.catalog) == 1) and
                  isinstance(tr.catalog[0], (ProfileGalaxy, PointSource))) or
                 ((len(tr.catalog) == 2) and
                  isinstance(tr.catalog[0], (ProfileGalaxy, PointSource, type(None))) and
                  isinstance(tr.catalog[1], ConstantSurfaceBrightness))
                 )):
            print('GpuFriendlyOptimizer: falling back to super.  Images frozen: %s, Cat len: %i' %
                  (tr.isParamFrozen('images'), len(tr.catalog)))
            for src in tr.catalog:
                print('  src', src, 'is gal/psf: %s; is CSB: %s' %
                      (isinstance(src, (ProfileGalaxy, PointSource)),
                       isinstance(src, ConstantSurfaceBrightness)))
            return super().getLinearUpdateDirection(tr, priors=priors, get_icov=get_icov,
                                                    **kwargs)
        assert(get_icov == False)
        return self.one_source_update(tr, priors=priors, **kwargs)

    def tryUpdates(self, tr, X, **kwargs):
        if not (tr.isParamFrozen('images') and
                (((len(tr.catalog) == 1) and
                  isinstance(tr.catalog[0], (ProfileGalaxy, PointSource))) or
                 ((len(tr.catalog) == 2) and
                  isinstance(tr.catalog[0], (ProfileGalaxy, PointSource, type(None))) and
                  isinstance(tr.catalog[1], ConstantSurfaceBrightness))
                 )):
            return super().tryUpdates(tr, X, **kwargs)
        return self.one_source_try_updates(tr, X, **kwargs)

    def one_source_update(self, tr, **kwargs):
        return super().getLinearUpdateDirection(tr, **kwargs)

    def one_source_try_updates(self, tr, X, **kwargs):
        return super().tryUpdates(tr, X, **kwargs)

class GpuOptimizer(GpuFriendlyOptimizer):

    def __init__(self, cp, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_cp(cp)

    def __getstate__(self):
        d = self.__dict__.copy()
        # Can't pickle a module
        del d['cp']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.set_cp(self.cp_orig)

    def set_cp(self, cp):
        if cp == 'cupy':
            self.cp_orig = cp
            import cupy as cp
            cp.get = lambda x: x.get()
        elif cp == 'numpy':
            self.cp_orig = cp
            import numpy as np
            np.get = lambda x: x
            cp = np
        else:
            self.cp_orig = None
        self.cp = cp

    def __repr__(self):
        return 'GpuOptimizer(' + (self.cp_orig or str(self.cp)) + ')'

    def one_source_try_updates(self, tr, X, **kwargs):
        return self.gpu_one_source_try_updates(tr, X, **kwargs)

    def one_source_update(self, tr, **kwargs):
        return self.gpu_one_source_update(tr, **kwargs)

    def _gpu_checks(self, tr):
        # Assume single source, optionally with a ConstantSurfaceBrightness
        assert(len(tr.catalog) in [1,2])
        tims = tr.images

        # Assume galaxy or point source, or None (when we're fitting no-source + SB)
        src = tr.catalog[0]
        is_galaxy = isinstance(src, ProfileGalaxy)
        is_psf = isinstance(src, PointSource)
        is_none = (src is None)
        assert(is_galaxy or is_psf or is_none)

        sb = None
        if len(tr.catalog) == 2:
            sb = tr.catalog[1]
            assert(isinstance(sb, ConstantSurfaceBrightness))
        if is_none:
            assert(sb)
            # ... otherwise, what are we doing?

        # Assume model masks are set (ie, pixel ROIs of interest are defined)
        #masks = [tr._getModelMaskByIdx(i, src) for i in range(len(tr.images))]
        if not is_none:
            masks = [tr._getModelMaskFor(tim, src) for tim in tims]
        else:
            masks = [tr._getModelMaskFor(tim, sb) for tim in tims]
        if any(m is None for m in masks):
            debug('One or more modelMasks is None in GPU code -- cutting images')
            goodtims = []
            goodmasks = []
            for tim,mask in zip(tims,masks):
                if mask is None:
                    continue
                goodtims.append(tim)
                goodmasks.append(mask)
            if len(goodtims) == 0:
                debug('After removing None modelMasks, no images remain!')
                return None, None
            debug('Cut from %i to %i images with good modelMasks' % (len(tr.images), len(goodtims)))
            tims = goodtims
            masks = goodmasks
            del goodtims,goodmasks
        assert(all([m is not None for m in masks]))
        return tims, masks

    def gpu_one_source_try_updates(self, tr, X, alphas=None, **kwargs):
        cp = self.cp
        na = cp.newaxis
        tims,masks = self._gpu_checks(tr)
        if tims is None:
            return 0., 0.
        src = tr.catalog[0]
        is_galaxy = isinstance(src, ProfileGalaxy)
        is_psf = isinstance(src, PointSource)
        sb = None
        is_none = (src is None)
        if len(tr.catalog) == 2:
            sb = tr.catalog[1]
        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T

        if src:
            # Pixel positions (at initial params)
            pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
                   for tim in tims]
            px, py = np.array(pxy, dtype=np.float32).T
        else:
            # fake source position in middle of modelMasks
            px = (x0+x1)/2.
            py = (y0+y1)/2.
        ipx = px.round().astype(np.int32)
        ipy = py.round().astype(np.int32)

        halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py]))

        # Pre-process image: PSF, padded pix, etc
        img_params = self.gpu_setup_image_params(tims, halfsize, extents, ipx, ipy)

        p0 = tr.getParams()

        # FIXME -- check_step??
        # Get lists of alpha values and parameters to try
        steps = [(0., p0, False, False)] + self.getParameterSteps(tr, X, alphas)
        if len(steps) == 0:
            self.last_step_hit_limit = True
            self.hit_limit = True
            return 0., 0.

        counts_grid = np.zeros((len(tims), len(steps)), np.float32)
        if sb:
            sb_counts_grid = np.zeros((len(tims), len(steps)), np.float32)

        # We need to transpose the mogs: need Nimages x Nprofiles(aka Nsteps)
        mogs = [[] for tim in tims]

        idx_grid = np.zeros((len(tims), len(steps)), np.int32)
        idy_grid = np.zeros((len(tims), len(steps)), np.int32)

        mux_grid = np.zeros((len(tims), len(steps)), np.float32)
        muy_grid = np.zeros((len(tims), len(steps)), np.float32)

        Nmods = len(steps)
        logpriors = np.zeros(Nmods, np.float32)
        for istep, (_, p, _, _) in enumerate(steps):
            tr.setParams(p)
            logpriors[istep] = tr.getLogPrior()

            if src:
                # Pixel positions
                pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
                       for tim in tims]
                px, py = np.array(pxy, dtype=np.float32).T
                # Round source pixel position to nearest integer
                ix = px.round().astype(np.int32)
                iy = py.round().astype(np.int32)
                idx_grid[:, istep] = ix - ipx
                idy_grid[:, istep] = iy - ipy

                # subpixel portion: shifted via Lanczos interpolation
                mux = px - ix
                muy = py - iy
                assert(np.abs(mux).max() <= 0.5)
                assert(np.abs(muy).max() <= 0.5)
                mux_grid[:, istep] = mux
                muy_grid[:, istep] = muy

                counts_grid[:, istep] = np.array(
                    [tim.getPhotoCal().brightnessToCounts(src.brightness)
                     for tim in tims])

            if sb:
                sb_counts_grid[:, istep] = np.array(
                    [tim.getPhotoCal().brightnessToCounts(sb.brightness) *
                     tim.getWcs().pixel_scale()**2
                     for tim in tims])

            if is_galaxy:
                for itim,(tim,x,y) in enumerate(zip(tims, px, py)):
                    mogs[itim].append(src._getShearedProfile(tim,x,y))

        if is_psf:
            mods = self.gpu_get_unitflux_psf(img_params, mux_grid, muy_grid)
        elif is_galaxy:
            mods = self.gpu_get_unitflux_galaxy_profiles(mogs, img_params,
                                                         mux_grid, muy_grid)
        if is_psf or is_galaxy:
            mods *= cp.array(counts_grid)[..., na, na]
        # add sb below to make shifting better

        # mods shape: Nimages x Nmod x pH x pW
        pH, pW = img_params.pH, img_params.pW
        Nimages = len(tims)
        assert(img_params.padpix.shape == (Nimages, pH, pW))
        assert(img_params.padie.shape  == (Nimages, pH, pW))
        if is_psf or is_galaxy:
            assert(mods.shape == (Nimages,Nmods,pH,pW))
        chisqs = cp.zeros(Nmods, np.float32)

        #plt.clf()
        #chia = dict(interpolation='nearest', origin='lower', vmin=-3, vmax=+3)
        # FIXME -- quick path if all idx,idy == 0, or if all for one model == 0 ?
        chi_work = cp.empty((pH, pW), np.float32)
        for imod in range(Nmods):
            for itim in range(Nimages):
                dx = idx_grid[itim, imod]
                dy = idy_grid[itim, imod]

                #plt.subplot(Nmods, Nimages, imod*Nimages + itim + 1)
                #plt.title('dx,dy = %i,%i' % (dx, dy))

                if dx == 0 and dy == 0:
                    if sb and is_none:
                        # only SB
                        chisqs[imod] += cp.sum(((img_params.padpix[itim, ...] -
                                                 sb_counts_grid[itim, imod]) *
                                                img_params.padie[itim, ...])**2)
                    elif sb:
                        chisqs[imod] += cp.sum(((img_params.padpix[itim, ...] -
                                                 (mods[itim, imod, ...] + sb_counts_grid[itim, imod])) *
                                                img_params.padie[itim, ...])**2)
                    else:
                        chisqs[imod] += cp.sum(((img_params.padpix[itim, ...] -
                                                 mods[itim, imod, ...]) *
                                                img_params.padie[itim, ...])**2)
                    # DEBUG
                    #chi = (img_params.padpix[itim, ...] - mods[itim, imod, ...]) * img_params.padie[itim, ...]
                    #plt.imshow(chi, **chia)
                    continue
                pix = img_params.padpix[itim, ...]
                ie  = img_params.padie [itim, ...]

                if sb:
                    chi_work[:,:] = pix - sb_counts_grid[itim, imod]
                else:
                    chi_work[:,:] = pix

                if not is_none:
                    mod = mods[itim, imod, ...]
                    if dx > 0:
                        x_pix_slice = slice(dx, None)
                        x_mod_slice = slice(-dx)
                    elif dx == 0:
                        x_pix_slice = slice(None)
                        x_mod_slice = slice(None)
                    else:
                        x_pix_slice = slice(dx)
                        x_mod_slice = slice(-dx, None)

                    if dy > 0:
                        y_pix_slice = slice(dy, None)
                        y_mod_slice = slice(-dy)
                    elif dy == 0:
                        y_pix_slice = slice(None)
                        y_mod_slice = slice(None)
                    else:
                        y_pix_slice = slice(dy)
                        y_mod_slice = slice(-dy, None)
                    chi_work[y_pix_slice, x_pix_slice] -= mod[y_mod_slice, x_mod_slice]
                chisqs[imod] += cp.sum((chi_work * ie)**2)
                #chi = (pix - mod) * ie
                #chisqs[imod] += cp.sum(((pix - mod) * ie).astype(cp.float64)**2)
                #plt.imshow(chi_work, **chia)
        #ps.savefig()

        #print('chisqs:', chisqs)
        #print('loglikes:', -0.5 * chisqs)
        #print('delta-lnls:', -0.5 * (chisqs[1:] - chisqs[0]))
        chisqs = cp.get(chisqs)
        logprobs = logpriors - 0.5 * chisqs

        logprob_best = logprobs[0]
        alpha_best = None
        p_best = p0
        for istep, ((alpha, p, step_limit, hit_limit), logprob) in enumerate(
            zip(steps, logprobs)):
            if istep == 0:
                continue
            if hit_limit:
                self.hit_limit = True
            if not np.isfinite(logprob):
                tr.setParams(p)
                print('GPUOptimizer: log-prob %s.  Prior %s, Likelihood %s. Source: %s' %
                      (logprob, logpriors[istep], -0.5*chisqs[istep], str(src)))
                break
            if logprob < (logprob_best - 1.):
                # We're getting significantly worse -- quit line search
                break
            if logprob > logprob_best:
                # Best we've found so far -- accept this step!
                self.last_step_hit_limit = hit_limit
                alpha_best = alpha
                logprob_best = logprob
                p_best = p
        tr.setParams(p_best)
        if alpha_best is None:
            return 0., 0.

        return logprob_best - logprobs[0], alpha_best

    def gpu_one_source_update(self, tr, priors=True, get_A=False, **kwargs):
        cp = self.cp
        nu = cp.newaxis
        t0 = time.time()

        tims,masks = self._gpu_checks(tr)
        if tims is None or len(tims) == 0:
            return None
        src = tr.catalog[0]
        is_galaxy = isinstance(src, ProfileGalaxy)
        is_psf = isinstance(src, PointSource)
        is_none = (src is None)
        sb = None
        if len(tr.catalog) == 2:
            sb = tr.catalog[1]

        Nimages = len(tims)
        extents = [mm.extent for mm in masks]
        x0, x1, y0, y1 = np.asarray(extents).T

        if src:
            # Pixel positions
            pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
                   for tim in tims]
            px, py = np.array(pxy, dtype=np.float32).T
            # WCS inv(CD) matrix
            img_cdi = [tim.getWcs().cdInverseAtPixel(x,y) for tim,x,y in zip(tims, px, py)]
            # Current counts
            img_counts = [tim.getPhotoCal().brightnessToCounts(src.brightness)
                          for tim in tims]
            src_bands = src.getBrightness().getParamNames()
            fluxes = cp.array(img_counts)
        else:
            # fake source position in middle of modelMasks
            px = (x0+x1)/2.
            py = (y0+y1)/2.
            src_bands = []
        # Round source pixel position to nearest integer
        ipx = px.round().astype(np.int32)
        ipy = py.round().astype(np.int32)
        ## Here, we're basically just rounding up the modelMask size.
        ## This affects which terms in the Gaussian mixture model of the galaxy
        ## profile get rendered with FFT vs Gaussians.
        halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py]))

        # Pre-process image: PSF, padded pix, etc
        img_params = self.gpu_setup_image_params(tims, halfsize, extents, ipx, ipy)

        if is_galaxy:
            # Get galaxy profiles for shape derivatives.
            amixes = [[('current', src._getShearedProfile(tim,x,y), 0.)] +
                    src.getDerivativeShearedProfiles(tim,x,y)
                    for tim,x,y in zip(tims, px, py)]
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
        elif is_psf:
            G = self.gpu_get_unitflux_psf(img_params, mux, muy)

        # Now we have computed the galaxy profiles for the finite-differences steps.
        # Turn these into derivatives and solve the update matrix.
        pH, pW = img_params.pH, img_params.pW
        assert(img_params.padpix.shape == (Nimages, pH, pW))
        assert(img_params.padie.shape  == (Nimages, pH, pW))

        # We have either RaDecPos or GaiaPositions for the pos -- 2 or 0 params.
        Npos = 0
        if src and src.isParamThawed('pos'):
            Npos = src.pos.numberOfParams()
        assert(Npos in [0, 2])

        # List of the band for each tim
        tim_bands = [tim.getPhotoCal().band for tim in tims]
        # The unique set of bands in all tims.  This is the number of flux parameters we
        # will fit for in the A matrix, in this order.
        tim_ubands = list(np.unique(tim_bands))
        # List of the index into tim_ubands for each tim
        # This is what we'll use to build the A matrix
        tim_band_index = [tim_ubands.index(b) for b in tim_bands]
        Nbands = len(tim_ubands)

        if is_galaxy:
            # Nprofiles - 1: profile 0 is the current galaxy shape, the rest are shape/sersic derivs.
            Nshapes = (Nprofiles - 1)
        else:
            Nshapes = 0

        Nsb = 0
        if sb:
            sb_counts = np.array([tim.getPhotoCal().brightnessToCounts(sb.brightness)
                                  for tim in tims])
            # formally, only pixscale_at(x,y) is defined in ducks.py....
            pixscales = np.array([tim.getWcs().pixel_scale()
                                  for tim in tims])
            sb_counts_per_pix = cp.array(sb_counts * pixscales**2)
            Nsb = Nbands

        if is_none:
            Nbands = 0

        Nderivs = Npos + Nbands + Nshapes + Nsb

        # We'll need both directions of the mapping between
        # source parameter index and index in our A matrix
        # due to "tims" having only a subset of the bands in the source's params.
        # List of (source parameter index, A matrix index)
        # This is baking in the assumption that source parameters are in the order:
        # pos, brightness, shape
        # this is in the (source, fitting) order
        param_indices = []
        if Npos:
            param_indices.append((0,0))
            param_indices.append((1,1))
        for i,band in enumerate(src_bands):
            if band in tim_ubands:
                param_indices.append((Npos + i, Npos + tim_ubands.index(band)))
        for i in range(Nshapes):
            param_indices.append((Npos + len(src_bands) + i, Npos + Nbands + i))
        N_sb_bands = 0
        if sb:
            sb_bands = sb.brightness.getParamNames()
            N_sb_bands = len(sb_bands)
            for i,band in enumerate(sb_bands):
                if band in tim_ubands:
                    param_indices.append((Npos + len(src_bands) + Nshapes + i,
                                          Npos + Nbands + Nshapes + tim_ubands.index(band)))

        # Mappings between the parameters we're going to fit for (fit params)
        # and source parameters (src params).  Eg, a source might have a flux
        # parameter that is not constrained by the images we are fitting.
        src_to_fit_param = dict(param_indices)
        fit_to_src_param = dict([(f,s) for s,f in param_indices])

        assert(all([isinstance(tim.getPhotoCal(), LinearPhotoCal) for tim in tims]))
        padie = img_params.padie
        padpix = img_params.padpix
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
            dx[:, 1:-1, 1:-1] = (G[:, 0, 1:-1,  :-2] - G[:, 0, 1:-1, 2:  ]) / 2.
            dy[:, 1:-1, 1:-1] = (G[:, 0,  :-2, 1:-1] - G[:, 0, 2:,   1:-1]) / 2.

            cdi = cp.array(img_cdi)
            assert(cdi.shape == (Nimages,2,2))

            A[:Npix_total, 0] = (fluxes[:, nu, nu] *
                                 (dx * cdi[:, 0, 0][:, nu, nu] +
                                  dy * cdi[:, 1, 0][:, nu, nu]) * padie).ravel()
            A[:Npix_total, 1] = (fluxes[:, nu, nu] *
                                 (dx * cdi[:, 0, 1][:, nu, nu] +
                                  dy * cdi[:, 1, 1][:, nu, nu]) * padie).ravel()

        if src:
            # Flux derivatives.
            for i in range(Nimages):
                # Image i fills in the column corresponding to its flux
                # and a block of rows corresponding to its pixels.
                col = Npos + tim_band_index[i]
                A[i * pH*pW: (i+1) * pH*pW, col] = (G[i, 0, :, :] * padie[i, :, :]).ravel()
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
                                                     padie).ravel()

        if sb:
            # Surface-brightness derivatives.
            for i in range(Nimages):
                # Image i fills in the column corresponding to its sb flux
                # and a block of rows corresponding to its pixels.
                col = Npos + Nbands + Nshapes + tim_band_index[i]
                # This might not be strictly necessary to mask by "padmask", but it makes the results
                # match more closely with the traditional code, because the colscales are the same.
                # FIXME -- here we must be assuming LinearPhotoCal with scale = 1!
                A[i * pH*pW: (i+1) * pH*pW, col] = (pixscales[i]**2 * padie[i, :, :].ravel())

        if sb and src:
            # add current surface brightnesses to the model
            B[:Npix_total] = ((padpix - (fluxes[:, nu, nu] * G[:, 0, :, :] + sb_counts_per_pix[:, nu, nu]))
                              * padie).ravel()
        elif sb:
            B[:Npix_total] = ((padpix - sb_counts_per_pix[:, nu, nu]) * padie).ravel()
        else:
            B[:Npix_total] = ((padpix - fluxes[:, nu, nu] * G[:, 0, :, :]) * padie).ravel()

        if Npriors > 0:
            rA, cA, vA, pb, mub = priorVals
            for ri,ci,vi,bi in zip(rA, cA, vA, pb):
                for rij,vij,bij in zip(ri, vi, bi):
                    # Map the source's parameter index to our local fitting parameter index
                    ci = src_to_fit_param[ci]
                    A[Npix_total + rij, ci] = vij
                    B[Npix_total + rij] += bij
            del priorVals, rA, cA, vA, pb, mub

        # Precondition the A matrix (simply) by normalizing the column scales (L2 norms)
        colscales = cp.sqrt((A**2).sum(axis=0))

        # Remove any columns with zero colscale (ie, no actual derivatives)
        if cp.any(colscales == 0):
            colscales = cp.get(colscales)
            I = np.flatnonzero(colscales)
            if len(I) == 0:
                debug('All derivatives are zero.')
                return None
            new_fit_to_src_param = {}
            new_src_to_fit_param = {}
            for i_new,i_old in enumerate(I):
                s = fit_to_src_param[i_old]
                new_fit_to_src_param[i_new] = s
                new_src_to_fit_param[s] = i_new
            fit_to_src_param = new_fit_to_src_param
            src_to_fit_param = new_src_to_fit_param
            A = A[:, I]
            colscales = colscales[I]
            colscales = cp.array(colscales)

        # Cut A down to only active rows
        goodrows = cp.any(A, axis=1)
        if not cp.all(goodrows):
            A = A[goodrows, :]
            B = B[goodrows]
            del goodrows

        A /= colscales[nu, :]
        X,_,_,_ = cp.linalg.lstsq(A, B, rcond=None)
        X /= colscales

        X = cp.get(X)
        # Parameter indices in the tractor params of our vector X:
        I = np.array([fit_to_src_param[i] for i in range(len(X))])
        # The vector we're going to return:
        N_src_params = (src and src.numberOfParams() or 0)
        Nout = N_src_params + N_sb_bands
        sX = np.zeros(Nout, np.float32)
        sX[I] = X
        assert(np.all(np.isfinite(sX)))

        if get_A:
            sA = np.zeros((Nrows, Nout), np.float32)
            sA[:,I] = cp.get(A)
            s_scales = np.zeros(Nout, np.float32)
            s_scales[I] = cp.get(colscales)
            B = cp.get(B)
            return sA, B, sX, s_scales, pH,pW, img_params.padslices #cp.get(img_params.padmask)

        return sX

    def gpu_setup_image_params(self, tims, halfsize, extents, ipx, ipy):
        '''
        ipx,ipy: arrays of length = len(tims), of the tim pixel coordinate
                 that will be placed in the padded image coord cx,cy.
                 ie, the center of the padded patches that we'll compute.
        '''
        cp = self.cp
        # Assume no (varying) sky background levels
        assert(all([isinstance(tim.sky, ConstantSky) for tim in tims]))
        # Assume sky levels are zero.
        assert(all([tim.getSky().getValue() == 0 for tim in tims]))

        psfs = [tim.getPsf() for tim in tims]
        # Assume hybrid PSF model
        assert(all([isinstance(psf, HybridPSF) for psf in psfs]))

        img_params = Duck()
        psfH, psfW = np.array([psf.shape for psf in psfs]).T
        P, (cx, cy), (pH, pW), (v, w), psf_mogs, psf_pad = self.get_vectorized_psfs(psfs, halfsize)
        assert(pW % 2 == 0)
        assert(pH % 2 == 0)
        Nimages = len(tims)
        assert(P.shape == (Nimages,len(w),len(v)))

        # Embed pix and ie in images the same size as pW,pH.
        # FIXME -- should be able to cache this; rationalize pixel transfer to GPU.
        padpix  = cp.zeros((Nimages, pH,pW), cp.float32)
        padie   = cp.zeros((Nimages, pH,pW), cp.float32)
        padmask = cp.zeros((Nimages, pH,pW), bool)
        padslices = []
        for i,(tim,(x0,x1,y0,y1)) in enumerate(zip(tims, extents)):
            pix = tim.getImage()
            ie = tim.getInvError()

            dx = cx - ipx[i]
            dy = cy - ipy[i]
            p = pix[y0:y1, x0:x1]
            padpix [i, y0+dy : y1+dy, x0+dx : x1+dx] = cp.array(p)
            p =  ie[y0:y1, x0:x1]
            padie  [i, y0+dy : y1+dy, x0+dx : x1+dx] = cp.array(p)
            padmask[i, y0+dy : y1+dy, x0+dx : x1+dx] = True
            padslices.append((slice(y0+dy, y1+dy), slice(x0+dx, x1+dx)))

        # GPU arrays:
        img_params.psf_pad = psf_pad
        img_params.P = P
        img_params.v = v
        img_params.w = w
        img_params.padpix = padpix
        img_params.padie  = padie
        img_params.padmask = padmask
        # numpy arrays:
        img_params.psf_mogs = psf_mogs
        img_params.padslices = padslices # debug
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

        assert(mux.shape == muy.shape)
        if len(mux.shape) == 1:
            assert(len(mux) == nimg)
            npro = 1
            in_img = padpsf.reshape((nimg, npro, h, w))
        else:
            assert(len(mux.shape) == 2)
            assert(mux.shape[0] == nimg)
            _,npro = mux.shape
            # hmmm, unnecessary copy...
            #in_img = padpsf[:, cp.newaxis, :, :].reshape((nimg, npro, h, w))
            in_img = padpsf[:, cp.newaxis, :, :].repeat(npro, axis=1)
        out_img = cp.empty((nimg, npro, h, w), cp.float32)
        self.lanczos_shift_images_gpu(in_img, out_img, mux, muy)
        return out_img

    def gpu_get_unitflux_galaxy_profiles(self, mogs, img_params, mux, muy):
        '''
        Computes galaxy profiles for a grid of Nimages x Nprofiles.

        "img_params" contains (padded) image pixels and processed PSFs;
        Nimages in size.

        "mogs" gives the galaxy (mixture-of-Gaussian) profiles, for the
        grid of (Nimages x Nprofiles).

        "mux" and "muy" are sub-pixel shifts, and are each either of
        length Nimages, or size (Nimage x Nprofiles); they just get
        passed to the Lanczos method which can handle either.
        '''
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
        IM = (sz**2 < (nsigma2**2 * vv)) * (mix_amps != 0)
        IF = (sz**2 > (nsigma1**2 * vv)) * (mix_amps != 0)
        ramp = np.any((IM*IF))

        # Assume the MoG components are sorted by increasing variance; terms we'll evaluate
        # with the FFT will be at the front, and MoG at the back.
        # (BUT, because we pad out the arrays to the max K of MoG components, the end of the array
        # can contain zeros!)
        Kmog = np.flatnonzero(IM.max(axis=(0,1)))
        Kfft = np.flatnonzero(IF.max(axis=(0,1)))
        if len(Kmog) == 0:
            Nmog = 0
        else:
            Nmog = Kmax - min(Kmog)
        if len(Kfft) == 0:
            Nfft = 0
        else:
            Nfft = 1 + max(Kfft)
        assert(np.all(Kmog >= (Kmax-Nmog)))
        assert(np.all(Kfft < Nfft))

        mog_mix_vars = mix_vars[:, :, -Nmog:, :]
        fft_mix_vars = mix_vars[:, :, :Nfft:, :]
        mog_mix_amps = mix_amps[:, :, -Nmog:]
        fft_mix_amps = mix_amps[:, :, :Nfft]

        # Assert that the block of the MoG model that we're pulling off to process with
        # the MoG / FFT methods include all of the instances where we want to run those methods.
        assert(np.sum(IM) == np.sum(IM[:, :, -Nmog:]))
        assert(np.sum(IF) == np.sum(IF[:, :, :Nfft ]))

        if ramp:
            # ramp between MOG and FFT for intermediate-sigma components
            mogweights = np.ones(vv.shape, np.float32)
            fftweights = np.ones(vv.shape, np.float32)
            ns = sz / np.maximum(1e-6, np.sqrt(vv))
            mogweights = (np.clip((nsigma2 - ns) / (nsigma2 - nsigma1), 0., 1.) * IM)[..., -Nmog:]
            fftweights = (np.clip((ns - nsigma1) / (nsigma2 - nsigma1), 0., 1.) * IF)[..., :Nfft]
            mogweights *= mog_mix_amps
            mog_mix_amps = mogweights
            fftweights *= fft_mix_amps
            fft_mix_amps = fftweights
            # BRUTAL -- mog_mix_amps and fft_mix_amps are VIEWS into the same mix_amps array!
            #mog_mix_amps *= mogweights[:, :, -Nmog:]
            #fft_mix_amps *= fftweights[:, :, :Nfft]
            del mogweights, fftweights

        nu = cp.newaxis

        if Nfft > 0:
            g_fft_mix_amps = cp.array(fft_mix_amps)
            a = cp.array(fft_mix_vars[:, :, :, 0])
            b = cp.array(2. * fft_mix_vars[:, :, :, 1])
            d = cp.array(fft_mix_vars[:, :, :, 2])

            v, w = img_params.v, img_params.w
            Nv = len(v)
            Nw = len(w)

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
        else:
            G = None

        pH, pW = img_params.pH, img_params.pW
        if Nmog > 0:
            ## FIXME -- trim these arrays to just non-zero weighted pixels???
            # (ie, the non-padded rectangles of the images)
            psf_amps, psf_vars = img_params.psf_mogs
            _,Npsfmog = psf_amps.shape
            if G is not None:
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
            # because we pad out these arrays to the max sizes, dets can be zero.
            assert(np.all(np.logical_or(det > 0, (det == 0) & (conv_amps == 0))))
            # avoid 0 * inf (amp=0 * 1/(det=0))
            det = np.maximum(det, 1e-30)
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
            xx = cp.arange(0-cx, pW-cx, dtype=cp.float32)
            yy = cp.arange(0-cy, pH-cy, dtype=cp.float32)
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
            if G is None:
                G = cp.sum(distsq * scale[..., nu, nu], axis=2)
            else:
                G += cp.sum(distsq * scale[..., nu, nu], axis=2)
            del distsq
        assert(G.shape == (Nimages,Nprofiles,pH,pW))

        # FIXME -- check that this all remains float32 through the computations
        self.lanczos_shift_images_gpu(G, G, mux, muy)

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
        varrs = np.zeros((N, maxK, 3))
        for i,psfmog in enumerate(psfmogs):
            amps [i, :psfmog.K] = psfmog.amp
            assert(np.all(psfmog.mean == 0))
            varrs[i, :psfmog.K, 0] = psfmog.var[:, 0, 0]
            varrs[i, :psfmog.K, 1] = psfmog.var[:, 0, 1]
            varrs[i, :psfmog.K, 2] = psfmog.var[:, 1, 1]
        psf_mogs = amps,varrs

        imsize = psfs[0].img.shape
        for psf in psfs:
            assert(psf.sampling == 1.)

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

    def lanczos_shift_images_gpu(self, in_img, out_img, x, y, work=None):
        cp = self.cp
        '''
        Only vectorized in the specific way we need:

        in_img images:
        (Nimages x Nmodels x H x W)

        x, y:
        each of length (Nimages,)
        OR
        of size (Nimage x Nmodels)

        work:
        same shape as G; pre-allocated work array.

        out_img can be = in_img for in-place shifting.
        '''
        assert(len(in_img.shape) == 4)
        if work is not None:
            assert(work.shape == in_img.shape)
        Nim, Nmod, H, W = in_img.shape
        assert(out_img.shape == in_img.shape)
        assert(x.shape == y.shape)

        if len(x.shape) == 1:
            assert(len(x) == Nim)
            # Create Lanczos filter arrays, shape (Nim, 7)
            fx = np.arange(-3, +4)[np.newaxis, :] + x[:, np.newaxis]
            fy = np.arange(-3, +4)[np.newaxis, :] + y[:, np.newaxis]
        else:
            assert(x.shape == (Nim, Nmod))
            # Create Lanczos filter arrays, shape (Nim, Nmod, 7)
            fx = np.arange(-3, +4)[np.newaxis, np.newaxis, :] + x[..., np.newaxis]
            fy = np.arange(-3, +4)[np.newaxis, np.newaxis, :] + y[..., np.newaxis]

        fx = lanczos_filter(3, fx)
        fy = lanczos_filter(3, fy)
        self.correlate7_2d_gpu(in_img, out_img, fx, fy, work=work)
        del work

    def correlate7_2d_gpu(self, in_img, out_img, fx, fy, work=None):
        cp = self.cp
        na = cp.newaxis
        if work is None:
            work = cp.empty_like(in_img)
        else:
            ## FIXME - only really need work array to be larger; use a view
            assert(work.shape == in_img.shape)

        fx = cp.array(fx)
        fy = cp.array(fy)

        assert(len(in_img.shape) == 4)
        Nim,Nmod,H,W = in_img.shape
        assert(out_img.shape == in_img.shape)

        if len(fx.shape) == 2:
            assert(len(fx.shape) == 2)
            assert(fx.shape == fy.shape)
            Nim2,K = fx.shape
            assert(Nim2 == Nim)
            assert(K == 7)

            # Apply X filter

            # Special handling - left edge.
            work[:, :, :, 0] = cp.sum(in_img[:, :, :, :4] * fx[:, na, na, 3:], axis=-1)
            work[:, :, :, 1] = cp.sum(in_img[:, :, :, :5] * fx[:, na, na, 2:], axis=-1)
            work[:, :, :, 2] = cp.sum(in_img[:, :, :, :6] * fx[:, na, na, 1:], axis=-1)

            # Special handling - right edge.
            work[:, :, :, -1] = cp.sum(in_img[:, :, :, -4:] * fx[:, na, na, :4], axis=-1)
            work[:, :, :, -2] = cp.sum(in_img[:, :, :, -5:] * fx[:, na, na, :5], axis=-1)
            work[:, :, :, -3] = cp.sum(in_img[:, :, :, -6:] * fx[:, na, na, :6], axis=-1)

            # Middle
            work[:, :, :, 3:-3]  = in_img[:, :, :,  :-6] * fx[:, na, na, na, 0]
            work[:, :, :, 3:-3] += in_img[:, :, :, 1:-5] * fx[:, na, na, na, 1]
            work[:, :, :, 3:-3] += in_img[:, :, :, 2:-4] * fx[:, na, na, na, 2]
            work[:, :, :, 3:-3] += in_img[:, :, :, 3:-3] * fx[:, na, na, na, 3]
            work[:, :, :, 3:-3] += in_img[:, :, :, 4:-2] * fx[:, na, na, na, 4]
            work[:, :, :, 3:-3] += in_img[:, :, :, 5:-1] * fx[:, na, na, na, 5]
            work[:, :, :, 3:-3] += in_img[:, :, :, 6:  ] * fx[:, na, na, na, 6]

            # Apply Y filter

            # Special handling - bottom edge.
            out_img[:, :, 0, :] = cp.sum(work[:, :, :4, :] * fy[:, na, 3:, na], axis=-2)
            out_img[:, :, 1, :] = cp.sum(work[:, :, :5, :] * fy[:, na, 2:, na], axis=-2)
            out_img[:, :, 2, :] = cp.sum(work[:, :, :6, :] * fy[:, na, 1:, na], axis=-2)

            # Special handling - top edge.
            out_img[:, :, -1, :] = cp.sum(work[:, :, -4:, :] * fy[:, na, :4, na], axis=-2)
            out_img[:, :, -2, :] = cp.sum(work[:, :, -5:, :] * fy[:, na, :5, na], axis=-2)
            out_img[:, :, -3, :] = cp.sum(work[:, :, -6:, :] * fy[:, na, :6, na], axis=-2)

            # Middle
            out_img[:, :, 3:-3, :]  = work[:, :,  :-6, :] * fy[:, na, na, 0, na]
            out_img[:, :, 3:-3, :] += work[:, :, 1:-5, :] * fy[:, na, na, 1, na]
            out_img[:, :, 3:-3, :] += work[:, :, 2:-4, :] * fy[:, na, na, 2, na]
            out_img[:, :, 3:-3, :] += work[:, :, 3:-3, :] * fy[:, na, na, 3, na]
            out_img[:, :, 3:-3, :] += work[:, :, 4:-2, :] * fy[:, na, na, 4, na]
            out_img[:, :, 3:-3, :] += work[:, :, 5:-1, :] * fy[:, na, na, 5, na]
            out_img[:, :, 3:-3, :] += work[:, :, 6:  , :] * fy[:, na, na, 6, na]

        else:
            assert(len(fx.shape) == 3)
            assert(fx.shape == fy.shape)
            Nim2,Nmod2,K = fx.shape
            assert(Nim2 == Nim)
            assert(Nmod2 == Nmod)
            assert(K == 7)

            # Apply X filter

            # Special handling - left edge.
            work[..., :, 0] = cp.sum(in_img[..., :, :4] * fx[..., na, 3:], axis=-1)
            work[..., :, 1] = cp.sum(in_img[..., :, :5] * fx[..., na, 2:], axis=-1)
            work[..., :, 2] = cp.sum(in_img[..., :, :6] * fx[..., na, 1:], axis=-1)

            # Special handling - right edge.
            work[..., :, -1] = cp.sum(in_img[..., :, -4:] * fx[..., na, :4], axis=-1)
            work[..., :, -2] = cp.sum(in_img[..., :, -5:] * fx[..., na, :5], axis=-1)
            work[..., :, -3] = cp.sum(in_img[..., :, -6:] * fx[..., na, :6], axis=-1)

            # Middle
            work[..., :, 3:-3]  = in_img[..., :,  :-6] * fx[..., na, na, 0]
            work[..., :, 3:-3] += in_img[..., :, 1:-5] * fx[..., na, na, 1]
            work[..., :, 3:-3] += in_img[..., :, 2:-4] * fx[..., na, na, 2]
            work[..., :, 3:-3] += in_img[..., :, 3:-3] * fx[..., na, na, 3]
            work[..., :, 3:-3] += in_img[..., :, 4:-2] * fx[..., na, na, 4]
            work[..., :, 3:-3] += in_img[..., :, 5:-1] * fx[..., na, na, 5]
            work[..., :, 3:-3] += in_img[..., :, 6:  ] * fx[..., na, na, 6]

            # Apply Y filter

            # Special handling - bottom edge.
            out_img[..., 0, :] = cp.sum(work[..., :4, :] * fy[..., 3:, na], axis=-2)
            out_img[..., 1, :] = cp.sum(work[..., :5, :] * fy[..., 2:, na], axis=-2)
            out_img[..., 2, :] = cp.sum(work[..., :6, :] * fy[..., 1:, na], axis=-2)

            # Special handling - top edge.
            out_img[..., -1, :] = cp.sum(work[..., -4:, :] * fy[..., :4, na], axis=-2)
            out_img[..., -2, :] = cp.sum(work[..., -5:, :] * fy[..., :5, na], axis=-2)
            out_img[..., -3, :] = cp.sum(work[..., -6:, :] * fy[..., :6, na], axis=-2)

            # Middle
            out_img[..., 3:-3, :]  = work[...,  :-6, :] * fy[..., na, 0, na]
            out_img[..., 3:-3, :] += work[..., 1:-5, :] * fy[..., na, 1, na]
            out_img[..., 3:-3, :] += work[..., 2:-4, :] * fy[..., na, 2, na]
            out_img[..., 3:-3, :] += work[..., 3:-3, :] * fy[..., na, 3, na]
            out_img[..., 3:-3, :] += work[..., 4:-2, :] * fy[..., na, 4, na]
            out_img[..., 3:-3, :] += work[..., 5:-1, :] * fy[..., na, 5, na]
            out_img[..., 3:-3, :] += work[..., 6:  , :] * fy[..., na, 6, na]

    def gpu_one_source_models(self, tr):
        cp = self.cp
        na = cp.newaxis
        tims,masks = self._gpu_checks(tr)
        if tims is None:
            return None
        src = tr.catalog[0]
        is_galaxy = isinstance(src, ProfileGalaxy)
        is_psf = isinstance(src, PointSource)
        extents = [mm.extent for mm in masks]

        # Pixel positions (at initial params)
        pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
               for tim in tims]
        px, py = np.array(pxy, dtype=np.float32).T
        ipx = px.round().astype(np.int32)
        ipy = py.round().astype(np.int32)

        x0, x1, y0, y1 = np.asarray(extents).T
        halfsize = np.max(([(x1-x0)/2, (y1-y0)/2,
                            1+px-x0, 1+x1-px, 1+py-y0, 1+y1-py]))

        # Pre-process image: PSF, padded pix, etc
        img_params = self.gpu_setup_image_params(tims, halfsize, extents, ipx, ipy)

        p0 = tr.getParams()

        steps = [p0]

        counts_grid = np.zeros((len(tims), len(steps)), np.float32)

        # We need to transpose the mogs: need Nimages x Nprofiles(aka Nsteps)
        mogs = [[] for tim in tims]

        # idx_grid = np.zeros((len(tims), len(steps)), np.int32)
        # idy_grid = np.zeros((len(tims), len(steps)), np.int32)

        mux_grid = np.zeros((len(tims), len(steps)), np.float32)
        muy_grid = np.zeros((len(tims), len(steps)), np.float32)

        Nmods = len(steps)
        logpriors = np.zeros(Nmods, np.float32)
        for istep, p in enumerate(steps):
            tr.setParams(p)
            logpriors[istep] = tr.getLogPrior()

            # Pixel positions
            pxy = [tim.getWcs().positionToPixel(src.getPosition(), src)
                   for tim in tims]
            px, py = np.array(pxy, dtype=np.float32).T
            # Round source pixel position to nearest integer
            ix = px.round().astype(np.int32)
            iy = py.round().astype(np.int32)
            # idx_grid[:, istep] = ix - ipx
            # idy_grid[:, istep] = iy - ipy

            # subpixel portion: shifted via Lanczos interpolation
            mux = px - ix
            muy = py - iy
            assert(np.abs(mux).max() <= 0.5)
            assert(np.abs(muy).max() <= 0.5)
            mux_grid[:, istep] = mux
            muy_grid[:, istep] = muy

            counts_grid[:, istep] = np.array(
                [tim.getPhotoCal().brightnessToCounts(src.brightness)
                 for tim in tims])

            if is_galaxy:
                for itim,(tim,x,y) in enumerate(zip(tims, px, py)):
                    mogs[itim].append(src._getShearedProfile(tim,x,y))

        if is_psf:
            mods = self.gpu_get_unitflux_psf(img_params, mux_grid, muy_grid)
        else:
            mods = self.gpu_get_unitflux_galaxy_profiles(mogs, img_params,
                                                         mux_grid, muy_grid)
        mods *= counts_grid[..., na, na]

        # mods shape: Nimages x Nmod x pH x pW
        pH, pW = img_params.pH, img_params.pW
        Nimages = len(tims)
        assert(img_params.padpix.shape == (Nimages, pH, pW))
        assert(img_params.padie.shape  == (Nimages, pH, pW))

        assert(mods.shape == (Nimages,Nmods,pH,pW))

        mods = cp.get(mods[:, 0, :, :])
        patches = []
        cx,cy = img_params.cx, img_params.cy
        for i, (x0,x1,y0,y1) in enumerate(extents):
            dx = cx - ipx[i]
            dy = cy - ipy[i]
            mod = mods[i, y0+dy : y1+dy, x0+dx : x1+dx]
            patches.append(Patch(x0, y0, mod))
        return patches


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
    from tractor.galaxy import ExpGalaxy, DevGalaxy
    from tractor.sersic import SersicGalaxy, SersicIndex
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
    from tractor.basics import ConstantSurfaceBrightness
    from tractor import ParamList
    import os
    import sys
    import time
    import pylab as plt
    from astrometry.util.util import Tan

    #from tractor.utils.cupy_wrapper import cp

    t = time.time()
    t = np.fmod(t, 1.)
    t = int(t * 1e6)
    seed = t
    # 540632
    #print('Random seed', seed)
    seed = 540632
    print('Setting seed', seed)
    np.random.seed(seed)
    

    from astrometry.util.plotutils import PlotSequence
    ps = PlotSequence('chi')
    
    # import pickle
    # opt = GpuOptimizer('cupy')
    # s = pickle.dumps(opt)
    # opt2 = pickle.loads(s)
    # 
    # print('Opt:', opt, opt.cp)
    # print('Opt2:', opt2, opt2.cp)
    # 
    # opt = GpuOptimizer('numpy')
    # s = pickle.dumps(opt)
    # opt2 = pickle.loads(s)
    # 
    # print('Opt:', opt, opt.cp)
    # print('Opt2:', opt2, opt2.cp)
    # sys.exit(0)
    
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

    #brightness = NanoMaggies(g=1000., r=2000., z=500.)
    brightness = NanoMaggies(g=2000., r=4000., z=500.)

    shape = EllipseWithPriors(np.log(5.), 0.1, 0.4)
    pos = RaDecPos(ra_cen - 25.*arcsec, dec_cen)
    gpos = GaiaPosition(ra_cen - 25.*arcsec, dec_cen, 2016.0, 0., 0., 0.)
    #gal = ExpGalaxy(pos, brightness, shape)
    #gal = DevGalaxy(pos, brightness, shape)
    #param_scales = [1./3600, 1./3600., 100., 100., 100., 1., 0.1, 0.1]
    #ptsrc = PointSource(pos, brightness)
    #param_scales = [1./3600, 1./3600, 100., 100., 100]
    #gal = ExpGalaxy(gpos, brightness, shape)
    gal = DevGalaxy(gpos, brightness, shape)
    param_scales = [100., 100., 100., 1., 0.1, 0.1]

    gal = SersicGalaxy(pos, brightness, shape, SersicIndex(3.5))
    param_scales = [1./3600, 1./3600, 100., 100., 100., 1., 0.1, 0.1, 0.1]
    #print('Sersic profile:', ser.getProfile())
    
    b2 = NanoMaggies(g=20., r=40., z=60.)
    sb = ConstantSurfaceBrightness(b2)
    param_scales += [5.] * sb.numberOfParams()

    #cat = [gal]
    #cat = [ptsrc, sb]
    cat = [gal, sb]

    optargs = dict(alphas=[0.1, 0.3, 1.0], dchisq=0.1, shared_params=False, priors=True)

    true_cat = [s.copy() for s in cat]

    psf = NormalizedPixelizedPsfEx(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                        'test',
                                        'psfex-decam-00392360-S31.fits'))
    psf = HybridPixelizedPSF(psf,
                             gauss=NCircularGaussianPSF([psf.fwhm / 2.35], [1.]))
    psf = psf.constantPsfAt(w/2, h/2)

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

    print('True source parameters:')
    for src in cat:
        print('  ', src)

    ## move initial params away from truth!

    p0 = tr.getParams() + np.random.normal(size=len(param_scales)) * np.array(param_scales)
    tr.setParams(p0)

    print('Perturbed source parameters:')
    for src in cat:
        print('  ', src)

    #cp.get = lambda x: x.get()
    #gpu_opt = GpuOptimizer(cp)

    np.get = lambda x: x
    np_gpu_opt = GpuOptimizer(np)

    src = cat[0]
    mm = ModelMask(110, 10, 80, 80)
    tr.setModelMasks([dict([(s,mm) for s in cat])
                      for tim in tr.images])

    print()
    print('CPU patches...')
    patches = [src.getModelPatch(tim, modelMask=tr._getModelMaskFor(tim, src))
               for tim in tr.images]

    print()
    print('GPU patches...')
    gpu_patches = np_gpu_opt.gpu_one_source_models(tr)

    print('Patches:', patches)
    print('GPU patches:', gpu_patches)

    for p,gp in zip(patches, gpu_patches):
        plt.clf()
        plt.subplot(1,3,1)
        mx = p.patch.max()
        ima = dict(interpolation='nearest', origin='lower', vmin=0, vmax=mx)
        plt.imshow(p.patch, **ima)
        plt.title('cpu')
        plt.subplot(1,3,2)
        plt.imshow(gp.patch, **ima)
        plt.title('gpu')
        plt.subplot(1,3,3)
        scale = 0.1
        plt.imshow(gp.patch - p.patch, interpolation='nearest', origin='lower',
                   vmin=-scale*mx, vmax=+scale*mx)
        plt.title('diff')
        ps.savefig()
    
    alphas = [0.1, 0.3, 1.0]
    optargs = dict(shared_params=False, priors=True, alphas=alphas)

    orig_opt = tr.optimizer

    tr.setModelMasks(None)

    up0 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('LSQR Update:', up0)

    tr.setParams(p0)

    src = cat[0]
    mm = ModelMask(110, 10, 80, 80)
    tr.setModelMasks([dict([(s,mm) for s in cat])
                      for tim in tr.images])

    up0m = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('LSQR Update w/ modelMasks:', up0m)
    up0 = up0m

    sm_opt = SmarterDenseOptimizer()
    tr.optimizer = sm_opt

    allderivs = tr.getDerivs()
    up1,ic,colmap = tr.optimizer.getUpdateDirection(tr, allderivs, get_cov=True, **optargs)
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('Smarter update:', up1)
    n = len(up1)
    ic1 = np.zeros((n,n,), np.float32)
    for j,i in enumerate(colmap):
        ic1[i,colmap] = ic[j,:]
    ic = ic1

    tr.setParams(p0)

    #tr.optimizer = gpu_opt
    #up2 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    #print('GPU Update:', up2)

    tr.optimizer = np_gpu_opt
    up3 = tr.optimizer.getLinearUpdateDirection(tr, **optargs)
    print('GPU Update (w/numpy):', up3)

    tr.setParams(p0)

    print('Fractional difference (LSQR-SM):', difference(up0, up1))
    compare('LSQR', 'SM',  up0, up1, ic)

    #print('Fractional difference (LSQR-GPU):', difference(up0, up2))
    #print('Fractional difference (SM-GPU):', difference(up1, up2))
    print('Fractional difference (LSQR-GPU[NP]):', difference(up0, up3))
    print('Fractional difference (SM-GPU[NP])  :', difference(up1, up3))

    #compare('LSQR', 'GPU', up0, up2, ic)
    #compare('SM',   'GPU', up1, up2, ic)
    compare('LSQR', 'GPU[NP]', up0, up3, ic)
    compare('SM',   'GPU[NP]', up1, up3, ic)

    tr.setParams(p0)

    A,B,X,scales,pH,pW,padslices = np_gpu_opt.gpu_one_source_update(tr, get_A=True, **optargs)
    print('A', A.shape)
    print('B', B.shape)
    print('pH,pW', pH,pW)
    A *= scales[np.newaxis, :]
    I = np.flatnonzero(scales)
    A = A[:, I]
    X = X[I]
    scales = scales[I]
    print('scales:', scales)
    AX = A @ X
    npix,ncols = A.shape
    plt.clf()
    nims = 2
    R,C = nims*2, (ncols+3 + 1)//2
    k = 1
    for j in range(nims):
        for i in range(ncols):
            plt.subplot(R, C, k)
            k += 1
            deriv = A[j*pH*pW : (j+1)*pH*pW, i].reshape(pH,pW)
            mx = np.max(np.abs(deriv))
            plt.imshow(deriv, vmin=-mx, vmax=mx,
                       interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])

        plt.subplot(R, C, k)
        k += 1
        diff = B[j*pH*pW : (j+1)*pH*pW].reshape(pH,pW)
        mx = np.max(np.abs(diff))
        plt.imshow(diff, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('B')
        Bsub = diff
        bmx = mx

        plt.subplot(R, C, k)
        k += 1
        diff = AX[j*pH*pW : (j+1)*pH*pW].reshape(pH,pW)
        mx = np.max(np.abs(diff))
        plt.imshow(diff, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('A*X')
        AXsub = diff

        plt.subplot(R, C, k)
        k += 1
        mx = bmx
        plt.imshow(AXsub - Bsub, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('A*X - B')

        if ncols % 2 == 0:
            k += 1
    plt.savefig('a.png')

    plt.clf()
    nims = 2
    R,C = nims*2, (ncols+3 + 1)//2
    k = 1

    for j in range(nims):
        for i in range(ncols):
            plt.subplot(R, C, k)
            k += 1
            deriv = A[j*pH*pW : (j+1)*pH*pW, i].reshape(pH,pW)
            deriv = deriv[padslices[j]]
            mx = np.max(np.abs(deriv))
            plt.imshow(deriv, vmin=-mx, vmax=mx,
                       interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])

        plt.subplot(R, C, k)
        k += 1
        diff = B[j*pH*pW : (j+1)*pH*pW].reshape(pH,pW)
        diff = diff[padslices[j]]
        mx = np.max(np.abs(diff))
        plt.imshow(diff, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('B')
        Bsub = diff
        bmx = mx

        plt.subplot(R, C, k)
        k += 1
        diff = AX[j*pH*pW : (j+1)*pH*pW].reshape(pH,pW)
        diff = diff[padslices[j]]
        mx = np.max(np.abs(diff))
        plt.imshow(diff, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('A*X')
        AXsub = diff

        plt.subplot(R, C, k)
        k += 1
        mx = bmx
        plt.imshow(AXsub - Bsub, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('A*X - B')

        if ncols % 2 == 0:
            k += 1
    plt.savefig('a-pad.png')

    print('sum((0 - B)**2):', np.sum(B**2))
    print('sum((AX - B)**2):', np.sum((AX - B)**2))
    print('X:', X)

    px = tr.getParams()
    print('px', px)
    print('p0', p0)
    tr.setParams(p0)
    
    allderivs = tr.getDerivs()
    A,B,X,scales,img_layout = sm_opt.getUpdateDirection(tr, allderivs, get_A=True, **optargs)
    print('A', A.shape)
    print('B', B.shape)
    print('scales:', scales)
    A *= scales[np.newaxis, :]
    AX = A @ X
    npix,ncols = A.shape
    plt.clf()
    nims = 2
    R,C = nims*2, (ncols+3 + 1)//2
    k = 1
    for j,tim in enumerate(tr.images):
        offset,(x0,x1,y0,y1) = img_layout[tim]
        for i in range(ncols):
            plt.subplot(R, C, k)
            k += 1
            w,h = x1-x0, y1-y0
            deriv = A[offset : offset + w*h, i].reshape(h,w)
            mx = np.max(np.abs(deriv))
            plt.imshow(deriv, vmin=-mx, vmax=mx,
                       interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])

        plt.subplot(R, C, k)
        k += 1
        diff = B[offset : offset + w*h].reshape(h,w)
        mx = np.max(np.abs(diff))
        plt.imshow(diff, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('B')
        Bsub = diff
        bmx = mx

        plt.subplot(R, C, k)
        k += 1
        diff = AX[offset : offset + w*h].reshape(h,w)
        mx = np.max(np.abs(diff))
        plt.imshow(diff, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('A*X')
        AXsub = diff

        plt.subplot(R, C, k)
        k += 1
        mx = bmx
        plt.imshow(AXsub - Bsub, vmin=-mx, vmax=mx,
                   interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        plt.title('A*X - B')

        if ncols % 2 == 0:
            k += 1
    plt.savefig('a-sm.png')

    print('sum((0 - B)**2):', np.sum(B**2))
    print('sum((AX - B)**2):', np.sum((AX - B)**2))
    print('X:', X)
    
    mod1 = tr.getModelImage(0)
    mod2 = tr.getModelImage(1)
    mn,mx = np.percentile(tim1.getImage().ravel(), [2,98])
    ima = dict(interpolation='nearest', origin='lower', vmin=mn, vmax=mx)
    chima = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=+5)
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(mod1, **ima)
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.imshow(mod2, **ima)
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.imshow(tim1.getImage(), **ima)
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.imshow(tim2.getImage(), **ima)
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.imshow((tim1.getImage() - mod1) * tim1.getInvError(), **chima)
    plt.subplot(2,3,6)
    plt.imshow((tim2.getImage() - mod2) * tim2.getInvError(), **chima)
    plt.savefig('before.png')

    tr.optimizer = np_gpu_opt
    tr.setParams(p0)

    print('Starting optimize_loop with GPU[NP]')
    print('Start:', src)
    tr.optimize_loop(**optargs)
    print('Done optimize_loop with GPU[NP]')

    # dlnp, X, alpha = tr.optimize(**optargs)
    # print('Stepped alpha', alpha, 'for dlogprob', dlnp)
    # print('Step:', src)
    # p = tr.getParams()
    # tr.setParams(np.array(p) + np.array(X))
    # print('Full X step would be:', src)
    # tr.setParams(p)

    print('After optimization (GPU):', tr.catalog[0])
    for src in tr.catalog:
        print('  ', src)

    mod1 = tr.getModelImage(0)
    mod2 = tr.getModelImage(1)
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(mod1, **ima)
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.imshow(mod2, **ima)
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.imshow(tim1.getImage(), **ima)
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.imshow(tim2.getImage(), **ima)
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.imshow((tim1.getImage() - mod1) * tim1.getInvError(), **chima)
    plt.subplot(2,3,6)
    plt.imshow((tim2.getImage() - mod2) * tim2.getInvError(), **chima)
    plt.savefig('after.png')

    tr.optimizer = sm_opt
    tr.setParams(p0)

    print('Starting optimize_loop with SM')
    tr.optimize_loop(**optargs)
    print('Done optimize_loop with SM')

    # dlnp, X, alpha = tr.optimize(**optargs)
    # print('Stepped alpha', alpha, 'for dlogprob', dlnp)
    # print('Step:', src)
    # p = tr.getParams()
    # tr.setParams(np.array(p) + np.array(X))
    # print('Full X step would be:', src)
    # tr.setParams(p)

    print('After optimization (SM):', tr.catalog[0])
    for src in tr.catalog:
        print('  ', src)

    mod1 = tr.getModelImage(0)
    mod2 = tr.getModelImage(1)
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(mod1, **ima)
    plt.colorbar()
    plt.subplot(2,3,4)
    plt.imshow(mod2, **ima)
    plt.colorbar()
    plt.subplot(2,3,2)
    plt.imshow(tim1.getImage(), **ima)
    plt.colorbar()
    plt.subplot(2,3,5)
    plt.imshow(tim2.getImage(), **ima)
    plt.colorbar()
    plt.subplot(2,3,3)
    plt.imshow((tim1.getImage() - mod1) * tim1.getInvError(), **chima)
    plt.subplot(2,3,6)
    plt.imshow((tim2.getImage() - mod2) * tim2.getInvError(), **chima)
    plt.savefig('after-sm.png')
        
