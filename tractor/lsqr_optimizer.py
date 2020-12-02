from __future__ import print_function
import numpy as np
from astrometry.util.ttime import Time
from tractor.engine import logverb, isverbose, logmsg
from tractor.optimize import Optimizer
from tractor.utils import listmax


class LsqrOptimizer(Optimizer):

    def _optimize_forcedphot_core(
            self, tractor,
            result, umodels, imlist, mod0, scales, skyderivs, minFlux,
            nonneg=None, wantims0=None, wantims1=None,
            negfluxval=None, rois=None, priors=None, sky=None,
            justims0=None, subimgs=None, damp=None, alphas=None,
            Nsky=None, mindlnp=None, shared_params=None):

        # print(len(umodels), 'umodels, lengths', [len(x) for x in umodels])
        if len(umodels) == 0:
            return
        Nsourceparams = len(umodels[0])
        imgs = tractor.images

        #t0 = Time()
        derivs = [[] for i in range(Nsourceparams)]
        for tim, umods, scale in zip(imlist, umodels, scales):
            for um, dd in zip(umods, derivs):
                if um is None:
                    continue
                dd.append((um * scale, tim))
        #logverb('forced phot: derivs', Time() - t0)
        if sky:
            # Sky derivatives are part of the image derivatives, so go
            # first in the derivative list.
            derivs = skyderivs + derivs
        assert(len(derivs) == tractor.numberOfParams())
        self._lsqr_forced_photom(
            tractor, result, derivs, mod0, imgs, umodels,
            rois, scales, priors, sky, minFlux, justims0, subimgs,
            damp, alphas, Nsky, mindlnp, shared_params)

    def _lsqr_forced_photom(self, tractor, result, derivs, mod0, imgs, umodels,
                            rois, scales,
                            priors, sky, minFlux, justims0, subimgs,
                            damp, alphas, Nsky, mindlnp, shared_params):
        # About rois and derivs: we call
        #   getUpdateDirection(derivs, ..., chiImages=[chis])
        # And this uses the "img" objects in "derivs" to decide on the region
        # that is being optimized; the number of rows = total number of pixels.
        # We have to make sure that "chiImages" matches that size.
        #
        # We shift the unit-flux models (above, um.x0 -= x0) to be
        # relative to the ROI.

        # debugging images
        ims0 = None
        imsBest = None

        lnp0 = None
        chis0 = None
        quitNow = False

        # FIXME -- this should depend on the PhotoCal scalings!
        damp0 = 1e-3
        damping = damp

        while True:
            # A flag to try again even if the lnprob got worse
            tryAgain = False

            p0 = tractor.getParams()
            if sky:
                p0sky = p0[:Nsky]
                p0 = p0[Nsky:]

            if lnp0 is None:
                #t0 = Time()
                lnp0, chis0, ims0 = self._lnp_for_update(
                    tractor,
                    mod0, imgs, umodels, None, None, p0, rois, scales,
                    None, None, priors, sky, minFlux)
                #logverb('forced phot: initial lnp = ',
                #        lnp0, 'took', Time() - t0)
                assert(np.isfinite(lnp0))

            if justims0:
                result.lnp0 = lnp0
                result.chis0 = chis0
                result.ims0 = ims0
                return

            # print('Starting opt loop with')
            # print('  p0', p0)
            # print('  lnp0', lnp0)
            # print('  chisqs', [(chi**2).sum() for chi in chis0])
            # print('chis0:', chis0)

            # Ugly: getUpdateDirection calls self.getImages(), and
            # ASSUMES they are the same as the images referred-to in
            # the "derivs", to figure out which chi image goes with
            # which image.  Temporarily set images = subimages
            if rois is not None:
                realims = tractor.images
                tractor.images = subimgs

            #logverb('forced phot: getting update with damp=', damping)
            #t0 = Time()
            X = self.getUpdateDirection(tractor, derivs, damp=damping,
                                        priors=priors,
                                        scale_columns=False, chiImages=chis0,
                                        shared_params=shared_params)
            if X is None or len(X) == 0:
                print('Error getting update direction')
                break

            #topt = Time() - t0
            #logverb('forced phot: opt:', topt)
            #print('forced phot: update', X)
            if rois is not None:
                tractor.images = realims

            # tryUpdates():
            if alphas is None:
                # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
                alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

            if sky:
                # Split the sky-update parameters from the source parameters
                Xsky = X[:Nsky]
                X = X[Nsky:]
            else:
                p0sky = Xsky = None

            # Check whether the update produces all fluxes above the
            # minimums: if so we should be able to take the step with
            # alpha=1 and quit.

            if (minFlux is None) or np.all((p0 + X) >= minFlux):
                #print('Update produces non-negative fluxes; accepting with alpha=1')
                alphas = [1.]
                quitNow = True
            else:
                print('Some too-negative fluxes requested:')
                print('Fluxes:', p0)
                print('Update:', X)
                print('Total :', p0 + X)
                print('MinFlux:', minFlux)
                if damp == 0.0:
                    damping = damp0
                    damp0 *= 10.
                    print('Setting damping to', damping)
                    if damp0 < 1e3:
                        tryAgain = True

            lnpBest = lnp0
            alphaBest = None
            chiBest = None

            for alpha in alphas:
                #t0 = Time()
                lnp, chis, ims = self._lnp_for_update(
                    tractor,
                    mod0, imgs, umodels, X, alpha, p0, rois, scales,
                    p0sky, Xsky, priors, sky, minFlux)
                logverb('Forced phot: stepped with alpha', alpha,
                        'for lnp', lnp, ', dlnp', lnp - lnp0)
                #logverb('Took', Time() - t0)
                if lnp < (lnpBest - 1.):
                    logverb('lnp', lnp, '< lnpBest-1', lnpBest - 1.)
                    break
                if not np.isfinite(lnp):
                    break
                if lnp > lnpBest:
                    alphaBest = alpha
                    lnpBest = lnp
                    chiBest = chis
                    imsBest = ims

            if alphaBest is not None:
                # Clamp fluxes up to zero
                if minFlux is not None:
                    pa = [max(minFlux, p + alphaBest * d)
                          for p, d in zip(p0, X)]
                else:
                    pa = [p + alphaBest * d for p, d in zip(p0, X)]
                tractor.catalog.setParams(pa)

                if sky:
                    tractor.images.setParams([p + alpha * d for p, d
                                              in zip(p0sky, Xsky)])

                dlogprob = lnpBest - lnp0
                alpha = alphaBest
                lnp0 = lnpBest
                chis0 = chiBest
                # print('Accepting alpha =', alpha)
                # print('new lnp0', lnp0)
                # print('new chisqs', [(chi**2).sum() for chi in chis0])
                # print('new params', self.getParams())
            else:
                dlogprob = 0.
                alpha = 0.

                # ??
                if sky:
                    # Revert -- recall that we change params while probing in
                    # lnpForUpdate()
                    tractor.images.setParams(p0sky)

            #tstep = Time() - t0
            #print('forced phot: line search:', tstep)
            #print('forced phot: alpha', alphaBest, 'for delta-lnprob', dlogprob)
            if dlogprob < mindlnp:
                if not tryAgain:
                    break

            if quitNow:
                break
        result.ims0 = ims0
        result.ims1 = imsBest

    def _lnp_for_update(self, tractor, mod0, imgs, umodels, X, alpha, p0, rois,
                        scales, p0sky, Xsky, priors, sky, minFlux):
        if X is None:
            pa = p0
        else:
            pa = [p + alpha * d for p, d in zip(p0, X)]
        if Xsky is not None:
            tractor.images.setParams(
                [p + alpha * d for p, d in zip(p0sky, Xsky)])

        lnp = 0.
        if priors:
            lnp += tractor.getLogPrior()
            if not np.isfinite(lnp):
                return lnp, None, None

        # Recall that "umodels" is a full matrix (shape (Nimage,
        # Nsrcs)) of patches, so we just go through each image,
        # ignoring None entries and building up model images from
        # the scaled unit-flux patches.

        ims = self._getims(pa, imgs, umodels, mod0, scales, sky, minFlux, rois)
        chisq = 0.
        chis = []
        for nil, nil, nil, chi, roi in ims:
            chis.append(chi)
            chisq += (chi.astype(np.float64)**2).sum()
        lnp += -0.5 * chisq
        return lnp, chis, ims

    def optimize(self, tractor, alphas=None, damp=0, priors=True,
                 scale_columns=True,
                 shared_params=True, variance=False, just_variance=False,
                 **nil):
        #logverb(tractor.getName() + ': Finding derivs...')
        #t0 = Time()
        allderivs = tractor.getDerivs()
        #tderivs = Time() - t0
        #print(Time() - t0)
        #print('allderivs:', allderivs)
        # for d in allderivs:
        #   for (p,im) in d:
        #       print('patch mean', np.mean(p.patch))
        #logverb('Finding optimal update direction...')
        #t0 = Time()
        X = self.getUpdateDirection(tractor, allderivs, damp=damp,
                                    priors=priors,
                                    scale_columns=scale_columns,
                                    shared_params=shared_params,
                                    variance=variance)
        #print('Update:', X)
        if X is None:
            # Failure
            return (0., None, 0.)
        if variance:
            if len(X) == 0:
                return 0, X, 0, None
            X, var = X
            if just_variance:
                return var
        #print(Time() - t0)
        #topt = Time() - t0
        #print('X:', X)
        if len(X) == 0:
            return 0, X, 0.
        #logverb('X: len', len(X), '; non-zero entries:', np.count_nonzero(X))
        logverb('Finding optimal step size...')
        #t0 = Time()
        (dlogprob, alpha) = self.tryUpdates(tractor, X, alphas=alphas)
        #tstep = Time() - t0
        #logverb('Finished opt2.')
        #logverb('  alpha =', alpha)
        #logverb('  Tderiv', tderivs)
        #logverb('  Topt  ', topt)
        #logverb('  Tstep ', tstep)
        # print('Stepped alpha=', alpha, 'for dlogprob', dlogprob)
        if variance:
            return dlogprob, X, alpha, var
        return dlogprob, X, alpha

    def optimize_loop(self, tractor, dchisq=0., steps=50, **kwargs):
        R = {}
        for step in range(steps):
            dlnp, X, alpha = self.optimize(tractor, **kwargs)
            # print('Opt step: dlnp', dlnp,
            #      ', '.join([str(src) for src in tractor.getCatalog()]))
            if dlnp <= dchisq:
                break
        R.update(steps=step)
        return R

    def getUpdateDirection(self, tractor, allderivs, damp=0., priors=True,
                           scale_columns=True, scales_only=False,
                           chiImages=None, variance=False,
                           shared_params=True,
                           get_A_matrix=False):
        #
        # Returns: numpy array containing update direction.
        # If *variance* is True, return    (update,variance)
        # If *get_A_matrix* is True, returns the sparse matrix of derivatives.
        # If *scale_only* is True, return column scalings
        # In cases of an empty matrix, returns the list []
        #
        # allderivs: [
        #    (param0:)  [  (deriv, img), (deriv, img), ... ],
        #    (param1:)  [],
        #    (param2:)  [  (deriv, img), ],
        # ]
        # The "img"s may repeat
        # "deriv" are Patch objects.

        # Each position in the "allderivs" array corresponds to a
        # model parameter that we are optimizing

        # We want to minimize:
        #   || chi + (d(chi)/d(params)) * dparams ||^2
        # So  b = chi
        #     A = -d(chi)/d(params)
        #     x = dparams
        #
        # chi = (data - model) / std = (data - model) * inverr
        # derivs = d(model)/d(param)
        # A matrix = -d(chi)/d(param)
        #          = + (derivs) * inverr

        # Parameters to optimize go in the columns of matrix A
        # Pixels go in the rows.

        if shared_params:
            # Find shared parameters
            p0 = tractor.getParams()
            tractor.setParams(np.arange(len(p0)))
            p1 = tractor.getParams()
            tractor.setParams(p0)
            U, I = np.unique(p1, return_inverse=True)
            logverb(len(p0), 'params;', len(U), 'unique')
            paramindexmap = I
            #print('paramindexmap:', paramindexmap)
            #print('p1:', p1)

        # Build the sparse matrix of derivatives:
        sprows = []
        spcols = []
        spvals = []

        # Keep track of row offsets for each image.
        imgoffs = {}
        nextrow = 0
        for param in allderivs:
            for deriv, img in param:
                if img in imgoffs:
                    continue
                imgoffs[img] = nextrow
                #print('Putting image', img.name, 'at row offset', nextrow)
                nextrow += img.numberOfPixels()
        Nrows = nextrow
        del nextrow
        Ncols = len(allderivs)

        if variance:
            var = np.zeros(Ncols, np.float32)

        # FIXME -- shared_params should share colscales!

        colscales = np.ones(Ncols)
        for col, param in enumerate(allderivs):
            RR = []
            VV = []
            WW = []
            for (deriv, img) in param:
                inverrs = img.getInvError()
                (H, W) = img.shape
                row0 = imgoffs[img]
                deriv.clipTo(W, H)
                pix = deriv.getPixelIndices(img)
                if len(pix) == 0:
                    #print('This param does not influence this image!')
                    continue

                assert(np.all(pix < img.numberOfPixels()))
                # (grab non-zero indices)
                dimg = deriv.getImage()
                nz = np.flatnonzero(dimg)
                #print('  source', j, 'derivative', p, 'has', len(nz), 'non-zero entries')
                if len(nz) == 0:
                    continue
                rows = row0 + pix[nz]
                #print('Adding derivative', deriv.getName(), 'for image', img.name)
                vals = dimg.flat[nz]
                w = inverrs[deriv.getSlice(img)].flat[nz]
                assert(vals.shape == w.shape)
                # if not scales_only:
                RR.append(rows)
                VV.append(vals)
                WW.append(w)

            # massage, re-scale, and clean up matrix elements
            if len(VV) == 0:
                continue
            rows = np.hstack(RR)
            VV = np.hstack(VV)
            WW = np.hstack(WW)
            vals = VV * WW

            # shouldn't be necessary since we check len(nz)>0 above
            # if len(vals) == 0:
            #   continue
            mx = np.max(np.abs(vals))
            if mx == 0:
                logmsg('mx == 0:', len(np.flatnonzero(VV)), 'of', len(VV), 'non-zero derivatives,',
                       len(np.flatnonzero(WW)), 'of', len(
                           WW), 'non-zero weights;',
                       len(np.flatnonzero(vals)), 'non-zero products')
                continue
            # MAGIC number: near-zero matrix elements -> 0
            # 'mx' is the max value in this column.
            FACTOR = 1.e-10
            I = (np.abs(vals) > (FACTOR * mx))
            rows = rows[I]
            vals = vals[I]
            # L2 norm
            scale = np.sqrt(np.dot(vals,vals))
            colscales[col] = scale
            if scales_only:
                continue

            if variance:
                var[col] = scale**2

            sprows.append(rows)
            spcols.append(col)
            if scale_columns:
                if scale == 0.:
                    spvals.append(vals)
                else:
                    spvals.append(vals / scale)
            else:
                spvals.append(vals)

        if scales_only:
            return colscales

        b = None
        if priors:
            # We don't include the priors in the "colscales"
            # computation above, mostly because the priors are
            # returned as sparse additions to the matrix, and not
            # necessarily column-oriented the way the other params
            # are.  It would be possible to make it work, but dstn is
            # not convinced it's worth the effort right now.
            X = tractor.getLogPriorDerivatives()
            if X is not None:
                rA, cA, vA, pb, mub = X
                sprows.extend([[rij + Nrows for rij in ri] for ri in rA])
                spcols.extend(cA)
                spvals.extend([vi / colscales[ci] for vi, ci in zip(vA, cA)])
                #print('Prior: adding sparse vals', [vi / colscales[ci] for vi, ci in zip(vA, cA)])
                #print(' with b', pb)
                oldnrows = Nrows
                nr = listmax(rA, -1) + 1
                Nrows += nr
                logverb('Nrows was %i, added %i rows of priors => %i' %
                        (oldnrows, nr, Nrows))
                # if len(cA) == 0:
                #     Ncols = 0
                # else:
                #     Ncols = 1 + max(cA)

                b = np.zeros(Nrows)
                b[oldnrows:] = np.hstack(pb)

        if len(spcols) == 0:
            logverb("len(spcols) == 0")
            return []

        # 'spcols' has one integer per 'sprows' block.
        # below we hstack the rows, but before doing that, remember how
        # many rows are in each chunk.
        spcols = np.array(spcols)
        nrowspercol = np.array([len(x) for x in sprows])

        if shared_params:
            # Apply shared parameter map
            #print('Before applying shared parameter map:')
            #print('spcols:', len(spcols), 'elements')
            #print('  ', len(set(spcols)), 'unique')
            spcols = paramindexmap[spcols]
            # print('After:')
            #print('spcols:', len(spcols), 'elements')
            #print('  ', len(set(spcols)), 'unique')
            Ncols = np.max(spcols) + 1
            logverb('Set Ncols=', Ncols)

        # b = chi
        #
        # FIXME -- we could be much smarter here about computing
        # just the regions we need!
        #
        if b is None:
            b = np.zeros(Nrows)

        chimap = {}
        if chiImages is not None:
            for img, chi in zip(tractor.getImages(), chiImages):
                chimap[img] = chi

        # FIXME -- could compute just chi ROIs here.

        # iterating this way avoids setting the elements more than once
        for img, row0 in imgoffs.items():
            chi = chimap.get(img, None)
            if chi is None:
                #print('computing chi image')
                chi = tractor.getChiImage(img=img)
            chi = chi.ravel()
            NP = len(chi)
            # we haven't touched these pix before
            assert(np.all(b[row0: row0 + NP] == 0))
            assert(np.all(np.isfinite(chi)))
            #print('Setting [%i:%i) from chi img' % (row0, row0+NP))
            b[row0: row0 + NP] = chi
        # Zero out unused rows -- FIXME, is this useful??
        # print('Nrows', Nrows, 'vs len(urows)', len(urows))
        # bnz = np.zeros(Nrows)
        # bnz[urows] = b[urows]
        # print('b', len(b), 'vs bnz', len(bnz))
        # b = bnz
        assert(np.all(np.isfinite(b)))

        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr

        spvals = np.hstack(spvals)
        if not np.all(np.isfinite(spvals)):
            print('Warning: infinite derivatives; bailing out')
            return None
        assert(np.all(np.isfinite(spvals)))

        sprows = np.hstack(sprows)  # hogg's lovin' hstack *again* here
        assert(len(sprows) == len(spvals))

        # For LSQR, expand 'spcols' to be the same length as 'sprows'.
        cc = np.empty(len(sprows))
        i = 0
        for c, n in zip(spcols, nrowspercol):
            cc[i: i + n] = c
            i += n
        spcols = cc
        assert(i == len(sprows))
        assert(len(sprows) == len(spcols))

        logverb('  Number of sparse matrix elements:', len(sprows))
        urows = np.unique(sprows)
        ucols = np.unique(spcols)
        logverb('  Unique rows (pixels):', len(urows))
        logverb('  Unique columns (params):', len(ucols))
        if len(urows) == 0 or len(ucols) == 0:
            return []
        logverb('  Max row:', urows[-1])
        logverb('  Max column:', ucols[-1])
        logverb('  Sparsity factor (possible elements / filled elements):',
                float(len(urows) * len(ucols)) / float(len(sprows)))

        # FIXME -- does it make LSQR faster if we remap the row and column
        # indices so that no rows/cols are empty?

        # FIXME -- we could probably construct the CSC matrix ourselves!

        # Build sparse matrix
        #A = csc_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))
        A = csr_matrix((spvals, (sprows, spcols)), shape=(Nrows, Ncols))

        if get_A_matrix:
            return A

        lsqropts = dict(show=isverbose(), damp=damp)

        # Run lsqr()
        logverb('LSQR: %i cols (%i unique), %i elements' %
                (Ncols, len(ucols), len(spvals) - 1))

        # print('A matrix:')
        # print(A.todense())
        # print('vector b:')
        # print(b)

        bail = False
        try:
            # lsqr can trigger floating-point errors
            oldsettings = np.seterr(all='print')
            (X, istop, niters, r1norm, r2norm, anorm, acond,
             arnorm, xnorm, lsqrvar) = lsqr(A, b, **lsqropts)
        except ZeroDivisionError:
            print('ZeroDivisionError caught.  Returning zero.')
            bail = True
        # finally:
        np.seterr(**oldsettings)

        del A
        del b

        if bail:
            if shared_params:
                return np.zeros(len(paramindexmap))
            return np.zeros(len(allderivs))

        # print('LSQR results:')
        # print('  istop =', istop)
        # print('  niters =', niters)
        # print('  r1norm =', r1norm)
        # print('  r2norm =', r2norm)
        # print('  anorm =', anorm)
        # print('  acord =', acond)
        # print('  arnorm =', arnorm)
        # print('  xnorm =', xnorm)
        # print('  var =', var)

        logverb('scaled  X=', X)
        X = np.array(X)

        if shared_params:
            # Unapply shared parameter map -- result is duplicated
            # result elements.
            # logverb('shared_params: before, X len', len(X), 'with',
            #         np.count_nonzero(X), 'non-zero entries')
            # logverb('paramindexmap: len', len(paramindexmap),
            #         'range', paramindexmap.min(), paramindexmap.max())
            X = X[paramindexmap]
            # logverb('shared_params: after, X len', len(X), 'with',
            #         np.count_nonzero(X), 'non-zero entries')

        if scale_columns:
            X[colscales > 0] /= colscales[colscales > 0]
        logverb('  X=', X)

        if variance:
            if shared_params:
                # Unapply shared parameter map.
                var = var[paramindexmap]

            return X, 1./np.array(var)

        return X

    # def getParameterScales(self):
    #     print(self.getName()+': Finding derivs...')
    #     allderivs = self.getDerivs()
    #     print('Finding column scales...')
    #     s = self.getUpdateDirection(allderivs, scales_only=True)
    #     return s
