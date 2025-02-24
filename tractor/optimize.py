from __future__ import print_function
import numpy as np
from astrometry.util.ttime import Time
from tractor.engine import logverb, OptResult, logmsg


class Optimizer(object):
    def optimize(self, tractor, alphas=None, damp=0, priors=True,
                 scale_columns=True, shared_params=True, variance=False,
                 just_variance=False):
        pass

    def optimize_loop(self, tractor, **kwargs):
        pass

    def forced_photometry(self, tractor,
                          alphas=None, damp=0, priors=False,
                          minsb=0.,
                          mindlnp=1.,
                          rois=None,
                          sky=False,
                          minFlux=None,
                          fitstats=False,
                          fitstat_extras=None,
                          justims0=False,
                          variance=False,
                          skyvariance=False,
                          shared_params=True,
                          nonneg=False,
                          nilcounts=-1e30,
                          wantims=True,
                          negfluxval=None,
                          **kwargs
                          ):
        from tractor.basics import LinearPhotoCal, ShiftedWcs

        result = OptResult()

        scales = []
        imgs = tractor.getImages()
        for img in imgs:
            assert(isinstance(img.getPhotoCal(), LinearPhotoCal))
            scales.append(img.getPhotoCal().getScale())

        if rois is not None:
            assert(len(rois) == len(imgs))

        # HACK -- if sky=True, assume we are fitting the sky in ALL images.
        # We could ask which ones are thawed...
        if sky:
            for img in imgs:
                # FIXME -- would be nice to allow multi-param linear sky models
                assert(img.getSky().numberOfParams() == 1)

        Nsourceparams = tractor.catalog.numberOfParams()
        srcs = list(tractor.catalog.getThawedSources())

        # Render unit-flux models for each source.
        #t0 = Time()
        (umodels, umodtosource, umodsforsource
         ) = self._get_umodels(tractor, srcs, imgs, minsb, rois, **kwargs)
        for umods in umodels:
            assert(len(umods) == Nsourceparams)
        #tmods = Time() - t0
        #print('forced phot: getting unit-flux models:', tmods)
        #print('Number of sources:', len(srcs))
        #print('Number of source params:', Nsourceparams)

        subimgs = []
        if rois is not None:
            for i, img in enumerate(imgs):
                from tractor.image import Image
                roi = rois[i]
                y0 = roi[0].start
                x0 = roi[1].start
                y1 = roi[0].stop
                x1 = roi[1].stop
                subimg = img.subimage(x0, x1, y0, y1)
                subimgs.append(subimg)
            imlist = subimgs
        else:
            imlist = imgs

        #t0 = Time()
        fsrcs = list(tractor.catalog.getFrozenSources())
        mod0 = []
        for img in imlist:
            # "sky = not sky": I'm not just being contrary :)
            # If we're fitting sky, we'll do a setParams() and get the
            # sky models to render themselves when evaluating lnProbs,
            # rather than pre-computing the nominal value here and
            # then computing derivatives.
            mod0.append(tractor.getModelImage(
                img, fsrcs, minsb=minsb, sky=not sky))
        #tmod = Time() - t0
        #logverb('forced phot: getting frozen-source model:', tmod)

        skyderivs = None
        if sky:
            #t0 = Time()
            # build the derivative list as required by getUpdateDirection:
            #    (param0) ->  [  (deriv, img), (deriv, img), ...   ], ... ],
            skyderivs = []
            for img in imlist:
                dskys = img.getSky().getParamDerivatives(tractor, img, None)
                for dsky in dskys:
                    skyderivs.append([(dsky, img)])
            Nsky = len(skyderivs)
            assert(Nsky == tractor.images.numberOfParams())
            assert(Nsky + Nsourceparams == tractor.numberOfParams())
            #logverb('forced phot: sky derivs', Time() - t0)
        else:
            Nsky = 0

        wantims0 = wantims1 = wantims
        if fitstats:
            wantims1 = True

        #t0 = Time()
        self._optimize_forcedphot_core(
            tractor, result, umodels, imlist, mod0, scales, skyderivs, minFlux,
            nonneg=nonneg, wantims0=wantims0, wantims1=wantims1,
            negfluxval=negfluxval, rois=rois, priors=priors, sky=sky,
            justims0=justims0, subimgs=subimgs, damp=damp, alphas=alphas,
            Nsky=Nsky, mindlnp=mindlnp, shared_params=shared_params, **kwargs)
        #print('Optimize_forcedphot_core:', Time()-t0)

        if variance:
            # Inverse variance
            #t0 = Time()
            result.IV = self._get_iv(sky, skyvariance, Nsky, skyderivs, Nsourceparams,
                                     imlist, umodels, scales)
            #logverb('forced phot: variance:', Time() - t0)

        imsBest = getattr(result, 'ims1', None)
        if fitstats and imsBest is None:
            print('Warning: fit stats not computed because imsBest is None')
            result.fitstats = None
        elif fitstats:
            #t0 = Time()
            result.fitstats = self._get_fitstats(
                tractor.catalog, imsBest, srcs, imlist, umodsforsource,
                umodels, scales, nilcounts, extras=fitstat_extras)
            #logverb('forced phot: fit stats:', Time() - t0)
        return result

    def _get_umodels(self, tractor, srcs, imgs, minsb, rois, **kwargs):
        # Here we build up the "umodels" nested list, which has shape
        # (if it were a numpy array) of (len(images), len(Nsrcparams))
        # where each element is None, or a Patch with the unit-flux
        # model of that source in that image.
        #
        # For sources that have a single brightness param, Nsrcparams
        # = Nsources.
        #
        umodels = []
        umodtosource = {}
        umodsforsource = [[] for s in srcs]

        for i, img in enumerate(imgs):
            umods = []
            pcal = img.getPhotoCal()
            nvalid = 0
            nallzero = 0
            nzero = 0
            if rois is not None:
                roi = rois[i]
                y0 = roi[0].start
                x0 = roi[1].start
            else:
                x0 = y0 = 0
            for si, src in enumerate(srcs):
                counts = sum([pcal.brightnessToCounts(b)
                              for b in src.getBrightnesses()])
                if counts <= 0:
                    mv = 1e-3
                else:
                    # we will scale the PSF by counts and we want that
                    # scaled min val to be less than minsb
                    mv = minsb / counts
                mask = tractor._getModelMaskFor(img, src)
                ums = src.getUnitFluxModelPatches(
                    img, minval=mv, modelMask=mask, **kwargs)

                isvalid = False
                isallzero = False

                for ui, um in enumerate(ums):
                    if um is None:
                        continue
                    if um.patch is None:
                        continue
                    um.x0 -= x0
                    um.y0 -= y0
                    isvalid = True
                    if um.patch.sum() == 0:
                        nzero += 1
                    else:
                        isallzero = False

                    if not np.all(np.isfinite(um.patch)):
                        print('Non-finite patch for source', src)
                        print('In image', img)
                        assert(False)

                # first image only:
                if i == 0:
                    for ui in range(len(ums)):
                        umodtosource[len(umods) + ui] = si
                        umodsforsource[si].append(len(umods) + ui)

                umods.extend(ums)

                if isvalid:
                    nvalid += 1
                    if isallzero:
                        nallzero += 1
            # print 'Img', i, 'has', nvalid, 'of', len(srcs), 'sources'
            # print '  ', nallzero, 'of which are all zero'
            # print '  ', nzero, 'components are zero'
            umodels.append(umods)
        return umodels, umodtosource, umodsforsource

    def _optimize_forcedphot_core(
            self, tractor,
            result, umodels, imlist, mod0, scales, skyderivs, minFlux,
            nonneg=None, wantims0=None, wantims1=None,
            negfluxval=None, rois=None, priors=None, sky=None,
            justims0=None, subimgs=None, damp=None, alphas=None,
            Nsky=None, mindlnp=None, shared_params=None):
        raise RuntimeError('Unimplemented')

    def _get_fitstats(self, catalog, imsBest, srcs, imlist, umodsforsource,
                      umodels, scales, nilcounts, extras=[]):
        '''
        extras: [ ('key', [eim0,eim1,eim2]), ... ]

        Extra fields to add to the "FitStats" object, populated by
        taking the profile-weighted sum of 'eim*'.  The number of
        these images *must* match the number and shape of Tractor
        images.
        '''
        if extras is None:
            extras = []

        class FitStats(object):
            pass
        fs = FitStats()

        # Per-image stats:
        imchisq = []
        imnpix = []
        for img, mod, ie, chi, roi in imsBest:
            imchisq.append((chi**2).sum())
            imnpix.append((ie > 0).sum())
        fs.imchisq = np.array(imchisq)
        fs.imnpix = np.array(imnpix)

        # Per-source stats:

        # profile-weighted chi-squared (unit-model weighted chi-squared)
        fs.prochi2 = np.zeros(len(srcs))
        # profile-weighted number of pixels: ~ 1 if all pixels are masked.
        fs.pronpix = np.zeros(len(srcs))
        fs.promasked = np.zeros(len(srcs))
        # profile-weighted sum of (flux from other sources / my flux)
        fracflux_num = np.zeros(len(srcs))
        fracflux_den = np.zeros(len(srcs))
        fs.proflux = np.zeros(len(srcs))
        # total number of pixels touched by this source
        fs.npix = np.zeros(len(srcs), int)

        fracin_num = np.zeros(len(srcs))
        fracin_den = np.zeros(len(srcs))

        for key, x in extras:
            setattr(fs, key, np.zeros(len(srcs)))

        # subtract sky from models before measuring others' flux
        # within my profile
        skies = []
        for tim, (img, mod, ie, chi, roi) in zip(imlist, imsBest):
            tim.getSky().addTo(mod, scale=-1.)
            skies.append(tim.getSky().val)
        fs.sky = np.array(skies)

        # Some fancy footwork to convert from umods to sources
        # (eg, composite galaxies that can have multiple umods)

        # keep reusing these arrays
        srcmods = [np.zeros_like(chi) for (img, mod, ie, chi, roi) in imsBest]

        # for each source:
        for si, uis in enumerate(umodsforsource):
            # print 'fit stats for source', si, 'of', len(umodsforsource)
            src = catalog[si]
            # for each image
            for imi, (umods, scale, tim, (img, mod, ie, chi, roi)) in enumerate(
                    zip(umodels, scales, imlist, imsBest)):
                # just use 'scale'?
                pcal = tim.getPhotoCal()
                cc = [pcal.brightnessToCounts(b)
                      for b in src.getBrightnesses()]
                sourcecounts = sum(cc)
                if sourcecounts == 0:
                    continue
                # Still want to measure objects with negative flux
                # if csum < nilcounts:
                #     continue

                srcmod = srcmods[imi]
                xlo, xhi, ylo, yhi = None, None, None, None
                # for each component (usually just one)
                for ui, counts in zip(uis, cc):
                    if counts == 0:
                        continue
                    um = umods[ui]
                    if um is None:
                        continue
                    # track total ROI.
                    x0, y0 = um.x0, um.y0
                    uh, uw = um.shape
                    if xlo is None or x0 < xlo:
                        xlo = x0
                    if xhi is None or x0 + uw > xhi:
                        xhi = x0 + uw
                    if ylo is None or y0 < ylo:
                        ylo = y0
                    if yhi is None or y0 + uh > yhi:
                        yhi = y0 + uh
                    # accumulate this unit-flux model into srcmod
                    (um * counts).addTo(srcmod)
                # Divide by total flux, not flux within this image; sum <= 1.
                if xlo is None or xhi is None or ylo is None or yhi is None:
                    continue
                slc = slice(ylo, yhi), slice(xlo, xhi)

                srcmod[slc] /= sourcecounts

                nz = np.flatnonzero((srcmod[slc] != 0) * (ie[slc] > 0))
                if len(nz) == 0:
                    srcmod[slc] = 0.
                    continue

                fs.prochi2[si] += np.sum(np.abs(srcmod[slc].flat[nz])
                                         * chi[slc].flat[nz]**2)
                fs.pronpix[si] += np.sum(np.abs(srcmod[slc].flat[nz]))
                fs.promasked[si] += np.sum(np.abs(srcmod[slc][ie[slc] == 0]))
                # (mod - srcmod*sourcecounts) is the model for everybody else
                fracflux_num[si] += (np.sum((np.abs(mod[slc] / sourcecounts - srcmod[slc]) * np.abs(srcmod[slc])).flat[nz])
                                     / np.sum((srcmod[slc]**2).flat[nz]))
                fracflux_den[si] += np.sum(np.abs(srcmod[slc]
                                                  ).flat[nz] / np.abs(sourcecounts))
                # scale to nanomaggies, weight by profile
                fs.proflux[si] += np.sum((np.abs((mod[slc] - srcmod[slc]
                                                  * sourcecounts) / scale) * np.abs(srcmod[slc])).flat[nz])
                fs.npix[si] += len(nz)

                fracin_num[si] += np.sum(np.abs(srcmod[slc]))
                fracin_den[si] += 1.

                for key, extraims in extras:
                    x = getattr(fs, key)
                    x[si] += np.sum(np.abs(srcmod[slc].flat[nz])
                                    * extraims[imi][slc].flat[nz])

                srcmod[slc] = 0.

        fs.profracflux = fracflux_num / np.maximum(1, fracflux_den)
        fs.fracin = fracin_num / np.maximum(1, fracin_den)

        # re-add sky
        for tim, (img, mod, ie, chi, roi) in zip(imlist, imsBest):
            tim.getSky().addTo(mod)

        return fs

    def _get_iv(self, sky, skyvariance, Nsky, skyderivs, Nsourceparams,
                imlist, umodels, scales):
        if sky and skyvariance:
            pass
        else:
            Nsky = 0

        IV = np.zeros(Nsky + Nsourceparams)
        # sky derivs first
        if sky and skyvariance:
            for di, (dsky, tim) in enumerate(skyderivs):
                ie = tim.getInvError()
                if dsky.shape == tim.shape:
                    dchi2 = np.sum((dsky.patch * ie)**2)
                else:
                    mm = np.zeros(tim.shape)
                    dsky.addTo(mm)
                    dchi2 = np.sum((mm * ie)**2)
                IV[di] = dchi2


        ## Covariance calculation - combine source models that perfectly overlap 
        ## more stable covariance calculation..
        # dimension of the covariance matrix 
        D = len(umodels[0]) 
        models_cov = np.zeros(shape=(D, imlist[0].data.shape[0], imlist[0].data.shape[1]))

        # dictionary for tracking unique, non-overlapping sources in the FIM
        uniq_source_dict = {}
        # dictionary for tracking overlapping sources
        overlap_dict = {}

        # source params next
        for i, (tim, umods, scale) in enumerate(zip(imlist, umodels, scales)):
            mm = np.zeros(tim.shape)
            ie = tim.getInvError()
            for ui, um in enumerate(umods):
                if um is None:
                    continue
                # print 'deriv: sum', um.patch.sum(), 'scale', scale, 'shape', um.shape,
                um.addTo(mm)
                x0, y0 = um.x0, um.y0
                uh, uw = um.shape
                # If the unit-flux model stars off this image (x0,y0<0),
                # trim the slice.
                # (slice indexing correctly handles when the umod extends off
                # the positive end)

                # also need to slice the unit flux model properly if they extend off the image, 
                # From Dan Masters
                if x0 < 0:
                    uw += x0
                    x0 = 0
                    um_start_x = -x0
                else: 
                    um_start_x = 0

                if y0 < 0:
                    uh += y0
                    y0 = 0
                    um_start_y = -y0
                else: 
                    um_start_y = 0

                x1 = np.min([x0+uw, tim.shape[1]])
                y1 = np.min([y0+uh, tim.shape[0]])

                slc = slice(y0, y0 + uh), slice(x0, x0 + uw)
                slc_model = slice(um_start_y, um_start_y+(y1-y0)), slice(um_start_x, um_start_x+(x1-x0))

                # dimension key for this model
                key = (x0, y0, uh, uw)

                # if this source is of the same dimension as any previous ones,
                if key in uniq_source_dict:
                    # the previous matched model index
                    prev_idx = uniq_source_dict[key][0]
                    # the difference between the two matched models
                    diff_map = abs(um.getImage() - umods[prev_idx].getImage())

                    # check if the PSF models are identical
                    if np.allclose(diff_map, 0, atol=1e-5):
                        ## no need to do anything since PSF models are identical. 
                        # # append to the previous matched model
                        # models_cov[prev_idx][slc] += um.getImage()[slc_model]
                        # # re-normalization
                        # models_cov[prev_idx] /= np.sum(models_cov[prev_idx])

                        # update the overlap dict
                        if prev_idx in overlap_dict:
                            overlap_dict[prev_idx].append(ui)
                        else:
                            # register in the overlap dict
                            overlap_dict[prev_idx] = [ui]

                    # if the PSF models are differnet, then treat them as distinct sources
                    else:
                        models_cov[ui][slc] += um.getImage()[slc_model]
                        # register as a separate source
                        uniq_source_dict[key].append(ui) 
                else:
                    # new unique source
                    models_cov[ui][slc] += um.getImage()[slc_model]
                    # register as a separate source
                    uniq_source_dict[key] = [ui]
                
        # remove duplicate sources from models_cov
        if overlap_dict:
            models_cov = np.delete(models_cov, 
                                np.concatenate(list(overlap_dict.values())), 
                                axis=0)
            # new dimension after removing duplicates
            new_D = D - len(np.concatenate(list(overlap_dict.values())))
        else:
            new_D = D # if no overlaps, keep original dimensions

        # faster implementation of the fisher information matrix ,using Einstein summation
        F = np.zeros(shape=(new_D, new_D))

        # double check dimension of models_cov for FIM calculation
        assert models_cov.shape == (new_D, imlist[0].data.shape[0], imlist[0].data.shape[1])

        # construct FIM
        F[:new_D, :new_D] = -np.einsum('ijk,ljk->il', 
                                        models_cov * ie, 
                                        models_cov * ie)

        if np.any(np.isnan(F)) or np.any(np.isinf(F)):
            raise ValueError("Fisher matrix contains NaNs or Infs.")

        # Calculate covariance matrix by inverting FIM
        try:
            C = np.linalg.inv(F)

        except np.linalg.LinAlgError as e:
            # Handle the case where F is not invertible
            C = np.nan + np.zeros_like(F)

        # might be a truncated list due to overlapping sources
        var = -np.diag(C)
        # reconstruct a variance array with the input number of sources
        var_all = np.zeros(shape=D) + np.nan
        var_all[np.concatenate(list(uniq_source_dict.values()))] = var
        # set any overlapping sources to be nan
        var_all[list(overlap_dict.keys())] = np.nan
        
        IV[Nsky:] = 1/var_all # inverse variance. 

        # remove any repetitive sources in the uniq source dict that are in the overlap dict
        uniq_source_dict = {key: [idx for idx in value if idx not in overlap_dict] for key, value in uniq_source_dict.items()}

        ## TODO: what to be returned here?
        ## to keep track of overlapping (junk) sources, would be useful to return 
        ## the unique and overlapping source dictionary 
        ## uniq_source_dict, overlap_dict... 
        return IV

    def tryUpdates(self, tractor, X, alphas=None):
        if alphas is None:
            # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        pBefore = tractor.getLogProb()
        logverb('  log-prob before:', pBefore)
        pBest = pBefore
        alphaBest = None
        p0 = tractor.getParams()
        for alpha in alphas:
            logverb('  Stepping with alpha =', alpha)
            pa = [p + alpha * d for p, d in zip(p0, X)]
            tractor.setParams(pa)
            pAfter = tractor.getLogProb()
            logverb('  Log-prob after:', pAfter)
            logverb('  delta log-prob:', pAfter - pBefore)

            #print('Step', alpha, 'p', pAfter, 'dlnp', pAfter-pBefore)
            
            if not np.isfinite(pAfter):
                logmsg('  Got bad log-prob', pAfter)
                break

            if pAfter < (pBest - 1.):
                break

            if pAfter > pBest:
                alphaBest = alpha
                pBest = pAfter

        # if alphaBest is None or alphaBest == 0:
        #     print "Warning: optimization is borking"
        #     print "Parameter direction =",X
        #     print "Parameters, step sizes, updates:"
        #     for n,p,s,x in zip(tractor.getParamNames(), tractor.getParams(), tractor.getStepSizes(), X):
        #         print n, '=', p, '  step', s, 'update', x
        if alphaBest is None:
            tractor.setParams(p0)
            return 0, 0.

        logverb('  Stepping by', alphaBest,
                'for delta-logprob', pBest - pBefore)
        pa = [p + alphaBest * d for p, d in zip(p0, X)]
        tractor.setParams(pa)
        return pBest - pBefore, alphaBest

    def _getims(self, fluxes, imgs, umodels, mod0, scales, sky, minFlux, rois):
        ims = []
        for i, (img, umods, m0, scale
                ) in enumerate(zip(imgs, umodels, mod0, scales)):
            roi = None
            if rois:
                roi = rois[i]
            mod = m0.copy()
            assert(np.all(np.isfinite(mod)))
            if sky:
                img.getSky().addTo(mod)
                assert(np.all(np.isfinite(mod)))
            for f, um in zip(fluxes, umods):
                if um is None:
                    continue
                if um.patch is None:
                    continue
                if minFlux is not None:
                    f = max(f, minFlux)
                counts = f * scale
                if counts == 0.:
                    continue
                if not np.isfinite(counts):
                    print('Warning: counts', counts, 'f', f, 'scale', scale)
                assert(np.isfinite(counts))
                assert(np.all(np.isfinite(um.patch)))
                # print 'Adding umod', um, 'with counts', counts, 'to mod', mod.shape
                (um * counts).addTo(mod)

            ie = img.getInvError()
            im = img.getImage()
            if roi is not None:
                ie = ie[roi]
                im = ie[roi]
            chi = (im - mod) * ie

            # DEBUG
            if not np.all(np.isfinite(chi)):
                print('Chi has non-finite pixels:')
                print(np.unique(chi[np.logical_not(np.isfinite(chi))]))
                print('Inv error range:', ie.min(), ie.max())
                print('All finite:', np.all(np.isfinite(ie)))
                print('Mod range:', mod.min(), mod.max())
                print('All finite:', np.all(np.isfinite(mod)))
                print('Img range:', im.min(), im.max())
                print('All finite:', np.all(np.isfinite(im)))
            assert(np.all(np.isfinite(chi)))
            ims.append((im, mod, ie, chi, roi))
        return ims
