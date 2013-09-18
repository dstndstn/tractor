import logging
import numpy as np
from tractor import *
from ceres import *

def logverb(*args):
    msg = ' '.join([str(x) for x in args])
    logging.debug(msg)
def logmsg(*args):
    msg = ' '.join([str(x) for x in args])
    logging.info(msg)

class CeresTractor(Tractor):
    def __init__(self, *args, **kwargs):
        super(CeresTractor, self).__init__(*args, **kwargs)
        self.BW, self.BH = 50,50
        
    def optimize_forced_photometry(self, damp=0, priors=False,
                                   minsb=0.,
                                   mindlnp=1.,
                                   rois=None,
                                   sky=False,
                                   minFlux=None,
                                   fitstats=False,
                                   justims0=False,
                                   variance=False,
                                   skyvariance=False,
                                   shared_params=True, **kwargs):
        from tractor.basics import LinearPhotoCal, ShiftedWcs

        assert(not priors)
        assert(rois is None)
        
        scales = []
        imgs = self.getImages()
        for img in imgs:
            assert(isinstance(img.getPhotoCal(), LinearPhotoCal))
            scales.append(img.getPhotoCal().getScale())

        # HACK -- if sky=True, assume we are fitting the sky in ALL images.
        # We could ask which ones are thawed...
        if sky:
            for img in imgs:
                # FIXME -- would be nice to allow multi-param linear sky models
                assert(img.getSky().numberOfParams() == 1)

        Nsourceparams = self.catalog.numberOfParams()
        
        #
        # Here we build up the "umodels" nested list, which has shape
        # (if it were a numpy array) of (len(images), len(srcs))
        # where each element is None, or a Patch with the unit-flux model
        # of that source in that image.
        #
        t0 = Time()
        umodels = []
        subimgs = []
        srcs = list(self.catalog.getThawedSources())

        umodtosource = {}
        umodsforsource = [[] for s in srcs]

        for i,img in enumerate(imgs):
            umods = []
            pcal = img.getPhotoCal()
            for si,src in enumerate(srcs):
                counts = sum([pcal.brightnessToCounts(b)
                              for b in src.getBrightnesses()])
                if counts <= 0:
                    mv = 0.
                else:
                    # we will scale the PSF by counts and we want that
                    # scaled min val to be less than minsb
                    mv = minsb / counts
                ums = src.getUnitFluxModelPatches(img, minval=mv)

                # first image only:
                if i == 0:
                    for ui in range(len(ums)):
                        umodtosource[len(umods) + ui] = si
                        umodsforsource[si].append(len(umods) + ui)

                umods.extend(ums)
            assert(len(umods) == Nsourceparams)
            umodels.append(umods)
        tmods = Time()-t0
        logverb('forced phot: getting unit-flux models:', tmods)

        imlist = imgs
        
        t0 = Time()
        fsrcs = list(self.catalog.getFrozenSources())
        mod0 = []
        for img in imlist:
            # "sky = not sky": I'm not just being contrary :) If we're
            # fitting sky, we'll do a setParams() and get the sky
            # models to render themselves when evaluating lnProbs,
            # rather than pre-computing the nominal value here and
            # then computing derivatives.
            mod0.append(self.getModelImage(img, fsrcs, minsb=minsb, sky=not sky))
        tmod = Time() - t0
        logverb('forced phot: getting frozen-source model:', tmod)

        if sky:
            t0 = Time()
            skyderivs = []
            # build the derivative list as required by getUpdateDirection:
            #    (param0) ->  [  (deriv, img), (deriv, img), ...   ], ... ],
            for img in imlist:
                dskys = img.getSky().getParamDerivatives(self, img, None)
                for dsky in dskys:
                    skyderivs.append([(dsky, img)])
            Nsky = len(skyderivs)
            assert(Nsky == self.images.numberOfParams())
            assert(Nsky + Nsourceparams == self.numberOfParams())
            logverb('forced phot: sky derivs', Time()-t0)
        else:
            Nsky = 0

        t0 = Time()
        derivs = [[] for i in range(Nsourceparams)]
        for i,(tim,umods,scale) in enumerate(zip(imlist, umodels, scales)):
            for um,dd in zip(umods, derivs):
                if um is None:
                    continue
                dd.append((um * scale, tim))
        logverb('forced phot: derivs', Time()-t0)

        t0 = Time()
        
        if sky:
            # Sky derivatives are part of the image derivatives, so go first in
            # the derivative list.
            derivs = skyderivs + derivs

        assert(len(derivs) == self.numberOfParams())
        allderivs = derivs

        ceresType = np.float32

        #
        # allderivs: [
        #    (param0:)  [  (deriv, img), (deriv, img), ... ],
        #    (param1:)  [],
        #    (param2:)  [  (deriv, img), ],
        #
        blocks = []

        usedParamMap = {}
        k = 0
        for i,derivs in enumerate(allderivs):
            if len(derivs) == 0:
                continue
            usedParamMap[i] = k
            k += 1

        blockstart = {}
        BW,BH = self.BW, self.BH
        for i,derivs in enumerate(allderivs):
            parami = usedParamMap[i]
            for deriv,img in derivs:
                H,W = img.shape
                if img in blockstart:
                    (b0,nbw,nbh) = blockstart[img]
                else:
                    # Dice up the image
                    nbw = int(np.ceil(W / float(BW)))
                    nbh = int(np.ceil(H / float(BH)))
                    b0 = len(blocks)
                    blockstart[img] = (b0, nbw, nbh)
                    for iy in range(nbh):
                        for ix in range(nbw):
                            x0 = ix * BW
                            y0 = iy * BH
                            slc = (slice(y0, min(y0+BH, H)),
                                   slice(x0, min(x0+BW, W)))
                            imi = imlist.index(img)
                            m0 = mod0[imi]
                            data = (x0, y0,
                                    img.getImage()[slc].astype(ceresType),
                                    m0[slc].astype(ceresType),
                                    img.getInvError()[slc].astype(ceresType))
                            blocks.append((data, []))

                # Dice up the deriv
                deriv.clipTo(W,H)
                if deriv.patch is None:
                    continue
                ph,pw = deriv.shape
                bx0 = np.clip(int(np.floor( deriv.x0       / float(BW))),
                              0, nbw-1)
                bx1 = np.clip(int(np.ceil ((deriv.x0 + pw) / float(BW))),
                              0, nbw-1)
                by0 = np.clip(int(np.floor( deriv.y0       / float(BH))),
                              0, nbh-1)
                by1 = np.clip(int(np.ceil ((deriv.y0 + ph) / float(BH))),
                              0, nbh-1)

                for by in range(by0, by1+1):
                    for bx in range(bx0, bx1+1):
                        bi = by * nbw + bx
                        dd = (parami, deriv.x0, deriv.y0,
                              deriv.patch.astype(ceresType))
                        blocks[b0 + bi][1].append(dd)
        logverb('forced phot: dicing up', Time()-t0)
                        
        t0 = Time()
        params = self.getParams()
        ims0 = self._getims(params, imgs, umodels, mod0, scales,
                            sky, minFlux)
        logverb('forced phot: ims0', Time()-t0)
                        
        t0 = Time()
        fluxes = np.zeros(len(usedParamMap))
        print 'Ceres forced phot:'
        print len(blocks), ('image blocks (%ix%i), %i params' %
                            (self.BW,self.BH, len(fluxes)))
        x = ceres_forced_phot(blocks, fluxes)
        print 'Fluxes:', fluxes
        logverb('forced phot: ceres', Time()-t0)

        t0 = Time()
        params = np.zeros(len(allderivs))
        for i,k in usedParamMap.items():
            params[i] = fluxes[k]

        self.setParams(params)

        ims1 = self._getims(params, imgs, umodels, mod0, scales,
                            sky, minFlux)
        imsBest = ims1
        logverb('forced phot: ims1:', Time()-t0)
        
        rtn = (ims0,imsBest)
        if variance:
            # Inverse variance
            t0 = Time()
            IV = None
            if sky and skyvariance:
                NS = Nsky
            else:
                NS = 0
            IV = np.zeros(len(srcs) + NS)
            if sky and skyvariance:
                for di,(dsky,tim) in enumerate(skyderivs):
                    ie = tim.getInvError()
                    if dsky.shape == tim.shape:
                        dchi2 = np.sum((dsky.patch * ie)**2)
                    else:
                        mm = np.zeros(tim.shape)
                        dsky.addTo(mm)
                        dchi2 = np.sum((mm * ie)**2)
                    IV[di] = dchi2
                
            for i,(tim,umods,scale) in enumerate(zip(imlist, umodels, scales)):
                mm = np.zeros(tim.shape)
                ie = tim.getInvError()
                for ui,um in enumerate(umods):
                    if um is None:
                        continue
                    um.addTo(mm)
                    x0,y0 = um.x0,um.y0
                    uh,uw = um.shape
                    slc = slice(y0, y0+uh), slice(x0,x0+uw)
                    dchi2 = np.sum((mm[slc] * scale * ie[slc]) ** 2)
                    IV[NS + ui] += dchi2
                    mm[slc] = 0.
            logverb('forced phot: invvar', Time()-t0)

            rtn = rtn + (IV,)

        if fitstats and imsBest is None:
            rtn = rtn + (None,)
        elif fitstats:
            t0 = Time()
            class FitStats(object):
                pass
            fs = FitStats()

            # Per-image stats:
            imchisq = []
            imnpix = []
            for img,mod,ie,chi,roi in imsBest:
                imchisq.append((chi**2).sum())
                imnpix.append((ie > 0).sum())
            fs.imchisq = np.array(imchisq)
            fs.imnpix = np.array(imnpix)

            # Per-source stats:
            
            # profile-weighted chi-squared (unit-model weighted chi-squared)
            fs.prochi2 = np.zeros(len(srcs))
            # profile-weighted number of pixels
            fs.pronpix = np.zeros(len(srcs))
            # profile-weighted sum of (flux from other sources / my flux)
            fs.profracflux = np.zeros(len(srcs))
            fs.proflux = np.zeros(len(srcs))
            # total number of pixels touched by this source
            fs.npix = np.zeros(len(srcs), int)

            # subtract sky from models before measuring others' flux within my profile
            skies = []
            for tim,(img,mod,ie,chi,roi) in zip(imlist, imsBest):
                tim.getSky().addTo(mod, scale=-1.)
                skies.append(tim.getSky().val)
            fs.sky = np.array(skies)

            # Some fancy footwork to convert from umods to sources
            # (eg, composite galaxies that can have multiple umods)

            # keep reusing these arrays
            srcmods = [np.zeros_like(chi) for (img,mod,ie,chi,roi) in imsBest]
            
            # for each source:
            for si,uis in enumerate(umodsforsource):
                print 'fit stats for source', si, 'of', len(umodsforsource)
                src = self.catalog[si]
                # for each image
                for X in enumerate(zip(umodels, scales, imlist, imsBest)):
                    imi,(umods,scale,tim,(img,mod,ie,chi,roi)) = X
                    # just use 'scale'?
                    pcal = tim.getPhotoCal()
                    cc = [pcal.brightnessToCounts(b) for b in src.getBrightnesses()]
                    csum = sum(cc)
                    if csum == 0:
                        continue

                    srcmod = srcmods[i]
                    xlo,xhi,ylo,yhi = None,None,None,None
                    # for each component (usually just one)
                    for ui,counts in zip(uis, cc):
                        if counts == 0:
                            continue
                        um = umods[ui]
                        if um is None:
                            continue
                        # track total ROI.
                        x0,y0 = um.x0,um.y0
                        uh,uw = um.shape
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
                    slc = slice(ylo,yhi),slice(xlo,xhi)
                    
                    srcmod[slc] /= csum

                    nz = np.flatnonzero((srcmod[slc] > 0) * (ie[slc] > 0))
                    if len(nz) == 0:
                        srcmod[slc] = 0.
                        continue

                    fs.prochi2[si] += np.sum(srcmod[slc].flat[nz] * chi[slc].flat[nz]**2)
                    fs.pronpix[si] += np.sum(srcmod[slc].flat[nz])
                    # (mod - srcmod*csum) is the model for everybody else
                    fs.profracflux[si] += np.sum(((mod[slc] / csum - srcmod[slc]) * srcmod[slc]).flat[nz])
                    # scale to nanomaggies, weight by profile
                    fs.proflux[si] += np.sum((((mod[slc] - srcmod[slc]*csum) / scale) * srcmod[slc]).flat[nz])
                    fs.npix[si] += len(nz)
                    srcmod[slc] = 0.

            # re-add sky
            for tim,(img,mod,ie,chi,roi) in zip(imlist, imsBest):
                tim.getSky().addTo(mod)

            logverb('forced phot: fit stats:', Time()-t0)

            rtn = rtn + (fs,)

        return rtn
        

    def _getims(self, params, imgs, umodels, mod0, scales, sky, minFlux):
        ims = []
        pa = params
        for i,(img,umods,m0,scale) in enumerate(zip(imgs, umodels, mod0, scales)):
            mod = m0.copy()
            if sky:
                img.getSky().addTo(mod)
            for b,um in zip(pa,umods):
                if um is None:
                    continue
                if minFlux is not None:
                    b = max(b, minFlux)
                counts = b * scale
                if counts == 0.:
                    continue
                if not np.isfinite(counts):
                    print 'Warning: counts', counts, 'b', b, 'scale', scale
                assert(np.isfinite(counts))
                assert(np.all(np.isfinite(um.patch)))
                (um * counts).addTo(mod)
            ie = img.getInvError()
            im = img.getImage()
            chi = (im - mod) * ie
            roi = None
            ims.append((im, mod, ie, chi, roi))
        return ims
                
