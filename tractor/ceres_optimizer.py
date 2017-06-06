from __future__ import print_function
import numpy as np
from collections import Counter

from astrometry.util.ttime import Time

from .engine import logverb
from .optimize import Optimizer

class CeresOptimizer(Optimizer):

    def __init__(self, BW=10, BH=10):
        super(CeresOptimizer, self).__init__()
        self.BW = 10
        self.BH = 10
        self.ceresType = np.float32
        
    def getDynamicScales(self, tractor):
        '''
        Returns parameter step sizes that will result in changes in
        chi^2 of about 1.0
        '''
        scales = np.zeros(tractor.numberOfParams())
        for i in range(tractor.getNImages()):
            derivs = self._getOneImageDerivs(tractor, i)
            for j,x0,y0,der in derivs:
                scales[j] += np.sum(der**2)
        scales = np.sqrt(scales)
        I = (scales != 0)
        if any(I):
            scales[I] = 1./scales[I]
        I = (scales == 0)
        if any(I):
            scales[I] = np.array(tractor.getStepSizes())[I]
        return scales
    
    def _optimize_forcedphot_core(self, tractor, result, *args, **kwargs):
        x = self._ceres_forced_photom(tractor, result, *args, **kwargs)
        result.ceres_status = x

    def optimize(self, tractor, **kwargs):
        X = self._ceres_opt(tractor, **kwargs)
        #print('optimize:', X)
        chisq0 = X['initial_cost']
        chisq1 = X['final_cost']
        # dlnp, dparams, alpha
        return chisq0 - chisq1, None, 1

    def optimize_loop(self, tractor, **kwargs):
        X = self._ceres_opt(tractor, **kwargs)
        return X
    
    def _ceres_opt(self, tractor, variance=False, scale_columns=True,
                   numeric=False, scaled=True, numeric_stepsize=0.1,
                   dynamic_scale=True,
                   dlnp = 1e-3, max_iterations=0, print_progress=True,
                   priors=False, bounds=False, **nil):
        from ceres import ceres_opt

        pp = tractor.getParams()
        if len(pp) == 0:
            return None

        if scaled:
            p0 = np.array(pp)

            if dynamic_scale:
                scales = self.getDynamicScales(tractor)
                # print('Dynamic scales:', scales)
            else:
                scales = np.array(tractor.getStepSizes())
            
            # Offset all the parameters so that Ceres sees them all
            # with value 1.0
            p0 -= scales
            params = np.ones_like(p0)

        else:
            params = np.array(pp)
            p0 = 0
            scales = np.ones(len(pp), float)
            
        trwrapper = CeresTractorAdapter(tractor, self, p0, scales)
            
        variance_out = None
        if variance:
            variance_out = np.zeros_like(params)

        gpriors = None
        if priors:
            gpriors = tractor.getGaussianPriors()
            # print('Gaussian priors:', gpriors)

        lubounds = None
        if bounds:
            lowers = tractor.getLowerBounds()
            uppers = tractor.getUpperBounds()
            print('Lower bounds:', lowers)
            print('Upper bounds:', uppers)
            assert(len(lowers) == len(pp))
            assert(len(uppers) == len(pp))
            lubounds = ([(i,float(b),True) for i,b in enumerate(lowers)
                         if b is not None] +
                        [(i,float(b),False) for i,b in enumerate(uppers)
                         if b is not None])
            print('lubounds:', lubounds)
            
        R = ceres_opt(trwrapper, tractor.getNImages(), params, variance_out,
                      (1 if scale_columns else 0),
                      (1 if numeric else 0), numeric_stepsize,
                      dlnp, max_iterations, gpriors, lubounds,
                      print_progress)
        if variance:
            R['variance'] = variance_out

        if scaled:
            print('Opt. in scaled space:', params)
            tractor.setParams(p0 + params * scales)
            if variance:
                variance_out *= scales**2
            R['params0'] = p0
            R['scales'] = scales
        else:
            tractor.setParams(params)

        return R
        
    # This function is called-back by _ceres_opt; it is called from
    # ceres-tractor.cc via ceres.i .
    def _getOneImageDerivs(self, tractor, imgi):
        # Returns:
        #     [  (param-index, deriv_x0, deriv_y0, deriv), ... ]
        # not necessarily in order of param-index
        # Where deriv_x0, deriv_y0 are integer pixel offsets of the "deriv" image.
        #
        # NOTE, this scales the derivatives by inverse-error and -1 to
        # yield derivatives of CHI with respect to PARAMs; NOT the
        # model image wrt params.
        #
        allderivs = []

        # First, derivs for Image parameters (because 'images' comes
        # first in the tractor's parameters)
        parami = 0
        img = tractor.images[imgi]
        cat = tractor.catalog
        if not tractor.isParamFrozen('images'):
            for i in tractor.images.getThawedParamIndices():
                if i == imgi:
                    # Give the image a chance to compute its own derivs
                    derivs = img.getParamDerivatives(tractor, cat)
                    needj = []
                    for j,deriv in enumerate(derivs):
                        if deriv is None:
                            continue
                        if deriv is False:
                            needj.append(j)
                            continue
                        allderivs.append((parami + j, deriv))

                    if len(needj):
                        mod0 = tractor.getModelImage(i)
                        p0 = img.getParams()
                        ss = img.getStepSizes()
                    for j in needj:
                        step = ss[j]
                        img.setParam(j, p0[j]+step)
                        modj = tractor.getModelImage(i)
                        img.setParam(j, p0[j])
                        deriv = Patch(0, 0, (modj - mod0) / step)
                        allderivs.append((parami + j, deriv))

                parami += tractor.images[i].numberOfParams()

            assert(parami == tractor.images.numberOfParams())
            
        srcs = list(tractor.catalog.getThawedSources())
        for src in srcs:
            derivs = tractor._getSourceDerivatives(src, img)
            for j,deriv in enumerate(derivs):
                if deriv is None:
                    continue
                allderivs.append((parami + j, deriv))
            parami += src.numberOfParams()

        assert(parami == tractor.numberOfParams())
        # Clip and unpack the (x0,y0,patch) elements for ease of use from C (ceres)
        # Also scale by -1 * inverse-error to get units of dChi here.
        ie = img.getInvError()
        H,W = img.shape
        chiderivs = []
        for ind,d in allderivs:
            d.clipTo(W,H)
            if d.patch is None:
                continue
            deriv = -1. * d.patch.astype(np.float64) * ie[d.getSlice()]
            chiderivs.append((ind, d.x0, d.y0, deriv))

        # print('_getOneImageDerivs: image', tractor.images[imgi],
        #       ':', len(chiderivs))
        # for ind,x0,y0,deriv in chiderivs:
        #     print('  ', deriv.shape)
            
        return chiderivs
    
            
    def _ceres_forced_photom(self, tractor, result, umodels,
                             imlist, mods0, scales,
                             skyderivs, minFlux,
                             nonneg = False,
                             wantims0 = True,
                             wantims1 = True,
                             negfluxval = None,
                             verbose = False,
                             **kwargs
                             ):
        '''
        negfluxval: when 'nonneg' is set, the flux value to give sources that went
        negative in an unconstrained fit.
        '''
        from ceres import ceres_forced_phot

        t0 = Time()
        blocks = []
        blockstart = {}
        usedParamMap = {}
        nextparam = 0
        # umodels[ imagei, srci ] = Patch
        Nsky = 0
        Z = []
        if skyderivs is not None:
            # skyderivs = [ (param0:)[ (deriv,img), ], (param1:)[ (deriv,img), ], ...]
            # Reorg them to be in img-major order
            skymods = [ [] for im in imlist ]
            for skyd in skyderivs:
                for (deriv,img) in skyd:
                    imi = imlist.index(img)
                    skymods[imi].append(deriv)

            for mods,im,mod0 in zip(skymods, imlist, mods0):
                Z.append((mods, im, 1., mod0, Nsky))
                Nsky += len(mods)

        Z.extend(zip(umodels, imlist, scales, mods0, np.zeros(len(imlist),int)+Nsky))

        sky = (skyderivs is not None)

        umod_npix = []
        umod_sizes = []
        for zi,(umods,img,scale,mod0, paramoffset) in enumerate(Z):
            for umod in umods:
                if umod is None:
                    continue
                h,w = umod.shape
                umod_npix.append(h*w)
                umod_sizes.append(umod.shape)
        umod_npix = np.array(umod_npix)
        I = np.argsort(-umod_npix)
        #print('Largest umods:', [umod_sizes[i] for i in I[:20]])

        # umod_sizes = Counter()
        # for zi,(umods,img,scale,mod0, paramoffset) in enumerate(Z):
        #     umod_sizes.update([umod.shape for umod in umods if umod is not None])
        # print('Unit-model sizes')
        # print(umod_sizes.most_common())

            
        
        for zi,(umods,img,scale,mod0, paramoffset) in enumerate(Z):
            H,W = img.shape
            if img in blockstart:
                (b0,nbw,nbh) = blockstart[img]
            else:
                # Dice up the image
                nbw = int(np.ceil(W / float(self.BW)))
                nbh = int(np.ceil(H / float(self.BH)))
                b0 = len(blocks)
                blockstart[img] = (b0, nbw, nbh)
                for iy in range(nbh):
                    for ix in range(nbw):
                        x0 = ix * self.BW
                        y0 = iy * self.BH
                        slc = (slice(y0, min(y0+self.BH, H)),
                               slice(x0, min(x0+self.BW, W)))
                        data = (x0, y0,
                                img.getImage()[slc].astype(self.ceresType),
                                mod0[slc].astype(self.ceresType),
                                img.getInvError()[slc].astype(self.ceresType))
                        blocks.append((data, []))

            for modi,umod in enumerate(umods):
                if umod is None:
                    continue
                # DEBUG
                if len(umod.shape) != 2:
                    print('zi', zi)
                    print('modi', modi)
                    print('umod', umod)
                umod.clipTo(W,H)
                umod.trimToNonZero()
                if umod.patch is None:
                    continue
                # Dice up the model
                ph,pw = umod.shape
                bx0 = np.clip(int(np.floor( umod.x0       / float(self.BW))),
                              0, nbw-1)
                bx1 = np.clip(int(np.ceil ((umod.x0 + pw) / float(self.BW))),
                              0, nbw-1)
                by0 = np.clip(int(np.floor( umod.y0       / float(self.BH))),
                              0, nbh-1)
                by1 = np.clip(int(np.ceil ((umod.y0 + ph) / float(self.BH))),
                              0, nbh-1)

                parami = paramoffset + modi
                if parami in usedParamMap:
                    ceresparam = usedParamMap[parami]
                else:
                    usedParamMap[parami] = nextparam
                    ceresparam = nextparam
                    nextparam += 1

                cmod = (umod.patch * scale).astype(self.ceresType)
                for by in range(by0, by1+1):
                    for bx in range(bx0, bx1+1):
                        bi = by * nbw + bx
                        #if type(umod.x0) != int or type(umod.y0) != int:
                        #    print('umod:', umod.x0, umod.y0, type(umod.x0), type(umod.y0))
                        #    print('umod:', umod)
                        dd = (ceresparam, int(umod.x0), int(umod.y0), cmod)
                        blocks[b0 + bi][1].append(dd)
        logverb('forced phot: dicing up', Time()-t0)
                        
        rtn = []
        if wantims0:
            t0 = Time()
            params = tractor.getParams()
            result.ims0 = self._getims(params, imlist, umodels, mods0, scales,
                                       sky, minFlux, None)
            logverb('forced phot: ims0', Time()-t0)

        t0 = Time()
        fluxes = np.zeros(len(usedParamMap))
        logverb('Ceres forced phot:')
        logverb(len(blocks), ('image blocks (%ix%i), %i params' %
                              (self.BW, self.BH, len(fluxes))))
        if len(blocks) == 0 or len(fluxes) == 0:
            logverb('Nothing to do!')
            return
        # init fluxes passed to ceres
        p0 = tractor.getParams()
        for i,k in usedParamMap.items():
            fluxes[k] = p0[i]

        iverbose = 1 if verbose else 0
        nonneg = int(nonneg)
        if nonneg:
            # Initial run with nonneg=False, to get in the ballpark
            x = ceres_forced_phot(blocks, fluxes, 0, iverbose)
            assert(x == 0)
            logverb('forced phot: ceres initial run', Time()-t0)
            t0 = Time()
            if negfluxval is not None:
                fluxes = np.maximum(fluxes, negfluxval)

        x = ceres_forced_phot(blocks, fluxes, nonneg, iverbose)
        #print('Ceres forced phot:', x)
        logverb('forced phot: ceres', Time()-t0)

        t0 = Time()
        params = np.zeros(len(p0))
        for i,k in usedParamMap.items():
            params[i] = fluxes[k]
        tractor.setParams(params)
        logverb('forced phot: unmapping params:', Time()-t0)

        if wantims1:
            t0 = Time()
            result.ims1 = self._getims(params, imlist, umodels, mods0, scales,
                                       sky, minFlux, None)
            logverb('forced phot: ims1:', Time()-t0)
        return x

class CeresTractorAdapter(object):
    def __init__(self, tractor, ceresopt, p0, scales):
        self.tractor = tractor
        self.ceresopt = ceresopt
        self.offset = p0
        self.scale = scales

    def getImage(self, i):
        print('CeresTractorAdapter: getImage(%i)' % i)
        return self.tractor.getImage(i)

    def getChiImage(self, i):
        print('CeresTractorAdapter: getChiImage(%i)' % i)
        return self.tractor.getChiImage(i)

    def _getOneImageDerivs(self, i):
        derivs = self.ceresopt._getOneImageDerivs(self.tractor, i)
        for (ind, x0, y0, der) in derivs:
            der *= self.scale[ind]
        return derivs

    def setParams(self, p):
        print('CeresTractorAdapter: setParams:', self.offset + self.scale * p)
        return self.tractor.setParams(self.offset + self.scale * p)
