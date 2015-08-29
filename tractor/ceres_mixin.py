import numpy as np

from astrometry.util.ttime import *

from engine import *

class TractorCeresMixin(object):

    def getDynamicScales(self):
        '''
        Returns parameter step sizes that will result in changes in
        chi^2 of about 1.0
        '''
        scales = np.zeros(self.numberOfParams())
        for i in range(self.getNImages()):
            derivs = self._getOneImageDerivs(i)
            for j,x0,y0,der in derivs:
                scales[j] += np.sum(der**2)
        scales = np.sqrt(scales)
        I = (scales != 0)
        if any(I):
            scales[I] = 1./scales[I]
        I = (scales == 0)
        if any(I):
            scales[I] = np.array(self.getStepSizes())[I]
        return scales
    
    def _optimize_forcedphot_core(self, result, *args, **kwargs):
        x = self._ceres_forced_photom(result, *args, **kwargs)
        result.ceres_status = x


    def _ceres_opt(self, variance=False, scale_columns=True,
                   numeric=False, scaled=True, numeric_stepsize=0.1,
                   dynamic_scale=True,
                   dlnp = 1e-3, max_iterations=0):
        from ceres import ceres_opt

        pp = self.getParams()
        if len(pp) == 0:
            return None

        if scaled:
            p0 = np.array(pp)

            if dynamic_scale:
                scales = self.getDynamicScales()
                # print 'Dynamic scales:', scales

            else:
                scales = np.array(self.getStepSizes())
            
            # Offset all the parameters so that Ceres sees them all
            # with value 1.0
            p0 -= scales
            params = np.ones_like(p0)
        
            scaler = ScaledTractor(self, p0, scales)
            tractor = scaler

        else:
            params = np.array(pp)
            tractor = self

        variance_out = None
        if variance:
            variance_out = np.zeros_like(params)

        R = ceres_opt(tractor, self.getNImages(), params, variance_out,
                      (1 if scale_columns else 0),
                      (1 if numeric else 0), numeric_stepsize,
                      dlnp, max_iterations)
        if variance:
            R['variance'] = variance_out

        if scaled:
            print 'Opt. in scaled space:', params
            self.setParams(p0 + params * scales)
            if variance:
                variance_out *= scales**2
            R['params0'] = p0
            R['scales'] = scales

        return R
        
    # This function is called-back by _ceres_opt; it is called from
    # ceres-tractor.cc via ceres.i .
    def _getOneImageDerivs(self, imgi):
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
        img = self.images[imgi]
        cat = self.catalog
        if not self.isParamFrozen('images'):
            for i in self.images.getThawedParamIndices():
                if i == imgi:
                    # Give the image a chance to compute its own derivs
                    derivs = img.getParamDerivatives(self, cat)
                    needj = []
                    for j,deriv in enumerate(derivs):
                        if deriv is None:
                            continue
                        if deriv is False:
                            needj.append(j)
                            continue
                        allderivs.append((parami + j, deriv))

                    if len(needj):
                        mod0 = self.getModelImage(i)
                        p0 = img.getParams()
                        ss = img.getStepSizes()
                    for j in needj:
                        step = ss[j]
                        img.setParam(j, p0[j]+step)
                        modj = self.getModelImage(i)
                        img.setParam(j, p0[j])
                        deriv = Patch(0, 0, (modj - mod0) / step)
                        allderivs.append((parami + j, deriv))

                parami += self.images[i].numberOfParams()

            assert(parami == self.images.numberOfParams())
            
        srcs = list(self.catalog.getThawedSources())
        for src in srcs:
            derivs = self._getSourceDerivatives(src, img)
            for j,deriv in enumerate(derivs):
                if deriv is None:
                    continue
                allderivs.append((parami + j, deriv))
            parami += src.numberOfParams()

        assert(parami == self.numberOfParams())
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
            
        return chiderivs
    
            
    def _ceres_forced_photom(self, result, umodels,
                             imlist, mods0, scales,
                             skyderivs, minFlux,
                             BW, BH,
                             ceresType = np.float32,
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
        if BW is None:
            BW = 50
        if BH is None:
            BH = 50
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

        for zi,(umods,img,scale,mod0, paramoffset) in enumerate(Z):
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
                        data = (x0, y0,
                                img.getImage()[slc].astype(ceresType),
                                mod0[slc].astype(ceresType),
                                img.getInvError()[slc].astype(ceresType))
                        blocks.append((data, []))

            for modi,umod in enumerate(umods):
                if umod is None:
                    continue
                # DEBUG
                if len(umod.shape) != 2:
                    print 'zi', zi
                    print 'modi', modi
                    print 'umod', umod
                umod.clipTo(W,H)
                umod.trimToNonZero()
                if umod.patch is None:
                    continue
                # Dice up the model
                ph,pw = umod.shape
                bx0 = np.clip(int(np.floor( umod.x0       / float(BW))),
                              0, nbw-1)
                bx1 = np.clip(int(np.ceil ((umod.x0 + pw) / float(BW))),
                              0, nbw-1)
                by0 = np.clip(int(np.floor( umod.y0       / float(BH))),
                              0, nbh-1)
                by1 = np.clip(int(np.ceil ((umod.y0 + ph) / float(BH))),
                              0, nbh-1)

                parami = paramoffset + modi
                if parami in usedParamMap:
                    ceresparam = usedParamMap[parami]
                else:
                    usedParamMap[parami] = nextparam
                    ceresparam = nextparam
                    nextparam += 1

                cmod = (umod.patch * scale).astype(ceresType)
                for by in range(by0, by1+1):
                    for bx in range(bx0, bx1+1):
                        bi = by * nbw + bx
                        #if type(umod.x0) != int or type(umod.y0) != int:
                        #    print 'umod:', umod.x0, umod.y0, type(umod.x0), type(umod.y0)
                        #    print 'umod:', umod
                        dd = (ceresparam, int(umod.x0), int(umod.y0), cmod)
                        blocks[b0 + bi][1].append(dd)
        logverb('forced phot: dicing up', Time()-t0)
                        
        rtn = []
        if wantims0:
            t0 = Time()
            params = self.getParams()
            result.ims0 = self._getims(params, imlist, umodels, mods0, scales,
                                       sky, minFlux, None)
            logverb('forced phot: ims0', Time()-t0)

        t0 = Time()
        fluxes = np.zeros(len(usedParamMap))
        logverb('Ceres forced phot:')
        logverb(len(blocks), ('image blocks (%ix%i), %i params' % (BW, BH, len(fluxes))))
        if len(blocks) == 0 or len(fluxes) == 0:
            logverb('Nothing to do!')
            return
        # init fluxes passed to ceres
        p0 = self.getParams()
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
        #print 'Ceres forced phot:', x
        logverb('forced phot: ceres', Time()-t0)

        t0 = Time()
        params = np.zeros(len(p0))
        for i,k in usedParamMap.items():
            params[i] = fluxes[k]
        self.setParams(params)
        logverb('forced phot: unmapping params:', Time()-t0)

        if wantims1:
            t0 = Time()
            result.ims1 = self._getims(params, imlist, umodels, mods0, scales,
                                       sky, minFlux, None)
            logverb('forced phot: ims1:', Time()-t0)
        return x
