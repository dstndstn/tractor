

class CeresTractor(Tractor):
    def __init__(self, *args, **kwargs):
        super(CeresTractor, self).__init__(*args, **kwargs)
        self.doingForcedPhot = False
        self.BW, self.BH = 50,50
        
    def optimize_forced_photometry(self, **kwargs):
        self.doingForcedPhot = True
        super(CeresTractor, self).optimize_forced_photometry(**kwargs)
        self.doingForcedPhot = False

    def getUpdateDirection(self, *args, **kwargs):
        if not self.doingForcedPhot:
            return super(CeresTractor, self).getUpdateDirection(*args, **kwargs)
        return self.ceresGetUpdateDirection(*args, **kwargs)
        
    def ceresGetUpdateDirection(self, allderivs, damp=0., priors=True,
                                scale_columns=True, scales_only=False,
                                chiImages=None, variance=False,
                                shared_params=True, **kwargs):
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
                if img in blockstart:
                    (b0,nbw,nbh) = blockstart[img]
                else:
                    H,W = img.shape
                    nbw = int(np.ceil(W / float(BW)))
                    nbh = int(np.ceil(H / float(BH)))
                    blockstart[img] = (len(blocks), nbw, nbh)

                    for iy in range(nbh):
                        for ix in range(nbw):
                            x0 = ix * BW
                            y0 = iy * BH
                            slc = (slice(y0, min(y0+BH, H)),
                                   slice(x0, min(x0+BW, W)))
                            data = (x0, y0,
                                    img.getImage()[slc].astype(np.float64),
                                    img.getInvError()[slc].astype(np.float64))
                            blocks.append((data, []))



                    
