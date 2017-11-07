from __future__ import print_function


class TractorMultiprocMixin(object):

    def __init__(self, mp=None, **kwargs):
        super(TractorMultiprocMixin, self).__init__(**kwargs)
        if mp is None:
            from astrometry.util.multiproc import multiproc
            mp = multiproc()

    def _map(self, func, iterable):
        return self.mp.map(func, iterable)

    def _map_async(self, func, iterable):
        return self.mp.map_async(func, iterable)

    def getModelImages(self):
        # avoid shipping my images...
        allimages = self.getImages()
        self.images = Images()
        args = [(self, im) for im in allimages]
        # print 'Calling _map:', getmodelimagefunc2
        # print 'args:', args
        mods = self._map(getmodelimagefunc2, args)
        self.images = allimages

    def getDerivs(self):
        allderivs = []

        if self.isParamFrozen('catalog'):
            srcs = []
        else:
            srcs = list(self.catalog.getThawedSources())

        allsrcs = self.catalog

        # First, derivs for Image parameters (because 'images'
        # comes first in the tractor's parameters)
        if self.isParamFrozen('images'):
            ims = []
            imjs = []
        else:
            imjs = [i for i in self.images.getThawedParamIndices()]
            ims = [self.images[j] for j in imjs]
        imderivs = self._map(getimagederivs, [(imj, im, self, allsrcs)
                                              for im, imj in zip(ims, imjs)])

        needimjs = []
        needims = []
        needparams = []

        for derivs, im, imj in zip(imderivs, ims, imjs):
            need = []
            for k, d in enumerate(derivs):
                if d is False:
                    need.append(k)
            if len(need):
                needimjs.append(imj)
                needims.append(im)
                needparams.append(need)

        # initial models...
        logverb('Getting', len(needimjs),
                'initial models for image derivatives')
        mod0s = self._map_async(
            getmodelimagefunc, [(self, imj) for imj in needimjs])
        # stepping each (needed) param...
        args = []
        # for j,im in enumerate(ims):
        #   p0 = im.getParams()
        #   #print 'Image', im
        #   #print 'Step sizes:', im.getStepSizes()
        #   #print 'p0:', p0
        #   for k,step in enumerate(im.getStepSizes()):
        #       args.append((self, j, k, p0[k], step))
        for im, imj, params in zip(needims, needimjs, needparams):
            p0 = im.getParams()
            ss = im.getStepSizes()
            for i in params:
                args.append((self, imj, i, p0[i], ss[i]))
        # reverse the args so we can pop() below.
        logverb('Stepping in', len(args), 'model parameters for derivatives')
        mod1s = self._map_async(getmodelimagestep, reversed(args))

        # Next, derivs for the sources.
        args = []
        for j, src in enumerate(srcs):
            for i, img in enumerate(self.images):
                args.append((src, img))

        # if modelMasks are set, need to send those across the multiprocessing...
        assert(self.modelMasks is None)
        sderivs = self._map_async(getsrcderivs, reversed(args))

        # Wait for and unpack the image derivatives...
        mod0s = mod0s.get()
        mod1s = mod1s.get()
        # convert to a imj->mod0 map
        assert(len(mod0s) == len(needimjs))
        mod0s = dict(zip(needimjs, mod0s))

        for derivs, im, imj in zip(imderivs, ims, imjs):
            for k, d in enumerate(derivs):
                if d is False:
                    mod0 = mod0s[imj]
                    nm = im.getParamNames()[k]
                    step = im.getStepSizes()[k]
                    mod1 = mod1s.pop()
                    d = Patch(0, 0, (mod1 - mod0) / step)
                    d.name = 'd(im%i)/d(%s)' % (j, nm)
                allderivs.append([(d, im)])
        # popped all
        assert(len(mod1s) == 0)

        # for i,(j,im) in enumerate(zip(imjs,ims)):
        #   mod0 = mod0s[i]
        #   p0 = im.getParams()
        #   for k,(nm,step) in enumerate(zip(im.getParamNames(), im.getStepSizes())):
        #       mod1 = mod1s.pop()
        #       deriv = Patch(0, 0, (mod1 - mod0) / step)
        #       deriv.name = 'd(im%i)/d(%s)' % (j,nm)
        #       allderivs.append([(deriv, im)])

        # Wait for source derivs...
        sderivs = sderivs.get()

        for j, src in enumerate(srcs):
            srcderivs = [[] for i in range(src.numberOfParams())]
            for i, img in enumerate(self.images):
                # Get derivatives (in this image) of params
                derivs = sderivs.pop()
                # derivs is a list of Patch objects or None, one per parameter.
                assert(len(derivs) == src.numberOfParams())
                for k, deriv in enumerate(derivs):
                    if deriv is None:
                        continue
                    # if deriv is False:
                    #     # compute it now
                    #   (THIS IS WRONG; mod0s only has initial models for images
                    #    that need it -- those that are unfrozen)
                    #     mod0 = mod0s[i]
                    #     nm = src.getParamNames()[k]
                    #     step = src.getStepSizes()[k]
                    #     mod1 = tra.getModelImage(img)
                    #     d = Patch(0, 0, (mod1 - mod0) / step)
                    #     d.name = 'd(src(im%i))/d(%s)' % (i, nm)
                    #     deriv = d

                    if not np.all(np.isfinite(deriv.patch.ravel())):
                        print('Derivative for source', src)
                        print('deriv index', i)
                        assert(False)
                    srcderivs[k].append((deriv, img))
            allderivs.extend(srcderivs)

        assert(len(allderivs) == self.numberOfParams())
        return allderivs


# These are free functions for multiprocessing in "getderivs2()"
def getmodelimagestep(X):
    (tr, j, k, p0, step) = X
    im = tr.getImage(j)
    im.setParam(k, p0 + step)
    mod = tr.getModelImage(im)
    im.setParam(k, p0)
    return mod


def getmodelimagefunc2(X):
    (tr, im) = X
    try:
        return tr.getModelImage(im)
    except:
        import traceback
        print('Exception in getmodelimagefun2:')
        traceback.print_exc()
        raise


def getimagederivs(X):
    (imj, img, tractor, srcs) = X
    # FIXME -- avoid shipping all images...
    return img.getParamDerivatives(tractor, srcs)


def getsrcderivs(X):
    (src, img) = X
    return src.getParamDerivatives(img)


def getmodelimagefunc(X):
    (tr, imj) = X
    return tr.getModelImage(imj)
