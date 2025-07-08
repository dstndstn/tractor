from __future__ import print_function
import numpy as np

from tractor.utils import MultiParams
from tractor.patch import ModelMask
from tractor import ducks

ct1 = np.zeros(2,dtype=np.int32)

class BasicSource(ducks.Source):
    def getPosition(self):
        return self.pos

    def setPosition(self, position):
        self.pos = position


class SingleProfileSource(BasicSource):
    '''
    A mix-in class for Source objects that have a single profile, eg, PointSources,
    Dev, Exp, and Sersic galaxies, and also FixedCompositeGalaxy (surprising but true)
    but not CompositeGalaxy.
    '''

    def getBrightness(self):
        return self.brightness

    def setBrightness(self, brightness):
        self.brightness = brightness

    def getBrightnesses(self):
        return [self.getBrightness()]

    def getUnitFluxModelPatches(self, *args, **kwargs):
        return [self.getUnitFluxModelPatch(*args, **kwargs)]

    def getModelPatch(self, img, minsb=None, modelMask=None, **kwargs):
        ct1[0] += 1
        #print ("CT1:", ct1)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None
        # HACK
        if not np.isfinite(np.float32(counts)):
            return None
        if minsb is None:
            minsb = img.modelMinval
        minval = minsb / counts
        upatch = self.getUnitFluxModelPatch(img, minval=minval,
                                            modelMask=modelMask, **kwargs)

        if upatch is None:
            return None
        if upatch.patch is not None:
            assert(np.all(np.isfinite(upatch.patch)))

        # print('getModelPatch: unit-patch sum', np.sum(upatch.patch), 'counts', counts,
        #      'scaled sum:', np.sum(upatch.patch * counts))

        p = upatch * counts
        if p.patch is not None:
            assert(np.all(np.isfinite(p.patch)))
        return p


class PointSource(MultiParams, SingleProfileSource):
    '''
    An implementation of a point source, characterized by its position
    and brightness.
    '''

    def __init__(self, pos, br):
        '''
        PointSource(pos, brightness)
        '''
        super(PointSource, self).__init__(pos, br)
        # if not None, fixedRadius determines the size of unit-flux
        # model Patches produced for this PointSource.
        self.fixedRadius = None
        # if not None, minradius determines the minimum size of unit-flux
        # models
        self.minRadius = None

    def copy(self):
        c = super(PointSource, self).copy()
        c.fixedRadius = self.fixedRadius
        c.minRadius = self.minRadius
        return c

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getSourceType(self):
        return 'PointSource'

    def __str__(self):
        return (self.getSourceType() + ' at ' + str(self.pos) +
                ' with ' + str(self.brightness))

    def __repr__(self):
        return (self.getSourceType() + '(' + repr(self.pos) + ', ' +
                repr(self.brightness) + ')')

    def getUnitFluxModelPatch(self, img, minval=0., derivs=False,
                              modelMask=None, **kwargs):
        ct1[1] += 1
        #print ("CT1:", ct1)
        (px, py) = img.getWcs().positionToPixel(self.getPosition(), self)
        H, W = img.shape
        psf = self._getPsf(img)
        # quit early if the requested position is way outside the image bounds
        r = self.fixedRadius
        #print ("R1", r)
        if r is None:
            r = psf.getRadius()
            #print ("R2", r)
        #print ("PSFR", psf.getRadius())
        #print ("PX", px, py, r)
        #print (px+r, px-r, py+r, py-r)
        if px + r < 0 or px - r > W or py + r < 0 or py - r > H:
            return None

        clipExtent = None
        #print ("MM", modelMask)
        if modelMask is None:
            # max extent
            clipExtent = [0, W, 0, H]

        #print ("PSF", type(psf), psf)
        #print (psf.getPointSourcePatch)
        patch = psf.getPointSourcePatch(px, py, minval=minval,
                                        radius=self.fixedRadius,
                                        derivs=derivs,
                                        minradius=self.minRadius,
                                        modelMask=modelMask,
                                        clipExtent=clipExtent)
        return patch

    def _getPsf(self, img):
        return img.getPsf()

    def getParamDerivatives(self, img, fastPosDerivs=True, modelMask=None,
                            **kwargs):
        '''
        returns [ Patch, Patch, ... ] of length numberOfParams().
        '''
        # Short-cut the case where we're only fitting fluxes, and the
        # band of the image is not being fit.

        pos_frozen = self.isParamFrozen('pos')
        bright_frozen = self.isParamFrozen('brightness')

        counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
        if pos_frozen and not bright_frozen:
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            allzero = True
            #print (f'{bsteps=} {bvals=}')
            #print (type(bsteps), type(bvals))
            for i, bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                #print (f'{oldval=} {counts0=} {countsi=}')
                if countsi != counts0:
                    allzero = False
                    #print ("NONZERO")
                    break
            if allzero:
                return [None] * self.numberOfParams()

        pos = self.getPosition()
        wcs = img.getWcs()

        minsb = img.modelMinval
        if counts0 > 0:
            minval = minsb / counts0
        else:
            minval = None

        derivs = (not pos_frozen) and fastPosDerivs
        patchdx, patchdy = None, None
        #print ("PARAM DERIVS", minsb, minval, derivs)
        #import traceback
        #traceback.print_stack()

        if derivs:
            #print("A1")
            patches = self.getUnitFluxModelPatch(img, minval=minval,
                                                 derivs=True,
                                                 modelMask=modelMask,
                                                 **kwargs)
            if patches is None:
                #print("A2")
                return [None] * self.numberOfParams()
            if not isinstance(patches, tuple):
                #print("A3")
                patch0 = patches
            else:
                #print("A4")
                patch0, patchdx, patchdy = patches
        else:
            #print("A5")
            patch0 = self.getUnitFluxModelPatch(img, minval=minval,
                                                modelMask=modelMask,
                                                **kwargs)

        if patch0 is None:
            #print("A6")
            return [None] * self.numberOfParams()
        # check for intersection of patch0 with img
        H, W = img.shape
        if not patch0.overlapsBbox((0, W, 0, H)):
            #print("A7")
            return [None] * self.numberOfParams()

        derivs = []

        # Position
        if not pos_frozen:
            #print ("NOT FROZEN")
            if patchdx is not None and patchdy is not None:
                derivs.extend(wcs.pixelDerivsToPositionDerivs(pos, self, counts0, patch0,
                                                              patchdx, patchdy))
            elif counts0 == 0:
                derivs.extend([None] * pos.numberOfParams())
            else:
                # Step params
                psteps = pos.getStepSizes(img)
                pvals = pos.getParams()
                for i, pstep in enumerate(psteps):
                    pdiff = pstep
                    p0 = patch0
                    oldval = pos.setParam(i, pvals[i] + pstep)
                    patchx = self.getUnitFluxModelPatch(img, minval=minval,
                                                        modelMask=modelMask,
                                                        **kwargs)
                    if getattr(pos, 'symmetric_derivs', False):
                        pos.setParam(i, pvals[i] - pstep)
                        patchmx = self.getUnitFluxModelPatch(img, minval=minval,
                                                             modelMask=modelMask,
                                                             **kwargs)
                        p0 = patchmx
                        pdiff = pstep * 2

                    pos.setParam(i, oldval)
                    if patchx is None:
                        dx = patch0 * (-1 * counts0 / pstep)
                    else:
                        dx = (patchx - p0) * (counts0 / pdiff)
                    dx.setName('d(ptsrc)/d(pos%i)' % i)
                    derivs.append(dx)

        # Brightness
        if not bright_frozen:
            #print ("BF")
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            for i, bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts0) / bstep)
                df.setName('d(ptsrc)/d(bright%i)' % i)
                derivs.append(df)
        return derivs
