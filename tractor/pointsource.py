import numpy as np

from .utils import MultiParams
import ducks

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

    def getModelPatch(self, img, minsb=None, modelMask=None):
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None

        ## HACK
        if not np.isfinite(np.float32(counts)):
            return None

        if minsb is None:
            minsb = img.modelMinval
        minval = minsb / counts
        upatch = self.getUnitFluxModelPatch(img, minval=minval,
                                            modelMask=modelMask)
        if upatch is None:
            return None

        if upatch.patch is not None:
            assert(np.all(np.isfinite(upatch.patch)))

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

    def getUnitFluxModelPatch(self, img, minval=0., derivs=False, modelMask=None):
        (px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
        H,W = img.shape
        psf = self._getPsf(img)
        # quit early if the requested position is way outside the image bounds
        r = self.fixedRadius
        if r is None:
            r = psf.getRadius()
        if px + r < 0 or px - r > W or py + r < 0 or py - r > H:
            return None
        patch = psf.getPointSourcePatch(px, py, minval=minval, extent=[0,W,0,H],
                                        radius=self.fixedRadius, derivs=derivs,
                                        minradius=self.minRadius, modelMask=modelMask)
        return patch

    def _getPsf(self, img):
        return img.getPsf()

    def getParamDerivatives(self, img, fastPosDerivs=True, modelMask=None):
        '''
        returns [ Patch, Patch, ... ] of length numberOfParams().
        '''
        # Short-cut the case where we're only fitting fluxes, and the
        # band of the image is not being fit.
        counts0 = img.getPhotoCal().brightnessToCounts(self.brightness)
        if self.isParamFrozen('pos') and not self.isParamFrozen('brightness'):
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            allzero = True
            for i,bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                if countsi != counts0:
                    allzero = False
                    break
            if allzero:
                return [None]*self.numberOfParams()

        pos = self.getPosition()
        wcs = img.getWcs()

        minsb = img.modelMinval
        if counts0 > 0:
            minval = minsb / counts0
        else:
            minval = None

        derivs = (not self.isParamFrozen('pos')) and fastPosDerivs
        patchdx,patchdy = None,None
        
        if derivs:
            patches = self.getUnitFluxModelPatch(img, minval=minval, derivs=True,
                                                 modelMask=modelMask)
            #print 'minval=', minval, 'Patches:', patches
            if patches is None:
                return [None]*self.numberOfParams()
            if not isinstance(patches, tuple):
                patch0 = patches
                #print 'img:', img
                #print 'counts0:', counts0
            else:
                patch0, patchdx, patchdy = patches

        else:
            patch0 = self.getUnitFluxModelPatch(img, minval=minval, modelMask=modelMask)

        if patch0 is None:
            return [None]*self.numberOfParams()
        # check for intersection of patch0 with img
        H,W = img.shape
        if not patch0.overlapsBbox((0, W, 0, H)):
            return [None]*self.numberOfParams()
        
        derivs = []

        # Position
        if not self.isParamFrozen('pos'):

            if patchdx is not None and patchdy is not None:

                # Convert x,y derivatives to Position derivatives

                px,py = wcs.positionToPixel(pos, self)
                cd = wcs.cdAtPixel(px, py)
                cdi = np.linalg.inv(cd)
                # Get thawed Position parameter indices
                thawed = pos.getThawedParamIndices()
                for i,pname in zip(thawed, pos.getParamNames()):
                    deriv = (patchdx * cdi[0,i] + patchdy * cdi[1,i]) * counts0
                    deriv.setName('d(ptsrc)/d(pos.%s)' % pname)
                    derivs.append(deriv)

            elif counts0 == 0:
                derivs.extend([None] * pos.numberOfParams())
            else:
                psteps = pos.getStepSizes(img)
                pvals = pos.getParams()
                for i,pstep in enumerate(psteps):
                    oldval = pos.setParam(i, pvals[i] + pstep)
                    patchx = self.getUnitFluxModelPatch(img, minval=minval,
                                                        modelMask=modelMask)
                
                    pos.setParam(i, oldval)
                    if patchx is None:
                        dx = patch0 * (-1 * counts0 / pstep)
                    else:
                        dx = (patchx - patch0) * (counts0 / pstep)
                    dx.setName('d(ptsrc)/d(pos%i)' % i)
                    derivs.append(dx)

        # Brightness
        if not self.isParamFrozen('brightness'):
            bsteps = self.brightness.getStepSizes(img)
            bvals = self.brightness.getParams()
            for i,bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, bvals[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts0) / bstep)
                df.setName('d(ptsrc)/d(bright%i)' % i)
                derivs.append(df)
        return derivs
