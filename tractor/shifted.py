from __future__ import print_function
from tractor.utils import BaseParams
from tractor import ducks

# class SubImage(Image):
#   def __init__(self, im, roi,
#                skyclass=SubSky,
#                psfclass=SubPsf,
#                wcsclass=SubWcs):
#       (x0,x1,y0,y1) = roi
#       slc = (slice(y0,y1), slice(x0,x1))
#       data = im.getImage[slc]
#       invvar = im.getInvvar[slc]
#       sky = skyclass(im.getSky(), roi)
#       psf = psfclass(im.getPsf(), roi)
#       wcs = wcsclass(im.getWcs(), roi)
#       pcal = im.getPhotoCal()
#       super(SubImage, self).__init__(data=data, invvar=invvar, psf=psf,
#                                      wcs=wcs, sky=sky, photocal=photocal,
#                                      name='sub'+im.name)
#
# class SubSky(object):
#   def __init__(self, sky, roi):
#       self.sky = sky
#       self.roi = roi
#
#   #def getParamDerivatives(self, img):
#   def addTo(self, mod):


class ParamsWrapper(BaseParams):
    def __init__(self, real):
        self.real = real

    def hashkey(self):
        return self.real.hashkey()

    def getLogPrior(self):
        return self.real.getLogPrior()

    def getLogPriorDerivatives(self):
        return self.real.getLogPriorDerivatives()

    def getParams(self):
        return self.real.getParams()

    def setParams(self, x):
        return self.real.setParams(x)

    def setParam(self, i, x):
        return self.real.setParam(i, x)

    def numberOfParams(self):
        return self.real.numberOfParams()

    def getParamNames(self):
        return self.real.getParamNames()

    def getStepSizes(self, *args, **kwargs):
        return self.real.getStepSizes(*args, **kwargs)


class ShiftedPsf(ParamsWrapper, ducks.ImageCalibration):
    def __init__(self, psf, x0, y0):
        super(ShiftedPsf, self).__init__(psf)
        self.psf = psf
        self.x0 = x0
        self.y0 = y0

        if hasattr(psf, 'getMixtureOfGaussians'):
            self.getMixtureOfGaussians = self._getMixtureOfGaussians

    def __str__(self):
        return ('ShiftedPsf: %i,%i + ' % (self.x0, self.y0)) + str(self.psf)

    def hashkey(self):
        return ('ShiftedPsf', self.x0, self.y0) + self.psf.hashkey()

    def getPointSourcePatch(self, px, py, extent=None, derivs=False,
                            modelMask=None, **kwargs):
        if extent is not None:
            (ex0, ex1, ey0, ey1) = extent
            extent = (ex0 + self.x0, ex1 + self.x0,
                      ey0 + self.y0, ey1 + self.y0)
        mm = None
        if modelMask is not None:
            from .patch import Patch
            mm = Patch(modelMask.x0 + self.x0, modelMask.y0 +
                       self.y0, modelMask.patch)

        p = self.psf.getPointSourcePatch(self.x0 + px, self.y0 + py,
                                         extent=extent, derivs=derivs,
                                         modelMask=mm, **kwargs)
        # Now we have to shift the patch back too
        if p is None:
            return None
        if derivs and isinstance(p, tuple):
            p, dx, dy = p
            p.x0 -= self.x0
            p.y0 -= self.y0
            dx.x0 -= self.x0
            dx.y0 -= self.y0
            dy.x0 -= self.x0
            dy.y0 -= self.y0
            return p, dx, dy
        p.x0 -= self.x0
        p.y0 -= self.y0
        return p

    def getRadius(self):
        return self.psf.getRadius()

    def _getMixtureOfGaussians(self, px=None, py=None, **kwargs):
        if px is not None:
            px = px + self.x0
        if py is not None:
            py = py + self.y0
        return self.psf.getMixtureOfGaussians(px=px, py=py, **kwargs)


class ScaledPhotoCal(ParamsWrapper, ducks.ImageCalibration):
    def __init__(self, photocal, factor):
        super(ScaledPhotoCal, self).__init__(photocal)
        self.pc = photocal
        self.factor = factor

    def hashkey(self):
        return ('ScaledPhotoCal', self.factor) + self.pc.hashkey()

    def brightnessToCounts(self, brightness):
        return self.factor * self.pc.brightnessToCounts(brightness)


class ScaledWcs(ParamsWrapper, ducks.ImageCalibration):
    def __init__(self, wcs, factor):
        super(ScaledWcs, self).__init__(wcs)
        self.factor = factor
        self.wcs = wcs

    def hashkey(self):
        return ('ScaledWcs', self.factor) + tuple(self.wcs.hashkey())

    def cdAtPixel(self, x, y):
        x, y = (x + 0.5) / self.factor - 0.5, (y + 0.5) * self.factor - 0.5
        cd = self.wcs.cdAtPixel(x, y)
        return cd / self.factor

    def positionToPixel(self, pos, src=None):
        x, y = self.wcs.positionToPixel(pos, src=src)
        # Or somethin'
        return ((x + 0.5) * self.factor - 0.5,
                (y + 0.5) * self.factor - 0.5)


class ShiftedWcs(ParamsWrapper, ducks.ImageCalibration):
    '''
    Wraps a WCS in order to use it for a subimage.
    '''

    def __init__(self, wcs, x0, y0):
        super(ShiftedWcs, self).__init__(wcs)
        self.x0 = x0
        self.y0 = y0
        self.wcs = wcs

    def toFitsHeader(self, hdr, prefix=''):
        tt = type(self.wcs)
        sub_type = '%s.%s' % (tt.__module__, tt.__name__)
        hdr.add_record(dict(name=prefix + 'SUB', value=sub_type,
                            comment='ShiftedWcs sub-type'))
        hdr.add_record(dict(name=prefix + 'X0', value=self.x0,
                            comment='ShiftedWcs x0'))
        hdr.add_record(dict(name=prefix + 'Y0', value=self.y0,
                            comment='ShiftedWcs y0'))
        print('Sub wcs:', self.wcs)
        self.wcs.toFitsHeader(hdr, prefix=prefix)

    def hashkey(self):
        return ('ShiftedWcs', self.x0, self.y0) + tuple(self.wcs.hashkey())

    def cdAtPixel(self, x, y):
        return self.wcs.cdAtPixel(x + self.x0, y + self.y0)

    def positionToPixel(self, pos, src=None):
        x, y = self.wcs.positionToPixel(pos, src=src)
        return (x - self.x0, y - self.y0)

    def pixelToPosition(self, x, y, src=None):
        pos = self.wcs.pixelToPosition(x + self.x0, y + self.y0, src=src)
        return pos
