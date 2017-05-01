from __future__ import print_function
import numpy as np

from .utils import BaseParams, ParamList, MultiParams, ArithmeticParams
from tractor import ducks

class NullWCS(BaseParams, ducks.WCS):
    '''
    The "identity" WCS -- useful when you are using raw pixel
    positions rather than RA,Decs.
    '''
    def __init__(self, pixscale=1., dx=0., dy=0.):
        '''
        pixscale: [arcsec/pix]
        '''
        self.pixscale = pixscale
        self.dx = dx
        self.dy = dy
    def hashkey(self):
        return ('NullWCS', self.dx, self.dy)
    def positionToPixel(self, pos, src=None):
        return pos.x + self.dx, pos.y + self.dy
    def pixelToPosition(self, x, y, src=None):
        return x - self.dx, y - self.dy
    def cdAtPixel(self, x, y):
        return np.array([[1.,0.],[0.,1.]]) * self.pixscale / 3600.
    def pixscale_at(self, x, y):
        return self.pixscale
    def shifted(self, x, y):
        return self.copy()

class WcslibWcs(BaseParams, ducks.ImageCalibration):
    '''
    A WCS implementation that wraps a FITS WCS object (with a pixel
    offset), delegating to wcslib.

    FIXME: we could use the "wcssub()" functionality to handle subimages
    rather than x0,y0.

    FIXME: we could implement anwcs_copy() using wcscopy().
    
    '''
    def __init__(self, filename, hdu=0, wcs=None):
        self.x0 = 0.
        self.y0 = 0.
        if wcs is not None:
            self.wcs = wcs
        else:
            from astrometry.util.util import anwcs
            wcs = anwcs(filename, hdu)
            self.wcs = wcs

    # pickling
    def __getstate__(self):
        #print 'pickling wcslib header...'
        s = self.wcs.getHeaderString()
        #print 'pickled wcslib header: len', len(s)
        # print 'Pickling WcslibWcs: string'
        # print '------------------------------------------------'
        # print s
        # print '------------------------------------------------'
        return (self.x0, self.y0, s)

    def __setstate__(self, state):
        (x0,y0,hdrstr) = state
        self.x0 = x0
        self.y0 = y0
        from astrometry.util.util import anwcs_from_string
        # print 'Creating WcslibWcs from header string:'
        # print '------------------------------------------------'
        # print hdrstr
        # print '------------------------------------------------'
        #print 'Unpickling: wcslib header string length:', len(hdrstr)
        self.wcs = anwcs_from_string(hdrstr)
        #print 'unpickling done'
        
    def copy(self):
        raise RuntimeError('unimplemented')

    def __str__(self):
        return ('WcslibWcs: x0,y0 %.3f,%.3f' % (self.x0,self.y0))

    def debug(self):
        from astrometry.util.util import anwcs_print_stdout
        print('WcslibWcs:')
        anwcs_print_stdout(self.wcs)

    def pixel_scale(self):
        #from astrometry.util.util import anwcs_pixel_scale
        #return anwcs_pixel_scale(self.wcs)
        cd = self.cdAtPixel(self.x0, self.y0)
        return np.sqrt(np.abs(cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0])) * 3600.

    def setX0Y0(self, x0, y0):
        '''
        Sets the pixel offset to apply to pixel coordinates before putting
        them through the wrapped WCS.  Useful when using a cropped image.
        '''
        self.x0 = x0
        self.y0 = y0

    def positionToPixel(self, pos, src=None):
        ok,x,y = self.wcs.radec2pixelxy(pos.ra, pos.dec)
        # MAGIC: 1 for FITS coords.
        return x - 1. - self.x0, y - 1. - self.y0

    def pixelToPosition(self, x, y, src=None):
        # MAGIC: 1 for FITS coords.
        ok,ra,dec = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 1. + self.y0)
        return RaDecPos(ra, dec)

    def cdAtPixel(self, x, y):
        '''
        Returns the ``CD`` matrix at the given ``x,y`` pixel position.

        (Returns the constant ``CD`` matrix elements)
        '''
        ok,ra0,dec0 = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 1. + self.y0)
        ok,ra1,dec1 = self.wcs.pixelxy2radec(x + 2. + self.x0, y + 1. + self.y0)
        ok,ra2,dec2 = self.wcs.pixelxy2radec(x + 1. + self.x0, y + 2. + self.y0)

        cosdec = np.cos(np.deg2rad(dec0))

        return np.array([[(ra1 - ra0)*cosdec, (ra2 - ra0)*cosdec],
                 [dec1 - dec0,        dec2 - dec0]])


class ConstantFitsWcs(ParamList, ducks.ImageCalibration):
    '''
    A WCS implementation that wraps a FITS WCS object (with a pixel
    offset).
    '''
    def __init__(self, wcs):
        '''
        Creates a new ``ConstantFitsWcs`` given an underlying WCS object.
        '''
        self.x0 = 0
        self.y0 = 0
        super(ConstantFitsWcs, self).__init__()
        self.wcs = wcs

        cd = self.wcs.get_cd()
        self.cd = np.array([[cd[0], cd[1]], [cd[2],cd[3]]])
        self.pixscale = self.wcs.pixel_scale()
        
    def hashkey(self):
        return (self.x0, self.y0, id(self.wcs))

    def copy(self):
        copy = ConstantFitsWcs(self.wcs)
        copy.x0 = self.x0
        copy.y0 = self.y0
        return copy

    def shifted(self, dx, dy):
        copy = self.copy()
        x,y = self.getX0Y0()
        copy.setX0Y0(x+dx, y+dy)
        return copy
        
    def __str__(self):
        from .utils import getClassName
        return ('%s: x0,y0 %.3f,%.3f, WCS ' % (getClassName(self), self.x0,self.y0)
                + str(self.wcs))

    def getX0Y0(self):
        return self.x0,self.y0

    def setX0Y0(self, x0, y0):
        '''
        Sets the pixel offset to apply to pixel coordinates before putting
        them through the wrapped WCS.  Useful when using a cropped image.
        '''
        self.x0 = x0
        self.y0 = y0

    def positionToPixel(self, pos, src=None):
        '''
        Converts an :class:`tractor.RaDecPos` to a pixel position.
        Returns: tuple of floats ``(x, y)``
        '''
        X = self.wcs.radec2pixelxy(pos.ra, pos.dec)
        # handle X = (ok,x,y) and X = (x,y) return values
        if len(X) == 3:
            ok,x,y = X
        else:
            assert(len(X) == 2)
            x,y = X
        # MAGIC: subtract 1 to convert from FITS to zero-indexed pixels.
        return x - 1 - self.x0, y - 1 - self.y0

    def pixelToPosition(self, x, y, src=None):
        '''
        Converts floats ``x``, ``y`` to a
        :class:`tractor.RaDecPos`.
        '''
        # MAGIC: add 1 to convert from zero-indexed to FITS pixels.
        r,d = self.wcs.pixelxy2radec(x + 1 + self.x0, y + 1 + self.y0)
        return RaDecPos(r,d)

    def cdAtPixel(self, x, y):
        '''
        Returns the ``CD`` matrix at the given ``x,y`` pixel position.

        (Returns the constant ``CD`` matrix elements)
        '''
        return self.cd

    def pixel_scale(self):
        return self.pixscale

    def pixscale_at(self, x, y):
        '''
        Returns the local pixel scale at the given *x*,*y* pixel coords,
        in *arcseconds* per pixel.
        '''
        return self.pixscale
    
    def toStandardFitsHeader(self, hdr):
        if self.x0 != 0 or self.y0 != 0:
            wcs = self.wcs.get_subimage(self.x0, self.y0,
                                        self.wcs.get_width()-self.x0,
                                        self.wcs.get_height()-self.y0)
        else:
            wcs = self.wcs
        wcs.add_to_header(hdr)

    
class TanWcs(ConstantFitsWcs, ducks.ImageCalibration):
    '''
    The WCS object must be an astrometry.util.util.Tan object, or a
    convincingly quacking duck.

    You can also give it a filename and HDU.
    '''

    @staticmethod
    def getNamedParams():
        return dict(crval1=0, crval2=1, crpix1=2, crpix2=3,
                    cd1_1=4, cd1_2=5, cd2_1=6, cd2_2=7)

    def __init__(self, wcs, hdu=0):
        '''
        Creates a new ``FitsWcs`` given a :class:`~astrometry.util.util.Tan`
        object.  To create one of these from a filename and FITS HDU extension,

        ::
            from astrometry.util.util import Tan

            fn = 'my-file.fits'
            ext = 0
            TanWcs(Tan(fn, ext))

        To create one from WCS parameters,

            tanwcs = Tan(crval1, crval2, crpix1, crpix2,
                cd11, cd12, cd21, cd22, imagew, imageh)
            TanWcs(tanwcs)

        '''
        if hasattr(self, 'x0'):
            print('TanWcs has an x0 attr:', self.x0)
        self.x0 = 0
        self.y0 = 0

        if isinstance(wcs, basestring):
            from astrometry.util.util import Tan
            wcs = Tan(wcs, hdu)

        super(TanWcs, self).__init__(wcs)
        # ParamList keeps its params in a list; we don't want to do that.
        del self.vals

    def copy(self):
        from astrometry.util.util import Tan
        wcs = self.__class__(Tan(self.wcs))
        wcs.setX0Y0(self.x0, self.y0)
        return wcs

    def shifted(self, dx, dy):
        copy = self.copy()
        x,y = self.getX0Y0()
        copy.setX0Y0(x+dx, y+dy)
        return copy

    # Here we ASSUME TAN WCS!
    # oi vey...
    def _setThing(self, i, val):
        w = self.wcs
        if i in [0,1]:
            crv = w.crval
            crv[i] = val
            w.set_crval(*crv)
        elif i in [2,3]:
            crp = w.crpix
            crp[i-2] = val
            w.set_crpix(*crp)
        elif i in [4,5,6,7]:
            cd = list(w.get_cd())
            cd[i-4] = val
            w.set_cd(*cd)
        elif i == 8:
            self.x0 = val
        elif i == 9:
            self.y0 = val
        else:
            raise IndexError
    def _getThing(self, i):
        w = self.wcs
        if i in [0,1]:
            crv = w.crval
            return crv[i]
        elif i in [2,3]:
            crp = w.crpix
            return crp[i-2]
        elif i in [4,5,6,7]:
            cd = w.get_cd()
            return cd[i-4]
        elif i == 8:
            return self.x0
        elif i == 9:
            return self.y0
        else:
            raise IndexError
    def _getThings(self):
        w = self.wcs
        return list(w.crval) + list(w.crpix) + list(w.cd) + [self.x0, self.y0]
    def _numberOfThings(self):
        return 10
    def getStepSizes(self, *args, **kwargs):
        pixscale = self.wcs.pixel_scale()
        # we set everything ~ 1 pixel
        dcrval = pixscale / 3600.
        # CD matrix: ~1 pixel over image size (call it 1000)
        dcd = pixscale / 3600. / 1000.
        ss = [dcrval, dcrval, 1., 1., dcd, dcd, dcd, dcd, 1., 1.]
        return list(self._getLiquidArray(ss))

class PixPos(ParamList):
    '''
    A Position implementation using pixel positions.
    '''
    @staticmethod
    def getNamedParams():
        return dict(x=0, y=1)
    def __init__(self, *args):
        super(PixPos, self).__init__(*args)
        self.stepsize = [0.1, 0.1]
    def __str__(self):
        return 'pixel (%.2f, %.2f)' % (self.x, self.y)
    def getDimension(self):
        return 2

class RaDecPos(ArithmeticParams, ParamList):
    '''
    A Position implementation using RA,Dec positions, in degrees.

    Attributes:
      * ``.ra``
      * ``.dec``
    '''
    @staticmethod
    def getName():
        return "RaDecPos"
    @staticmethod
    def getNamedParams():
        return dict(ra=0, dec=1)
    def __str__(self):
        return '%s: RA, Dec = (%.5f, %.5f)' % (self.getName(), self.ra, self.dec)
    def __init__(self, *args, **kwargs):
        super(RaDecPos,self).__init__(*args,**kwargs)
        #self.setStepSizes(1e-4)
        delta = 1e-4
        self.setStepSizes([delta / np.cos(np.deg2rad(self.dec)), delta])
    def getDimension(self):
        return 2
    #def setStepSizes(self, delta):
    #    self.stepsizes = [delta / np.cos(np.deg2rad(self.dec)),delta]

    def distanceFrom(self, pos):
        from astrometry.util.starutil_numpy import degrees_between
        return degrees_between(self.ra, self.dec, pos.ra, pos.dec)


