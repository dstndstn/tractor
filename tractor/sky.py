from tractor.utils import BaseParams, ScalarParam
from tractor.patch import Patch
from tractor import ducks

class NullSky(BaseParams, ducks.Sky):
    '''
    A Sky implementation that does nothing; the background level is
    zero.
    '''
    pass
    
class ConstantSky(ScalarParam, ducks.ImageCalibration):
    '''
    A simple constant sky level across the whole image.

    This sky object has one parameter, the constant level.
    
    The sky level is specified in the same units as the image
    ("counts").
    '''
    def getParamDerivatives(self, tractor, img, srcs):
        import numpy as np
        p = Patch(0, 0, np.ones_like(img.getImage()))
        p.setName('dsky')
        return [p]
    def addTo(self, img, scale=1.):
        if self.val == 0:
            return
        img += (self.val * scale)
    def getParamNames(self):
        return ['sky']

    def getConstant(self):
        return self.val

    def subtract(self, con):
        self.val -= con

    def scale(self, s):
        '''
        Scales this sky model by a factor of *s*.
        '''
        self.val *= s

    def shift(self, x0,y0):
        pass

    def shifted(self, x0, y0):
        s = self.copy()
        s.shift(x0, y0)
        return s
    
    def toStandardFitsHeader(self, hdr):
        hdr.add_record(dict(name='SKY', comment='Sky value in Tractor model',
                            value=self.val))

    def write_fits(self, filename, hdr=None):
        import fitsio
        
        tt = type(self)
        sky_type = '%s.%s' % (tt.__module__, tt.__name__)
        if hdr is None:
            hdr = fitsio.FITSHDR()
        hdr.add_record(dict(name='SKY', value=sky_type,
                            comment='Sky class'))
        self.toFitsHeader(hdr, prefix='SKY_')

        fitsio.write(filename, None, header=hdr, clobber=True)

