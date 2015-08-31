from .utils import BaseParams, ScalarParam
import ducks

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

    def toStandardFitsHeader(self, hdr):
        hdr.add_record(dict(name='SKY', comment='Sky value in Tractor model',
                            value=self.val))
