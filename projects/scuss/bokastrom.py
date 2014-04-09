import numpy as np

def precess(ra, dec, epoch1, epoch2):
    if epoch1 == epoch2:
        return ra,dec
    arc=45. / np.arctan(1.)
    r2 = ra  / arc
    d2 = dec / arc
    r0 = [ np.cos(r2) * np.cos(d2),
           np.sin(r2) * np.cos(d2),
           np.sin(d2) ]
    if epoch1 != 2000.:
        p = astrox(epoch1)
        r1 = [ p[0][0] * r0[0] + p[0][1] * r0[1] + p[0][2] * r0[2],
               p[1][0] * r0[0] + p[1][1] * r0[1] + p[1][2] * r0[2],
               p[2][0] * r0[0] + p[2][1] * r0[1] + p[2][2] * r0[2],
               ]
        r0[0] = r1[0]
        r0[1] = r1[1]
        r0[2] = r1[2]

    if epoch2 != 2000.:
        p = astrox(epoch2)
        r1 = [ p[0][0] * r0[0] + p[1][0] * r0[1] + p[2][0] * r0[2],
               p[0][1] * r0[0] + p[1][1] * r0[1] + p[2][1] * r0[2],
               p[0][2] * r0[0] + p[1][2] * r0[1] + p[2][2] * r0[2],
               ]
        r0[0] = r1[0]
        r0[1] = r1[1]
        r0[2] = r1[2];

    ra = np.arctan2(r0[1], r0[0]) * arc
    dec = np.arcsin(r0[2]) * arc
    return ra,dec

def _astrox():
    pass


# cache astrom coefficients
def _bok_xytoad(self, x):
    # x: a8
    z = x[0]*x[3] - x[2]*x[1]
    b8 = [ x[3]/z,
           -x[1]/z,
           -x[2]/z,
           x[0]/z,
           (x[2]*x[5]-x[4]*x[3])/z,
           (x[4]*x[1]-x[0]*x[5])/z 
           ]
    return b8


class BokAstrom(object):
    def __init__(self, a8, epoch=2000.0):
        self.a8 = a8
        self.b8 = _bok_xytoad(self.a8)
        # Epoch of this image's WCS header
        self.epoch = epoch
        
    # bok_ad2xy.c : rad_xy    
    def radec2pixelxy(self, ra, dec, epoch=2000.):
        if epoch != self.epoch:
            ra,dec = precess(ra, dec, epoch, self.epoch)
        ac, dc = self.a8[6], self.a8[7]
        
        cx = np.pi/12.
        cy = np.pi/180.
        bok = 48.
        rar = ra  * cx
        der = dec * cy
        tmp = np.arctan(np.tan(der) / np.cos(rar-ac))
        xi = np.cos(tmp) * np.tan(rar-ac) / np.cos(tmp-dc)
        xn = np.tan(tmp-dc)
        tmp = 1. + bok*(xi*xi+xn*xn)
        xi *= tmp
        xn *= tmp

        x = self.b8[0] * xi + self.b8[2] * xn + self.b8[4]
        y = self.b8[1] * xi + self.b8[3] * xn + self.b8[5]
    
        return x,y

