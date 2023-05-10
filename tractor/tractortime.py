import numpy as np

from tractor.utils import ScalarParam, ArithmeticParams

class TAITimeMeta(type):
    def __getattr__(cls, name):
        if name == 'mjd2k':
            if name not in cls.__dict__:
                from astrometry.util.starutil_numpy import datetomjd, J2000
                setattr(cls, name, datetomjd(J2000))
        if not name in cls.__dict__:
            raise AttributeError()
        return cls.__dict__[name]

class TAITime(ScalarParam, ArithmeticParams, metaclass=TAITimeMeta):
    '''
    This is TAI as used in the SDSS 'frame' headers; eg

    TAI     =        4507681767.55 / 1st row - 
    Number of seconds since Nov 17 1858

    And http://mirror.sdss3.org/datamodel/glossary.html#tai
    says:

    MJD = TAI/(24*3600)
    '''
    equinox = 53084.28  # mjd of the spring equinox in 2004
    daysperyear = 365.25  # Julian years, by definition

    def __init__(self, t, mjd=None, date=None):
        if t is None:
            from astrometry.util.starutil_numpy import datetomjd
            if date is not None:
                mjd = datetomjd(date)
            if mjd is not None:
                t = mjd * 24. * 3600.
        super(TAITime, self).__init__(t)

    def toMjd(self):
        return self.getValue() / (24. * 3600.)

    def getSunTheta(self):
        mjd = self.toMjd()
        th = 2. * np.pi * (mjd - TAITime.equinox) / TAITime.daysperyear
        th = np.fmod(th, 2. * np.pi)
        return th

    def toYears(self):
        ''' years since Nov 17, 1858 ?'''
        return float(self.getValue() / (24. * 3600. * TAITime.daysperyear))

    def toYear(self):
        ''' to proper year '''
        return self.toYears() - TAITime(None, mjd=TAITime.mjd2k()).toYears() + 2000.0

if __name__ == '__main__':
    print(TAITime.mjd2k)
    print(TAITime.mjd2k)
    t = TAITime(100.)
    print(t)
    print(TAITime.equinox)
    print(TAITime.daysperyear)

    print(TAITime.xxx)
