import numpy as np

from astrometry.util.starutil_numpy import datetomjd, J2000

from tractor.utils import ScalarParam, ArithmeticParams

class TAITime(ScalarParam, ArithmeticParams):
    '''
    This is TAI as used in the SDSS 'frame' headers; eg

    TAI     =        4507681767.55 / 1st row - 
    Number of seconds since Nov 17 1858

    And http://mirror.sdss3.org/datamodel/glossary.html#tai
    says:

    MJD = TAI/(24*3600)
    '''
    equinox = 53084.28 # mjd
    daysperyear = 365.25

    mjd2k = datetomjd(J2000)

    def __init__(self, t, mjd=None, date=None):
        if t is None:
            if date is not None:
                mjd = datetomjd(date)
            if mjd is not None:
                t = mjd * 24. * 3600.
        super(TAITime, self).__init__(t)

    def toMjd(self):
        return self.getValue() / (24.*3600.)

    def getSunTheta(self):
        mjd = self.toMjd()
        th = 2. * np.pi * (mjd - TAITime.equinox) / TAITime.daysperyear
        th = np.fmod(th, 2.*np.pi)
        return th

    def toYears(self):
        ''' years since Nov 17, 1858 ?'''
        return float(self.getValue() / (24.*3600. * TAITime.daysperyear))

    def toYear(self):
        ''' to proper year '''
        return self.toYears() - TAITime(None, mjd=TAITime.mjd2k).toYears() + 2000.0


