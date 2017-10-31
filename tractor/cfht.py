from __future__ import print_function
from .utils import BaseParams
from .brightness import LinearPhotoCal


def parse_head_file(fn):
    import fitsio
    f = open(fn, 'r')
    hdrs = []
    #s = ''
    hdr = None
    for line in f.readlines():
        #print('Read %i chars' % len(line), ': ' + line[:25]+'...')
        line = line.strip()
        #print('Stripped to %i' % len(line))
        #line = line + ' ' * (80 - len(line))
        #assert(len(line) == 80)
        #s += line
        if line.startswith('END'):
            #print('Found END')
            hdr = None
            continue
        if hdr is None:
            hdr = fitsio.FITSHDR()
            hdrs.append(hdr)
            #print('Started header number', len(hdrs))
        hdr.add_record(line)
    return hdrs


class CfhtLinearPhotoCal(LinearPhotoCal):
    def __init__(self, hdr, bandname=None, scale=1.):
        import numpy as np
        from tractor.brightness import NanoMaggies

        if hdr is not None:
            self.exptime = hdr['EXPTIME']
            self.phot_c = hdr['PHOT_C']
            self.phot_k = hdr['PHOT_K']
            self.airmass = hdr['AIRMASS']
            print('CFHT photometry:', self.exptime,
                  self.phot_c, self.phot_k, self.airmass)

            zpt = (2.5 * np.log10(self.exptime) + self.phot_c +
                   self.phot_k * (self.airmass - 1))
            print('-> zeropoint', zpt)
            scale = NanoMaggies.zeropointToScale(zpt)
            print('-> scale', scale)

        super(CfhtLinearPhotoCal, self).__init__(scale, band=bandname)

    def copy(self):
        c = self.__class__(None, bandname=self.band, scale=self.getScale())
        c.exptime = self.exptime
        c.phot_c = self.phot_c
        c.phot_k = self.phot_k
        c.airmass = self.airmass
        return c


class CfhtPhotoCal(BaseParams):
    def __init__(self, hdr=None, bandname=None):
        self.bandname = bandname
        if hdr is not None:
            self.exptime = hdr['EXPTIME']
            self.phot_c = hdr['PHOT_C']
            self.phot_k = hdr['PHOT_K']
            self.airmass = hdr['AIRMASS']
            print('CFHT photometry:', self.exptime,
                  self.phot_c, self.phot_k, self.airmass)
        # FIXME -- NO COLOR TERMS (phot_x)!
        '''
		COMMENT   Formula for Photometry, based on keywords given in this header:
		COMMENT   m = -2.5*log(DN) + 2.5*log(EXPTIME)
		COMMENT   M = m + PHOT_C + PHOT_K*(AIRMASS - 1) + PHOT_X*(PHOT_C1 - PHOT_C2)
		'''

    def hashkey(self):
        return ('CfhtPhotoCal', self.exptime, self.phot_c, self.phot_k, self.airmass)

    def copy(self):
        return CfhtPhotoCal(hdr=dict(EXPTIME=self.exptime,
                                     PHOT_C=self.phot_c,
                                     PHOT_K=self.phot_k,
                                     AIRMASS=self.airmass), bandname=self.bandname)

    def getParams(self):
        return [self.phot_c, ]

    def getStepSizes(self, *args, **kwargs):
        return [0.01]

    def setParam(self, i, p):
        assert(i == 0)
        self.phot_c = p

    def getParamNames(self):
        return ['phot_c']

    def brightnessToCounts(self, brightness):
        M = brightness.getMag(self.bandname)
        logc = (M - self.phot_c - self.phot_k * (self.airmass - 1.)) / -2.5
        return self.exptime * 10.**logc

    # def countsToBrightness(self, counts):
    #	return Mag(-2.5 * np.log10(counts / self.exptime) +
    #			   self.phot_c + self.phot_k * (self.airmass - 1.))
