"""
This file is part of the Tractor project.
Copyright 2011, 2012 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`basics.py`
===========

Generally useful generic implementations of things the Tractor needs:
Magnitudes, (RA,Dec) positions, FITS WCS, and so on.

"""

import numpy as np

from .image import Image
from .patch import Patch
from .utils import *
from . import mixture_profiles as mp

from .tractortime import TAITime
from .psf import PixelizedPSF, GaussianMixturePSF, GaussianMixtureEllipsePSF, NCircularGaussianPSF
from .wcs import NullWCS, WcslibWcs, ConstantFitsWcs, TanWcs, PixPos, RaDecPos
from .sky import NullSky, ConstantSky
from .brightness import Mag, Flux, Mags, Fluxes, NanoMaggies, FluxesPhotoCal, MagsPhotoCal, NullPhotoCal, LinearPhotoCal
from .pointsource import BasicSource, SingleProfileSource, PointSource
from .shifted import ParamsWrapper, ShiftedPsf, ScaledPhotoCal, ScaledWcs, ShiftedWcs

class TractorWCSWrapper(object):
    '''
    Wraps a Tractor WCS object to look like an
    astrometry.util.util.Tan/Sip object.
    '''
    def __init__(self, wcs, w, h, x0=0, y0=0):
        self.wcs = wcs
        self.imagew = w
        self.imageh = h
        self.x0 = x0
        self.y0 = y0
    def pixelxy2radec(self, x, y):
        # FITS pixels x,y
        rd = self.wcs.pixelToPosition(x+self.x0-1, y+self.y0-1)
        return rd.ra, rd.dec
    def radec2pixelxy(self, ra, dec):
        # Vectorized?
        if hasattr(ra, '__len__') or hasattr(dec, '__len__'):
            try:
                b = np.broadcast(ra, dec)
                ok = np.ones(b.shape, bool)
                x  = np.zeros(b.shape)
                y  = np.zeros(b.shape)
                rd = RaDecPos(0.,0.)
                for i,(r,d) in enumerate(b):
                    rd.ra = r
                    rd.dec = d
                    xi,yi = self.wcs.positionToPixel(rd)
                    x.flat[i] = xi
                    y.flat[i] = yi
                x += 1 - self.x0
                y += 1 - self.y0
                return ok,x,y
            except:
                pass
            
        x,y = self.wcs.positionToPixel(RaDecPos(ra, dec))
        return True, x-self.x0+1, y-self.y0+1


def getParamTypeTree(param):
    mytype = type(param)
    if isinstance(param, MultiParams):
        return [mytype] + [getParamTypeTree(s) for s in param._getActiveSubs()]
    return [mytype]


