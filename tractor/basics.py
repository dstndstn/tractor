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

from tractor.image import Image
from tractor.patch import Patch
from tractor.utils import MultiParams
from tractor import mixture_profiles as mp

from tractor.tractortime import TAITime
from tractor.psf import (PixelizedPSF, GaussianMixturePSF,
                         GaussianMixtureEllipsePSF, NCircularGaussianPSF)
from tractor.wcs import (NullWCS, WcslibWcs, ConstantFitsWcs, TanWcs,
                         PixPos, RaDecPos)
from tractor.sky import NullSky, ConstantSky
from tractor.brightness import (Mag, Flux, Mags, Fluxes, NanoMaggies,
                                FluxesPhotoCal, MagsPhotoCal, NullPhotoCal,
                                LinearPhotoCal)
from tractor.pointsource import BasicSource, SingleProfileSource, PointSource
from tractor.shifted import (ParamsWrapper, ShiftedPsf, ScaledPhotoCal,
                             ScaledWcs, ShiftedWcs)
from tractor.ducks import Source

class ConstantSurfaceBrightness(MultiParams, Source):
    '''
    This is a source that represents a constant surface brightness / background level.

    The brightness is in per-square-arcsec units.
    '''
    def __init__(self, br):
        '''
        ConstantSurfaceBrightness(pos, brightness)
        '''
        super(ConstantSurfaceBrightness, self).__init__(br)

    #def copy(self):

    @staticmethod
    def getNamedParams():
        return dict(brightness=0)

    def getSourceType(self):
        return 'ConstantSurfaceBrightness'

    def __str__(self):
        return self.getSourceType() + ': ' + str(self.brightness) + ' mag/arcsec^2'

    def __repr__(self):
        return self.getSourceType() + '(' + repr(self.brightness) + ')'

    def getModelPatch(self, img, modelMask=None, **kwargs):
        if modelMask is None:
            mh,mw = img.shape
            mx0,my0 = 0,0
        else:
            mh,mw = modelMask.shape
            mx0,my0 = modelMask.x0, modelMask.y0
        cx,cy = mx0 + mw/2, my0 + mh/2
        pixscale = img.getWcs().pixscale_at(cx, cy)
        # this is in per-square-arcsec
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        ## FIXME -- we could return a duck-typed Patch that doesn't
        ## actually contain this array of pixels!!
        mod = np.empty((mh, mw), np.float32)
        # scale to counts-per-pixel
        mod[:,:] = counts * pixscale**2
        return Patch(mx0, my0, mod)

    def getParamDerivatives(self, img, modelMask=None, **kwargs):
        bright_frozen = self.isParamFrozen('brightness')
        if bright_frozen:
            return []

        photo = img.getPhotoCal()
        counts0 = photo.brightnessToCounts(self.brightness)
        bsteps = self.brightness.getStepSizes(img)
        bvals = self.brightness.getParams()
        scales = np.zeros(len(bsteps), np.float32)
        # Step each param (eg, each band) and see if it affects this image...
        for i, bstep in enumerate(bsteps):
            oldval = self.brightness.setParam(i, bvals[i] + bstep)
            countsi = photo.brightnessToCounts(self.brightness)
            self.brightness.setParam(i, oldval)
            if countsi == counts0:
                scales[i] = 0
            else:
                scales[i] = (countsi - counts0) / bstep
        if np.all(scales == 0):
            return [None] * len(scales)

        if modelMask is None:
            mh,mw = img.shape
            mx0,my0 = 0,0
        else:
            mh,mw = modelMask.shape
            mx0,my0 = modelMask.x0, modelMask.y0
        cx,cy = mx0 + mw/2, my0 + mh/2
        pixscale = img.getWcs().pixscale_at(cx, cy)


        derivs = []
        for i,s in enumerate(scales):
            if s == 0:
                derivs.append(None)
                continue
            mod = np.empty((mh, mw), np.float32)
            # for LinearPhotoCal, eg, s will end up being the linear factor
            mod[:,:] = s * pixscale**2
            df = Patch(mx0, my0, mod)
            df.setName('d(ptsrc)/d(bright%i)' % i)
            derivs.append(df)
        return derivs

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
        rd = self.wcs.pixelToPosition(x + self.x0 - 1, y + self.y0 - 1)
        return rd.ra, rd.dec

    def radec2pixelxy(self, ra, dec):
        # Vectorized?
        if hasattr(ra, '__len__') or hasattr(dec, '__len__'):
            try:
                b = np.broadcast(ra, dec)
                ok = np.ones(b.shape, bool)
                x = np.zeros(b.shape)
                y = np.zeros(b.shape)
                rd = RaDecPos(0., 0.)
                for i, (r, d) in enumerate(b):
                    rd.ra = r
                    rd.dec = d
                    xi, yi = self.wcs.positionToPixel(rd)
                    x.flat[i] = xi
                    y.flat[i] = yi
                x += 1 - self.x0
                y += 1 - self.y0
                return ok, x, y
            except:
                pass

        x, y = self.wcs.positionToPixel(RaDecPos(ra, dec))
        return True, x - self.x0 + 1, y - self.y0 + 1


def getParamTypeTree(param):
    mytype = type(param)
    if isinstance(param, MultiParams):
        return [mytype] + [getParamTypeTree(s) for s in param._getActiveSubs()]
    return [mytype]
