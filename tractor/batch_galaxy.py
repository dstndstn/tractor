"""
This file is part of the Tractor project.
Copyright 2011, 2012, 2013 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`galaxy.py`
================

Exponential and deVaucouleurs galaxy model classes.

These use the slightly modified versions of the exp and dev profiles
from the SDSS /Photo/ software; we use multi-Gaussian approximations
of these.
"""
import numpy as np
from cupy_wrapper import cp
#import cupy as cp

from tractor.galaxy import HoggGalaxy
from tractor.sersic import SersicGalaxy
from tractor import batch_mixture_profiles as mp
from tractor.utils import ParamList, MultiParams, ScalarParam, BaseParams
from tractor.patch import Patch, add_patches, ModelMask
from tractor.basics import SingleProfileSource, BasicSource
from tractor.batch_mixture_profiles import ImageDerivs, BatchImageParams, BatchMixtureOfGaussians

debug_ps = None
import time

def getShearedProfileGPU(galaxy, imgs, px, py):
    galmix = galaxy.getProfile()
    galmix = BatchMixtureOfGaussians(galmix.amp, galmix.mean, galmix.var, quick=True) 
    cdinv = cp.array([img.getWcs().cdInverseAtPixel(px[i], py[i]) for i, img in enumerate(imgs)])
    G = cp.asarray(galaxy.shape.getRaDecBasis())
    Tinv = cp.dot(cdinv, G)
    #print (type(galmix))
    amix = galmix.apply_shear_GPU(Tinv)
    return amix

def getDerivativeShearedProfilesGPU(galaxy, img, px, py):
    # Returns a list of sheared profiles that will be needed to compute
    # derivatives for this source; this is assumed in addition to the
    # sheared profile at the current parameter settings.
    derivs = []
    if galaxy.isParamThawed('shape'):
        gsteps = galaxy.shape.getStepSizes()
        gnames = galaxy.shape.getParamNames()
        oldvals = galaxy.shape.getParams()
        for i, gstep in enumerate(gsteps):
            oldval = galaxy.shape.setParam(i, oldvals[i] + gstep)
            pro = getShearedProfileGPU(galaxy, img, px, py)
            #print('Param', gnames[i], 'was', oldval, 'stepped to', oldvals[i]+gstep,
            #      '-> profile', pro.var.ravel())
            galaxy.shape.setParam(i, oldval)
            derivs.append(('shape.'+gnames[i], pro, gstep))
    if type(galaxy) == SersicGalaxy:
        #print ("SERSIC")
        if galaxy.isParamThawed('sersicindex'):
            steps = galaxy.sersicindex.getStepSizes()
            inames = galaxy.sersicindex.getParamNames()
            oldvals = galaxy.sersicindex.getParams()
            ups = galaxy.sersicindex.getUpperBounds()
            los = galaxy.sersicindex.getLowerBounds()
            n0 = len(pro.amp)
            #n0 = len(self._getShearedProfile(img, px, py).amp)

            for i,step in enumerate(steps):
                # Assume step is positive, and check whether stepping
                # would exceed the upper bound.
                newval = oldvals[i] + step
                if newval > ups[i]:
                    step *= -1.
                    newval = oldvals[i] + step
                    assert(newval > los[i])
                oldval = galaxy.sersicindex.setParam(i, newval)
                pro = getShearedProfileGPU(galaxy, img, px, py)
                if len(pro.amp) != n0:
                    # If we can, try stepping the opposite direction to avoid changing the
                    # number of Gaussian components
                    new2 = oldvals[i] - step
                    if new2 < ups[i] and new2 > los[i]:
                        galaxy.sersicindex.setParam(i, new2)
                        pro2 = getShearedProfileGPU(galaxy, img, px, py)
                        if len(pro2.amp) == n0:
                            # use this one instead
                            pro = pro2
                            step *= -1

                galaxy.sersicindex.setParam(i, oldval)
                derivs.append(('sersicindex.'+inames[i], pro, step))
    return derivs
