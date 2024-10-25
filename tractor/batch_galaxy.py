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
import cupy as cp

from tractor.galaxy import HoggGalaxy
from tractor import batch_mixture_profiles as mp
from tractor.utils import ParamList, MultiParams, ScalarParam, BaseParams
from tractor.patch import Patch, add_patches, ModelMask
from tractor.basics import SingleProfileSource, BasicSource
from tractor.batch_mixture_profiles import ImageDerivs, BatchImageParams, BatchMixtureOfGaussians

debug_ps = None
import time
tbs1 = np.zeros(10)
tbs2 = np.zeros(10)
tbs3 = np.zeros(10)


def print_tbs():
    print ("TBS1:", tbs1)
    print ("TBS2:", tbs2)
    print ("TBS3:", tbs3)


def getShearedProfileGPU(galaxy, imgs, px, py):
    tbs3[9] += 1
    t = time.time()
    galmix = galaxy.getProfile()
    galmix = BatchMixtureOfGaussians(galmix.amp, galmix.mean, galmix.var, quick=True) 
    tbs3[0] += time.time()-t
    t = time.time()
    cdinv = cp.array([img.getWcs().cdInverseAtPixel(px[i], py[i]) for i, img in enumerate(imgs)])
    tbs3[1] += time.time()-t
    t = time.time()
    G = cp.asarray(galaxy.shape.getRaDecBasis())
    tbs3[2] += time.time()-t
    t = time.time()
    Tinv = cp.dot(cdinv, G)
    tbs3[3] += time.time()-t
    t = time.time()
    #print (type(galmix))
    amix = galmix.apply_shear_GPU(Tinv)
    tbs3[4] += time.time()-t
    return amix

def getDerivativeShearedProfilesGPU(galaxy, img, px, py):
    # Returns a list of sheared profiles that will be needed to compute
    # derivatives for this source; this is assumed in addition to the
    # sheared profile at the current parameter settings.
    tbs2[9] += 1
    derivs = []
    if galaxy.isParamThawed('shape'):
        t = time.time()
        gsteps = galaxy.shape.getStepSizes()
        gnames = galaxy.shape.getParamNames()
        oldvals = galaxy.shape.getParams()
        tbs2[0] += time.time()-t
        for i, gstep in enumerate(gsteps):
            t = time.time()
            oldval = galaxy.shape.setParam(i, oldvals[i] + gstep)
            tbs2[1] += time.time()-t
            t = time.time()
            pro = getShearedProfileGPU(galaxy, img, px, py)
            tbs2[2] += time.time()-t
            t = time.time()
            #print('Param', gnames[i], 'was', oldval, 'stepped to', oldvals[i]+gstep,
            #      '-> profile', pro.var.ravel())
            galaxy.shape.setParam(i, oldval)
            derivs.append(('shape.'+gnames[i], pro, gstep))
            tbs2[3] += time.time()-t
    return derivs
