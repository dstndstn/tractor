"""
This file is part of the Tractor project.
Copyright 2011, 2012, 2013 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`sdss.py`
===========

SDSS-specific implementation of Tractor elements:

 - retrieves and reads SDSS images, converting them to Tractor language
 - knows about SDSS photometric and astrometric calibration conventions
 - for DR7-specifics, see sdss_dr7.
 
"""
from __future__ import print_function
import os
import sys
from math import pi, sqrt, ceil, floor
from datetime import datetime

import pylab as plt
import numpy as np

from .engine import *
from .basics import *
from .imageutils import interpret_roi
from .galaxy import *

## FIXME -- these PSF params are not Params
class SdssBrightPSF(ParamsWrapper):
    def __init__(self, real, a1, s1, a2, s2, a3, sigmap, beta):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.s1 = s1
        self.s2 = s2
        self.sigmap = sigmap
        self.beta = beta
        super(SdssBrightPSF,self).__init__(real)

    def getRadius(self):
        return self.real.getRadius()

    def getMixtureOfGaussians(self, **kwargs):
        return self.real.getMixtureOfGaussians(**kwargs)
        
    def getBrightPointSourcePatch(self, px, py, dc):
        if dc > self.a3:
            R = 25.
        else:
            R = np.sqrt( self.beta * self.sigmap**2 *
                         ( (dc / self.a3)**(-2./self.beta) - 1.) )
        # print('beta', self.beta)
        # print('sigmap', self.sigmap)
        # print('p0:', self.a3)
        # print('dc', dc)
        # print('Bright ps patch: R=', R)

        R = np.clip(R, 25., 200.)

        x0,x1 = int(floor(px-R)), int(ceil(px+R))
        y0,y1 = int(floor(py-R)), int(ceil(py+R))
        # clip to image bounds?

        #mog = MixtureOfGaussians([self.a1, self.a2], np.zeros((2,2)),
        #                        [np.diag([s1,s1]), np.diag([s2,s2])])
        #grid = mog.evaluate_grid_dstn(x0-px, x1-px, y0-py, y1-py)
        #patch = Patch(x0, y0, grid)

        X,Y = np.meshgrid(np.arange(x0,x1+1), np.arange(y0,y1+1))
        #print('patch shape', patch.shape)
        #print('X,Y shape', X.shape)
        R2 = ((X-px)**2 + (Y-py)**2)
        # According to RHL, these are denormalized Gaussians
        P = (self.a1 * np.exp(-0.5 * R2 / (self.s1**2)) +
             self.a2 * np.exp(-0.5 * R2 / (self.s2**2)) +
             self.a3 * ((1. + R2 / (self.beta * self.sigmap**2))**(-self.beta/2.)))
        P /= np.sum(P)
        return Patch(x0, y0, P)

    def getPointSourcePatch(self, px, py, **kwargs):
        return self.real.getPointSourcePatch(px, py, **kwargs)
        
'''
Warning: Bright point sources do NOT produce correct derivatives!!
'''

class SdssPointSource(PointSource):
    '''
    Knows about the SDSS 2-Gaussian-plus-power-law PSF model and
    switches (smoothly) to it for bright sources.

    *thresh* is the threshold, in COUNTS, for using the bright PSF
    model.
    
    *thresh2* is the lower threshold, in COUNTS, for when to start
    switching to the bright PSF model.  We use linear interpolation
    for the *thresh2 < b < thresh* range.  If not specified, defaults
    to factor (1./2.5) of *thresh*, ie, about a mag.
    '''
    def __init__(self, pos, bright, thresh=None, thresh2=None):
        super(SdssPointSource, self).__init__(pos, bright)
        self.thresh1 = thresh
        if thresh2 is not None:
            self.thresh2 = thresh2
            assert(self.thresh2 <= thresh)
        elif thresh is not None:
            self.thresh2 = (thresh / 2.5)
        else:
            self.thresh2 = None

    def copy(self):
        return SdssPointSource(pos, bright, self.thresh1, self.thresh2)

    def getModelPatch(self, img, modelMask=None):
        assert(modelMask is None)
        (px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)
        if counts == 0:
            return None
        if self.thresh1 is not None and counts >= self.thresh1:
            fb = 1.
            fa = 0.
        if self.thresh2 is not None and counts >= self.thresh2:
            fb = (counts - self.thresh2) / (self.thresh1 - self.thresh2)
            fb = np.clip(fb, 0., 1.)
            fa = 1. - fb
        else:
            fb = 0.
            fa = 1.

        patch = None
        if fb != 0.:
            ## HACK -- precision
            #print('Counts:', counts)
            #print('sigma:', img.getMedianPixelNoise())
            # after scaling by counts, we want to have a small fraction
            # of a sigma beyond the cutoff
            dc = 1e-3 * img.getMedianPixelNoise() / counts
            patchb = img.getPsf().getBrightPointSourcePatch(px, py, dc)
            patch = patchb * fb

        if fa != 0:
            patcha = img.getPsf().getPointSourcePatch(px, py)
            if patch is None:
                patch = patcha
            else:
                patch += patcha * fa

        #print('PointSource: PSF patch has sum', patch.getImage().sum())
        return patch * counts
    

def _check_sdss_files(sdss, run, camcol, field, bandname, filetypes,
                      retrieve=True, tryopen=False):
    from astrometry.sdss import band_index

    bandnum = band_index(bandname)
    for filetype in filetypes:
        fn = sdss.getPath(filetype, run, camcol, field, bandname)
        print('Looking for file', fn)
        exists = os.path.exists(fn)
        retrieveKwargs = {}
        if exists and tryopen:
            # This doesn't catch *all* types of errors you can imagine...
            cmd = 'fitsverify -q %s' % fn
            try:
                print('Running:', cmd)
                rtn = os.system(cmd)
                print('rtn:', rtn)
                if rtn != 0:
                    exists = False
            except:
                print('Failed to fitsverify', fn, ': maybe corrupt.')
                exists = False
            if not exists:
                retrieveKwargs.update(skipExisting=False)
                
        if (not exists) and retrieve:
            print('Retrieving', fn)
            res = sdss.retrieve(filetype, run, camcol, field, bandnum,
                                **retrieveKwargs)
            if res is False:
                raise RuntimeError('No such file on SDSS DAS: %s, ' % filetype +
                                   'rcfb %i/%i/%i/%s' %
                                   (run, camcol, field, bandname))
        elif not exists:
            raise OSError('no such file: "%s"' % fn)

def _get_sources(run, camcol, field, bandname='r', sdss=None, release='DR7',
                 objs=None,
                 retrieve=True, checkFiles=True,
                 curl=False, roi=None,
                 radecroi=None,
                 radecrad=None,
                 bands=None,
                 badmag=25, nanomaggies=False,
                 getobjs=False, getsourceobjs=False, getobjinds=False,
                 extrabands=None,
                 fixedComposites=False,
                 forcePointSources=False,
                 useObjcType=False,
                 objCuts=True,
                 classmap={},
                 ellipse=GalaxyShape,
                 cutToPrimary=False):
    '''
    If set,

    radecrad = (ra,dec,rad)

    returns sources within "rad" degrees of the given RA,Dec (in degrees)
    center.

    WARNING, this method alters the "objs" argument, if given.
    Consider calling objs.copy() before calling.

    -"bandname" is the SDSS band used to cut on position, select
     star/gal/exp/dev, and set galaxy shapes.

    -"bands" are the bands to include in the returned Source objects;
     they will be initialized from the SDSS bands.

    -"extrabands" are also included in the returned Source objects;
     they will be initialized to the SDSS flux for either the first of
     "bands", if given, or "bandname".
    '''
    from astrometry.sdss import (DR7, DR8, DR9, band_names, band_index,
                                 photo_flags1_map)
    
    #   brightPointSourceThreshold=0.):

    if sdss is None:
        dr = dict(DR7=DR7, DR8=DR8, DR9=DR9)[release]
        sdss = dr(curl=curl)
    drnum = sdss.getDRNumber()
    isdr7 = (drnum == 7)
    
    if bands is None:
        bands = band_names()
    bandnum = band_index(bandname)

    bandnums = np.array([band_index(b) for b in bands])
    bandnames = bands

    if extrabands is None:
        extrabands = []
    
    if objs is None:
        from astrometry.util.fits import fits_table
        if isdr7:
            # FIXME
            rerun = 0
            if checkFiles:
                _check_sdss_files(sdss, run, camcol, field, bandnum,
                                  ['tsObj', 'tsField'],
                                  retrieve=retrieve, tryopen=True)
            tsf = sdss.readTsField(run, camcol, field, rerun)
            objfn = sdss.getPath('tsObj', run, camcol, field,
                                 bandname, rerun=rerun)
        else:
            if checkFiles:
                _check_sdss_files(sdss, run, camcol, field, bandnum,
                                  ['photoObj'],
                                  retrieve=retrieve, tryopen=True)
            objfn = sdss.getPath('photoObj', run, camcol, field)
            
        objs = fits_table(objfn)
        if objs is None:
            print('No sources in SDSS file', objfn)
            return []

    objs.index = np.arange(len(objs))
    if getobjs:
        allobjs = objs.copy()
        
    if roi is not None:
        x0,x1,y0,y1 = roi
        # FIXME -- we keep only the sources whose centers are within
        # the ROI box.  Should instead do some ellipse-overlaps
        # geometry.
        x = objs.colc[:,bandnum]
        y = objs.rowc[:,bandnum]
        objs.cut((x >= x0) * (x < x1) * (y >= y0) * (y < y1))

    if radecroi is not None:
        r0,r1,d0,d1 = radecroi
        objs.cut((objs.ra >= r0) * (objs.ra <= r1) *
                 (objs.dec >= d0) * (objs.dec <= d1))

    if radecrad is not None:
        from astrometry.libkd.spherematch import match_radec
        (ra,dec,rad) = radecrad
        I,J,d = match_radec(ra, dec, objs.ra, objs.dec, rad)
        objs.cut(J)
        del I
        del d
        
    if objCuts:
        # Only deblended children;
        # No BRIGHT sources
        bright = photo_flags1_map.get('BRIGHT')
        objs.cut((objs.nchild == 0) * ((objs.objc_flags & bright) == 0))

    if cutToPrimary:
        objs.cut((objs.resolve_status & 256) > 0)

    if len(objs) == 0:
        sources = []
        if not (getobjs or getobjinds or getsourceobjs):
            return sources
        rtn = [sources]
        if getobjs:
            rtn.append(None)
        if getobjinds:
            rtn.append(None)
        if getsourceobjs:
            rtn.append(None)
        return rtn
    
    if isdr7:
        objs.rename('phi_dev', 'phi_dev_deg')
        objs.rename('phi_exp', 'phi_exp_deg')
        objs.rename('r_dev', 'theta_dev')
        objs.rename('r_exp', 'theta_exp')
        
    # SDSS and Tractor have different opinions on which way this rotation goes
    objs.phi_dev_deg *= -1.
    objs.phi_exp_deg *= -1.

    # MAGIC -- minimum size of galaxy.
    objs.theta_dev = np.maximum(objs.theta_dev, 1./30.)
    objs.theta_exp = np.maximum(objs.theta_exp, 1./30.)

    if forcePointSources:
        Lstar = np.ones(len(objs), float)
        Lgal = np.zeros(len(objs), float)
        Ldev = Lexp = Lgal
    else:
        if useObjcType:
            objs.cut(np.logical_or(objs.objc_type == 6,
                                   objs.objc_type == 3))
            Lstar = (objs.objc_type == 6)
            Lgal = (objs.objc_type == 3)
        else:
            Lstar = (objs.prob_psf[:,bandnum] == 1) * 1.0
            Lgal  = (objs.prob_psf[:,bandnum] == 0) * 1.0

        if isdr7:
            fracdev = objs.fracpsf[:,bandnum]
        else:
            fracdev = objs.fracdev[:,bandnum]
        Ldev = Lgal * fracdev
        Lexp = Lgal * (1. - fracdev)

    if isdr7:
        from .sdss_dr7 import _dr7_getBrightness
        if nanomaggies:
            raise RuntimeError('Nanomaggies not supported for DR7 (yet)')
        def lup2bright(lups):
            counts = [tsf.luptitude_to_counts(lup,j)
                      for j,lup in enumerate(lups)]
            counts = np.array(counts)
            bright = _dr7_getBrightness(counts, tsf, bandnames, extrabands)
            return bright
        flux2bright = lup2bright
        starflux = objs.psfcounts
        compflux = objs.counts_model
        devflux = objs.counts_dev
        expflux = objs.counts_exp
        def comp2bright(lups, Ldev, Lexp):
            counts = [tsf.luptitude_to_counts(lup,j)
                      for j,lup in enumerate(lups)]
            counts = np.array(counts)
            dcounts = counts * Ldev
            ecounts = counts * Lexp
            dbright = _dr7_getBrightness(dcounts, tsf, bands, extrabands)
            ebright = _dr7_getBrightness(ecounts, tsf, bands, extrabands)
            return dbright, ebright
    else:
        def nmgy2bright(flux):
            if len(bandnums):
                flux = flux[bandnums]
            else:
                flux = flux[np.array([bandnum])]
            bb = bandnames + extrabands
            if nanomaggies:
                if len(extrabands):
                    if len(bandnums) == 0:
                        # Only "extrabands", no SDSS bands.
                        flux = np.zeros(len(extrabands)) + flux[0]
                    else:
                        flux = np.append(flux, np.zeros(len(extrabands)))
                bright = NanoMaggies(order=bb, **dict(zip(bb, flux)))
            else:
                I = (flux > 0)
                mag = np.zeros_like(flux) + badmag
                mag[I] = sdss.nmgy_to_mag(flux[I])
                if len(extrabands):
                    mag = np.append(mag, np.zeros(len(extrabands)) + badmag)
                bright = Mags(order=bb, **dict(zip(bb,mag)))
            return bright
        def comp2bright(flux, Ldev, Lexp):
            dflux = flux * Ldev
            eflux = flux * Lexp
            dbright = nmgy2bright(dflux)
            ebright = nmgy2bright(eflux)
            return dbright, ebright
        
        flux2bright = nmgy2bright
        starflux = objs.psfflux
        compflux = objs.cmodelflux
        devflux = objs.devflux
        expflux = objs.expflux
        
    sources = []
    nstars, ndev, nexp, ncomp = 0, 0, 0, 0
    isources = []

    ptsrcclass = classmap.get(PointSource, PointSource)

    for i in range(len(objs)):
        if Lstar[i]:
            pos = RaDecPos(objs.ra[i], objs.dec[i])
            flux = starflux[i,:]
            bright = flux2bright(flux)
            # This should work, I just don't feel like testing it now...
            # if brightPointSourceThreshold:
            #   ps = SdssPointSource(pos, bright, thresh=brightPointSourceThreshold)
            # else:
            #   ps = PointSource(pos, bright)
            sources.append(ptsrcclass(pos, bright))
            nstars += 1
            isources.append(i)
            continue

        hasdev = (Ldev[i] > 0)
        hasexp = (Lexp[i] > 0)
        iscomp = (hasdev and hasexp)
        pos = RaDecPos(objs.ra[i], objs.dec[i])
        if iscomp:
            flux = compflux[i,:]
        elif hasdev:
            flux = devflux[i,:]
        elif hasexp:
            flux = expflux[i,:]
        else:
            print('Skipping object with Lstar = %g, Ldev = %g, Lexp = %g (fracdev=%g)'
                  % (Lstar[i], Ldev[i], Lexp[i], fracdev[i]))
            continue

        isources.append(i)
        if iscomp:
            if fixedComposites:
                bright = flux2bright(flux)
                fdev = (Ldev[i] / (Ldev[i] + Lexp[i]))
            else:
                dbright,ebright = comp2bright(flux, Ldev[i], Lexp[i])
        else:
            bright = flux2bright(flux)

        if hasdev:
            re  = objs.theta_dev  [i,bandnum]
            ab  = objs.ab_dev     [i,bandnum]
            phi = objs.phi_dev_deg[i,bandnum]
            dshape = ellipse(re, ab, phi)
        if hasexp:
            re  = objs.theta_exp  [i,bandnum]
            ab  = objs.ab_exp     [i,bandnum]
            phi = objs.phi_exp_deg[i,bandnum]
            eshape = ellipse(re, ab, phi)

        if iscomp:
            if fixedComposites:
                gal = FixedCompositeGalaxy(pos, bright, fdev, eshape, dshape)
            else:
                gal = CompositeGalaxy(pos, ebright, eshape, dbright, dshape)
            ncomp += 1
        elif hasdev:
            gal = DevGalaxy(pos, bright, dshape)
            ndev += 1
        elif hasexp:
            gal = ExpGalaxy(pos, bright, eshape)
            nexp += 1
        sources.append(gal)

    print('Created', ndev, 'deV,', nexp, 'exp,', ncomp, 'composite',)
    print('(total %i) galaxies and %i stars' % (ndev+nexp+ncomp, nstars))
    
    if not (getobjs or getobjinds or getsourceobjs):
        return sources

    if nstars + ndev + nexp + ncomp < len(objs):
        objs = objs[np.array(isources)]
    
    rtn = [sources]
    if getobjs:
        rtn.append(allobjs)
    if getobjinds:
        rtn.append(objs.index if len(objs) else np.array([]))
    if getsourceobjs:
        rtn.append(objs)
    return rtn

def get_tractor_sources_dr8(*args, **kwargs):
    #brightPointSourceThreshold=0.):
    #brightPointSourceBand='r'):
    '''
    get_tractor_sources_dr8(run, camcol, field, bandname='r', sdss=None,
                            retrieve=True, curl=False, roi=None, bands=None,
                            badmag=25, nanomaggies=False,
                            getobjs=False, getobjinds=False)

    Creates tractor.Source objects corresponding to objects in the SDSS catalog
    for the given field.

    bandname: "canonical" band from which to get galaxy shapes, positions, etc

    '''
    kwargs.update(release='DR8')
    return _get_sources(*args, **kwargs)

def get_tractor_sources_dr9(*args, **kwargs):
    kwargs.update(release='DR9')
    return _get_sources(*args, **kwargs)

def get_tractor_sources_cas_dr9(table, bandname='r', bands=None,
                                extrabands=None,
                                nanomaggies=False):
    '''
    table: filename or astrometry.util.fits.fits_table() object
    '''
    from astrometry.sdss import band_names
    from astrometry.util.fits import fits_table

    if isinstance(table, basestring):
        cas = fits_table(table)
    else:
        cas = table

    # Make it look like a "photoObj" file.
    cols = cas.get_columns()
    N = len(cas)

    T = tabledata()
    T.ra = cas.ra
    T.dec = cas.dec
    
    # nchild
    if not 'nchild' in cols:
        T.nchild = np.zeros(N,int)
    else:
        T.nchild = cas.nchild
    # rowc,colc -- shouldn't be necessary...
    if not 'objc_flags' in cols:
        T.objc_flags = np.zeros(N,int)
    else:
        T.objc_flags = cas.objc_flags

    allbands = band_names()
    nbands = len(allbands)

    colmap = [('phi_dev_deg', 'devphi'),
              ('phi_exp_deg', 'expphi'),
              ('theta_dev',   'devrad'),
              ('theta_exp',   'exprad'),
              ('ab_dev',      'devab'),
              ('ab_exp',      'expab'),
              ('psfflux',     'psfflux'),
              ('cmodelflux',  'cmodelflux'),
              ('devflux',     'devflux'),
              ('expflux',     'expflux'),
              ('fracdev',     'fracdev'),
              ('prob_psf',     'probpsf'),
            ]

    for c1,c2 in colmap:
        T.set(c1, np.zeros((N, nbands)))

    for bi,b in enumerate(allbands):
        for c1,c2 in colmap:
            cname = '%s_%s' % (c2, b)
            if cname in cols:
                T.get(c1)[:,bi] = cas.get(cname)

    return _get_sources(-1, -1, -1, release='DR9', objs=T, sdss=DR9(),
                        bandname=bandname, bands=bands, nanomaggies=nanomaggies,
                        extrabands=extrabands)

def scale_sdss_image(tim, S):
    '''
    Returns a rebinned "tim" by integer scale "S" (ie S=2 reduces the
    image size by a factor of 2x2).

    The rescaling preserves intensity.
    
    We drop remainder pixels.
    '''

    # drop remainder pixels (for all image-shaped data items)
    H,W = tim.shape
    data = tim.getImage()
    invvar = tim.getInvvar()
    if H%S:
        data = data    [:-(H%S),:]
        invvar = invvar[:-(H%S),:]
    if W%S:
        data = data    [:,:-(W%S)]
        invvar = invvar[:,:-(W%S)]
    H,W = data.shape
    assert((H % S) == 0)
    assert((W % S) == 0)

    # rebin image data
    sH,sW = H/S, W/S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    newiv = np.zeros((sH,sW), dtype=invvar.dtype)
    for i in range(S):
        for j in range(S):
            iv = invvar[i::S, j::S]
            newdata += data[i::S, j::S] * iv
            newiv += iv
    newdata /= (newiv + (newiv == 0)*1.)
    newdata[newiv == 0] = 0.
    data = newdata
    invvar = newiv
    psf = tim.getPsf().scaleBy(1./S)
    wcs = ScaledWcs(tim.getWcs(), 1./S)
    # We're assuming ConstantSky here
    sky = tim.getSky().copy()
    photocal = ScaledPhotoCal(tim.getPhotoCal(), (1./S)**2)
    return Image(data=data, invvar=invvar, psf=psf, wcs=wcs,
                 sky=sky, photocal=photocal, name='Scaled(%i) '%S + tim.name,
                 zr=tim.zr)

def get_tractor_image_dr8(*args, **kwargs):
    '''
    Creates a tractor.Image given an SDSS field identifier.

    If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
    in the image, in zero-indexed pixel coordinates.  x1,y1 are
    NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.

    psf can be:
      "dg" for double-Gaussian
      "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

      "bright-*", "*" one of the above PSFs, with special handling at
      the bright end.

    "roiradecsize" = (ra, dec, half-size in pixels) indicates that you
    want to grab a ROI around the given RA,Dec.

    "roiradecbox" = (ra0, ra1, dec0, dec1) indicates that you
    want to grab a ROI containing the given RA,Dec ranges.

    "invvarAtCentr" -- get a scalar constant inverse-variance

    "invvarAtCenterImage" -- get a scalar constant inverse-variance
    but still make an image out of it.

    Returns: (tractor.Image, dict)

    dict contains useful details like:
      'sky'
      'skysig'
    '''
    retry = kwargs.pop('retry_retrieve', True)
    if retry:
        try:
            return _get_tractor_image_dr8(*args, **kwargs)
        except:
            import traceback
            print('First get_tractor_image_dr8() failed -- trying to re-retrieve data')
            print(traceback.print_exc())
            
            # Re-retrieve the data
            sdss = kwargs.get('sdss', None)
            if sdss is None:
                from astrometry.sdss import DR8
                curl = kwargs.get('curl', False)
                sdss = DR8(curl=curl)
            (run,camcol,field,bandname) = args
            for ft in ['psField', 'fpM', 'frame']:
                fn = sdss.retrieve(ft, run, camcol, field, bandname,
                                   skipExisting=False)
            # and continue...
    return _get_tractor_image_dr8(*args, **kwargs)

def _get_tractor_image_dr8(run, camcol, field, bandname, sdss=None,
                          roi=None, psf='kl-gm', roiradecsize=None,
                          roiradecbox=None,
                          savepsfimg=None, curl=False,
                          nanomaggies=False,
                          zrange=[-3,10],
                          invvarIgnoresSourceFlux=False,
                          invvarAtCenter=False,
                          invvarAtCenterImage=False,
                          retrieve=True,
                          imargs={}):
    from astrometry.sdss import band_index

    origpsf = psf
    if psf.startswith('bright-'):
        psf = psf[7:]
        brightpsf = True
        print('Setting bright PSF handling')
    else:
        brightpsf = False

    valid_psf = ['dg', 'kl-gm', 'kl-pix']
    if psf not in valid_psf:
        raise RuntimeError('PSF must be in ' + str(valid_psf))

    if sdss is None:
        from astrometry.sdss import DR8
        sdss = DR8(curl=curl)

    bandnum = band_index(bandname)

    if retrieve:
        for ft in ['psField', 'fpM']:
            fn = sdss.retrieve(ft, run, camcol, field, bandname)
        fn = sdss.retrieve('frame', run, camcol, field, bandname)
    else:
        fn = sdss.getPath('frame', run, camcol, field, bandname)

    # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
    frame = sdss.readFrame(run, camcol, field, bandname, filename=fn)
    H,W = frame.getImageShape()
    info = dict()
    hdr = frame.getHeader()
    tai = hdr.get('TAI')
    stripe = hdr.get('STRIPE')
    strip = hdr.get('STRIP')
    obj = hdr.get('OBJECT')
    info.update(tai=tai, stripe=stripe, strip=strip, object=obj, hdr=hdr)

    astrans = frame.getAsTrans()
    wcs = SdssWcs(astrans)
    #print('Created SDSS Wcs:', wcs)
    #print('(x,y) = 1,1 -> RA,Dec', wcs.pixelToPosition(1,1))

    X = interpret_roi(wcs, (H,W), roi=roi, roiradecsize=roiradecsize,
                               roiradecbox=roiradecbox)
    if X is None:
        return None,None
    roi,hasroi = X
    info.update(roi=roi)
    x0,x1,y0,y1 = roi

    # Mysterious half-pixel shift.  asTrans pixel coordinates?
    wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

    if nanomaggies:
        photocal = LinearPhotoCal(1., band=bandname)
    else:
        photocal = MagsPhotoCal(bandname, 22.5)

    sky = 0.
    skyobj = ConstantSky(sky)

    calibvec = frame.getCalibVec()

    invvarAtCenter = invvarAtCenter or invvarAtCenterImage

    psfield = sdss.readPsField(run, camcol, field)
    iva = dict(ignoreSourceFlux=invvarIgnoresSourceFlux)
    if invvarAtCenter:
        if hasroi:
            iva.update(constantSkyAt=(int((x0+x1)/2.), int((y0+y1)/2.)))
        else:
            iva.update(constantSkyAt=(int(W/2.), int(H/2.)))
    invvar = frame.getInvvar(psfield, bandnum, **iva)
    invvar = invvar.astype(np.float32)
    if not invvarAtCenter:
        assert(invvar.shape == (H,W))

    # Could get this from photoField instead
    # http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html
    gain = psfield.getGain(bandnum)
    darkvar = psfield.getDarkVariance(bandnum)

    meansky = np.mean(frame.sky)
    meancalib = np.mean(calibvec)
    skysig = sqrt((meansky / gain) + darkvar) * meancalib

    info.update(sky=sky, skysig=skysig)
    zr = np.array(zrange)*skysig + sky
    info.update(zr=zr)

    # http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
    fpM = sdss.readFpM(run, camcol, field, bandname)

    if not hasroi:
        image = frame.getImage()

    else:
        roislice = (slice(y0,y1), slice(x0,x1))
        image = frame.getImageSlice(roislice).astype(np.float32)
        if invvarAtCenterImage:
            invvar = invvar + np.zeros(image.shape, np.float32)
        elif invvarAtCenter:
            pass
        else:
            invvar = invvar[roislice].copy()
        H,W = image.shape
            
    if (not invvarAtCenter) or invvarAtCenterImage:
        for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
            fpM.setMaskedPixels(plane, invvar, 0, roi=roi)

    dgpsf = psfield.getDoubleGaussian(bandnum, normalize=True)
    info.update(dgpsf=dgpsf)
            
    if psf == 'kl-pix':
        # Pixelized KL-PSF
        klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
        # Trim symmetric zeros
        sh,sw = klpsf.shape
        while True:
            if (np.all(klpsf[0,:] == 0.) and
                np.all(klpsf[:,0] == 0.) and
                np.all(klpsf[-1,:] == 0.) and
                np.all(klpsf[:,-1] == 0.)):
                klpsf = klpsf[1:-1, 1:-1]
            else:
                break
        mypsf = PixelizedPSF(klpsf)
        
    elif psf == 'kl-gm':
        from emfit import em_fit_2d
        from fitpsf import em_init_params
        
        # Create Gaussian mixture model PSF approximation.
        klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
        S = klpsf.shape[0]
        # number of Gaussian components
        K = 3
        w,mu,sig = em_init_params(K, None, None, None)
        II = klpsf.copy()
        II /= II.sum()
        # HIDEOUS HACK
        II = np.maximum(II, 0)
        #print('Multi-Gaussian PSF fit...')
        xm,ym = -(S/2), -(S/2)
        if savepsfimg is not None:
            plt.clf()
            plt.imshow(II, interpolation='nearest', origin='lower')
            plt.title('PSF image to fit with EM')
            plt.savefig(savepsfimg)
        res = em_fit_2d(II, xm, ym, w, mu, sig)
        #print('em_fit_2d result:', res)
        if res == 0:
            # print('w,mu,sig', w,mu,sig)
            mypsf = GaussianMixturePSF(w, mu, sig)
            mypsf.computeRadius()
        else:
            # Failed!  Return 'dg' model instead?
            print('PSF model fit', psf, 'failed!  Returning DG model instead')
            psf = 'dg'
    if psf == 'dg':
        print('Creating double-Gaussian PSF approximation')
        (a,s1, b,s2) = dgpsf
        mypsf = NCircularGaussianPSF([s1, s2], [a, b])

    if brightpsf:
        print('Wrapping PSF in SdssBrightPSF')
        (a1,s1, a2,s2, a3,sigmap,beta) = psfield.getPowerLaw(bandnum)
        mypsf = SdssBrightPSF(mypsf, a1,s1,a2,s2,a3,sigmap,beta)
        print('PSF:', mypsf)

    timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
                 sky=skyobj, photocal=photocal,
                 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
                       (run, camcol, field, bandname)),
                 time=TAITime(tai),
                 **imargs)
    timg.zr = zr
    return timg,info

def get_tractor_image_dr9(*args, **kwargs):
    sdss = kwargs.get('sdss', None)
    if sdss is None:
        from astrometry.sdss import DR9
        curl = kwargs.pop('curl', False)
        kwargs['sdss'] = DR9(curl=curl)
    return get_tractor_image_dr8(*args, **kwargs)

class SdssWcs(ParamList):
    pnames = ['a', 'b', 'c', 'd', 'e', 'f',
              'drow0', 'drow1', 'drow2', 'drow3',
              'dcol0', 'dcol1', 'dcol2', 'dcol3',
              'csrow', 'cscol', 'ccrow', 'cccol',
              'x0', 'y0']

    @staticmethod
    def getNamedParams():
        # node and incl are properties of the survey geometry, not params.
        # riCut... not clear.
        # Note that we omit x0,y0 from this list
        return dict([(k,i) for i,k in enumerate(SdssWcs.pnames[:-2])])

    def __init__(self, astrans):
        self.x0 = 0
        self.y0 = 0
        super(SdssWcs, self).__init__(self.x0, self.y0, astrans)
        # ParamList keeps its params in a list; we don't want to do that.
        del self.vals
        self.astrans = astrans
        # if not None, cd_at_pixel() returns this constant value; set via
        # setConstantCd()
        self.constant_cd = None

    def setConstantCd(self, x, y):
        self.constant_cd = self.cdAtPixel(x, y)

    def _setThing(self, i, val):
        N = len(SdssWcs.pnames)
        if i == N-2:
            self.x0 = val
        elif i == N-1:
            self.y0 = val
        else:
            t = self.astrans.trans
            t[SdssWcs.pnames[i]] = val
    def _getThing(self, i):
        N = len(SdssWcs.pnames)
        if i == N-2:
            return self.x0
        elif i == N-1:
            return self.y0
        t = self.astrans.trans
        return t[SdssWcs.pnames[i]]
    def _getThings(self):
        t = self.astrans.trans
        return [t[nm] for nm in SdssWcs.pnames[:-2]] + [self.x0, self.y0]
    def _numberOfThings(self):
        return len(SdssWcs.pnames)

    def getStepSizes(self, *args, **kwargs):
        deg = 0.396 / 3600. # deg/pix
        P = 2000. # ~ image size
        # a,d: degrees
        # b,c,e,f: deg/pixels
        # drowX: 1/(pix ** (X-1)
        # dcolX: 1/(pix ** (X-1)
        # c*row,col: 1.
        ss = [ deg, deg/P, deg/P, deg, deg/P, deg/P,
               1., 1./P, 1./P**2, 1./P**3,
               1., 1./P, 1./P**2, 1./P**3,
               1., 1., 1., 1.,
               1., 1.]
        return list(self._getLiquidArray(ss))

    def setX0Y0(self, x0, y0):
        self.x0 = x0
        self.y0 = y0

    # This function is not used by the tractor, and it works in
    # *original* pixel coords (no x0,y0 offsets)
    # (x,y) to RA,Dec in deg
    def pixelToRaDec(self, x, y):
        ra,dec = self.astrans.pixel_to_radec(x, y)
        return ra,dec

    def cdAtPixel(self, x, y):
        if self.constant_cd is not None:
            return self.constant_cd
        return self.astrans.cd_at_pixel(x + self.x0, y + self.y0)

    def pixscale_at(self, x, y):
        cd = self.cdAtPixel(x,y)
        pixscale = np.sqrt(np.abs(cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]))
        return pixscale
    
    # RA,Dec in deg to pixel x,y.
    # HACK -- color -- debug
    def positionToPixel(self, pos, src=None, color=0.):
        ## FIXME -- color.
        if color != 0.:
            x,y = self.astrans.radec_to_pixel_single_py(pos.ra, pos.dec, color)
        else:
            x,y = self.astrans.radec_to_pixel_single(pos.ra, pos.dec)
        return x - self.x0, y - self.y0

    # (x,y) to RA,Dec in deg
    def pixelToPosition(self, x, y, src=None):
        ## FIXME -- color.
        ra,dec = self.pixelToRaDec(x + self.x0, y + self.y0)
        return RaDecPos(ra, dec)
