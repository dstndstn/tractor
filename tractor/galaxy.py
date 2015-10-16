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
from __future__ import print_function

import numpy as np

from astrometry.util.miscutils import get_overlapping_region

from . import mixture_profiles as mp
from .engine import *
from .utils import *
from .cache import *
from .patch import *
from .basics import SingleProfileSource, BasicSource

_galcache = Cache(maxsize=10000)
def get_galaxy_cache():
    return _galcache

def set_galaxy_cache_size(N=10000):
    global _galcache
    _galcache = Cache(maxsize=N)

enable_galaxy_cache = set_galaxy_cache_size

def disable_galaxy_cache():
    global _galcache
    _galcache = None

class GalaxyShape(ParamList):
    '''
    A naive representation of an ellipse (describing a galaxy shape),
    using effective radius (in arcsec), axis ratio, and position angle.

    For better ellipse parameterizations, see ellipses.py
    '''
    @staticmethod
    def getName():
        return "Galaxy Shape"

    @staticmethod
    def getNamedParams():
        '''
        re: arcsec
        ab: axis ratio, dimensionless, in [0,1]
        phi: deg, "E of N", 0=direction of increasing Dec,
        90=direction of increasing RA
        '''
        return dict(re=0, ab=1, phi=2)

    def __repr__(self):
        return 're=%g, ab=%g, phi=%g' % (self.re, self.ab, self.phi)

    def __str__(self):
        return ('%s: re=%.2f, ab=%.2f, phi=%.1f' %
                (self.getName(), self.re, self.ab, self.phi))

    def getTensor(self, cd):
        # G takes unit vectors (in r_e) to degrees (~intermediate world coords)
        G = self.getRaDecBasis()
        # "cd" takes pixels to degrees (intermediate world coords)
        # T takes pixels to unit vectors.
        T = np.dot(np.linalg.inv(G), cd)
        return T

    def getRaDecBasis(self):
        '''
        Returns a transformation matrix that takes vectors in r_e
        to delta-RA, delta-Dec vectors.
        '''
        # # convert re, ab, phi into a transformation matrix
        phi = np.deg2rad(90 - self.phi)
        # # convert re to degrees
        # # HACK -- bring up to a minimum size to prevent singular
        # # matrix inversions
        re_deg = max(1./30, self.re) / 3600.
        cp = np.cos(phi)
        sp = np.sin(phi)
        # Squish, rotate, and scale into degrees.
        # resulting G takes unit vectors (in r_e) to degrees
        # (~intermediate world coords)
        return re_deg * np.array([[cp, sp*self.ab], [-sp, cp*self.ab]])


plotnum = 0

class Galaxy(MultiParams, SingleProfileSource):
    '''
    Generic Galaxy profile with position, brightness, and shape.
    '''
    def __init__(self, *args):
        super(Galaxy, self).__init__(*args)
        self.name = self.getName()
        self.dname = self.getDName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, shape=2)

    def getName(self):
        return 'Galaxy'

    def getDName(self):
        '''
        Name used in labeling the derivative images d(Dname)/dx, eg
        '''
        return 'gal'
        
    def getSourceType(self):
        return self.name

    def getShape(self):
        return self.shape

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness)
                + ' and ' + str(self.shape))
    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', shape=' + repr(self.shape))

    def getUnitFluxModelPatch(self, img, **kwargs):
        raise RuntimeError('getUnitFluxModelPatch unimplemented in' +
                           self.getName())

    # returns [ Patch, Patch, ... ] of length numberOfParams().
    # Galaxy.
    def getParamDerivatives(self, img, modelMask=None):
        pos0 = self.getPosition()
        (px0,py0) = img.getWcs().positionToPixel(pos0, self)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)

        minsb = img.modelMinval
        if counts > 0:
            minval = minsb / counts
        else:
            minval = None

        patch0 = self.getUnitFluxModelPatch(img, px0, py0, minval=minval,
                                            modelMask=modelMask)
        if patch0 is None:
            return [None] * self.numberOfParams()
        derivs = []

        extent = patch0.getExtent()
        
        # derivatives wrt position

        ## FIXME -- would we be better to do central differences in
        ## pixel space, and convert to Position via CD matrix?

        psteps = pos0.getStepSizes()
        if not self.isParamFrozen('pos'):
            params = pos0.getParams()
            if counts == 0:
                derivs.extend([None] * len(params))
                psteps = []
            for i,pstep in enumerate(psteps):
                oldval = pos0.setParam(i, params[i]+pstep)
                (px,py) = img.getWcs().positionToPixel(pos0, self)
                pos0.setParam(i, oldval)

                patchx = self.getUnitFluxModelPatch(img, px, py, minval=minval,
                                                    extent=extent, modelMask=modelMask)
                if patchx is None or patchx.getImage() is None:
                    derivs.append(None)
                    continue

                # Plot derivatives.
                if False:
                    import pylab as plt
                    plt.clf()
                    global plotnum
                    plt.subplot(2,2,1)
                    mn = patch0.patch.min()
                    mx = patch0.patch.max()
                    plt.imshow(patch0.patch, extent=patch0.getExtent(),
                               interpolation='nearest', origin='lower',
                               vmin=mn, vmax=mx)
                    plt.colorbar()
                    plt.subplot(2,2,2)
                    plt.imshow(patchx.patch, extent=patchx.getExtent(),
                               interpolation='nearest', origin='lower',
                               vmin=mn, vmax=mx)
                    plt.colorbar()
                    plt.subplot(2,2,3)
                    diff = patchx.patch - patch0.patch
                    mx = np.max(np.abs(diff))
                    plt.imshow(diff, extent=patchx.getExtent(),
                               interpolation='nearest', origin='lower',
                               vmin=-mx, vmax=mx)
                    plt.colorbar()
                    fn = 'galdiff-%03i.png' % plotnum
                    plt.suptitle('dpos of %s in %s' % (str(self), img.name))
                    plt.savefig(fn)
                    print('wrote', fn)
                    plotnum += 1
                               
                dx = (patchx - patch0) * (counts / pstep)

                if modelMask is None:
                    # We evaluated patch0 and patchx on the same extent,
                    # so they are pixel aligned.  Take the intersection of
                    # the pixels they evaluated (>minval) to avoid jumps.
                    dx.patch *= ((patch0.patch > 0) * (patchx.patch > 0))

                dx.setName('d(%s)/d(pos%i)' % (self.dname, i))
                derivs.append(dx)

        # derivatives wrt brightness
        bsteps = self.brightness.getStepSizes()
        if not self.isParamFrozen('brightness'):
            params = self.brightness.getParams()
            for i,bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, params[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                df = patch0 * ((countsi - counts) / bstep)
                df.setName('d(%s)/d(bright%i)' % (self.dname, i))
                derivs.append(df)

        # derivatives wrt shape
        gsteps = self.shape.getStepSizes()
        if not self.isParamFrozen('shape'):
            gnames = self.shape.getParamNames()
            oldvals = self.shape.getParams()
            # print('Galaxy.getParamDerivatives:', self.getName())
            # print('  oldvals:', oldvals)
            if counts == 0:
                derivs.extend([None] * len(oldvals))
                gsteps = []
            for i,gstep in enumerate(gsteps):
                oldval = self.shape.setParam(i, oldvals[i]+gstep)
                #print('  stepped', gnames[i], 'by', gsteps[i],)
                #print('to get', self.shape)
                patchx = self.getUnitFluxModelPatch(img, px0, py0, minval=minval,
                                                    extent=extent, modelMask=modelMask)
                self.shape.setParam(i, oldval)
                if patchx is None:
                    print('patchx is None:')
                    print('  ', self)
                    print('  stepping galaxy shape', self.shape.getParamNames()[i])
                    print('  stepped', gsteps[i])
                    print('  to', self.shape.getParams()[i])
                    derivs.append(None)
                    continue

                dx = (patchx - patch0) * (counts / gstep)
                dx.setName('d(%s)/d(%s)' % (self.dname, gnames[i]))
                derivs.append(dx)
        return derivs


from astrometry.util.plotutils import *
psfft = PlotSequence('fft')

do_fft_timing = False
if do_fft_timing:
    from astrometry.util.ttime import *
    fft_timing = []
    fft_timing_id = 0

    
class ProfileGalaxy(object):
    '''
    A mix-in class that renders itself based on a Mixture-of-Gaussians
    profile.
    '''
    def getName(self):
        return 'ProfileGalaxy'

    def getProfile(self):
        return None

    # Here's the main method to override;
    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        return None
    
    def _getUnitFluxDeps(self, img, px, py):
        return None

    def _getUnitFluxPatchSize(self, img, minval):
        return 0
    
    def getUnitFluxModelPatch(self, img, px=None, py=None, minval=0.0,
                              extent=None, modelMask=None):
        if px is None or py is None:
            (px,py) = img.getWcs().positionToPixel(self.getPosition(), self)
        #
        if _galcache is None:
            return self._realGetUnitFluxModelPatch(img, px, py, minval,
                                                   extent=extent, modelMask=modelMask)
        
        deps = self._getUnitFluxDeps(img, px, py)
        #print('deps', deps, '->',)
        #deps = hash(deps)
        #print(deps)

        try:
            # FIXME -- what about when the extent was specified for
            # the cached entry but not specified for this call?
            #print('Searching cache for:', deps)
            (cached,mv) = _galcache.get(deps)
            # if cached is None:
            #     print('Cache hit:', cached)
            # else:
            #     print('Cache hit:', cached.shape)
            #     if modelMask is not None:
            #         assert(cached.shape == modelMask.shape)

            if mv <= minval:
                if extent is None:
                    if cached is None:
                        return None
                    return cached.copy()
                if cached is not None:
                    # do the extents overlap?
                    (x0,x1,y0,y1) = extent
                    (cx0,cx1,cy0,cy1) = cached.getExtent()
                    if cx0 <= x0 and cx1 >= x1 and cy0 <= y0 and cy1 >= y1:
                        pat = cached.copy()
                        if cx0 != x0 or cx1 != x1 or cy0 != y0 or cy1 != y1:
                            pat.clipToRoi(*extent)
                        return pat
        except KeyError:
            pass

        patch = self._realGetUnitFluxModelPatch(img, px, py, minval,
                                                extent=extent, modelMask=modelMask)
        if patch is not None:
            patch = patch.copy()
        # print('Adding to cache:', deps,)
        # if patch is not None:
        #     print('patch shape', patch.shape)
        # else:
        #     print('patch is None')
        if patch is not None and modelMask is not None:
            assert(patch.shape == modelMask.shape)
        # print('modelMask:', modelMask)
        _galcache.put(deps, (patch,minval))
        return patch

    def _realGetUnitFluxModelPatch(self, img, px, py, minval, extent=None,
                                   modelMask=None):
        '''
        extent: if not None, [x0,x1,y0,y1], where the range to render
        is [x0, x1), [y0,y1).
        '''

        #####
        global psfft

        if do_fft_timing:
            global fft_timing
            global fft_timing_id
            fft_timing_id += 1
            timing_id = fft_timing_id
            tpatch = CpuMeas()
            fft_timing.append((timing_id, 'get_unit_patch', 0,
                               (self,)))
        
        if modelMask is None:
            # now choose the patch size
            halfsize = self._getUnitFluxPatchSize(img, px, py, minval)

        if modelMask is not None:
            x0,y0 = modelMask.x0, modelMask.y0
        elif extent is None:
            # find overlapping pixels to render
            (outx, inx) = get_overlapping_region(
                int(np.floor(px-halfsize)), int(np.ceil(px+halfsize+1)),
                0, img.getWidth())
            (outy, iny) = get_overlapping_region(
                int(np.floor(py-halfsize)), int(np.ceil(py+halfsize+1)),
                0, img.getHeight())
            if inx == [] or iny == []:
                # no overlap

                if do_fft_timing:
                    fft_timing.append((timing_id, 'no_overlap', CpuMeas().cpu_seconds_since(tpatch)))

                return None
            x0,x1 = outx.start, outx.stop
            y0,y1 = outy.start, outy.stop
        else:
            x0,x1,y0,y1 = extent
        psf = img.getPsf()

        # We have two methods of rendering profile galaxies: If the
        # PSF can be represented as a mixture of Gaussians, then we do
        # the analytic Gaussian convolution, producing a larger
        # mixture of Gaussians, and we render that.  Otherwise
        # (pixelized PSFs), we FFT the PSF, multiply by the analytic
        # FFT of the galaxy, and IFFT back to get the rendered
        # profile.

        if hasattr(psf, 'getMixtureOfGaussians'):
            amix = self._getAffineProfile(img, px, py)
            # now convolve with the PSF, analytically
            psfmix = psf.getMixtureOfGaussians(px=px, py=py)
            cmix = amix.convolve(psfmix)

            # print('galaxy affine mixture:', amix)
            # print('psf mixture:', psfmix)
            # print('convolved mixture:', cmix)
            # print('_realGetUnitFluxModelPatch: extent', x0,x1,y0,y1)
            if modelMask is None:
                return mp.mixture_to_patch(cmix, x0, x1, y0, y1, minval,
                                           exactExtent=(extent is not None))
            else:
                # The convolved mixture *already* has the px,py offset added
                # (via px,py to amix) so set px,py=0,0 in this call.
                p = cmix.evaluate_grid_masked(x0, y0, modelMask.patch, 0., 0.)
                assert(p.shape == modelMask.shape)
                return p


        # Otherwise, FFT:
        imh,imw = img.shape

        haveExtent = (modelMask is not None) or (extent is not None)

        if not haveExtent:
            halfsize = self._getUnitFluxPatchSize(img, px, py, minval)

            # Avoid huge galaxies -> huge halfsize in a tiny image (blob)
            imsz = max(imh,imw)
            halfsize = min(halfsize, imsz)

        else:
            # FIXME -- max of modelMask, PSF, and Galaxy sizes!

            if modelMask is not None:
                mh,mw = modelMask.shape
                x1 = x0 + mw
                y1 = y0 + mh
            else:
                # x0,x1,y0,y1 were set to extent, above.
                mw = x1 - x0
                mh = y1 - y0

            # is the source center outside the modelMask?
            sourceOut = (px < x0 or px > x1-1 or py < y0 or py > y1-1)
            
            if sourceOut:
                # FIXME -- could also *think* about switching to a
                # Gaussian approximation when very far from the source
                # center...
                #
                #print('modelMask does not contain source center!  Fetching bigger model...')
                # If the source is *way* outside the patch, return zero.
                neardx,neardy = 0., 0.
                if px < x0:
                    neardx = x0 - px
                if px > x1:
                    neardx = px - x1
                if py < y0:
                    neardy = y0 - py
                if py > y1:
                    neardy = py - y1
                nearest = np.hypot(neardx, neardy)
                if nearest > self.getRadius():

                    if do_fft_timing:
                        fft_timing.append((timing_id, 'source_way_outside', CpuMeas().cpu_seconds_since(tpatch)))
                    return None

                # how far is the furthest point from the source center?
                farw = max(abs(x0 - px), abs(x1 - px))
                farh = max(abs(y0 - py), abs(y1 - py))
                bigx0 = int(np.floor(px - farw))
                bigx1 = int(np.ceil (px + farw))
                bigy0 = int(np.floor(py - farh))
                bigy1 = int(np.ceil (py + farh))
                bigw = 1 + bigx1 - bigx0
                bigh = 1 + bigy1 - bigy0
                boffx = x0 - bigx0
                boffy = y0 - bigy0
                assert(bigw >= mw)
                assert(bigh >= mh)
                assert(boffx >= 0)
                assert(boffy >= 0)
                if modelMask is None:
                    bigMask = np.ones((bigh,bigw), bool)
                else:
                    bigMask = np.zeros((bigh,bigw), bool)
                    bigMask[boffy:boffy+mh, boffx:boffx+mw] = modelMask.patch
                bigMask = Patch(bigx0, bigy0, bigMask)

                if do_fft_timing:
                    fft_timing.append((timing_id, 'calling_sourceout', None))
                    t0 = CpuMeas()

                # print('Recursing:', self, ':', (mh,mw), 'to', (bigh,bigw))
                bigmodel = self._realGetUnitFluxModelPatch(
                    img, px, py, minval, extent=None, modelMask=bigMask)

                if do_fft_timing:
                    t1 = CpuMeas()
                    fft_timing.append((timing_id, 'sourceout', t1.cpu_seconds_since(t0),
                                       (bigMask.shape, (mh,mw))))

                return Patch(x0, y0,
                             bigmodel.patch[boffy:boffy+mh, boffx:boffx+mw])
            
            halfsize = max(mh/2., mw/2.)

            psfh,psfw = psf.shape
            halfsize = max(halfsize, max(psfw/2., psfh/2.))

        if do_fft_timing:
            t0 = CpuMeas()

        P,(cx,cy),(pH,pW),(v,w) = psf.getFourierTransform(px, py, halfsize)

        #print('Computing', self, ': halfsize=', halfsize, 'FFT', (pH,pW))
        
        if do_fft_timing:
            t1 = CpuMeas()
            fft_timing.append((timing_id, 'psf_fft', t1.cpu_seconds_since(t0),
                               (haveExtent, halfsize)))

        # print('PSF Fourier transform size:', P.shape)
        # print('Padded size:', pH,pW)
        # print('PSF center in padded image:', cx,cy)
        # print('Source center px,py', px,py)

        # One example:
        # pH,pW = (256, 256)
        # P.shape = (256, 129)
        # cx,cy = (127, 127)
        
        dx = px - cx
        dy = py - cy
        if haveExtent:
            # the Patch we return *must* have this origin.
            ix0 = x0
            iy0 = y0

            # Put the difference into the galaxy FFT.
            mux = dx - ix0
            muy = dy - iy0

            # ASSUME square PSF
            assert(pH == pW)
            psfh,psfw = psf.shape
            # How much padding on the PSF image?
            psfmargin = cx - psfw/2

            gx0 = gy0 = 0
            if abs(mux) >= psfmargin or abs(muy) >= psfmargin:
                # Wrap-around is possible (likely).  Compute a shifted image
                # and then copy it into the result.
                gx0 = int(np.round(mux))
                gy0 = int(np.round(muy))
                mux -= gx0
                muy -= gy0
                
        else:
            # Put the integer portion of the offset into Patch x0,y0
            ix0 = int(np.round(dx))
            iy0 = int(np.round(dy))

            # Put the subpixel portion into the galaxy FFT.
            mux = dx - ix0
            muy = dy - iy0

        if do_fft_timing:
            t0 = CpuMeas()
        
        amix = self._getAffineProfile(img, mux, muy)

        if do_fft_timing:
            t1 = CpuMeas()
        
        Fsum = amix.getFourierTransform(v, w)
        
        if do_fft_timing:
            t2 = CpuMeas()

            fft_timing.append((timing_id, 'get_affine', t1.cpu_seconds_since(t0),
                               (haveExtent,)))
            fft_timing.append((timing_id, 'get_ft', t2.cpu_seconds_since(t1),
                               (haveExtent,)))

        if False:
            # for fakedx in []:#0]:#, 1, 10]:

            amix2 = self._getAffineProfile(img, mux + fakedx, muy)
            Fsum2 = amix2.getFourierTransform(v, w)

            plt.clf()
            plt.subplot(3,3,1)
            plt.imshow(Fsum2.real, interpolation='nearest', origin='lower')
            plt.title('Galaxy FFT')
            plt.subplot(3,3,2)
            plt.imshow(P.real, interpolation='nearest', origin='lower')
            plt.title('PSF FFT')
            plt.subplot(3,3,3)
            plt.imshow((Fsum2 * P).real,
                       interpolation='nearest', origin='lower')
            plt.title('Galaxy * PSF FFT')
            plt.subplot(3,3,4)
            plt.imshow(Fsum2.imag, interpolation='nearest', origin='lower')
            plt.title('Galaxy FFT')
            plt.subplot(3,3,5)
            plt.imshow(P.imag, interpolation='nearest', origin='lower')
            plt.title('PSF FFT')
            plt.subplot(3,3,6)
            plt.imshow((Fsum2 * P).imag,
                       interpolation='nearest', origin='lower')
            plt.title('Galaxy * PSF FFT')

            plt.subplot(3,3,7)
            plt.imshow(np.fft.irfft2(Fsum2, s=(pH,pW)),
                       interpolation='nearest', origin='lower')
            plt.title('iFFT Galaxy')
            plt.subplot(3,3,8)
            plt.imshow(np.fft.irfft2(P, s=(pH,pW)),
                       interpolation='nearest', origin='lower')
            plt.title('iFFT PSF')
            plt.subplot(3,3,9)
            plt.imshow(np.fft.irfft2(Fsum2*P, s=(pH,pW)),
                       interpolation='nearest', origin='lower')
            plt.title('iFFT Galaxy*PSF')
            
            plt.suptitle('dx = %i pixel' % fakedx)
            psfft.savefig()

        
        
        if haveExtent:

            if False:
                plt.clf()
                plt.imshow(np.fft.irfft2(Fsum * P, s=(pH,pW)),
                           interpolation='nearest', origin='lower')
                plt.title('iFFT in PSF shape')
                psfft.savefig()

            if do_fft_timing:
                t0 = CpuMeas()

            G = np.fft.irfft2(Fsum * P, s=(pH,pW))

            if do_fft_timing:
                t1 = CpuMeas()
                fft_timing.append((timing_id, 'irfft2', t1.cpu_seconds_since(t0),
                                   (haveExtent, (pH,pW))))
            

            gh,gw = G.shape

            if gx0 != 0 or gy0 != 0:
                print('gx0,gy0', gx0,gy0)
                yi,yo = get_overlapping_region(-gy0, -gy0+mh-1, 0, gh-1)
                xi,xo = get_overlapping_region(-gx0, -gx0+mw-1, 0, gw-1)

                # shifted
                shG = np.zeros((mh,mw), G.dtype)
                shG[yo,xo] = G[yi,xi]
                G = shG
            
            if gh > mh or gw > mw:
                G = G[:mh,:mw]
            if modelMask is not None:
                assert(G.shape == modelMask.shape)
            else:
                assert(G.shape == (mh,mw))
            
        else:
            #print('iFFT', (pW,pH))

            # psfim = np.fft.irfft2(P)
            # print('psf iFFT', psfim.shape, psfim.sum())
            # psfim /= psfim.sum()
            # xx,yy = np.meshgrid(np.arange(pW), np.arange(pH))
            # print('centroid', np.sum(psfim*xx), np.sum(psfim*yy))

            if do_fft_timing:
                t0 = CpuMeas()

            G = np.fft.irfft2(Fsum * P, s=(pH,pW))

            if do_fft_timing:
                t1 = CpuMeas()
                fft_timing.append((timing_id, 'irfft2_b', t1.cpu_seconds_since(t0),
                                   (haveExtent, (pH,pW))))

            # print('Evaluating iFFT with shape', pH,pW)
            # print('G shape:', G.shape)

            if False:
                plt.clf()
                plt.imshow(G, interpolation='nearest', origin='lower')
                plt.title('iFFT in padded PSF shape')
                psfft.savefig()
            
            # Clip down to suggested "halfsize"
            if x0 > ix0:
                G = G[:,x0 - ix0:]
                ix0 = x0
            if y0 > iy0:
                G = G[y0 - iy0:, :]
                iy0 = y0
            gh,gw = G.shape
            if gw+ix0 > x1:
                G = G[:,:x1-ix0]
            if gh+iy0 > y1:
                G = G[:y1-iy0,:]

        if do_fft_timing:
            fft_timing.append((timing_id, 'get_unit_patch_finished', CpuMeas().cpu_seconds_since(tpatch),
                               (self,)))

        return Patch(ix0, iy0, G)
                    

class HoggGalaxy(ProfileGalaxy, Galaxy):

    def getName(self):
        return 'HoggGalaxy'

    def getRadius(self):
        return self.nre * self.shape.re
    
    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        # shift and squash
        cd = img.getWcs().cdAtPixel(px, py)
        galmix = self.getProfile()
        Tinv = np.linalg.inv(self.shape.getTensor(cd))
        amix = galmix.apply_affine(np.array([px,py]), Tinv.T)
        amix.symmetrize()
        return amix

    def _getUnitFluxDeps(self, img, px, py):
        # return ('unitpatch', self.getName(), px, py,
        return hash(('unitpatch', self.getName(), px, py,
                img.getWcs().hashkey(),
                img.getPsf().hashkey(), self.shape.hashkey())
                    )

    def _getUnitFluxPatchSize(self, img, px, py, minval):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        cd = img.getWcs().cdAtPixel(px, py)
        pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
        halfsize = max(1., self.getRadius() / 3600. / pixscale)
        psf = img.getPsf()
        halfsize += psf.getRadius()
        halfsize = int(np.ceil(halfsize))
        return halfsize

class GaussianGalaxy(HoggGalaxy):
    nre = 6.
    profile = mp.MixtureOfGaussians(np.array([1.]), np.zeros((1,2)),
                                    np.array([[[1.,0.],[0.,1.]]]))
    profile.normalize()
    def __init__(self, *args, **kwargs):
        self.nre = GaussianGalaxy.nre
        super(GaussianGalaxy,self).__init__(*args, **kwargs)
    def getName(self):
        return 'GaussianGalaxy'
    def getProfile(self):
        return GaussianGalaxy.profile

class ExpGalaxy(HoggGalaxy):
    nre = 4.
    profile = mp.get_exp_mixture()
    profile.normalize()
    @staticmethod
    def getExpProfile():
        return ExpGalaxy.profile
    def __init__(self, *args, **kwargs):
        self.nre = ExpGalaxy.nre
        super(ExpGalaxy,self).__init__(*args, **kwargs)
    def getName(self):
        return 'ExpGalaxy'
    def getProfile(self):
        return ExpGalaxy.profile

class DevGalaxy(HoggGalaxy):
    nre = 8.
    profile = mp.get_dev_mixture()
    profile.normalize()
    @staticmethod
    def getDevProfile():
        return DevGalaxy.profile
    def __init__(self, *args, **kwargs):
        self.nre = DevGalaxy.nre
        super(DevGalaxy,self).__init__(*args, **kwargs)
    def getName(self):
        return 'DevGalaxy'
    def getProfile(self):
        return DevGalaxy.profile

class FracDev(ScalarParam):
    stepsize = 0.01
    def clipped(self):
        f = self.getValue()
        return np.clip(f, 0., 1.)

    def derivative(self):
        return 1.

class SoftenedFracDev(FracDev):
    '''
    Implements a "softened" version of the deV-to-total fraction.

    The sigmoid function is scaled and shifted so that S(0) ~ 0.1 and
    S(1) ~ 0.9.

    Use the 'clipped()' function to get the un-softened fracDev
    clipped to [0,1].
    '''
    def clipped(self):
        f = self.getValue()
        return 1./(1 + np.exp(4.*(0.5 - f)))
    
    def derivative(self):
        f = self.getValue()
        # Thanks, Sage
        ef = np.exp(-4.*f + 2)
        return 4. * ef / ((ef + 1)**2)

    
class FixedCompositeGalaxy(MultiParams, ProfileGalaxy, SingleProfileSource):
    '''
    A galaxy with Exponential and deVaucouleurs components where the
    brightnesses of the deV and exp components are defined in terms of
    a total brightness and a fraction of that total that goes to the
    deV component.
    
    The two components share a position (ie the centers are the same),
    but have different shapes.  The galaxy has a single brightness
    that is split between the components.

    This is like CompositeGalaxy, but more useful for getting
    consistent colors from forced photometry, because one can freeze
    the fracDev to keep the deV+exp profile fixed.
    '''
    def __init__(self, pos, brightness, fracDev, shapeExp, shapeDev):
        # handle passing fracDev as a float
        if not isinstance(fracDev, BaseParams):
            fracDev = FracDev(fracDev)
        MultiParams.__init__(self, pos, brightness, fracDev,
                             shapeExp, shapeDev)
        self.name = self.getName()

    def getRadius(self):
        return max(self.shapeExp.re * ExpGalaxy.nre, self.shapeDev.re * DevGalaxy.nre)

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1, fracDev=2, shapeExp=3, shapeDev=4)

    def getName(self):
        return 'FixedCompositeGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness) 
                + ', ' + str(self.fracDev)
                + ', exp ' + str(self.shapeExp)
                + ', deV ' + str(self.shapeDev))
    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) +
                ', fracDev=' + repr(self.fracDev) + 
                ', shapeExp=' + repr(self.shapeExp) +
                ', shapeDev=' + repr(self.shapeDev) + ')')

    def _getAffineProfile(self, img, px, py):
        f = self.fracDev.clipped()
        profs = []
        if f > 0.:
            profs.append((f, DevGalaxy.profile, self.shapeDev))
        if f < 1.:
            profs.append((1.-f, ExpGalaxy.profile, self.shapeExp))

        cd = img.getWcs().cdAtPixel(px, py)
        mix = []
        for f,p,s in profs:
            Tinv = np.linalg.inv(s.getTensor(cd))
            amix = p.apply_affine(np.array([px,py]), Tinv.T)
            amix.symmetrize()
            amix.amp *= f
            mix.append(amix)
            #print('affine profile: shape', s, 'weight', f, '->', amix)
            #print('amp sum:', np.sum(amix.amp))
        if len(mix) == 1:
            return mix[0]
        smix = mix[0] + mix[1]
        #print('Summed profiles:', smix)
        #print('amp sum', np.sum(smix.amp))
        return mix[0] + mix[1]

    def _getUnitFluxPatchSize(self, img, px, py, minval):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        cd = img.getWcs().cdAtPixel(px, py)
        pixscale = np.sqrt(np.abs(np.linalg.det(cd)))
        f = self.fracDev.clipped()
        r = 1.
        if f < 1.:
            s = self.shapeExp
            rexp = ExpGalaxy.nre * s.re
            r = max(r, rexp)
        if f > 0.:
            s = self.shapeDev
            rdev = max(r, DevGalaxy.nre * s.re)
            r = max(r, rdev)
        halfsize = r / 3600. / pixscale
        psf = img.getPsf()
        halfsize += psf.getRadius()
        return halfsize
    
    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(),
                     px, py, img.getWcs().hashkey(),
                     img.getPsf().hashkey(),
                     self.shapeDev.hashkey(),
                     self.shapeExp.hashkey(),
                     self.fracDev.hashkey()))
    
    def getParamDerivatives(self, img, modelMask=None):
        e = ExpGalaxy(self.pos, self.brightness, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightness, self.shapeDev)
        e.dname = 'fcomp.exp'
        d.dname = 'fcomp.dev'

        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')

        dexp = e.getParamDerivatives(img, modelMask=modelMask)
        ddev = d.getParamDerivatives(img, modelMask=modelMask)

        # print('FixedCompositeGalaxy.getParamDerivatives.')
        # print('tim shape', img.shape)
        # print('exp deriv extents:')
        # for deriv in dexp + ddev:
        #     print('  ', deriv.name, deriv.getExtent())

        # fracDev scaling
        f = self.fracDev.clipped()
        for deriv in dexp:
            if deriv is not None:
                deriv *= (1.-f)
        for deriv in ddev:
            if deriv is not None:
                deriv *= f
        
        derivs = []
        i0 = 0
        if not self.isParamFrozen('pos'):
            # "pos" is shared between the models, so add the derivs.
            npos = self.pos.numberOfParams()
            for i in range(npos):
                ii = i0+i
                dsum = add_patches(dexp[ii], ddev[ii])
                if dsum is not None: 
                    dsum.setName('d(fcomp)/d(pos%i)' % i)
                derivs.append(dsum)
            i0 += npos

        if not self.isParamFrozen('brightness'):
            # shared between the models, so add the derivs.
            nb = self.brightness.numberOfParams()
            for i in range(nb):
                ii = i0+i
                dsum = add_patches(dexp[ii], ddev[ii])
                if dsum is not None: 
                    dsum.setName('d(fcomp)/d(bright%i)' % i)
                derivs.append(dsum)
            i0 += nb
                
        if not self.isParamFrozen('fracDev'):
            counts = img.getPhotoCal().brightnessToCounts(self.brightness)
            if counts == 0.:
                derivs.append(None)
            else:
                ## FIXME -- should be possible to avoid recomputing these...
                ue = e.getUnitFluxModelPatch(img, modelMask=modelMask)
                ud = d.getUnitFluxModelPatch(img, modelMask=modelMask)

                df = self.fracDev.derivative()

                if ue is not None:
                    ue *= -df
                if ud is not None:
                    ud *= +df
                df = add_patches(ud, ue)
                if df is None:
                    derivs.append(None)
                else:
                    df *= counts
                    df.setName('d(fcomp)/d(fracDev)')
                    derivs.append(df)

        if not self.isParamFrozen('shapeExp'):
            derivs.extend(dexp[i0:])
        if not self.isParamFrozen('shapeDev'):
            derivs.extend(ddev[i0:])

        # print('Returning derivs:')
        # for deriv in derivs:
        #     print('  ', deriv.name, deriv.getExtent())
            
        return derivs


class CompositeGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''
    def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2, brightnessDev=3, shapeDev=4)

    def getName(self):
        return 'CompositeGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' ' + str(self.shapeExp)
                + ' and deV ' + str(self.brightnessDev) + ' ' + str(self.shapeDev))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) + 
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev))

    def getBrightness(self):
        ''' This makes some assumptions about the ``Brightness`` / ``PhotoCal`` and
        should be treated as approximate.'''
        return self.brightnessExp + self.brightnessDev

    def getBrightnesses(self):
        return [self.brightnessExp, self.brightnessDev]

    def _getModelPatches(self, img, minsb=0., modelMask=None):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if minsb == 0.:
            kw = {}
        else:
            kw = dict(minsb=minsb/2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe,pd)
    
    def getModelPatch(self, img, minsb=0., modelMask=None):
        pe,pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask)
        return add_patches(pe,pd)

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None):
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        return (e.getUnitFluxModelPatches(img, minval=minval, modelMask=modelMask) +
                d.getUnitFluxModelPatches(img, minval=minval, modelMask=modelMask))

    def getUnitFluxModelPatch(self, img, px=None, py=None, modelMask=None):
        # this code is un-tested
        assert(False)
        fe = self.brightnessExp / (self.brightnessExp + self.brightnessDev)
        fd = 1. - fe
        assert(fe >= 0.)
        assert(fe <= 1.)
        e = ExpGalaxy(self.pos, fe, self.shapeExp)
        d = DevGalaxy(self.pos, fd, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        pe = e.getModelPatch(img, px, py)
        pd = d.getModelPatch(img, px, py)
        if pe is None:
            return pd
        if pd is None:
            return pe
        return pe + pd

    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None):
        #print('CompositeGalaxy: getParamDerivatives')
        #print('  Exp brightness', self.brightnessExp, 'shape', self.shapeExp)
        #print('  Dev brightness', self.brightnessDev, 'shape', self.shapeDev)
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        e.dname = 'comp.exp'
        d.dname = 'comp.dev'
        if self.isParamFrozen('pos'):
            e.freezeParam('pos')
            d.freezeParam('pos')
        if self.isParamFrozen('brightnessExp'):
            e.freezeParam('brightness')
        if self.isParamFrozen('shapeExp'):
            e.freezeParam('shape')
        if self.isParamFrozen('brightnessDev'):
            d.freezeParam('brightness')
        if self.isParamFrozen('shapeDev'):
            d.freezeParam('shape')

        de = e.getParamDerivatives(img, modelMask=modelMask)
        dd = d.getParamDerivatives(img, modelMask=modelMask)

        if self.isParamFrozen('pos'):
            derivs = de + dd
        else:
            derivs = []
            # "pos" is shared between the models, so add the derivs.
            npos = len(self.pos.getStepSizes())
            for i in range(npos):
                dp = add_patches(de[i], dd[i])
                if dp is not None: 
                    dp.setName('d(comp)/d(pos%i)' % i)
                derivs.append(dp)
            derivs.extend(de[npos:])
            derivs.extend(dd[npos:])

        return derivs

if __name__ == '__main__':
    from astrometry.util.plotutils import PlotSequence
    import matplotlib
    from basics import GaussianMixturePSF, PixPos, Flux, NullPhotoCal, NullWCS, ConstantSky
    from engine import Image
    matplotlib.use('Agg')
    import pylab as plt
    ps = PlotSequence('gal')
    
    # example PSF (from WISE W1 fit)
    w = np.array([ 0.77953706,  0.16022146,  0.06024237])
    mu = np.array([[-0.01826623, -0.01823262],
                   [-0.21878855, -0.0432496 ],
                   [-0.83365747, -0.13039277]])
    sigma = np.array([[[  7.72925584e-01,   5.23305564e-02],
                       [  5.23305564e-02,   8.89078473e-01]],
                       [[  9.84585869e+00,   7.79378820e-01],
                       [  7.79378820e-01,   8.84764455e+00]],
                       [[  2.02664489e+02,  -8.16667434e-01],
                        [ -8.16667434e-01,   1.87881670e+02]]])
    
    psf = GaussianMixturePSF(w, mu, sigma)

    shape = GalaxyShape(10., 0.5, 30.)
    pos = PixPos(100, 50)
    bright = Flux(1000.)
    egal = ExpGalaxy(pos, bright, shape)

    data = np.zeros((100, 200))
    tim = Image(data=data, psf=psf)

    p0 = egal.getModelPatch(tim)
    
    p1 = egal.getModelPatch(tim, 1e-3)

    bright.setParams([100.])

    p2 = egal.getModelPatch(tim, 1e-3)

    print('p0', p0.patch.sum())
    print('p1', p1.patch.sum())
    print('p2', p2.patch.sum())
    
    plt.clf()
    ima = dict(interpolation='nearest', origin='lower')
    plt.subplot(2,2,1)
    plt.imshow(np.log10(np.maximum(1e-16, p0.patch)), **ima)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.log10(np.maximum(1e-16, p1.patch)), **ima)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(np.log10(np.maximum(1e-16, p2.patch)), **ima)
    plt.colorbar()
    ps.savefig()
