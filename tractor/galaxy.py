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

from tractor import mixture_profiles as mp
from tractor.utils import ParamList, MultiParams, ScalarParam, BaseParams
from tractor.patch import Patch, add_patches, ModelMask
from tractor.basics import SingleProfileSource, BasicSource
import time
#from tractor.utils import savetxt_cpu_append

debug_ps = None


def get_galaxy_cache():
    return None


def set_galaxy_cache_size(N=10000):
    # Ugh, dealing with caching + extents / modelMasks was too much
    pass


enable_galaxy_cache = set_galaxy_cache_size

def disable_galaxy_cache():
    pass


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
        re_deg = max(1. / 30, self.re) / 3600.
        cp = np.cos(phi)
        sp = np.sin(phi)
        # Squish, rotate, and scale into degrees.
        # resulting G takes unit vectors (in r_e) to degrees
        # (~intermediate world coords)
        return re_deg * np.array([[cp, sp * self.ab], [-sp, cp * self.ab]])


class Galaxy(MultiParams, SingleProfileSource):
    '''
    Generic Galaxy profile with position, brightness, and shape.
    '''

    def __init__(self, *args):
        super().__init__(*args)
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
                ', shape=' + repr(self.shape) + ')')

    def getUnitFluxModelPatch(self, img, **kwargs):
        raise RuntimeError('getUnitFluxModelPatch unimplemented in' +
                           self.getName())

    # returns [ Patch, Patch, ... ] of length numberOfParams().
    # Galaxy.
    def getParamDerivatives(self, img, modelMask=None, **kwargs):
        pos0 = self.getPosition()
        wcs = img.getWcs()
        (px0, py0) = wcs.positionToPixel(pos0, self)
        counts = img.getPhotoCal().brightnessToCounts(self.brightness)

        minsb = img.modelMinval
        if counts > 0:
            minval = minsb / counts
        else:
            minval = None

        padded = False
        if modelMask is not None:
            # grow mask by 1 pixel in each direction (for spatial derivs)
            mh,mw = modelMask.shape
            mm = ModelMask(modelMask.x0 - 1, modelMask.y0 - 1, mw + 2, mh + 2)
            patch0 = self.getUnitFluxModelPatch(img, px=px0, py=py0, minval=minval,
                                                modelMask=mm, **kwargs)
            padded = True
        else:
            patch0 = self.getUnitFluxModelPatch(img, px=px0, py=py0, minval=minval,
                                                modelMask=modelMask, **kwargs)
        if patch0 is None:
            return [None] * self.numberOfParams()

        if modelMask is None:
            x0,x1,y0,y1 = patch0.getExtent()
            modelMask = ModelMask.fromExtent(x0,x1,y0,y1)
            #modelMask = ModelMask.fromExtent([x0,x1,y0,y1])
        assert(modelMask is not None)

        derivs = []
        # derivatives wrt position
        if not self.isParamFrozen('pos'):
            if counts == 0:
                derivs.extend([None] * len(pos0.getParams()))
            else:
                p0 = patch0.patch

                if padded:
                    dx = (p0[1:-1,   :-2] - p0[1:-1, 2:  ]) / 2.
                    dy = (p0[ :-2 , 1:-1] - p0[2:  , 1:-1]) / 2.
                    assert(dx.shape == modelMask.shape)
                    assert(dy.shape == modelMask.shape)
                    x0,y0 = modelMask.x0,modelMask.y0
                    patchdx = Patch(x0, y0, dx)
                    patchdy = Patch(x0, y0, dy)
                    # Undo the padding on patch0.
                    patch0  = Patch(x0, y0, patch0.patch[1:-1, 1:-1])
                    assert(patch0.shape == modelMask.shape)
                else:
                    dx = np.zeros_like(p0)
                    dx[:,1:-1] = (p0[:, :-2] - p0[:, 2:]) / 2.
                    dy = np.zeros_like(p0)
                    dy[1:-1,:] = (p0[:-2, :] - p0[2:, :]) / 2.
                    patchdx = Patch(patch0.x0, patch0.y0, dx)
                    patchdy = Patch(patch0.x0, patch0.y0, dy)
                #savetxt_cpu_append('cdx.txt', dx)
                #savetxt_cpu_append('cdy.txt', dy)
                #savetxt_cpu_append('cp0.txt', p0)
                #savetxt_cpu_append('cpatch.txt', patch0.patch)
                del dx, dy
                derivs.extend(wcs.pixelDerivsToPositionDerivs(pos0, self, counts,
                                                              patch0, patchdx, patchdy))
                del patchdx, patchdy

        # derivatives wrt brightness
        if not self.isParamFrozen('brightness'):
            bsteps = self.brightness.getStepSizes()
            params = self.brightness.getParams()
            for i, bstep in enumerate(bsteps):
                oldval = self.brightness.setParam(i, params[i] + bstep)
                countsi = img.getPhotoCal().brightnessToCounts(self.brightness)
                self.brightness.setParam(i, oldval)
                if countsi == counts:
                    df = None
                else:
                    #print (f'{bstep=} {countsi=} {counts=}')
                    df = patch0 * ((countsi - counts) / bstep)
                    df.setName('d(%s)/d(bright%i)' % (self.dname, i))
                    #savetxt_cpu_append('cdf.txt', df.patch)
                derivs.append(df)

        # derivatives wrt shape
        if not self.isParamFrozen('shape'):
            gsteps = self.shape.getStepSizes()
            gnames = self.shape.getParamNames()
            oldvals = self.shape.getParams()
            if counts == 0:
                derivs.extend([None] * len(oldvals))
                gsteps = []
            for i, gstep in enumerate(gsteps):
                oldval = self.shape.setParam(i, oldvals[i] + gstep)
                patchx = self.getUnitFluxModelPatch(
                    img, px=px0, py=py0, minval=minval, modelMask=modelMask,
                    **kwargs)
                self.shape.setParam(i, oldval)
                if patchx is None:
                    print('patchx is None:')
                    print('  ', self)
                    print('  stepping galaxy shape',
                          self.shape.getParamNames()[i])
                    print('  stepped', gsteps[i])
                    print('  to', self.shape.getParams()[i])
                    derivs.append(None)
                    continue
                #print (f'{counts=} {gstep=}')
                dx = (patchx - patch0) * (counts / gstep)
                dx.setName('d(%s)/d(%s)' % (self.dname, gnames[i]))
                derivs.append(dx)
                #savetxt_cpu_append('cdx3.txt', dx.patch)
                #savetxt_cpu_append('cpatchx3.txt', patchx.patch)
                #savetxt_cpu_append('cpatch03.txt', patch0.patch)
        return derivs


class ProfileGalaxy(object):
    '''
    A mix-in class that renders itself based on a Mixture-of-Gaussians
    profile.
    '''

    def getName(self):
        return 'ProfileGalaxy'

    def getProfile(self):
        return None

    # Here are the two main methods to override;
    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        return None

    def _getShearedProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        shear-transformed into the pixel space of the image.
        At px,py (but not offset to px,py).
        '''
        return None

    def _getUnitFluxDeps(self, img, px, py):
        return None

    def _getUnitFluxPatchSize(self, img, **kwargs):
        return 0

    def getUnitFluxModelPatch(self, img, px=None, py=None, minval=0.0,
                              modelMask=None, **kwargs):
        if px is None or py is None:
            (px, py) = img.getWcs().positionToPixel(self.getPosition(), self)
        patch = self._realGetUnitFluxModelPatch(
            img, px, py, minval, modelMask=modelMask, **kwargs)
        if patch is not None and modelMask is not None:
            assert(patch.shape == modelMask.shape)
        return patch

    def _realGetUnitFluxModelPatch(self, img, px, py, minval, modelMask=None,
                                   inner_real_nsigma = 3.,
                                   outer_real_nsigma = 4.,
                                   force_halfsize=None,
                                   **kwargs):
        from astrometry.util.miscutils import get_overlapping_region
        if modelMask is not None:
            x0, y0 = modelMask.x0, modelMask.y0
        else:
            # choose the patch size
            halfsize = self._getUnitFluxPatchSize(img, px=px, py=py, minval=minval)
            # find overlapping pixels to render
            (outx, inx) = get_overlapping_region(
                int(np.floor(px - halfsize)), int(np.ceil(px + halfsize + 1)),
                0, img.getWidth())
            (outy, iny) = get_overlapping_region(
                int(np.floor(py - halfsize)), int(np.ceil(py + halfsize + 1)),
                0, img.getHeight())
            if inx == [] or iny == []:
                # no overlap
                return None
            x0, x1 = outx.start, outx.stop
            y0, y1 = outy.start, outy.stop

        psf = img.getPsf()

        # We have two methods of rendering profile galaxies: If the
        # PSF can be represented as a mixture of Gaussians, then we do
        # the analytic Gaussian convolution, producing a larger
        # mixture of Gaussians, and we render that.  Otherwise
        # (pixelized PSFs), we FFT the PSF, multiply by the analytic
        # FFT of the galaxy, and IFFT back to get the rendered
        # profile.

        # The "HybridPSF" class is just a marker to indicate whether this
        # code should treat the PSF as a hybrid.
        from tractor.psf import HybridPSF
        hybrid = isinstance(psf, HybridPSF)

        def run_mog(amix=None, mm=None):
            ''' This runs the mixture-of-Gaussians convolution method.
            '''
            if amix is None:
                amix = self._getAffineProfile(img, px, py)
            #print('Evaluating MoG mixture:', len(amix.amp))
            #print('amps:', amix.amp)
            if mm is None:
                mm = modelMask
            # now convolve with the PSF, analytically
            # (note that the psf's center is *not* set to px,py; that's just
            #  the position to use for spatially-varying PSFs)
            psfmix = psf.getMixtureOfGaussians(px=px, py=py)
            cmix = amix.convolve(psfmix)
            if mm is None:
                #print('Mixture to patch: amix', amix, 'psfmix', psfmix, 'cmix', cmix)
                return mp.mixture_to_patch(cmix, x0, x1, y0, y1, minval)
            # The convolved mixture *already* has the px,py offset added
            # (via px,py to amix) so set px,py=0,0 in this call.
            if mm.mask is not None:
                p = cmix.evaluate_grid_masked(mm.x0, mm.y0, mm.mask, 0., 0.)
            else:
                p = cmix.evaluate_grid(mm.x0, mm.x1, mm.y0, mm.y1, 0., 0.)
            assert(p.shape == mm.shape)
            return p

        if hasattr(psf, 'getMixtureOfGaussians') and not hybrid:
            return run_mog(mm=modelMask)

        # Otherwise, FFT:
        imh, imw = img.shape
        if modelMask is None:
            # Avoid huge galaxies -> huge halfsize in a tiny image (blob)
            imsz = max(imh, imw)
            halfsize = min(halfsize, imsz)
            # FIXME -- should take some kind of combination of
            # modelMask, PSF, and Galaxy sizes!

        else:
            # ModelMask sets the sizes.
            mh, mw = modelMask.shape
            x1 = x0 + mw
            y1 = y0 + mh

            halfsize = max(mh / 2., mw / 2.)
            # How far from the source center to furthest modelMask edge?
            # FIXME -- add 1 for Lanczos margin?
            halfsize = max(halfsize, max(max(1 + px - x0, 1 + x1 - px),
                                         max(1 + py - y0, 1 + y1 - py)))
            psfh, psfw = psf.shape
            halfsize = max(halfsize, max(psfw / 2., psfh / 2.))
            #print('Halfsize:', halfsize)
            if force_halfsize is not None:
                halfsize = force_halfsize
            #if not hasattr(img, 'halfsize'):
            #    img.halfsize = halfsize
            #elif int(np.ceil(np.log2(halfsize))) < int(np.ceil(np.log2(img.halfsize))):
            #    print (f'Adjusting halfsize from {halfsize=} to {img.halfsize=}')
            #    halfsize = img.halfsize
            #halfsize += 1
            # is the source center outside the modelMask?
            sourceOut = (px < x0 or px > x1 - 1 or py < y0 or py > y1 - 1)
            #print (f'{px=} {x0=} {x1=} {py=} {y0=} {y1=} {sourceOut=}')
            # print('mh,mw', mh,mw, 'sourceout?', sourceOut)
            sourceOut = False

            if sourceOut:
                if hybrid:
                    return run_mog(mm=modelMask)

                # Super Yuck -- FFT, modelMask, source is outside the
                # box.
                neardx, neardy = 0., 0.
                if px < x0:
                    neardx = x0 - px
                if px > x1:
                    neardx = px - x1
                if py < y0:
                    neardy = y0 - py
                if py > y1:
                    neardy = py - y1
                nearest = np.hypot(neardx, neardy)
                #print('Nearest corner:', nearest, 'vs radius', self.getRadius())
                if nearest > self.getRadius():
                    return None
                # how far is the furthest point from the source center?
                farw = max(abs(x0 - px), abs(x1 - px))
                farh = max(abs(y0 - py), abs(y1 - py))
                bigx0 = int(np.floor(px - farw))
                bigx1 = int(np.ceil(px + farw))
                bigy0 = int(np.floor(py - farh))
                bigy1 = int(np.ceil(py + farh))
                bigw = 1 + bigx1 - bigx0
                bigh = 1 + bigy1 - bigy0
                boffx = x0 - bigx0
                boffy = y0 - bigy0
                assert(bigw >= mw)
                assert(bigh >= mh)
                assert(boffx >= 0)
                assert(boffy >= 0)
                bigMask = np.zeros((bigh, bigw), bool)
                if modelMask.mask is not None:
                    bigMask[boffy:boffy + mh,
                            boffx:boffx + mw] = modelMask.mask
                else:
                    bigMask[boffy:boffy + mh, boffx:boffx + mw] = True
                bigMask = ModelMask(bigx0, bigy0, bigMask)
                # print('Recursing:', self, ':', (mh,mw), 'to', (bigh,bigw))
                bigmodel = self._realGetUnitFluxModelPatch(
                    img, px, py, minval, modelMask=bigMask)
                #print ("PATCH")
                return Patch(x0, y0,
                             bigmodel.patch[boffy:boffy + mh, boffx:boffx + mw])

        #print('Getting Fourier transform of PSF at', px,py)
        #print(type(psf))
        #print (psf.getFourierTransform)
        # print('Tim shape:', img.shape)
        P, (cx, cy), (pH, pW), (v, w) = psf.getFourierTransform(px, py, halfsize)

        dx = px - cx
        dy = py - cy
        if modelMask is not None:
            # the Patch we return *must* have this origin.
            ix0 = x0
            iy0 = y0
            # the difference that we have to handle by shifting the model image
            mux = dx - ix0
            muy = dy - iy0
            # we will handle the integer portion by computing a shifted image
            # and copying it into the result
            sx = int(np.round(mux))
            sy = int(np.round(muy))
            # the subpixel portion will be handled with a Lanczos interpolation
            mux -= sx
            muy -= sy
        else:
            # Put the integer portion of the offset into Patch x0,y0
            ix0 = int(np.round(dx))
            iy0 = int(np.round(dy))
            # the subpixel portion will be handled with a Lanczos interpolation
            mux = dx - ix0
            muy = dy - iy0

        # At this point, mux,muy are both in [-0.5, 0.5]
        assert(np.abs(mux) <= 0.5)
        assert(np.abs(muy) <= 0.5)

        amix = self._getShearedProfile(img, px, py)
        #print ("CPU AMIX", amix.var, amix.var.shape)
        fftmix = amix
        mogmix = None

        if hybrid and inner_real_nsigma is not None and outer_real_nsigma is not None:
            # Split "amix" into terms that we will evaluate using MoG
            # vs FFT.
            vv = amix.var[:, 0, 0] + amix.var[:, 1, 1]
            # Ramp between:
            nsigma1 = inner_real_nsigma
            nsigma2 = outer_real_nsigma
            # Terms that will wrap-around significantly if evaluated
            # with FFT...  We want to know: at the outer edge of this
            # patch, how many sigmas out are we?  If small (ie, the
            # edge still has a significant fraction of the flux),
            # render w/ MoG.
            #pold = pW
            #pW = 128
            IM = ((pW/2)**2 < (nsigma2**2 * vv))
            IF = ((pW/2)**2 > (nsigma1**2 * vv))
            #print ("pW", pW, "N1", nsigma1)
            #print ("vv", vv, vv.shape)
            #pW = pold
            #print('Evaluating', np.sum(IM), 'terms as MoG,', np.sum(IF), 'with FFT,',
            #      np.sum(IM) + np.sum(IF) - len(IM), 'with both')
            #print('  sizes vs PSF size', pW, ':', ', '.join(['%.3g' % s for s in np.sqrt(vv)]))
            ramp = np.any(IM*IF)

            if np.any(IM):
                #print ("CMOG AMP", amix.amp)
                #print ("CMOG VAR", amix.var)
                #print ("CMOG MEAN", amix.mean)
                amps = amix.amp[IM]
                #print ("IM", IM, IM.sum())
                if ramp:
                    ns = (pW/2) / np.maximum(1e-6, np.sqrt(vv))
                    mogweights = np.minimum(1., (nsigma2 - ns[IM]) / (nsigma2 - nsigma1))
                    fftweights = np.minimum(1., (ns[IF] - nsigma1) / (nsigma2 - nsigma1))
                    assert(np.all(mogweights > 0.))
                    assert(np.all(mogweights <= 1.))
                    assert(np.all(fftweights > 0.))
                    assert(np.all(fftweights <= 1.))
                    amps *= mogweights
                mogmix = mp.MixtureOfGaussians(amps,
                                               amix.mean[IM, :] + np.array([px, py])[np.newaxis, :],
                                               amix.var[IM, :, :], quick=True)

            if np.any(IF):
                #print ("CFFT AMP", amix.amp)
                #print ("CFFT VAR", amix.var)
                #print ("CFFT MEAN", amix.mean)
                amps = amix.amp[IF]
                #print ("IF", IF, IF.sum())
                if ramp:
                    amps *= fftweights
                fftmix = mp.MixtureOfGaussians(amps, amix.mean[IF, :], amix.var[IF, :, :],
                                                quick=True)
            else:
                fftmix = None

        if fftmix is not None:
            #print('Evaluating FFT mixture:', len(fftmix.amp), 'components in size', pH,pW)
            #print('Amps:', fftmix.amp)
            #print ("FFTMIX")
            Fsum = fftmix.getFourierTransform(v, w, zero_mean=True)
            # In Intel's mkl_fft library, the irfftn code path is faster than irfft2
            # (the irfft2 version sets args (to their default values) which triggers padding
            #  behavior, changing the FFT size and copying behavior)
            #G = np.fft.irfft2(Fsum * P, s=(pH, pW))
            #savetxt_cpu_append('cfsum.txt', Fsum)
            #savetxt_cpu_append('cp.txt', P)
            G = np.fft.irfftn(Fsum * P)

            assert(G.shape == (pH,pW))
            # FIXME -- we could try to be sneaky and Lanczos-interp
            # after cutting G down to nearly its final size... tricky
            # tho

            # Lanczos-3 interpolation in ~the same way we do for
            # pixelized PSFs.
            from tractor.psf import lanczos_shift_image
            G = G.astype(np.float32)
            #savetxt_cpu_append('cg.txt', G)
            if mux != 0.0 or muy != 0.0:
                lanczos_shift_image(G, mux, muy, inplace=True)
        else:
            G = np.zeros((pH, pW), np.float32)
        #savetxt_cpu_append('cg.txt', G)

        if modelMask is not None:
            gh, gw = G.shape
            assert((gw == pW) and (gh == pH))
            if sx != 0 or sy != 0:
                yi, yo = get_overlapping_region(-sy, -sy + mh - 1, 0, gh - 1)
                xi, xo = get_overlapping_region(-sx, -sx + mw - 1, 0, gw - 1)
                # shifted
                # FIXME -- are yo,xo always the whole image?  If so, optimize
                shG = np.zeros((mh, mw), G.dtype)
                shG[yo, xo] = G[yi, xi]

                if debug_ps is not None:
                    _fourier_galaxy_debug_plots(G, shG, xi, yi, xo, yo, P, Fsum,
                                                pW, pH, psf)

                G = shG
            if gh > mh or gw > mw:
                G = G[:mh, :mw]
            assert(G.shape == modelMask.shape)

        else:
            # Clip down to suggested "halfsize"
            if x0 > ix0:
                G = G[:, x0 - ix0:]
                ix0 = x0
            if y0 > iy0:
                G = G[y0 - iy0:, :]
                iy0 = y0
            gh, gw = G.shape
            if gw + ix0 > x1:
                G = G[:, :x1 - ix0]
            if gh + iy0 > y1:
                G = G[:y1 - iy0, :]

        if mogmix is not None:
            if modelMask is not None:
                mogpatch = run_mog(amix=mogmix, mm=modelMask)
            else:
                gh, gw = G.shape
                mogpatch = run_mog(amix=mogmix, mm=ModelMask(ix0, iy0, gw, gh))
            assert(mogpatch.patch.shape == G.shape)
            #savetxt_cpu_append('cmogpatch.txt', mogpatch.patch)
            G += mogpatch.patch
        #savetxt_cpu_append('cg2.txt', G)

        return Patch(ix0, iy0, G)

def _fourier_galaxy_debug_plots(G, shG, xi, yi, xo, yo, P, Fsum,
                                pW, pH, psf):
    import pylab as plt
    mx = G.max()
    ima = dict(vmin=np.log10(mx) - 6,
               vmax=np.log10(mx),
               interpolation='nearest', origin='lower')
    plt.clf()
    plt.subplot(1, 2, 1)
    #plt.imshow(shG, interpolation='nearest', origin='lower')
    plt.imshow(np.log10(shG), **ima)
    ax = plt.axis()
    plt.plot([xo.start, xo.start, xo.stop - 1, xo.stop - 1, xo.start],
             [yo.start, yo.stop - 1, yo.stop - 1, yo.start, yo.start],
             'r-')
    plt.axis(ax)
    plt.title('shG')
    plt.subplot(1, 2, 2)
    #plt.imshow(G, interpolation='nearest', origin='lower')
    plt.imshow(np.log10(G), **ima)
    ax = plt.axis()
    plt.plot([xi.start, xi.start, xi.stop - 1, xi.stop - 1, xi.start],
             [yi.start, yi.stop - 1, yi.stop - 1, yi.start, yi.start],
             'r-')
    plt.axis(ax)
    plt.title('G')
    debug_ps.savefig()

    def plot_real_imag(F, name):
        plt.clf()
        plt.subplot(2, 2, 1)
        print(name, 'real range', F.real.min(), F.real.max())
        plt.imshow(np.log10(np.abs(F.real)),
                   interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('log(abs(%s.real))' % name)
        plt.subplot(2, 2, 3)
        plt.imshow(np.sign(F.real),
                   vmin=-1, vmax=1,
                   interpolation='nearest', origin='lower', cmap='RdBu')
        plt.xticks([])
        plt.yticks([])
        plt.title('sign(%s.real)' % name)
        print(name, 'imag range', F.imag.min(), F.imag.max())
        plt.subplot(2, 2, 2)
        mx = np.abs(F.imag).max()
        plt.imshow(np.log10(np.abs(F.imag)),
                   interpolation='nearest', origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        plt.title('log(abs(%s.imag))' % name)
        plt.subplot(2, 2, 4)
        plt.imshow(np.sign(F.imag),
                   vmin=-1, vmax=1,
                   interpolation='nearest', origin='lower', cmap='RdBu')
        plt.xticks([])
        plt.yticks([])
        plt.title('sign(%s.imag)' % name)

    plot_real_imag(P, 'PSF')
    debug_ps.savefig()
    plot_real_imag(Fsum, 'FFT(Galaxy)')
    debug_ps.savefig()
    plot_real_imag(P * Fsum, 'FFT(PSF * Galaxy)')
    debug_ps.savefig()

    plt.clf()
    p = np.fft.irfft2(P, s=(pH, pW))
    ax = plt.axis([pW // 2 - 7, pW // 2 + 7, pH // 2 - 7, pH // 2 + 7])
    plt.subplot(1, 3, 1)
    plt.imshow(p, interpolation='nearest', origin='lower')
    plt.axis(ax)
    plt.title('psf (real space)')
    # This is in the corners...
    g = np.fft.irfft2(Fsum, s=(pH, pW))
    plt.subplot(1, 3, 2)
    plt.imshow(g, interpolation='nearest', origin='lower')
    plt.title('galaxy (real space)')
    c = np.fft.irfft2(Fsum * P, s=(pH, pW))
    plt.subplot(1, 3, 3)
    plt.imshow(c, interpolation='nearest', origin='lower')
    plt.axis(ax)
    plt.title('convolved (real space)')
    debug_ps.savefig()

    # # What kind of artifacts do we get from the iFFT(FFT(PSF)) - PSF ?
    # p = np.fft.irfft2(P, s=(pH,pW))
    # plt.clf()
    # plt.subplot(1,3,1)
    # # ASSUME PixelizedPSF
    # pad,cx,cy = psf._padInImage(pW,pH)
    # plt.imshow(pad, interpolation='nearest', origin='lower')
    # plt.subplot(1,3,2)
    # plt.imshow(p, interpolation='nearest', origin='lower')
    # plt.subplot(1,3,3)
    # plt.imshow(pad - p, interpolation='nearest', origin='lower')
    # plt.colorbar()
    # debug_ps.savefig()

    # plt.clf()
    # plt.subplot(1,2,1)
    # mx = np.abs(P).max()
    # plt.imshow(np.log10(np.abs(P)),
    #            interpolation='nearest', origin='lower')
    # plt.xticks([]); plt.yticks([])
    # plt.colorbar()
    # plt.title('log(abs(FFT(PSF)))')
    # plt.subplot(1,2,2)
    # plt.imshow(np.arctan2(P.imag, P.real),
    #            vmin=-np.pi, vmax=np.pi,
    #            interpolation='nearest', origin='lower', cmap='RdBu')
    # plt.xticks([]); plt.yticks([])
    # plt.title('phase(FFT(PSF))')
    # debug_ps.savefig()


class HoggGalaxy(ProfileGalaxy, Galaxy):

    def getName(self):
        return 'HoggGalaxy'

    def getRadius(self):
        return self.nre * self.shape.re

    def _getAffineProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        affine-transformed into the pixel space of the image.
        '''
        galmix = self.getProfile()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        G = self.shape.getRaDecBasis()
        Tinv = np.dot(cdinv, G)
        amix = galmix.apply_affine(np.array([px, py]), Tinv)
        return amix

    def _getShearedProfileGPU(self, imgs, px, py):
        import cupy as cp
        galmix = self.getProfile()
        cdinv = cp.array([img.getWcs().cdInverseAtPixel(px[i], py[i]) for i, img in enumerate(imgs)])
        G = cp.asarray(self.shape.getRaDecBasis())
        Tinv = cp.dot(cdinv, G)
        amix = galmix.apply_shear_GPU(Tinv)
        return amix

    def _getShearedProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        shear-transformed into the pixel space of the image.
        At px,py (but not offset to px,py).
        '''
        galmix = self.getProfile()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        G = self.shape.getRaDecBasis()
        Tinv = np.dot(cdinv, G)
        amix = galmix.apply_shear(Tinv)
        return amix

    def _getUnitFluxDeps(self, img, px, py):
        # return ('unitpatch', self.getName(), px, py,
        return hash(('unitpatch', self.getName(), px, py,
                     img.getWcs().hashkey(),
                     img.getPsf().hashkey(), self.shape.hashkey())
                    )

    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py)
        halfsize = max(1., self.getRadius() / pixscale)
        halfsize += img.psf.getRadius()
        halfsize = int(np.ceil(halfsize))
        return halfsize

    def getDerivativeShearedProfiles(self, img, px, py):
        # Returns a list of sheared profiles that will be needed to compute
        # derivatives for this source; this is assumed in addition to the
        # sheared profile at the current parameter settings.
        derivs = []
        if self.isParamThawed('shape'):
            gsteps = self.shape.getStepSizes()
            gnames = self.shape.getParamNames()
            oldvals = self.shape.getParams()
            for i, gstep in enumerate(gsteps):
                oldval = self.shape.setParam(i, oldvals[i] + gstep)
                pro = self._getShearedProfile(img, px, py)
                #print('Param', gnames[i], 'was', oldval, 'stepped to', oldvals[i]+gstep,
                #      '-> profile', pro.var.ravel())
                self.shape.setParam(i, oldval)
                derivs.append(('shape.'+gnames[i], pro, gstep))
        return derivs

class GaussianGalaxy(HoggGalaxy):
    nre = 6.
    profile = mp.MixtureOfGaussians(np.array([1.]), np.zeros((1, 2)),
                                    np.array([[[1., 0.], [0., 1.]]]))
    profile.normalize()

    def __init__(self, *args, **kwargs):
        self.nre = GaussianGalaxy.nre
        super().__init__(*args, **kwargs)

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
        super().__init__(*args, **kwargs)

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
        super().__init__(*args, **kwargs)

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
        return 1. / (1 + np.exp(4. * (0.5 - f)))

    def derivative(self):
        f = self.getValue()
        # Thanks, Sage
        ef = np.exp(-4. * f + 2)
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
        return max(self.shapeExp.re * ExpGalaxy.nre,
                   self.shapeDev.re * DevGalaxy.nre)

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
        # f = self.fracDev.clipped()
        # profs = []
        # if f > 0.:
        #     profs.append((f, DevGalaxy.profile, self.shapeDev))
        # if f < 1.:
        #     profs.append((1. - f, ExpGalaxy.profile, self.shapeExp))
        # cdinv = img.getWcs().cdInverseAtPixel(px, py)
        # mix = []
        # for f, p, s in profs:
        #     G = s.getRaDecBasis()
        #     Tinv = np.dot(cdinv, G)
        #     amix = p.apply_affine(np.array([px, py]), Tinv)
        #     amix.amp = amix.amp * f
        #     mix.append(amix)

        f = self.fracDev.clipped()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        mix = []
        if f > 0.:
            amix = DevGalaxy.profile.apply_affine(np.array([px, py]), np.dot(cdinv, self.shapeDev.getRaDecBasis()))
            amix.amp = amix.amp * f
            mix.append(amix)
        if f < 1.:
            amix = ExpGalaxy.profile.apply_affine(np.array([px, py]), np.dot(cdinv, self.shapeExp.getRaDecBasis()))
            amix.amp = amix.amp * (1. - f)
            mix.append(amix)
        if len(mix) == 1:
            return mix[0]
        return mix[0] + mix[1]

    def _getShearedProfile(self, img, px, py):
        ''' Returns a MixtureOfGaussians profile that has been
        shear-transformed into the pixel space of the image.
        At px,py (but not offset to px,py).
        '''
        # f = self.fracDev.clipped()
        # profs = []
        # if f > 0.:
        #     profs.append((f, DevGalaxy.profile, self.shapeDev))
        # if f < 1.:
        #     profs.append((1. - f, ExpGalaxy.profile, self.shapeExp))
        # cdinv = img.getWcs().cdInverseAtPixel(px, py)
        # mix = []
        # for f, p, s in profs:
        #     G = s.getRaDecBasis()
        #     Tinv = np.dot(cdinv, G)
        #     amix = p.apply_shear(Tinv)
        #     amix.amp = amix.amp * f
        #     mix.append(amix)
        # if len(mix) == 1:
        #     return mix[0]
        # return mix[0] + mix[1]

        f = self.fracDev.clipped()
        cdinv = img.getWcs().cdInverseAtPixel(px, py)
        mix = []
        if f > 0.:
            amix = DevGalaxy.profile.apply_shear(np.dot(cdinv, self.shapeDev.getRaDecBasis()))
            amix.amp = amix.amp * f
            mix.append(amix)
        if f < 1.:
            amix = ExpGalaxy.profile.apply_shear(np.dot(cdinv, self.shapeExp.getRaDecBasis()))
            amix.amp = amix.amp * (1. - f)
            mix.append(amix)
        if len(mix) == 1:
            return mix[0]
        return mix[0] + mix[1]

    def _getUnitFluxPatchSize(self, img, px=0., py=0., minval=0.):
        if hasattr(self, 'halfsize'):
            return self.halfsize
        pixscale = img.wcs.pixscale_at(px, py)
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
        halfsize = r / pixscale
        halfsize += img.psf.getRadius()
        return halfsize

    def _getUnitFluxDeps(self, img, px, py):
        return hash(('unitpatch', self.getName(),
                     px, py, img.getWcs().hashkey(),
                     img.getPsf().hashkey(),
                     self.shapeDev.hashkey(),
                     self.shapeExp.hashkey(),
                     self.fracDev.hashkey()))

    def getParamDerivatives(self, img, modelMask=None, **kwargs):
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

        if hasattr(self, 'halfsize'):
            e.halfsize = self.halfsize
            d.halfsize = self.halfsize

        dexp = e.getParamDerivatives(img, modelMask=modelMask, **kwargs)
        ddev = d.getParamDerivatives(img, modelMask=modelMask, **kwargs)

        # print('FixedCompositeGalaxy.getParamDerivatives.')
        # print('tim shape', img.shape)
        # print('exp deriv extents:')
        # for deriv in dexp + ddev:
        #     print('  ', deriv.name, deriv.getExtent())

        # fracDev scaling
        f = self.fracDev.clipped()
        for deriv in dexp:
            if deriv is not None:
                deriv *= (1. - f)
        for deriv in ddev:
            if deriv is not None:
                deriv *= f

        derivs = []
        i0 = 0
        if not self.isParamFrozen('pos'):
            # "pos" is shared between the models, so add the derivs.
            npos = self.pos.numberOfParams()
            for i in range(npos):
                ii = i0 + i
                dsum = add_patches(dexp[ii], ddev[ii])
                if dsum is not None:
                    dsum.setName('d(fcomp)/d(pos%i)' % i)
                derivs.append(dsum)
            i0 += npos

        if not self.isParamFrozen('brightness'):
            # shared between the models, so add the derivs.
            nb = self.brightness.numberOfParams()
            for i in range(nb):
                ii = i0 + i
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
                # FIXME -- should be possible to avoid recomputing these...
                ue = e.getUnitFluxModelPatch(img, modelMask=modelMask, **kwargs)
                ud = d.getUnitFluxModelPatch(img, modelMask=modelMask, **kwargs)

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
        return derivs


class CompositeGalaxy(MultiParams, BasicSource):
    '''
    A galaxy with Exponential and deVaucouleurs components.

    The two components share a position (ie the centers are the same),
    but have different brightnesses and shapes.
    '''

    def __init__(self, pos, brightnessExp, shapeExp, brightnessDev, shapeDev):
        MultiParams.__init__(self, pos, brightnessExp, shapeExp,
                             brightnessDev, shapeDev)
        self.name = self.getName()

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightnessExp=1, shapeExp=2,
                    brightnessDev=3, shapeDev=4)

    def getName(self):
        return 'CompositeGalaxy'

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with Exp ' + str(self.brightnessExp) + ' '
                + str(self.shapeExp)
                + ' and deV ' + str(self.brightnessDev) + ' '
                + str(self.shapeDev))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightnessExp=' + repr(self.brightnessExp) +
                ', shapeExp=' + repr(self.shapeExp) +
                ', brightnessDev=' + repr(self.brightnessDev) +
                ', shapeDev=' + repr(self.shapeDev))

    def getBrightness(self):
        ''' This makes some assumptions about the
        ``Brightness`` / ``PhotoCal`` and should be treated as
        approximate.'''
        return self.brightnessExp + self.brightnessDev

    def getBrightnesses(self):
        return [self.brightnessExp, self.brightnessDev]

    def _getModelPatches(self, img, minsb=0., modelMask=None, **kwargs):
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        kw = kwargs.copy()
        if minsb:
            kw.update(minsb=minsb / 2.)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        pe = e.getModelPatch(img, modelMask=modelMask, **kw)
        pd = d.getModelPatch(img, modelMask=modelMask, **kw)
        return (pe, pd)

    def getModelPatch(self, img, minsb=0., modelMask=None, **kwargs):
        pe, pd = self._getModelPatches(img, minsb=minsb, modelMask=modelMask,
                                       **kwargs)
        return add_patches(pe, pd)

    def getUnitFluxModelPatches(self, img, minval=0., modelMask=None, **kwargs):
        # Needed for forced photometry
        if minval > 0:
            # allow each component half the error
            minval = minval * 0.5
        e = ExpGalaxy(self.pos, self.brightnessExp, self.shapeExp)
        d = DevGalaxy(self.pos, self.brightnessDev, self.shapeDev)
        if hasattr(self, 'halfsize'):
            e.halfsize = d.halfsize = self.halfsize
        return (e.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask, **kwargs) +
                d.getUnitFluxModelPatches(img, minval=minval,
                                          modelMask=modelMask, **kwargs))

    # MAGIC: ORDERING OF EXP AND DEV PARAMETERS
    # MAGIC: ASSUMES EXP AND DEV SHAPES SAME LENGTH
    # CompositeGalaxy.
    def getParamDerivatives(self, img, modelMask=None, **kwargs):
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

        de = e.getParamDerivatives(img, modelMask=modelMask, **kwargs)
        dd = d.getParamDerivatives(img, modelMask=modelMask, **kwargs)

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
