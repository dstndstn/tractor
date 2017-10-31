from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt

import os
import tempfile
import numpy as np
import sys
from glob import glob
import logging

import fitsio

from astrometry.util.fits import *
from astrometry.util.plotutils import *

import tractor
from tractor import *
from tractor.sdss import *


def galex_read_image(basefn, radecroi=None, nanomaggies=True,
                     band='NUV'):
    '''

    basefn like 'AIS_104/AIS_104_sg17-nd'

    '''

    # http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html
    '''
    To convert from GALEX counts per second (CPS) to flux:
      FUV: Flux [erg sec-1 cm-2 -1] = 1.40 x 10-15 x CPS
      NUV: Flux [erg sec-1 cm-2 -1] = 2.06 x 10-16 x CPS
    To convert from GALEX counts per second (CPS) to magnitudes in the AB
    system (Oke 1990):
      FUV: mAB = -2.5 x log10(CPS) + 18.82
      NUV: mAB = -2.5 x log10(CPS) + 20.08
    For this purpose, we have taken the relative response of all locations
    on the detector as 1. The current estimates are that the zero-points
    defined here are accurate to 10% (1 sigma).
    To convert from flux to AB magnitudes:
      FUV: mAB = -2.5 x log10(FluxFUV / 1.40 x 10-15 erg sec-1 cm-2 Within -1) + 18.82
      NUV: mAB = -2.5 x log10(FluxNUV / 2.06 x 10-16 erg sec-1 cm-2 -1) + 20.08
    '''

    intfn = basefn + '-int.fits.gz'

    #Fint = fitsio.FITS(intfn)
    #H,W = Fint[0].get_info()['dims']
    intimg = pyfits.open(intfn)[0].data
    H, W = intimg.shape
    print('Intensity img:', H, W)

    phdr = pyfits.getheader(intfn, 0)

    if intfn.endswith('.gz'):
        tf = tempfile.NamedTemporaryFile()
        tfn = tf.name
        tf.close()
        phdr.tofile(tfn)
        wcsfn = tfn
    else:
        wcsfn = intfn

    # wcs = Tan(tfn, 0)
    # twcs = FitsWcs(wcs)

    twcs = WcslibWcs(wcsfn, 0)

    if radecroi is not None:
        ralo, rahi, declo, dechi = radecroi
        xy = [twcs.positionToPixel(RaDecPos(r, d))
              for r, d in [(ralo, declo), (ralo, dechi), (rahi, declo), (rahi, dechi)]]
        xy = np.array(xy)
        x0, x1 = xy[:, 0].min(), xy[:, 0].max()
        y0, y1 = xy[:, 1].min(), xy[:, 1].max()
        print('RA,Dec ROI', ralo, rahi, declo, dechi,
              'becomes x,y ROI', x0, x1, y0, y1)
        roi = (x0, x1 + 1, y0, y1 + 1)

    if roi is not None:
        x0, x1, y0, y1 = roi
        x0 = int(np.floor(x0))
        x1 = int(np.ceil(x1))
        y0 = int(np.floor(y0))
        y1 = int(np.ceil(y1))
        roi = (x0, x1, y0, y1)
        # Clip to image size...
        x0 = np.clip(x0, 0, W)
        x1 = np.clip(x1, 0, W)
        y0 = np.clip(y0, 0, H)
        y1 = np.clip(y1, 0, H)
        if x0 == x1 or y0 == y1:
            print('ROI is empty')
            return None
        assert(x0 < x1)
        assert(y0 < y1)
        #roi = (x0,x1,y0,y1)
        twcs.setX0Y0(x0, y0)

    else:
        x0, x1, y0, y1 = 0, W, 0, H

    data = intimg[y0:y1, x0:x1]

    # HACK!
    invvar = np.ones_like(data)

    assert(nanomaggies)

    zps = dict(NUV=20.08, FUV=18.82)
    zp = zps[band]

    photocal = LinearPhotoCal(NanoMaggies.zeropointToScale(zp),
                              band=band)

    tsky = ConstantSky(0.)

    name = 'GALEX ' + phdr['OBJECT'] + ' ' + band

    # HACK -- circular Gaussian PSF of fixed size...
    # in arcsec
    fwhms = dict(NUV=6.0, FUV=6.0)
    # -> sigma in pixels
    sig = fwhms[band] / 2.35 / twcs.pixel_scale()
    tpsf = NCircularGaussianPSF([sig], [1.])

    zr = [-3, 10]

    tim = Image(data=data, invvar=invvar, psf=tpsf, wcs=twcs,
                sky=tsky, photocal=photocal, name=name, zr=zr)
    tim.extent = [x0, x1, y0, y1]
    return tim


if __name__ == '__main__':

    # Region in proposal
    # sz = 21 # sdss pix
    sz = 500  # sdss pix
    sz *= 0.396 / 3600
    ra, dec = 213.031963607, 51.2037580743
    cosd = np.cos(np.deg2rad(dec))
    rdroi = [ra - sz / cosd, ra + sz / cosd, dec - sz, dec + sz]

    #rdroi = [212.96, 213.06, 51.18, 51.25]

    #gbase = 'AIS_104/AIS_104_sg17-'
    gbase = 'GI6_061027_F_VOID_025/GI6_061027_F_VOID_025-'
    #gbase = 'GI5_028149_W3_00862_0105o/GI5_028149_W3_00862_0105o-'

    # oob
    #rdroi = [213.0, 213.5, 50.25, 50.5]

    #rdroi = [215.75, 216.25, 51.25, 51.5]
    #gbase = 'AIS_105/AIS_105_sg97-'

    #rdroi = [213.0, 213.5, 50.9, 51.1]

    gbase = 'GROTH_00_css9950/GROTH_00_css9950-'
    #rdroi = [214.75, 215.25, 52.5, 52.75]
    rdroi = [215.08, 215.11, 52.71, 52.73]

    #bands = ['NUV', 'FUV']
    bands = ['NUV', ]

    r0, r1, d0, d1 = rdroi

    sfn = 'objs-eboss-w3-dr9.fits'

    S = fits_table(sfn, columns=['ra', 'dec'])
    print('EBOSS region:', S.ra.min(), S.ra.max(), S.dec.min(), S.dec.max())

    mr = 0.01
    md = 0.01

    I = np.flatnonzero((S.ra > (r0 - mr)) * (S.ra < (r1 + mr)) *
                       (S.dec > (d0 - md)) * (S.dec < (d1 + md)))
    print('Reading', len(I), 'in range')

    S = fits_table(sfn, rows=I, column_map=dict(r_dev='theta_dev',
                                                r_exp='theta_exp',
                                                fracpsf='fracdev'))
    S.cmodelflux = S.modelflux

    sband = 'r'

    # force everything to be ptsrc?
    ptsrc = False

    # NOTE, this method CUTS the "S" arg
    cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                  objs=S, bands=[], nanomaggies=True, extrabands=bands,
                                  fixedComposites=True, forcePointSources=ptsrc)
    print('Created', len(cat), 'Tractor sources')
    assert(len(cat) == len(S))

    ps = PlotSequence('galex')
    # plt.figure(figsize=(2,2))
    # plt.figure(figsize=(4,4))
    plt.figure(figsize=(6, 6))
    plt.clf()
    #plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    for band in bands:
        if band == 'NUV':
            tag = 'nd'
        elif band == 'FUV':
            tag = 'fd'
        else:
            assert(0)

        tim = galex_read_image(gbase + tag, band=band, radecroi=rdroi)

        # print '200,500', tim.getWcs().pixelToPosition(200,500)
        # print '250,550', tim.getWcs().pixelToPosition(250,550)

        tractor = Tractor([tim], cat)
        print('Created', tractor)

        # ima = dict(interpolation='nearest', origin='lower', vmin=tim.zr[0], vmax=tim.zr[1])
        #ima = dict(interpolation='nearest', origin='upper', vmin=0, vmax=0.1)
        ima = dict(interpolation='nearest', origin='upper', vmin=0, vmax=0.03)

        im = tim.getImage()
        print('Image range:', im.min(), im.max())

        plt.clf()
        plt.imshow(tim.getImage(), **ima)
        plt.gray()
        # plt.colorbar()
        #plt.title(tim.name + ': data')
        ps.savefig()

        mod = tractor.getModelImage(0)

        plt.clf()
        plt.imshow(mod, **ima)
        plt.gray()
        # plt.colorbar()
        #plt.title(tim.name + ': model')
        ps.savefig()
