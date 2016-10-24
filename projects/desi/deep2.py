from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *
from astrometry.util.ttime import *

from tractor import *

from wise.allwisecat import allwise_catalog_radecbox
from wise.unwise import *

import logging

def deep2(deep2field, ps):
    tiledir = 'wise-coadds'

    T = fits_table('pcat.%i.fits' % deep2field)
    print('Read', len(T), 'sources')

    # DEEP2, Field 1 part 1
    #ralo, rahi  = 213.2, 214.5
    #declo,dechi =  51.9,  52.4

    ralo, rahi = T.ra.min(), T.ra.max()
    declo,dechi = T.dec.min(), T.dec.max()
    print('RA,Dec range', ralo,rahi, declo,dechi)

    # unwise tile half-size, in degrees
    ddec = (1024 * 2.75 / 3600)
    dra  = ddec / np.cos(np.deg2rad((declo+dechi)/2.))

    A = fits_table(os.path.join(tiledir, 'tiles.fits'))
    A.cut((A.ra  + dra  > ralo ) * (A.ra  - dra  < rahi ) *
          (A.dec + ddec > declo) * (A.dec - ddec < dechi))
    print(len(A), 'tiles overlap:', A.coadd_id)
    tiles = A.coadd_id

    T.cut(T.magr < 24.1)
    #T.cut(T.magr < 21.)
    T.cut(T.magr > 0.)
    print(len(T), 'sources in mag range')
    print('Mag range r', T.magr.min(), T.magr.max())

    for band in [1,2]:
        tims = []
        for tile in tiles:
            tim = get_unwise_tractor_image(tiledir, tile, band,
                                           roiradecbox=[ralo, rahi, declo, dechi],
                                           bandname='w')
            tim.psf.radius = 25
            tims.append(tim)
            print('tim', tim)

        if band == 1:
            plt.clf()
            plt.plot(T.ra, T.dec, 'r.', alpha=0.1)
            for tim in tims:
                h,w = tim.shape
                xy = [(0,0),(0,h),(w,h),(w,0),(0,0)]
                rd = [tim.wcs.pixelToPosition(x,y) for x,y in xy]
                rr = [x.ra for x in rd]
                dd = [x.dec for x in rd]
                plt.plot(rr,dd, 'b-')
            ps.savefig()

        cat = []
        flux = NanoMaggies.magToNanomaggies(T.magr)
        for i in range(len(T)):
            cat.append(PointSource(RaDecPos(T.ra[i], T.dec[i]),
                                   NanoMaggies(w=flux[i] * 40.)))

        fitsky = False

        tractor = Tractor(tims, cat)
        tractor.freezeParamsRecursive('*')
        tractor.thawPathsTo('w')
        if fitsky:
            tractor.thawPathsTo('sky')

        t0 = Time()

        minsb = 0.1 * tims[0].sig1
        R = tractor.optimize_forced_photometry(
            minsb=minsb, mindlnp=1., fitstats=True,
            fitstat_extras=[('pronexp', [tim.nims for tim in tims])],
            variance=True, shared_params=False,
            use_ceres=True, BW=8, BH=8, wantims=True,
            sky=fitsky)
        print('That took', Time()-t0)

        for im0,im1 in zip(R.ims0, R.ims1):
            #(dat,mod,ie,chi,roi) = im0
            (dat,mod,ie,chi,roi) = im1

            ima = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=50)
            plt.clf()
            plt.imshow(dat, **ima)
            plt.title(tim.name)
            ps.savefig()

            # plt.clf()
            # plt.imshow(mod, **ima)
            # plt.title(tim.name + ': initial model')
            # ps.savefig()

            plt.clf()
            plt.imshow(mod, **ima)
            plt.title(tim.name + ': fit model')
            ps.savefig()

        nm = np.array([src.getBrightness().getBand('w') for src in cat])
        nm_ivar = R.IV
    
        wband = 'w%i' % band
        T.set(wband + '_nanomaggies', nm.astype(np.float32))
        T.set(wband + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
        dnm = np.zeros(len(nm_ivar), np.float32)
        okiv = (nm_ivar > 0)
        dnm[okiv] = (1./np.sqrt(nm_ivar[okiv])).astype(np.float32)
        okflux = (nm > 0)
        mag = np.zeros(len(nm), np.float32)
        mag[okflux] = (NanoMaggies.nanomaggiesToMag(nm[okflux])).astype(np.float32)
        dmag = np.zeros(len(nm), np.float32)
        ok = (okiv * okflux)
        dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dnm[ok] / nm[ok])).astype(np.float32)
    
        mag[np.logical_not(okflux)] = np.nan
        dmag[np.logical_not(ok)] = np.nan
            
        T.set(wband + '_mag', mag)
        T.set(wband + '_mag_err', dmag)
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix', 'pronexp']
        for k in fskeys:
            T.set(wband + '_' + k, getattr(R.fitstats, k).astype(np.float32))

    T.writeto('deep2-wise-%i.fits' % deep2field)



if __name__ == '__main__':
    ps = PlotSequence('deep2')
    for cat in [11, 12, 13, 14, 21, 22, 23, 31, 32, 33, 41, 42, 43]:
        deep2(cat, ps)


