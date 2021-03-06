from __future__ import print_function
import os
import numpy as np
from astrometry.util.fits import *


def _read_wise_cats_wcs(wcs,
                        pattern, rdpattern, decranges, cols=None, pixelmargin=0):
    r0, r1, d0, d1 = wcs.radec_bounds()
    # if the field were rotated 45 degrees, the margin would be sqrt(2) times bigger...
    margin = np.sqrt(2.) * pixelmargin * wcs.pixel_scale() / 3600.
    W, H = wcs.get_width(), wcs.get_height()
    d0 -= margin
    d1 += margin
    # Dec bounds are useful, but ignore RA
    TT = []
    for i, (dlo, dhi) in enumerate(decranges):
        if dlo > d1 or dhi < d0:
            continue
        fn = rdpattern % (i + 1)
        T = fits_table(fn)
        print('Read', len(T), 'from', fn)
        I = np.flatnonzero((T.dec >= d0) * (T.dec <= d1))
        ok, x, y = wcs.radec2pixelxy(T.ra[I], T.dec[I])
        J = (ok *
             (x >= (0.5 - pixelmargin)) *
             (y >= (0.5 - pixelmargin)) *
             (x <= (W + 0.5 + pixelmargin)) *
             (y <= (H + 0.5 + pixelmargin)))
        I = I[J]
        print('found', len(I), 'within WCS')
        if len(I) == 0:
            continue
        fn = pattern % (i + 1)
        T = fits_table(fn, rows=I, columns=cols)
        TT.append(T)
    if len(TT) == 0:
        return None
    if len(TT) == 1:
        return TT[0]
    return merge_tables(TT)


def _read_wise_cats(r0, r1, d0, d1,
                    pattern, rdpattern, decranges, cols=None):
    if r1 - r0 > 180:
        print('WARNING: wise_catalog_radecbox: RA range',
              r0, 'to', r1, ': maybe wrap-around?')
    TT = []
    for i, (dlo, dhi) in enumerate(decranges):
        if dlo > d1 or dhi < d0:
            continue
        fn = rdpattern % (i + 1)
        T = fits_table(fn)
        print('Read', len(T), 'from', fn)
        I = np.flatnonzero((T.ra >= r0) * (T.ra <= r1) *
                           (T.dec >= d0) * (T.dec <= d1))
        print('found', len(I), 'in range')
        if len(I) == 0:
            continue
        fn = pattern % (i + 1)
        T = fits_table(fn, rows=I, columns=cols)
        TT.append(T)
    if len(TT) == 0:
        return None
    if len(TT) == 1:
        return TT[0]
    return merge_tables(TT)


def wise_catalog_radecbox(r0, r1, d0, d1,
                          path='wise-cats', cols=None):
    return _read_wise_cats(r0, r1, d0, d1,
                           os.path.join(path, 'wise-allsky-cat-part%02i.fits'),
                           os.path.join(
                               path, 'wise-allsky-cat-part%02i-radec.fits'),
                           wise_catalog_dec_range, cols=cols)


wise_catalog_dec_range = [
    (-90.,        -74.4136),
    (-74.413600,  -68.5021),
    (-68.502100,  -63.9184),
    (-63.918400,  -59.9494),
    (-59.949400,  -56.4176),
    (-56.417600,  -53.2041),
    (-53.204100,  -50.1737),
    (-50.173700,  -47.2370),
    (-47.237000,  -44.3990),
    (-44.399000,  -41.6214),
    (-41.621400,  -38.9014),
    (-38.901400,  -36.2475),
    (-36.247500,  -33.6239),
    (-33.623900,  -31.0392),
    (-31.039200,  -28.4833),
    (-28.483300,  -25.9208),
    (-25.920800,  -23.3635),
    (-23.363500,  -20.8182),
    (-20.818200,  -18.2736),
    (-18.273600,  -15.7314),
    (-15.731400,  -13.1899),
    (-13.189900,  -10.6404),
    (-10.640400,  -8.09100),
    (-8.091000,  -5.52990),
    (-5.529900,  -2.94460),
    (-2.944600,  -0.36360),
    (-0.363600,  2.225200),
    (2.225200,  4.838400),
    (4.838400,  7.458000),
    (7.458000,  10.09010),
    (10.090100,  12.73250),
    (12.732500,  15.40770),
    (15.407700,  18.10780),
    (18.107800,  20.83530),
    (20.835300,  23.59230),
    (23.592300,  26.38530),
    (26.385300,  29.21470),
    (29.214700,  32.08660),
    (32.086600,  35.00620),
    (35.006200,  37.98660),
    (37.986600,  41.04520),
    (41.045200,  44.18640),
    (44.186400,  47.38530),
    (47.385300,  50.67800),
    (50.678000,  54.08870),
    (54.088700,  57.69760),
    (57.697600,  61.79110),
    (61.791100,  66.58320),
    (66.583200,  73.30780),
    (73.307800,  90.), ]
