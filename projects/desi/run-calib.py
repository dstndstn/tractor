#! /usr/bin/env python

import os
import numpy as np
from astrometry.util.fits import fits_table

from common import decals_dir, run_calibs, DecamImage, Decals

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    opt,args = parser.parse_args()

    D = Decals()
    T = D.get_ccds()
    print len(T), 'CCDs'
    #ccdsfn = os.path.join(decals_dir, 'decals-ccds.fits')
    #T = fits_table(ccdsfn)

    for a in args:
        i = int(a)
        t = T[i]
        print 'Running', t.calname

        im = DecamImage(t)

        pixscale = np.sqrt(np.abs(t.cd1_1 * t.cd2_2 - t.cd1_2 * t.cd2_1))
        pixscale *= 3600.
        print 'Pixscale', pixscale, 'arcsec/pix'
        mock_psf = False
        kwargs = dict()

        # kwargs.update(psfex=False, psfexfit=False)

        run_calibs((im, kwargs, t.ra, t.dec, pixscale, mock_psf))
