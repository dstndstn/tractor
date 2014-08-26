import os
import numpy as np
from astrometry.util.fits import fits_table

from common import decals_dir, run_calibs, DecamImage

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser()
    opt,args = parser.parse_args()

    ccdsfn = os.path.join(decals_dir, 'decals-ccds.fits')
    T = fits_table(ccdsfn)

    for a in args:
        i = int(a)
        t = T[i]
        print 'Running', t.calname

        im = DecamImage(t)

        pixscale = np.sqrt(np.abs(t.cd1_1 * t.cd2_2 - t.cd1_2 * t.cd2_1))
        print 'Pixscale', pixscale
        print 'in arcsec:', pixscale * 3600.

        run_calibs(im, t.ra_bore, t.dec_bore, pixscale)
