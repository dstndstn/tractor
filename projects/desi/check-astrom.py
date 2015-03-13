import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from common import *

import fitsio
import numpy as np



if __name__ == '__main__':
    D = Decals()

    C = D.get_ccds()

    i = np.flatnonzero((C.expnum == 257212) * (C.extname == 'N16'))
    assert(len(i) == 1)
    i = i[0]

    ccd = C[i]

    fn = os.path.join(D.decals_dir, 'images', ccd.cpimage.strip())
    if not os.path.exists(fn):
        print 'Does not exist:', fn
        fn = fn.replace('.fits.fz', '.fits')
    #img = fitsio.read(fn, ext=ccd.cpimage_hdu)
    #print 'img', img.shape

    im = DecamImage(ccd)
    tim = im.get_tractor_image(D)
    print 'Tim', tim.shape

    calname = ccd.calname.strip()
    print 'cal', calname
    
    wcsfn = os.path.join(D.decals_dir, 'calib', 'decam', 'astrom', calname + '.wcs.fits')
    print 'Trying', wcsfn
    wcs = Sip(wcsfn)
    print 'Got', wcs

    xyfn = os.path.join(D.decals_dir, 'calib', 'decam', 'sextractor', calname + '.fits')
    print 'Trying', xyfn
    xy = fits_table(xyfn, hdu=2)
    print 'Read', len(xy), 'sources'

    ra,dec = wcs.radec_center()
    rdfn = 'rd.fits'
    cmd = ('query-starkd -o %s -r %f -d %f -R 0.2 /project/projectdirs/desi/users/dstn/ps1-astrometry-index/index-ps1-hp17-2.fits' %
           (rdfn, ra, dec))
    print cmd
    rtn = os.system(cmd)
    assert(rtn == 0)

    rd = fits_table(rdfn)
    ok,xx,yy = wcs.radec2pixelxy(rd.ra, rd.dec)

    plt.figure(figsize=(6,12))
    #plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.clf()
    dimshow(tim.getImage(), vmin=-0.2, vmax=1)
    ax = plt.axis()
    plt.plot(xy.x_image, xy.y_image, 'ro', mec='r', mfc='none')
    plt.plot(xx-1, yy-1, 'g+')
    plt.axis(ax)
    #plt.axis([0,1000,0,1000])
    plt.savefig('1.png')


