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

    srdfn = 'srd.fits'
    cmd = ('query-starkd -o %s -r %f -d %f -R 0.2 /project/projectdirs/desi/users/dstn/sdss-astrometry-index/r2/index-sdss2-r-hp17-2.fits' %
           (srdfn, ra, dec))
    print cmd
    rtn = os.system(cmd)
    assert(rtn == 0)

    srd = fits_table(srdfn)
    ok,sx,sy = wcs.radec2pixelxy(srd.ra, srd.dec)

    H,W = tim.shape
    wfn = 'pvsip.wcs'
    cmd = ('~/astrometry/blind/wcs-pv2sip -e %i -W %i -H %i -X %i -Y %i %s %s' %
           (ccd.cpimage_hdu, W, H, W, H, fn, wfn))
    print cmd
    rtn = os.system(cmd)
    assert(rtn == 0)

    pvsip = Sip(wfn)
    ok,x2,y2 = pvsip.radec2pixelxy(rd.ra, rd.dec)


    ok,stx,sty = pvsip.radec2pixelxy(srd.ra, srd.dec)

    S = fits_table()
    S.x = stx
    S.y = sty
    S.writeto('stxy.fits')

    #plt.figure(figsize=(6,12))
    plt.figure(figsize=(10,20))
    #plt.figure(figsize=(6,6))
    #plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.98)
    plt.clf()
    dimshow(tim.getImage(), vmin=-0.1, vmax=0.5)
    ax = plt.axis()
    plt.title('Expnum %i, Extname %s' % (ccd.expnum, ccd.extname))
    plt.savefig('1.png')

    p0 = plt.plot(xy.x_image, xy.y_image, 'ro', mec='r', mfc='none', ms=8)
    plt.axis(ax)
    #plt.title('SourceExtractor sources')
    plt.legend((p0), ('SourceExtractor sources'), 'upper left')
    plt.savefig('2.png')

    ## sx: SDSS -> SIP
    p3 = plt.plot(sx-1, sy-1, 'o', mec='c', mfc='none')

    ## xx: PS1 -> SIP
    p1 = plt.plot(xx-1, yy-1, 'g+')

    ## x2: PS1 -> (TV->SIP)
    p2 = plt.plot(x2-1, y2-1, 'mx')
    plt.legend((p0[0],p3[0],p1[0],p2[0]),
               ('SourceExtractor sources',
                'SDSS -> Astrometry.net',
                'PS1 -> Astrometry.net',
                'PS1 -> (TPV->SIP)'), 'upper left')
    plt.axis(ax)
    plt.savefig('3.png')
    #plt.axis([0,1000,0,1000])


