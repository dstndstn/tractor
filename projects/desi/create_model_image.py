import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.util import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *

from common import *
from desi_common import *

from runbrick import get_sdss_sources

def main():
    decals = Decals()

    catpattern = 'pipebrick-cats/tractor-phot-b%06i.fits'
    ra,dec = 242, 7

    #roi = None
    roi = [500, 1000, 500, 1000]

    if roi is not None:
        x0,x1,y0,y1 = roi

    #expnum = 346623
    #ccdname = 'N12'
    #chips = decals.find_ccds(expnum=expnum, extname=ccdname)
    #print 'Found', len(chips), 'chips for expnum', expnum, 'extname', ccdname
    #if len(chips) != 1:
    #return False

    chips = decals.get_ccds()
    D = np.argsort(np.hypot(chips.ra - ra, chips.dec - dec))
    print 'Closest chip:', chips[D[0]]
    chips = [chips[D[0]]]

    im = DecamImage(chips[0])
    print 'Image:', im

    targetwcs = Sip(im.wcsfn)
    if roi is not None:
        targetwcs = targetwcs.get_subimage(x0, y0, x1-x0, y1-y0)

    r0,r1,d0,d1 = targetwcs.radec_bounds()
    # ~ 30-pixel margin
    margin = 2e-3
    if r0 > r1:
        # RA wrap-around
        TT = [brick_catalog_for_radec_box(ra,rb, d0-margin,d1+margin,
                                          decals, catpattern)
                for (ra,rb) in [(0, r1+margin), (r0-margin, 360.)]]
        T = merge_tables(TT)
        T._header = TT[0]._header
    else:
        T = brick_catalog_for_radec_box(r0-margin,r1+margin,d0-margin,
                                        d1+margin, decals, catpattern)

    print 'Got', len(T), 'catalog entries within range'
    cat = read_fits_catalog(T, T._header)
    print 'Got', len(cat), 'catalog objects'

    print 'Switching ellipse parameterizations'
    # Switch ellipse parameterizations
    keepcat = []
    for src in cat:
        if isinstance(src, FixedCompositeGalaxy):
            f = src.fracDev.getClippedValue()
            if f == 0.:
                src = ExpGalaxy(src.pos, src.brightness, src.shapeExp)
            elif f == 1.:
                src = DevGalaxy(src.pos, src.brightness, src.shapeDev)

        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            newshape = EllipseESoft.fromEllipseE(src.shape)
            if not np.all(np.isfinite(newshape.getParams())):
                print 'Shape has infinite term: orig', src.shape, 'new', newshape
                print 'src:', src
                continue
            src.shape = newshape
            keepcat.append(src)
        elif isinstance(src, FixedCompositeGalaxy):
            newshape = EllipseESoft.fromEllipseE(src.shapeDev)
            if not np.all(np.isfinite(newshape.getParams())):
                print 'ShapeDev has infinite term: orig', src.shapeDev, 'new', newshape
                print 'src:', src
                continue
            src.shapeDev = newshape
            newshape = EllipseESoft.fromEllipseE(src.shapeExp)
            if not np.all(np.isfinite(newshape.getParams())):
                print 'ShapeExp has infinite term: orig', src.shapeExp, 'new', newshape
                print 'src:', src
                continue
            src.shapeExp = newshape
            keepcat.append(src)
    cat = keepcat

    slc = None
    if roi is not None:
        slc = slice(y0,y1), slice(x0,x1)
    tim = im.get_tractor_image(decals, slc=slc)
    print 'Got', tim
    tim.psfex.fitSavedData(*tim.psfex.splinedata)
    tim.psfex.radius = 20
    tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)
    
    tractor = Tractor([tim], cat)
    print 'Created', tractor

    mod = tractor.getModelImage(0)

    plt.clf()
    dimshow(tim.getImage(), **tim.ima)
    plt.savefig('1.png')

    plt.clf()
    dimshow(mod, **tim.ima)
    plt.savefig('2.png')

    
    ok,x,y = targetwcs.radec2pixelxy([src.getPosition().ra  for src in cat],
                                  [src.getPosition().dec for src in cat])
    ax = plt.axis()
    plt.plot(x, y, 'rx')
    plt.savefig('3.png')
    plt.axis(ax)
    plt.savefig('4.png')
    
    bands = [im.band]
    scat,T = get_sdss_sources(bands, targetwcs, local=False)
    print 'Got', len(scat), 'SDSS sources in bounds'
    
    stractor = Tractor([tim], scat)
    print 'Created', stractor
    smod = stractor.getModelImage(0)

    plt.clf()
    dimshow(smod, **tim.ima)
    plt.savefig('5.png')

    

if __name__ == '__main__':
    main()
    