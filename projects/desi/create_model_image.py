
from astrometry.util.util import *

from common import *

def main():
    catpattern = 'pipebrick-cats/tractor-phot-%06i.fits'
    expnum = 346623
    ccdname = 'N12'

    decals = Decals()
    chips = decals.find_ccds(expnum=expnum, extname=ccdname)
    print 'Found', len(chips), 'chips for expnum', expnum, 'extname', ccdname
    if len(chips) != 1:
        return False

    im = DecamImage(chips[0])
    print 'Image:', im

    wcs = Sip(im.wcsfn)
    r0,r1,d0,d1 = wcs.radec_bounds()
    # ~ 30-pixel margin
    margin = 2e-3
    if r0 > r1:
        # RA wrap-around
        T = merge_tables([
            brick_catalog_for_radec_box(ra,rb, d0-margin,d1+margin,
                                        decals, catpattern)
            for (ra,rb) in [(0, r1+margin), (r0-margin, 360.)]])
    else:
        T = brick_catalog_for_radec_box(r0-margin,r1+margin,d0-margin,
                                        d1+margin, decals, catpattern)

    print 'Got', len(T), 'catalog entries within range'
        
