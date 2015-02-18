
from common import *

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser(usage='%prog <brick name>')
    opt,args = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(-1)

    decals = Decals()
    CCDs = decals.get_ccds()
    for brickname in args:
        brick = decals.get_brick_by_name(brickname)
        print '# Brick', brickname, 'RA,Dec', brick.ra, brick.dec
        wcs = wcs_for_brick(brick)
        I = ccds_touching_wcs(wcs, CCDs)
        for i in I:
            print i, CCDs.filter[i], CCDs.expnum[i], CCDs.extname[i], CCDs.cpimage[i], CCDs.cpimage_hdu[i], CCDs.calname[i]
