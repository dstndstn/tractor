import sys
import os
import numpy as np
from astrometry.util.fits import fits_table
from common import * #Decals, wcs_for_brick, ccds_touching_wcs

'''
python projects/desi/queue-calibs.py  | qdo load cal -
qdo launch cal 1 --pbsopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial"

python projects/desi/queue-calibs.py  | qdo load bricks -
qdo launch bricks 1 --pbsopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial -o pipebrick-logs -j oe -l pvmem=6GB" \
    --script projects/desi/pipebrick.sh
'''

from astrometry.libkd.spherematch import *

import matplotlib
matplotlib.use('Agg')
import pylab as plt

if __name__ == '__main__':
    D = Decals()
    B = D.get_bricks()

    # I,J,d,counts = match_radec(B.ra, B.dec, T.ra, T.dec, 0.2, nearest=True, count=True)
    # plt.clf()
    # plt.hist(counts, counts.max()+1)
    # plt.savefig('bricks.png')
    # B.cut(I[counts >= 9])
    # plt.clf()
    # plt.plot(B.ra, B.dec, 'b.')
    # #plt.scatter(B.ra[I], B.dec[I], c=counts)
    # plt.savefig('bricks2.png')

    #B.cut((B.ra > 240) * (B.ra < 250) * (B.dec > 5) * (B.dec < 12))

    # 535 bricks, ~7000 CCDs
    rlo,rhi = 240,245
    dlo,dhi =   5, 12
    
    # 56 bricks, ~725 CCDs
    #B.cut((B.ra > 240) * (B.ra < 242) * (B.dec > 5) * (B.dec < 7))
    # 240 bricks, ~3000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 244) * (B.dec > 5) * (B.dec < 9))
    # 535 bricks, ~7000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 245) * (B.dec > 5) * (B.dec < 12))
    B.cut((B.ra > rlo) * (B.ra < rhi) * (B.dec > dlo) * (B.dec < dhi))
    #print len(B), 'bricks in range'

    bricksize = 0.25
    # how many bricks wide?
    bw,bh = int(np.ceil((rhi - rlo) / bricksize)), int(np.ceil((dhi - dlo) / bricksize))
    # how big are the postage stamps?
    stampsize = 100
    stampspace = 100

    html = ('<html><body>' +
            '<div style="width:%i; height:%i; position:relative">' % (bw*stampspace, bh*stampspace))

    for b in B:
        pngfn = 'pipebrick-plots/brick-%06i-00.png' % b.brickid
        stampfn = 'brick-%06i-00-stamp.jpg' % b.brickid
        if not os.path.exists(stampfn) and os.path.exists(pngfn):
            cmd = 'pngtopnm %s | pamcut -top 50 | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (pngfn, stampfn)
            print cmd
            os.system(cmd)
        if not os.path.exists(stampfn):
            continue

        jpgfn = 'brick-%06i-00.jpg' % b.brickid
        if not os.path.exists(jpgfn):
            cmd = 'pngtopnm %s | pamcut -top 50 | pnmtojpeg -quality 90 > %s' % (pngfn, jpgfn)
            print cmd
            os.system(cmd)

        modpngfn = 'pipebrick-plots/brick-%06i-02.png' % b.brickid
        modstampfn = 'brick-%06i-02-stamp.jpg' % b.brickid
        if not os.path.exists(modstampfn) and os.path.exists(modpngfn):
            cmd = 'pngtopnm %s | pamcut -top 50 | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (modpngfn, modstampfn)
            print cmd
            os.system(cmd)


        bottom = int(stampspace * (b.dec1 - dlo) / bricksize)
        #left   = int(stampspace * (b.ra1  - rlo) / bricksize)
        left   = int(stampspace * (rhi - b.ra1) / bricksize)
        html += ('<a href="%s"><img src="%s" ' % (jpgfn, stampfn) +
                 "onmouseenter=\"this.src='%s\';\" onmouseleave=\"this.src='%s';\" " % (modstampfn, stampfn) +
                 'style="position:absolute; bottom:%i; left:%i; width=%i; height=%i " /></a>' %
                 (bottom, left, stampsize, stampsize))
    html += ('</div>' + 
            '</body></html>')

    f = open('bricks.html', 'w')
    f.write(html)
    f.close()

    sys.exit(0)


    T = D.get_ccds()

    for b in B:
        fn = 'pipebrick-cats/tractor-phot-b%06i.fits' % b.brickid
        if os.path.exists(fn):
            print >> sys.stderr, 'exists:', fn
            continue
        #print b
        # Don't try bricks for which the zeropoints are missing.
        wcs = wcs_for_brick(b)
        I = ccds_touching_wcs(wcs, T)
        im = None
        try:
            for t in T[I]:
                im = DecamImage(t)
                zp = D.get_zeropoint_for(im)
        except:
            print >> sys.stderr, 'Brick', b.brickid, ': Failed to get zeropoint for', im
            #import traceback
            #traceback.print_exc()
            continue

        # Ok
        print b.brickid

    sys.exit(0)

    allI = set()
    for b in B:
        wcs = wcs_for_brick(b)
        I = ccds_touching_wcs(wcs, T)
        allI.update(I)
    #print 'Total of', len(allI), 'CCDs touch'
    #T.cut(np.array(list(allI)))

    print >>sys.stderr, len(B), 'bricks,', len(allI), 'CCDs'

    for i in list(allI):
        print 'python projects/desi/run-calib.py %i' % i

    sys.exit(0)

    # g,r,z full focal planes, 2014-08-18
    #I = np.flatnonzero(T.expnum == 349664)
    #I = np.flatnonzero(T.expnum == 349667)
    #I = np.flatnonzero(T.expnum == 349589)

    #for im in T.cpimage[:10]:
    #    print >>sys.stderr, 'im >>%s<<' % im, im.startswith('CP20140818')
    #I = np.flatnonzero(np.array([im.startswith('CP20140818') for im in T.cpimage]))

    # images touching brick X
    B = D.get_bricks()
    #ii = 380155
    ii = 377305
    targetwcs = wcs_for_brick(B[ii])
    I = ccds_touching_wcs(targetwcs, T)
    #print len(I), 'CCDs touching'

    print >>sys.stderr, len(I), 'in cut'
    for i in I:
        print 'python projects/desi/run-calib.py %i' % i

