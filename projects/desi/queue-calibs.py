import sys
import os
import numpy as np
from collections import OrderedDict

from astrometry.util.fits import fits_table
from common import * #Decals, wcs_for_brick, ccds_touching_wcs


'''
python projects/desi/queue-calibs.py  | qdo load cal -
qdo launch cal 1 --batchopts "-A cosmo -t 1-50" --walltime=24:00:00 --batchqueue serial --script projects/desi/run-calib.py
#qdo launch cal 1 --batchopts "-A cosmo -t 1-10" --walltime=24:00:00 --batchqueue serial

python projects/desi/queue-calibs.py  | qdo load bricks -
qdo launch bricks 1 --batchopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial -o pipebrick-logs -j oe -l pvmem=6GB" \
    --script projects/desi/pipebrick.sh
'''

from astrometry.libkd.spherematch import *

import matplotlib
matplotlib.use('Agg')
import pylab as plt
from glob import glob

if __name__ == '__main__':

    if False:
        # tune-ups:
        fns = glob('pipebrick-cats/tractor-phot-b??????.fits')
        fns.sort()
        for fn in fns:
            fn = fn.replace('pipebrick-cats/tractor-phot-b', '')
            fn = fn.replace('.fits', '')
            brickid = int(fn, 10)
            print brickid
        sys.exit(0)

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

    B.writeto('edr-bricks.fits')

    if True:
        bricksize = 0.25
        # how many bricks wide?
        bw,bh = int(np.ceil((rhi - rlo) / bricksize)), int(np.ceil((dhi - dlo) / bricksize))
        # how big are the postage stamps?
        stampsize = 100
        stampspace = 100
    
        html = ('<html><body>' +
                '<div style="width:%i; height:%i; position:relative">' % (bw*stampspace, bh*stampspace))
    
        for b in B:
            #pngfn = 'pipebrick-plots/brick-%06i-00.png' % b.brickid
            #stampfn = 'jpegs/brick-%06i-00-stamp.jpg' % b.brickid
            pngfn = 'tunebrick/coadd/plot-%06i-00.png' % b.brickid
            stampfn = 'tunebrick/web/image-%06i-stamp.jpg' % b.brickid
            if not os.path.exists(stampfn) and os.path.exists(pngfn):
                #cmd = 'pngtopnm %s | pamcut -top 50 | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (pngfn, stampfn)
                cmd = 'pngtopnm %s | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (pngfn, stampfn)
                print cmd
                os.system(cmd)
            if not os.path.exists(stampfn):
                continue

            #modpngfn = 'pipebrick-plots/brick-%06i-02.png' % b.brickid
            #modstampfn = 'jpegs/brick-%06i-02-stamp.jpg' % b.brickid
            modpngfn = 'tunebrick/coadd/plot-%06i-03.png' % b.brickid
            modstampfn = 'tunebrick/web/model-%06i-stamp.jpg' % b.brickid
            if not os.path.exists(modstampfn) and os.path.exists(modpngfn):
                #cmd = 'pngtopnm %s | pamcut -top 50 | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (modpngfn, modstampfn)
                cmd = 'pngtopnm %s | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (modpngfn, modstampfn)
                print cmd
                os.system(cmd)

            # 1000 x 1000 image
            #jpgfn = 'jpegs/brick-%06i-00.jpg' % b.brickid
            jpgfn = 'tunebrick/web/image-%06i.jpg' % b.brickid
            if not os.path.exists(jpgfn):
                #cmd = 'pngtopnm %s | pamcut -top 50 | pnmtojpeg -quality 90 > %s' % (pngfn, jpgfn)
                cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (pngfn, jpgfn)
                print cmd
                os.system(cmd)

            # 1000 x 1000 model
            #png2fn = 'pipebrick-plots/brick-%06i-02.png' % b.brickid
            #jpg2fn = 'jpegs/brick-%06i-02.jpg' % b.brickid
            png2fn = modpngfn
            jpg2fn = 'tunebrick/web/model-%06i.jpg' % b.brickid
            if not os.path.exists(jpg2fn):
                #cmd = 'pngtopnm %s | pamcut -top 50 | pnmtojpeg -quality 90 > %s' % (png2fn, jpg2fn)
                cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (png2fn, jpg2fn)
                print cmd
                os.system(cmd)

            bottom = int(stampspace * (b.dec1 - dlo) / bricksize)
            #left   = int(stampspace * (b.ra1  - rlo) / bricksize)
            left   = int(stampspace * (rhi - b.ra1) / bricksize)
            html += ('<a href="%s"><img src="%s" ' % (jpgfn, stampfn) +
                     "onmouseenter=\"this.src='%s\';\" onmouseleave=\"this.src='%s';\" " % (modstampfn, stampfn) +
                     'style="position:absolute; bottom:%i; left:%i; width=%i; height=%i " /></a>' %
                     (bottom, left, stampsize, stampsize))
            html = html.replace('tunebrick/web/', '')
        html += ('</div>' + 
                '</body></html>')
    
        f = open('tunebrick/web/bricks.html', 'w')
        f.write(html)
        f.close()
    
        #sys.exit(0)


    T = D.get_ccds()

    allI = set()
    for b in B:
        wcs = wcs_for_brick(b)
        I = ccds_touching_wcs(wcs, T)
        print len(I), 'CCDs for brick', b.brickid
        allI.update(I)
    allI = list(allI)
    allI.sort()
    T.cut(allI)
    T.writeto('edr-ccds.fits')
    sys.exit(0)

    for b in B:
        #fn = 'pipebrick-cats/tractor-phot-b%06i.fits' % b.brickid
        #fn = 'pipebrick-plots/brick-%06i-02.png' % b.brickid
        fn = 'tunebricks-cats/tractor-phot-b%06i.fits' % b.brickid
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

    #allI = set()
    allI = OrderedDict()
    
    for b in B:
        wcs = wcs_for_brick(b)
        I = ccds_touching_wcs(wcs, T)
        print >> sys.stderr, 'Brick', b, ':', len(I), 'CCDs'
        #allI.update(I)
        allI.update([(i,True) for i in I])
    #print 'Total of', len(allI), 'CCDs touch'
    #T.cut(np.array(list(allI)))

    print >>sys.stderr, len(B), 'bricks,', len(allI), 'CCDs'

    #for i in list(allI):
    for i in allI.keys():

        im = DecamImage(T[i])
        if not run_calibs(im, None, None, None, just_check=True):
            continue
        #print 'python projects/desi/run-calib.py %i' % i
        print i

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

