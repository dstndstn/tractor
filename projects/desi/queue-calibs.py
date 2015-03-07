import sys
import os
import numpy as np
from collections import OrderedDict

from astrometry.util.fits import fits_table
from common import * #Decals, wcs_for_brick, ccds_touching_wcs


'''
This script (with manual editing) can produce lists of CCD indices for calibration:

python projects/desi/queue-calibs.py  | qdo load cal -
qdo launch cal 1 --batchopts "-A cosmo -t 1-50" --walltime=24:00:00 --batchqueue serial --script projects/desi/run-calib.py
#qdo launch cal 1 --batchopts "-A cosmo -t 1-10" --walltime=24:00:00 --batchqueue serial

Or
qdo launch cal 8 --batchopts "-A cosmo -t 1-6" --pack --walltime=30:00 --batchqueue debug --script projects/desi/run-calib.py


Or lists of bricks to run in production:

python projects/desi/queue-calibs.py  | qdo load bricks -
qdo launch bricks 1 --batchopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial -o pipebrick-logs -j oe -l pvmem=6GB" \
    --script projects/desi/pipebrick.sh
'''

from astrometry.libkd.spherematch import *

import matplotlib
matplotlib.use('Agg')
import pylab as plt
from glob import glob

def log(*s):
    print >>sys.stderr, ' '.join([str(ss) for ss in s])

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

    # EDR:
    # 535 bricks, ~7000 CCDs
    rlo,rhi = 240,245
    dlo,dhi =   5, 12

    # 860 bricks
    # ~10,000 CCDs
    # rlo,rhi = 239,246
    # dlo,dhi =   5, 13

    # Arjun says 3x3 coverage area is roughly
    # RA=240-252 DEC=6-12 (but not completely rectangular)

    # COSMOS
    #rlo,rhi = 148.9, 151.2
    #dlo,dhi = 0.9, 3.5

    # A nice well-behaved region (EDR2/3)
    # rlo,rhi = 243.6, 244.6
    # dlo,dhi = 8.1, 8.6

    # DES Stripe82
    #rlo,rhi = 316., 6.
    # rlo,rhi = 350.,360.
    # dlo,dhi = -6., 4.

    # 56 bricks, ~725 CCDs
    #B.cut((B.ra > 240) * (B.ra < 242) * (B.dec > 5) * (B.dec < 7))
    # 240 bricks, ~3000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 244) * (B.dec > 5) * (B.dec < 9))
    # 535 bricks, ~7000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 245) * (B.dec > 5) * (B.dec < 12))

    if rlo < rhi:
        B.cut((B.ra > rlo) * (B.ra < rhi) * (B.dec > dlo) * (B.dec < dhi))
    else: # RA wrap
        B.cut(np.logical_or(B.ra > rlo, B.ra < rhi) * (B.dec > dlo) * (B.dec < dhi))
    log(len(B), 'bricks in range')

    for b in B:
        print b.brickname
    sys.exit(0)
    
    #B.cut(B.brickname == '1498p017')
    #log(len(B), 'bricks for real')


    T = D.get_ccds()
    log(len(T), 'CCDs')
    
    bands = 'grz'
    log('Filters:', np.unique(T.filter))
    T.cut(np.flatnonzero(np.array([f in bands for f in T.filter])))
    log('Cut to', len(T), 'CCDs in filters', bands)

    allI = set()
    for b in B:
        wcs = wcs_for_brick(b)
        I = ccds_touching_wcs(wcs, T)
        log(len(I), 'CCDs for brick', b.brickid, 'RA,Dec (%.2f, %.2f)' % (b.ra, b.dec))

        #if len(I):
        #    print b.brickname

        allI.update(I)
    allI = list(allI)
    allI.sort()

    f = open('jobs','w')
    log('Total of', len(allI), 'CCDs')
    for i in allI:
        #im = DecamImage(T[i])
        #if not im.run_calibs(im, None, None, None, just_check=True,
        #                     psfex=False, psfexfit=False):
        #    continue
        f.write('%i\n' % i)
    f.close()
    print 'Wrote "jobs"'

    sys.exit(0)
    


    # Various tune-ups and other stuff below here...

    #B.writeto('edrplus-bricks.fits')

    if False:
        for b in B:
            #fn = 'tunebrick/coadd/image2-%06i.png' % b.brickid
            #fn = 'cosmos/coadd/image2-%06i.png' % b.brickid
            #fn = 'tunebrick/coadd/image2-%06i-g.fits' % b.brickid
            #if os.path.exists(fn):
            #    continue
            print b.brickid
    
            wcs = wcs_for_brick(b)
            for band in 'grz':
                #fn = 'cosmos/coadd/image2-%06i-%s.fits' % (b.brickid, band)
                fn = 'tunebrick/coadd/image2-%06i-%s.fits' % (b.brickid, band)
                if not os.path.exists(fn):
                    continue
                for key,val in [('CTYPE1', 'RA---TAN'),
                                ('CTYPE2', 'DEC--TAN'),
                                ('CRVAL1', wcs.crval[0]),
                                ('CRVAL2', wcs.crval[1]),
                                ('CRPIX1', wcs.crpix[0]),
                                ('CRPIX2', wcs.crpix[1]),
                                ('CD1_1', wcs.cd[0]),
                                ('CD1_2', wcs.cd[1]),
                                ('CD2_1', wcs.cd[2]),
                                ('CD2_2', wcs.cd[3]),
                                ('IMAGEW', wcs.imagew),
                                ('IMAGEH', wcs.imageh),]:
                    cmd = 'modhead %s %s %s' % (fn, key, val)
                    print cmd
                    os.system(cmd)
    
        sys.exit(0)


    if False:
        bricksize = 0.25
        # how many bricks wide?
        bw,bh = int(np.ceil((rhi - rlo) / bricksize)), int(np.ceil((dhi - dlo) / bricksize))
        # how big are the postage stamps?
        stampsize = 100
        stampspace = 100
    
        html = ('<html><body>' +
                '<div style="width:%i; height:%i; position:relative">' % (bw*stampspace, bh*stampspace))
    
        for b in B:

            fn = 'tunebrick/coadd/image2-%06i.png' % b.brickid
            if not os.path.exists(fn):
                continue

            modpngfn = 'tunebrick/coadd/plot-%06i-03.png' % b.brickid
            modstampfn = 'tunebrick/web/model-%06i-stamp.jpg' % b.brickid
            png2fn = modpngfn
            jpg2fn = 'tunebrick/web/model-%06i.jpg' % b.brickid
            mod = (modpngfn, jpg2fn, modstampfn)

            for pngfn, jpgfn, stampfn in [
                mod,
                ('tunebrick/coadd/plot-%06i-00.png' % b.brickid,
                 'tunebrick/web/image-%06i.jpg' % b.brickid,
                 'tunebrick/web/image-%06i-stamp.jpg' % b.brickid),
                ('tunebrick/coadd/image2-%06i.png' % b.brickid,
                 'tunebrick/web/image2-%06i.jpg' % b.brickid,
                 'tunebrick/web/image2-%06i-stamp.jpg' % b.brickid),
                ]:
                if not os.path.exists(stampfn) and os.path.exists(pngfn):
                    cmd = 'pngtopnm %s | pnmscale 0.1 | pnmtojpeg -quality 90 > %s' % (pngfn, stampfn)
                    print cmd
                    os.system(cmd)

                # 1000 x 1000 image
                if os.path.exists(pngfn) and not os.path.exists(jpgfn):
                    cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (pngfn, jpgfn)
                    print cmd
                    os.system(cmd)
            # Note evilness: we use the loop variables outside the loop!


            bottom = int(stampspace * (b.dec1 - dlo) / bricksize)
            left   = int(stampspace * (rhi - b.ra1) / bricksize)

            if os.path.exists(modstampfn):
                mouse = "onmouseenter=\"this.src='%s\';\" onmouseleave=\"this.src='%s';\" " % (modstampfn, stampfn)
            else:
                mouse = ''
            html += ('<a href="%s"><img src="%s" ' % (jpgfn, stampfn) +
                     mouse +
                     'style="position:absolute; bottom:%i; left:%i; width=%i; height=%i " /></a>' %
                     (bottom, left, stampsize, stampsize))
            html = html.replace('tunebrick/web/', '')
        html += ('</div>' + 
                '</body></html>')
    
        #fn = 'tunebrick/web/bricks.html'
        fn = 'bricks2.html'
        f = open(fn, 'w')
        f.write(html)
        f.close()
        print 'Wrote', fn
    
        sys.exit(0)
        
    #T.cut(allI)
    #T.writeto('edr-ccds.fits')
    sys.exit(0)

    f = open('jobs','w')
    log('Total of', len(allI), 'CCDs')
    for i in allI:
        im = DecamImage(T[i])
        if not im.run_calibs(im, None, None, None, just_check=True,
                             psfex=False, psfexfit=False):
            continue
        f.write('%i\n' % i)
        #print i
    f.close()
    print 'Wrote "jobs"'
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

