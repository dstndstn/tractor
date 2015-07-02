import sys
import os
import numpy as np
from collections import OrderedDict

from astrometry.util.fits import fits_table
from astrometry.util.file import *
from common import * #Decals, wcs_for_brick, ccds_touching_wcs

from astrometry.libkd.spherematch import *

'''
This script (with manual editing) can produce lists of CCD indices for calibration:

python projects/desi/queue-calibs.py  | qdo load cal -

dr1(d):
qdo launch bricks 16 --mpack 6 --batchopts "-A desi" --walltime=24:00:00 --script projects/desi/pipebrick.sh --batchqueue regular --verbose

qdo launch cal 1 --batchopts "-A cosmo -t 1-50" --walltime=24:00:00 --batchqueue serial --script projects/desi/run-calib.py
#qdo launch cal 1 --batchopts "-A cosmo -t 1-10" --walltime=24:00:00 --batchqueue serial

Or
qdo launch cal 8 --batchopts "-A cosmo -t 1-6" --pack --walltime=30:00 --batchqueue debug --script projects/desi/run-calib.py


Or lists of bricks to run in production:

python projects/desi/queue-calibs.py  | qdo load bricks -
qdo launch bricks 1 --batchopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial -o pipebrick-logs -j oe -l pvmem=6GB" \
    --script projects/desi/pipebrick.sh
'''

import matplotlib
matplotlib.use('Agg')
import pylab as plt
from glob import glob

def log(*s):
    print >>sys.stderr, ' '.join([str(ss) for ss in s])

if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('--calibs', action='store_true', default=False,
                      help='Output CCDs that need to be calibrated.')

    parser.add_option('--forced', action='store_true', default=False,
                      help='Output forced-photometry commands')

    parser.add_option('--lsb', action='store_true', default=False,
                      help='Output Low-Surface-Brightness commands')
    
    parser.add_option('--touching', action='store_true', default=False,
                      help='Cut to only CCDs touching selected bricks')

    parser.add_option('--check', action='store_true', default=False,
                      help='Check which calibrations actually need to run.')
    parser.add_option('--check-coadd', action='store_true', default=False,
                      help='Check which caoadds actually need to run.')
    parser.add_option('--out', help='Output filename for calibs, default %default',
                      default='jobs')
    parser.add_option('--command', action='store_true', default=False,
                      help='Write out full command-line to run calib')
    
    parser.add_option('--maxdec', type=float, help='Maximum Dec to run')
    parser.add_option('--mindec', type=float, help='Minimum Dec to run')

    parser.add_option('--region', help='Region to select')

    parser.add_option('--bricks', help='Set bricks.fits file to load')
    parser.add_option('--ccds', help='Set ccds.fits file to load')

    parser.add_option('--delete-sky', default=False, action='store_true',
                      help='Delete any existing sky calibration files')
    parser.add_option('--delete-pvastrom', default=False, action='store_true',
                      help='Delete any existing PV WCS calibration files')

    parser.add_option('--write-ccds', help='Write CCDs list as FITS table?')

    opt,args = parser.parse_args()

    decals = Decals()
    if opt.bricks is not None:
        B = fits_table(opt.bricks)
        log('Read', len(B), 'from', opt.bricks)
    else:
        B = decals.get_bricks()

    if opt.ccds is not None:
        T = fits_table(opt.ccds)
        log('Read', len(T), 'from', opt.ccds)
    else:
        T = decals.get_ccds()
        log(len(T), 'CCDs')
    T.index = np.arange(len(T))

    # I,J,d,counts = match_radec(B.ra, B.dec, T.ra, T.dec, 0.2, nearest=True, count=True)
    # plt.clf()
    # plt.hist(counts, counts.max()+1)
    # plt.savefig('bricks.png')
    # B.cut(I[counts >= 9])
    # plt.clf()
    # plt.plot(B.ra, B.dec, 'b.')
    # #plt.scatter(B.ra[I], B.dec[I], c=counts)
    # plt.savefig('bricks2.png')


    # DES Stripe82
    #rlo,rhi = 350.,360.
    # rlo,rhi = 300., 10.
    # dlo,dhi = -6., 4.
    # TINY bit
    #rlo,rhi = 350.,351.1
    #dlo,dhi = 0., 1.1

    # EDR+
    # 860 bricks
    # ~10,000 CCDs
    #rlo,rhi = 239,246
    #dlo,dhi =   5, 13

    # DR1
    #rlo,rhi = 0, 360
    # part 1
    #dlo,dhi = 25, 40
    # part 2
    #dlo,dhi = 20,25
    # part 3
    #dlo,dhi = 15,20
    # part 4
    #dlo,dhi = 10,15
    # part 5
    #dlo,dhi = 5,10
    # the rest
    #dlo,dhi = -11, 5
    #dlo,dhi = 15,25.5

    dlo,dhi = -15, 40
    rlo,rhi = 0, 360

    # Arjun says 3x3 coverage area is roughly
    # RA=240-252 DEC=6-12 (but not completely rectangular)

    # COSMOS
    #rlo,rhi = 148.9, 151.2
    #dlo,dhi = 0.9, 3.5

    # A nice well-behaved region (EDR2/3)
    # rlo,rhi = 243.6, 244.6
    # dlo,dhi = 8.1, 8.6

    # 56 bricks, ~725 CCDs
    #B.cut((B.ra > 240) * (B.ra < 242) * (B.dec > 5) * (B.dec < 7))
    # 240 bricks, ~3000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 244) * (B.dec > 5) * (B.dec < 9))
    # 535 bricks, ~7000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 245) * (B.dec > 5) * (B.dec < 12))


    if opt.region in ['test1', 'test2', 'test3', 'test4']:
        nm = dict(test1='2446p115', # weird stuff around bright star
                  test2='1183p292', # faint sources around bright galaxy
                  test3='3503p005', # DES
                  test4='1163p277', # Pollux
                  )[opt.region]

        B.cut(np.flatnonzero(np.array([s == nm for s in B.brickname])))
        log('Cut to', len(B), 'bricks')
        print B.ra, B.dec
        dlo,dhi = -90,90
        rlo,rhi = 0, 360

    elif opt.region == 'edr':
        # EDR:
        # 535 bricks, ~7000 CCDs
        rlo,rhi = 240,245
        dlo,dhi =   5, 12

    elif opt.region == 'dr1a':
        rlo,rhi = 0, 360
        dlo,dhi = 30, 40
    # elif opt.region == 'dr1b':
    #     rlo,rhi = 0, 360
    #     dlo,dhi = 25,30
    # elif opt.region == 'dr1b':
    #     rlo,rhi = 0, 360
    #     dlo,dhi = 25,30

    elif opt.region == 'des':
        dlo, dhi = -6., 4.
        rlo, rhi = 317., 7.

        T.cut(np.flatnonzero(np.array(['CPDES82' in fn for fn in T.cpimage])))
        log('Cut to', len(T), 'CCDs with "CPDES82" in filename')

    elif opt.region == 'subdes':
        rlo,rhi = 320., 360.
        dlo,dhi = -1.25, 1.25

    elif opt.region == 'grz':
        # Bricks with grz coverage.
        # Be sure to use  --bricks decals-bricks-in-dr1.fits
        # which has_[grz] columns.
        B.cut((B.has_g == 1) * (B.has_r == 1) * (B.has_z == 1))
        log('Cut to', len(B), 'bricks with grz coverage')
        
    elif opt.region == 'nogrz':
        # Bricks without grz coverage.
        # Be sure to use  --bricks decals-bricks-in-dr1.fits
        # which has_[grz] columns.
        B.cut(np.logical_not((B.has_g == 1) * (B.has_r == 1) * (B.has_z == 1)))
        log('Cut to', len(B), 'bricks withOUT grz coverage')
    elif opt.region == 'deep2':
        rlo,rhi = 250,260
        dlo,dhi = 30,35

    elif opt.region == 'virgo':
        rlo,rhi = 185,190
        dlo,dhi =  10, 15

    elif opt.region == 'virgo2':
        rlo,rhi = 182,192
        dlo,dhi =   8, 18

    elif opt.region == 'lsb':
        rlo,rhi = 147.2, 147.8
        dlo,dhi = -0.4, 0.4

    if opt.mindec is not None:
        dlo = opt.mindec
    if opt.maxdec is not None:
        dhi = opt.maxdec

    if rlo < rhi:
        B.cut((B.ra >= rlo) * (B.ra <= rhi) * (B.dec >= dlo) * (B.dec <= dhi))
    else: # RA wrap
        B.cut(np.logical_or(B.ra >= rlo, B.ra <= rhi) * (B.dec >= dlo) * (B.dec <= dhi))
    log(len(B), 'bricks in range')

    I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, 0.25)
    keep = np.zeros(len(B), bool)
    for i in I:
        keep[i] = True
    B.cut(keep)
    log('Cut to', len(B), 'bricks near CCDs')

    if opt.touching:
        keep = np.zeros(len(T), bool)
        for j in J:
            keep[j] = True
        T.cut(keep)
        log('Cut to', len(T), 'CCDs near bricks')
    
    # Aside -- how many near DR1=1 CCDs?
    if False:
        T2 = D.get_ccds()
        log(len(T2), 'CCDs')
        T2.cut(T2.dr1 == 1)
        log(len(T2), 'CCDs marked DR1=1')
        log(len(B), 'bricks in range')
        I,J,d = match_radec(B.ra, B.dec, T2.ra, T2.dec, 0.25)
        keep = np.zeros(len(B), bool)
        for i in I:
            keep[i] = True
        B2 = B[keep]
        log('Total of', len(B2), 'bricks near CCDs with DR1=1')
        for band in 'grz':
            Tb = T2[T2.filter == band]
            log(len(Tb), 'in filter', band)
            I,J,d = match_radec(B2.ra, B2.dec, Tb.ra, Tb.dec, 0.25)
            good = np.zeros(len(B2), np.uint8)
            for i in I:
                good[i] = 1
            B2.set('has_' + band, good)

        B2.writeto('decals-bricks-in-dr1.fits')
        sys.exit(0)

    # sort by dec decreasing
    B.cut(np.argsort(-B.dec))

    for b in B:
        if opt.check:
            fn = 'dr1n/tractor/%s/tractor-%s.fits' % (b.brickname[:3], b.brickname)
            if os.path.exists(fn):
                print >> sys.stderr, 'Exists:', fn
                continue
        if opt.check_coadd:
            fn = 'dr1b/coadd/%s/%s/decals-%s-image.jpg' % (b.brickname[:3], b.brickname, b.brickname)
            if os.path.exists(fn):
                print >> sys.stderr, 'Exists:', fn
                continue

        print b.brickname

    if not (opt.calibs or opt.forced or opt.lsb):
        sys.exit(0)
    
    if 'dr1' in T.columns():
        T.cut(T.dr1 == 1)
        log(len(T), 'photometric for DR1')
    
    bands = 'grz'
    log('Filters:', np.unique(T.filter))
    T.cut(np.flatnonzero(np.array([f in bands for f in T.filter])))
    log('Cut to', len(T), 'CCDs in filters', bands)

    if opt.touching:
        allI = set()
        for b in B:
            wcs = wcs_for_brick(b)
            I = ccds_touching_wcs(wcs, T)
            log(len(I), 'CCDs for brick', b.brickid, 'RA,Dec (%.2f, %.2f)' % (b.ra, b.dec))
            if len(I) == 0:
                continue
            allI.update(I)
        allI = list(allI)
        allI.sort()
    else:
        allI = np.arange(len(T))

    if opt.write_ccds:
        T[allI].writeto(opt.write_ccds)
        print 'Wrote', opt.write_ccds

    ## Be careful here -- T has been cut; we want to write out T.index.
    ## 'allI' contains indices into T.

    if opt.forced:
        print 'Writing forced-photometry commands to', opt.out
        f = open(opt.out,'w')
        log('Total of', len(allI), 'CCDs')
        for j,i in enumerate(allI):
            expstr = '%08i' % T.expnum[i]
            #outdir = os.path.join('forced', expstr[:5], expstr)
            #trymakedirs(outdir)
            outfn = os.path.join('forced', expstr[:5], expstr,
                                 'decam-%s-%s-forced.fits' %
                                 (expstr, T.extname[i]))
            imgfn = os.path.join(decals.decals_dir, 'images',
                                 T.cpimage[i].strip())
            if (not os.path.exists(imgfn) and 
                imgfn.endswith('.fz') and
                os.path.exists(imgfn[:-3])):
                imgfn = imgfn[:-3]

            f.write('python projects/desi/forced-photom-decam.py %s %i DR1 %s\n' % 
                    (imgfn, T.cpimage_hdu[i], outfn))

        f.close()
        print 'Wrote', opt.out
        sys.exit(0)

    if opt.lsb:
        print 'Writing LSB commands to', opt.out
        f = open(opt.out,'w')
        log('Total of', len(allI), 'CCDs')
        for j,i in enumerate(allI):
            exp = T.expnum[i]
            ext = T.extname[i].strip()
            outfn = 'lsb/lsb-%s-%s.fits' % (exp, ext)
            f.write('python projects/desi/lsb.py --expnum %i --extname %s --out %s -F -n > lsb/lsb-%s-%s.log 2>&1\n' % (exp, ext, outfn, exp, ext))
        f.close()
        print 'Wrote', opt.out
        sys.exit(0)
        

    print 'Writing calibs to', opt.out
    f = open(opt.out,'w')
    log('Total of', len(allI), 'CCDs')
    for j,i in enumerate(allI):

        if opt.delete_sky or opt.delete_pvastrom:
            print j+1, 'of', len(allI)
            im = DecamImage(decals, T[i])
            if opt.delete_sky and os.path.exists(im.skyfn):
                print '  deleting:', im.skyfn
                os.unlink(im.skyfn)
            if opt.delete_pvastrom and os.path.exists(im.pvwcsfn):
                print '  deleting:', im.pvwcsfn
                os.unlink(im.pvwcsfn)

        if opt.check:
            print j+1, 'of', len(allI)
            im = DecamImage(decals, T[i])
            if not im.run_calibs(im, None, None, None, just_check=True,
                                 astrom=False):
                print 'Calibs for', im.expnum, im.extname, im.calname, 'already done'
                continue

        if opt.command:
            f.write('python projects/desi/run-calib.py --expnum %i --extname %s\n' %
                    (T.expnum[i], T.extname[i]))
        else:
            f.write('%i\n' % T.index[i])
        if opt.check:
            f.flush()
    f.close()
    print 'Wrote', opt.out

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
        im = DecamImage(decals, T[i])
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
                im = DecamImage(decals, t)
                zp = decals.get_zeropoint_for(im)
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
    B = decals.get_bricks()
    #ii = 380155
    ii = 377305
    targetwcs = wcs_for_brick(B[ii])
    I = ccds_touching_wcs(targetwcs, T)
    #print len(I), 'CCDs touching'

    print >>sys.stderr, len(I), 'in cut'
    for i in I:
        print 'python projects/desi/run-calib.py %i' % i

