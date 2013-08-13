import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt

import fitsio

from scipy.ndimage.morphology import binary_dilation

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from runslice import *
from astrometry.util.starutil_numpy import *

from tractor import *
from tractor.ttime import *

from wise3 import get_l1b_file

def main():
    ps = PlotSequence('co')
    
    # Read Atlas Image table
    T = fits_table('wise_allsky_4band_p3as_cdd.fits')
    T.row = np.arange(len(T))
    
    plt.clf()
    plt.plot(T.ra, T.dec, 'r.', ms=4, alpha=0.5)
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Atlas tile centers')
    plt.axis([360,0,-90,90])
    ps.savefig()
    
    # W3
    r0,r1 = 210.593,  219.132
    d0,d1 =  51.1822,  54.1822
    
    margin = 1.
    
    T.cut((T.ra + margin > r0) *
          (T.ra - margin < r1) *
          (T.dec + margin > d0) *
          (T.dec - margin < d1))
    print 'Cut to', len(T), 'Atlas tiles near RA,Dec box'
    
    # Read WISE frame metadata
    wisedir = 'wise-frames'
    WISE = fits_table(os.path.join(wisedir, 'WISE-index-L1b.fits'))
    print 'Read', len(WISE), 'WISE L1b frames'
    #WISE.about()
    WISE.row = np.arange(len(WISE))


    ##### HACK
    # I = np.array([5659184, 3566872, 3561340, 5657601] +
    #              [8343194, 8343192, 8342266, 8344122] +
    #              [8273238, 8275835, 8281051, 8276780] +
    #              [8271344, 8267564, 8265670, 8270399])
    # WISE.cut(I)


    
    margin = 2.
    WISE.cut((WISE.ra + margin > r0) *
             (WISE.ra - margin < r1) *
             (WISE.dec + margin > d0) *
             (WISE.dec - margin < d1))
    print 'Cut to', len(WISE), 'WISE frames near RA,Dec box'
    
    outdir = 'wise-coadds'
    
    pixscale = 2.75 / 3600.
    W,H = 2048, 2048
    #W,H = 512, 512

    # Save the original array
    allWISE = WISE
    
    for ti in T:
        print 'RA,Dec', ti.ra, ti.dec
        cowcs = Tan(ti.ra, ti.dec, (W+1)/2., (H+1)/2.,
                    -pixscale, 0., 0., pixscale, W, H)
    
        copoly = np.array([cowcs.pixelxy2radec(x,y) for x,y in [(1,1), (W,1), (W,H), (1,H)]])
        print 'copoly', copoly
    
        margin = 2.
        for band in [1,2,3,4]:
            # cut
            WISE = allWISE
            WISE = WISE[WISE.band == band]
            WISE.cut(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec) < margin)
            print 'Found', len(WISE), 'WISE frames in range and in band W%i' % band
            # reorder by dist from center
            I = np.argsort(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec))
            WISE.cut(I)
        
            res = []
            for wi,wise in enumerate(WISE):
                print
                print (wi+1), 'of', len(WISE)
                intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
                print 'intfn', intfn
                wcs = Sip(intfn)
                #print 'Wcs:', wcs
                h,w = wcs.get_height(), wcs.get_width()
                poly = np.array([wcs.pixelxy2radec(x,y) for x,y in [(1,1), (w,1), (w,h), (1,h)]])
                if not polygons_intersect(copoly, poly):
                    print 'Image does not intersect target'
                    res.append(None)
                    continue
                F = fitsio.FITS(intfn)
                ihdr = F[0].read_header()
                zp = ihdr['MAGZP']
                res.append((intfn, wcs, w, h, poly, zp))
        
            I = np.flatnonzero(np.array([r is not None for r in res]))
            WISE.cut(I)
            print 'Cut to', len(WISE), 'intersecting target'
            res = [r for r in res if r is not None]
            WISE.intfn = np.array([r[0] for r in res])
            #WISE.rdpoly = np.array([r[4] for r in res])
            WISE.zeropoint = np.array([r[5] for r in res])
        
            print 'WISE table rows:', WISE.row
    
            coim,coiv,copp,masks = coadd_wise(cowcs, WISE, ps, band)
    
            coadd_id = ti.coadd_id.replace('_ab41', '')
    
            ofn = os.path.join(outdir, 'coadd-%s-w%i-img.fits' % (coadd_id, band))
            fitsio.write(ofn, coim, clobber=True)
            ofn = os.path.join(outdir, 'coadd-%s-w%i-invvar.fits' % (coadd_id, band))
            fitsio.write(ofn, coiv, clobber=True)
            ofn = os.path.join(outdir, 'coadd-%s-w%i-ppstd.fits' % (coadd_id, band))
            fitsio.write(ofn, copp, clobber=True)
    
            WISE.coadd_sky = np.array([m[1] for m in masks])
            WISE.coadd_dsky = np.array([m[2] for m in masks])
    
            for i,(omask, sky, dsky) in enumerate(masks):
                ofn = os.path.basename(WISE.intfn[i]).replace('-int-', '-msk-rchi-%s-1b.fits' % (coadd_id))
                ofn = os.path.join(outdir, ofn)
                fitsio.write(ofn, omask, clobber=True)
    
            ofn = os.path.join(outdir, 'wise-frames-%s-w%i.fits' % (coadd_id, band))
            WISE.writeto(ofn)


def coadd_wise(cowcs, WISE, ps, band):
    mask_gz = True
    unc_gz = True
    L = 3

    W = cowcs.get_width()
    H = cowcs.get_height()

    coimg  = np.zeros((H,W))
    coimg2 = np.zeros((H,W))
    cow     = np.zeros((H,W))

    rimgs = []
    
    for wi,wise in enumerate(WISE):
        print
        print (wi+1), 'of', len(WISE)
        intfn = wise.intfn
        uncfn = intfn.replace('-int-', '-unc-')
        if unc_gz:
            uncfn = uncfn + '.gz'
        maskfn = intfn.replace('-int-', '-msk-')
        if mask_gz:
            maskfn = maskfn + '.gz'

        print 'intfn', intfn
        print 'uncfn', uncfn
        print 'maskfn', maskfn

        wcs = Sip(intfn)
        print 'Wcs:', wcs
        h,w = wcs.get_height(), wcs.get_width()
        
        F = fitsio.FITS(intfn)
        img = F[0].read()
        ihdr = F[0].read_header()
        mask = fitsio.FITS(maskfn)[0].read()
        unc  = fitsio.FITS(uncfn) [0].read()

        zp = ihdr['MAGZP']
        zpscale = NanoMaggies.zeropointToScale(zp)
        print 'Zeropoint:', zp, '-> scale', zpscale

        goodmask = ((mask & sum([1<<bit for bit in [0,1,2,3,4,5,6,7, 9,
                                                    10,11,12,13,14,15,16,17,18,
                                                    21,26,27,28]])) == 0)
        goodmask[unc == 0] = False
        goodmask[np.logical_not(np.isfinite(img))] = False

        # Patch masked pixels so we can interpolate
        patchimg = img.copy()
        ok = patch_image(patchimg, goodmask)
        assert(ok)
        assert(np.all(np.isfinite(patchimg)))

        sig1 = np.median(unc[goodmask])

        # HACK -- estimate sky level via clipped medians
        med = np.median(patchimg)
        ok = np.flatnonzero(np.abs(patchimg - med) < 3.*sig1)
        sky = np.median(patchimg.flat[ok])
        print 'Estimated sky level:', sky

        patchimg = (patchimg - sky) * zpscale
        sig1 *= zpscale

        w = (1./sig1**2)

        #name = 'WISE ' + ihdr['COADDID'] + ' W%i' % band
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(cowcs, wcs, [patchimg], L)
        except OverlapError:
            print 'No overlap; skipping'
            rimgs.append(None)
            continue
        rim = rims[0]
        assert(np.all(np.isfinite(rim)))

        print 'Pixels in range:', len(Yo)
        coimg [Yo,Xo] += w * rim
        coimg2[Yo,Xo] += w * (rim**2)
        cow   [Yo,Xo] += w

        # save for later...
        rmask = np.zeros((H,W), np.bool)
        rmask[Yo,Xo] = True
        rimg = np.zeros_like(coimg)
        rimg[Yo,Xo] = rim

        rimgs.append((rmask, rimg, w, maskfn, wcs, sky, zpscale))

        # plt.clf()
        # plt.subplot(2,2,1)
        # plt.imshow(coimg / np.maximum(con, 1), interpolation='nearest', origin='lower')
        # plt.subplot(2,2,2)
        # plt.imshow(cowimg / np.maximum(cow, 1e-16), interpolation='nearest', origin='lower')
        # plt.subplot(2,2,3)
        # plt.imshow(con, interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.suptitle('%i images' % nwi)
        # ps.savefig()


    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16
    coimg1 = coimg / np.maximum(cow, tinyw)
    cow1 = cow.copy()

    # Per-pixel std
    coppstd = np.sqrt(coimg2 / np.maximum(cow, tinyw) - coimg1**2)
    costd1 = np.median(coppstd)
    print 'Median coadd per-pixel std:', costd1
    comed = np.median(coimg1)

    ima = dict(interpolation='nearest', origin='lower',
               vmin=comed - 3.*costd1, vmax=comed + 10.*costd1)

    # plt.clf()
    # plt.imshow(coimg1, **ima)
    # plt.colorbar()
    # plt.title('Coadd')
    # ps.savefig()

    # plt.clf()
    # plt.imshow(coppstd, interpolation='nearest', origin='lower')
    # plt.colorbar()
    # plt.title('Coadd per-pixel std')
    # ps.savefig()

    # Using the difference between the coadd and the resampled
    # individual images ("rchi"), mask additional pixels and redo the
    # coadd.
    coimg [:,:] = 0
    coimg2[:,:] = 0
    cow   [:,:] = 0
    badpixmasks = []
    dskys = []
    for (rmask, rimg, w, nil,nil,nil, zpscale) in rimgs:

        # like in the WISE Atlas Images, estimate sky difference via difference
        # of medians in overlapping area.
        dsky = np.median(rimg[rmask]) - np.median(coimg1[rmask])
        print 'Sky difference:', dsky

        dskys.append(dsky / zpscale)

        rchi = (rimg - dsky - coimg1) * rmask * (cow1 > 0) / np.maximum(coppstd, 1e-6)
        assert(np.all(np.isfinite(rchi)))
        badpix = (np.abs(rchi) >= 5.)
        print 'Number of rchi-bad pixels:', np.count_nonzero(badpix)

        # plt.clf()
        # plt.imshow(rimg - dsky, **ima)
        # plt.title('rimg - dsky')
        # plt.colorbar()
        # ps.savefig()

        # plt.clf()
        # plt.imshow(rimg - dsky - coimg1, **ima)
        # plt.title('rimg - dsky - coimg1')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(rchi, interpolation='nearest', origin='lower', vmin=-10, vmax=10)
        # plt.title('rchi')
        # plt.colorbar()
        # ps.savefig()

        # plt.clf()
        # plt.imshow(badpix, interpolation='nearest', origin='lower', vmin=0, vmax=1,
        #            cmap='gray')
        # plt.title('badpix')
        # plt.colorbar()
        # ps.savefig()

        # Bit 1: rchi >= 5
        badpixmask = badpix.astype(np.uint8)

        # grow by a small margin
        badpix = binary_dilation(badpix)

        # Bit 2: grown
        badpixmask += (2 * badpix)

        # plt.clf()
        # plt.imshow(badpixmask, interpolation='nearest', origin='lower', vmin=0, vmax=3,
        #            cmap='gray')
        # plt.title('badpixmask')
        # plt.colorbar()
        # ps.savefig()

        badpixmasks.append(badpixmask)
        notbad = np.logical_not(badpix)

        print 'Notbad:', np.count_nonzero(notbad), 'set', np.count_nonzero(np.logical_not(notbad)), 'zero'
        print 'Badpix:', np.count_nonzero(badpix), 'set', np.count_nonzero(np.logical_not(badpix)), 'zero'
        ok = patch_image(rimg, notbad, required=badpix)
        assert(ok)

        # plt.clf()
        # plt.imshow(rimg - dsky, **ima)
        # plt.title('patched rimg - dsky')
        # plt.colorbar()
        # ps.savefig()

        coimg  += w * rimg
        coimg2 += w * rimg**2
        cow[rmask] += w

    coimg = (coimg / np.maximum(cow, tinyw))
    coinvvar = cow

    # plt.clf()
    # plt.imshow(coimg1, **ima)
    # plt.colorbar()
    # plt.title('Coadd round 1')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.imshow(coimg, **ima)
    # plt.colorbar()
    # plt.title('Coadd round 2')
    # ps.savefig()

    #print 'Second-round coadd:'
    #sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
    #print 'Coadd sig:', sig
    # per-pixel variance
    coppstd = np.sqrt(coimg2 / (np.maximum(cow, tinyw)) - coimg**2)

    # 2. Apply rchi masks to individual images
    print 'Applying rchi masks to images...'

    masks = []
    for (rmask,rimg,w,maskfn,wcs,sky,zpscale),badpix,dsky in zip(rimgs, badpixmasks, dskys):
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, cowcs, [], None)

        omask = np.zeros((wcs.get_height(), wcs.get_width()), badpix.dtype)
        omask[Yo,Xo] = badpix[Yi,Xi]

        masks.append((omask, sky, dsky))

    return coimg, coinvvar, coppstd, masks



def trymain():
    try:
        main()
    except:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':

    if True:
        import cProfile
        from datetime import tzinfo, timedelta, datetime
        pfn = 'prof-%s.dat' % (datetime.now().isoformat())
        cProfile.run('trymain()', pfn)
        print 'Wrote', pfn
    else:
        main()

