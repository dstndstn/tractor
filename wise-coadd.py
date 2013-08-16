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

#median_f = np.median
median_f = flat_median_f

class Duck():
    pass

def main():
    t00 = Time()

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
    I = np.array([5659184, 3566872, 3561340, 5657601] +
                 [8343194, 8343192, 8342266, 8344122] +
                 [8273238, 8275835, 8281051, 8276780] +
                 [8271344, 8267564, 8265670, 8270399])
    I = np.array([5659184, 8273238, 3566872, 3561340, 8275835, 8281051, 5657601, 5660763, 8271344, 3631120])
    #I = I[:4]
    # I = np.array([ 5659184,
    #                8895297,
    #                3525776,
    #                8290996,
    #                ])
    #WISE.cut(I)

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

    print 'Entering main loop:'
    print Time() - t00

    for ti in T:
        print
        coadd_id = ti.coadd_id.replace('_ab41', '')
        print 'Starting coadd tile', coadd_id
        print 'RA,Dec', ti.ra, ti.dec
        print
        cowcs = Tan(ti.ra, ti.dec, (W+1)/2., (H+1)/2.,
                    -pixscale, 0., 0., pixscale, W, H)
    
        copoly = np.array([cowcs.pixelxy2radec(x,y) for x,y in [(1,1), (W,1), (W,H), (1,H)]])
        #print 'copoly', copoly
    
        margin = 2.
        #for band in [1,2,3,4]:
        for band in [2]:

            t0 = Time()

            # cut
            WISE = allWISE
            WISE = WISE[WISE.band == band]
            WISE.cut(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec) < margin)
            print 'Found', len(WISE), 'WISE frames in range and in band W%i' % band
            # reorder by dist from center
            I = np.argsort(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec))
            WISE.cut(I)

            # *inclusive* coordinates of the bounding-box in the coadd of this image
            # (x0,x1,y0,y1)
            WISE.coextent = np.zeros((len(WISE), 4), int)

            # *inclusive* coordinates of the bounding-box in the image overlapping coadd
            WISE.imextent = np.zeros((len(WISE), 4), int)

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
                res.append((intfn, wcs, w, h, poly))

                cpoly = clip_polygon(poly, copoly)

                xy = np.array([cowcs.radec2pixelxy(r,d)[1:] for r,d in cpoly])
                xy -= 1
                x0,y0 = np.floor(xy.min(axis=0)).astype(int)
                x1,y1 = np.ceil (xy.max(axis=0)).astype(int)
                WISE.coextent[wi,:] = [np.clip(x0, 0, W-1),
                                       np.clip(x1, 0, W-1),
                                       np.clip(y0, 0, H-1),
                                       np.clip(y1, 0, H-1)]

                xy = np.array([wcs.radec2pixelxy(r,d)[1:] for r,d in cpoly])
                xy -= 1
                x0,y0 = np.floor(xy.min(axis=0)).astype(int)
                x1,y1 = np.ceil (xy.max(axis=0)).astype(int)
                WISE.imextent[wi,:] = [np.clip(x0, 0, w-1),
                                       np.clip(x1, 0, w-1),
                                       np.clip(y0, 0, h-1),
                                       np.clip(y1, 0, h-1)]

                print 'wi', wi
                print 'row', WISE.row[wi]
                print 'Image extent:', WISE.imextent[wi,:]

            # plt.clf()
            # jj = np.array([0,1,2,3,0])
            # plt.plot(copoly[jj,0], copoly[jj,1], 'b-')
            # for r in res:
            #     if r is None:
            #         continue
            #     poly = r[-1]
            #     plt.plot(poly[jj,0], poly[jj,1], 'r-', alpha=0.1)
            # ps.savefig()
            # 
            # plt.clf()
            # jj = np.array([0,1,2,3,0])
            # plt.plot(copoly[jj,0], copoly[jj,1], 'b-')
            # for r in res:
            #     if r is None:
            #         continue
            #     poly = r[-1]
            #     try:
            #         CC = clip_polygon(poly, copoly)
            #         plt.plot([c[0] for c in CC] + [CC[0][0]], [c[1] for c in CC] + [CC[0][1]], 'r-', alpha=0.1)
            #     except:
            #         plt.plot(poly[jj,0], poly[jj,1], 'k-')
            # ps.savefig()
        
            I = np.flatnonzero(np.array([r is not None for r in res]))
            WISE.cut(I)
            print 'Cut to', len(WISE), 'intersecting target'
            res = [r for r in res if r is not None]
            WISE.intfn = np.array([r[0] for r in res])
            #WISE.rdpoly = np.array([r[4] for r in res])
        
            print 'WISE table rows:', WISE.row

            #WISE.cut(np.arange(10))

            t1 = Time()
            print 'Up to coadd_wise:'
            print t1 - t0

            # table vs no-table: ~ zero difference except in cores of v.bright stars

            coim,coiv,copp,masks = coadd_wise(cowcs, WISE, ps, band)

            t2 = Time()
            print 'coadd_wise:'
            print t2 - t1
    
            prefix = os.path.join(outdir, 'coadd-%s-w%i' % (coadd_id, band))
                
            ofn = prefix + '-img.fits'
            fitsio.write(ofn, coim.astype(np.float32), clobber=True)
            ofn = prefix + '-invvar.fits'
            fitsio.write(ofn, coiv.astype(np.float32), clobber=True)
            ofn = prefix + '-ppstd.fits'
            fitsio.write(ofn, copp.astype(np.float32), clobber=True)

            ii = []
            for i,(mm,r) in enumerate(zip(masks, res)):
                if mm is None:
                    continue
                ii.append(i)

                ofn = WISE.intfn[i].replace('-int', '')
                ofn = os.path.join(outdir, 'coadd-mask-' + coadd_id + '-' + os.path.basename(ofn))

                #(omask, sky, dsky, zp, ncoadd, nrchi) = mm
                (nil,wcs,w,h,poly) = r
                fullmask = np.zeros((h,w), mm.omask.dtype)
                x0,x1,y0,y1 = WISE.imextent[i,:]
                fullmask[y0:y1+1, x0:x1+1] = mm.omask

                fitsio.write(ofn, fullmask, clobber=True)
                #fitsio.write(ofn, omask, clobber=True)

            WISE.cut(np.array(ii))
            masks = [masks[i] for i in ii]

            WISE.coadd_sky  = np.array([m.sky for m in masks])
            WISE.coadd_dsky = np.array([m.dsky for m in masks])
            WISE.zeropoint  = np.array([m.zp for m in masks])
            WISE.npixoverlap = np.array([m.ncopix for m in masks])
            WISE.npixpatched = np.array([m.npatched for m in masks])
            WISE.npixrchi    = np.array([m.nrchipix for m in masks])

            ofn = prefix + '-frames.fits'
            WISE.writeto(ofn)

            ######## HACK
            #break
        #break
    ################

    print 'Whole shebang:'
    print Time() - t00


def coadd_wise(cowcs, WISE, ps, band, table=True):
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
        t00 = Time()
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

        rr = Duck()

        wcs = Sip(intfn)
        print 'Wcs:', wcs
        h,w = wcs.get_height(), wcs.get_width()

        x0,x1,y0,y1 = wise.imextent

        F = fitsio.FITS(intfn)
        img = F[0][y0:y1+1, x0:x1+1]
        ihdr = F[0].read_header()
        mask = fitsio.FITS(maskfn)[0][y0:y1+1, x0:x1+1]
        unc  = fitsio.FITS(uncfn) [0][y0:y1+1, x0:x1+1]
        print 'Img:', img.shape, img.dtype
        print 'Unc:', unc.shape, unc.dtype
        print 'Mask:', mask.shape, mask.dtype

        wcs = wcs.get_subimage(int(x0), int(y0), int(1+x1-x0), int(1+y1-y0))

        zp = ihdr['MAGZP']
        zpscale = NanoMaggies.zeropointToScale(zp)
        print 'Zeropoint:', zp, '-> scale', zpscale

        goodmask = ((mask & sum([1<<bit for bit in [0,1,2,3,4,5,6,7, 9,
                                                    10,11,12,13,14,15,16,17,18,
                                                    21,26,27,28]])) == 0)
        goodmask[unc == 0] = False
        goodmask[np.logical_not(np.isfinite(img))] = False

        sig1 = median_f(unc[goodmask])
        print 'sig1:', sig1

        del mask
        del unc

        #patchimg[np.logical_not(goodmask)] = 0.
        #print 'Unpatched image:', patchimg.min(), patchimg.max()
        #assert(np.all(np.isfinite(patchimg)))
        #patchimg1 = patchimg.copy()

        # Patch masked pixels so we can interpolate
        rr.npatched = np.count_nonzero(np.logical_not(goodmask))
        ok = patch_image(img, goodmask)
        assert(ok)
        assert(np.all(np.isfinite(img)))

        #print 'Patched image:', img.min(), img.max()

        # HACK -- estimate sky level via clipped medians
        med = median_f(img)
        ok = np.flatnonzero(np.abs(img - med) < 3.*sig1)
        sky = median_f(img.flat[ok])
        print 'Estimated sky level:', sky

        # Convert to nanomaggies
        img = (img - sky) * zpscale
        sig1 *= zpscale

        # coadd subimage
        cox0,cox1,coy0,coy1 = wise.coextent
        cosubwcs = cowcs.get_subimage(int(cox0), int(coy0), int(1+cox1-cox0), int(1+coy1-coy0))
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(cosubwcs, wcs, [img], L,
                                                 table=table)
        except OverlapError:
            print 'No overlap; skipping'
            rimgs.append(None)
            continue
        rim = rims[0]
        assert(np.all(np.isfinite(rim)))

        print 'Pixels in range:', len(Yo)
        #print 'Added to coadd: range', rim.min(), rim.max(), 'mean', np.mean(rim), 'median', np.median(rim)
        w = (1./sig1**2)
        coimg [coy0 + Yo, cox0 + Xo] += w * rim
        coimg2[coy0 + Yo, cox0 + Xo] += w * (rim**2)
        cow   [coy0 + Yo, cox0 + Xo] += w

        # save for later...
        rr.rmask = np.zeros((1+coy1-coy0, 1+cox1-cox0), np.bool)
        rr.rmask[Yo, Xo] = True
        rr.rimg = np.zeros((1+coy1-coy0, 1+cox1-cox0), img.dtype)
        rr.rimg[Yo, Xo] = rim

        rr.w = w
        rr.wcs = wcs
        rr.sky = sky
        rr.zpscale = zpscale
        rr.zp = zp
        rr.ncopix = len(Yo)
        rr.coextent = wise.coextent
        rr.cosubwcs = cosubwcs
        rimgs.append(rr)

        #rimgs.append((rmask, rimg, w, maskfn, wcs, sky, zpscale, zp, len(Yo),
        #             wise.coextent, cosubwcs))

        # plt.clf()
        # plt.subplot(1,2,1)
        # plt.imshow(coimg / np.maximum(cow, 1e-16), interpolation='nearest', origin='lower',
        #            extent=[0, W, 0, H], vmin=0, vmax=10)
        # plt.axis([0,W, 0, H])
        # plt.subplot(1,2,2)
        # plt.imshow(rimg, interpolation='nearest', origin='lower',
        #            extent=[cox0, cox1, coy0, coy1], vmin=0, vmax=10)
        # plt.axis([0,W, 0, H])
        # ps.savefig()

        print 'coadd_wise image', wi, 'first pass'
        print Time() - t00


    #print 'Coadd (before normalizing) range:', coimg.min(), coimg.max(), 'mean', np.mean(coimg), 'median', np.median(coimg)
    #print 'Coadd weight range:', cow.min(), cow.max(), 'median', np.median(cow)

    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16
    coimg1 = coimg / np.maximum(cow, tinyw)
    cow1 = cow.copy()
    #print 'Coadd range:', coimg1.min(), coimg1.max(), 'mean', np.mean(coimg1), 'median', np.median(coimg1)

    # plt.clf()
    # plt.imshow(coimg2 / np.maximum(cow,tinyw),
    #            interpolation='nearest', origin='lower',
    #            vmin=0, vmax=100)
    # plt.colorbar()
    # plt.title('coimg2 / cow')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.imshow(coimg1**2,
    #            interpolation='nearest', origin='lower',
    #            vmin=0, vmax=100)
    # plt.colorbar()
    # plt.title('coimg1**2')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.imshow(coimg2 / np.maximum(cow, tinyw) - coimg1**2,
    #            interpolation='nearest', origin='lower',
    #            vmin=-10, vmax=10)
    # plt.colorbar()
    # plt.title('coppvar')
    # ps.savefig()

    # Per-pixel std
    coppstd = np.sqrt(np.maximum(0, coimg2 / np.maximum(cow, tinyw) - coimg1**2))
    #print 'Coadd per-pixel range:', coppstd.min(), coppstd.max()
    #costd1 = np.median(coppstd)
    #print 'Median coadd per-pixel std:', costd1
    #comed = np.median(coimg1)
    # ima = dict(interpolation='nearest', origin='lower',
    #            vmin=comed - 3.*costd1, vmax=comed + 10.*costd1)
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

    masks = []
    ri = 0
    #for ri,rr in enumerate(rimgs):
    while len(rimgs):
        rr = rimgs.pop(0)
        ri += 1

        t00 = Time()

        if rr is None:
            masks.append(None)
            continue
        #(rmask, rimg, w, nil, wcs, sky, zpscale, zp, nadded, coext, cosubwcs) = rimg
        mm = Duck()
        mm.npatched = rr.npatched
        mm.ncopix = rr.ncopix
        mm.sky = rr.sky
        mm.zp = rr.zp

        cox0,cox1,coy0,coy1 = rr.coextent
        subco = coimg1 [coy0:coy1+1, cox0:cox1+1]
        subw = cow1    [coy0:coy1+1, cox0:cox1+1]
        subpp = coppstd[coy0:coy1+1, cox0:cox1+1]

        # like in the WISE Atlas Images, estimate sky difference via difference
        # of medians in overlapping area... or is it median of differences?
        #dsky = np.median(rr.rimg[rr.rmask]) - np.median(subco[rr.rmask])
        dsky = median_f(rr.rimg[rr.rmask] - subco[rr.rmask])
        print 'Sky difference:', dsky
        dsky /= rr.zpscale
        print 'scaled:', dsky
        mm.dsky = dsky

        # print 'rimg:', rimg.shape, rimg.min(), rimg.max()
        # print 'subco:', subco.shape, subco.min(), subco.max()
        # print 'subw:', subw.shape, subw.min(), subw.max()
        # print 'rmask:', rmask.shape, rmask.min(), rmask.max()
        # print 'subpp:', subpp.shape, subpp.min(), subpp.max()

        rchi = (rr.rimg - dsky - subco) * rr.rmask * (subw > 0) * (subpp > 0) / np.maximum(subpp, 1e-6)
        #print 'rchi', rchi.min(), rchi.max()
        assert(np.all(np.isfinite(rchi)))
        badpix = (np.abs(rchi) >= 5.)
        #print 'Number of rchi-bad pixels:', np.count_nonzero(badpix)
        mm.nrchipix = np.count_nonzero(badpix)

        # plt.clf()
        # plt.imshow(rimg - dsky, **ima)
        # plt.title('rimg - dsky')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(rimg - dsky, interpolation='nearest', origin='lower')
        # plt.title('rimg - dsky')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(subco, **ima)
        # plt.title('subco')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(subco, interpolation='nearest', origin='lower')
        # plt.title('subco')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(rimg - dsky - subco, interpolation='nearest', origin='lower')
        # plt.title('rimg - dsky - subco')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(rmask, interpolation='nearest', origin='lower')
        # plt.title('rmask')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(subw > 0, interpolation='nearest', origin='lower')
        # plt.title('subw > 0')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(subpp, interpolation='nearest', origin='lower')
        # plt.title('subpp')
        # plt.colorbar()
        # ps.savefig()

        # plt.clf()
        # plt.imshow(rimg - dsky - coimg1, **ima)
        # plt.title('rimg - dsky - coimg1')
        # plt.colorbar()
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(rchi, interpolation='nearest', origin='lower')#, vmin=-10, vmax=10)
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

        # HACK--
        imgcopy = rr.rimg.copy()
        maskcopy = np.logical_not(badpix).copy()
        reqcopy = badpix.copy()

        ok = patch_image(rr.rimg, np.logical_not(badpix), required=badpix)
        if not ok:
            print 'Writing out failing patch_image inputs'
            fitsio.write('patch-image-img.fits', imgcopy)
            fitsio.write('patch-image-mask.fits', maskcopy.astype(np.uint8))
            fitsio.write('patch-image-required.fits', reqcopy)
        assert(ok)

        # plt.clf()
        # plt.imshow(rimg - dsky, **ima)
        # plt.title('patched rimg - dsky')
        # plt.colorbar()
        # ps.savefig()

        coimg [coy0: coy1+1, cox0: cox1+1] += w * rr.rimg
        coimg2[coy0: coy1+1, cox0: cox1+1] += w * rr.rimg**2
        cow   [coy0: coy1+1, cox0: cox1+1][rr.rmask] += rr.w

        # print 'Applying rchi masks to images...'
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(rr.wcs, rr.cosubwcs, [], None)
        mm.omask = np.zeros((rr.wcs.get_height(), rr.wcs.get_width()), badpixmask.dtype)
        mm.omask[Yo,Xo] = badpixmask[Yi,Xi]

        masks.append(mm)

        del badpix
        del badpixmask

        print 'coadd_wise image', ri, 'second pass'
        print Time() - t00

    # print 'Coadd (before normalizing) range:', coimg.min(), coimg.max(), 'mean', np.mean(coimg), 'median', np.median(coimg)
    # print 'Coadd weight range:', cow.min(), cow.max(), 'median', np.median(cow)

    coimg = (coimg / np.maximum(cow, tinyw))
    coinvvar = cow
    # print 'Coadd range:', coimg1.min(), coimg1.max(), 'mean', np.mean(coimg1), 'median', np.median(coimg1)

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
    var = coimg2 / (np.maximum(cow, tinyw)) - coimg**2
    # print 'Coadd per-pix variance:', var.min(), var.max(), 'mean', np.mean(var), 'median', np.median(var)
    coppstd = np.sqrt(np.maximum(0, var))
    #print 'Coadd per-pixel range:', coppstd.min(), coppstd.max()
    #costd1 = np.median(coppstd)
    #print 'Median coadd per-pixel std:', costd1

    return coimg, coinvvar, coppstd, masks



def trymain():
    try:
        main()
    except:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':

    Time.add_measurement(MemMeas)

    if True:
        import cProfile
        from datetime import tzinfo, timedelta, datetime
        pfn = 'prof-%s.dat' % (datetime.now().isoformat())
        cProfile.run('trymain()', pfn)
        print 'Wrote', pfn
    else:
        main()

