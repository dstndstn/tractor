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
    # I = np.array([5659184, 8273238, 3566872, 3561340, 8275835, 8281051, 5657601, 5660763, 8271344,
    #               3631120, 3627936, 3633504, 8267564, 3580336, 3577176, 3572436, 3636684, 3555016, 
    #               5664353, 3574016, 8276780, 5661078, 3626344, 8265670, 8277727, 3586264, 3575596, 
    #               8270399, 3639468, 8282943, 8274892, 8343194, 3551860, 3568448, 3629532, 5665723, 
    #               8286727, 8269454, 3642316, 8268507, 5663089, 8280106, 8272291, 3565288, 3548700, 
    #               8284833, 8269452, 3612756, 5658865, 8343192, 3591784, 8895234, 3632312, 5659499, 
    #               3559756, 3568452, 3645500, 3556596, 8277725, 8274890, 8283890, 8342266, 8272289, 
    #               3562916, 3581920, 8281996, 8264002, 3545940, 3571216, 5656337, 3594588, 8344122, 
    #               5661825, 3612752, 3635096, 8895360, 3648684, 3550280, 8284835, 5658022, 3577180, 
    #               3585076, 8266617, 8282941, 3597744, 8276782, 3543156, 8285782, 3627940, 8266615, 
    #               8263057, 8271346, 3547516, 3651868, 8290051, 3638276, 3629528, 5677371, 8280104, 
    #               8281998, 3635092, 8258800, 3551864, 3631124, 3626340, 3580340, 3548704, 3587448, 
    #               3638272, 3545944, 3556592, 3555020, 3543160, 3600896, 3540804, 8256910, 5653920, 
    #               3581916, 8290053, 3578756, 8293606, 5664774, 5679616, 3540808, 3572432, 5662246, 
    #               8267562, 8265672, 3558180, 3640728, 3633508, 8289108, 8273236, 3570028, 8894761, 
    #               3654716, 3583500, 5665408, 3643908, 3561344, 3537652, 8258802, 8268509, 8270397, 
    #               8275833, 3553436, 3590208, 8262112, 5653286, 5666144, 3537648, 8342268, 5669835, 
    #               3566868, 8261167, 3614332, 3534496, 3647092, 3564108, 3586268, 8281053, 3636688, 
    #               5666982, 3643912, 3564104, 8290998, 8285780, 3657508, 8895171, 3539228, 3531340, 
    #               3550276, 3566876, 3650276, 8896875, 8281049, 8257855, 3593356, 8297388, 3596164, 
    #               3607208, 3648680, 8253125, 3645496, 8275837, 5651286, 3588632, 8295496, 8295498, 
    #               8894887, 3561336, 5679280, 3654712, 3651864, 5680038, 3639472, 8273240, 3657504, 
    #               3636680, 3653460, 5649601, 8344120, 3647096, 5667312, 3536072, 3591780, 3570032, 
    #               3599320, 3660692, 5654758, 3558176, 8283892, 3588628, 3597740, 3596168, 3633500, 
    #               3642320, 5684939, 8299280, 3600892, 3586260, 3609992, 8267566, 8895297, 8286725, 
    #               3591788, 3655916, 3525776, 3583496, 3604048, 3572440, 3602472, 8293604, 8249836, 
    #               3531336, 8292659, 8253127, 8255962, 8270401, 3631116, 8294551, 8292661, 8283888, 
    #               3663100, 3607204, 8271342, 5663510, 8895471, 5682161, 5650865, 5656758, 3663104, 
    #               3575600, 3627932, 3544364, 3624752, 3645504, 5648118, 3599324, 3626348, 5671419, 
    #               8290996, 8289110, 3659100, 3594592, 3605628, 3666288, 8254072, 8344124, 8268505, 
    #               8345056, 3577172, 3528176, 8296445, 8276778, 8894824, 3551856, 8247942, 8262115, 
    #               3529756, 8255020, 8282945, 3666292, 8252180, 3578760, 3574012, 8297390, 5676630, 
    #               3519852, 3669476, 8302825, 3541992, 8342264, 3608788, 3661884, 8263055, 3571212, 
    #               5655073, 8255017, 8280108, 3525772, 3544368, 8896275, 3550284, 3547520, 3553440, 
    #               3624756, 3541996, 3672660, 8896086, 8257857, 3632316, 8254070, 3526960, 3556600, 
    #               3517080, 3611176, 3539232, 8284831, 8304717, 8246054, 8343196, 3614328, 3568444, 
    #               3664696, 3539224, 8294553, 8293608, 8896812, 3536076, 3559760, 8298335, 5654337, 
    #               3612760, 8272293, 8247944, 8244357, 8269456, 8894950, 8345058, 8300225, 3675848, 
    #               3532920, 3522612, 3565284, 3562920, 8246052, 3513924, 3524192, 8249834, 8256912, 
    #               8264000, 3529760, 3667884, 8269450, 8302827, 3565292, 5651707, 3543152, 3678279, 
    #               8343190, 8301882, 8895013, 8248889, 8291000, 5660342, 3562912, 8272287, 8893924, 
    #               8249838, 3526964, 8266619, 8264005, 8274888, 8277723, 3519848, 3510764, 5683009, 
    #               8895534, 3521032, 8246997, 5672155, 8299282, 3655912, 3647088, 3524196, 3650272, 
    #               8248887, 5680464, 3571220, 3540800, 3681463, 5684513, 3531344, 3640724, 3559752, 
    #               3534500, 3638268, 3574020, 8284837, 8252178, 3532916, 8252182, 3615900, 3528180, 
    #               5652971, 3537656, 8300227, 8261165, 8253123, 3661880, 3593352, 3635088, 3543164, 
    #               8290055, 5648961, 3521036, 5646641, 3507600, 3522616, 3596160, 8282000, 3590204, 
    #               8309937, 3518656, 3602468, 8303772, 3599316, 3587444, 8245300, 8244355, 3545948, 
    #               5652128, 8893798, 3585072, 3577184, 3632308, 3684263, 8296447, 5686006, 8896212, 
    #               5647382, 5657179, 5670256, 8240572, 8258804, 3629524, 3581912, 3518660, 8242465, 
    #               3548708, 3627944, 8893672, 3667880, 3608784, 8308045, 8255960, 5655920, 5675355, 
    #               8295500, 5650022, 3517084, 8261170, 8271348, 3504768, 5675777, 5671734, 3578752, 
    #               8894698, 8304719, 3513920, 8247940, 3515500, 3555024, 3513928, 3671068, 5674929, 
    #               3534492, 3515504, 3553432, 3660688, 3657500, 8244359, 3663096, 8311829, 3575592, 
    #               8265674, 8246995, 5661403, 5686432, 5644960, 8340667, 8262110, 8342270, 8900471, 
    #               8243410, 8292663, 3526956, 3654708, 8268511, 8267560, 3669472, 3674252, 8245302, 
    #               8893987, 3558184, 3502004, 5688129, 8256907, 3512340, 8270395, 3570024, 8893861, 
    #               3507604, 5655494, 5645377, 3675844, 3651860, 8301884, 3510760, 8894635, 5689302, 
    #               3648676, 3531332, 8275831, 8246999, 3678275, 8238682, 5659920, 3604044, 3504772, 
    #               3509180, 5680886, 8281055, 3498844, 3600888, 8308992, 8286730, 3547512, 3679867, 
    #               8235402, 3623156, 8239627, 3597736, 3564100, 3594584, 3642312, 3509176, 5642624, 
    #               8235404, 8263060, 8313721, 3639464, 5666561, 3506020, 8896686, 3683047, 8233516, 
    #               8896149, 3561332, 5641152, 8347411, 3636676, 3588624, 8273242, 3506016, 5665089, 
    #               5691557, 3503580, 8231879, 3558172, 3633496, 8267568, 8270403, 3493332, 8310884, ])
    # WISE.cut(I[160:165])
    
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

            #WISE.cut(np.arange(10))
    
            for ttag,table in [('table',True),('notable',False)]:
                coim,coiv,copp,masks = coadd_wise(cowcs, WISE, ps, band, table=table)
    
                coadd_id = ti.coadd_id.replace('_ab41', '')

                prefix = os.path.join(outdir, 'coadd-%s-w%i' % (coadd_id, band))
                
                prefix += '-' + ttag
                
                ofn = prefix + '-img.fits'
                fitsio.write(ofn, coim.astype(np.float32), clobber=True)
                ofn = prefix + '-invvar.fits'
                fitsio.write(ofn, coiv.astype(np.float32), clobber=True)
                ofn = prefix + '-ppstd.fits'
                fitsio.write(ofn, copp.astype(np.float32), clobber=True)

                Sky= []
                Dsky = []
                ii = []
                for i,mm in enumerate(masks):
                    if mm is None:
                        continue
                    (omask, sky, dsky) = mm
                    Sky.append(sky)
                    Dsky.append(dsky)
                    ii.append(i)

                    ofn = os.path.basename(WISE.intfn[i]).replace('-int-', '-msk-rchi-%s-1b-%s.fits' % (coadd_id, ttag))
                    ofn = os.path.join(outdir, ofn)
                    fitsio.write(ofn, omask, clobber=True)

                WISE.cut(np.array(ii))
                WISE.coadd_sky = np.array(Sky)
                WISE.coadd_dsky = np.array(Dsky)

                ofn = prefix + '-frames.fits'
                WISE.writeto(ofn)


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
        #goodmask[np.abs(img) > 1e10] = False

        # Patch masked pixels so we can interpolate
        patchimg = img.copy()

        patchimg[np.logical_not(goodmask)] = 0.
        print 'Unpatched image:', patchimg.min(), patchimg.max()
        assert(np.all(np.isfinite(patchimg)))

        patchimg1 = patchimg.copy()

        ok = patch_image(patchimg, goodmask)
        assert(ok)
        assert(np.all(np.isfinite(patchimg)))

        print 'Patched image:', patchimg.min(), patchimg.max()

        sig1 = np.median(unc[goodmask])
        print 'sig1:', sig1

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
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(cowcs, wcs, [patchimg], L,
                                                 table=table)
        except OverlapError:
            print 'No overlap; skipping'
            rimgs.append(None)
            continue
        rim = rims[0]
        assert(np.all(np.isfinite(rim)))

        print 'Pixels in range:', len(Yo)
        print 'Added to coadd: range', rim.min(), rim.max(), 'mean', np.mean(rim), 'median', np.median(rim)
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


    print 'Coadd (before normalizing) range:', coimg.min(), coimg.max(), 'mean', np.mean(coimg), 'median', np.median(coimg)
    print 'Coadd weight range:', cow.min(), cow.max(), 'median', np.median(cow)

    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16
    coimg1 = coimg / np.maximum(cow, tinyw)
    cow1 = cow.copy()
    print 'Coadd range:', coimg1.min(), coimg1.max(), 'mean', np.mean(coimg1), 'median', np.median(coimg1)

    # Per-pixel std
    coppstd = np.sqrt(coimg2 / np.maximum(cow, tinyw) - coimg1**2)
    print 'Coadd per-pixel range:', coppstd.min(), coppstd.max()
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

    #badpixmasks = []
    #dskys = []

    masks = []
    for rimg in rimgs:
        if rimg is None:
            masks.append(None)
            continue
        (rmask, rimg, w, nil, wcs, sky, zpscale) = rimg
        # like in the WISE Atlas Images, estimate sky difference via difference
        # of medians in overlapping area.
        dsky = np.median(rimg[rmask]) - np.median(coimg1[rmask])
        print 'Sky difference:', dsky

        dsky /= zpscale
        #dskys.append(dsky / zpscale)

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

        #badpixmasks.append(badpixmask)

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

        # print 'Applying rchi masks to images...'
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, cowcs, [], None)
        omask = np.zeros((wcs.get_height(), wcs.get_width()), badpixmask.dtype)
        omask[Yo,Xo] = badpixmask[Yi,Xi]
        masks.append((omask, sky, dsky))


    print 'Coadd (before normalizing) range:', coimg.min(), coimg.max(), 'mean', np.mean(coimg), 'median', np.median(coimg)
    print 'Coadd weight range:', cow.min(), cow.max(), 'median', np.median(cow)

    coimg = (coimg / np.maximum(cow, tinyw))
    coinvvar = cow

    print 'Coadd range:', coimg1.min(), coimg1.max(), 'mean', np.mean(coimg1), 'median', np.median(coimg1)

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

    print 'Coadd per-pixel range:', coppstd.min(), coppstd.max()
    costd1 = np.median(coppstd)
    print 'Median coadd per-pixel std:', costd1

    return coimg, coinvvar, coppstd, masks



def trymain():
    try:
        main()
    except:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':

    if False:
        import cProfile
        from datetime import tzinfo, timedelta, datetime
        pfn = 'prof-%s.dat' % (datetime.now().isoformat())
        cProfile.run('trymain()', pfn)
        print 'Wrote', pfn
    else:
        main()

