#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
import tempfile
from scipy.ndimage.morphology import binary_dilation

import fitsio


if __name__ == '__main__':
    arr = os.environ.get('PBS_ARRAYID')
    d = os.environ.get('PBS_O_WORKDIR')
    if arr is not None and d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())


from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *

from astrometry.blind.plotstuff import *

from tractor import *
from tractor.ttime import *

from wise3 import get_l1b_file

import logging
lvl = logging.INFO
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

#median_f = np.median
median_f = flat_median_f


# GLOBALS
outdir = 'wise-coadds'
pixscale = 2.75 / 3600.
W,H = 2048, 2048
bands = [1,2,3,4]
# WISE Level 1b inputs
wisedir = 'wise-frames'
mask_gz = True
unc_gz = True


class Duck():
    pass

def get_atlas_tiles(r0,r1,d0,d1):
    # Read Atlas Image table
    T = fits_table('wise_allsky_4band_p3as_cdd.fits', columns=['coadd_id', 'ra', 'dec'])
    T.row = np.arange(len(T))
    print 'Read', len(T), 'Atlas tiles'

    margin = max(W,H) * pixscale / 2.
    cosdec = np.cos(np.deg2rad(max(abs(d0),abs(d1))))
    mr = margin / cosdec
    
    T.cut((T.ra + mr > r0) *
          (T.ra - mr < r1) *
          (T.dec + margin > d0) *
          (T.dec - margin < d1))
    print 'Cut to', len(T), 'Atlas tiles near RA,Dec box'

    T.coadd_id = np.array([c.replace('_ab41','') for c in T.coadd_id])

    return T


def get_wise_frames(r0,r1,d0,d1):
    # Read WISE frame metadata
    WISE = fits_table(os.path.join(wisedir, 'WISE-index-L1b.fits'))
    print 'Read', len(WISE), 'WISE L1b frames'
    WISE.row = np.arange(len(WISE))

    # Testing subsets of rows...
    #I = np.array([3580337,3577177])
    #WISE.cut(I)

    cosdec = np.cos(np.deg2rad(max(abs(d0),abs(d1))))

    margin = 2.
    WISE.cut((WISE.ra + margin/cosdec > r0) *
             (WISE.ra - margin/cosdec < r1) *
             (WISE.dec + margin > d0) *
             (WISE.dec - margin < d1))
    print 'Cut to', len(WISE), 'WISE frames near RA,Dec box'

    return WISE

def check_md5s(WISE):

    from astrometry.util.run_command import run_command
    from astrometry.util.file import read_file

    for i in np.lexsort((WISE.band, WISE.frame_num, WISE.scan_id)):
        intfn = get_l1b_file(wisedir, WISE.scan_id[i], WISE.frame_num[i], WISE.band[i])
        uncfn = intfn.replace('-int-', '-unc-')
        if unc_gz:
            uncfn = uncfn + '.gz'
        maskfn = intfn.replace('-int-', '-msk-')
        if mask_gz:
            maskfn = maskfn + '.gz'
        #print 'intfn', intfn
        #print 'uncfn', uncfn
        #print 'maskfn', maskfn

        instr = ''
        for fn in [intfn,uncfn,maskfn]:
            if not os.path.exists(fn):
                print '%s: DOES NOT EXIST' % fn
                continue
            md5 = read_file(fn + '.md5')
            instr += '%s  %s\n' % (md5, fn)
        if len(instr):
            cmd = "echo '%s' | md5sum -c" % instr
            rtn,out,err = run_command(cmd)
            print out, err
            if rtn:
                print 'ERROR: return code', rtn


def main():
    t00 = Time()

    ps = PlotSequence('co')
    
    # plt.clf()
    # plt.plot(T.ra, T.dec, 'r.', ms=4, alpha=0.5)
    # plt.xlabel('RA (deg)')
    # plt.ylabel('Dec (deg)')
    # plt.title('Atlas tile centers')
    # plt.axis([360,0,-90,90])
    # ps.savefig()
    
    # W3
    #r0,r1 = 210.593,  219.132
    #d0,d1 =  51.1822,  54.1822

    # SEQUELS
    r0,r1 = 120.0, 200.0
    d0,d1 =  45.0,  60.0

    T = get_atlas_tiles(r0,r1,d0,d1)
    allWISE = get_wise_frames(r0,r1,d0,d1)

    plot_region(r0,r1,d0,d1, ps, T, allWISE[allWISE.band == bands[0]], None)

    print 'Entering main loop:', Time() - t00
    for ti in T:
        print
        print 'Starting coadd tile', ti.coadd_id
        print 'RA,Dec', ti.ra, ti.dec
        print
        for band in bands:
            one_coadd(ti, band, allWISE, ps)
    print 'Whole enchilada:', Time() - t00


def one_coadd(ti, band, WISE, ps):
    print 'Coadd tile', ti.coadd_id
    print 'RA,Dec', ti.ra, ti.dec
    print 'Band', band

    tag = 'coadd-%s-w%i' % (ti.coadd_id, band)
    prefix = os.path.join(outdir, tag)
    ofn = prefix + '-img.fits'
    if os.path.exists(ofn):
        print 'Output file exists:', ofn
        return

    cowcs = Tan(ti.ra, ti.dec, (W+1)/2., (H+1)/2.,
                -pixscale, 0., 0., pixscale, W, H)
    
    copoly = np.array([cowcs.pixelxy2radec(x,y) for x,y in [(1,1), (W,1), (W,H), (1,H)]])
    
    margin = (1.1 # safety margin
              * (np.sqrt(2.) / 2.) # diagonal
              * (max(W,H) + 1016) # WISE FOV, coadd FOV side length
              * pixscale) # in deg
    print 'Margin:', margin
    t0 = Time()

    # cut
    WISE = WISE[WISE.band == band]
    WISE.cut(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec) < margin)
    print 'Found', len(WISE), 'WISE frames in range and in band W%i' % band
    # reorder by dist from center
    I = np.argsort(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec))
    WISE.cut(I)

    #WISE.cut(np.arange(20))

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
    WISE.wcs = np.array([r[1] for r in res])

    #print 'WISE table rows:', WISE.row

    t1 = Time()
    print 'Up to coadd_wise:'
    print t1 - t0

    # table vs no-table: ~ zero difference except in cores of v.bright stars

    try:
        coim,coiv,copp, coimb,coivb,masks = coadd_wise(cowcs, WISE, ps, band)
    except:
        print 'coadd_wise failed:'
        import traceback
        traceback.print_exc()

        print 'time up to failure:'
        t2 = Time()
        print t2 - t1

        return
    t2 = Time()
    print 'coadd_wise:'
    print t2 - t1

    f,wcsfn = tempfile.mkstemp()
    os.close(f)
    cowcs.write_to(wcsfn)
    hdr = fitsio.read_header(wcsfn)
    os.remove(wcsfn)
        

    tag = 'coadd-%s-w%i' % (ti.coadd_id, band)
    prefix = os.path.join(outdir, tag)

    ofn = prefix + '-img.fits'
    fitsio.write(ofn, coim.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-invvar.fits'
    fitsio.write(ofn, coiv.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-ppstd.fits'
    fitsio.write(ofn, copp.astype(np.float32), header=hdr, clobber=True)

    ofn = prefix + '-img-w.fits'
    fitsio.write(ofn, coimb.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-invvar-w.fits'
    fitsio.write(ofn, coivb.astype(np.float32), header=hdr, clobber=True)

    ii = []
    for i,(mm,r) in enumerate(zip(masks, res)):
        if mm is None:
            continue
        ii.append(i)

        maskdir = os.path.join(outdir, 'masks-' + tag)
        if not os.path.exists(maskdir):
            os.mkdir(maskdir)

        ofn = WISE.intfn[i].replace('-int', '')
        ofn = os.path.join(maskdir, 'coadd-mask-' + ti.coadd_id + '-' + os.path.basename(ofn))

        (nil,wcs,w,h,poly) = r
        fullmask = np.zeros((h,w), mm.omask.dtype)
        x0,x1,y0,y1 = WISE.imextent[i,:]
        fullmask[y0:y1+1, x0:x1+1] = mm.omask

        fitsio.write(ofn, fullmask, clobber=True)

        cmd = 'gzip %s' % ofn
        print 'Running:', cmd
        rtn = os.system(cmd)
        print 'Result:', rtn

    WISE.cut(np.array(ii))
    masks = [masks[i] for i in ii]

    WISE.coadd_sky  = np.array([m.sky for m in masks])
    WISE.coadd_dsky = np.array([m.dsky for m in masks])
    WISE.zeropoint  = np.array([m.zp for m in masks])
    WISE.npixoverlap = np.array([m.ncopix for m in masks])
    WISE.npixpatched = np.array([m.npatched for m in masks])
    WISE.npixrchi    = np.array([m.nrchipix for m in masks])

    WISE.delete_column('wcs')

    ofn = prefix + '-frames.fits'
    WISE.writeto(ofn)



def plot_region(r0,r1,d0,d1, ps, T, WISE, wcsfns):
    maxcosdec = np.cos(np.deg2rad(min(abs(d0),abs(d1))))
    plot = Plotstuff(outformat='png', size=(800,800),
                     rdw=((r0+r1)/2., (d0+d1)/2., 1.05*max(d1-d0, (r1-r0)*maxcosdec)))

    for i in range(3):
        if i in [0,2]:
            plot.color = 'verydarkblue'
        else:
            plot.color = 'black'
        plot.plot('fill')
        plot.color = 'white'
        out = plot.outline

        if i == 0:
            if T is None:
                continue
            plot.alpha = 0.5
            for ti in T:
                cowcs = Tan(ti.ra, ti.dec, (W+1)/2., (H+1)/2.,
                            -pixscale, 0., 0., pixscale, W, H)
                out.wcs = anwcs_new_tan(cowcs)
                out.fill = 1
                plot.plot('outline')
                out.fill = 0
                plot.plot('outline')
        elif i == 1:
            if WISE is None:
                continue
            # cut
            #WISE = WISE[WISE.band == band]
            plot.alpha = (3./256.)
            out.fill = 1
            print 'Plotting', len(WISE), 'exposures'
            wcsparams = []
            fns = []
            for wi,wise in enumerate(WISE):
                if wi % 10 == 0:
                    print '.',
                if wi % 1000 == 0:
                    print wi, 'of', len(WISE)

                if wi and wi % 10000 == 0:
                    fn = ps.getnext()
                    plot.write(fn)
                    print 'Wrote', fn

                    wp = np.array(wcsparams)
                    WW = fits_table()
                    WW.crpix  = wp[:, 0:2]
                    WW.crval  = wp[:, 2:4]
                    WW.cd     = wp[:, 4:8]
                    WW.imagew = wp[:, 8]
                    WW.imageh = wp[:, 9]
                    WW.intfn = np.array(fns)
                    WW.writeto('sequels-wcs.fits')

                intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, wise.band)
                try:
                    wcs = Tan(intfn, 0, 1)
                except:
                    import traceback
                    traceback.print_exc()
                    continue
                out.wcs = anwcs_new_tan(wcs)
                plot.plot('outline')

                wcsparams.append((wcs.crpix[0], wcs.crpix[1], wcs.crval[0], wcs.crval[1],
                                  wcs.cd[0], wcs.cd[1], wcs.cd[2], wcs.cd[3],
                                  wcs.imagew, wcs.imageh))
                fns.append(intfn)

            wp = np.array(wcsparams)
            WW = fits_table()
            WW.crpix  = wp[:, 0:2]
            WW.crval  = wp[:, 2:4]
            WW.cd     = wp[:, 4:8]
            WW.imagew = wp[:, 8]
            WW.imageh = wp[:, 9]
            WW.intfn = np.array(fns)
            WW.writeto('sequels-wcs.fits')

            fn = ps.getnext()
            plot.write(fn)
            print 'Wrote', fn

        elif i == 2:
            if wcsfns is None:
                continue
            plot.alpha = 0.5
            for fn in wcsfns:
                out.set_wcs_file(fn, 0)
                out.fill = 1
                plot.plot('outline')
                out.fill = 0
                plot.plot('outline')


        plot.color = 'gray'
        plot.alpha = 1.
        grid = plot.grid
        grid.ralabeldir = 2
        grid.ralo = 120
        grid.rahi = 200
        grid.declo = 30
        grid.dechi = 60
        plot.plot_grid(5, 5, 20, 10)
        plot.color = 'red'
        plot.apply_settings()
        plot.line_constant_dec(d0, r0, r1)
        plot.stroke()
        plot.line_constant_ra(r1, d0, d1)
        plot.stroke()
        plot.line_constant_dec(d1, r1, r0)
        plot.stroke()
        plot.line_constant_ra(r0, d1, d0)
        plot.stroke()
        fn = ps.getnext()
        plot.write(fn)
        print 'Wrote', fn


def coadd_wise(cowcs, WISE, ps, band, table=True):
    L = 3
    W = cowcs.get_width()
    H = cowcs.get_height()
    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16

    rimgs, coimg1, cow1, coppstd1 = _coadd_wise_round1(cowcs, WISE, ps, band, table, L,
                                                       tinyw)

    # Using the difference between the coadd and the resampled
    # individual images ("rchi"), mask additional pixels and redo the
    # coadd.
    coimg  = np.zeros((H,W))
    coimg2 = np.zeros((H,W))
    cow    = np.zeros((H,W))
    coimgb = np.zeros_like(coimg)
    cowb   = np.zeros_like(cow)

    masks = []
    ri = 0
    while len(rimgs):
        rr = rimgs.pop(0)
        ri += 1

        print
        print 'Coadd round 2, image', (ri+1), 'of', len(WISE)

        t00 = Time()

        if rr is None:
            masks.append(None)
            continue

        mm = Duck()
        mm.npatched = rr.npatched
        mm.ncopix = rr.ncopix
        mm.sky = rr.sky
        mm.zp = rr.zp

        cox0,cox1,coy0,coy1 = rr.coextent
        subco = coimg1  [coy0:coy1+1, cox0:cox1+1].astype(np.float32)
        subw  = cow1    [coy0:coy1+1, cox0:cox1+1]
        subpp = coppstd1[coy0:coy1+1, cox0:cox1+1]

        # like in the WISE Atlas Images, estimate sky difference via
        # median difference in the overlapping area.
        dsky = median_f(rr.rimg[rr.rmask] - subco[rr.rmask])
        print 'Sky difference:', dsky
        dsky /= rr.zpscale
        print 'scaled:', dsky
        mm.dsky = dsky

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

        ok = patch_image(rr.rimg, np.logical_not(badpix), required=badpix)
        if not ok:
            print 'patch_image failed; continuing'
            masks.append(None)
            continue

        # plt.clf()
        # plt.imshow(rimg - dsky, **ima)
        # plt.title('patched rimg - dsky')
        # plt.colorbar()
        # ps.savefig()

        coimg [coy0: coy1+1, cox0: cox1+1] += rr.w * rr.rimg
        coimg2[coy0: coy1+1, cox0: cox1+1] += rr.w * rr.rimg**2
        # About the [rr.rmask]: that is the area where [rr.rimg] != 0
        cow   [coy0: coy1+1, cox0: cox1+1][rr.rmask] += rr.w

        # Add rchi-masked pixels to the mask
        rr.rmask2[badpix] = False

        coimgb[coy0: coy1+1, cox0: cox1+1] += rr.w * rr.rimg * rr.rmask2
        cowb  [coy0: coy1+1, cox0: cox1+1] += rr.w * rr.rmask2

        # print 'Applying rchi masks to images...'
        mm.omask = np.zeros((rr.wcs.get_height(), rr.wcs.get_width()), badpixmask.dtype)
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(rr.wcs, rr.cosubwcs, [], None)
            mm.omask[Yo,Xo] = badpixmask[Yi,Xi]
        except OverlapError:
            import traceback
            print 'WARNING: Caught OverlapError resampling rchi mask'
            print 'rr WCS', rr.wcs
            print 'shape', mm.omask.shape
            print 'cosubwcs:', rr.cosubwcs
            traceback.print_exc()

        masks.append(mm)

        del badpix
        del badpixmask

        #print 'coadd_wise image', ri, 'second pass'
        print Time() - t00

    # print 'Coadd (before normalizing) range:', coimg.min(), coimg.max(), 'mean', np.mean(coimg), 'median', np.median(coimg)
    # print 'Coadd weight range:', cow.min(), cow.max(), 'median', np.median(cow)
    coimg /= np.maximum(cow, tinyw)
    coinvvar = cow
    coimgb /= np.maximum(cowb, tinyw)
    coinvvarb = cowb
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
    coppstd = np.sqrt(np.maximum(0, coimg2 / (np.maximum(cow, tinyw)) - coimg**2))
    #print 'Coadd per-pixel range:', coppstd.min(), coppstd.max()

    return coimg, coinvvar, coppstd, coimgb, coinvvarb, masks


def _coadd_wise_round1(cowcs, WISE, ps, band, table, L,
                       tinyw):
    W = cowcs.get_width()
    H = cowcs.get_height()
    coimg  = np.zeros((H,W))
    coimg2 = np.zeros((H,W))
    cow    = np.zeros((H,W))
    rimgs = []
    
    for wi,wise in enumerate(WISE):
        t00 = Time()
        print
        print 'Coadd round 1, image', (wi+1), 'of', len(WISE)
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

        #wcs = Sip(intfn)
        #print 'Wcs:', wcs
        wcs = wise.wcs
        h,w = wcs.get_height(), wcs.get_width()

        x0,x1,y0,y1 = wise.imextent

        wcs = wcs.get_subimage(int(x0), int(y0), int(1+x1-x0), int(1+y1-y0))

        with fitsio.FITS(intfn) as F:
            img = F[0][y0:y1+1, x0:x1+1]
            ihdr = F[0].read_header()
        mask = fitsio.FITS(maskfn)[0][y0:y1+1, x0:x1+1]
        unc  = fitsio.FITS(uncfn) [0][y0:y1+1, x0:x1+1]
        print 'Img:', img.shape, img.dtype
        print 'Unc:', unc.shape, unc.dtype
        print 'Mask:', mask.shape, mask.dtype

        zp = ihdr['MAGZP']
        zpscale = NanoMaggies.zeropointToScale(zp)
        print 'Zeropoint:', zp, '-> scale', zpscale

        goodmask = ((mask & sum([1<<bit for bit in [0,1,2,3,4,5,6,7, 9,
                                                    10,11,12,13,14,15,16,17,18,
                                                    21,26,27,28]])) == 0)
        goodmask[unc == 0] = False
        goodmask[np.logical_not(np.isfinite(img))] = False
        goodmask[np.logical_not(np.isfinite(unc))] = False

        sig1 = median_f(unc[goodmask])
        print 'sig1:', sig1

        del mask
        del unc

        # Patch masked pixels so we can interpolate
        rr.npatched = np.count_nonzero(np.logical_not(goodmask))
        ok = patch_image(img, goodmask)
        if not ok:
            print 'WARNING: Patching failed:'
            print 'Image size:', img.shape
            print 'Number to patch:', rr.npatched
            continue
        assert(np.all(np.isfinite(img)))

        # Estimate sky level via clipped medians
        med = median_f(img)
        ok = np.flatnonzero(np.abs(img - med) < 3.*sig1)
        sky = median_f(img.flat[ok])
        print 'Estimated sky level:', sky

        # Convert to nanomaggies
        img -= sky
        img *= zpscale
        #img = (img - sky) * zpscale
        sig1 *= zpscale

        # coadd subimage
        cox0,cox1,coy0,coy1 = wise.coextent
        cosubwcs = cowcs.get_subimage(int(cox0), int(coy0), int(1+cox1-cox0), int(1+coy1-coy0))
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(cosubwcs, wcs, [img], L, table=table)
                                                 
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
        rr.rmask2 = np.zeros((1+coy1-coy0, 1+cox1-cox0), np.bool)
        rr.rmask2[Yo, Xo] = goodmask[Yi, Xi]
        rr.w = w
        rr.wcs = wcs
        rr.sky = sky
        rr.zpscale = zpscale
        rr.zp = zp
        rr.ncopix = len(Yo)
        rr.coextent = wise.coextent
        rr.cosubwcs = cosubwcs
        rimgs.append(rr)

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
    coimg /= np.maximum(cow, tinyw)
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
    coppstd = np.sqrt(np.maximum(0, coimg2 / np.maximum(cow, tinyw) - coimg**2))
    # costd1 = np.median(coppstd)
    # comed = np.median(coimg1)
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

    return rimgs, coimg, cow, coppstd



def trymain():
    try:
        main()
    except:
        import traceback
        traceback.print_exc()

def _bounce_one_coadd(A):
    try:
        one_coadd(*A)
    except:
        import traceback
        print 'one_coadd failed:'
        traceback.print_exc()

if __name__ == '__main__':
    import optparse
    from astrometry.util.multiproc import multiproc

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--threads', dest='threads', type=int, help='Multiproc',
                      default=None)
    opt,args = parser.parse_args()
    if opt.threads:
        mp = multiproc(opt.threads)
    else:
        mp = multiproc()

    Time.add_measurement(MemMeas)

    batch = False
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        batch = True

    # d = os.environ.get('PBS_O_WORKDIR')
    # if batch and d is not None:
    #     os.chdir(d)
    #     sys.path.append(os.getcwd())

    if not batch:
        import cProfile
        from datetime import tzinfo, timedelta, datetime
        pfn = 'prof-%s.dat' % (datetime.now().isoformat())
        cProfile.run('trymain()', pfn)
        print 'Wrote', pfn
        sys.exit(0)

    dataset = 'sequels'

    # SEQUELS
    r0,r1 = 120.0, 200.0
    d0,d1 =  45.0,  60.0

    fn = '%s-atlas.fits' % dataset
    if os.path.exists(fn):
        print 'Reading', fn
        T = fits_table(fn)
    else:
        T = get_atlas_tiles(r0,r1,d0,d1)
        T.writeto(fn)

    fn = '%s-frames.fits' % dataset
    if os.path.exists(fn):
        print 'Reading', fn
        WISE = fits_table(fn)
    else:
        WISE = get_wise_frames(r0,r1,d0,d1)
        WISE.writeto(fn)

    #WISE.cut(np.logical_or(WISE.band == 1, WISE.band == 2))
    #check_md5s(WISE)

    ps = PlotSequence(dataset)

    if arr == 0:
        # Check which tiles still need to be done.
        need = []
        for band in bands:
            fns = []
            for i in range(len(T)):
                tag = 'coadd-%s-w%i' % (T.coadd_id[i], band)
                prefix = os.path.join(outdir, tag)
                ofn = prefix + '-img.fits'
                if os.path.exists(ofn):
                    print 'Output file exists:', ofn
                    fns.append(ofn)
                    continue
                need.append(band*1000 + i)

            if band == bands[0]:
                plot_region(r0,r1,d0,d1, ps, T, None, fns)
            else:
                plot_region(r0,r1,d0,d1, ps, None, None, fns)

        print ' '.join('%i' %i for i in need)

        # write out scripts
        for i in need:
            script = '\n'.join(['#! /bin/bash',
                                ('#PBS -N %s-%i' % (dataset, i)),
                                '#PBS -l cput=1:00:00',
                                '#PBS -l pvmem=4gb',
                                'cd $PBS_O_WORKDIR',
                                ('export PBS_ARRAYID=%i' % i),
                                './wise-coadd.py',
                                ''])
                                
            sfn = 'pbs-%s-%i.sh' % (dataset, i)
            write_file(script, sfn)
            os.system('chmod 755 %s' % sfn)

        # Collapse contiguous ranges
        strings = []
        start = need.pop(0)
        end = start
        while len(need):
            x = need.pop(0)
            if x == end + 1:
                # extend this run
                end = x
            else:
                # run finished; output and start new one.
                if start == end:
                    strings.append('%i' % start)
                else:
                    strings.append('%i-%i' % (start, end))
                start = end = x
        # done; output
        if start == end:
            strings.append('%i' % start)
        else:
            strings.append('%i-%i' % (start, end))
        print ','.join(strings)
        sys.exit(0)
            
    if len(args):
        A = []
        for a in args:
            tileid = int(a)
            #print 'Running index', tileid
            band = tileid / 1000
            tileid = tileid % 1000
            print 'Doing coadd tile', T.coadd_id[tileid], 'band', band
            A.append((T[tileid], band, WISE, ps))
        mp.map(_bounce_one_coadd, A)

        sys.exit(0)


    #plot_region(r0,r1,d0,d1, ps, T, WISE, None)

    band = arr / 1000
    assert(band in bands)
    tile = arr % 1000
    assert(tile < len(T))

    print 'Doing coadd tile', T.coadd_id[tile], 'band', band

    t0 = Time()
    one_coadd(T[tile], band, WISE, ps)
    print 'Tile', T.coadd_id[tile], 'band', band, 'took:', Time()-t0

