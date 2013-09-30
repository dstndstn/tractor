#! /usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
import tempfile
from scipy.ndimage.morphology import binary_dilation
import datetime

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
from astrometry.util.run_command import *
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
default_outdir = 'wise-coadds'
pixscale = 2.75 / 3600.

W,H = 2048, 2048
#W,H = 1024,1024

bands = [1,2,3,4]
# WISE Level 1b inputs
wisedir = 'wise-frames'
mask_gz = True
unc_gz = True


class Duck():
    pass

def get_coadd_tile_wcs(ra, dec):
    cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                -pixscale, 0., 0., pixscale, W, H)
    return cowcs

def walk_wcs_boundary(wcs, step=1024, margin=0):
    '''
    Walk the image boundary counter-clockwise.
    '''
    W = wcs.get_width()
    H = wcs.get_height()
    xlo = 1
    xhi = W
    ylo = 1
    yhi = H
    if margin:
        xlo -= margin
        ylo -= margin
        xhi += margin
        yhi += margin
    
    xx,yy = [],[]
    xwalk = np.linspace(xlo, xhi, int(np.ceil((1+xhi-xlo)/float(step)))+1)
    ywalk = np.linspace(ylo, yhi, int(np.ceil((1+yhi-ylo)/float(step)))+1)
    # bottom edge
    x = xwalk[:-1]
    y = ylo
    xx.append(x)
    yy.append(np.zeros_like(x) + y)
    # right edge
    x = xhi
    y = ywalk[:-1]
    xx.append(np.zeros_like(y) + x)
    yy.append(y)
    # top edge
    x = list(reversed(xwalk))[:-1]
    y = yhi
    xx.append(x)
    yy.append(np.zeros_like(x) + y)
    # left edge
    x = xlo
    y = list(reversed(ywalk))[:-1]
    # (note, NOT closed)
    xx.append(np.zeros_like(y) + x)
    yy.append(y)
    #
    rr,dd = wcs.pixelxy2radec(np.hstack(xx), np.hstack(yy))
    return rr,dd

def get_wcs_radec_bounds(wcs):
    rr,dd = walk_wcs_boundary(wcs)
    r0,r1 = rr.min(), rr.max()
    d0,d1 = dd.min(), dd.max()
    return r0,r1,d0,d1

def get_atlas_tiles(r0,r1,d0,d1):
    '''
    Select Atlas Image tiles touching a desired RA,Dec box.
    '''
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

    # Some of them don't *actually* touch our RA,Dec box...
    print 'Checking tile RA,Dec bounds...'
    keep = []
    for i in range(len(T)):
        wcs = get_coadd_tile_wcs(T.ra[i], T.dec[i])
        R0,R1,D0,D1 = get_wcs_radec_bounds(wcs)
        if R1 < r0 or R0 > r1 or D1 < d0 or D0 > d1:
            print 'Coadd tile', T.coadd_id[i], 'is outside RA,Dec box'
            continue
        keep.append(i)
    T.cut(np.array(keep))
    print 'Cut to', len(T), 'tiles'

    return T


def get_wise_frames(r0,r1,d0,d1, margin=2.):
    '''
    Returns WISE frames touching the given RA,Dec box plus margin.
    '''
    # Read WISE frame metadata
    WISE = fits_table(os.path.join(wisedir, 'WISE-index-L1b.fits'))
    print 'Read', len(WISE), 'WISE L1b frames'
    WISE.row = np.arange(len(WISE))

    # Coarse cut on RA,Dec box.
    cosdec = np.cos(np.deg2rad(max(abs(d0),abs(d1))))
    WISE.cut((WISE.ra + margin/cosdec > r0) *
             (WISE.ra - margin/cosdec < r1) *
             (WISE.dec + margin > d0) *
             (WISE.dec - margin < d1))
    print 'Cut to', len(WISE), 'WISE frames near RA,Dec box'

    # Join to WISE Single-Frame Metadata Tables
    WISE.qual_frame = np.zeros(len(WISE), np.int16) - 1
    WISE.moon_masked = np.zeros(len(WISE), bool)
    WISE.dtanneal = np.zeros(len(WISE), np.float32)

    WISE.matched = np.zeros(len(WISE), bool)
    
    for nbands in [2,3,4]:
        fn = os.path.join(wisedir, 'WISE-l1b-metadata-%iband.fits' % nbands)
        T = fits_table(fn, columns=['ra', 'dec', 'scan_id', 'frame_num',
                                    'qual_frame', 'moon_masked', 'dtanneal'])
        print 'Read', len(T), 'from', fn
        # Cut with extra large margins
        T.cut((T.ra  + 2.*margin/cosdec > r0) *
              (T.ra  - 2.*margin/cosdec < r1) *
              (T.dec + 2.*margin > d0) *
              (T.dec - 2.*margin < d1))
        print 'Cut to', len(T), 'near RA,Dec box'
        if len(T) == 0:
            continue

        I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 60./3600.)
        print 'Matched', len(I)
        K = np.flatnonzero((WISE.scan_id  [I] == T.scan_id  [J]) *
                           (WISE.frame_num[I] == T.frame_num[J]))
        I = I[K]
        J = J[K]
        print 'Cut to', len(I), 'matching scan/frame'

        for band in [1,2,3,4]:
            K = (WISE.band[I] == band)
            print 'Band', band, ':', sum(K)
            if sum(K) == 0:
                continue
            WISE.qual_frame [I[K]] = T.qual_frame [J[K]].astype(WISE.qual_frame.dtype)
            moon = T.moon_masked[J[K]]
            print 'Moon:', np.unique(moon)
            print 'moon[%i]:' % (band-1), np.unique([m[band-1] for m in moon])
            WISE.moon_masked[I[K]] = np.array([m[band-1] == '1' for m in moon]).astype(WISE.moon_masked.dtype)
            WISE.dtanneal   [I[K]] = T.dtanneal[J[K]].astype(WISE.dtanneal.dtype)
            print 'moon_masked:', np.unique(WISE.moon_masked)
            WISE.matched[I[K]] = True

    print np.sum(WISE.matched), 'of', len(WISE), 'matched to metadata tables'
    assert(np.sum(WISE.matched) == len(WISE))
    WISE.delete_column('matched')
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

def one_coadd(ti, band, WISE, ps, wishlist, outdir, mp, do_cube):
    print 'Coadd tile', ti.coadd_id
    print 'RA,Dec', ti.ra, ti.dec
    print 'Band', band

    version = {}
    rtn,out,err = run_command('svn info')
    assert(rtn == 0)
    lines = out.split('\n')
    lines = [l for l in lines if len(l)]
    for l in lines:
        words = l.split(':', 1)
        words = [w.strip() for w in words]
        version[words[0]] = words[1]
    print 'SVN version info:', version

    tag = 'unwise-%s-w%i' % (ti.coadd_id, band)
    prefix = os.path.join(outdir, tag)
    ofn = prefix + '-img.fits'
    if os.path.exists(ofn):
        print 'Output file exists:', ofn
        return

    cowcs = get_coadd_tile_wcs(ti.ra, ti.dec)

    copoly = np.array(zip(*walk_wcs_boundary(cowcs, step=W/2., margin=10)))
    print 'Copoly:', copoly

    if ps:
        plt.clf()
        plt.plot(WISE.ra, WISE.dec, 'b.', ms=10, alpha=0.5)
        plt.plot(np.append(copoly[:,0],copoly[0,0]),
                 np.append(copoly[:,1],copoly[0,1]), 'r-')
        plt.title('Before cuts')
        ps.savefig()
    
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
    #I = np.argsort(degrees_between(ti.ra, ti.dec, WISE.ra, WISE.dec))
    #WISE.cut(I)
    # DEBUG
    I = np.argsort(-WISE.dec)
    WISE.cut(I)
    
    if ps:
        plt.clf()
        plt.plot(WISE.ra, WISE.dec, 'b.', ms=10, alpha=0.5)
        plt.plot(np.append(copoly[:,0],copoly[0,0]),
                 np.append(copoly[:,1],copoly[0,1]), 'r-')
        plt.title('Circle cut')

        inter = np.zeros(len(WISE), bool)
        for wi,wise in enumerate(WISE):
            intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
            wcs = Sip(intfn)
            h,w = wcs.get_height(), wcs.get_width()
            poly = np.array(zip(*walk_wcs_boundary(wcs, step=2.*w, margin=10)))
            intersects = polygons_intersect(copoly, poly)
            inter[wi] = intersects
            cc = 'b'
            alpha = 0.1
            if not intersects:
                cc = 'r'
                alpha = 0.5
            plt.plot(np.append(poly[:,0],poly[0,0]),
                     np.append(poly[:,1],poly[0,1]), '-', color=cc, alpha=alpha)
        plt.plot(WISE.ra[inter==False], WISE.dec[inter==False], 'r.', ms=10, alpha=0.5)
        print sum(inter), 'intersecting fields'
        ps.savefig()

    # cut on RA,Dec box too
    r0,d0 = copoly.min(axis=0)
    r1,d1 = copoly.max(axis=0)
    dd = np.sqrt(2.) * (1016./2.) * pixscale * 1.01 # safety
    dr = dd / min([np.cos(np.deg2rad(d)) for d in [d0,d1]])
    WISE.cut((WISE.ra  + dr >= r0) * (WISE.ra  - dr <= r1) *
             (WISE.dec + dd >= d0) * (WISE.dec - dd <= d1))
    print 'cut to', len(WISE), 'in RA,Dec box'

    if ps:
        plt.clf()
        plt.plot(WISE.ra, WISE.dec, 'b.', ms=10, alpha=0.5)
        plt.plot(np.append(copoly[:,0],copoly[0,0]),
                 np.append(copoly[:,1],copoly[0,1]), 'r-')
        plt.title('Box cut')

        inter = np.zeros(len(WISE), bool)
        for wi,wise in enumerate(WISE):
            intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
            wcs = Sip(intfn)
            h,w = wcs.get_height(), wcs.get_width()
            poly = np.array(zip(*walk_wcs_boundary(wcs, step=2.*w, margin=10)))
            intersects = polygons_intersect(copoly, poly)
            inter[wi] = intersects
            cc = 'b'
            alpha = 0.1
            if not intersects:
                cc = 'r'
                alpha = 0.5
            plt.plot(np.append(poly[:,0],poly[0,0]),
                     np.append(poly[:,1],poly[0,1]), '-', color=cc, alpha=alpha)
        print sum(inter), 'intersecting fields'
        plt.plot(WISE.ra[inter==False], WISE.dec[inter==False], 'r.', ms=10, alpha=0.5)

        ps.savefig()

    print 'Qual_frame scores:', np.unique(WISE.qual_frame)
    WISE.cut(WISE.qual_frame > 0)
    print 'Cut out qual_frame = 0;', len(WISE), 'remaining'
    
    print 'Moon_masked:', np.unique(WISE.moon_masked)
    WISE.cut(WISE.moon_masked == False)
    print 'Cut moon_masked:', len(WISE), 'remaining'

    if band in [3,4]:
        WISE.cut(WISE.dtanneal > 2000.)
        print 'Cut out dtanneal <= 2000 seconds:', len(WISE), 'remaining'

    if band == 4:
        ok = np.array([np.logical_or(s < '03752a', s > '03761b')
                       for s in WISE.scan_id])
        WISE.cut(ok)
        print 'Cut out bad scans in W4:', len(WISE), 'remaining'

    if ps:
        plt.clf()
        plt.plot(WISE.ra, WISE.dec, 'b.', ms=10, alpha=0.5)
        plt.plot(np.append(copoly[:,0],copoly[0,0]),
                 np.append(copoly[:,1],copoly[0,1]), 'r-')
        plt.title('Quality cuts')

        inter = np.zeros(len(WISE), bool)
        for wi,wise in enumerate(WISE):
            intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
            wcs = Sip(intfn)
            h,w = wcs.get_height(), wcs.get_width()
            poly = np.array(zip(*walk_wcs_boundary(wcs, step=2.*w, margin=10)))
            intersects = polygons_intersect(copoly, poly)
            inter[wi] = intersects
            cc = 'b'
            alpha = 0.1
            if not intersects:
                cc = 'r'
                alpha = 0.5
            plt.plot(np.append(poly[:,0],poly[0,0]),
                     np.append(poly[:,1],poly[0,1]), '-', color=cc, alpha=alpha)
        print sum(inter), 'intersecting fields'
        plt.plot(WISE.ra[inter==False], WISE.dec[inter==False], 'r.', ms=10, alpha=0.5)

        ps.savefig()

    if wishlist:
        for wise in WISE:
            intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
            if not os.path.exists(intfn):
                print 'Need:', intfn
        return

    # *inclusive* coordinates of the bounding-box in the coadd of this image
    # (x0,x1,y0,y1)
    WISE.coextent = np.zeros((len(WISE), 4), int)
    # *inclusive* coordinates of the bounding-box in the image overlapping coadd
    WISE.imextent = np.zeros((len(WISE), 4), int)

    if ps:
        plt.clf()
        plt.plot(np.append(copoly[:,0],copoly[0,0]),
                 np.append(copoly[:,1],copoly[0,1]), 'r-')
        ninter = 0

    res = []
    for wi,wise in enumerate(WISE):
        print
        print (wi+1), 'of', len(WISE)
        intfn = get_l1b_file(wisedir, wise.scan_id, wise.frame_num, band)
        print 'intfn', intfn
        wcs = Sip(intfn)

        h,w = wcs.get_height(), wcs.get_width()
        poly = np.array(zip(*walk_wcs_boundary(wcs, step=2.*w, margin=10)))
        intersects = polygons_intersect(copoly, poly)

        if ps:
            cc = 'b'
            alpha = 0.1
            if not intersects:
                cc = 'r'
                alpha = 0.5
            else:
                ninter += 1
            plt.plot(np.append(poly[:,0],poly[0,0]),
                     np.append(poly[:,1],poly[0,1]), '-', color=cc, alpha=alpha)

        #print 'poly:', poly
        if not intersects:
            print 'Image does not intersect target'
            res.append(None)
            continue
        res.append((intfn, wcs, w, h, poly))

        cpoly = clip_polygon(copoly, poly)
        #print 'Clipped polygon:', cpoly
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
        print 'Coadd extent:', WISE.coextent[wi,:]

    if ps:
        print 'Number intersecting:', ninter
        ps.savefig()

        m = 0.05
        plt.axis([r0-m, r1+m, d0-m, d1+m])
        ps.savefig()

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

    t1 = Time()
    print 'Up to coadd_wise:'
    print t1 - t0

    try:
        (coim,coiv,copp,con, coimb,coivb,coppb,conb,masks, cube,
         )= coadd_wise(cowcs, WISE, ps, band, mp, do_cube)
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

    hdr.add_record(dict(name='UNW_VER', value=version['Revision'],
                        comment='unWISE code SVN revision'))
    hdr.add_record(dict(name='UNW_URL', value=version['URL'], comment='SVN URL'))
    hdr.add_record(dict(name='UNW_DATE', value=datetime.datetime.now().isoformat(),
                        comment='unWISE run time'))

    ofn = prefix + '-img.fits'
    fitsio.write(ofn, coim.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-invvar.fits'
    fitsio.write(ofn, coiv.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-ppstd.fits'
    fitsio.write(ofn, copp.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-n.fits'
    fitsio.write(ofn, con.astype(np.int16), header=hdr, clobber=True)

    ofn = prefix + '-img-w.fits'
    fitsio.write(ofn, coimb.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-invvar-w.fits'
    fitsio.write(ofn, coivb.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-ppstd-w.fits'
    fitsio.write(ofn, coppb.astype(np.float32), header=hdr, clobber=True)
    ofn = prefix + '-n-w.fits'
    fitsio.write(ofn, conb.astype(np.int16), header=hdr, clobber=True)

    if do_cube:
        ofn = prefix + '-cube.fits'
        fitsio.write(ofn, cube.astype(np.float32), header=hdr, clobber=True)

    ii = []
    for i,(mm,r) in enumerate(zip(masks, res)):
        if mm is None:
            continue
        ii.append(i)

        if not mm.included:
            continue

        maskdir = os.path.join(outdir, 'masks-' + tag)
        if not os.path.exists(maskdir):
            os.mkdir(maskdir)

        ofn = WISE.intfn[i].replace('-int', '')
        ofn = os.path.join(maskdir, 'unwise-mask-' + ti.coadd_id + '-'
                           + os.path.basename(ofn))
        (nil,wcs,w,h,poly) = r
        fullmask = np.zeros((h,w), mm.omask.dtype)
        x0,x1,y0,y1 = WISE.imextent[i,:]
        fullmask[y0:y1+1, x0:x1+1] = mm.omask
        fitsio.write(ofn, fullmask, clobber=True)

        cmd = 'gzip -f %s' % ofn
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
    WISE.included    = np.array([m.included for m in masks]).astype(np.uint8)

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
                cowcs = get_coadd_tile_wcs(ti.ra, ti.dec)
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


def _coadd_one_round2((rr, cow1, cowimg1, cowimgsq1, tinyw, plotfn)):
    if rr is None:
        return None
    t00 = Time()
    mm = Duck()
    mm.npatched = rr.npatched
    mm.ncopix = rr.ncopix
    mm.sky = rr.sky
    mm.zp = rr.zp
    mm.included = True

    cox0,cox1,coy0,coy1 = rr.coextent
    coslc = slice(coy0, coy1+1), slice(cox0, cox1+1)
    # Remove this image from the per-pixel std calculation...
    subw  = np.maximum(cow1[coslc] - rr.w, tinyw)
    subco = (cowimg1  [coslc] - (rr.w * rr.rimg   )) / subw
    subsq = (cowimgsq1[coslc] - (rr.w * rr.rimg**2)) / subw
    subpp = np.sqrt(np.maximum(0, subsq - subco**2))

    # like in the WISE Atlas Images, estimate sky difference via
    # median difference in the overlapping area.
    dsky = median_f((rr.rimg[rr.rmask] - subco[rr.rmask]).astype(np.float32))
    print 'Sky difference:', dsky

    rchi = ((rr.rimg - dsky - subco) * rr.rmask * (subw > 0) * (subpp > 0) /
            np.maximum(subpp, 1e-6))
    #print 'rchi', rchi.min(), rchi.max()
    assert(np.all(np.isfinite(rchi)))
    badpix = (np.abs(rchi) >= 5.)
    #print 'Number of rchi-bad pixels:', np.count_nonzero(badpix)
    mm.nrchipix = np.count_nonzero(badpix)

    # Bit 1: abs(rchi) >= 5
    badpixmask = badpix.astype(np.uint8)
    # grow by a small margin
    badpix = binary_dilation(badpix)
    # Bit 2: grown
    badpixmask += (2 * badpix)
    # Add rchi-masked pixels to the mask
    rr.rmask2[badpix] = False
    # print 'Applying rchi masks to images...'
    mm.omask = np.zeros((rr.wcs.get_height(), rr.wcs.get_width()),
                        badpixmask.dtype)
    try:
        Yo,Xo,Yi,Xi,nil = resample_with_wcs(rr.wcs, rr.cosubwcs, [], None)
        mm.omask[Yo,Xo] = badpixmask[Yi,Xi]
    except OverlapError:
        import traceback
        print 'WARNING: Caught OverlapError resampling rchi mask'
        print 'rr WCS', rr.wcs
        print 'shape', mm.omask.shape
        print 'cosubwcs:', rr.cosubwcs
        traceback.print_exc(None, sys.stdout)

    if mm.nrchipix > mm.ncopix * 0.01:
        print (('WARNING: dropping exposure scan %s frame %i band %i:' +
                + '# nrchi pixels %i') % (
                    WISE.scan_id[ri], WISE.frame_num[ri], band, mm.nrchipix))
        mm.included = False

    if mm.included:
        ok = patch_image(rr.rimg, np.logical_not(badpix),
                         required=(badpix * rr.rmask))
        if not ok:
            print 'patch_image failed'
            return None

        rimg = (rr.rimg - dsky)
        #rr.rimg[rr.rmask] -= dsky

        mm.coslc = coslc
        mm.coimgsq = rr.rmask * rr.w * rimg**2
        mm.coimg   = rr.rmask * rr.w * rimg
        mm.cow     = rr.rmask * rr.w
        mm.con     = rr.rmask
        mm.rmask2  = rr.rmask2
        # mm.coimgsqb = rr.rmask2 * rr.w * rimg**2
        # mm.coimgb   = rr.rmask2 * rr.w * rimg
        # mm.cowb     = rr.rmask2 * rr.w
        # mm.conb     = rr.rmask2

    dsky /= rr.zpscale
    print 'scaled:', dsky
    mm.dsky = dsky

        
    if plotfn:

        # HACK
        rchihistrange = 6
        rchihistargs = dict(range=(-rchihistrange,rchihistrange), bins=100)
        rchihist = None
        rchihistedges = None

        R,C = 3,3
        plt.clf()
        plt.subplot(R,C,1)
        I = rr.rimg - dsky
        plo,phi = [np.percentile(I[rr.rmask], p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('rimg')
        plt.subplot(R,C,2)
        I = subco
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('subco')
        plt.subplot(R,C,3)
        I = subpp
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('subpp')
        plt.subplot(R,C,4)
        plt.imshow(rchi, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=-5, vmax=5)
        plt.xticks([]); plt.yticks([])
        plt.title('rchi (%i)' % mm.nrchipix)

        plt.subplot(R,C,5)
        I = rr.img
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('img')

        plt.subplot(R,C,6)
        I = mm.omask
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=0, vmax=3)
        plt.xticks([]); plt.yticks([])
        plt.title('omask')

        plt.subplot(R,C,7)
        I = rr.rimg
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.xticks([]); plt.yticks([])
        plt.title('patched rimg')

        # plt.subplot(R,C,8)
        # I = (coimgb / np.maximum(cowb, tinyw))
        # plo,phi = [np.percentile(I, p) for p in [25,99]]
        # plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
        #            vmin=plo, vmax=phi)
        # plt.xticks([]); plt.yticks([])
        # plt.title('coimgb')

        I = (rchi != 0.)
        n,e = np.histogram(np.clip(rchi[I], -rchihistrange, rchihistrange),
                           **rchihistargs)
        if rchihist is None:
            rchihist, rchihistedges = n,e
        else:
            rchihist += n

        plt.subplot(R,C,9)
        e = rchihistedges
        e = (e[:-1]+e[1:])/2.
        #plt.semilogy(e, np.maximum(0.1, rchihist), 'b-')
        plt.semilogy(e, np.maximum(0.1, n), 'b-')
        plt.axvline(5., color='r')
        plt.xlim(-(rchihistrange+1), rchihistrange+1)
        plt.yticks([])
        plt.title('rchi')

        inc = ''
        if not mm.included:
            inc = '(not incl)'
        plt.suptitle('%s %i %s' % (WISE.scan_id[ri], WISE.frame_num[ri], inc))
        plt.savefig(plotfn)

    print Time() - t00
        
    return mm
        

def coadd_wise(cowcs, WISE, ps, band, mp, do_cube, table=True):
    L = 3
    W = cowcs.get_width()
    H = cowcs.get_height()
    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16

    # DEBUG
    #WISE = WISE[:10]
    # DEBUG -- scan closest to outlier 03833a
    #WISE.hexscan = np.array([int(s, 16) for s in WISE.scan_id])
    #WISE.cut(np.lexsort((WISE.frame_num, np.abs(WISE.hexscan - int('03833a', 16)))))
    #WISE.cut(np.lexsort((WISE.frame_num, WISE.scan_id)))

    (rimgs, coimg1, cow1, coppstd1, cowimgsq1
     )= _coadd_wise_round1(cowcs, WISE, ps, band, table, L, tinyw, mp)
    cowimg1 = coimg1 * cow1

    # Using the difference between the coadd and the resampled
    # individual images ("rchi"), mask additional pixels and redo the
    # coadd.

    assert(len(rimgs) == len(WISE))

    cube = None
    if do_cube:
        cube = np.zeros((len(rimgs), H, W), np.float32)
        cubei = 0


    # If we're not multiprocessing, do the loop manually to reduce
    # memory usage (we don't need to keep all "rr" inputs and "masks"
    # outputs in memory at once).
    if not mp.pool:
        masks = []
        ri = -1
        while len(rimgs):
            ri += 1
            rr = rimgs.pop(0)
            print
            print 'Coadd round 2, image', (ri+1), 'of', len(WISE)
            if ps:
                plotfn = ps.getnext()
            else:
                plotfn = None
            masks.append(_coadd_one_round2(
                (rr, cow1, cowimg1, cowimgsq1, tinyw, plotfn)))
    else:
        args = []
        for rr in rimgs:
            if ps:
                plotfn = ps.getnext()
            else:
                plotfn = None
            args.append((rr, cow1, cowimg1, cowimgsq1, tinyw, plotfn))
        masks = mp.map(_coadd_one_round2, args)

    coimg    = np.zeros((H,W))
    coimgsq  = np.zeros((H,W))
    cow      = np.zeros((H,W))
    con      = np.zeros((H,W), np.int16)
    coimgb   = np.zeros((H,W))
    coimgsqb = np.zeros((H,W))
    cowb     = np.zeros((H,W))
    conb     = np.zeros((H,W), np.int16)

    for mm in masks:
        if mm is None or not mm.included:
            continue
        coimgsq [mm.coslc] += mm.coimgsq
        coimg   [mm.coslc] += mm.coimg
        cow     [mm.coslc] += mm.cow
        con     [mm.coslc] += mm.con
        coimgsqb[mm.coslc] += mm.rmask2 * mm.coimgsq
        coimgb  [mm.coslc] += mm.rmask2 * mm.coimg
        cowb    [mm.coslc] += mm.rmask2 * mm.cow
        conb    [mm.coslc] += mm.rmask2 * mm.con
        
        if do_cube:
            cube[(cubei,) + coslc] = (mm.coimgb).astype(cube.dtype)
            cubei += 1

    coimg /= np.maximum(cow, tinyw)
    coinvvar = cow

    coimgb /= np.maximum(cowb, tinyw)
    coinvvarb = cowb

    # per-pixel variance
    coppstd  = np.sqrt(np.maximum(0, coimgsq  / (np.maximum(cow,  tinyw)) - coimg **2))
    coppstdb = np.sqrt(np.maximum(0, coimgsqb / (np.maximum(cowb, tinyw)) - coimgb**2))

    if ps:
        plt.clf()
        I = coimg1
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 1')
        ps.savefig()

        plt.clf()
        I = coppstd1
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd per-pixel std 1')
        ps.savefig()

        plt.clf()
        I = coimg
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2')
        ps.savefig()

        plt.clf()
        I = coimgb
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 (weighted)')
        ps.savefig()

        imlo,imhi = plo,phi

        plt.clf()
        I = coppstd
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 per-pixel std')
        ps.savefig()

        plt.clf()
        I = coppstdb
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 per-pixel std (weighted)')
        ps.savefig()

        nmax = max(con.max(), conb.max())

        plt.clf()
        I = coppstd
        plo,phi = [np.percentile(I, p) for p in [25,99]]
        plt.imshow(I, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=plo, vmax=phi)
        plt.colorbar()
        plt.title('Coadd round 2 per-pixel std')
        ps.savefig()


    return coimg, coinvvar, coppstd, con, coimgb, coinvvarb, coppstdb, conb, masks, cube


def _coadd_one_round1((i, N, wise, table, L, ps, band, cowcs)):
    t00 = Time()
    print
    print 'Coadd round 1, image', (i+1), 'of', N
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

    wcs = wise.wcs
    x0,x1,y0,y1 = wise.imextent
    cox0,cox1,coy0,coy1 = wise.coextent

    coW = int(1 + cox1 - cox0)
    coH = int(1 + coy1 - coy0)

    wcs = wcs.get_subimage(int(x0), int(y0), int(1+x1-x0), int(1+y1-y0))
    with fitsio.FITS(intfn) as F:
        img = F[0][y0:y1+1, x0:x1+1]
        ihdr = F[0].read_header()
    mask = fitsio.FITS(maskfn)[0][y0:y1+1, x0:x1+1]
    unc  = fitsio.FITS(uncfn) [0][y0:y1+1, x0:x1+1]
    #print 'Img:', img.shape, img.dtype
    #print 'Unc:', unc.shape, unc.dtype
    #print 'Mask:', mask.shape, mask.dtype
    zp = ihdr['MAGZP']
    zpscale = 1. / NanoMaggies.zeropointToScale(zp)
    print 'Zeropoint:', zp, '-> scale', zpscale

    if band == 4:
        # In W4, the WISE single-exposure images are binned down
        # 2x2, so we are effectively splitting each pixel into 4
        # sub-pixels.  Spread out the flux.
        zpscale *= 0.25

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

    # our return value (quack):
    rr = Duck()
    # Patch masked pixels so we can interpolate
    rr.npatched = np.count_nonzero(np.logical_not(goodmask))
    print 'Pixels to patch:', rr.npatched
    if rr.npatched > 100000:
        print 'WARNING: too many pixels to patch:', rr.npatched
        return None
    ok = patch_image(img, goodmask)
    if not ok:
        print 'WARNING: Patching failed:'
        print 'Image size:', img.shape
        print 'Number to patch:', rr.npatched
        return None
    assert(np.all(np.isfinite(img)))

    # Estimate sky level via clipped medians
    med = median_f(img)
    ok = np.flatnonzero(np.abs(img - med) < 3.*sig1)
    sky = median_f(img.flat[ok])
    print 'Estimated sky level:', sky

    # Convert to nanomaggies
    img -= sky
    img *= zpscale
    sig1 *= zpscale

    # coadd subimage
    cosubwcs = cowcs.get_subimage(int(cox0), int(coy0), coW, coH)
    try:
        Yo,Xo,Yi,Xi,rims = resample_with_wcs(cosubwcs, wcs, [img], L, table=table)
    except OverlapError:
        print 'No overlap; skipping'
        return None
    rim = rims[0]
    assert(np.all(np.isfinite(rim)))

    print 'Pixels in range:', len(Yo)
    #print 'Added to coadd: range', rim.min(), rim.max(), 'mean', np.mean(rim), 'median', np.median(rim)

    if ps:
        # save for later...
        rr.img = img

    # Scalar!
    rr.w = (1./sig1**2)
    rr.rmask = np.zeros((coH, coW), np.bool)
    rr.rmask[Yo, Xo] = True
    rr.rimg = np.zeros((coH, coW), img.dtype)
    rr.rimg[Yo, Xo] = rim
    rr.rmask2 = np.zeros((coH, coW), np.bool)
    rr.rmask2[Yo, Xo] = goodmask[Yi, Xi]
    rr.wcs = wcs
    rr.sky = sky
    rr.zpscale = zpscale
    rr.zp = zp
    rr.ncopix = len(Yo)
    rr.coextent = wise.coextent
    rr.cosubwcs = cosubwcs

    print 'coadd_wise image', (i+1), 'first pass'
    print Time() - t00

    return rr


def _coadd_wise_round1(cowcs, WISE, ps, band, table, L,
                       tinyw, mp):
    W = cowcs.get_width()
    H = cowcs.get_height()
    coimg  = np.zeros((H,W))
    coimgsq = np.zeros((H,W))
    cow    = np.zeros((H,W))

    args = []
    for wi,wise in enumerate(WISE):
        args.append((wi, len(WISE), wise, table, L, ps, band, cowcs))
    rimgs = mp.map(_coadd_one_round1, args)
    del args

    for rr in rimgs:
        if rr is None:
            continue
        cox0,cox1,coy0,coy1 = rr.coextent
        slc = slice(coy0,coy1+1), slice(cox0,cox1+1)

        # note, rr.w is a scalar.
        coimgsq[slc] += rr.w * (rr.rimg**2)
        coimg  [slc] += rr.w *  rr.rimg
        cow    [slc] += rr.w *  rr.rmask

    print 'Min cow (round 1):', cow.min()
        
    coimg /= np.maximum(cow, tinyw)
    # Per-pixel std
    coppstd = np.sqrt(np.maximum(0, coimgsq / np.maximum(cow, tinyw) - coimg**2))

    return rimgs, coimg, cow, coppstd, coimgsq



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

def main():
    import optparse
    from astrometry.util.multiproc import multiproc

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--threads', dest='threads', type=int, help='Multiproc',
                      default=None)
    parser.add_option('--todo', dest='todo', action='store_true', default=False,
                      help='Print and plot fields to-do')
    parser.add_option('-w', dest='wishlist', action='store_true', default=False,
                      help='Print needed frames and exit?')
    parser.add_option('--plots', dest='plots', action='store_true', default=False)

    parser.add_option('--plot-prefix', dest='plotprefix', default=None)

    parser.add_option('--outdir', dest='outdir', default=default_outdir,
                      help='Output directory: default %default')

    parser.add_option('--size', dest='size', default=None, type=int,
                      help='Set output image size -- DEBUGGING ONLY!')
    parser.add_option('--cube', dest='cube', action='store_true', default=False,
                      help='Save & write out image cube')

    opt,args = parser.parse_args()
    if opt.threads:
        mp = multiproc(opt.threads)
    else:
        mp = multiproc()

    if opt.size:
        global W,H
        W = H = opt.size

    Time.add_measurement(MemMeas)

    batch = False
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        batch = True

    # if not batch:
    #     import cProfile
    #     from datetime import tzinfo, timedelta, datetime
    #     pfn = 'prof-%s.dat' % (datetime.now().isoformat())
    #     cProfile.run('trymain()', pfn)
    #     print 'Wrote', pfn
    #     sys.exit(0)

    dataset = 'sequels'

    # SEQUELS
    r0,r1 = 120.0, 210.0
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

    if opt.plotprefix is None:
        opt.plotprefix = dataset
    ps = PlotSequence(opt.plotprefix, format='%03i')

    if opt.todo:
        # Check which tiles still need to be done.
        need = []
        for band in bands:
            fns = []
            for i in range(len(T)):
                tag = 'coadd-%s-w%i' % (T.coadd_id[i], band)
                prefix = os.path.join(opt.outdir, tag)
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
        if len(need):
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
        else:
            print 'Done (party now)'
        sys.exit(0)


    if not opt.plots:
        ps = None
            
    if len(args):
        A = []
        for a in args:
            tileid = int(a)
            band = tileid / 1000
            tileid = tileid % 1000
            print 'Doing coadd tile', T.coadd_id[tileid], 'band', band
            one_coadd(T[tileid], band, WISE, ps, opt.wishlist, opt.outdir, mp,
                      opt.cube)
            #A.append((T[tileid], band, WISE, ps, opt.wishlist, opt.outdir, mp))
        #mp.map(_bounce_one_coadd, A)
        sys.exit(0)

    if arr is None:
        print 'No tile(s) specified'
        parser.print_help()
        sys.exit(0)

    band = arr / 1000
    assert(band in bands)
    tile = arr % 1000
    assert(tile < len(T))

    print 'Doing coadd tile', T.coadd_id[tile], 'band', band

    t0 = Time()
    one_coadd(T[tile], band, WISE, ps, False, opt.outdir, mp, opt.cube)
    print 'Tile', T.coadd_id[tile], 'band', band, 'took:', Time()-t0


if __name__ == '__main__':
    main()
    
