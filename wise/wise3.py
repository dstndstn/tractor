if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import logging
import tempfile
import tractor
import pyfits
import pylab as plt
import numpy as np
import sys
from glob import glob
from scipy.ndimage.measurements import label,find_objects
from collections import Counter

from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.libkd.spherematch import match_radec, cluster_radec
from astrometry.util.util import *
from astrometry.blind.plotstuff import *
from astrometry.util.resample import *
from astrometry.util.multiproc import *
from astrometry.util.stages import *
from astrometry.util.ttime import *

from tractor import *
from tractor.sdss import *
from tractor.galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

import wise

def get_l1b_file(basedir, scanid, frame, band):
    scangrp = scanid[-2:]
    return os.path.join(basedir, scangrp, scanid, '%03i' % frame, 
                        '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))


# Find WISE images in range
# Read WISE sources in range
def stage100(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             indexfn=None,
             **kwa):
    bandnum = opt.bandnum
    band = 'w%i' % bandnum
    wisedatadirs = opt.wisedatadirs

    print 'RA,Dec range', ralo, rahi, declo, dechi

    roipoly = np.array([(ralo,declo),(ralo,dechi),(rahi,dechi),(rahi,declo)])

    TT = []
    for d,tag in wisedatadirs:
        if indexfn is None:
            ifn = os.path.join(d, 'WISE-index-L1b.fits')
        else:
            ifn = indexfn
        T = fits_table(ifn, columns=['ra','dec','scan_id','frame_num', 'band'])
        print 'Read', len(T), 'from WISE index', ifn

        # Add a margin around the CRVAL so we catch all fields that touch the RA,Dec box.
        # Magic numbers 1016 * 2.75 = image size * pixel scale of W1 = FOV
        margin = (1016. * 2.75 * np.sqrt(2.) / 3600.) / 2.
        cosdec = np.cos(np.deg2rad((declo + dechi) / 2.))
        print 'Margin:', margin, 'degrees'

        r0 = ralo - margin/cosdec
        r1 = rahi + margin/cosdec
        d0 = declo - margin
        d1 = dechi + margin

        I = np.flatnonzero(T.band == bandnum)
        print len(I), 'band', band
        T.cut(I)
        I = np.flatnonzero((T.ra > r0) * (T.ra < r1) * (T.dec > d0) * (T.dec < d1))
        print len(I), 'overlap RA,Dec box'
        T.cut(I)
        T.tag = [tag] * len(T)

        assert(len(np.unique([s + '%03i' % f for s,f in zip(T.scan_id, T.frame_num)])) == len(T))


        Igood = []
        for i,(sid,fnum) in enumerate(zip(T.scan_id, T.frame_num)):
            # HACK -- uncertainty image faulty
            #if ((sid == '11301b' and fnum == 57) or
            #    (sid == '11304a' and fnum == 30)):
            #    print 'WARNING: skipping bad data:', sid, fnum
            #    continue
            Igood.append(i)
        if len(Igood) != len(T):
            T.cut(np.array(Igood))

        fns = []
        for sid,fnum in zip(T.scan_id, T.frame_num):
            print 'scan,frame', sid, fnum

            fn = get_l1b_file(d, sid, fnum, bandnum)
            print '-->', fn
            assert(os.path.exists(fn))
            fns.append(fn)
        T.filename = np.array(fns)
        TT.append(T)
    T = merge_tables(TT)

    wcses = []
    corners = []
    ii = []
    for i in range(len(T)):
        wcs = Sip(T.filename[i], 0)
        W,H = wcs.get_width(), wcs.get_height()
        rd = []
        for x,y in [(1,1),(1,H),(W,H),(W,1)]:
            rd.append(wcs.pixelxy2radec(x,y))
        rd = np.array(rd)
        if polygons_intersect(roipoly, rd):
            wcses.append(wcs)
            corners.append(rd)
            ii.append(i)

    print 'Found', len(wcses), 'overlapping'
    I = np.array(ii)
    T.cut(I)

    outlines = corners
    corners = np.vstack(corners)

    if ps:
        r0,r1 = corners[:,0].min(), corners[:,0].max()
        d0,d1 = corners[:,1].min(), corners[:,1].max()
        print 'RA,Dec extent', r0,r1, d0,d1

        plot = Plotstuff(outformat='png', ra=(r0+r1)/2., dec=(d0+d1)/2., width=d1-d0, size=(800,800))
        out = plot.outline
        plot.color = 'white'
        plot.alpha = 0.07
        plot.apply_settings()
        for wcs in wcses:
            out.wcs = anwcs_new_sip(wcs)
            out.fill = False
            plot.plot('outline')
            out.fill = True
            plot.plot('outline')
        plot.color = 'gray'
        plot.alpha = 1.0
        plot.lw = 1
        plot.plot_grid(1, 1, 1, 1)
        pfn = ps.getnext()
        plot.write(pfn)
        print 'Wrote', pfn


    # Read WISE sources in the ROI
    if opt.wsources is not None:
        print 'Looking for WISE sources in', opt.wsources
    if opt.wsources is not None and os.path.exists(opt.wsources):
        W = fits_table(opt.wsources)
        W.cut((W.ra > ralo) * (W.ra < rahi) * (W.dec > declo) * (W.dec < dechi))
    else:
        from wisecat import wise_catalog_radecbox
        cols=['cntr', 'ra', 'dec', 'sigra', 'sigdec', 'cc_flags',
              'ext_flg', 'var_flg', 'moon_lev', 'ph_qual',
              'w1mpro', 'w1sigmpro', 'w1sat', 'w1nm', 'w1m', 
              'w1snr', 'w1cov', 'w1mag', 'w1sigm', 'w1flg',
              'w2mpro', 'w2sigmpro', 'w2sat', 'w2nm', 'w2m',
              'w2snr', 'w2cov', 'w2mag', 'w2sigm', 'w2flg',
              'w3mpro', 'w3sigmpro', 'w3sat', 'w3nm', 'w3m', 
              'w3snr', 'w3cov', 'w3mag', 'w3sigm', 'w3flg',
              'w4mpro', 'w4sigmpro', 'w4sat', 'w4nm', 'w4m',
              'w4snr', 'w4cov', 'w4mag', 'w4sigm', 'w4flg', ]
        W = wise_catalog_radecbox(ralo, rahi, declo, dechi, cols=cols)
        if opt.wsources is not None:
            W.writeto(opt.wsources)
            print 'Wrote', opt.wsources
        
    return dict(opt100=opt, rd=(ralo,rahi,declo,dechi), T=T, outlines=outlines,
                wcses=wcses, bandnum=bandnum, band=band, W=W)

# Read WISE images
# Apply ROI mask
def stage101(opt=None, ps=None, T=None, outlines=None, wcses=None, rd=None,
             band=None, bandnum=None,
             **kwa):
    r0,r1,d0,d1 = rd

    xyrois = []
    subwcses = []
    tims = []

    # Margin 1: grab WISE images that extend outside the RA,Dec box.

    margin1 = 10./3600.

    cosdec = np.cos(np.deg2rad((d0+d1)/2.))
    
    rm0 = r0 - margin1/cosdec
    rm1 = r1 + margin1/cosdec
    dm0 = d0 - margin1
    dm1 = d1 + margin1

    ninroi = []
    # Find the pixel ROI in each image containing the RA,Dec ROI.
    for i,(Ti,wcs) in enumerate(zip(T,wcses)):
        xy = []
        for r,d in [(rm0,dm0),(rm0,dm1),(rm1,dm1),(rm1,dm0)]:
            ok,x,y = wcs.radec2pixelxy(r,d)
            xy.append((x,y))
        xy = np.array(xy)
        x0,y0 = xy.min(axis=0)
        x1,y1 = xy.max(axis=0)
        W,H = int(wcs.get_width()), int(wcs.get_height())
        x0 = np.clip(int(np.floor(x0)), 0, W-1)
        y0 = np.clip(int(np.floor(y0)), 0, H-1)
        x1 = np.clip(int(np.ceil (x1)), 0, W-1)
        y1 = np.clip(int(np.ceil (y1)), 0, H-1)
        assert(x0 <= x1)
        assert(y0 <= y1)
        x1 += 1
        y1 += 1
        
        xyrois.append([x0,x1,y0,y1])

        tim = wise.read_wise_level1b(Ti.filename.replace('-int-1b.fits',''),
                                     nanomaggies=True, mask_gz=True, unc_gz=True,
                                     sipwcs=True, constantInvvar=True,
                                     roi=[x0,x1,y0,y1],
                                     zrsigs = [-2, 5])
        print 'Read', tim

        # Mask pixels outside the RA,Dec ROI
        if x0 > 0 or y0 > 0 or x1 < W-1 or y1 < H-1:
            print 'Image was clipped -- masking pixels outside ROI'
            h,w = tim.shape
            print 'Clipped size:', w,'x',h
            wcs = tim.getWcs()
            x0,y0 = wcs.getX0Y0()
            XX,YY = np.meshgrid(np.arange(x0, x0+w), np.arange(y0, y0+h))
            # approximate, but *way* faster than doing full WCS per pixel!
            J = point_in_poly(XX.ravel(), YY.ravel(), xy)
            K = J.reshape(XX.shape)
            iv = tim.getInvvar()
            tim.setInvvar(iv * K)
            tim.rdmask = K
            ninroi.append(np.sum(J))
        else:
            h,w = tim.shape
            ninroi.append(w*h)

        tims.append(tim)

    T.extents = np.array([tim.extent for tim in tims])
    T.pixinroi = np.array(ninroi)

    return dict(opt101=opt, tims=tims, margin1=margin1)

# Read SDSS (or other given) sources in range
# Create Tractor sources
def stage102(opt=None, ps=None, T=None, outlines=None, wcses=None, rd=None,
             tims=None, band=None, margin1=None,
             **kwa):
    r0,r1,d0,d1 = rd

    # Read SDSS sources in range.

    S = fits_table(opt.sources, columns=['ra','dec'])
    print 'Read', len(S), 'sources from', opt.sources

    margin2 = margin1 * 2.
    cosdec = np.cos(np.deg2rad((d0+d1)/2.))
    mr = margin2 / cosdec
    md = margin2
    I = np.flatnonzero((S.ra  > (r0-mr)) * (S.ra  < (r1+mr)) *
                       (S.dec > (d0-md)) * (S.dec < (d1+md)))
    print 'Reading', len(I), 'in range'

    S = fits_table(opt.sources, rows=I,
                   column_map=dict(r_dev='theta_dev',
                                   r_exp='theta_exp',
                                   fracpsf='fracdev'))
    S.row = I
    S.inblock = ((S.ra  >= r0) * (S.ra  < r1) *
                 (S.dec >= d0) * (S.dec < d1))


    if opt.nonsdss:
        cat = []
        for r,d in zip(S.ra, S.dec):
            cat.append(PointSource(RaDecPos(r,d), NanoMaggies(**{ band: 1. })))

    else:
        S.cmodelflux = S.modelflux
        sband = 'r'
        # NOTE, this method CUTS the "S" arg
        cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                      objs=S, bands=[], nanomaggies=True,
                                      extrabands=[band],
                                      fixedComposites=True,
                                      forcePointSources=opt.ptsrc,
                                      useObjcType=True)
    print 'Created', len(cat), 'Tractor sources'
    assert(len(cat) == len(S))

    tractor = Tractor(tims, cat)
    cat = tractor.catalog

    ## Give each source a minimum brightness
    minbright = 250.
    cat.freezeParamsRecursive('*')
    cat.thawPathsTo(band)
    p0 = cat.getParams()
    cat.setParams(np.maximum(minbright, p0))

    return dict(opt102=opt, tractor=tractor, S=S, margin2=margin2)

# Load PSF model
def stage103(opt=None, ps=None, tractor=None, band=None, bandnum=None,
             **kwa):
    tims = tractor.images

    # Load the spatially-varying PSF model
    from wise_psf import WisePSF
    psf = WisePSF(bandnum, savedfn='w%ipsffit.fits' % bandnum)
    # disabled
    assert(not opt.pixpsf)

    # Instantiate a (non-varying) mixture-of-Gaussians PSF at the
    # middle of each image's patch
    for tim in tims:
        x0,y0 = tim.getWcs().getX0Y0()
        h,w = tim.shape
        x,y = x0+w/2, y0+h/2
        tim.psf = psf.mogAt(x, y)

    # Mask inf pixels
    for tim in tims:
        # print 'Checking', tim
        I = np.flatnonzero(np.logical_not(np.isfinite(tim.getImage())))
        if len(I):
            print 'Found', len(I), 'inf pixels in', tim.name
            tim.getImage().flat[I] = 0
            iv = tim.getInvvar()
            iv.flat[I] = 0.
            tim.setInvvar(iv)
        assert(np.all(np.isfinite(tim.getInvvar())))
        
    return dict(opt103=opt)

# Create coadd, mask discrepant pixels
def stage104(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             R=None, imstats=None, T=None, S=None, bandnum=None, band=None,
             tractor=None, ims1=None,
             mp=None,
             **kwa):
    '''
    Build a co-add, find discrepant pixels in the individual exposures
    and mask them, then redo the coadd.
    '''
    # Create WCS into which we will coadd
    pixscale = 2.75 / 3600.
    # W4 is binned-down 2x2
    if bandnum == 4:
        pixscale *= 2

    ra  = (ralo  + rahi)  / 2.
    dec = (declo + dechi) / 2.
    W,H = (rahi - ralo) * np.cos(np.deg2rad(dec)) / pixscale, (dechi - declo) / pixscale
    W,H = int(np.ceil(W)), int(np.ceil(H))
    cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2., pixscale, 0., 0., pixscale, W,H)
    print 'Target WCS:', cowcs

    tims = tractor.getImages()

    # Resample
    ims = mp.map(_resample_one, [(tim, None, cowcs, True) for tim in tims])

    # Coadd
    lancsum  = np.zeros((H,W))
    lancsum2 = np.zeros((H,W))
    wsum     = np.zeros((H,W))
    for i,d in enumerate(ims):
        if d is None:
            print 'No overlap:', tims[i]
            print 'image shape:', tims[i].shape
            print '# valid pix:', np.sum(tims[i].invvar > 0)
            continue
        lancsum  += (d.rimg    * d.ww)
        lancsum2 += (d.rimg**2 * d.ww)
        wsum     += d.ww

        # plt.clf()
        # plt.subplot(1,2,1)
        # plt.imshow(lancsum, interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.imshow(wsum, interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.suptitle('coadd step %i' % i)
        # ps.savefig()
        # print 'median ww:', np.median(d.ww)
        # print 'sig1:', d.sig1

    # For W4, single-image ww is ~ 1e-10
    tinyw = 1e-16
    coimg   = (lancsum / np.maximum(wsum, tinyw))
    coinvvar = wsum
    coimg1 = coimg
    
    sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
    print 'Coadd sig:', sig
    # Per-pixel std
    coppstd = np.sqrt(lancsum2 / (np.maximum(wsum, tinyw)) - coimg**2)
    coppstd1 = coppstd
    sig1 = sig
    
    # Using the difference between the coadd and the resampled
    # individual images ("rchi"), mask additional pixels and redo the
    # coadd.
    nnsum    = np.zeros((H,W))
    lancsum [:,:] = 0
    lancsum2[:,:] = 0
    wsum    [:,:] = 0
    for d in ims:
        if d is None:
            continue
        rchi = (d.rimg - coimg) * d.mask / np.maximum(coppstd, 1e-6)
        badpix = (np.abs(rchi) >= 5.)
        # grow by a small margin
        badpix = binary_dilation(badpix)
        notbad = np.logical_not(badpix)
        d.rchi = rchi
        d.mask *= notbad
        w = (1. / d.sig1**2)
        ww = w * d.mask
        # update d.ww?
        nnsum    += (d.nnimg   * ww)
        lancsum  += (d.rimg    * ww)
        lancsum2 += (d.rimg**2 * ww)
        wsum     += ww
    conn  = (nnsum   / np.maximum(wsum, tinyw))
    coimg = (lancsum / np.maximum(wsum, tinyw))
    coinvvar = wsum

    print 'Second-round coadd:'
    sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
    print 'Coadd sig:', sig
    # per-pixel variance
    coppstd = np.sqrt(lancsum2 / (np.maximum(wsum, tinyw)) - coimg**2)

    # 2. Apply rchi masks to individual images
    print 'Applying rchi masks to images...'
    tims = tractor.getImages()

    args = []
    for i,(tim, d) in enumerate(zip(tims, ims)):
        if d is None:
            args.append((tim, None, cowcs))
            continue
        args.append((tim, d.mask, cowcs))
    rmasks = mp.map(_rev_resample_mask, args)

    for i,(mask,tim) in enumerate(zip(rmasks, tims)):
        if mask is None:
            tim.coaddmask = None
            continue
        tim.coaddmask = mask
        tim.orig_invvar = tim.invvar
        if mask is not None:
            tim.setInvvar(tim.invvar * (mask > 0))
        else:
            tim.setInvvar(tim.invvar)

    if ps:
        # Mosaic of all individual exposures
        nims = len(tims)
        cols = int(np.ceil(np.sqrt(nims)))
        rows = int(np.ceil(nims / float(cols)))
        plt.clf()
        for i,(tim,d) in enumerate(zip(tims, ims)):
            if d is None:
                continue
            plt.subplot(rows, cols, i+1)
            ima = dict(interpolation='nearest', origin='lower',
                        vmin=-2.*d.sig1, vmax=5.*d.sig1, cmap='gray')
            plt.imshow(d.rimg, **ima)
            plt.xticks([]); plt.yticks([])
        plt.suptitle('Individual exposures: %s' % band)
        ps.savefig()

        # wmax = np.max([d.ww.max() for d in ims])
        # plt.clf()
        # for i,(tim,d) in enumerate(zip(tims, ims)):
        #     plt.subplot(rows, cols, i+1)
        #     ima = dict(interpolation='nearest', origin='lower',
        #                vmin=0, vmax=wmax, cmap='gray')
        #     plt.imshow(d.ww, **ima)
        #     plt.xticks([]); plt.yticks([])
        # plt.suptitle('Individual weights: %s' % band)
        # ps.savefig()
        # 
        # # (non-resampled)
        # plt.clf()
        # for i,(tim,d) in enumerate(zip(tims, ims)):
        #     plt.subplot(rows, cols, i+1)
        #     ima = dict(interpolation='nearest', origin='lower',
        #                vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
        #     plt.imshow(tim.getImage(), **ima)
        #     plt.xticks([]); plt.yticks([])
        # plt.suptitle('Individual exposures (not resampled): %s' % band)
        # ps.savefig()
        # 
        # # invvars
        # plt.clf()
        # ivmax = np.max([tim.orig_invvar.max() for tim in tims])
        # print 'ivmax:', ivmax
        # for i,(tim,d) in enumerate(zip(tims, ims)):
        #     plt.subplot(rows, cols, i+1)
        #     ima = dict(interpolation='nearest', origin='lower',
        #                vmin=0, vmax=ivmax, cmap='gray')
        #     #plt.imshow(tim.getInvvar(), **ima)
        #     plt.imshow(tim.orig_invvar, **ima)
        #     plt.xticks([]); plt.yticks([])
        # plt.suptitle('Individual invvars: %s' % band)
        # ps.savefig()

        # mask bits
        # for bit in range(32):
        #     val = (1 << bit)
        #     anyset = np.any([np.any(tim.maskplane & val)for tim in tims])
        #     if not anyset:
        #         print 'Mask bit', bit, ': none set'
        #         continue
        #     plt.clf()
        #     for i,(tim,d) in enumerate(zip(tims, ims)):
        #         plt.subplot(rows, cols, i+1)
        #         ima = dict(interpolation='nearest', origin='lower',
        #                    vmin=0, vmax=1, cmap='gray')
        #         plt.imshow(tim.maskplane & val, **ima)
        #         plt.xticks([]); plt.yticks([])
        #     plt.suptitle('Individual mask bit %i: %s' % (bit, band))
        #     ps.savefig()

        # for tim in tims:
        #     print 'sigma1:', tim.sigma1
        #     print 'goodmask pixels:', len(np.flatnonzero(tim.goodmask))
        #     print 'invvar > 0 pixels:', len(np.flatnonzero(tim.cinvvar))
        #     print 'invvar > 0 pixels:', len(np.flatnonzero(tim.vinvvar))
        #     print 'invvar > 0 pixels:', len(np.flatnonzero(tim.orig_invvar))
        #     print 'invvar > 0 pixels:', len(np.flatnonzero(tim.getInvvar()))

        # First-round coadd
        plt.clf()
        plt.imshow(coimg1, interpolation='nearest', origin='lower',
                   vmin=-2.*sig1, vmax=5.*sig1, cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.title('Initial Coadd: %s' % band)
        ps.savefig()

        # # Mosaic of rchi images
        # plt.clf()
        # for i,(tim,d) in enumerate(zip(tims, ims)):
        #     plt.subplot(rows, cols, i+1)
        #     ima = dict(interpolation='nearest', origin='lower',
        #                vmin=-5, vmax=5, cmap='gray')
        #     plt.imshow(d.rchi, **ima)
        #     plt.xticks([]); plt.yticks([])
        # plt.suptitle('rchi: %s' % band)
        # ps.savefig()

        # Second-round Coadd
        plt.clf()
        plt.imshow(coimg, interpolation='nearest', origin='lower',
                   vmin=-2.*sig, vmax=5.*sig, cmap='gray')
        plt.xticks([]); plt.yticks([])
        plt.title('Coadd: %s' % band)
        ps.savefig()



    return dict(coimg=coimg, coinvvar=coinvvar, coppstd=coppstd,
                conn=conn,
                cowcs=cowcs, opt104=opt,
                coimg1=coimg1, coppstd1=coppstd1,
                resampled=ims)

# Add WISE objects with no "SDSS" counterpart
def stage105(opt=None, ps=None, tractor=None, band=None, bandnum=None, T=None,
             S=None, ri=None, di=None, W=None,
             **kwa):
    tims = tractor.images
    cat = tractor.getCatalog()
    cat1 = cat.copy()
    sdss = S

    # Find WISE objs with no SDSS counterpart
    if len(W):
        I,J,d = match_radec(W.ra, W.dec, sdss.ra, sdss.dec, 4./3600.)
        unmatched = np.ones(len(W), bool)
        unmatched[I] = False
        UW = W[unmatched]
        # 1. Create tractor PointSource objects for each WISE-only object
        wcat = []
        for i in range(len(UW)):
            mag = UW.get('w%impro' % bandnum)[i]
            nm = NanoMaggies.magToNanomaggies(mag)
            src = PointSource(RaDecPos(UW.ra[i], UW.dec[i]),
                              NanoMaggies(**{band: nm}))
            wcat.append(src)
        srcs = [src for src in cat] + wcat
        tractor.setCatalog(Catalog(*srcs))
    else:
        UW = W.copy()

    return dict(cat1=cat1, tractor=tractor, UW=UW)


# Run forced photometry (simultaneously on all exposures)
def stage106(opt=None, ps=None, tractor=None, band=None, bandnum=None, T=None,
             S=None, UW=None, ceres=True, **kwa):
    tims = tractor.images
    minFlux = opt.minflux
    if minFlux is not None:
        minFlux = np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val
                             for tim in tims])
        print 'minFlux:', minFlux, 'nmgy'

    goodtims = []
    for tim in tims:
        assert(np.all(np.isfinite(tim.getImage())))
        assert(np.all(np.isfinite(tim.getInvvar())))
        #print 'tim', tim.name, 'invvar', tim.getInvvar().min(), tim.getInvvar().max()
        if tim.getInvvar().max() == 0:
            print 'Dropping tim', tim.name, ': no non-zero invvar pixels'
            continue

        if not np.isfinite(tim.getSky().getValue()):
            print 'Infinite sky value in', tim.name
            tim.getSky().setValue(0.)

        goodtims.append(tim)
    tractor.setImages(Images(*goodtims))
    tims = tractor.getImages()

    t0 = Time()
    tractor.freezeParamsRecursive('*')
    tractor.thawPathsTo('sky')
    tractor.thawPathsTo(band)
    try:
        #ims0,ims1,IV,fs
        kwa = {}
        if ceres:
            cblock = 10
            kwa.update(use_ceres=True, BW=cblock, BH=cblock)
            
        R = tractor.optimize_forced_photometry(
            minsb=opt.minsb, mindlnp=1., sky=True, minFlux=minFlux,
            fitstats=True, variance=True,
            shared_params=False, **kwa)
        ims0 = R.ims0
        ims1 = R.ims1
        IV = R.IV
        fs = R.fitstats
    except:
        import traceback
        traceback.print_exc()

        cat = tractor.getCatalog()
        for tim in tims:
            plt.clf()
            plt.imshow(tim.getImage(), interpolation='nearest', origin='lower', cmap='gray')
            plt.title('Image ' + tim.name)
            ps.savefig()

            plt.clf()
            plt.imshow(tim.getInvvar(), interpolation='nearest', origin='lower', cmap='gray')
            plt.colorbar()
            plt.title('Invvar ' + tim.name)
            x,y = [],[]
            wcs = tim.getWcs()
            for src in cat:
                xx,yy = wcs.positionToPixel(src.getPosition())
                x.append(xx)
                y.append(yy)
            ax = plt.axis()
            plt.plot(x, y, 'ro')
            plt.axis(ax)
            ps.savefig()
        raise

    print 'Forced phot took', Time()-t0

    cat = tractor.catalog
    assert(len(cat) == len(S) + len(UW))
    assert(len(cat) == len(IV))

    # The parameters are stored in the order: sky, then fluxes
    #print 'Variance vector:', len(V)
    #print '# images:', len(tims)
    #print '# sources:', len(cat)

    cat2 = cat.copy()

    R = tabledata()
    R.ra  = np.array([src.getPosition().ra  for src in cat])
    R.dec = np.array([src.getPosition().dec for src in cat])
    R.sdss = np.array([1] * len(S) + [0] * len(UW)).astype(np.uint8)
    R.set(band, np.array([src.getBrightness().getBand(band) for src in cat]))
    R.set(band + '_ivar', IV)
    R.row = np.hstack((S.row, np.array([-1] * len(UW))))
    R.inblock = np.hstack((S.inblock, np.array([1] * len(UW)))).astype(np.uint8)

    imstats = tabledata()
    if fs is not None:
        for k in ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
            R.set(k, getattr(fs, k))
        for k in ['imchisq', 'imnpix', 'sky']:
            X = getattr(fs, k)
            imstats.set(k, X)
        imstats.scan_id = T.scan_id
        imstats.frame_num = T.frame_num

    return dict(R=R, imstats=imstats, ims0=ims0, ims1=ims1, cat2=cat2)


class Duck(object):
    pass

def get_sip_subwcs(wcs, extent):
    # Create sub-WCS
    (x0, x1, y0, y1) = extent
    return wcs.get_subimage(int(x0), int(y0), int(x1-x0), int(y1-y0))

def _resample_one((tim, mod, targetwcs, spline)):
    print 'Resampling', tim.name
    wcs2 = get_sip_subwcs(tim.getWcs().wcs, tim.extent)
    try:
        yo,xo,yi,xi,nil = resample_with_wcs(targetwcs, wcs2, [],[], spline=spline)
    except OverlapError:
        return None
    W,H = targetwcs.get_width(), targetwcs.get_height()
    nnim = np.zeros((H,W))
    # print 'extent', tim.extent
    # print 'tim shape', tim.shape, tim.data.shape
    # print 'wcs2:', wcs2
    # print 'xi', xi.min(), xi.max()
    # print 'yi', yi.min(), yi.max()
    # print 'target W,H', W,H
    # print 'xo', xo.min(), xo.max()
    # print 'yo', yo.min(), yo.max()
    nnim[yo,xo] = tim.data[yi,xi]
    iv = np.zeros((H,W))
    iv[yo,xo] = tim.invvar[yi,xi]
    # Patch masked pixels so we can interpolate
    patchimg = tim.data.copy()
    ok = patch_image(patchimg, tim.invvar > 0,
                     required=tim.rdmask)
    if not ok:
        print 'WARNING: patching failed.  Image size', patchimg.shape
        print 'Wanted to patch', np.count_nonzero(tim.rdmask), 'pixels'
        return None

    # Resample
    Lorder = 3
    inims = [patchimg]
    if mod is not None:
        inims.append(mod)
    try:
        yo,xo,yi,xi,rpix = resample_with_wcs(targetwcs, wcs2, inims, Lorder, spline=spline)
    except OverlapError:
        return None
    rpatch = np.zeros((H,W))
    rpatch[yo,xo] = rpix[0]

    sig1 = tim.sigma1
    sky = tim.getSky().getValue()
    # photocal.getScale() takes nanomaggies to image counts; we want to convert
    # images to nanomaggies (per pix)
    scale = 1. / tim.getPhotoCal().getScale()
    sig1 = sig1 * scale

    rmod = None
    if mod is not None:
        rmod = np.zeros((H,W))
        rmod[yo,xo] = rpix[1]
        rmod = (rmod   - sky) * scale
    #print 'scale', scale, 'scaled sig1:', sig1
    w = (1. / sig1**2)
    ww = w * (iv > 0)

    d = Duck()
    d.nnimg = (nnim   - sky) * scale
    d.rimg  = (rpatch - sky) * scale
    d.rmod  =  rmod
    d.ww = ww
    d.mask = (iv > 0)
    d.sig1 = sig1
    d.name = tim.name
    d.mod = mod
    d.img = tim.data
    d.invvar = tim.invvar
    d.sky = sky
    d.scale = scale
    d.npix1 = np.sum(tim.getInvError() > 0)
    if rmod is not None:
        d.lnp1 = np.sum(((mod - tim.getImage()) * tim.getInvError())**2)
        d.lnp2 = np.sum(((rmod - rpatch)**2 * iv))
    else:
        d.lnp1 = 0.
        d.lnp2 = 0.
    d.npix2 = np.sum(iv > 0)
    return d

def _resample_mod((tim, mod, targetwcs, spline)):
    W,H = targetwcs.get_width(), targetwcs.get_height()
    wcs2 = get_sip_subwcs(tim.getWcs().wcs, tim.extent)
    Lorder = 3
    try:
        yo,xo,yi,xi,rpix = resample_with_wcs(targetwcs, wcs2, [mod], Lorder, spline=spline)
    except OverlapError:
        return None
    rmod = np.zeros((H,W))
    rmod[yo,xo] = rpix[0]
    iv = np.zeros((H,W))
    iv[yo,xo] = tim.invvar[yi,xi]
    sig1 = tim.sigma1
    w = (1. / sig1**2)
    ww = w * (iv > 0)
    return rmod,ww

def _rev_resample_mask((tim, mask, targetwcs)):
    if mask is None:
        return None
    wcs2 = get_sip_subwcs(tim.getWcs().wcs, tim.extent)
    try:
        yo,xo,yi,xi,nil = resample_with_wcs(wcs2, targetwcs, [],[])
    except OverlapError:
        return None
    w,h = int(wcs2.get_width()), int(wcs2.get_height())
    rmask = np.zeros((h,w))
    rmask[yo,xo] = mask[yi, xi]
    return rmask




def stage107(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             R=None, imstats=None, T=None, S=None, bandnum=None, band=None,
             tractor=None, ims1=None,
             mp=None,
             coimg=None, coinvvar=None, coppstd=None, cowcs=None,
             #comod=None,
             #ims2=None,
             **kwa):
    r0,r1,d0,d1 = ralo,rahi,declo,dechi
    sdss = S
    cat = tractor.getCatalog()
    W,H = cowcs.get_width(), cowcs.get_height()
    tims = tractor.getImages()
    #cochi = (coimg - comod) * np.sqrt(coinvvar)

    args = []
    #for i,(tim, (nil,mod,ie,chi,roi), (nil,mod0,nil,nil,nil)) in enumerate(zip(tims, ims2, ims1)):
    for i,(tim, (nil,mod,ie,chi,roi)) in enumerate(zip(tims, ims1)):

        sky = tim.getSky().getValue()
        scale = 1. / tim.getPhotoCal().getScale()
        modx  = (mod - sky) * scale

        args.append((tim, modx, cowcs, True))
        
        if i < 10 and ps is not None:

            ima = dict(interpolation='nearest', origin='lower',
                       vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
            #vmin=tim.sky - tim.sigma1 * 2,
            #vmax=tim.sky + tim.sigma1 * 5)

            plt.clf()
            plt.subplot(2,2,1)
            plt.imshow(tim.data, **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('data')
            plt.colorbar()
            plt.subplot(2,2,2)
            plt.imshow(tim.invvar, interpolation='nearest', origin='lower',
                       cmap='gray', vmin=0)
            plt.xticks([]); plt.yticks([])
            plt.title('invvar')
            plt.colorbar()
            #plt.subplot(2,2,3)
            #plt.imshow(mod0)
            #plt.colorbar()
            plt.subplot(2,2,4)
            plt.imshow(mod, **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('model')
            plt.colorbar()
            plt.suptitle(tim.name)
            ps.savefig()
    
    mims = mp.map(_resample_mod, args)

    # redo = []
    # for i,(mim,(t,m,c,nil)) in enumerate(zip(mims,args)):
    #     if mim is None:
    #         redo.append((t,m,c,False))
    # if len(redo):
    #     mims2 = mp.map(_resample_mod, redo)
    #     j = 0
    #     for i,mim in enumerate(mims):
    #         if mim is None:
    #             mims[i] = mims2[j]
    #             j += 1

    modsum2 = np.zeros((H,W))
    wsum2 = np.zeros((H,W))
    for mim in mims:
        if mim is None:
            continue
        rmod,ww = mim
        modsum2 += (rmod * ww)
        wsum2   += ww
    comod2 = modsum2 / np.maximum(wsum2, 1e-12)
    cochi2 = (coimg - comod2) * np.sqrt(coinvvar)

    if ps is not None:
        sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=-2*sig, vmax=10*sig)

        plt.clf()
        plt.imshow(comod2, **ima)
        plt.title('Coadded model')
        plt.colorbar()
        ps.savefig()
        
        # plt.clf()
        # plt.imshow(cochi, interpolation='nearest', origin='lower',
        #            vmin=-10., vmax=10., cmap='gray')
        # plt.title('Chi (before)')
        # plt.colorbar()
        # ps.savefig()

        plt.clf()
        plt.imshow(cochi2, interpolation='nearest', origin='lower',
                   vmin=-10., vmax=10., cmap='gray')
        plt.title('Coadded chi')
        plt.colorbar()
        ps.savefig()

        # plt.clf()
        # plt.imshow(comod, **ima)
        # plt.title('Model (before)')
        # plt.colorbar()
        # ps.savefig()
    

    return dict(comod2=comod2)

def stage108(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             R=None, imstats=None, T=None, S=None, bandnum=None, band=None,
             tractor=None, ims1=None,
             mp=None,
             coimg=None, coinvvar=None, comod=None, coppstd=None, cowcs=None,
             comod2=None,
             ims2=None, cat2=None,
             UW=None, ceres=None,
             **kwa):
    r0,r1,d0,d1 = ralo,rahi,declo,dechi
    sdss = S
    cat = tractor.getCatalog()
    S = cowcs.get_width()
    tims = tractor.getImages()
    #cochi = (coimg - comod) * np.sqrt(coinvvar)
    cochi2 = (coimg - comod2) * np.sqrt(coinvvar)

    # 4. Run forced photometry on coadd
    from wise_psf import WisePSF
    psf = WisePSF(bandnum, savedfn='w%ipsffit.fits' % bandnum)

    wcs = ConstantFitsWcs(cowcs)
    pcal = LinearPhotoCal(1., band=band)
    sky = ConstantSky(0.)
    # HACK
    psf = psf.mogAt(500., 500.)

    coimg = coimg.copy()
    coinvvar = coinvvar.copy()
    bad = np.flatnonzero(np.logical_not(np.isfinite(coimg)))
    print 'Zeroing out', len(bad), 'non-finite pixels in coadd'
    coimg.flat[bad] = 0.0
    coinvvar.flat[bad] = 0.0

    coim = Image(data=coimg, invvar=coinvvar, wcs=wcs, photocal=pcal, sky=sky,
                 psf=psf, name='coadd', domask=False)

    tr = Tractor([coim], cat)
    tr.freezeParamsRecursive('*')
    tr.thawPathsTo('sky')
    tr.thawPathsTo(band)

    minsb = 0.005
    minFlux = None
    
    try:
        t0 = Time()
        kwa = {}
        if ceres:
            cblock = 10
            kwa.update(use_ceres=True, BW=cblock, BH=cblock)

        R = tr.optimize_forced_photometry(
            minsb=minsb, mindlnp=1.,
            sky=True, minFlux=minFlux,
            fitstats=True, variance=True,
            shared_params=False, **kwa)
        ims0,ims1 = R.ims0, R.ims1
        IV,fs = R.IV, R.fitstats
        print 'Forced phot on coadd took', Time()-t0
    except:
        import traceback
        traceback.print_exc()

        plt.clf()
        plt.imshow(coimg, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('Coimg')
        ps.savefig()

        plt.clf()
        plt.imshow(coinvvar, interpolation='nearest', origin='lower', cmap='gray')
        plt.colorbar()
        plt.title('Coinvvar')
        ps.savefig()

        x,y = [],[]
        for src in cat:
            xx,yy = wcs.positionToPixel(src.getPosition())
            x.append(xx)
            y.append(yy)
        ax = plt.axis()
        plt.plot(x, y, 'ro')
        plt.axis(ax)
        ps.savefig()

        raise

    print 'Sky level in coadd fit:', coim.getSky()

    cat3 = cat.copy()

    if ps is not None:
        sig = 1./np.sqrt(np.median(coinvvar[coinvvar > 0]))
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=-2*sig, vmax=10*sig)
        imchi = dict(interpolation='nearest', origin='lower',
                     vmin=-5, vmax=5, cmap='gray')

        (im,mod0,ie,chi0,roi) = ims0[0]
        if ims1 is not None:
            (im,mod1,ie,chi1,roi) = ims1[0]

        plt.clf()
        plt.imshow(im, **ima)
        plt.title('coadd: data')
        ps.savefig()

        plt.clf()
        plt.imshow(mod0, **ima)
        plt.title('coadd: initial model')
        ps.savefig()

        if ims1 is not None:
            plt.clf()
            plt.imshow(mod1, **ima)
            plt.title('coadd: final model')
            ps.savefig()

        plt.clf()
        plt.imshow(chi0, **imchi)
        plt.title('coadd: initial chi')
        ps.savefig()

        if ims1 is not None:
            plt.clf()
            plt.imshow(chi1, **imchi)
            plt.title('coadd: final chi')
            ps.savefig()

    m2,m3 = [],[]
    for s2,s3 in zip(cat2, cat3):
        m2.append(NanoMaggies.nanomaggiesToMag(s2.getBrightness().getBand(band)))
        m3.append(NanoMaggies.nanomaggiesToMag(s3.getBrightness().getBand(band)))
    m2 = np.array(m2)
    m3 = np.array(m3)

    if ps is not None:
        plt.clf()
        plt.plot(m2, m3, 'b.', ms=8)
        if bandnum in [1,2]:
            lo,hi = 12,24
        else:
            lo,hi = 8,20
        plt.plot([lo,hi],[lo,hi], 'k-', lw=2, alpha=0.3)
        plt.axis([lo,hi,lo,hi])
        plt.xlabel('Individual images photometry (mag)')
        plt.ylabel('Coadd photometry (mag)')
        ps.savefig()

    print 'SDSS sources:', len(sdss)
    print 'UW sources:', len(UW)
    print 'cat2:', len(cat2)
    print 'cat3:', len(cat3)

    rd = np.array([(s.getPosition().ra, s.getPosition().dec) for s in cat2])
    ra,dec = rd[:,0], rd[:,1]
    I = ((ra >= r0) * (ra <= r1) * (dec >= d0) * (dec <= d1))
    inbounds = np.flatnonzero(I)
    J = np.arange(len(I)) < len(sdss)

    if ps is not None:
        plt.clf()
        p1 = plt.plot(m2[I*J], (m3-m2)[I*J], 'b.', ms=8)
        nJ = np.logical_not(J)
        p2 = plt.plot(m2[nJ], (m3-m2)[nJ], 'g.', ms=8)
        I = np.logical_not(I)
        p3 = plt.plot(m2[I*J], (m3-m2)[I*J], 'r.', ms=8)
        if bandnum in [1,2]:
            lo,hi = 12,24
        else:
            lo,hi = 8,20
        plt.plot([lo,hi],[0, 0], 'k-', lw=2, alpha=0.3)
        plt.axis([lo,hi,-2,2])
        plt.xlabel('Individual images photometry (mag)')
        plt.ylabel('Coadd photometry - Individual (mag)')
        plt.legend((p1,p2,p3),('In bounds', 'WISE-only', 'Out-of-bounds'))
        ps.savefig()

        # Show locations of largest changes
    
        plt.clf()
        plt.imshow(im, **ima)
        plt.xticks([]); plt.yticks([])
        plt.gray()
        plt.title('Coadd %s: data' % band)
        ps.savefig()
        ax = plt.axis()
        J = np.argsort(-np.abs((m3-m2)[inbounds]))
        for j in J[:5]:
            ii = inbounds[j]
            pos = cat2[ii].getPosition()
            x,y = coim.getWcs().positionToPixel(pos)
            plt.text(x, y, '%.1f/%.1f' % (m2[ii], m3[ii]), color='r')
            plt.plot(x, y, 'r+', ms=15, lw=1.5)
        plt.axis(ax)
        ps.savefig()

        if ims1 is not None:
            plt.clf()
            plt.imshow(mod1, **ima)
            plt.xticks([]); plt.yticks([])
            plt.gray()
            ax = plt.axis()
            J = np.argsort(-np.abs((m3-m2)[inbounds]))
            for j in J[:5]:
                ii = inbounds[j]
                pos = cat2[ii].getPosition()
                x,y = coim.getWcs().positionToPixel(pos)
                plt.text(x, y, '%.1f/%.1f' % (m2[ii], m3[ii]), color='r')
                plt.plot(x, y, 'r+', ms=15, lw=1.5)
            plt.axis(ax)
            plt.title('Coadd %s: model' % band)
            ps.savefig()

        plt.clf()
        plt.imshow(comod2, **ima)
        plt.xticks([]); plt.yticks([])
        plt.gray()
        plt.title('individual frames: model')
        ps.savefig()
    
        if ims1 is not None:
            plt.clf()
            plt.imshow(chi1, **imchi)
            plt.xticks([]); plt.yticks([])
            ax = plt.axis()
            J = np.argsort(-np.abs((m3-m2)[inbounds]))
            for j in J[:5]:
                ii = inbounds[j]
                pos = cat2[ii].getPosition()
                x,y = coim.getWcs().positionToPixel(pos)
                plt.text(x, y, '%.1f/%.1f' % (m2[ii], m3[ii]), color='r')
                plt.plot(x, y, 'r+', ms=15, lw=1.5)
            plt.axis(ax)
            plt.title('Coadd %s: chi' % band)
            ps.savefig()

    return dict(ims3=ims1, cotr=tr, cat3=cat3)



### Forced photometry on one epoch at a time
def stage109(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             R=None, imstats=None, T=None, S=None, bandnum=None, band=None,
             tractor=None, ims1=None,
             mp=None,
             coimg=None, coinvvar=None, comod=None, coppstd=None, cowcs=None,
             comod2=None,
             ims2=None, cat2=None,
             UW=None,
             **kwa):
    r0,r1,d0,d1 = ralo,rahi,declo,dechi

    cat = tractor.getCatalog()
    tims = tractor.getImages()

    cats4 = []
    for tim in tims:
        # Reset all brightnesses before running?
        cati = cat.copy()
        cati.freezeParamsRecursive('*')
        cati.thawPathsTo(band)
        cati.setParams(np.array([100.] * cati.numberOfParams()))

        tr = Tractor([tim], cati)
        tr.freezeParamsRecursive('*')
        tr.thawPathsTo('sky')
        tr.thawPathsTo(band)

        npix = sum(tim.getInvvar() > 0)
        nparams = tr.numberOfParams()
        print 'N pixels:', npix
        print 'N params:', nparams

        if npix < nparams:
            cats4.append(tr.getCatalog())
            continue

        minsb = 0.005
        minFlux = None
        t0 = Time()
        ims0,ims1,IV,fs = tr.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
                                                        sky=True, minFlux=minFlux,
                                                        fitstats=True,
                                                        variance=True)
        print 'Forced phot on coadd took', Time()-t0
        cats4.append(tr.getCatalog())

    if ps:
        m2,m4s = [],[]
        for s2 in cat2:
            m2.append(NanoMaggies.nanomaggiesToMag(s2.getBrightness().getBand(band)))
        for cat4 in cats4:
            m4 = []
            for s in cat4:
                m4.append(NanoMaggies.nanomaggiesToMag(s.getBrightness().getBand(band)))
            m4s.append(np.array(m4))
        m2 = np.array(m2)

        plt.clf()
        for m4 in m4s:
            plt.plot(m2, m4, 'b.', ms=8)
        if bandnum in [1,2]:
            lo,hi = 12,24
        else:
            lo,hi = 8,20
        plt.plot([lo,hi],[lo,hi], 'k-', lw=2, alpha=0.3)
        plt.axis([lo,hi,lo,hi])
        plt.xlabel('Simultaneous photometry (mag)')
        plt.ylabel('Image-at-a-time photometry (mag)')
        ps.savefig()

    return dict(cats4=cats4)


# Sampling with emcee
def stage110(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             R=None, imstats=None, T=None, S=None, bandnum=None, band=None,
             tractor=None, ims1=None,
             mp=None,
             coimg=None, coinvvar=None, comod=None, coppstd=None, cowcs=None,
             comod2=None,
             ims2=None, cat2=None,
             UW=None,
             **kwa):
    r0,r1,d0,d1 = ralo,rahi,declo,dechi

    cat = tractor.getCatalog()
    tims = tractor.getImages()

    import emcee

    tr = tractor
    tr.freezeParamsRecursive('*')
    tr.thawPathsTo('sky')
    tr.thawPathsTo(band)

    p0 = np.array(tr.getParams())
    ndim = len(p0)
    nw = 50
    print 'N params', ndim, 'N walkers', nw
    sampler = emcee.EnsembleSampler(nw, ndim, tr, threads=8)
    steps = np.array(tr.getStepSizes())
    colscales = tr.getParameterScales()
    pp0 = np.vstack([p0 + 1e-4 * steps / colscales *
                     np.random.normal(size=len(steps))
                     for i in range(nw)])
    alllnp = []
    allp = []
    lnp = None
    pp = pp0
    rstate = None
    for step in range(1001):
        print 'Taking emcee step', step
        pp,lnp,rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        #print 'lnprobs:', lnp
        alllnp.append(lnp.copy())
        allp.append(pp.copy())

    return dict(allp=allp, alllnp=alllnp)



def stage205(opt=None, ps=None, tractor=None, band=None, bandnum=None, T=None,
             S=None, ri=None, di=None,
             ims0=None, ims1=None,
             ttsuf='', pcat=[], addSky=False,
             **kwa):

    if ps is not None:
        ptims = tractor.images

        imas = [dict(interpolation='nearest', origin='lower',
                     vmin=tim.zr[0], vmax=tim.zr[1]) for tim in ptims]
        imchis = [dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)]*len(ptims)

        tt = 'Block ' + str(ri) + ', ' + str(di) + ttsuf

        if addSky:
            # in engine.py, we subtracted the sky when computing per-image
            for tim,(img,mod,ie,chi,roi) in zip(ptims, ims1):
                tim.getSky().addTo(mod)
    
        _plot_grid([img for (img, mod, ie, chi, roi) in ims0], imas)
        plt.suptitle('Data: ' + tt)
        ps.savefig()

        if ims1 is not None:
            _plot_grid2(ims1, pcat, ptims, imas)
            plt.suptitle('Forced-phot model: ' + tt)
            ps.savefig()

            _plot_grid2(ims1, pcat, ptims, imchis, ptype='chi')
            plt.suptitle('Forced-phot chi: ' + tt)
            ps.savefig()


def stage204(opt=None, ps=None, tractor=None, band=None, bandnum=None, **kwa):
    tims = tractor.images

    tractor.freezeParam('images')

    minFlux = opt.minflux
    if minFlux is not None:
        minFlux = np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val for tim in tims])
        print 'minFlux:', minFlux, 'nmgy'

    t0 = Time()
    ims0,ims1 = tractor.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
                                                   sky=False, minFlux=minFlux)
    print 'Forced phot took', Time()-t0


def stage304(opt=None, ps=None, tractor=None, band=None, bandnum=None, rd=None, **kwa):

    r0,r1,d0,d1 = rd
    cat = tractor.catalog

    ra  = np.array([src.getPosition().ra  for src in tractor.catalog])
    dec = np.array([src.getPosition().dec for src in tractor.catalog])

    # W = fits_table('/home/boss/products/NULL/wise/trunk/fits/wise-allsky-cat-part45-radec.fits')
    # print 'Read', len(W), 'WISE'
    # W.cut((W.ra > r0) * (W.ra < r1) * (W.dec > d0) * (W.dec < d1))
    # print 'Cut to', len(W), 'WISE'
    # I,J,d = match_radec(ra, dec, W.ra, W.dec, 4./3600.)
    # print len(I), 'matches to WISE sources'
    
    print 'Clustering with radius', opt.wrad, 'arcsec'
    Wrad = opt.wrad / 3600.
    groups,singles = cluster_radec(ra, dec, Wrad, singles=True)
    #print 'Source clusters:', groups
    #print 'Singletons:', singles
    print 'Source clusters:', len(groups)
    print 'Singletons:', len(singles)
    
    print 'Group size histogram:'
    ng = Counter()
    for g in groups:
        ng[len(g)] += 1
    kk = ng.keys()
    kk.sort()
    for k in kk:
        print '  ', k, 'sources:', ng[k], 'groups'

    tims = tractor.images
    tractor.freezeParam('images')

    minFlux = opt.minflux
    if minFlux is not None:
        minFlux = np.median([tim.sigma1 * minFlux / tim.getPhotoCal().val for tim in tims])
        print 'minFlux:', minFlux, 'nmgy'

    dpix = opt.wrad / 2.75

    sgroups = [[i] for i in singles]

    NG = len(sgroups) + len(groups)
    
    for gi,X in enumerate(sgroups + groups):

        print 'Group', gi, 'of', NG, 'groups;', len(X), 'sources'

        mysrcs = [cat[i] for i in X]
        mytims = []
        myrois = []
        for tim in tims:
            wcs = tim.getWcs()
            xy = []
            for src in mysrcs:
                xy.append(wcs.positionToPixel(src.getPosition()))
            xy = np.array(xy)
            xi,yi = xy[:,0], xy[:,1]
            H,W = tim.shape
            x0 = np.clip(int(np.floor(xi.min() - dpix)), 0, W-1)
            y0 = np.clip(int(np.floor(yi.min() - dpix)), 0, H-1)
            x1 = np.clip(int(np.ceil (xi.max() + dpix)), 0, W-1)
            y1 = np.clip(int(np.ceil (yi.max() + dpix)), 0, H-1)
            if x0 == x1 or y0 == y1:
                continue
            #myrois.append([x0,x1,y0,y1])
            myrois.append((slice(y0,y1+1), slice(x0,x1+1)))
            mytims.append(tim)

        # FIXME -- Find sources nearby!
        
        subtr = Tractor(mytims, mysrcs)
        subtr.freezeParamsRecursive('*')
        subtr.thawPathsTo(band)

        t0 = Time()
        ims0,ims1 = subtr.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
                                                     sky=False, minFlux=minFlux,
                                                     rois=myrois)
        print 'Forced phot took', Time()-t0


def stage305(opt=None, ps=None, tractor=None, band=None, bandnum=None, rd=None, **kwa):
    r0,r1,d0,d1 = rd
    cat = tractor.catalog
    tims = tractor.images

    cat.freezeParamsRecursive('*')
    tractor.thawPathsTo('sky')

    for ti,tim in enumerate(tims):
        print 'Image', ti, 'of', len(tims)

        subtr = Tractor([tim], cat)
        t0 = Time()
        ims0,ims1 = subtr.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
                                                     sky=True, minFlux=None)
        print 'Optimizing sky took', Time()-t0


def stage306(opt=None, ps=None, tractor=None, band=None, bandnum=None, rd=None, **kwa):
    r0,r1,d0,d1 = rd
    cat = tractor.catalog
    tims = tractor.images

    cat.freezeParamsRecursive('*')
    cat.thawPathsTo(band)
    tractor.thawPathsTo('sky')

    p0 = tractor.getParams()

    perimparams = []
    perimfit = []
    perimsky = []

    cat0 = cat.getParams()

    for ti,tim in enumerate(tims):
        print 'Image', ti, 'of', len(tims)

        cat.setParams(cat0)

        wcs = tim.getWcs()
        H,W = tim.shape
        margin = 5.
        srcs, fsrcs = [],[]
        ii = []
        for i,src in enumerate(cat):
            x,y = wcs.positionToPixel(src.getPosition())
            if x > 0 and x < W and y > 0 and y < H:
                srcs.append(src)
                ii.append(i)
            elif x > -margin and x < (W+margin) and y > -margin and y < (H+margin):
                fsrcs.append(src)
        print len(srcs), 'in bounds plus', len(fsrcs), 'nearby'
        subcat = Catalog(*(srcs + fsrcs))
        print len(subcat), 'in subcat'
        for i in range(len(fsrcs)):
            subcat.freezeParam(len(srcs) + i)

        ### We should do something about sources that live in regions of ~ 0 invvar!


        subtr = Tractor([tim], subcat)
        t0 = Time()
        ims0,ims1 = subtr.optimize_forced_photometry(minsb=opt.minsb, mindlnp=1.,
                                                     sky=True, minFlux=None)
        print 'Forced phot took', Time()-t0

        continue

        imas = [dict(interpolation='nearest', origin='lower',
                     vmin=tim.zr[0], vmax=tim.zr[1])]
        imchis = [dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5)]
        tt = tim.name

        pcat = []
        ptims = [tim]

        _plot_grid([img for (img, mod, ie, chi, roi) in ims0], imas)
        plt.suptitle('Data: ' + tt)
        ps.savefig()

        _plot_grid2(ims0, pcat, ptims, imas)
        plt.suptitle('Initial model: ' + tt)
        ps.savefig()

        _plot_grid2(ims0, pcat, ptims, imchis, ptype='chi')
        plt.suptitle('Initial chi: ' + tt)
        ps.savefig()

        if ims1 is not None:
            _plot_grid2(ims1, pcat, ptims, imas)
            plt.suptitle('Forced-phot model: ' + tt)
            ps.savefig()

            _plot_grid2(ims1, pcat, ptims, imchis, ptype='chi')
            plt.suptitle('Forced-phot chi: ' + tt)
            ps.savefig()

        perimparams.append(cat.getParams())
        perimfit.append(ii)
        perimsky.append(tim.getParams())

    return dict(perimflux=perimparams,
                periminds=perimfit,
                perimsky=perimsky)
                


def stage402(opt=None, ps=None, T=None, outlines=None, wcses=None, rd=None,
             band=None, bandnum=None, tims=None,
             rcf=None, cat2=None,
             **kwa):
    r0,r1,d0,d1 = rd
    # Coadd images
    ra  = (r0 + r1) / 2.
    dec = (d0 + d1) / 2.
    cosd = np.cos(np.deg2rad(dec))

    coadds = []

    for coi,pixscale in enumerate([2.75 / 3600., 0.4 / 3600]):
        W = int(np.ceil((r1 - r0) * cosd / pixscale))
        H = int(np.ceil((d1 - d0) / pixscale))
        cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                    -pixscale, 0., 0., pixscale,
                    W, H)
        print 'Target WCS:', cowcs
        coadd = np.zeros((H,W))
        comod = np.zeros((H,W))
        con   = np.zeros((H,W), int)
        for i,(tim) in enumerate(tims):
            print 'coadding', i
            # Create sub-WCS
            x0,x1,y0,y1 = tim.extent
            wcs = tim.getWcs().wcs
            wcs2 = Sip(wcs)
            cpx,cpy = wcs2.crpix
            wcs2.set_crpix((cpx - x0, cpy - y0))
            h,w = tim.shape
            wcs2.set_width(w)
            wcs2.set_height(h)
            print 'wcs2:', wcs2
            yo,xo,yi,xi,nil = resample_with_wcs(cowcs, wcs2, [], 0, spline=False)
            if yo is None:
                continue
            ok = (tim.invvar[yi,xi] > 0)
            coadd[yo,xo] += (tim.data[yi,xi] * ok)
            con  [yo,xo] += ok

            tractor = Tractor([tim], cat2)
            mod = tractor.getModelImage(0)
            comod[yo,xo] += (mod[yi,xi] * ok)

        coadd /= np.maximum(con, 1)
        comod /= np.maximum(con, 1)

        n = np.median(con)
        print 'median of', n, 'exposures'
        mn = np.median(coadd)
        st = tims[0].sigma1 / np.sqrt(n)

        plt.clf()
        plt.imshow(coadd, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=mn-2.*st, vmax=mn+5.*st)
        plt.title('WISE coadd: %s' % band)
        ps.savefig()

        plt.clf()
        plt.imshow(coadd, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=mn-2.*st, vmax=mn+20.*st)
        plt.title('WISE coadd: %s' % band)
        ps.savefig()

        plt.clf()
        plt.imshow(comod, interpolation='nearest', origin='lower', cmap='gray',
                   vmin=mn-2.*st, vmax=mn+20.*st)
        plt.title('WISE model coadd: %s' % band)
        ps.savefig()

        coadds.append(('WISE', band, pixscale, coadd, mn, st, cowcs))

        coadds.append(('WISE model 2', band, pixscale, comod, mn, st, cowcs))


        if bandnum != 1:
            continue

        r,c,f = rcf
        for sband in ['u','g','r','i','z']:
            tim,inf = get_tractor_image_dr9(r,c,f, sband, psf='dg', nanomaggies=True)
            mn = inf['sky']
            st = inf['skysig']

            print 'SDSS image:', tim
            h,w = tim.getImage().shape
            wcs = tim.getWcs()
            wcs.astrans._cache_vals()
            wcs = AsTransWrapper(wcs.astrans, w, h)
            yo,xo,yi,xi,nil = resample_with_wcs(cowcs, wcs, [],[], spline=False)
            if yo is None:
                print 'WARNING: No overlap with SDSS image?!'
            sco = np.zeros((H,W))
            sco[yo,xo] = tim.getImage()[yi,xi]
                                    
            plt.clf()
            plt.imshow(sco, interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn-2.*st, vmax=mn+5.*st)
            plt.title('SDSS image: %s band' % sband)
            ps.savefig()

            coadds.append(('SDSS', sband, pixscale, sco, mn, st, cowcs))
    return dict(coadds=coadds)


def stage403(coadds=None, **kwa):
    # Merge W2 coadds into W1 results.
    P = unpickle_from_file('w3-target-10-w2-stage402.pickle')
    co = P['coadds']
    coadds += co
    #print 'Coadds:', coadds
    return dict(coadds=coadds)

def stage404(coadds=None, ps=None, targetrd=None,
             W=None, S=None, rd=None, rcf=None,
             tims=None, band=None, cat2=None,
             **kwa):
    r0,r1,d0,d1 = rd

    #print 'Available kwargs:', kwa.keys()

    z = fits_table('zans-plates7027-7032-zscan.fits')
    I,J,d = match_radec(z.plug_ra, z.plug_dec, targetrd[0], targetrd[1],
                        1./3600.)
    print 'N matches', len(I)
    print 'zans', z[I[0]]
    z[I[0]].about()
    
    plt.subplots_adjust(hspace=0, wspace=0,
                        left=0, right=1,
                        bottom=0, top=1)
    
    wfn = 'wise-objs-w3.fits'
    W = fits_table(wfn)
    print 'Read', len(W), 'from', wfn
    W.cut((W.ra > r0) * (W.ra < r1) * (W.dec > d0) * (W.dec < d1))
    print 'Cut to', len(W), 'in RA,Dec box'
    Wise = W

    S.cut((S.ra > r0) * (S.ra < r1) * (S.dec > d0) * (S.dec < d1))
    print 'Cut to', len(S), 'SDSS'
    sdss = S

    #sdss.about()
    sdss.inblock = sdss.inblock.astype(np.uint8)
    sdss.writeto('sdss-objs.fits')
    
    fine = 0.4/3600.
    ss = []

    CC = tabledata()
    for i,c in enumerate(['name', 'band', 'pixscale', 'mean', 'std']):
        CC.set(c, np.array([x[i] for x in coadds]))

    for b in 'irg':
        sx = [(im,mn,st,wcs) for src,bb,pixscale,im,mn,st,wcs in coadds
              if src == 'SDSS' and bb == b and pixscale == fine]
        assert(len(sx) == 1)
        cowcs = sx[0][-1]
        ss.append(sx[0][:3])
        
    si,sr,sg = ss
    ww = []
    for b in ['w1','w2']:
        sx = [(im,mn,st) for src,bb,pixscale,im,mn,st,wcs in coadds
              if src == 'WISE' and bb == b and pixscale == fine]
        assert(len(sx) == 1)
        ww.append(sx[0])
    w1,w2 = ww

    # Grab post-transient-pixel-masking coadds...
    # P = unpickle_from_file('w3-target-10-w1-stage105.pickle')
    # coimg = P['coimg']
    # coinvvar = P['coinvvar']
    # w1b = coimg
    # w1b = (w1b, np.median(w1b), 1./np.sqrt(np.median(coinvvar)))
    # # Merge W2 coadds into W1 results.
    # P = unpickle_from_file('w3-target-10-w2-stage105.pickle')
    # co = P['coimg']
    # coiv = P['coinvvar']
    # w2b = co
    # w2b = (w2b, np.median(w2b), 1./np.sqrt(np.median(coiv)))

    H,W = si[0].shape
    print 'Coadd size', W, H

    sRGB = np.zeros((H,W,3))
    (im,mn,st) = si
    r = im
    print 'i-band std', st
    sRGB[:,:,0] = (im - mn) / st
    (im,mn,st) = sr
    g = im
    print 'r-band std', st
    sRGB[:,:,1] = (im - mn) / st
    (im,mn,st) = sg
    b = im
    print 'g-band std', st
    sRGB[:,:,2] = (im - mn) / st

    if False:
        # plt.clf()
        # plt.hist((r * 1.0).ravel(), bins=50, histtype='step', color='r')
        # plt.hist((g * 1.5).ravel(), bins=50, histtype='step', color='g')
        # plt.hist((b * 2.5).ravel(), bins=50, histtype='step', color='b')
        # ps.savefig()
    
        #B = 0.02
        B = 0.
    
        r = np.maximum(r * 1.0 + B, 0)
        g = np.maximum(g * 1.5 + B, 0)
        b = np.maximum(b * 2.5 + B, 0)
        I = (r+g+b)/3.
    
        #alpha = 1.5
        alpha = 2.5
        Q = 20
        m2 = 0.
        fI = np.arcsinh(alpha * Q * (I - m2)) / np.sqrt(Q)
        I += (I == 0.) * 1e-6
        R = fI * r / I
        G = fI * g / I
        B = fI * b / I
        maxrgb = reduce(np.maximum, [R,G,B])
        J = (maxrgb > 1.)
        R[J] = R[J]/maxrgb[J]
        G[J] = G[J]/maxrgb[J]
        B[J] = B[J]/maxrgb[J]
        lupRGB = np.clip(np.dstack([R,G,B]), 0., 1.)

        # plt.clf()
        # plt.imshow(lupRGB, interpolation='nearest', origin='lower')
        # ps.savefig()

    # img = np.clip(sRGB / 5., 0., 1.)
    # plt.clf()
    # plt.imshow(img, interpolation='nearest', origin='lower')
    # ps.savefig()

    img = np.clip((sRGB + 2) / 12., 0., 1.)
    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower')
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    green = (0,1,0)

    ok,sx,sy = wcs.radec2pixelxy(sdss.ra, sdss.dec)
    ax = plt.axis()
    plt.plot(sx-1, sy-1, 'o', mfc='none', mec=green, ms=30, mew=2)
    plt.axis(ax)
    ps.savefig()
    
    # img = plt.imread('sdss2.png')
    # img = np.flipud(img)
    # plt.clf()
    # plt.imshow(img, interpolation='nearest', origin='lower')
    # ps.savefig()

    wRGB = np.zeros((H,W,3))
    (im,mn,st) = w1
    wRGB[:,:,2] = (im - mn) / st
    (im,mn,st) = w2
    wRGB[:,:,0] = (im - mn) / st
    wRGB[:,:,1] = (wRGB[:,:,0] + wRGB[:,:,2]) / 2.

    img = np.clip((wRGB + 1) / 6., 0., 1.)

    plt.clf()
    plt.imshow(img, interpolation='nearest', origin='lower')
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    ok,x,y = wcs.radec2pixelxy(Wise.ra, Wise.dec)
    ax = plt.axis()
    plt.plot(x-1, y-1, 'o', mfc='none', mec=green, ms=30, mew=2)
    plt.axis(ax)
    ps.savefig()

    print 'WISE mags: W1', Wise.w1mpro
    print 'WISE mags: W2', Wise.w2mpro

    print 'x', x
    print 'y', y
    
    # img = np.clip(wRGB / 10., 0., 1.)
    # plt.clf()
    # plt.imshow(img, interpolation='nearest', origin='lower')
    # ps.savefig()

    # wRGB2 = np.zeros((H,W,3))
    # (im,mn,st) = w1b
    # wRGB2[:,:,2] = (im - mn) / st
    # (im,mn,st) = w2b
    # wRGB2[:,:,0] = (im - mn) / st
    # wRGB2[:,:,1] = (wRGB2[:,:,0] + wRGB2[:,:,2]) / 2.
    # img = np.clip(wRGB2 / 5., 0., 1.)
    # plt.clf()
    # plt.imshow(img, interpolation='nearest', origin='lower')
    # ps.savefig()
    # img = np.clip(wRGB2 / 10., 0., 1.)
    # plt.clf()
    # plt.imshow(img, interpolation='nearest', origin='lower')
    # ps.savefig()

    (im,mn,st) = sr

    ima = dict(interpolation='nearest', origin='lower', 
               vmin=mn-1.*st, vmax=mn+5.*st, cmap='gray')
    # SDSS r-band
    plt.clf()
    plt.imshow(im, **ima)
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    sband = 'r'
    # oof, the previous get_tractor_sources_dr9 did a *= -1 on the angles...
    sdss.phi_dev_deg *= -1
    sdss.phi_exp_deg *= -1
    cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                  objs=sdss, bands=['r'], nanomaggies=True,
                                  extrabands=[band],
                                  fixedComposites=True, forcePointSources=False)
    print 'Created', len(cat), 'Tractor sources'

    r,c,f = rcf
    stim,inf = get_tractor_image_dr9(r,c,f, sband, psf='dg', nanomaggies=True)
    sig1 = inf['skysig']

    H,W = im.shape
    wcs = FitsWcs(cowcs)
    tim = Image(data=np.zeros_like(im),invvar=np.zeros_like(im) + (1./sig1)**2,
                wcs=wcs, photocal=stim.photocal, sky=stim.sky, psf=stim.psf,
                domask=False)
    tractor = Tractor([tim], cat)
    rmodel = tractor.getModelImage(0)
    
    # SDSS r-band model (on coadd wcs)
    # + noise
    plt.clf()
    plt.imshow(rmodel + np.random.normal(size=rmodel.shape, scale=sig1), **ima)
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    plt.clf()
    plt.imshow(rmodel, **ima)
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    # WISE, single exposure, model (resampled to coadd wcs)
    for i,tim in enumerate(tims):
        print '  ', i, tim.name, '# pix', len(np.flatnonzero(tim.invvar))
    tim = tims[0]
    print 'Catalog:'
    for src in cat:
        print '  ', src
    tractor = Tractor([tim], cat)
    cat = tractor.catalog
    cat.freezeParamsRecursive('*')
    cat.thawPathsTo(band)
    p0 = cat.getParams()
    #minbright = 250. # mag 16.5
    minbright = 500. # mag 16.5
    cat.setParams(np.maximum(minbright, p0))

    wmod0 = tractor.getModelImage(0)

    tractor.setCatalog(cat2)
    wmod1 = tractor.getModelImage(0)
    
    sky = tim.getSky().getValue()
    imw = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=sky - 1.*tim.sigma1, vmax=sky + 5.*tim.sigma1)
    # plt.clf()
    # plt.imshow(wmod0 + 3.*tim.sigma1 * (tim.invvar == 0), **imw)
    # plt.xticks([]); plt.yticks([])
    # ps.savefig()

    print 'WISE x0,y0', tim.getWcs().getX0Y0()
    x0,y0 = tim.getWcs().getX0Y0()
    # Create sub-WCS
    wcs = tim.getWcs().wcs
    wcs2 = Sip(wcs)
    cpx,cpy = wcs2.crpix
    wcs2.set_crpix((cpx - x0, cpy - y0))
    h,w = tim.shape
    wcs2.set_width(w)
    wcs2.set_height(h)

    yo,xo,yi,xi,nil = resample_with_wcs(cowcs, wcs2, [],[])
    if yo is None:
        print 'WARNING: No overlap with WISE model?'
    rwmod0 = np.zeros_like(rmodel)
    rwmod1 = np.zeros_like(rmodel)
    rwimg = np.zeros_like(rmodel)
    rwmod0[yo,xo] = wmod0[yi,xi]
    rwmod1[yo,xo] = wmod1[yi,xi]
    rwimg[yo,xo] = tim.getImage()[yi,xi]
    
    plt.clf()
    plt.imshow(rwmod0, **imw)
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    imw2 = dict(interpolation='nearest', origin='lower', cmap='gray',
                vmin=sky - 1.*tim.sigma1, vmax=sky + 5.*tim.sigma1)
    
    plt.clf()
    plt.imshow(rwimg, **imw2)
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    plt.clf()
    plt.imshow(rwmod1, **imw2)
    plt.xticks([]); plt.yticks([])
    ps.savefig()

    print 'band', band
    
    I = np.flatnonzero((CC.name == 'WISE model 2') * (CC.pixscale == fine) *
                       (CC.band == band))
    assert(len(I) == 1)
    nil,nil,nil,wcomod,mn,st,nil = coadds[I[0]]
    I = np.flatnonzero((CC.name == 'WISE') * (CC.pixscale == fine) *
                       (CC.band == band))
    assert(len(I) == 1)
    nil,nil,nil,wcoimg,mn,st,nil = coadds[I[0]]

    imw = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=mn - 3.*st, vmax=mn + 15.*st)
    
    plt.clf()
    plt.imshow(wcomod, **imw)
    ps.savefig()

    plt.clf()
    plt.imshow(wcoimg, **imw)
    ps.savefig()

    ax = plt.axis()
    plt.plot(sx-1, sy-1, 'o', mfc='none', mec=green, ms=30, mew=2)
    plt.axis(ax)
    ps.savefig()
    
    
def stage509(cat1=None, cat2=None, cat3=None, bandnum=None,
             band=None, S=None,
             **kwa):
    # Would it still pass the QSO selection cuts?  Write out FITS tables, run selection
    # --> yes.
    for i,cat in [(1,cat1), (2,cat2), (3,cat3)]:
        R = tabledata()
        R.ra  = np.array([src.getPosition().ra  for src in cat])
        R.dec = np.array([src.getPosition().dec for src in cat])
        R.set(band, np.array([src.getBrightness().getBand(band) for src in cat]))
        #R.set(band + '_ivar', IV)
        R.writeto('w3-tr-cat%i-w%i.fits' % (i, bandnum))

    S.writeto('w3-tr-sdss.fits')

def stage510(S=None, W=None, targetrd=None, **kwa):
    for cat in [1,2,3, 4,5]:
        if cat in [4,5]:
            I,J,d = match_radec(S.ra, S.dec, W.ra, W.dec, 4./3600.)
            print len(I), 'matches'
            S.cut(I)
            W.cut(J)
            if cat == 4:
                S.w1 = NanoMaggies.magToNanomaggies(W.w1mpro)
                S.w2 = NanoMaggies.magToNanomaggies(W.w2mpro)
            elif cat == 5:
                S.w1 = NanoMaggies.magToNanomaggies(W.w1mag)
                S.w2 = NanoMaggies.magToNanomaggies(W.w2mag)
        else:
            W1,W2 = [fits_table('w3-tr-cat%i-w%i.fits' % (cat,bandnum)) for
                     bandnum in [1,2]]
            NS = len(S)
            assert(np.all(W1.ra[:NS] == S.ra))
            assert(np.all(W1.dec[:NS] == S.dec))
            assert(np.all(W2.ra[:NS] == S.ra))
            assert(np.all(W2.dec[:NS] == S.dec))
            W1 = W1[:NS]
            W2 = W2[:NS]
            S.w1 = W1.w1
            S.w2 = W2.w2

        fluxtomag = NanoMaggies.nanomaggiesToMag

        wmag = (S.w1 * 1.0 + S.w2 * 0.5) / 1.5
        I = np.flatnonzero(wmag)
        Si = S[I]

        Si.wise = fluxtomag(wmag[I])
        Si.optpsf = fluxtomag((Si.psfflux[:,1] * 0.8 +
                               Si.psfflux[:,2] * 0.6 +
                               Si.psfflux[:,3] * 1.0) / 2.4)
        Si.optmod = fluxtomag((Si.modelflux[:,1] * 0.8 +
                               Si.modelflux[:,2] * 0.6 +
                               Si.modelflux[:,3] * 1.0) / 2.4)
        Si.gpsf = fluxtomag(Si.psfflux[:,1])
        Si.rpsf = fluxtomag(Si.psfflux[:,2])
        Si.ipsf = fluxtomag(Si.psfflux[:,3])
        Si.ispsf = (Si.objc_type == 6)
        Si.isgal = (Si.objc_type == 3)

        in1 = ( ((Si.gpsf - Si.ipsf) < 1.5) *
                (Si.optpsf > 17.) *
                (Si.optpsf < 22.) *
                ((Si.optpsf - Si.wise) > ((Si.gpsf - Si.ipsf) + 3)) *
                np.logical_or(Si.ispsf, (Si.optpsf - Si.optmod) < 0.1) )

        (r,d) = targetrd

        I,J,d = match_radec(Si.ra, S.dec, np.array([r]), np.array([d]), 1./3600.)
        print 'Matched', len(I)
        
        print 'in1:', in1[I]

        print 'W1', Si.w1[I], fluxtomag(Si.w1[I])
        print 'W2', Si.w2[I], fluxtomag(Si.w2[I])


# Plots for LBL talk
# FIXME -- copy'n'pasted, doesn't work
def stage606():

    plt.figure(figsize=(8,8))
    ima = dict(interpolation='nearest', origin='lower',
               vmin=-2*sig, vmax=10*sig)
    R,C = 2,3
    plt.clf()
    plt.suptitle('First-round Coadds')
    plt.subplot(R,C,1)
    plt.imshow(nnimg, **ima)
    #plt.colorbar()
    plt.title('NN data')
    plt.subplot(R,C,2)
    plt.imshow(coimg, **ima)
    #plt.colorbar()
    plt.title('Data')
    plt.subplot(R,C,3)
    plt.imshow(comod, **ima)
    #plt.colorbar()
    plt.title('Model')
    plt.subplot(R,C,4)
    plt.imshow(coppstd, interpolation='nearest', origin='lower')
    #plt.colorbar()
    plt.title('Coadd std')
    plt.subplot(R,C,5)
    plt.imshow(cochi, interpolation='nearest', origin='lower',
               vmin=-5., vmax=5.)
    #plt.colorbar()
    plt.title('Chi')
    plt.subplot(R,C,6)
    plt.imshow(cochi, interpolation='nearest', origin='lower',
               vmin=-20., vmax=20.)
    #plt.colorbar()
    plt.title('Chi (b)')
    ps.savefig()
    
    ima = dict(interpolation='nearest', origin='lower',
               vmin=-2*sig, vmax=10*sig)
    
    R,C = 2,3
    
    plt.clf()
    plt.suptitle('Second-round Coadds')
    plt.subplot(R,C,1)
    plt.imshow(nn, **ima)
    # plt.colorbar()
    plt.title('NN data')
    plt.subplot(R,C,2)
    plt.imshow(coimg, **ima)
    # plt.colorbar()
    plt.title('Data')
    plt.subplot(R,C,3)
    plt.imshow(comod, **ima)
    # plt.colorbar()
    plt.title('Model')
    plt.subplot(R,C,4)
    plt.imshow(coppstd, interpolation='nearest', origin='lower')
    # plt.colorbar()
    plt.title('Coadd std')
    plt.subplot(R,C,5)
    plt.imshow(cochi, interpolation='nearest', origin='lower',
               vmin=-5., vmax=5.)
    # plt.colorbar()
    plt.title('Chi')
    plt.subplot(R,C,6)
    plt.imshow(cochi, interpolation='nearest', origin='lower',
               vmin=-20., vmax=20.)
    # plt.colorbar()
    plt.title('Chi (b)')
    ps.savefig()

    # 2. Apply rchi masks to individual images
    tims = tractor.getImages()

    args = []
    for i,(tim, d) in enumerate(zip(tims, ims)):
        args.append((tim, d.mask, cowcs))
    rmasks = mp.map(_rev_resample_mask, args)
    for i,(mask,tim) in enumerate(zip(rmasks, tims)):
        # if i < 10:
        #     plt.clf()
        #     plt.subplot(1,2,1)
        #     plt.imshow(ims[i].mask)
        #     plt.colorbar()
        #     if mask is not None:
        #         plt.subplot(1,2,2)
        #         plt.imshow(mask)
        #         plt.colorbar()
        #     ps.savefig()

        # LBNL
        if i != 62:
            continue
    
        tim.coaddmask = mask
        if mask is not None:
            tim.setInvvar(tim.invvar * (mask > 0))
        else:
            tim.setInvvar(tim.invvar)


        d = ims[i]
        sig1 = tim.sigma1 * d.scale
        R,C = 2,3
        plt.clf()
        plt.subplot(R,C,1)
        plt.imshow(d.rimg, interpolation='nearest', origin='lower',
                   vmin=-2*sig1, vmax=5*sig1)
        plt.title('resamp data')
    
        plt.subplot(R,C,2)
        plt.imshow(d.rmod, interpolation='nearest', origin='lower',
                   vmin=-2*sig1, vmax=5*sig1)
        plt.title('resamp mod')
    
        plt.subplot(R,C,3)
        chi = (d.rimg - d.rmod) * d.mask / sig1
        plt.imshow(chi, interpolation='nearest', origin='lower',
                   vmin=-5, vmax=5, cmap='gray')
        plt.title('chi2: %.1f' % np.sum(chi**2))

        # grab original rchi
        rchi = rchis[i]
        rchi2 = np.sum(rchi**2) / np.sum(d.mask)

        plt.subplot(R,C,4)
        plt.imshow(rchi, interpolation='nearest', origin='lower',
                   vmin=-5, vmax=5, cmap='gray')
        plt.title('rchi2 vs coadd: %.2f' % rchi2)

        plt.subplot(R,C,5)
        plt.imshow(np.abs(rchi) > 5, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('abs(rchi) > 5')

        plt.subplot(R,C,6)
        plt.imshow(d.mask, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('mask')

        plt.suptitle(d.name)
        ps.savefig()

        if i == 62:
            # LBNL plots
            plt.subplots_adjust(hspace=0, wspace=0,
                                left=0, right=1,
                                bottom=0, top=1)
            plt.clf()
            plt.imshow(coimg1, cmap='gray', **ima)
            ps.savefig()

            plt.clf()
            plt.imshow(coimg, cmap='gray', **ima)
            ps.savefig()

            plt.clf()
            plt.imshow(d.rimg, interpolation='nearest', origin='lower',
                       vmin=-2*sig1, vmax=5*sig1, cmap='gray')
            ps.savefig()

            plt.clf()
            plt.imshow(rchi, interpolation='nearest', origin='lower',
                       vmin=-5, vmax=5, cmap='gray')
            ps.savefig()

            plt.clf()
            plt.imshow(d.mask, interpolation='nearest', origin='lower', cmap='gray')
            ps.savefig()


# plots from stage 10?; reduce pickle size
def stage700(opt=None, ps=None, tractor=None, band=None, bandnum=None, T=None,
             S=None, ri=None, di=None, UW=None,
             coimg1=None, coinvvar=None, coimg=None, cowcs=None,
             resampled=None,
             ims1=None,
             **kwa):

    # for k,v in kwa.items():
    #     print
    #     print k, '='
    #     print
    #     print v
    #     print

    if ps:
        cosig = 1./np.sqrt(np.median(coinvvar))
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=-2*cosig, vmax=10*cosig)
        plt.clf()
        plt.imshow(coimg1, **ima)
        plt.gray()
        plt.title('Initial coadd')
        ps.savefig()
    
        plt.clf()
        plt.imshow(coimg, **ima)
        plt.gray()
        plt.title('Final coadd')
        ps.savefig()
    
        ax = plt.axis()
        ok,X,Y = cowcs.radec2pixelxy(S.ra, S.dec)
        plt.plot(X-1, Y-1, 'b+')
        ok,X,Y = cowcs.radec2pixelxy(UW.ra, UW.dec)
        plt.plot(X-1, Y-1, 'r+')
        plt.axis(ax)
        ps.savefig()

    tims = tractor.getImages()

    for i,(d,tim, (nil,mod,nil,chi,nil)) in enumerate(zip(resampled, tims, ims1)):
        if d is None:
            continue
        #print 'd:', dir(d)
        d.rimg = d.rimg.astype(np.float32)
        d.rchi = d.rchi.astype(np.float32)
        delattr(d, 'nnimg')
        delattr(d, 'invvar')
        delattr(d, 'ww')
        delattr(d, 'mod')
        delattr(d, 'img')

        # print 'd:'
        # for x in dir(d):
        #     if x.startswith('_'):
        #         continue
        #     try:
        #         X = getattr(d, x)
        #         dt = X.dtype
        #         sh = None
        #         try:
        #             sh = X.shape
        #         except:
        #             pass
        #         print '  ', x, dt,
        #         if sh is not None:
        #             print sh
        #         else:
        #             print
        #     except:
        #         continue
        if tim.coaddmask is None:
            continue

        tim.coaddmask = tim.coaddmask.astype(bool)
        del tim.cinvvar
        del tim.inverr
        del tim.origInvvar
        del tim.orig_invvar
        del tim.starMask
        del tim.uncplane
        del tim.maskplane
        del tim.vinvvar
        del tim.rdmask
        del tim.goodmask

        if i >= 10 or ps is None:
            del tim.coaddmask
            del d.rchi
            continue

        sig = tim.sigma1
        imm = dict(interpolation='nearest', origin='lower',
                   vmin=0, vmax=1, cmap='gray')
        imchi = dict(interpolation='nearest', origin='lower',
                     vmin=-5., vmax=5., cmap='gray')
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=-2*sig, vmax=10*sig, cmap='gray')

        R,C = 2,3
        plt.clf()
        plt.subplot(R,C,1)
        plt.imshow(d.rimg, **ima)
        plt.title('resampled img')
        plt.subplot(R,C,2)
        plt.imshow(d.rchi, **imchi)
        plt.title('chi vs coadd')
        #plt.subplot(R,C,3)
        #plt.imshow(tim.coaddmask, **imm)
        #plt.title('masked from coadd')
        plt.subplot(R,C,3)
        plt.imshow(d.mask, **imm)
        plt.title('masked')
        plt.subplot(R,C,4)
        plt.imshow(tim.data, **ima)
        plt.title('image')
        plt.subplot(R,C,5)
        plt.imshow(mod, **ima)
        plt.title('model')
        plt.subplot(R,C,6)
        plt.imshow(chi, **imchi)
        plt.title('chi')
        plt.suptitle(tim.name)
        ps.savefig()

        del tim.coaddmask
        del d.rchi


    return dict(tractor=tractor, resampled=resampled,
                coppstd1=None,
                ims0=None, ims1=None,
                tims=None)


def stage701(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             band=None, bandnum=None,
             tractor=None, 
             coimg=None, coinvvar=None, cowcs=None,
             R=None,
             **kwa):

    # 4. Run forced photometry on coadd
    from wise_psf import WisePSF
    psf = WisePSF(bandnum, savedfn='w%ipsffit.fits' % bandnum)
    wcs = ConstantFitsWcs(cowcs)
    pcal = LinearPhotoCal(1., band=band)
    sky = ConstantSky(0.)
    # HACK
    psf = psf.mogAt(500., 500.)

    coim = Image(data=coimg, invvar=coinvvar, wcs=wcs, photocal=pcal, sky=sky,
                 psf=psf, name='coadd', domask=False)

    cat = tractor.getCatalog().copy()
    tr = Tractor([coim], cat)
    tr.freezeParamsRecursive('*')
    tr.thawPathsTo('sky')
    tr.thawPathsTo(band)

    mod0 = tr.getModelImage(0)

    minsb = 0.005
    minFlux = None
    t0 = Time()
    ims0,ims1,IV,fs = tr.optimize_forced_photometry(minsb=minsb, mindlnp=1.,
                                                    sky=True, minFlux=minFlux,
                                                    fitstats=True,
                                                    variance=True)
    print 'Forced phot on coadd took', Time()-t0

    if ps:
        mod1 = tr.getModelImage(0)
        chi0 = (coimg - mod0) * coim.getInvError()
        chi1 = (coimg - mod1) * coim.getInvError()

        cosig = 1./np.sqrt(np.median(coinvvar))
        ima = dict(interpolation='nearest', origin='lower',
                   vmin=-2*cosig, vmax=10*cosig)
        imchi = dict(interpolation='nearest', origin='lower',
                     vmin=-5., vmax=5., cmap='gray')
        rows,cols = 2,3
        plt.clf()
        plt.subplot(rows,cols,1)
        plt.imshow(coimg, **ima)
        plt.gray()
        plt.title('Coadd')
        plt.subplot(rows,cols,2)
        plt.imshow(mod0, **ima)
        plt.gray()
        plt.title('Model 0')
        plt.subplot(rows,cols,3)
        plt.imshow(chi0, **imchi)
        plt.gray()
        plt.title('Chi 0')
        plt.subplot(rows,cols,5)
        plt.imshow(mod1, **ima)
        plt.gray()
        plt.title('Model 1 (sky: %.1f)' % coim.getSky().getValue())
        plt.subplot(rows,cols,6)
        plt.imshow(chi1, **imchi)
        plt.gray()
        plt.title('Chi 1')
        ps.savefig()

        m1,m2 = [],[]
        for s1,s2 in zip(tractor.getCatalog(), cat):
            m1.append(NanoMaggies.nanomaggiesToMag(s1.getBrightness().getBand(band)))
            m2.append(NanoMaggies.nanomaggiesToMag(s2.getBrightness().getBand(band)))
        m1 = np.array(m1)
        m2 = np.array(m2)

        plt.clf()
        I = (R.inblock > 0) * (R.sdss > 0)
        p1 = plt.plot(m1[I], (m2-m1)[I], 'b.')
        I = np.logical_not(R.inblock)
        p2 = plt.plot(m1[I], (m2-m1)[I], 'g.')
        I = np.logical_not(R.sdss)
        p3 = plt.plot(m1[I], (m2-m1)[I], 'r.')
        lo,hi = 12,24
        plt.plot([lo,hi],[0, 0], 'k-', lw=3, alpha=0.3)
        plt.axis([lo,hi,-2,2])
        plt.xlabel('Individual images photometry (mag)')
        plt.ylabel('Coadd photometry - Individual (mag)')
        plt.legend((p1,p2,p3),('In bounds', 'WISE-only', 'Out-of-bounds'))
        ps.savefig()

    R.set('coadd_' + band, np.array([src.getBrightness().getBand(band) for src in cat]))
    R.set('coadd_' + band + '_ivar', IV)

    # for f1,f2,e1,e2 in zip(R.get(band), R.get('coadd_'+band),
    #                        R.get(band+'_ivar'), R.get('coadd_'+band+'_ivar')):
    #     print '  %8.2f, %8.2f, %8.2f, %8.2f' % (f1,f2,e1,e2)

    return dict(cat2=cat, R=R, cotim=coim)

def stage702(opt=None, ps=None, ralo=None, rahi=None, declo=None, dechi=None,
             band=None, bandnum=None,
             tractor=None, 
             coimg=None, coinvvar=None, cowcs=None,
             R=None,
             cotim=None,
             **kwa):

    # Try treating some sources at point sources rather than galaxies:

    # - for SDSS size < X
    # - SDSS objs with no WISE match (ie, faint in WISE)?
    # - SDSS objs with mag < XXX (ie, SDSS shapes dubious)

    for tim in tractor.getImages():
        tim.setInvvar(tim.invvar)

    cat = tractor.getCatalog()
    for src in cat:
        if isinstance(src, FixedCompositeGalaxy):
            print 'Comp', src.shapeDev.re, src.shapeExp.re
        elif isinstance(src, ExpGalaxy):
            print 'Exp', src.re
        elif isinstance(src, DevGalaxy):
            print 'Dev', src.re

    pointsources = []
    
    nchanged = 0
    for size in [ 0.1, 0.2, 0.4, 0.8 ]:
        cati = []

        for src in cat:
            if isinstance(src, FixedCompositeGalaxy):
                #print 'Comp', src.shapeDev.re, src.shapeExp.re
                if src.shapeDev.re < size and src.shapeExp.re < size:
                    cati.append(PointSource(src.pos, src.brightness))
                    nchanged += 1
                else:
                    cati.append(src.copy())
            elif isinstance(src, ExpGalaxy) or isinstance(src, DevGalaxy):
                if src.re < size:
                    cati.append(PointSource(src.pos, src.brightness))
                    nchanged += 1
                else:
                    cati.append(src.copy())
            else:
                cati.append(src.copy())

        print 'Changed', nchanged, 'galaxies to point sources'
        if nchanged == 0:
            continue

        tr = Tractor(tractor.getImages(), cati)

        t0 = Time()
        tractor.freezeParamsRecursive('*')
        tractor.thawPathsTo('sky')
        tractor.thawPathsTo(band)
        minsb = 0.005
        minFlux = None
        ims0,ims1,IV,fs = tractor.optimize_forced_photometry(
            minsb=opt.minsb, mindlnp=1., sky=True, minFlux=minFlux,
            fitstats=True, variance=True)
        print 'Forced phot took', Time()-t0

        lnp = tr.getLogProb()
        print 'Log prob:', lnp

        if ps:
            mod = tr.getModelImage(cotim)
            chi = (cotim.getImage() - mod) * cotim.getInvError()
            cosig = 1./np.median(cotim.getInvError())
            ima = dict(interpolation='nearest', origin='lower',
                       vmin=-2*cosig, vmax=10*cosig)
            imchi = dict(interpolation='nearest', origin='lower',
                         vmin=-5., vmax=5., cmap='gray')
            rows,cols = 1,3
            plt.clf()
            plt.subplot(rows,cols,1)
            plt.imshow(cotim.getImage(), **ima)
            plt.gray()
            plt.title('Coadd')
            plt.subplot(rows,cols,2)
            plt.imshow(mod, **ima)
            plt.gray()
            plt.title('Model')
            plt.subplot(rows,cols,3)
            plt.imshow(chi, **imchi)
            plt.gray()
            plt.title('Galaxies < %f -> ptsrcs; lnp %.1f' % (size, lnp))
            ps.savefig()

        pointsources.append(('size', size, cati, lnp))


        
def main():

    #plt.figure(figsize=(12,12))
    #plt.figure(figsize=(10,10))
    plt.figure(figsize=(8,8))


    import optparse
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-v', dest='verbose', action='store_true')

    parser.add_option('--ri', dest='ri', type=int,
                      default=0, help='RA slice')
    parser.add_option('--di', dest='di', type=int,
                      default=0, help='Dec slice')

    parser.add_option('-S', '--stage', dest='stage', type=int,
                      default=0, help='Run to stage...')
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
                      help="Force re-running the given stage(s) -- don't read from pickle.")

    parser.add_option('--nw', dest='write', action='store_false', default=True,
                      help='Do not write stage pickle files')

    parser.add_option('-w', dest='bandnum', type=int, default=1,
                      help='WISE band (default %default)')

    parser.add_option('--ppat', dest='picklepat', default=None,
                      help='Stage pickle pattern')

    parser.add_option('--threads', dest='threads', type=int, help='Multiproc')

    parser.add_option('--osources', dest='osources',
                      help='File containing competing measurements to produce a model image for')

    parser.add_option('-s', dest='sources',
                      help='Input SDSS source list')

    parser.add_option('--not-sdss', dest='nonsdss', action='store_true',
                      help='File given in "-s" are not SDSS objects; just read RA,Dec and make point sources')

    parser.add_option('-W', dest='wsources',
                      help='WISE source list')

    parser.add_option('-i', dest='individual', action='store_true',
                      help='Fit individual images?')

    parser.add_option('-n', dest='name', default='wise',
                      help='Base filename for outputs (plots, stage pickles)')

    parser.add_option('-P', dest='ps', default=None,
                      help='Filename pattern for plots')

    
    parser.add_option('-M', dest='plotmask', action='store_true',
                      help='Plot mask plane bits?')

    parser.add_option('--ptsrc', dest='ptsrc', action='store_true',
                      help='Set all sources to point sources')
    parser.add_option('--pixpsf', dest='pixpsf', action='store_true',
                      help='Use pixelized PSF -- use with --ptsrc')

    parser.add_option('--nonconst-invvar', dest='constInvvar', action='store_false',
                      default=True, help='Do not set the invvar constant')

    parser.add_option('--wrad', dest='wrad', default=15., type=float,
                      help='WISE radius: look at a box this big in arcsec around the source position')
    parser.add_option('--srad', dest='srad', default=0., type=float,
                      help='SDSS radius: grab SDSS sources within this radius in arcsec.  Default: --wrad + 5')

    parser.add_option('--minsb', dest='minsb', type=float, default=0.05,
                      help='Minimum surface-brightness approximation, default %default')

    parser.add_option('--minflux', dest='minflux', type=str, default="-5",
                      help='Minimum flux a source is allowed to have, in sigma; default %default; "none" for no limit')

    parser.add_option('-p', dest='plots', action='store_true',
                      help='Make result plots?')
    parser.add_option('-r', dest='result',
                      help='result file to compare', default='measurements-257.fits')
    parser.add_option('-m', dest='match', action='store_true',
                      help='do RA,Dec match to compare results; else assume 1-to-1')
    parser.add_option('-N', dest='nearest', action='store_true', default=False,
                      help='Match nearest, or all?')

    #parser.add_option('--noResampleSpline', dest='resampleSpline', 

    opt,args = parser.parse_args()

    if opt.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.picklepat is None:
        opt.picklepat = opt.name + '-stage%0i.pickle'
    if opt.ps is None:
        opt.ps = opt.name

    if opt.threads:
        mp = multiproc(opt.threads)
    else:
        mp = multiproc(1)

    if opt.minflux in ['none','None']:
        opt.minflux = None
    else:
        opt.minflux = float(opt.minflux)

    # W3 area
    r0,r1 = 210.593,  219.132
    d0,d1 =  51.1822,  54.1822
    dd = np.linspace(d0, d1, 51)
    rr = np.linspace(r0, r1, 91)

    #basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise1test'
    #wisedatadirs = [(os.path.join(basedir, 'allsky'), 'cryo'),
    #                (os.path.join(basedir, 'prelim_postcryo'), 'post-cryo'),]
    basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise_frames'
    wisedatadirs = [(basedir, 'merged'),]

    opt.wisedatadirs = wisedatadirs


    ri = opt.ri
    di = opt.di
    if ri == -1 and True:

        # T = fits_table('/clusterfs/riemann/raid006/bosswork/boss/spectro/redux/current/7027/v5_6_0/spZbest-7027-56448.fits')
        # print 'Read', len(T), 'spZbest'
        # P = fits_table('/clusterfs/riemann/raid006/bosswork/boss/spectro/redux/current/7027/spPlate-7027-56448.fits', hdu=5)
        # print 'Read', len(P), 'spPlate'

        T = fits_table('/home/schlegel/wise1ext/sdss/zans-plates7027-7032-zscan.fits', column_map={'class':'clazz'})
        print 'Read', len(T), 'zscan'

        I = np.flatnonzero(T.zwarning == 0)
        print len(I), 'with Zwarning = 0'

        print 'Classes:', np.unique(T.clazz)

        qso = 'QSO   '
        I = np.flatnonzero((T.zwarning == 0) * (T.clazz == qso))
        print len(I), 'QSO with Zwarning = 0'

        print 'Typescans:', np.unique(T.typescan)
        qsoscan = 'QSO    '
        I = np.flatnonzero((T.zwarning == 0) * (T.clazz == qso) * (T.typescan == qsoscan))
        print len(I), 'QSO, scan QSO, with Zwarning = 0'

        for zcut in [2, 2.3, 2.5]:
            I = np.flatnonzero((T.zwarning == 0) * (T.clazz == qso) * (T.typescan == qsoscan) * (T.z > zcut))
            print len(I), 'QSO, scan QSO, with Zwarning = 0, z >', zcut

            I = np.flatnonzero((T.zwarning == 0) * (T.clazz == qso) * (T.typescan == qsoscan) * (T.zscan > zcut))
            print len(I), 'QSO, scan QSO, with Zwarning = 0, z and zscan >', zcut

        #zcut = 2.3
        zcut = 2.0
        I = np.flatnonzero((T.zwarning == 0) * (T.clazz == qso) * (T.typescan == qsoscan) * (T.zscan > zcut))
        print len(I), 'QSO, scan QSO, with Zwarning = 0, z and zscan >', zcut

        T.cut(I)

        # S = fits_table('ancil/objs-eboss-w3-dr9.fits')
        # print 'read', len(S), 'SDSS objs'
        # I,J,d = match_radec(T.plug_ra, T.plug_dec, S.ra, S.dec, 1./3600.)
        # print len(I), 'match'
        # print len(np.unique(I)), 'unique fibers'
        # print len(np.unique(J)), 'unique SDSS'
        # S.cut(J)
        # T.cut(I)
        
        A = fits_table('ancil/ancil-QSO-eBOSS-W3-ADM-dr8.fits')
        print 'Read', len(A), 'targets'
        I,J,d = match_radec(T.plug_ra, T.plug_dec, A.ra, A.dec, 1./3600.)
        print len(I), 'matched'
        print len(np.unique(I)), 'unique fibers'
        print len(np.unique(J)), 'unique targets'

        A.cut(J)
        #S.cut(I)
        T.cut(I)

        I = np.flatnonzero(A.w3bitmask == 4)
        print 'Selected by WISE only:', len(I)
        A.cut(I)
        T.cut(I)
        T.ra = T.plug_ra
        T.dec = T.plug_dec

        # #for j,i in enumerate(I):
        # for j,i in enumerate(range(len(T))):
        #     ra,dec = T.ra[i], T.dec[i]
        #     jfn = 'sdss-%i.jpg' % (j)
        #     if os.path.exists(jfn):
        #         continue
        #     url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=%f&dec=%f&scale=0.4&width=200&height=200&opt=G' % (ra,dec)
        #     cmd = 'wget "%s" -O %s' % (url, jfn)
        #     os.system(cmd)
        #     print j, 'RA,Dec', ra, dec
        #     
        # print '|              ra|              dec|'
        # print '|          double|           double|'
        # print '|             deg|              deg|'
        # for j,i in enumerate(range(len(T))):
        #     ra,dec = T.ra[i], T.dec[i]
        #     print ' %16.12f %16.12f' % (ra,dec)
        # print

        import fitsio

        atlas = fits_table('w3-atlas.fits')
        for j,i in enumerate(range(len(T))):

            if j != 9:
                continue

            print
            print 'source', j
            print
            T[i].about()
            print

            ra,dec = T.ra[i], T.dec[i]
            pixscale = 2.75
            for k,(rc,dc) in enumerate(zip(atlas.ra, atlas.dec)):
                W,H = 2048,2048
                cowcs = Tan(rc, dc, (W+1)/2., (H+1)/2.,
                            -pixscale/3600., 0., 0., pixscale/3600., W, H)
                if not cowcs.is_inside(ra, dec):
                    continue
                
                co = atlas.coadd_id[k]
                print 'source', j, 'RA,Dec', ra, dec, 'tile index', k, co
                imgs = []
                ok, x,y = cowcs.radec2pixelxy(ra, dec)
                print 'x,y', x,y
                x = int(np.round(x)-1)
                y = int(np.round(y)-1)
                S = 6
                ix0 = max(0, x-S)
                iy0 = max(0, y-S)
                slc = (slice(max(0, y-S), min(H, y+S+1)),
                       slice(max(0, x-S), min(W, x+S+1)))
                imextent = [slc[1].start - 0.5, slc[1].stop - 0.5,
                            slc[0].start - 0.5, slc[0].stop - 0.5]
                # imextent = [slc[1].start - 1, slc[1].stop - 1,
                #             slc[0].start - 1, slc[0].stop - 1]
                #imextent = [slc[1].start - 1, slc[1].stop - 1,
                #            slc[0].start - 1, slc[0].stop - 1]

                # Get matching SDSS image
                rs,ds = cowcs.pixelxy2radec(ix0+S + 1, iy0+S + 1)
                #rs,ds = ra,dec
                SW,SH = 180,180
                pscale = (2*S+1) * 2.75 / SW

                pngfn = 'sdss-%i.png' % j
                if not os.path.exists(pngfn):
                    jfn = 'sdss-%i.jpg' % (j)
                    url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=%f&dec=%f&scale=%f&width=%i&height=%i' % (rs,ds,pscale,SW,SH)
                    cmd = 'wget "%s" -O %s' % (url, jfn)
                    os.system(cmd)
                    cmd = 'jpegtopnm %s | pnmtopng > %s' % ('sdss-%i.jpg' % j, 'sdss-%i.png' % j)
                    os.system(cmd)

                for band in [1,2]:
                    fn = 'unwise-coadds/%s/%s/unwise-%s-w%i-img-m.fits' % (co[:3], co, co, band)
                    if not os.path.exists(fn):
                        print 'Does not exist:', fn
                        continue
                    img = fitsio.read(fn)
                    subimg = img[slc]
                    #print 'Subimg:', subimg.shape
                    imgs.append(subimg)

                if len(imgs) != 2:
                    continue
                sdss = plt.imread(pngfn)

                mods = []
                for band in [1,2]:
                    modfn = 'fit-%s-w%i-mod.fits' % (co, band)
                    if os.path.exists(modfn):
                        mod = fitsio.read(modfn)
                        #print 'Read mod', modfn, mod.shape
                        mod = mod[slc]
                        #print 'cut to', mod.shape
                        mods.append(mod)

                chis = []
                for band in [1,2]:
                    chifn = 'fit-%s-w%i-chi.fits' % (co, band)
                    if os.path.exists(chifn):
                        chi = fitsio.read(chifn)
                        chi = chi[slc]
                        chis.append(chi)


                wsrcs = None
                wfn = 'w3-phot-temp/wise-sources-%s.fits' % co
                if os.path.exists(wfn):
                    W = fits_table(wfn)
                    W.cut(degrees_between(ra, dec, W.ra, W.dec) < S*2*2.75/3600.)
                    ok,x,y = cowcs.radec2pixelxy(W.ra, W.dec)
                    wsrcs = (x,y, (W.w1mpro, W.w2mpro))


                ima = dict(interpolation='nearest', origin='lower',
                           vmin=-2, vmax=50., cmap='gray', extent=imextent)


                phot = None
                pfn = 'w3-phot/phot-%s.fits' % co
                if os.path.exists(pfn):
                    P = fits_table(pfn)
                    P.cut(degrees_between(ra, dec, P.ra, P.dec) < S*2*2.75/3600.)
                    ok,x,y = cowcs.radec2pixelxy(P.ra, P.dec)
                    phot = (x, y, (P.w1_mag, P.w2_mag))

                    mod2 = []

                    imods = []

                    #
                    for band in [1,2]:
                        img = imgs[band-1]
                        psf = fits_table('wise-psf-avg.fits', hdu=band)
                        psf = GaussianMixturePSF(psf.amp, psf.mean, psf.var)
                        twcs = ConstantFitsWcs(cowcs)
                        twcs.setX0Y0(ix0, iy0)
                        tsky = ConstantSky(0.)
                        wanyband = 'w'
                        tim = Image(data=img, invvar=np.ones_like(img), psf=psf, wcs=twcs,
                                    sky=tsky, photocal=LinearPhotoCal(1., band=wanyband),
                                    domask=False)
                        cat = []
                        iskip = []
                        nm = P.get('w%i_nanomaggies' % band)
                        for ii,(r,d,w) in enumerate(zip(P.ra, P.dec, nm)):
                            dd = (np.abs(r-ra) + np.abs(d-dec)) * 3600.
                            print 'dist', dd
                            if dd < 0.5:
                                print 'Skipping', r, d
                                iskip.append(ii)
                                continue
                            cat.append(PointSource(RaDecPos(r,d),
                                                   NanoMaggies(**{wanyband: w})))
                        tr = Tractor([tim], cat)
                        mod2.append(tr.getModelImage(0))

                        bandmods = []
                        plt.figure(figsize=(4,4))
                        cat = [PointSource(RaDecPos(r,d),
                                           NanoMaggies(**{wanyband: w}))
                               for r,d,w in zip(P.ra[iskip], P.dec[iskip], nm[iskip])] + cat

                        for si,src in enumerate(cat):
                            tr = Tractor([tim], [src])
                            imod = tr.getModelImage(0)
                            bandmods.append(imod)
                            plt.clf()
                            plt.imshow(imod, **ima)
                            plt.title('source %i band w%i' % (si, band))
                            plt.savefig('imod-%i-%02i-w%i.png' % (j, si, band))
                        imods.append(bandmods)


                #for jj,(mods1,mods2) in enumerate(zip(*imods)):
                mods1,mods2 = imods
                print 'mods1:', len(mods1), 'mods2:', len(mods2)
                rgbs = []
                for ii,(mod1,mod2) in enumerate(zip(mods1 + [mods[0], imgs[0]],
                                                    mods2 + [mods[1], imgs[1]])):
                    mh,mw = mod1.shape
                    rgb = np.zeros((mh,mw,3), np.float32)
                    #rgb[:,:,0] = (mod2  + 2.) / 30.
                    #rgb[:,:,2] = (mod1  + 2.) / 30.
                    rgb[:,:,0] = (mod2  + 10.) / 50.
                    rgb[:,:,2] = (mod1  + 10.) / 50.
                    rgb[:,:,1] = (rgb[:,:,0] + rgb[:,:,2]) / 2.
                    rgbs.append(rgb)

                # plt.clf()
                # plt.imshow(np.clip(rgb, 0, 1), interpolation='nearest', origin='lower')
                # fn = 'imod-%i-%02i.png' % (j, ii)
                # plt.savefig(fn)
                # print 'wrote', fn

                plt.figure(figsize=(3,3))
                plt.subplots_adjust(left=0.01, right=0.99,
                                    bottom=0.01, top=0.99)
                                    
                plt.clf()
                plt.imshow(np.clip(sdss*6, 0, 1), interpolation='nearest', extent=imextent)
                # ax = plt.axis()
                # ok,x,y = cowcs.radec2pixelxy(ra, dec)
                # plt.plot(x-1, y-1, 'o', mec='g', mfc='none', ms=20, mew=3)
                # plt.axis(ax)
                plt.savefig('imod-%i-sdss.pdf' % j)

                datargb = rgbs[-1]
                modrgb = rgbs[-2]
                targetrgb = rgbs[0]

                mima = dict(interpolation='nearest', origin='lower',
                            extent=imextent)

                plt.clf()
                plt.imshow(np.clip(datargb, 0, 1), **mima)
                ax = plt.axis()
                x,y,w = wsrcs
                plt.plot(x-1, y-1, 'x', mec='r', mfc='none', ms=20, mew=3)

                # plt.plot(ix0, iy0, 'r+')
                # plt.plot(ix0 + 2*S, iy0 + 2*S, 'r+')

                plt.axis(ax)
                plt.savefig('imod-%i-data.pdf' % j)

                plt.clf()
                plt.imshow(np.clip(modrgb, 0, 1), **mima)
                ax = plt.axis()
                ok,x,y = cowcs.radec2pixelxy(ra, dec)
                plt.plot(x-1, y-1, 'o', mec='g', mfc='none', ms=20, mew=3)
                x2,y2,ww = phot
                I = np.flatnonzero((x2-x)**2 + (y2-y)**2 > 1.)
                plt.plot(x2[I]-1, y2[I]-1, 'g+', ms=20, mew=3)
                plt.axis(ax)
                plt.savefig('imod-%i-mod.pdf' % j)

                plt.clf()
                plt.imshow(np.clip(targetrgb, 0, 1), **mima)
                #ax = plt.axis()
                #ok,x,y = cowcs.radec2pixelxy(ra, dec)
                #plt.plot(x-1, y-1, 'o', mec='b', mfc='none', ms=20, mew=3)
                #plt.axis(ax)
                plt.savefig('imod-%i-target.pdf' % j)

                # fn = 'imod-%i-%02i.png' % (j, ii)
                # plt.savefig(fn)
                # print 'wrote', fn




                imd = ima.copy()
                imd.update(vmin=-50, vmax=50)
                imchi = ima.copy()
                imchi.update(vmin=-5, vmax=5)

                plt.figure(figsize=(8,8))

                plt.clf()
                plt.subplot(3,3,1)
                plt.imshow(np.clip(sdss*3, 0, 1))

                for k,im in enumerate(imgs):
                    plt.subplot(3,3,2+k)
                    plt.imshow(im, **ima)
                    if wsrcs is not None:
                        ax = plt.axis()
                        x,y,ww = wsrcs
                        plt.plot(x-1, y-1, 'o', mec='r', mfc='none')
                        w = ww[k]
                        for x,y,m in zip(x,y,w):
                            if not np.isfinite(m):
                                continue
                            plt.text(x-1, y-1, '%.1f' % m, color='r')
                        plt.axis(ax)
                plt.subplot(3,3,3)
                plt.imshow(imgs[1], **ima)
                for k,m in enumerate(mods):
                    print 'mod', m.shape
                    plt.subplot(3,3,5+k)
                    plt.imshow(m, **ima)
                    ax = plt.axis()
                    ok,x,y = cowcs.radec2pixelxy(ra, dec)
                    plt.plot(x-1, y-1, 'o', mec='b', mfc='none')
                    if phot is not None:
                        x,y,ww = phot
                        plt.plot(x-1, y-1, 'b+')
                        w = ww[k]
                        for x,y,m in zip(x,y,w):
                            if not np.isfinite(m):
                                continue
                            plt.text(x-1, y-1, '%.1f' % m, color='g', rotation=90)
                    plt.axis(ax)

                    if phot is not None:
                        for i,(sp,m2) in enumerate(zip([4, 7], mod2)):
                            plt.subplot(3,3,sp)
                            plt.imshow(imgs[i] - m2, **imd)
                            ax = plt.axis()
                            ok,x,y = cowcs.radec2pixelxy(ra, dec)
                            plt.plot(x-1, y-1, 'o', mec='b', mfc='none')
                            plt.axis(ax)
                        
                for k,m in enumerate(chis):
                    plt.subplot(3,3,8+k)
                    plt.imshow(m, **imchi)
                    ax = plt.axis()
                    ok,x,y = cowcs.radec2pixelxy(ra, dec)
                    plt.plot(x-1, y-1, 'o', mec='b', mfc='none')
                    if phot is not None:
                        x,y,ww = phot
                        plt.plot(x-1, y-1, 'b+')
                        #w = ww[k]
                        #for x,y,m in zip(x,y,w):
                        #    plt.text(x-1, y-1, '%.1f' % m, color='g')
                    plt.axis(ax)
                plt.savefig('both-%i.png' % j)
            


        
        # ares = []
        # for j,i in enumerate(I):
        #     ra,dec = A.ra[i], A.dec[i]
        #     print '<img src="http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx?ra=%f&dec=%f&scale=0.2&width=200&height=200&opt=G"><br />' % (ra,dec)
        # 
        #     # plots zoom-in
        #     #ra,dec = 214.770, 53.047
        #     ddec = 0.01
        #     dra = ddec / np.cos(np.deg2rad(dec))
        #     rlo,rhi =  ra -  dra,  ra +  dra
        #     dlo,dhi = dec - ddec, dec + ddec
        # 
        #     opt.name = 'w3-target-%02i' % j
        #     opt.picklepat = opt.name + '-stage%0i.pickle'
        #     opt.ps = opt.name
        #     #ar = mp.apply(runtostage, (opt.stage, opt, mp, rlo,rhi,dlo,dhi), kwargs=dict(rcf=(A.run[i],A.camcol[i],A.field[i])))
        #     ar = mp.apply(runtostage, (opt.stage, opt, None, rlo,rhi,dlo,dhi), kwargs=dict(rcf=(A.run[i],A.camcol[i],A.field[i])))
        #     ares.append(ar)
        # for ar in ares:
        #     ar.get()

        # nice one!
        good = 10

        j,i = good,I[good]
        ra,dec = A.ra[i], A.dec[i]
        print '<img src="http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx?ra=%f&dec=%f&scale=0.2&width=200&height=200&opt=G"><br />' % (ra,dec)
        ddec = 0.007
        dra = ddec / np.cos(np.deg2rad(dec))
        rlo,rhi =  ra -  dra,  ra +  dra
        dlo,dhi = dec - ddec, dec + ddec
        rcf = (A.run[i],A.camcol[i],A.field[i])
        
    elif ri == -1:
        rlo,rhi = 218.035321632, 218.059043959
        dlo,dhi =  53.8245423746, 53.8385423746
        rcf = (3712, 6, 214)
        j = 10
        ra  = (rlo + rhi) / 2.
        dec = (dlo + dhi) / 2.
        
    if ri == -1:
        print 'rlo,rhi', rlo,rhi
        print 'dlo,dhi', dlo,dhi
        print 'R,C,F', (A.run[i],A.camcol[i],A.field[i])
        
        #opt.name = 'w3-target-%02i-w1' % j
        opt.name = 'w3-target-%02i-w%i' % (j, opt.bandnum)
        opt.picklepat = opt.name + '-stage%0i.pickle'
        opt.ps = opt.name
        runtostage(opt.stage, opt, mp, rlo,rhi,dlo,dhi, rcf=rcf,
                   targetrd=(ra,dec))

        # opt.bandnum = 2
        # opt.name = 'w3-target-%02i-w2' % j
        # opt.picklepat = opt.name + '-stage%0i.pickle'
        # opt.ps = opt.name
        # runtostage(opt.stage, opt, mp, rlo,rhi,dlo,dhi, rcf=(A.run[i],A.camcol[i],A.field[i]))

        sys.exit(0)

    elif ri == -2:
        # streak in 11317b060
        ra,dec = 215.208333, 51.171111
        ddec = 0.06
        dra = ddec / np.cos(np.deg2rad(dec))
        rlo,rhi =  ra -  dra,  ra +  dra
        dlo,dhi = dec - ddec, dec + ddec
        
        
    else:
        rlo,rhi = rr[ri],rr[ri+1]
        dlo,dhi = dd[di],dd[di+1]

    
    runtostage(opt.stage, opt, mp, rlo,rhi,dlo,dhi)



def runtostage(stage, opt, mp, rlo,rhi,dlo,dhi, **kwa):

    class MyCaller(CallGlobal):
        def getkwargs(self, stage, **kwargs):
            kwa = self.kwargs.copy()
            kwa.update(kwargs)
            if opt.ps is not None:
                kwa.update(ps = PlotSequence(opt.ps + '-s%i' % stage, format='%03i'))
            return kwa

    prereqs = { 
        100: None,
        204: 103,
        205: 104,
        304: 103,

        402: 108,
        509: 108,
        700: 106,
        }

    runner = MyCaller('stage%i', globals(), opt=opt, mp=mp,
                      declo=dlo, dechi=dhi, ralo=rlo, rahi=rhi,
                      ri=opt.ri, di=opt.di, **kwa)

    R = runstage(stage, opt.picklepat, runner, force=opt.force, prereqs=prereqs,
                 write=opt.write)
    return R

if __name__ == '__main__':
    main()
    
