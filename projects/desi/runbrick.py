import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import tempfile
import os

import fitsio

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing

from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.miscutils import clip_polygon
from astrometry.util.resample import resample_with_wcs
from astrometry.libkd.spherematch import match_radec

# from astrometry.util.file import *
# from astrometry.util.util import *
# from astrometry.util.starutil_numpy import *

from astrometry.util.ttime import Time, MemMeas
from astrometry.sdss.fields import read_photoobjs_in_wcs
from astrometry.sdss import DR9

from tractor import *
from tractor.galaxy import *
from tractor.source_extractor import *
from tractor.sdss import get_tractor_sources_dr9

from common import *

mp = None

photoobjdir = 'photoObjs-new'

def get_rgb(imgs, bands, mnmx=None, arcsinh=None):
    '''
    Given a list of images in the given bands, returns a scaled RGB
    image.
    '''
    # for now...
    assert(''.join(bands) == 'grz')

    scales = dict(g = (2, 0.0066),
                  r = (1, 0.01),
                  z = (0, 0.025),
                  )
    h,w = imgs[0].shape
    rgb = np.zeros((h,w,3), np.float32)
    # Convert to ~ sigmas
    for im,band in zip(imgs, bands):
        plane,scale = scales[band]
        rgb[:,:,plane] = (im / scale).astype(np.float32)

        print 'rgb: plane', plane, 'range', rgb[:,:,plane].min(), rgb[:,:,plane].max()

    if mnmx is None:
        mn,mx = -3, 10
    else:
        mn,mx = mnmx

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        rgb = nlmap(rgb)
        mn = nlmap(mn)
        mx = nlmap(mx)

    rgb = (rgb - mn) / (mx - mn)
    return np.clip(rgb, 0., 1.)
    

def set_globals():
    global imx
    global imchi
    
    plt.figure(figsize=(12,9));
    #plt.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.95,
    #                    hspace=0.05, wspace=0.05)

    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        #hspace=0.05, wspace=0.05)
                        hspace=0.2, wspace=0.05)
    imx = dict(interpolation='nearest', origin='lower')
    imchi = dict(interpolation='nearest', origin='lower', cmap='RdBu',
                vmin=-5, vmax=5)

def check_photometric_calib(ims, cat, ps):
    # Check photometric calibrations
    lastband = None

    for im in ims:
        band = im.band
        cat = fits_table(im.morphfn, hdu=2, columns=[
            'mag_psf','x_image', 'y_image', 'mag_disk', 'mag_spheroid', 'flags',
            'flux_psf' ])
        print 'Read', len(cat), 'from', im.morphfn
        if len(cat) == 0:
            continue
        cat.cut(cat.flags == 0)
        print '  Cut to', len(cat), 'with no flags set'
        if len(cat) == 0:
            continue
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)

        sdss = fits_table(im.sdssfn)


        I = np.flatnonzero(ZP.expnum == im.expnum)
        if len(I) > 1:
            I = np.flatnonzero((ZP.expnum == im.expnum) * (ZP.extname == im.extname))
        assert(len(I) == 1)
        I = I[0]
        magzp = ZP.zpt[I]
        print 'magzp', magzp
        exptime = ZP.exptime[I]
        magzp += 2.5 * np.log10(exptime)
        print 'magzp', magzp

        primhdr = im.read_image_primary_header()
        magzp0  = primhdr['MAGZERO']
        print 'header magzp:', magzp0

        I,J,d = match_radec(cat.ra, cat.dec, sdss.ra, sdss.dec, 1./3600.)

        flux = sdss.get('%s_psfflux' % band)
        mag = NanoMaggies.nanomaggiesToMag(flux)

        # plt.clf()
        # plt.plot(mag[J], cat.mag_psf[I] - mag[J], 'b.')
        # plt.xlabel('SDSS %s psf mag' % band)
        # plt.ylabel('SDSS - DECam mag')
        # plt.title(im.name)
        # plt.axhline(0, color='k', alpha=0.5)
        # plt.ylim(-2,2)
        # plt.xlim(15, 23)
        # ps.savefig()

        if band != lastband:
            if lastband is not None:
                ps.savefig()
            off = 0
            plt.clf()

        if off >= 8:
            continue

        plt.subplot(2,4, off+1)
        mag2 = -2.5 * np.log10(cat.flux_psf)
        p = plt.plot(mag[J], mag[J] - mag2[I], 'b.')
        plt.xlabel('SDSS %s psf mag' % band)
        if off in [0,4]:
            plt.ylabel('SDSS - DECam instrumental mag')
        plt.title(im.name)

        med = np.median(mag[J] - mag2[I])
        plt.axhline(med, color='k', alpha=0.25)

        plt.ylim(29,32)
        plt.xlim(15, 22)
        plt.axhline(magzp, color='r', alpha=0.5)
        plt.axhline(magzp0, color='b', alpha=0.5)

        off += 1
        lastband = band
    ps.savefig()

def get_se_sources(ims, catband, targetwcs, W, H):
    # FIXME -- we're only reading 'catband'-band catalogs, and all the fluxes
    # are initialized at that band's flux... should really read all bands!
        
    # Select SE catalogs to read
    catims = [im for im in ims if im.band == catband]
    print 'Reference catalog files:', catims
    # ... and read 'em
    cats = []
    extra_cols = []
    for im in catims:
        cat = fits_table(
            im.morphfn, hdu=2,
            columns=[x.upper() for x in
                     ['x_image', 'y_image', 'flags',
                      'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
                      'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
                      'disk_theta_world', 'spheroid_reff_world',
                      'spheroid_aspect_world', 'spheroid_theta_world',
                      'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
        print 'Read', len(cat), 'from', im.morphfn
        cat.cut(cat.flags == 0)
        print '  Cut to', len(cat), 'with no flags set'
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)
        cats.append(cat)
        
    # Plot all catalog sources and ROI
    # plt.clf()
    # for cat in cats:
    #     plt.plot(cat.ra, cat.dec, 'o', mec='none', mfc='b', alpha=0.5)
    # plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    # ps.savefig()
    # Cut catalogs to ROI
    for cat in cats:
        ok,x,y = targetwcs.radec2pixelxy(cat.ra, cat.dec)
        cat.cut((x > 0.5) * (x < (W+0.5)) * (y > 0.5) * (y < (H+0.5)))

    # Merge catalogs by keeping sources > 0.5" away from previous ones
    merged = cats[0]
    for cat in cats[1:]:
        if len(merged) == 0:
            merged = cat
            continue
        if len(cat) == 0:
            continue
        I,J,d = match_radec(merged.ra, merged.dec, cat.ra, cat.dec, 0.5/3600.)
        keep = np.ones(len(cat), bool)
        keep[J] = False
        if sum(keep):
            merged = merge_tables([merged, cat[keep]])
    
    # plt.clf()
    # plt.plot(merged.ra, merged.dec, 'o', mec='none', mfc='b', alpha=0.5)
    # plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    # ps.savefig()

    del cats
    # Create Tractor sources
    cat,isrcs = get_se_modelfit_cat(merged, maglim=90, bands=bands)
    print 'Tractor sources:', cat
    T = merged[isrcs]
    return cat, T

def get_sdss_sources(bands, targetwcs, W, H):
    # FIXME?
    margin = 0.

    sdss = DR9(basedir=photoobjdir)
    sdss.useLocalTree()

    cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
            'resolve_status', 'nchild', 'flags', 'objc_flags',
            'run','camcol','field','id',
            'psfflux', 'psfflux_ivar',
            'cmodelflux', 'cmodelflux_ivar',
            'modelflux', 'modelflux_ivar',
            'devflux', 'expflux']

    objs = read_photoobjs_in_wcs(targetwcs, margin, sdss=sdss, cols=cols)
    print 'Got', len(objs), 'photoObjs'

    srcs = get_tractor_sources_dr9(
        None, None, None, objs=objs, sdss=sdss,
        bands=bands,
        nanomaggies=True, fixedComposites=True,
        useObjcType=True,
        ellipse=EllipseESoft.fromRAbPhi)
    print 'Got', len(srcs), 'Tractor sources'

    cat = Catalog(*srcs)
    return cat, objs

def stage0(**kwargs):
    ps = PlotSequence('brick')
    t0 = tlast = Time()

    decals = Decals()

    B = decals.get_bricks()

    print 'Bricks:'
    B.about()

    # brick index...
    # One near the middle
    brickid = 377306
    # One near the edge and with little overlap
    #brickid = 380156
    ii = np.flatnonzero(B.brickid == brickid)[0]
    brick = B[ii]
    print 'Chosen brick:'
    brick.about()

    W,H = 3600,3600
    #W,H = 400,400

    bands = ['g','r','z']
    catband = 'r'

    targetwcs = wcs_for_brick(brick, W=W, H=H)
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    pixscale = targetwcs.pixel_scale()
    print 'pixscale', pixscale

    T = decals.get_ccds()
    T.cut(ccds_touching_wcs(targetwcs, T))
    print len(T), 'CCDs nearby'

    ims = []
    for band in bands:
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        for t in TT:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(t)
            ims.append(im)

    # Check that the CCDs_touching cuts are correct.
    if False:
        #from astrometry.blind.plotstuff import Plotstuff
        #plot = Plotstuff(outformat='png', size=(800,800), ra=brick.ra, dec=brick.dec,
        #                 width=pixscale*W)
        T2 = decals.get_ccds()

        T3 = T2[ccds_touching_wcs(targetwcs, T2, polygons=False)]
        T4 = T2[ccds_touching_wcs(targetwcs, T2)]
        print len(T3), 'on RA,Dec box'
        print len(T4), 'polygon'
        ccmap = dict(r='r', g='g', z='m')
        for band in bands:

            plt.clf()

            TT2 = T3[T3.filter == band]
            print len(TT2), 'in', band, 'band'
            plt.plot(TT2.ra, TT2.dec, 'o', color=ccmap[band], alpha=0.5)

            for t in TT2:
                im = DecamImage(t)

                run_calibs(im, brick.ra, brick.dec, pixscale, morph=False, se2=False,
                           psfex=False)

                wcs = im.read_wcs()
                r,d = wcs.pixelxy2radec([1,1,t.width,t.width,1], [1,t.height,t.height,1,1])
                plt.plot(r, d, '-', color=ccmap[band], alpha=0.3, lw=2)

            TT2 = T4[T4.filter == band]
            print len(TT2), 'in', band, 'band; polygon'
            plt.plot(TT2.ra, TT2.dec, 'x', color=ccmap[band], alpha=0.5, ms=15)

            for t in TT2:
                im = DecamImage(t)
                wcs = im.read_wcs()
                r,d = wcs.pixelxy2radec([1,1,t.width,t.width,1], [1,t.height,t.height,1,1])
                plt.plot(r, d, '-', color=ccmap[band], lw=1.5)

            TT2.about()

            plt.plot(brick.ra, brick.dec, 'k.')
            plt.plot(targetrd[:,0], targetrd[:,1], 'k-')
            plt.xlabel('RA')
            plt.ylabel('Dec')
            ps.savefig()
        sys.exit(0)


    print 'Finding images touching brick:', Time()-tlast
    tlast = Time()

    args = []
    for im in ims:
        if mp is not None:
            args.append((im, brick.ra, brick.dec, pixscale))
        else:
            run_calibs(im, brick.ra, brick.dec, pixscale)
    if mp is not None:
        mp.map(bounce_run_calibs, args)

    print 'Calibrations:', Time()-tlast
    tlast = Time()

    #check_photometric_calib(ims, cat, ps)
    #cat,T = get_se_sources(ims, catband, targetwcs, W, H)

    cat,T = get_sdss_sources(bands, targetwcs, W, H)

    print 'SDSS sources:', Time()-tlast
    tlast = Time()

    # record coordinates in target brick image
    ok,T.tx,T.ty = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.tx -= 1
    T.ty -= 1
    T.itx = np.clip(np.round(T.tx).astype(int), 0, W-1)
    T.ity = np.clip(np.round(T.ty).astype(int), 0, H-1)

    nstars = sum([1 for src in cat if isinstance(src, PointSource)])
    print 'Number of point sources:', nstars

    #T.about()
    # for c in T.get_columns():
    #     plt.clf()
    #     plt.hist(T.get(c), 50)
    #     plt.xlabel(c)
    #     ps.savefig()

    # Read images, clip to ROI
    tims = []
    for im in ims:
        band = im.band
        wcs = im.read_wcs()
        imh,imw = wcs.imageh,wcs.imagew
        imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
        ok,tx,ty = wcs.radec2pixelxy(targetrd[:-1,0], targetrd[:-1,1])
        tpoly = zip(tx,ty)
        clip = clip_polygon(imgpoly, tpoly)
        clip = np.array(clip)
        #print 'Clip', clip
        if len(clip) == 0:
            continue
        x0,y0 = np.floor(clip.min(axis=0)).astype(int)
        x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
        slc = slice(y0,y1+1), slice(x0,x1+1)

        ## FIXME -- it seems I got lucky and the cross product is
        ## negative == clockwise, as required by clip_polygon. One
        ## could check this and reverse the polygon vertex order.
        # dx0,dy0 = tx[1]-tx[0], ty[1]-ty[0]
        # dx1,dy1 = tx[2]-tx[1], ty[2]-ty[1]
        # cross = dx0*dy1 - dx1*dy0
        # print 'Cross:', cross

        img,imghdr = im.read_image(header=True, slice=slc)
        invvar = im.read_invvar(slice=slc)
        #print 'Image ', img.shape

        # header 'FWHM' is in pixels
        psf_fwhm = imghdr['FWHM']
        primhdr = im.read_image_primary_header()

        magzp = decals.get_zeropoint_for(im)
        print 'magzp', magzp
        zpscale = NanoMaggies.zeropointToScale(magzp)
        #print 'zpscale', zpscale

        medsky = np.median(img)
        img -= medsky

        # Scale images to Nanomaggies
        img /= zpscale
        invvar *= zpscale**2
        orig_zpscale = zpscale
        zpscale = 1.
        sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))

        # Clamp near-zero (incl negative!) invvars to zero
        thresh = 0.2 * (1./sig1**2)
        invvar[invvar < thresh] = 0

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        # get full image size for PsfEx
        info = im.get_image_info()
        fullh,fullw = info['dims']
        psfex = PsfEx(im.psffn, fullw, fullh, scale=False, nx=9, ny=17)
        #psfex = ShiftedPsf(psfex, x0, y0)
        # HACK -- highly approximate PSF here!
        psf_sigma = psf_fwhm / 2.35
        psf = NCircularGaussianPSF([psf_sigma],[1.])

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=ConstantSky(0.), name=im.name + ' ' + band)
        tim.zr = [-3. * sig1, 10. * sig1]
        tim.sig1 = sig1
        tim.band = band
        tim.psf_fwhm = psf_fwhm
        tim.psf_sigma = psf_sigma
        tim.sip_wcs = wcs
        tim.x0,tim.y0 = int(x0),int(y0)
        tim.psfex = psfex
        mn,mx = tim.zr
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        tims.append(tim)

        # tractor = Tractor([tim], cat)
        # plt.clf()
        # plt.subplot(1,2,1)
        # plt.imshow(tim.getImage(), **ima)
        # mod = tractor.getModelImage(tim)
        # plt.subplot(1,2,2)
        # plt.imshow(mod, **ima)
        # plt.suptitle(tim.name)
        # ps.savefig()
        # plt.clf()
        # plt.imshow(invvar, interpolation='nearest', origin='lower')
        # plt.colorbar()
        # plt.title('weight map: ' + im.name)
        # ps.savefig()
        # 
        # plt.clf()
        # plt.hist(invvar.ravel(), 100)
        # plt.xlabel('invvar')
        # ps.savefig()

        # plt.clf()
        # plt.imshow(tim.getImage(), **ima)
        # plt.suptitle(tim.name)
        # ps.savefig()

    print 'Read images:', Time()-tlast
    tlast = Time()

    # save resampling params
    for tim in tims:
        wcs = tim.sip_wcs
        x0,y0 = int(tim.x0),int(tim.y0)
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(x0, y0, subw, subh)
        tim.subwcs = subwcs
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue
        tim.resamp = (Yo,Xo,Yi,Xi)

    print 'Computing resampling:', Time()-tlast
    tlast = Time()

    # Produce per-band coadds, for plots
    coimgs = []
    cons = []
    for ib,band in enumerate(bands):
        coimg = np.zeros((H,W), np.float32)
        con   = np.zeros((H,W), np.uint8)
        for tim in tims:
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            nn = (tim.getInvvar()[Yi,Xi] > 0)
            coimg[Yo,Xo] += tim.getImage ()[Yi,Xi] * nn
            con  [Yo,Xo] += nn
        coimg /= np.maximum(con,1)
        coimgs.append(coimg)
        cons  .append(con)

    print 'Coadds:', Time()-tlast
    tlast = Time()

    # Render the detection maps
    detmaps = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    detivs  = dict([(b, np.zeros((H,W), np.float32)) for b in bands])
    for tim in tims:
        iv = tim.getInvvar()
        psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
        detim = tim.getImage().copy()
        detim[iv == 0] = 0.
        detim = gaussian_filter(detim, tim.psf_sigma) / psfnorm**2
        detsig1 = tim.sig1 / psfnorm
        subh,subw = tim.shape
        detiv = np.zeros((subh,subw), np.float32) + (1. / detsig1**2)
        detiv[iv == 0] = 0.
        (Yo,Xo,Yi,Xi) = tim.resamp
        detmaps[tim.band][Yo,Xo] += detiv[Yi,Xi] * detim[Yi,Xi]
        detivs [tim.band][Yo,Xo] += detiv[Yi,Xi]

    print 'Detmaps:', Time()-tlast
    tlast = Time()

    # -find significant peaks in the per-band detection maps and SED-matched (hot)
    # -segment into blobs
    # -blank out blobs containing a catalog source
    # -create sources for any remaining peaks
    hot = np.zeros((H,W), bool)
    sedmap = np.zeros((H,W), np.float32)
    sediv  = np.zeros((H,W), np.float32)
    for band in bands:
        detmap = detmaps[band] / np.maximum(1e-16, detivs[band])
        detsn = detmap * np.sqrt(detivs[band])
        hot |= (detsn > 5.)
        sedmap += detmaps[band]
        sediv  += detivs [band]
        detmaps[band] = detmap
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
    hot |= (sedsn > 5.)
    peaks = hot.copy()

    plt.clf()
    dimshow(np.round(sedsn), vmin=0, vmax=10, cmap='hot')
    plt.title('SED-matched detection filter (flat SED)')
    ps.savefig()

    crossa = dict(ms=10, mew=1.5)
    plt.clf()
    dimshow(peaks)
    ax = plt.axis()
    plt.plot(T.itx, T.ity, 'r+', **crossa)
    plt.axis(ax)
    plt.title('Detection blobs')
    ps.savefig()
    
    blobs,nblobs = label(hot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    for x,y in zip(T.itx, T.ity):
        # blob number
        bb = blobs[y,x]
        if bb == 0:
            continue
        # un-set 'peaks' within this blob
        slc = blobslices[bb-1]
        peaks[slc][blobs[slc] == bb] = 0

    plt.clf()
    dimshow(peaks)
    ax = plt.axis()
    plt.plot(T.itx, T.ity, 'r+', **crossa)
    plt.axis(ax)
    plt.title('Detection blobs minus catalog sources')
    ps.savefig()
        
    # zero out the edges(?)
    peaks[0 ,:] = peaks[:, 0] = 0
    peaks[-1,:] = peaks[:,-1] = 0
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (sedsn[1:-1,1:-1] >= sedsn[1:-1,2:  ])
    pki = np.flatnonzero(peaks)
    peaky,peakx = np.unravel_index(pki, peaks.shape)
    print len(peaky), 'peaks'

    print 'Peaks:', Time()-tlast
    tlast = Time()

    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    ax = plt.axis()
    plt.plot(T.tx, T.ty, 'r+', **crossa)
    plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
    plt.axis(ax)
    plt.title('Catalog + SED-matched detections')
    ps.savefig()
    
    # Grow the 'hot' pixels by dilating by a few pixels
    rr = 2.0
    RR = int(np.ceil(rr))
    S = 2*RR+1
    struc = (((np.arange(S)-RR)**2)[:,np.newaxis] +
             ((np.arange(S)-RR)**2)[np.newaxis,:]) <= rr**2
    hot = binary_dilation(hot, structure=struc)
    #iterations=int(np.ceil(2. * psf_sigma)))

    # Add sources for the new peaks we found
    # make their initial fluxes ~ 5-sigma
    fluxes = dict([(b,[]) for b in bands])
    for tim in tims:
        psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
        fluxes[tim.band].append(5. * tim.sig1 / psfnorm)
    fluxes = dict([(b, np.mean(fluxes[b])) for b in bands])
    pr,pd = targetwcs.pixelxy2radec(peakx+1, peaky+1)
    print 'Adding', len(pr), 'new sources'
    # Also create FITS table for new sources
    Tnew = fits_table()
    Tnew.ra  = pr
    Tnew.dec = pd
    Tnew.tx = peakx
    Tnew.ty = peaky
    Tnew.itx = np.clip(np.round(Tnew.tx).astype(int), 0, W-1)
    Tnew.ity = np.clip(np.round(Tnew.ty).astype(int), 0, H-1)
    for i,(r,d,x,y) in enumerate(zip(pr,pd,peakx,peaky)):
        cat.append(PointSource(RaDecPos(r,d),
                               NanoMaggies(order=bands, **fluxes)))

    print 'Existing source table:'
    T.about()
    print 'New source table:'
    Tnew.about()

    T = merge_tables([T, Tnew], columns='fillzero')

    # Segment, and record which sources fall into each blob
    blobs,nblobs = label(hot)
    print 'N detected blobs:', nblobs
    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]
    blobsrcs = []
    blobflux = []
    for blob in range(1, nblobs+1):
        blobsrcs.append(np.flatnonzero(T.blob == blob))
        bslc = blobslices[blob-1]
        # not really 'flux' per se...
        blobflux.append(np.sum(sedsn[bslc][blobs[bslc] == blob]))

    print 'Segmentation:', Time()-tlast
    tlast = Time()

    if False:
        plt.clf()
        plt.imshow(hot, cmap='gray', **imx)
        plt.title('Segmentation')
        ps.savefig()

    cat.freezeAllParams()
    tractor = Tractor(tims, cat)
    tractor.freezeParam('images')
    
    rtn = dict()
    for k in ['T', 'sedsn', 'coimgs', 'cons', 'detmaps', 'detivs',
              'nblobs','blobsrcs','blobflux','blobslices', 'blobs',
              'tractor', 'cat', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
              'bands', 'tims', 'ps']:
        rtn[k] = locals()[k]
    return rtn

# Check out the PsfEx models
def stage101(T=None, sedsn=None, coimgs=None, con=None, coimas=None,
             detmaps=None, detivs=None,
             rgbim=None,
             nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
             tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None, tims=None,
             **kwargs):
    # sort sources by their sedsn values.
    fluxes = sedsn[T.ity, T.itx]

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]

    for srci in np.argsort(-fluxes)[:20]:
        cat.freezeAllParams()
        cat.thawParam(srci)
                    
        print 'Fitting:'
        tractor.printThawedParams()
        for itim,tim in enumerate(tims):
            ox0,oy0 = orig_wcsxy0[itim]
            x,y = tim.wcs.positionToPixel(cat[srci].getPosition())
            psfimg = tim.psfex.instantiateAt(ox0+x, oy0+y, nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg)
            tim.psf = subpsf

        for step in range(10):
            dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break
        
        chis1 = tractor.getChiImages()
        mods1 = tractor.getModelImages()


        for itim,tim in enumerate(tims):
            ox0,oy0 = orig_wcsxy0[itim]
            x,y = tim.wcs.positionToPixel(cat[srci].getPosition())
            psfimg = tim.psfex.instantiateAt(ox0+x, oy0+y, nativeScale=True)
            subpsf = PixelizedPSF(psfimg)
            tim.psf = subpsf
        for step in range(10):
            dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break
        
        chis2 = tractor.getChiImages()
        mods2 = tractor.getModelImages()

        
        subchis = []
        submods = []
        subchis2 = []
        submods2 = []
        subimgs = []
        for i,(chi,mod) in enumerate(zip(chis1, mods1)):
            x,y = tims[i].wcs.positionToPixel(cat[srci].getPosition())
            x = int(x)
            y = int(y)
            S = 15
            th,tw = tims[i].shape
            x0 = max(x-S, 0)
            y0 = max(y-S, 0)
            x1 = min(x+S, tw)
            y1 = min(y+S, th)
            subchis.append(chi[y0:y1, x0:x1])
            submods.append(mod[y0:y1, x0:x1])
            subimgs.append(tims[i].getImage()[y0:y1, x0:x1])
            subchis2.append(chis2[i][y0:y1, x0:x1])
            submods2.append(mods2[i][y0:y1, x0:x1])

        mxchi = max([np.abs(chi).max() for chi in subchis])

        # n = len(subchis)
        # cols = int(np.ceil(np.sqrt(n)))
        # rows = int(np.ceil(float(n) / cols))
        # plt.clf()
        # for i,chi in enumerate(subchis):
        #     plt.subplot(rows, cols, i+1)
        #     plt.imshow(-chi, vmin=-mxchi, vmax=mxchi, cmap='RdBu', **imx)
        #     plt.colorbar()
        # ps.savefig()

        cols = len(subchis)
        rows = 3
        rows = 5
        plt.clf()
        ta = dict(fontsize=8)
        for i,(chi,mod,img) in enumerate(zip(subchis,submods,subimgs)):
            mx = img.max()
            def nl(x):
                return np.log10(np.maximum(tim.sig1, x + 5.*tim.sig1))

            plt.subplot(rows, cols, i+1)
            plt.imshow(nl(img), vmin=nl(0), vmax=nl(mx), **imx)
            plt.xticks([]); plt.yticks([])
            plt.title(tims[i].name, **ta)

            plt.subplot(rows, cols, i+1+cols)
            plt.imshow(nl(mod), vmin=nl(0), vmax=nl(mx), **imx)
            plt.xticks([]); plt.yticks([])
            if i == 0:
                plt.title('MoG PSF', **ta)

            plt.subplot(rows, cols, i+1+cols*2)
            mxchi = 5.
            plt.imshow(-chi, vmin=-mxchi, vmax=mxchi, cmap='RdBu', **imx)
            plt.xticks([]); plt.yticks([])
            #plt.colorbar()
            if i == 0:
                plt.title('MoG chi', **ta)

            # pix
            plt.subplot(rows, cols, i+1+cols*3)
            plt.imshow(nl(submods2[i]), vmin=nl(0), vmax=nl(mx), **imx)
            plt.xticks([]); plt.yticks([])
            if i == 0:
                plt.title('Pixelized PSF', **ta)

            plt.subplot(rows, cols, i+1+cols*4)
            mxchi = 5.
            plt.imshow(-subchis2[i], vmin=-mxchi, vmax=mxchi, cmap='RdBu', **imx)
            plt.xticks([]); plt.yticks([])
            if i == 0:
                plt.title('Pixelized chi', **ta)

        rd = cat[srci].getPosition()
        plt.suptitle('Source at RA,Dec = (%.4f, %.4f)' % (rd.ra, rd.dec))
            
        ps.savefig()


def _plot_mods(tims, mods, titles, bands, coimgs, cons, bslc, blobw, blobh, ps):
    subims = [[] for m in mods]
    chis = dict([(b,[]) for b in bands])
    for iband,band in enumerate(bands):
        comods = [np.zeros((blobh,blobw)) for m in mods]
        cochis = [np.zeros((blobh,blobw)) for m in mods]
        comodn = np.zeros((blobh,blobw))

        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            rechi = np.zeros((blobh,blobw))
            chilist = []
            comodn[Yo,Xo] += 1
            for imod,mod in enumerate(mods):
                chi = ((tim.getImage()[Yi,Xi] - mod[itim][Yi,Xi]) *
                       tim.getInvError()[Yi,Xi])
                rechi[Yo,Xo] = chi
                chilist.append(rechi.copy())
                cochis[imod][Yo,Xo] += chi
                comods[imod][Yo,Xo] += mod[itim][Yi,Xi]
            chis[band].append(chilist)
            mn,mx = -10.*tim.sig1, 30.*tim.sig1

        for comod in comods:
            comod /= np.maximum(comodn, 1)
        ima = dict(vmin=mn, vmax=mx)
        coimg = coimgs[iband][bslc]
        coimgn = cons[iband][bslc]
        for subim,comod,cochi in zip(subims, comods, cochis):
            subim.append((coimg, coimgn, comod, ima, cochi))

    # Plot per-band image, model, and chi coadds, and RGB images
    for i,subim in enumerate(subims):
        plt.clf()
        rows,cols = 3,5
        imgs = []
        mods = []
        resids = []
        for j,(img,imgn,mod,ima,chi) in enumerate(subim):
            imgs.append(img)
            mods.append(mod)
            resid = img - mod
            resid[imgn == 0] = np.nan
            resids.append(resid)
            plt.subplot(rows,cols,1 + j + 0)
            dimshow(img, **ima)
            plt.subplot(rows,cols,1 + j + cols)
            dimshow(mod, **ima)
            plt.subplot(rows,cols,1 + j + cols*2)
            #dimshow(-chi, **imchi)
            #dimshow(imgn, vmin=0, vmax=3)
            dimshow(resid, nancolor='r')
        plt.subplot(rows,cols, 4)
        dimshow(get_rgb(imgs, bands))
        plt.subplot(rows,cols, cols+4)
        dimshow(get_rgb(mods, bands))
        plt.subplot(rows,cols, cols*2+4)
        dimshow(get_rgb(resids, bands, mnmx=(-10,10)))

        mnmx = -5,300
        kwa = dict(mnmx=mnmx, arcsinh=1)
        plt.subplot(rows,cols, 5)
        print
        print 'RGB image'
        dimshow(get_rgb(imgs, bands, **kwa))
        plt.subplot(rows,cols, cols+5)
        print
        print 'RGB model'
        dimshow(get_rgb(mods, bands, **kwa))
        print
        plt.subplot(rows,cols, cols*2+5)
        mnmx = -100,100
        kwa = dict(mnmx=mnmx, arcsinh=1)
        dimshow(get_rgb(resids, bands, **kwa))
        plt.suptitle(titles[i])
        ps.savefig()

    # Plot per-image chis
    cols = max(len(v) for v in chis.values())
    rows = len(bands)
    for i in range(len(mods)):
        plt.clf()
        for row,band in enumerate(bands):
            sp0 = 1 + cols*row
            for col,cc in enumerate(chis[band]):
                chi = cc[i]
                plt.subplot(rows, cols, sp0 + col)
                plt.imshow(-chi, **imchi)
        plt.suptitle(titles[i])
        ps.savefig()


def stage1(T=None, sedsn=None, coimgs=None, cons=None,
           detmaps=None, detivs=None,
           nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
           tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
           W=None,H=None,
           bands=None, ps=None, tims=None,
           plots=False,
           **kwargs):

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]

    tlast = Time()

    # Fit in order of flux
    for blobnumber,iblob in enumerate(np.argsort(-np.array(blobflux))):
        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        if len(Isrcs) == 0:
            continue

        print
        print 'Blob', blobnumber, 'of', len(blobflux), ':', len(Isrcs), 'sources'
        print

        # blob bbox in target coords
        sy,sx = bslc
        y0,y1 = sy.start, sy.stop
        x0,x1 = sx.start, sx.stop
        blobh,blobw = y1 - y0, x1 - x0

        rr,dd = targetwcs.pixelxy2radec([x0,x0,x1,x1],[y0,y1,y1,y0])

        ###
        # FIXME -- We create sub-images for each blob here.
        # What we don't do, though, is mask out the invvar pixels
        # that are within the blob bounding-box but not within the
        # blob itself.  Does this matter?
        ###

        alphas = [0.1, 0.3, 1.0]
        
        subtims = []
        for itim,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            if sx1 < 0 or sy1 < 0 or sx1 > w or sy1 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            #print 'image subregion', sx0,sx1,sy0,sy1

            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subiv  = tim.getInvvar()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[itim]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            # FIXME --
            #subpsf = tim.psfex.mogAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.)
            #subpsf = tim.getPsf()

            psfimg = tim.psfex.instantiateAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.,
                                             nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg)

            subtim = Image(data=subimg, invvar=subiv, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky(), name=tim.name)
            subtim.band = tim.band

            (Yo,Xo,Yi,Xi) = tim.resamp
            I = np.flatnonzero((Yi >= sy0) * (Yi < sy1) * (Xi >= sx0) * (Xi < sx1) *
                               (Yo >=  y0) * (Yo <  y1) * (Xo >=  x0) * (Xo <  x1))
            Yo = Yo[I] - y0
            Xo = Xo[I] - x0
            Yi = Yi[I] - sy0
            Xi = Xi[I] - sx0
            subtim.resamp = (Yo, Xo, Yi, Xi)
            subtim.sig1 = tim.sig1


            subtims.append(subtim)

        subtr = Tractor(subtims, cat)
        subtr.freezeParam('images')

        if plots:
            # before-n-after plots
            mod0 = subtr.getModelImages()
        print 'Sub-image initial lnlikelihood:', subtr.getLogLikelihood()

        mod3 = None

        # Optimize individual sources in order of flux
        fluxes = []
        for i in Isrcs:
            br = cat[i].getBrightness()
            # HACK -- here we just *sum* the nanomaggies in each band.  Bogus!
            flux = sum([br.getFlux(band) for band in bands])
            fluxes.append(flux)
        Ibright = Isrcs[np.argsort(-np.array(fluxes))]
        cat.freezeAllParams()
        for i in Ibright:
            tsrc = Time()
            print 'Fitting source', i
            print cat[i]
            cat.freezeAllBut(i)
            print 'Optimizing:', subtr
            subtr.printThawedParams()
            for step in range(5):
                dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                print 'dlnp:', dlnp
                if dlnp < 0.1:
                    break
            print 'Fitting source took', Time()-tsrc
            print cat[i]

        if plots:
            modS = subtr.getModelImages()
        print 'Sub-image individual-source fit lnlikelihood:', subtr.getLogLikelihood()

        # Optimize all at once?
        cat.freezeAllParams()
        print 'Fitting:'
        for i in Isrcs:
            cat.thawParams(i)
            print cat[i]
        print 'Optimizing:', subtr
        subtr.printThawedParams()
        
        for step in range(10):
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                          alphas=alphas)
            print 'dlnp:', dlnp
            if dlnp == 0.0 and plots and False:
                # Borked -- take the step and render the models.
                p0 = subtr.getParams()
                subtr.setParams(p0 + X)
                mod3 = [tractor.getModelImage(tim) for tim in tims]
                subtr.setParams(p0)

                derivs = subtr.getDerivs()
                for i,(paramname,derivlist) in enumerate(zip(subtr.getParamNames(), derivs)):
                    if len(derivlist) == 0:
                        continue
                    plt.clf()
                    n = len(derivlist)
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(float(n) / cols))
                    for j,(deriv,tim) in enumerate(derivlist):
                        plt.subplot(rows,cols, j+1)
                        plt.imshow(deriv.patch, cmap='RdBu', **imx)
                        plt.colorbar()
                        plt.title(tim.name)
                    plt.suptitle('Borked optimization: derivs for ' + paramname)
                    ps.savefig()

            if dlnp < 0.1:
                break

        if plots:
            mod1 = subtr.getModelImages()
        print 'Sub-image first fit lnlikelihood:', subtr.getLogLikelihood()
        #print 'First fit chi-squared:', tractor.getLogLikelihood()

        # Forced-photometer bands individually
        for band in bands:
            cat.freezeAllRecursive()
            for i in Isrcs:
                cat.thawParam(i)
                cat[i].thawPathsTo(band)
            #cat.thawPathsTo(band)
            bandtims = []
            for tim in subtims:
                if tim.band == band:
                    bandtims.append(tim)
            print
            print 'Fitting', band, 'band:'
            btractor = Tractor(bandtims, cat)
            btractor.freezeParam('images')
            btractor.printThawedParams()
            B = 8
            X = btractor.optimize_forced_photometry(shared_params=False, use_ceres=True,
                                                    BW=B, BH=B, wantims=False)
        cat.thawAllRecursive()
        #print 'Forced-phot chi-squared:', tractor.getLogLikelihood()
        print 'Forced-phot lnlikelihood:', subtr.getLogLikelihood()

        if plots:
            mod2 = subtr.getModelImages()

            if mod3 is None:
                mods = [mod0, modS, mod1, mod2]
                tt = ['initial', 'single sources', 'all sources', 'per band']
            else:
                mods = [mod0, modS, mod1, mod3, mod2]
                tt = ['initial', 'single sources', 'all sources', 'borked', 'per band']

            _plot_mods(subtims, mods, tt, bands, coimgs, cons, bslc, blobw, blobh, ps)

            if blobnumber >= 10:
                plots = False


        print 'Blob', blobnumber, 'finished:', Time()-tlast
        tlast = Time()

    rtn = dict()
    for k in ['tractor','tims','ps']:
        rtn[k] = locals()[k]
    return rtn

class BrightPointSource(PointSource):
    def _getPsf(self, img):
        return img.brightPsf
    def getSourceType(self):
        return 'BrightPointSource'

def stage2(T=None, sedsn=None, coimgs=None, cons=None,
           detmaps=None, detivs=None,
           nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
           cat=None, targetrd=None, pixscale=None, targetwcs=None,
           W=None,H=None,
           bands=None, ps=None,
           plots=False, tims=None, tractor=None,
           **kwargs):

    # For bright sources, use more MoG components, or use pixelized PSF model?
    fluxes = []
    for src in cat:
        br = src.getBrightness()
        fluxes.append([br.getFlux(b) for b in bands])
    fluxes = np.array(fluxes)

    for i,b in enumerate(bands):
        ii = np.argsort(-fluxes[:,i])
        print
        print 'Brightest in band', b
        for j in ii[:10]:
            print j, cat[j].getBrightness()


    # HACK -- define "bright" limits
    bright = dict(g = 20.5, r = 20, z = 19.5)

    ibright = []
    for band in bands:
        brightmag = bright[band]
        for i,src in enumerate(cat):
            br = src.getBrightness()
            if br.getMag(band) < brightmag:
                ibright.append(i)
    ibright = np.unique(ibright)

    print 'Bright sources:', ibright

    bcat = []
    for i,src in enumerate(cat):
        # if i in ibright:
        #     if isinstance(src, PointSource):
        #         bcat.append(BrightPointSource(src.pos, src.brightness))
        #     else:
        #         ### FIXME -- model selection??
        #         print 'Trying to replace bright source', src, 'with point source'
        #         bcat.append(BrightPointSource(src.getPosition(), src.getBrightness()))

        if i in ibright and isinstance(src, PointSource):
            bcat.append(BrightPointSource(src.pos, src.brightness))
        else:
            bcat.append(src)
    bcat = Catalog(*bcat)

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
                
    for iblob,(bslc,Isrcs) in enumerate(zip(blobslices, blobsrcs)):
        if not len(set(ibright).intersection(set(Isrcs))):
            continue
        print 'Re-fitting blob', iblob, 'with', len(Isrcs), 'sources'

        bcat.freezeAllParams()
        print 'Fitting:'
        for i in Isrcs:
            bcat.thawParams(i)
            print bcat[i]
            
        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])

        ###
        # FIXME -- We create sub-image for each blob here.
        # What wo don't do, though, is mask out the invvar pixels
        # that are within the blob bounding-box but not within the
        # blob itself.  Does this matter?
        ###

        alphas = [0.1, 0.3, 1.0]
        
        subtims = []
        for itim,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            if sx1 < 0 or sy1 < 0 or sx1 > w or sy1 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            #print 'image subregion', sx0,sx1,sy0,sy1

            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subiv  = tim.getInvvar()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[itim]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            # FIXME --
            #subpsf = tim.psfex.mogAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.)
            #subpsf = tim.getPsf()

            psfimg = tim.psfex.instantiateAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.,
                                             nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg)

            #subtim = BrightPsfImage(data=subimg, invvar=subiv, wcs=subwcs,
            subtim = Image(data=subimg, invvar=subiv, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky(), name=tim.name)
            subtim.extent = (sx0, sx1, sy0, sy1)
            subtim.band = tim.band

            (Yo,Xo,Yi,Xi) = tim.resamp
            I = np.flatnonzero((Yi >= sy0) * (Yi < sy1) * (Xi >= sx0) * (Xi < sx1) *
                               (Yo >= by0) * (Yo < by1) * (Xo >= bx0) * (Xo < bx1))
            Yo = Yo[I] - by0
            Xo = Xo[I] - bx0
            Yi = Yi[I] - sy0
            Xi = Xi[I] - sx0
            subtim.resamp = (Yo, Xo, Yi, Xi)
            subtim.sig1 = tim.sig1

            subtim.brightPsf = PixelizedPsfEx(tim.psfex, ox0 + sx0, oy0 + sy0)
            #subtim.brightPsf = PixelizedPSF(psfimg)
            #subtim.brightPsf = GaussianMixturePSF.fromStamp(psfimg, N=5)

            subtims.append(subtim)

        subtr = Tractor(subtims, bcat)
        subtr.freezeParam('images')
        print 'Optimizing:', subtr
        subtr.printThawedParams()

        if plots:
            otractor = Tractor(subtims, cat)
            modx = otractor.getModelImages()

            # before-n-after plots
            mod0 = subtr.getModelImages()
        print 'Sub-image initial lnlikelihood:', subtr.getLogLikelihood()

        for i in Isrcs:
            print bcat[i]

        for step in range(10):
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                          alphas=alphas)
            print 'dlnp:', dlnp
            if dlnp < 0.1:
                break

        if plots:
            mod1 = subtr.getModelImages()
        print 'Sub-image first fit lnlikelihood:', subtr.getLogLikelihood()

        for i in Isrcs:
            print bcat[i]

        # Forced-photometer bands individually
        for band in bands:
            bcat.freezeAllRecursive()
            for i in Isrcs:
                bcat.thawParam(i)
                bcat[i].thawPathsTo(band)
            bandtims = []
            for tim in subtims:
                if tim.band == band:
                    bandtims.append(tim)
            print
            print 'Fitting', band, 'band:'
            btractor = Tractor(bandtims, bcat)
            btractor.freezeParam('images')
            btractor.printThawedParams()
            B = 8
            X = btractor.optimize_forced_photometry(shared_params=False, use_ceres=True,
                                                    BW=B, BH=B, wantims=False)
        bcat.thawAllRecursive()
        print 'Sub-image forced-phot lnlikelihood:', subtr.getLogLikelihood()
        for i in Isrcs:
            print bcat[i]

        if plots:
            mod2 = subtr.getModelImages()

        if plots:
            mods = [modx, mod0, mod1, mod2]
            _plot_mods(subtims, mods, ['' for m in mods], bands, coimgs, cons, bslc, blobw, blobh, ps)

    rtn = dict()
    for k in ['tractor','tims', 'bcat', 'ps']:
        rtn[k] = locals()[k]
    return rtn


class PixelizedPsfEx(object):
    def __init__(self, psfex, x0, y0):
        self.psfex = psfex
        self.x0 = x0
        self.y0 = y0
        
    def getPointSourcePatch(self, px, py, minval=0., extent=None, radius=None):
        pix = self.psfex.instantiateAt(self.x0 + px, self.y0 + py, nativeScale=True)
        return PixelizedPSF(pix).getPointSourcePatch(px, py, radius=radius, extent=extent)

def stage103(T=None, sedsn=None, coimgs=None, con=None, coimas=None,
             detmaps=None, detivs=None,
             rgbim=None,
             nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
             cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None,
             plots=False, tims=None, tractor=None, bcat=None,
             **kwargs):

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    for itim,tim in enumerate(tims):
        ox0,oy0 = orig_wcsxy0[itim]
        # # HACK -- instantiate pixelized PSF at center of tim
        # r,d = targetwcs.pixelxy2radec(W/2., H/2.)
        # ok,cx,cy = tim.subwcs.radec2pixelxy(r, d)
        # psfimg = tim.psfex.instantiateAt(ox0+cx, oy0+cy, nativeScale=True)
        # tim.brightPsf = PixelizedPsfEx(psfimg)

        tim.brightPsf = PixelizedPsfEx(tim.psfex, ox0, oy0)

    cat = tractor.catalog = bcat

    print 'Sources:'
    for i,src in enumerate(cat):
        print '  ', i, src

    stage102(tractor=tractor, tims=tims, H=H, W=W, bands=bands,
             rgbim=rgbim, cat=cat, ps=ps, coimgs=coimgs, con=con,
             targetwcs=targetwcs)


def stage102(T=None, sedsn=None, coimgs=None, con=None, coimas=None,
             detmaps=None, detivs=None,
             rgbim=None,
             nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
             cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None,
             plots=False, tims=None, tractor=None,
             **kwargs):

    mod = [tractor.getModelImage(tim) for tim in tims]

    # After plot
    rgbmod = np.zeros((H,W,3))
    rgbmod2 = np.zeros((H,W,3))
    rgbresids = np.zeros((H,W,3))
    rgbresids2 = np.zeros((H,W,3))

    rgbmodsig = np.zeros((H,W,3))
    rgbimgsig = np.zeros((H,W,3))

    for iband,band in enumerate(bands):
        coimg = coimgs[iband]
        comod = np.zeros((H,W))
        comod2 = np.zeros((H,W))
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            comod[Yo,Xo] += mod[itim][Yi,Xi]
            ie = tim.getInvError()
            noise = np.random.normal(size=ie.shape) / ie
            noise[ie == 0] = 0.
            comod2[Yo,Xo] += mod[itim][Yi,Xi] + noise[Yi,Xi]
            mn,mx = tim.zr
            sig1 = tim.sig1
            mn2,mx2 = -10.*sig1, 30.*sig1
        comod  /= np.maximum(con, 1)
        comod2 /= np.maximum(con, 1)
        c = 2-iband

        rgbmodsig[:,:,c] = comod / sig1
        rgbimgsig[:,:,c] = coimg / sig1

        rgbmod[:,:,c] = np.clip((comod - mn) / (mx - mn), 0., 1.)
        rgbmod2[:,:,c] = np.clip((comod2 - mn) / (mx - mn), 0., 1.)
        rgbresids[:,:,c] = np.clip((coimg - comod - mn) / (mx - mn), 0., 1.)

        # Blank out residual pixels where there is no data
        rgbresids[:,:,c][con == 0] = np.clip((0 - mn) / (mx - mn), 0., 1.)

        rgbresids2[:,:,c] = np.clip((coimg - comod - mn2) / (mx2 - mn2), 0., 1.)
        rgbresids2[:,:,c][con == 0] = np.clip((0 - mn2) / (mx2 - mn2), 0., 1.)

    plt.clf()
    dimshow(rgbim)
    plt.title('Images')
    ps.savefig()

    ax = plt.axis()
    cat = tractor.getCatalog()
    for i,src in enumerate(cat):
        rd = src.getPosition()
        ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, '+', color=cc, ms=10, mew=1.5)
        else:
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', ms=10, mew=1.5)
        # plt.text(x, y, '%i' % i, color=cc, ha='center', va='bottom')
    plt.axis(ax)
    ps.savefig()

    plt.clf()
    dimshow(rgbmod)
    plt.title('Model')
    ps.savefig()

    plt.clf()
    dimshow(rgbmod2)
    plt.title('Model + Noise')
    ps.savefig()

    plt.clf()
    dimshow(rgbresids)
    plt.title('Residuals')
    ps.savefig()

    plt.clf()
    dimshow(rgbresids2)
    plt.title('Residuals (2)')
    ps.savefig()

    mn,mx = -20.,50.
    plt.clf()
    dimshow(np.clip((rgbimgsig - mn) / (mx - mn), 0., 1.))
    plt.title('Image (2)')
    ps.savefig()

    plt.clf()
    dimshow(np.clip((rgbmodsig - mn) / (mx - mn), 0., 1.))
    plt.title('Model (2)')
    ps.savefig()

    plt.clf()
    dimshow(np.clip((rgbimgsig - rgbmodsig - mn) / (mx - mn), 0., 1.))
    plt.title('Residuals (2)')
    ps.savefig()



if __name__ == '__main__':
    from astrometry.util.stages import *
    import optparse
    import logging
    
    parser = optparse.OptionParser()
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
                      help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_option('-s', '--stage', dest='stage', default=1, type=int,
                      help="Run up to the given stage")
    parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')
    parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
                      help='Make more verbose')
    parser.add_option('--threads', type=int, help='Run multi-threaded')
    parser.add_option('-p', '--plots', dest='plots', action='store_true',
                      help='Per-blob plots?')
    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='runbrick-s%03i.pickle')

    opt,args = parser.parse_args()

    Time.add_measurement(MemMeas)

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    if opt.threads and opt.threads > 1:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads)

    set_globals()
    stagefunc = CallGlobal('stage%i', globals())
    prereqs = {101: 0, 102: 1, 103: 2}
    opt.force.append(opt.stage)
    
    runstage(opt.stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
             prereqs=prereqs, plots=opt.plots)
    
