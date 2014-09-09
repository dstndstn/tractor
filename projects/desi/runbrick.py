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
from astrometry.util.resample import resample_with_wcs,OverlapError
from astrometry.libkd.spherematch import match_radec
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

def _print_struc(X):
    if X is None:
        print 'None',
    elif type(X) in (list,tuple):
        islist = (type(X) is list)
        if islist:
            print '[',
        else:
            print '(',
        for x in X:
            _print_struc(x)
            print ',',
        if islist:
            print ']',
        else:
            print ')',
    else:
        print type(X),

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
        #print 'rgb: plane', plane, 'range', rgb[:,:,plane].min(), rgb[:,:,plane].max()

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
    global imchi
    plt.figure(figsize=(12,9));
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        hspace=0.2, wspace=0.05)
    imchi = dict(cmap='RdBu', vmin=-5, vmax=5)

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

def stage0(W=3600, H=3600, brickid=None, **kwargs):
    ps = PlotSequence('brick')
    t0 = tlast = Time()

    decals = Decals()

    B = decals.get_bricks()

    print 'Bricks:'
    B.about()

    # brick index...
    # One near the middle
    #brickid = 377306
    # One near the edge and with little overlap
    #brickid = 380156
    ii = np.flatnonzero(B.brickid == brickid)[0]
    brick = B[ii]
    print 'Chosen brick:'
    brick.about()

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

    print 'Read images:', Time()-tlast
    tlast = Time()

    # save resampling params
    for tim in tims:
        wcs = tim.sip_wcs
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        tim.subwcs = subwcs
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
        except OverlapError:
            print 'No overlap'
            continue
        if len(Yo) == 0:
            continue
        tim.resamp = (Yo,Xo,Yi,Xi)

        # # Resampling the reverse direction
        # try:
        #     Yo,Xo,Yi,Xi,rims = resample_with_wcs(subwcs, targetwcs, [], 2)
        # except OverlapError:
        #     print 'No overlap'
        #     continue
        # if len(Yo) == 0:
        #     continue
        # tim.reverseresamp = (Yo,Xo,Yi,Xi)

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
        dimshow(hot)
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

def stage101(coimgs=None, cons=None, bands=None, ps=None, **kwargs):
    # RGB image
    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    ps.savefig()

    # cluster zoom-in
    plt.clf()
    dimshow(get_rgb(coimgs, bands)[200:1200, 1700:2700,:])
    ps.savefig()


def _plot_mods(tims, mods, titles, bands, coimgs, cons, bslc, blobw, blobh, ps,
               chi_plots=True):
    subims = [[] for m in mods]
    chis = dict([(b,[]) for b in bands])
    
    make_coimgs = (coimgs is None)
    if make_coimgs:
        coimgs = [np.zeros((blobh,blobw)) for b in bands]
        cons   = [np.zeros((blobh,blobw)) for b in bands]

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
                chilist.append((rechi.copy(), itim))
                cochis[imod][Yo,Xo] += chi
                comods[imod][Yo,Xo] += mod[itim][Yi,Xi]
            chis[band].append(chilist)
            mn,mx = -10.*tim.sig1, 30.*tim.sig1

            if make_coimgs:
                coimgs[iband][Yo,Xo] += tim.getImage()[Yi,Xi]
                cons  [iband][Yo,Xo] += 1
                
        if make_coimgs:
            coimgs[iband] /= np.maximum(cons[iband], 1)
            coimg  = coimgs[iband]
            coimgn = cons  [iband]
        else:
            coimg = coimgs[iband][bslc]
            coimgn = cons[iband][bslc]
            
        for comod in comods:
            comod /= np.maximum(comodn, 1)
        ima = dict(vmin=mn, vmax=mx)
        for subim,comod,cochi in zip(subims, comods, cochis):
            subim.append((coimg, coimgn, comod, ima, cochi))

    # Plot per-band image, model, and chi coadds, and RGB images
    for i,subim in enumerate(subims):
        plt.clf()
        rows,cols = 3,5
        imgs = []
        themods = []
        resids = []
        for j,(img,imgn,mod,ima,chi) in enumerate(subim):
            imgs.append(img)
            themods.append(mod)
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
        dimshow(get_rgb(themods, bands))
        plt.subplot(rows,cols, cols*2+4)
        dimshow(get_rgb(resids, bands, mnmx=(-10,10)))

        mnmx = -5,300
        kwa = dict(mnmx=mnmx, arcsinh=1)
        plt.subplot(rows,cols, 5)
        dimshow(get_rgb(imgs, bands, **kwa))
        plt.subplot(rows,cols, cols+5)
        dimshow(get_rgb(themods, bands, **kwa))
        plt.subplot(rows,cols, cols*2+5)
        mnmx = -100,100
        kwa = dict(mnmx=mnmx, arcsinh=1)
        dimshow(get_rgb(resids, bands, **kwa))
        plt.suptitle(titles[i])
        ps.savefig()

    if not chi_plots:
        return
    # Plot per-image chis: in a grid with band along the rows and images along the cols
    cols = max(len(v) for v in chis.values())
    rows = len(bands)
    for imod in range(len(mods)):
        plt.clf()
        for row,band in enumerate(bands):
            sp0 = 1 + cols*row
            # chis[band] = [ (one for each tim:) [ (one for each mod:) (chi,itim), (chi,itim) ], ...]
            for col,chilist in enumerate(chis[band]):
                chi,itim = chilist[imod]
                plt.subplot(rows, cols, sp0 + col)
                dimshow(-chi, **imchi)
                plt.xticks([]); plt.yticks([])
                plt.title(tims[itim].name)
        plt.suptitle(titles[imod])
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

    # Fit a MoG PSF model to the PSF in the middle of each tim.
    initial_psf_mog = []
    for itim,tim in enumerate(tims):
        ox0,oy0 = orig_wcsxy0[itim]
        h,w = tim.shape
        psfimg = tim.psfex.instantiateAt(ox0+(w/2), oy0+h/2, nativeScale=True)
        subpsf = GaussianMixturePSF.fromStamp(psfimg, emsteps=1000)
        initial_psf_mog.append((subpsf.mog.amp, subpsf.mog.mean, subpsf.mog.var))


    # Fit in order of flux
    for blobnumber,iblob in enumerate(np.argsort(-np.array(blobflux))):
        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        if len(Isrcs) == 0:
            continue

        tblob = Time()
        print
        print 'Blob', blobnumber, 'of', len(blobflux), ':', len(Isrcs), 'sources'
        print

        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])

        alphas = [0.1, 0.3, 1.0]

        if plots and False:
            imgs = [coimgs[i][bslc] for i in range(len(bands))]
            rgb = get_rgb(imgs, bands)
            rgb1 = rgb.copy()
            for i,cc in enumerate([0,1,0]):
                rgb[:,:,i][blobs[bslc] != (iblob+1)] = cc
            plt.clf()
            plt.subplot(1,3,1)
            dimshow(rgb1)
            plt.subplot(1,3,2)
            dimshow(blobs[bslc] == (iblob+1))
            plt.subplot(1,3,3)
            dimshow(rgb)
            plt.suptitle('blob (target coords)')
            ps.savefig()

        tlast = Time()
        subtims = []
        for itim,tim in enumerate(tims):
            ttim = Time()

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

            print 'tim clip:', Time()-ttim
            ttim = Time()

            # Mask out invvar for pixels that are not within the blob.
            subtarget = targetwcs.get_subimage(bx0, by0, blobw, blobh)
            subsubwcs = tim.subwcs.get_subimage(int(sx0), int(sy0), int(sx1-sx0), int(sy1-sy0))
            try:
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(subsubwcs, subtarget, [], 2)
            except OverlapError:
                print 'No overlap'
                continue
            if len(Yo) == 0:
                continue
            subiv2 = np.zeros_like(subiv)
            I = np.flatnonzero(blobs[bslc][Yi, Xi] == (iblob+1))
            subiv2[Yo[I],Xo[I]] = subiv[Yo[I],Xo[I]]
            subiv = subiv2

            print 'tim mask iv:', Time()-ttim
            ttim = Time()

            if plots and False:
                plt.clf()
                plt.subplot(1,2,1)
                dimshow(subimg)
                plt.subplot(1,2,2)
                dimshow(subiv)
                plt.suptitle('blob (subtim)')
                ps.savefig()

            # FIXME --
            #subpsf = tim.psfex.mogAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.)
            #subpsf = tim.getPsf()


            ttim = Time()

            psfimg = tim.psfex.instantiateAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.,
                                             nativeScale=True)

            print 'tim instantiate PSF:', Time()-ttim
            ttim = Time()

            if False:
                (w,mu,var) = initial_psf_mog[itim]
                thepsf = GaussianMixturePSF(w.copy(), mu.copy(), var.copy())
                psftim = Image(data=psfimg, invvar=np.zeros(psfimg.shape)+1e4,
                               psf=thepsf)
                ph,pw = psfimg.shape
                psftractor = Tractor([psftim], [PointSource(PixPos(pw/2., ph/2.), Flux(1.))])
                psftractor.freezeParam('catalog')
                psftim.freezeAllBut('psf')
                print 'Optimizing:'
                psftractor.printThawedParams()
                for step in range(100):
                    dlnp,X,alpha = psftractor.optimize(priors=False, shared_params=False)
                    print 'dlnp:', dlnp
                    if dlnp < 0.1:
                        break
                print 'Tractor fit PSF:'
                print thepsf
                print 'tim PSF fitting via Tractor:', Time()-ttim
                ttim = Time()

            # Note, initial_psf_mog is probably modified in this process!
            subpsf = GaussianMixturePSF.fromStamp(psfimg, P0=initial_psf_mog[itim])

            print 'EM fit PSF:'
            print subpsf
            
            print 'tim fit PSF:', Time()-ttim
            print 'psfimg shape', psfimg.shape
            ttim = Time()

            subtim = Image(data=subimg, invvar=subiv, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky(), name=tim.name)
            subtim.band = tim.band

            (Yo,Xo,Yi,Xi) = tim.resamp
            I = np.flatnonzero((Yi >= sy0) * (Yi < sy1) * (Xi >= sx0) * (Xi < sx1) *
                               (Yo >=  by0) * (Yo <  by1) * (Xo >=  bx0) * (Xo <  bx1))
            Yo = Yo[I] - by0
            Xo = Xo[I] - bx0
            Yi = Yi[I] - sy0
            Xi = Xi[I] - sx0
            subtim.resamp = (Yo, Xo, Yi, Xi)
            subtim.sig1 = tim.sig1

            print 'tim resamp:', Time()-ttim

            subtims.append(subtim)

        print 'subtims:', Time()-tlast
        #tlast = Time()

        subcat = Catalog(*[cat[i] for i in Isrcs])
        subtr = Tractor(subtims, subcat)
        subtr.freezeParam('images')

        if plots:
            plotmods = []
            plotmodnames = []
            plotmods.append(subtr.getModelImages())
            plotmodnames.append('Initial')
        print 'Sub-image initial lnlikelihood:', subtr.getLogLikelihood()

        # Optimize individual sources in order of flux
        fluxes = []
        for src in subcat:
            # HACK -- here we just *sum* the nanomaggies in each band.  Bogus!
            br = src.getBrightness()
            flux = sum([br.getFlux(band) for band in bands])
            fluxes.append(flux)
        Ibright = np.argsort(-np.array(fluxes))


        if len(Ibright) >= 5:
            # -Remember the original subtim images
            # -Compute initial models for each source (in each tim)
            # -Subtract initial models from images
            # -During fitting, for each source:
            #   -add back in the source's initial model (to each tim)
            #   -fit, with Catalog([src])
            #   -subtract final model (from each tim)
            # -Replace original subtim images
            #
            # --Might want to omit newly-added detection-filter sources, since their
            # fluxes are bogus.

            # Remember original tim images
            orig_timages = [tim.getImage().copy() for tim in subtims]
            initial_models = []

            # Create initial models for each tim x each source
            tt = Time()
            for tim in subtims:
                mods = []
                for src in subcat:
                    mod = src.getModelPatch(tim)
                    mods.append(mod)
                    if mod is not None:
                        mod.addTo(tim.getImage(), scale=-1)
                initial_models.append(mods)
            print 'Subtracting initial models:', Time()-tt

            # For sources in decreasing order of brightness
            for numi,i in enumerate(Ibright):
                tsrc = Time()
                print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
                src = subcat[i]
                print src

                srctractor = Tractor(subtims, [src])
                srctractor.freezeParams('images')

                # Add this source's initial model back in.
                for tim,mods in zip(subtims, initial_models):
                    mod = mods[i]
                    if mod is not None:
                        mod.addTo(tim.getImage())

                print 'Optimizing:', srctractor
                srctractor.printThawedParams()

                if plots:
                    spmods = [srctractor.getModelImages()]
                    spnames = ['Initial']
    
                for step in range(10):
                    dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                                  alphas=alphas)
                    print 'dlnp:', dlnp
                    if dlnp < 0.1:
                        break

                if plots:
                    spmods.append(srctractor.getModelImages())
                    spnames.append('Fit')
                    _plot_mods(subtims, spmods, spnames, bands, None, None, bslc, blobw, blobh, ps,
                               chi_plots=False)

                for tim in subtims:
                    mod = src.getModelPatch(tim)
                    if mod is not None:
                        mod.addTo(tim.getImage(), scale=-1)

                if plots:
                    _plot_mods(subtims, [srctractor.getModelImages()], ['Residuals'],
                               bands, None, None, bslc, blobw, blobh, ps, chi_plots=False)

                print 'Fitting source took', Time()-tsrc
                print src
    
            for tim,img in zip(subtims, orig_timages):
                tim.data = img

            del orig_timages
            del initial_models
            
        else:
            # Fit sources one at a time, but don't subtract other models
            subcat.freezeAllParams()
            for numi,i in enumerate(Ibright):
                tsrc = Time()
                print 'Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright))
                print subcat[i]
                subcat.freezeAllBut(i)
                print 'Optimizing:', subtr
                subtr.printThawedParams()
                for step in range(10):
                    dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                                  alphas=alphas)
                    print 'dlnp:', dlnp
                    if dlnp < 0.1:
                        break
                print 'Fitting source took', Time()-tsrc
                print subcat[i]

        if plots:
            plotmods.append(subtr.getModelImages())
            plotmodnames.append('Per Source')
        print 'Sub-image individual-source fit lnlikelihood:', subtr.getLogLikelihood()

        if len(Isrcs) > 1 and len(Isrcs) <= 10:
            tfit = Time()
            # Optimize all at once?
            subcat.thawAllParams()
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
                    plotmods.append(subtr.getModelImages())
                    plotmodnames.append('Borked')
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
                            dimshow(deriv.patch, cmap='RdBu')
                            plt.colorbar()
                            plt.title(tim.name)
                        plt.suptitle('Borked optimization: derivs for ' + paramname)
                        ps.savefig()
                if dlnp < 0.1:
                    break

            print 'Simultaneous fit took:', Time()-tfit
            if plots:
                plotmods.append(subtr.getModelImages())
                plotmodnames.append('All Sources')
            print 'Sub-image first fit lnlikelihood:', subtr.getLogLikelihood()

        # FIXME -- for large blobs, fit strata of sources simultaneously?

        if False:
            print 'Starting forced phot: time since blob start', Time()-tblob
            # Forced-photometer bands individually?
            for band in bands:
                tp = Time()
    
                subcat.freezeAllRecursive()
                subcat.thawPathsTo(band)
                bandtims = []
                for tim in subtims:
                    if tim.band == band:
                        bandtims.append(tim)
                print
                print 'Fitting', band, 'band:', len(bandtims), 'images'
                btractor = Tractor(bandtims, subcat)
                btractor.freezeParam('images')
                btractor.printThawedParams()
                B = 8
                X = btractor.optimize_forced_photometry(shared_params=False, use_ceres=True,
                                                        BW=B, BH=B, wantims=False)
                print 'Forced phot of', band, 'band took', Time()-tp
    
            subcat.thawAllRecursive()
            print 'Forced-phot lnlikelihood:', subtr.getLogLikelihood()
    
            if plots:
                plotmods.append(subtr.getModelImages())
                plotmodnames.append('Forced phot')

        if plots:
            _plot_mods(subtims, plotmods, plotmodnames, bands, coimgs, cons, bslc, blobw, blobh, ps)
            if blobnumber >= 10:
                plots = False

        print 'Blob', blobnumber, 'finished:', Time()-tlast
        tlast = Time()

    rtn = dict()
    for k in ['tractor','tims','ps']:
        rtn[k] = locals()[k]
    return rtn




def stage102(T=None, sedsn=None, coimgs=None, cons=None,
             detmaps=None, detivs=None,
             nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
             cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None,
             plots=False, tims=None, tractor=None,
             **kwargs):

    coimgs = [im.astype(np.float32) for im in coimgs]
    cons   = [im.astype(np.float32) for im in cons]

    return dict(sedsn=None, detmaps=None, detivs=None,
                nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
                coimgs=coimgs, cons=cons)

def stage203(T=None, coimgs=None, cons=None,
             cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None,
             plots=False, tims=None, tractor=None,
             brickid=None,
             **kwargs):
    print 'kwargs:', kwargs.keys()
    del kwargs

    from desi_common import prepare_fits_catalog

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    # Fit a MoG PSF model to the PsfEx model in the middle of each tim.
    for itim,tim in enumerate(tims):
        ox0,oy0 = orig_wcsxy0[itim]
        h,w = tim.shape
        psfimg = tim.psfex.instantiateAt(ox0+(w/2), oy0+h/2, nativeScale=True)
        subpsf = GaussianMixturePSF.fromStamp(psfimg, emsteps=1000)
        tim.psf = subpsf

    
    #cat.freezeAllRecursive()
    #cat.thawPathsTo(*bands)
    #print 'Variances...'
    #flux_var = tractor.optimize(priors=False, shared_params=False,
    #                            variance=True, just_variance=True)
    # print 'Opt forced photom...'
    # R = tractor.optimize_forced_photometry(
    #     shared_params=False, wantims=False, fitstats=True, variance=True,
    #     use_ceres=True, BW=8,BH=8)
    # flux_iv,fs = R.IV, R.fitstats
    #flux_iv = 1./flux_var

    print 'Variances...'
    cat.freezeAllRecursive()
    cat.thawPathsTo(*bands)
    flux_iv = []
    for isrc,src in enumerate(cat):
        print 'Variance for source', isrc, 'of', len(cat)
        srctr = Tractor(tims, [src])
        srctr.freezeParam('images')

        # flux_var = srctr.optimize(priors=False, shared_params=False,
        #                           variance=True, just_variance=True)
        # vars.append(flux_var)
        # print 'Flux variance:', flux_var

        chisqderivs = []
        for band in bands:
            src.freezeAllRecursive()
            src.thawPathsTo(band)

            # bandtims = [tim for tim in tims if tim.band == band]
            # btr = Tractor(bandtims, [src])
            # btr.freezeParam('images')
            # p0 = src.getParams()
            # R = btr.optimize_forced_photometry(variance=True, shared_params=False,
            #                                    wantims=False)
            # flux_iv = R.IV
            # print 'IV:', flux_iv
            # src.setParams(p0)
            # band_var = srctr.optimize(priors=False, shared_params=False,
            #                           variance=True, just_variance=True)
            # print 'Variance for', band, 'band:', band_var

            dchisq = 0
            for tim in tims:
                if tim.band != band:
                    continue
                derivs = src.getParamDerivatives(tim)
                # just the flux
                assert(len(derivs) == 1)
                H,W = tim.shape
                for deriv in derivs:
                    if deriv is None:
                        continue
                    if not deriv.clipTo(W,H):
                        continue
                    chi = deriv.patch * tim.getInvError()[deriv.getSlice()]
                    dchisq += (chi**2).sum()
            flux_iv.append(dchisq)
    flux_iv = np.array(flux_iv)
    assert(len(flux_iv) == len(cat)*len(bands))

    fs = None

    # HACK -- temp, until this propagates through the stages...
    if brickid is None:
        brickid = 377306

    TT = T.copy()
    for k in ['itx','ity','index']:
        TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob']:
            TT.rename(col, 'sdss_%s' % col)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.objid   = np.arange(len(TT)).astype(np.int32)

    for src in cat:
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseE.fromEllipseESoft(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
            src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)

    cat.freezeAllRecursive()
    cat.thawPathsTo(*bands)

    hdr = None
    T2,hdr = prepare_fits_catalog(cat, 1./flux_iv, TT, hdr, bands, fs)
    for k in ['ra_var', 'dec_var']:
        T2.set(k, T2.get(k).astype(np.float32))
    T2.writeto('tractor-phot-b%i.fits' % brickid, header=hdr)

    return dict(flux_iv=flux_iv, tims=tims, cat=cat)


def stage204(T=None, flux_iv=None, tims=None, cat=None,
             bands=None, brickid=None, **kwargs):
    from desi_common import prepare_fits_catalog

    fs = None

    # HACK -- temp, until this propagates through the stages...
    if brickid is None:
        brickid = 377306

    TT = T.copy()
    for k in ['itx','ity','index']:
        TT.delete_column(k)
    for col in TT.get_columns():
        if not col in ['tx', 'ty', 'blob']:
            TT.rename(col, 'sdss_%s' % col)

    TT.brickid = np.zeros(len(TT), np.int32) + brickid
    TT.objid   = np.arange(len(TT)).astype(np.int32)

    for src in cat:
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseE.fromEllipseESoft(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
            src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)

    cat.freezeAllRecursive()
    cat.thawPathsTo(*bands)

    hdr = None
    T2,hdr = prepare_fits_catalog(cat, 1./flux_iv, TT, hdr, bands, fs)
    for k in ['ra_var', 'dec_var', 'tx', 'ty']:
        T2.set(k, T2.get(k).astype(np.float32))
    T2.writeto('tractor-phot-b%i.fits' % brickid, header=hdr)


def stage103(T=None, coimgs=None, cons=None,
             cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None,
             plots=False, tims=None, tractor=None,
             **kwargs):

    print 'kwargs:', kwargs.keys()
    del kwargs

    # for tim in tims:
    #     print 'Fitting PsfEx model for', tim.name
    #     tim.psfex.ensureFit()
    #     tim.psf = tim.psfex

    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    plt.title('Image')
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

    mnmx = -5,300
    arcsinha = dict(mnmx=mnmx, arcsinh=1)

    plt.clf()
    dimshow(get_rgb(coimgs, bands, **arcsinha))
    plt.title('Image')
    ps.savefig()

    # After plot
    rgbmod = []
    rgbmod2 = []
    rgbresids = []

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    for iband,band in enumerate(bands):
        coimg = coimgs[iband]
        comod = np.zeros((H,W), np.float32)
        comod2 = np.zeros((H,W), np.float32)
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue

            # Fit a MoG PSF model to the PsfEx model in the middle of the tim
            ox0,oy0 = orig_wcsxy0[itim]
            h,w = tim.shape
            psfimg = tim.psfex.instantiateAt(ox0+(w/2), oy0+h/2, nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg, emsteps=1000)
            tim.psf = subpsf

            if False:
                print 'Fitting PsfEx model for', tim.name
                tim.psfex.savesplinedata = True
                tim.psfex.ensureFit()
                tim.psf = tim.psfex

            mod = tractor.getModelImage(tim)

            if plots:
                plt.clf()
                dimshow(tim.getImage(), **tim.ima)
                plt.title(tim.name)
                ps.savefig()
                plt.clf()
                dimshow(mod, **tim.ima)
                plt.title(tim.name)
                ps.savefig()
                plt.clf()
                dimshow((tim.getImage() - mod) * tim.getInvError(), **imchi)
                plt.title(tim.name)
                ps.savefig()

            (Yo,Xo,Yi,Xi) = tim.resamp
            comod[Yo,Xo] += mod[Yi,Xi]
            ie = tim.getInvError()
            noise = np.random.normal(size=ie.shape) / ie
            noise[ie == 0] = 0.
            comod2[Yo,Xo] += mod[Yi,Xi] + noise[Yi,Xi]
        comod  /= np.maximum(cons[iband], 1)
        comod2 /= np.maximum(cons[iband], 1)

        rgbmod.append(comod)
        rgbmod2.append(comod2)
        resid = coimg - comod
        resid[cons[iband] == 0] = np.nan
        rgbresids.append(resid)

        fitsio.write('image-coadd-%s.fits' % band, comod)
        fitsio.write('model-coadd-%s.fits' % band, coimg)
        fitsio.write('resid-coadd-%s.fits' % band, resid)

    plt.clf()
    dimshow(get_rgb(rgbmod, bands))
    plt.title('Model')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbmod2, bands))
    plt.title('Model + Noise')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbmod2, bands, **arcsinha))
    plt.title('Model + Noise')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbresids, bands))
    plt.title('Residuals')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbresids, bands, mnmx=(-30,30)))
    plt.title('Residuals (2)')
    ps.savefig()

    return dict(tims=tims)

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

    parser.add_option('-b', '--brick', type=int, help='Brick ID to run: default %default',
                      default=377306)

    parser.add_option('--threads', type=int, help='Run multi-threaded')
    parser.add_option('-p', '--plots', dest='plots', action='store_true',
                      help='Per-blob plots?')
    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='runbrick-s%03i.pickle')

    parser.add_option('-W', type=int, default=3600, help='Target image width (default %default)')
    parser.add_option('-H', type=int, default=3600, help='Target image height (default %default)')

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
    prereqs = {101: 0, 102:1, 203:102 }
    opt.force.append(opt.stage)
    
    runstage(opt.stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
             prereqs=prereqs, plots=opt.plots, W=opt.W, H=opt.H, brickid=opt.brick)
    
