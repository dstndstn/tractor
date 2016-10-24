from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os

os.environ['TMPDIR'] = 'tmp'

import fitsio

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_closing, binary_erosion

from astrometry.util.util import *
from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.resample import resample_with_wcs,OverlapError
from astrometry.util.starutil_numpy import *
from astrometry.util.miscutils import clip_polygon
from astrometry.libkd.spherematch import match_radec

from astrometry.util.ttime import Time, MemMeas
from astrometry.util.stages import *

from tractor import *
from tractor.galaxy import *
from tractor.sdss import get_tractor_sources_dr9

from common import *

from runbrick import get_rgb, get_sdss_sources

green = (0,1,0)

plt.figure(figsize=(10,8))

class CfhtDecals(Decals):
    def get_zeropoint_for(self, im):
        hdr = im.read_image_header()
        magzp = hdr['PHOT_C'] + 2.5 * np.log10(hdr['EXPTIME'])
        return magzp

def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[], type=int,
                      help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_option('-s', '--stage', dest='stage', default=[], type=int, action='append',
                      help="Run up to the given stage(s)")
    parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')
    parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
                      help='Make more verbose')
    opt,args = parser.parse_args()

    stagefunc = CallGlobal('stage%i', globals())

    if len(opt.stage) == 0:
        opt.stage.append(1)
    opt.force.extend(opt.stage)
    opt.picklepat = 'pickles/cfht-s%03i.pickle'
    prereqs = {}

    kwargs = {}
    for stage in opt.stage:
        runstage(stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
                 prereqs=prereqs, **kwargs)


def stage0(**kwargs):
    ps = PlotSequence('cfht')

    decals = CfhtDecals()
    B = decals.get_bricks()
    print('Bricks:')
    B.about()

    ra,dec = 190.0, 11.0

    #bands = 'ugri'
    bands = 'gri'
    
    B.cut(np.argsort(degrees_between(ra, dec, B.ra, B.dec)))
    print('Nearest bricks:', B.ra[:5], B.dec[:5], B.brickid[:5])

    brick = B[0]
    pixscale = 0.186
    #W,H = 1024,1024
    #W,H = 2048,2048
    #W,H = 3600,3600
    W,H = 4800,4800

    targetwcs = wcs_for_brick(brick, pixscale=pixscale, W=W, H=H)
    ccdfn = 'cfht-ccds.fits'
    if os.path.exists(ccdfn):
        T = fits_table(ccdfn)
    else:
        T = get_ccd_list()
        T.writeto(ccdfn)
    print(len(T), 'CCDs')
    T.cut(ccds_touching_wcs(targetwcs, T))
    print(len(T), 'CCDs touching brick')

    T.cut(np.array([b in bands for b in T.filter]))
    print(len(T), 'in bands', bands)

    ims = []
    for t in T:
        im = CfhtImage(t)
        # magzp = hdr['PHOT_C'] + 2.5 * np.log10(hdr['EXPTIME'])
        # fwhm = t.seeing / (pixscale * 3600)
        # print '-> FWHM', fwhm, 'pix'
        im.seeing = t.seeing
        im.pixscale = t.pixscale
        print('seeing', t.seeing)
        print('pixscale', im.pixscale*3600, 'arcsec/pix')
        im.run_calibs(t.ra, t.dec, im.pixscale, W=t.width, H=t.height)
        ims.append(im)


    # Read images, clip to ROI
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    keepims = []
    tims = []
    for im in ims:
        print()
        print('Reading expnum', im.expnum, 'name', im.extname, 'band', im.band, 'exptime', im.exptime)
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

        print('Image slice: x [%i,%i], y [%i,%i]' % (x0,x1, y0,y1))
        print('Reading image from', im.imgfn, 'HDU', im.hdu)
        img,imghdr = im.read_image(header=True, slice=slc)
        goodpix = (img != 0)
        print('Number of pixels == 0:', np.sum(img == 0))
        print('Number of pixels != 0:', np.sum(goodpix))
        if np.sum(goodpix) == 0:
            continue
        # print 'Image shape', img.shape
        print('Image range', img.min(), img.max())
        print('Goodpix image range:', (img[goodpix]).min(), (img[goodpix]).max())
        if img[goodpix].min() == img[goodpix].max():
            print('No dynamic range in image')
            continue
        # print 'Reading invvar from', im.wtfn, 'HDU', im.hdu
        # invvar = im.read_invvar(slice=slc)
        # # print 'Invvar shape', invvar.shape
        # # print 'Invvar range:', invvar.min(), invvar.max()
        # invvar[goodpix == 0] = 0.
        # if np.all(invvar == 0.):
        #     print 'Skipping zero-invvar image'
        #     continue
        # assert(np.all(np.isfinite(img)))
        # assert(np.all(np.isfinite(invvar)))
        # assert(not(np.all(invvar == 0.)))
        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        # slice1 = (slice(0,-5,10),slice(0,-5,10))
        # slice2 = (slice(5,None,10),slice(5,None,10))
        # # print 'sliced shapes:', img[slice1].shape, img[slice2].shape
        # # print 'good shape:', (goodpix[slice1] * goodpix[slice2]).shape
        # # print 'good values:', np.unique(goodpix[slice1] * goodpix[slice2])
        # # print 'sliced[good] shapes:', (img[slice1] -  img[slice2])[goodpix[slice1] * goodpix[slice2]].shape
        # mad = np.median(np.abs(img[slice1] - img[slice2])[goodpix[slice1] * goodpix[slice2]].ravel())
        # sig1 = 1.4826 * mad / np.sqrt(2.)
        # print 'MAD sig1:', sig1
        # # invvar was 1 or 0
        # invvar *= (1./(sig1**2))
        # medsky = np.median(img[goodpix])

        # Read full image for sig1 and sky estimate
        fullimg = im.read_image()
        fullgood = (fullimg != 0)
        # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(fullimg[slice1] - fullimg[slice2])[fullgood[slice1] * fullgood[slice2]].ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        print('MAD sig1:', sig1)
        medsky = np.median(fullimg[fullgood])
        invvar = np.zeros_like(img)
        invvar[goodpix] = 1./sig1**2

        # Median-smooth sky subtraction
        plt.clf()
        dimshow(np.round((img-medsky) / sig1), vmin=-3, vmax=5)
        plt.title('Scalar median: %s' % im.name)
        ps.savefig()

        # medsky = np.zeros_like(img)
        # # astrometry.util.util
        # median_smooth(img, np.logical_not(goodpix), 256, medsky)
        fullmed = np.zeros_like(fullimg)
        median_smooth(fullimg - medsky, np.logical_not(fullgood), 256, fullmed)
        fullmed += medsky
        medimg = fullmed[slc]
        
        plt.clf()
        dimshow(np.round((img - medimg) / sig1), vmin=-3, vmax=5)
        plt.title('Median filtered: %s' % im.name)
        ps.savefig()
        
        #print 'Subtracting median:', medsky
        #img -= medsky
        img -= medimg
        
        primhdr = im.read_image_primary_header()

        magzp = decals.get_zeropoint_for(im)
        print('magzp', magzp)
        zpscale = NanoMaggies.zeropointToScale(magzp)
        print('zpscale', zpscale)

        # Scale images to Nanomaggies
        img /= zpscale
        sig1 /= zpscale
        invvar *= zpscale**2
        orig_zpscale = zpscale

        zpscale = 1.
        assert(np.sum(invvar > 0) > 0)
        print('After scaling:')
        print('sig1', sig1)
        print('invvar range', invvar.min(), invvar.max())
        print('image range', img.min(), img.max())

        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(np.isfinite(sig1))

        plt.clf()
        lo,hi = -5*sig1, 10*sig1
        n,b,p = plt.hist(img[goodpix].ravel(), 100, range=(lo,hi), histtype='step', color='k')
        xx = np.linspace(lo, hi, 200)
        plt.plot(xx, max(n)*np.exp(-xx**2 / (2.*sig1**2)), 'r-')
        plt.xlim(lo,hi)
        plt.title('Pixel histogram: %s' % im.name)
        ps.savefig()

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        info = im.get_image_info()
        fullh,fullw = info['dims']

        # read fit PsfEx model
        psfex = PsfEx.fromFits(im.psffitfn)
        print('Read', psfex)

        # HACK -- highly approximate PSF here!
        #psf_fwhm = imghdr['FWHM']
        #psf_fwhm = im.seeing

        psf_fwhm = im.seeing / (im.pixscale * 3600)
        print('PSF FWHM', psf_fwhm, 'pixels')
        psf_sigma = psf_fwhm / 2.35
        psf = NCircularGaussianPSF([psf_sigma],[1.])

        print('img type', img.dtype)
        
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
        tim.imobj = im
        mn,mx = tim.zr
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        tims.append(tim)
        keepims.append(im)

    ims = keepims

    print('Computing resampling...')
    # save resampling params
    for tim in tims:
        wcs = tim.sip_wcs
        subh,subw = tim.shape
        subwcs = wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        tim.subwcs = subwcs
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, subwcs, [], 2)
        except OverlapError:
            print('No overlap')
            continue
        if len(Yo) == 0:
            continue
        tim.resamp = (Yo,Xo,Yi,Xi)

    print('Creating coadds...')
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
            if len(Yo) == 0:
                continue
            nn = (tim.getInvvar()[Yi,Xi] > 0)
            coimg[Yo,Xo] += tim.getImage ()[Yi,Xi] * nn
            con  [Yo,Xo] += nn

            # print
            # print 'tim', tim.name
            # print 'number of resampled pix:', len(Yo)
            # reim = np.zeros_like(coimg)
            # ren  = np.zeros_like(coimg)
            # reim[Yo,Xo] = tim.getImage()[Yi,Xi] * nn
            # ren[Yo,Xo] = nn
            # print 'number of resampled pix with positive invvar:', ren.sum()
            # plt.clf()
            # plt.subplot(2,2,1)
            # mn,mx = [np.percentile(reim[ren>0], p) for p in [25,95]]
            # print 'Percentiles:', mn,mx
            # dimshow(reim, vmin=mn, vmax=mx)
            # plt.colorbar()
            # plt.subplot(2,2,2)
            # dimshow(con)
            # plt.colorbar()
            # plt.subplot(2,2,3)
            # dimshow(reim, vmin=tim.zr[0], vmax=tim.zr[1])
            # plt.colorbar()
            # plt.subplot(2,2,4)
            # plt.hist(reim.ravel(), 100, histtype='step', color='b')
            # plt.hist(tim.getImage().ravel(), 100, histtype='step', color='r')
            # plt.suptitle('%s: %s' % (band, tim.name))
            # ps.savefig()

        coimg /= np.maximum(con,1)
        coimgs.append(coimg)
        cons  .append(con)

    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    ps.savefig()

    plt.clf()
    for i,b in enumerate(bands):
        plt.subplot(2,2,i+1)
        dimshow(cons[i], ticks=False)
        plt.title('%s band' % b)
        plt.colorbar()
    plt.suptitle('Number of exposures')
    ps.savefig()

    print('Grabbing SDSS sources...')
    bandlist = [b for b in bands]
    cat,T = get_sdss_sources(bandlist, targetwcs)
    # record coordinates in target brick image
    ok,T.tx,T.ty = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.tx -= 1
    T.ty -= 1
    T.itx = np.clip(np.round(T.tx).astype(int), 0, W-1)
    T.ity = np.clip(np.round(T.ty).astype(int), 0, H-1)

    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    ax = plt.axis()
    plt.plot(T.tx, T.ty, 'o', mec=green, mfc='none', ms=10, mew=1.5)
    plt.axis(ax)
    plt.title('SDSS sources')
    ps.savefig()

    print('Detmaps...')
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

    rtn = dict()
    for k in ['T', 'coimgs', 'cons', 'detmaps', 'detivs',
              'targetrd', 'pixscale', 'targetwcs', 'W','H',
              'bands', 'tims', 'ps', 'brick', 'cat']:
        rtn[k] = locals()[k]
    return rtn


def stage1(T=None, coimgs=None, cons=None, detmaps=None, detivs=None,
           targetrd=None, pixscale=None, targetwcs=None, W=None,H=None,
           bands=None, tims=None, ps=None, brick=None, cat=None):
    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    hot = np.zeros((H,W), np.float32)

    for band in bands:
        detmap = detmaps[band] / np.maximum(1e-16, detivs[band])
        detsn = detmap * np.sqrt(detivs[band])
        hot = np.maximum(hot, detsn)
        detmaps[band] = detmap

    ### FIXME -- ugri
    for sedname,sed in [('Flat', (1.,1.,1.)), ('Red', (2.5, 1.0, 0.4))]:
        sedmap = np.zeros((H,W), np.float32)
        sediv  = np.zeros((H,W), np.float32)
        for iband,band in enumerate(bands):
            # We convert the detmap to canonical band via
            #   detmap * w
            # And the corresponding change to sig1 is
            #   sig1 * w
            # So the invvar-weighted sum is
            #    (detmap * w) / (sig1**2 * w**2)
            #  = detmap / (sig1**2 * w)
            sedmap += detmaps[band] * detivs[band] / sed[iband]
            sediv  += detivs [band] / sed[iband]**2
        sedmap /= np.maximum(1e-16, sediv)
        sedsn   = sedmap * np.sqrt(sediv)
        hot = np.maximum(hot, sedsn)

        plt.clf()
        dimshow(np.round(sedsn), vmin=0, vmax=10, cmap='hot')
        plt.title('SED-matched detection filter: %s' % sedname)
        ps.savefig()

    peaks = (hot > 4)
    blobs,nblobs = label(peaks)
    print('N detected blobs:', nblobs)
    blobslices = find_objects(blobs)
    # Un-set catalog blobs
    for x,y in zip(T.itx, T.ity):
        # blob number
        bb = blobs[y,x]
        if bb == 0:
            continue
        # un-set 'peaks' within this blob
        slc = blobslices[bb-1]
        peaks[slc][blobs[slc] == bb] = 0

    # Now, after having removed catalog sources, crank up the detection threshold
    peaks &= (hot > 5)
        
    # zero out the edges(?)
    peaks[0 ,:] = peaks[:, 0] = 0
    peaks[-1,:] = peaks[:,-1] = 0
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[0:-2,1:-1])
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[2:  ,1:-1])
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[1:-1,0:-2])
    peaks[1:-1, 1:-1] &= (hot[1:-1,1:-1] >= hot[1:-1,2:  ])

    # These are our peaks
    pki = np.flatnonzero(peaks)
    peaky,peakx = np.unravel_index(pki, peaks.shape)
    print(len(peaky), 'peaks')

    crossa = dict(ms=10, mew=1.5)
    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    ax = plt.axis()
    plt.plot(T.tx, T.ty, 'r+', **crossa)
    plt.plot(peakx, peaky, '+', color=green, **crossa)
    plt.axis(ax)
    plt.title('SDSS + SED-matched detections')
    ps.savefig()


    ### HACK -- high threshold again

    # Segment, and record which sources fall into each blob
    blobs,nblobs = label((hot > 20))
    print('N detected blobs:', nblobs)
    blobslices = find_objects(blobs)
    T.blob = blobs[T.ity, T.itx]
    blobsrcs = []
    blobflux = []
    fluximg = coimgs[1]
    for blob in range(1, nblobs+1):
        blobsrcs.append(np.flatnonzero(T.blob == blob))
        bslc = blobslices[blob-1]
        blobflux.append(np.sum(fluximg[bslc][blobs[bslc] == blob]))

    # Fit the SDSS sources
    
    for tim in tims:
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = tim.psfex
        
    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1
    srcvariances = [[] for src in cat]
    # Fit in order of flux
    for blobnumber,iblob in enumerate(np.argsort(-np.array(blobflux))):

        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        if len(Isrcs) == 0:
            continue

        print()
        print('Blob', blobnumber, 'of', len(blobflux), ':', len(Isrcs), 'sources')
        print('Source indices:', Isrcs)
        print()

        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])
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
            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subie  = tim.getInvError()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[itim]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            # Mask out inverr for pixels that are not within the blob.
            subtarget = targetwcs.get_subimage(bx0, by0, blobw, blobh)
            subsubwcs = tim.subwcs.get_subimage(int(sx0), int(sy0), int(sx1-sx0), int(sy1-sy0))
            try:
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(subsubwcs, subtarget, [], 2)
            except OverlapError:
                print('No overlap')
                continue
            if len(Yo) == 0:
                continue
            subie2 = np.zeros_like(subie)
            I = np.flatnonzero(blobs[bslc][Yi, Xi] == (iblob+1))
            subie2[Yo[I],Xo[I]] = subie[Yo[I],Xo[I]]
            subie = subie2
            # If the subimage (blob) is small enough, instantiate a
            # constant PSF model in the center.
            if sy1-sy0 < 100 and sx1-sx0 < 100:
                subpsf = tim.psf.mogAt(ox0 + (sx0+sx1)/2., oy0 + (sy0+sy1)/2.)
            else:
                # Otherwise, instantiate a (shifted) spatially-varying
                # PsfEx model.
                subpsf = ShiftedPsf(tim.psf, ox0+sx0, oy0+sy0)

            subtim = Image(data=subimg, inverr=subie, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky(), name=tim.name)
            subtim.band = tim.band
            subtim.sig1 = tim.sig1
            subtim.modelMinval = tim.modelMinval
            subtims.append(subtim)

        subcat = Catalog(*[cat[i] for i in Isrcs])
        subtr = Tractor(subtims, subcat)
        subtr.freezeParam('images')
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
            for tim in subtims:
                mods = []
                for src in subcat:
                    mod = src.getModelPatch(tim)
                    mods.append(mod)
                    if mod is not None:
                        if not np.all(np.isfinite(mod.patch)):
                            print('Non-finite mod patch')
                            print('source:', src)
                            print('tim:', tim)
                            print('PSF:', tim.getPsf())
                        assert(np.all(np.isfinite(mod.patch)))
                        mod.addTo(tim.getImage(), scale=-1)
                initial_models.append(mods)
            # For sources in decreasing order of brightness
            for numi,i in enumerate(Ibright):
                tsrc = Time()
                print('Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright)))
                src = subcat[i]
                print(src)

                srctractor = Tractor(subtims, [src])
                srctractor.freezeParams('images')
                
                # Add this source's initial model back in.
                for tim,mods in zip(subtims, initial_models):
                    mod = mods[i]
                    if mod is not None:
                        mod.addTo(tim.getImage())

                print('Optimizing:', srctractor)
                srctractor.printThawedParams()
                for step in range(50):
                    dlnp,X,alpha = srctractor.optimize(priors=False, shared_params=False,
                                                  alphas=alphas)
                    print('dlnp:', dlnp, 'src', src)
                    if dlnp < 0.1:
                        break

                for tim in subtims:
                    mod = src.getModelPatch(tim)
                    if mod is not None:
                        mod.addTo(tim.getImage(), scale=-1)
    
            for tim,img in zip(subtims, orig_timages):
                tim.data = img

            del orig_timages
            del initial_models
        else:
            # Fit sources one at a time, but don't subtract other models
            subcat.freezeAllParams()
            for numi,i in enumerate(Ibright):
                tsrc = Time()
                print('Fitting source', i, '(%i of %i in blob)' % (numi, len(Ibright)))
                print(subcat[i])
                subcat.freezeAllBut(i)
                print('Optimizing:', subtr)
                subtr.printThawedParams()
                for step in range(10):
                    dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                                  alphas=alphas)
                    print('dlnp:', dlnp)
                    if dlnp < 0.1:
                        break
                print('Fitting source took', Time()-tsrc)
                print(subcat[i])
        if len(Isrcs) > 1 and len(Isrcs) <= 10:
            tfit = Time()
            # Optimize all at once?
            subcat.thawAllParams()
            print('Optimizing:', subtr)
            subtr.printThawedParams()
            for step in range(20):
                dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                              alphas=alphas)
                print('dlnp:', dlnp)
                if dlnp < 0.1:
                    break

        # Variances
        subcat.thawAllRecursive()
        subcat.freezeAllParams()
        for isub,srci in enumerate(Isrcs):
            print('Variances for source', srci)
            subcat.thawParam(isub)

            src = subcat[isub]
            print('Source', src)
            print('Params:', src.getParamNames())
            
            if isinstance(src, (DevGalaxy, ExpGalaxy)):
                src.shape = EllipseE.fromEllipseESoft(src.shape)
            elif isinstance(src, FixedCompositeGalaxy):
                src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
                src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)

            print('Converted ellipse:', src)

            allderivs = subtr.getDerivs()
            for iparam,derivs in enumerate(allderivs):
                dchisq = 0
                for deriv,tim in derivs:
                    h,w = tim.shape
                    deriv.clipTo(w,h)
                    ie = tim.getInvError()
                    slc = deriv.getSlice(ie)
                    chi = deriv.patch * ie[slc]
                    dchisq += (chi**2).sum()
                if dchisq == 0.:
                    v = np.nan
                else:
                    v = 1./dchisq
                srcvariances[srci].append(v)
            assert(len(srcvariances[srci]) == subcat[isub].numberOfParams())
            subcat.freezeParam(isub)

    cat.thawAllRecursive()

    for i,src in enumerate(cat):
        print('Source', i, src)
        print('variances:', srcvariances[i])
        print(len(srcvariances[i]), 'vs', src.numberOfParams())
        if len(srcvariances[i]) != src.numberOfParams():
            # This can happen for sources outside the brick bounds: they never get optimized?
            print('Warning: zeroing variances for source', src)
            srcvariances[i] = [0]*src.numberOfParams()
            if isinstance(src, (DevGalaxy, ExpGalaxy)):
                src.shape = EllipseE.fromEllipseESoft(src.shape)
            elif isinstance(src, FixedCompositeGalaxy):
                src.shapeExp = EllipseE.fromEllipseESoft(src.shapeExp)
                src.shapeDev = EllipseE.fromEllipseESoft(src.shapeDev)
        assert(len(srcvariances[i]) == src.numberOfParams())

    variances = np.hstack(srcvariances)
    assert(len(variances) == cat.numberOfParams())

    return dict(cat=cat, variances=variances)

def stage2(cat=None, variances=None, T=None, bands=None, ps=None,
           targetwcs=None, **kwargs):
    print('kwargs:', kwargs.keys())
    #print 'variances:', variances

    from desi_common import prepare_fits_catalog
    TT = T.copy()
    hdr = None
    fs = None
    T2,hdr = prepare_fits_catalog(cat, 1./np.array(variances), TT, hdr, bands, fs)
    T2.about()
    T2.writeto('cfht.fits')
    
    ccmap = dict(g='g', r='r', i='m')

    bandlist = [b for b in bands]
    scat,S = get_sdss_sources(bandlist, targetwcs)
    S.about()
    I,J,d = match_radec(T2.ra, T2.dec, S.ra, S.dec, 1./3600.)
    M = fits_table()
    M.ra = S.ra[J]
    M.dec = S.dec[J]
    M.cfhtI = I
    for band in bands:
        mag = T2.get('decam_%s_mag' % band)[I]
        sflux = np.array([s.getBrightness().getBand(band) for s in scat])[J]
        smag = NanoMaggies.nanomaggiesToMag(sflux)
        M.set('mag_%s' % band, mag)
        M.set('smag_%s' % band, smag)
        cc = ccmap[band]
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(smag, mag, '.', color=cc, alpha=0.5)
        lo,hi = 16,24
        plt.plot([lo,hi],[lo,hi], 'k-')
        plt.xlabel('SDSS mag')
        plt.ylabel('CFHT mag')
        plt.axis([lo,hi,lo,hi])

        plt.subplot(2,1,2)
        plt.plot(smag, mag - smag, '.', color=cc, alpha=0.5)
        lo,hi = 16,24
        plt.plot([lo,hi],[0, 0], 'k-')
        plt.xlabel('SDSS mag')
        plt.ylabel('CFHT - SDSS mag')
        plt.axis([lo,hi,-1,1])

        plt.suptitle('%s band' % band)
        ps.savefig()
    M.writeto('cfht-matched.fits')
        
    plt.clf()
    lp,lt = [],[]
    for band in bands:
        sn = T2.get('decam_%s_nanomaggies' % band) * np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band))
        #mag = T2.get('decam_%s_mag_corr' % band)
        mag = T2.get('decam_%s_mag' % band)
        print('band', band)
        print('Mags:', mag)
        print('SN:', sn)
        cc = ccmap[band]
        p = plt.semilogy(mag, sn, '.', color=cc, alpha=0.5)
        lp.append(p[0])
        lt.append('%s band' % band)
    plt.xlabel('mag')
    plt.ylabel('Flux Signal-to-Noise')
    tt = [1,2,3,4,5,10,20,30,40,50]
    plt.yticks(tt, ['%i' % t for t in tt])
    plt.axhline(5., color='k')
    #plt.axis([21, 26, 1, 20])
    plt.legend(lp, lt, loc='upper right')
    plt.title('CFHT depth')
    ps.savefig()

    # ['tims', 'cons', 'pixscale', 'H', 'coimgs', 'detmaps', 'W', 'brick', 'detivs', 'targetrd']

    return dict(T2=T2, M=M, tims=None, detmaps=None, detivs=None,
                cons=None, coimgs=None)

def stage3(cat=None, variances=None, T=None, bands=None, ps=None,
           targetwcs=None, T2=None, M=None, **kwargs):

    ccmap = dict(g='g', r='r', i='m')

    for band in bands:
        mag = M.get('mag_%s' % band)
        # sdss
        smag = M.get('smag_%s' % band)
        ok = np.flatnonzero(mag != smag)
        mag = mag[ok]
        smag = smag[ok]
        
        cc = ccmap[band]
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(smag, mag, '.', color=cc, alpha=0.5)
        lo,hi = 16,24
        plt.plot([lo,hi],[lo,hi], 'k-')
        plt.xlabel('SDSS mag')
        plt.ylabel('CFHT mag')
        plt.axis([lo,hi,lo,hi])

        plt.subplot(2,1,2)
        plt.plot(smag, mag - smag, '.', color=cc, alpha=0.5)
        lo,hi = 16,24
        plt.plot([lo,hi],[0, 0], 'k-')
        plt.xlabel('SDSS mag')
        plt.ylabel('CFHT - SDSS mag')
        plt.axis([lo,hi,-1,1])

        plt.suptitle('%s band' % band)
        ps.savefig()

    plt.clf()
    lp,lt = [],[]
    for band in bands:
        sn = (T2.get('decam_%s_nanomaggies' % band) * 
              np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band)))
        mag = T2.get('decam_%s_mag' % band)
        cc = ccmap[band]
        p = plt.semilogy(mag, sn, '.', color=cc, alpha=0.5)
        lp.append(p[0])
        lt.append('%s band' % band)
    plt.xlabel('mag')
    plt.ylabel('Flux Signal-to-Noise')
    #tt = [1,2,3,4,5,10,20,30,40,50]
    tt = [1,3,5,10,30,50,100,300,500,1000,3000,5000,10000,30000,50000]
    plt.yticks(tt, ['%i' % t for t in tt])
    plt.axhline(5., color='k')
    #plt.axis([21, 26, 1, 20])
    plt.legend(lp, lt, loc='upper right')
    plt.title('CFHT depth')
    lo,hi = 16,27
    plt.xlim(lo,hi)
    ps.savefig()


    # zps = {}
    # plt.clf()
    # lp,lt = [],[]
    # for band in bands:
    #     sn = (T2.get('decam_%s_nanomaggies' % band) * 
    #           np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band)))
    #     mag = T2.get('decam_%s_mag' % band)
    #     cc = ccmap[band]
    #     I = (np.isfinite(mag) * (T2.type == 'S') * np.isfinite(sn))
    #     Ti = T2[I]
    #     Ti.mag = mag[I]
    #     Ti.sn = sn[I]
    #     zps[band] = np.median(np.log10(Ti.sn) + 0.4*Ti.mag)
    #     plt.plot(Ti.mag, np.log10(Ti.sn) + 0.4*Ti.mag, '.', color=cc)
    # ps.savefig()
        

    metadata = dict(
        g=(1080946, 634),
        r=(1617879, 445),
        i=(1080306, 411),
        )
    
    zps = {}
    lo,hi = 16,27
    plt.clf()
    lp,lt = [],[]
    for band in bands:
        sn = (T2.get('decam_%s_nanomaggies' % band) * 
              np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band)))
        mag = T2.get('decam_%s_mag' % band)
        cc = ccmap[band]
        I = (np.isfinite(mag) * (T2.type == 'S') * np.isfinite(sn))
        Ti = T2[I]
        Ti.mag = mag[I]
        Ti.sn = sn[I]

        zp = np.median(np.log10(Ti.sn) + 0.4*Ti.mag)
        zps[band] = zp
        print('zp', zp)
        xx = np.array([lo,hi])
        plt.plot(xx, 10.**(zp - 0.4*xx), '-', alpha=0.4, color=cc)
        plt.plot(xx, 10.**(zp - 0.4*xx), 'k-', alpha=0.4)
        
        p = plt.semilogy(Ti.mag, Ti.sn, '.', color=cc, alpha=0.5)

        (expnum, exptime) = metadata[band]
        depth = (zp - np.log10(5.)) / 0.4
        lp.append(p[0])
        lt.append('%s band: exptime %i, depth %.2f (exp %i)' %
                  (band, exptime, depth, expnum))
        plt.plot([depth,depth], [1,5], color=cc, alpha=0.5)
        
    plt.xlabel('mag')
    plt.ylabel('Flux Signal-to-Noise')
    #tt = [1,2,3,4,5,10,20,30,40,50]
    tt = [1,3,5,10,30,50,100,300,500,1000,3000,5000,10000,30000,50000]
    plt.yticks(tt, ['%i' % t for t in tt])
    plt.axhline(5., color='k')
    #plt.axis([21, 26, 1, 20])
    plt.xlim(lo,hi)
    #ylo,yhi = plt.ylim()
    #plt.ylim(1, yhi)
    plt.ylim(1, 1e5)
    plt.legend(lp, lt, loc='upper right')
    plt.title('CFHT depth (point sources)')
    ps.savefig()



    plt.clf()
    lp,lt = [],[]
    for band in bands:
        sn = (T2.get('decam_%s_nanomaggies' % band) * 
              np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band)))
        mag = T2.get('decam_%s_mag' % band)
        cc = ccmap[band]
        I = (np.isfinite(mag) * (T2.type != 'S') * np.isfinite(sn))
        Ti = T2[I]
        Ti.mag = mag[I]
        Ti.sn = sn[I]
        xx = np.array([lo,hi])
        zp = zps[band]
        plt.plot(xx, 10.**(zp - 0.4*xx), '-', alpha=0.4, color=cc)
        plt.plot(xx, 10.**(zp - 0.4*xx), 'k-', alpha=0.4)
        
        p = plt.semilogy(Ti.mag, Ti.sn, '.', color=cc, alpha=0.5)

        lp.append(p[0])
        lt.append('%s band' % band)
        
    plt.xlabel('mag')
    plt.ylabel('Flux Signal-to-Noise')
    tt = [1,3,5,10,30,50,100,300,500,1000,3000,5000,10000,30000,50000]
    plt.yticks(tt, ['%i' % t for t in tt])
    plt.axhline(5., color='k')
    plt.xlim(lo,hi)
    # ylo,yhi = plt.ylim()
    # plt.ylim(1, yhi)
    plt.ylim(1, 1e5)
    plt.legend(lp, lt, loc='upper right')
    plt.title('CFHT depth (extended sources)')
    ps.savefig()

    plt.subplots_adjust(hspace=0.)
    plt.clf()
    lp,lt = [],[]
    for i,band in enumerate(bands):
        plt.subplot(3,1,i+1)
        sn = (T2.get('decam_%s_nanomaggies' % band) * 
              np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band)))
        mag = T2.get('decam_%s_mag' % band)
        cc = ccmap[band]
        I = (np.isfinite(mag) * (T2.type != 'S') * np.isfinite(sn))
        Ti = T2[I]
        Ti.mag = mag[I]
        Ti.sn = sn[I]
        xx = np.array([lo,hi])
        zp = zps[band]
        plt.plot(xx, [1.,1.], '-', alpha=0.4, color=cc)
        plt.plot(xx, [1.,1.], 'k-', alpha=0.4)

        plt.plot(xx, [0.4,0.4],   'k-', alpha=0.1)
        plt.plot(xx, [0.16,0.16], 'k-', alpha=0.1)
        
        p = plt.semilogy(Ti.mag, Ti.sn / 10.**(zp - 0.4*Ti.mag),
                         '.', color=cc, alpha=0.5)
        lp.append(p[0])
        lt.append('%s band' % band)

        if i == 1:
            plt.ylabel('Flux Signal-to-Noise vs Point Source')
        if i != 2:
            plt.xticks([])
            
        plt.ylim(0.08, 1.2)
        plt.xlim(lo,hi)
        
    plt.xlabel('mag')
    #tt = [1,3,5,10,30,50,100,300,500,1000,3000,5000,10000,30000,50000]
    #plt.yticks(tt, ['%i' % t for t in tt])
    plt.figlegend(lp, lt, loc='upper right')
    plt.suptitle('CFHT depth (extended sources)')
    ps.savefig()

    plt.clf()
    lp,lt = [],[]
    for i,band in enumerate(bands):
        sn = (T2.get('decam_%s_nanomaggies' % band) * 
              np.sqrt(T2.get('decam_%s_nanomaggies_invvar' % band)))
        mag = T2.get('decam_%s_mag' % band)
        cc = ccmap[band]
        I = (np.isfinite(mag) * (T2.type != 'S') * np.isfinite(sn))
        Ti = T2[I]
        Ti.mag = mag[I]
        Ti.sn = sn[I]
        zp = zps[band]
        snloss = Ti.sn / 10.**(zp - 0.4*Ti.mag)

        for J,name,style,deV in [
                (np.flatnonzero(Ti.type == 'D'), 'deV', 'x', True),
                (np.flatnonzero(Ti.type == 'E'), 'exp', 'o', False),
                (np.flatnonzero((Ti.type == 'C') * (Ti.fracDev > 0.5)),
                 'comp/deV', 's', True),
                (np.flatnonzero((Ti.type == 'C') * (Ti.fracDev <= 0.5)),
                 'comp/exp', '^', False),
                ]:
            print(len(J), name)
            if deV:
                size = Ti.shapeDev[:,0]
            else:
                size = Ti.shapeExp[:,0]

            ylo = 5e-2
            p = plt.loglog(np.clip(size[J], 1e-2, 1e3),
                           np.clip(snloss[J], ylo, 10.),
                           style, color=cc, mfc='none', mec=cc, alpha=0.5)

            if i == 0:
                lp.append(p[0])
                lt.append(name)

    plt.axhline(0.4**0, color='k', alpha=0.1)
    plt.axhline(0.4**1, color='k', alpha=0.1)
    plt.axhline(0.4**2, color='k', alpha=0.1)
    plt.axhline(0.4**3, color='k', alpha=0.1)
                
    plt.xlim(0.9*1e-2, 1.1*1e3)
    plt.ylim(0.9*ylo, 1.2)
    plt.xlabel('Galaxy effective radius (arcsec)')
    plt.ylabel('S/N loss vs point source')
    plt.legend(lp,lt, loc='upper right')
    plt.title('CFHT extended sources: S/N vs size')
    ps.savefig()
        
    
def get_ccd_list():
    expnums = [ 1080306, 1080946, 1168106, 1617879 ]

    #seeings =  [ 0.58, 0.67, 0.76, 0.61 ]
    seeings =   [ 0.61, 0.69, 0.79, 0.63 ]

    # 1168106 | 10AQ01 Feb 21 04:21:30 10 | P03 NGVS+2-1   | 12:39:58.2 11:03:49 2000 |  582 | u | 1.06 | 0.76 0.79   180 |P 1 V D|
    # 1080946 | 09AQ09 May 24 23:36:20 09 | P03 NGVS+2-1   | 12:39:55.2 11:02:49 2000 |  634 | g | 1.28 | 0.67 0.69  1191 |P 1 V Q|
    # 1617879 | 13AQ05 Apr 18 22:43:46 13 | P03 NGVS+2-1   | 12:39:57.2 11:01:49 2000 |  445 | r | 1.02 | 0.61 0.63  2240 |P 1 V D|
    # 1080306 | 09AQ09 May 19 21:35:36 09 | P03 NGVS+2-1   | 12:39:55.2 11:02:49 2000 |  411 | i | 1.01 | 0.58 0.61  1923 |P 1 V Q|

    imfns = [ 'cfht/%ip.fits' % i for i in expnums ]
    T = fits_table()
    T.cpimage = []
    T.cpimage_hdu = []
    T.filter = []
    T.exptime = []
    T.ra = []
    T.dec = []
    T.width = []
    T.height = []
    T.expnum = []
    T.extname = []
    T.calname = []
    T.seeing = []
    T.pixscale = []
    T.crval1 = []
    T.crval2 = []
    T.crpix1 = []
    T.crpix2 = []
    T.cd1_1 = []
    T.cd1_2 = []
    T.cd2_1 = []
    T.cd2_2 = []
    for i,(fn,expnum,seeing) in enumerate(zip(imfns, expnums, seeings)):
        F = fitsio.FITS(fn)
        primhdr = F[0].read_header()
        filter = primhdr['FILTER'].split('.')[0]
        exptime = primhdr['EXPTIME']
        pixscale = primhdr['PIXSCAL1'] / 3600.
        print('Pixscale:', pixscale * 3600, 'arcsec/pix')

        for hdu in range(1, len(F)):
        #for hdu in [13]:
            hdr = F[hdu].read_header()
            T.cpimage.append(fn)
            T.cpimage_hdu.append(hdu)
            T.filter.append(filter)
            T.exptime.append(exptime)
            args = []
            for k in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2',
                      'CD2_1', 'CD2_2' ]:
                val = hdr[k]
                T.get(k.lower()).append(val)
                args.append(val)
            wcs = Tan(*(args + [hdr[k] for k in ['NAXIS1', 'NAXIS2']]))
            print('WCS pixscale', wcs.pixel_scale())
            W,H = wcs.get_width(), wcs.get_height()
            ra,dec = wcs.radec_center()
            T.ra.append(ra)
            T.dec.append(dec)
            T.width.append(W)
            T.height.append(H)
            T.seeing.append(seeing)
            T.expnum.append(expnum)
            extname = hdr['EXTNAME']
            T.extname.append(extname)
            T.calname.append('cfht/%i/cfht-%i-%s' % (expnum, expnum, extname))
            T.pixscale.append(pixscale)
    T._length = len(T.cpimage)
    T.to_np_arrays()
    return T
    



class CfhtImage(DecamImage):
    def read_invvar(self, slice=None, **kwargs):
        print('CFHT read_invvar')
        info = self.get_image_info()
        H,W = info['dims']
        ### FIXME!
        iv = np.ones((H,W), np.float32)
        if slice is not None:
            iv = iv[slice]
        return iv

    def read_image(self, header=False, **kwargs):
        X = super(CfhtImage, self).read_image(header=header, **kwargs)
        if header:
            img,hdr = X
            return img.astype(np.float32),hdr
        else:
            img = X
            return img.astype(np.float32)

    def run_calibs(self, ra, dec, pixscale, W=2112, H=4644, se=True,
                   astrom=True, psfex=True, psfexfit=True):
        '''
        pixscale: in degrees/pixel
        '''
        for fn in [self.wcsfn,self.sefn,self.psffn,self.psffitfn]:
            print('exists?', os.path.exists(fn), fn)
        self.makedirs()
    
        run_fcopy = False
        run_se = False
        run_astrom = False
        run_psfex = False
        run_psfexfit = False

        sedir = 'NGVS-g-Single'
    
        if not all([os.path.exists(fn) for fn in [self.sefn]]):
            run_se = True
            run_fcopy = True
        if not all([os.path.exists(fn) for fn in [self.wcsfn,self.corrfn,self.sdssfn]]):
            run_astrom = True
        if not os.path.exists(self.psffn):
            run_psfex = True
        if not os.path.exists(self.psffitfn):
            run_psfexfit = True

        if run_fcopy and (run_se and se):
            tmpimgfn  = create_temp(suffix='.fits')
            cmd = 'imcopy %s"+%i" %s' % (self.imgfn, self.hdu, tmpimgfn)
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
        if run_astrom or run_se:
            # grab header values...
            primhdr = self.read_image_primary_header()
            hdr     = self.read_image_header()
            magzp = hdr['PHOT_C'] + 2.5 * np.log10(hdr['EXPTIME'])
            seeing = self.seeing
            print('Seeing', seeing, 'arcsec')
    
        if run_se and se:
            #'-SEEING_FWHM %f' % seeing,
            #'-PIXEL_SCALE 0',
            #'-PIXEL_SCALE %f' % (pixscale * 3600),
            #'-MAG_ZEROPOINT %f' % magzp,
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'psfex.sex'),
                '-PARAMETERS_NAME', os.path.join(sedir, 'psfex.param'),
                '-FILTER_NAME', os.path.join(sedir, 'default.conv'),
                '-CATALOG_NAME', self.sefn,
                tmpimgfn])
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_astrom and astrom:
            cmd = ' '.join([
                'solve-field --config', an_config, '-D . --temp-dir', tempdir,
                '--ra %f --dec %f' % (ra,dec), '--radius 1',
                '-L %f -H %f -u app' % (0.9 * pixscale * 3600, 1.1 * pixscale * 3600),
                '--continue --no-plots --no-remove-lines --uniformize 0',
                '--no-fits2fits',
                '-X x_image -Y y_image -s flux_auto --extension 2',
                '--width %i --height %i' % (W,H),
                '--crpix-center',
                '-N none -U none -S none -M none --rdls', self.sdssfn,
                '--corr', self.corrfn, '--wcs', self.wcsfn, 
                '--temp-axy', '--tag-all', self.sefn])
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_psfex and psfex:
            cmd = ('psfex -c %s -PSF_DIR %s %s -NTHREADS 1' %
                   (os.path.join(sedir, 'gradmap.psfex'),
                    os.path.dirname(self.psffn), self.sefn))
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)

            dotpsf = self.psffn.replace('.fits', '.psf')
            if os.path.exists(dotpsf) and not os.path.exists(self.psffn):
                cmd = 'mv %s %s' % (dotpsf, self.psffn)
                print(cmd)
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)
                
    
        if run_psfexfit and psfexfit:
            print('Fit PSF...')
    
            from tractor.basics import GaussianMixtureEllipsePSF, GaussianMixturePSF
            from tractor.psfex import PsfEx
    
            iminfo = self.get_image_info()
            #print 'img:', iminfo
            H,W = iminfo['dims']
            psfex = PsfEx(self.psffn, W, H, ny=13, nx=7,
                          psfClass=GaussianMixtureEllipsePSF)
            psfex.savesplinedata = True
            print('Fitting MoG model to PsfEx')
            psfex._fitParamGrid(damp=1)
            pp,XX,YY = psfex.splinedata
    
            # Convert to GaussianMixturePSF
            ppvar = np.zeros_like(pp)
            for iy in range(psfex.ny):
                for ix in range(psfex.nx):
                    psf = GaussianMixtureEllipsePSF(*pp[iy, ix, :])
                    mog = psf.toMog()
                    ppvar[iy,ix,:] = mog.getParams()
            psfexvar = PsfEx(self.psffn, W, H, ny=psfex.ny, nx=psfex.nx,
                             psfClass=GaussianMixturePSF)
            psfexvar.splinedata = (ppvar, XX, YY)
            psfexvar.toFits(self.psffitfn, merge=True)
            print('Wrote', self.psffitfn)
            
    




if __name__ == '__main__':
    main()
    
