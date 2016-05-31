import matplotlib
matplotlib.use('Agg')
import pylab as plt

from scipy.ndimage.morphology import *
from scipy.ndimage.measurements import *
from scipy.ndimage.filters import *
from scipy.interpolate import RectBivariateSpline

import fitsio

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.ttime import *

from tractor import *

import os
os.environ['DECALS_DIR'] = 'decals-lsb'

from legacypipe.common import *
from legacypipe.desi_common import *

def bin_image(data, S):
    # rebin image data
    H,W = data.shape
    sH,sW = (H+S-1)/S, (W+S-1)/S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    count = np.zeros((sH,sW), int)
    for i in range(S):
        for j in range(S):
            sub = data[i::S, j::S]
            subh,subw = sub.shape
            newdata[:subh,:subw] += sub
            count[:subh,:subw] += 1
    newdata /= count
    return newdata

def bin_image_2(data, S):
    # rebin image data, padding with zeros
    H,W = data.shape
    sH,sW = (H+S-1)/S, (W+S-1)/S
    newdata = np.zeros((sH,sW), dtype=data.dtype)
    for i in range(S):
        for j in range(S):
            sub = data[i::S, j::S]
            subh,subw = sub.shape
            newdata[:subh,:subw] += sub
    return newdata / (S*S)



def stage_1(expnum=431202, extname='S19', plotprefix='lsb', plots=False,
            brightstars = 'bright.fits',
            pixscale=0.27,
            **kwa):
    if plots:
        ps = PlotSequence(plotprefix)
    else:
        ps = None

    survey = LegacySurveyData()
    C = survey.find_ccds(expnum=expnum, ccdname=extname)
    print len(C), 'CCDs'
    im = survey.get_image_object(C[0])
    print 'im', im
    
    #(x0,x1,y0,y1) = opt.zoom
    #zoomslice = (slice(y0,y1), slice(x0,x1))
    zoomslice = None
    
    tim = im.get_tractor_image(gaussPsf=True, splinesky=True, slc=zoomslice)
    print 'Tim', tim
    
    cats = []
    bricks = bricks_touching_wcs(tim.subwcs, survey=survey)
    bricknames = bricks.brickname
    for b in bricknames:
        fn = survey.find_file('tractor', brick=b)
        if not os.path.exists(fn):
            print 'WARNING: file does not exist:', fn
            continue
        print 'Reading', fn
        cat = fits_table(fn)
        print 'Read', len(cat), 'sources'
        if cat is None or len(cat) == 0:
            continue
        cats.append(cat)
    if len(cats):
        T = merge_tables(cats)
        T._header = cats[0]._header
        
        # margin
        M = 20
        ok,x,y = tim.subwcs.radec2pixelxy(T.ra, T.dec)
        x -= 1.
        y -= 1.
        T.x = x
        T.y = y
        H,W = tim.shape
        T.cut((x > -M) * (x < (W+M)) * (y > -M) * (y < (H+M)))
        print 'Cut to', len(T), 'within image bounds'
    
        T.cut(T.brick_primary)
        print 'Cut to', len(T), 'brick_primary'
        T.cut((T.out_of_bounds == False) * (T.left_blob == False))
        print 'Cut to', len(T), 'not out-of-bound or left-blob'
        
        T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
        T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T
        
        print 'Brightest z-band:', np.max(T.decam_flux[:,4])
        print 'Brightest r-band:', np.max(T.decam_flux[:,2])
    
        orig_catalog = T.copy()
        
        # Cut to compact sources
        T.cut(np.maximum(T.shapeexp_r, T.shapedev_r) < 3.)
        print 'Cut to', len(T), 'compact'
        
        cat = read_fits_catalog(T)
    else:
        cat = []
        orig_catalog = fits_table()
    
    print len(cat), 'catalog objects'
    
    if plots:
        plt.clf()
        plt.imshow(tim.getImage(), **tim.ima)
        plt.title('Orig data')
        ps.savefig()
    
    # Mask out bright pixels.
    mask = np.zeros(tim.shape, np.bool)
    bright = fits_table(brightstars)
    print 'Read', len(bright), 'SDSS bright stars'
    ok,bx,by = tim.subwcs.radec2pixelxy(bright.ra, bright.dec)
    bx = np.round(bx).astype(int)
    by = np.round(by).astype(int)
    
    H,W = mask.shape
    bright.modelmag = 22.5 - 2.5*np.log10(bright.modelflux)
    mag = bright.modelmag[:,2]
    radius = (10. ** (3.5 - 0.15 * mag) / pixscale).astype(np.int)

    I = np.flatnonzero(
        ok *
        (radius > 0) *
        (bx + radius > 0) * (bx - radius < W) *
        (by + radius > 0) * (by - radius < H))
    print len(I), 'bright stars are near the image'

    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    for x,y,r in zip(bx[I], by[I], radius[I]):
        mask[(xx - x)**2 + (yy - y)**2 < r**2] = True

    mask[tim.inverr == 0] = True
    tim.inverr[mask] = 0.
    tim.data[mask] = 0.
    
    if plots:
        plt.clf()
        plt.imshow(mask, interpolation='nearest', origin='lower', vmin=0, vmax=1, cmap='gray')
        plt.title('Mask')
        ps.savefig()
    
        plt.clf()
        plt.imshow(tim.getImage(), **tim.ima)
        plt.title('Masked')
        ps.savefig()
    
    tr = Tractor([tim], cat)
    mod = tr.getModelImage(tim)
    
    if False:
        # OLD DEBUGGING
        print 'Model median:', np.median(mod)
        rawimg = fitsio.read('decals-lsb/images/decam/CP20150407/c4d_150410_035040_ooi_z_v1.fits.fz', ext=im.hdu)
        print 'Image median:', np.median(rawimg)
        print 'mid sky', tim.midsky
        rawmod = mod * tim.zpscale + tim.midsky
        print 'Model median:', np.median(rawmod)
        fitsio.write('model.fits', rawmod, clobber=True)
    
    if plots:
        plt.clf()
        plt.imshow(mod, **tim.ima)
        plt.title('Model')
        ps.savefig()
    
    mod[mask] = 0.
    
    if plots:
        plt.clf()
        plt.imshow(mod, **tim.ima)
        plt.title('Masked model')
        ps.savefig()
    
        imchi = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5, cmap='RdBu')

        # plt.clf()
        # plt.imshow((tim.getImage() - mod) * tim.getInvError(), **imchi)
        # plt.title('Chi')
        # plt.colorbar()
        # ps.savefig()
    
        plt.clf()
        plt.imshow((tim.getImage() - mod), **tim.ima)
        plt.title('Residuals')
        ps.savefig()
    
    resid = tim.getImage() - mod
    
    sky = np.zeros_like(resid)
    median_smooth(resid, mask, 256, sky)

    if plots:
        plt.clf()
        plt.imshow(sky, **tim.ima)
        plt.title('Smoothed residuals (sky)')
        ps.savefig()

    resid -= sky
    # Re-apply mask
    resid[mask] = 0.
    
    if plots:
        plt.clf()
        plt.imshow(resid, **tim.ima)
        plt.title('Residual - sky')
        ps.savefig()
    
    return dict(resid=resid, sky=sky, ps=ps, tim=tim,
                tr=tr, mod=mod, mask=mask,
                orig_catalog = orig_catalog, pixscale=pixscale)


def stage_2(resid=None, sky=None, ps=None, tim=None, mask=None,
            plots=False,
            **kwa):

    # Run a detection filter -- smooth by PSF and high threshold.
    
    psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
    det = gaussian_filter(resid, tim.psf_sigma) / psfnorm**2
    detsig1 = tim.sig1 / psfnorm

    thresh = 10. * detsig1

    hot = (det > thresh)
    growsize = int(tim.psf_fwhm * 2)
    hot = binary_dilation(hot, iterations=growsize)

    if plots:
        plt.clf()
        plt.imshow(resid * hot, **tim.ima)
        plt.title('Detected sources')
        ps.savefig()

    sourcepix = (hot * resid)
        
    mask[hot] = True
    resid[mask] = 0.
    
    if plots:
        plt.clf()
        plt.imshow(resid, **tim.ima)
        plt.title('Sources masked')
        ps.savefig()

    return dict(sourcepix=sourcepix)
    
def stage_3(resid=None, sky=None, ps=None, tim=None, mask=None,
            plots=False, plotprefix=None, pixscale=None,
            **kwa):
    if plots and ps is None:
        ps = PlotSequence(plotprefix)

    radii_arcsec = np.sqrt(2.) ** np.arange(15)
    
    filters = []

    keep = np.logical_not(mask)
    
    if plots and False:
        plt.clf()
        plt.subplot(2,1,1)
        plt.hist(resid[keep] / tim.sig1, 50, range=(-8,8))
        plt.subplot(2,1,2)
        plt.hist(resid[keep] / tim.sig1, 50, range=(-8,8), log=True)
        ps.savefig()

    binning = 1
    img = resid

    for i,r in enumerate(radii_arcsec):
        sigma = r / pixscale
        print 'Filtering at', r, 'arcsec'
        if i and i%2 == 0:
            img = bin_image_2(img, 2)
            binning *= 2
            blurred = gaussian_filter(img, sigma/binning, mode='constant')
        else:
            blurred = gaussian_filter(img, sigma / binning, mode='constant')

        bh,bw = img.shape
        H,W = resid.shape
        bx = (np.arange(bw)+0.5) * binning - 0.5
        by = (np.arange(bh)+0.5) * binning - 0.5
        spline = RectBivariateSpline(bx, by, blurred.T)
        blurred = spline(np.arange(W), np.arange(H)).T
        knorm =  1./(2. * np.sqrt(np.pi) * sigma)
        sn = blurred / (knorm * tim.sig1)
        print 'Max S/N:', sn.max()
        
        if plots:
            plt.clf()
            plt.imshow(sn, interpolation='nearest', origin='lower',
                       vmin=-2., vmax=32., cmap='gray')
            plt.title('Filtered at %f arcsec: S/N' % r)
            ps.savefig()
            
        filters.append(sn.astype(np.float32))

    return dict(filters=filters, radii_arcsec=radii_arcsec)

def stage_4(resid=None, sky=None, ps=None, tim=None,
            mask=None, sourcepix=None,
            filters=None, radii_arcsec=None,
            orig_catalog=None, plots=False,
            lsbcat='lsb.fits',
            expnum=None, extname=None,
            pixscale=None,
            **kwa):

    #ok,x,y = tim.subwcs.radec2pixelxy(188.7543, 13.3847)
    #print 'X,Y', x,y
    
    fstack = np.dstack(filters)
    amax = np.argmax(fstack, axis=2)
    fmax = np.max(fstack, axis=2)

    if plots:
        sna = dict(interpolation='nearest', origin='lower',
                   vmin=-2., vmax=32.,
                   cmap='gray')
        plt.clf()
        plt.imshow(fmax, **sna)
        plt.title('Filter max')
        ps.savefig()

        from matplotlib.ticker import FixedFormatter
        radformat = FixedFormatter(['%i' % r for r in radii_arcsec])
    
        plt.clf()
        plt.imshow(amax, interpolation='nearest', origin='lower',
                   cmap='jet')
        plt.title('Filter argmax')
        plt.colorbar(ticks=np.arange(amax.max()+1), format=radformat)
        ps.savefig()

        peak_amax = np.zeros_like(amax)

    hot = (fmax > 10.)
    blobs, nblobs = label(hot)
    print 'Nblobs', nblobs
    #print 'blobs max', blobs.max()
    peaks = []
    for blob in range(1, nblobs+1):
        I = (blobs == blob)
        imax = np.argmax(fmax * I)
        py,px = np.unravel_index(imax, fmax.shape)
        peaks.append((px,py))
        #print 'blob imax', imax
        print 'px,py', px,py
        print 'blob max', fmax.flat[imax]
        ifilt = amax.flat[imax]
        print 'blob argmax', ifilt
        #print '  =', amax[py,px]
        
        if plots:
            peak_amax[I] = ifilt

    px = [x for x,y in peaks]
    py = [y for x,y in peaks]
        
    if plots:
        plt.clf()
        plt.imshow(amax * hot, interpolation='nearest', origin='lower',
                   cmap='jet')
        ax = plt.axis()
        plt.plot(px, py, '+', color='w')
        plt.axis(ax)
        plt.title('Filter argmax')
        plt.colorbar(ticks=np.arange((amax*hot).max()+1), format=radformat)
        ps.savefig()

    # Total flux estimate
    # fmax is S/N in the amax filter.  Back out...
    fluxes = []
    for x,y in zip(px,py):
        print 'peak', x,y
        ifilt = amax[y,x]
        print 'ifilt', ifilt
        sn = fmax[y,x]
        print 'S/N', sn
        r = radii_arcsec[ifilt]
        sigma = r / pixscale
        knorm = 1./(2. * np.sqrt(np.pi) * sigma)
        blurred = sn * (knorm * tim.sig1)
        fluxest = blurred * 2.*np.pi * (2. * sigma**2)
        fluxes.append(fluxest)

    fluxes = np.array(fluxes)
    mags = 22.5 - 2.5 * np.log10(fluxes)

    if plots:
        for bg in [1,2]:
            plt.clf()
            if bg == 1:
                plt.imshow(peak_amax, interpolation='nearest', origin='lower',
                           cmap='jet')
            else:
                plt.imshow(tim.getImage(), **tim.ima)
            ax = plt.axis()
            plt.plot(px, py, '+', color='w', ms=10, mew=2)
            # Extended sources in the catalog
            if len(orig_catalog):
                E = orig_catalog[np.maximum(orig_catalog.shapeexp_r, orig_catalog.shapedev_r) >= 3.]
                plt.plot(E.x, E.y, 'x', color='k', ms=10, mew=2)
    
            for x,y,m in zip(px, py, mags):
                ra,dec = tim.subwcs.pixelxy2radec(x+1, y+1)
                plt.text(x, y, '%.1f (%.2f,%.2f)' % (m,ra,dec), color='w', ha='left', fontsize=12)
    
            T = fits_table('evcc.fits')
            ok,x,y = tim.subwcs.radec2pixelxy(T.ra, T.dec)
            x = x[ok]
            y = y[ok]
            plt.plot(x, y, 'o', mec=(0,1,0), mfc='none', ms=50, mew=5)
            T = fits_table('vcc.fits')
            ok,x,y = tim.subwcs.radec2pixelxy(T.ra, T.dec)
            x = x[ok]
            y = y[ok]
            plt.plot(x, y, 'o', mec=(0,1,1), mfc='none', ms=50, mew=5)
    
            plt.axis(ax)
            plt.title('Peaks')
            #plt.title('Filter peak argmax')
            #plt.colorbar(ticks=np.arange((peak_amax*hot).max()+1), format=radformat)
            ps.savefig()

    LSB = fits_table()
    LSB.filter = np.array([tim.band] * len(px))
    LSB.expnum  = np.array([expnum ] * len(px)).astype(np.int32)
    LSB.extname = np.array([extname] * len(px))
    LSB.x = np.array(px).astype(np.int16)
    LSB.y = np.array(py).astype(np.int16)
    ra,dec = tim.subwcs.pixelxy2radec(LSB.x, LSB.y)
    LSB.ra = ra
    LSB.dec = dec
    LSB.flux = fluxes.astype(np.float32)
    LSB.mag = mags.astype(np.float32)
    LSB.radius = np.array(radii_arcsec)[amax[py,px]].astype(np.float32)
    LSB.sn = fmax[py,px].astype(np.float32)
            
    # Apertures, radii in ARCSEC.
    apertures_arcsec = np.array([0.5, 0.75, 1., 1.5, 2., 3.5, 5., 7.])
    apertures = apertures_arcsec / pixscale
    apxy = np.vstack((px, py)).T
    
    import photutils
    with np.errstate(divide='ignore'):
        imsigma = 1.0 / tim.getInvError()
        imsigma[tim.getInvError() == 0] = 0

    for photimg, err, name, errname in [
            (resid, imsigma, 'apflux', 'apflux_ivar'),
            (tim.getImage() - sky, None, 'apimgflux', None),
            ((tim.getImage() - sky) * mask, None, 'apmaskedimgflux', None),
            (1. - (mask * 1.), None, 'apgoodpix', None),
            (sourcepix, None, 'apsources', None),
            ]:

        apimg = []
        apimgerr = []
        for rad in apertures:
            aper = photutils.CircularAperture(apxy, rad)
            p = photutils.aperture_photometry(photimg, aper, error=err)
            apimg.append(p.field('aperture_sum'))
            if err is not None:
                apimgerr.append(p.field('aperture_sum_err'))
        ap = np.vstack(apimg).T
        ap[np.logical_not(np.isfinite(ap))] = 0.
        LSB.set(name, ap.astype(np.float32))
        if err is not None:
            apiv = 1./(np.vstack(apimgerr).T)**2
            apiv[np.logical_not(np.isfinite(apiv))] = 0.
            LSB.set(errname, apiv.astype(np.float32))

    LSB.cut(np.argsort(-LSB.sn))
    LSB.writeto(lsbcat)
    
    return dict(LSB=LSB, filters=None, mod=None, sky=None)

def stage_5(LSB=None, resid=None, tim=None, mask=None, ps=None, **kwa):

    print 'resid', resid.dtype
    
    # For each detected LSB source, try to fit it using the tractor...
    orig_image = tim.getImage()
    tim.data = resid
    tim.inverr[mask] = 0.

    lsba = dict(interpolation='nearest', origin='lower', vmin=0,
                vmax=0.5*tim.sig1, cmap='gray')
    chia = dict(interpolation='nearest', origin='lower', vmin=-3.,
                vmax=3., cmap='RdBu')

    plt.clf()
    plt.imshow(tim.getImage(), **lsba)
    plt.title('Data for LSB fits')
    ps.savefig()
    
    for i,lsb in enumerate(LSB):
        source = ExpGalaxy(RaDecPos(lsb.ra, lsb.dec),
                           NanoMaggies(**{ tim.band: lsb.flux }),
                           EllipseESoft.fromRAbPhi(lsb.radius * 2.35/2., 1., 0.))
        print 'Source:', source

        #EllipseE(lsb.radius * 2.35 / 2., 0., 0.)

        tr = Tractor([tim], [source])
        tr.freezeParam('images')
        mod = tr.getModelImage(0)

        plt.clf()
        plt.imshow(mod, **lsba)
        plt.title('Initial model: LSB candidate %i' % i)
        ps.savefig()

        chi0 = (tim.getImage() - mod) * tim.inverr

        while True:
            dlnp,X,alpha = tr.optimize(priors=False, shared_params=False)
            print 'dlnp:', dlnp
            print 'source:', source
            if dlnp < 1e-3:
                break

        mod = tr.getModelImage(0)

        plt.clf()
        plt.imshow(mod, **lsba)
        plt.title('Final model: LSB candidate %i' % i)
        ps.savefig()

        plt.clf()
        plt.imshow(mod, **tim.ima)
        plt.title('Final model: LSB candidate %i' % i)
        ps.savefig()
        
        plt.clf()
        plt.imshow(chi0, **chia)
        plt.title('Initial chi: LSB candidate %i' % i)
        ps.savefig()
        
        chi = (tim.getImage() - mod) * tim.inverr
        plt.clf()
        plt.imshow(chi, **chia)
        plt.title('Final chi: LSB candidate %i' % i)
        ps.savefig()

        if i == 1:
            break
        

from astrometry.util.stages import *

plt.figure(figsize=(8,16))

import optparse
parser = optparse.OptionParser()
parser.add_option('-f', '--force-stage', dest='force', action='append', default=[],
                  help="Force re-running the given stage(s) -- don't read from pickle.")
parser.add_option('-F', '--force-all', dest='forceall', action='store_true',
                  help='Force all stages to run')
parser.add_option('-s', '--stage', dest='stage', default=[], action='append',
                  help="Run up to the given stage(s)")
parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')

parser.add_option('--expnum', type=int, default=431202,
                  help='CCD: Exposure number to run')
parser.add_option('--extname', default='S19',
                  help='CCD name to run')
parser.add_option('--prefix', default='lsb', help='Plot & pickle filename prefix')
parser.add_option('--plots', action='store_true', default=False,
                  help='Enable plots')
parser.add_option('--bright', default='bright.fits',
                  help='SDSS bright star filename, default %default')

parser.add_option('--out', default='lsb.fits',
                  help='Output catalog filename, default %default')

opt,args = parser.parse_args()

stagefunc = CallGlobal('stage_%s', globals())

kwargs = dict(expnum=opt.expnum, extname=opt.extname,
              plotprefix=opt.prefix, plots=opt.plots,
              brightstars=opt.bright, lsbcat=opt.out)

if len(opt.stage) == 0:
    opt.stage.append('4')
opt.force.extend(opt.stage)
if opt.forceall:
    kwargs.update(forceall=True)

for s in opt.stage:
    runstage(s, 'pickles/%s-%%(stage)s.pickle' % opt.prefix, stagefunc,
             prereqs={ '5':'4', '4':'3', '3':'2', '2':'1', '1':None },
             force=opt.force, write=opt.write, **kwargs)
