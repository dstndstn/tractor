import matplotlib
matplotlib.use('Agg')
import pylab as plt

from scipy.ndimage.morphology import *
from scipy.ndimage.measurements import *
from scipy.ndimage.filters import *

import fitsio

from astrometry.util.fits import *
from astrometry.util.plotutils import *

import os
os.environ['DECALS_DIR'] = 'decals-lsb'

from common import *
from desi_common import *

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


def stage_1():
    ps = PlotSequence('lsb')
    
    
    decals = Decals()
    C = decals.find_ccds(expnum=431202, extname='S19')
    print len(C), 'CCDs'
    im = DecamImage(C[0])
    print 'im', im
    
    #(x0,x1,y0,y1) = opt.zoom
    #zoomslice = (slice(y0,y1), slice(x0,x1))
    zoomslice = None
    
    tim = im.get_tractor_image(decals, const2psf=True, pvwcs=True, slc=zoomslice) #, nanomaggies=False)
    print 'Tim', tim
    
    cats = []
    for b in ['1864p102', '1862p102']:
        fn = os.path.join(decals.decals_dir, 'tractor', b[:3],
                          'tractor-%s.fits' % b)
        print 'Reading', fn
        cats.append(fits_table(fn))
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
    
    print len(cat), 'catalog objects'
    
    plt.clf()
    plt.imshow(tim.getImage(), **tim.ima)
    plt.title('Orig data')
    ps.savefig()
    
    # Mask out bright pixels.
    #mask = (tim.getImage() > 50. * tim.sig1)
    #mask = binary_dilation(mask, iterations=20)

    mask = np.zeros(tim.shape, np.bool)
    
    bright = fits_table('bright-virgo.fits')
    ok,bx,by = tim.subwcs.radec2pixelxy(bright.ra, bright.dec)
    bx = np.round(bx).astype(int)
    by = np.round(by).astype(int)
    
    H,W = mask.shape
    bright.modelmag = 22.5 - 2.5*np.log10(bright.modelflux)
    mag = bright.modelmag[:,2]
    radius = (10. ** (3.5 - 0.15 * mag) / 0.27).astype(np.int)

    I = np.flatnonzero(
        (radius > 0) *
        (bx + radius > 0) * (bx - radius < W) *
        (by + radius > 0) * (by - radius < H))

    xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    for x,y,r in zip(bx[I], by[I], radius[I]):
        #mask[max(y-r,0):min(y+r+1, H),
        #    max(x-r,0):min(x+r+1, W)] = True
        mask[(xx - x)**2 + (yy - y)**2 < r**2] = True
    # log(r) = 3.5 - 0.15 * m

    mask[tim.inverr == 0] = True
    
    tim.inverr[mask] = 0.
    tim.data[mask] = 0.
    
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
    
    print 'Model median:', np.median(mod)
    rawimg = fitsio.read('decals-lsb/images/decam/CP20150407/c4d_150410_035040_ooi_z_v1.fits.fz', ext=im.hdu)
    print 'Image median:', np.median(rawimg)
    
    print 'mid sky', tim.midsky
    rawmod = mod * tim.zpscale + tim.midsky
    print 'Model median:', np.median(rawmod)
    
    fitsio.write('model.fits', rawmod, clobber=True)
    
    plt.clf()
    plt.imshow(mod, **tim.ima)
    plt.title('Model')
    ps.savefig()
    
    mod[mask] = 0.
    
    plt.clf()
    plt.imshow(mod, **tim.ima)
    plt.title('Masked model')
    ps.savefig()
    
    # ax = plt.axis()
    # plt.plot(T.x, T.y, 'r.')
    # plt.axis(ax)
    # ps.savefig()
    
    imchi = dict(interpolation='nearest', origin='lower', vmin=-5, vmax=5, cmap='RdBu')
    
    plt.clf()
    plt.imshow((tim.getImage() - mod) * tim.getInvError(), **imchi)
    plt.title('Chi')
    #ps.savefig()
    
    plt.colorbar()
    ps.savefig()
    
    plt.clf()
    plt.imshow((tim.getImage() - mod), **tim.ima)
    plt.title('Residuals')
    ps.savefig()
    
    resid = tim.getImage() - mod
    
    smoo = np.zeros_like(resid)
    median_smooth(resid, mask, 256, smoo)
    
    plt.clf()
    plt.imshow(smoo, **tim.ima)
    plt.title('Smoothed residuals (sky)')
    ps.savefig()

    resid -= smoo
    # Re-apply mask
    resid[mask] = 0.
    
    plt.clf()
    plt.imshow(resid, **tim.ima)
    plt.title('Residual - sky')
    ps.savefig()
    
    return dict(resid=resid, sky=smoo, ps=ps, tim=tim,
                tr=tr, mod=mod, mask=mask,
        orig_catalog = orig_catalog)

######

def stage_2(resid=None, sky=None, ps=None, tim=None, mask=None,
            **kwa):

    # Run a detection filter -- smooth by PSF and high threshold.
    
    psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
    det = gaussian_filter(resid, tim.psf_sigma) / psfnorm**2
    detsig1 = tim.sig1 / psfnorm

    thresh = 10. * detsig1

    hot = (det > thresh)
    growsize = int(tim.psf_fwhm * 2)
    hot = binary_dilation(hot, iterations=growsize)

    plt.clf()
    plt.imshow(resid * hot, **tim.ima)
    plt.title('Detected sources')
    ps.savefig()

    mask[hot] = True
    resid[mask] = 0.
    
    plt.clf()
    plt.imshow(resid, **tim.ima)
    plt.title('Sources masked')
    ps.savefig()

    
def stage_3(resid=None, sky=None, ps=None, tim=None, mask=None,
            **kwa):

    radii_arcsec = [1., 2., 4., 6., 8., 10., 13., 16.,
                    20., 25, 32., 45., 60., 120.]

    filters = []

    from scipy import signal

    keep = np.logical_not(mask)
    
    plt.clf()
    plt.subplot(2,1,1)
    plt.hist(resid[keep] / tim.sig1, 50, range=(-8,8))
    plt.subplot(2,1,2)
    plt.hist(resid[keep] / tim.sig1, 50, range=(-8,8), log=True)
    ps.savefig()
    
    for i,r in enumerate(radii_arcsec):
        sigma = r / 0.27
        print 'Filtering at', r

        # k1 = signal.gaussian(r*10, r)
        # k1 /= k1.sum()
        # print 'Kernel 1:', k1.shape
        # k1 = np.atleast_2d(k1)
        # print 'Kernel 1:', k1.shape
        # 
        # blurred = signal.fftconvolve(resid, k1, mode='same')
        # blurred = signal.fftconvolve(blurred, k1.T, mode='same')
        #ksum = kernel.sum()
        
        kernel = np.outer(signal.gaussian(sigma*10, sigma),
                          signal.gaussian(sigma*10, sigma))
        kernel /= kernel.sum()
        blurred = signal.fftconvolve(resid, kernel, mode='same')

        #kernel_norm = 1./(2. * np.sqrt(np.pi) * sigma)
        #print 'r=', r, 'sigma=', sigma, 'kernel sum=', ksum, 'kernel norm=', kernel_norm
        #blurred /= kernel_norm**2
        #sn = blurred / (tim.sig1 / kernel_norm)

        # Source has profile:
        #
        # s = ftotal * N(0, s^2)
        #
        # Convolve by N(0, sigma^2)
        #
        # ->   c = ftotal * N(0, s^2 + sigma^2)

        knorm = np.sqrt(np.sum(kernel**2))
        print 'Knorm', knorm
        print 'knorm', 1./(2. * np.sqrt(np.pi) * sigma)

        sn = blurred / (knorm * tim.sig1)
        
        # plt.clf()
        # plt.subplot(2,1,1)
        # plt.hist(sn[keep], 50, range=(-8,8))
        # plt.subplot(2,1,2)
        # plt.hist(sn[keep], 50, range=(-8,8), log=True)
        # ps.savefig()
        #blurred = blurred.astype(np.float32)
        
        #filters.append(gaussian_filter(resid, sigma,
        #        mode='constant'))
        #filters.append(blurred)

        print 'Max S/N:', sn.max()
        
        plt.clf()
        plt.imshow(sn, interpolation='nearest', origin='lower',
                   vmin=-2., vmax=32.,
                   cmap='gray')
        plt.title('Filtered at %f arcsec: S/N' % r)
        ps.savefig()

        filters.append(sn.astype(np.float32))
        
        

    return dict(filters=filters, radii_arcsec=radii_arcsec)

def stage_4(resid=None, sky=None, ps=None, tim=None,
            filters=None, radii_arcsec=None,
            orig_catalog=None,
            **kwa):

    # for f,r in zip(filters, radii_arcsec):
    #     plt.clf()
    #     #plt.imshow(f, **tim.ima)
    #     #dimshow(f, vmin=np.percentile(f, 25),
    #     #        vmax=np.percentile(f, 99))
    #     plt.imshow(f, interpolation='nearest', origin='lower',
    #                vmin=-1.*tim.sig1, vmax=5.*tim.sig1,
    #         cmap='gray')
    #     plt.title('Filtered at %f arcsec' % r)
    #     ps.savefig()

    fstack = np.dstack(filters)
    print 'fstack shape', fstack.shape

    amax = np.argmax(fstack, axis=2)
    fmax = np.max(fstack, axis=2)

    print 'fmax shape', fmax.shape
    print 'amax shape', amax.shape, amax.dtype
    
    sna = dict(interpolation='nearest', origin='lower',
               vmin=-2., vmax=32.,
               cmap='gray')
    
    plt.clf()
    plt.imshow(fmax, **sna)
    plt.title('Filter max')
    ps.savefig()

    from matplotlib.ticker import FixedFormatter

    #def get_formatter(mx):
    #    return FixedFormatter(['%i' % r for r in radii_arcsec])
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
    print 'blobs max', blobs.max()
    peaks = []
    for blob in range(1, nblobs+1):
        I = (blobs == blob)
        imax = np.argmax(fmax * I)
        py,px = np.unravel_index(imax, fmax.shape)
        peaks.append((px,py))
        print 'blob imax', imax
        print 'px,py', px,py
        print 'blob max', fmax.flat[imax]
        ifilt = amax.flat[imax]
        print 'blob argmax', ifilt
        print '  =', amax[py,px]
        
        peak_amax[I] = ifilt

    px = [x for x,y in peaks]
    py = [y for x,y in peaks]
        
    plt.clf()
    plt.imshow(amax * hot, interpolation='nearest', origin='lower',
               cmap='jet')
    ax = plt.axis()
    plt.plot(px, py, '+', color='w')
    plt.axis(ax)
    plt.title('Filter argmax')
    plt.colorbar(ticks=np.arange((amax*hot).max()+1), format=radformat)
    ps.savefig()

    # Extended sources in the catalog
    E = orig_catalog[np.maximum(orig_catalog.shapeexp_r, orig_catalog.shapedev_r) >= 3.]

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
        sigma = r / 0.27
        knorm = 1./(2. * np.sqrt(np.pi) * sigma)
        blurred = sn * (knorm * tim.sig1)
        fluxest = blurred * 2.*np.pi * (2. * sigma**2)
        fluxes.append(fluxest)

    fluxes = np.array(fluxes)
    mags = 22.5 - 2.5 * np.log10(fluxes)
        
    plt.clf()
    plt.imshow(peak_amax, interpolation='nearest', origin='lower',
               cmap='jet')
    ax = plt.axis()
    plt.plot(px, py, '+', color='w', ms=10, mew=2)
    plt.plot(E.x, E.y, 'x', color='k', ms=10, mew=2)

    for x,y,m in zip(px, py, mags):
        ra,dec = tim.subwcs.pixelxy2radec(x+1, y+1)
        plt.text(x, y, '%.1f (%.2f,%.2f)' % (m,ra,dec), color='w', ha='left', fontsize=12)

    plt.axis(ax)
    plt.title('Filter peak argmax')
    plt.colorbar(ticks=np.arange((peak_amax*hot).max()+1), format=radformat)
    ps.savefig()


    LSB = fits_table()
    LSB.x = np.array(px)
    LSB.y = np.array(py)
    ra,dec = tim.subwcs.pixelxy2radec(LSB.x, LSB.y)
    LSB.ra = ra
    LSB.dec = dec
    LSB.flux = fluxes
    LSB.mag = mags
    LSB.radius = np.array(radii_arcsec)[amax[py,px]]
    LSB.sn = fmax[py,px]

    LSB.cut(np.argsort(-LSB.sn))
    
    LSB.writeto('lsb.fits')
    
    # for f1,f2,r1,r2 in zip(filters, filters[1:],
    #                        radii_arcsec, radii_arcsec[1:]):
    #     plt.clf()
    #     plt.imshow(f2-f1,
    #                interpolation='nearest', origin='lower',
    #                vmin=-1.*tim.sig1, vmax=5.*tim.sig1,
    #                cmap='gray')
    #     plt.title('Filtered at %.1f - %.1f arcsec' %
    #               (r2, r1))
    #     ps.savefig()

        
        
    # bin = bin_image(resid, 8)
    # 
    # plt.clf()
    # dimshow(bin,
    #         vmin=np.percentile(bin,25),
    #         vmax=np.percentile(bin,99))
    # plt.title('Binned by 8')
    # ps.savefig()
    # 
    # bs = gaussian_filter(bin, 25)
    # 
    # plt.clf()
    # dimshow(bs,
    #         vmin=np.percentile(bs,25),
    #         vmax=np.percentile(bs,99))
    # plt.title('Binned by 8, Gaussian smoothed')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.imshow(tim.getImage(), **tim.ima)
    # ax = plt.axis()
    # plt.plot(bx, by, 'r+', mew=2, ms=10)
    # plt.axis(ax)
    # plt.title('SDSS Bright Stars')
    # ps.savefig()

    return dict(LSB=LSB, filters=None, mod=None, sky=None)

def stage_5(LSB=None, resid=None, tim=None, mask=None, ps=None, **kwa):

    print 'resid', resid.dtype
    
    # For each detected LSB source, try to fit it using the tractor...
    orig_image = tim.getImage()
    tim.data = resid
    tim.inverr[mask] = 0.

    from tractor import *

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
opt,args = parser.parse_args()

stagefunc = CallGlobal('stage_%s', globals())

kwargs = {}

if len(opt.stage) == 0:
    opt.stage.append('3')
opt.force.extend(opt.stage)
if opt.forceall:
    kwargs.update(forceall=True)

for s in opt.stage:
    runstage(s, 'lsb-%(stage)s.pickle', stagefunc,
             prereqs={ '5':'4', '4':'3', '3':'2', '2':'1', '1':None },
             force=opt.force, write=opt.write)
