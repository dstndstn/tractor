import matplotlib
matplotlib.use('Agg')
import pylab as plt

from scipy.ndimage.morphology import *
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
    
    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T
    
    print 'Brightest z-band:', np.max(T.decam_flux[:,4])
    print 'Brightest r-band:', np.max(T.decam_flux[:,2])
    
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
    mask = (tim.getImage() > 50. * tim.sig1)
    mask = binary_dilation(mask, iterations=20)
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
    median_smooth(resid, None, 256, smoo)
    
    plt.clf()
    plt.imshow(smoo, **tim.ima)
    plt.title('Smoothed residuals')
    ps.savefig()
    
    plt.clf()
    plt.imshow(resid - smoo, **tim.ima)
    plt.title('Residual - smoothed')
    ps.savefig()
    
    ######
    
    smoo2 = np.zeros_like(resid)
    median_smooth(resid - smoo, None, 10, smoo2)
    
    plt.clf()
    plt.imshow(smoo2, **tim.ima)
    plt.title('smoothed(Residual - smoothed)')
    ps.savefig()
    
    plt.clf()
    dimshow(smoo2)
    plt.title('smoothed(Residual - smoothed)')
    ps.savefig()
    
    return dict(resid=resid, smoo=smoo, ps=ps, tim=tim,
                tr=tr, mod=mod)

######

def stage_2(resid=None, smoo=None, ps=None, tim=None,
            **kwa):

    bin = bin_image(resid - smoo, 8)

    plt.clf()
    dimshow(bin,
            vmin=np.percentile(bin,25),
            vmax=np.percentile(bin,99))
    plt.title('Binned by 8')
    ps.savefig()

    bs = gaussian_filter(bin, 25)
    
    plt.clf()
    dimshow(bs,
            vmin=np.percentile(bs,25),
            vmax=np.percentile(bs,99))
    plt.title('Binned by 8, Gaussian smoothed')
    ps.savefig()

    bright = fits_table('bright-virgo.fits')
    ok,bx,by = tim.subwcs.radec2pixelxy(bright.ra, bright.dec)
    
    plt.clf()
    plt.imshow(tim.getImage(), **tim.ima)
    ax = plt.axis()
    plt.plot(bx, by, 'r+', mew=2, ms=10)
    plt.axis(ax)
    plt.title('SDSS Bright Stars')
    ps.savefig()


from astrometry.util.stages import *

stagefunc = CallGlobal('stage_%s', globals())

runstage('2', 'lsb-%(stage)s.pickle', stagefunc,
         prereqs={ '2':'1', '1':None },
        force=['2'])
