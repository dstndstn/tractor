import os
import tempfile

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import Tan, Sip
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.miscutils import polygons_intersect, estimate_mode, clip_polygon

from tractor.basics import ConstantSky, NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.engine import get_class_from_name, Image
from tractor.psfex import PsfEx

tempdir = os.environ['TMPDIR']
decals_dir = os.environ.get('DECALS_DIR')
calibdir = os.path.join(decals_dir, 'calib', 'decam')
sedir    = os.path.join(decals_dir, 'calib', 'se-config')
an_config= os.path.join(decals_dir, 'calib', 'an-config', 'cfg')

def get_rgb(imgs, bands, mnmx=None, arcsinh=None):
    '''
    Given a list of images in the given bands, returns a scaled RGB
    image.
    '''
    bands = ''.join(bands)
    if bands == 'grz':
        scales = dict(g = (2, 0.0066),
                      r = (1, 0.01),
                      z = (0, 0.025),
                      )
    elif bands == 'gri':
        # scales = dict(g = (2, 0.004),
        #               r = (1, 0.0066),
        #               i = (0, 0.01),
        #               )
        scales = dict(g = (2, 0.002),
                      r = (1, 0.004),
                      i = (0, 0.005),
                      )
    else:
        assert(False)
        
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
    

def switch_to_soft_ellipses(cat):
    from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy
    from tractor.ellipses import EllipseESoft
    for src in cat:
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = EllipseESoft.fromEllipseE(src.shape)
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeDev = EllipseESoft.fromEllipseE(src.shapeDev)
            src.shapeExp = EllipseESoft.fromEllipseE(src.shapeExp)

def brick_catalog_for_radec_box(ralo, rahi, declo, dechi,
                                decals, catpattern, bricks=None):
    '''
    Merges multiple Tractor brick catalogs to cover an RA,Dec
    bounding-box.

    No cleverness with RA wrap-around; assumes ralo < rahi.

    decals: Decals object
    
    bricks: table of bricks, eg from Decals.get_bricks()

    catpattern: filename pattern of catalog files to read,
        eg "pipebrick-cats/tractor-phot-%06i.its"
    
    '''
    assert(ralo < rahi)
    assert(declo < dechi)

    if bricks is None:
        bricks = decals.get_bricks()
    I = decals.bricks_touching_radec_box(bricks, ralo, rahi, declo, dechi)
    print len(I), 'bricks touch RA,Dec box'
    TT = []
    hdr = None
    for i in I:
        brick = bricks[i]
        fn = catpattern % brick.brickid
        print 'Catalog', fn
        if not os.path.exists(fn):
            print 'Warning: catalog does not exist:', fn
            continue
        T = fits_table(fn, header=True)
        if T is None or len(T) == 0:
            print 'Warning: empty catalog', fn
            continue
        T.cut((T.ra  >= ralo ) * (T.ra  <= rahi) *
              (T.dec >= declo) * (T.dec <= dechi))
        TT.append(T)
    if len(TT) == 0:
        return None
    T = merge_tables(TT)
    # arbitrarily keep the first header
    T._header = TT[0]._header
    return T
    
def ccd_map_image(valmap, empty=0.):
    '''
    valmap: { 'N7' : 1., 'N8' : 17.8 }

    Returns: a numpy image (shape (12,14)) with values mapped to their CCD locations.
    '''
    img = np.empty((12,14))
    img[:,:] = empty
    for k,v in valmap.items():
        x0,x1,y0,y1 = ccd_map_extent(k)
        #img[y0+6:y1+6, x0+7:x1+7] = v
        img[y0:y1, x0:x1] = v
    return img

def ccd_map_center(extname):
    x0,x1,y0,y1 = ccd_map_extent(extname)
    return (x0+x1)/2., (y0+y1)/2.

def ccd_map_extent(extname, inset=0.):
    assert(extname.startswith('N') or extname.startswith('S'))
    num = int(extname[1:])
    assert(num >= 1 and num <= 31)
    if num <= 7:
        x0 = 7 - 2*num
        y0 = 0
    elif num <= 13:
        x0 = 6 - (num - 7)*2
        y0 = 1
    elif num <= 19:
        x0 = 6 - (num - 13)*2
        y0 = 2
    elif num <= 24:
        x0 = 5 - (num - 19)*2
        y0 = 3
    elif num <= 28:
        x0 = 4 - (num - 24)*2
        y0 = 4
    else:
        x0 = 3 - (num - 28)*2
        y0 = 5
    if extname.startswith('N'):
        (x0,x1,y0,y1) = (x0, x0+2, -y0-1, -y0)
    else:
        (x0,x1,y0,y1) = (x0, x0+2, y0, y0+1)

    # Shift from being (0,0)-centered to being aligned with the ccd_map_image() image.
    x0 += 7
    x1 += 7
    y0 += 6
    y1 += 6
    
    if inset == 0.:
        return (x0,x1,y0,y1)
    return (x0+inset, x1-inset, y0+inset, y1-inset)

def wcs_for_brick(b, W=3600, H=3600, pixscale=0.262):
    '''
    b: row from decals-bricks.fits file
    W,H: size in pixels
    pixscale: pixel scale in arcsec/pixel.

    Returns: Tan wcs object
    '''
    pixscale = pixscale / 3600.
    return Tan(b.ra, b.dec, W/2.+0.5, H/2.+0.5,
               -pixscale, 0., 0., pixscale,
               float(W), float(H))

def ccds_touching_wcs(targetwcs, T, ccdrad=0.17, polygons=True):
    '''
    targetwcs: wcs object describing region of interest
    T: fits_table object of CCDs

    ccdrad: radius of CCDs, in degrees.  Default 0.17 is for DECam.
    #If None, computed from T.

    Returns: index array I of CCDs within range.
    '''
    trad = targetwcs.radius()
    if ccdrad is None:
        ccdrad = max(np.sqrt(np.abs(T.cd1_1 * T.cd2_2 - T.cd1_2 * T.cd2_1)) *
                     np.hypot(T.width, T.height) / 2.)

    rad = trad + ccdrad
    #r,d = targetwcs.crval
    r,d = targetwcs.radec_center()
    #print len(T), 'ccds'
    #print 'trad', trad, 'ccdrad', ccdrad
    I = np.flatnonzero(np.abs(T.dec - d) < rad)
    #print 'Cut to', len(I), 'on Dec'
    I = I[degrees_between(T.ra[I], T.dec[I], r, d) < rad]
    #print 'Cut to', len(I), 'on RA,Dec'

    if not polygons:
        return I
    # now check actual polygon intersection
    tw,th = targetwcs.imagew, targetwcs.imageh
    targetpoly = [(0.5,0.5),(tw+0.5,0.5),(tw+0.5,th+0.5),(0.5,th+0.5)]
    cd = targetwcs.get_cd()
    tdet = cd[0]*cd[3] - cd[1]*cd[2]
    #print 'tdet', tdet
    if tdet > 0:
        targetpoly = list(reversed(targetpoly))
    targetpoly = np.array(targetpoly)

    keep = []
    for i in I:
        W,H = T.width[i],T.height[i]
        wcs = Tan(*[float(x) for x in
                    [T.crval1[i], T.crval2[i], T.crpix1[i], T.crpix2[i], T.cd1_1[i],
                     T.cd1_2[i], T.cd2_1[i], T.cd2_2[i], W, H]])
        cd = wcs.get_cd()
        wdet = cd[0]*cd[3] - cd[1]*cd[2]
        #print 'wdet', wdet
        poly = []
        for x,y in [(0.5,0.5),(W+0.5,0.5),(W+0.5,H+0.5),(0.5,H+0.5)]:
            rr,dd = wcs.pixelxy2radec(x,y)
            ok,xx,yy = targetwcs.radec2pixelxy(rr,dd)
            poly.append((xx,yy))
        if wdet > 0:
            poly = list(reversed(poly))
        poly = np.array(poly)
        if polygons_intersect(targetpoly, poly):
            keep.append(i)
    I = np.array(keep)
    #print 'Cut to', len(I), 'on polygons'
    return I

def create_temp(**kwargs):
    f,fn = tempfile.mkstemp(dir=tempdir, **kwargs)
    os.close(f)
    os.unlink(fn)
    return fn

class Decals(object):
    def __init__(self):
        self.decals_dir = decals_dir
        self.ZP = None
        
    def get_bricks(self):
        return fits_table(os.path.join(self.decals_dir, 'decals-bricks.fits'))

    def get_brick(self, brickid):
        B = self.get_bricks()
        I = np.flatnonzero(B.brickid == brickid)
        if len(I) == 0:
            return None
        return B[I[0]]

    def bricks_touching_radec_box(self, bricks,
                                  ralo, rahi, declo, dechi):
        '''
        Returns an index vector of the bricks that touch the given RA,Dec box.
        '''
        I = np.flatnonzero((bricks.ra1  <= rahi ) * (bricks.ra2  >= ralo) *
                           (bricks.dec1 <= dechi) * (bricks.dec2 >= declo))
        return I
    
    def get_ccds(self):
        T = fits_table(os.path.join(self.decals_dir, 'decals-ccds.fits'))
        T.extname = np.array([s.strip() for s in T.extname])
        return T

    def find_ccds(self, expnum=None, extname=None):
        T = self.get_ccds()
        if expnum is not None:
            T.cut(T.expnum == expnum)
        if extname is not None:
            T.cut(T.extname == extname)
        return T
    
    def get_zeropoint_for(self, im):
        if self.ZP is None:
            zpfn = os.path.join(self.decals_dir, 'calib', 'decam', 'photom', 'zeropoints.fits')
            #print 'Reading zeropoints:', zpfn
            self.ZP = fits_table(zpfn)

            if 'ccdname' in self.ZP.get_columns():
                # 'N4 ' -> 'N4'
                self.ZP.ccdname = np.array([s.strip() for s in self.ZP.ccdname])

            #self.ZP.about()

        I = np.flatnonzero(self.ZP.expnum == im.expnum)
        #print 'Got', len(I), 'matching expnum', im.expnum
        if len(I) > 1:
            #I = np.flatnonzero((self.ZP.expnum == im.expnum) * (self.ZP.extname == im.extname))
            I = np.flatnonzero((self.ZP.expnum == im.expnum) * (self.ZP.ccdname == im.extname))
            #print 'Got', len(I), 'matching expnum', im.expnum, 'and extname', im.extname

        elif len(I) == 0:
            print 'WARNING: using header zeropoints for', im
            # No updated zeropoint -- use header MAGZERO from primary HDU.
            hdr = im.read_image_primary_header()
            magzero = hdr['MAGZERO']
            #exptime = hdr['EXPTIME']
            #magzero += 2.5 * np.log10(exptime)
            return magzero

        assert(len(I) == 1)
        I = I[0]

        # Arjun says use CCDZPT
        magzp = self.ZP.ccdzpt[I]

        # magzp = self.ZP.zpt[I]
        # print 'Raw magzp', magzp
        # if magzp == 0:
        #     print 'Magzp = 0; using ccdzpt'
        #     magzp = self.ZP.ccdzpt[I]
        #     print 'Got', magzp
        exptime = self.ZP.exptime[I]
        magzp += 2.5 * np.log10(exptime)
        #print 'magzp', magzp
        return magzp

class DecamImage(object):
    def __init__(self, t):
        imgfn, hdu, band, expnum, extname, calname, exptime = (
            t.cpimage.strip(), t.cpimage_hdu, t.filter.strip(), t.expnum,
            t.extname.strip(), t.calname.strip(), t.exptime)

        self.imgfn = os.path.join(decals_dir, 'images', 'decam', imgfn)
        self.hdu   = hdu
        self.expnum = expnum
        self.extname = extname
        self.band  = band
        self.exptime = exptime
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            print attr, '->', fn
            if os.path.exists(fn):
                print 'Exists.'
                continue
            if fn.endswith('.fz'):
                fun = fn[:-3]
                if os.path.exists(fun):
                    print 'Using      ', fun
                    print 'rather than', fn
                    setattr(self, attr, fun)
            fn = getattr(self, attr)
            print attr, fn
            print '  exists? ', os.path.exists(fn)

        ibase = os.path.basename(imgfn)
        ibase = ibase.replace('.fits.fz', '')
        idirname = os.path.basename(os.path.dirname(imgfn))
        #self.name = dirname + '/' + base + ' + %02i' % hdu
        #print 'dir,base', idirname, ibase
        #print 'calibdir', calibdir

        self.calname = calname
        self.name = '%08i-%s' % (expnum, extname)
        #print 'Calname', calname
        
        extnm = '.ext%02i' % hdu
        self.wcsfn = os.path.join(calibdir, 'astrom', calname + '.wcs.fits')
        self.corrfn = self.wcsfn.replace('.wcs.fits', '.corr.fits')
        self.sdssfn = self.wcsfn.replace('.wcs.fits', '.sdss.fits')
        self.sefn = os.path.join(calibdir, 'sextractor', calname + '.fits')
        self.se2fn = os.path.join(calibdir, 'sextractor2', calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', calname + '.fits')
        self.psffitfn = os.path.join(calibdir, 'psfexfit', calname + '.fits')
        self.psffitellfn = os.path.join(calibdir, 'psfexfit', calname + '-ell.fits')
        self.skyfn = os.path.join(calibdir, 'sky', calname + '.fits')
        self.morphfn = os.path.join(calibdir, 'morph', calname + '.fits')

    def __str__(self):
        return self.name
    def __repr__(self):
        return str(self)

    def get_tractor_image(self, decals, slc=None, radecpoly=None, mock_psf=False):
        '''
        slc: y,x slices
        '''
        band = self.band
        imh,imw = self.get_image_shape()
        wcs = self.read_wcs()
        x0,y0 = 0,0
        if slc is None and radecpoly is not None:
            imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
            ok,tx,ty = wcs.radec2pixelxy(radecpoly[:-1,0], radecpoly[:-1,1])
            tpoly = zip(tx,ty)
            clip = clip_polygon(imgpoly, tpoly)
            clip = np.array(clip)
            if len(clip) == 0:
                return None
            x0,y0 = np.floor(clip.min(axis=0)).astype(int)
            x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
            slc = slice(y0,y1+1), slice(x0,x1+1)

            if y1 - y0 < 5 or x1 - x0 < 5:
                print 'Skipping tiny subimage'
                return None
        if slc is not None:
            sy,sx = slc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop
        
        print 'Reading image from', self.imgfn, 'HDU', self.hdu
        img,imghdr = self.read_image(header=True, slice=slc)
        print 'Reading invvar from', self.wtfn, 'HDU', self.hdu
        invvar = self.read_invvar(slice=slc, clip=True)

        print 'Invvar range:', invvar.min(), invvar.max()
        if np.all(invvar == 0.):
            print 'Skipping zero-invvar image'
            return None
        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(not(np.all(invvar == 0.)))

        # header 'FWHM' is in pixels
        psf_fwhm = imghdr['FWHM']
        psf_sigma = psf_fwhm / 2.35
        primhdr = self.read_image_primary_header()

        magzp = decals.get_zeropoint_for(self)
        print 'magzp', magzp
        zpscale = NanoMaggies.zeropointToScale(magzp)
        print 'zpscale', zpscale

        sky = self.read_sky_model()
        midsky = sky.getConstant()
        img -= midsky
        sky.subtract(midsky)

        # Scale images to Nanomaggies
        img /= zpscale
        invvar *= zpscale**2
        orig_zpscale = zpscale
        zpscale = 1.
        assert(np.sum(invvar > 0) > 0)
        sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(np.isfinite(sig1))

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        if mock_psf:
            from tractor.basics import NCircularGaussianPSF
            psfex = None
            psf = NCircularGaussianPSF([1.5], [1.0])
            print 'WARNING: using mock PSF:', psf
        else:
            # read fit PsfEx model -- with ellipse representation
            psfex = PsfEx.fromFits(self.psffitellfn)
            print 'Read', psfex
            psf = psfex

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=sky, name=self.name + ' ' + band)
        assert(np.all(np.isfinite(tim.getInvError())))
        tim.zr = [-3. * sig1, 10. * sig1]
        tim.midsky = midsky
        tim.sig1 = sig1
        tim.band = band
        tim.psf_fwhm = psf_fwhm
        tim.psf_sigma = psf_sigma
        tim.sip_wcs = wcs
        tim.x0,tim.y0 = int(x0),int(y0)
        tim.psfex = psfex
        tim.imobj = self
        mn,mx = tim.zr
        subh,subw = tim.shape
        tim.subwcs = tim.sip_wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        return tim
    
    def makedirs(self):
        for dirnm in [os.path.dirname(fn) for fn in
                      [self.wcsfn, self.corrfn, self.sdssfn, self.sefn, self.psffn, self.morphfn,
                       self.se2fn, self.psffitfn, self.skyfn]]:
            if not os.path.exists(dirnm):
                try:
                    os.makedirs(dirnm)
                except:
                    pass

    def _read_fits(self, fn, hdu, slice=None, header=None, **kwargs):
        if slice is not None:
            f = fitsio.FITS(fn)[hdu]
            img = f[slice]
            rtn = img
            if header:
                hdr = f.read_header()
                return (img,hdr)
            return img
        return fitsio.read(fn, ext=hdu, header=header, **kwargs)

    def read_image(self, **kwargs):
        return self._read_fits(self.imgfn, self.hdu, **kwargs)

    def get_image_info(self):
        return fitsio.FITS(self.imgfn)[self.hdu].get_info()

    def get_image_shape(self):
        ''' Returns image H,W '''
        return self.get_image_info()['dims']
    
    def read_image_primary_header(self, **kwargs):
        return fitsio.read_header(self.imgfn)

    def read_image_header(self, **kwargs):
        return fitsio.read_header(self.imgfn, ext=self.hdu)

    def read_dq(self, **kwargs):
        return self._read_fits(self.dqfn, self.hdu, **kwargs)
    #return fitsio.FITS(self.dqfn)[self.hdu].read()

    def read_invvar(self, clip=False, **kwargs):
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        if clip:
            sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
            # Clamp near-zero (incl negative!) invvars to zero
            thresh = 0.2 * (1./sig1**2)
            invvar[invvar < thresh] = 0
        return invvar
    #return fitsio.FITS(self.wtfn)[self.hdu].read()

    def read_wcs(self):
        return Sip(self.wcsfn)

    def read_sdss(self):
        S = fits_table(self.sdssfn)
        # ugh!
        if S.objc_type.min() > 128:
            S.objc_type -= 128
        return S

    def read_sky_model(self):
        hdr = fitsio.read_header(self.skyfn)
        skyclass = hdr['SKY']
        clazz = get_class_from_name(skyclass)
        fromfits = getattr(clazz, 'fromFitsHeader')
        skyobj = fromfits(hdr, prefix='SKY_')
        return skyobj

    def run_calibs(self, ra, dec, pixscale, mock_psf,
                   W=2048, H=4096, se=True,
                   astrom=True, psfex=True, sky=True,
                   morph=False, se2=False, psfexfit=True,
                   funpack=True, fcopy=False, use_mask=True,
                   just_check=False):
        '''
        pixscale: in arcsec/pixel

        just_check: if True, returns True if calibs need to be run.
        '''
        print 'run_calibs:', str(self), 'near RA,Dec', ra,dec, 'with pixscale', pixscale, 'arcsec/pix'

        for fn in [self.wcsfn, self.sefn, self.psffn, self.psffitfn, self.skyfn]:
            print 'exists?', os.path.exists(fn), fn
        self.makedirs()

        if mock_psf:
            psfex = False
            psfexfit = False
    
        run_funpack = False
        run_se = False
        run_se2 = False
        run_astrom = False
        run_psfex = False
        run_psfexfit = False
        run_morph = False
        run_sky = False
    
        if se and not all([os.path.exists(fn) for fn in [self.sefn]]):
            run_se = True
            run_funpack = True
        if se2 and not all([os.path.exists(fn) for fn in [self.se2fn]]):
            run_se2 = True
            run_funpack = True
        #if not all([os.path.exists(fn) for fn in [self.wcsfn,self.corrfn,self.sdssfn]]):
        if astrom and not os.path.exists(self.wcsfn):
            run_astrom = True
        if psfex and not os.path.exists(self.psffn):
            run_psfex = True
        if psfexfit and not (os.path.exists(self.psffitfn) and os.path.exists(self.psffitellfn)):
            run_psfexfit = True
        if morph and not os.path.exists(self.morphfn):
            run_morph = True
            run_funpack = True
        if sky and not os.path.exists(self.skyfn):
            run_sky = True

        if just_check:
            return (run_se or run_se2 or run_astrom or run_psfex or run_psfexfit
                    or run_morph or run_sky)

        if run_funpack and (funpack or fcopy):
            tmpimgfn  = create_temp(suffix='.fits')
            tmpmaskfn = create_temp(suffix='.fits')
    
            if funpack:
                cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpimgfn, self.imgfn)
            else:
                cmd = 'imcopy %s"+%i" %s' % (self.imgfn, self.hdu, tmpimgfn)
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
            if use_mask:
                cmd = 'funpack -E %i -O %s %s' % (self.hdu, tmpmaskfn, self.dqfn)
                print cmd
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)
    
        if run_astrom or run_morph or run_se or run_se2:
            # grab header values...
            primhdr = self.read_image_primary_header()
            hdr     = self.read_image_header()
    
            magzp  = primhdr['MAGZERO']
            fwhm = hdr['FWHM']
            seeing = pixscale * fwhm
            print 'FWHM', fwhm, 'pix'
            print 'pixscale', pixscale, 'arcsec/pix'
            print 'Seeing', seeing, 'arcsec'
    
        if run_se:
            maskstr = ''
            if use_mask:
                maskstr = '-FLAG_IMAGE ' + tmpmaskfn
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'DECaLS-v2.sex'),
                maskstr, '-SEEING_FWHM %f' % seeing,
                '-PIXEL_SCALE 0',
                #'-PIXEL_SCALE %f' % (pixscale),
                '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', self.sefn,
                tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_se2:
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'DECaLS-v2-2.sex'),
                '-FLAG_IMAGE', tmpmaskfn, '-SEEING_FWHM %f' % seeing,
                '-PIXEL_SCALE %f' % (pixscale),
                '-MAG_ZEROPOINT %f' % magzp, '-CATALOG_NAME', self.se2fn,
                tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_astrom:
            cmd = ' '.join([
                'solve-field --config', an_config, '-D . --temp-dir', tempdir,
                '--ra %f --dec %f' % (ra,dec), '--radius 1',
                '-L %f -H %f -u app' % (0.9 * pixscale, 1.1 * pixscale),
                '--continue --no-plots --no-remove-lines --uniformize 0',
                '--no-fits2fits',
                '-X x_image -Y y_image -s flux_auto --extension 2',
                '--width %i --height %i' % (W,H),
                '--crpix-center',
                '-N none -U none -S none -M none',
                #'--rdls', self.sdssfn,
                #'--corr', self.corrfn,
                '--rdls none --corr none',
                '--wcs', self.wcsfn, 
                '--temp-axy', '--tag-all', self.sefn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)

            if not os.path.exists(self.wcsfn):
                # Run a second phase...
                an_config_2 = os.path.join(decals_dir, 'calib', 'an-config', 'cfg2')
                cmd = ' '.join([
                    'solve-field --config', an_config_2, '-D . --temp-dir', tempdir,
                    '--ra %f --dec %f' % (ra,dec), '--radius 1',
                    '-L %f -H %f -u app' % (0.9 * pixscale, 1.1 * pixscale),
                    '--continue --no-plots --uniformize 0',
                    '--no-fits2fits',
                    '-X x_image -Y y_image -s flux_auto --extension 2',
                    '--width %i --height %i' % (W,H),
                    '--crpix-center',
                    '-N none -U none -S none -M none',
                    '--rdls none --corr none',
                    '--wcs', self.wcsfn, 
                    '--temp-axy', '--tag-all', self.sefn])
                    #--no-remove-lines 
                print cmd
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)

        if run_psfex:
            cmd = ('psfex -c %s -PSF_DIR %s %s' %
                   (os.path.join(sedir, 'DECaLS-v2.psfex'),
                    os.path.dirname(self.psffn), self.sefn))
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_psfexfit:
            print 'Fit PSF...'
    
            from tractor.basics import GaussianMixtureEllipsePSF, GaussianMixturePSF
            from tractor.psfex import PsfEx
    
            iminfo = self.get_image_info()
            #print 'img:', iminfo
            H,W = iminfo['dims']
            psfex = PsfEx(self.psffn, W, H, ny=13, nx=7,
                          psfClass=GaussianMixtureEllipsePSF)
            psfex.savesplinedata = True
            print 'Fitting MoG model to PsfEx'
            psfex._fitParamGrid(damp=1)
            pp,XX,YY = psfex.splinedata

            psfex.toFits(self.psffitellfn, merge=True)
            print 'Wrote', self.psffitellfn
    
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
            print 'Wrote', self.psffitfn
            
        if run_morph:
            cmd = ' '.join(['sex -c', os.path.join(sedir, 'CS82_MF.sex'),
                            '-FLAG_IMAGE', tmpmaskfn,
                            '-SEEING_FWHM %f' % seeing,
                            '-MAG_ZEROPOINT %f' % magzp,
                            '-PSF_NAME', self.psffn,
                            '-CATALOG_NAME', self.morphfn,
                            tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)

        if run_sky:
            img = self.read_image()
            wt = self.read_invvar()
            img = img[wt > 0]
            try:
                skyval = estimate_mode(img, raiseOnWarn=True)
            except:
                skyval = np.median(img)
            sky = ConstantSky(skyval)
            tt = type(sky)
            sky_type = '%s.%s' % (tt.__module__, tt.__name__)
            hdr = fitsio.FITSHDR()
            hdr.add_record(dict(name='SKY', value=sky_type, comment='Sky class'))
            sky.toFitsHeader(hdr, prefix='SKY_')
            fits = fitsio.FITS(self.skyfn, 'rw', clobber=True)
            fits.write(None, header=hdr)
            
def run_calibs(X):
    im = X[0]
    kwargs = X[1]
    args = X[2:]
    print 'run_calibs:', X
    print 'im', im
    print 'args', args
    print 'kwargs', kwargs
    return im.run_calibs(*args, **kwargs)


