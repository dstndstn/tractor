from .brightness import Mags

def _dr7_getBrightness(counts, tsf, bands, extrabands):
    from astrometry.sdss import band_name

    allcounts = counts
    order = []
    kwargs = {}

    BAD_MAG = 99.
    
    for i,counts in enumerate(allcounts):
        bandname = band_name(i)
        if not bandname in bands:
            continue
        if counts == 0:
            mag = BAD_MAG
        else:
            mag = tsf.counts_to_mag(counts, i)
        if not np.isfinite(mag):
            mag = BAD_MAG
        order.append(bandname)
        kwargs[bandname] = mag
        #print('Band', bandname, 'counts', counts, 'mag', mag)
    #print('creating mags:', kwargs)
    for b in extrabands:
        order.append(b)
        kwargs.update(b=BAD_MAG)
    m = Mags(order=order, **kwargs)
    #print('created', m)
    return m


def get_tractor_sources_dr7(*args, **kwargs):
    '''
    get_tractor_sources_dr7(run, camcol, field, bandname='r',
                            sdss=None, release='DR7',
                            retrieve=True, curl=False, roi=None, bands=None)

    Creates tractor.Source objects corresponding to objects in the SDSS catalog
    for the given field.

    bandname: "canonical" band from which to get galaxy shapes, positions, etc
    '''
    from .sdss import _get_sources
    kwargs.update(release='DR7')
    return _get_sources(*args, **kwargs)

def get_tractor_image_dr7(run, camcol, field, bandname, 
                      sdssobj=None, release='DR7',
                      retrieve=True, curl=False,
                      psf='kl-gm', useMags=False,
                      roi=None,
                      roiradecsize=None,
                      roiradecbox=None,
                      nanomaggies=False,
                      savepsfimg=None, zrange=[-3,10]):
    '''
    Creates a tractor.Image given an SDSS field identifier.

    If not None, roi = (x0, x1, y0, y1) defines a region-of-interest
    in the image, in zero-indexed pixel coordinates.  x1,y1 are
    NON-inclusive; roi=(0,100,0,100) will yield a 100 x 100 image.
    
    psf can be:
      "dg" for double-Gaussian
      "kl-gm" for SDSS KL-decomposition approximated as a Gaussian mixture

    "roiradecsize" = (ra, dec, half-size in pixels) indicates that you
    want to grab a ROI around the given RA,Dec.

    Returns:
      (tractor.Image, dict)

    dict contains useful details like:
      'sky'
      'skysig'
    '''
    from astrometry.sdss import DR7, band_index

    if sdssobj is None:
        # Ugly
        if release != 'DR7':
            raise RuntimeError('We only support DR7 currently')
        sdss = DR7(curl=curl)
    else:
        sdss = sdssobj

    valid_psf = ['dg', 'kl-gm']
    if psf not in valid_psf:
        raise RuntimeError('PSF must be in ' + str(valid_psf))
    # FIXME
    rerun = 0

    bandnum = band_index(bandname)

    _check_sdss_files(sdss, run, camcol, field, bandname,
                      ['fpC', 'tsField', 'psField', 'fpM'],
                      retrieve=retrieve)
    fpC = sdss.readFpC(run, camcol, field, bandname)
    hdr = fpC.getHeader()
    fpC = fpC.getImage()
    fpC = fpC.astype(np.float32) - sdss.softbias
    image = fpC
    (H,W) = image.shape

    info = dict()
    tai = hdr.get('TAI')
    stripe = hdr.get('STRIPE')
    strip = hdr.get('STRIP')
    obj = hdr.get('OBJECT')
    info.update(tai=tai, stripe=stripe, strip=strip, object=obj, hdr=hdr)

    tsf = sdss.readTsField(run, camcol, field, rerun)
    astrans = tsf.getAsTrans(bandnum)
    wcs = SdssWcs(astrans)
    #print('Created SDSS Wcs:', wcs)

    X = interpret_roi(wcs, (H,W), roi=roi, roiradecsize=roiradecsize,
                      roiradecbox=roiradecbox)
    if X is None:
        return None,None
    roi,hasroi = X
    info.update(roi=roi)
    x0,x1,y0,y1 = roi
        
    # Mysterious half-pixel shift.  asTrans pixel coordinates?
    wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

    if nanomaggies:
        zp = tsf.get_zeropoint(bandnum)
        photocal = LinearPhotoCal(NanoMaggies.zeropointToScale(zp),
                                  band=bandname)
    elif useMags:
        photocal = SdssMagsPhotoCal(tsf, bandname)
    else:
        photocal = SdssFluxPhotoCal()

    psfield = sdss.readPsField(run, camcol, field)
    sky = psfield.getSky(bandnum)
    skysig = sqrt(sky)
    skyobj = ConstantSky(sky)
    zr = sky + np.array(zrange) * skysig
    info.update(sky=sky, skysig=skysig, zr=zr)

    fpM = sdss.readFpM(run, camcol, field, bandname)
    gain = psfield.getGain(bandnum)
    darkvar = psfield.getDarkVariance(bandnum)
    skyerr = psfield.getSkyErr(bandnum)
    invvar = sdss.getInvvar(fpC, fpM, gain, darkvar, sky, skyerr)

    dgpsf = psfield.getDoubleGaussian(bandnum, normalize=True)
    info.update(dgpsf=dgpsf)
    
    if roi is not None:
        roislice = (slice(y0,y1), slice(x0,x1))
        image = image[roislice].copy()
        invvar = invvar[roislice].copy()

    if psf == 'kl-gm':
        from emfit import em_fit_2d
        from fitpsf import em_init_params

        # Create Gaussian mixture model PSF approximation.
        H,W = image.shape
        klpsf = psfield.getPsfAtPoints(bandnum, x0+W/2, y0+H/2)
        S = klpsf.shape[0]
        # number of Gaussian components
        K = 3
        w,mu,sig = em_init_params(K, None, None, None)
        II = klpsf.copy()
        II /= II.sum()
        # HIDEOUS HACK
        II = np.maximum(II, 0)
        #print('Multi-Gaussian PSF fit...')
        xm,ym = -(S/2), -(S/2)
        if savepsfimg is not None:
            plt.clf()
            plt.imshow(II, interpolation='nearest', origin='lower')
            plt.title('PSF image to fit with EM')
            plt.savefig(savepsfimg)
        res = em_fit_2d(II, xm, ym, w, mu, sig)
        print('em_fit_2d result:', res)
        if res == 0:
            # print('w,mu,sig', w,mu,sig)
            mypsf = GaussianMixturePSF(w, mu, sig)
            mypsf.computeRadius()
        else:
            # Failed!  Return 'dg' model instead?
            print('PSF model fit', psf, 'failed!  Returning DG model instead')
            psf = 'dg'
    if psf == 'dg':
        print('Creating double-Gaussian PSF approximation')
        (a,s1, b,s2) = dgpsf
        mypsf = NCircularGaussianPSF([s1, s2], [a, b])
        
    timg = Image(data=image, invvar=invvar, psf=mypsf, wcs=wcs,
                 sky=skyobj, photocal=photocal,
                 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
                       (run, camcol, field, bandname)))
    timg.zr = zr
    return timg,info



class SdssMagsPhotoCal(BaseParams):
    '''
    A photocal that uses Mags objects.
    '''
    def __init__(self, tsfield, bandname):
        from astrometry.sdss import band_index
        
        self.bandname = bandname
        self.band = band_index(bandname)

        #self.tsfield = tsfield
        band = self.band
        self.exptime = tsfield.exptime
        self.aa = tsfield.aa[band]
        self.kk = tsfield.kk[band]
        self.airmass = tsfield.airmass[band]

        super(SdssMagsPhotoCal,self).__init__()

    # @staticmethod
    # def getNamedParams():
    #   return dict(aa=0)
    # # These underscored versions are for use by NamedParams(), and ignore
    # # the active/inactive state.
    # def _setThing(self, i, val):
    #   assert(i == 0)
    #   self.aa = val
    # def _getThing(self, i):
    #   assert(i == 0)
    #   return self.aa
    # def _getThings(self):
    #   return [self.aa]
    # def _numberOfThings(self):
    #   return 1

    # to implement Params
    def getParams(self):
        return [self.aa]
    def getStepSizes(self, *args, **kwargs):
        return [0.01]
    def setParam(self, i, p):
        assert(i == 0)
        self.aa = p

    def getParamNames(self):
        return ['aa']

    def hashkey(self):
        return ('SdssMagsPhotoCal', self.bandname, #self.tsfield)
                self.exptime, self.aa, self.kk, self.airmass)
    
    def brightnessToCounts(self, brightness):
        mag = brightness.getMag(self.bandname)
        if not np.isfinite(mag):
            return 0.
        # MAGIC
        if mag > 50.:
            return 0.
        #return self.tsfield.mag_to_counts(mag, self.band)

        # FROM astrometry.sdss.common.TsField.mag_to_counts:
        logcounts = (-0.4 * mag + np.log10(self.exptime)
                     - 0.4*(self.aa + self.kk * self.airmass))
        rtn = 10.**logcounts
        return rtn

