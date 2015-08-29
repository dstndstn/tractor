from .utils import MultiParams, _isint, listmax, get_class_from_name

class Image(MultiParams):
    '''
    An image plus its calibration information.  An ``Image`` has
    pixels, inverse-variance map, WCS, PSF, photometric calibration
    information, and sky level.  All these things are ``Params``
    instances, and ``Image`` is a ``MultiParams`` so that the Tractor
    can optimize them.
    '''
    def __init__(self, data=None, invvar=None, inverr=None,
                 psf=None, wcs=None, sky=None,
                 photocal=None, name=None, time=None, **kwargs):
        '''
        Args:
          * *data*: numpy array: the image pixels
          * *invvar*: numpy array: the image inverse-variance
          * *inverr*: numpy array: the image inverse-error
          * *psf*: a :class:`tractor.PSF` duck
          * *wcs*: a :class:`tractor.WCS` duck
          * *sky*: a :class:`tractor.Sky` duck
          * *photocal*: a :class:`tractor.PhotoCal` duck
          * *name*: string name of this image.
          * *zr*: plotting range ("vmin"/"vmax" in matplotlib.imshow)

        Only one of *invvar* and *inverr* should be given.  If both
        are given, inverr takes precedent.

        If *wcs* is not given, assumes pixel space.

        If *sky* is not given, assumes zero sky.
        
        If *photocal* is not given, assumes count units.
        '''
        self.data = data
        if inverr is not None:
            self.inverr = inverr
        elif invvar is not None:
            self.inverr = np.sqrt(invvar)
            
        self.name = name
        self.zr = kwargs.pop('zr', None)
        self.time = time

        # Fill in defaults, if necessary.
        if wcs is None:
            from .basics import NullWCS
            wcs = NullWCS()
        if sky is None:
            from .basics import NullSky
            sky = NullSky()
        if photocal is None:
            from .basics import NullPhotoCal
            photocal = NullPhotoCal()

        # acceptable approximation level when rendering this model
        # image
        self.modelMinval = 0.
            
        super(Image, self).__init__(psf, wcs, photocal, sky)

    def __str__(self):
        return 'Image ' + str(self.name)

    @staticmethod
    def getNamedParams():
        return dict(psf=0, wcs=1, photocal=2, sky=3)

    def getTime(self):
        return self.time
    
    def getParamDerivatives(self, tractor, srcs):
        '''
        Returns a list of Patch objects, one per numberOfParams().
        Note that this means you have to pay attention to the
        frozen/thawed state.

        Can return None for no derivative, or False if you want the
        Tractor to compute the derivatives for you.
        '''
        derivs = []
        for s in self._getActiveSubs():
            if hasattr(s, 'getParamDerivatives'):
                #print 'Calling getParamDerivatives on', s
                sd = s.getParamDerivatives(tractor, self, srcs)
                assert(len(sd) == s.numberOfParams())
                derivs.extend(sd)
            else:
                derivs.extend([False] * s.numberOfParams())
        # print 'Image.getParamDerivatives: returning', derivs
        return derivs

    def getSky(self):
        return self.sky

    def setSky(self, sky):
        self.sky = sky

    def setPsf(self, psf):
        self.psf = psf

    @property
    def shape(self):
        return self.getShape()

    @property
    def invvar(self):
        return self.inverr**2
    
    # Numpy arrays have shape H,W
    def getWidth(self):
        return self.getShape()[1]
    def getHeight(self):
        return self.getShape()[0]
    def getShape(self):
        if 'shape' in self.__dict__:
            return self.shape
        return self.data.shape

    def getModelShape(self):
        return self.getShape()
    
    def hashkey(self):
        return ('Image', id(self.data), id(self.inverr), self.psf.hashkey(),
                self.sky.hashkey(), self.wcs.hashkey(),
                self.photocal.hashkey())

    def numberOfPixels(self):
        (H,W) = self.data.shape
        return W*H

    def getInvError(self):
        return self.inverr
    def getInvvar(self):
        return self.inverr**2

    def getImage(self):
        return self.data
    def getPsf(self):
        return self.psf
    def getWcs(self):
        return self.wcs
    def getPhotoCal(self):
        return self.photocal

    @staticmethod
    def readFromFits(fits, prefix=''):
        hdr = fits[0].read_header()
        pix = fits[1].read()
        iv = fits[2].read()
        assert(pix.shape == iv.shape)

        def readObject(prefix):
            k = prefix
            objclass = hdr[k]
            clazz = get_class_from_name(objclass)
            fromfits = getattr(clazz, 'fromFitsHeader')
            print 'fromFits:', fromfits
            obj = fromfits(hdr, prefix=prefix + '_')
            print 'Got:', obj
            return obj

        psf = readObject(prefix + 'PSF')
        wcs = readObject(prefix + 'WCS')
        sky = readObject(prefix + 'SKY')
        pcal = readObject(prefix + 'PHO')

        return Image(data=pix, invvar=iv, psf=psf, wcs=wcs, sky=sky,
                     photocal=pcal)
        
    def toFits(self, fits, prefix='', primheader=None, imageheader=None,
               invvarheader=None):
        hdr = self.getFitsHeader(header=primheader, prefix=prefix)
        fits.write(None, header=hdr)
        fits.write(self.getImage(), header=imageheader)
        fits.write(self.getInvvar(), header=invvarheader)

    def getFitsHeader(self, header=None, prefix=''):
        psf = self.getPsf()
        wcs = self.getWcs()
        sky = self.getSky()
        pcal = self.getPhotoCal()
        
        if header is None:
            import fitsio
            hdr = fitsio.FITSHDR()
        else:
            hdr = header
        tt = type(psf)
        psf_type = '%s.%s' % (tt.__module__, tt.__name__)
        tt = type(wcs)
        wcs_type = '%s.%s' % (tt.__module__, tt.__name__)
        tt = type(sky)
        sky_type = '%s.%s' % (tt.__module__, tt.__name__)
        tt = type(pcal)
        pcal_type = '%s.%s' % (tt.__module__, tt.__name__)
        hdr.add_record(dict(name=prefix + 'PSF', value=psf_type,
                            comment='PSF class'))
        hdr.add_record(dict(name=prefix + 'WCS', value=wcs_type,
                            comment='WCS class'))
        hdr.add_record(dict(name=prefix + 'SKY', value=sky_type,
                            comment='Sky class'))
        hdr.add_record(dict(name=prefix + 'PHO', value=pcal_type,
                            comment='PhotoCal class'))
        psf.toFitsHeader(hdr,  prefix + 'PSF_')
        wcs.toFitsHeader(hdr,  prefix + 'WCS_')
        sky.toFitsHeader(hdr,  prefix + 'SKY_')
        pcal.toFitsHeader(hdr, prefix + 'PHO_')
        return hdr

    def getStandardFitsHeader(self, header=None):
        if header is None:
            import fitsio
            hdr = fitsio.FITSHDR()
        else:
            hdr = header
        psf = self.getPsf()
        wcs = self.getWcs()
        sky = self.getSky()
        pcal = self.getPhotoCal()
        psf.toStandardFitsHeader(hdr)
        wcs.toStandardFitsHeader(hdr)
        sky.toStandardFitsHeader(hdr)
        pcal.toStandardFitsHeader(hdr)
        return hdr

    
