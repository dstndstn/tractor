import matplotlib
matplotlib.use('Agg')

if True:

    import numpy as np
    import pylab as plt
    import fitsio
    from tractor import *
    from tractor.galaxy import *

    mydir = os.path.dirname(__file__)

    tims = []
    bands = 'ugrizy'
    nepochs = 3
    for band in bands:
        fn = os.path.join(mydir, 'galsim', 'output', 'demo12b_%s.fits' % band)
        print 'Band', band
        print 'Reading', fn
        cube,hdr = fitsio.read(fn, header=True)
        print 'Read', cube.shape

        pixscale = hdr['GS_SCALE']
        print 'Pixel scale:', pixscale, 'arcsec/pix'

        #pixnoise = 0.1
        pixnoise = 0.02
        psf_fwhm = 0.6 / pixscale
        psf_sigma = psf_fwhm / 2.35

        nims,h,w = cube.shape
        assert(nims == nepochs)

        for i in range(nims):
            image = cube[i,:,:]
            tim = Image(data=image, invvar=np.ones_like(image) / pixnoise**2,
                        photocal=FluxesPhotoCal(band),
                        wcs=NullWCS(pixscale=pixscale),
                        # Hack up a multi-Gaussian PSF
                        #psf=NCircularGaussianPSF([psf_sigma, psf_sigma*2], [0.8, 0.2]))
                        psf=NCircularGaussianPSF([psf_sigma], [1.0]))
            tims.append(tim)

    # galaxy = DevGalaxy(PixPos(w/2, h/2), Fluxes(**dict([(band, 10.) for band in bands])),
    # EllipseESoft(0., 0., 0.))

    galaxy = CompositeGalaxy(PixPos(w/2, h/2),
                             Fluxes(**dict([(band, 10.) for band in bands])),
                             EllipseESoft(0., 0., 0.),
                             Fluxes(**dict([(band, 10.) for band in bands])),
                             EllipseESoft(0., 0., 0.))

    tractor = Tractor(tims, [galaxy])

    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-5.*pixnoise, vmax=20.*pixnoise)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.clf()
    for i,band in enumerate(bands):
        for e in range(nepochs):
            plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
            plt.imshow(tims[nepochs*i + e].getImage(), **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('%s #%i' % (band, e+1))
    plt.savefig('8.png')

    mods = [tractor.getModelImage(i) for i in range(len(tims))]

    plt.clf()
    for i,band in enumerate(bands):
        for e in range(nepochs):
            plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
            plt.imshow(mods[nepochs*i + e], **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('%s #%i' % (band, e+1))
    plt.suptitle('Initial models')
    plt.savefig('9.png')

    # Freeze all image calibration parameters
    tractor.freezeParam('images')

    # Take several linearized least squares steps
    for i in range(20):

        print 'Optimization step', i
        print 'Before:', galaxy

        dlnp,X,alpha = tractor.optimize()
        print 'dlnp', dlnp

        print 'After:', galaxy

        if dlnp < 1e-3:
            break

    mods = [tractor.getModelImage(i) for i in range(len(tims))]

    plt.clf()
    for i,band in enumerate(bands):
        for e in range(nepochs):
            plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
            plt.imshow(mods[nepochs*i + e], **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('%s #%i' % (band, e+1))
    plt.suptitle('Optimized models')
    plt.savefig('10.png')

    plt.clf()
    for i,band in enumerate(bands):
        for e in range(nepochs):
            plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
            mod = mods[nepochs*i + e]
            plt.imshow(mod + pixnoise * np.random.normal(size=mod.shape), **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('%s #%i' % (band, e+1))
    plt.suptitle('Optimized models + noise')
    plt.savefig('11.png')
