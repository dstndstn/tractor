import matplotlib
matplotlib.use('Agg')

if True:

    import numpy as np
    import pylab as plt
    import fitsio
    from tractor import *
    from tractor.galaxy import *

    # These match the values in galsim/demo12.py
    pixnoise = 0.02
    psf_sigma = 1.5
    bands = 'ugrizy'
    nepochs = 3
    
    # Read multiple epochs of imaging for each band.
    mydir = os.path.dirname(__file__)
    tims = []
    for band in bands:
        fn = os.path.join(mydir, 'galsim', 'output', 'demo12b_%s.fits' % band)
        print 'Band', band, 'Reading', fn
        cube,hdr = fitsio.read(fn, header=True)
        print 'Read', cube.shape
        pixscale = hdr['GS_SCALE']
        print 'Pixel scale:', pixscale, 'arcsec/pix'
        nims,h,w = cube.shape
        assert(nims == nepochs)
        for i in range(nims):
            image = cube[i,:,:]
            tim = Image(data=image, inverr=np.ones_like(image) / pixnoise,
                        photocal=FluxesPhotoCal(band),
                        wcs=NullWCS(pixscale=pixscale),
                        psf=NCircularGaussianPSF([psf_sigma], [1.0]))
            tims.append(tim)

    # We create a dev+exp galaxy with made-up initial parameters.
    galaxy = CompositeGalaxy(PixPos(w/2, h/2),
                             Fluxes(**dict([(band, 10.) for band in bands])),
                             EllipseESoft(0., 0., 0.),
                             Fluxes(**dict([(band, 10.) for band in bands])),
                             EllipseESoft(0., 0., 0.))

    tractor = Tractor(tims, [galaxy])

    # Plot images
    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-5.*pixnoise, vmax=20.*pixnoise)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92)
    plt.clf()
    for i,band in enumerate(bands):
        for e in range(nepochs):
            plt.subplot(nepochs, len(bands), e*len(bands) + i +1)
            plt.imshow(tims[nepochs*i + e].getImage(), **ima)
            plt.xticks([]); plt.yticks([])
            plt.title('%s #%i' % (band, e+1))
    plt.suptitle('Images')
    plt.savefig('8.png')

    # Plot initial models:
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

    print 'Thawed params:'
    tractor.printThawedParams()
    print

    
    # Take several linearized least squares steps
    for i in range(20):
        dlnp,X,alpha = tractor.optimize(shared_params=False)
        print 'dlnp', dlnp
        if dlnp < 1e-3:
            break

    # Plot optimized models:
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

    # Plot optimized models + noise:
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
