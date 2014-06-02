import matplotlib
matplotlib.use('Agg')

if True:
    from tractor import *
    source = PointSource(PixPos(17., 27.4), Flux(23.9))
    photocal = NullPhotoCal()
    wcs = NullWCS()
    counts = photocal.brightnessToCounts(source.getBrightness())
    x,y = wcs.positionToPixel(source.getPosition())
    print 'source', source
    print 'counts', counts, 'x,y', x,y
    print
    print 'source', source
    print 'photocal', photocal
    print 'wcs', wcs
    print 'counts', counts
    print 'x,y', x,y

if True:
    from tractor import *
    from astrometry.util.util import Tan
    source = PointSource(RaDecPos(42.3, 9.7), Mags(r=21.5, i=21.3))
    photocal = MagsPhotoCal('r', 22.5)
    wcs = FitsWcs(Tan(42.0, 9.0, 100., 100., 0.1, 0., 0., 0.1, 200., 200.))
    counts = photocal.brightnessToCounts(source.getBrightness())
    x,y = wcs.positionToPixel(source.getPosition())

    print 'source', source
    print 'counts', counts, 'x,y', x,y

    print 'source', source
    print 'photocal', photocal
    print 'wcs', wcs
    print 'counts', counts
    print 'x,y', x,y
    
if True:
    from tractor import *
    pos = RaDecPos(42.3, 9.7)
    print 'Pos:', pos
    print 'Params:', pos.getParams()
    print 'Param names:', pos.getParamNames()
    print 'Step sizes:', pos.getStepSizes()
    pos.setParams([42.7, 9.3])
    print 'After setParams:', pos
    pos.setParam(1, 10.0)
    print 'After setParam:', pos

if True:
    from tractor import *
    source = PointSource(RaDecPos(42.3, 9.7), Mags(r=99.9))
    print source
    print source.pos
    print source.brightness
    print source.pos.ra
    print source.brightness.r
    print source.getParams()
    print zip(source.getParamNames(), source.getParams())


from tractor import *
cat = Catalog(PointSource(RaDecPos(42.3, 9.7), Mags(r=99.9)))
print cat
print zip(cat.getParamNames(), cat.getParams())
cat[0].freezeParam('pos')
print zip(cat.getParamNames(), cat.getParams())
#cat[0].freezeParam('brightness')
cat[0].pos.freezeParam('ra')
cat[0].thawParam('pos')
print zip(cat.getParamNames(), cat.getParams())

cat.thawAllRecursive()
print zip(cat.getParamNames(), cat.getParams())
cat.freezeAllRecursive()
cat.thawPathsTo('r')
print zip(cat.getParamNames(), cat.getParams())
print 'Thawed(self)   Thawed(parent)   Param'
for param, tself, tparent in cat.getParamStateRecursive():
    print '   %5s      %5s           ' % (tself, tparent), param

cat[0].thawParam('pos')
#print zip(cat.getParamNames(), cat.getParams())
cat.printThawedParams()
cat[0].pos.thawAllParams()
print zip(cat.getParamNames(), cat.getParams())
cat.printThawedParams()



if True:
    import numpy as np
    import pylab as plt
    from tractor import *

    # Size of image, centroid and flux of source
    W,H = 25,25
    cx,cy = 12.8, 14.3
    flux = 12.
    # PSF size
    psfsigma = 2.
    # Per-pixel image noise
    noisesigma = 0.01
    # Create synthetic Gaussian star image
    G = np.exp(((np.arange(W)-cx)[np.newaxis,:]**2 +
                (np.arange(H)-cy)[:,np.newaxis]**2)/(-2.*psfsigma**2))
    trueimage = flux * G/G.sum()
    image = trueimage + noisesigma * np.random.normal(size=trueimage.shape)

    # Create Tractor Image
    tim = Image(data=image, invvar=np.ones_like(image) / (noisesigma**2),
                psf=NCircularGaussianPSF([psfsigma], [1.]),
                wcs=NullWCS(), photocal=NullPhotoCal(),
                sky=ConstantSky(0.))

    # Create Tractor source with approximate position and flux
    src = PointSource(PixPos(W/2., H/2.), Flux(10.))

    # Create Tractor object itself
    tractor = Tractor([tim], [src])

    # Render the model image
    mod0 = tractor.getModelImage(0)
    chi0 = tractor.getChiImage(0)

    # Plots
    ima = dict(interpolation='nearest', origin='lower', cmap='gray',
               vmin=-2*noisesigma, vmax=5*noisesigma)
    imchi = dict(interpolation='nearest', origin='lower', cmap='gray',
                 vmin=-5, vmax=5)
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(trueimage, **ima)
    plt.title('True image')
    plt.subplot(2,2,2)
    plt.imshow(image, **ima)
    plt.title('Image')
    plt.subplot(2,2,3)
    plt.imshow(mod0, **ima)
    plt.title('Tractor model')
    plt.subplot(2,2,4)
    plt.imshow(chi0, **imchi)
    plt.title('Chi')
    plt.savefig('1.png')
    
    # Freeze all image calibration params -- just fit source params
    tractor.freezeParam('images')

    # Save derivatives for later plotting...
    derivs = tractor.getDerivs()

    # Take several linearized least squares steps
    for i in range(10):
        dlnp,X,alpha = tractor.optimize()
        print 'dlnp', dlnp
        if dlnp < 1e-3:
            break

    # Get the fit model and residual images for plotting
    mod = tractor.getModelImage(0)
    chi = tractor.getChiImage(0)
    # Plots
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(trueimage, **ima)
    plt.title('True image')
    plt.subplot(2,2,2)
    plt.imshow(image, **ima)
    plt.title('Image')
    plt.subplot(2,2,3)
    plt.imshow(mod, **ima)
    plt.title('Tractor model')
    plt.subplot(2,2,4)
    plt.imshow(chi, **imchi)
    plt.title('Chi')
    plt.savefig('2.png')
    
    # Plot the derivatives we saved earlier
    def showpatch(patch, ima):
        im = patch.patch
        h,w = im.shape
        ext = [patch.x0,patch.x0+w, patch.y0,patch.y0+h]
        plt.imshow(im, extent=ext, **ima)
        plt.title(patch.name)
    imderiv = dict(interpolation='nearest', origin='lower', cmap='gray',
                   vmin=-0.05, vmax=0.05)
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(mod0, **ima)
    ax = plt.axis()
    plt.title('Initial Tractor model')
    for i in range(3):
        plt.subplot(2,2,2+i)
        showpatch(derivs[i][0][0], imderiv)
        plt.axis(ax)
    plt.savefig('3.png')

    
    
