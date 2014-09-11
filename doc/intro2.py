import matplotlib
matplotlib.use('Agg')

if True:
    import numpy as np
    import pylab as plt
    from tractor import *
    from tractor.galaxy import *
    from tractor.sersic import *

    # size of image
    W,H = 40,40

    # PSF size
    psfsigma = 1.

    # per-pixel noise
    noisesigma = 0.01

    # create tractor.Image object for rendering synthetic galaxy
    # images 
    tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)) / noisesigma,
                psf=NCircularGaussianPSF([psfsigma], [1.]))

    # Exponential galaxy -- GalaxyShape params are r_e [arcsec], ab, phi [deg]
    exp = ExpGalaxy(PixPos(10,10), Flux(10.), GalaxyShape(3., 0.5, 45.))

    # Composite (deV+exp) galaxy -- params are position, exp
    # brightness, exp shape, dev brightness, dev shape.  Notice that
    # we're using the "EllipseE" ellipse -- r, e1, e2.
    comp = CompositeGalaxy(PixPos(10,30),
                           Flux(10.), EllipseE(3., 0.5, 0.),
                           Flux(10.), EllipseE(3., 0., -0.5))

    # Sersic galaxy -- this time using the "softened" ellipse param.
    sersic = SersicGalaxy(PixPos(30,10), Flux(10.),
                          EllipseESoft(1., 0.5, 0.5), SersicIndex(3.))

    # FixedComposite 
    fixed = FixedCompositeGalaxy(PixPos(30,30), Flux(10.), 0.8,
                                 EllipseE(2., 0., 0.), EllipseE(1., 0., 0.))

    tractor = Tractor([tim], [exp, comp, sersic, fixed])

    mod = tractor.getModelImage(0)

    # Plot
    plt.clf()
    plt.imshow(np.log(mod + noisesigma),
               interpolation='nearest', origin='lower', cmap='gray')
    plt.title('Galaxies')
    plt.savefig('7.png')
    
