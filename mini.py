import numpy as np

from astrometry.util.util import Tan

from tractor.sdss_galaxy import *
from tractor.engine import *
from tractor.basics import *


if __name__ == '__main__':

    # image size
    W,H = 50,50
    # fake pixel data
    data = np.zeros((H,W))

    # inverse variance; sigma = 1.
    iv = np.ones((H,W))

    # two-Gaussian PSF
    psf = GaussianMixturePSF([0.9, 0.1], np.zeros((2,2)),
                             np.array([np.eye(2), np.eye(2) * 3]))

    # simple tangent-plane projection
    ra,dec = 90.,0.
    # pixel scale 1 arcsecond per pixel
    pscale = 1. / 3600.
    wcs = FitsWcs(Tan(ra, dec, (1.+W)/2., (1.+H)/2.,
                      pscale, 0., 0., pscale, W, H))

    # image sensitivity:
    band = 'r'
    photocal = LinearPhotoCal(10., band=band)

    # flat sky
    sky = ConstantSky(0.)
    
    # Create tractor.Image object
    tim = Image(data, iv, psf, wcs, sky, photocal)

    def brightness(x):
        return NanoMaggies(**{band: x})

    # Create some sources
    ptsrc = PointSource(RaDecPos(ra, dec), brightness(10.))

    gal1 = ExpGalaxy(RaDecPos(ra - 10 * pscale, dec),
                     brightness(50.),
                     GalaxyShape(5., 0.5, 45.))
    gal2 = DevGalaxy(RaDecPos(ra + 10 * pscale, dec),
                     brightness(50.),
                     GalaxyShape(5., 0.25, 135.))

    gal3 = FixedCompositeGalaxy(RaDecPos(ra, dec + 10 * pscale),
                                brightness(50.),
                                0.6,
                                GalaxyShape(5., 0.6, 30.),
                                GalaxyShape(5., 0.3, 45.))

    cat = Catalog(ptsrc, gal1, gal2, gal3)

    # Create Tractor object
    tr = Tractor([tim], cat)
    
    # Evaluate likelihood
    lnp = tr.getLogProb()
    print 'Logprob:', lnp


    # Or, without creating a Tractor object:

    model = np.zeros_like(data)
    sky.addTo(model)
    for src in cat:
        patch = src.getModelPatch(tim)
        if patch is None:
            continue
        patch.addTo(model)
    lnp = -0.5 * ((model - data)**2 * iv).sum()
    print 'Logprob:', lnp

    

    # plot model image
    import pylab as plt
    mod = tr.getModelImage(0)
    #print 'Mod', mod.min(), mod.max()
    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower', vmin=0, vmax=10.)
    plt.colorbar()
    plt.title('model')
    plt.savefig('mod.png')
    

