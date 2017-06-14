from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
import os

'''
In this example script, we take a tiny subimage of real DECam data,
from the DECam Legacy Survey (DECaLS), Data Release 3.  Occasionally,
the DECaLS pipeline fails to separate nearby sources, or just misses
some sources.  Here, we try adding in a new source and re-fitting the
existing sources to produce a better fit.  We read sources from the
original catalog and add them in to initialize, and then add a new
source and re-optimize.
'''

from tractor import (
    Image, ConstantFitsWcs, RaDecPos, LinearPhotoCal,
    ConstantSky, NanoMaggies, DevGalaxy, ExpGalaxy,
    EllipseE, Tractor, PointSource)
from tractor.psfex import PixelizedPsfEx

from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.fits import fits_table

'''
Image data are from:
  The "image-sub" and "invvar-sub" files are from:
  fitscopy image-decam-520206-S16-z.fits.gz"[35:104, 1465:1534]" decam-520206-S16-image-sub.fits
  fitscopy iv-decam-520206-S16-z.fits.gz"[35:104, 1465:1534]" decam-520206-S16-invvar-sub.fits
Catalog data are from: DECaLS DR3 (legacysurvey.org/dr3)
  tractor-1816p325-sub.fits
  cut to:
  I = np.flatnonzero(np.hypot(T.ra - 181.6309, T.dec - 32.5399) < 2e-3)
  -> [667, 668]
'''

def main():
    # Where are the data?
    datadir = os.path.join(os.path.dirname(__file__), 'data-decam')
    name = 'decam-520206-S16'
    imagefn  = os.path.join(datadir, '%s-image-sub.fits'  % name)
    invvarfn = os.path.join(datadir, '%s-invvar-sub.fits' % name)
    psfexfn  = os.path.join(datadir, '%s-psfex.fits'  % name)
    catfn    = os.path.join(datadir, 'tractor-1816p325-sub.fits')
    
    # Read the image and inverse-variance maps.
    image  = fitsio.read(imagefn)
    invvar = fitsio.read(invvarfn)
    # The DECam inverse-variance maps are unfortunately corrupted
    # by fpack, causing zeros to become negative.  Fix those.
    invvar[invvar < np.median(invvar)*0.1] = 0.
    H,W = image.shape
    print('Subimage size:', image.shape)

    # For the PSF model, we need to know what subimage region this is:
    subimage_offset = (35, 1465)
    # We also need the calibrated zeropoint.
    zeropoint = 24.7787

    # What filter was this image taken in?  (z)
    prim_header = fitsio.read_header(imagefn)
    band = prim_header['FILTER'].strip()[0]
    print('Band:', band)
    # These DECam images were calibrated so that the zeropoints need
    # an exposure-time factor, so add that in.
    exptime = prim_header['EXPTIME']
    zeropoint += 2.5 * np.log10(exptime)
    
    # Read the PsfEx model file
    psf = PixelizedPsfEx(psfexfn)
    # Instantiate a constant pixelized PSF at the image center
    # (of the subimage)
    x0,y0 = subimage_offset
    psf = psf.constantPsfAt(x0 + W/2., y0 + H/2.)

    # Load the WCS model from the header
    # We convert from the RA---TPV type to RA---SIP
    header = fitsio.read_header(imagefn, ext=1)
    wcsobj = wcs_pv2sip_hdr(header, stepsize=10)

    # We'll just use a rough sky estimate...
    skyval = np.median(image)

    # Create the Tractor Image (tim).
    tim = Image(data=image, invvar=invvar, psf=psf,
                wcs=ConstantFitsWcs(wcsobj),
                sky=ConstantSky(skyval),
                photocal=LinearPhotoCal(NanoMaggies.zeropointToScale(zeropoint), band=band)
                )

    # Read the official DECaLS DR3 catalog -- it has only two sources in this subimage.
    catalog = fits_table(catfn)
    print('Read', len(catalog), 'sources')
    print('Source types:', catalog.type)

    # Create Tractor sources corresponding to these two catalog
    # entries.

    # In DECaLS, the "SIMP" type is a round Exponential galaxy with a
    # fixed 0.45" radius, but we'll treat it as a general Exp galaxy.

    sources = []
    for c in catalog:
        # Create a "position" object given the catalog RA,Dec
        position = RaDecPos(c.ra, c.dec)
        # Create a "brightness" object; in the catalog, the fluxes are
        # stored in a [ugrizY] array, so pull out the right index
        band_index = 'ugrizY'.index(band)
        flux = c.decam_flux[band_index]
        brightness = NanoMaggies(**{ band: flux})

        # Depending on the source classification in the catalog, pull
        # out different fields for the galaxy shape, and for the
        # galaxy type.  The DECaLS catalogs, conveniently, store
        # galaxy shapes as (radius, e1, e2) ellipses.
        if c.type.strip() == 'DEV':
            shape = EllipseE(c.shapedev_r, c.shapedev_e1, c.shapedev_e2)
            galclass = DevGalaxy
        elif c.type.strip() == 'SIMP':
            shape = EllipseE(c.shapeexp_r, c.shapeexp_e1, c.shapeexp_e2)
            galclass = ExpGalaxy
        else:
            assert(False)
        # Create the tractor galaxy object
        source = galclass(position, brightness, shape)
        print('Created', source)
        sources.append(source)

    # Create the Tractor object -- a list of tractor Images and a list of tractor sources.
    tractor = Tractor([tim], sources)
    
    # Render the initial model image.
    print('Getting initial model...')
    mod = tractor.getModelImage(0)
    make_plot(tim, mod, 'Initial Scene', 'mod0.png')

    # Instantiate a new source at the location of the unmodelled peak.
    print('Adding new source...')
    # Find the peak very naively...
    ipeak = np.argmax((image - mod) * tim.inverr)
    iy,ix = np.unravel_index(ipeak, tim.shape)
    print('Residual peak at', ix,iy)
    # Compute the RA,Dec location of the peak...
    radec = tim.getWcs().pixelToPosition(ix, iy)
    print('RA,Dec', radec)

    # Try modelling it as a point source.
    # We'll initialize the brightness arbitrarily to 1 nanomaggy (= mag 22.5)
    brightness = NanoMaggies(**{ band: 1.})
    source = PointSource(radec, brightness)

    # Add it to the catalog!
    tractor.catalog.append(source)

    # Render the new model image with this source added.
    mod = tractor.getModelImage(0)
    make_plot(tim, mod, 'New Source (Before Fit)', 'mod1.png')

    print('Fitting new source...')
    # Now we're going to fit for the properties of the new source we
    # added.
    # We don't want to fit for any of the image calibration properties:
    tractor.freezeParam('images')
    # And we don't (yet) want to fit the existing sources.  The new
    # source is index number 2, so freeze everything else in the catalog.
    tractor.catalog.freezeAllBut(2)

    print('Fitting parameters:')
    tractor.printThawedParams()

    # Do the actual optimization:
    tractor.optimize_loop()

    mod = tractor.getModelImage(0)
    make_plot(tim, mod, 'New Source Fit', 'mod2.png')

    print('Fitting sources simultaneously...')
    # Now let's unfreeze all the sources and fit them simultaneously.
    tractor.catalog.thawAllParams()
    tractor.printThawedParams()

    tractor.optimize_loop()

    mod = tractor.getModelImage(0)
    make_plot(tim, mod, 'Simultaneous Fit', 'mod3.png')


def make_plot(tim, mod, title, filename):
    image = tim.getImage()
    # imshow arguments for images
    mn,mx = np.percentile(image, [25,99])
    ima = dict(interpolation='nearest', origin='lower',
               cmap='gray', vmin=mn, vmax=mx)
    # imshow arguments for chi images
    chia = dict(interpolation='nearest', origin='lower',
               cmap='gray', vmin=-5, vmax=+5)
    
    plt.clf()
    plt.subplot(2,2,1)
    plt.imshow(image, **ima)
    plt.xticks([]); plt.yticks([])
    plt.title('Data')
    plt.subplot(2,2,2)
    plt.imshow(mod, **ima)
    plt.xticks([]); plt.yticks([])
    plt.title('Model')
    plt.subplot(2,2,3)
    inverr = tim.getInvError()
    plt.imshow((image - mod) * inverr, **chia)
    plt.xticks([]); plt.yticks([])
    plt.title('Chi')
    plt.subplot(2,2,4)
    noise = np.random.normal(size=tim.shape)
    noise[inverr == 0] = 0.
    noise[inverr != 0] /= inverr[inverr != 0]
    plt.imshow(mod + noise, **ima)
    plt.xticks([]); plt.yticks([])
    plt.title('Model + Noise')
    plt.suptitle(title)
    plt.savefig(filename)
    
if __name__ == '__main__':
    main()
