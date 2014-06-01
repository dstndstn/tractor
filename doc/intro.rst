Introduction to using The Tractor
=================================

The Tractor is a code for optimizing or sampling from *models* of
astronomical objects.  The approach is *generative*: given
astronomical sources and a description of the image properties, the
code produces pixel-space estimates or predictions of what will be
observed in the images.  We use this estimate to produce a likelihood
for the observed data given the model: assuming our model space
actually includes the truth (it doesn't, in detail), then if we had
the optimal model parameters, the predicted image would only differ
from the actually observed image by noise.  Given a noise model of the
instrument and assuming pixelwise independent noise, the
log-likelihood is just the negative chi-squared difference: (image -
model) / noise.

To actually use the Tractor code to infer the properties of
astronomical objects in your images, you will probably have to write a
*driver* script, which will read in your data of interest, create
*tractor.Image* objects describing your images, and source objects
describing the astronomical sources of interest.  The Tractor does not
(at present) create sources itself; you have to initialize it with
reasonable guesses about the objects in your images.

*tractor.Image* objects carry the data, per-pixel noise sigma (we
 usually work with inverse-variance), and a number of *calibration*
 parameters.  These include the PSF model, astrometric calibration
 (WCS), photometric calibration, and sky background model.  Each of
 these calibrations can be parameterized and its parameters fit
 alongside the properties of the astronomical sources.

*tractor.Source* objects are rather nebulously defined, as we will see
 below.  A simple example is the *tractor.PointSource* class, which
 has a "position" and a "brightness".  A *Source* object must be able
 to *render* its appearance in a given *tractor.Image*; that is, is
 must be able to produce a pixel-space model of what it would look
 like in the given image.  To do this, it must use the image's
 calibration objects to convert the source's representation of its
 position into pixel space (via the image's WCS), convert its
 representation of its brightness into pixel counts (via the image's
 photometric calibration or "photoCal").  It also needs the image's
 PSF model.

The core Tractor code does not know or care about the exact types
(python classes) you use to represent the position and brightness.
The only requirement for a "position" or "brightness" class is that it
have the right "duck type", and that the image's PhotoCal or WCS
objects can convert it to image space.  That is, the class you use for
the "position" of sources must match the class you use for the "WCS"
of the images, and the "brightness" of the sources must match the
"PhotoCal" of the images.  Let's see an example to clarify this.

In this example, we are working in pixel space and raw counts; we use
the *PixPos* class to represent pixel positions, and the *Flux* class
to represent the image counts.  We can then use the "null" calibration
classes, which just pass through the position and flux values
unmodified.

::

    from tractor import *

    source = PointSource(PixPos(17., 27.4), Flux(23.9))

    photocal = NullPhotoCal()
    wcs = NullWCS()

    counts = photocal.brightnessToCounts(source.getBrightness())
    x,y = wcs.positionToPixel(source.getPosition())


Instead, we could chose to work in RA,Dec coordinates, so we would use
the *RaDecPos* class to represent the positions of sources in
celestial coordinates, and one of the WCS calibration classes that
expect celestial coordinates.  Similarly, we could decide to work with
brightness represented in *Mags*, and use a *MagsPhotoCal*.

::

    from tractor import *
    from astrometry.util.util import Tan

    source = PointSource(RaDecPos(42.3, 9.7), Mags(r=12., i=11.3))

    photocal = MagsPhotoCal('r', 22.5)
    wcs = FitsWcs(Tan(42.0, 9.0, 100., 100., 0.1, 0., 0., 0.1, 200., 200.))

    counts = photocal.brightnessToCounts(source.getBrightness())
    x,y = wcs.positionToPixel(source.getPosition())



