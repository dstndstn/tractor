
.. _basics_images:

Basic Image calibrations
========================

Sky
---

The "Sky" describes the "background" in your images---what the images
would look like in the absence of noise or astronomical sources.

.. autoclass:: tractor.NullSky
   :members:
.. autoclass:: tractor.ConstantSky
   :members:

Astrometry (World Coordinate System, WCS)
-----------------------------------------

.. autoclass:: tractor.NullWCS
   :members:

.. autoclass:: tractor.ConstantFitsWcs
   :members:

.. autoclass:: tractor.TanWcs
   :members:

Photometry calibration ("PhotoCal")
-----------------------------------

.. autoclass:: tractor.NullPhotoCal
   :members:

.. autoclass:: tractor.LinearPhotoCal
   :members:

.. autoclass:: tractor.MagsPhotoCal
   :members:

Point-spread function (PSF)
---------------------------

.. autoclass:: tractor.NCircularGaussianPSF
   :members:
.. autoclass:: tractor.GaussianMixturePSF
   :members:

