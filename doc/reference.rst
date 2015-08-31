API Reference
=============

* :ref:`ducks` -- code-as-documentation descriptions of the types of objects using by the Tractor
* :ref:`utils` -- ParamList, MultiParams, other utility types
* :ref:`basics` -- Types for standard images, magnitudes, WCSes
* :ref:`engine` -- Core Tractor routines
* :ref:`galaxy` -- SDSS exp & deV galaxies
* :ref:`sdss` -- Specific data types for handling SDSS images and catalogs
* :ref:`cfht` -- Specific data types for handling data from the Canada-France-Hawaii Telescope

Flat list
---------

* :class:`~tractor.Brightness`
* :class:`~tractor.galaxy.CompositeGalaxy`
* :class:`~tractor.Catalog`
* :class:`~tractor.ConstantSky`
* :class:`~tractor.galaxy.DevGalaxy`
* :class:`~tractor.galaxy.ExpGalaxy`
* :class:`~tractor.FitsWcs`
* :class:`~tractor.Flux`
* :class:`~tractor.GaussianMixturePSF`
* :class:`~tractor.Image`
* :class:`~tractor.Images`
* :class:`~tractor.Mag`
* :class:`~tractor.Mags`
* :class:`~tractor.MagsPhotoCal`
* :class:`~tractor.MultiParams`
* :class:`~tractor.NamedParams`
* :class:`~tractor.NCircularGaussianPSF`
* :class:`~tractor.NullPhotoCal`
* :class:`~tractor.NullWCS`
* :class:`~tractor.ParamList`
* :class:`~tractor.Params`
* :class:`~tractor.Patch`
* :class:`~tractor.PhotoCal`
* :class:`~tractor.PixPos`
* :class:`~tractor.PointSource`
* :class:`~tractor.PSF`
* :class:`~tractor.RaDecPos`
* :class:`~tractor.Sky`
* :class:`~tractor.Source`
* :class:`~tractor.Tractor`
* :class:`~tractor.WCS`



.. _ducks:

Ducks
-----
.. autoclass:: tractor.Params
   :members:

.. autoclass:: tractor.Source
   :members:

.. autoclass:: tractor.Sky
   :members:

.. autoclass:: tractor.Brightness
   :members:

.. autoclass:: tractor.PhotoCal
   :members:

.. autoclass:: tractor.Position
   :members:

.. autoclass:: tractor.WCS
   :members:

.. autoclass:: tractor.PSF
   :members:


.. _utils:

Utilities
---------
.. autoclass:: tractor.BaseParams
   :members:
.. autoclass:: tractor.NamedParams
   :members:
.. autoclass:: tractor.ScalarParam
   :members:
.. autoclass:: tractor.ParamList
   :members:
.. autoclass:: tractor.MultiParams
   :members:


.. _engine:

Core Tractor routines
---------------------
.. autoclass:: tractor.Tractor
   :members:

.. autoclass:: tractor.Catalog
   :members:

.. autoclass:: tractor.Images
   :members:

.. autoclass:: tractor.Image
   :members:

.. autoclass:: tractor.Patch
   :members:



.. _galaxy:

Galaxies
--------
.. automodule:: tractor.galaxy
   :members:
   :undoc-members:

.. _sdss:

SDSS images & catalogs
----------------------
.. automodule:: tractor.sdss
   :members:
   :undoc-members:


.. _cfht:

CFHT images & catalogs
----------------------
.. automodule:: tractor.cfht
   :members:
   :undoc-members:




