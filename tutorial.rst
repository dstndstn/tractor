
Tutorial for the Tractor
========================

This tutorial (really more of a case study) goes through extending the
Tractor for a new kind of astronomical source.  Hi, Phil!

We want to create a new kind of :class:`~tractor.Source` object: a
strongly gravitationally lensed quasar.  The lens will have a visible
component: a :class:`~tractor.sdss_galaxy.DevGalaxy` galaxy profile,
as well as a dark component that defines its mass.  The lens will
produce 2 to 4 images of the quasar.

The lens model produces only approximate estimates of the brightness
of the multiple quasar images, so we will need a "fudge factor" for
the magnitudes predicted by the lens model.

We want to create a class to hold our Lensed Quasar::

    from tractor import MultiParams
    from tractor.sdss_galaxy import DevGalaxy

    class LensedQuasar(MultiParams):
        @staticmethod
        def getNamedParams():
            return dict(light=0, mass=1, quasar=2, magfudge=3)
 
We chose to make it inherit from :class:`~tractor.MultiParams` because
we want to think of it as being *composed* of the *light* (visible
component of the lens that determines its appearance), *mass* (the
dark mass of the lens that determines its lensing behavior), and the
*quasar* being lensed.  We also have *magfudge*, our fudge-factor for
the quasar's brightness at each image.

We want our ``LensedQuasar`` to be a :class:`~tractor.Source` that can
be manipulated by the Tractor, though.  We therefore have to implement
the interface described by ``Source``; we have to make our
``LensedQuasar`` quack like a ``Source`` duck.

To do this, we add the methods defined in ``Source`` to our ``LensedQuasar``::

    from tractor import PointSource

    class LensedQuasar(MultiParams):
        # ... as before ...

        def getModelPatch(self, img):
            # We start by rendering the visible lens galaxy.
            patch = self.light.getModelPatch(img)

            # We will use the lens model to predict the quasar's image positions.
            positions,mags = self.mass.getLensedImages(self.light.position, self.quasar)
            # 'positions' should be a list of RaDecPos objects
            # 'mags' should be a list of Mags objects

            for pos,mag,fudge in zip(positions, mags, self.magfudge):
                # For each image of the quasar, we will create a PointSource
                ps = PointSource(pos, mag + fudge)
                # ... and add it to the patch.
                patch += ps.getModelPatch(img)

            return patch

        def getParamDerivatives(img):
            pass


In the ``getModelPatch`` method, we have to return a
:class:`~tractor.Patch` object: a synthetic rendering of our
``LensedQuasar`` as it would appear in the given
:class:`~tractor.Image`.  We will do that by combining the appearance
of ``self.light`` -- the visible component of the lens -- with the
multiple images of the quasar whose positions and brightnesses are
estimated by the lensing model, ``self.mass``.


Now, what is the ``mass`` going to look like?  It is going to have
parameters that we want the Tractor to be able to optimize, so it has
to be a :class:`~tractor.Params`.  Actually, as you might have
guessed, it just has to quack like a :class:`~tractor.Params`.  Since
our ``mass`` is just going to have a few parameters, we could inherit
from :class:`~tractor.ParamList`::

    from tractor import ParamList

    class LensingMass(ParamList):

        @staticmethod
        def getNamedParams():
            return dict(mass=0, radius=1)

        def getStepSizes(self):
            '''We're using units of solar masses and arcsec'''
            return [1e12, 0.1]

        def getLensedImages(self, mypos, quasar):
            pass

The ``getLensedImages`` function is the one we're going to call from
``LensedQuasar.getModelPatch()`` to predict the lensed image
properties.

Let's fill in the blanks and get the code to run.  To create a
``LensedQuasar`` object, we'll have to create its components.  We will
mock up the ``Quasar`` and ``MagFudge`` classes.  Currently ``Quasar``
doesn't even have any parameters, and that's ok::

    from tractor import RaDecPos, Mags
    from tractor.sdss_galaxy import GalaxyShape

    class Quasar(ParamList):
        pass
    
    class MagFudge(ParamList):
        pass
    

    if __name__ == '__main__':
        # Create properties of the lensing galaxy:
        pos = RaDecPos(234.5, 17.9)
        bright = Mags(r=17.4, g=18.9, order=['g','r'])
        # GalaxyShape( re [arcsec], ab ratio, phi [deg] )
        shape = GalaxyShape(2., 0.5, 48.)
        light = DevGalaxy(pos, bright, shape)
        
        mass = LensingMass(1e14, 0.1)
        
        quasar = Quasar()
        
        # Four parameters for up to four images.
        fudge = MagFudge(0., 0., 0., 0.)
        
        # Create a LensedQuasar object from its components.
        lq = LensedQuasar(light, mass, quasar, fudge)
        
        print 'LensedQuasar params:'
        for nm,val in zip(lq.getParamNames(), lq.getParams()):
            print '  ', nm, '=', val
    

and this will print::

    LensedQuasar params:
       light.pos.ra = 234.5
       light.pos.dec = 17.9
       light.brightness.g = 18.9
       light.brightness.r = 17.4
       light.shape.re = 2.0
       light.shape.ab = 0.5
       light.shape.phi = 48.0
       mass.mass = 1e+14
       mass.radius = 0.1
       magfudge.param0 = 0.0
       magfudge.param1 = 0.0
       magfudge.param2 = 0.0
       magfudge.param3 = 0.0
    
