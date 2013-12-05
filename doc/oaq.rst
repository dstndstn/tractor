
Once-asked Questions
====================

Q: My images have some crazy WCS that your code doesn't understand.  What do I do?
----------------------------------------------------------------------------------

My code looks like this::

    from astrometry.util.util import Tan
    from tractor import FitsWcs

    wcs = FitsWcs(Tan('myimage.fits'))

and it's failing like::

    blah

What do I do?

A:
--

One option is to create a ``Tan`` object yourself and populate it with
the required parameters::

    t = Tan()
    t.set_crpix(24, 530)
    t.set_crval(234.66, 47.8765)
    t.set_cd(1., 0., 0., 1.)
    t.set_imagesize(1024, 1024)
    wcs = FitsWcs(t)

And you'll probably actually do that by opening your image file and
parsing its crazy header cards, converting them to TAN as understood
by our code::

    import pyfits

    hdr = pyfits.open('myimage.fits')[0].header

    t = Tan()
    t.set_crpix(hdr.get('CRPIX1'), hdr.get('CRPIX2'))
    t.set_crval(hdr.get('CRVAL1'), hdr.get('CRVAL2'))
    cd1 = hdr.get('CDELT1')
    cd2 = hdr.get('CDELT2')
    # assume your images have no rotation...
    t.set_cd(cd1, 0., 0., cd2)
    t.set_imagesize(1024, 1024)
    wcs = FitsWcs(t)
    


