if True:
    from tractor import *
    source = PointSource(PixPos(17., 27.4), Flux(23.9))
    photocal = NullPhotoCal()
    wcs = NullWCS()
    counts = photocal.brightnessToCounts(source.getBrightness())
    x,y = wcs.positionToPixel(source.getPosition())
    print 'source', source
    print 'photocal', photocal
    print 'wcs', wcs
    print 'counts', counts
    print 'x,y', x,y

if True:
    from tractor import *
    from astrometry.util.util import Tan
    source = PointSource(RaDecPos(42.3, 9.7), Mags(r=12., i=11.3))
    photocal = MagsPhotoCal('r', 22.5)
    wcs = FitsWcs(Tan(42.0, 9.0, 100., 100., 0.1, 0., 0., 0.1, 200., 200.))
    counts = photocal.brightnessToCounts(source.getBrightness())
    x,y = wcs.positionToPixel(source.getPosition())

    print 'source', source
    print 'photocal', photocal
    print 'wcs', wcs
    print 'counts', counts
    print 'x,y', x,y
    
