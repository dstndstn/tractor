from astrometry.util.util import *
from astrometry.util.fits import *

N_subtiles = 4
unwise_atlas = 'allsky-atlas.fits'
decam_pixscale = 0.27

def get_subtile_wcs(name, x, y):
    '''
    pixscale: arcsec/pixel
    '''
    nsub = N_subtiles
    pixscale = decam_pixscale

    wcs = unwise_wcs_from_name(name)
    W,H = wcs.get_width(), wcs.get_height()
    # Tweak to DECam pixel scale and number of pixels.
    D = int(np.ceil((W * wcs.pixel_scale() / pixscale) / nsub)) * nsub
    DW,DH = D,D
    wcs.set_crpix(DW/2 + 1.5, DH/2 + 1.5)
    pixscale = pixscale / 3600.
    wcs.set_cd(-pixscale, 0., 0., pixscale)
    wcs.set_imagesize(DW, DH)
    W,H = wcs.get_width(), wcs.get_height()

    subw, subh = W/nsub, H/nsub
    subwcs = Tan(wcs)
    subwcs.set_crpix(wcs.crpix[0] - x * subw, wcs.crpix[1] - y * subh)
    subwcs.set_imagesize(subw, subh)
    return subwcs

def unwise_wcs_from_name(name, atlas=unwise_atlas):
    print 'Reading', atlas
    T = fits_table(atlas)
    print 'Read', len(T), 'WISE tiles'
    I = np.flatnonzero(name == T.coadd_id)
    if len(I) != 1:
        raise RuntimeError('Failed to find WISE tile "%s"' % name)
    I = I[0]
    tra,tdec = T.ra[I],T.dec[I]
    return unwise_tile_wcs(tra, tdec)

# from unwise_coadd.py : get_coadd_tile_wcs()
def unwise_tile_wcs(ra, dec, W=2048, H=2048, pixscale=2.75):
    '''
    Returns a Tan WCS object at the given RA,Dec center, axis aligned, with the
    given pixel W,H and pixel scale in arcsec/pixel.
    '''
    cowcs = Tan(ra, dec, (W+1)/2., (H+1)/2.,
                -pixscale/3600., 0., 0., pixscale/3600., W, H)
    return cowcs
