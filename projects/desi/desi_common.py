import fitsio

from astrometry.util.util import *
from astrometry.util.fits import *

from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

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


def get_fits_catalog(cat, var, T, hdr, filts, fs):
    if T is None:
        T = fits_table()
    if hdr is None:
        hdr = fitsio.FITSHDR()

    typemap = { PointSource: 'S', ExpGalaxy: 'E', DevGalaxy: 'D',
                FixedCompositeGalaxy: 'C' }

    # Find a source of each type and query its parameter names, for the header.
    # ASSUMES the catalog contains at least one object of each type
    for t,ts in typemap.items():
        for src in cat:
            if type(src) != t:
                continue
            print 'Parameters for', t, src
            sc = src.copy()
            sc.thawAllRecursive()
            for i,nm in enumerate(sc.getParamNames()):
                hdr.add_record(dict(name='TR_%s_P%i' % (ts, i), value=nm,
                                    comment='Tractor param name'))
            def flatten_node(node):
                return reduce(lambda x,y: x+y,
                              [flatten_node(c) for c in node[1:]],
                              [node[0]])
            tree = getParamTypeTree(sc)
            print 'Source param types:', tree
            types = flatten_node(tree)
            #print 'Flat:', types
            for i,t in enumerate(types):
                hdr.add_record(dict(name='TR_%s_T%i' % (ts, i),
                                    value=t.replace("'", '"'),
                                    comment='Tractor param types'))
            break
    print 'Header:', hdr

    params0 = cat.getParams()

    for filt in filts:
        flux = np.array([sum(b.getFlux(filt) for b in src.getBrightnesses())
                         for src in cat])

        # Oh my, this is tricky... set parameter values to the variance
        # vector so that we can read off the parameter variances via the
        # python object apis.
        cat.setParams(var)
        fluxvar = np.array([sum(b.getFlux(filt) for b in src.getBrightnesses())
                            for src in cat])
        cat.setParams(params0)
    
        flux_iv = 1./np.array(fluxvar)
        mag,dmag = NanoMaggies.fluxErrorsToMagErrors(flux, flux_iv)
    
        T.set('decam_%s_nanomaggies'        % filt, flux)
        T.set('decam_%s_nanomaggies_invvar' % filt, flux_iv)
        T.set('decam_%s_mag'                % filt, mag)
        T.set('decam_%s_mag_err'            % filt, dmag)

    if fs is not None:
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
        for k in fskeys:
            x = getattr(fs, k)
            x = np.array(x).astype(np.float32)
            T.set('decam_%s_%s' % (tim.filter, k), x.astype(np.float32))
    
    return T, hdr
        
