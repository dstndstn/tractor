import fitsio

from astrometry.util.util import *
from astrometry.util.fits import *

from tractor import *
from tractor.galaxy import *
from tractor.ellipses import *

N_subtiles = 4
unwise_atlas = 'allsky-atlas.fits'
decam_pixscale = 0.27

# FITS catalogs
fits_typemap = { PointSource: 'S', ExpGalaxy: 'E', DevGalaxy: 'D',
                 FixedCompositeGalaxy: 'C' }

def typestring(t):
    t = repr(t).replace("<class '", '').replace("'>", "")
    return t

ellipse_types = dict([(typestring(t), t) for t in
                      [ EllipseESoft, EllipseE,
                        ]])
#{ 'tractor.ellipses.EllipseESoft': EllipseESoft,
#  'tractor.ellipses.EllipseE': EllipseE,
#  }

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

def source_param_types(src):
    def flatten_node(node):
        return reduce(lambda x,y: x+y,
                      [flatten_node(c) for c in node[1:]],
                      [node[0]])
    tree = getParamTypeTree(src)
    print 'Source param types:', tree
    types = flatten_node(tree)
    return types
    

def prepare_fits_catalog(cat, var, T, hdr, filts, fs):
    if T is None:
        T = fits_table()
    if hdr is None:
        hdr = fitsio.FITSHDR()

    hdr.add_record(dict(name='TR_VER', value=1, comment='Tractor output format version'))

    # Find a source of each type and query its parameter names, for the header.
    # ASSUMES the catalog contains at least one object of each type
    for t,ts in fits_typemap.items():
        for src in cat:
            if type(src) != t:
                continue
            print 'Parameters for', t, src
            sc = src.copy()
            sc.thawAllRecursive()
            for i,nm in enumerate(sc.getParamNames()):
                hdr.add_record(dict(name='TR_%s_P%i' % (ts, i), value=nm,
                                    comment='Tractor param name'))

            for i,t in enumerate(source_param_types(sc)):
                t = typestring(t)
                hdr.add_record(dict(name='TR_%s_T%i' % (ts, i),
                                    value=t, comment='Tractor param type'))
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

    get_tractor_fits_values(T, cat, '%s')
    cat.setParams(var)
    get_tractor_fits_values(T, cat, '%s_var')
    cat.setParams(params0)
    # Heh, no uncertainty here!
    T.delete_column('type_var')
    
    return T, hdr
        
# def convert_source_for_output(src):
#     '''
#     Converts a tractor source from our internal representation to
#     output format.
# 
#     Specifically, converts EllipseESoft to EllipseE
#     '''
#     if instance(src, (DevGalaxy, ExpGalaxy)):
#         src.shape = EllipseE.fromEllipeESoft(src.shape)
#     elif instance(src, FixedCompositeGalaxy):
#         src.shapeExp = EllipseE.fromEllipeESoft(src.shapeExp)
#         src.shapeDev = EllipseE.fromEllipeESoft(src.shapeDev)

# We'll want to compute errors in our native representation, so have a
# FITS output routine that can convert those into output format.

def get_tractor_fits_values(T, cat, pat):
    T.set(pat % 'type', np.array([fits_typemap[type(src)] for src in cat]))

    T.set(pat % 'ra',  np.array([src.getPosition().ra  for src in cat]))
    T.set(pat % 'dec', np.array([src.getPosition().dec for src in cat]))

    shapeExp = np.zeros((len(T), 3))
    shapeDev = np.zeros((len(T), 3))
    fracDev  = np.zeros(len(T))

    for i,src in enumerate(cat):
        if isinstance(src, ExpGalaxy):
            shapeExp[i,:] = src.shape.getAllParams()
        elif isinstance(src, DevGalaxy):
            shapeDev[i,:] = src.shape.getAllParams()
            fracDev[i] = 1.
        elif isinstance(src, FixedCompositeGalaxy):
            shapeExp[i,:] = src.shapeExp.getAllParams()
            shapeDev[i,:] = src.shapeDev.getAllParams()
            fracDev[i] = src.fracDev.getValue()

    T.set(pat % 'shapeExp', shapeExp)
    T.set(pat % 'shapeDev', shapeDev)
    T.set(pat % 'fracDev', fracDev)
    return




def read_fits_catalog(T, hdr):
    '''
    This is currently a weird hybrid of dynamic and hard-coded.
    '''
    rev_typemap = dict([(v,k) for k,v in fits_typemap.items()])

    bands = []
    bandcols = []
    for col in T.get_columns():
        pre,post = 'decam_', '_nanomaggies'
        if col.startswith(pre) and col.endswith(post) and len(col) == len(pre+post)+1:
            band = col[len(pre)]
            bands.append(band)
            bandcols.append(pre + band + post)
    print 'Found bands:', bands, 'in', bandcols

    cat = []
    for i,t in enumerate(T):
        clazz = rev_typemap[t.type]
        pos = RaDecPos(t.ra, t.dec)
        br = NanoMaggies(**dict([(b,t.get(c)) for b,c in zip(bands,bandcols)]))
        params = [pos, br]
        if issubclass(clazz, (DevGalaxy, ExpGalaxy)):
            # hard-code knowledge that third param is the ellipse
            eclazz = hdr['TR_%s_T3' % t.type]
            # look up that string... to avoid eval()
            eclazz = ellipse_types[eclazz]
            if issubclass(clazz, DevGalaxy):
                ell = eclazz(*t.shapedev)
            else:
                ell = eclazz(*t.shapeexp)
            params.append(ell)
            
        elif issubclass(clazz, FixedCompositeGalaxy):
            # hard-code knowledge that params are fracDev, shapeE, shapeD
            params.append(t.fracdev)
            expeclazz = hdr['TR_%s_T4' % t.type]
            deveclazz = hdr['TR_%s_T5' % t.type]
            expeclazz = ellipse_types[expeclazz]
            deveclazz = ellipse_types[deveclazz]
            ee = expeclazz(*t.shapeexp)
            de = deveclazz(*t.shapedev)
            params.append(ee)
            params.append(de)

        elif issubclass(clazz, PointSource):
            pass
        else:
            raise RuntimeError('Unknown class %s' % str(clazz))

        src = clazz(*params)
        #print 'Created source', src
        cat.append(src)
    return cat



if __name__ == '__main__':
    T=fits_table('3524p000-0-12-n16-sdss-cat.fits')
    cat = read_fits_catalog(T, T.get_header())
    print 'Read catalog:'
    for src in cat:
        print ' ', src
