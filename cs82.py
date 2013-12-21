import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import logging
from glob import glob

from astrometry.util.fits import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.file import *
from astrometry.util.plotutils import *
from astrometry.util.util import *
from astrometry.util.ttime import *
from astrometry.sdss import *
from astrometry.libkd.spherematch import *

from tractor import *
from tractor.sdss import *
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

from photoobjs import *

data_dir = 'data/cs82'
window_flist = 'window_flist.fits'

def get_cs82_sources(T, maglim=25, bands=['u','g','r','i','z']):
    srcs = Catalog()
    isrcs = []
    for i,t in enumerate(T):
        if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
            #print 'PSF'
            themag = t.mag_psf
            nm = NanoMaggies.magToNanomaggies(themag)
            m = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
            srcs.append(PointSource(RaDecPos(t.ra, t.dec), m))
            isrcs.append(i)
            continue

        if t.mag_disk > maglim and t.mag_spheroid > maglim:
            #print 'Faint'
            continue

        # deV: spheroid
        # exp: disk

        dmag = t.mag_spheroid
        emag = t.mag_disk

        # SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

        if dmag <= maglim:
            shape_dev = GalaxyShape(t.spheroid_reff_world * 3600.,
                                    t.spheroid_aspect_world,
                                    t.spheroid_theta_world + 90.)

        if emag <= maglim:
            shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600.,
                                    t.disk_aspect_world,
                                    t.disk_theta_world + 90.)

        pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

        isrcs.append(i)
        if emag > maglim and dmag <= maglim:
            nm = NanoMaggies.magToNanomaggies(dmag)
            m_dev = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
            srcs.append(DevGalaxy(pos, m_dev, shape_dev))
            continue
        if emag <= maglim and dmag > maglim:
            nm = NanoMaggies.magToNanomaggies(emag)
            m_exp = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
            srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
            continue

        # print 'Composite'
        nmd = NanoMaggies.magToNanomaggies(dmag)
        nme = NanoMaggies.magToNanomaggies(emag)
        nm = nmd + nme
        fdev = (nmd / nm)
        m = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
        srcs.append(FixedCompositeGalaxy(pos, m, fdev, shape_exp, shape_dev))

    #print 'Sources:', len(srcs)
    return srcs, np.array(isrcs)


def getTables(cs82field, enclosed=True, extra_cols=[]):
    fn = os.path.join(data_dir, 'masked.%s_y.V2.7A.swarp.cut.deVexp.fit' % cs82field)
    print 'Reading', fn
    T = fits_table(fn,
            hdu=2, column_map={'ALPHA_J2000':'ra',
                               'DELTA_J2000':'dec'},
            columns=[x.upper() for x in
                     ['ALPHA_J2000', 'DELTA_J2000',
                      'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
                      'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
                      'disk_theta_world', 'spheroid_reff_world',
                      'spheroid_aspect_world', 'spheroid_theta_world',
                      'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
    ra0,ra1 = T.ra.min(), T.ra.max()
    dec0,dec1 = T.dec.min(), T.dec.max()
    print 'RA', ra0,ra1
    print 'Dec', dec0,dec1
    T.index = np.arange(len(T))

    # ASSUME no RA wrap-around in the catalog
    trad = 0.5 * np.hypot(ra1 - ra0, dec1 - dec0)
    tcen = radectoxyz((ra1+ra0)*0.5, (dec1+dec0)*0.5)

    frad = 0.5 * np.hypot(13., 9.) / 60.

    fn = 'sdssfield-%s.fits' % cs82field
    if os.path.exists(fn):
        print 'Reading', fn
        F = fits_table(fn)
    else:
        F = fits_table(window_flist)

        # These runs don't appear in DAS
        F.cut(F.rerun != "157")

        # For Stripe 82, mu-nu is aligned with RA,Dec.
        rd = []
        rd.append(munu_to_radec_deg(F.mu_start, F.nu_start, F.node, F.incl))
        rd.append(munu_to_radec_deg(F.mu_end,   F.nu_end,   F.node, F.incl))
        rd = np.array(rd)
        F.ra0  = np.min(rd[:,0,:], axis=0)
        F.ra1  = np.max(rd[:,0,:], axis=0)
        F.dec0 = np.min(rd[:,1,:], axis=0)
        F.dec1 = np.max(rd[:,1,:], axis=0)

        I = np.flatnonzero((F.ra0 <= T.ra.max()) *
                           (F.ra1 >= T.ra.min()) *
                           (F.dec0 <= T.dec.max()) *
                           (F.dec1 >= T.dec.min()))
        print 'Possibly overlapping fields:', len(I)
        F.cut(I)

        # When will I ever learn not to cut on RA boxes when there is wrap-around?
        xyz = radectoxyz(F.ra, F.dec)
        r2 = np.sum((xyz - tcen)**2, axis=1)
        I = np.flatnonzero(r2 < deg2distsq(trad + frad))
        print 'Possibly overlapping fields:', len(I)
        F.cut(I)

        F.enclosed = ((F.ra0 >= T.ra.min()) *
                      (F.ra1 <= T.ra.max()) *
                      (F.dec0 >= T.dec.min()) *
                      (F.dec1 <= T.dec.max()))
        
        # Sort by distance from the center of the field.
        ra  = (T.ra.min()  + T.ra.max() ) / 2.
        dec = (T.dec.min() + T.dec.max()) / 2.
        I = np.argsort( ((F.ra0  + F.ra1 )/2. - ra )**2 +
                        ((F.dec0 + F.dec1)/2. - dec)**2 )
        F.cut(I)

        F.writeto(fn)
        print 'Wrote', fn

    if enclosed:
        F.cut(F.enclosed)
        print 'Enclosed fields:', len(F)
        
    return T,F


def main(opt, cs82field):
    t0 = Time()
    
    bands = opt.bands

    ps = PlotSequence('cs82')

    version = get_svn_version()
    print 'SVN version info:', version
    
    T,F = getTables(cs82field, enclosed=False)

    sdss = DR9(basedir='data/unzip')
    if opt.local:
        sdss.useLocalTree()
        sdss.saveUnzippedFiles('data/unzip')

    ### HACK -- ignore 0/360 issues
    ra0 = T.ra.min()
    ra1 = T.ra.max()
    dec0 = T.dec.min()
    dec1 = T.dec.max()
    print 'RA range:', ra0, ra1
    print 'Dec range:', dec0, dec1
    # check for wrap-around
    assert(ra1 - ra0 < 2.)

    # Read SDSS objects to initialize fluxes (and fill in holes?)
    # create fake WCS for this area...
    pixscale = 1./3600.
    decpix = int(np.ceil((dec1 - dec0) / pixscale))
    # HACK -- ignoring cos(dec)
    rapix = int(np.ceil((ra1 - ra0) / pixscale))
    wcs = Tan((ra0 + ra1)/2., (dec0+dec1)/2., rapix/2 + 1, decpix/2 + 1,
              pixscale, 0., 0., pixscale, rapix, decpix)
    pa = PrimaryArea()
    S = read_photoobjs(sdss, wcs, 1./3600., pa=pa, cols=['ra','dec','cmodelflux',
                                                         'resolve_status'])
    print 'Read', len(S), 'SDSS objects'

    plt.clf()
    plothist(T.ra, T.dec, 200, imshowargs=dict(cmap='gray'))
    #plt.plot([ra0,ra0,ra1,ra1,ra0], [dec0,dec1,dec1,dec0,dec0], 'r-')
    for f in F:
        plt.plot([f.ra0,f.ra0,f.ra1,f.ra1,f.ra0], [f.dec0,f.dec1,f.dec1,f.dec0,f.dec0], 'b-', alpha=0.5)
    plt.title('%s: %i SDSS fields' % (cs82field, len(F)))
    ps.savefig()


    #decs = np.linspace(dec0, dec1, 20)
    #ras  = np.linspace(ra0,  ra1, 20)
    decs = np.linspace(dec0, dec1, 2)
    #ras  = np.linspace(ra0,  ra1, 41)
    # DEBUG -- ++quickness
    ras  = np.linspace(ra0,  ra1, 101)

    print 'Score range:', F.score.min(), F.score.max()
    print 'Before score cut:', len(F)
    F.cut(F.score > 0.5)
    print 'Cut on score:', len(F)

    T.phot_done = np.zeros(len(T), bool)

    Tcats = []

    for decslice,(dlo,dhi) in enumerate(zip(decs, decs[1:])):
        print 'Dec slice:', dlo, dhi
        for raslice,(rlo,rhi) in enumerate(zip(ras, ras[1:])):
            print 'RA slice:', rlo, rhi

            tslice0 = Time()

            # in deg
            margin = 15. / 3600.
            Ti = T[((T.dec + margin) >= dlo) * ((T.dec - margin) <= dhi) *
                   ((T.ra  + margin) >= rlo) * ((T.ra  - margin) <= rhi)]
            Ti.marginal = np.logical_not((Ti.dec >= dlo) * (Ti.dec <= dhi) *
                                         (Ti.ra  >= rlo) * (Ti.ra  <= rhi))
            print len(Ti), 'sources in RA,Dec slice'
            print len(np.flatnonzero(Ti.marginal)), 'are in the margins'

            Fi = F[np.logical_not(np.logical_or(F.dec0 > dhi, F.dec1 < dlo)) *
                   np.logical_not(np.logical_or(F.ra0  > rhi, F.ra1  < rlo))]
            print len(Fi), 'fields in RA,Dec slice'

            plt.clf()
            plothist(Ti.ra, Ti.dec, 200, imshowargs=dict(cmap='gray'))
            plt.plot([rlo,rlo,rhi,rhi,rlo], [dlo,dhi,dhi,dlo,dlo], 'r-')
            for f in Fi:
                plt.plot([f.ra0,f.ra0,f.ra1,f.ra1,f.ra0], [f.dec0,f.dec1,f.dec1,f.dec0,f.dec0], 'b-', alpha=0.5)
            plt.title('%s slice d%i r%i: %i SDSS fields' % (cs82field, decslice, raslice, len(Fi)))
            ps.savefig()

            print 'Creating Tractor sources...'
            maglim = 24
            cat,icat = get_cs82_sources(Ti, maglim=maglim, bands=bands)
            print 'Got', len(cat), 'sources'
            Tcat = Ti[icat]
            print len(Tcat), 'in table'
            Tcats.append(Tcat)

            print 'Matching to SDSS sources...'
            print 'N cat', len(Tcat)
            print 'N SDSS', len(S)
            I,J,d = match_radec(Tcat.ra, Tcat.dec, S.ra, S.dec, 1./3600., nearest=True)
            print 'found', len(I), 'matches'
            # initialize fluxes based on SDSS matches
            flux = S.cmodelflux
            for i,j in zip(I, J):
                #print 'src', cat[i]
                for band in bands:
                    bi = 'ugriz'.index(band)
                    setattr(cat[i].getBrightness(), band, flux[j, bi])
                #print 'src', cat[i]

            # FIXME -- freeze marginal sources!
            
            for band in bands:
                cat.freezeParamsRecursive('*')
                cat.thawPathsTo(band)

                tb0 = Time()

                tims = []
                sigs = []
                npix = 0
                for i,(r,c,f) in enumerate(zip(Fi.run, Fi.camcol, Fi.field)):
                    print 'Reading', (i+1), 'of', len(Fi), ':', r,c,f,band
                    tim,inf = get_tractor_image_dr9(r, c, f, band, sdss=sdss,
                                                    nanomaggies=True, zrange=[-2,5],
                                                    roiradecbox=[rlo,rhi,dlo,dhi],
                                                    invvarIgnoresSourceFlux=True)
                    if tim is None:
                        continue

                    # Get SDSS sources to fill in holes....?

                    (H,W) = tim.shape
                    print 'Tim', tim.shape
                    tim.wcs.setConstantCd(W/2., H/2.)
                    del tim.origInvvar
                    del tim.starMask
                    del tim.mask
                    # needed for optimize_forced_photometry with rois
                    #del tim.invvar
                    tims.append(tim)
                    sigs.append(1./np.sqrt(np.median(tim.invvar)))
                    npix += (H*W)
                    print 'got', (H*W), 'pixels, total', npix
                    print 'Read image', i+1, 'in band', band, ':', Time()-tb0

                print 'Read', len(tims), 'images'
                print 'total of', npix, 'pixels'

                if False:
                    plt.clf()
                    plothist(Ti.ra, Ti.dec, 200, imshowargs=dict(cmap='gray'))
                    plt.plot([rlo,rlo,rhi,rhi,rlo], [dlo,dhi,dhi,dlo,dlo], 'r-')
                    for tim in tims:
                        H,W = tim.shape
                        rd0 = tim.getWcs().pixelToPosition(0,0)
                        rd1 = tim.getWcs().pixelToPosition(W-1,H-1)
                        plt.plot([rd0.ra,rd0.ra,rd1.ra,rd1.ra,rd0.ra], [rd0.dec,rd1.dec,rd1.dec,rd0.dec,rd0.dec], 'b-', alpha=0.5)
                    plt.title('%s slice d%i r%i: %i SDSS fields' % (cs82field, decslice, raslice, len(tims)))
                    ps.savefig()

                sig1 = np.median(sigs)
                minsig = 0.1
                minsb= minsig * sig1
                print 'Sigma1:', sig1, 'minsig', minsig, 'minsb', minsb
                
                tractor = Tractor(tims, cat)
                tractor.freezeParam('images')
                sz = 8
                wantims = True

                tp0 = Time()
                print 'Starting forced phot:', Time()-tb0
                print '(since start of band)'

                R = tractor.optimize_forced_photometry(
                    minsb=minsb, mindlnp=1., wantims=wantims,
                    fitstats=True, variance=True,
                    shared_params=False, use_ceres=True,
                    BW=sz, BH=sz)

                print 'Forced phot finished:', Time()-tb0

                IV = R.IV
                fitstats = R.fitstats

                nm = np.array([src.getBrightness().getBand(band)
                               for src in tractor.getCatalog()])
                nm_ivar = IV
                assert(len(nm) == len(Tcat))
                assert(len(nm_ivar) == len(Tcat))

                tag = ''

                Tcat.set('sdss_%s_nanomaggies%s' % (band, tag), nm)
                Tcat.set('sdss_%s_nanomaggies_invvar%s' % (band, tag), nm_ivar)
                dnm = 1./np.sqrt(nm_ivar)
                mag = NanoMaggies.nanomaggiesToMag(nm)
                dmag = np.abs((-2.5 / np.log(10.)) * dnm / nm)
                Tcat.set('sdss_%s_mag%s' % (band, tag), mag)
                Tcat.set('sdss_%s_mag_err%s' % (band, tag), dmag)
                if fitstats is not None:
                    fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
                    for k in fskeys:
                        Tcat.set(k + '_' + band + tag, getattr(fitstats, k).astype(np.float32))
                stat = R.ceres_status
                func_tol = (stat['termination'] == 2)
                steps = stat['steps_successful']
                Tcat.set('fit_ok_%s%s' % (band, tag), np.array([(func_tol and steps > 0)] * len(T)))
                if wantims:
                    ims0 = R.ims0
                    ims1 = R.ims1

                    nims = len(tims)
                    cols = int(np.ceil(np.sqrt(nims)))
                    rows = int(np.ceil(nims / float(cols)))
                    
                    plt.clf()
                    for i,tim in enumerate(tims):
                        plt.subplot(rows, cols, i+1)
                        ima = dict(interpolation='nearest', origin='lower',
                                   vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
                        img = tim.getImage()
                        plt.imshow(img, **ima)
                        plt.xticks([]); plt.yticks([])
                        plt.title(tim.name)
                    plt.suptitle('Data: SDSS %s' % band)
                    ps.savefig()

                    plt.clf()
                    print 'ims1:', len(ims1)
                    print 'tims:', len(tims)
                    # for i,tim in enumerate(tims):
                    #     plt.subplot(rows, cols, i+1)
                    #     ima = dict(interpolation='nearest', origin='lower',
                    #                vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
                    #     mod = tractor.getModelImage(i)
                    #     plt.imshow(mod, **ima)
                    #     plt.xticks([]); plt.yticks([])
                    #     plt.title(tim.name)
                    for i,(im,mod,ie,chi,roi) in enumerate(ims1):
                        plt.subplot(rows, cols, i+1)
                        ima = dict(interpolation='nearest', origin='lower',
                                   vmin=tim.zr[0], vmax=tim.zr[1], cmap='gray')
                        plt.imshow(mod, **ima)
                        plt.xticks([]); plt.yticks([])
                        plt.title(tims[i].name)
                    plt.suptitle('Models: SDSS %s' % band)
                    # for i,tim in enumerate(tims):
                    #     plt.subplot(rows, cols, i+1)
                    #     ax = plt.axis()
                    #     xy = np.array([tim.getWcs().positionToPixel(RaDecPos(r,d))
                    #                    for r,d in zip(T.ra, T.dec)])
                    #     plt.plot(xy[:,0], xy[:,1], 'r+', ms=15, mew=1.)
                    #     if len(sdssobjs):
                    #         xy = np.array([tim.getWcs().positionToPixel(s.getPosition())
                    #                        for s in sdssobjs])
                    #         plt.plot(xy[:,0], xy[:,1], 'gx', ms=10, mew=1.)
                    #     plt.axis(ax)
                    ps.savefig()

                    del ims0
                    del ims1
                del R
                del tims
                del tractor
            
            print 'Slice:', Time()-tslice0
            print 'Total:', Time()-t0

            Tall = merge_tables(Tcats)
            print 'Total of', len(Tall), 'results vs', len(T), 'in catalog'
            print 'Tall:', len(Tall)
            Tall.about()
            print 'T:', len(T)
            T.about()
            Tcols = T.get_columns()
            Tx = T.copy()
            for c in Tall.get_columns():
                if c in Tcols:
                    #print 'Skipping column', c
                    print 'Updating column', c
                    Tx.get(c)[Tall.index] = Tall.get(c)
                    continue
                print 'Setting column', c
                X = np.zeros(len(T), Tall.get(c).dtype)
                print 'col', c, 'X:', len(X), X.dtype
                print 'index:', len(Tall.index), Tall.index.dtype
                X[Tall.index] = Tall.get(c)
                Tx.set(c, X)
            Tx.phot_done[Tall.index] = True
            fn = 'cs82-phot-%s-slice%i.fits' % (cs82field, decslice * (len(ras)-1) + raslice)
            Tx.writeto(fn)
            Tx.about()
            print 'Wrote', fn
            Tx.cut(Tx.phot_done)
            fn = 'cs82-phot-%s-slice%i-cut.fits' % (cs82field, decslice * (len(ras)-1) + raslice)
            Tx.writeto(fn)
            Tx.about()
            print 'Wrote', fn
                
            del Tx


    Tall = merge_tables(Tcats)
    print 'Total of', len(Tall), 'results vs', len(T), 'in catalog'
    Tcols = T.get_columns()
    for c in Tall.get_columns():
        if c in Tcols:
            continue
        X = np.zeros_like(len(T), Tall.get(c).dtype)
        X[Tall.index] = Tall.get(c)
        T.set(c, Tall)

    fn = 'cs82-phot-%s.fits' % cs82field
    T.writeto(fn)
    print 'Wrote', fn
    return

if __name__ == '__main__':
    import optparse
    Time.add_measurement(MemMeas)
    sdss = DR9()
    url = sdss.dasurl
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-b', dest='bands', type=str, default='ugriz',
                      help='SDSS bands (default %default)')
    parser.add_option('-l', dest='local', action='store_true', default=False,
                      help='Use local SDSS tree?')
    parser.add_option('--das', default=url,
                      help='SDSS DAS url: default %default')

    opt,args = parser.parse_args()

    #lvl = logging.INFO
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    cs82field = 'S82p18p'
    T = main(opt, cs82field)
    
