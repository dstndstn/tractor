if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from astrometry.util.util import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.ttime import Time, MemMeas

from common import *
from desi_common import *

from tractor.galaxy import *

import runbrick
from runbrick import stage_tims, _map, _get_mod, runbrick_global_init, get_rgb
from runbrick import tim_get_resamp, set_source_radii

def stage_cat(brickid=None, target_extent=None,
              targetwcs=None, tims=None, **kwargs):

    catpattern = 'pipebrick-cats/tractor-phot-b%06i.fits'

    catfn = catpattern % brickid
    T = fits_table(catfn)
    cat,invvars = read_fits_catalog(T, invvars=True)
    print 'Got catalog:', len(cat), 'sources'

    #invvars = np.array(invvars)
    print 'Invvars:', invvars

    assert(len(Catalog(*cat).getParams()) == len(invvars))
    assert(len(cat) == len(T))
    Tcat = T
    
    if target_extent is not None:
        #x0,x1,y0,y1 = target_extent
        W,H = int(targetwcs.get_width()), int(targetwcs.get_height())
        print 'W,H', W,H
        x0,x1,y0,y1 = 1,W, 1,H
        r,d = targetwcs.pixelxy2radec(np.array([x0,x0,x1,x1]),np.array([y0,y1,y1,y0]))
        r0,r1 = r.min(),r.max()
        d0,d1 = d.min(),d.max()
        margin = 0.002
        ikeep = []
        keepcat = []
        keepivs = []
        iterinvvars = invvars
        for i,src in enumerate(cat):
            N = src.numberOfParams()
            iv = iterinvvars[:N]
            iterinvvars = iterinvvars[N:]
            pos = src.getPosition()
            if (pos.ra  > r0-margin and pos.ra  < r1+margin and
                pos.dec > d0-margin and pos.dec < d1+margin):
                keepcat.append(src)
                keepivs.extend(iv)
                ikeep.append(i)
        cat = keepcat
        Tcat.cut(np.array(ikeep))
        invvars = np.array(keepivs)
        print 'Keeping', len(cat), 'sources within range'
        
    assert(Catalog(*cat).numberOfParams() == len(invvars))

    print len(tims), 'tims'
    print 'Sizes:', [tim.shape for tim in tims]

    for tim in tims:
        from tractor.psfex import CachingPsfEx
        tim.psfex.radius = 20
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)

    return dict(cat=cat, Tcat=Tcat, invvars=invvars)


def stage_tune(tims=None, cat=None, targetwcs=None, coimgs=None, cons=None,
               bands=None, invvars=None, brickid=None,
               Tcat=None, version_header=None, ps=None, **kwargs):
    tstage = t0 = Time()
    print 'kwargs:', kwargs.keys()

    #print 'invvars:', invvars

    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1

    # Caching PSF
    for tim in tims:
        from tractor.psfex import CachingPsfEx
        tim.psfex.radius = 20
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    set_source_radii(bands, orig_wcsxy0, tims, cat, minsigma)

    plt.figure(figsize=(10,10))
    plt.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)

    plt.clf()
    rgb = get_rgb(coimgs, bands)
    dimshow(rgb)
    #plt.title('Image')
    ps.savefig()

    tmpfn = create_temp(suffix='.png')
    plt.imsave(tmpfn, rgb)
    del rgb
    cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > tunebrick/coadd/image-%06i-full.jpg' % (tmpfn, brickid)
    os.system(cmd)
    os.unlink(tmpfn)

    pla = dict(ms=5, mew=1)

    ax = plt.axis()
    for i,src in enumerate(cat):
        rd = src.getPosition()
        ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, '+', color=cc, **pla)
        else:
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', **pla)
        # plt.text(x, y, '%i' % i, color=cc, ha='center', va='bottom')
    plt.axis(ax)
    ps.savefig()

    print 'Plots:', Time()-t0

    # print 'Catalog:'
    # for src in cat:
    #     print '  ', src
    # switch_to_soft_ellipses(cat)

    assert(Catalog(*cat).numberOfParams() == len(invvars))

    keepcat = []
    keepinvvars = []
    iterinvvars = invvars
    ikeep = []
    for i,src in enumerate(cat):
        N = src.numberOfParams()
        iv = iterinvvars[:N]
        iterinvvars = iterinvvars[N:]
        if not np.all(np.isfinite(src.getParams())):
            print 'Dropping source:', src
            continue
        keepcat.append(src)
        keepinvvars.extend(iv)
        #print 'Keep:', src
        #print 'iv:', iv
        #print 'sigma', 1./np.sqrt(np.array(iv))
        ikeep.append(i)
    cat = keepcat
    Tcat.cut(np.array(ikeep))
    invvars = keepinvvars
    print len(cat), 'sources with finite params'
    assert(Catalog(*cat).numberOfParams() == len(invvars))
    assert(len(iterinvvars) == 0)

    print 'Rendering model images...'
    t0 = Time()
    mods = _map(_get_mod, [(tim, cat) for tim in tims])
    print 'Getting model images:', Time()-t0

    wcsW = targetwcs.get_width()
    wcsH = targetwcs.get_height()

    t0 = Time()
    comods = []
    for iband,band in enumerate(bands):
        comod  = np.zeros((wcsH,wcsW), np.float32)
        for itim, (tim,mod) in enumerate(zip(tims, mods)):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            comod[Yo,Xo] += mod[Yi,Xi]
        comod  /= np.maximum(cons[iband], 1)
        comods.append(comod)
    print 'Creating model coadd:', Time()-t0

    plt.clf()
    dimshow(get_rgb(comods, bands))
    plt.title('Model')
    ps.savefig()
    del comods

    t0 = Time()
    keepinvvars = []
    keepcat = []
    iterinvvars = invvars
    ikeep = []
    for isrc,src in enumerate(cat):
        newiv = None
        N = src.numberOfParams()

        gc = get_galaxy_cache()
        print 'Galaxy cache:', gc
        if gc is not None:
            gc.clear()

        print 'Checking source', isrc, 'of', len(cat), ':', src
        #print 'N params:', N
        #print 'iterinvvars:', len(iterinvvars)

        oldiv = iterinvvars[:N]
        iterinvvars = iterinvvars[N:]
        recompute_iv = False

        if isinstance(src, FixedCompositeGalaxy):
            # Obvious simplification: for composite galaxies with fracdev
            # out of bounds, convert to exp or dev.
            f = src.fracDev.getClippedValue()
            if f == 0.:
                oldsrc = src
                src = ExpGalaxy(oldsrc.pos, oldsrc.brightness, oldsrc.shapeExp)
                print 'Converted comp to exp'
                #print '   ', oldsrc
                #print ' ->', src
                # pull out the invvar elements!
                pp = src.getParams()
                oldsrc.setParams(oldiv)
                newiv = oldsrc.pos.getParams() + oldsrc.brightness.getParams() + oldsrc.shapeExp.getParams()
                src.setParams(pp)
            elif f == 1.:
                oldsrc = src
                src = DevGalaxy(oldsrc.pos, oldsrc.brightness, oldsrc.shapeDev)
                print 'Converted comp to dev'
                ##print '   ', oldsrc
                print ' ->', src
                pp = src.getParams()
                oldsrc.setParams(oldiv)
                newiv = oldsrc.pos.getParams() + oldsrc.brightness.getParams() + oldsrc.shapeDev.getParams()
                src.setParams(pp)

        # treated_as_pointsource: do the bright-star check at least!
        if not isinstance(src, PointSource):
            # This is the check we use in unWISE
            if src.getBrightness().getMag('r') < 12.5:
                oldsrc = src
                src = PointSource(oldsrc.pos, oldsrc.brightness)
                print 'Bright star: replacing', oldsrc
                print 'With', src
                # Not QUITE right.
                #oldsrc.setParams(oldiv)
                #newiv = oldsrc.pos.getParams() + oldsrc.brightness.getParams()
                recompute_iv = True

        #print 'Try removing source:', src
        tsrc = Time()

        srcmodlist = []
        for itim,tim in enumerate(tims):
            patch = src.getModelPatch(tim)
            if patch is None:
                continue
            if patch.patch is None:
                continue

            # HACK -- this shouldn't be necessary, but seems to be!
            # FIXME -- track down why patches are being made with extent outside
            # that of the parent!
            H,W = tim.shape
            if patch.x0 < 0 or patch.y0 < 0 or patch.x1 > W or patch.y1 > H:
                print 'Warning: Patch extends outside tim bounds:'
                print 'patch extent:', patch.getExtent()
                print 'image size:', W, 'x', H
            patch.clipTo(W,H)
            ph,pw = patch.shape
            if pw*ph == 0:
                continue
            srcmodlist.append((itim, patch))
    
        # Try removing the source from the model;
        # check chi-squared change in the patches.
        sdlnp = 0.
        for itim,patch in srcmodlist:
            tim = tims[itim]
            mod = mods[itim]
            slc = patch.getSlice(tim)
            simg = tim.getImage()[slc]
            sie  = tim.getInvError()[slc]
            chisq0 = np.sum(((simg - mod[slc]) * sie)**2)
            chisq1 = np.sum(((simg - (mod[slc] - patch.patch)) * sie)**2)
            sdlnp += -0.5 * (chisq1 - chisq0)
        print 'Removing source: dlnp =', sdlnp
        print 'Testing source removal:', Time()-tsrc
    
        if sdlnp > 0:
            #print 'Removing source!'
            for itim,patch in srcmodlist:
                patch.addTo(mods[itim], scale=-1)
            continue

        # Try some model changes...
        newsrcs = []
        if isinstance(src, FixedCompositeGalaxy):
            newsrcs.append(ExpGalaxy(src.pos, src.brightness, src.shapeExp))
            newsrcs.append(DevGalaxy(src.pos, src.brightness, src.shapeDev))
            newsrcs.append(PointSource(src.pos, src.brightness))
        elif isinstance(src, (DevGalaxy, ExpGalaxy)):
            newsrcs.append(PointSource(src.pos, src.brightness))

        bestnew = None
        bestdlnp = 0.
        bestdpatches = None

        srcmodlist2 = [None for tim in tims]
        for itim,patch in srcmodlist:
            srcmodlist2[itim] = patch

        for newsrc in newsrcs:

            dpatches = []
            dlnp = 0.
            for itim,tim in enumerate(tims):
                patch = newsrc.getModelPatch(tim)
                if patch is not None:
                    if patch.patch is None:
                        patch = None
                if patch is not None:
                    # HACK -- this shouldn't be necessary, but seems to be!
                    # FIXME -- track down why patches are being made with extent outside
                    # that of the parent!
                    H,W = tim.shape
                    patch.clipTo(W,H)
                    ph,pw = patch.shape
                    if pw*ph == 0:
                        patch = None

                oldpatch = srcmodlist2[itim]
                if oldpatch is None and patch is None:
                    continue

                # Find difference in models
                if oldpatch is None:
                    dpatch = patch
                elif patch is None:
                    dpatch = oldpatch * -1.
                else:
                    dpatch = patch - oldpatch
                dpatches.append((itim, dpatch))
                
                mod = mods[itim]
                slc = dpatch.getSlice(tim)
                simg = tim.getImage()[slc]
                sie  = tim.getInvError()[slc]
                chisq0 = np.sum(((simg - mod[slc]) * sie)**2)
                chisq1 = np.sum(((simg - (mod[slc] + dpatch.patch)) * sie)**2)
                dlnp += -0.5 * (chisq1 - chisq0)

            #print 'Trying source change:'
            #print 'from', src
            #print '  to', newsrc
            print 'Trying source change to', type(newsrc).__name__, ': dlnp =', dlnp

            if dlnp >= bestdlnp:
                bestnew = newsrc
                bestdlnp = dlnp
                bestdpatches = dpatches

        if bestnew is not None:
            print 'Found model improvement!  Switching to',
            print bestnew
            for itim,dpatch in bestdpatches:
                dpatch.addTo(mods[itim])
            src = bestnew
            recompute_iv = True

        del srcmodlist
        del srcmodlist2

        if recompute_iv:
            dchisqs = np.zeros(src.numberOfParams())
            for tim in tims:
                derivs = src.getParamDerivatives(tim)
                h,w = tim.shape
                ie = tim.getInvError()
                for i,deriv in enumerate(derivs):
                    if deriv is None:
                        continue
                    deriv.clipTo(w,h)
                    slc = deriv.getSlice(ie)
                    chi = deriv.patch * ie[slc]
                    dchisqs[i] += (chi**2).sum()
            newiv = dchisqs

        if newiv is None:
            keepinvvars.append(oldiv)
        else:
            keepinvvars.append(newiv)
    
        keepcat.append(src)
        ikeep.append(isrc)
    cat = keepcat
    Tcat.cut(np.array(ikeep))

    gc = get_galaxy_cache()
    print 'Galaxy cache:', gc
    if gc is not None:
        gc.clear()

    assert(len(iterinvvars) == 0)
    keepinvvars = np.hstack(keepinvvars)
    assert(Catalog(*keepcat).numberOfParams() == len(keepinvvars))
    invvars = keepinvvars
    assert(len(cat) == len(Tcat))
    print 'Model selection:', Time()-t0

    t0 = Time()
    # WCS header for these images
    hdr = fitsio.FITSHDR()
    targetwcs.add_to_header(hdr)
    fwa = dict(clobber=True, header=hdr)

    comods = []
    for iband,band in enumerate(bands):
        comod  = np.zeros((wcsH,wcsW), np.float32)
        cochi2 = np.zeros((wcsH,wcsW), np.float32)
        coiv   = np.zeros((wcsH,wcsW), np.float32)
        detiv   = np.zeros((wcsH,wcsW), np.float32)
        for itim, (tim,mod) in enumerate(zip(tims, mods)):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            comod[Yo,Xo] += mod[Yi,Xi]
            ie = tim.getInvError()
            cochi2[Yo,Xo] += ((tim.getImage()[Yi,Xi] - mod[Yi,Xi]) * ie[Yi,Xi])**2
            coiv[Yo,Xo] += ie[Yi,Xi]**2

            psfnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
            detsig1 = tim.sig1 / psfnorm
            detiv[Yo,Xo] += (ie[Yi,Xi] > 0) * (1. / detsig1**2)

        comod  /= np.maximum(cons[iband], 1)
        comods.append(comod)
        del comod

        fn = 'tunebrick/coadd/chi2-%06i-%s.fits' % (brickid, band)
        fitsio.write(fn, cochi2, **fwa)
        del cochi2
        print 'Wrote', fn

        fn = 'tunebrick/coadd/image-%06i-%s.fits' % (brickid, band)
        fitsio.write(fn, coimgs[iband], **fwa)
        print 'Wrote', fn
        fitsio.write(fn, coiv, clobber=False)
        print 'Appended ivar to', fn
        del coiv

        fn = 'tunebrick/coadd/depth-%06i-%s.fits' % (brickid, band)
        fitsio.write(fn, detiv, **fwa)
        print 'Wrote', fn
        del detiv

        fn = 'tunebrick/coadd/model-%06i-%s.fits' % (brickid, band)
        fitsio.write(fn, comods[iband], **fwa)
        print 'Wrote', fn

        fn = 'tunebrick/coadd/nexp-b%06i-%s.fits' % (brickid, band)
        fitsio.write(fn, cons[iband], **fwa)
        print 'Wrote', fn

    plt.clf()
    rgb = get_rgb(comods, bands)
    dimshow(rgb)
    plt.title('Model')
    ps.savefig()
    del comods
    
    # Plot sources over top
    ax = plt.axis()
    for i,src in enumerate(cat):
        rd = src.getPosition()
        ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, '+', color=cc, **pla)
        else:
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', **pla)
        # plt.text(x, y, '%i' % i, color=cc, ha='center', va='bottom')
    plt.axis(ax)
    ps.savefig()

    tmpfn = create_temp(suffix='.png')
    plt.imsave(tmpfn, rgb)
    del rgb
    cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > tunebrick/coadd/model-%06i-full.jpg' % (tmpfn, brickid)
    os.system(cmd)
    os.unlink(tmpfn)

    assert(len(cat) == len(Tcat))
    print 'Coadd FITS files and plots:', Time()-t0

    print 'Whole stage:', Time()-tstage

    return dict(cat=cat, Tcat=Tcat, invvars=invvars)

def stage_writecat2(cat=None, Tcat=None, invvars=None, version_header=None,
                    bands=None, targetwcs=None, brickid=None,
                    **kwargs):
    t0 = Time()
    
    # Write catalog
    hdr = version_header
    thdr = Tcat._header

    hdr.add_record(dict(name='RB_TRACV', value=thdr['TRACTORV'],
                        comment='Tractor version when runbrick.py was run'))
    if 'DECALSDT' in thdr:
        hdr.add_record(dict(name='RB_DATE', value=thdr['DECALSDT'],
                            comment='Date when runbrick.py was run'))
    assert(len(cat) == len(Tcat))

    # Keep these columns...
    TT = fits_table()
    for k in ['blob','brickid','objid','sdss_run', 'sdss_camcol', 'sdss_field', 'sdss_objid',
              'sdss_cmodelflux', 'sdss_cmodelflux_ivar', 'sdss_ra', 'sdss_dec',
              'sdss_modelflux', 'sdss_modelflux_ivar',
              'sdss_psfflux', 'sdss_psfflux_ivar', 'sdss_extinction', 'sdss_flags',
              'sdss_objc_flags', 'sdss_objc_type', 'tx','ty']:
        TT.set(k, Tcat.get(k))
    TT._length = len(Tcat)

    print 'TT:', len(TT)
    print 'cat:', len(cat)

    print 'params:', Catalog(*cat).numberOfParams()
    print 'invvars:', len(invvars), invvars

    ## FIXME -- we don't update these fit-stats
    fs = None
    T,hdr = prepare_fits_catalog(Catalog(*cat), invvars, TT, hdr, bands, fs)

    ok,x,y = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.x = x.astype(np.float32)
    T.y = y.astype(np.float32)
    T.ra_ivar  = T.ra_ivar.astype(np.float32)
    T.dec_ivar = T.dec_ivar.astype(np.float32)

    decals = Decals()
    brick = decals.get_brick(brickid)
    T.brick_primary = ((T.ra  >= brick.ra1 ) * (T.ra  < brick.ra2) *
                       (T.dec >= brick.dec1) * (T.dec < brick.dec2))
    T.brickname = np.array([brick.brickname] * len(T))

    # Unpack shape columns
    T.shapeExp_r  = T.shapeExp[:,0]
    T.shapeExp_e1 = T.shapeExp[:,1]
    T.shapeExp_e2 = T.shapeExp[:,2]
    T.shapeDev_r  = T.shapeExp[:,0]
    T.shapeDev_e1 = T.shapeExp[:,1]
    T.shapeDev_e2 = T.shapeExp[:,2]
    T.shapeExp_r_ivar  = T.shapeExp_ivar[:,0]
    T.shapeExp_e1_ivar = T.shapeExp_ivar[:,1]
    T.shapeExp_e2_ivar = T.shapeExp_ivar[:,2]
    T.shapeDev_r_ivar  = T.shapeExp_ivar[:,0]
    T.shapeDev_e1_ivar = T.shapeExp_ivar[:,1]
    T.shapeDev_e2_ivar = T.shapeExp_ivar[:,2]

    for k in ['shapeExp', 'shapeDev', 'shapeExp_ivar', 'shapeDev_ivar']:
        T.delete_column(k)
              
    fn = 'tunebrick/tractor/tractor-%06i.fits' % brickid
    T.writeto(fn, header=hdr, columns=(
        'brickid brickname objid ra dec ra_ivar dec_ivar type ' +
        'x y brick_primary blob tx ty ' +
        'decam_flux decam_flux_ivar ' +
        'fracDev fracDev_ivar ' +
        'shapeExp_r shapeExp_e1 shapeExp_e2 shapeDev_r shapeDev_e1 shapeDev_e2 ' +
        'shapeExp_r_ivar shapeExp_e1_ivar shapeExp_e2_ivar ' +
        'shapeDev_r_ivar shapeDev_e1_ivar shapeDev_e2_ivar ' +
        'sdss_run sdss_camcol sdss_field sdss_objid sdss_cmodelflux ' +
        'sdss_cmodelflux_ivar sdss_ra sdss_dec sdss_modelflux sdss_modelflux_ivar ' +
        'sdss_psfflux sdss_psfflux_ivar sdss_flags sdss_objc_flags sdss_objc_type ' +
        'sdss_extinction').split(' '))
    print 'Wrote', fn
    print 'Writing catalog:', Time()-t0

def stage_recoadd(tims=None, bands=None, targetwcs=None, ps=None, brickid=None,
                  basedir=None,
                  **kwargs):
    #print 'kwargs:', kwargs.keys()
    if targetwcs is None:
        # can happen if no CCDs overlap...
        import sys
        sys.exit(0)
    
    W = targetwcs.get_width()
    H = targetwcs.get_height()

    coimgs = []
    # moo
    cowimgs = []
    #nimgs = []
    wimgs = []
    for iband,band in enumerate(bands):
        coimg  = np.zeros((H,W), np.float32)
        cowimg  = np.zeros((H,W), np.float32)
        wimg  = np.zeros((H,W), np.float32)
        nimg  = np.zeros((H,W), np.uint8)
        for tim in tims:
            if tim.band != band:
                continue
            print 'Coadding', tim.name
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            coimg[Yo,Xo] += tim.getImage()[Yi,Xi]
            nimg[Yo,Xo] += 1
            cowimg[Yo,Xo] += tim.getImage()[Yi,Xi] * tim.getInvvar()[Yi,Xi]
            wimg[Yo,Xo] += tim.getInvvar()[Yi,Xi]
            del R,Yo,Xo,Yi,Xi
        coimg /= np.maximum(nimg, 1)
        cowimg /= np.maximum(wimg, 1e-16)
        coimgs.append(coimg)
        cowimgs.append(cowimg)
        #nimgs.append(nimg)
        wimgs.append(wimg)

    for i,(wimg,cowimg,coimg) in enumerate(zip(wimgs, cowimgs, coimgs)):
        cowimg[wimg == 0] = coimg[wimg == 0]
    del wimgs
    del coimgs
    del wimg
    del coimg

    try:
        os.path.makedirs(os.path.join(basedir, 'coadd'))
    except:
        pass

    # WCS header for these images
    hdr = fitsio.FITSHDR()
    targetwcs.add_to_header(hdr)
    fwa = dict(clobber=True, header=hdr)

    for band,cow in zip(bands, cowimgs):
        fn = os.path.join(basedir, 'coadd', 'image2-%06i-%s.fits' % (brickid,band))
        fitsio.write(fn, cow, **fwa)
        print 'Wrote', fn

    return dict(coimgs=cowimgs, tims=None)

def stage_rergb(coimgs=None, bands=None, basedir=None, brickid=None,
                ps=None, **kwargs):
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)

    rgb = get_rgb(coimgs, bands)

    plt.clf()
    dimshow(rgb)
    fn = os.path.join(basedir, 'coadd', 'image2-%06i.png' % brickid)
    plt.savefig(fn)
    print 'Saved', fn

    tmpfn = create_temp(suffix='.png')
    plt.imsave(tmpfn, rgb)
    del rgb
    fn = os.path.join(basedir, 'coadd', 'image2-%06i-full.jpg' % brickid)
    cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, fn)
    os.system(cmd)
    os.unlink(tmpfn)
    print 'Wrote', fn


def stage_primage(coimgs=None, bands=None, ps=None, basedir=None,
                  **kwargs):

    '''
    Nice PR image for NOAO.  Make a synthetic brick centered on a nice galaxy in 374451

    bricks.txt:
    # brickid ra dec ra1 ra2 dec1 dec2
    1 244.70 7.41 244.5739 244.8261 7.285 7.535
    2 244.70 7.41 244.5739 244.8261 7.285 7.535

    text2fits.py -f jdddddd PR/bricks.txt PR/decals-bricks.fits
    cp decals/decals-ccds.fits PR
    (cd PR; ln -s ~/cosmo/work/decam/versions/work/calib .)
    (cd PR; ln -s ~/cosmo/work/decam/versions/work/images .)
    export DECALS_DIR=$(pwd)/PR
    python -u projects/desi/tunebrick.py -b      1 -s primage -P "pickles/PR-%(brick)06i-%%(stage)s.pickle"
    python -u projects/desi/tunebrick.py -b 374441 -s primage -P "pickles/PR-%(brick)06i-%%(stage)s.pickle"

    '''


    print 'kwargs:', kwargs.keys()

    rgb = get_rgb(coimgs, bands, mnmx=(0., 100.), arcsinh=1.)
    
    plt.clf()
    dimshow(rgb)
    ps.savefig()

    fn = ps.getnext()
    plt.imsave(fn, rgb, origin='lower')

    jpegfn = fn.replace('.png','.jpg')
    cmd = 'pngtopnm %s | pnmtojpeg -quality 80 > %s' % (fn, jpegfn)
    print cmd
    os.system(cmd)
    


def main():
    import optparse
    from astrometry.util.stages import *

    parser = optparse.OptionParser()
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[],
                      help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_option('-s', '--stage', dest='stage', default=[], action='append',
                      help="Run up to the given stage(s)")
    parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')
    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='pickles/tunebrick-%(brick)06i-%%(stage)s.pickle')

    parser.add_option('-b', '--brick', type=int, help='Brick ID to run: default %default',
                      default=377306)
    parser.add_option('-p', '--plots', dest='plots', action='store_true')
    #parser.add_option('--stamp', action='store_true')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 3600 0 3600")')
    parser.add_option('-W', type=int, default=3600, help='Target image width (default %default)')
    parser.add_option('-H', type=int, default=3600, help='Target image height (default %default)')

    parser.add_option('--bands', help='Bands to process; default "%default"', default='grz')

    parser.add_option('--plot-base', default='plot-%(brick)06i', #'tunebrick/coadd/plot-%(brick)06i',
                      help='Plot filenames; default %default')

    parser.add_option('--threads', type=int, help='Run multi-threaded')

    parser.add_option('--base-dir', dest='basedir', default='tunebrick',
                      help='Base output directory; default %default')

    parser.add_option('--mock-psf', dest='mock_psf', action='store_true',
                      help='Use fake PSF?')

    opt,args = parser.parse_args()
    Time.add_measurement(MemMeas)

    stagefunc = CallGlobal('stage_%s', globals())

    if len(opt.stage) == 0:
        opt.stage.append('writecat2')
    opt.force.extend(opt.stage)

    opt.picklepat = opt.picklepat % dict(brick=opt.brick)

    prereqs = {'tims': None,
               'cat': 'tims',
               'tune': 'cat',
               'writecat2': 'tune',

               'recoadd': 'tims',
               'rergb': 'recoadd',

               'primage': 'recoadd',
               }

    ps = PlotSequence(opt.plot_base % dict(brick=opt.brick))
    initargs = dict(ps=ps)
    initargs.update(W=opt.W, H=opt.H, brickid=opt.brick, target_extent=opt.zoom,
                    program_name = 'tunebrick.py', pipe=True,
                    bands=opt.bands,
                    mock_psf=opt.mock_psf)
    kwargs = {}
    kwargs.update(basedir=opt.basedir)

    if opt.threads and opt.threads > 1:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(opt.threads, init=runbrick_global_init, initargs=())
        runbrick.mp = mp
    else:
        runbrick_global_init()

    t0 = Time()
    for stage in opt.stage:
        runstage(stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
                 prereqs=prereqs, initial_args=initargs, **kwargs)
                 
               #tune(opt.brick, target_extent=opt.zoom)
    print 'Total:', Time()-t0

if __name__ == '__main__':
    main()
