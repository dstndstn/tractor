from __future__ import print_function
# class BlobTractor(Tractor):
#     def __init__(self, *args, **kwargs):
#         super(BlobTractor, self).__init__(*args, **kwargs)
#         if nocache:
#             self.disable_cache()
#             
#     def getLogLikelihood(self):
#         assert(not self.is_multiproc())
#         chisq = 0.
# 
#         sky=True
#         minsb=None
#         srcs=None
#         if srcs is None:
#             srcs = self.catalog
# 
#         for img in self.images:
#             mod = np.zeros(img.getModelShape(), self.modtype)
#             if sky:
#                 img.getSky().addTo(mod)
# 
#             # For flux outside blob:
#             # ASSUME the PSF conv galaxy is done via MoG;
#             psfmog = img.getPsf().getMixtureOfGaussians()
#             psfsum = np.sum(psfmog.amp)
#             # FIXME -- render PSF model within source's modelMask?
#             # modelrad = 8
#             # H,W = img.shape
#             # cx,cy = int(W/2), int(H/2)
#             # psfpatch = img.getPsf().getPointSourcePatch(
#             #     cx, cy, extent=[cx-modelrad, cx+modelrad+1, cy-modelrad, cy+modelrad+1])
#             # print 'PSF sum:', psfsum
#             # print 'PSF patch sum:', psfpatch.patch.sum()
#             #psfsum = psfpatch.patch.sum()
# 
#             for src in srcs:
#                 if src is None:
#                     continue
#                 patch = self.getModelPatch(img, src, minsb=minsb)
#                 if patch is None:
#                     continue
#                 patch.addTo(mod)
#                 # Also tally up the flux outside the blob...
#                 #
#                 # One issue here could be images that do not fully
#                 # cover the blob... the model gets penalized for
#                 # predicting flux outside the image, but the image is
#                 # smaller than the blob; the model is supposed to
#                 # predict flux outside that image.  Edges suck.  You
#                 # could imagine this pulling the flux estimate low.
#                 # Could not impose this penalty for sources close to
#                 # the image edges (or outside the image), but hard
#                 # thresholds suck too.  Apodize the penalty near the
#                 # image edges?
#                 #
#                 # Nastily, flux outside the blob is the
#                 # brightness->counts * PSF sum - patch sum, since real
#                 # PSFs need not sum to 1.0.
#                 #
#                 # ASSUME the source has a getBrightness() method
#                 counts = img.getPhotoCal().brightnessToCounts(src.getBrightness())
#                 # Flux outside the blob is...
#                 excess = counts * psfsum - patch.patch.sum()
#                 # APPROX - guess that the excess flux is spread into N 1-sigma pixels
#                 # print 'Source:', src
#                 # print 'Source flux:', counts/img.sig1, 'sigma'
#                 # print 'Excess flux: %.2f' % (excess/img.sig1), 'sigma (%.0f%%)' % (100.*excess/counts), ': counts %.2f'%counts, 'x PSF sum %.3f'% psfsum, '= %.2f' % (counts*psfsum), 'vs patch %.2f' % patch.patch.sum(), '= %.2f' % excess
# 
#                 # Here's an attempt at apodization...
#                 #ix,iy = img.getWcs().positionToPixel(src.getPosition())
#                 #ix = int(np.round(ix))
#                 #iy = int(np.round(iy))
#                 #H,W = img.shape
#                 #edgedist = max(0, min(ix, iy, H-1-iy, W-1-ix))
#                 #print 'edge distance:', edgedist, '-- x,y', (ix,iy), 'size', W,'x',H
#                 # soften = 1.
#                 # if edgedist < 5:
#                 #     #soften = np.exp(-0.5 * (5. - edgedist)**2 / 2.**2)
#                 #     soften = 0.2 * edgedist
#                 #     print 'soften:', soften
#                 # chisq += soften * (np.abs(excess) / img.sig1)
#                 
#                 chisq += (np.abs(excess) / img.sig1)
# 
#             chisq += (((img.getImage() - mod) * img.getInvError())**2).sum()
# 
#         return -0.5 * chisq
    
def check_touching(decals, targetwcs, bands, brick, pixscale, ps):
    T2 = decals.get_ccds()

    T3 = T2[ccds_touching_wcs(targetwcs, T2, polygons=False)]
    T4 = T2[ccds_touching_wcs(targetwcs, T2)]
    print(len(T3), 'on RA,Dec box')
    print(len(T4), 'polygon')
    ccmap = dict(r='r', g='g', z='m')
    for band in bands:

        plt.clf()

        TT2 = T3[T3.filter == band]
        print(len(TT2), 'in', band, 'band')
        plt.plot(TT2.ra, TT2.dec, 'o', color=ccmap[band], alpha=0.5)

        for t in TT2:
            im = DecamImage(decals, t)

            run_calibs(im, brick.ra, brick.dec, pixscale, morph=False, se2=False,
                       psfex=False)

            wcs = im.read_wcs()
            r,d = wcs.pixelxy2radec([1,1,t.width,t.width,1], [1,t.height,t.height,1,1])
            plt.plot(r, d, '-', color=ccmap[band], alpha=0.3, lw=2)

        TT2 = T4[T4.filter == band]
        print(len(TT2), 'in', band, 'band; polygon')
        plt.plot(TT2.ra, TT2.dec, 'x', color=ccmap[band], alpha=0.5, ms=15)

        for t in TT2:
            im = DecamImage(decals, t)
            wcs = im.read_wcs()
            r,d = wcs.pixelxy2radec([1,1,t.width,t.width,1], [1,t.height,t.height,1,1])
            plt.plot(r, d, '-', color=ccmap[band], lw=1.5)

        TT2.about()

        plt.plot(brick.ra, brick.dec, 'k.')
        plt.plot(targetrd[:,0], targetrd[:,1], 'k-')
        plt.xlabel('RA')
        plt.ylabel('Dec')
        ps.savefig()



def check_photometric_calib(ims, cat, ps):
    # Check photometric calibrations
    lastband = None

    for im in ims:
        band = im.band
        cat = fits_table(im.morphfn, hdu=2, columns=[
            'mag_psf','x_image', 'y_image', 'mag_disk', 'mag_spheroid', 'flags',
            'flux_psf' ])
        print('Read', len(cat), 'from', im.morphfn)
        if len(cat) == 0:
            continue
        cat.cut(cat.flags == 0)
        print('  Cut to', len(cat), 'with no flags set')
        if len(cat) == 0:
            continue
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)

        sdss = fits_table(im.sdssfn)


        I = np.flatnonzero(ZP.expnum == im.expnum)
        if len(I) > 1:
            I = np.flatnonzero((ZP.expnum == im.expnum) * (ZP.extname == im.extname))
        assert(len(I) == 1)
        I = I[0]
        magzp = ZP.zpt[I]
        print('magzp', magzp)
        exptime = ZP.exptime[I]
        magzp += 2.5 * np.log10(exptime)
        print('magzp', magzp)

        primhdr = im.read_image_primary_header()
        magzp0  = primhdr['MAGZERO']
        print('header magzp:', magzp0)

        I,J,d = match_radec(cat.ra, cat.dec, sdss.ra, sdss.dec, 1./3600.)

        flux = sdss.get('%s_psfflux' % band)
        mag = NanoMaggies.nanomaggiesToMag(flux)

        # plt.clf()
        # plt.plot(mag[J], cat.mag_psf[I] - mag[J], 'b.')
        # plt.xlabel('SDSS %s psf mag' % band)
        # plt.ylabel('SDSS - DECam mag')
        # plt.title(im.name)
        # plt.axhline(0, color='k', alpha=0.5)
        # plt.ylim(-2,2)
        # plt.xlim(15, 23)
        # ps.savefig()

        if band != lastband:
            if lastband is not None:
                ps.savefig()
            off = 0
            plt.clf()

        if off >= 8:
            continue

        plt.subplot(2,4, off+1)
        mag2 = -2.5 * np.log10(cat.flux_psf)
        p = plt.plot(mag[J], mag[J] - mag2[I], 'b.')
        plt.xlabel('SDSS %s psf mag' % band)
        if off in [0,4]:
            plt.ylabel('SDSS - DECam instrumental mag')
        plt.title(im.name)

        med = np.median(mag[J] - mag2[I])
        plt.axhline(med, color='k', alpha=0.25)

        plt.ylim(29,32)
        plt.xlim(15, 22)
        plt.axhline(magzp, color='r', alpha=0.5)
        plt.axhline(magzp0, color='b', alpha=0.5)

        off += 1
        lastband = band
    ps.savefig()

def get_se_sources(ims, catband, targetwcs, W, H):
    # FIXME -- we're only reading 'catband'-band catalogs, and all the fluxes
    # are initialized at that band's flux... should really read all bands!
        
    # Select SE catalogs to read
    catims = [im for im in ims if im.band == catband]
    print('Reference catalog files:', catims)
    # ... and read 'em
    cats = []
    extra_cols = []
    for im in catims:
        cat = fits_table(
            im.morphfn, hdu=2,
            columns=[x.upper() for x in
                     ['x_image', 'y_image', 'flags',
                      'chi2_psf', 'chi2_model', 'mag_psf', 'mag_disk',
                      'mag_spheroid', 'disk_scale_world', 'disk_aspect_world',
                      'disk_theta_world', 'spheroid_reff_world',
                      'spheroid_aspect_world', 'spheroid_theta_world',
                      'alphamodel_j2000', 'deltamodel_j2000'] + extra_cols])
        print('Read', len(cat), 'from', im.morphfn)
        cat.cut(cat.flags == 0)
        print('  Cut to', len(cat), 'with no flags set')
        wcs = Sip(im.wcsfn)
        cat.ra,cat.dec = wcs.pixelxy2radec(cat.x_image, cat.y_image)
        cats.append(cat)
        
    # Plot all catalog sources and ROI
    # plt.clf()
    # for cat in cats:
    #     plt.plot(cat.ra, cat.dec, 'o', mec='none', mfc='b', alpha=0.5)
    # plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    # ps.savefig()
    # Cut catalogs to ROI
    for cat in cats:
        ok,x,y = targetwcs.radec2pixelxy(cat.ra, cat.dec)
        cat.cut((x > 0.5) * (x < (W+0.5)) * (y > 0.5) * (y < (H+0.5)))

    # Merge catalogs by keeping sources > 0.5" away from previous ones
    merged = cats[0]
    for cat in cats[1:]:
        if len(merged) == 0:
            merged = cat
            continue
        if len(cat) == 0:
            continue
        I,J,d = match_radec(merged.ra, merged.dec, cat.ra, cat.dec, 0.5/3600.)
        keep = np.ones(len(cat), bool)
        keep[J] = False
        if sum(keep):
            merged = merge_tables([merged, cat[keep]])
    
    # plt.clf()
    # plt.plot(merged.ra, merged.dec, 'o', mec='none', mfc='b', alpha=0.5)
    # plt.plot(targetrd[:,0], targetrd[:,1], 'r-')
    # ps.savefig()

    del cats
    # Create Tractor sources
    cat,isrcs = get_se_modelfit_cat(merged, maglim=90, bands=bands)
    print('Tractor sources:', cat)
    T = merged[isrcs]
    return cat, T

# Check out the PsfEx models
def stage101(T=None, sedsn=None, coimgs=None, con=None, coimas=None,
             detmaps=None, detivs=None,
             rgbim=None,
             nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
             tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None, tims=None,
             **kwargs):
    # sort sources by their sedsn values.
    fluxes = sedsn[T.ity, T.itx]

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]

    for srci in np.argsort(-fluxes)[:20]:
        cat.freezeAllParams()
        cat.thawParam(srci)
                    
        print('Fitting:')
        tractor.printThawedParams()
        for itim,tim in enumerate(tims):
            ox0,oy0 = orig_wcsxy0[itim]
            x,y = tim.wcs.positionToPixel(cat[srci].getPosition())
            psfimg = tim.psfex.instantiateAt(ox0+x, oy0+y, nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg)
            tim.psf = subpsf

        for step in range(10):
            dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            print('dlnp:', dlnp)
            if dlnp < 0.1:
                break
        
        chis1 = tractor.getChiImages()
        mods1 = tractor.getModelImages()


        for itim,tim in enumerate(tims):
            ox0,oy0 = orig_wcsxy0[itim]
            x,y = tim.wcs.positionToPixel(cat[srci].getPosition())
            psfimg = tim.psfex.instantiateAt(ox0+x, oy0+y, nativeScale=True)
            subpsf = PixelizedPSF(psfimg)
            tim.psf = subpsf
        for step in range(10):
            dlnp,X,alpha = tractor.optimize(priors=False, shared_params=False)
            print('dlnp:', dlnp)
            if dlnp < 0.1:
                break
        
        chis2 = tractor.getChiImages()
        mods2 = tractor.getModelImages()

        
        subchis = []
        submods = []
        subchis2 = []
        submods2 = []
        subimgs = []
        for i,(chi,mod) in enumerate(zip(chis1, mods1)):
            x,y = tims[i].wcs.positionToPixel(cat[srci].getPosition())
            x = int(x)
            y = int(y)
            S = 15
            th,tw = tims[i].shape
            x0 = max(x-S, 0)
            y0 = max(y-S, 0)
            x1 = min(x+S, tw)
            y1 = min(y+S, th)
            subchis.append(chi[y0:y1, x0:x1])
            submods.append(mod[y0:y1, x0:x1])
            subimgs.append(tims[i].getImage()[y0:y1, x0:x1])
            subchis2.append(chis2[i][y0:y1, x0:x1])
            submods2.append(mods2[i][y0:y1, x0:x1])

        mxchi = max([np.abs(chi).max() for chi in subchis])

        # n = len(subchis)
        # cols = int(np.ceil(np.sqrt(n)))
        # rows = int(np.ceil(float(n) / cols))
        # plt.clf()
        # for i,chi in enumerate(subchis):
        #     plt.subplot(rows, cols, i+1)
        #     dimshow(-chi, vmin=-mxchi, vmax=mxchi, cmap='RdBu')
        #     plt.colorbar()
        # ps.savefig()

        cols = len(subchis)
        rows = 3
        rows = 5
        plt.clf()
        ta = dict(fontsize=8)
        for i,(chi,mod,img) in enumerate(zip(subchis,submods,subimgs)):
            mx = img.max()
            def nl(x):
                return np.log10(np.maximum(tim.sig1, x + 5.*tim.sig1))

            plt.subplot(rows, cols, i+1)
            dimshow(nl(img), vmin=nl(0), vmax=nl(mx))
            plt.xticks([]); plt.yticks([])
            plt.title(tims[i].name, **ta)

            plt.subplot(rows, cols, i+1+cols)
            dimshow(nl(mod), vmin=nl(0), vmax=nl(mx))
            plt.xticks([]); plt.yticks([])
            if i == 0:
                plt.title('MoG PSF', **ta)

            plt.subplot(rows, cols, i+1+cols*2)
            mxchi = 5.
            dimshow(-chi, vmin=-mxchi, vmax=mxchi, cmap='RdBu')
            plt.xticks([]); plt.yticks([])
            #plt.colorbar()
            if i == 0:
                plt.title('MoG chi', **ta)

            # pix
            plt.subplot(rows, cols, i+1+cols*3)
            dimshow(nl(submods2[i]), vmin=nl(0), vmax=nl(mx))
            plt.xticks([]); plt.yticks([])
            if i == 0:
                plt.title('Pixelized PSF', **ta)

            plt.subplot(rows, cols, i+1+cols*4)
            mxchi = 5.
            dimshow(-subchis2[i], vmin=-mxchi, vmax=mxchi, cmap='RdBu')
            plt.xticks([]); plt.yticks([])
            if i == 0:
                plt.title('Pixelized chi', **ta)

        rd = cat[srci].getPosition()
        plt.suptitle('Source at RA,Dec = (%.4f, %.4f)' % (rd.ra, rd.dec))
            
        ps.savefig()


class BrightPointSource(PointSource):
    def _getPsf(self, img):
        return img.brightPsf
    def getSourceType(self):
        return 'BrightPointSource'

def stage2(T=None, sedsn=None, coimgs=None, cons=None,
           detmaps=None, detivs=None,
           nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
           cat=None, targetrd=None, pixscale=None, targetwcs=None,
           W=None,H=None,
           bands=None, ps=None,
           plots=False, tims=None, tractor=None,
           **kwargs):

    # For bright sources, use more MoG components, or use pixelized PSF model?
    fluxes = []
    for src in cat:
        br = src.getBrightness()
        fluxes.append([br.getFlux(b) for b in bands])
    fluxes = np.array(fluxes)

    for i,b in enumerate(bands):
        ii = np.argsort(-fluxes[:,i])
        print()
        print('Brightest in band', b)
        for j in ii[:10]:
            print(j, cat[j].getBrightness())


    # HACK -- define "bright" limits
    bright = dict(g = 20.5, r = 20, z = 19.5)

    ibright = []
    for band in bands:
        brightmag = bright[band]
        for i,src in enumerate(cat):
            br = src.getBrightness()
            if br.getMag(band) < brightmag:
                ibright.append(i)
    ibright = np.unique(ibright)

    print('Bright sources:', ibright)

    bcat = []
    for i,src in enumerate(cat):
        # if i in ibright:
        #     if isinstance(src, PointSource):
        #         bcat.append(BrightPointSource(src.pos, src.brightness))
        #     else:
        #         ### FIXME -- model selection??
        #         print 'Trying to replace bright source', src, 'with point source'
        #         bcat.append(BrightPointSource(src.getPosition(), src.getBrightness()))

        if i in ibright and isinstance(src, PointSource):
            bcat.append(BrightPointSource(src.pos, src.brightness))
        else:
            bcat.append(src)
    bcat = Catalog(*bcat)

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
                
    for iblob,(bslc,Isrcs) in enumerate(zip(blobslices, blobsrcs)):
        if not len(set(ibright).intersection(set(Isrcs))):
            continue
        print('Re-fitting blob', iblob, 'with', len(Isrcs), 'sources')

        bcat.freezeAllParams()
        print('Fitting:')
        for i in Isrcs:
            bcat.thawParams(i)
            print(bcat[i])
            
        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])

        ###
        # FIXME -- We create sub-image for each blob here.
        # What wo don't do, though, is mask out the invvar pixels
        # that are within the blob bounding-box but not within the
        # blob itself.  Does this matter?
        ###

        alphas = [0.1, 0.3, 1.0]
        
        subtims = []
        for itim,tim in enumerate(tims):
            h,w = tim.shape
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            if sx1 < 0 or sy1 < 0 or sx1 > w or sy1 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            #print 'image subregion', sx0,sx1,sy0,sy1

            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subiv  = tim.getInvvar()[subslc]
            subwcs = tim.getWcs().copy()
            ox0,oy0 = orig_wcsxy0[itim]
            subwcs.setX0Y0(ox0 + sx0, oy0 + sy0)

            # FIXME --
            #subpsf = tim.psfex.mogAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.)
            #subpsf = tim.getPsf()

            psfimg = tim.psfex.instantiateAt(ox0+(sx0+sx1)/2., oy0+(sy0+sy1)/2.,
                                             nativeScale=True)
            subpsf = GaussianMixturePSF.fromStamp(psfimg)

            #subtim = BrightPsfImage(data=subimg, invvar=subiv, wcs=subwcs,
            subtim = Image(data=subimg, invvar=subiv, wcs=subwcs,
                           psf=subpsf, photocal=tim.getPhotoCal(),
                           sky=tim.getSky(), name=tim.name)
            subtim.extent = (sx0, sx1, sy0, sy1)
            subtim.band = tim.band

            (Yo,Xo,Yi,Xi) = tim.resamp
            I = np.flatnonzero((Yi >= sy0) * (Yi < sy1) * (Xi >= sx0) * (Xi < sx1) *
                               (Yo >= by0) * (Yo < by1) * (Xo >= bx0) * (Xo < bx1))
            Yo = Yo[I] - by0
            Xo = Xo[I] - bx0
            Yi = Yi[I] - sy0
            Xi = Xi[I] - sx0
            subtim.resamp = (Yo, Xo, Yi, Xi)
            subtim.sig1 = tim.sig1

            subtim.brightPsf = PixelizedPsfEx(tim.psfex, ox0 + sx0, oy0 + sy0)
            #subtim.brightPsf = PixelizedPSF(psfimg)
            #subtim.brightPsf = GaussianMixturePSF.fromStamp(psfimg, N=5)

            subtims.append(subtim)

        subtr = Tractor(subtims, bcat)
        subtr.freezeParam('images')
        print('Optimizing:', subtr)
        subtr.printThawedParams()

        if plots:
            otractor = Tractor(subtims, cat)
            modx = otractor.getModelImages()

            # before-n-after plots
            mod0 = subtr.getModelImages()
        print('Sub-image initial lnlikelihood:', subtr.getLogLikelihood())

        for i in Isrcs:
            print(bcat[i])

        for step in range(10):
            dlnp,X,alpha = subtr.optimize(priors=False, shared_params=False,
                                          alphas=alphas)
            print('dlnp:', dlnp)
            if dlnp < 0.1:
                break

        if plots:
            mod1 = subtr.getModelImages()
        print('Sub-image first fit lnlikelihood:', subtr.getLogLikelihood())

        for i in Isrcs:
            print(bcat[i])

        # Forced-photometer bands individually
        for band in bands:
            bcat.freezeAllRecursive()
            for i in Isrcs:
                bcat.thawParam(i)
                bcat[i].thawPathsTo(band)
            bandtims = []
            for tim in subtims:
                if tim.band == band:
                    bandtims.append(tim)
            print()
            print('Fitting', band, 'band:')
            btractor = Tractor(bandtims, bcat)
            btractor.freezeParam('images')
            btractor.printThawedParams()
            B = 8
            X = btractor.optimize_forced_photometry(shared_params=False, use_ceres=True,
                                                    BW=B, BH=B, wantims=False)
        bcat.thawAllRecursive()
        print('Sub-image forced-phot lnlikelihood:', subtr.getLogLikelihood())
        for i in Isrcs:
            print(bcat[i])

        if plots:
            mod2 = subtr.getModelImages()

        if plots:
            mods = [modx, mod0, mod1, mod2]
            _plot_mods(subtims, mods, ['' for m in mods], bands, coimgs, cons, bslc, blobw, blobh, ps)

    rtn = dict()
    for k in ['tractor','tims', 'bcat', 'ps']:
        rtn[k] = locals()[k]
    return rtn


class PixelizedPsfEx(object):
    def __init__(self, psfex, x0, y0):
        self.psfex = psfex
        self.x0 = x0
        self.y0 = y0
        
    def getPointSourcePatch(self, px, py, minval=0., extent=None, radius=None):
        pix = self.psfex.instantiateAt(self.x0 + px, self.y0 + py, nativeScale=True)
        return PixelizedPSF(pix).getPointSourcePatch(px, py, radius=radius, extent=extent)

def stage103(T=None, sedsn=None, coimgs=None, con=None, coimas=None,
             detmaps=None, detivs=None,
             rgbim=None,
             nblobs=None,blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
             cat=None, targetrd=None, pixscale=None, targetwcs=None,
             W=None,H=None,
             bands=None, ps=None,
             plots=False, tims=None, tractor=None, bcat=None,
             **kwargs):

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    for itim,tim in enumerate(tims):
        ox0,oy0 = orig_wcsxy0[itim]
        # # HACK -- instantiate pixelized PSF at center of tim
        # r,d = targetwcs.pixelxy2radec(W/2., H/2.)
        # ok,cx,cy = tim.subwcs.radec2pixelxy(r, d)
        # psfimg = tim.psfex.instantiateAt(ox0+cx, oy0+cy, nativeScale=True)
        # tim.brightPsf = PixelizedPsfEx(psfimg)

        tim.brightPsf = PixelizedPsfEx(tim.psfex, ox0, oy0)

    cat = tractor.catalog = bcat

    print('Sources:')
    for i,src in enumerate(cat):
        print('  ', i, src)

    stage102(tractor=tractor, tims=tims, H=H, W=W, bands=bands,
             rgbim=rgbim, cat=cat, ps=ps, coimgs=coimgs, con=con,
             targetwcs=targetwcs)


