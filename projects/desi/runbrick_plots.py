import pylab as plt
from common import *

def _psf_check_plots(tims):
    # HACK -- check PSF models
    plt.figure(num=2, figsize=(7,4.08))
    for im,tim in zip(ims,tims):
        print
        print 'Image', tim.name

        plt.subplots_adjust(left=0, right=1, bottom=0, top=0.95,
                            hspace=0, wspace=0)
        W,H = 2048,4096
        psfex = PsfEx(im.psffn, W, H)

        psfim0 = psfim = psfex.instantiateAt(W/2, H/2)
        # trim
        psfim = psfim[10:-10, 10:-10]

        tfit = Time()
        psffit2 = GaussianMixtureEllipsePSF.fromStamp(psfim, N=2)
        print 'Fitting PSF mog:', psfim.shape, Time()-tfit

        psfim = psfim0[5:-5, 5:-5]
        tfit = Time()
        psffit2 = GaussianMixtureEllipsePSF.fromStamp(psfim, N=2)
        print 'Fitting PSF mog:', psfim.shape, Time()-tfit

        ph,pw = psfim.shape
        psffit = GaussianMixtureEllipsePSF.fromStamp(psfim, N=3)

        #mx = 0.03
        mx = psfim.max()

        mod3 = np.zeros_like(psfim)
        p = psffit.getPointSourcePatch(pw/2, ph/2, radius=pw/2)
        p.addTo(mod3)
        mod2 = np.zeros_like(psfim)
        p = psffit2.getPointSourcePatch(pw/2, ph/2, radius=pw/2)
        p.addTo(mod2)

        plt.clf()
        plt.subplot(2,3,1)
        dimshow(psfim, vmin=0, vmax=mx, ticks=False)
        plt.subplot(2,3,2)
        dimshow(mod3, vmin=0, vmax=mx, ticks=False)
        plt.subplot(2,3,3)
        dimshow(mod2, vmin=0, vmax=mx, ticks=False)
        plt.subplot(2,3,5)
        dimshow(psfim-mod3, vmin=-mx/2, vmax=mx/2, ticks=False)
        plt.subplot(2,3,6)
        dimshow(psfim-mod2, vmin=-mx/2, vmax=mx/2, ticks=False)
        ps.savefig()
        #continue

        for round in [1,2,3,4,5]:
            plt.clf()
            k = 1
            #rows,cols = 10,5
            rows,cols = 7,4
            for iy,y in enumerate(np.linspace(0, H, rows).astype(int)):
                for ix,x in enumerate(np.linspace(0, W, cols).astype(int)):
                    psfimg = psfex.instantiateAt(x, y)
                    # trim
                    psfimg = psfimg[5:-5, 5:-5]
                    print 'psfimg', psfimg.shape
                    ph,pw = psfimg.shape
                    psfimg2 = tim.psfex.getPointSourcePatch(x, y, radius=pw/2)
                    mod = np.zeros_like(psfimg)
                    h,w = mod.shape
                    #psfimg2.x0 -= x
                    #psfimg2.x0 += w/2
                    #psfimg2.y0 -= y
                    #psfimg2.y0 += h/2
                    psfimg2.x0 = 0
                    psfimg2.y0 = 0
                    print 'psfimg2:', (psfimg2.x0,psfimg2.y0)
                    psfimg2.addTo(mod)
                    print 'psfimg:', psfimg.min(), psfimg.max(), psfimg.sum()
                    print 'psfimg2:', psfimg2.patch.min(), psfimg2.patch.max(), psfimg2.patch.sum()
                    print 'mod:', mod.min(), mod.max(), mod.sum()

                    #plt.subplot(rows, cols, k)
                    plt.subplot(cols, rows, k)
                    k += 1
                    kwa = dict(vmin=0, vmax=mx, ticks=False)
                    if round == 1:
                        dimshow(psfimg, **kwa)
                        plt.suptitle('PsfEx')
                    elif round == 2:
                        dimshow(mod, **kwa)
                        plt.suptitle('varying MoG')
                    elif round == 3:
                        dimshow(psfimg - mod, vmin=-mx/2, vmax=mx/2, ticks=False)
                        plt.suptitle('PsfEx - varying MoG')
                    elif round == 4:
                        dimshow(psfimg - mod3, vmin=-mx/2, vmax=mx/2, ticks=False)
                        plt.suptitle('PsfEx - const MoG(3)')
                    elif round == 5:
                        dimshow(psfimg - mod2, vmin=-mx/2, vmax=mx/2, ticks=False)
                        plt.suptitle('PsfEx - const MoG(2)')
            ps.savefig()


def _debug_plots(srctractor, ps):
    thislnp0 = srctractor.getLogProb()
    p0 = np.array(srctractor.getParams())
    print 'logprob:', p0, '=', thislnp0

    print 'p0 type:', p0.dtype
    px = p0 + np.zeros_like(p0)
    srctractor.setParams(px)
    lnpx = srctractor.getLogProb()
    assert(lnpx == thislnp0)
    print 'logprob:', px, '=', lnpx

    scales = srctractor.getParameterScales()
    print 'Parameter scales:', scales
    print 'Parameters:'
    srctractor.printThawedParams()

    # getParameterScales better not have changed the params!!
    assert(np.all(p0 == np.array(srctractor.getParams())))
    assert(srctractor.getLogProb() == thislnp0)

    pfinal = srctractor.getParams()
    pnames = srctractor.getParamNames()

    plt.figure(3, figsize=(8,6))

    plt.clf()
    for i in range(len(scales)):
        plt.plot([(p[i] - pfinal[i])*scales[i] for lnp,p in params],
                 [lnp for lnp,p in params], '-', label=pnames[i])
    plt.ylabel('lnp')
    plt.legend()
    plt.title('scaled')
    ps.savefig()

    for i in range(len(scales)):
        plt.clf()
        #plt.subplot(2,1,1)
        plt.plot([p[i] for lnp,p in params], '-')
        plt.xlabel('step')
        plt.title(pnames[i])
        ps.savefig()

        plt.clf()
        plt.plot([p[i] for lnp,p in params],
                 [lnp for lnp,p in params], 'b.-')

        # We also want to know about d(lnp)/d(param)
        # and d(lnp)/d(X)
        step = 1.1
        steps = 1.1 ** np.arange(-20, 21)
        s2 = np.linspace(0, steps[0], 10)[1:-1]
        steps = reduce(np.append, [-steps[::-1], -s2[::-1], 0, s2, steps])
        print 'Steps:', steps

        plt.plot(p0[i], thislnp0, 'bx', ms=20)

        print 'Stepping in param', pnames[i], '...'
        pp = p0.copy()
        lnps,parms = [],[]
        for s in steps:
            parm = p0[i] + s / scales[i]
            pp[i] = parm
            srctractor.setParams(pp)
            lnp = srctractor.getLogProb()
            parms.append(parm)
            lnps.append(lnp)
            print 'logprob:', pp, '=', lnp
            
        plt.plot(parms, lnps, 'k.-')
        j = np.argmin(np.abs(steps - 1.))
        plt.plot(parms[j], lnps[j], 'ko')

        print 'Stepping in X...'
        lnps,parms = [],[]
        for s in steps:
            pp = p0 + s * X
            srctractor.setParams(pp)
            lnp = srctractor.getLogProb()
            parms.append(pp[i])
            lnps.append(lnp)
            print 'logprob:', pp, '=', lnp


        ##
        s3 = s2[:2]
        ministeps = reduce(np.append, [-s3[::-1], 0, s3])
        print 'mini steps:', ministeps
        for s in ministeps:
            pp = p0 + s * X
            srctractor.setParams(pp)
            lnp = srctractor.getLogProb()
            print 'logprob:', pp, '=', lnp

        rows = len(ministeps)
        cols = len(srctractor.images)

        plt.figure(4, figsize=(8,6))
        plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.01,
                            right=0.99, bottom=0.01, top=0.99)
        plt.clf()
        k = 1
        mods = []
        for s in ministeps:
            pp = p0 + s * X
            srctractor.setParams(pp)
            print 'ministep', s
            print 'log prior', srctractor.getLogPrior()
            print 'log likelihood', srctractor.getLogLikelihood()
            mods.append(srctractor.getModelImages())
            chis = srctractor.getChiImages()
            # for chi in chis:
            #     plt.subplot(rows, cols, k)
            #     k += 1
            #     dimshow(chi, ticks=False, vmin=-10, vmax=10, cmap='jet')
            print 'chisqs:', [(chi**2).sum() for chi in chis]
            print 'sum:', sum([(chi**2).sum() for chi in chis])

        mod0 = mods[len(ministeps)/2]
        for modlist in mods:
            for mi,mod in enumerate(modlist):
                plt.subplot(rows, cols, k)
                k += 1
                m0 = mod0[mi]
                rng = m0.max() - m0.min()
                dimshow(mod - mod0[mi], vmin=-0.01*rng, vmax=0.01*rng,
                        ticks=False, cmap='gray')
        ps.savefig()
        plt.figure(3)
        
        plt.plot(parms, lnps, 'r.-')

        print 'Stepping in X by alphas...'
        lnps = []
        for cc,ss in [('m',0.1), ('m',0.3), ('r',1)]:
            pp = p0 + ss*X
            srctractor.setParams(pp)
            lnp = srctractor.getLogProb()
            print 'logprob:', pp, '=', lnp

            plt.plot(p0[i] + ss * X[i], lnp, 'o', color=cc)
            lnps.append(lnp)

        px = p0[i] + X[i]
        pmid = (px + p0[i]) / 2.
        dp = np.abs((px - pmid) * 2.)
        hi,lo = max(max(lnps), thislnp0), min(min(lnps), thislnp0)
        lnpmid = (hi + lo) / 2.
        dlnp = np.abs((hi - lo) * 2.)

        plt.ylabel('lnp')
        plt.title(pnames[i])
        ps.savefig()

        plt.axis([pmid - dp, pmid + dp, lnpmid-dlnp, lnpmid+dlnp])
        ps.savefig()

    srctractor.setParams(p0)
    ### DEBUG

def _plot_derivs(subtims, newsrc, srctractor, ps):
    plt.clf()
    rows = len(subtims)
    cols = 1 + newsrc.numberOfParams()
    for it,tim in enumerate(subtims):
        derivs = srctractor._getSourceDerivatives(newsrc, tim)
        c0 = 1 + cols*it
        mod = srctractor.getModelPatchNoCache(tim, src)
        if mod is not None and mod.patch is not None:
            plt.subplot(rows, cols, c0)
            dimshow(mod.patch, extent=mod.getExtent())
        c0 += 1
        for ip,deriv in enumerate(derivs):
            if deriv is None:
                continue
            plt.subplot(rows, cols, c0+ip)
            mx = np.max(np.abs(deriv.patch))
            dimshow(deriv.patch, extent=deriv.getExtent(), vmin=-mx, vmax=mx)
    plt.title('Derivatives for ' + name)
    ps.savefig()
    plt.clf()
    modimgs = srctractor.getModelImages()
    comods,nil = compute_coadds(subtims, bands, subtarget, images=modimgs)
    dimshow(get_rgb(comods, bands))
    plt.title('Initial ' + name)
    ps.savefig()
            
def _plot_mods(tims, mods, titles, bands, coimgs, cons, bslc, blobw, blobh, ps,
               chi_plots=True, rgb_plots=False, main_plot=True,
               rgb_format='%s'):
    import numpy as np

    subims = [[] for m in mods]
    chis = dict([(b,[]) for b in bands])
    
    make_coimgs = (coimgs is None)
    if make_coimgs:
        print '_plot_mods: blob shape', (blobh, blobw)
        coimgs = [np.zeros((blobh,blobw)) for b in bands]
        cons   = [np.zeros((blobh,blobw)) for b in bands]

    for iband,band in enumerate(bands):
        comods = [np.zeros((blobh,blobw)) for m in mods]
        cochis = [np.zeros((blobh,blobw)) for m in mods]
        comodn = np.zeros((blobh,blobw))
        mn,mx = 0,0
        sig1 = 1.
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp

            rechi = np.zeros((blobh,blobw))
            chilist = []
            comodn[Yo,Xo] += 1
            for imod,mod in enumerate(mods):
                chi = ((tim.getImage()[Yi,Xi] - mod[itim][Yi,Xi]) *
                       tim.getInvError()[Yi,Xi])
                rechi[Yo,Xo] = chi
                chilist.append((rechi.copy(), itim))
                cochis[imod][Yo,Xo] += chi
                comods[imod][Yo,Xo] += mod[itim][Yi,Xi]
            chis[band].append(chilist)
            # we'll use 'sig1' of the last tim in the list below...
            mn,mx = -10.*tim.sig1, 30.*tim.sig1
            sig1 = tim.sig1
            if make_coimgs:
                nn = (tim.getInvError()[Yi,Xi] > 0)
                coimgs[iband][Yo,Xo] += tim.getImage()[Yi,Xi] * nn
                cons  [iband][Yo,Xo] += nn
                
        if make_coimgs:
            coimgs[iband] /= np.maximum(cons[iband], 1)
            coimg  = coimgs[iband]
            coimgn = cons  [iband]
        else:
            coimg = coimgs[iband][bslc]
            coimgn = cons[iband][bslc]
            
        for comod in comods:
            comod /= np.maximum(comodn, 1)
        ima = dict(vmin=mn, vmax=mx, ticks=False)
        resida = dict(vmin=-5.*sig1, vmax=5.*sig1, ticks=False)
        for subim,comod,cochi in zip(subims, comods, cochis):
            subim.append((coimg, coimgn, comod, ima, cochi, resida))

    # Plot per-band image, model, and chi coadds, and RGB images
    rgba = dict(ticks=False)
    rgbs = []
    rgbnames = []
    plt.figure(1)
    for i,subim in enumerate(subims):
        plt.clf()
        rows,cols = 3,5
        imgs = []
        themods = []
        resids = []
        for j,(img,imgn,mod,ima,chi,resida) in enumerate(subim):
            imgs.append(img)
            themods.append(mod)
            resid = img - mod
            resid[imgn == 0] = np.nan
            resids.append(resid)

            if main_plot:
                plt.subplot(rows,cols,1 + j + 0)
                dimshow(img, **ima)
                plt.subplot(rows,cols,1 + j + cols)
                dimshow(mod, **ima)
                plt.subplot(rows,cols,1 + j + cols*2)
                # dimshow(-chi, **imchi)
                # dimshow(imgn, vmin=0, vmax=3)
                dimshow(resid, nancolor='r', **resida)
        rgb = get_rgb(imgs, bands)
        if i == 0:
            rgbs.append(rgb)
            rgbnames.append(rgb_format % 'Image')
        if main_plot:
            plt.subplot(rows,cols, 4)
            dimshow(rgb, **rgba)
        rgb = get_rgb(themods, bands)
        rgbs.append(rgb)
        rgbnames.append(rgb_format % titles[i])
        if main_plot:
            plt.subplot(rows,cols, cols+4)
            dimshow(rgb, **rgba)
            plt.subplot(rows,cols, cols*2+4)
            dimshow(get_rgb(resids, bands, mnmx=(-10,10)), **rgba)

            mnmx = -5,300
            kwa = dict(mnmx=mnmx, arcsinh=1)
            plt.subplot(rows,cols, 5)
            dimshow(get_rgb(imgs, bands, **kwa), **rgba)
            plt.subplot(rows,cols, cols+5)
            dimshow(get_rgb(themods, bands, **kwa), **rgba)
            plt.subplot(rows,cols, cols*2+5)
            mnmx = -100,100
            kwa = dict(mnmx=mnmx, arcsinh=1)
            dimshow(get_rgb(resids, bands, **kwa), **rgba)
            plt.suptitle(titles[i])
            ps.savefig()

    if rgb_plots:
        # RGB image and model
        plt.figure(2)
        for rgb,tt in zip(rgbs, rgbnames):
            plt.clf()
            dimshow(rgb, **rgba)
            plt.title(tt)
            ps.savefig()

    if not chi_plots:
        return

    imchi = dict(cmap='RdBu', vmin=-5, vmax=5)

    plt.figure(1)
    # Plot per-image chis: in a grid with band along the rows and images along the cols
    cols = max(len(v) for v in chis.values())
    rows = len(bands)
    for imod in range(len(mods)):
        plt.clf()
        for row,band in enumerate(bands):
            sp0 = 1 + cols*row
            # chis[band] = [ (one for each tim:) [ (one for each mod:) (chi,itim), (chi,itim) ], ...]
            for col,chilist in enumerate(chis[band]):
                chi,itim = chilist[imod]
                plt.subplot(rows, cols, sp0 + col)
                dimshow(-chi, **imchi)
                plt.xticks([]); plt.yticks([])
                plt.title(tims[itim].name)
        #plt.suptitle(titles[imod])
        ps.savefig()




'''
PSF plots
'''
def stage_psfplots(
    T=None, sedsn=None, coimgs=None, cons=None,
    detmaps=None, detivs=None,
    blobsrcs=None,blobflux=None,blobslices=None, blobs=None,
    tractor=None, cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None, brickid=None,
    bands=None, ps=None, tims=None,
    plots=False,
    **kwargs):

    tim = tims[0]
    tim.psfex.fitSavedData(*tim.psfex.splinedata)
    spl = tim.psfex.splines[0]
    print 'Spline:', spl
    knots = spl.get_knots()
    print 'knots:', knots
    tx,ty = knots
    k = 3
    print 'interior knots x:', tx[k+1:-k-1]
    print 'additional knots x:', tx[:k+1], 'and', tx[-k-1:]
    print 'interior knots y:', ty[k+1:-k-1]
    print 'additional knots y:', ty[:k+1], 'and', ty[-k-1:]

    for itim,tim in enumerate(tims):
        psfex = tim.psfex
        psfex.fitSavedData(*psfex.splinedata)
        if plots:
            print
            print 'Tim', tim
            print
            pp,xx,yy = psfex.splinedata
            ny,nx,nparams = pp.shape
            assert(len(xx) == nx)
            assert(len(yy) == ny)
            psfnil = psfex.psfclass(*np.zeros(nparams))
            names = psfnil.getParamNames()
            xa = np.linspace(xx[0], xx[-1],  50)
            ya = np.linspace(yy[0], yy[-1], 100)
            #xa,ya = np.meshgrid(xa,ya)
            #xa = xa.ravel()
            #ya = ya.ravel()
            print 'xa', xa
            print 'ya', ya
            for i in range(nparams):
                plt.clf()
                plt.subplot(1,2,1)
                dimshow(pp[:,:,i])
                plt.title('grid fit')
                plt.colorbar()
                plt.subplot(1,2,2)
                sp = psfex.splines[i](xa, ya)
                sp = sp.T
                print 'spline shape', sp.shape
                assert(sp.shape == (len(ya),len(xa)))
                dimshow(sp, extent=[xx[0],xx[-1],yy[0],yy[-1]])
                plt.title('spline')
                plt.colorbar()
                plt.suptitle('tim %s: PSF param %s' % (tim.name, names[i]))
                ps.savefig()

def stage_initplots(
    coimgs=None, cons=None, bands=None, ps=None,
    targetwcs=None,
    blobs=None,
    T=None, cat=None, tims=None, tractor=None, **kwargs):
    # RGB image
    # plt.clf()
    # dimshow(get_rgb(coimgs, bands))
    # ps.savefig()

    print 'T:'
    T.about()

    # cluster zoom-in
    #x0,x1, y0,y1 = 1700,2700, 200,1200
    #x0,x1, y0,y1 = 1900,2500, 400,1000
    #x0,x1, y0,y1 = 1900,2400, 450,950
    x0,x1, y0,y1 = 0,500, 0,500

    clco = [co[y0:y1, x0:x1] for co in coimgs]
    clW, clH = x1-x0, y1-y0
    clwcs = targetwcs.get_subimage(x0, y0, clW, clH)

    plt.figure(figsize=(6,6))
    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    ps.suffixes = ['png','pdf']

    # cluster zoom-in
    rgb = get_rgb(clco, bands)
    plt.clf()
    dimshow(rgb, ticks=False)
    ps.savefig()

    # blobs
    #b0 = blobs
    #b1 = binary_dilation(blobs, np.ones((3,3)))
    #bout = np.logical_and(b1, np.logical_not(b0))
    # b0 = blobs
    # b1 = binary_erosion(b0, np.ones((3,3)))
    # bout = np.logical_and(b0, np.logical_not(b1))
    # # set green
    # rgb[:,:,0][bout] = 0.
    # rgb[:,:,1][bout] = 1.
    # rgb[:,:,2][bout] = 0.
    # plt.clf()
    # dimshow(rgb, ticks=False)
    # ps.savefig()

    # Initial model (SDSS only)
    try:
        # convert from string to int
        T.objid = np.array([int(x) if len(x) else 0 for x in T.objid])
    except:
        pass
    scat = Catalog(*[cat[i] for i in np.flatnonzero(T.objid)])
    sedcat = Catalog(*[cat[i] for i in np.flatnonzero(T.objid == 0)])

    print len(cat), 'total sources'
    print len(scat), 'SDSS sources'
    print len(sedcat), 'SED-matched sources'
    tr = Tractor(tractor.images, scat)

    comods = []
    comods2 = []
    for iband,band in enumerate(bands):
        comod = np.zeros((clH,clW))
        comod2 = np.zeros((clH,clW))
        con = np.zeros((clH,clW))
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            (Yo,Xo,Yi,Xi) = tim.resamp
            mod = tr.getModelImage(tim)
            Yo -= y0
            Xo -= x0
            K, = np.nonzero((Yo >= 0) * (Yo < clH) * (Xo >= 0) * (Xo < clW))
            Xo,Yo,Xi,Yi = Xo[K],Yo[K],Xi[K],Yi[K]
            comod[Yo,Xo] += mod[Yi,Xi]
            ie = tim.getInvError()
            noise = np.random.normal(size=ie.shape) / ie
            noise[ie == 0] = 0.
            comod2[Yo,Xo] += mod[Yi,Xi] + noise[Yi,Xi]
            con[Yo,Xo] += 1
        comod /= np.maximum(con, 1)
        comods.append(comod)
        comod2 /= np.maximum(con, 1)
        comods2.append(comod2)
    
    plt.clf()
    dimshow(get_rgb(comods2, bands), ticks=False)
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(comods, bands), ticks=False)
    ps.savefig()

    # Overplot SDSS sources
    ax = plt.axis()
    for src in scat:
        rd = src.getPosition()
        ok,x,y = clwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', ms=10, mew=1.5)
        else:
            plt.plot(x-1, y-1, 'o', mec='r', mfc='none', ms=10, mew=1.5)
    plt.axis(ax)
    ps.savefig()

    # Add SED-matched detections
    for src in sedcat:
        rd = src.getPosition()
        ok,x,y = clwcs.radec2pixelxy(rd.ra, rd.dec)
        plt.plot(x-1, y-1, 'o', mec='c', mfc='none', ms=10, mew=1.5)
    plt.axis(ax)
    ps.savefig()

    # Mark SED-matched detections on image
    plt.clf()
    dimshow(get_rgb(clco, bands), ticks=False)
    ax = plt.axis()
    for src in sedcat:
        rd = src.getPosition()
        ok,x,y = clwcs.radec2pixelxy(rd.ra, rd.dec)
        #plt.plot(x-1, y-1, 'o', mec='c', mfc='none', ms=10, mew=1.5)
        x,y = x-1, y-1
        hi,lo = 20,7
        # plt.plot([x-lo,x-hi],[y,y], 'c-')
        # plt.plot([x+lo,x+hi],[y,y], 'c-')
        # plt.plot([x,x],[y+lo,y+hi], 'c-')
        # plt.plot([x,x],[y-lo,y-hi], 'c-')
        plt.annotate('', (x,y+lo), xytext=(x,y+hi),
                     arrowprops=dict(color='c', width=1, frac=0.3, headwidth=5))
    plt.axis(ax)
    ps.savefig()

    # plt.clf()
    # dimshow(get_rgb([gaussian_filter(x,1) for x in clco], bands), ticks=False)
    # ps.savefig()

    # Resid
    # plt.clf()
    # dimshow(get_rgb([im-mo for im,mo in zip(clco,comods)], bands), ticks=False)
    # ps.savefig()

    # find SDSS fields within that WCS
    #sdss = DR9(basedir=photoobjdir)
    #sdss.useLocalTree()
    sdss = DR9(basedir='tmp')
    sdss.saveUnzippedFiles('tmp')

    #wfn = sdss.filenames.get('window_flist', None)
    wfn = os.path.join(os.environ['PHOTO_RESOLVE'], 'window_flist.fits')

    from astrometry.sdss.fields import radec_to_sdss_rcf
    
    clra,cldec = clwcs.radec_center()
    clrad = clwcs.radius()
    clrad = clrad + np.hypot(10.,14.)/2./60.
    print 'Searching for run,camcol,fields with radius', clrad, 'deg'
    RCF = radec_to_sdss_rcf(clra, cldec, radius=clrad*60., tablefn=wfn)
    print 'Found %i fields possibly in range' % len(RCF)

    sdsscoimgs = [np.zeros((clH,clW),np.float32) for band in bands]
    sdsscons   = [np.zeros((clH,clW),np.float32) for band in bands]
    for run,camcol,field,r,d in RCF:
        for iband,band in enumerate(bands):
            bandnum = band_index(band)
            sdss.retrieve('frame', run, camcol, field, band)
            frame = sdss.readFrame(run, camcol, field, bandnum)
            print 'Got frame', frame
            h,w = frame.getImageShape()
            simg = frame.getImage()
            wcs = AsTransWrapper(frame.astrans, w, h, 0.5, 0.5)
            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(clwcs, wcs, [], 3)
            except OverlapError:
                continue
            sdsscoimgs[iband][Yo,Xo] += simg[Yi,Xi]
            sdsscons  [iband][Yo,Xo] += 1
    for co,n in zip(sdsscoimgs, sdsscons):
        co /= np.maximum(1e-6, n)

    plt.clf()
    dimshow(get_rgb(sdsscoimgs, bands, **rgbkwargs), ticks=False)
    #plt.title('SDSS')
    ps.savefig()


'''
Plots; single-image image,invvar,model FITS files
'''
def stage_fitplots(
    T=None, coimgs=None, cons=None,
    cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None, brickid=None,
    plots=False, plots2=False, tims=None, tractor=None,
    pipe=None,
    outdir=None,
    **kwargs):

    for tim in tims:
        print 'Tim', tim, 'PSF', tim.getPsf()
        
    writeModels = False

    if pipe:
        t0 = Time()
        # Produce per-band coadds, for plots
        coimgs,cons = compute_coadds(tims, bands, targetwcs)
        print 'Coadds:', Time()-t0

    plt.figure(figsize=(10,10.5))
    #plt.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.998)
    plt.subplots_adjust(left=0.002, right=0.998, bottom=0.002, top=0.95)

    plt.clf()
    dimshow(get_rgb(coimgs, bands))
    plt.title('Image')
    ps.savefig()

    ax = plt.axis()
    cat = tractor.getCatalog()
    for i,src in enumerate(cat):
        rd = src.getPosition()
        ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
        cc = (0,1,0)
        if isinstance(src, PointSource):
            plt.plot(x-1, y-1, '+', color=cc, ms=10, mew=1.5)
        else:
            plt.plot(x-1, y-1, 'o', mec=cc, mfc='none', ms=10, mew=1.5)
        # plt.text(x, y, '%i' % i, color=cc, ha='center', va='bottom')
    plt.axis(ax)
    ps.savefig()

    mnmx = -5,300
    arcsinha = dict(mnmx=mnmx, arcsinh=1)

    # After plot
    rgbmod = []
    rgbmod2 = []
    rgbresids = []
    rgbchisqs = []

    chibins = np.linspace(-10., 10., 200)
    chihist = [np.zeros(len(chibins)-1, int) for band in bands]

    wcsW = targetwcs.get_width()
    wcsH = targetwcs.get_height()
    print 'Target WCS shape', wcsW,wcsH

    t0 = Time()
    mods = _map(_get_mod, [(tim, cat) for tim in tims])
    print 'Getting model images:', Time()-t0

    orig_wcsxy0 = [tim.wcs.getX0Y0() for tim in tims]
    for iband,band in enumerate(bands):
        coimg = coimgs[iband]
        comod  = np.zeros((wcsH,wcsW), np.float32)
        comod2 = np.zeros((wcsH,wcsW), np.float32)
        cochi2 = np.zeros((wcsH,wcsW), np.float32)
        for itim, (tim,mod) in enumerate(zip(tims, mods)):
            if tim.band != band:
                continue

            #mod = tractor.getModelImage(tim)

            if plots2:
                plt.clf()
                dimshow(tim.getImage(), **tim.ima)
                plt.title(tim.name)
                ps.savefig()
                plt.clf()
                dimshow(mod, **tim.ima)
                plt.title(tim.name)
                ps.savefig()
                plt.clf()
                dimshow((tim.getImage() - mod) * tim.getInvError(), **imchi)
                plt.title(tim.name)
                ps.savefig()

            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            comod[Yo,Xo] += mod[Yi,Xi]
            ie = tim.getInvError()
            noise = np.random.normal(size=ie.shape) / ie
            noise[ie == 0] = 0.
            comod2[Yo,Xo] += mod[Yi,Xi] + noise[Yi,Xi]
            chi = ((tim.getImage()[Yi,Xi] - mod[Yi,Xi]) * tim.getInvError()[Yi,Xi])
            cochi2[Yo,Xo] += chi**2
            chi = chi[chi != 0.]
            hh,xe = np.histogram(np.clip(chi, -10, 10).ravel(), bins=chibins)
            chihist[iband] += hh

            if not writeModels:
                continue

            im = tim.imobj
            fn = 'image-b%06i-%s-%s.fits' % (brickid, band, im.name)

            wcsfn = create_temp()
            wcs = tim.getWcs().wcs
            x0,y0 = orig_wcsxy0[itim]
            h,w = tim.shape
            subwcs = wcs.get_subimage(int(x0), int(y0), w, h)
            subwcs.write_to(wcsfn)

            primhdr = fitsio.FITSHDR()
            primhdr.add_record(dict(name='X0', value=x0, comment='Pixel origin of subimage'))
            primhdr.add_record(dict(name='Y0', value=y0, comment='Pixel origin of subimage'))
            xfn = im.wcsfn.replace(decals_dir+'/', '')
            primhdr.add_record(dict(name='WCS_FILE', value=xfn))
            xfn = im.psffn.replace(decals_dir+'/', '')
            primhdr.add_record(dict(name='PSF_FILE', value=xfn))
            primhdr.add_record(dict(name='INHERIT', value=True))

            imhdr = fitsio.read_header(wcsfn)
            imhdr.add_record(dict(name='EXTTYPE', value='IMAGE', comment='This HDU contains image data'))
            ivhdr = fitsio.read_header(wcsfn)
            ivhdr.add_record(dict(name='EXTTYPE', value='INVVAR', comment='This HDU contains an inverse-variance map'))
            fits = fitsio.FITS(fn, 'rw', clobber=True)
            tim.toFits(fits, primheader=primhdr, imageheader=imhdr, invvarheader=ivhdr)

            imhdr.add_record(dict(name='EXTTYPE', value='MODEL', comment='This HDU contains a Tractor model image'))
            fits.write(mod, header=imhdr)
            print 'Wrote image and model to', fn
            
        comod  /= np.maximum(cons[iband], 1)
        comod2 /= np.maximum(cons[iband], 1)

        rgbmod.append(comod)
        rgbmod2.append(comod2)
        resid = coimg - comod
        resid[cons[iband] == 0] = np.nan
        rgbresids.append(resid)
        rgbchisqs.append(cochi2)

        # Plug the WCS header cards into these images
        wcsfn = create_temp()
        targetwcs.write_to(wcsfn)
        hdr = fitsio.read_header(wcsfn)
        os.remove(wcsfn)

        if outdir is None:
            outdir = '.'
        wa = dict(clobber=True, header=hdr)
        for name,img in [('image', coimg), ('model', comod), ('resid', resid), ('chi2', cochi2)]:
            fn = os.path.join(outdir, '%s-coadd-%06i-%s.fits' % (name, brickid, band))
            fitsio.write(fn, img,  **wa)
            print 'Wrote', fn

    del cons

    plt.clf()
    dimshow(get_rgb(rgbmod, bands))
    plt.title('Model')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbmod2, bands))
    plt.title('Model + Noise')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbresids, bands))
    plt.title('Residuals')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbresids, bands, mnmx=(-30,30)))
    plt.title('Residuals (2)')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(coimgs, bands, **arcsinha))
    plt.title('Image (stretched)')
    ps.savefig()

    plt.clf()
    dimshow(get_rgb(rgbmod2, bands, **arcsinha))
    plt.title('Model + Noise (stretched)')
    ps.savefig()

    del coimgs
    del rgbresids
    del rgbmod
    del rgbmod2

    plt.clf()
    g,r,z = rgbchisqs
    im = np.log10(np.dstack((z,r,g)))
    mn,mx = 0, im.max()
    dimshow(np.clip((im - mn) / (mx - mn), 0., 1.))
    plt.title('Chi-squared')
    ps.savefig()

    plt.clf()
    xx = np.repeat(chibins, 2)[1:-1]
    for y,cc in zip(chihist, 'grm'):
        plt.plot(xx, np.repeat(np.maximum(0.1, y),2), '-', color=cc)
    plt.xlabel('Chi')
    plt.yticks([])
    plt.axvline(0., color='k', alpha=0.25)
    ps.savefig()

    plt.yscale('log')
    mx = np.max([max(y) for y in chihist])
    plt.ylim(1, mx * 1.05)
    ps.savefig()

    return dict(tims=tims)

