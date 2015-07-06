import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import sys
import os

import fitsio

from astrometry.util.fits import fits_table
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.libkd.spherematch import *

from astrometry.util.util import *

from tractor import *

from common import *

def subplot_grid(ny, nx, i):
    y = i/nx
    x = i%nx
    plt.subplot(ny, nx, 1 + ((ny-1)-y)*nx + x)


if __name__ == '__main__':
    decals = Decals()

    ps = PlotSequence('psf')

    #B = decals.get_bricks()

    T = decals.get_ccds()
    T.cut(T.extname == 'S1')
    print 'Cut to', len(T)
    #print 'Expnums:', T[:10]
    T.cut(T.expnum == 348233)
    print 'Cut to', len(T)
    T.about()

    band = T.filter[0]
    print 'Band:', band

    im = DecamImage(decals, T[0])
    print 'Reading', im.imgfn

    # Get approximate image center for astrometry
    hdr = im.read_image_header()
    wcs = Tan(hdr['CRVAL1'], hdr['CRVAL2'], hdr['CRPIX1'], hdr['CRPIX2'],
              hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2'],
              hdr['NAXIS1'], hdr['NAXIS2'])
    print 'WCS:', wcs
    r,d = wcs.pixelxy2radec(wcs.imagew/2, wcs.imageh/2)

    pixscale = wcs.pixel_scale()
    run_calibs(im, r, d, pixscale, astrom=True, morph=False, se2=False)


    iminfo = im.get_image_info()
    print 'img:', iminfo
    H,W = iminfo['dims']
    #psfex = PsfEx(im.psffn, W, H, nx=6)
    #psfex = PsfEx(im.psffn, W, H, ny=13, nx=7)
    #psfex = PsfEx(im.psffn, W, H, ny=9, nx=5)

    #psfex = PsfEx(im.psffn, W, H, ny=17, nx=9,
    #              psfClass=GaussianMixtureEllipsePSF)
    psfex = PsfEx(im.psffn, W, H, ny=13, nx=7,
                  psfClass=GaussianMixtureEllipsePSF)
    
    fn = 'psfex-ellipses.fits'
    if os.path.exists(fn):

        plt.figure(figsize=(5,10))
        plt.subplots_adjust(left=0.01, bottom=0.01, top=0.95, right=0.99,
                            wspace=0.05, hspace=0.025)


        pp = fitsio.read(fn)
        print 'Read parameters:', pp.shape
        ny,nx,nparams = pp.shape
        print 'nx,ny', nx,ny
        print 'nparams', nparams
        psfex.ny = ny
        psfex.nx = nx

        YY = np.linspace(0, psfex.H, psfex.ny)
        XX = np.linspace(0, psfex.W, psfex.nx)
        #psfex.splinedata = (pp, XX, YY)

        # Convert to GaussianMixturePSF
        ppvar = np.zeros_like(pp)
        for iy in range(ny):
            for ix in range(nx):
                psf = GaussianMixtureEllipsePSF(*pp[iy, ix, :])
                mog = psf.toMog()
                ppvar[iy,ix,:] = mog.getParams()
        psfexvar = PsfEx(im.psffn, W, H, ny=psfex.ny, nx=psfex.nx,
                         psfClass=GaussianMixturePSF)
        psfexvar.splinedata = (ppvar, XX, YY)
        #psfexvar.fitSavedData(ppvar, XX, YY)
        #fitsio.write('psfex-variances.fits', ppvar, clobber=True)
        
        psfexvar.toFits('psfex-var-2.fits')

        psfexvar2 = PsfEx.fromFits('psfex-var-2.fits')
        
        sys.exit(0)
        

        psfex.fitSavedData(pp, XX, YY)
        
        subpp = pp[::2, ::2, :]
        subXX, subYY = XX[::2], YY[::2]
        print 'subpp:', subpp.shape
        subpsfex = PsfEx(im.psffn, W, H, ny=len(subYY), nx=len(subXX),
                         psfClass=GaussianMixtureEllipsePSF)
        subpsfex.fitSavedData(subpp, subXX, subYY)

        ppvar = np.zeros_like(pp)
        for iy in range(ny):
            for ix in range(nx):
                psf = GaussianMixtureEllipsePSF(*pp[iy, ix, :])
                #print 'PSF:', psf
                mog = psf.toMog()
                #print 'MoG:', mog
                #print 'Params:', mog.getParams()
                ppvar[iy,ix,:] = mog.getParams()
        psfexvar = PsfEx(im.psffn, W, H, ny=17, nx=9,
                         psfClass=GaussianMixturePSF)
        #psfexvar = VaryingGaussianPSF(W, H, ny=17, nx=9)
        psfexvar.fitSavedData(ppvar, XX, YY)

        subvar = ppvar[::2, ::2, :]
        subpsfexvar = PsfEx(im.psffn, W, H, ny=len(subYY), nx=len(subXX),
                            psfClass=GaussianMixturePSF)
        subpsfexvar.fitSavedData(subvar, subXX, subYY)
        

        psfgrid = []
        psfcropgrid = []
        crop = 10
        for y in YY:
            for x in XX:
                psfimg = psfex.instantiateAt(x, y)
                psfgrid.append(psfimg)
                h,w = psfimg.shape
                img = psfimg[h/2-crop:h/2+crop+1, w/2-crop:w/2+crop+1]
                psfcropgrid.append(img)
        mx = np.max([psfimg.max() for psfimg in psfgrid])
        logmx = np.log10(mx)
        plt.clf()
        for i,psfimg in enumerate(psfcropgrid):
            #plt.subplot(len(YY), len(XX), i+1)
            subplot_grid(len(YY), len(XX), i)
            dimshow(np.log10(np.maximum(psfimg, mx*1e-16)),
                    vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
        plt.suptitle('PsfEx models')
        ps.savefig()


        modnames = ['Dense-grid MoG', 'Coarse-grid MoG',
                    'Dense-grid MoG (variance)', 'Coarse-grid MoG (variance)']
        models = [ psfex, subpsfex, psfexvar, subpsfexvar ]

        modgrids = [[] for m in models]
        
        for iy,y in enumerate(YY):
            for ix,x in enumerate(XX):

                for model,grid in zip(models, modgrids):
                    psf = model.psfAt(x, y)
                    mod = psf.getPointSourcePatch(0., 0., radius=crop)
                    assert(mod.shape == psfcropgrid[0].shape)
                    grid.append(mod.patch)
                

        for name,modgrid in zip(modnames, modgrids):
            plt.clf()
            for i,psfimg in enumerate(modgrid):
                subplot_grid(len(YY), len(XX), i)
                dimshow(np.log10(np.maximum(psfimg, mx*1e-16)),
                        vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
            plt.suptitle(name)
            ps.savefig()

        for name,modgrid in zip(modnames, modgrids):
            plt.clf()
            for i,(psfimg,modimg) in enumerate(zip(psfcropgrid,modgrid)):
                subplot_grid(len(YY), len(XX), i)
                diff = psfimg - modimg
                #print 'Max diff:', np.abs(diff).max()
                dimshow(psfimg - modimg, vmin=-0.001, vmax=0.001,
                        ticks=False, cmap='RdBu')
            plt.suptitle('PsfEx - %s' % name)
            ps.savefig()

        
        sys.exit(0)


    
    # for ibase,basis in enumerate(psfex.psfbases):
    #     plt.clf()
    #     plt.subplot(2,1,1)
    #     mx = np.percentile(np.abs(basis), 99.5)
    #     if ibase == 0:
    #         dimshow(basis, ticks=False)
    #     else:
    #         dimshow(basis, ticks=False, vmin=-mx, vmax=mx)
    #     #plt.colorbar(fraction=0.2)
    #     plt.suptitle('PsfEx eigen-PSF %i' % ibase)
    #     plt.subplot(2,1,2)
    # 
    #     xx,yy = np.meshgrid(np.linspace(0, W, 50), np.linspace(0, H, 50))
    #     print 'xx,yy', xx.shape, yy.shape
    #     amp = np.zeros_like(xx)
    #     print 'amp', amp.shape
    #     
    #     dx = (xx - psfex.x0) / psfex.xscale
    #     dy = (yy - psfex.y0) / psfex.yscale
    #     for d in range(psfex.degree + 1):
    #         print 'degree', d
    #         for j in range(d+1):
    #             k = d - j
    #             print 'j', j, 'k', k
    #             # PSFEx manual pg. 111 ?
    #             ii = j + (psfex.degree+1) * k - (k * (k-1))/ 2
    #             print 'ii', ii
    #             if ii != ibase:
    #                 continue
    #             amp += dx**j * dy**k
    #     dimshow(amp, extent=[xx[0,0],xx[0,-1],yy[0,0],yy[-1,0]])
    #     plt.colorbar(fraction=0.25)
    #     ps.savefig()

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92,
                        wspace=0.1, hspace=0.2)
    
    nbases = len(psfex.psfbases)
    cols = int(np.ceil(np.sqrt(nbases) * 1.3))
    rows = int(np.ceil(nbases / float(cols)))
    plt.clf()
    for ibase,basis in enumerate(psfex.psfbases):
        plt.subplot(rows, cols, ibase+1)
        mx = np.percentile(np.abs(basis), 99.5)
        if ibase == 0:
            dimshow(basis, ticks=False)
        else:
            dimshow(basis, ticks=False, vmin=-mx, vmax=mx)

        for d in range(psfex.degree + 1):
            print 'degree', d
            for j in range(d+1):
                k = d - j
                print 'j', j, 'k', k
                # PSFEx manual pg. 111 ?
                ii = j + (psfex.degree+1) * k - (k * (k-1))/ 2
                print 'ii', ii
                if ii == ibase:
                    xoyo = (j,k)
        plt.title('$x^%i y^%i$' % xoyo, fontdict=dict(fontsize=10))
    plt.suptitle('PsfEx eigen-PSFs')
    ps.savefig()
    
    plt.figure(figsize=(5,10))
    plt.subplots_adjust(left=0.01, bottom=0.01, top=0.95, right=0.99,
                        wspace=0.05, hspace=0.025)

    S = im.read_sdss()
    print len(S), 'SDSS sources'
    S.cut(S.objc_type == 6)
    print len(S), 'SDSS stars'
    S.flux = S.get('%s_psfflux' % band)

    wcs = im.read_wcs()

    img = im.read_image()
    sky = np.median(img)
    img -= sky
    invvar = im.read_invvar(clip=True)
    sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
    sigoff = 3
    img += sig1 * sigoff
    # convert to sigmas
    img /= sig1
    invvar *= sig1**2

    H,W = img.shape
    sz = 22


    # Read sources detected in DECam image too
    T = fits_table(im.sefn, hdu=2)
    print 'Got', len(T), 'DECam sources'
    T.about()
    T.ra,T.dec = wcs.pixelxy2radec(T.x_image, T.y_image)
    I,J,d = match_radec(S.ra, S.dec, T.ra, T.dec, 1./3600., nearest=True)
    print 'Matched', len(I)

    # Replace SDSS RA,Dec by DECam RA,Dec
    S.cut(I)
    S.ra  = T.ra [J]
    S.dec = T.dec[J]

    
    ok,S.x,S.y = wcs.radec2pixelxy(S.ra, S.dec)
    S.x -= 1
    S.y -= 1
    S.ix, S.iy = np.round(S.x).astype(int), np.round(S.y).astype(int)
    S.cut((S.ix >= sz) * (S.iy >= sz) * (S.ix < W-sz) * (S.iy < H-sz))
    print len(S), 'SDSS stars in bounds'

    S.cut(invvar[S.iy, S.ix] > 0)
    print len(S), 'SDSS stars not in masked regions'
    S.cut(np.argsort(-S.flux))

    rows,cols = 4,5
    #rows,cols = 2,2

    
    subimgs = []
    for i in range(rows*cols):
        s = S[i]
        subimg = img[s.iy - sz : s.iy + sz+1, s.ix - sz : s.ix + sz+1]
        subimgs.append(subimg)
    
    # plt.clf()
    # for i,subimg in enumerate(subimgs):
    #     plt.subplot(rows, cols, 1+i)
    #     dimshow(subimg, ticks=False)
    #     #plt.colorbar()
    # ps.savefig()
    
    maxes = []
    # plt.clf()
    for i,subimg in enumerate(subimgs):
        # plt.subplot(rows, cols, 1+i)
        mx = subimg.max()
        maxes.append(mx)
        # logmx = np.log10(mx)
        # dimshow(np.log10(np.maximum(subimg, mx*1e-16)), vmin=0, vmax=logmx,
        #         ticks=False)
    # ps.savefig()

    origpsfimgs = []
    unitpsfimgs = []
    psfimgs = []

    YY = np.linspace(0, psfex.H, psfex.ny)
    XX = np.linspace(0, psfex.W, psfex.nx)
    psfgrid = []
    psfcropgrid = []
    crop = 10
    for y in YY:
        for x in XX:
            psfimg = psfex.instantiateAt(x, y)
            psfgrid.append(psfimg)
            h,w = psfimg.shape
            img = psfimg[h/2-crop:h/2+crop+1, w/2-crop:w/2+crop+1]
            psfcropgrid.append(img)
    mx = np.max([psfimg.max() for psfimg in psfgrid])
    logmx = np.log10(mx)
    plt.clf()
    for i,psfimg in enumerate(psfcropgrid):
        #plt.subplot(len(YY), len(XX), i+1)
        subplot_grid(len(YY), len(XX), i)
        dimshow(np.log10(np.maximum(psfimg, mx*1e-16)),
                vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
    plt.suptitle('PsfEx models')
    ps.savefig()


    psfex.savesplinedata = True
    # psfex.ensureFit()
    psfex._fitParamGrid(damp=1)
    pp,XX,YY = psfex.splinedata


    # Convert to GaussianMixturePSF
    ppvar = np.zeros_like(pp)
    for iy in range(ny):
        for ix in range(nx):
            psf = GaussianMixtureEllipsePSF(*pp[iy, ix, :])
            mog = psf.toMog()
            ppvar[iy,ix,:] = mog.getParams()
    psfexvar = PsfEx(im.psffn, W, H, ny=psfex.ny, nx=psfex.nx,
                     psfClass=GaussianMixturePSF)
    #psfexvar.fitSavedData(ppvar, XX, YY)
    fitsio.write('psfex-variances.fits', ppvar, clobber=True)
    
    modgrid = []
    
    for iy,y in enumerate(YY):
        for ix,x in enumerate(XX):
            psf = psfex.psfAt(x,y)
            psf.radius = crop
            modimg = psf.getPointSourcePatch(0., 0.)
            modgrid.append(modimg.patch)

    
    # pp = []
    # px0 = None
    # for iy,y in enumerate(YY):
    #     pprow = []
    #     p0 = px0
    #     for ix,x in enumerate(XX):
    #         psfimg = psfex.instantiateAt(x, y)
    #         h,w = psfimg.shape
    #         cropped = psfimg[h/2-crop:h/2+crop+1, w/2-crop:w/2+crop+1]
    # 
    #         epsf = GaussianMixtureEllipsePSF.fromStamp(psfimg, P0=p0, damp=1.)
    #         p0 = epsf.getParams()
    #         if ix == 0:
    #             px0 = p0
    # 
    #         epsf.radius = crop
    #         modimg = epsf.getPointSourcePatch(0., 0.)
    #         modgrid.append(modimg.patch)
    #             
    #         if iy == 0 and False:
    #             plt.clf()
    #             plt.subplot(3,1,1)
    #             dimshow(np.log10(np.maximum(cropped, mx*1e-16)),
    #                     vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
    #             plt.subplot(3,1,2)
    #             dimshow(np.log10(np.maximum(modimg.patch, mx*1e-16)),
    #                     vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
    #             ax = plt.axis()
    #             angle = np.linspace(0., 2.*np.pi, 20)
    #             xx,yy = np.sin(angle), np.cos(angle)
    #             xy = np.vstack((xx,yy))
    #             for i,ell in enumerate(epsf.ellipses):
    #                 mx,my = epsf.mog.mean[i,:]
    #                 T = ell.getRaDecBasis()
    #                 T *= 3600.
    #                 txy = np.dot(T, xy)
    #                 plt.plot(crop + mx + txy[0,:],
    #                          crop + my + txy[1,:], 'k-', lw=1.5)
    #             plt.axis(ax)
    #             plt.subplot(3,1,3)
    #             dimshow(cropped - modimg.patch,
    #                     vmin=-0.001, vmax=0.001, ticks=False, cmap='RdBu')
    #             ps.savefig()
    # 
    #         print 'Fit PSF:', epsf
    #         #print 'Params:', epsf.getAllParams()
    #         #repsf = GaussianMixtureEllipsePSF(*epsf.getAllParams())
    #         #print 'Reconstructed:', repsf
    #         
    #         params = np.array(epsf.getAllParams())
    #         pprow.append(params)
    #     pp.append(pprow)
    # pp = np.array(pp)
    # print 'pp', pp.shape

    
    fitsio.write('psfex-ellipses.fits', pp, clobber=True)
    
    plt.clf()
    for i,modimg in enumerate(modgrid):
        subplot_grid(len(YY), len(XX), i)
        #plt.subplot(len(YY), len(XX), i+1)
        dimshow(np.log10(np.maximum(modimg, mx*1e-16)),
                vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
    plt.suptitle('PsfEx: Mixture-of-Gaussian fits')
    ps.savefig()

    plt.clf()
    for i,(psfimg,modimg) in enumerate(zip(psfcropgrid,modgrid)):
        subplot_grid(len(YY), len(XX), i)
        #plt.subplot(len(YY), len(XX), i+1)
        diff = psfimg - modimg
        print 'Max diff:', np.abs(diff).max()
        dimshow(psfimg - modimg, vmin=-0.001, vmax=0.001,
                ticks=False, cmap='RdBu')
    plt.suptitle('PsfEx: Pixelized - Mixture-of-Gaussian')
    ps.savefig()
    
    ny,nx,nparams = pp.shape
    names = psf.getParamNames()

    plt.figure(figsize=(10,10))
    
    
    #iii = [ [i + j for j in [0,3,4,9,10,11]] for i in [0,1,2] ]
    iii = [ [0, 3,4, 9,10,11],
            [1, 5,6, 12,13,14],
            [2, 7,8, 15,16,17] ]

    for ii in iii:
        plt.clf()
        for j,ip in enumerate(ii):
            plt.subplot(2,3, j+1)

            print 'Param', names[ip]
            print pp[:,:,ip]

            # param values from other components
            kpp = np.hstack([pp[:,:,kii[j]].ravel() for kii in iii])
            mn,mx = kpp.min(), kpp.max()
            
            dimshow(pp[:,:,ip], cmap='jet', ticks=False, vmin=mn, vmax=mx)
            plt.colorbar()
            plt.title(names[ip])
        plt.suptitle('Mixture of Gaussian models: spatial variation of parameters')
        ps.savefig()
    
    # for ip in range(nparams):
    #     plt.clf()
    #     dimshow(pp[:,:,ip], ticks=False)
    #     plt.colorbar()
    #     plt.title(names[ip])
    #     ps.savefig()
    


    sys.exit(0)
    
    psfex.savesplinedata = True
    print 'Fitting PsfEx model...'
    psfex.ensureFit()

    modgrid = []
    for y in YY:
        for x in XX:
            mog = psfex.psfAt(x, y)
            mog.radius = crop
            modimg = mog.getPointSourcePatch(0., 0.)
            print 'Patch shape', modimg.shape
            modgrid.append(modimg.patch)
    plt.clf()
    for i,modimg in enumerate(modgrid):
        plt.subplot(len(YY), len(XX), i+1)
        #h,w = psfimg.shape
        #img = psfimg[h/2-crop:h/2+crop, w/2-crop:w/2+crop]
        dimshow(np.log10(np.maximum(modimg, mx*1e-16)),
                vmax=logmx, vmin=logmx-4, ticks=False, cmap='jet')
    plt.suptitle('PsfEx: Mixture-of-Gaussian fits')
    ps.savefig()

    plt.clf()
    for i,(psfimg,modimg) in enumerate(zip(psfcropgrid,modgrid)):
        plt.subplot(len(YY), len(XX), i+1)
        diff = psfimg - modimg
        print 'Max diff:', np.abs(diff).max()
        dimshow(psfimg - modimg, vmin=-0.01, vmax=0.01,
                ticks=False, cmap='jet')
    plt.suptitle('PsfEx: Pixelized - Mixture-of-Gaussian')
    ps.savefig()
    
    sys.exit(0)

    
    plt.clf()
    for i,subimg in enumerate(subimgs):
        s = S[i]
        plt.subplot(rows, cols, 1+i)

        # Sum the flux near the core...
        ss = 5
        flux = np.sum(img[s.iy-ss:s.iy+ss+1, s.ix-ss:s.ix+ss+1])
        # subtract off the 3*sig we added
        flux -= (2*ss+1)**2 * sigoff

        psfimg = psfex.instantiateAt(s.x, s.y)

        origpsfimgs.append(psfimg)
        
        if True:
            from astrometry.util.util import lanczos3_interpolate
            dx,dy = s.x - s.ix, s.y - s.iy
            #print 'dx,dy', dx,dy
            ph,pw = psfimg.shape
            ix,iy = np.meshgrid(np.arange(pw), np.arange(ph))
            ix = ix.ravel().astype(np.int32)
            iy = iy.ravel().astype(np.int32)
            nn = len(ix)
            laccs = [np.zeros(nn, np.float32)]
            rtn = lanczos3_interpolate(ix, iy,
                                       -dx+np.zeros(len(ix),np.float32),
                                       -dy+np.zeros(len(ix),np.float32),
                                       laccs, [psfimg.astype(np.float32)])
            psfimg = laccs[0].reshape(psfimg.shape)

        unitpsfimgs.append(psfimg)
        psfimg = psfimg * flux + sigoff
        psfimgs.append(psfimg)
        
        mx = maxes[i]
        #mx = psfimg.max()
        logmx = np.log10(mx)
        dimshow(np.log10(np.maximum(psfimg, mx*1e-16)), vmin=0, vmax=logmx,
                ticks=False)
    ps.savefig()

    # plt.clf()
    # for i,(subimg,psfimg) in enumerate(zip(subimgs, psfimgs)):
    #     plt.subplot(rows, cols, 1+i)
    #     dimshow(subimg - psfimg, ticks=False)
    # ps.savefig()

    plt.clf()
    for i,(subimg,psfimg) in enumerate(zip(subimgs, psfimgs)):

        # Re-scale psfimgs
        h,w = subimg.shape
        A = np.zeros((h*w, 2))
        A[:,0] = 1.
        A[:,1] = psfimg.ravel()
        b = subimg.ravel()
        x,resid,rank,s = np.linalg.lstsq(A, b)
        #print 'x', x
        #psfimg = x[0] + psfimg * x[1]
        psfimg *= x[1]
        psfimg += x[0]
        
        res = subimg-psfimg
        #print 'range', res.min(), res.max()
        
        plt.subplot(rows, cols, 1+i)
        dimshow(subimg - psfimg, ticks=False, vmin=-5, vmax=5)
    plt.suptitle('Image - PixPSF')
    ps.savefig()

    # Re-fit the centers
    srcs = []
    tims = []
    
    plt.clf()
    for i,(subimg,upsfimg,opsfimg) in enumerate(zip(
            subimgs, unitpsfimgs, origpsfimgs)):

        s = S[i]
        print
        print
        A = np.zeros((h*w, 2))
        A[:,0] = 1.
        A[:,1] = upsfimg.ravel()
        b = subimg.ravel()
        x,resid,rank,nil = np.linalg.lstsq(A, b)
        sky  = x[0]
        flux = x[1]
        print 'Flux', flux
        print 'Sky', sky
        
        tim = Image(data=subimg, invvar=np.ones_like(subimg),
                    psf=PixelizedPSF(opsfimg), sky=ConstantSky(sky))
        tim.modelMinval = 1e-8
        tim.getWcs().pixscale = 3600.
        h,w = tim.shape
        src = PointSource(PixPos(w/2 + (s.x-s.ix), h/2 + (s.y-s.iy)), 
                                 Flux(flux))
        tr = Tractor([tim],[src])
        tr.freezeParam('images')
        print 'src', src
        for step in range(20):
            dlnp,X,alpha = tr.optimize(shared_params=False)
            print 'dlnp', dlnp, 'src', src
            if dlnp < 1e-6:
                break

        srcs.append(src)
        tims.append(tim)
        
        plt.subplot(rows, cols, 1+i)
        dimshow(subimg - tr.getModelImage(0), ticks=False, vmin=-5, vmax=5)
    plt.suptitle('Image - PixPSF model')
    ps.savefig()
            

    # Fit MoG model to each one
    mogs = []
    
    plt.clf()
    for i,(subimg,psfimg,tim,src) in enumerate(zip(
            subimgs, origpsfimgs, tims, srcs)):
        s = S[i]

        print 'PSF image sum', psfimg.sum()
        
        #mog = psfex.psfAt(s.x, s.y)
        mog = GaussianMixturePSF.fromStamp(psfimg)
        mogs.append(mog)
        
        tim.opsf = tim.psf
        tim.psf = mog
        tr = Tractor([tim],[src])
        
        plt.subplot(rows, cols, 1+i)
        dimshow(subimg - tr.getModelImage(0), ticks=False, vmin=-5, vmax=5)
    plt.suptitle('Image - MoG model')
    ps.savefig()
        
    plt.clf()
    for i,(mog,psfimg) in enumerate(zip(mogs, origpsfimgs)):

        ph,pw = psfimg.shape
        sz = pw/2
        #mog.radius = sz
        
        mogimg = mog.getPointSourcePatch(0., 0., radius=sz)
        #, extent=[-sz,sz,-sz,sz])
        mogimg = mogimg.patch
        mx = 0.002
        
        plt.subplot(rows, cols, 1+i)
        dimshow(psfimg - mogimg, ticks=False, vmin=-mx, vmax=mx)
        #plt.colorbar()
    plt.suptitle('PixPSF - MoG')
    ps.savefig()

    orig_mogs = [mog.copy() for mog in mogs]
    
    # Re-fit the MoG PSF to the pixelized postage stamp
    plt.clf()
    for i,(mog,psfimg) in enumerate(zip(mogs, origpsfimgs)):
        print
        print
        
        tim = Image(data=psfimg, invvar=1e6*np.ones_like(psfimg),
                    psf=mog)

        h,w = psfimg.shape
        src = PointSource(PixPos(w/2,h/2), Flux(1.))
        tr = Tractor([tim], [src])

        tr.freezeParam('catalog')
        tim.freezeAllBut('psf')

        print 'MoG', mog
        for step in range(20):
            dlnp,X,alpha = tr.optimize(shared_params=False)
            print 'dlnp', dlnp, 'mog', mog
            if dlnp < 1e-6:
                break
            
        sz = w/2
        mogimg = mog.getPointSourcePatch(0., 0., radius=sz)
        mogimg = mogimg.patch
        mx = 0.002
        
        plt.subplot(rows, cols, 1+i)
        dimshow(psfimg - mogimg, ticks=False, vmin=-mx, vmax=mx)
    plt.suptitle('PixPSF - MoG')
    ps.savefig()


    # Image - MoG model

    plt.clf()
    for i,(subimg,tim,src) in enumerate(zip(
            subimgs, tims, srcs)):
        tr = Tractor([tim],[src])
        plt.subplot(rows, cols, 1+i)
        dimshow(subimg - tr.getModelImage(0), ticks=False, vmin=-5, vmax=5)
    plt.suptitle('Image - MoG model')
    ps.savefig()



    ############## 

    # Re-fit the MoGs using EllipseESoft basis.
    epsfs = []
    plt.clf()
    for i,(mog,psfimg) in enumerate(zip(orig_mogs, origpsfimgs)):
        print
        print
        
        ells = [EllipseESoft.fromCovariance(cov)
                for cov in mog.mog.var]
        psf = GaussianMixtureEllipsePSF(mog.mog.amp, mog.mog.mean, ells)

        tim = Image(data=psfimg, invvar=1e6*np.ones_like(psfimg),
                    psf=psf)
        epsfs.append(psf)
        
        h,w = psfimg.shape
        src = PointSource(PixPos(w/2,h/2), Flux(1.))
        tr = Tractor([tim], [src])
        tr.freezeParam('catalog')
        tim.freezeAllBut('psf')

        print 'PSF', psf
        for step in range(20):
            dlnp,X,alpha = tr.optimize(shared_params=False)
            print 'dlnp', dlnp, 'psf', psf
            if dlnp < 1e-6:
                break
            
        sz = w/2
        mogimg = psf.getPointSourcePatch(0., 0., radius=sz)
        mogimg = mogimg.patch
        mx = 0.002
        
        plt.subplot(rows, cols, 1+i)
        dimshow(psfimg - mogimg, ticks=False, vmin=-mx, vmax=mx)
    plt.suptitle('PixPSF - MoG (ellipse)')
    ps.savefig()

    # Update the 'tims' with these newly-found PSFs
    for psf,tim in zip(epsfs, tims):
        tim.mogpsf = tim.psf
        tim.psf = psf

    # Image - MoG model
    # plt.clf()
    # for i,(subimg,tim,src) in enumerate(zip(
    #         subimgs, tims, srcs)):
    #     tr = Tractor([tim],[src])
    #     plt.subplot(rows, cols, 1+i)
    #     mod = tr.getModelImage(0)
    #     dimshow(mod, ticks=False)
    #     #plt.colorbar()
    # plt.suptitle('MoG (ellipse) model')
    # ps.savefig()
    
    plt.clf()
    for i,(subimg,tim,src,psf) in enumerate(zip(
            subimgs, tims, srcs,epsfs)):
        print 'Source:', src
        print 'PSF:', tim.getPsf()
        print 'subimage sum:', subimg.sum()
        tr = Tractor([tim],[src])
        plt.subplot(rows, cols, 1+i)
        mod = tr.getModelImage(0)
        print 'mod sum:', mod.sum()
        dimshow(subimg - mod, ticks=False, vmin=-5, vmax=5)
    plt.suptitle('Image - MoG (ellipse) model')
    ps.savefig()
    
