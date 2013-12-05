import matplotlib
matplotlib.use('Agg')
import pylab as plt

import sys
import numpy as np
import logging
lvl = logging.INFO
#lvl = logging.DEBUG
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

#from astrometry.util.plotutils import *

from tractor import *
from ceres import *

# H,W = 100,100
# NS = 10
#H,W = 13,25
#H,W = 25,25
#NS = 25
#NS = 5

# H,W = 5,25
# NS = 25

H,W = 5,5
NS = 11

f = 100.
img = np.zeros((H,W))
tim1 = Image(data=img, invvar=np.ones_like(img),
             psf=NCircularGaussianPSF([1.],[1.]),
             wcs=NullWCS(),
             photocal=LinearPhotoCal(1.),
             sky=ConstantSky(0.))

for seed in range(100):

    print
    print
    print 'Starting with seed', seed
    print
    
    np.random.seed(seed)

    #srcs = [PointSource(PixPos(np.random.randint(W), np.random.randint(H)),
    #                    Flux(f)) for x in range(NS)]
    srcs = [PointSource(PixPos(np.random.uniform(W), np.random.uniform(H)),
                        Flux(f)) for x in range(NS)]

    tractor = Tractor([tim1], srcs)
    tractor.freezeParam('images')
    for src in srcs:
        src.freezeParam('pos')

    mod = tractor.getModelImage(0)
    mod += np.random.normal(size=mod.shape)

    tim1.data = mod
    H,W = mod.shape
    iv = np.ones_like(mod)

    BH,BW = 5,5
    nbw = int(np.ceil(W / float(BW)))
    nbh = int(np.ceil(H / float(BH)))

    blocks = []
    for iy in range(nbh):
        for ix in range(nbw):
            x0 = ix * BW
            y0 = iy * BH
            slc = slice(y0, min(y0+BH, H)), slice(x0, min(x0+BW, W))
            dat = mod[slc].astype(np.float32)
            #mod0 = np.zeros_like(dat)
            mod0 = None
            print 'data type', dat.dtype
            data = (x0, y0, dat, mod0,
                    np.sqrt(iv[slc]).astype(np.float32))
            blocks.append((data, []))

    for i,src in enumerate(srcs):
        patches = src.getUnitFluxModelPatches(tim1)
        assert(len(patches) == 1)
        patch = patches[0]
        patch.clipTo(W,H)
        ph,pw = patch.shape
        bx0 = np.clip(int(np.floor( patch.x0       / float(BW))), 0, nbw-1)
        bx1 = np.clip(int(np.ceil ((patch.x0 + pw) / float(BW))), 0, nbw-1)
        by0 = np.clip(int(np.floor( patch.y0       / float(BH))), 0, nbh-1)
        by1 = np.clip(int(np.ceil ((patch.y0 + ph) / float(BH))), 0, nbh-1)
        #print 'Patch touches x blocks [%i,%i], y blocks [%i,%i]' % (bx0, bx1, by0, by1)

        for by in range(by0, by1+1):
            for bx in range(bx0, bx1+1):
                bi = by * nbw + bx
                deriv = (i, patch.x0, patch.y0, patch.patch.astype(np.float32))
                blocks[bi][1].append(deriv)


    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower',
               extent=[0, W, 0, H], cmap='gray')
    for block in blocks:
        ((x0,y0,img,mod0,iv), srclist) = block
        h,w = img.shape
        plt.plot([x0,x0+w,x0+w,x0,x0], [y0,y0,y0+h,y0+h,y0], 'r-', alpha=0.5)
        for (i,x0,y0,p) in srclist:
            h,w = p.shape
            #plt.plot([x0,x0+w,x0+w,x0,x0], [y0,y0,y0+h,y0+h,y0], 'b-', alpha=0.5)
            #plt.plot([x0+w/2.]*2, [y0,y0+h], 'b-', alpha=0.25)
            #plt.plot([x0,x0+w], [y0+h/2.]*2, 'b-', alpha=0.25)
            pass
        for i,src in enumerate(srcs):
            pos = src.getPosition()
            x,y = pos.x, pos.y
            plt.plot(x, y, 'b+', ms=12)
    plt.savefig('c%02i.png' % seed)
                
    nonneg = 1
    
    print 'Ceres forced phot:'
    fluxes = np.zeros(len(srcs)) + 10.
    x = ceres_forced_phot(blocks, fluxes, nonneg)
    print 'Fluxes:', fluxes

    tractor.setParams(np.zeros(tractor.numberOfParams()) + 10.)
    X2 = tractor.optimize_forced_photometry()
    print 'optimize_forced_photometry() fluxes:', tractor.getParams()

    tractor.setParams(np.zeros(tractor.numberOfParams()) + 10.)
    X2 = tractor.optimize_forced_photometry(use_ceres=True, nonneg=nonneg)
    print 'Ceres optimize_forced_photometry() fluxes:', tractor.getParams()

    
