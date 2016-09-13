from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from tractor import *
from astrometry.util.plotutils import *

def main():
    ps = PlotSequence('forced')
    
    W,H = 50,50
    sig1 = 1.
    flux = 1000.
    
    tim = Image(data=np.zeros((H,W)), inverr=np.ones((H,W)) / sig1,
                psf=NCircularGaussianPSF([0.8], [1.]),
                photocal=LinearPhotoCal(1.))
    tim.sig1 = sig1
    
    x1 = 20
    src1 = PointSource(PixPos(x1, H/2), Flux(flux))
    
    # What happens if we're missing one of the sources in the model?
    for x2 in [30, 25, 24, 23, 22, 20]:
        src2 = PointSource(PixPos(x2, H/2), Flux(flux))
    
        fitsrc1 = PointSource(PixPos(x1, H/2), Flux(1.))
        fitsrc1.freezeParam('pos')
        
    
        (mod,modx,chix,flux0,fluxes) = runtest([tim], [src1, src2], [fitsrc1], 250)
    
        plt.clf()
        plt.imshow(mod, interpolation='nearest', origin='lower')
        plt.title('Model: src2 distance = %i' % (x2 - x1))
        ps.savefig()
    
        plt.clf()
        plt.imshow(modx, interpolation='nearest', origin='lower')
        plt.title('Model: src2 distance = %i' % (x2 - x1))
        ps.savefig()
    
        plt.clf()
        plt.imshow(chix, interpolation='nearest', origin='lower', vmin=-3, vmax=3)
        plt.title('Chi: src2 distance = %i' % (x2 - x1))
        ps.savefig()
    
        xm = np.mean(fluxes)
        st = np.std(fluxes)
        xl = xm - 4.*st
        xh = xm + 4.*st        
        plt.clf()
        plt.hist(fluxes, bins=21, range=(xl,xh))
        plt.axvline(flux0, color='r')
        plt.axvline(flux, color='r', linestyle='--')
        plt.xlim(xl,xh)
        plt.xlabel('Fit Flux')
        plt.title('Fits: src2 distance = %i' % (x2 - x1))
        ps.savefig()


    # What happens if we have the source position a little wrong?
    chislices = []
    for fx1 in [20, 21, 22, 23, 24]:
        fitsrc1 = PointSource(PixPos(fx1, H/2), Flux(1.))
        fitsrc1.freezeParam('pos')
    
        (mod,modx,chix,flux0,fluxes) = runtest([tim], [src1], [fitsrc1], 250)
    
        plt.clf()
        plt.imshow(mod, interpolation='nearest', origin='lower')
        plt.title('Model: pos error = %i' % (fx1 - x1))
        ps.savefig()
    
        plt.clf()
        plt.imshow(modx, interpolation='nearest', origin='lower')
        plt.title('Model: pos error = %i' % (fx1 - x1))
        ps.savefig()
    
        plt.clf()
        plt.imshow(chix, interpolation='nearest', origin='lower', vmin=-3, vmax=3)
        plt.title('Chi: pos error = %i' % (fx1 - x1))
        ps.savefig()

        chislices.append(chix[H/2, :])
        
        xm = np.mean(fluxes)
        st = np.std(fluxes)
        xl = xm - 4.*st
        xh = xm + 4.*st        
        plt.clf()
        plt.hist(fluxes, bins=21, range=(xl,xh))
        plt.axvline(flux0, color='r')
        plt.axvline(flux, color='r', linestyle='--')
        plt.xlim(xl,xh)
        plt.xlabel('Fit Flux')
        plt.title('Fits: pos error = %i' % (fx1 - x1))
        ps.savefig()

    plt.clf()
    for chi in chislices:
        plt.plot(chi)
    plt.xlabel('Slice in x direction of image')
    plt.ylabel('Chi of best fit')
    ps.savefig()
        
    
def runtest(tims, realsrcs, fitsrcs, niters):
    tr = Tractor(tims, realsrcs)
    mod = tr.getModelImage(0)

    tr = Tractor(tims, fitsrcs)
    tr.freezeParam('images')
    #tr.printThawedParams()
    
    fluxes = []
    for i in range(niters):
        for tim in tims:
            if i == 0:
                noise = 0.
            else:
                noise = np.random.normal(scale=tim.sig1, size=tim.shape)
            tim.data = mod + noise

        if i == 1:
            modx = tims[0].data
    
        # while True:
        #     dlnp,X,alpha = tr.optimize(alphas=[0.1, 0.3, 1.])
        #     print 'dlnp', dlnp
        #     if dlnp < 0.1:
        #         break

        tr.optimize_forced_photometry()
    
        if i == 1:
            chix = tr.getChiImage(0)
            
        if i > 0:
            fluxes.append(tr.getParams()[0])
        else:
            flux0 = tr.getParams()[0]
            print('Noise-free fit flux:', flux0)

    return mod, modx, chix, flux0, fluxes
            

    
if __name__ == '__main__':
    main()
    
