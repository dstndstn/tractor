from tractor.dense_optimizer import ConstrainedDenseOptimizer
import numpy as np

'''
A mixin class for LsqrOptimizer that does the linear update direction step
by factorizing over images -- it solves the linear problem for each image
independently, and then combines those results (via their covariances) into
the overall result.
'''
class FactoredOptimizer(object):

    def getSingleImageUpdateDirection(self, tr, **kwargs):
        #print('getSingleImageUpdateDirection( kwargs=', kwargs, ')')
        allderivs = tr.getDerivs()
        x,A = self.getUpdateDirection(tr, allderivs, get_A_matrix=True, **kwargs)
        icov = np.matmul(A.T, A)
        del A
        return x, icov

    def getLinearUpdateDirection(self, tr, **kwargs):
        #print('getLinearUpdateDirection( kwargs=', kwargs, ')')
        img_opts = []
        from tractor import Images

        imgs = tr.images
        for i,img in enumerate(imgs):
            tr.images = Images(img)
            x,x_icov = self.getSingleImageUpdateDirection(tr, **kwargs)
            # print('Opt for img', i, ':')
            # print(x)
            # print('And icov')
            # print(x_icov)
            img_opts.append((x,x_icov))
        tr.images = imgs

        # ~ inverse-covariance-weighted sum of img_opts...
        xicsum = 0
        icsum = 0
        for x,ic in img_opts:
            xicsum = xicsum + np.dot(ic, x)
            icsum = icsum + ic
        C = np.linalg.inv(icsum)
        x = np.dot(C, xicsum)
        # print('Total opt:')
        # print(x)
        return x


class FactoredDenseOptimizer(FactoredOptimizer, ConstrainedDenseOptimizer):
    pass


if __name__ == '__main__':

    import pylab as plt
    from tractor import Image, PixPos, Flux, Tractor, NullWCS, NCircularGaussianPSF, PointSource
    
    n_ims = 2
    sig1s = [3., 10.]
    H,W = 50,50
    cx,cy = 23,27
    psf_sigmas = [2., 1.]
    fluxes = [500., 500.]
    
    tims = []
    for i in range(n_ims):
        x = np.arange(W)
        y = np.arange(H)
        data = np.exp(-0.5 * ((x[np.newaxis,:] - cx)**2 + (y[:,np.newaxis] - cy)**2) /
                      psf_sigmas[i]**2)
        data *= fluxes[i] / (2. * np.pi * psf_sigmas[i]**2)
        data += np.random.normal(size=(50,50)) * sig1s[i]

        tims.append(Image(data=data, inverr=np.ones_like(data) / sig1s[i],
                          psf=NCircularGaussianPSF([psf_sigmas[i]], [1.]),
                          wcs=NullWCS()))
    src = PointSource(PixPos(W//2, H//2), Flux(100.))

    opt = FactoredDenseOptimizer()

    opt2 = ConstrainedDenseOptimizer()

    tr = Tractor(tims, [src], optimizer=opt)
    tr2 = Tractor(tims, [src], optimizer=opt2)
    tr.freezeParam('images')
    tr2.freezeParam('images')

    mods = list(tr.getModelImages())
    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods[i], **ima)
    plt.savefig('1.png')

    fit_kwargs = dict(shared_params=False, priors=False)
    up1 = tr.optimizer.getLinearUpdateDirection(tr, **fit_kwargs)
    up2 = tr2.optimizer.getLinearUpdateDirection(tr2, **fit_kwargs)

    print('Update directions:')
    print(up1)
    print(up2)
    
    tr.optimize_loop(**fit_kwargs)
    
    mods = list(tr.getModelImages())
    plt.clf()
    for i in range(n_ims):
        ima = dict(interpolation='nearest', origin='lower', vmin=-3.*sig1s[i],
                   vmax=5.*sig1s[i])
        plt.subplot(2,2, i*2 + 1)
        plt.imshow(tims[i].data, **ima)
        plt.subplot(2,2, i*2 + 2)
        plt.imshow(mods[i], **ima)
    plt.savefig('2.png')
    
