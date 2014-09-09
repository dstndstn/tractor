import os

from astrometry.util.plotutils import *
from astrometry.util.ttime import *
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from tractor.psfex import *
from tractor import *

from ngmix.em import *
#from ngmix.observation import *

def plot_result(gpsf, psfimg):
    psfimg2 = np.maximum(psfimg, 0)
    psfimg2 /= psfimg2.sum()

    #gpsf.mog.normalize()

    mx = np.max(np.log10(psfimg[psfimg>0] + 1e-3))
    #mn = np.min(np.log10(psfimg + 1e-3))
    mn = np.log10(5e-4)
    
    plt.clf()
    plt.subplot(2,3,1)
    dimshow(np.log10(psfimg + 1e-3), vmin=mn, vmax=mx)

    plt.subplot(2,3,2)
    dimshow(np.log10(psfimg2 + 1e-3), vmin=mn, vmax=mx)
    plt.subplot(2,3,3)
    gpsf.radius = pw/2
    img = gpsf.getPointSourcePatch(0.,0.).patch
    dimshow(np.log10(img + 1e-3), vmin=mn, vmax=mx)

    ax = plt.axis()
    vv = gpsf.mog.var
    mu = gpsf.mog.mean
    K  = gpsf.mog.K
    h,w = psfimg.shape
    cc = 'k'
    aa=1.0
    for k in range(K):
        v = vv[k,:,:]
        u,s,v = np.linalg.svd(v)
        angle = np.linspace(0., 2.*np.pi, 200)
        u1 = u[0,:]
        u2 = u[1,:]
        s1,s2 = np.sqrt(s)
        xy = (u1[np.newaxis,:] * s1 * np.cos(angle)[:,np.newaxis] +
              u2[np.newaxis,:] * s2 * np.sin(angle)[:,np.newaxis])
        plt.plot(xy[:,0]+w/2+mu[k,0], xy[:,1]+h/2+mu[k,1], '-',
                 color=cc, alpha=aa)
    plt.axis(ax)

    mx = 1e-3
    plt.subplot(2,3,4)
    dimshow(psfimg - img, vmin=-mx, vmax=mx)
    plt.colorbar()

    plt.subplot(2,3,5)
    dimshow(psfimg2 - img, vmin=-mx, vmax=mx)
    plt.colorbar()
    
    plt.subplot(2,3,6)
    plt.hist((psfimg - img).ravel(), 50, log=True, range=(-mx,mx),
             histtype='step', color='b', lw=2, alpha=0.5)
    plt.hist((psfimg2 - img).ravel(), 50, log=True, range=(-mx,mx),
             histtype='step', color='r')
    plt.yticks([])
    plt.xlim(-mx,mx)
    

if __name__ == '__main__':
    ps = PlotSequence('test-em')

    truepsf = GaussianMixturePSF(np.array([1.]),
                                 np.array([[0.1, 0.3]]),
                                 np.array([[[4.,0.5],[0.5,4.0]]]))
    truepsf.radius = 10
    img = truepsf.getPointSourcePatch(0., 0.)
    img = img.patch

    fsa = dict(v2=True, approx=1e-8, N=1)
    
    fit1 = GaussianMixturePSF.fromStamp(img, **fsa)

    print 'truth:', truepsf
    print 'fit1 :', fit1

    for mu,sigma in [#(0, 1e-6), (1e-6,1e-6), (1e-5,1e-6),
                     (0.,1e-4), #(1e-4,1e-4), (2e-4,1e-4), (-1e-4,1e-4), (-2e-4,1e-4),
                     (0., 1e-3),
                     (0.,1e-2),
        # (1e-2, 1e-2), #(0, 2e-2)
        # (-1e-2, 1e-2),
                     ]:
        print
        print 'Fit with noise mu,sigma', mu,sigma
        h,w = img.shape
        plt.clf()
        plt.subplot(2,1,1)
        fitparams = []
        for iters in range(50):
            noise = np.random.normal(size=img.shape) * sigma + mu
            plt.plot(img[h/2,:] + noise[h/2,:], 'g-', alpha=0.1)
            fit2,sky = GaussianMixturePSF.fromStamp(img + noise, **fsa)
            fit2.radius = truepsf.radius
            print 'fit sky=', sky
            print fit2
            fitparams.append(fit2.getParams())
            
            fitimg = fit2.getPointSourcePatch(0.,0.)
            fitimg = fitimg.patch
            plt.plot(fitimg[h/2,:], 'r-', alpha=0.2)

        plt.plot(img[h/2,:], 'b-', lw=2)
        plt.suptitle('Noise mu=%.2g, sigma=%.2g' % (mu,sigma))
        #plt.yscale('symlog', linthreshy=sigma)
        fitparams = np.array(fitparams)
        plt.subplot(2,1,2)
        iparam = -3
        plt.hist(fitparams[:,iparam], 10)
        trueval = truepsf.getParams()[iparam]
        xl,xh = plt.xlim()
        mx = max(np.abs(xl-trueval), np.abs(xh-trueval))
        plt.xlim(trueval-mx, trueval+mx)
        plt.axvline(trueval, color='r')
        plt.xlabel(truepsf.getParamNames()[iparam])
        ps.savefig()
            
    import sys    
    sys.exit(0)
    
    



    
    fn = os.path.join(os.path.dirname(__file__),
                      'c4d_140818_002108_ooi_z_v1.ext27.psf')
    psf = PsfEx(fn, 2048, 4096)

    nrounds = 1
    
    psfimg = psf.instantiateAt(100,100)
    print 'psfimg sum', psfimg.sum()
    print '  ', psfimg.min(), psfimg.max()
    plt.clf()
    dimshow(np.log10(psfimg + 1e-3))
    ps.savefig()

    from tractor.fitpsf import em_init_params
    N = 3
    w,mu,var = em_init_params(N, None, None, None)
    print 'w,mu,var', w.shape,mu.shape,var.shape
    
    ph,pw = psfimg.shape
    parm = np.hstack([[w[i], mu[i,1] + ph/2, mu[i,0] + pw/2,
                       var[i,1,1],var[i,0,1],var[i,0,0]]
                       for i in range(len(w))])
    print 'parm', parm

    t0 = Time()
    for i in range(nrounds):
        imx,sky = prep_image(psfimg)
        ob = Observation(imx)
        mix = GMixEM(ob)
        start = GMix(pars=parm)
        mix.run_em(start, sky, maxiter=10000)
        res = mix.get_gmix()
    print 'Sheldon:', Time()-t0
    print 'Result:', res
    parm = res.get_full_pars()
    print 'Params:', parm
    meta = mix.get_result()
    print 'Sky:', meta['psky']
    w = np.array(parm[0::6])
    w /= w.sum()
    mu = np.array([parm[2::6]-pw/2, parm[1::6]-ph/2]).T
    var = np.array([ [[cc, rc], [rc,rr]]
                    for rr,rc,cc in zip(parm[3::6], parm[4::6], parm[5::6])])
    #print 'w,mu,var', w,mu,var
    #print w.shape, mu.shape, var.shape
    espsf = GaussianMixturePSF(w, mu, var)
    print 'ES psf:', espsf

    plot_result(espsf, psfimg)
    plt.suptitle('Sheldon')
    ps.savefig()



    if False:
        # Pull the minimum pixel lower and lower, and see what happens.
        psfimgx = psfimg.copy()
        psfimgx[1,1] = psfimgx.min()
        print 'Min:', psfimgx.min()
        for i in range(10):
            print
            print 'Round', i
            psfimgx[1,1] *= 2.
            print 'Min:', psfimgx.min()
            imx,sky = prep_image(psfimgx)
            print 'Sky:', sky
            ob = Observation(imx)
            mix = GMixEM(ob)
            start = GMix(pars=parm)
            mix.run_em(start, sky, maxiter=10000)
            res = mix.get_gmix()
            print 'Result:', res
            parm = res.get_full_pars()
            print 'Params:', parm
            meta = mix.get_result()
            print 'Sky:', meta['psky']
            w = np.array(parm[0::6])
            w /= w.sum()
            mu = np.array([parm[2::6]-pw/2, parm[1::6]-ph/2]).T
            var = np.array([ [[cc, rc], [rc,rr]]
                             for rr,rc,cc in zip(parm[3::6], parm[4::6], parm[5::6])])
            espsf = GaussianMixturePSF(w, mu, var)
            print 'ES psf:', espsf
            plot_result(espsf, psfimgx)
            plt.suptitle('Sheldon: round %i' % i)
            ps.savefig()
    
        sys.exit(0)







    
    fsa = dict(emsteps=1000)
    
    t0 = Time()
    for i in range(nrounds):
        gpsf1 = GaussianMixturePSF.fromStamp(psfimg, **fsa)
    print 'fromStamp:', Time()-t0
    print gpsf1

    plot_result(gpsf1, psfimg)
    plt.suptitle('fromStamp (orig)')
    ps.savefig()

    if False:
        t0 = Time()
        for i in range(nrounds):
            gpsf2 = GaussianMixturePSF.fromStamp(psfimg, v2=True, **fsa)
        print 'fromStamp (v2):', Time()-t0
        print gpsf2
    
        plot_result(gpsf2, psfimg)
        plt.suptitle('fromStamp (v2)')
        ps.savefig()
        
        t0 = Time()
        for i in range(nrounds):
            gpsf3 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-8, **fsa)
        print 'fromStamp (v3):', Time()-t0
        print gpsf3

        plot_result(gpsf3, psfimg)
        plt.suptitle('fromStamp (v2, 1e-8)')
        ps.savefig()

    t0 = Time()
    for i in range(nrounds):
        gpsf4,sky = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-6, **fsa)
    print 'fromStamp (v4):', Time()-t0
    print gpsf4
    plot_result(gpsf4, psfimg)
    plt.suptitle('fromStamp (v2, 1e-6)')
    ps.savefig()

    import sys
    sys.exit(0)
    
    t0 = Time()
    for i in range(nrounds):
        gpsf5 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-4, **fsa)
    print 'fromStamp (v5):', Time()-t0
    print gpsf5
    plot_result(gpsf5, psfimg)
    plt.suptitle('fromStamp (v2, 1e-4)')
    ps.savefig()

    
    if False:
        # non-positive determinant!
        t0 = Time()
        for i in range(nrounds):
            gpsf5 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-6,
                                                 clamp=False, **fsa)
        print 'fromStamp (v5):', Time()-t0
        print gpsf5
        plot_result(gpsf5, psfimg)
        plt.suptitle('fromStamp (v5, 1e-6, no clamp)')
        ps.savefig()

    
    t0 = Time()
    #w,mu,var = em_init_params(N, None, None, None)
    #thepsf = GaussianMixturePSF(w.copy(), mu.copy(), var.copy())

    thepsf = gpsf4.copy()
    psftim = Image(data=psfimg, invvar=np.zeros(psfimg.shape)+1e6,
                   psf=thepsf)
    ph,pw = psfimg.shape
    psftractor = Tractor([psftim], [PointSource(PixPos(pw/2, ph/2), Flux(1.))])
    psftractor.freezeParam('catalog')
    psftim.freezeAllBut('psf')
    #print 'Optimizing:'
    #psftractor.printThawedParams()

    tpsfs = []
    for step in range(100):
        #print 'Log-prob:', psftractor.getLogLikelihood()
        #tpsfs.append(psftim.psf.copy())
        dlnp,X,alpha = psftractor.optimize(priors=False, shared_params=False,
                                           damp=0.1, alphas=[0.1, 0.3, 1.0, 2.0])
        print 'dlnp:', dlnp
        if dlnp < 1e-6:
            break
    print 'Tractor fit PSF:'
    print thepsf
    print 'tim PSF fitting via Tractor:', Time()-t0

    plot_result(thepsf, psfimg)
    plt.suptitle('Tractor')
    ps.savefig()
    
    
    
    #for gpsf in [espsf, gpsf1, gpsf2, gpsf3, gpsf4, thepsf] + tpsfs:
    # plt.clf()
    # for i,psf in enumerate([gpsf, gpsf2]):
    #     plt.subplot(1,2,i+1)
    #     patch = psf.getPointSourcePatch(0., 0.)
    #     #dimshow(patch.patch)
    #     dimshow(np.log10(patch.patch + 1e-3))
    # ps.savefig()

    
