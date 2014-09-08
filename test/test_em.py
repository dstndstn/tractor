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

if __name__ == '__main__':
    fn = os.path.join(os.path.dirname(__file__),
                      'c4d_140818_002108_ooi_z_v1.ext27.psf')
    psf = PsfEx(fn, 2048, 4096)

    ps = PlotSequence('test-em')
    
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
    for i in range(10):
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
    w = np.array(parm[0::6])
    w /= w.sum()
    mu = np.array([parm[2::6]-pw/2, parm[1::6]-ph/2]).T
    var = np.array([ [[cc, rc], [rc,rr]]
                    for rr,rc,cc in zip(parm[3::6], parm[4::6], parm[5::6])])
    #print 'w,mu,var', w,mu,var
    #print w.shape, mu.shape, var.shape
    espsf = GaussianMixturePSF(w, mu, var)
    print 'ES psf:', espsf

    fsa = dict(emsteps=1000)
    
    t0 = Time()
    for i in range(10):
        gpsf1 = GaussianMixturePSF.fromStamp(psfimg, **fsa)
    print 'fromStamp:', Time()-t0
    print gpsf1

    t0 = Time()
    for i in range(10):
        gpsf2 = GaussianMixturePSF.fromStamp(psfimg, v2=True, **fsa)
    print 'fromStamp (v2):', Time()-t0
    print gpsf2

    t0 = Time()
    for i in range(10):
        gpsf3 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-8, **fsa)
    print 'fromStamp (v3):', Time()-t0
    print gpsf3

    t0 = Time()
    for i in range(10):
        gpsf4 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-6, **fsa)
    print 'fromStamp (v4):', Time()-t0
    print gpsf4

    t0 = Time()
    w,mu,var = em_init_params(N, None, None, None)
    thepsf = GaussianMixturePSF(w.copy(), mu.copy(), var.copy())
    psftim = Image(data=psfimg, invvar=np.zeros(psfimg.shape)+1e4,
                   psf=thepsf)
    ph,pw = psfimg.shape
    psftractor = Tractor([psftim], [PointSource(PixPos(pw/2, ph/2), Flux(1.))])
    psftractor.freezeParam('catalog')
    psftim.freezeAllBut('psf')
    print 'Optimizing:'
    psftractor.printThawedParams()

    tpsfs = []
    for step in range(100):
        tpsfs.append(psftim.psf.copy())
        dlnp,X,alpha = psftractor.optimize(priors=False, shared_params=False,
                                           damp=0.1, alphas=[0.1, 0.3, 1.0, 2.0])
        print 'dlnp:', dlnp
        if dlnp < 1e-6:
            break
    print 'Tractor fit PSF:'
    print thepsf
    print 'tim PSF fitting via Tractor:', Time()-t0

    

    for gpsf in [espsf, gpsf1, gpsf2, gpsf3, gpsf4, thepsf] + tpsfs:
        plt.clf()
        plt.subplot(2,2,1)
        mx = np.max(np.log10(psfimg + 1e-3))
        mn = np.min(np.log10(psfimg + 1e-3))
        dimshow(np.log10(psfimg + 1e-3), vmin=mn, vmax=mx)
        plt.subplot(2,2,2)
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

        plt.subplot(2,2,3)
        mx = 1e-3
        dimshow(psfimg - img, vmin=-mx, vmax=mx)
        plt.colorbar()
        plt.subplot(2,2,4)
        plt.hist((psfimg - img).ravel(), 50, log=True, range=(-mx,mx))
        plt.yticks([])
        plt.xlim(-mx,mx)
        ps.savefig()
    

    
    plt.clf()
    for i,psf in enumerate([gpsf, gpsf2]):
        plt.subplot(1,2,i+1)
        patch = psf.getPointSourcePatch(0., 0.)
        #dimshow(patch.patch)
        dimshow(np.log10(patch.patch + 1e-3))
    ps.savefig()

    
