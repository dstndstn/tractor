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
        mix.run_em(start, sky, maxiter=10000, tol=1e-8)
        res = mix.get_gmix()
    print 'Sheldon:', Time()-t0
    print 'Result:', res
    parm = res.get_full_pars()
    w = np.array(parm[0::6])
    w /= w.sum()
    mu = np.array([parm[2::6]-pw/2, parm[1::6]-ph/2]).T
    var = np.array([ [[cc, rc], [rc,cc]]
                    for rr,rc,cc in zip(parm[3::6], parm[4::6], parm[5::6])])
    #print 'w,mu,var', w,mu,var
    #print w.shape, mu.shape, var.shape
    espsf = GaussianMixturePSF(w, mu, var)
    print 'ES psf:', espsf

    t0 = Time()
    for i in range(10):
        gpsf = GaussianMixturePSF.fromStamp(psfimg)
    print 'fromStamp:', Time()-t0
    print gpsf

    t0 = Time()
    for i in range(10):
        gpsf2 = GaussianMixturePSF.fromStamp(psfimg, v2=True)
    print 'fromStamp (v2):', Time()-t0
    print gpsf2

    t0 = Time()
    for i in range(10):
        gpsf3 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-8)
    print 'fromStamp (v3):', Time()-t0
    print gpsf3

    t0 = Time()
    for i in range(10):
        gpsf4 = GaussianMixturePSF.fromStamp(psfimg, v2=True, approx=1e-6)
    print 'fromStamp (v4):', Time()-t0
    print gpsf4
    
    for gpsfi in [espsf, gpsf, gpsf2, gpsf3, gpsf4]:
        plt.clf()
        plt.subplot(1,2,1)
        gpsfi.radius = pw/2
        img = gpsfi.getPointSourcePatch(0.,0.).patch
        dimshow(np.log10(img + 1e-3))
        plt.subplot(1,2,2)
        dimshow(psfimg - img)
        plt.colorbar()
        ps.savefig()
    

    
    plt.clf()
    for i,psf in enumerate([gpsf, gpsf2]):
        plt.subplot(1,2,i+1)
        patch = psf.getPointSourcePatch(0., 0.)
        #dimshow(patch.patch)
        dimshow(np.log10(patch.patch + 1e-3))
    ps.savefig()

    
