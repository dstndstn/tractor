import os

from astrometry.util.plotutils import *
from astrometry.util.ttime import *
import matplotlib
matplotlib.use('Agg')
import pylab as plt

from tractor.psfex import *
from tractor import *

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

    t0 = Time()
    for i in range(10):
        gpsf = GaussianMixturePSF.fromStamp(psfimg)
    print 'fromStamp:', Time()-t0

    t0 = Time()
    for i in range(10):
        gpsf2 = GaussianMixturePSF.fromStamp(psfimg, v2=True)
    print 'fromStamp (v2):', Time()-t0

    plt.clf()
    for i,psf in enumerate([gpsf, gpsf2]):
        plt.subplot(1,2,i+1)
        patch = psf.getPointSourcePatch(0., 0.)
        #dimshow(patch.patch)
        dimshow(np.log10(patch.patch + 1e-3))
    ps.savefig()
