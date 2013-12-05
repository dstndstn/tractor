if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

from astrometry.util.multiproc import *

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.ttime import *

from sequels import one_tile

def _bounce_one_tile((args, kwargs)):
    return one_tile(*args, **kwargs)

T = fits_table('fordustin.fits', columns=['ra','dec'])
T.about()

T.cosmos_row = np.arange(len(T))
T.objc_flags = np.zeros(len(T), int)
T.nchild = np.zeros(len(T), int)
T.objc_type = np.zeros(len(T), int) + 6
T.fracdev = np.ones((len(T),5), np.float32)

T.phi_dev_deg = np.zeros(len(T), np.float32)
T.phi_exp_deg = np.zeros(len(T), np.float32)
T.theta_dev = np.zeros(len(T), np.float32)
T.theta_exp = np.zeros(len(T), np.float32)
T.treated_as_pointsource = np.ones(len(T), bool)
T.objid = np.zeros(len(T), int)
T.index = np.zeros(len(T), int)
T.run = np.zeros(len(T), int)
T.camcol = np.zeros(len(T), int)
T.field = np.zeros(len(T), int)
T.id = np.zeros(len(T), int)

A = fits_table('cosmos-atlas.fits')

class Duck():
    pass

opt = Duck()
opt.bands = [1,2]
opt.save_wise_output = '%s'
opt.savewise = False
opt.splitrcf = False
opt.ceres = True
opt.ceresblock = 8
opt.errfrac = 0.
opt.blocks = 1
opt.minsig1 = 0.1
opt.minsig2 = 0.1
opt.nonneg = False
opt.pickle2 = False
opt.sky = False
opt.cells = []
opt.bright1 = None
opt.minradius = 2
opt.photoObjsOnly = False
opt.output = 'cosmos-phot-%s.fits'
opt.outdir = '.'
tiledir = 'wise-coadds'
outdir = '.'
tempoutdir = 'phot-temp'
#pobjoutdir = '%s-pobj'
Time.add_measurement(MemMeas)

ps = PlotSequence('cosmos')

mp = multiproc(8)

mp.map(_bounce_one_tile,
       [((tile, opt, False, ps, A, tiledir, tempoutdir), dict(T=T))
        for tile in A])

#for i,tile in enumerate(A):
#    one_tile(tile, opt, False, ps, A, tiledir, tempoutdir, T=T)
    
