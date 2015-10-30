from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        'tractor/engine.py',
        'tractor/basics.py',
        'tractor/utils.py',
        'tractor/patch.py',
        'tractor/mixture_profiles.py',
        'tractor/ellipses.py',
        'tractor/galaxy.py',
        'tractor/cache.py',
        'tractor/psfex.py',
        'tractor/brightness.py',
        # 'tractor/ceres.py',
        'tractor/ceres_optimizer.py',
        'tractor/ducks.py',
        'tractor/image.py',
        'tractor/lsqr_optimizer.py',
        # 'tractor/motion.py',
        'tractor/optimize.py',
        # 'tractor/ordereddict.py',
        'tractor/pointsource.py',
        'tractor/psf.py',
        # 'tractor/sdss.py',
        # 'tractor/sersic.py',
        'tractor/shifted.py',
        'tractor/sky.py',
        'tractor/splinesky.py',
        # 'tractor/total_ordering.py',
        'tractor/tractortime.py',
        'tractor/wcs.py',
        ])
)

