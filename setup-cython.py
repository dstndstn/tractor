from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize([
        'tractor/patch.pyx',

        'tractor/galaxy.py',
        'tractor/basics.py',
        'tractor/brightness.py',
        'tractor/ceres_optimizer.py',
        'tractor/ducks.py',
        'tractor/ellipses.py',
        'tractor/engine.py',

        'tractor/sersic.py',
        'tractor/image.py',
        'tractor/imageutils.py',
        'tractor/lsqr_optimizer.py',
        'tractor/mixture_profiles.py',
        'tractor/motion.py',
        'tractor/optimize.py',
        #'tractor/patch.py',
        'tractor/pointsource.py',
        'tractor/psf.py',
        'tractor/psfex.py',
        'tractor/sersic.py',
        'tractor/sfd.py',
        'tractor/shifted.py',
        'tractor/sky.py',
        'tractor/splinesky.py',
        'tractor/tractortime.py',
        'tractor/utils.py',
        'tractor/wcs.py',
        ], annotate=True, compiler_directives=dict(language_level=3,
#infer_types=True,
#profile=True
                                                   ))
    )
