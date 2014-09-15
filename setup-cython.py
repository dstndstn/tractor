from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(['tractor/engine.py', 'tractor/basics.py',
                             'tractor/utils.py', 'tractor/patch.py',
                             'tractor/mixture_profiles.py',
                             'tractor/ellipses.py',
                             'tractor/galaxy.py',
                             'tractor/cache.py',
                             'tractor/psfex.py',
                             ])
)

