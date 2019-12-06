import os
from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy as np
numpy_inc = [np.get_include()]

kwargs = {}
if os.environ.get('CC') == 'icc':
    kwargs.update(extra_compile_args=['-g', '-xhost', '-axMIC-AVX512'],
                  extra_link_args=['-g', '-lsvml'])
else:
    kwargs.update(extra_compile_args=['-g', '-std=c99'],
                  extra_link_args=['-g'])

module_fourier = Extension('tractor._mp_fourier',
                           sources = ['tractor/mp_fourier.i'],
                           include_dirs = numpy_inc,
                           undef_macros=['NDEBUG'],
                           **kwargs)
module_mix = Extension('tractor._mix',
                       sources = ['tractor/mix.i'],
                       include_dirs = numpy_inc,
                       extra_objects = [],
                       undef_macros=['NDEBUG'],
    )
#extra_compile_args=['-O0','-g'],
#extra_link_args=['-O0', '-g'],

module_em = Extension('tractor._emfit',
                      sources = ['tractor/emfit.i' ],
                      include_dirs = numpy_inc,
                      extra_objects = [],
                      undef_macros=['NDEBUG'],
                      )

mods = [module_mix, module_em, module_fourier]

nthreads = 4
comdir2 = dict(language_level=3,
               profile=True)

cymod2 = cythonize(
    ['tractor/galaxy.py',],
    annotate=True,
    compiler_directives=comdir2,
    nthreads=nthreads)

comdir1 = dict(language_level=3,
               infer_types=True,
               profile=True)
cymod1 = cythonize(
    [
        'tractor/patch.pyx',
        #'tractor/galaxy.py',
        #'tractor/patch.py',
        'tractor/basics.py',
        'tractor/brightness.py',
        'tractor/ceres_optimizer.py',
        'tractor/constrained_optimizer.py',
        'tractor/dense_optimizer.py',
        'tractor/ducks.py',
        'tractor/ellipses.py',
        'tractor/engine.py',
        'tractor/image.py',
        'tractor/imageutils.py',
        'tractor/lsqr_optimizer.py',
        'tractor/mixture_profiles.py',
        'tractor/motion.py',
        'tractor/optimize.py',
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
    ],
    annotate=True,
    compiler_directives=comdir1,
    nthreads=nthreads)

# Reach into the distutils.core.Extension objects and set the compiler options...
for ext in cymod1 + cymod2:
    for k,v in kwargs.items():
        setattr(ext, k, v)

setup(
name="tractor",
    version="git",
    packages=['tractor'], #, 'wise'],
    package_dir={'wise':'wise', 'tractor':'tractor'},
    ext_modules = cymod1 + cymod2 + mods,
    )
