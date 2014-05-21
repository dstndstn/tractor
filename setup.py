from distutils.core import setup, Extension
from distutils.command.build_ext import *
from distutils.dist import Distribution

from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()

import sys

eigen_inc = os.environ.get('EIGEN_INC', None)
if eigen_inc is None:
    eigen_inc = check_output('pkg-config --cflags eigen3')

ceres_inc = os.environ.get('CERES_INC', None)

ceres_lib = os.environ.get('CERES_LIB', None)

inc = [eigen_inc]
if ceres_inc is not None:
    inc.append(ceres_inc)

link = []
if ceres_lib is not None:
    link.append(ceres_lib)

module_ceres = Extension('tractor._ceres',
                         sources=['tractor/ceres-tractor.cc', 'tractor/ceres.i'],
                         include_dirs = numpy_inc,
                         extra_compile_args = inc,
                         extra_link_args = link,
                         language = 'c++',
                         swig_opts=['-c++'],
                         )

module_mix = Extension('tractor._mix',
                              sources = ['tractor/mix_wrap.c' ],
                              include_dirs = numpy_inc,
                              extra_objects = [],
                              undef_macros=['NDEBUG'],
                              #extra_compile_args=['-O0','-g'],
                              #extra_link_args=['-O0', '-g'],
                              )

module_em = Extension('tractor._emfit',
                             sources = ['tractor/emfit_wrap.c' ],
                             include_dirs = numpy_inc,
                             extra_objects = [],
                             undef_macros=['NDEBUG'],
                             #extra_compile_args=['-O0','-g'],
                             #extra_link_args=['-O0', '-g'],
                             )

class MyDistribution(Distribution):
    display_options = Distribution.display_options + [
        ('with-ceres', None, 'build Ceres module?'),
        ]


## Distutils is so awkward to work with that THIS is the easiest way to add
# an extra command-line arg!

mods = [module_mix, module_em]
key = '--with-ceres'
if key in sys.argv:
    sys.argv.remove(key)
    mods.append(module_ceres)


setup(
    distclass=MyDistribution,
    name="the Tractor",
    version="git",
    author="Dustin Lang (CMU) and David W. Hogg (NYU)",
    author_email="dstn@cmu.edu",
    packages=["tractor"],
    ext_modules = mods,
    url="http://theTractor.org/",
    license="GPLv2",
    description="probabilistic astronomical image analysis",
    long_description="Attempt at replacing heuristic astronomical catalogs with models built with specified likelihoods, priors, and utilities.",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
