from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

numpy_inc = get_numpy_include_dirs()

c_swig_module_mix = Extension('tractor._mix',
                              sources = ['tractor/mix_wrap.c' ],
                              include_dirs = numpy_inc,
                              extra_objects = [],
                              undef_macros=['NDEBUG'],
                              #extra_compile_args=['-O0','-g'],
                              #extra_link_args=['-O0', '-g'],
                              )

c_swig_module_em = Extension('tractor._emfit',
                             sources = ['tractor/emfit_wrap.c' ],
                             include_dirs = numpy_inc,
                             extra_objects = [],
                             undef_macros=['NDEBUG'],
                             #extra_compile_args=['-O0','-g'],
                             #extra_link_args=['-O0', '-g'],
                             )

setup(
    name="the Tractor",
    version="n/a",
    author="Dustin Lang (CMU) and David W. Hogg (NYU)",
    author_email="dstn@cmu.edu",
    packages=["tractor"],
    ext_modules = [c_swig_module_mix, c_swig_module_em],
    url="http://theTractor.org/",
    license="GPLv2",
    description="probabilistic astronomical image analysis",
    long_description="Our first attempt at replacing heuristic astronomical catalogs with models built with specified likelihoods, priors, and utilities.",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
