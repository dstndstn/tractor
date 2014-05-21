from distutils.core import setup, Extension
from distutils.command.build_ext import *

from numpy.distutils.misc_util import get_numpy_include_dirs
numpy_inc = get_numpy_include_dirs()

import os
from subprocess import check_output

#print 'bools:', dir(build_ext)
#print build_ext.boolean_options

# pymod_lib = os.environ.get('PYMOD_LIB', None)
# if pymod_lib is None:
#     pymod_lib = ('-L' + check_output('python-config --prefix') + '/lib' +
#                  ' ' + check_output('python-config --libs'))
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

class mybuild(build_ext):

    #boolean_options = build_ext.boolean_options + ['with-ceres']
    user_options = build_ext.user_options + [('with-ceres', None,
                                              'Build Ceres module'),]

    def __init__(self, *args, **kwargs):
        print 'mybuild(), args', args, 'kwargs', kwargs
        #super(mybuild,self).__init__(*args, **kwargs)
        build_ext.__init__(self, *args, **kwargs)

    def initialize_options(self):
        #super(mybuild, self).initialize_options()
        build_ext.initialize_options(self)
        self.with_ceres = False

    #def finalize_options(self):
    #if self.with_ceres:

    def build_extensions(self):
        print 'MyBuild: build_extensions', self.extensions
        self.check_extensions_list(self.extensions)
        #super(mybuild, self).build_extensions()
        #build_ext.build_extensions(self)
        print 'With Ceres:', self.with_ceres
        for ext in self.extensions:
            if ext == module_ceres and not self.with_ceres:
                print 'Skipping Ceres'
                continue
            self.build_extension(ext)

    def run(self):
        print 'MyBuild: run()'
        #super(mybuild, self).run()
        build_ext.run(self)


setup(
    #options={'build_ext':{'swig_opts':'-c++'}},
    cmdclass={'build_ext': mybuild},
    name="the Tractor",
    version="n/a",
    author="Dustin Lang (CMU) and David W. Hogg (NYU)",
    author_email="dstn@cmu.edu",
    packages=["tractor"],
    ext_modules = [module_mix, module_em, module_ceres],
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
