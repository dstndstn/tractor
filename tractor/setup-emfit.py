from distutils.core import setup, Extension
import numpy

numpy_inc = [numpy.get_include()]

c_swig_module = Extension('_emfit',
                          sources=['emfit.i'],
                          include_dirs=numpy_inc,
                          extra_objects=[],
                          undef_macros=['NDEBUG'],
                          # extra_compile_args=['-O0','-g'],
                          #extra_link_args=['-O0', '-g'],
                          )

setup(name='EM fit of Gaussian mixture',
      version='1.0',
      description='',
      author='Lang & Hogg',
      author_email='dstn@astro.princeton.edu',
      url='http://astrometry.net',
      py_modules=[],
      ext_modules=[c_swig_module])
