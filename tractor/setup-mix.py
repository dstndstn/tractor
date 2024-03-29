from distutils.core import setup, Extension
import numpy

numpy_inc = [numpy.get_include()]

#sources = ['mix_wrap.c' ],
c_swig_module = Extension('_mix',
                          sources=['mix.i'],
                          include_dirs=numpy_inc,
                          extra_objects=[],
                          )
# undef_macros=['NDEBUG'],
# extra_compile_args=['-O0','-g'],
#extra_link_args=['-O0', '-g'],


setup(name='Gaussian mixtures',
      version='1.0',
      description='',
      author='Lang & Hogg',
      author_email='dstndstn@gmail.com',
      url='http://thetractor.org',
      py_modules=[],
      ext_modules=[c_swig_module])
