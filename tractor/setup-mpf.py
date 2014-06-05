from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

numpy_inc = get_numpy_include_dirs()

c_swig_module = Extension('_mp_fourier',
						  sources = ['mp_fourier_wrap.c' ],
						  include_dirs = numpy_inc,
    )
#extra_objects = [],
#undef_macros=['NDEBUG'],
#extra_compile_args=['-O0','-g'],
#extra_link_args=['-O0', '-g'],


setup(name = 'Gaussian mixtures -- Fourier transform',
	  version = '1.0',
	  description = '',
	  author = 'Lang & Hogg',
	  author_email = 'dstndstn@gmail.com',
	  url = 'http://astrometry.net',
	  ext_modules = [c_swig_module])
#py_modules = [],

