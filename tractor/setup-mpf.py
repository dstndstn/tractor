from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

numpy_inc = get_numpy_include_dirs()

c_swig_module = Extension('_mp_fourier',
						  sources = ['mp_fourier.i' ],
						  include_dirs = numpy_inc,
                          extra_compile_args=['-g'],
                          extra_link_args=['-g'],
    )
my_swig_mod = Extension('_c_mp_fourier',
                        sources=['c_mp_fourier.i'],
                        include_dirs=numpy_inc,
                        extra_compile_args=['-g', '-xhost', '-qopt-report=5'],
                        extra_link_args=['-g', '-lsvml'])

setup(name = 'Gaussian mixtures -- Fourier transform',
	  version = '1.0',
	  description = '',
	  author = 'Lang & Hogg',
	  author_email = 'dstndstn@gmail.com',
	  url = 'http://astrometry.net',
	  ext_modules = [c_swig_module, my_swig_mod])
