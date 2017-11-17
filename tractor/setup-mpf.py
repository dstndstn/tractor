from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
import os

numpy_inc = get_numpy_include_dirs()

kwargs = {}
if os.environ.get('CC') == 'icc':
    kwargs.update(extra_compile_args=['-g', '-xhost', '-axMIC-AVX512'],
                  extra_link_args=['-g', '-lsvml'])
else:
    kwargs.update(extra_compile_args=['-g', '-std=c99'],
                  extra_link_args=['-g'])

mpf_module = Extension('_mp_fourier',
                       sources = ['mp_fourier.i' ],
                       include_dirs = numpy_inc,
                       **kwargs)

setup(name = 'Gaussian mixtures -- Fourier transform',
	  version = '1.0',
	  description = '',
	  author = 'Lang & Hogg',
	  author_email = 'dstndstn@gmail.com',
	  url = 'http://astrometry.net',
	  ext_modules = [mpf_module])
