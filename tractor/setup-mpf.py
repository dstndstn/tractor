from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
import os

numpy_inc = get_numpy_include_dirs()

cflags = os.environ.get('CFLAGS')

kwargs = {}
if os.environ.get('CC') == 'icc':
    if cflags is None:
        cflags = '-g -xhost -axMIC-AVX512'
    else:
        print('mp_fourier: using user-specified CFLAGS')
    cflags = cflags.split()
    kwargs.update(extra_compile_args=cflags,
                  extra_link_args=['-g', '-lsvml'])
else:
    cflags = cflags.split()
    kwargs.update(extra_compile_args=cflags,
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
