import os
import sys
from distutils.core import setup, Extension
from distutils.dist import Distribution

# from http://stackoverflow.com/questions/12491328/python-distutils-not-include-the-swig-generated-module
from distutils.command.build import build
class CustomBuild(build):
    sub_commands = [
        ('build_ext', build.has_ext_modules),
        ('build_py', build.has_pure_modules),
        ('build_clib', build.has_c_libraries),
        ('build_scripts', build.has_scripts),
    ]

def main():
    import numpy as np
    numpy_inc = [np.get_include()]

    if os.environ.get('CC') == 'icc':
        compile_args = ['-g']
        cpp_compile_args = compile_args
        link_args = ['-g', '-lsvml']
    else:
        compile_args = ['-g', '-std=c99']
        cpp_compile_args = ['-g']
        link_args = ['-g']

    kwargs = dict(extra_compile_args=compile_args,
                  extra_link_args=link_args)

    # Swig extensions
    module_fourier = Extension('tractor._mp_fourier',
                               sources = ['tractor/mp_fourier.i'],
                               include_dirs = numpy_inc,
                               undef_macros=['NDEBUG'],
                               swig_opts=['-outdir', 'tractor'],
                               **kwargs)
    module_mix = Extension('tractor._mix',
                           sources = ['tractor/mix.i'],
                           include_dirs = numpy_inc,
                           extra_objects = [],
                           undef_macros=['NDEBUG'],
                           swig_opts=['-outdir', 'tractor'],
                           **kwargs)
    module_em = Extension('tractor._emfit',
                          sources = ['tractor/emfit.i' ],
                          include_dirs = numpy_inc,
                          extra_objects = [],
                          undef_macros=['NDEBUG'],
                          swig_opts=['-outdir', 'tractor'],
                          **kwargs)
    mods = [module_mix, module_em, module_fourier]
    pymods = ['tractor.mix', 'tractor.emfit', 'tractor.mp_fourier']

    class MyDistribution(Distribution):
        display_options = Distribution.display_options + [
            ('with-ceres', None, 'build Ceres module?'),
            ('with-cython', None, 'build using Cython?'),
            ]

    with_ceres = True
    if with_ceres:
        sys.argv.remove(key)

        ## Ceres module
        from subprocess import check_output
        eigen_inc = os.environ.get('EIGEN_INC', None)
        if eigen_inc is None:
            try:
                eigen_inc = check_output(['pkg-config', '--cflags', 'eigen3']).strip()
                # py3
                eigen_inc = eigen_inc.decode()
            except:
                eigen_inc = ''

        import shlex
        inc = shlex.split(eigen_inc)
        ceres_inc = os.environ.get('CERES_INC', None)
        ceres_lib = os.environ.get('CERES_LIB', '-lceres')
        if ceres_inc is not None:
            inc.extend(shlex.split(ceres_inc))
        link = []
        if ceres_lib is not None:
            link.extend(shlex.split(ceres_lib))
        module_ceres = Extension('tractor._ceres',
                                 sources=['tractor/ceres-tractor.cc', 'tractor/ceres.i'],
                                 include_dirs = numpy_inc,
                                 extra_compile_args = cpp_compile_args + inc,
                                 extra_link_args = link_args + link,
                                 language = 'c++',
                                 swig_opts=['-c++', '-outdir', 'tractor'],
                                 )
        mods.append(module_ceres)
        pymods.append('tractor.ceres')

    with_cython = True
    if with_cython:
        sys.argv.remove(key)
        from Cython.Build import cythonize

        nthreads = 4
        comdir2 = dict(language_level=3,
                       profile=True)
        # galaxy.py for some reason can't handle infer_types.
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
        mods = mods + cymod1 + cymod2

    cmd = 'echo "git_version = \'$(git describe)\'" > tractor/gitversion.py && cat tractor/gitversion.py tractor/version_post.py > tractor/version.py'
    print('Running', cmd)
    rtn = os.system(cmd)
    print('Return value:', rtn)
    pymods.append('tractor.version')
    version = '0.0'
    try:
        sys.path.append('.')
        from tractor.version import version as v
        version = v
        print('Set version', version)
    except:
        import traceback
        traceback.print_exc()
        pass

    setup(
        name='tractor',
        author='Dustin Lang (Perimeter Institute) and David W. Hogg (NYU/MPIA/Flatiron)',
        author_email="dstndstn@gmail.com",
        distclass=MyDistribution,
        cmdclass={'build': CustomBuild},
        version=version,
        packages=['tractor', 'wise'],
        package_dir={'wise':'wise', 'tractor':'tractor'},
        package_data={'wise':['wise-psf-avg.fits', 'allsky-atlas.fits']},
        ext_modules=mods,
        py_modules=pymods,
        zip_safe=False,
    )

if __name__ == '__main__':
    main()
