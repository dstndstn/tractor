# the Tractor

Probabilistic astronomical source detection & measurement

[![Build status (CircleCI)](https://circleci.com/gh/dstndstn/tractor/tree/main.svg?style=shield)](https://circleci.com/gh/dstndstn/tractor/tree/main)
[![Docs](https://readthedocs.org/projects/thetractor/badge/?version=latest)](http://thetractor.readthedocs.org/en/latest/)
[![Coverage](https://coveralls.io/repos/github/dstndstn/tractor/badge.svg?branch=main)](https://coveralls.io/github/dstndstn/tractor)
[![codecov](https://codecov.io/gh/dstndstn/tractor/branch/main/graph/badge.svg?token=FvbnHgYbxp)](https://codecov.io/gh/dstndstn/tractor)

## authors & license

Copyright 2011-2023 Dustin Lang (Perimeter Institute) & David W. Hogg (NYU/Flatiron)

Licensed under GPLv2; see LICENSE.

## install

First, install the Astrometry.net code.  (https://astrometry.net/downloads or https://github.com/dstndstn/astrometry.net/tags).  You can do this by grabbing the code and using "make", or you can install the python code directly using pip:

    pip install git+https://github.com/dstndstn/astrometry.net.git

Then grab the Tractor code:

    git clone git@github.com:dstndstn/tractor.git
    cd tractor
    make

It is possible to run directly out of the checked-out *tractor*
directory.  But if you want to install it, you can use pip, optionally with flags to enable Ceres Solver (requires the Ceres library), and Cython:

    pip install -v --install-option="--with-ceres" --install-option="--with-cython" .

There is a test script that renders SDSS images:

    python examples/tractor-sdss-synth.py


Prereqs:

* scipy
* numpy
* astrometry.net

Other packages used in various places include:

* fitsio: https://github.com/esheldon/fitsio
* emcee: http://dan.iel.fm/emcee/current

## documentation:

Horribly incomplete, but online at http://thetractor.org/doc and http://thetractor.readthedocs.org/en/latest/

## collaboration:

We are trying to move from a "Research code that only we use" phase to
"Research code that other people can use".  We are happy to hear your
feedback; please feel free to file Issues.  And naturally we will be
pleased to see pull requests!

