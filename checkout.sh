#! /bin/bash
set -e

mkdir tractor
cd tractor
pwd

git clone https://github.com/dstndstn/tractor.git .
svn co -N http://astrometry.net/svn/trunk/src/astrometry
(cd astrometry && svn up util)
(cd astrometry && svn up libkd)
(cd astrometry && svn up qfits-an)
(cd astrometry && svn up catalogs)
(cd astrometry && svn up gsl-an)
(cd astrometry && svn up sdss)

(cd astrometry && make pyutil)
(cd astrometry/libkd && make pyspherematch)
(cd astrometry/sdss && make)
make

echo 'Setting up FITSIO:'
echo '  mkdir fitsio-git'
echo '  git clone https://github.com/esheldon/fitsio.git fitsio-git/src'
echo '  (P=$(pwd) cd fitsio-git/src && python setup.py install --home=$P/fitsio-git)'
echo '  mv fitsio-git/lib*/python/fitsio .'
