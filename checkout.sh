#! /bin/bash
set -e

mkdir tractor
cd tractor
pwd

git clone https://github.com/dstndstn/tractor.git .

wget "http://astrometry.net/downloads/astrometry.net-latest.tar.gz"
tar xzf astrometry.net-latest.tar.gz
mv astrometry.net-?.?? astrometry
(cd astrometry && make py)

make

echo 'Setting up FITSIO:'
echo '  mkdir fitsio-git'
echo '  git clone https://github.com/esheldon/fitsio.git fitsio-git/src'
echo '  (P=$(pwd) cd fitsio-git/src && python setup.py install --home=$P/fitsio-git)'
echo '  mv fitsio-git/lib*/python/fitsio .'
