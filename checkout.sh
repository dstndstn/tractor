#! /bin/bash
set -e

mkdir tractor
cd tractor

svn co http://astrometry.net/svn/trunk/projects/tractor .
svn co -N http://astrometry.net/svn/trunk/src/astrometry
(cd astrometry && svn up util)
(cd astrometry && svn up libkd)
(cd astrometry && svn up qfits-an)
(cd astrometry && svn up gsl-an)
(cd astrometry && svn up sdss)

(cd astrometry && make pyutil)
make

