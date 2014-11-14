#! /bin/bash

echo "PWD: $(pwd)"
export PYTHONPATH=${PYTHONPATH}:.
echo "Modules:"
module list 2>&1
echo

echo "Environment:"
set

# echo MKL $MKL
# echo MKL_HOME $MKL_HOME
# echo LD_LIBRARY_PATH $LD_LIBRARY_PATH
# 
# echo ldd:
# ldd tractor/_ceres.so
# 
# python -c "from tractor.ceres import *"

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

brick="$1"

#python -u projects/desi/tunebrick.py -n -b $brick > tunebrick-logs/$brick.log 2>&1
python -u projects/desi/tunebrick.py -n -b $brick -s recoadd > tunebrick-logs/$brick-recoadd.log 2>&1

# qdo launch bricks 1 --batchopts "-l pvmem=10GB -t 1-20" --batchqueue serial --walltime=24:00:00 --script projects/desi/tunebrick.sh
