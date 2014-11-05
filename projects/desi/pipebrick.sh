#! /bin/bash

echo "PWD: $(pwd)"
export PYTHONPATH=${PYTHONPATH}:.
echo "Modules:"
module list 2>&1
echo

echo "Environment:"
set

echo MKL $MKL
echo MKL_HOME $MKL_HOME
echo LD_LIBRARY_PATH $LD_LIBRARY_PATH

echo ldd:
ldd tractor/_ceres.so

python -c "from tractor.ceres import *"

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

brick="$1"

python -u projects/desi/pipebrick.py --no-ceres $brick > pipebrick-logs/$brick.log 2>&1

#python -u projects/desi/pipebrick.py $brick > pipebrick-logs/$brick.log 2>&1

# qdo launch bricks 1 --batchopts "-l pvmem=6GB -A cosmo -t 1-20 -q serial" --walltime=24:00:00 --script projects/desi/pipebrick.sh --verbose


# Why did they fail?

# tail $(qdo tasks bricks --state=Failed | tail -n +2 | awk '{printf("pipebrick-logs/%s.log ",$2)}')
