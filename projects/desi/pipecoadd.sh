#! /bin/bash

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

# $SCRATCH/dr1
#outdir=/scratch1/scratchdirs/dstn/dr1b
outdir=$GSCRATCH/dr1b

mkdir -p $outdir/logs
brick="$1"
log="$outdir/logs/co-$brick.log"

echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n\n\n\n\n\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo -e "\n\n\n\n\n\n\n\n\n\n" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set >> $log
echo >> $log

echo >> $log
ulimit -a >> $log
echo >> $log


echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

python -u projects/desi/runbrick.py --force-all --no-write --brick $brick --outdir $outdir --skip-coadd --stage image_coadds >> $log 2>&1

rtnval=$?

echo "runbrick.py exited with status $rtnval" >> $log
exit $rtnval
