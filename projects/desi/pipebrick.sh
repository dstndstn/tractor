#! /bin/bash

export PYTHONPATH=${PYTHONPATH}:.

echo "PWD: $(pwd)"
echo "Modules:"
module list 2>&1
echo
echo "Environment:"
set
echo

echo
ulimit -a
echo

# 
# echo MKL $MKL
# echo MKL_HOME $MKL_HOME
# echo LD_LIBRARY_PATH $LD_LIBRARY_PATH
# echo ldd:
# ldd tractor/_ceres.so
# python -c "from tractor.ceres import *"

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

brick="$1"

outdir=edr4
mkdir -p $outdir/logs

#python -u projects/desi/runbrick.py --force-all --no-write --stage writecat --brick $brick --outdir $outdir > $outdir/logs/$brick.log 2>&1

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" > $outdir/logs/$brick.log

python -u projects/desi/runbrick.py --force-all --no-write --stage writecat --brick $brick --outdir $outdir --threads 8 >> $outdir/logs/$brick.log 2>&1

# Try 8 threads on edison nodes (packing 3 of those per 24-core node)?
#qdo launch bricks 3 --mpack 8 --batchopts "-A desi -t 1-10" --walltime=24:00:00 --script projects/desi/pipebrick-edison.sh --batchqueue regular

# with 8 threads: 3 GB per core * 8 cores (most of the carver nodes have 24 GB)
# qdo launch bricks 1 --batchopts "-l pvmem=3GB -l nodes=1:ppn=8 -A desi -t 1-20 -q regular" --walltime=48:00:00 --script projects/desi/pipebrick.sh

# qdo launch bricks 1 --batchopts "-l pvmem=6GB -A cosmo -t 1-20 -q serial" --walltime=24:00:00 --script projects/desi/pipebrick.sh --verbose

# or maybe
#qdo launch bricks 4 --mpack 2 --batchopts "-l pvmem=6GB -A cosmo -t 1-2" --walltime=24:00:00 --script projects/desi/pipebrick.sh --batchqueue regular

# Why did they fail?
# tail $(qdo tasks bricks --state=Failed | tail -n +2 | awk '{printf("pipebrick-logs/%s.log ",$2)}')
