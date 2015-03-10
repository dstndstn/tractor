#! /bin/bash

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

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

brick="$1"

# $SCRATCH/dr1
outdir=/scratch1/scratchdirs/dstn/dr1
mkdir -p $outdir/logs

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" > $outdir/logs/$brick.log

python -u projects/desi/runbrick.py --force-all --no-write --stage writecat --brick $brick --outdir $outdir --threads 6 >> $outdir/logs/$brick.log 2>&1

# Edison: 6 threads per job, 4 jobs per node = 24 cores.
# here we ask for 4 nodes * 4 jobs = 16 jobs total.
# qdo launch bricks 16 --mpack 6 --batchopts "-A desi" --walltime=4:00:00 --script projects/desi/pipebrick.sh --batchqueue regular --verbose


# with 8 threads: 3 GB per core * 8 cores (most of the carver nodes have 24 GB)
# qdo launch bricks 1 --batchopts "-l pvmem=3GB -l nodes=1:ppn=8 -A desi -t 1-20 -q regular" --walltime=48:00:00 --script projects/desi/pipebrick.sh

# qdo launch bricks 1 --batchopts "-l pvmem=6GB -A cosmo -t 1-20 -q serial" --walltime=24:00:00 --script projects/desi/pipebrick.sh --verbose

# or maybe
#qdo launch bricks 4 --mpack 2 --batchopts "-l pvmem=6GB -A cosmo -t 1-2" --walltime=24:00:00 --script projects/desi/pipebrick.sh --batchqueue regular

# Why did they fail?
# tail $(qdo tasks bricks --state=Failed | tail -n +2 | awk '{printf("pipebrick-logs/%s.log ",$2)}')
