#! /bin/bash

export PYTHONPATH=${PYTHONPATH}:.

# Force MKL single-threaded
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

outdir=$SCRATCH/dr1j

brick="$1"

logdir=$(echo $brick | head -c 3)
mkdir -p $outdir/logs/$logdir
log="$outdir/logs/$logdir/$brick.log"

echo Logging to: $log
echo Running on ${NERSC_HOST} $(hostname)

echo -e "\n\n\n\n\n\n\n\n\n\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log
echo "PWD: $(pwd)" >> $log
echo "Modules:" >> $log
module list >> $log 2>&1
echo >> $log
echo "Environment:" >> $log
set >> $log
echo >> $log
ulimit -a >> $log
echo >> $log

echo -e "\nStarting on ${NERSC_HOST} $(hostname)\n" >> $log
echo "-----------------------------------------------------------------------------------------" >> $log

echo "Astrometry.net path:"
python -c "import astrometry; print astrometry.__file__"

python projects/desi/runbrick.py --force-all --no-write --brick $brick --outdir $outdir --threads 12 --nsigma 6 --skip --pipe >> $log 2>&1

# dr1j:
# qdo launch edr 32 --mpack 3 --walltime=48:00:00 --script projects/desi/pipebrick.sh --batchqueue regular --verbose

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
