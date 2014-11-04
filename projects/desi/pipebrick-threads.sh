#! /bin/bash

echo "PWD: $(pwd)"
export PYTHONPATH=${PYTHONPATH}:.
echo "Modules:"
module list 2>&1
echo

# We do our own multi-threaded, so don't let MKL use OpenMP multi-threading
# https://software.intel.com/en-us/articles/using-threaded-intel-mkl-in-multi-thread-application
export MKL_NUM_THREADS=1

brick="$1"
python -u projects/desi/pipebrick.py --threads 8 $brick > pipebrick-logs/$brick.log 2>&1


# qdo launch bricks 1 --batchopts "-A cosmo -t 1-10" --walltime=4:00:00 --batchqueue regular --script projects/desi/pipebrick-threads.sh

#  > qdo launch bricks 2 --mpack 4 --batchopts "-A cosmo -t 1-10 -q regular" --walltime=4:00:00 --script projects/desi/pipebrick.sh --verbose
#  Cores per node: 8                                                                                                                         
#  Cores per NUMA node: 8                                                                                                                    
#  Workers per node: 2                                                                                                                       
#  Workers per NUMA node: 2                                                                                                                  
#  Nodes: 1                                                                                                                                  
#  MPIRUN: mpirun -np 2 -npernode 2 -report-bindings                                                                                         
#  qsub -N bricks -V -j oe -l nodes=1:ppn=2 -A cosmo -t 1-10 -q regular -l walltime=4:00:00 /global/homes/d/dstn/qdo/bin/qdojob              
#
#  qalter 10521298[] "-l pvmem=10GB"
