#! /bin/bash

echo "PWD: $(pwd)"
export PYTHONPATH=${PYTHONPATH}:.
echo "Modules:"
module list 2>&1
echo

brick="$1"
python -u projects/desi/pipebrick.py $brick > pipebrick-logs/$brick.log 2>&1

# qdo launch bricks 1 --batchopts "-l pvmem=6GB -A cosmo -t 1-20 -q serial" --walltime=24:00:00 --script projects/desi/pipebrick.sh --verbose


# Why did they fail?

# tail $(qdo tasks bricks --state=Failed | tail -n +2 | awk '{printf("pipebrick-logs/%s.log ",$2)}')
