import matplotlib
matplotlib.use('Agg')
import numpy as np

from tractor.sdss_galaxy import get_galaxy_cache, disable_galaxy_cache, set_galaxy_cache_size

import resource
import os
import subprocess
import time
from random import randint


import gc

gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_INSTANCES | gc.DEBUG_OBJECTS | gc.DEBUG_SAVEALL)

from garbagetrack import track

from tractor import Tractor, Images, Catalog
from tractor.cache import *

tractor=None

def memuse():
	#cmd = '/bin/ps -p %i -o rss,size,sz,vsz,command' % os.getpid()
	cmd = '/bin/ps -p %i -o rss,size,sz,vsz h' % os.getpid()
	res = subprocess.check_output(cmd, shell=True)
	#print 'ps says:', res
	mem = [int(x) for x in res.split()]
	mem.append(resource.getrusage(resource.RUSAGE_SELF)[2])
	global test
	if test is not None:
		mem.append(test.totalSize())
		mem.append(len(test))
	else:
		mem.append(0)
		mem.append(0)

	return mem



test = Cache()
mem = []

for j in range(3):
    mem.append(memuse())
    for i in range(80000):
	    test.put(randint(0,1000000),np.zeros(10000))
    mem.append(memuse())
    test.clear()
    mem.append(memuse())
mem.append(memuse())
print mem
