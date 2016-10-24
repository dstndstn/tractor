from __future__ import print_function
if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')

import numpy as np

from tractor import *
from tractor import sdss as st
from tractor.cache import *
from astrometry.util.ttime import *
import gc
#gc.set_debug(gc.DEBUG_LEAK)
#gc.enable()

run,camcol,field = 3384, 4, 198
bandname = 'r'

timg,info = st.get_tractor_image(run, camcol, field, bandname, useMags=True)
sources = st.get_tractor_sources(run, camcol, field,bandname, bands=[bandname])

# mags = []
# for i,src in enumerate(sources):
# 	m = getattr(src.getBrightness(), bandname)
# 	mags.append(m)
# J = np.argsort(mags)

#sources = [sources[J[i]] for i in range(10)]

sources = sources[:20]

tractor = Tractor([timg], sources)
tractor.cache = Cache(maxsize=100)

for step in range(10):
	tractor.getLogProb()
	print('After getLogProb() number', (step+1))
	print(tractor.cache)
	print('Items:')
	tractor.cache.printItems()
	print()

print('Before clearing cache...')
print(tractor.cache)
memusage()

tractor.cache.clear()

gc.collect()

print('After clearing cache...')
print(tractor.cache)
memusage()

#print 'garbage:'
#for x in gc.garbage:
#	print x
