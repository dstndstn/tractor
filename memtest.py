# vg0.log:
#
# from tractor.sdss import get_tractor_sources_dr8
# srcs = get_tractor_sources_dr8(3063,4,60,'g')
# del srcs

# LEAK SUMMARY:
#    definitely lost: 5,205 bytes in 127 blocks
#    indirectly lost: 43,006 bytes in 190 blocks
#      possibly lost: 6,182,807 bytes in 30,911 blocks
#    still reachable: 9,293,066 bytes in 58,867 blocks
#         suppressed: 0 bytes in 0 blocks

# vg1.log:
# 
# from tractor.sdss import get_tractor_sources_dr8, get_tractor_image_dr8
# from tractor import Tractor, Images, Catalog
# srcs = get_tractor_sources_dr8(3063,4,60,'g')
# im,inf = get_tractor_image_dr8(3063,4,60,'g', roi=[100,200,100,200])
# del srcs
# del im
# del inf

# LEAK SUMMARY:
#    definitely lost: 5,205 bytes in 127 blocks
#    indirectly lost: 43,006 bytes in 190 blocks
#      possibly lost: 6,201,669 bytes in 30,940 blocks
#    still reachable: 9,312,232 bytes in 59,062 blocks
#         suppressed: 0 bytes in 0 blocks
		

# vg2.log:
# from tractor.sdss import get_tractor_sources_dr8, get_tractor_image_dr8
# from tractor import Tractor, Images, Catalog
# srcs = get_tractor_sources_dr8(3063,4,60,'g')
# im,inf = get_tractor_image_dr8(3063,4,60,'g', roi=[100,200,100,200])
# tractor = Tractor(Images(im), Catalog(*srcs))
# mod = tractor.getModelImage(im)
# tractor.clearCache()
# del srcs
# del im
# del inf
# del mod
# del tractor

# LEAK SUMMARY:
#    definitely lost: 5,165 bytes in 126 blocks
#    indirectly lost: 42,446 bytes in 188 blocks
#      possibly lost: 6,245,200 bytes in 31,037 blocks
#    still reachable: 9,438,753 bytes in 60,207 blocks
#         suppressed: 0 bytes in 0 blocks

# vg3.log:
# srcs = get_tractor_sources_dr8(3063,4,60,'g', roi=[100,200,100,200])
# LEAK SUMMARY:
#    definitely lost: 5,205 bytes in 127 blocks
#    indirectly lost: 43,006 bytes in 190 blocks
#      possibly lost: 6,239,158 bytes in 31,026 blocks
#    still reachable: 9,307,420 bytes in 59,010 blocks
#         suppressed: 0 bytes in 0 blocks

# vg4.log:

# added:
# tractor.freezeParam('images')
# tractor.optimize(alphas=[1e-3, 1e-2, 0.1, 1])
	
# LEAK SUMMARY:
#    definitely lost: 5,205 bytes in 127 blocks
#    indirectly lost: 43,006 bytes in 190 blocks
#      possibly lost: 6,299,543 bytes in 31,147 blocks
#    still reachable: 9,323,115 bytes in 59,059 blocks
#         suppressed: 0 bytes in 0 blocks
	
# vg5.log: x10 optimize()
# LEAK SUMMARY:
#    definitely lost: 5,205 bytes in 127 blocks
#    indirectly lost: 43,006 bytes in 190 blocks
#      possibly lost: 6,300,475 bytes in 31,141 blocks
#    still reachable: 9,352,515 bytes in 59,242 blocks
#         suppressed: 0 bytes in 0 blocks

import matplotlib
matplotlib.use('Agg')
import numpy as np

from tractor.sdss_galaxy import get_galaxy_cache, disable_galaxy_cache
tractor = None

import resource
import os
import subprocess
def memuse():
	#cmd = '/bin/ps -p %i -o rss,size,sz,vsz,command' % os.getpid()
	cmd = '/bin/ps -p %i -o rss,size,sz,vsz h' % os.getpid()
	res = subprocess.check_output(cmd, shell=True)
	#print 'ps says:', res
	mem = [int(x) for x in res.split()]
	mem.append(resource.getrusage(resource.RUSAGE_SELF)[2])
	global tractor
	if tractor is not None:
		mem.append(tractor.cache.totalSize())
		mem.append(len(tractor.cache))
	else:
		mem.append(0)
		mem.append(0)
	mem.append(get_galaxy_cache().totalSize())
	mem.append(len(get_galaxy_cache()))

	return mem

import gc
#gc.set_debug(gc.DEBUG_LEAK) # gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_INSTANCES) #
gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_INSTANCES | gc.DEBUG_OBJECTS | gc.DEBUG_SAVEALL)

from garbagetrack import track

labels = []
mem = []
mem.append(memuse())

from tractor.sdss import get_tractor_sources_dr8, get_tractor_image_dr8
from tractor import Tractor, Images, Catalog
from tractor.cache import *

import sys
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stdout)

mem.append(memuse())

track('start', doprint=False)

srcs = get_tractor_sources_dr8(3063,4,60,'g', roi=[100,300,100,300])

labels.append((len(mem), 'sources'))
mem.append(memuse())

track('sources')

im,inf = get_tractor_image_dr8(3063,4,60,'g', roi=[100,300,100,300])

labels.append((len(mem), 'image'))
mem.append(memuse())

track('image')

tractor = Tractor(Images(im), Catalog(*srcs))

#tractor.cache = NullCache()
#disable_galaxy_cache()

labels.append((len(mem), 'tractor'))
mem.append(memuse())

mod = tractor.getModelImage(im)

labels.append((len(mem), 'model'))
mem.append(memuse())

track('mod')

tractor.freezeParam('images')
#for x in range(10):
for x in range(2):
	tractor.optimize(alphas=[1e-3, 1e-2, 0.1, 1])
	labels.append((len(mem), 'opt %i' % x))
	mem.append(memuse())

	nun = gc.collect()
	print nun, 'unreachable objects'

	track('opt')

if False:
	for x in range(10):
		p0 = np.array(tractor.getParams())
		ss = np.array(tractor.getStepSizes())
		p0 += ss * np.random.normal(0., 1e-3, size=len(p0))
		tractor.setParams(p0)
		mod = tractor.getModelImage(im)
	
		labels.append((len(mem), 'mod %i' % x))
		mem.append(memuse())
		track('mod')
	
	for x in range(10):
		p0 = np.array(tractor.getParams())
		ss = np.array(tractor.getStepSizes())
		p0 += ss * np.random.normal(0., 1e-3, size=len(p0))
		tractor.setParams(p0)
		dervs = tractor.getDerivs()
	
		labels.append((len(mem), 'derivs %i' % x))
		mem.append(memuse())
		track('derivs')
	
	
	for x in range(10):
		X = tractor.getUpdateDirection(dervs)
		labels.append((len(mem), 'upd %i' % x))
		mem.append(memuse())
		track('upd')




tractor.clearCache()
labels.append((len(mem), 'clear cache'))
mem.append(memuse())
track('clear tractor cache')

get_galaxy_cache().clear()
labels.append((len(mem), 'clear gal cache'))
mem.append(memuse())
track('clear galaxy cache')

del srcs
del im
del inf
del mod
del tractor
tractor = None

labels.append((len(mem), 'del'))
mem.append(memuse())
track('del')

nun = gc.collect()
print nun, 'unreachable objects'
labels.append((len(mem), 'gc'))
mem.append(memuse())
track('end')

# print
# print
# print 'Unreachable objects:'
# print
# 
# garbage = gc.garbage
# 
# for i,obj in enumerate(garbage):
# 	print
# 	print
# 	print
# 	print i, type(obj)
# 	refs = []
# 	rr = gc.get_referents(obj)
# 	#try:
# 	#	refs.append(garbage.index(r))
# 	#except ValueError:
# 	#	pass
# 	for j,o in enumerate(garbage):
# 		if i == j:
# 			continue
# 		if any([o is r for r in rr]):
# 			refs.append(j)
# 		#if o in rr:
# 		#	refs.append(j)
# 	print 'Refers to:', refs
# 	print str(obj)[:256]
# 	print repr(obj)[:256]


import pylab as plt
plt.clf()
mem = np.array(mem)
print 'mem', mem.shape
scales = [ 1e-3, 1e-6, 1. ]
for i,(nm,j) in enumerate([('rss',0), ('size',0), ('sz',0), ('vsz',0), ('maxrss',0),
						   ('tcache pix',1), ('tcache N',2),
						   ('galcache pix',1), ('galcache N',2)]):
	#plt.plot(mem[:,i], '-', label=nm)
	plt.subplot(3,1, j+1)
	plt.plot(mem[:,i] * scales[j], '-', label=nm)

plt.subplot(3,1, 1)
ax = plt.axis()
for x,txt in labels:
	plt.text(x, 0.1 * ax[3], txt, rotation='vertical', va='bottom')
labs = ['MB', 'Mpixels', 'number of cache entries']
for j in range(3):
	plt.subplot(3,1, j+1)
	plt.ylabel(labs[j])
	plt.legend(loc='upper left')
plt.savefig('mem.png')

