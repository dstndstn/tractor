#! /usr/bin/env python

import os
import sys

batch = False

d = os.environ.get('PBS_O_WORKDIR')
if d is not None:
	os.chdir(d)
	sys.path.append(os.getcwd())
	batch = True

# print 'args:', sys.argv
# print 'environ:'
# for k,v in os.environ.items():
# 	print '  ', k, '=', v
# print

import numpy as np
import logging
from wise3 import *

NDEC = 50
NRA = 90

arr = os.environ.get('PBS_ARRAYID')
if arr is None:
	#arr = 0
	arr = 125
else:
	arr = int(arr)
	
band = int(arr / 100)
rslice = arr % 100

print 'Band', band
print 'Dec slice', dslice

# duck-type command-line options
class myopts(object):
	pass
opt = myopts()

opt.minflux = None
opt.bandnum = band
opt.osources = None
opt.sources = 'objs-eboss-w3-dr9.fits'
opt.ptsrc = False
opt.pixpsf = False

#opt.minsb = 0.05
opt.minsb = 0.005

lvl = logging.INFO
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

# W3 area
r0,r1 = 210.593,  219.132
d0,d1 =  51.1822,  54.1822

dd = np.linspace(d0, d1, NDEC + 1)
rr = np.linspace(r0, r1, NRA  + 1)

print 'RA steps:', rr
print 'Dec steps:', dd

#dlo,dhi = dd[dslice], dd[dslice+1]
#print 'My dec slice:', dlo, dhi

ri = rslice
rlo,rhi = rr[rslice], rr[rslice+1]

basename = 'ebossw3-v4'

for di,(dlo,dhi) in enumerate(zip(dd[:-1], dd[1:])):

	fn = '%s-r%02i-d%02i-w%i.fits' % (basename, ri, di, opt.bandnum)
	if os.path.exists(fn):
		print 'Output file exists:', fn
		print 'Skipping'
		if batch:
			continue

	try:
		P = dict(ralo=rlo, rahi=rhi, declo=dlo, dechi=dhi,
				 opt=opt)

		R = stage100(**P)
		P.update(R)
		R = stage101(**P)
		P.update(R)
		R = stage102(**P)
		P.update(R)
		R = stage103(**P)
		P.update(R)
		R = stage104(**P)
		P.update(R)

		R = P['R']
		R.writeto(fn)
		print 'Wrote', fn

		imst = P['imstats']
		fn = '%s-r%02i-d%02i-w%i-imstats.fits' % (basename, ri, di, opt.bandnum)
		imst.writeto(fn)
		print 'Wrote', fn

	except:
		import traceback
		print '---------------------------------------------------'
		print 'FAILED: dec slice', di, 'ra slice', ri
		print rlo,rhi, dlo,dhi
		print '---------------------------------------------------'
		traceback.print_exc()
		print '---------------------------------------------------'
		if not batch:
			raise
