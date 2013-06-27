#! /usr/bin/env python

'''
This is a PBS driver script we used on riemann to perform
WISE-from-SDSS forced photometry.  

Jobs are submitted like:
qsub -l "nodes=1:ppn=1" -l "walltime=3:00:00" -N w1v4 -o w1v4.log -q batch -t 100-190 ./runslice.py

The "-t" option (job arrays) causes the PBS_ARRAYID environment
variable to be set for each job.  We use that to determine which chunk
of work that job is going to do.

In the case of the eBOSS W3 test, we split the region into boxes in
RA,Dec and used the job id for the Dec slice (90 slices).

(I was also using this as a non-PBS command-line driver; batch=False
then)

'''

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
	#arr = 125
	# HACK!
	arr = 147
else:
	arr = int(arr)
	
band = int(arr / 100)
ri = arr % 100

print 'Band', band
print 'RA slice', ri

# duck-type command-line options
class myopts(object):
	pass
opt = myopts()

basename = 'ebossw3-v5'

if not batch:
	basename = 'ebossw3-tst'

opt.minflux = None
opt.bandnum = band
opt.osources = None
opt.sources = 'objs-eboss-w3-dr9.fits'

opt.ptsrc = False
# v5
#opt.ptsrc = True

opt.pixpsf = False

#opt.minsb = 0.05
opt.minsb = 0.005

lvl = logging.INFO
logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

# W3 area
r0,r1 = 210.593,  219.132
d0,d1 =  51.1822,  54.1822
basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise1test'
wisedatadirs = [(os.path.join(basedir, 'allsky'), 'cryo'),
                (os.path.join(basedir, 'prelim_postcryo'), 'post-cryo'),]


dd = np.linspace(d0, d1, NDEC + 1)
rr = np.linspace(r0, r1, NRA  + 1)

#print 'RA steps:', rr
#print 'Dec steps:', dd

rlo,rhi = rr[ri], rr[ri+1]

for di,(dlo,dhi) in enumerate(zip(dd[:-1], dd[1:])):

	fn = '%s-r%02i-d%02i-w%i.fits' % (basename, ri, di, opt.bandnum)
	if os.path.exists(fn):
		print 'Output file exists:', fn
		print 'Skipping'
		if batch:
			continue

	# HACK!!
	if not batch and di != 25:
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

		pfn = '%s-r%02i-d%02i-w%i.pickle' % (basename, ri, di, opt.bandnum)

		tractor = P['tractor']
		ims1 = P['ims1']

		res1 = []
		for tim,(img,mod,ie,chi,roi) in zip(tractor.images, ims1):
			print 'Tim:', dir(tim)
			for k in ['origInvvar', 'starMask', 'inverr', 'cinvvar', 'goodmask',
					  'mask', 'maskplane', 'rdmask', 'uncplane', 'vinvvar']:
				try:
					delattr(tim, k)
				except:
					pass
			print 'Tim:', dir(tim)
			res1.append((tim, mod, roi))

		PP = dict(res1=res1, cat=tractor.getCatalog(), rd=P['rd'], ri=ri, di=di,
				  bandnum=opt.bandnum, S=P['S'])
		pickle_to_file(PP, pfn)


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
