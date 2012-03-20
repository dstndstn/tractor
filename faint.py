import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from glob import glob
from astrometry.util.pyfits_utils import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.file import *
from astrometry.util.util import *
from astrometry.sdss import *
from tractor import *
from tractor import cfht as cf
from tractor import sdss as st
from tractor.sdss_galaxy import *
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params
#import emcee

def getdata(RA, DEC):
	rcf = radec_to_sdss_rcf(RA, DEC, radius=0., tablefn='s82fields.fits',
							contains=True)
	print 'SDSS fields nearby:', len(rcf)
	rcf = [(r,c,f,ra,dec) for r,c,f,ra,dec in rcf if r != 206]
	print 'Filtering out run 206:', len(rcf)

	sdss = DR7()
	sdss.setBasedir('faint')
	band = 'z'
	skies = []
	ims = []
	tais = []
	rcfs = []
	#S = 50
	S = 40
	for i,(r,c,f,ra,dec) in enumerate(rcf):
		print
		print 'Retrieving', (i+1), 'of', len(rcf), r,c,f,band
		try:
			im,info = st.get_tractor_image(r, c, f, band, psf='dg', useMags=True,
										   sdssobj=sdss,
										   roiradecsize=(RA, DEC, S/2))
		except:
			continue
		skies.append((info['sky'], info['skysig']))
		tais.append(info['tai'])
		rcfs.append((r,c,f))
		ims.append(im)

		### 
		#if i == 9:
		#	break
			
	return ims,skies,tais,rcfs

def mysavefig(fn):
	for suff in ['.png']:#,'.pdf']:
		plt.savefig(fn + suff)
		print 'Wrote', fn + suff

def main():
	# faint
	#RA,DEC,N = 17.5600, 1.1053, 1
	# bright neighbour
	#RA,DEC,N = 358.5436, 0.7213, 2
	# known; too bright
	#RA,DEC,N = 52.64687, -0.42678, 3
	# known; z = 19.5; parallax -0.003+-0.142;
	#   dRA/dt +0.084+-0.010; dDec/dt 0.011+-0.009
	RA,DEC,N = 342.47285, 0.73458, 4

	pfn = 'faint%i.pickle' % N
	if os.path.exists(pfn):
		print 'Reading pickle', pfn
		X = unpickle_from_file(pfn)
	else:
		#X = getdata(358.5436, 0.7213)
		X = getdata(RA, DEC)
		pickle_to_file(X, pfn)
	(ims,skies,tais,rcfs) = X
	print 'Got', len(rcfs), 'fields'

	omit = [94, 5759]

	uruns = set()
	I = []
	for i,(r,c,f) in enumerate(rcfs):
		if r in uruns:
			continue
		if r in omit:
			print 'Omitting run', r
			continue
		uruns.add(r)
		I.append(i)
	print 'Cut to', len(uruns), 'unique runs'
	rcfs = [rcfs[i] for i in I]
	ims = [ims[i] for i in I]
	skies = [skies[i] for i in I]
	tais = [tais[i] for i in I]

	I = np.argsort(tais)
	rcfs = [rcfs[i] for i in I]
	ims = [ims[i] for i in I]
	skies = [skies[i] for i in I]
	tais = [tais[i] for i in I]

	zrs = [np.array([-1.,+5.]) * std + sky for sky,std in skies]

	plt.figure(figsize=(4,4))
	plt.clf()
	plotpos0 = [0.01, 0.01, 0.98, 0.92]

	imsum = None
	slo,shi = None,None

	for i,im in enumerate(ims):

		zr = zrs[i]
		ima = dict(interpolation='nearest', origin='lower',
				   vmin=zr[0], vmax=zr[1], cmap='gray')

		data = im.getImage()

		if imsum is None:
			imsum = data
			slo,shi = zr[0],zr[1]
		else:
			imsum += data
			slo += zr[0]
			shi += zr[1]
	
		plt.clf()
		plt.gca().set_position(plotpos0)
		plt.imshow(data, **ima)
		#plt.title('Data %s' % im.name)
		plt.xticks([],[])
		plt.yticks([],[])
		mysavefig('faint-data%02i' % i)

	plt.clf()
	plt.gca().set_position(plotpos0)
	plt.imshow(imsum, interpolation='nearest', origin='lower',
			   vmin=slo, vmax=shi, cmap='gray')
	plt.title('Data sum')
	plt.xticks([],[])
	plt.yticks([],[])
	mysavefig('faint-data-sum')

	shi = slo + (shi - slo) / np.sqrt(len(ims))
	plt.clf()
	plt.gca().set_position(plotpos0)
	plt.imshow(imsum, interpolation='nearest', origin='lower',
			   vmin=slo, vmax=shi, cmap='gray')
	plt.title('Data sum')
	plt.xticks([],[])
	plt.yticks([],[])
	mysavefig('faint-data-sum2')

	plt.clf()
	plt.gca().set_position(plotpos0)
	plt.imshow(imsum, interpolation='nearest', origin='lower',
			   cmap='gray')
	plt.title('Data sum')
	plt.xticks([],[])
	plt.yticks([],[])
	mysavefig('faint-data-sum3')


if __name__ == '__main__':
	main()
