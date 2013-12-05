import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import pyfits

from astrometry.util.fits import *
from astrometry.util.plotutils import *

'''
Plots for Rachel's talk to CS/graphics

on sesame3:

setup -t HSC pipe_tasks
setup -t HSC obs_subaru
setup astrometry_net_data sdss-dr8

rsync -Rarvz /scratch/lustre/hsc/Subaru/SUPA/./ACTJ0022M0036/2010-12-05/00867/W-J-B/SUPA01269265.fits SUP-ACT
rsync -Rarvz /scratch/lustre/hsc/Subaru/SUPA/./CALIB/calibRegistry.sqlite3 SUP-ACT
rsync -Rarvz /scratch/lustre/hsc/Subaru/SUPA/./CALIB/FLAT/2010-12-05/W-J-B/000/FLAT-00000005.fits SUP-ACT
rsync -Rarvz /scratch/lustre/hsc/Subaru/SUPA/./ACTJ0022M0036/2010-12-05/00867/W-J-B/SUPA01269265.fits SUP-ACT 

> cat myconf.py
import lsst.meas.extensions.shapeHSM
root.measurement.algorithms.names |= ['shape.hsm.regauss'] #lsst.meas.extensions.shapeHSM.algorithms

$PIPE_TASKS_DIR/bin/processCcd.py suprimecam SUP-ACT/ --id visit=126926 ccd=5 --output SUP-ACT/out/ -C myconf.py

'''

imc = dict(interpolation='nearest', origin='lower',
		   norm=ArcsinhNormalize(mean=0., std=20.),
		   vmin=-20, vmax=2000.)#vmax=1000.)

ps = PlotSequence('gals')

P=pyfits.open('CORR01269265.fits')

var = P[2].data
sigma = np.sqrt(np.median(var))
print 'Sigma:', sigma

FW,FH = 14,10

plt.figure(figsize=(FW,FH))
plt.subplots_adjust(hspace=0.01, wspace=0.01,
					left=0.01, right=0.99,
					bottom=0.01, top=0.99)
#left=0.05, right=0.95,
#bottom=0.05, top=0.95)

x0,y0 = 0,100
im=P[1].data
sub = im[y0:2100, :]
sub = sub[:FH*100, :FW*100]
H,W = sub.shape
T=fits_table('SRC01269265.fits')

# plt.clf()
# plt.hist(im.ravel(), 100, range=(-100, 1000))
# ps.savefig()

plt.clf()
plt.imshow(sub, extent=[x0,x0+W, y0,y0+H], **imc)
plt.gray()
plt.xticks([])
plt.yticks([])
ps.savefig()

ax = plt.axis()
xy = T.centroid_sdss
#plt.plot(xy[:,0], xy[:,1], 'ro', mec='r', ms=6, mfc='none')
#plt.axis(ax)
#ps.savefig()

good = T.copy()
good.about()
SN = good.multishapelet_combo_flux / good.multishapelet_combo_flux_err
good.cut(SN > 10)
good.SN = good.multishapelet_combo_flux / good.multishapelet_combo_flux_err

good.x = 0.5 + good.shape_hsm_regauss_centroid[:,0]
good.y = 0.5 + good.shape_hsm_regauss_centroid[:,1]

ms = dict(marker='o', mec='g', ms=15, mfc='none', mew=2, linestyle='none')
ms2 = dict(marker='o', mec='r', ms=12, mfc='none', mew=2, linestyle='none')

plt.clf()
plt.imshow(sub, extent=[x0,x0+W, y0,y0+H], **imc)
plt.gray()
plt.plot(good.x, good.y, **ms)
plt.axis(ax)
ps.savefig()

I = (good.shape_hsm_regauss_resolution > 0.3)
bad = good[np.logical_not(I)]
good.cut(I)

plt.clf()
plt.imshow(sub, extent=[x0,x0+W, y0,y0+H], **imc)
plt.gray()
plt.plot(good.x, good.y, **ms)
plt.plot(bad.x, bad.y, **ms2)
plt.axis(ax)
ps.savefig()

# good.cut(good.classification_extendedness == 1)
# 
# plt.clf()
# plt.imshow(sub, extent=[x0,x0+W, y0,y0+H], **imc)
# plt.gray()
# xy = good.centroid_sdss
# plt.plot(xy[:,0], xy[:,1], 'ro', mec='r', ms=6, mfc='none')
# plt.axis(ax)
# ps.savefig()

#x,y = xy[:,0], xy[:,1]
#good.cut((x >= x0) * (x < x0+W) * (y >= y0) * (y < y0+H))
#xy = good.centroid_sdss
#x,y = xy[:,0], xy[:,1]

# margin
S = 15
good.cut((good.x > (x0+S)) * (good.x < (x0+W-S)) *
		 (good.y > (y0+S)) * (good.y < (y0+H-S)))
print len(good), 'good galaxies in range'

R,C = FH,FW
#ima = dict(interpolation='nearest', origin='lower',
#		   norm=ArcsinhNormalize(mean=0., std=10.),
#		   vmin=-30, vmax=500.)
ima = imc
S = 10
I = np.argsort(-good.SN)

for jveto in [ [], [ 4,5,6,7,13,15,18,23,25,32,37,
					 21,33,45,54] ]:
	plt.clf()
	k = 0
	for j,i in enumerate(I):
		if good.flags[i,10]: # flags_pixel_interpolated_center
			continue
		if good.flags[i,11]: # flags_pixel_saturated_any
			continue
		if j in jveto:
			continue
		if k == R*C:
			print 'Lowest S/N:', good.SN[i]
			break
		plt.subplot(R,C, k+1)
		k += 1
		ix,iy = int(good.x[i]) - x0, int(good.y[i]) - y0
		print 'galaxy at', ix,iy
		plt.imshow(sub[iy-S: iy+S, ix-S: ix+S], **ima)
	
		#plt.text(S, S, '%i' % j, color='g')
		#	bbox=dict(facecolor='1', alpha=0.8))
	
		# interpolation='nearest',
		# origin='lower', vmin=0, vmax=100)
		plt.xticks([])
		plt.yticks([])
	ps.savefig()
	
