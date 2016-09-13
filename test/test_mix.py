from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from tractor import mixture_profiles as mp

from tractor import *
from astrometry.util.plotutils import *
import sys
import os

fn = os.path.join(os.path.dirname(__file__),
                  'c4d_140818_002108_ooi_z_v1.ext27.psf')
psf = PsfEx(fn, 2048, 4096)
psfimg = psf.instantiateAt(100,100)
gpsf = GaussianMixturePSF.fromStamp(psfimg)

print('PSF:', gpsf)

ps = PlotSequence('test-mix')

patch0 = gpsf.getPointSourcePatch(0., 0., radius=20)

approx = 1e-4

patch1 = gpsf.getPointSourcePatch(0., 0., radius=20, v3=True, minval=approx)
patch2,dx2,dy2 = gpsf.getPointSourcePatch(0., 0., radius=20, v3=True,
                                          minval=approx, derivs=True)

mn,mx = patch0.patch.min(), patch0.patch.max()
#ima = dict(vmin=mn, vmax=mx)
floor = approx * 1e-2
ima = dict(vmin=np.log10(max(floor, mn)), vmax=np.log10(mx))

imda = dict(vmin=-approx, vmax=approx)

for patch in [patch1, patch2]:
    print('Patch range', patch.patch.min(), patch.patch.max())
    plt.clf()
    plt.subplot(2,2,1)
    dimshow(np.log10(np.maximum(floor, patch0.patch)), **ima)
    plt.colorbar()
    plt.title('p0')
    plt.subplot(2,2,2)
    dimshow(np.log10(np.maximum(floor, patch.patch)), **ima)
    plt.colorbar()

    plt.subplot(2,2,4)
    dimshow(patch.patch > 0, vmin=0, vmax=1)
    
    plt.title('patch')
    plt.subplot(2,2,3)
    diff = patch.patch - patch0.patch
    print('Diff range:', diff.min(), diff.max())
    dimshow(diff, **imda)
    plt.colorbar()
    plt.title('difference')
    ps.savefig()


plt.clf()
plt.subplot(2,2,2)
dimshow(np.log10(np.maximum(floor, patch2.patch)), **ima)
plt.subplot(2,2,3)
dimshow(dx2.patch)
plt.subplot(2,2,4)
dimshow(dy2.patch)
ps.savefig()

sys.exit(0)




mg = mp.MixtureOfGaussians([1.], [0.,0.], np.array([1.]))
x = mg.evaluate(np.array([0,0]))
x = mg.evaluate(np.array([0.,1.]))
x = mg.evaluate(np.array([-3.,2.]))
x = mg.evaluate(np.array([[-3.,2.], [-17,4], [4,-2]]))

mg = mp.MixtureOfGaussians([1.], [0.,1.], np.array([2.]))
x = mg.evaluate(np.array([0.,1.]))
x = mg.evaluate(np.array([0,0]))
x = mg.evaluate(np.array([-3.,2.]))
x = mg.evaluate(np.array([[-3.,2.], [-17,4], [4,-2]]))

mg = mp.MixtureOfGaussians([1.], [0.,1.], np.array([ [[1.3, 0.1],[0.1,3.1]], ]))
x = mg.evaluate(np.array([0.,1.]))
x = mg.evaluate(np.array([0,0]))
x = mg.evaluate(np.array([-3.,2.]))
x = mg.evaluate(np.array([[-3.,2.], [-17,4], [4,-2]]))

mg = mp.MixtureOfGaussians([1., 0.5], np.array([ [0.,1.], [-0.3,2] ]),
						   np.array([ [[1.3, 0.1],[0.1,3.1]], [[1.2, -0.8],[-0.8, 2.4]], ]))
x = mg.evaluate(np.array([0.,1.]))
x = mg.evaluate(np.array([0,0]))
x = mg.evaluate(np.array([-3.,2.]))
x = mg.evaluate(np.array([[-3.,2.], [-17,4], [4,-2]]))

# via test_hogg_galaxy.py

mg = mp.MixtureOfGaussians(
	np.array([  4.31155865e-05,   1.34300460e-03,   1.62488556e-02,
				1.13537806e-01,   4.19327122e-01,   4.49500096e-01]),
	np.array([[ 50., 66.],
			  [ 50., 66.],
			  [ 50., 66.],
			  [ 50., 66.],
			  [ 50., 66.],
			  [ 50., 66.]]),
	np.array([[[  4.00327202e+00, -2.42884668e-03],
			   [ -2.42884668e-03,  4.00607661e+00]],
			  [[  4.04599449e+00, -3.41420550e-02],
			   [ -3.41420550e-02,  4.08541834e+00]],
			  [[  4.29345271e+00, -2.17832145e-01],
			   [ -2.17832145e-01,  4.54498361e+00]],
			  [[  5.32854173e+00, -9.86186474e-01],
			   [ -9.86186474e-01,  6.46729178e+00]],
			  [[  8.90680650e+00, -3.64235921e+00],
			   [ -3.64235921e+00,  1.31126406e+01]],
			  [[  2.00488533e+01, -1.19131840e+01],
			   [ -1.19131840e+01,  3.38050133e+01]]]))

	   
x = mg.evaluate(np.array([0.,1.]))
x = mg.evaluate(np.array([0,0]))
x = mg.evaluate(np.array([-3.,2.]))
x = mg.evaluate(np.array([[-3.,2.], [-17,4], [4,-2]]))

x = mg.evaluate(np.array([[ 27.,  43.],
						  [ 28.,  43.],
						  [ 29.,  43.],
						  [ 72.,  90.],
						  [ 73.,  90.],
						  [ 74.,  90.]]))

X,Y = np.meshgrid(np.arange(27, 75),
				  np.arange(43, 91))
XY = np.vstack((X.ravel(), Y.ravel())).T
print(XY.shape)
x = mg.evaluate(XY)







mg = mp.MixtureOfGaussians(
	np.array([  4.31155865e-05,
				]),
	#1.34300460e-03,   1.62488556e-02,
	#1.13537806e-01,   4.19327122e-01,   4.49500096e-01]),
	np.array([[ 50., 66.],
			  ]),
	#[ 50., 66.],
	#[ 50., 66.],
	#[ 50., 66.],
	#[ 50., 66.],
	#[ 50., 66.]]),
	np.array([[[  4.00327202e+00, -2.42884668e-03],
			   [ -2.42884668e-03,  4.00607661e+00]],
			  ])
	)
	#[[  4.04599449e+00, -3.41420550e-02],
	#		   [ -3.41420550e-02,  4.08541834e+00]],
	#			  [[  4.29345271e+00, -2.17832145e-01],
	#		   [ -2.17832145e-01,  4.54498361e+00]],
	#		  [[  5.32854173e+00, -9.86186474e-01],
	#		   [ -9.86186474e-01,  6.46729178e+00]],
	#		  [[  8.90680650e+00, -3.64235921e+00],
	#		   [ -3.64235921e+00,  1.31126406e+01]],
	#		  [[  2.00488533e+01, -1.19131840e+01],
	#		   [ -1.19131840e+01,  3.38050133e+01]]]))
	
X,Y = np.meshgrid(np.arange(27, 75), np.arange(43, 91))
XY = np.vstack((X.ravel(), Y.ravel())).T
print(XY.shape)
x = mg.evaluate(XY)

#Z = np.array([[27.,43.], [27.,44.], [27.,44.]])
Z = XY[0,:]
print(Z.shape)
x = mg.evaluate(Z)
