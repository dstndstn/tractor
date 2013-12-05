import mixture_profiles as mp
import numpy as np

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
print XY.shape
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
print XY.shape
x = mg.evaluate(XY)

#Z = np.array([[27.,43.], [27.,44.], [27.,44.]])
Z = XY[0,:]
print Z.shape
x = mg.evaluate(Z)
