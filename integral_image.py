from numpy import *

def integral_image(I):
	return cumsum(cumsum(I, axis=0), axis=1)

def intimg_rect(img, xx0, xx1, yy0, yy1):
	'''
	Computes the area of the rectangular region in an integral image.

	(xx0,yy0) are one less than the first pixel included in the sum.
	(xx1,yy1) are the last pixel included in the sum.

	(Note that the inclusiveness is opposite from usual.)

	These can be (1-d) arrays, but they must be the same shape.

	The return value is:

        sum_{i=xx0 + 1}^{xx1} sum_{j = yy0 + 1}^{yy1} I[j,i]

    where I is the original image

	>>> if True:
	...     from numpy.random import random_sample, seed
	...     seed(42)
	...     I = random_sample((10,10))
	...     # This is the explicit sum
	...     R0 = sum(I[3:5, 4:8])
	...     print round(R0, 5)
	...     # This is the integral-image sum
	...     intimg = integral_image(I)
	...     R1 = intimg_rect(intimg, 4-1, 8-1, 3-1, 5-1)
	...     print round(R1, 5)
	3.9294
	3.9294
	
	'''
	x0 = atleast_1d(xx0).astype(int)
	x1 = atleast_1d(xx1).astype(int)
	y0 = atleast_1d(yy0).astype(int)
	y1 = atleast_1d(yy1).astype(int)
	assert(x0.shape == x1.shape)
	assert(x0.shape == y0.shape)
	assert(x0.shape == y1.shape)
	assert(all(x1 >= 0))
	assert(all(y1 >= 0))
	assert(all(x1 > x0))
	assert(all(y1 > y0))
	A = img[y1,x1]
	I = logical_and((x0 >= 0), (y0 >= 0))
	A[I] += img[y0[I],x0[I]]
	I = (x0 >= 0)
	A[I] -= img[y1[I],x0[I]]
	I = (y0 >= 0)
	A[I] -= img[y0[I],x1[I]]
	if isscalar(xx0):
		return A[0]
	return A


if __name__ == '__main__':
	import doctest
	doctest.testmod()
