from numpy import *
from integral_image import intimg_rect

import unittest

# Compute the area of the rectangular region in a first-quadrant
# integral image.
# (x0,y0) is the pixel one less than the first pixel included in the sum
# (x1,y1) is the last  pixel included in the sum
def symm_intimg_rect(img, xx0, xx1, yy0, yy1, midline):
	'''
	Computes the area of a given rectangle, given an integral image for
	an image that is assumed to be mirror-symmetric in both axes.

	That is, if the image size is N x N, then I[i, j] == I[N-i-1, j]
	and I[i, j] == I[i, N-j-1].

	The image must be odd-sized and square.

	"midline" is (image size - 1)/2; if the image is 2N-1, midline == N.

	"img" must be an integral image of size at least midline+1 x midline+1.

	(xx0,yy0) are one less than the first pixel included in the sum.
	(xx1,yy1) are the last pixel included in the sum.

	(Note that the inclusiveness is opposite from convention.)

	These can be (1-d) arrays, but they must be the same shape.

	The return value is:

        sum_{i=xx0 + 1}^{xx1} sum_{j = yy0 + 1}^{yy1} I[j,i]


	>>> if True:
	...     from numpy import fliplr, flipud
	...     from numpy.random import random_sample, seed
	...     from integral_image import *
	...     seed(42)
	...     I = random_sample((11,11))
	...     # symmetrize by copying the first quadrant
	...     I[:,6:] = fliplr(I[:,0:5])
	...     I[6:,:] = flipud(I[0:5,:])
	...     intimg = integral_image(I)
	...     symmimg = integral_image(I[:6,:6])
	...     for (x0,x1,y0,y1) in [(-1,0,-1,0), (3,7,4,8), (-1,10,-1,10), (5,6,-1,0)]:
	...         R0 = intimg_rect(intimg, x0, x1, y0, y1)
	...         R1 = symm_intimg_rect(symmimg, x0, x1, y0, y1, 5)
	...         print round(R0,5), round(R1,5), round(R0,5) == round(R1,5)
	0.37454 0.37454 True
	5.76586 5.76586 True
	58.09467 58.09467 True
	0.15602 0.15602 True
	
	'''
	# This might not be strictly necessary -- you could imaging wanting
	# to compute a negative area -- but we don't need it for now, so assert
	# some simplicity.
	x0 = atleast_1d(xx0).astype(int)
	x1 = atleast_1d(xx1).astype(int)
	y0 = atleast_1d(yy0).astype(int)
	y1 = atleast_1d(yy1).astype(int)
	assert(all(x1 >= x0))
	assert(all(y1 >= y0))
	R = zeros(len(x0))

	## FIXME -- This could be written much more cleanly, perhaps by treating
	## everything as straddling both midlines.

	# box A (quadrant 1): un-flipped
	IA = logical_and(y0 < midline, x0 < midline)
	if any(IA):
		ax1 = minimum(x1[IA], midline)
		ay1 = minimum(y1[IA], midline)
		R[IA] += intimg_rect(img, x0[IA], ax1, y0[IA], ay1)

	# box B (quadrant 2): x flipped
	IB = logical_and(x1 > midline, y0 < midline)
	if any(IB):
		flipx0 = 2*midline - maximum(midline+1, x0[IB]+1)
		flipx1 = (2*midline - x1[IB]) - 1
		by1 = minimum(y1[IB], midline)
		R[IB] += intimg_rect(img, flipx1, flipx0, y0[IB], by1)

	# box C (quadrant 4): y flipped.
	IC = logical_and(y1 > midline, x0 < midline)
	if any(IC):
		flipy0 = 2*midline - maximum(midline+1, y0[IC]+1)
		flipy1 = (2*midline - y1[IC]) - 1
		cx1 = minimum(x1[IC], midline)
		R[IC] += intimg_rect(img, x0[IC], cx1, flipy1, flipy0)

	# box D (quadrant 3): both flipped.
	ID = logical_and(y1 > midline, x1 > midline)
	if any(ID):
		flipy0 = 2*midline - maximum(midline+1, y0[ID] + 1)
		flipy1 = (2*midline - y1[ID]) - 1
		flipx0 = 2*midline - maximum(midline+1, x0[ID] + 1)
		flipx1 = (2*midline - x1[ID]) - 1
		R[ID] += intimg_rect(img, flipx1, flipx0, flipy1, flipy0)
	if isscalar(xx0):
		return R[0]
	return R

	


# Reference an element in a symmetric integral image
def symm_intimg_ref(img, x, y, midline):
	if x <= midline and y <= midline:
		return img[y,x]

	elif x <= midline and y > midline:
		ry = 2*midline - y
		I = img[midline,x] + img[midline-1,x]
		if ry > 0:
			I -= img[ry-1,x]
		return I

	elif y <= midline and x > midline:
		rx = 2*midline - x
		I = img[y,midline] + img[y,midline-1]
		if rx > 0:
			I -= img[y,rx-1]
		return I

	else:
		rx = 2*midline - x
		ry = 2*midline - y
		I = (img[midline,midline] + img[midline,midline-1]
			 + img[midline-1,midline] + img[midline-1,midline-1])
		if rx > 0 and ry > 0:
			I += img[ry-1,rx-1]
		if rx > 0:
			I -= (img[midline,rx-1] + img[midline-1,rx-1])
		if ry > 0:
			I -= (img[ry-1,midline] + img[ry-1,midline-1])
		return I



class testsymmintimg(unittest.TestCase):
    def test_rectangles(self):
		from numpy import fliplr, flipud
		from numpy.random import random_sample, seed
		from integral_image import *
		seed(42)
		I = random_sample((11,11))
		# symmetrize by copying the first quadrant
		I[:,6:] = fliplr(I[:,0:5])
		I[6:,:] = flipud(I[0:5,:])
		intimg = integral_image(I)
		symmimg = integral_image(I[:6,:6])
		
		for (x0,x1,y0,y1) in [(-1,0,-1,0), # first element
							  (1,4,1,4), # first quadrant (simple)
							  (-1,5,-1,5), # first quadrant (edges)
							  (6,9,1,4), # quadrant 2 (simple)
							  (5,10,1,4), # quadrant 2 (edges)
							  (4,5,-1,5), # quadrant 1/2 boundary
							  (4,6,-1,5), # quadrant 1/2 boundary + 1 over
							  (-1,10,-1,5), # quadrants 1,2 (up to edges)
							  (7,9,7,9), # quadrant 3
							  (1,4,7,9), # quadrant 4
							  (3,7,4,8), #
							  (-1,10,-1,10), #
							  (5,6,-1,0)]:
			R0 = intimg_rect(intimg, x0, x1, y0, y1)
			R1 = symm_intimg_rect(symmimg, x0, x1, y0, y1, 5)
			print round(R0,5), round(R1,5), round(R0,5) == round(R1,5)
			self.assertAlmostEqual(R1, R0, 8)

if __name__ == '__main__':
	import doctest
	doctest.testmod()
	unittest.main()
