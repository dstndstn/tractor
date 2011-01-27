import sys
import pyfits

from numpy import *
import numpy

import numpy as np

from galaxy_profiles import *
from integral_image import *
from symmetric_integral_image import *
from astrometry.util.miscutils import *


class CompiledProfile():
	def __init__(self, filename=None, hdu=0, **kwargs):
		if filename is not None:
			self._read(filename, hdu)
		else:
			self._create(**kwargs)

	def __str__(self):
		return 'CompiledProfile: %s, re %g, nr %g, ab %g, csize %i, cn %i' % (self.modelname, self.cre, self.cnrad, self.cab, self.csize, self.cn)

	def _read(self, filename=None, hdu=0):
		p = pyfits.open(filename)[hdu]
		hdr = p.header
		self.intimg = p.data
		assert(self.intimg.shape[1] == self.intimg.shape[0])
		self.modelname = hdr['PROF_NM']
		self.cre = hdr['PROF_RE']
		self.cnrad = hdr['PROF_NR']
		self.cab = hdr['PROF_AB']
		## Here we assume symmetry (thus also that cab == 1) and that
		# the image is a symmetric integral image.
		assert(self.cab == 1)
		self.csize = self.intimg.shape[0] - 1
		self.cn = self.csize * 2 + 1

	def _create(self, profile_func=None, modelname=None, re=None, nrad=None, ab=1.0,
				subnpix=10, nsub=101):
		assert(ab == 1)
		self.modelname = modelname
		self.cre = float(re)
		self.cnrad = nrad
		self.cab = float(ab)
		self.csize = int(ceil(self.cnrad * self.cre))
		self.cn = 2*self.csize + 1

		# Sample the whole thing... we will overwrite the subsampled region.
		(gridi,gridj) = meshgrid(arange(self.csize+1)-self.csize,
								 arange(self.csize+1)-self.csize)
		R = sqrt((gridi/self.cab)**2 + gridj**2)
		profile = profile_func(R/self.cre)

		# Subsample the bottom-right corner.
		g = (arange(nsub)-(nsub-1)/2) / float(nsub)
		(subgridi,subgridj) = meshgrid(g, g)
		#print 'g: min %g, max %g, len %i' % (g.min(), g.max(), len(g))
		for subi in range(subnpix):
			for subj in range(subi+1): #subnpix:
				R = sqrt(((subi + subgridi) / self.cab)**2 + (subj + subgridj)**2)
				p = mean(profile_func(R/self.cre))
				#print 'subsampling pixel', subi,subj, ('(%i,%i)' % (self.csize-subi, self.csize-subj)),
				#print ' :  %g -> %g' % (profile[self.csize-subi, self.csize-subj], p)
				profile[self.csize-subi, self.csize-subj] = p
				profile[self.csize-subj, self.csize-subi] = p

		# Normalize by the sum of the whole (mirrored) profile.
		profile /= (sum(profile) + 3 * sum(profile[:-1,:-1]))

		self.intimg = integral_image(profile)

	def get_hdu(self, hdutype=pyfits.PrimaryHDU):
		hdu = hdutype(self.intimg)
		hdr = hdu.header
		hdr.update('PROF_NM', self.modelname)
		hdr.update('PROF_RE', self.cre)
		hdr.update('PROF_NR', self.cnrad)
		hdr.update('PROF_AB', self.cab)
		return hdu

	def get_compiled_ab(self):
		return self.cab

	def sample_transform(self, T, re_pix, ab, x, y, outw, outh, margin,
						 debugstep=0):
		'''
		T: transforms pixels in output image coords to units of r_e
	       in the compiled profile.

	    re_pix: r_e in pixels to set the returned patch size
		'''
		# target (output) size
		tsize = ceil(self.cnrad * re_pix) + margin
		tn = 2*tsize + 1
		#print 'tn', tn

		ix = round(x)
		iy = round(y)

		# Sample only within ~ the image bounds
		# (we will trim more carefully below)
		sxlo = floor(max(ix - tsize - 1, -margin))
		sxhi = ceil (min(ix + tsize + 1, outw - 1 + margin))
		sylo = floor(max(iy - tsize - 1, -margin))
		syhi = ceil (min(iy + tsize + 1, outh - 1 + margin))
		#print 'sample range x', sxlo, sxhi, 'y', sylo, syhi

		# gridi,gridj are integers (-tsize to +tsize); shift to the
		# correct subpixel positions.
		(gridi, gridj) = meshgrid(arange(sxlo, sxhi+1)-ix,
								  arange(sylo, syhi+1)-iy)
		# The integer pixel offsets (ix,iy) get added in to x0,y0 at the
		# very end.  The fractional parts (fx,fy) get dealt with here
		fx = x - ix
		fy = y - iy
		gridi -= fx
		gridj -= fy
		gridshape = gridi.shape
		gridi = gridi.ravel()
		gridj = gridj.ravel()
		# Now we compute where those sample points land in the
		# compiled profile image.
		ij = vstack((gridi, gridj))
		#print 'ij range:', gridi.min(), gridi.max(), gridj.min(), gridj.max()

		Tij = np.dot(T, ij)
		#print 'Tij', Tij.shape
		#print 'Tij range', Tij[0,:].min(), Tij[0,:].max(), Tij[1,:].min(), Tij[1,:].max()
		# At this point, Tij has shape (2,N), and is in units of r_e in
		# the compiled profile.
		# Multiplying by the compiled r_e will make them in units
		# of pixels in the compiled profile.

		ab_factor = self.cab / max(ab, 1./float(2. * self.cn))
		re_factor = self.cre / (max(re_pix, 1./float(2. * self.cn)))
		# cpix{w,h} are the width,height of sample pixels in the
		# compiled profile
		# FIXME -- it's not at all clear that this is the right way to
		# assign the axis ratio!  Should change wrt angle phi.
		cpixw = re_factor * ab_factor
		cpixh = re_factor
		# Now compute ii,jj, the pixel centers in the compiled image.
		# Tij was in units of (compiled) r_e, so scale to pixels and shift
		# to the center...
		cx0 = cy0 = self.csize
		ii = Tij[0,:] * self.cre + cx0
		jj = Tij[1,:] * self.cre + cy0
		xlo = numpy.round(ii - cpixw/2.).astype(int) - 1
		xhi = numpy.round(ii + cpixw/2.).astype(int)
		ylo = numpy.round(jj - cpixh/2.).astype(int) - 1
		yhi = numpy.round(jj + cpixh/2.).astype(int)

		if debugstep == 1:
			return (xlo,xhi,ylo,yhi, self.cre, self.cn, cpixw, cpixh,
					re_factor, ab_factor, Tij, ii, jj)
		# Clip the boxes that are completely out of bounds:
		ib = logical_and(logical_and(xlo < self.cn, ylo < self.cn),
						 logical_and(xhi >= 0, yhi >= 0))
		# Compute the pixel areas before clamping...
		A = (yhi[ib] - ylo[ib]).astype(float) * (xhi[ib] - xlo[ib])
		# ... then clamp to the edges of the image.
		xlo = clip(xlo[ib], -1, self.cn-2)
		xhi = clip(xhi[ib],  0, self.cn-1)
		ylo = clip(ylo[ib], -1, self.cn-2)
		yhi = clip(yhi[ib],  0, self.cn-1)
		assert(all((xlo<xhi)*(ylo<yhi)))

		profile = zeros_like(gridi.ravel())
		# Take mean over the rectangular region...
		profile[ib] = (symm_intimg_rect(self.intimg, xlo, xhi, ylo, yhi, self.csize)
					   / A)
		profile = profile.reshape(gridshape)
		# correct for the count-density of the compiled model...
		profile *= (cpixw * cpixh)

		(outx, inx) = get_overlapping_region(sxlo, sxhi+1, -margin, outw-1 + margin)
		(outy, iny) = get_overlapping_region(sylo, syhi+1, -margin, outh-1 + margin)
		if inx == [] or iny == []:
			return (None, 0, 0)
		x0 = outx.start
		y0 = outy.start
		profile = profile[iny,inx]
		return (profile, x0, y0)


		return None

	def sample(self, r_e, ab, phi, x, y, outw, outh, margin):
		# target (output) size: number of r_e in the compiled profile
		# times r_e of the galaxy to be sampled.
		tsize = ceil(self.cnrad * r_e) + margin
		tn = 2*tsize + 1

		ix = round(x)
		iy = round(y)
		#print 'Sampling a galaxy profile: %s, r_e %g, ab %g, phi %g.  Size %i' % (self.modelname, r_e, ab, phi, tn)
		# Sample only within ~ the image bounds (we will trim more carefully below)
		sxlo = floor(max(ix - tsize - 1, -margin))
		sxhi = ceil (min(ix + tsize + 1, outw - 1 + margin))
		sylo = floor(max(iy - tsize - 1, -margin))
		syhi = ceil (min(iy + tsize + 1, outh - 1 + margin))

		(gridi, gridj) = meshgrid(arange(sxlo, sxhi+1)-ix, arange(sylo, syhi+1)-iy)
		#(gridi,gridj) = meshgrid(arange(tn)-tsize, arange(tn)-tsize)
		# gridi,gridj are integers (-tsize to +tsize); shift to the
		# correct subpixel positions.
		fx = x - ix
		fy = y - iy
		# The integer pixel offsets (ix,iy) get added in to x0,y0 at the
		# very end.  The fractional parts (fx,fy) get dealt with here:
		gridi -= fx
		gridj -= fy
		gridshape = gridi.shape
		gridi = gridi.ravel()
		gridj = gridj.ravel()

		# Now we compute where those sample points land in the
		# compiled profile image.
		ij = vstack((gridi, gridj))
		# stretch by 1/ab (carefully)
		# If ab~=0, we want to stretch so that the whole compiled profile
		# fits in one sampled pixel.  We can overshoot that.
		ab_factor = self.cab / max(ab, 1./float(2. * self.cn))
		Tab = array([[ab_factor, 0.,], [0., 1.]])
		# rotate
		phirad = deg2rad(90 - phi)
		Trot = array([[cos(phirad), -sin(phirad)],
					  [sin(phirad),  cos(phirad)]])
		T = dot(Tab, Trot)
		Tij = dot(T, ij)
		# At this point, Tij has shape (2,N), and is in units of the
		# sample pixels; dividing by r_e will make them in units of r_e,
		# and multiplying by the compiled r_e will make them in units
		# of pixels in the compiled profile.

		# Scale by r_e (carefully) -- the smallest r_e we want to allow
		# would make the whole compiled profile fit in one pixel.
		re_factor = self.cre / (max(r_e, 1./float(2. * self.cn)))

		# Compute rectangular pixel sizes in the "cprof" image.  NOTE
		# that this is an approximation -- the pixels are really
		# sheared (parallelograms), but here we use (axis-aligned)
		# rectangles.

		# cpix{w,h} are the width,height of sample pixels in the
		# compiled profile
		cpixw = re_factor * ab_factor
		cpixh = re_factor
		# Now compute ii,jj, the pixel centers in the compiled image.
		# Tij was in units of (compiled) r_e, so scale to pixels and shift
		# to the center...
		cx0 = cy0 = self.csize
		ii = Tij[0,:] * re_factor + cx0
		jj = Tij[1,:] * re_factor + cy0
		xlo = numpy.round(ii - cpixw/2.).astype(int) - 1
		xhi = numpy.round(ii + cpixw/2.).astype(int)
		ylo = numpy.round(jj - cpixh/2.).astype(int) - 1
		yhi = numpy.round(jj + cpixh/2.).astype(int)

		# Clip the boxes that are completely out of bounds:
		ib = logical_and(logical_and(xlo < self.cn, ylo < self.cn),
						 logical_and(xhi >= 0, yhi >= 0))
		# Compute the pixel areas before clamping...
		A = (yhi[ib] - ylo[ib]).astype(float) * (xhi[ib] - xlo[ib])
		# ... then clamp to the edges of the image.
		xlo = clip(xlo[ib], -1, self.cn-2)
		xhi = clip(xhi[ib],  0, self.cn-1)
		ylo = clip(ylo[ib], -1, self.cn-2)
		yhi = clip(yhi[ib],  0, self.cn-1)
		assert(all((xlo<xhi)*(ylo<yhi)))

		profile = zeros_like(gridi.ravel())
		# Take mean over the rectangular region...
		profile[ib] = (symm_intimg_rect(self.intimg, xlo, xhi, ylo, yhi, self.csize)
					   / A)
		profile = profile.reshape(gridshape)
		# correct for the count-density of the compiled model...
		profile *= (cpixw * cpixh)

		(outx, inx) = get_overlapping_region(sxlo, sxhi+1, -margin, outw-1 + margin)
		(outy, iny) = get_overlapping_region(sylo, syhi+1, -margin, outh-1 + margin)
		if inx == [] or iny == []:
			return (None, 0, 0)
		x0 = outx.start
		y0 = outy.start
		profile = profile[iny,inx]
		return (profile, x0, y0)


if __name__ == '__main__':
	import astrometry.util.defaults
	from pylab import *

	# Create profiles.
	pexp = CompiledProfile(modelname='exp', profile_func=profile_exp, re=100, nrad=4)
	print 'Created:', pexp
	hdu = pexp.get_hdu()
	hdu.writeto('exp.fits', clobber=True)

	pdev = CompiledProfile(modelname='deV', profile_func=profile_dev, re=100, nrad=8)
	print 'Created:', pdev
	hdu = pdev.get_hdu()
	hdu.writeto('dev.fits', clobber=True)


	# Test profiles.
	pexp = CompiledProfile('exp.fits')
	print 'Read:', pexp
	pdev = CompiledProfile('dev.fits')
	print 'Read:', pdev

	re = 5
	ab = 0.4
	phi = -25.
	x = 25.25
	y = 20.5
	outw = 1000
	outh = 1000
	margin = 10

	(P1,x0,y0) = pexp.sample(re, ab, phi, x, y, outw, outh, margin)
	P1 *= 1e6
	clf()
	imshow(log(P1+1), vmin=0)
	plot([x-x0],[y-y0], 'r.')
	#axis([x-x0-5, x-x0+5, y-y0-5, y-y0+5])
	savefig('exp.png')

	(P2,x0,y0) = pdev.sample(re, ab, phi, x, y, outw, outh, margin)
	P2 *= 1e6
	clf()
	imshow(log(P2+1), vmin=0)
	plot([x-x0],[y-y0], 'r.')
	#axis([x-x0-5, x-x0+5, y-y0-5, y-y0+5])
	savefig('dev.png')


	R = range(10, 100, 5)
	esum = []
	for r in R:
		(P1,x0,y0) = pexp.sample(r, ab, phi, 1000, 1000, 2000, 2000, 1)
		print 'Radius', r, 'shape', P1.shape, 'exp sum', sum(P1)
		esum.append(sum(P1))

	clf()
	plot(R, esum, 'r.')
	savefig('edsum.png')

	print
	dsum = []
	for r in R:
		(P2,x0,y0) = pdev.sample(r, ab, phi, 1000, 1000, 2000, 2000, 1)
		print 'Radius', r, 'shape', P2.shape, 'dev sum', sum(P2)
		dsum.append(sum(P2))

	clf()
	plot(R, esum, 'r.')
	plot(R, dsum, 'b.')
	savefig('edsum.png')
	
