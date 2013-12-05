if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
import numpy as np
import pylab as plt
from astrometry.util.file import *
from tractor import *
from tractor import sdss as st









def main():
	from optparse import OptionParser
	import sys

	parser = OptionParser(usage=('%prog <tractorX.pickle>'))
	parser.add_option('--derivs', dest='derivs', action='store_true',
					  default=False, help='Plot derivatives for each source?')
	parser.add_option('--optstep', dest='optstep', action='store_true',
					  default=False,
					  help='Plot difference in one optimization step')
	parser.add_option('--source', '-s', dest='sources', action='append',
					  default=[], type=int,
					  help='Process only the given sources')

	opt,args = parser.parse_args()
	if len(args) != 1:
		parser.print_help()
		print 'Need tractor.pickle file'
		sys.exit(-1)

	pfn = args[0]
	tractor = unpickle_from_file(pfn)

	srci = opt.sources
	if len(srci) == 0:
		srci = range(len(tractor.getCatalog()))

	if opt.optstep:
		mods0 = tractor.getModelImages()
		allparams = tractor.getAllDerivs()
		X = tractor.optimize(allparams)
		oldp = tractor.stepParams(X)
		mods1 = tractor.getModelImages()
		tractor.revertParams(oldp)
		tractor.stepParams(X, alpha=1e-3)
		mods2 = tractor.getModelImages()
		tractor.revertParams(oldp)

		for ii,(m0,m1,m2) in enumerate(zip(mods0, mods1, mods2)):
			ima = dict(interpolation='nearest', origin='lower')
			xx = m0.copy().ravel()
			xx.sort()
			mn = xx[int(0.25 * len(xx))]
			mx = xx[int(0.99 * len(xx))]

			plt.clf()
			plt.imshow(m0, vmin=mn, vmax=mx, **ima)
			plt.gray()
			plt.colorbar()
			plt.savefig('tractor-i%i-o1.png' % (ii))

			plt.clf()
			plt.imshow(m1, vmin=mn, vmax=mx, **ima)
			plt.gray()
			plt.colorbar()
			plt.savefig('tractor-i%i-o2.png' % (ii))

			plt.clf()
			plt.imshow(m2, vmin=mn, vmax=mx, **ima)
			plt.gray()
			plt.colorbar()
			plt.savefig('tractor-i%i-o4.png' % (ii))

			np.seterr(all='warn')

			dd = np.abs(m1-m0).ravel()
			dd.sort()
			mx = dd[int(0.9 * len(dd))]
			print 'dd range', dd[0], dd[-1]
			plt.clf()
			plt.imshow(m1-m0, vmin=-mx, vmax=mx, **ima)
			plt.gray()
			plt.colorbar()
			plt.savefig('tractor-i%i-o3.png' % (ii))

			dd = np.abs(m2-m0).ravel()
			dd.sort()
			mx = dd[int(0.9 * len(dd))]
			print 'dd range', dd[0], dd[-1]
			plt.clf()
			plt.imshow(m2-m0, vmin=-mx, vmax=mx, **ima)
			plt.gray()
			plt.colorbar()
			plt.savefig('tractor-i%i-o5.png' % (ii))



	for si in srci:
		s = tractor.getCatalog()[si]
		print
		print 'Making plots for', s
		for ii,img in enumerate(tractor.getImages()):
			print '  In image', img
			patch = s.getModelPatch(img)

			ima = dict(interpolation='nearest', origin='lower')

			plt.clf()
			plt.imshow(patch.getImage(), extent=patch.getExtent(), **ima)
			plt.gray()
			plt.savefig('tractor-s%i-i%i-a.png' % (si, ii))

			image = img.getImage()
			I = np.zeros_like(image)
			patch.addTo(I)
			plt.clf()
			plt.imshow(I, **ima)
			plt.savefig('tractor-s%i-i%i-b.png' % (si, ii))

			if opt.derivs:
				print 'Source', si
				print 'Getting derivatives for', s
				p0 = s.getParams()
				print '  params:', s.getParams()
				print '  step sizes:', s.getStepSizes(img)
				#print '  hash key:', s.hashkey()
				if isinstance(s, st.ExpGalaxy) or isinstance(s, st.DevGalaxy):
					print ' -> pos', s.pos
					#print '    pos params', s.pos.getParams()
					print ' -> flux', s.flux
					#print '    flux params', s.flux.getParams()
					shape = s.shape
					print ' -> shape', shape
					#print '    shape params', shape.getParams()
					#print ' -> shape.vals', shape.vals
					#print ' -> shape.re', shape.re
					#print ' -> shape.ab', shape.ab
					#print ' -> shape.phi', shape.phi
					
				derivs = s.getParamDerivatives(img)
				p1 = s.getParams()
				assert(p0 == p1)
				for di,d in enumerate(derivs):
					if d is None:
						continue
					plt.clf()
					dimg = d.getImage()
					mx = max(abs(dimg.min()), abs(dimg.max()))
					plt.imshow(dimg, vmin=-mx, vmax=mx,
							   extent=patch.getExtent(), **ima)
					plt.title('Derivative: %s' % d.getName())
					plt.savefig('tractor-s%i-i%i-d%i.png' % (si, ii, di))

					


	
if __name__ == '__main__':
	main()
	
