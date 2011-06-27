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

	#for si,s in enumerate(tractor.getCatalog()):
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
				print '  params:', s.getParams()
				print '  step sizes:', s.getStepSizes(img)
				print '  hash key:', s.hashkey()
				if isinstance(s, st.ExpGalaxy) or isinstance(s, st.DevGalaxy):
					print ' -> pos', s.pos
					print ' -> flux', s.flux
					shape = s.shape
					print ' -> shape', shape
					print ' -> shape.vals', shape.vals
					print ' -> shape.re', shape.re
					print ' -> shape.ab', shape.ab
					print ' -> shape.phi', shape.phi
					
				derivs = s.getParamDerivatives(img)
				for di,d in enumerate(derivs):
					if d is None:
						continue
					plt.clf()
					plt.imshow(d.getImage(), **ima)
					plt.title('Derivative: %s' % d.getName())
					plt.savefig('tractor-s%i-i%i-d%i.png' % (si, ii, di))
			


	
if __name__ == '__main__':
	main()
	
