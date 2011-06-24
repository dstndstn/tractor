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
					  
	opt,args = parser.parse_args()
	if len(args) != 1:
		parser.print_help()
		print 'Need tractor.pickle file'
		sys.exit(-1)

	pfn = args[0]
	tractor = unpickle_from_file(pfn)

	for si,s in enumerate(tractor.getCatalog()):
		print
		print 'Making plots for', s
		for ii,img in enumerate(tractor.getImages()):
			print '  In image', img
			patch = s.getModelPatch(img)

			ima = dict(interpolation='nearest', origin='lower')

			plt.clf()
			plt.imshow(patch.getImage(), extent=patch.getExtent(), **ima)
			plt.savefig('tractor-s%i-i%i-a.png' % (si, ii))

			image = img.getImage()
			I = np.zeros_like(image)
			patch.addTo(I)
			plt.clf()
			plt.imshow(I, **ima)
			plt.savefig('tractor-s%i-i%i-b.png' % (si, ii))


	
if __name__ == '__main__':
	main()
	
