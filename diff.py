

if __name__ == '__main__':
	import matplotlib
	matplotlib.use('Agg')
from math import pi, sqrt, ceil, floor
import pyfits
import pylab as plt
import numpy as np
import matplotlib

from astrometry.sdss import * #DR7, band_name, band_index
from astrometry.util.pyfits_utils import *
from astrometry.util.file import *
from astrometry.util.util import *

from tractor import *
from sdsstractor import *


def main():
	sdss = DR7()

	bandname = 'g'
	bandnum = band_index(bandname)

	# ref
	run,camcol,field = 6955,3,809
	rerun = 0

	ra,dec = 53.202125, -0.365361
	
	fpC = sdss.readFpC(run, camcol, field, bandname).getImage()
	fpC = fpC.astype(float) - sdss.softbias

	###
	wcs = Tan(sdss.getFilename('fpC', run, camcol, field, bandname,
							   rerun=rerun))
	x,y = wcs.radec2pixelxy(ra,dec)
	print 'WCS x,y', x,y

	objs = fits_table(sdss.getFilename('tsObj', run, camcol, field,
									   bandname, rerun=rerun))
									   
	print 'objs', objs

	tsf = sdss.readTsField(run, camcol, field, rerun)
	print 'tsField:', tsf
	astrans = tsf.getAsTrans(bandnum)
	print 'astrans', astrans

	# crazy!
	x1,y1 = astrans.radec_to_pixel(ra, dec)
	print 'x1,y1', x1,y1

	## ok
	r1,d1 = astrans.pixel_to_radec(x,y)
	print 'r1,d1', r1,d1

	## round-trips successfully
	xp,yp = astrans.pixel_to_prime(x, y)
	print 'xp,yp', xp,yp
	x2,y2 = astrans.prime_to_pixel(xp,yp)
	print 'x2,y2', x2,y2

	m1,n1 = astrans.radec_to_munu(ra, dec)
	print 'm1,n1', m1,n1
	r2,d2 = astrans.munu_to_radec(m1,n1)
	print 'r2,d2', r2,d2
	# unchanged... because incl=0 ?

	# crazy
	print
	mu,nu = m1,n1
	xp1,yp1 = astrans.munu_to_prime(mu,nu)
	print 'xp1,yp1', xp1,yp1



if __name__ == '__main__':
	main()

