from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import unittest

from tractor import *
from tractor.sdss import *
from tractor.galaxy import *

class TractorSdssTest(unittest.TestCase):
    def test_pixpsf(self):
        tim,tinf = get_tractor_image_dr8(94, 2, 520, 'i', psf='kl-pix',
                                         roi=[500,600,500,600], nanomaggies=True)
        psf = tim.getPsf()
        print('PSF', psf)

        for i,(dx,dy) in enumerate([
                (0.,0.), (0.2,0.), (0.4,0), (0.6,0),
                (0., -0.2), (0., -0.4), (0., -0.6)]):
            px,py = 50.+dx, 50.+dy
            patch = psf.getPointSourcePatch(px, py)
            print('Patch size:', patch.shape)
            print('x0,y0', patch.x0, patch.y0)
            H,W = patch.shape
            XX,YY = np.meshgrid(np.arange(W), np.arange(H))
            im = patch.getImage()
            cx = patch.x0 + (XX * im).sum() / im.sum()
            cy = patch.y0 + (YY * im).sum() / im.sum()
            print('cx,cy', cx,cy)
            print('px,py', px,py)

            self.assertLess(np.abs(cx - px), 0.1)
            self.assertLess(np.abs(cy - py), 0.1)
            
            plt.clf()
            plt.imshow(patch.getImage(), interpolation='nearest', origin='lower')
            plt.title('dx,dy %f, %f' % (dx,dy))
            plt.savefig('pixpsf-%i.png' % i)
        




if __name__ == '__main__':
    unittest.main()
