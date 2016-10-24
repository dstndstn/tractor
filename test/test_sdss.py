from __future__ import print_function

import unittest
import os

from tractor import *
from tractor.sdss import *
from tractor.galaxy import *

from astrometry.sdss import DR8

class SdssTest(unittest.TestCase):
    def setUp(self):
        tdir = os.path.join(os.path.dirname(__file__), 'data-sdss')
        self.sdss = DR8(basedir=tdir)
        #self.sdss.saveUnzippedFiles(tdir)
        
    def test_image(self):
        run,camcol,field = (1000,1,100)
        args = (run, camcol, field, 'r')
        roi = (0,100,0,100)
        # Manually created sub-frame file, so the sky vector isn't the right
        # size... hack via invvarAtCenter.
        
        kwargs = dict(roi=roi, psf='dg', sdss=self.sdss,
                      retry_retrieve=False, invvarAtCenter=True, retrieve=False)
        tim1,i1 = get_tractor_image_dr8(*args, nanomaggies=True, **kwargs)
        tim2,i2 = get_tractor_image_dr8(*args, nanomaggies=False, **kwargs)
        print('tim1', tim1.name)
        print('tim2', tim2.name)

        self.assertIsInstance(tim1.photocal, LinearPhotoCal)
        self.assertIsInstance(tim2.photocal, MagsPhotoCal)
        self.assertEqual(tim1.shape, (100,100))
        self.assertEqual(tim2.shape, (100,100))
        
        # Slightly larger ROI to get some
        roi = (0,150,0,150)
        args = (run, camcol, field)
        kwargs = dict(bandname='r', sdss=self.sdss, retrieve=False,
                      roi=roi, useObjcType=True, checkFiles=False)
        srcs1 = get_tractor_sources_dr8(*args, nanomaggies=True,  **kwargs)
        srcs2 = get_tractor_sources_dr8(*args, nanomaggies=False, **kwargs)

        print('srcs1:', srcs1)
        print('srcs2:', srcs2)

        self.assertEqual(len(srcs1), 6)
        self.assertEqual(len(srcs2), 6)
        self.assertIsInstance(srcs1[0], PointSource)
        self.assertIsInstance(srcs2[0], PointSource)
        p1 = srcs1[0]
        p2 = srcs2[0]
        self.assertIsInstance(p1.getBrightness(), NanoMaggies)
        self.assertIsInstance(p2.getBrightness(), Mags)

        # The important thing is interoperability of Brightness and Photocal...
        counts1 = tim1.getPhotoCal().brightnessToCounts(p1.getBrightness())
        print('Counts1:', counts1)
        counts2 = tim2.getPhotoCal().brightnessToCounts(p2.getBrightness())
        print('Counts2:', counts2)

        self.assertTrue(np.abs(counts1 - counts2) < 1e-3)
        self.assertTrue(np.abs(counts1 - 541.697) < 1e-3)
        
        
if __name__ == '__main__':
    unittest.main()
