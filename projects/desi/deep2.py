if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys

import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *
from astrometry.util.ttime import *

from tractor import *

from wise.allwisecat import allwise_catalog_radecbox
from wise.unwise import *

import logging

if __name__ == '__main__':
    tiledir = 'wise-coadds'

    tile = '2145p530'
    band = 1
    
    tim = get_unwise_tractor_image(tiledir, tile, band,
                                   roiradecbox=[213.2, 214.5, 51.9, 52.4])
    print 'tim', tim
    print 'tim ROI', tim.roi
    
