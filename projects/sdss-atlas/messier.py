#Messier Objects and NGC numbers (and names)
#
#M31 - NGC 224 - (Andromeda Galaxy) 945 fields
#M32 - NGC 221 - Near andromeda-worked
#M33 - NGC 598 - (Triangulum Galaxy) 146 fields
#M49 - NGC 4472 - Memory error- Segmentation fault (core dumped)- 7/11, worked -7/12
#M51 - NGC 5194 - (Whirlpool Galaxy) - worked 7/10
#M58 - NGC 4579 - Memory error - worked 7/7 !
#M59 - NGC 4621 - Worked
#M60 - NGC 4649 - Neighboring Galaxy-Worked, bad fit
#M61 - NGC 4303 - Assertion Error 7/9, 7/12
#M63 - NGC 5055 - (Sunflower Galaxy) - worked - 7/11 
#M64 - NGC 4826 - (Black Eye Galaxy) - Memory error - worked -7/11
#M65 - NGC 3623 - Memory error - worked 7/10
#M66 - NGC 3627 - Memory error - worked 7/11
#M74 - NGC 628 - memory error 7/11- worked 7/12
#M77 - NGC 1068 85 fields
#M81 - NGC 3031 - (Bode's Galaxy) 22 fields
#M82 - NGC 3034 - (Cigar Galaxy) - Segmentation fault(core dumped) - worked 7/13
#M83 - NGC 5236 - (Southern Pinwheel Galaxy) Not in SDSS!!!
#M84 - NGC 4374 - worked
#M85 - NGC 4382 - mem error- worked 7/11
#M86 - NGC 4406 - worked 7/11
#M87 - NGC 4486 - (Virgo A) - mem error- worked 7/12
#M88 - NGC 4501 - worked
#M89 - NGC 4552 - memory error 7/9 - worked 7/12
#M90 - NGC 4569- 9 fields - mem error 7/12 -worked
#M91 - NGC 4548 - worked
#M94 - NGC 4736 - worked 7/5
#M95 - NGC 3351 - worked
#M96 - NGC 3368 - worked
#M98 - NGC 4192 - worked 7/5  
#M99 - NGC 4254 - worked
#M100 - NGC 4321 - worked 7/5
#M101 - NGC 5457 - 26 fields
#M104 - NGC 4594 - (sombrero galaxy) - worked 7/7  
#M105 - NGC 3379 - worked 7/7
#M106 - NGC 4258 - 20 fields
#M108 - NGC 3556 -  8 fields - worked 7/12
#M109 - NGC 3992 - worked
#M110 - NGC 205 - 23 fields

# -*- mode: python; indent-tabs-mode: nil -*-
# (this tells emacs to indent with spaces)
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import logging
import urllib2
import tempfile
import numpy as np
import pylab as plt
import pyfits

from astrometry.util.file import *
from astrometry.util.multiproc import multiproc

from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import galaxy as sg
from tractor import basics as ba
from tractor.overview import fieldPlot
from tractor.tychodata import tychoMatch
from tractor.rc3 import getNGC
from astrometry.util.ngc2000 import *
from astrometry.sdss.fields import *
import optparse


def main():
    #ngcs=[4472,5055,4826,3627,628,3034,4382,4406,4486]
    #done 4736,4192,4321,4594, 4579, 3379, 3623, 5194
    #next =[5194,5055,3034,4826]
    ngcs=[3623,3627,4569,4406]
    for ngc in ngcs:
        # try:
        #     j = getNGC(ngc)
        #     #print j
        #     ra = float(j['RA'][0])
        #     dec = float(j['DEC'][0])
        #     radius = (10.**j['LOG_D25'][0])/10.
        #     rcfs = radec_to_sdss_rcf(ra,dec,radius=math.hypot(radius,13.2/2.),tablefn='dr8fields.fits')
        #     if len(rcfs) > 4:
        #         print "Too many matches NGC: %d" % ngc
        #         print len(rcfs)
        #     else:
        #         print "Good amount of matches NGC: %d" % ngc
        #         print len(rcfs)
        # except:
        #     print "NOT FOUND: NGC: %d" % ngc
        print("Galaxy: NGC-%d" % ngc)
        os.system("python -u ngc.py %d --threads 4 --itune1 6 --itune2 6 1>%d.log 2>%d_err.log" % (ngc, ngc, ngc))
   #    os.system('cp flip-ngc%d.pdf messier' % ngc)
   #    os.system('cp ngc-%d.png messier' % ngc)
   #    os.system('cp ngc-%d.pickle messier' % ngc)
        os.system('cp flip-ngc%d.pdf TractorOutput' % ngc)
        os.system('cp ngc-%d.png TractorOutput' % ngc)
        os.system('cp flip-ngc%d.tex TractorOutput' % ngc)
     #  os.system('cp ngc-%d.pickle TractorOutput' % ngc)

        #assert(False)

if __name__ == '__main__':
    main()
