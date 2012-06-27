#Messier Objects and NGC numbers (and names)
#
#M31 - NGC 224 - (Andromeda Galaxy) 945 fields
#M32 - NGC 221
#M33 - NGC 598 - (Triangulum Galaxy) 146 fields
#M49 - NGC 4472 - Memory error
#M51 - NGC 5194 - (Whirlpool Galaxy) 
#M58 - NGC 4579
#M59 - NGC 4621
#M60 - NGC 4649
#M61 - NGC 4303
#M63 - NGC 5055 - (Sunflower Galaxy)
#M64 - NGC 4826 - (Black Eye Galaxy)
#M65 - NGC 3623
#M66 - NGC 3627
#M74 - NGC 628
#M77 - NGC 1068 85 fields
#M81 - NGC 3031 - (Bode's Galaxy) 22 fields
#M82 - NGC 3034 - (Cigar Galaxy)
#M83 - NGC 5236 - (Southern Pinwheel Galaxy) Not in SDSS!!!
#M84 - NGC 4374
#M85 - NGC 4382
#M86 - NGC 4406
#M87 - NGC 4486 - (Virgo A)
#M88 - NGC 4501
#M89 - NGC 4552
#M90 - NGC 4569
#M91 - NGC 4548
#M94 - NGC 4736
#M95 - NGC 3351
#M96 - NGC 3368
#M98 - NGC 4192
#M99 - NGC 4254
#M100 - NGC 4321
#M101 - NGC 5457 - 26 fields
#M104 - NGC 4594
#M105 - NGC 3379
#M106 - NGC 4258 - 20 fields
#M108 - NGC 3556
#M109 - NGC 3992
#M110 - NGC 205 - 23 fields

# -*- mode: python; indent-tabs-mode: nil -*-
# (this tells emacs to indent with spaces)
import matplotlib
matplotlib.use('Agg')

import os
import logging
import urllib2
import tempfile
import numpy as np
import pylab as plt
import pyfits


def main():
    #ngcs = [224,221,598,4472,5194,4579,4621,4649,4303,5055,4826,3623,3627,628,1068,3031,3034,5236,4374,4382,4406,4486,4501,4552,4569,4548,4736,3351,3368,4192,4254,4321,5457,4594,3379,4258,3556,3992,205]
    #ngcs = [4472,5194,4579,4621,4649,4303,5055,4826,3623,3627,628,3034,4374,4382,4406,4486,4501,4552,4569,4528,4736,3351,3368,4192,4254,4321,4594,3379,3556,3992]
    ngcs = [4579,4621,4649]
    for ngc in ngcs:
        # try:
        #     j = getNGC(ngc)
        #     print j
        #     ra = float(j['RA'][0])
        #     dec = float(j['DEC'][0])
        #     radius = (10.**j['LOG_D25'][0])/10.
        #     rcfs = radec_to_sdss_rcf(ra,dec,radius=math.hypot(radius,13.2/2.),tablefn='dr8fields.fits')
        #     if len(rcfs) > 10:
        #         print "Too many matches NGC: %d" % ngc
        #         print len(rcfs)
        # except:
        #     print "NOT FOUND: NGC: %d" % ngc
        print "Galaxy: NGC-%d" % ngc
        os.system("python -u ngc.py %d --threads 4 --itune1 6 --itune2 6 1>%d.log 2>%d_err.log" % (ngc, ngc, ngc))
        os.system('mv flip-ngc%d messier' % ngc)
        os.system('mv ngc-%d.png messier' % ngc)
        os.system('mv ngc-%d.pickle messier' % ngc)
        #assert(False)

if __name__ == '__main__':
    main()
