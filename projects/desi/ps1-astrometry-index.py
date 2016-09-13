from __future__ import print_function
import matplotlib
matplotlib.use('Agg')

import os
import sys
from glob import glob

from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.sdss import *

# Creates Astrometry.net index files from a Pan-STARRS1 catalog:
# Email from Arjun 2014-09-26
# I just updated my subset of Doug's PS1 catalog. The file is called:
#        ~arjundey/ps1_DECaLS_2.fits
# This contains all PS1 sources that satisfy
#         -7 < DEC < 32
#         15 < {median mag} < 23 and nmag_ok > 0 in EITHER g, r OR z
# and combines Doug's original 2 files (one each for NGC and SGC) into
# a single file.
#
# I did:
#     from astrometry.util.fits import *
#     T=fits_table('~arjundey/ps1_DECaLS_2.fits')
#     T.rmag = T.mean[:,1]
#     T.writeto('tmp/ps1-decals-3.fits')




def _run_one((cmd, outfn)):
    if os.path.exists(outfn):
        print('Output file exists:', outfn, '; not running command')
        return
    os.system(cmd)

def main():
    
    reffn = 'tmp/ps1-decals-3.fits'

    splitpat = 'tmp/ps1-hp%02i.fits'
    
    if False:
        cmd = 'hpsplit %s -o %s -n 2' % (reffn, splitpat)
        print(cmd)
        os.system(cmd)
    
    # hpsplit data/decam/sdss-indexes/calibObj-merge-both{,-2}.fits -o data/decam/sdss-indexes/sdss-hp%02i-ns2.fits -n 2

    # build-astrometry-index -o data/decam/sdss-indexes/index-sdss-z-hp00-2.fits -P 2 -i data/decam/sdss-indexes/sdss-stars-hp00-ns2.fits -S z_psf -H 0 -s 2 -L 20 -I 1408120 -t data/tmp

    cmds = []
    scale = 2
    for hp in range(48):
        indfn = 'tmp/index-ps1-hp%02i-%i.fits' % (hp, scale)
        if os.path.exists(indfn):
            print('Exists:', indfn)
            continue
        catfn = splitpat % hp
        if not os.path.exists(catfn):
            print('No input catalog:', catfn)
            continue
        nstars = 30
        cmd = ('build-astrometry-index -o %s -P %i -i %s -S rmag -H %i -s 2 -L 20 -I 1409260 -t tmp -n %i'
               % (indfn, scale, catfn, hp, nstars))
        print(cmd)
        cmds.append((cmd,indfn))

    mp = multiproc(8)
    mp.map(_run_one, cmds)
        

if __name__ == '__main__':
    main()
    
