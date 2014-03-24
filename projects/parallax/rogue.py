import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import numpy as np
import pylab as plt

import os

import fitsio

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.util.run_command import *
from astrometry.util.starutil_numpy import *
from astrometry.util.ttime import *
from astrometry.libkd.spherematch import *
from astrometry.blind.plotstuff import *

from unwise_coadd import get_wise_frames, get_l1b_file
from wise.wise import *

if __name__ == '__main__':
    r,d = 133.795, -7.245
    sz = 0.01

    wisedir = 'wise-frames'
    wfn = 'rogue-frames.fits'
    if os.path.exists(wfn):
        W = fits_table(wfn)
    else:
        W = get_wise_frames(r-sz, r+sz, d-sz, d+sz, margin=1.2)
        W.writeto(wfn)
    print len(W), 'WISE frames'

    band = 2
    W.cut(W.band == band)
    roi = [r-sz, r+sz, d-sz, d+sz]

    if not 'inroi' in W.get_columns():
        W.inroi = np.zeros(len(W), bool)
        for i,w in enumerate(W):
            fn = get_l1b_file(wisedir, w.scan_id, w.frame_num, band)
            print fn

            basefn = fn.replace('-int-1b.fits', '')

            fns = [fn, basefn + '-unc-1b.fits.gz', basefn + '-msk-1b.fits.gz']
            for fn in fns:
                if not os.path.exists(fn):
                    cmd = 'rsync -RLrvz carver:unwise/./%s .' % fn
                    print cmd
                    os.system(cmd)

            tim = read_wise_level1b(basefn, radecroi=roi, nanomaggies=True,
                                    mask_gz=True, unc_gz=True)
            
            print 'Got', tim
            if tim is None:
                continue
            
            W.inroi[i] = True
        W.writeto(wfn)
    
    W.cut(W.inroi)

    tims = []
    for i,w in enumerate(W):
        fn = get_l1b_file(wisedir, w.scan_id, w.frame_num, band)
        basefn = fn.replace('-int-1b.fits', '')
        tim = read_wise_level1b(basefn, radecroi=roi, nanomaggies=True,
                                mask_gz=True, unc_gz=True)
        print 'Got', tim
        tims.append(tim)
