if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

import os
import numpy as np
import pylab as plt
import pyfits
import sys

from general import generalNSAtlas
from halflight import halflight
#from addtodb import add_to_table_nsatlas

def main():
    #nsgals = [167,20315,24188,39462,44000,131926,133189,133385,133469,133678,140045,149936,151192,151193,152228,158585]
    nsgals2 = [39462,131926,140045,149936,151192,151193,152228,158585]

    #133678 seems not to be in the table???
    for entry in nsgals2:
        try:
            print entry
            newentry = 'NSA_ID_%s' % entry
            print 'running tractor for %s' %entry
            generalNSAtlas(entry,itune1=6,itune2=6,nocache=True)
            os.system('cp flip-%s.pdf NSAtlas_Output' % newentry)
            os.system('cp %s.png NSAtlas_Output' % newentry)
            os.system('cp %s.pickle NSAtlas_Output' % newentry)
            halflight(newentry,direc='NSAtlas_Output')
        except AssertionError:
            print 'Tractor failed on %s' %entry
        #add_to_table_nsatlas(newentry)

if __name__ == '__main__':
    main()
