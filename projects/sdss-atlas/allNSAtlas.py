from __future__ import print_function
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

import os
import numpy as np
import pylab as plt
import pyfits
import sys
import traceback

from general import generalNSAtlas
from halflight import halflight
#from addtodb import add_to_table_nsatlas

def main():
    nsgals2=[]
    f = open("targets2.txt","r")
    for line in f:
        nsgals2.append(int(line))
    print(nsgals2)
    for entry in nsgals2:
        try:
            print(entry)
            newentry = 'NSA_ID_%s' % entry
            if os.path.exists("NSAtlas_Output/%s.pickle" % newentry):
                print("%s has run already!" % newentry)
                continue
            print('running tractor for %s' %entry)
            generalNSAtlas(entry,itune1=6,itune2=6,nocache=True)
            os.system('cp flip-%s.pdf NSAtlas_Output' % newentry)
            os.system('cp %s.png NSAtlas_Output' % newentry)
            os.system('cp %s.pickle NSAtlas_Output' % newentry)
            halflight(newentry,direc='NSAtlas_Output')
        except AssertionError:
            print(traceback.print_exc())
            print("Tractor failed on entry: %s" % newentry)
            continue
        except IndexError:
            print(traceback.print_exc())
            print("Tractor failed on entry: %s" % newentry)
            continue
        #add_to_table_nsatlas(newentry)

if __name__ == '__main__':
    main()
