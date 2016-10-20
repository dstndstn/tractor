from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import logging
import numpy as np
import pylab as plt

import pyfits

from astrometry.util.file import *

direc = 'RC3_Output/uppickle/'
files = os.listdir(direc)

colors=[]
concs=[]

for galaxy in files:
    CG,r50s,r90s,conc = unpickle_from_file(os.path.join(direc,galaxy))
    colors.append(CG.getBrightness()[1]-CG.getBrightness()[3])
    concs.append(conc[3])

print(colors)
print(concs)

plt.plot(colors,concs, 'r+')
plt.savefig("colorvconci.png")
    
