import numpy as np
import pylab as plt
import pyfits



def getNGC(ngc):

    rc3 = pyfits.open("rc3.fits")


    print rc3[1].columns

    data = rc3[1].data
    name = 'NGC %d' % (ngc)
    names = data.field("NAME")

    mask = data.field("NAME") == name

    record = data[mask]
    return record




