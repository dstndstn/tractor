import numpy as np
import pylab as plt
import pyfits



def getNGC(ngc):

    rc3 = pyfits.open("rc3.fits")



    data = rc3[1].data
    name = 'NGC %d' % (ngc)
    names = data.field("NAME")

    mask = data.field("NAME") == name

    record = data[mask]
    rc3.close()
    return record


def getName(name):

    rc3 = pyfits.open("rc3limited.fits")

    data = rc3[1].data
    names = data.field("NAME")

    mask = data.field("NAME") == name

    record = data[mask]
    print mask
    print record
    if len(record) ==0:
        print "NONE"
        altnames = data.field("ALT_NAME_1")
        mask = data.field("ALT_NAME_1") == name
        record = data[mask]
    rc3.close()
    return record
