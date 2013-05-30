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


def getName(name,fn="newrc3limited.fits"):

    rc3 = pyfits.open(fn)

    data = rc3[1].data
    names = data.field("NAME")

    mask = data.field("NAME") == name

    record = data[mask]
    print record
    if len(record) ==0:
        print "NONE"
        altnames = data.field("ALT_NAME_1")
        mask = data.field("ALT_NAME_1") == name
        record = data[mask]
    if len(record) ==0:
        print "NONE"
        altnames = data.field("ALT_NAME_2")
        mask = data.field("ALT_NAME_2") == name
        record = data[mask]
    if len(record) ==0:
        print "NONE"
        altnames = data.field("PGC_NAME")
        mask = data.field("PGC_NAME") == name
        record = data[mask]

    rc3.close()
    return record
