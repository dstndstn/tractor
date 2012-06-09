import matplotlib
import numpy as np
import pylab as plt
import pyfits as pyf

from astrometry.libkd import spherematch as sm


def tychoMatch(ra,dec,rad):
    data = pyf.open("tycho2-cut.fits")[1].data
    RA=data.field('RA')
    DEC=data.field('DEC')
    MAG=data.field('MAG')

    ra1=ra
    dec1=dec
    matchrad = rad
    I1,I2,d = sm.match_radec(ra1,dec1, RA,DEC, matchrad)

    print pyf.open("tycho2-cut.fits")[1].columns.names

    
    return RA[I2],DEC[I2],MAG[I2]




#find all bright stars that are:
#between specified RA and DEC
#that is saturated
#that is a child
