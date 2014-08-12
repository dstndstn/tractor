import numpy as np
import matplotlib.pyplot as plt
from astrometry.sdss.fields import *
from astrometry.util.ngc2000 import *
import os
from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import galaxy as sg
from tractor import basics as ba


def fieldPlot(x0,y0,radius,ngc):

    rcfs = radec_to_sdss_rcf(x0,y0,tablefn="dr8fields.fits",radius=hypot(radius,13./2.))
    print rcfs

    radius = radius/60.

    bandname = 'r'
    width = 2049
    height = 1489

    print x0
    print y0


    x,y = [],[]
    for theta in np.linspace(0,2*np.pi,100):
        ux = np.cos(theta)
        uy = np.sin(theta)
        x.append(ux*radius+x0)
        y.append(uy*radius+y0)

    plt.plot(x,y)
    plt.plot(x0,y0,'o')



#    colors = ['r','g','y','b','k','m','c']

#    for col,rcf in zip(colors,rcfs):
    for rcf in rcfs:
        timg,info = st.get_tractor_image(rcf[0],rcf[1],rcf[2],bandname,useMags=True)
        wcs = timg.getWcs()
        rd = wcs.pixelToPosition(0,0)
        rd2 = wcs.pixelToPosition(width,height)
        print rd
        print rd2

        plt.plot((rd.ra,rd.ra,rd2.ra,rd2.ra,rd.ra),(rd.dec,rd2.dec,rd2.dec,rd.dec,rd.dec))
        plt.text((rd.ra+rd2.ra)/2.,(rd.dec+rd2.dec)/2.,"Run: %s, Camcol: %s, Field: %s" % (rcf[0],rcf[1],rcf[2]),fontsize=10.,va='center',ha='center')

    #    plt.plot(rd.ra,rd.dec,'+',color=col)
    #    plt.plot(rd2.ra,rd.dec,'+',color=col)
    #    plt.plot(rd.ra,rd2.dec,'+',color=col)
    #    plt.plot(rd2.ra,rd2.dec,'+',color=col)


    print "Saving figure"
    plt.savefig("fieldplot-ngc%d.png" % (ngc))

def main():
    from optparse import OptionParser
    import sys

    parser = OptionParser(usage=('%prog'))
    parser.add_option('-n','--ngc',dest='ngc',type='int')

    opt,args = parser.parse_args()

    ngc = opt.ngc

    j = get_ngc(ngc)
    x0 = j.ra
    y0 = j.dec
    size = j.size
    radius=size/4.


    print x0
    print y0
    print size
    print radius



    if ngc is None:
        parser.print_help()
        print 'Please give an id'
        sys.exit(-1)

    fieldPlot(x0,y0,radius,ngc)


if __name__ == '__main__':
    main()
