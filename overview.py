import numpy as np
import matplotlib.pyplot as plt
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.ngc2000 import *
import os
from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import sdss_galaxy as sg
from tractor import basics as ba


for i,j in enumerate(ngc2000):
    if j['id'] == 4258 and j['is_ngc']:
        print j
        break


cons = .5
print j['ra']
print j['dec']

rcfs = radec_to_sdss_rcf(j['ra'],j['dec'])
print rcfs

x0 = j['ra']
y0 = j['dec']
radius = (98./60.)/60.
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

colors = ['r','g']

for col,rcf in zip(colors,rcfs):
    timg,info = st.get_tractor_image(rcf[0],rcf[1],rcf[2],bandname,useMags=True)
    wcs = timg.getWcs()
    rd = wcs.pixelToPosition(0,0)
    rd2 = wcs.pixelToPosition(width,height)
    print rd
    print rd2

    plt.plot(rd.ra,rd.dec,'+',color=col)
    plt.plot(rd2.ra,rd.dec,'+',color=col)
    plt.plot(rd.ra,rd2.dec,'+',color=col)
    plt.plot(rd2.ra,rd2.dec,'+',color=col)

plt.savefig("view.png")

