#used to find red galaxies and run them in tractor and then compare their results to NSA
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
import pyfits as pyf
from astropysics.obstools import *
import operator
import math
import os
import random
from random import choice
import itertools

if __name__ == '__main__':
    data = pyf.open("nsa-short.fits.gz")[1].data
    a=data.field('RA')
    b=data.field('DEC')
    y = data.field('SERSICFLUX')
    radius = data.field('SERSIC_TH50')
    n=data.field('SERSIC_N')
    p50=data.field('PETROTH50')
    p90=data.field('PETROTH90')
    nsaid=data.field('NSAID')
    extinction=data.field('EXTINCTION')
    good=np.array([True for x in data.field('RA')])
    indx1=np.where(y[:,5] <= 0)
    good[indx1]=False
    indx2=np.where(y[:,3] <= 0)
    good[indx2]=False
    indx3=np.where(radius > 158)
    good[indx3]=False
    print radius[good].shape


#fnugriz
#0123456    
    def Mag1(y):
        return 22.5-2.5*np.log10(np.abs(y))     
    def SB(y):
        return 2.5*np.log10(2*np.pi*y)
    def concentration(x,y):
    	return x/y
    	
    
#magnitudes
umag=Mag1(y[:,2][good])
gmag=Mag1(y[:,3][good])
rmag=Mag1(y[:,4][good])
imag=Mag1(y[:,5][good])
zmag=Mag1(y[:,6][good])
gminusr=gmag-rmag
rminusi=rmag-imag
uminusr=umag-rmag
u=umag-extinction[:,2][good]
g=gmag-extinction[:,3][good]
r=rmag-extinction[:,4][good]
i=imag-extinction[:,5][good]
z=zmag-extinction[:,6][good]
badu=Mag1(y[:,2][good==False])-extinction[:,2][good==False]
badg=Mag1(y[:,3][good==False])-extinction[:,3][good==False]
badr=Mag1(y[:,4][good==False])-extinction[:,4][good==False]
badi=Mag1(y[:,5][good==False])-extinction[:,5][good==False]
badz=Mag1(y[:,6][good==False])-extinction[:,6][good==False]
   
#colors
gi=g-i
ug=u-g
gr=g-r
ri=r-i 
iz=i-z
ur=u-r
badgi=badg-badi
badug=badu-badg
badgr=badg-badr
badri=badr-badi
badiz=badi-badz 

fig1=plt.figure(1)
plt.plot(badgr,badri,'m.', alpha=0.5)
plt.plot(gr, ri,'k.', alpha=0.5)
plt.xlabel(r"$g-r$")
plt.ylabel(r"$r-i$")
plt.xlim(0,1.2)
plt.ylim(-.1,0.8)

#22883, 93093, 129429
color=[x for x in xrange(len(nsaid[good])) if nsaid[good][x]==23057]
for x in color:
    print gr[x],ri[x],ur[x]
    print 'done'
assert(False)
fig1=plt.figure(1)
plt.plot(badgr,badri,'m.', alpha=0.5)
plt.plot(gr, ri,'k.', alpha=0.5)
plt.xlabel(r"$g-r$")
plt.ylabel(r"$r-i$")
plt.xlim(0,1.2)
plt.ylim(-.1,0.8)


#UNPICKLE STARTS HERE
ngcs=[221,628,3034,3351,3368,3379,3623,3556,3627,3992,4192,4254,4321,4374,4382,4406,4472,4486,4501,4548,4552,4569,4579,4594,4621,4736,4826,5055,5194,4148, 521,681]

for ngc in ngcs:
    print ngc
    CG=unpickle_from_file('ngc-%s.pickle'%(ngc)) 
    tot=CG.getBrightness()
    print tot
    pos=CG.getPosition()
    print pos
    dev=CG.brightnessDev
    exp=CG.brightnessExp
    print CG
    assert(False)

#extinction values by filter for Sloan
    sloanu=5.155
    sloang=3.793
    sloanr=2.751
    sloani=2.086
    sloanz=1.479

#get extinction from SFD
    galactic=radectolb(pos[0],pos[1])
    print galactic
    x=get_SFD_dust(galactic[0], galactic[1],dustmap='ebv',interpolate=True)
    correction=[x*sloanu,x*sloang,x*sloanr,x*sloani,x*sloanz]
    corrected_mags=map(operator.sub,tot,correction)
    print 'corrected mags',corrected_mags
    tracgr=tot[1]-tot[2]
    tracri=tot[2]-tot[3]
    tracur=tot[0]-tot[2]
    print 'g-r from trac',tracgr
    print 'r-i from trac',tracri
    print 'u-r from trac', tracur
    gr_corrected=corrected_mags[1]-corrected_mags[2]
    ri_corrected=corrected_mags[2]-corrected_mags[3]
    ur_corrected=corrected_mags[0]-corrected_mags[2]
    print gr_corrected, 'CORRECTED GR COLOR'
    print ri_corrected, 'CORRECTED RI COLOR'
    print ur_corrected, 'CORRECTED UR COLOR'

    color1=[gr_corrected]
    color2=[ri_corrected]
    color3=[tracgr]
    color4=[tracri]
    xyz=['red','yellow','blue','cyan','magenta','green','purple','orange']
    rando=choice(xyz)
    choosing=[t for t in xrange(len(xyz)) if rando == xyz[t]]

    #this plots de-reddened NSA vs. de-reddened tractor
    for t in choosing:
        plt.plot(color1, color2,'*',linestyle='-', color=rando, ms=12, markeredgecolor=xyz[t] , markeredgewidth=1,markerfacecolor='none')
    plt.savefig('locater.pdf')
    os.system('cp locater.pdf public_html/')


    # plt.savefig('locate.pdf')
    # os.system('cp locate.pdf public_html/')

    # this plots de-reddened NS Atlas vs. original tractor colors
    # for t in choosing:
    #     plt.plot(color3, color4,'*',linestyle='-', color=rando, ms=12, markeredgecolor=xyz[t] , markeredgewidth=1,markerfacecolor='none')
    # plt.savefig('locate2.pdf')
    # os.system('cp locate2.pdf public_html/')
 

# locate1=[i for i in xrange(len(gr))]
# for i in locate1:
#      if 0.877 < gr[i] < 0.88:
#          if 0.44 < ri[i] < 0.47:
#              print gr[i],ri[i],nsaid[good][i],a[good][i],b[good][i] 
#              plt.plot(gr[i],ri[i],'bs',ms=10,markeredgecolor='blue',markeredgewidth=1,markerfacecolor='none')


