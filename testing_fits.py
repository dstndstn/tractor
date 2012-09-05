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
    data=pyf.open('large_galaxies.fits')
    data2=data[1].data
    mags=data2.field('CG_TOTMAGS')
    extinction=data2.field('CG_EXTINCTION')
    name=data2.field('RC3_NAME')
    sb=data2.field('CG I-SB')
    half_light=data2.field('CG_R50S')
    concentration=data2.field('CG_CONC')
    #print mags, len(mags)
    gr_original=mags[:,1]-mags[:,2]
    ri_original=mags[:,2]-mags[:,3]
    corrected_g=mags[:,1]-extinction[:,1]
    corrected_r=mags[:,2]-extinction[:,2]
    corrected_i=mags[:,3]-extinction[:,3]
    gr_corrected=corrected_g-corrected_r
    ri_corrected=corrected_r-corrected_i
    gi_corrected=corrected_g-corrected_i
    print len(gr_original)
    #assert(False)
    #print len(gr_corrected)
    
    def mu_50(i,r):
        return i+2.5*(log10(pi*r**2))


    # outlier=[i for i in xrange(len(ri_corrected))]
    # for i in outlier:
    #     if ri_corrected[i] > 0.8:
    #         print gr_corrected[i],ri_corrected[i], name[i]
    #     #if ri_corrected[i] > 0.8
    # outlier2=[i for i in xrange(len(gr_original))]
    # for i in outlier2:
    #     if gr_original[i] > 1.0:
    #         print gr_original[i],ri_original[i],name[i]
    # plt.plot(gr_original,ri_original,'r.',alpha=0.5,label='non-dereddened')
    # plt.plot(gr_corrected,ri_corrected,'b.',alpha=0.5,label='dereddened')
    # plt.annotate('PGC 70104',xy=(0.2,-0.00762749),xytext=(0.4,0.0),arrowprops=dict(facecolor='black'))
    # plt.annotate('UGC 11861',xy=(1.119,.6218),xytext=(0.7,.6218),arrowprops=dict(facecolor='black'))
    # plt.annotate('MCG 1-57-?',xy=(.62,.850),xytext=(.4,.85),arrowprops=dict(facecolor='black'))
    # plt.xlabel(r"$g-r$")
    # plt.ylabel(r"$r-i$")
    # plt.title('all RC3 galaxies to date')
    # plt.legend()
    # plt.savefig('allRC3_colorplot.pdf')
    # os.system('cp allRC3_colorplot.pdf public_html/')
    

plt.subplots_adjust(wspace=0,hspace=0)
matplotlib.rcParams['font.size'] = 9

plt.subplot(441)
plt.xlabel(r"$g-r$")
plt.ylabel(r"$\mu_i$")
plt.xlim(0.2,0.9)
plt.yticks([18,19,20,21,22],[18,19,20,21,22])
plt.ylim(17,23)
plt.plot(gr_corrected,sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(442)
plt.tick_params(axis='both',labelbottom='off',labelleft='off')
plt.xlabel(r"$g-i$")
plt.xlim(0.3,1.4)
plt.ylim(17,23)
plt.plot(gi_corrected,sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(443)
plt.tick_params(axis='both',labelbottom='off',labelleft='off')
plt.xlabel(r"$r-i$")
plt.xlim(0,80)
plt.ylim(17,23)
plt.plot(half_light[:,3],sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(444)
plt.tick_params(axis='both',labelleft='off')
plt.xticks([2.5,3.0,3.5,4.0,4.5],[2.5,3.0,3.5,4.0,4.5])
plt.xlabel(r"$C$")
plt.xlim(2,5)
plt.ylim(17,23)
plt.plot(concentration[:,3],sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(445)
# plt.plot(badug,badiz,'m.',alpha=0.5,ms=1.5)
# plt.plot(ug,iz, 'k.',alpha=0.5,ms=1.5)
plt.xlim(0.2,0.9)
plt.ylim(2,5)
plt.yticks([2.5,3.0,3.5,4.0,4.5],[2.5,3.0,3.5,4.0,4.5])
plt.ylabel(r"$C(r90/r50)$")
plt.plot(gr_corrected,concentration[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(446)
plt.tick_params(axis='both',labelbottom='off',labelleft='off')
#plt.plot(badgr,badiz,'m.',alpha=0.5,ms=1.5)
#plt.plot(gr,iz,'k.',alpha=0.5,ms=1.5)
plt.xlim(0.3,1.4)
plt.ylim(2,5)
plt.xlabel(r"$g-r$")
plt.plot(gi_corrected,concentration[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(447)
plt.tick_params(axis='both',labelleft='off')
# plt.plot(badri,badiz,'m.',alpha=0.5,ms=1.5)
# plt.plot(ri,iz,'k.',alpha=0.5,ms=1.5)
plt.xlim(0,80)
plt.xticks([10,20,30,40,50,60,70],[10,20,30,40,50,60,70])
plt.ylim(2,5)
plt.xlabel(r"$half-light\,radius$")
plt.plot(half_light[:,3],concentration[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(449)
# plt.plot(badug,badri,'m.',alpha=0.5,ms=1.5)
# plt.plot(ug,ri,'k.',alpha=0.5,ms=1.5)
plt.xlim(0.2,0.9)
plt.ylim(0,80)
plt.yticks([10,20,30,40,50,60,70],[10,20,30,40,50,60,70])
plt.ylabel(r"$half-light\,radius$")
plt.plot(gr_corrected,half_light[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(4,4,10)
plt.tick_params(axis='both',labelleft='off')
# plt.plot(badgr,badri,'m.',alpha=0.5,ms=1.5)
# plt.plot(gr,ri,'k.',alpha=0.5,ms=1.5)		
plt.xlim(0.3,1.4)
# plt.xticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
plt.ylim(0,80)
plt.xlabel(r"$g-i$")
plt.plot(gi_corrected,half_light[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')

plt.subplot(4,4,13)
# plt.plot(badug,badgr,'m.',alpha=0.5,ms=1.5)
# plt.plot(ug,gr,'k.',alpha=0.5,ms=1.5)
plt.xlim(0.2,0.9)
plt.xlabel(r"$g-r$")
# plt.yticks([0.2,0.4,0.6,0.8,1.0],[0.2,0.4,0.6,0.8,1.0])
plt.ylim(0.3,1.4)
# plt.xticks([0.5,1.0,1.5,2.0],[0.5,1.0,1.5,2.0])
plt.ylabel(r"$g-i$")
plt.plot(gr_corrected,gi_corrected,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
plt.suptitle('Comparison of all objects in RC3 to date')

plt.savefig('triangle_rc3.pdf')
os.system('cp triangle_rc3.pdf public_html/')
