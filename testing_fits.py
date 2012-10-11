from unpickle_scale import unpickle4color
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
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
    ra=data2.field('RC3_RA')
    ra2=data2.field('CG_RA')
    dec=data2.field('RC3_DEC')
    sb=data2.field('CG I-SB')
    half_light=data2.field('CG_R50S')
    r90=data2.field('CG_R90S')
    concentration=data2.field('CG_CONC')
    gr_original=mags[:,1]-mags[:,2]
    ri_original=mags[:,2]-mags[:,3]
    corrected_g=mags[:,1]-extinction[:,1]
    corrected_r=mags[:,2]-extinction[:,2]
    corrected_i=mags[:,3]-extinction[:,3]
    gr_corrected=corrected_g-corrected_r
    ri_corrected=corrected_r-corrected_i
    gi_corrected=corrected_g-corrected_i
    print len(name)
    print len(ra)
    nsadata=pyf.open('nsa_galaxies.fits')
    data3=nsadata[1].data
    #cols=data3[1].columns
    #print data3[6]
    #print data3[6]
    #print cols.names
    


    def mu_50(i,r):
        return i+2.5*(log10(pi*r**2))

    def quantile(x,q):
        #print len(x)
        i=len(sorted(x))*q
        z=int(floor(i))
        #print z
        return sorted(x)[z]
    
    def plotq(z):
        plt.axvline(x=quantile(z,0.25),color='black',alpha=0.5)
        plt.axvline(x=quantile(z,0.5),color='black',alpha=0.5)
        plt.axvline(x=quantile(z,0.75),color='black',alpha=0.5)
        return 'plotted'
    
    def qofq(f,g):
        x1,x2=plt.xlim()
        y1,y2=plt.ylim()
        a=[t for t in xrange(len(f)) if f[t] < quantile(f,0.25)]
        sample00=sorted(g[a])
        #print sample00, len(sample00)
        print quantile(sample00,0.25)
        print quantile(sample00,0.5)
        print quantile(sample00,0.75)
        plt.axhline(y=quantile(sample00,0.25),xmin=0,xmax=(quantile(f,0.25)-x1)/(x2-x1),color='black',alpha=0.5)
        plt.axhline(y=quantile(sample00,0.5),xmin=0,xmax=(quantile(f,0.25)-x1)/(x2-x1),color='black',alpha=0.5)
        plt.axhline(y=quantile(sample00,0.75),xmin=0,xmax=(quantile(f,0.25)-x1)/(x2-x1),color='black',alpha=0.5)

        b=[t for t in xrange(len(f)) if quantile(f,0.25) < f[t] < quantile(f,0.5)]
        sample1=sorted(g[b])
        print quantile(sample1,0.25)
        print quantile(sample1,0.5)
        print quantile(sample1,.75)
        plt.axhline(y=quantile(sample1,0.25),xmin=(quantile(f,0.25)-x1)/(x2-x1),xmax=(quantile(f,0.5)-x1)/(x2-x1),color='black',alpha=0.5)
        plt.axhline(y=quantile(sample1,0.5),xmin=(quantile(f,0.25)-x1)/(x2-x1),xmax=(quantile(f,0.5)-x1)/(x2-x1),color='black',alpha=0.5)
        plt.axhline(y=quantile(sample1,0.75),xmin=(quantile(f,0.25)-x1)/(x2-x1),xmax=(quantile(f,0.5)-x1)/(x2-x1),color='black',alpha=0.5)        

        c=[t for t in xrange(len(f)) if quantile(f,0.5) < f[t] < quantile(f,0.75)]
        sample2=sorted(g[c])
        print quantile(sample2,0.25)
        print quantile(sample2,0.5)
        print quantile(sample2,0.75)
        plt.axhline(y=quantile(sample2,0.25),xmin=(quantile(f,0.5)-x1)/(x2-x1),xmax=(quantile(f,0.75)-x1)/(x2-x1),color='black',alpha=0.5)
        plt.axhline(y=quantile(sample2,0.5),xmin=(quantile(f,0.5)-x1)/(x2-x1),xmax=(quantile(f,0.75)-x1)/(x2-x1),color='black',alpha=0.5)
        plt.axhline(y=quantile(sample2,0.75),xmin=(quantile(f,0.5)-x1)/(x2-x1),xmax=(quantile(f,0.75)-x1)/(x2-x1),color='black',alpha=0.5)        
        
        d=[t for t in xrange(len(f)) if f[t] > quantile(f,0.75)]
        sample3=sorted(g[d])
        print quantile(sample3,0.25)
        print quantile(sample3,0.5)
        print quantile(sample3,0.75)
        plt.axhline(y=quantile(sample3,0.25),xmin=(quantile(f,0.75)-x1)/(x2-x1),xmax=1,color='black',alpha=0.5)
        plt.axhline(y=quantile(sample3,0.5),xmin=(quantile(f,0.75)-x1)/(x2-x1),xmax=1,color='black',alpha=0.5)
        plt.axhline(y=quantile(sample3,0.75),xmin=(quantile(f,0.75)-x1)/(x2-x1),xmax=1,color='black',alpha=0.5)        

        return None

    galaxies=np.array([True for x in ra])
    leoa=np.where(name=='LEO A')
    ugc7332=np.where(name=='UGC 7332')
    ugc9394=np.where(name=='UGC 9394')
    ngc52=np.where(name=='NGC 52')
    ngc1022=np.where(name=='NGC 1022')
    a=[leoa,ugc7332,ugc9394,ngc52]
    for x in a:
        print gr_corrected[x], gi_corrected[x]
    print half_light[:,3][ngc1022]
    assert(False)
    # size=np.where(half_light[:,3] < 60.)
    # sblimit=np.where(sb < 23.)
    # galaxies[size]=False
    # galaxies[sblimit]=False
    # print name[galaxies]
    # grlimit=np.where(gr_corrected >.62)
    # grlimit2=np.where(gr_corrected < .6)
    # galaxies[grlimit]=False
    # galaxies[grlimit2]=False
    # gilimit=np.where(gi_corrected > 1.0)
    # gilimit2=np.where(gi_corrected <0.95)
    # galaxies[gilimit]=False
    # galaxies[gilimit2]=False
    # radius=np.where(half_light[:,2] > 150.)
    # galaxies[radius]=False
                                    
    # plt.annotate('PGC 70104',xy=(0.2,-0.00762749),xytext=(0.4,0.0),arrowprops=dict(facecolor='black'))
    # plt.annotate('UGC 11861',xy=(1.119,.6218),xytext=(0.7,.6218),arrowprops=dict(facecolor='black'))
    # plt.annotate('MCG 1-57-?',xy=(.62,.850),xytext=(.4,.85),arrowprops=dict(facecolor='black'))
    # plt.xlabel(r"$g-r$")
    # plt.ylabel(r"$r-i$")
    # plt.title('all RC3 galaxies to date')
    # plt.legend()
    # plt.savefig('allRC3_colorplot.pdf')
    # os.system('cp allRC3_colorplot.pdf public_html/')
    
plt.figure(1)
plt.subplots_adjust(wspace=0,hspace=0)
matplotlib.rcParams['font.size'] = 9

# plt.subplot(441)
# plt.xlabel(r"$g-r$")
# plt.ylabel(r"$\mu_i$")
# plt.xlim(0.2,0.9)
# plt.yticks([18,19,20,21,22],[18,19,20,21,22])
# plt.ylim(17,23)
# plt.plot(gr_corrected,sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gr_corrected)
# qofq(gr_corrected,sb)


# plt.subplot(442)
# plt.tick_params(axis='both',labelbottom='off',labelleft='off')
# plt.xlabel(r"$g-i$")
# plt.xlim(0.3,1.4)
# plt.ylim(17,23)
# plt.plot(gi_corrected,sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gi_corrected)
# qofq(gi_corrected,sb)

# plt.subplot(443)
# plt.tick_params(axis='both',labelbottom='off',labelleft='off')
# plt.xlabel(r"$r-i$")
# plt.xlim(0,80)
# plt.ylim(17,23)
# plt.plot(half_light[:,3],sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(half_light[:,3])
# qofq(half_light[:,3],sb)

# plt.subplot(444)
# plt.tick_params(axis='both',labelleft='off')
# plt.xticks([2.5,3.0,3.5,4.0,4.5],[2.5,3.0,3.5,4.0,4.5])
# plt.xlabel(r"$C$")
# plt.xlim(2,5)
# plt.ylim(17,23)
# plt.plot(concentration[:,3],sb,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(concentration[:,3])
# qofq(concentration[:,3],sb)

# plt.subplot(445)
# plt.xlim(0.2,0.9)
# plt.ylim(2,5)
# plt.yticks([2.5,3.0,3.5,4.0,4.5],[2.5,3.0,3.5,4.0,4.5])
# plt.ylabel(r"$C(r90/r50)$")
# plt.plot(gr_corrected,concentration[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gr_corrected)
# qofq(gr_corrected,concentration[:,3])

# plt.subplot(446)
# plt.tick_params(axis='both',labelbottom='off',labelleft='off')
# plt.xlim(0.3,1.4)
# plt.ylim(2,5)
# plt.xlabel(r"$g-r$")
# plt.plot(gi_corrected,concentration[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gi_corrected)
# qofq(gi_corrected,concentration[:,3])

# plt.subplot(447)
# plt.tick_params(axis='both',labelleft='off')
# plt.xlim(0,80)
# plt.xticks([10,20,30,40,50,60,70],[10,20,30,40,50,60,70])
# plt.ylim(2,5)
# plt.xlabel(r"$half-light\,radius$")
# plt.plot(half_light[:,3],concentration[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(half_light[:,3])
# qofq(half_light[:,3],concentration[:,3])

# plt.subplot(449)
# plt.xlim(0.2,0.9)
# plt.ylim(0,80)
# plt.yticks([10,20,30,40,50,60,70],[10,20,30,40,50,60,70])
# plt.ylabel(r"$half-light\,radius$")
# plt.plot(gr_corrected,half_light[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gr_corrected)
# qofq(gr_corrected,half_light[:,3])

# plt.subplot(4,4,10)
# plt.tick_params(axis='both',labelleft='off')
# plt.xlim(0.3,1.4)
# plt.ylim(0,80)
# plt.xlabel(r"$g-i$")
# plt.plot(gi_corrected,half_light[:,3],'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gi_corrected)
# qofq(gi_corrected,half_light[:,3])

# plt.subplot(4,4,13)
# plt.xlim(0.2,0.9)
# plt.xlabel(r"$g-r$")
# plt.ylim(0.3,1.4)
# plt.ylabel(r"$g-i$")
# plt.plot(gr_corrected,gi_corrected,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gr_corrected)
# qofq(gr_corrected,gi_corrected)
# plt.suptitle('Comparison of all objects in RC3 to date')

# #plt.savefig('triangle_rc3quantiles_3.pdf')
# #os.system('cp triangle_rc3quantiles_3.pdf public_html/')



data = pyf.open("nsa-short.fits.gz")[1].data
a=data.field('RA')
b=data.field('DEC')
y = data.field('SERSICFLUX')
z = data.field('SERSIC_TH50')
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
indx3=np.where(z > 158)
good[indx3]=False


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

# plt.figure(2)
# plt.subplots_adjust(wspace=0,hspace=0)

# plt.subplot(211)
# plt.xlim(0,1.1)
# plt.xlabel(r"$g-r$")
# plt.ylim(-.1,1.7)
# plt.ylabel(r"Tractor $g-i$")
# plt.plot(gr_corrected,gi_corrected,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
# plotq(gr_corrected)
# qofq(gr_corrected,gi_corrected)

# plt.subplot(212)
# plt.plot(badgr,badgi,'m.', alpha=0.5)
# plt.plot(gr, gi,'k.', alpha=0.5)
# plt.xlabel(r"$g-r$")
# plt.xlim(0,1.1)
# plt.ylim(-.1,1.7)
# plt.ylabel(r"NS Atlas $g-i$")
# plotq(gr_corrected)
# qofq(gr_corrected,gi_corrected)

# #plt.savefig('color_color1_3.pdf')
# #os.system('cp color_color1_3.pdf public_html')


plt.figure(3)
plt.plot(badgr,badgi,'m.', alpha=0.5)
plt.plot(gr, gi,'k.', alpha=0.5)
plt.plot(gr_corrected,gi_corrected,'o',ms=2,markeredgecolor='blue',markeredgewidth=0.9,markerfacecolor='none')
plt.xlabel(r"$g-r$")
plt.ylabel(r"$g-i$")
plt.xlim(0,1.1)
plt.ylim(-.1,1.7)
plotq(gr_corrected)
qofq(gr_corrected,gi_corrected)

plt.plot(unpickle4color('NGC_1022-scale1.pickle')[0],unpickle4color('NGC_1022-scale1.pickle')[1],'*', ms=8,markeredgecolor='yellow',markeredgewidth=0.9,markerfacecolor='none',label='NGC 1022 (1)')

plt.plot(unpickle4color('NGC_1022-scale2.pickle')[0],unpickle4color('NGC_1022-scale2.pickle')[1],'*', ms=8,markeredgecolor='red',markeredgewidth=0.9,markerfacecolor='none',label='NGC 1022 (2)')

plt.plot(unpickle4color('NGC_1022-scale4.pickle')[0],unpickle4color('NGC_1022-scale4.pickle')[1],'*', ms=8,markeredgecolor='orange',markeredgewidth=0.9,markerfacecolor='none',label='NGC 1022 (4)')

plt.plot(unpickle4color('NGC_5457-scale2.pickle')[0],unpickle4color('NGC_5457-scale2.pickle')[1],'*', ms=8,markeredgecolor='cyan',markeredgewidth=0.9,markerfacecolor='none',label='M101 (2)')

plt.plot(unpickle4color('NGC_5457-scale4.pickle')[0],unpickle4color('NGC_5457-scale4.pickle')[1],'*', ms=8,markeredgecolor='purple',markeredgewidth=0.9,markerfacecolor='none',label='M101 (4)')

# plt.plot(gr_corrected[ugc7332],gi_corrected[ugc7332],'*', ms=8,markeredgecolor='yellow',markeredgewidth=0.9,markerfacecolor='none',label='UGC 7332')
# plt.plot(gr_corrected[leoa],gi_corrected[leoa],'*', ms=8,markeredgecolor='red',markeredgewidth=0.9,markerfacecolor='none',label='LEO A')
# plt.plot(gr_corrected[ngc52],gi_corrected[ngc52],'*', ms=8,markeredgecolor='orange',markeredgewidth=0.9,markerfacecolor='none',label='NGC 52')
plt.legend(loc='lower right')

# plt.savefig('color_color2_sb.pdf')
# os.system('cp color_color2_sb.pdf public_html/')
#os.chdir("../")
plt.savefig('plot_scale.pdf')
os.system('cp plot_scale.pdf public_html/')


# plt.figure(4)
# plt.plot(ra,dec, 'o',ms=2,markeredgecolor='cyan',markeredgewidth=0.9,markerfacecolor='none')
# plt.title('RA vs. Dec from RC3')
# #plt.savefig('ra_dec.pdf')
# #os.system('cp ra_dec.pdf public_html/')

plt.figure(5)
plt.subplot(211)
plt.xlabel(r"$r_{50,r}$(arcsec)")
plt.ylabel(r"log(n)")
plt.hist(half_light[:,2],bins=60,log=True)
#plt.subplot(212)
#plt.hist(half_light[:,2])
pdf=PdfPages('r50r90.pdf')
pdf.savefig()

plt.clf()
plt.hist(half_light[:,3],bins=60, log=True,color='green')
plt.xlabel(r"$r_{50,i}$(arcsec)")
plt.ylabel(r"log(n)")
pdf.savefig()

plt.clf()
plt.hist(r90[:,2],bins=60,log=True,color='purple')
plt.xlabel(r"$r_{90,r}$(arcsec)")
plt.ylabel(r"log(n)")
pdf.savefig()

plt.clf()
plt.hist(r90[:,3],bins=60,log=True,color='yellow')
plt.xlabel(r"$r_{90,i}$(arcsec)")
plt.ylabel(r"log(n)")
pdf.savefig()

plt.clf()
plt.plot(half_light[:,2],r90[:,2], 'ro',alpha=0.5,ms=4)
plt.xlabel(r"$r_{50,r}$(arcsec)")
plt.ylabel(r"$r_{90,r}$(arcsec)")
plt.xlim(0,150)
plt.ylim(0,450)
plotq(half_light[:,2])
qofq(half_light[:,2],r90[:,2])
pdf.savefig()

plt.clf()
plt.plot(half_light[:,3],r90[:,3], 'co',alpha=0.5,ms=4)
plt.xlabel(r"$r_{50,i}$(arcsec)")
plt.ylabel(r"$r_{90,i}$(arcsec)")
plt.xlim(0,150)
plt.ylim(0,400)
plotq(half_light[:,3])
qofq(half_light[:,3],r90[:,3])
pdf.savefig()

plt.clf()
plt.plot(half_light[:,3],sb,'bo',alpha=0.5,ms=4)
plt.ylabel(r"$\mu_{50,i}$")
plt.xlabel(r"$r_{50,i}$(arcsec)")
plt.xlim(0,150)
plt.ylim(16,24)
plotq(half_light[:,3])
qofq(half_light[:,3],sb)
pdf.savefig()

pdf.close()
os.system('cp r50r90.pdf public_html')
