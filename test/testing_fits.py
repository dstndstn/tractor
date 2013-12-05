#testing large_galaxies.fits (the objects from rc3 catalog)
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot
import pylab as plt
import triangle 
import pyfits as pyf
from astropysics.obstools import *
import operator
import math
import os
from matplotlib.ticker import MaxNLocator, AutoLocator


if __name__ == '__main__':
    data=pyf.open('large_galaxies.fits')
    data2=data[1].data
    mags=data2.field('CG_TOTMAGS')
    extinction=data2.field('CG_EXTINCTION')
    name=data2.field('RC3_NAME')
    print len(name)
    assert(False)
    ra=data2.field('RC3_RA')
    ra2=data2.field('CG_RA')
    dec=data2.field('RC3_DEC')
    dec2=data2.field('CG_DEC')
    sb=data2.field('CG I-SB')
    half_light=data2.field('CG_R50S')
    r90=data2.field('CG_R90S')
    concentration=data2.field('CG_CONC')
    gr_original=mags[:,1]-mags[:,2]
    ri_original=mags[:,2]-mags[:,3]
    corrected_u=mags[:,0]-extinction[:,0]
    corrected_g=mags[:,1]-extinction[:,1]
    corrected_r=mags[:,2]-extinction[:,2]
    corrected_i=mags[:,3]-extinction[:,3]
    corrected_z=mags[:,4]-extinction[:,4]
    gr_corrected=corrected_g-corrected_r
    ri_corrected=corrected_r-corrected_i
    gi_corrected=corrected_g-corrected_i 
    ug_corrected=corrected_u-corrected_g
    iz_corrected=corrected_i-corrected_z

    print half_light
    assert(False)

    def mu_50(i,r):
        return i+2.5*(log10(pi*r**2))

    def quantile(x,q):
        i=len(sorted(x))*q
        z=int(floor(i))
        #print z
        #print len(x)
        return ((sorted(x))[z])
    
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

        print len(sample1)
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
    mice=np.where(name=='NGC 4676A')
    # galaxies[radius]=False
    print gr_corrected[mice],gi_corrected[mice]
    mice2=np.where(name=='NGC 4676B')
    # galaxies[radius]=False
    print gr_corrected[mice2],gi_corrected[mice2]
    #assert(False)
    # plt.annotate('PGC 70104',xy=(0.2,-0.00762749),xytext=(0.4,0.0),arrowprops=dict(facecolor='black'))
    # plt.annotate('UGC 11861',xy=(1.119,.6218),xytext=(0.7,.6218),arrowprops=dict(facecolor='black'))
    # plt.annotate('MCG 1-57-?',xy=(.62,.850),xytext=(.4,.85),arrowprops=dict(facecolor='black'))

mag=[mags[:,0],mags[:,1],mags[:,2],mags[:,3],mags[:,4]]
# os.chdir('RC3_Output/')
# for t in mag:
#     print 'next'

#     select=[x for x in xrange(len(name)) if math.isinf(t[x])]
#     for x in select:
#         print name[x]
#         replace=name[x].replace(' ','_')
#         os.system('cp flip-%s.pdf ~/penguin/tractor/infflux/' %(replace))

sb=[0 if math.isnan(x) else x for x in sb]
sb=[0 if math.isinf(x) else x for x in sb]
gr_corrected=[0 if math.isnan(x) else x for x in gr_corrected]
gr_corrected=[0 if math.isinf(x) else x for x in gr_corrected]
ug_corrected=[0 if math.isinf(x) else x for x in ug_corrected]
ug_corrected=[0 if math.isnan(x) else x for x in ug_corrected]
gi_corrected=[0 if math.isinf(x) else x for x in gi_corrected]
gi_corrected=[0 if math.isnan(x) else x for x in gi_corrected]
iz_corrected=[0 if math.isinf(x) else x for x in iz_corrected]
iz_corrected=[0 if math.isnan(x) else x for x in iz_corrected]
concentration[:,3]=[0 if math.isnan(x) else x for x in concentration[:,3]]
concentration[:,3]=[0 if math.isinf(x) else x for x in concentration[:,3]]
mags[:,3]=[0 if math.isinf(x) else x for x in mags[:,3]]
mags[:,3]=[0 if math.isnan(x) else x for x in mags[:,3]]

plt.figure(1)
plt.clf()
plt.plot(ra,dec, 'o',ms=2,markeredgecolor='cyan',markeredgewidth=0.9,markerfacecolor='none')
plt.title('RA vs. Dec from RC3')
pdf2=PdfPages('color_color.pdf')
pdf2.savefig()

data = pyf.open("nsa-short.fits.gz")[1].data
a=data.field('RA')
b=data.field('DEC')
y = data.field('SERSICFLUX')
z = data.field('SERSIC_TH50')
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

plt.clf()
plt.subplots_adjust(wspace=0,hspace=0)
plt.subplot(211)
plt.xlim(0,1.1)
plt.xlabel(r"$g-r$")
plt.ylim(-.1,1.7)
plt.ylabel(r"Tractor $g-i$")
plt.plot(gr_corrected,gi_corrected,'o',ms=2,markeredgecolor='black',markeredgewidth=0.9,markerfacecolor='none')
plotq(gr_corrected)
#qofq(gr_corrected,gi_corrected)

plt.subplot(212)
plt.plot(badgr,badgi,'m.', alpha=0.5)
plt.plot(gr, gi,'k.', alpha=0.5)
plt.xlabel(r"$g-r$")
plt.xlim(0,1.1)
plt.ylim(-.1,1.7)
plt.ylabel(r"NS Atlas $g-i$")
plotq(gr_corrected)
#qofq(gr_corrected,gi_corrected)
pdf2.savefig()

plt.clf()
#plt.plot(badgr,badgi,'m.', alpha=0.5)
#plt.plot(gr, gi,'k.', alpha=0.5)
plt.plot(gr_corrected,gi_corrected,'k.',alpha=0.5)
#'o',ms=2,markeredgecolor='blue',markeredgewidth=0.9,markerfacecolor='none')
plt.xlabel(r"$g-r$")
plt.ylabel(r"$g-i$")
plt.plot(0.6119,1.0635,'g*',ms=10)
plt.plot(.8090,1.2804,'r*',ms=10)
plt.xlim(0,1.1)
plt.ylim(-.1,1.7)
plt.savefig('mice.pdf')
os.system('cp mice.pdf public_html/')
assert(False)
#qofq(gr_corrected,gi_corrected)

plt.legend(loc='lower right')
pdf2.savefig()
pdf2.close()
os.system('cp color_color.pdf public_html/')



# plt.figure(3)
# pdf3=PdfPages('corner_plots.pdf')
# cornerdata=np.array((concentration[:,3],sb,gr_corrected,gi_corrected,half_light[:,3]),dtype=float) 
# print len(cornerdata)
# triangle.corner(cornerdata,labels=[r"$C_i$",r"$\mu_{50,i}$",r"$g-r$",r"$g-i$",r"$R_{50,i}$"], extents=[(2,5),(17,23),(0.2,0.9),(0.3,1.4),(0,100)], bins=80)
# pdf3.savefig()

# plt.clf()
# cornerdata=np.array((corrected_i,ug_corrected,gr_corrected,gi_corrected,iz_corrected),dtype=float)
# triangle.corner(cornerdata,labels=[r"$i-mag$",r"$u-g$",r"$g-r$",r"$g-i$",r"$i-z$"],extents=[(9,14.5),(0.5,2),(0.2,0.9),(0.3,1.4),(0,0.4)],bins=80)
# pdf3.savefig()
# pdf3.close()
# os.system('cp corner_plots.pdf public_html/')


nsa_data=pyf.open('nsa_galaxies.fits')
nsa_data2=nsa_data[1].data
nsa_mags=nsa_data2.field('CG_TOTMAGS')
nsa_extinction=nsa_data2.field('CG_EXTINCTION')
nsa_name=nsa_data2.field('NSA_NAME')

nsa_ra=nsa_data2.field('NSA_RA')
nsa_ra2=nsa_data2.field('CG_RA')
nsa_dec=nsa_data2.field('NSA_DEC')
nsa_dec2=nsa_data2.field('CG_DEC')
nsa_sb=nsa_data2.field('CG I-SB')
nsa_half_light=nsa_data2.field('CG_R50S')
nsa_r90=nsa_data2.field('CG_R90S')
nsa_concentration=nsa_data2.field('CG_CONC')

nsa_corrected_u=nsa_mags[:,0]-nsa_extinction[:,0]
nsa_corrected_g=nsa_mags[:,1]-nsa_extinction[:,1]
nsa_corrected_r=nsa_mags[:,2]-nsa_extinction[:,2]
nsa_corrected_i=nsa_mags[:,3]-nsa_extinction[:,3]
nsa_corrected_z=nsa_mags[:,4]-nsa_extinction[:,4]
nsa_gr_corrected=nsa_corrected_g-nsa_corrected_r
nsa_ri_corrected=nsa_corrected_r-nsa_corrected_i
nsa_gi_corrected=nsa_corrected_g-nsa_corrected_i 
nsa_ug_corrected=nsa_corrected_u-nsa_corrected_g
nsa_iz_corrected=nsa_corrected_i-nsa_corrected_z

nsa_sb=[0 if math.isnan(x) else x for x in nsa_sb]
nsa_sb=[0 if math.isinf(x) else x for x in nsa_sb]
nsa_gr_corrected=[0 if math.isnan(x) else x for x in nsa_gr_corrected]
nsa_gr_corrected=[0 if math.isinf(x) else x for x in nsa_gr_corrected]
nsa_ug_corrected=[0 if math.isinf(x) else x for x in nsa_ug_corrected]
nsa_ug_corrected=[0 if math.isnan(x) else x for x in nsa_ug_corrected]
nsa_gi_corrected=[0 if math.isinf(x) else x for x in nsa_gi_corrected]
nsa_gi_corrected=[0 if math.isnan(x) else x for x in nsa_gi_corrected]
nsa_iz_corrected=[0 if math.isinf(x) else x for x in nsa_iz_corrected]
nsa_iz_corrected=[0 if math.isnan(x) else x for x in nsa_iz_corrected]
nsa_concentration[:,3]=[0 if math.isnan(x) else x for x in nsa_concentration[:,3]]
nsa_concentration[:,3]=[0 if math.isinf(x) else x for x in nsa_concentration[:,3]]
nsa_mags[:,3]=[0 if math.isinf(x) else x for x in nsa_mags[:,3]]
nsa_mags[:,3]=[0 if math.isnan(x) else x for x in nsa_mags[:,3]]


c1=[float(i) for i in concentration[:,3]]
c2=[float(i) for i in nsa_concentration[:,3]]
c1+=c2

sb+=nsa_sb
gr_corrected+=nsa_gr_corrected
gi1=[float(i) for i in gi_corrected]
gi2=[float(i) for i in nsa_gi_corrected]
gi1+=gi2

r1=[float(i) for i in half_light[:,3]]
r2=[float(i) for i in nsa_half_light[:,3]]
r1+=r2

i1=[float(i) for i in corrected_i]
i2=[float(i) for i in nsa_corrected_i]
i1+=i2

ug1=[float(i) for i in ug_corrected]
ug2=[float(i) for i in nsa_ug_corrected]
ug1+=ug2

iz1=[float(i) for i in iz_corrected]
iz2=[float(i) for i in nsa_iz_corrected]
iz1+=iz2

print len(gi1),len(r1),len(i1),len(ug1),len(iz1), len(sb)


plt.figure(4)
pdf4=PdfPages('corner_all.pdf')
cornerdata=np.array((c1,sb,gr_corrected,gi1,r1),dtype=float) 
print len(cornerdata)
triangle.corner(cornerdata,labels=[r"$C_i$",r"$\mu_{50,i}$",r"$g-r$",r"$g-i$",r"$R_{50,i}$"], extents=[(2,5),(17,23),(0.2,0.9),(0.3,1.4),(0,100)], bins=80)
pdf4.savefig()

plt.clf()
cornerdata=np.array((i1,ug1,gr_corrected,gi1,iz1),dtype=float)
triangle.corner(cornerdata,labels=[r"$i-mag$",r"$u-g$",r"$g-r$",r"$g-i$",r"$i-z$"],extents=[(9,14.5),(0.5,2),(0.2,0.9),(0.3,1.4),(0,0.4)],bins=80)
pdf4.savefig()
pdf4.close()
os.system('cp corner_all.pdf public_html/')
