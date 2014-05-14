#from tractor.rc3 import *
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
import matplotlib.pyplot
import pylab as plt
from pylab import *
import triangle 
import pyfits as pyf
from astropysics.obstools import *
import operator
import math
import os


if __name__ == '__main__':

    data=pyf.open('sdss_atlas.fits')
    data2=data[1].data
    mags=data2.field('CG_TOTMAGS')
    extinction=data2.field('CG_EXTINCTION')
    #name=data2.field('RC3_NAME')
    name=data2.field('NAME')
    #ra=data2.field('RC3_RA')
    ra2=data2.field('CG_RA')
    #dec=data2.field('RC3_DEC')    
    dec2=data2.field('CG_DEC')
    sb=data2.field('CG_I-SB')
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

    def mu_50(i,r):
        return i+2.5*(log10(pi*r**2))

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
print max(half_light[:,3])

plt.figure(1,figsize=(6,6))
plt.plot(ra2,dec2,'k.',alpha=0.5)
plt.xlim(360,0)
plt.ylim(-30,90)
a=np.arange(0,420,60)
b=np.arange(-10,100,20)
plt.xlabel('RA')
plt.ylabel('DEC')
plt.savefig('all_ra_dec.pdf')
os.system('cp all_ra_dec.pdf public_html/paper')
os.system('cp all_ra_dec.pdf ~/SloanAtlas/paper')

plt.figure(2)
plt.plot(half_light[:,3],corrected_i,'k.', alpha=0.5)
plt.xlim(0,260)
plt.ylim(0,20)
plt.xlabel(r'$R_{50,i}(arcsec)$')
plt.ylabel(r'$i-band\;magnitude$')
plt.savefig('all_r50_i.pdf')
os.system('cp all_r50_i.pdf public_html/paper')
os.system('cp all_r50_i.pdf ~/SloanAtlas/paper')

plt.figure(3)
plt.plot(r90[:,3],corrected_i,'k.', alpha=0.5)
plt.xlim(60,260)
plt.ylim(0,20)
plt.xlabel(r'$R_{90,i}(arcsec)$')
plt.ylabel(r'$i-band\;magnitude$')
plt.savefig('all_r90_i.pdf')
os.system('cp all_r90_i.pdf public_html/paper')
os.system('cp all_r90_i.pdf ~/SloanAtlas/paper')

plt.figure(4)
print len(gr_corrected), len(ri_corrected)
cornerdata=np.array([gr_corrected,ri_corrected,sb,concentration[:,3]],dtype=float)
cornerdata = cornerdata.T
print cornerdata.shape
triangle.corner(cornerdata, labels=[r'$g-r$',r'$r-i$',r'$\mu_{50,i}(mag\;arcsec^{-2})$',r'$R_{90,i}/R_{50,i}$'],extents=[(0.3,0.8),(0,0.5),(17,25),(2,5)],bins=60, scale_hist=True, quantiles=[0.25,0.5,0.75])
plt.savefig('all_triangle.pdf')
os.system('cp all_triangle.pdf public_html/paper')
os.system('cp all_triangle.pdf ~/SloanAtlas/paper')

plt.figure(5)
cornerdata=np.array((half_light[:,3],r90[:,3],concentration[:,3]),dtype=float)
cornerdata=cornerdata.T
triangle.corner(cornerdata,labels=[r'$r_{50,i}$',r'$r_{90,i}$','$C$'],extents=[(0,260),(0,260),(2,5)], bins=60,scale_hist=True, quantiles=[0.25,0.5,0.75])
plt.savefig('all_triangle2.pdf')
os.system('cp all_triangle2.pdf public_html/paper')
assert(False)

# #invalid now that we are using the 30 arcsec radius cut file
# nsa_data=pyf.open('nsa_galaxies.fits')
# nsa_data2=nsa_data[1].data
# nsa_mags=nsa_data2.field('CG_TOTMAGS')
# nsa_extinction=nsa_data2.field('CG_EXTINCTION')
# nsa_name=nsa_data2.field('NSA_NAME')
# print len(nsa_name)

# nsa_ra=nsa_data2.field('NSA_RA')
# nsa_ra2=nsa_data2.field('CG_RA')
# nsa_dec=nsa_data2.field('NSA_DEC')
# nsa_dec2=nsa_data2.field('CG_DEC')
# nsa_sb=nsa_data2.field('CG I-SB')
# nsa_half_light=nsa_data2.field('CG_R50S')
# nsa_r90=nsa_data2.field('CG_R90S')
# nsa_concentration=nsa_data2.field('CG_CONC')

# nsa_corrected_u=nsa_mags[:,0]-nsa_extinction[:,0]
# nsa_corrected_g=nsa_mags[:,1]-nsa_extinction[:,1]
# nsa_corrected_r=nsa_mags[:,2]-nsa_extinction[:,2]
# nsa_corrected_i=nsa_mags[:,3]-nsa_extinction[:,3]
# nsa_corrected_z=nsa_mags[:,4]-nsa_extinction[:,4]
# nsa_gr_corrected=nsa_corrected_g-nsa_corrected_r
# nsa_ri_corrected=nsa_corrected_r-nsa_corrected_i
# nsa_gi_corrected=nsa_corrected_g-nsa_corrected_i 
# nsa_ug_corrected=nsa_corrected_u-nsa_corrected_g
# nsa_iz_corrected=nsa_corrected_i-nsa_corrected_z

# nsa_sb=[0 if math.isnan(x) else x for x in nsa_sb]
# nsa_sb=[0 if math.isinf(x) else x for x in nsa_sb]
# nsa_gr_corrected=[0 if math.isnan(x) else x for x in nsa_gr_corrected]
# nsa_gr_corrected=[0 if math.isinf(x) else x for x in nsa_gr_corrected]
# nsa_ug_corrected=[0 if math.isinf(x) else x for x in nsa_ug_corrected]
# nsa_ug_corrected=[0 if math.isnan(x) else x for x in nsa_ug_corrected]
# nsa_gi_corrected=[0 if math.isinf(x) else x for x in nsa_gi_corrected]
# nsa_gi_corrected=[0 if math.isnan(x) else x for x in nsa_gi_corrected]
# nsa_iz_corrected=[0 if math.isinf(x) else x for x in nsa_iz_corrected]
# nsa_iz_corrected=[0 if math.isnan(x) else x for x in nsa_iz_corrected]
# nsa_ri_corrected=[0 if math.isinf(x) else x for x in nsa_ri_corrected]
# nsa_ri_corrected=[0 if math.isnan(x) else x for x in nsa_ri_corrected]

# nsa_concentration[:,3]=[0 if math.isnan(x) else x for x in nsa_concentration[:,3]]
# nsa_concentration[:,3]=[0 if math.isinf(x) else x for x in nsa_concentration[:,3]]
# nsa_mags[:,3]=[0 if math.isinf(x) else x for x in nsa_mags[:,3]]
# nsa_mags[:,3]=[0 if math.isnan(x) else x for x in nsa_mags[:,3]]


# c1=[float(i) for i in concentration[:,3]]
# c2=[float(i) for i in nsa_concentration[:,3]]
# c1+=c2

# sb+=nsa_sb
# gr1=[float(i) for i in gr_corrected]
# gr2=[float(i) for i in nsa_gr_corrected]
# gr1+=gr2

# gi1=[float(i) for i in gi_corrected]
# gi2=[float(i) for i in nsa_gi_corrected]
# gi1+=gi2

# r1=[float(i) for i in half_light[:,3]]
# r2=[float(i) for i in nsa_half_light[:,3]]
# r1+=r2

# i1=[float(i) for i in corrected_i]
# i2=[float(i) for i in nsa_corrected_i]
# i1+=i2

# ug1=[float(i) for i in ug_corrected]
# ug2=[float(i) for i in nsa_ug_corrected]
# ug1+=ug2

# iz1=[float(i) for i in iz_corrected]
# iz2=[float(i) for i in nsa_iz_corrected]
# iz1+=iz2

# ri1=[float(i) for i in ri_corrected]
# ri2=[float(i) for i in nsa_ri_corrected]
# ri1+=ri2
# print len(gi1),len(r1),len(i1),len(ug1),len(iz1), len(sb)

# ra2=[float(i) for i in ra2]
# nsa_ra2=[float(i) for i in nsa_ra2]
# ra2+=nsa_ra2
# print len(ra2)

# dec2=[float(i) for i in dec2]
# nsa_dec2=[float(i) for i in nsa_dec2]
# dec2+=nsa_dec2

# plt.figure(1,figsize=(6,6))
# plt.plot(ra2,dec2,'k.',alpha=0.5)
# plt.xlim(360,0)
# plt.ylim(-30,90)
# a=np.arange(0,420,60)
# b=np.arange(-10,100,20)
# print b
# #plt.xticks(a,a)
# plt.xlabel('RA')
# plt.ylabel('DEC')
# #plt.yticks(b,b)
# plt.savefig('ra_dec.pdf')
# os.system('cp ra_dec.pdf public_html/paper')
# os.system('cp ra_dec.pdf ~/SloanAtlas/paper')

# plt.figure(2,figsize=(6,6))
# plt.plot(gi1,ri1,'k.',alpha=0.5)
# plt.xlim(0.3,1.4)
# plt.xlabel(r'$g-i$')
# plt.ylabel(r'$r-i$')
# plt.ylim(0.1,0.5)
# plt.savefig('gi_ri.pdf')
# os.system('cp gi_ri.pdf public_html/paper')
# os.system('cp gi_ri.pdf ~/SloanAtlas/paper')

# plt.figure(3,figsize=(6,6))
# plt.plot(r1,i1, 'k.', alpha=0.5)
# plt.xlim(0,100)
# plt.ylim(8,18)
# plt.xlabel(r'$R_{50,i}(arcsec)$')
# plt.ylabel(r'$i-band\;magnitude$')
# plt.savefig('r50_i.pdf')
# os.system('cp r50_i.pdf public_html/paper')
# os.system('cp r50_i.pdf ~/SloanAtlas/paper')

# plt.figure(4)
# cornerdata=np.array((gi1,sb,c1), dtype=float)
# triangle.corner(cornerdata, labels=[r'$g-i$',r'$\mu_{50,i}(mag\;arcsec^{-2})$',r'$R_{90,i}/R_{50,i}$'],extents=[(0.3,1.4),(17,23),(2,5)],bins=80)
# plt.savefig('gi_mu_c.pdf')
# os.system('cp gi_mu_c.pdf public_html/paper')
# os.system('cp gi_mu_c.pdf ~/SloanAtlas/paper')
# assert(False)
# plt.figure(5)
# plt.hist(r1, cumulative=-1, bins=80)
# plt.xlabel(r'$R_{50,i}$')
# plt.ylabel(r'$N(R_{50,i})$')
# plt.savefig('cumulative_r50.pdf')
# os.system('cp cumulative_r50.pdf public_html/paper')


# d25=data2.field('RC3_LOG_D25')
# d25= (10**d25)
# print d25
# plt.figure(6)
# plt.loglog(d25,half_light[:,3],'k.',alpha=0.5)
# (m,b)=polyfit(log(d25),log(half_light[:,3]),1)
# yp=polyval([m,b],d25)
# print m
# print b
# plt.plot(d25,yp)
# plt.xlabel(r'$RC3\;log(d25)$')
# plt.ylabel(r'$R_{50,i}$')
# #plt.annotate('y=mx+b',xy=(900,900),xytext=None,xycoords='axes points', textcoords='axes points',arrowprops=None )
# plt.figtext(0.65,0.85,'y=%sx+%s'%(m,b),size='x-small')
# plt.savefig('d25_r.pdf')
# os.system('cp d25_r.pdf public_html/paper')

# print len(d25)
# #assert(False)
# rc3=pyf.open('newrc3limited.fits')
# color=rc3[1].data.field('BV_COLOR_TOT')
# cols=rc3[1].columns
# print cols.names
# print len(color)
# rc3_bv=[]
# bvdata=pyf.open('large_galaxies_bv.fits')
# bv=bvdata[1].data.field('RC3_BV')
# print len(bv)

# bv=data2.field('RC3_BV')
# print len(bv)
# print len(gr1)
# plt.figure(7)
# plt.plot(bv,gr_corrected, 'k.', alpha=0.5)
# plt.xlabel(r'$RC3\;B-V_{tot}$')
# plt.ylabel(r'$g-r$')
# plt.ylim(0,1)
# plt.xlim(0.2,1.2)
# plt.savefig('bv_gr.pdf')
# os.system('cp bv_gr.pdf public_html/paper')
