import inspect
import sys
import os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)

import matplotlib.pyplot as plt
import pyfits as pyf

def color_prop(fits,fn1,fn2, objects=[],labels=[]):
    table=pyf.open('%s.fits'%(fits))
    data=table[1].data
    mags=data.field('CG_TOTMAGS')
    extinction=data.field('CG_EXTINCTION')
    c=data.field('CG_CONC')
    c=c[:,3]
    mu50=data.field('CG_I-SB')
    u=mags[:,0]-extinction[:,0]
    g=mags[:,1]-extinction[:,1]
    r=mags[:,2]-extinction[:,2]
    i=mags[:,3]-extinction[:,3]
    z=mags[:,4]-extinction[:,4]

    
    plt.figure(figsize=(6,6))
    x=g-i
    plt.plot(x, mu50, 'k.', alpha=0.5)
    plt.xlim(0.1,1.4)
    plt.ylim(16,26)
    plt.xlabel(r'$g-i$')
    plt.ylabel(r'$\mu_{50,i}$')
    for j in range(0,len(objects)):
        x=g[j]-i[j]
        plt.plot(x, mu50[j], 'o', color='red',ms=5, markeredgecolor='red')
        plt.annotate(labels[j], xy=(x,mu50[j]), xytext=(x,mu50[j]+0.15),xycoords='data',textcoords='data',color='red',size='large')
    plt.savefig('%s.pdf' %(fn1))

    plt.figure(figsize=(6,6))
    x=g-i
    plt.plot(x, c, 'k.', alpha=0.5)
    plt.xlim(0.1,1.4)
    plt.ylim(2,7)
    plt.xlabel(r'$g-i$')
    plt.ylabel(r'$C_i$')
    for j in range(0,len(objects)):
        x=g[j]-i[j]
        plt.plot(x, c[j], 'o', color='red',ms=5, markeredgecolor='red')
        plt.annotate(labels[j], xy=(x,c[j]), xytext=(x,c[j]+0.15),xycoords='data',textcoords='data',color='red',size='large')
    plt.savefig('%s.pdf' %(fn2))
 
color_prop('sdss_atlas','tester','tester2', objects=[1,2,3,4,5], labels=['A','B','C','D','E'])

