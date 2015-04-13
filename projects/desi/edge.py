import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys

import fitsio
import optparse

parser = optparse.OptionParser()
#parser.add_option('--prefix', help='Plot prefix', default='edge')
opt,args = parser.parse_args()

for fn in args:
    #'c4d_131028_014102_ooi_r_v1.fits.fz')
    print 'Reading', fn
    F = fitsio.FITS(fn)
    mfn = fn.replace('_ooi_','_ood_')
    print 'Reading', mfn
    M = fitsio.FITS(mfn)
    print len(F), 'hdus'

    hdr = F[0].read_header()
    name = os.path.basename(fn)
    name = name.replace('.fits','').replace('.fz','')
    expnum = int(hdr['EXPNUM'])
    tt = 'Exp %i, %s' % (expnum, name)

    textargs = dict(ha='left', va='center', fontsize=8)

    for fig in [1,2,3,4]:
        plt.clf()

    yoffstep = 10
    for hdu in range(1, len(F)):
        I = F[hdu].read()
        hdr = F[hdu].read_header()
        mask = M[hdu].read()
        extname = hdr['EXTNAME']
        print 'Read', extname
        med = np.median(I, axis=1)
        mm = np.median(med)
        med -= mm
        H = len(med)
        x = np.arange(len(med))
        ok = np.median(mask == 0, axis=1).astype(bool)
    
        extname = extname.strip()
        if extname.startswith('N'):
            goodcolor = 'b'
        else:
            goodcolor = 'g'
        
        yoff = (hdu-1)*yoffstep
    
        N = 200
        cutx   = x  [:N]
        cutmed = med[:N]
        cutok  = ok [:N]
        cutnotok = np.logical_not(cutok)
    
        plt.figure(1)
        plt.plot(cutx, yoff + cutmed, 'k-')
        if sum(cutok):
            plt.plot(cutx[cutok], yoff + cutmed[cutok], '.', color=goodcolor, alpha=0.5)
        if sum(cutnotok):
            plt.plot(cutx[cutnotok], yoff + cutmed[cutnotok], 'r.', alpha=0.5)
        plt.text(N, yoff, extname, **textargs)
    
    
        cutx   = x  [-N:]
        cutmed = med[-N:]
        cutok  = ok [-N:]
        cutnotok = np.logical_not(cutok)
    
        plt.figure(2)
        plt.plot(cutx, yoff + cutmed, 'k-')
        if sum(cutok):
            plt.plot(cutx[cutok], yoff + cutmed[cutok], '.', color=goodcolor, alpha=0.5)
        if sum(cutnotok):
            plt.plot(cutx[cutnotok], yoff + cutmed[cutnotok], 'r.', alpha=0.5)
        plt.text(H, yoff, extname, **textargs)
    
        med = np.median(I, axis=0)
        mm = np.median(med)
        med -= mm
        W = len(med)
        x = np.arange(len(med))
        ok = np.median(mask == 0, axis=0).astype(bool)
    
        
        N = 200
        cutx   = x  [:N]
        cutmed = med[:N]
        cutok  = ok [:N]
        cutnotok = np.logical_not(cutok)
    
        plt.figure(3)
        plt.plot(cutx, yoff + cutmed, 'k-')
        if sum(cutok):
            plt.plot(cutx[cutok], yoff + cutmed[cutok], '.', color=goodcolor, alpha=0.5)
        if sum(cutnotok):
            plt.plot(cutx[cutnotok], yoff + cutmed[cutnotok], 'r.', alpha=0.5)
        plt.text(N, yoff, extname, **textargs)
    
        cutx   = x  [-N:]
        cutmed = med[-N:]
        cutok  = ok [-N:]
        cutnotok = np.logical_not(cutok)
    
        plt.figure(4)
        plt.plot(cutx, yoff + cutmed, 'k-')
        if sum(cutok):
            plt.plot(cutx[cutok], yoff + cutmed[cutok], '.', color=goodcolor, alpha=0.5)
        if sum(cutnotok):
            plt.plot(cutx[cutnotok], yoff + cutmed[cutnotok], 'r.', alpha=0.5)
        plt.text(W, yoff, extname, **textargs)
        
    plt.figure(1)
    plt.xlim(0, N)
    plt.ylim(-50, len(F)*yoffstep+50)
    plt.title('%s: bottom edge' % tt)
    plt.savefig('edge-%i-bottom.png' % expnum)
    
    plt.figure(2)
    plt.xlim(H-N, H)
    plt.ylim(-50, len(F)*yoffstep+50)
    plt.title('%s: top edge' % tt)
    plt.savefig('edge-%i-top.png' % expnum)

    plt.figure(3)
    plt.xlim(0, N)
    plt.ylim(-50, len(F)*yoffstep+50)
    plt.title('%s: left edge' % tt)
    plt.savefig('edge-%i-left.png' % expnum)
    
    plt.figure(4)
    plt.xlim(W-N, W)
    plt.ylim(-50, len(F)*yoffstep+50)
    plt.title('%s: right edge' % tt)
    plt.savefig('edge-%i-right.png' % expnum)





sys.exit(0)
    
I = fitsio.read('n26.fits')
    
tt = '247506-N26, c4d_131028_014102_ood_r_v1.fits.fz'
plt.clf()
plt.plot(np.sum(I, axis=1), 'b-')
plt.xlim(0, 4100)
plt.xlabel('y pixel')
plt.suptitle(tt)
plt.savefig('edge1.png')

plt.clf()
plt.subplot(1,2,1)
plt.plot(np.sum(I, axis=1), 'b-')
ax = plt.axis()
plt.axis([0,200,ax[2],ax[3]])
plt.xlabel('y pixel')
plt.subplot(1,2,2)
plt.plot(np.sum(I, axis=1), 'b-')
plt.axis([3900,4100,ax[2],ax[3]])
plt.xlabel('y pixel')
plt.suptitle(tt)
plt.savefig('edge2.png')

plt.clf()
plt.plot(np.median(I, axis=1), 'b-')
plt.xlim(0, 4100)
plt.xlabel('y pixel')
plt.suptitle(tt)
plt.savefig('edge3.png')

plt.clf()
plt.subplot(1,2,1)
plt.plot(np.median(I, axis=1), 'b-')
ax = plt.axis()
plt.axis([0,200,ax[2],ax[3]])
plt.xlabel('y pixel')
plt.subplot(1,2,2)
plt.plot(np.median(I, axis=1), 'b-')
plt.axis([3900,4100,ax[2],ax[3]])
plt.xlabel('y pixel')
plt.suptitle(tt)
plt.savefig('edge4.png')
