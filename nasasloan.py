if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import logging
import numpy as np
import pylab as plt

import pyfits

from astrometry.util.file import *

from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import sdss_galaxy as sg
from tractor import basics as ba
from tractor import engine as en
from astrometry.util.util import Tan
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

def main():
    ra = 126.925
    dec = 21.4833
    itune1 = 5
    itune2 = 5
    ntune = 0

    bands=['r']
    bandname = 'r'
    flipBands = ['r']

    rerun = 0

    TI = []
    sources = []

    table = pyfits.open("J082742.02+212844.7-r.fits")
    table.info()

    header = table[0].header
    data = table[0].data
    invvar=table[1].data
    skyobj = ba.ConstantSky(header['skyval'])
    psffn = 'J082742.02+212844.7-r-bpsf.fits.gz'
    psfimg = pyfits.open(psffn)[0].data
    print 'PSF image shape', psfimg.shape
    # number of Gaussian components
    PS = psfimg.shape[0]
    K = 3
    w,mu,sig = em_init_params(K, None, None, None)
    II = psfimg.copy()
    II /= II.sum()
    # HACK
    II = np.maximum(II, 0)
    print 'Multi-Gaussian PSF fit...'
    xm,ym = -(PS/2), -(PS/2)
    em_fit_2d(II, xm, ym, w, mu, sig)
    print 'w,mu,sig', w,mu,sig
    psf = GaussianMixturePSF(w, mu, sig)


    wcs = Tan("J082742.02+212844.7-r.fits",0)
    wcs = FitsWcs(wcs)
    wcs.setX0Y0(1.,1.)
    photocal = ba.NasaSloanPhotoCal(bandname) #Also probably not right
    TI.append(en.Image(data=data,invvar=invvar,sky=skyobj,psf=psf,wcs=wcs,photocal=photocal,name = "NASA-Sloan Test"))
    
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
    tims = [TI[0]]
    tractor = st.SDSSTractor(tims)
#    for source in sources:
#        tractor.addSources(source)

    zr = np.array([-5.,+5.])# * info['skysig']

    print bands

    prefix = 'ngc2595'
#    save('initial-'+prefix, tractor,zr,debug=True)
    bright = None
    lowbright = 1000
    sources=[]

    for timg,sources in zip(tims,sources):
        wcs = timg.getWcs()
        xtr,ytr = wcs.positionToPixel(RaDecPos(ra,dec))
    
        print xtr,ytr

        xt = xtr 
        yt = ytr
        r = 250.
        for src in sources:
            xs,ys = wcs.positionToPixel(src.getPosition(),src)
            if (xs-xt)**2+(ys-yt)**2 <= r**2:
                print "Removed:", src
                print xs,ys
                tractor.removeSource(src)

#    save('removed-'+prefix, tractor,zr,debug=True)
    newShape = sg.GalaxyShape(30.,1.,0.)
    newBright = ba.Mags(r=15.0,g=15.0,u=15.0,z=15.0,i=15.0)
    EG = st.ExpGalaxy(RaDecPos(ra,dec),newBright,newShape)
    print EG
    tractor.addSource(EG)


    save('added-'+prefix,tractor,zr,debug=True)

    for i in range(itune1):
        if (i % 5 == 0):
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[EG],sky=True)
        else:
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[EG],sky=False)
        tractor.clearCache()
        save('itune1-%d-' % (i+1)+prefix,tractor,zr,debug=True)
    
    CGPos = EG.getPosition()
    CGShape = EG.getShape()
    EGBright = EG.getBrightness()
    print EGBright
    CGg = EGBright[0]*1.25
    CGi = EGBright[1]*1.25
    CGr = EGBright[2]*1.25
    CGu = EGBright[3]*1.25
    CGz = EGBright[4]*1.25
    CGBright = ba.Mags(r=CGr,g=CGg,u=CGu,z=CGz,i=CGi)
    print EGBright
    print CGBright

    CG = st.CompositeGalaxy(CGPos,CGBright,CGShape,CGBright,CGShape)
    tractor.removeSource(EG)
    tractor.addSource(CG)

    for i in range(itune2):
        if (i % 5 == 0):
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=True)
        else:
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=False)
        tractor.clearCache()
        save('itune2-%d-' % (i+1)+prefix,tractor,zr,debug=True)

    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1,sky=True)
        save('ntune-%d-' % (i+1)+prefix,tractor,zr,debug=True)
        tractor.clearCache()

    makeflipbook(prefix,len(tractor.getImages()),itune1,itune2,ntune)

def makeflipbook(prefix,numImg,itune1=0,itune2=0,ntune=0):
    # Create a tex flip-book of the plots

    def allImages(title,imgpre):
        page = r'''
        \begin{frame}{%s}
        \plot{data-%s}
        \plot{model-%s} \\
        \plot{diff-%s}
        \plot{chi-%s} \\
        \end{frame}'''
        temp = ''
        for j in range(numImg):
            if j % 5 == 0:
                temp+= page % ((title+', %d' % (j),) + (imgpre+'-%d' % (j),)*4)
        return temp

    tex = r'''
    \documentclass[compress]{beamer}
    \usepackage{helvet}
    \newcommand{\plot}[1]{\includegraphics[width=0.5\textwidth]{#1}}
    \begin{document}
    '''
    
#    tex+=allImages('Initial Model','initial-'+prefix)
#    tex+=allImages('Removed','removed-'+prefix)
    tex+=allImages('Added','added-'+prefix)
    for i in range(itune1):
        tex+=allImages('Galaxy tuning, step %d' % (i+1),'itune1-%d-' %(i+1)+prefix)
    for i in range(itune2):
        tex+=allImages('Galaxy tuning (w/ Composite), step %d' % (i+1),'itune2-%d-' %(i+1)+prefix)
    for i in range(ntune):
        tex+=allImages('All tuning, step %d' % (i+1),'ntune-%d-' % (i+1)+prefix)
    
    tex += r'\end{document}' + '\n'
    fn = 'flip-' + prefix + '.tex'
    print 'Writing', fn
    open(fn, 'wb').write(tex)
    os.system("pdflatex '%s'" % fn)



if __name__ == '__main__':
    import cProfile
    import sys
    from datetime import tzinfo, timedelta, datetime
    cProfile.run('main()','prof-%s.dat' % (datetime.now().isoformat()))
    sys.exit(0)
