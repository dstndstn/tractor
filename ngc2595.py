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

def main():
    run = [4517,4576,4576]
    field = [103,99,100]
    camcol = [2,6,6]

    roi0 = [700,1400,0,800]
    roi1 = [1300,1800,1000,1500]
    roi2 = [1400,1800,0,400]
    roi = [roi0,roi1,roi2]
    
    ra = 126.925
    dec = 21.4833
    itune1 = 2
    itune2 = 1
    ntune = 0

    bands=['r','g','u','i','z']
    bandname = 'r'
    flipBands = ['r']

    rerun = 0

    TI = []
    sources = []
    for run,field,camcol,roi in zip(run,field,camcol,roi):
        TI.extend([st.get_tractor_image(run, camcol, field, bandname,roi=roi,useMags=True) for bandname in bands])
        sources.append(st.get_tractor_sources(run, camcol, field,bandname,roi=roi,bands=bands))

    timg,info = TI[0]
    photocal = timg.getPhotoCal()

    wcs = timg.getWcs()
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)

    tims = [timg for timg,tinf in TI]
    tractor = st.SDSSTractor(tims)
    for source in sources:
        tractor.addSources(source)

    zr = np.array([-5.,+5.]) * info['skysig']

    print bands

    prefix = 'ngc2595'
    saveAll('initial-'+prefix, tractor,zr,flipBands,debug=True)
    plotInvvar('initial-'+prefix,tractor)
    bright = None
    lowbright = 1000

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

    saveAll('removed-'+prefix, tractor,zr,flipBands,debug=True)
    newShape = sg.GalaxyShape(30.,1.,0.)
    newBright = ba.Mags(r=15.0,g=15.0,u=15.0,z=15.0,i=15.0)
    EG = st.ExpGalaxy(RaDecPos(ra,dec),newBright,newShape)
    print EG
    tractor.addSource(EG)


    saveAll('added-'+prefix,tractor,zr,flipBands,debug=True)

    for i in range(itune1):
        if (i % 5 == 0):
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[EG],sky=True)
        else:
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[EG],sky=False)
        tractor.clearCache()
        saveAll('itune1-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)
    
    CGPos = EG.getPosition()
    CGShape = EG.getShape()
    EGBright = EG.getBrightness()

    CGr = EGBright[0]*1.25
    CGg = EGBright[1]*1.25
    CGu = EGBright[2]*1.25
    CGz = EGBright[3]*1.25
    CGi = EGBright[4]*1.25
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
        saveAll('itune2-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)

    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1,sky=True)
        saveAll('ntune-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)
        tractor.clearCache()
    plotInvvar('final-'+prefix,tractor)
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
    
    tex+=allImages('Initial Model','initial-'+prefix)
    tex+=allImages('Removed','removed-'+prefix)
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
