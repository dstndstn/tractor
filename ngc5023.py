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

def main():
    run = [3818,3813,3813]
    field = [231,290,289]
    camcol = [3,3,3]
    x00 = 1400 #Field of view for the image
    x01 = 2000
    y00 = 900
    y01 = 1500
    

    roi0 = [x00,x01,y00,y01]
    roi1 = [0,600,0,400]
    roi2 = [0,500,1200,1500]
    roi = [roi0,roi1,roi2]
    
    ra = 198.05
    dec = 44.0333
    itune = 5
    ntune = 2

    bands=['r','g','u','i','z']
    bandname = 'r'
    flipBands = ['r']

    rerun = 0

    TI = []
    sources = []
    for run,field,camcol,roi in zip(run,field,camcol,roi):
        TI.extend([st.get_tractor_image(run, camcol, field, bandname,useMags=True,roi=roi) for bandname in bands])
        sources.append(st.get_tractor_sources(run, camcol, field,bandname,bands=bands,roi=roi))

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

    prefix = 'ngc5023'
    saveAll('initial-'+prefix, tractor,zr,flipBands,debug=True)
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
            xs,ys = wcs.positionToPixel(src.getPosition(), src)
            if (xs-xt)**2+(ys-yt)**2 <= r**2:
                if isinstance(src,st.CompositeGalaxy):
                    brightE = src.brightnessExp
                    brightD = src.brightnessDev
                    sumbright = sum([brightE.getMag(bandname)+brightD.getMag(bandname) for bandname in bands])
                    if sumbright < lowbright:
                        print("GREATER")
                        lowBrightE = brightE
                        lowBrightD = brightD
                        lowShapeE = src.shapeExp
                        lowShapeD = src.shapeDev
                print "Removed:", src
                print xs,ys
                tractor.removeSource(src)

    saveAll('removed-'+prefix, tractor,zr,flipBands,debug=True)
    CG = st.CompositeGalaxy(RaDecPos(ra,dec),lowBrightE,lowShapeE,lowBrightD,lowShapeD)
    print CG
    tractor.addSource(CG)


    saveAll('added-'+prefix,tractor,zr,flipBands,debug=True)


    for i in range(itune):
        tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=True)
        tractor.clearCache()
        saveAll('itune-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)

    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1,sky=True)
        saveAll('ntune-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)
        tractor.clearCache()

    makeflipbook(prefix,len(tractor.getImages()),itune,ntune)

def makeflipbook(prefix,numImg,itune=0,ntune=0):
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
    for i in range(itune):
        tex+=allImages('Galaxy tuning, step %d' % (i+1),'itune-%d-' %(i+1)+prefix)

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
