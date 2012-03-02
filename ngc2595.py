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
    itune = 6
    ntune = 3

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
    CG = st.ExpGalaxy(RaDecPos(ra,dec),newBright,newShape)
    print CG
    tractor.addSource(CG)


    saveAll('added-'+prefix,tractor,zr,flipBands,debug=True)

    for i in range(itune):
        if (i % 5 == 0):
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=True)
        else:
            tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=False)
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
