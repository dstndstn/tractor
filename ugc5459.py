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
    run = 2863
    field = 180
    camcol = 4
    x0 = 200 #Field of view for the image
    x1 = 900
    y0 = 400
    y1 = 850
    starx = 200. #Star location
    stary = 300.
    starr = 50. #Star radius

    roi = [x0,x1,y0,y1]
    
    bands=['r','g','i','u','z']
    bandname = 'r'
    flipBands = ['r']

    rerun = 0

    TI = []
    TI.extend([st.get_tractor_image(run, camcol, field, bandname,roi=roi,useMags=True) for bandname in bands])
    sources = st.get_tractor_sources(run, camcol, field,bandname, bands=bands,roi=roi)

    timg,info = TI[0]
    photocal = timg.getPhotoCal()

    wcs = timg.getWcs()
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)

    tims = [timg for timg,tinf in TI]
    tractor = st.SDSSTractor(tims)
    tractor.addSources(sources)

    zr = np.array([-5.,+5.]) * info['skysig']

    print bands

    ra,dec = 152.041958,53.083472

    r = 200
    itune = 10
    ntune = 5
    prefix = 'ugc5459'

    saveBands('initial-'+prefix, tractor,zr,flipBands,debug=True)

    xtr,ytr = wcs.positionToPixel(RaDecPos(ra,dec))
    
    print xtr,ytr
    bright = None
    lowbright = 1000

    xt = 250. #Moving to the left for better results
    yt = 210.
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

    saveBands('removed-'+prefix, tractor,zr,flipBands,debug=True)

#    bright = Mags(r=15.,u=15.,g=15.,z=15.,i=15.,order=['r','u','g','z','i'])
#    shape = st.GalaxyShape(re,ab,phi)
#    shape2 = st.GalaxyShape(30.,0.3,45.)
#    print bright
#    print shape
#   print shape2

    CG = st.CompositeGalaxy(RaDecPos(ra,dec),lowBrightE,lowShapeE,lowBrightD,lowShapeD)
    print CG
    tractor.addSource(CG)


    saveBands('added-'+prefix,tractor,zr,flipBands,debug=True)


    for img in tractor.getImages():
        star =  [(x,y) for x in range(img.getWidth()) for y in range(img.getHeight()) if (x-starx)**2+(y-stary)**2 <= starr**2]
        for (x,y) in star:
            img.getInvError()[y][x] = 0


    saveBands('nostar-'+prefix,tractor,zr,flipBands,debug=True)
    for i in range(itune):
        tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=False)
        tractor.clearCache()
        saveBands('itune-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)

    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1,sky=True)
        saveBands('ntune-%d-' % (i+1)+prefix,tractor,zr,flipBands,debug=True)
        tractor.clearCache()

    makeflipbook(prefix,flipBands,itune,ntune)

def makeflipbook(prefix,bands,itune=0,ntune=0):
    # Create a tex flip-book of the plots

    def allBands(title,imgpre):
        page = r'''
        \begin{frame}{%s}
        \plot{data-%s}
        \plot{model-%s} \\
        \plot{diff-%s}
        \plot{chi-%s} \\
        \end{frame}'''
        temp = ''
        for j,band in enumerate(bands):
            temp+= page % ((title+', Band: %s' % (band),) + (imgpre+'-%s-' % (band),)*4)
        return temp

    tex = r'''
    \documentclass[compress]{beamer}
    \usepackage{helvet}
    \newcommand{\plot}[1]{\includegraphics[width=0.5\textwidth]{#1}}
    \begin{document}
    '''
    
    tex+=allBands('Initial Model','initial-'+prefix)
    tex+=allBands('Removed','removed-'+prefix)
    tex+=allBands('Added','added-'+prefix)
    tex+=allBands('No Star','nostar-'+prefix)
    for i in range(itune):
        tex+=allBands('Galaxy tuning, step %d' % (i+1),'itune-%d-' %(i+1)+prefix)

    for i in range(ntune):
        tex+=allBands('All tuning, step %d' % (i+1),'ntune-%d-' % (i+1)+prefix)
    
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
