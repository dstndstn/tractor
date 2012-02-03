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
    x0 = 200
    x1 = 900
    y0 = 400
    y1 = 850

    roi = [x0,x1,y0,y1]
    
    bands=['r','g','i','u','z']
    bandname = 'r'

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

    # for src in sources:
    #     if isinstance(src,st.CompositeGalaxy):
    #         x,y = wcs.positionToPixel(src,src.getPosition())
    #         if (80 < x < 100 and 275 < y < 310):
    #             print src,x,y
    #             tractor.removeSource(src)


    ra,dec = 152.041958,53.083472

    r = 200
    itune = 10
    ntune = 5
    prefix = 'ugc5459'

    saveBands('initial-'+prefix, tractor,zr,bands,debug=True)

    xtr,ytr = wcs.positionToPixel(None,RaDecPos(ra,dec))
    
    print xtr,ytr

    xt = 250. #Moving to the left for better results
    yt = 210.
    for src in sources:
        xs,ys = wcs.positionToPixel(src,src.getPosition())
        if (xs-xt)**2+(ys-yt)**2 <= r**2:
            print "Removed:", src
            print xs,ys
            tractor.removeSource(src)

    saveBands('removed-'+prefix, tractor,zr,bands,debug=True)

    bright = Mags(r=20.,u=20.,g=20.,z=20.,i=20.,order=['r','u','g','z','i'])
    shape = st.GalaxyShape(60.,0.1,1.)
    shape2 = st.GalaxyShape(60.,0.3,89.)
    print bright
    print shape
    print shape2

    CG = st.CompositeGalaxy(RaDecPos(ra,dec),bright,shape,bright,shape2)
    print CG
    tractor.addSource(CG)


    saveBands('added-'+prefix,tractor,zr,bands,debug=True)
    for i in range(itune):
        tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=False)
        tractor.clearCache()
        saveBands('itune-%d-' % (i+1)+prefix,tractor,zr,bands,debug=True)

    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1,sky=True)
        saveBands('ntune-%d-' % (i+1)+prefix,tractor,zr,bands,debug=True)
        tractor.clearCache()

    makeflipbook(prefix,bands,itune,ntune)

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
