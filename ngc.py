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
from tractor.overview import fieldPlot
from tractor.tychodata import tychoMatch
from astrometry.util.ngc2000 import *
from astrometry.util.sdss_radec_to_rcf import *
import optparse

def main():

    import optparse
    parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')

    opt,args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)

    ngc = int(args[0])

    j = get_ngc(ngc)
    ra = j.ra
    dec = j.dec
    itune1 = 5
    itune2 = 5
    ntune = 0
    IRLS_scale = 25.
    radius = j.size/2.
    dr8 = True
    noarcsinh = False

    print radius
    print ra
    print dec

#    sras, sdecs, smags = tychoMatch(ra,dec,(radius*4.)/60.)
#    print sras
#    print sdecs
#    print smags

#    for sra,sdec,smag in zip(sras,sdecs,smags):
#        print sra,sdec,smag



    sra = [] #Temporary for now...
    sdec = []
    smag = [] 
    rcfs = radec_to_sdss_rcf(ra,dec,radius=math.hypot(radius,13./2.),tablefn="dr8fields.fits")
    print rcfs

    fieldPlot(ra,dec,radius,ngc)
    
    imkw = {}
    if dr8:
        getim = st.get_tractor_image_dr8
        getsrc = st.get_tractor_sources_dr8
        imkw.update(zrange=[-3,100])
    else:
        getim = st.get_tractor_image
        getsrc = st.get_tractor_sources_dr8
        imkw.update(useMags=True)

    bands=['u','g','r','i','z']
    bandname = 'r'
    flipBands = ['r']

    rerun = 0

    TI = []
    TItemps = []
    sources = []
    for rcf in rcfs:
        print rcf
        for band in bands:
            TItemp,tinf = getim(rcf[0], rcf[1], rcf[2], band,roiradecsize=(ra,dec,(radius*60.)/0.396),**imkw)
            TItemp.zr = tinf['zr']
            TItemps.append([TItemp,tinf])
        print TItemps
        print TItemps[0]

        timg,info = TItemps[0]
        if timg is None:
            print "Zero roi"
            continue

        print info['roi']
        sources.append(getsrc(rcf[0], rcf[1], rcf[2],bandname,roi=info['roi'],bands=bands))
        TI.extend(TItemps)

    print TI
    timg,info = TI[0]
    photocal = timg.getPhotoCal()

    wcs = timg.getWcs()
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
    tims = [timg for timg,tinf in TI]
    tractor = st.SDSSTractor(tims)
    for source in sources:
        print 'Adding sources', source
        tractor.addSources(source)

    sa = dict(debug=True, plotAll=False,plotBands=False)

    if noarcsinh:
        sa.update(nlscale=0)
    elif dr8:
        sa.update(chilo=-50.,chihi=50.)

    zr = info['zr']
    print "zr is: ",zr
    print info

    print bands

    timgs = tractor.getImages()
#    for timg,band in zip(timgs,bands):
#        data = timg.getImage()/np.sqrt(timg.getInvvar())
#        plt.hist(data,bins=100)
#        plt.savefig('hist-%s.png' % (band))

    prefix = 'ngc%d' % (ngc)
    saveAll('initial-'+prefix, tractor,**sa)
    plotInvvar('initial-'+prefix,tractor)
    bright = None
    lowbright = 1000


    starr = 25.
    for starx,stary in zip(sra,sdec):
        print starx,stary
        starx,stary = wcs.positionToPixel(RaDecPos(starx,stary))
        print starx,stary
        for img in tractor.getImages():
            star =  [(x,y) for x in range(img.getWidth()) for y in range(img.getHeight()) if (x-starx)**2+(y-stary)**2 <= starr**2]
            for (x,y) in star:
                img.getInvError()[y][x] = 0

    for timg,sources in zip(tims,sources):
        wcs = timg.getWcs()
        xtr,ytr = wcs.positionToPixel(RaDecPos(ra,dec))
    
        print xtr,ytr

        xt = xtr 
        yt = ytr
        r = (radius*60.)/.396 #radius in pixels
        for src in sources:
            xs,ys = wcs.positionToPixel(src.getPosition(),src)
            if (xs-xt)**2+(ys-yt)**2 <= r**2:
                print "Removed:", src
                print xs,ys
                tractor.removeSource(src)

    saveAll('removed-'+prefix, tractor,**sa)
    newShape = sg.GalaxyShape(30.,1.,0.)
    newBright = ba.Mags(r=15.0,g=15.0,u=15.0,z=15.0,i=15.0,order=['u','g','r','i','z'])
    EG = st.ExpGalaxy(RaDecPos(ra,dec),newBright,newShape)
    print EG
    tractor.addSource(EG)


    saveAll('added-'+prefix,tractor,**sa)

    for i in range(itune1):
        tractor.optimizeCatalogLoop(nsteps=1,srcs=[EG],sky=True)
        tractor.changeInvvar(IRLS_scale)
        saveAll('itune1-%d-' % (i+1)+prefix,tractor,flipBands,**sa)

    CGPos = EG.getPosition()
    CGShape1 = EG.getShape().copy()
    CGShape2 = EG.getShape().copy()
    EGBright = EG.getBrightness()

    CGu = EGBright[0] + 0.75
    CGg = EGBright[1] + 0.75
    CGr = EGBright[2] + 0.75
    CGi = EGBright[3] + 0.75
    CGz = EGBright[4] + 0.75
    CGBright1 = ba.Mags(r=CGr,g=CGg,u=CGu,z=CGz,i=CGi,order=['u','g','r','i','z'])
    CGBright2 = ba.Mags(r=CGr,g=CGg,u=CGu,z=CGz,i=CGi,order=['u','g','r','i','z'])
    print EGBright
    print CGBright1

    CG = st.CompositeGalaxy(CGPos,CGBright1,CGShape1,CGBright2,CGShape2)
    tractor.removeSource(EG)
    tractor.addSource(CG)

    for i in range(itune2):
        tractor.optimizeCatalogLoop(nsteps=1,srcs=[CG],sky=True)
        tractor.changeInvvar(IRLS_scale)
        saveAll('itune2-%d-' % (i+1)+prefix,tractor,**sa)

    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1,sky=True)
        tractor.changeInvvar(IRLS_scale)
        saveAll('ntune-%d-' % (i+1)+prefix,tractor,**sa)
    plotInvvar('final-'+prefix,tractor)
    sa.update(plotBands=True)
    saveAll('allBands-' + prefix,tractor,**sa)

    print CG
    print CG.getPosition()
    print CGBright1
    print CGBright2
    print CGShape1
    print CGShape2
    print CGBright1+CGBright2
    print CG.getBrightness()

    makeflipbook(prefix,len(tractor.getImages()),itune1,itune2,ntune)

def makeflipbook(prefix,numImg,itune1=0,itune2=0,ntune=0):
    # Create a tex flip-book of the plots

    def allImages(title,imgpre,allBands=False):
        page = r'''
        \begin{frame}{%s}
        \plot{data-%s}
        \plot{model-%s} \\
        \plot{diff-%s}
        \plot{chi-%s} \\
        \end{frame}'''
        temp = ''
        for j in range(numImg):
            if j % 5 == 2 or allBands:
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

    tex+=allImages('All Bands','allBands-'+prefix,True)
    
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
