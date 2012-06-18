# -*- mode: python; indent-tabs-mode: nil -*-
# (this tells emacs to indent with spaces)
import matplotlib
matplotlib.use('Agg')

import os
import logging
import urllib2
import tempfile
import numpy as np
import pylab as plt
import pyfits

from astrometry.util.file import *
from astrometry.util.multiproc import multiproc

from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import sdss_galaxy as sg
from tractor import basics as ba
from tractor.overview import fieldPlot
from tractor.tychodata import tychoMatch
from tractor.rc3 import getNGC
from astrometry.util.ngc2000 import *
from astrometry.util.sdss_radec_to_rcf import *
import optparse

def plotarea(ra, dec, radius, ngcnum, tims=None, rds=[]):
    from astrometry.util.util import Tan
    W,H = 512,512
    scale = (radius * 60. * 4) / float(W)
    print 'SDSS jpeg scale', scale
    imgfn = 'sdss-mosaic-ngc%04i.png' % ngcnum
    if not os.path.exists(imgfn):
        url = (('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?' +
                'ra=%f&dec=%f&scale=%f&width=%i&height=%i') %
               (ra, dec, scale, W, H))
        f = urllib2.urlopen(url)
        of,tmpfn = tempfile.mkstemp(suffix='.jpg')
        os.close(of)
        of = open(tmpfn, 'wb')
        of.write(f.read())
        of.close()
        cmd = 'jpegtopnm %s | pnmtopng > %s' % (tmpfn, imgfn)
        os.system(cmd)
    # Create WCS header for it
    cd = scale / 3600.
    args = (ra, dec, W/2. + 0.5, H/2. + 0.5, -cd, 0., 0., -cd, W, H)
    wcs = Tan(*[float(x) for x in args])

    plt.clf()
    I = plt.imread(imgfn)
    plt.imshow(I, interpolation='nearest', origin='lower')
    x,y = wcs.radec2pixelxy(ra, dec)
    R = radius * 60. / scale
    ax = plt.axis()
    plt.gca().add_artist(matplotlib.patches.Circle(xy=(x,y), radius=R, color='g',
                                                   lw=3, alpha=0.5, fc='none'))
    if tims is not None:
        print 'Plotting outlines of', len(tims), 'images'
        for tim in tims:
            H,W = tim.shape
            twcs = tim.getWcs()
            px,py = [],[]
            for x,y in [(1,1),(W,1),(W,H),(1,H),(1,1)]:
                rd = twcs.pixelToPosition(x,y)
                xx,yy = wcs.radec2pixelxy(rd.ra, rd.dec)
                print 'x,y', x, y
                x1,y1 = twcs.positionToPixel(rd)
                print '  x1,y1', x1,y1
                print '  r,d', rd.ra, rd.dec,
                print '  xx,yy', xx, yy
                px.append(xx)
                py.append(yy)
            plt.plot(px, py, 'g-', lw=3, alpha=0.5)

            # plot full-frame image outline too
            # px,py = [],[]
            # W,H = 2048,1489
            # for x,y in [(1,1),(W,1),(W,H),(1,H),(1,1)]:
            #     r,d = twcs.pixelToRaDec(x,y)
            #     xx,yy = wcs.radec2pixelxy(r,d)
            #     px.append(xx)
            #     py.append(yy)
            # plt.plot(px, py, 'g-', lw=1, alpha=1.)

    if rds is not None:
        px,py = [],[]
        for ra,dec in rds:
            print 'ra,dec', ra,dec
            xx,yy = wcs.radec2pixelxy(ra, dec)
            px.append(xx)
            py.append(yy)
        plt.plot(px, py, 'go')

    plt.axis(ax)
    fn = 'ngc-%04i.png' % ngcnum
    plt.savefig(fn)
    print 'saved', fn

def get_ims_and_srcs((r,c,f,rr,dd, bands, ra, dec, roipix, imkw, getim, getsrc)):
    tims = []
    roi = None
    for band in bands:
        tim,tinf = getim(r, c, f, band, roiradecsize=(ra,dec,roipix), **imkw)
        if tim is None:
            print "Zero roi"
            return None,None
        if roi is None:
            roi = tinf['roi']
        tim.zr = tinf['zr']
        tims.append(tim)
    s = getsrc(r, c, f, roi=roi, bands=bands)
    return (tims,s)

def main():
    import optparse
    parser = optparse.OptionParser(usage='%prog [options] <NGC-number>')
    parser.add_option('--threads', dest='threads', type=int, help='use multiprocessing')

    opt,args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)

    if opt.threads:
        mp = multiproc(nthreads=opt.threads)
    else:
        mp = multiproc()

    ngc = int(args[0])

    j = getNGC(ngc)
    print j
    ra = float(j['RA'][0])
    dec = float(j['DEC'][0])
    itune1 = 6
    itune2 = 6
    ntune = 0
    IRLS_scale = 25.
    radius = (10.**j['LOG_D25'][0])/10.
    dr8 = True
    noarcsinh = False

    print 'Radius', radius
    print 'RA,Dec', ra, dec

    sras, sdecs, smags = tychoMatch(ra,dec,(radius*4.)/60.)
    print sras
    print sdecs
    print smags
    sras=[]
    sdecs=[]
    smags=[]

    for sra,sdec,smag in zip(sras,sdecs,smags):
        print sra,sdec,smag



    rcfs = radec_to_sdss_rcf(ra,dec,radius=math.hypot(radius,13./2.),tablefn="dr8fields.fits")
    print rcfs

    #fieldPlot(ra,dec,radius,ngc)
    
    imkw = dict(psf='dg')
    if dr8:
        getim = st.get_tractor_image_dr8
        getsrc = st.get_tractor_sources_dr8
        imkw.update(zrange=[-3,100])
    else:
        getim = st.get_tractor_image
        getsrc = st.get_tractor_sources_dr8
        imkw.update(useMags=True)

    bands=['u','g','r','i','z']
    #bands=['r']
    bandname = 'r'
    flipBands = ['r']

    imsrcs = mp.map(get_ims_and_srcs, [(rcf + (bands, ra, dec, radius*60./0.396, imkw, getim, getsrc))
                                       for rcf in rcfs])
    timgs = []
    sources = []
    allsources = []
    for ims,s in imsrcs:
        if ims is None:
            continue
        timgs.extend(ims)
        allsources.extend(s)
        sources.append(s)

    #rds = [rcf[3:5] for rcf in rcfs]
    plotarea(ra, dec, radius, ngc, timgs) #, rds)
    
    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
    tractor = st.Tractor(timgs, allsources, mp=mp)

    sa = dict(debug=True, plotAll=False,plotBands=False)

    if noarcsinh:
        sa.update(nlscale=0)
    elif dr8:
        sa.update(chilo=-50.,chihi=50.)

    zr = timgs[0].zr
    print "zr is: ",zr

    print bands

    timgs = tractor.getImages()
    print "Number of images: ", len(timgs)
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
    for sra,sdec in zip(sras,sdecs):
        print sra,sdec

        for img in tractor.getImages():
            wcs = img.getWcs()
            invvar = img.getInvvar() 
            starx,stary = wcs.positionToPixel(RaDecPos(sra,sdec))
            star =  [(x,y) for x in range(img.getWidth()) for y in range(img.getHeight()) if (x-starx)**2+(y-stary)**2 <= starr**2]
            for (x,y) in star:
                img.getStarMask()[y][x] = 0

    for timgs,sources in imsrcs:
        timg = timgs[0]
        wcs = timg.getWcs()
        xtr,ytr = wcs.positionToPixel(RaDecPos(ra,dec))
    
        print xtr,ytr

        xt = xtr 
        yt = ytr
        r = ((radius*60.))/.396 #radius in pixels
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
        saveAll('itune1-%d-' % (i+1)+prefix,tractor,**sa)

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

    result = open('ngc-%s.txt' % (ngc),'w')

    result.write(str(CG))
    result.write(str(CG.getBrightness()))
    result.close()

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
    # To profile the code, you can do:
    #import cProfile
    #import sys
    #from datetime import tzinfo, timedelta, datetime
    #cProfile.run('main()','prof-%s.dat' % (datetime.now().isoformat()))
    #sys.exit(0)
    main()
    
