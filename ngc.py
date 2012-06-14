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

def plotarea(ra, dec, radius, ngcnum, tims=None):
    from astrometry.util.util import Tan
    W,H = 512,512
    scale = (radius * 60. * 4) / float(W)
    imgfn = 'sdss-mosaic-ngc%04i.png' % ngcnum
    if not os.path.exists(imgfn):
        url = ('http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?ra=%f&dec=%f&scale=%f&width=%i&height=%i' %
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
    plt.gca().add_artist(matplotlib.patches.Circle(xy=(x,y), radius=R, color='g', lw=3, alpha=0.5, fc='none'))
    if tims is not None:
        for tim in tims:
            H,W = tim.shape
            twcs = tim.getWcs()
            px,py = [],[]
            for x,y in [(1,1),(W,1),(W,H),(1,H),(1,1)]:
                rd = twcs.pixelToPosition(x,y)
                xx,yy = wcs.radec2pixelxy(rd.ra, rd.dec)
                px.append(xx)
                py.append(yy)
            plt.plot(px, py, 'g-', lw=3, alpha=0.5)
    plt.axis(ax)
    fn = 'ngc-%04i.png' % ngcnum
    plt.savefig(fn)
    print 'saved', fn

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
    itune1 = 7
    itune2 = 7
    ntune = 0
    IRLS_scale = 25.
    radius = j.size
    dr8 = True
    noarcsinh = False

    print 'Radius', radius
    print 'RA,Dec', ra, dec

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

    timgs = []
    sources = []
    allsources = []
    for rcf in rcfs:
        print rcf
        roi = None
        for band in bands:
            tim,tinf = getim(rcf[0], rcf[1], rcf[2], band,roiradecsize=(ra,dec,(radius*60.)/0.396),**imkw)
            if tim is None:
                print "Zero roi"
                break
            if roi is None:
                roi = tinf['roi']
            tim.zr = tinf['zr']
            timgs.append(tim)
        if tim is None:
            continue
        s = getsrc(rcf[0], rcf[1], rcf[2],bandname,roi=roi,bands=bands)
        sources.append(s)
        allsources.extend(s)

    plotarea(ra, dec, radius, ngc, timgs)

    lvl = logging.DEBUG
    logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
    tractor = st.SDSSTractor(timgs, allsources)

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
    for starx,stary in zip(sra,sdec):
        print starx,stary
        starx,stary = wcs.positionToPixel(RaDecPos(starx,stary))
        print starx,stary
        for img in tractor.getImages():
            star =  [(x,y) for x in range(img.getWidth()) for y in range(img.getHeight()) if (x-starx)**2+(y-stary)**2 <= starr**2]
            for (x,y) in star:
                img.getInvError()[y][x] = 0

    ### THIS ONLY MAKES SENSE FOR SINGLE-BAND -- there's one "timg" for each band
    ### but only one "sources" per r,c,f.
    for timg,sources in zip(timgs,sources):
        wcs = timg.getWcs()
        xtr,ytr = wcs.positionToPixel(RaDecPos(ra,dec))
    
        print xtr,ytr

        xt = xtr 
        yt = ytr
        r = ((radius*60.)/2.)/.396 #radius in pixels
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
