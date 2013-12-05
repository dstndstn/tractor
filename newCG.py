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
import resource
import gc

from astrometry.util.file import *
from astrometry.util.multiproc import multiproc

from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import sdss_galaxy as sg
from tractor import sdss as st
from tractor import basics as ba
#from tractor.overview import fieldPlot
from tractor.tychodata import tychoMatch
from tractor.rc3 import getName
from tractor.cache import *
from astrometry.util.sdss_radec_to_rcf import *
import optparse

def plotarea(ra, dec, radius, name, prefix, tims=None, rds=[]):
    from astrometry.util.util import Tan
    W,H = 512,512
    scale = (radius * 60. * 4) / float(W)
    print 'SDSS jpeg scale', scale
    imgfn = 'sdss-mosaic-%s.png' % prefix
    if not os.path.exists(imgfn):
        url = (('http://skyservice.pha.jhu.edu/DR9/ImgCutout/getjpeg.aspx?' +
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
    fn = '%s.png' % prefix
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
        tims.append(tim)
    s = getsrc(r, c, f, roi=roi, bands=bands)
    return (tims,s)


def generalRC3(name,threads=None,itune1=5,itune2=5,ntune=0,nocache=False,scale=1,ra=None,dec=None,ab=1.,angle=0.,radius=None):
    entry = getName(name,fn="mediumrc3.fits")
    print entry
    if ra is None:
        ra = entry['RA'][0]
    if dec is None:
        dec = entry['DEC'][0]
    log_ae = float(entry['LOG_AE'][0])
    log_d25 = float(entry['LOG_D25'][0])
    print 'LOG_AE is %s' % log_ae
    print 'LOG_D25 is %s' % log_d25
    
    
    if radius is not None:
        fieldradius = float(radius)
        remradius = float(radius)
#    if log_ae != 0:
        #fieldradius = 10.*(10.**log_ae)/10.
        #remradius = 10.*(10.**log_ae)/10.
    elif log_d25 !=0:
        #print 'No log_AE, using d25'
        fieldradius = (10.**log_d25)/10.
        remradius = (10.**log_d25)/10.
    else:
        print 'No d_25, using default values'
        fieldradius = 3.
        remradius = 2.        
    
    general(name,float(ra),float(dec),remradius,fieldradius,threads=threads,itune1=itune1,itune2=itune2,ntune=ntune,nocache=nocache,scale=scale,ab=float(ab),angle=float(angle))

def generalNSAtlas (nsid,threads=None,itune1=5,itune2=5,ntune=0,nocache=False,scale=1,fieldradius=0,ra=None,dec=None,ab=1,angle=0):
    data = pyfits.open("nsa-short.fits.gz")[1].data
    e=data.field('NSAID')

    mask = e == nsid
    record = data[mask]

    print record

    if fieldradius==0:
        fieldradius=record['SERSIC_TH50'][0]

    print "Radius is %e" % fieldradius
    if ra is None:
        ra = record['RA'][0]
    if dec is None:
        dec = record['DEC'][0]

    general("NSA_ID_%s" % nsid,float(ra),float(dec),fieldradius/60.,fieldradius/60.,threads=threads,itune1=itune1,itune2=itune2,ntune=ntune,nocache=nocache,scale=scale,ab=float(ab),angle=float(angle))


def general(name,ra,dec,remradius,fieldradius,threads=None,itune1=5,itune2=5,ntune=0,nocache=False,scale=1,ab=1.,angle=0.):
    #Radius should be in arcminutes
    if threads:
        mp = multiproc(nthreads=threads)
    else:
        mp = multiproc()

    IRLS_scale = 25.
    dr9 = True
    dr8 = False
    noarcsinh = False
    print name

    prefix = 'swapCG_%s' % (name.replace(' ', '_'))
    print 'Removal Radius', remradius
    print 'Field Radius', fieldradius
    print 'RA,Dec', ra, dec

    print os.getcwd()
    print ra,dec,math.hypot(fieldradius,13./2.)

    rcfs = radec_to_sdss_rcf(ra,dec,radius=math.hypot(fieldradius,13./2.),tablefn="dr9fields.fits")
    print 'RCFS:', rcfs
    print len(rcfs)
    assert(len(rcfs)>0)
    if 10 <= len(rcfs) < 20:
        scale = 2
    elif 20 <= len(rcfs) < 40:
        scale = 4
    elif 40 <= len(rcfs) < 80:
        scale = 8
    assert(len(rcfs)<80)

    sras, sdecs, smags = tychoMatch(ra,dec,(fieldradius*1.5)/60.)

    imkw = dict(psf='kl-gm')
    if dr9:
        getim = st.get_tractor_image_dr9
        getsrc = st.get_tractor_sources_dr9
        imkw.update(zrange=[-3,100])
    elif dr8:
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
    print rcfs

    imsrcs = mp.map(get_ims_and_srcs, [(rcf + (bands, ra, dec, fieldradius*60./0.396, imkw, getim, getsrc))
                                       for rcf in rcfs])
    timgs = []
    sources = []
    allsources = []
    for ims,s in imsrcs:
        if ims is None:
            continue
        if s is None:
            continue
        if scale > 1:
            for im in ims:
                timgs.append(st.scale_sdss_image(im,scale))
        else:
            timgs.extend(ims)
        allsources.extend(s)
        sources.append(s)

    #rds = [rcf[3:5] for rcf in rcfs]
    #plotarea(ra, dec, fieldradius, name, prefix, timgs) #, rds)
    
    #lvl = logging.DEBUG
    #logging.basicConfig(level=lvl,format='%(message)s',stream=sys.stdout)
    tractor = st.Tractor(timgs, allsources, mp=mp)

    sa = dict(debug=True, plotAll=False,plotBands=False)

    if noarcsinh:
        sa.update(nlscale=0)
    elif dr8 or dr9:
        sa.update(chilo=-8.,chihi=8.)

    if nocache:
        tractor.cache = NullCache()
        sg.disable_galaxy_cache()

    zr = timgs[0].zr
    print "zr is: ",zr

    print bands

    print "Number of images: ", len(timgs)
    #for timg,band in zip(timgs,bands):
    #    data = timg.getImage()/np.sqrt(timg.getInvvar())
    #    plt.hist(data,bins=100)
    #    plt.savefig('hist-%s.png' % (band))

    saveAll('initial-'+prefix, tractor,**sa)
    #plotInvvar('initial-'+prefix,tractor)

    

    for sra,sdec,smag in zip(sras,sdecs,smags):

        for img in tractor.getImages():
            wcs = img.getWcs()
            starx,stary = wcs.positionToPixel(RaDecPos(sra,sdec))
            starr=25*(2**(max(11-smag,0.)))
            if starx+starr<0. or starx-starr>img.getWidth() or stary+starr <0. or stary-starr>img.getHeight():
                continue
            X,Y = np.meshgrid(np.arange(img.getWidth()), np.arange(img.getHeight()))
            R2 = (X - starx)**2 + (Y - stary)**2
            img.getStarMask()[R2 < starr**2] = 0

    for timgs,sources in imsrcs:
        timg = timgs[0]
        wcs = timg.getWcs()
        xtr,ytr = wcs.positionToPixel(RaDecPos(ra,dec))
    
        xt = xtr 
        yt = ytr
        r = ((remradius*60.))/.396 #radius in pixels
        for src in sources:
            xs,ys = wcs.positionToPixel(src.getPosition(),src)
            if (xs-xt)**2+(ys-yt)**2 <= r**2:
                #print "Removed:", src
                #print xs,ys
                tractor.removeSource(src)

    #saveAll('removed-'+prefix, tractor,**sa)
    newShape = sg.GalaxyShape((remradius*60.)/10.,ab,angle)
    newBright = ba.Mags(r=15.0,g=15.0,u=15.0,z=15.0,i=15.0,order=['u','g','r','i','z'])
    EG = st.ExpGalaxy(RaDecPos(ra,dec),newBright,newShape)
    print EG
    tractor.addSource(EG)


    saveAll('added-'+prefix,tractor,**sa)

    #print 'Tractor has', tractor.getParamNames()

    for im in tractor.images:
        im.freezeAllParams()
        im.thawParam('sky')
    tractor.catalog.freezeAllBut(EG)

    #print 'Tractor has', tractor.getParamNames()
    #print 'values', tractor.getParams()

    for i in range(itune1):
        tractor.optimize()
        tractor.changeInvvar(IRLS_scale)
        saveAll('itune1-%d-' % (i+1)+prefix,tractor,**sa)
        tractor.clearCache()
        sg.get_galaxy_cache().clear()
        gc.collect()
        print resource.getpagesize()
        print resource.getrusage(resource.RUSAGE_SELF)[2]
        

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

    tractor.catalog.freezeAllBut(CG)
    print resource.getpagesize()
    print resource.getrusage(resource.RUSAGE_SELF)[2]


    for i in range(itune2):
        tractor.optimize()
        tractor.changeInvvar(IRLS_scale)
        saveAll('itune2-%d-' % (i+1)+prefix,tractor,**sa)
        tractor.clearCache()
        sg.get_galaxy_cache().clear()
        print resource.getpagesize()
        print resource.getrusage(resource.RUSAGE_SELF)[2]

    tractor.catalog.thawAllParams()
    for i in range(ntune):
        tractor.optimize()
        tractor.changeInvvar(IRLS_scale)
        saveAll('ntune-%d-' % (i+1)+prefix,tractor,**sa)
    #plotInvvar('final-'+prefix,tractor)
    sa.update(plotBands=True)
    saveAll('allBands-' + prefix,tractor,**sa)

    print "end of first round of optimization:", tractor.getLogLikelihood()
    print CG
    print CG.getPosition()
    print CGBright1
    print CGBright2
    print CGShape1
    print CGShape2
    print CGBright1+CGBright2
    print CG.getBrightness()

    pfn = '%s.pickle' % prefix
    pickle_to_file(CG,pfn)

    makeflipbook(prefix,len(tractor.getImages()),itune1,itune2,ntune)

    # now SWAP exp and dev and DO IT AGAIN
    newCG = st.CompositeGalaxy(CG.getPosition, CG.brightnessDev.copy(),
                               CG.shapeDev.copy(), CG.brightnessExp.copy(),
                               CG.shapeExp.copy())
    tractor.removeSource(CG)
    tractor.addSource(newCG)

    tractor.catalog.thawAllParams()
    for i in range(ntune):
        tractor.optimize()
        tractor.changeInvvar(IRLS_scale)
        saveAll('ntune-swap-%d-' % (i+1)+prefix,tractor,**sa)
    #plotInvvar('final-'+prefix,tractor)
    sa.update(plotBands=True)
    saveAll('allBands-swap-' + prefix,tractor,**sa)

    print "end of second (swapped) round of optimization:", tractor.getLogLikelihood()
    print newCG
    print newCG.getPosition()
    print newCG.getBrightness()

    pfn = '%s-swap.pickle' % prefix
    pickle_to_file(newCG,pfn)

    makeflipbook(prefix+"-swap",len(tractor.getImages()),itune1,itune2,ntune)

def main():
    import optparse
    parser = optparse.OptionParser(usage='%prog [options] <name>')
    parser.add_option('--threads', dest='threads', type=int, help='use multiprocessing')
    parser.add_option('--itune1',dest='itune1',type=int,help='Individual tuning, first stage',default=5)
    parser.add_option('--itune2',dest='itune2',type=int,help='Individual tuning, second stage',default=5)
    parser.add_option('--ntune',dest='ntune',type=int,help='All objects tuning',default=0)
    parser.add_option('--radius',dest='fradius',type=float,help='Search radius in arcseconds',default=1.)
    parser.add_option('--nocache',dest='nocache',action='store_true',default=False,help='Disable caching for memory reasons')
    parser.add_option('--nsatlas',dest='nsatlas',action='store_true',default=False,help='Use argument as Nasa-Sloan Atlas id')
    parser.add_option('--ab',dest='ab',type=float,help='ab ratio')
    parser.add_option('--angle',dest='angle',type=float,help='initial angle')
    parser.add_option('--ra',dest='ra',type=float,help='initial ra')
    parser.add_option('--dec',dest='dec',type=float,help='initial dec')

    opt,args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)

    threads=opt.threads

    
    itune1 = opt.itune1
    itune2 = opt.itune2
    ntune = opt.ntune
    nocache = opt.nocache
    if ab is None:
        ab=1.
    if angle is None:
        angle = 0.
    if opt.nsatlas:
        generalNSAtlas(int (args[0]),threads,itune1,itune2,ntune,nocache,fieldradius=opt.fradius)
    else:
        name = args[0].replace('_',' ')
        generalRC3(name,threads,itune1,itune2,ntune,nocache,ra=ra,dec=dec,ab=ab,angle=angle)



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
    #tex+=allImages('Removed','removed-'+prefix)
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

