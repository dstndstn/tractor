from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import logging
import numpy as np
import pylab as plt
import fitsio
import math

from astrometry.util.file import *
from astrometry.util.plotutils import ArcsinhNormalize
from astrometry.util.starutil_numpy import *
from astrometry.sdss.fields import *

from astrometry.sdss import DR9

from tractor import *
from tractor import sdss as st

def save(idstr, tractor, nlscale=1., debug=False, plotAll=False, imgi=0,
         chilo=-10., chihi=10., roi=None):
    #print "Index: ", imgi
    mod = tractor.getModelImage(imgi)
    chi = tractor.getChiImage(imgi=imgi)
    synthfn = 'synth-%s.fits' % idstr
    print('Writing synthetic image to', synthfn)
    fitsio.write(synthfn, mod, clobber=True)

    pfn = 'tractor-%s.pickle' % idstr
    print('Saving state to', pfn)
    pickle_to_file(tractor, pfn)

    plt.clf()
    plt.hist(chi.ravel(), range=(-10,10), bins=100)
    plt.savefig('chi2.png')

    timg = tractor.getImage(imgi)
    data = timg.getImage()
    print('Mod type:', mod.dtype)
    print('Chi type:', chi.dtype)
    print('Data type:', data.dtype)
    zr = timg.zr
    print('zr', zr)
    # Set up nonlinear mapping based on the statistics of the data image.
    #sigma = np.median(timg.getInvError())
    #print 'sigma', sigma
    ima = dict(interpolation='nearest', origin='lower')
    if nlscale == 0.:
        ima.update(vmin=zr[0], vmax=zr[1])
    else:
        print(data.shape)
        q1,q2,q3 = np.percentile(data.ravel(), [25, 50, 75])
        print('Data quartiles:', q1, q2, q3)
        ima.update(norm = ArcsinhNormalize(mean=q2, std=(1./nlscale) * (q3-q1)/2.,
                                           vmin=zr[0], vmax=zr[1]))

    if roi is not None:
        ima.update(extent=roi)

    imchi = ima.copy()
    if nlscale == 0.:
        imchi.update(vmin=chilo, vmax=chihi, norm=None)
    else:
        imchi.update(norm = ArcsinhNormalize(mean=0., std=1./nlscale, vmin=chilo, vmax=chihi))

    imdiff = ima.copy()
    dzr = (zr[1] - zr[0])/2.
    if nlscale == 0.:
        imdiff.update(vmin=-dzr, vmax=+dzr, norm=None)
    else:
        imdiff.update(norm = ArcsinhNormalize(mean=0., std=1./nlscale, vmin=-dzr, vmax=dzr))

    if debug:
        sources = tractor.getCatalog()
        wcs = timg.getWcs()
        allobjx = []
        allobjy = []
        allobjc = []
        pointx = []
        pointy = []
        xplotx = []
        xploty = []

        for obj in sources:
            if (isinstance(obj,PointSource)):
                xt,yt = wcs.positionToPixel(obj.getPosition(), obj)
                pointx.append(xt)
                pointy.append(yt)
                continue
            print(type(obj))
            shapes = []
            attrType = []
            if (isinstance(obj,st.CompositeGalaxy)):
                for attr in 'shapeExp', 'shapeDev':
                    shapes.append(getattr(obj, attr))
                    attrType.append(attr)
            else:
                shapes.append(getattr(obj,'shape'))
                attrType.append(' ')
            x0,y0 = wcs.positionToPixel(obj.getPosition(), obj)
            
            cd = timg.getWcs().cdAtPixel(x0,y0)
            print("CD",cd)
            for i,shape in enumerate(shapes):
                xplotx.append(x0)
                xploty.append(y0)
                T=np.linalg.inv(shape.getTensor(cd))
                print("Inverted tensor:",T)
                print(obj.getPosition())
                print(i)

                x,y = [],[]
                for theta in np.linspace(0,2*np.pi,100):
                    ux = np.cos(theta)
                    uy = np.sin(theta)
                    dx,dy = np.dot(T,np.array([ux,uy]))
                    x.append(x0+dx)
                    y.append(y0+dy)
                allobjx.append(x)
                allobjy.append(y)
                if (attrType[i] == 'shapeExp'):
                    allobjc.append('b')
                elif attrType[i] == 'shapeDev':
                    allobjc.append('g')
                else:
                    allobjc.append('r')

    def savepng(pre, img, title=None, **kwargs):
        fn = '%s-%s.png' % (pre, idstr)
        print('Saving', fn)
        plt.clf()
        plt.imshow(img, **kwargs)
        ax = plt.axis()
        if debug:
            print(len(xplotx),len(allobjx))
            for i,(objx,objy,objc) in enumerate(zip(allobjx,allobjy,allobjc)):
                plt.plot(objx,objy,'-',c=objc)
                tempx = []
                tempx.append(xplotx[i])
                tempx.append(objx[0])
                tempy = []
                tempy.append(xploty[i])
                tempy.append(objy[0])
                plt.plot(tempx,tempy,'-',c='purple')
            plt.plot(pointx,pointy,'y.')
            plt.plot(xplotx,xploty,'xg')
        plt.axis(ax)
        if title is not None:
            plt.title(title)
        plt.colorbar()
        plt.gray()
        plt.savefig(fn)

    savepng('data', data, title='Data ' + timg.name, **ima)
    savepng('model', mod, title='Model ' + timg.name, **ima)
    savepng('diff', data - mod, title='Data - Model, ' + timg.name, **imdiff)
    savepng('chi',  chi, title='Chi ' + timg.name, **imchi)
    if plotAll:
        debug = False
        for i,src in enumerate(tractor.getCatalog()):
            savepng('data-s%i'%(i+1),data - sky, title='Data '+timg.name,**ima)
            modelimg = tractor.getModelImage(timg, srcs=[src])
            savepng('model-s%i'%(i+1), modelimg - sky, title='Model-s%i'%(i+1),**ima) 
            savepng('diff-s%i'%(i+1), data - modelimg, title='Model-s%i'%(i+1),**imdiff)
            savepng('chi-s%i'%(i+1),tractor.getChiImage(imgi,srcs=[src]),title='Chi',**imchi)

def main():
    from optparse import OptionParser
    import sys

    # Otherwise plotting code can raise floating-point errors
    np.seterr(under='print')

    tune = []
    def store_value (option, opt, value, parser):
        if opt == '--ntune':
            tune.append(('n',value))
        elif opt == '--itune':
            tune.append(('i',value))

    parser = OptionParser(usage=('%prog [options] [-r <run> -c <camcol> -f <field> -b <band>]'))
    parser.add_option('-r', '--run', dest='run', type='int')
    parser.add_option('-c', '--camcol', dest='camcol', type='int')
    parser.add_option('-f', '--field', dest='field', type='int')
    parser.add_option('-b', '--band', dest='band', help='SDSS Band (u, g, r, i, z)')
    parser.add_option('--dg', action='store_true', default=False,
                      help='Use double-Gaussian PSF model (default is Gaussian mixture approximation of pixelized model')
    parser.add_option('-R', '--radec', dest='radec', nargs=2, type=str, help='RA,Dec center: float degrees or hh:mm:ss +-dd:mm:ss')
    parser.add_option('-s', '--size', dest='pixsize', type=int, help='Pixel size when using RA,Dec center option')
    parser.add_option('--drfields', dest='drfields', help='FITS table of SDSS fields: default dr%ifields.fits, etc')
    parser.add_option('--curl', dest='curl', action='store_true', default=False, help='Use "curl", not "wget", to download files')
    parser.add_option('--ntune', action='callback', callback=store_value, type=int,  help='Improve synthetic image by locally optimizing likelihood for nsteps iterations')
    parser.add_option('--itune', action='callback', callback=store_value, type=int, nargs=2, help='Optimizes each source individually')
    parser.add_option('--roi', dest='roi', type=int, nargs=4, help='Select an x0,x1,y0,y1 subset of the image')
    parser.add_option('--prefix', dest='prefix', help='Set output filename prefix; default is the SDSS  RRRRRR-BC-FFFF string (run, band, camcol, field)')
    parser.add_option('-v', '--verbose', dest='verbose', action='count', default=0,
                      help='Make more verbose')
    parser.add_option('-d','--debug',dest='debug',action='store_true',default=False,help="Trigger debug images")
    parser.add_option('--plotAll',dest='plotAll',action='store_true',default=False,help="Makes a plot for each source")
    parser.add_option('--no-arcsinh', dest='noarcsinh', action='store_true', help='Do not arcsinh-stretch plots')
    parser.add_option('--lbfgsb', dest='lbfgsb', action='store_true', default=False,
                      help='Use L-BFGS-B optimization method')
    parser.add_option('--scale', dest='scale', type=int, help='Scale images down by this factor')
    parser.add_option('--unzip', dest='unzip', help='Save unzipped frame files in this directory')
    parser.add_option('--no-flipbook', dest='flipbook', default=True,
                      action='store_false', help='Do not write PDF flip-book of plots')
    opt,args = parser.parse_args()

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    run = opt.run
    field = opt.field
    camcol = opt.camcol
    bands = []

    if run is None and field is None and camcol is None and opt.band is None:
        # Demo mode...
        ## FIXME -- find a nice field here
        run = 1752
        camcol = 3
        field = 164
        opt.band = 'r'
        opt.curl = True
        tune.extend([('i',[1,1]), ('n',1)])
        if opt.roi is None:
            opt.roi = [100,600,100,600]
        print()
        print('Demo mode: grabbing Run/Camcol/Field/Band %i/%i/%i/%s' % (run, camcol, field, opt.band))
        print('Using SDSS DR9 data')
        print()
        
    if opt.band is None:
        parser.print_help()
        print()
        print('Must supply band (-b [u | g |r |i | z])')
        print()
        sys.exit(-1)
    for char in opt.band:
        bands.append(char)
        if not char in ['u','g','r','i','z']:
            parser.print_help()
            print()
            print('Must supply band (u/g/r/i/z)')
            print()
            sys.exit(-1)
    rerun = 0
    usercf = (run is not None and field is not None and camcol is not None)
    userd = (opt.radec is not None and opt.pixsize is not None)
    if not (usercf or userd) or len(bands)==0:
        parser.print_help()
        print('Must supply (--run, --camcol, --field) or (--radec, --size), and --band')
        sys.exit(-1)
    bandname = bands[0] #initial position and shape
    prefix = opt.prefix
    if userd:
        ra = opt.radec[0]
        dec = opt.radec[1]
        size = opt.pixsize
    if prefix is None:
        if usercf:
            prefix = '%06i-%i-%04i' % (run,camcol, field)
        else:
            prefix = '%s-%s' % (ra, dec)

    if userd:
        # convert h:m:s to deg
        try:
            ra = float(ra)
        except:
            ra = ra.replace(':',' ')
            ra = hmsstring2ra(ra)
        try:
            dec = float(dec)
        except:
            dec = dec.replace(':',' ')
            dec = dmsstring2dec(dec)

    sdss = DR9()
    getim = st.get_tractor_image_dr9
    getsrc = st.get_tractor_sources_dr9
    imkw = dict(zrange=[-3,100], sdss=sdss)
    if opt.unzip:
        sdss.saveUnzippedFiles(opt.unzip)
        
    tims = []
    for bandname in bands:
        if userd:
            if not usercf:
                tfn = opt.drfields
                if tfn is None:
                    tfn = 'dr%ifields.fits' % drnum
                rcfs = radec_to_sdss_rcf(ra, dec, radius=math.hypot(radius,13./2.), tablefn=tfn)
                if len(rcfs) == 0:
                    print('RA,Dec (%.3f,%.3f): found no overlapping SDSS fields in file %s' % (ra, dec, tfn))
                    sys.exit(-1)
                if len(rcfs) > 1:
                    print('Found %i Run,Camcol,Fields overlapping RA,Dec (%.3f,%.3f): taking the first one' % (len(rcfs), ra, dec))
                rcfs = rcfs[0]
                run,camcol,field = rcfs[:3]
            imkw.update(roiradecsize = (ra, dec, opt.pixsize))
        if opt.dg:
            imkw.update(psf='dg')

        tim,tinf = getim(run, camcol, field, bandname, curl=opt.curl,
                         roi=opt.roi, **imkw)
        tim.zr = tinf['zr']
        if userd:
            opt.roi = tinf['roi']
        
        tims.append(tim)

    if opt.scale:
        print('Scaling images by', opt.scale)
        tims = [st.scale_sdss_image(tim, opt.scale) for tim in tims]
        
    sources = getsrc(run, camcol, field, bandname, bands=bands, curl=opt.curl, roi=opt.roi)
    tractor = Tractor(tims, sources)

    sa = dict(debug=opt.debug, plotAll=opt.plotAll, roi=opt.roi)
    if opt.noarcsinh:
        sa.update(nlscale=0)
        
    for j,band in enumerate(bands):
        save('initial-%s-' % (band) + prefix, tractor, imgi=j, **sa)

    lnp0 = tractor.getLogProb()
    print('Initial lnprob:', lnp0)

    for im in tractor.images:
        im.freezeAllParams()
        im.thawParam('sky')

    for count, each in enumerate(tune):
        if each[0]=='n':
            tractor.catalog.thawAllParams()
            tractor.images.thawParamsRecursive('sky')
            for i in range(each[1]):
                if opt.lbfgsb:
                    tractor.optimize_lbfgsb()
                else:
                    tractor.optimize()
                for j,band in enumerate(bands):
                    save('tune-%d-%d-%s-' % (count+1, i+1,band) + prefix, tractor, imgi=j, **sa)
                lnp1 = tractor.getLogProb()
                print('After optimization: lnprob', lnp1)
                print('  delta-lnprob:', lnp1 - lnp0)
    
        elif each[0]=='i':
            tractor.images.freezeParamsRecursive('sky')
            for i in range(each[1][0]):
                for j,src in enumerate(tractor.getCatalog()):
                    tractor.catalog.freezeAllBut(src)

                    if i < 6:
                        print('Freezing position')
                        src.freezeParam('pos')

                    print('Optimizing:')
                    for nm in tractor.getParamNames():
                        print(nm)

                    for step in range(each[1][1]):
                        if opt.lbfgsb:
                            tractor.optimize_lbfgsb(plotfn='synt-opt-i%i-s%04i.png' % (i,j))
                        else:
                            tractor.optimize()

                    src.unfreezeParam('pos')

                for j,band in enumerate(bands):
                    save('tune-%d-%d-%s-' % (count+1,i+1,band) + prefix, tractor, imgi=j, **sa)

                lnp1 = tractor.getLogProb()
                print('After optimization: lnprob', lnp1)
                print('  delta-lnprob:', lnp1 - lnp0)

    if opt.flipbook:
        makeflipbook(opt, prefix,tune,len(tractor.getCatalog()),bands)
        print()
        print('Created flip-book flip-%s.pdf' % prefix)
    print('Done')


def makeflipbook(opt, prefix,tune,numSrcs,bands):
    # Create a tex flip-book of the plots
    tex = r'''
    \documentclass[compress]{beamer}
    \usepackage{helvet}
    \newcommand{\plot}[1]{\includegraphics[width=0.5\textwidth]{#1}}
    \begin{document}
    '''
    if len(tune) != 0:
        tex += r'''\part{Tuning steps}\frame{\partpage}''' + '\n'
    page = r'''
    \begin{frame}{%s}
    \plot{data-%s}
    \plot{model-%s} \\
    \plot{diff-%s}
        \plot{chi-%s} \\
    \end{frame}'''
    for j,band in enumerate(bands):
        tex += page % (('Initial model, Band: %s' % (band),) + ('initial-%s-' % (band) + prefix,)*4)
    if opt.plotAll:
        for i in range(numSrcs):
            for j,band in enumerate(bands):
                tex += page % (('Source: %i, Band: %s' % (i+1,band),)+ ('s%d-initial-%s-' % (i+1,band) + prefix,)*4)
    for count,step in enumerate(tune):
        if step[0] == 'n':
            for i in range (step[1]):
                for k,band in enumerate(bands):
                    tex += page % (('Tuning set %i, Tuning step %i, Band: %s' % (count+1,i+1,band),) +
                       ('tune-%d-%d-%s-' % (count+1,i+1,band) + prefix,)*4)
                if opt.plotAll:
                    for j in range(numSrcs):
                        for k,band in enumerate(bands):
                            tex += page % (('Source: %i, Band: %s' % (j+1,band),)+ ('s%d-tune-%d-%d-%s-' % (j+1,count+1,i+1,band) + prefix,)*4)
        elif step[0] == 'i': 
            for i in range(step[1][0]):
                for band in bands:
                    tex += page % (('Tuning set %i, Individual tuning step %i, Band: %s' % (count+1,i+1,band),) + ('tune-%d-%d-%s-' % (count+1,i+1,band) + prefix,)*4)
                if opt.plotAll:
                    for j in range(numSrcs):
                        for band in bands:
                            tex += page % (('Source: %i, Band: %s' % (j+1,band),) + ('s%d-tune-%d-%d-%s-' % (j+1,count+1,i+1,band) + prefix,)*4)
    if len(tune) != 0: 
        last = tune[len(tune)-1]
        lastSet = len(tune) - 1
        if last[0] == 'n':
            final = last[1]
        elif last[0] == 'i':
            final = last[1][0]
        lBand = bands[0]
        # Finish with a 'blink'
        tex += r'''\part{Before-n-after}\frame{\partpage}''' + '\n'
        tex += (r'''
        \begin{frame}{Data}
        \plot{data-%s}
        \plot{data-%s} \\
        \plot{diff-%s}
        \plot{diff-%s}
        \end{frame}
        \begin{frame}{Before (left); After (right)}
        \plot{model-%s}
        \plot{model-%s} \\
        \plot{diff-%s}
        \plot{diff-%s}
        \end{frame}
        ''' % (('initial-%s-' % (lBand) + prefix,)*2 +
               ('initial-%s-' %(lBand) +prefix, 'tune-%d-%d-%s-' % (lastSet+1,final,lBand)+ prefix)*3))
    tex += r'\end{document}' + '\n'
    fn = 'flip-' + prefix + '.tex'
    print('Writing', fn)
    open(fn, 'wb').write(tex)
    os.system("pdflatex '%s'" % fn)

if __name__ == '__main__':
    import cProfile
    import sys
    from datetime import tzinfo, timedelta, datetime
    cProfile.run('main()', 'prof-%s.dat' % (datetime.now().isoformat()))
    sys.exit(0)
    #main()
