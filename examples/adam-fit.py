from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import sys
import logging
import numpy as np
import pylab as plt

import fitsio

from astrometry.util.file import *

from tractor import *
from tractor import sdss as st
from tractor.galaxy import *

# Assumes one image.
def save(idstr, tractor, zr):
    mod = tractor.getModelImage(0)

    synthfn = 'synth-%s.fits' % idstr
    print('Writing synthetic image to', synthfn)
    fitsio.write(synthfn, mod, clobber=True)

    pfn = 'tractor-%s.pickle' % idstr
    print('Saving state to', pfn)
    pickle_to_file(tractor, pfn)

    timg = tractor.getImage(0)
    data = timg.getImage()
    ima = dict(interpolation='nearest', origin='lower',
               vmin=zr[0], vmax=zr[1])

    def savepng(pre, img, title=None, **kwargs):
        fn = '%s-%s.png' % (pre, idstr)
        print('Saving', fn)
        plt.clf()
        plt.imshow(img, **kwargs)
        if title is not None:
            plt.title(title)
        plt.colorbar()
        plt.gray()
        plt.savefig(fn)

    sky = np.median(mod)
    savepng('data', data - sky, title='Data '+timg.name, **ima)
    savepng('model', mod - sky, title='Model', **ima)
    savepng('diff', data - mod, title='Data - Model', **ima)


def makeflipbook(ntune, prefix):
    # Create a tex flip-book of the plots
    tex = r'''
    \documentclass[compress]{beamer}
    \usepackage{helvet}
    \newcommand{\plot}[1]{\includegraphics[width=0.5\textwidth]{#1}}
    \begin{document}
    '''
    if ntune:
        tex += r'''\part{Tuning steps}\frame{\partpage}''' + '\n'
    page = r'''
    \begin{frame}{%s}
    \plot{data-%s}
    \plot{model-%s} \\
    \plot{diff-%s}
    \end{frame}'''
    tex += page % (('Initial model',) + (prefix,)*3)
    for i in range(ntune):
        tex += page % (('Tuning step %i' % (i+1),) +
                       ('tune-%d-' % (i+1) + prefix,)*3)
    if ntune:
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
        ''' % ((prefix,)*2 +
               (prefix, 'tune-%d-' % (ntune) + prefix)*3))
    tex += r'\end{document}' + '\n'
    fn = 'flip-' + prefix + '.tex'
    print('Writing', fn)
    open(fn, 'wb').write(tex)
    os.system("pdflatex '%s'" % fn)

if __name__ == '__main__':
    run = 3972
    camcol = 3
    field = 12
    band = 'r'
    roi = [1850, 1950, 620, 720]
    x0,y0 = roi[0],roi[2]

    # rd1 is near the position of the SDSS-catalog object
    # rd2 is four pixels above.
    rd1 = RaDecPos(225.67954, 11.265948)
    rd2 = RaDecPos(225.67991, 11.265852)

    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    bandname = band

    sdssprefix = '%06i-%s%i-%04i' % (run, bandname, camcol, field)

    timg,info = st.get_tractor_image_dr9(run, camcol, field, bandname,
                                         roi=roi)
    sources = st.get_tractor_sources_dr9(run, camcol, field, bandname,
                                     roi=roi)

    wcs = timg.getWcs()
    for source in sources:
        x,y = wcs.positionToPixel(source.getPosition(), src=source)
        print('  (%.2f, %.2f):' % (x+x0,y+y0), source)

    tractor = Tractor([timg], sources)

    lnlorig = tractor.getLogLikelihood()
    zr = np.array([-5.,+20.]) * info['skysig']
    save(sdssprefix+'-orig', tractor, zr)

    # find closest source
    mini = -1
    mind2 = 1e6
    for i,src in enumerate(sources):
        rd = src.getPosition()
        rascale = np.cos(np.deg2rad(rd.ra))
        d2 = ((rd.ra - rd1.ra)*rascale)**2 + (rd.dec - rd1.dec)**2
        if d2 < mind2:
            mind2 = d2
            mini = i
    gal = sources[mini]
    galorig = gal.copy()
    print('Closest source:', gal)

    # optimize original SDSS source (at fixed complexity)
    nsteps = 200
    mndl = 0. # 0 for machine precision

    tractor.freezeParam('images')
    tractor.catalog.freezeAllBut(mini)
    
    tractor.optimize_loop(steps=nsteps, mindlnprob=mndl)
    lnlopt = tractor.getLogLikelihood()
    galopt = gal.copy()
    save(sdssprefix+'-opt', tractor, zr)

    ##
    # -original SDSS catalog
    # -original SDSS - SDSS source + 2 pinned PointSources
    # -                            + pinned PointSource + pinned DeV
    # -                            + pinned DeV + pinned PointSource
    # -                            + 2 pinned Composite
    ##

    # set non-trivial initializations
    shape1 = galorig.shape.copy()
    shape1.re = 1. / 30.
    shape2 = galorig.shape.copy()
    bright = galorig.brightness
    newsrc1 = [
        PointSource(rd1.copy(), bright.copy() * 0.5),
        ExpGalaxy(rd1.copy(), bright.copy() * 0.5, shape1.copy()),
        DevGalaxy(rd1.copy(), bright.copy() * 0.5, shape1.copy()),
        CompositeGalaxy(rd1.copy(), bright.copy() * 0.25, shape1.copy(),
                        bright.copy() * 0.25, shape1.copy()),
        ]
    newsrc2 = [ 
        PointSource(rd2.copy(), bright.copy() * 0.5),
        ExpGalaxy(rd2.copy(), bright.copy() * 0.5, shape2.copy()),
        DevGalaxy(rd2.copy(), bright.copy() * 0.5, shape2.copy()),
        CompositeGalaxy(rd2.copy(), bright.copy() * 0.25, shape2.copy(),
                        bright.copy() * 0.25, shape2.copy()),
        ]

    # loop over source models, optimizing each in turn
    tractor.removeSource(gal)
    lnls = np.zeros((len(newsrc1), len(newsrc2)))
    optsrcs = []
    for i,ons1 in enumerate(newsrc1):
        for j,ons2 in enumerate(newsrc2):
            # add sources
            ns1 = ons1.copy()
            ns2 = ons2.copy()
            tractor.addSource(ns1)
            tractor.addSource(ns2)
            print('Created sources:')
            print(' ', ns1)
            print(' ', ns2)
            # pin source positions
            ns1.freezeParam('pos')
            ns2.freezeParam('pos')
            # optimize fluxes and shapes
            #D = tractor.getAllDerivs(srcs=[ns1,ns2], sky=False)
            #print D
            #for k in range(10):
            #   tractor.optimizeCatalogAtFixedComplexityStep(srcs=[ns1,ns2],
            #                                                sky=False)
            tractor.catalog.freezeAllBut(ns1,ns2)

            print('Fitting:')
            tractor.printThawedParams()

            tractor.optimize_loop(steps=nsteps, mindlnprob=mndl,
                                  shared_params=False)

            lnls[i, j] = tractor.getLogLikelihood()
            # make some freaky plots
            prefix = '%s-%d-%d' % (sdssprefix, i, j)
            print('Optimized sources:')
            print(' ', ns1)
            print(' ', ns2)
            print(prefix, lnls[i, j])
            save(prefix, tractor, zr)
            # remove sources again
            tractor.removeSource(ns1)
            tractor.removeSource(ns2)
            #
            optsrcs.append((prefix, i,j,ns1.copy(),ns2.copy(),lnls[i,j]))

    # output
    print()
    print(sdssprefix)
    print('original ln likelihood:', lnlorig)
    print('  ', galorig)
    print()
    print(sdssprefix)
    print('optimized (at original SDSS complexity) ln likelihood:', lnlopt)
    print('  ', galopt)

    for prefix,i,j,ns1,ns2,lnl in optsrcs:
        print()
        print(prefix)
        print(i,j, 'new ln likelihood:', lnl)
        print('  ', ns1)
        print('  ', ns2)

    print()
    # ------------------------------------------------------------------
    # die
    # ------------------------------------------------------------------
    sys.exit(0)

    ntune = 10
    for i in range(ntune):
        tractor.optimizeCatalogLoop(nsteps=1)
        print('Optimization step', i, 'lnl', tractor.getLogLikelihood())
        save('tune-%d-' % (i+1) + prefix, tractor, zr)
    
    makeflipbook(ntune, prefix)
    print()
    print('Created flip-book flip-%s.pdf' % prefix)

    lnl1 = tractor.getLogLikelihood()

    prefix = 'stargal-%06i-%s%i-%04i' % (run, bandname, camcol, field)

    #radec = wcs.pixelToPosition(None, (1908-x0, 673-y0))
    #print 'RA,Dec', radec
    #tractor.addSource(PointSource(radec, st.SdssFlux(10000. / st.SdssPhotoCal.scale)))
    cat0 = tractor.getCatalog().deepcopy()

    sc = st.SdssPhotoCal.scale
    pg = PinnedDevGalaxy(rdgal, gal.flux, gal.shape)
    ps = PinnedPointSource(rdstar, st.SdssFlux(10000. / sc))
    print('gal flux', gal.flux)
    print('gal shape', gal.shape)
    print('Adding pinned galaxy', pg)
    print('Adding pinned star', ps)

    print('number of gal params', gal.numberOfParams())
    print(gal.getParams())
    print('number of pinned gal params', pg.numberOfParams())
    print(pg.getParams())
    print('number of pinned star params', ps.numberOfParams())
    print(ps.getParams())

    tractor.addSource(pg)
    tractor.addSource(ps)
    sources = tractor.getCatalog()
    save(prefix, tractor, zr)

    lnl2 = tractor.getLogLikelihood()

    #for i in range(ntune):
    #   tractor.optimizeCatalogLoop(nsteps=1)
    #   save('tune-%d-' % (i+1) + prefix, tractor, zr)

    for i in range(ntune):
        tractor.optimizeCatalogAtFixedComplexityStep(srcs=[pg,ps])
        save('tune-%d-' % (i+1) + prefix, tractor, zr)
    lnl3 = tractor.getLogLikelihood()

    print('Final sources (gal + ps) A:')
    for source in sources:
        x,y = wcs.positionToPixel(source, source.getPosition())
        print('  (%.2f, %.2f):' % (x+x0,y+y0), source)

    # Now optimize everything...
    for i in range(ntune, ntune*2):
        tractor.optimizeCatalogAtFixedComplexityStep()
        save('tune-%d-' % (i+1) + prefix, tractor, zr)
    lnl3b = tractor.getLogLikelihood()

    print('Final sources (gal + ps) B:')
    for source in sources:
        x,y = wcs.positionToPixel(source, source.getPosition())
        print('  (%.2f, %.2f):' % (x+x0,y+y0), source)

    makeflipbook(ntune*2, prefix)
    print()
    print('Created flip-book flip-%s.pdf' % prefix)
    
    prefix = 'starstar-%06i-%s%i-%04i' % (run, bandname, camcol, field)
    ps2 = PinnedPointSource(rdgal, gal.flux)

    tractor.setCatalog(cat0.deepcopy())
    tractor.addSource(ps)
    #tractor.removeSource(pg)

    tractor.addSource(ps2)
    sources = tractor.getCatalog()
    save(prefix, tractor, zr)
    lnl4 = tractor.getLogLikelihood()
    for i in range(ntune):
        tractor.optimizeCatalogAtFixedComplexityStep(srcs=[ps, ps2])
        save('tune-%d-' % (i+1) + prefix, tractor, zr)
    lnl5 = tractor.getLogLikelihood()

    print('Final sources (2 x ps) A:')
    for source in sources:
        x,y = wcs.positionToPixel(source, source.getPosition())
        print('  (%.2f, %.2f):' % (x+x0,y+y0), source)

    for i in range(ntune, ntune*2):
        tractor.optimizeCatalogAtFixedComplexityStep()
        save('tune-%d-' % (i+1) + prefix, tractor, zr)
    lnl5b = tractor.getLogLikelihood()

    print('Final sources (2 x ps) B:')
    for source in sources:
        x,y = wcs.positionToPixel(source, source.getPosition())
        print('  (%.2f, %.2f):' % (x+x0,y+y0), source)

    makeflipbook(ntune*2, prefix)
    print()
    print('Created flip-book flip-%s.pdf' % prefix)

    print('Initial:', lnl0)
    print('Tuned:', lnl1)
    print('Pinned sources (gal + ps):', lnl2)
    print('Tuned:', lnl3)
    print('Tuned all:', lnl3b)
    print('Pinned sources (2 x ps):', lnl4)
    print('Tuned:', lnl5)
    print('Tuned all:', lnl5b)
