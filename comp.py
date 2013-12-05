import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as plt
from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *

import sys
#import urllib2

import fitsio

#  Ta = fits_table('phot-1384p454-w1-b16.fits')
#  Tb = fits_table('phot-1384p454-w1-b17.fits')
#  blocksa = 16
#  blocksb = 17

def reassemble_chunks(mods, blocks, imargin):
    rows = []
    j = 0
    while len(mods):
        #print 'row start'
        modrow = mods[:blocks]
        for i in range(1, len(modrow)):
            modrow[i] = modrow[i][:,imargin:]
        for i in range(0, len(modrow)-1):
            modrow[i] = modrow[i][:,:-imargin]
        if j > 0:
            for i in range(len(modrow)):
                modrow[i] = modrow[i][imargin:,:]
        if j < blocks-1:
            for i in range(len(modrow)):
                modrow[i] = modrow[i][:-imargin,:]
        j += 1
        mods = mods[blocks:]
        #for m in modrow:
        #    print 'shape:', m.shape
        row = np.hstack(modrow)
        #print 'row:', row.shape
        rows.append(row)
    mod = np.vstack(rows)
    #print 'moda:', moda.shape
    return mod

def wack(coadd_id, ps):

    #T = fits_table('photoobjs-%s.fits' % coadd_id)
    T = fits_table('phot-%s-b8.fits' % coadd_id)
    print 'Read', len(T), 'objects'
    G = T[T.objc_type == 3]
    print len(G), 'galaxies'
    # G = fits_table('sweeps-%s-gals.fits' % coadd_id)
    # print len(G), 'galaxies'
    G.cut((G.theta_dev[:,2] > 0) * (G.theta_exp[:,2] > 0))
    print len(G), 'galaxies with positive thetas'
    G.cut(G.modelflux[:,2] > 0)
    print len(G), 'galaxies with positive flux'

    b = 2
    gal = (G.objc_type == 3)
    dev = gal * (G.fracdev[:,b] >= 0.5)
    exp = gal * (G.fracdev[:,b] <  0.5)
    stars = (G.objc_type == 6)
    print sum(dev), 'deV,', sum(exp), 'exp, and', sum(stars), 'stars'
    print 'Total', len(G), 'sources'
    thetasn = np.zeros(len(G))
    G.theta_deverr[dev,b] = np.maximum(1e-6, G.theta_deverr[dev,b])
    G.theta_experr[exp,b] = np.maximum(1e-5, G.theta_experr[exp,b])
    # theta_experr nonzero: 1.28507e-05
    # theta_deverr nonzero: 1.92913e-06
    thetasn[dev] = G.theta_dev[dev,b] / G.theta_deverr[dev,b]
    thetasn[exp] = G.theta_exp[exp,b] / G.theta_experr[exp,b]
    print 'Theta S/N:', thetasn.min(), thetasn.max()
    assert(np.all(thetasn > 3.))


    ###
    G.rflux = np.array(G.modelflux[:,2], copy=True)
    G.modelflux[:,2] = np.maximum(1e-2, G.modelflux[:,2])
    G.modelflux[:,2] = np.minimum(1e4,  G.modelflux[:,2])
    G.theta_dev[:,2] = np.maximum(1e-2, G.theta_dev[:,2])
    G.theta_exp[:,2] = np.maximum(1e-2, G.theta_exp[:,2])

    G.fluxstr = np.array(['%.0f' % f for f in G.rflux])
    G.devaberrstr = np.array(['ab err %.2f' % f for f in G.ab_deverr[:,2]])
    G.expaberrstr = np.array(['ab err %.2f' % f for f in G.ab_experr[:,2]])

    Idev = (G.fracdev[:,2] > 0.5)
    Iexp = (G.fracdev[:,2] <= 0.5)
    # D.cut(D.theta_dev[:,2] > 0)
    # E.cut(E.theta_exp[:,2] > 0)
    # print len(D), 'dev', len(E), 'exp with positive theta'

    G.theta = np.zeros(len(G))
    G.thetaerr = np.zeros(len(G))
    G.theta[Idev] = G.theta_dev[Idev,2]
    G.theta[Iexp] = G.theta_exp[Iexp,2]
    G.thetaerr[Idev] = G.theta_deverr[Idev,2]
    G.thetaerr[Iexp] = G.theta_experr[Iexp,2]

    G.thetastr = np.array(['th %.2g +- %.2g' % (t, dt) for t,dt
                           in zip(G.theta, G.thetaerr)])

    G.ab = np.zeros(len(G))
    G.aberr = np.zeros(len(G))
    G.ab[Idev] = G.ab_dev[Idev,2]
    G.ab[Iexp] = G.ab_exp[Iexp,2]
    G.aberr[Idev] = G.ab_deverr[Idev,2]
    G.aberr[Iexp] = G.ab_experr[Iexp,2]

    G.abstr = np.array(['th %.2g +- %.2g' % (t, dt) for t,dt
                           in zip(G.ab, G.aberr)])


    D = G[Idev]
    E = G[Iexp]
    print len(D), 'dev', len(E), 'exp'

    #wack1(ps, G, D, E)
    ps.skipto(21)
    wack2(ps, G, D, E)

    
def wack1(ps, G, D, E):
    # Plot some wacky objects
    I = np.flatnonzero(D.modelflux[:,2] >= 1e3)
    T = D[I]
    T = T[np.argsort(T.rflux)]
    T.title = T.fluxstr
    print 'Bright dev: theta_dev=', T.theta_dev[:,2]
    rows,cols = 4,6
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.1, right=0.9,
                        bottom=0.1, top=0.9)
    plotobjs(rows, cols, T)
    plt.suptitle('deV "galaxies" with modelflux > 1e3')
    ps.savefig()

    I = np.flatnonzero((D.theta_dev[:,2] > 25) * (D.modelflux[:,2] >= 1.))
    T = D[I]
    T.title = T.fluxstr
    T = T[np.argsort(-T.rflux)]
    plotobjs(rows, cols, T)
    plt.suptitle('deVs with theta > 25" and flux > 1')
    ps.savefig()

    # I = np.flatnonzero((D.theta_dev[:,2] > 25) * (D.modelflux[:,2] >= 100.))
    # T = D[I]
    # T.title = T.fluxstr
    # T.cut(np.argsort(-T.rflux))
    # plotobjs(rows, cols, T)
    # plt.suptitle('deVs with theta > 25" and flux > 100')
    # ps.savefig()

    I = np.flatnonzero((E.theta_exp[:,2] > 25) * (E.modelflux[:,2] >= 100.))
    T = E[I]
    T.title = T.fluxstr
    T = T[np.argsort(-T.rflux)]
    plotobjs(rows, cols, T)
    plt.suptitle('exps with theta > 25" and flux > 100')
    ps.savefig()

    I = np.flatnonzero((D.theta_dev[:,2] > 5) * (D.theta_dev[:,2] < 10)
                       * (D.modelflux[:,2] > 1.) * (D.modelflux[:,2] < 10.))
    T = D[I]
    T.title = T.fluxstr
    plotobjs(rows, cols, T)
    plt.suptitle('deVs with theta [5,10] and flux [1,10]')
    ps.savefig()


    I = np.flatnonzero((D.theta_dev[:,2] > 10) * (D.theta_dev[:,2] < 20)
                       * (D.modelflux[:,2] > 10.) * (D.modelflux[:,2] < 100.))
    T = D[I]
    T.title = T.fluxstr
    plotobjs(rows, cols, T)
    plt.suptitle('deVs with theta [10,20] and flux [10,100]')
    ps.savefig()
    
    
    I = np.flatnonzero((E.ab_exp[:,2] < 0.07) * (E.modelflux[:,2] > 10.))
    T = D[I]
    T = T[np.argsort(-T.rflux)]
    T.title = T.expaberrstr #T.fluxstr
    plotobjs(rows, cols, T)
    plt.suptitle('exps with ab_exp < 0.07 and flux > 10')
    ps.savefig()
    
    I = np.flatnonzero((D.ab_dev[:,2] < 0.07) * (D.modelflux[:,2] > 10.))
    T = D[I]
    T = T[np.argsort(-T.rflux)]
    T.title = T.devaberrstr #T.fluxstr
    plotobjs(rows, cols, T)
    plt.suptitle('deVs with ab_dev < 0.07 and flux > 10')
    ps.savefig()



    
    # plt.clf()
    # plt.hist(G.theta_dev[:,2], 50, histtype='step', color='r')
    # plt.hist(D.theta_dev[:,2], 50, histtype='step', color='r')
    # plt.hist(G.theta_exp[:,2], 50, histtype='step', color='b')
    # plt.hist(E.theta_exp[:,2], 50, histtype='step', color='b')
    # plt.xlabel('theta')
    # ps.savefig()
    
    plt.clf()
    #plt.hist(np.log10(G.theta_dev[:,2]), 50, histtype='step', color='r')
    n,b,p1 = plt.hist(np.log10(D.theta_dev[:,2]), 50, histtype='step', color='r')
    #plt.hist(np.log10(G.theta_exp[:,2]), 50, histtype='step', color='b')
    n,b,p2 = plt.hist(np.log10(E.theta_exp[:,2]), 50, histtype='step', color='b')
    plt.xlabel('log theta')
    plt.title('Effective radii of galaxies')
    plt.legend((p1[0], p2[0]), ('deV', 'exp'))
    ps.savefig()
    
    plt.clf()
    # plt.hist(G.ab_dev[:,2], 100, histtype='step', color='r')
    # plt.hist(G.ab_exp[:,2], 100, histtype='step', color='b')
    n,b,p1 = plt.hist(D.ab_dev[:,2], 100, histtype='step', color='r')
    n,b,p2 = plt.hist(E.ab_exp[:,2], 100, histtype='step', color='b')
    plt.legend((p1[0], p2[0]), ('deV', 'exp'))
    plt.xlabel('ab')
    plt.title('Axis ratios of galaxies')
    ps.savefig()
    
    plt.clf()
    plt.hist(np.log10(G.modelflux[:,2]), 100, histtype='step', color='r')
    plt.xlabel('log modelflux')
    ps.savefig()
    
    # plt.clf()
    # loghist(D.theta_dev[:,2], D.modelflux[:,2], 100)
    # plt.xlabel('theta_dev')
    # plt.ylabel('modelflux')
    # ps.savefig()
    
    plt.clf()
    loghist(D.theta_dev[:,2], np.log10(D.modelflux[:,2]), 100)
    plt.xlabel('theta_dev')
    plt.ylabel('log modelflux')
    plt.title('Brightness vs Effective radius')
    ps.savefig()
    
    # plt.clf()
    # loghist(E.theta_exp[:,2], E.modelflux[:,2], 100)
    # plt.xlabel('theta_exp')
    # plt.ylabel('modelflux')
    # ps.savefig()
    
    plt.clf()
    loghist(E.theta_exp[:,2], np.log10(E.modelflux[:,2]), 100)
    plt.xlabel('theta_exp')
    plt.ylabel('log modelflux')
    plt.title('Brightness vs Effective radius')
    ps.savefig()
    
    # plt.clf()
    # loghist(E.ab_exp[:,2], E.modelflux[:,2], 100)
    # plt.xlabel('ab_exp')
    # plt.ylabel('modelflux')
    # ps.savefig()
    
    plt.clf()
    loghist(E.ab_exp[:,2], np.log10(E.modelflux[:,2]), 100)
    plt.xlabel('ab_exp')
    plt.ylabel('log modelflux')
    plt.title('Brightness vs Axis ratio')
    ps.savefig()

    plt.clf()
    loghist(D.ab_dev[:,2], np.log10(D.modelflux[:,2]), 100)
    plt.xlabel('ab_dev')
    plt.ylabel('log modelflux')
    plt.title('Brightness vs Axis ratio')
    ps.savefig()
    
    
    plt.clf()
    loghist(D.theta_dev[:,2], D.ab_dev[:,2], 100)
    plt.xlabel('theta_dev')
    plt.ylabel('ab_dev')
    plt.title('Axis ratio vs Radius')
    ps.savefig()
    
    plt.clf()
    loghist(E.theta_exp[:,2], E.ab_exp[:,2], 100)
    plt.xlabel('theta_exp')
    plt.ylabel('ab_exp')
    plt.title('Axis ratio vs Radius')
    ps.savefig()
    
    plt.clf()
    loghist(E.theta_exp[:,2], np.minimum(60., E.theta_experr[:,2]), 100)
    plt.xlabel('theta_exp')
    plt.ylabel('theta_exp err')
    ps.savefig()
    
    plt.clf()
    loghist(D.theta_dev[:,2], np.minimum(30., D.theta_deverr[:,2]), 100)
    plt.xlabel('theta_dev')
    plt.ylabel('theta_dev err')
    ps.savefig()
    
    plt.clf()
    loghist(E.ab_exp[:,2], np.minimum(1., E.ab_experr[:,2]), 100)
    plt.xlabel('ab_exp')
    plt.ylabel('ab_exp err')
    ps.savefig()
    
    plt.clf()
    loghist(D.ab_dev[:,2], np.minimum(1., D.ab_deverr[:,2]), 100)
    plt.xlabel('ab_dev')
    plt.ylabel('ab_dev err')
    ps.savefig()
    
    
    I = np.flatnonzero((D.ab_dev[:,2] < 0.1) * (D.theta_dev[:,2] > 2.))
    T = D[I]
    T = T[np.argsort(-T.rflux)]
    T.title = T.fluxstr
    print 'Flux:', T.rflux[:20]
    print 'Flux str:', T.title[:20]

    plotobjs(rows, cols, T)
    plt.suptitle('deV galaxies with ab < 0.1, theta > 2"')
    ps.savefig()

def wack2(ps, G, D, E):
    E.abexperr = np.minimum(1., E.ab_experr[:,2])
    D.abdeverr = np.minimum(1., D.ab_deverr[:,2])

    D.thetasn = D.theta_dev[:,2] / D.theta_deverr[:,2]
    E.thetasn = E.theta_exp[:,2] / E.theta_experr[:,2]

    plt.clf()
    loghist(np.log10(E.modelflux[:,2]), np.log10(E.thetasn))
    plt.xlabel('log model flux')
    plt.ylabel('log theta_exp S/N')
    ax = plt.axis()
    plt.plot([-2,4],[-2,4], 'b-')
    plt.axhline(np.log10(5.), color='b')
    plt.axhline(np.log10(3.), color='b')
    plt.axhline(np.log10(1.), color='b')
    plt.axis(ax)
    ps.savefig()

    plt.clf()
    loghist(np.log10(D.modelflux[:,2]), np.log10(D.thetasn))
    plt.xlabel('log model flux')
    plt.ylabel('log theta_dev S/N')
    ax = plt.axis()
    plt.plot([-2,4],[-2,4], 'b-')
    plt.axhline(np.log10(5.), color='b')
    plt.axhline(np.log10(3.), color='b')
    plt.axhline(np.log10(1.), color='b')
    plt.axis(ax)
    ps.savefig()


    rows,cols = 4,6
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.1, right=0.9,
                        bottom=0.1, top=0.9)
    I = np.argsort(-(D.thetasn - D.modelflux[:,2]))
    T = D[I]
    T.title = T.thetastr
    plotobjs(rows, cols, T)
    plt.suptitle('deV galaxies with theta S/N > modelflux')
    ps.savefig()

    I = np.argsort(-(E.thetasn - E.modelflux[:,2]))
    T = E[I]
    T.title = T.thetastr
    plotobjs(rows, cols, T)
    plt.suptitle('exp galaxies with theta S/N > modelflux')
    ps.savefig()

    D.cut(D.thetasn < D.modelflux[:,2])
    E.cut(E.thetasn < (10. * E.modelflux[:,2]))

    print 'Cut to', len(D), 'and', len(E), 'dev/exp'

    plt.clf()
    loghist(np.log10(E.modelflux[:,2]), np.log10(E.theta))
    plt.xlabel('log model flux')
    plt.ylabel('log theta_exp')
    ps.savefig()

    plt.clf()
    loghist(np.log10(D.modelflux[:,2]), np.log10(D.theta))
    plt.xlabel('log model flux')
    plt.ylabel('log theta_dev')
    ps.savefig()



    I = np.argsort(-D.theta_dev[:,2])
    T = D[I]
    print 'theta_devs:', T.theta[:20]
    T.title = T.thetastr
    plotobjs(rows, cols, T)
    plt.suptitle('dev galaxies with large theta')
    ps.savefig()

    I = np.argsort(-E.theta_exp[:,2])
    T = E[I]
    print 'theta_exps:', T.theta[:20]
    T.title = T.thetastr
    plotobjs(rows, cols, T)
    plt.suptitle('exp galaxies with large theta')
    ps.savefig()


    plt.clf()
    loghist(np.log10(E.modelflux[:,2]), E.abexperr)
    plt.xlabel('log model flux')
    plt.ylabel('ab_exp err')
    ps.savefig()

    plt.clf()
    loghist(np.log10(D.modelflux[:,2]), D.abdeverr)
    plt.xlabel('log model flux')
    plt.ylabel('ab_dev err')
    ps.savefig()



    I = np.argsort(D.aberr)
    T = D[I]
    T.title = T.abstr
    plotobjs(rows, cols, T)
    plt.suptitle('dev galaxies with small a/b err')
    ps.savefig()

    I = np.argsort(E.aberr)
    T = E[I]
    T.title = T.abstr
    plotobjs(rows, cols, T)
    plt.suptitle('exp galaxies with small a/b err')
    ps.savefig()





    # plt.clf()
    # loghist(np.log10(E.modelflux[:,2]), np.minimum(5., E.theta_experr[:,2]))
    # plt.xlabel('log model flux')
    # plt.ylabel('theta_exp err')
    # ps.savefig()
    # 
    # plt.clf()
    # loghist(np.log10(D.modelflux[:,2]), np.minimum(10., D.theta_deverr[:,2]))
    # plt.xlabel('log model flux')
    # plt.ylabel('theta_dev err')
    # ps.savefig()


    ha = dict(range=(-2,2), bins=50, histtype='step')

    plt.clf()
    n,b,p1 = plt.hist(np.log10(D.theta_dev[:,2]), color='r', **ha)
    I = np.flatnonzero(D.thetasn > 3.)
    n,b,p2 = plt.hist(np.log10(D.theta_dev[I,2]), color='r', lw=2, alpha=0.5, **ha)
    I = np.flatnonzero(D.thetasn > 5.)
    n,b,p3 = plt.hist(np.log10(D.theta_dev[I,2]), color='r', lw=3, alpha=0.3, **ha)

    n,b,p4 = plt.hist(np.log10(E.theta_exp[:,2]), color='b', **ha)
    I = np.flatnonzero(E.thetasn > 3.)
    n,b,p5 = plt.hist(np.log10(E.theta_exp[I,2]), color='b', lw=2, alpha=0.5, **ha)
    I = np.flatnonzero(E.thetasn > 5.)
    n,b,p6 = plt.hist(np.log10(E.theta_exp[I,2]), color='b', lw=3, alpha=0.3, **ha)
    plt.xlabel('log theta')
    plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0],p6[0]),
               ('deV (all)', 'deV (S/N > 3)', 'deV (S/N > 5)',
                'exp (all)', 'exp (S/N > 3)', 'exp (S/N > 5)'))
    plt.title('S/N cut effects on galaxy size distributions')
    ps.savefig()

    ha.update(normed=True)
    
    plt.clf()
    n,b,p1 = plt.hist(np.log10(D.theta_dev[:,2]), color='r', **ha)
    I = np.flatnonzero(D.thetasn > 3.)
    n,b,p2 = plt.hist(np.log10(D.theta_dev[I,2]), color='r', lw=2, alpha=0.5, **ha)
    I = np.flatnonzero(D.thetasn > 5.)
    n,b,p3 = plt.hist(np.log10(D.theta_dev[I,2]), color='r', lw=3, alpha=0.3, **ha)

    n,b,p4 = plt.hist(np.log10(E.theta_exp[:,2]), color='b', **ha)
    I = np.flatnonzero(E.thetasn > 3.)
    n,b,p5 = plt.hist(np.log10(E.theta_exp[I,2]), color='b', lw=2, alpha=0.5, **ha)
    I = np.flatnonzero(E.thetasn > 5.)
    n,b,p6 = plt.hist(np.log10(E.theta_exp[I,2]), color='b', lw=3, alpha=0.3, **ha)
    plt.xlabel('log theta')
    plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0],p6[0]),
               ('deV (all)', 'deV (S/N > 3)', 'deV (S/N > 5)',
                'exp (all)', 'exp (S/N > 3)', 'exp (S/N > 5)'))
    plt.title('S/N cut effects on galaxy size distributions')
    ps.savefig()

    D.absn = 1. / np.maximum(1e-3, D.abdeverr)
    E.absn = 1. / np.maximum(1e-3, E.abexperr)

    plt.clf()
    loghist(np.log10(D.thetasn), np.log10(D.absn), 100)
    plt.xlabel('log S/N in theta')
    plt.ylabel('log 1/error in ab (~ log S/N in ab)')
    ps.savefig()  

    plt.clf()
    loghist(np.log10(E.thetasn), np.log10(E.absn), 100)
    plt.xlabel('log S/N in theta')
    plt.ylabel('log 1/error in ab (~ log S/N in ab)')
    ps.savefig()  


    
def plotobjs(rows, cols, T):
    plt.clf()
    for i in range(min(len(T), rows * cols)):
        ra,dec = T.ra[i], T.dec[i]
        #url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=%g&dec=%g&scale=1&width=128&height=128' % (ra,dec)
        url = 'http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra=%g&dec=%g&scale=0.4&width=128&height=128' % (ra,dec)
        #fn = 'cutout-%g-%g.jpg' % (ra,dec)
        fn = 'cutoutB-%g-%g.png' % (ra,dec)
        if not os.path.exists(fn):
            #cmd = 'wget "%s" -O "%s"' % (url, fn)
            cmd = 'wget "%s" -O - | jpegtopnm | pnmtopng > "%s"' % (url, fn)
            print cmd
            os.system(cmd)
        I = plt.imread(fn)
        plt.subplot(rows, cols, i+1)
        plt.imshow(I, interpolation='nearest', origin='lower')
        plt.xticks([]); plt.yticks([])
        if 'title' in T.get_columns() and len(T.title[i]):
            plt.title(T.title[i], fontsize=8)
        


def sky_foolin(img):
    m = np.median(img)
    print 'median', m
    img -= m
    
    sim = np.sort(img.ravel())
    print 'sim', sim.shape, sim.dtype
    I = np.linspace(0, len(sim)-1, 500)
    
    sigest = sim[int(0.5 * len(sim))] - sim[int(0.16 * len(sim))]
    print 'sig est', sigest
    nsig = 0.1 * sigest
    
    I = np.linspace(0.25*len(sim), 0.55*len(sim), 11).astype(int)
    dn = []
    sumn = []
    for ii in I:
        X = sim[ii]
        # nlo = sum((sim > X-nsig) * (sim < X))
        # nhi = sum((sim > X) * (sim < X+nsig))
        # print 'nlo', nlo, 'nhi', nhi, 'diff', nlo-nhi
        # dn.append(nlo - nhi)
        #sumn.append(nlo+nhi)
        sumn.append(sum((sim > X-nsig) * (sim < X+nsig)))
    
    #plt.clf()
    #plt.plot(I, dn, 'r-')
    #ps.savefig()
    plt.clf()
    plt.plot(I, sumn, 'r-')
    ps.savefig()
    
    sumn = np.array(sumn)
    
    iscale = 0.5 * len(sim)
    xi = (I-mean(I))/iscale
    
    A = np.zeros((len(I), 3))
    A[:,0] = 1.
    A[:,1] = xi
    A[:,2] = xi**2
    
    b = sumn
    
    res = np.linalg.lstsq(A, b)
    X = res[0]
    print 'X', X
    
    plt.clf()
    plt.plot(xi, sumn, 'r-')
    xx = np.linspace(xi[0], xi[-1], 200)
    plt.plot(xx, X[0] + X[1] * xx + X[2] * xx**2, 'b-')
    ps.savefig()
    
    Imax = - X[1] / (2. * X[2])
    Imax = (Imax * iscale) + mean(I)
    i = int(np.round(Imax))
    print 'Imax', Imax
    mu = sim[i]
    print 'mu', mu
    
    plt.clf()
    plt.hist(img.ravel() - mu, 100, range=(-0.5,1), histtype='step')
    plt.xlim(-0.5,1)
    ps.savefig()
    
    sys.exit(0)
    
    lo,hi = 0.1,0.8
    for step in range(3):
        q = np.round(np.linspace(lo * len(sim), hi * len(sim), 21)).astype(int)
        Q = sim[q]
        # for i in range(len(Q)-5):
        #     QQ = Q[i : i+5]
        #     c = QQ[2]
        #     print 'diff', c - QQ
        #     d = c - QQ
        #     print (d[0] + d[4]) / 2., (d[1] + d[3]) / 2.
        
        dd = []
        for i in range(len(Q)-2):
            #QQ = Q[i : i+3]
            #print 'q', q[i], q[i+1], q[i+2]
            #print 'Q', Q[i], Q[i+1], Q[i+2]
            slope = (Q[i+2] - Q[i]) / float(q[i+2] - q[i])
            #print 'slope', slope
            Qmid = Q[i] + slope * (q[i+1] - q[i])
            #print 'Qmid', Qmid
            d = Q[i+1] - Qmid
            print 'd', d
            dd.append(d)
        dd = np.array(dd)
            
        plt.clf()
        plt.plot(q, Q, 'r.')
        ps.savefig()
        
        plt.clf()
        plt.plot(dd, 'r.')
        plt.axhline(0.)
        ps.savefig()
        
        I = np.flatnonzero((dd[:-1] > 0) * (dd[1:] < 0))
        print 'I', I
        I = I[0]
        lo,hi = q[I] / float(len(sim)), q[I+1] / float(len(sim))
        print 'lo,hi', lo,hi
        Qmid = sim[(q[I]+q[I+1])/2]
        print 'Q', Q[I], Qmid, Q[I+1]
        
        plt.clf()
        plt.hist(img.ravel() - Qmid, 100, range=(-0.5,1), histtype='step')
        plt.xlim(-0.5,1)
        ps.savefig()
    
    
    # quant = I/float(len(sim))
    # plt.clf()
    # plt.plot(quant, sim[np.round(I).astype(int)], 'r-')
    # plt.yscale('symlog')
    # ps.savefig()
    # 
    sys.exit(0)
    
    # pp = []
    # for p in range(35,55+1, 2):
    #     p0 = np.percentile(img, p)
    #     print 'percentile', p, p0
    #     pp.append(p0)
    
    plt.clf()
    plt.hist(img.ravel(), 100, range=(-0.5,1), histtype='step')
    for p in pp:
        plt.axvline(p, color='k')
    plt.xlim(-0.5,1)
    ps.savefig()
    
    plt.xlim(pp[0], pp[-1])
    ps.savefig()
    
    sys.exit(0)
    
    plt.clf()
    plt.hist(img.ravel(), 100, range=(-0.5,2), log=True)
    plt.xlim(-0.5,2)
    ps.savefig()




                

ps = PlotSequence('comp')

coadd_id = '1384p454'






sys.exit(0)

#wack(coadd_id, ps)
#sys.exit(0)

ps.skipto(50)

#band = 1
band = 2


blocksa = 8
blocksb = 7
#pat = 'phot-1384p454-b%i.fits'
pat = 'phot-1384p454-w2-b%i.fits'
fna = pat % blocksa
fnb = pat % blocksb
Ta = fits_table(fna)
Tb = fits_table(fnb)

tiledir = 'wise-coadds'
fn = os.path.join(tiledir, 'coadd-%s-w%i-img-w.fits' % (coadd_id, band))
print 'Reading', fn
wcs = Tan(fn)
H,W = wcs.get_height(), wcs.get_width()
print 'Shape', H,W
#H,W = 1024,1024

img = fitsio.read(fn)

# sky_foolin(img)

modsa,catsa,ta,srada = unpickle_from_file(fna.replace('.fits','.pickle'))
modsb,catsb,tb,sradb = unpickle_from_file(fnb.replace('.fits','.pickle'))

imargin = 12

# cell positions
Xa = np.round(np.linspace(0, W, blocksa+1)).astype(int)
Ya = np.round(np.linspace(0, H, blocksa+1)).astype(int)
Xb = np.round(np.linspace(0, W, blocksb+1)).astype(int)
Yb = np.round(np.linspace(0, H, blocksb+1)).astype(int)

#moda = reassemble_chunks(modsa, blocksa, imargin)
#modb = reassemble_chunks(modsb, blocksb, imargin)

plt.clf()
fluxa = Ta.get('w%i_nanomaggies' % band)
fluxb = Tb.get('w%i_nanomaggies' % band)
plt.plot(fluxa, fluxb, 'b.')
plt.xlabel('run A flux (nanomaggies)')
plt.ylabel('run B flux (nanomaggies)')
plt.xscale('symlog')
plt.yscale('symlog')
ps.savefig()

lo,hi = 0.5, 2.0
plt.clf()
plt.plot(fluxa, np.clip(fluxb / fluxa, lo, hi), 'b.')
plt.xlabel('a (nm)')
plt.ylabel('b/a (nm)')
plt.xscale('symlog')
plt.ylim(lo, hi)
ps.savefig()


ima = dict(interpolation='nearest', origin='lower',
           vmin=-0.5, vmax=0.5, cmap='gray')
           #vmin=-0.01, vmax=0.5, cmap='gray')

imd = dict(interpolation='nearest', origin='lower',
           vmin=-1e-2, vmax=1e-2, cmap='gray')

diff = fluxb - fluxa
I = np.argsort(-np.abs(diff))
print 'Largest diffs:', diff[I[:20]]

plt.clf()
plt.imshow(img, **ima)
ax = plt.axis()
plt.plot(Ta.x[I[:20]], Ta.y[I[:20]], 'ro', mec='r', mfc='none')
for i,ii in enumerate(I[:20]):
    plt.text(Ta.x[ii], Ta.y[ii], '%i' % i, color='r')
plt.axis(ax)
ps.savefig()



for i in I[:10]:
    ta = Ta[i]
    tb = Tb[i]
    print
    print 'RA,Dec', ta.ra, ta.dec
    print 'a', Ta.w1_nanomaggies[i], 'b', Tb.w1_nanomaggies[i]

    #print 'cells', ta.cell, tb.cell
    x0 = min(ta.cell_x0, tb.cell_x0)
    x1 = max(ta.cell_x1, tb.cell_x1)
    y0 = min(ta.cell_y0, tb.cell_y0)
    y1 = max(ta.cell_y1, tb.cell_y1)
    #ax = [x0, x1, y0, y1]
    M = 20
    ax = [x0-M, x1+M, y0-M, y1+M]

    moda = modsa[ta.cell]
    modb = modsb[tb.cell]
    #cata, ntargeta, nboxa, tima = catsa[ta.cell]
    #catb, ntargetb, nboxb, timb = catsb[tb.cell]
    ina, marga, wxa, wya = catsa[ta.cell][:4]
    inb, margb, wxb, wyb = catsb[tb.cell][:4]

    #print 'moda', moda.shape
    #print 'modb', modb.shape

    plt.clf()
    plt.imshow(img, **ima)
    plt.plot(ta.x, ta.y, 'rx')
    plt.axis(ax)
    ps.savefig()

    plt.clf()
    plt.imshow(moda, extent=[ta.cell_x0, ta.cell_x1, ta.cell_y0, ta.cell_y1], **ima)
    plt.plot(ta.x, ta.y, 'rx')
    for x in Xa:
        plt.axvline(x, color='c')
    for y in Ya:
        plt.axhline(y, color='c')

    print 'A:', len(ina), len(marga)
    print 'WISE:', len(wxa)
    #print 'x,y', Ta.x[ina], Ta.y[ina]

    #plt.plot(Ta.x[ina], Ta.y[ina], 'o', mec='g', mfc='none')
    #plt.plot(Ta.x[marga], Ta.y[marga], 'o', mec='b', mfc='none')
    plt.plot(wxa, wya, 'o', mec='m', mfc='none')
    #ptsrc = (Ta.objc_type[I] == 6)
    for I,cc in [(ina,'g'), (marga, 'b')]:
        x,y,r = Ta.x[I], Ta.y[I], srada[I]
        plt.plot(np.vstack([x-r, x+r]), np.vstack([y,y]), '-', color=cc)
        plt.plot(np.vstack([x,x]), np.vstack([y-r, y+r]), '-', color=cc)
        plt.plot(x, y, 'o', mec=cc, mfc='none')

    plt.axis(ax)
    ps.savefig()

    plt.clf()
    plt.imshow(modb, extent=[tb.cell_x0, tb.cell_x1, tb.cell_y0, tb.cell_y1], **ima)
    plt.plot(ta.x, ta.y, 'rx')
    for x in Xb:
        plt.axvline(x, color='c')
    for y in Yb:
        plt.axhline(y, color='c')

    #plt.plot(Tb.x[inb], Tb.y[inb], 'o', mec='g', mfc='none')
    #plt.plot(Tb.x[margb], Tb.y[margb], 'o', mec='b', mfc='none')
    plt.plot(wxb, wyb, 'o', mec='m', mfc='none')
    for I,cc in [(inb,'g'), (margb, 'b')]:
        x,y,r = Tb.x[I], Tb.y[I], sradb[I]
        plt.plot(np.vstack([x-r, x+r]), np.vstack([y,y]), '-', color=cc)
        plt.plot(np.vstack([x,x]), np.vstack([y-r, y+r]), '-', color=cc)
        plt.plot(x, y, 'o', mec=cc, mfc='none')

    plt.axis(ax)
    ps.savefig()

    dm = np.zeros_like(img)
    ma = np.zeros_like(img)
    mb = np.zeros_like(img)
    dm[ta.cell_y0:ta.cell_y1, ta.cell_x0:ta.cell_x1] += moda
    ma[ta.cell_y0:ta.cell_y1, ta.cell_x0:ta.cell_x1] = 1.
    dm[tb.cell_y0:tb.cell_y1, tb.cell_x0:tb.cell_x1] -= modb
    mb[tb.cell_y0:tb.cell_y1, tb.cell_x0:tb.cell_x1] = 1.
    dm *= (ma * mb)
    plt.clf()
    plt.imshow(dm[y0:y1,x0:x1], extent=[x0,x1,y0,y1], **imd)
    plt.plot(ta.x, ta.y, 'rx')
    for x in np.append(Xa, Xb):
        plt.axvline(x, color='c')
    for y in np.append(Ya, Yb):
        plt.axhline(y, color='c')
    plt.axis(ax)
    ps.savefig()

sys.exit(0)



hha,wwa = modsa[0].shape
hhb,wwb = modsb[0].shape
hh = min(hha,hhb)
ww = min(wwa,wwb)

print 'A:', wwa,hha
print 'B:', wwb,hhb

HH = max(hha,hhb)
WW = max(wwa,wwb)


ra,da,srcsa,tim = catsa[0]
rb,db,srcsb,tim = catsb[0]

print 'Catalog A:'
for src in srcsa:
    print '  ', src

print
print 'Catalog B:'
for src in srcsb:
    print '  ', src


print
I,J,d = match_radec(np.array(ra),np.array(da), np.array(rb),np.array(db), 1./3600.)
print len(I), 'matches'

for i,j in zip(I,J):
    print
    print srcsa[i]
    print srcsb[j]

print

sxa = Ta.x
sya = Ta.y
sxb = Tb.x
syb = Tb.y

for im,imargs,tt,cat in [(modsa[0], ima, 'Model A (%i), first block' % blocksa, catsa[0]),
                         (modsb[0], ima, 'Model B (%i), first block' % blocksb, catsb[0]),
                         (img, ima, 'Data', None),
                         (modsa[0][:hh,:ww] - modsb[0][:hh,:ww], imd, 'Model A-B', catsa[0])]:

    plt.clf()
    #plt.imshow(im[:hh,:ww], **ima)
    plt.imshow(im, **imargs)
    h,w = im.shape
    plt.axhline(h - imargin, color='b')
    plt.axvline(w - imargin, color='b')
    plt.title(tt)

    ax = [-10, WW+10, -10, HH+10]
    plt.axis(ax)

    ps.savefig()

    I = np.flatnonzero((sxa > -20) * (sxa < WW+20) * (sya > -20) * (sya < HH+20))
    x = sxa[I]
    y = sya[I]
    r = srada[I]

    plt.plot(x, y, 'r.')
    plt.plot(np.vstack([x-r, x+r]), np.vstack([y,y]), 'r-')
    plt.plot(np.vstack([x,x]), np.vstack([y-r, y+r]), 'r-')

    if cat is not None:
        r,d,srcs,tim = cat
        r = np.array(r)
        d = np.array(d)
        ok,x,y = wcs.radec2pixelxy(r,d)
        x -= 1
        y -= 1
        plt.plot(x, y, 'o', mec='g', mfc='none')

    plt.axis(ax)
    ps.savefig()



# plt.clf()
# plt.imshow(modsb[0][:hh,:ww], **ima)
# plt.title('Model B (%i), first block' % blocksb)
# ps.savefig()
# 
# plt.clf()
# plt.imshow(img[:hh,:ww], **ima)
# plt.title('Data A, one block')
# ps.savefig()

# plt.imshow(modsa[0][:hh,:ww] - modsb[0][:hh,:ww],
#            interpolation='nearest', origin='lower',
#            vmin=-1e-2, vmax=1e-2, cmap='gray')
# plt.title('Model A - Model B')
# ps.savefig()


# for img,a in [(moda,ima),(modb,ima),(moda-modb, dict(interpolation='nearest', origin='lower',
#                                                      vmin=-1e-3, vmax=1.e-3, cmap='gray'))]:
#     plt.clf()
#     plt.imshow(img, **a)
#     ax = plt.axis()
#     for x in Xa:
#         plt.axvline(x, color='b', alpha=0.5)
#     for y in Ya:
#         plt.axhline(y, color='b', alpha=0.5)
#     for x in Xb:
#         plt.axvline(x, color='r', alpha=0.5)
#     for y in Yb:
#         plt.axhline(y, color='r', alpha=0.5)
#     plt.axis(ax)
#     ps.savefig()
#     plt.axis([0,400,0,400])
#     ps.savefig()


# plt.clf()
# plt.plot(Ta.ra, Ta.dec, 'r.')
# plt.xlabel('RA')
# plt.ylabel('Dec')
# ps.savefig()

# plt.clf()
# ha = dict(bins=100, histtype='step')
# plt.hist(Ta.w1_nanomaggies, color='b', **ha)
# # plt.hist(Ta.w2_nanomaggies, color='g', **ha)
# # plt.hist(Ta.w3_nanomaggies, color='r', **ha)
# # plt.hist(Ta.w4_nanomaggies, color='m', **ha)
# ps.savefig()

plt.clf()
plt.plot(Ta.w1_nanomaggies, Tb.w1_nanomaggies, 'b.')
# plt.plot(Ta.w2_nanomaggies, Tb.w2_nanomaggies, 'g.')
# plt.plot(Ta.w3_nanomaggies, Tb.w3_nanomaggies, 'r.')
# plt.plot(Ta.w4_nanomaggies, Tb.w4_nanomaggies, 'm.')
plt.xlabel('run A flux (nanomaggies)')
plt.ylabel('run B flux (nanomaggies)')
plt.xscale('symlog')
plt.yscale('symlog')
ps.savefig()

lo,hi = 0.5, 2.0
plt.clf()
plt.plot(Ta.w1_nanomaggies, np.clip(Tb.w1_nanomaggies / Ta.w1_nanomaggies, lo, hi), 'b.')

# I = np.random.randint(len(Tb), size=(100,))
# d = 1./np.sqrt(Tb.w1_nanomaggies_ivar[I])
# y = Tb.w1_nanomaggies[I]
# x = Ta.w1_nanomaggies[I]
# plt.plot(np.vstack([x, x]), np.vstack([y-d, y+d]) / x, 'b-', alpha=0.5)
# 
# I = np.flatnonzero(np.logical_or(y / x > 2, y/x < 0.5))
# d = 1./np.sqrt(Tb.w1_nanomaggies_ivar[I])
# y = Tb.w1_nanomaggies[I]
# x = Ta.w1_nanomaggies[I]
# plt.plot(np.vstack([x, x]), np.vstack([y-d, y+d]) / x, 'b-', alpha=0.5)

plt.xlabel('a (nm)')
plt.ylabel('b/a (nm)')
plt.xscale('symlog')
#plt.yscale('symlog')
plt.ylim(lo, hi)
ps.savefig()

ratio = Tb.w1_nanomaggies / Ta.w1_nanomaggies
O = np.flatnonzero(np.logical_or(ratio > 2., ratio < 0.5))
print len(O), 'outliers'

# where are the outliers?
plt.clf()
plt.plot(Ta.ra, Ta.dec, 'r.', alpha=0.1)
plt.plot(Ta.ra[O], Ta.dec[O], 'b.', alpha=0.5)
plt.xlabel('RA')
plt.ylabel('Dec')
ps.savefig()

ok,X,Y = wcs.radec2pixelxy(Ta.ra, Ta.dec)
X -= 1.
Y -= 1.
Ta.x = X
Ta.y = Y

plt.clf()
#plt.plot(Ta.x[Ta.w1_ntimes == 1], Ta.y[Ta.w1_ntimes == 1], 'r.', alpha=0.1)
plt.plot(Ta.x, Ta.y, 'r.', alpha=0.1)
plt.plot(Ta.x[O], Ta.y[O], 'b.', alpha=0.5)
#plt.plot(Ta.x[Ta.w1_ntimes == 0], Ta.y[Ta.w1_ntimes == 0], 'g.', alpha=0.5)
ax = plt.axis()
for x in Xa:
    plt.axvline(x, color='k', alpha=0.5)
for y in Ya:
    plt.axhline(y, color='k', alpha=0.5)
for x in Xb:
    plt.axvline(x, color='g', alpha=0.5)
for y in Yb:
    plt.axhline(y, color='g', alpha=0.5)
plt.axis(ax)
plt.xlabel('x')
plt.ylabel('y')
ps.savefig()


