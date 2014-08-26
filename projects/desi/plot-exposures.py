import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.starutil_numpy import *
from astrometry.blind.plotstuff import *

if __name__ == '__main__':

    decals_dir = os.environ.get('DECALS_DIR')
    B = fits_table(os.path.join(decals_dir, 'decals-bricks.fits'))
    #B = fits_table('bricks.fits')
    #T = fits_table('ccds.fits')#-20140810.fits')
    T = fits_table(os.path.join(decals_dir, 'decals-ccds.fits'))
    
    # plt.clf()
    # plt.plot(T.ra, T.dec, 'b.')
    # plt.savefig('rd.png')

    if False:
        xyz = radectoxyz(T.ra, T.dec)
        print 'xyz', xyz.shape
        xyzc = np.mean(xyz, axis=0)
        xyzc /= np.sqrt(np.sum(xyzc**2))
        print 'xyzc', xyzc
        rc,dc = xyztoradec(xyzc)
        print 'rc,dc', rc,dc
        dist = np.max(np.sqrt(np.sum((xyz - xyzc)**2, axis=1)))
        print 'dist', dist
        width = dist2deg(dist)
    
    #allsky = True
    #gridsize = 10.
    #W,H = 1000,500

    allsky = False
    gridsize = 1.
    #rc,dc = 245,8
    #width = 6.
    rc,dc = 244,8
    #width = 1.
    width = 2.5
    W,H = 1000,1000
    sz = width * 0.6
    T.cut((np.abs(T.ra - rc) < sz) * (np.abs(T.dec - dc) < sz))
    print 'Cut to', len(T), 'in range'
    print 'Bands', np.unique(T.filter)

    B.index = np.arange(len(B))
    B.cut((np.abs(B.ra - rc) < sz) * (np.abs(B.dec - dc) < sz))
    print len(B), 'bricks in range'
    
    #r0,r1 = T.ra.min(),  T.ra.max()
    #d0,d1 = T.dec.min(), T.dec.max()
    #rc = (r0+r1)/2.
    #rc = 360.
    #width = max(r1-r0, d1-d0)
    #width = d1-d0
    #print 'RA range', r0,r1
    #print 'Dec range', d0,d1
    #print 'Plot width:', width
    #W,H = 800,800
    plot = Plotstuff(outformat='png', size=(W,H),
                     rdw=(rc, dc, width))

    if allsky:
        plot.wcs = anwcs_create_allsky_hammer_aitoff(rc, dc, W, H)

    cmap = { 'g':'green', 'r':'red', 'z':'magenta' }

    for band in np.unique(T.filter):
        TT = T[T.filter == band]
        print len(TT), 'in band', band
        
        plot.color = 'verydarkblue'
        plot.plot('fill')
    
        #plot.color = 'green'
        plot.color = cmap.get(band, 'white')
        out = plot.outline
        out.fill = 1
        plot.alpha = 0.3
    
        for t in TT:
            wcs = Tan(*[float(x) for x in
                        [t.crval1, t.crval2, t.crpix1, t.crpix2,
                         t.cd1_1, t.cd1_2, t.cd2_1, t.cd2_2, t.width, t.height]])
            out.wcs = anwcs_new_tan(wcs)

            #out.fill = 1
            #plot.alpha = 0.3
            plot.plot('outline')

            # out.fill = 0
            # plot.lw = 1
            # plot.alpha = 0.6
            # plot.plot('outline')

        fn = 'ccd-%s1.png' % band
        plot.write(fn)
        print 'Wrote', fn
        
        plot.color = 'gray'
        plot.plot_grid(gridsize, gridsize, gridsize*2, gridsize*2)

        plot.color = 'white'
        plot.alpha = 0.6
        plot.marker = 'circle'
        #plot.set_markersize(15)
        plot.set_markersize(3)
        plot.apply_settings()
        #for r,d in zip(B.ra, B.dec):
        #    plot.marker_radec(r,d)
        for r,d,ii in zip(B.ra, B.dec, B.index):
            plot.text_radec(r, d, '%i' % ii)
        plot.stroke()

        pixscale = 0.27 / 3600.
        out.fill = 0
        plot.alpha = 0.8
        for r,d in zip(B.ra, B.dec):
            #bw,bh = 2048, 2048
            bw,bh = 3600, 3600
            wcs = Tan(r, d, bw/2, bh/2, -pixscale, 0., 0., pixscale, bw,bh)
            out.wcs = anwcs_new_tan(wcs)
            plot.plot('outline')

        fn = 'ccd-%s2.png' % band
        plot.write(fn)
        print 'wrote', fn
