import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.starutil_numpy import *
from astrometry.blind.plotstuff import *
from astrometry.util.plotutils import *

from common import *

if __name__ == '__main__':

    decals = Decals()
    #decals_dir = os.environ.get('DECALS_DIR')
    #B = fits_table(os.path.join(decals_dir, 'decals-bricks.fits'))
    #B = fits_table('bricks.fits')
    #T = fits_table('ccds.fits')#-20140810.fits')
    #T = fits_table(os.path.join(decals_dir, 'decals-ccds.fits'))
    B = decals.get_bricks()
    T = decals.get_ccds()
    
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
    fmt = 'png'

    if True:
        ps = PlotSequence('ccdzoom')
        hammer = False
        allsky = False
        survey = False
        cut = False
        gridsize = 1.
        gridlabel = 1.
        #rc,dc = 245,8
        #width = 6.
        rc,dc = 244,8
        #width = 1.
        #width = 2.5
        width = 3
        W,H = 1000,1000
        sz = width * 0.6
        T.cut((np.abs(T.ra - rc) < sz) * (np.abs(T.dec - dc) < sz))
        print 'Cut to', len(T), 'in range'
        print 'Bands', np.unique(T.filter)
        plot_bricks = False

        #fmt = 'pdf'
        
        print 'Unique exposures:', np.unique(T.expnum)

        allTT = []
        band = 'g'
        T.cut(T.filter == band)
        print len(T), 'in', band

        expos = T.expnum[np.argsort(degrees_between(T.ra_bore, T.dec_bore, rc, dc))]
        closest = []
        for e in expos:
            if e in closest:
                continue
            closest.append(e)
            if len(closest) == 21:
                break
        print 'Closest exposures:', closest

        #[348256, 348235, 348279, 348278, 348234, 348277, 348257, 348255, 348260, 348282]
        closest = [closest[i] for i in [0,1,2, 7,4,3, 5,6,8,9,10,11,12,13,14,15,16,17,18,19,20]]
        
        ttsum = None
        for i in range(len(closest)):
            TT = T[T.expnum == closest[i]]
            if ttsum is None:
                ttsum = TT
            else:
                ttsum = merge_tables([ttsum, TT])
            allTT.append((ttsum, band))
        allTT = allTT[0:6] + [allTT[-1]]
            
    else:

        ps = PlotSequence('ccdall')
        cut = True
        hammer = True
        allsky = False
        survey = True
        gridsize = 30
        decgridsize = 15
        #rc,dc = 180, 0
        #width = 180
        #rc,dc = 280, 0
        #width = 120
        #rc,dc = 300, 0
        #width = 200
        #W,H = 1000,500

        rc,dc = 300, 15
        width = 170
        W,H = 1000,500
        sz = width

        plot_bricks = False
        

    if plot_bricks:
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

    #fmt = 'pdf'
    ps.suffixes = [fmt]
    
    plot = Plotstuff(outformat=fmt, size=(W,H), rdw=(rc, dc, width))

    if hammer:
        if allsky:
            plot.wcs = anwcs_create_allsky_hammer_aitoff(rc, dc, W, H)
        else:
            plot.wcs = anwcs_create_hammer_aitoff(rc, dc, (360./width), W, H,True)
                                                  
            
    #cmap = { 'g':'green', 'r':'red', 'z':'magenta' }
    cmap = { 'g':'darkgreen', 'r':'red', 'z':'magenta' }

    if allTT is None:
        allTT = []
        for band in np.unique(T.filter):
            TT = T[T.filter == band]
            print len(TT), 'in band', band
            allTT.append((TT,band))

    
    for TT,band in allTT:
        #plot.color = 'verydarkblue'
        plot.color = 'white'
        plot.plot('fill')
    
        #plot.color = 'green'
        plot.color = cmap.get(band, 'white')
        out = plot.outline
        out.fill = 1
        out.stepsize = 5000
        plot.alpha = 0.3

        #plot.bg_rgba = (0.,0.,0.,0.)
        plot.bg_rgba = (1.,1.,1.,1.)
        
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

        if survey:
            plot.color = 'red'
            plot.lw = 2
            plot.apply_settings()
            plot.line_constant_ra (120, -5, 30)
            plot.line_constant_dec( 30, 120, 270)
            plot.line_to_radec(270, 30)
            plot.line_to_radec(240,  0)
            plot.line_to_radec(230, -5)
            #plot.line_to_radec(120, -5)
            plot.line_constant_dec(-5, 230, 120)
            plot.stroke()
    
            plot.move_to_radec(315, -10)
    
            #plot.line_to_radec(315, 15)
            #plot.line_to_radec(330, 15)
            #plot.line_to_radec(330, 30)
            plot.line_constant_ra(315, -10, 30)
    
            plot.line_to_radec(315, 30)
            #plot.line_to_radec( 40, 30)
    
            #plot.line_constant_dec(30, 330, 361)
            plot.line_constant_dec(30, 315, 361)
            plot.line_to_radec(360, 30)
            plot.line_constant_dec(30, 0, 40)
    
            #plot.line_constant_dec2(30, 330-360, 40, 1)
    
            #plot.line_constant_dec(30, 0, 40)
            plot.line_to_radec( 40, 10)
            plot.line_to_radec( 65, 10)
            plot.line_to_radec( 65,-10)
            plot.line_to_radec( 45,-10)
            plot.line_to_radec( 45,  5)
            plot.line_to_radec(355,  5)
            plot.line_to_radec(355,-10)
            plot.line_to_radec(315,-10)
            plot.stroke()
    
        
        #plot.move_to_radec(315, -10)
        
        # fn = ps.getnext()
        # plot.write(fn)
        # print 'Wrote', fn
        
        plot.color = 'gray'
        plot.alpha = 0.5

        # plot.grid.ralo = 240.
        # plot.grid.rahi = 420.
        # plot.grid.declo = -15.
        # plot.grid.dechi =  30.

        if survey:
            plot.plot_grid(gridsize, decgridsize)
            plot.color = 'darkgray'
            #plot.fontsize = 12
            plot.apply_settings()
            #plot.valign = 'B'
            #plot.halign = 'L'
            #plot.label_offset_y = -5
            #plot.label_offset_x = 5
            off = 10
            plot.label_offset_y = -off
            plot.label_offset_x = 0
            plot.valign = 'B'
            plot.halign = 'C'
            for ra in range(180, 360, 30) + range(0, 90, 30):
                plot.text_radec(ra, 30, '%i' % ra)
            plot.valign = 'C'
            plot.halign = 'L'
            plot.label_offset_y = 0
            plot.label_offset_x = off
            for dec in range(-30, 60, 15):
                plot.text_radec(300, dec, '%i' % dec)
            plot.stroke()
        else:
            plot.color = 'darkgray'
            plot.plot_grid(gridsize, gridsize, gridlabel, gridlabel)
            
            
        plot.color = 'white'
        plot.alpha = 0.6
        plot.marker = 'circle'
        #plot.set_markersize(15)
        plot.set_markersize(3)
        plot.apply_settings()
        #for r,d in zip(B.ra, B.dec):
        #    plot.marker_radec(r,d)
        if plot_bricks:
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

        fn = ps.getnext()
        plot.write(fn)
        print 'wrote', fn

        if fmt == 'png' and cut:
            cutfn = ps.getnext()
            cmd = 'pngtopnm %s | pamcut -top 50 -bottom 400 | pnmtopng > %s' % (fn, cutfn)
            os.system(cmd)
        
