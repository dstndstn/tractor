'''
SELECT reference, dtpropid, surveyid, release_date, start_date, date_obs, dtpi, ra, dec, telescope, instrument, filter, exposure, obstype, obsmode, proctype, prodtype, seeing, depth, dtacqnam, reference 
AS archive_file, filesize 
FROM voi.siap 
WHERE (release_date < '2014-05-06') 
AND (date_obs < '2012-10-01')
AND (proctype = 'Raw' OR proctype = 'InstCal')
AND (prodtype  IS  NULL OR prodtype <> 'png')
AND (telescope = 'ct4m' AND instrument = 'decam')
AND (exposure>0)
AND (obstype = 'object')
ORDER BY date_obs ASC
limit 10000

-> 1766 rows -> decam-2012-09.vot

SELECT reference, dtpropid, surveyid, release_date, start_date, date_obs, dtpi, ra, dec, telescope, instrument, filter, exposure, obstype, obsmode, proctype, prodtype, seeing, depth, dtacqnam, reference 
AS archive_file, filesize 
FROM voi.siap 
WHERE (release_date < '2014-05-06') 
AND (date_obs between '2012-11-01' and '2012-12-01')
AND (proctype = 'Raw')
AND (prodtype = 'image')
AND (telescope = 'ct4m' AND instrument = 'decam')
AND (exposure>0)
AND (obstype = 'object')
ORDER BY date_obs ASC limit 10000

           to 2012-10-01 -> 1765 rows -> decam-2012-09.vot
2012-10-01 to 2012-11-01 -> 3166 rows -> decam-2012-10.vot
2012-11-01 to 2012-12-01 -> 6327 rows -> decam-2012-11.vot
2012-12-01 to 2013-01-01 -> 4540 rows -> decam-2012-12.vot
2013-01-01 to 2013-02-01 -> 3189 rows -> decam-2013-01.vot
2013-02-01 to 2013-03-01 -> 2990 rows -> decam-2013-02.vot
2013-03-01 to 2013-04-01 -> 1497 rows -> decam-2013-03.vot
2013-04-01 to 2013-05-01 -> 2015 rows -> decam-2013-04.vot
2013-05-01 to 2014-01-01 -> 6145 rows -> decam-2013-05.vot
2014-01-01 to 2014-05-01 -> 1925 rows -> decam-2014-01.vot
'''

'''
http://iraf-nvo.noao.edu/vo-cli/downloads/index.html
vodirectory -t image noao
-> http://archive.noao.edu/nvo/sim/voquery.php
'''

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys

from astrometry.util.siap import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
from astrometry.util.util import *

import fitsio

ps = PlotSequence('decov')


if True:
    # Plot the (WCS) headers extracted by collect-headers.py

    from astrometry.blind.plotstuff import *

    wcsfns = {}
    W,H = 2048,4096
    for dirpath,dirs,files in os.walk('headers-d2f3', followlinks=True):
        for fn in files:
            path = os.path.join(dirpath, fn)
            print 'Path', path
            try:
                hdr = fitsio.read_header(path)
                filt = hdr['FILTER']
                filt = filt.split()[0]
                print 'Filter', filt

                detpos = hdr['DETPOS']
                
                name = ''
                pth = os.path.dirname(path)
                for i in range(4):
                    nm = os.path.basename(pth)
                    print 'nm', nm
                    pth = os.path.dirname(pth)
                    if len(nm) == 3 and nm[0] == 'C':
                        name = nm
                        break

                if not filt in wcsfns:
                    wcsfns[filt] = []
                wcsfns[filt].append((path, name, detpos, W, H))
            except:
                import traceback
                traceback.print_exc()

    # WISE
    W = H = 2048
    for dirpath,dirs,files in os.walk('unwise', followlinks=True):
        for fn in files:
            if not '-img-m.fits' in fn:
                continue
            path = os.path.join(dirpath, fn)
            print 'Path', path
            if 'w1' in fn:
                filt = 'W1'
            elif 'w2' in fn:
                filt = 'W2'
            else:
                print '??', fn
                continue
            if not filt in wcsfns:
                wcsfns[filt] = []
            name = ''
            wcsfns[filt].append((path, name, '', W, H))

    #wcs = anwcs_create_allsky_hammer_aitoff(180., 0., 1000, 500)
    #wcs = anwcs_create_hammer_aitoff(180, -30, 2., 1000, 500, 1)
    #grid,gridlab = 30, 30

    # DEEP2 Field 3
    #wcs = anwcs_create_hammer_aitoff(352, 0., 10., 1000, 500, 1)
    #grid,gridlab = 1,3
    #wcs = anwcs_create_hammer_aitoff(352, 0., 40., 1000, 500, 1)
    #grid,gridlab = 1,1
    #wcs = anwcs_create_hammer_aitoff(352, 0., 60., 800, 800, 1)
    #grid,gridlab = 1,1
    wcs = anwcs_create_box_upsidedown(352.5, 0.25, 2.3, 800, 800)
    grid,gridlab = 1,1

    #plot = Plotstuff(outformat='png', size=(1000,500), rdw=(180, -30, 180))
    plot = Plotstuff(outformat='png')
    plot.wcs = wcs
    plot.set_size_from_wcs()

    for band in ['g','r','z']:
    #for band in ['g','r','z', 'W1','W2']:

        if not band in wcsfns:
            continue

        for label_detpos in [False, True]:
        
            plot.color = 'verydarkblue'
            plot.plot('fill')
            plot.color = 'white'
            plot.alpha = 0.3
            out = plot.outline
            out.fill = 1

            for path,name,detpos,W,H in wcsfns[band]:
                out.set_wcs_file(path, 0)
                out.set_wcs_size(W, H)
                plot.plot('outline')
                if label_detpos:
                    txt = detpos
                else:
                    txt = name
                if len(txt):
                    ok,r,d = out.wcs.pixelxy2radec(W/2, H/2)
                    plot.text_radec(r, d, txt)

            if False:
                # Plot 1/4 x 1/4 WISE tile
                plot.color = 'yellow'
                wcsfn = 'unwise/352/3524p000/unwise-3524p000-w1-img-m.fits'
                wcs = Tan(wcsfn)
                W,H = wcs.get_width(), wcs.get_height()
                nsub = 4
                subw, subh = W/nsub, H/nsub
                subx, suby = 0, 3
                subwcs = Tan(wcs)
                subwcs.set_crpix(wcs.crpix[0] - subx*subw, wcs.crpix[1] - suby*subh)
                subwcs.set_imagesize(subw, subh)
                ansub = anwcs_new_tan(subwcs)
                out.wcs = ansub
                out.fill = 0
                plot.plot('outline')
    
            plot.color = 'white'
            plot.alpha = 0.5
            plot.apply_settings()
            plot.plot_grid(grid, grid, gridlab, gridlab)
            fn = ps.getnext()
            plot.write(fn)
            print 'Wrote', fn

    sys.exit(0)





if False:
    T = fits_table('schlegel-z.fits')
    T.ra = np.array([float(s) for s in T.ra])
    T.dec = np.array([float(s) for s in T.dec])
    T.cut((T.ra > 148) * (T.ra < 152) * (T.dec > 0) * (T.dec < 4))
    T.about()
    print len(T), 'in range'
    print T.filename
    print T.ra
    print T.dec
    
    plt.clf()
    for fn in T.filename:
        pth = '/global/project/projectdirs/desi/imaging/redux/decam/proc/' + fn
        wcs = Sip(pth)
        r0,r1,d0,d1 = wcs.radec_bounds()
        plt.plot([r0,r1,r1,r0,r0], [d0,d0,d1,d1,d0], 'b-', alpha=0.25)
    ps.savefig()

# sys.exit(0)


decfn = 'decam.fits'
if not os.path.exists(decfn):
    TT = []
    for fn in ['decam-2012-09.vot',
               'decam-2012-10.vot',
               'decam-2012-11.vot',
               'decam-2012-12.vot',
               'decam-2013-01.vot',
               'decam-2013-02.vot',
               'decam-2013-03.vot',
               'decam-2013-04.vot',
               'decam-2013-05.vot',
               'decam-2014-01.vot',]:
        T = siap_parse_result(fn=fn)
        TT.append(T)
    T = merge_tables(TT)
    T.about()
    T.writeto(decfn)
else:
    T = fits_table(decfn)

print len(T), 'DECam exposures'

filts = np.array([s.split()[0] for s in T.filter])
T.filter = filts
ufilts = np.unique(T.filter)
print 'Unique filters:', ufilts

plt.clf()
plt.plot(T.ra, T.dec, 'b.')
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.title('DECam pointings')
ps.savefig()

plt.clf()
loghist(T.ra, T.dec, 200, range=((0,360),(-90,40)))
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.title('Number of DECam exposures')
ps.savefig()

plt.clf()
loghist(T.ra, T.dec, 200, range=((0,360),(-90,40)), weights=T.exposure)
plt.xlabel('RA (deg)')
plt.ylabel('Dec (deg)')
plt.title('DECam exposure time')
ps.savefig()

# ['Empty' 'None' 'VR' 'Y' 'g' 'i' 'n.a.' 'r' 'solid' 'u' 'z']
# HACK
ufilts = ['u','g','r','i','z','Y']

# ('DEEP2-F1', 214.25, 52.5),
# ('DEEP2-F3', 253.5, 0),
deeps = [('COSMOS', 150.116667, 2.205833),
         ('DEEP2-F3', 352.5, 0),
         ('DEEP2-F4', 37.5, 0),
         ('Deep QSO', 39, 0),
         ]

for f in ufilts:
    I = np.flatnonzero(T.filter == f)

    plt.clf()
    loghist(T.ra[I], T.dec[I], 200,
            range=((0,360),(-90,40)), weights=T.exposure[I])
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('DECam exposure time: filter %s' % f)
    ax = plt.axis()
    for name,ra,dec in deeps:
        plt.plot(ra,dec,'o', color='w', mec='w', mfc='none', ms=10)
        plt.text(ra+1, dec+1, name, color='w')
    plt.axis(ax)
    ps.savefig()
    

for name,ra,dec in deeps:
    I = np.flatnonzero(degrees_between(ra, dec, T.ra, T.dec) < 1.)
    print len(I), 'near', name
    Ti = T[I]

    for f in ufilts:
        I = np.flatnonzero(Ti.filter == f)

        print 'Filter', f
        print 'PI', [s.strip() for s in Ti.dtpi[I]]
        print 'Exposure', Ti.exposure[I]
        print 'Seeing', Ti.seeing[I]
        #print 'Obstype', Ti.obstype[I]
        #print 'Obsmode', Ti.obsmode[I]
        #print 'Proctype', Ti.proctype[I]
        #print 'Prodtype', Ti.prodtype[I]
        print 'Archive_file', [s.strip() for s in Ti.archive_file[I]]

        plt.clf()
        loghist(Ti.ra[I], Ti.dec[I], 200,
                range=((ra-1,ra+1),(dec-1,dec+1)), weights=Ti.exposure[I])
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        plt.title('%s: DECam exposure time: filter %s' % (name, f))
        ps.savefig()

    
