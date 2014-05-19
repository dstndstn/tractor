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

from astrometry.util.siap import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.starutil_numpy import *
from astrometry.util.util import *

ps = PlotSequence('decov')

T = fits_table('schlegel-z.fits')
T.ra = np.array([float(s) for s in T.ra])
T.dec = np.array([float(s) for s in T.dec])

#T.cut((T.ra > 148) * (T.ra < 152) * (T.dec > 0) * (T.dec < 4))
T.cut((T.ra > 350) * (T.ra < 355) * (T.dec > -2) * (T.dec < 2))

print len(T), 'in range'
print T.filename
print T.ra
print T.dec

#ralo,rahi = 149.25, 151.0
#declo,dechi = 1.5, 3.0
ralo,rahi = 351.37, 353.62
declo,dechi = -0.11, 0.44

overlap = []

plt.clf()
for i,fn in enumerate(T.filename):
    pth = '/global/project/projectdirs/desi/imaging/redux/decam/proc/' + fn
    wcs = Sip(pth)
    r0,r1,d0,d1 = wcs.radec_bounds()
    plt.plot([r0,r1,r1,r0,r0], [d0,d0,d1,d1,d0], 'b-', alpha=0.25)

    if r1 < ralo or r0 > rahi:
        continue
    if d1 < declo or d0 > dechi:
        continue
    overlap.append(i)
ps.savefig()

overlap = np.array(overlap)
T.cut(overlap)
print
print 'Overlapping:', len(T)
print T.filename
print T.ra
print T.dec

plt.clf()
for i,fn in enumerate(T.filename):
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

    
