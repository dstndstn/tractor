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

ps = PlotSequence('decam')

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

for f in ufilts:
    I = np.flatnonzero(T.filter == f)

    plt.clf()
    loghist(T.ra[I], T.dec[I], 200,
            range=((0,360),(-90,40)), weights=T.exposure[I])
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('DECam exposure time: filter %s' % f)
    ax = plt.axis()
    ra,dec = 150.116667, 2.205833
    plt.plot(ra,dec,'o', color='w', mec='w', mfc='none', ms=10)
    plt.text(ra+1, dec+1, 'COSMOS', color='w')
    plt.axis(ax)
    ps.savefig()
    
