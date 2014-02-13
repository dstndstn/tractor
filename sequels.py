#! /usr/bin/env python

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import pylab as plt
import os
import sys
from glob import glob

import fitsio

'''
SEQUELS target selection, "v5", 2013-10-29
scripts/sequels-12.pbs:
    python -u sequels.py -d data/sequels-phot-5 --blocks 1 --ceres -B 8 --dataset sequels -b 1234
python -u sequels.py -d data/sequels-phot-5 --finish --dataset sequels --pobj data/sequels-pobj-5/

# bonus flat file:
python -u sequels.py --dataset sequels -d data/sequels-phot-5 --finish --flat sequels-phot-v5.fits

wise-coadds -> /clusterfs/riemann/raid000/dstn/unwise/unwise/
data -> /clusterfs/riemann/raid000/dstn

(for x in data/sequels-phot-5/*.fits; do echo; echo $x; listhead $x | grep SEQ_; done) > sequels-phot-versions.txt
gzip sequels-phot-versions.txt

(for x in wise-coadds/*/*/*-img-m.fits; do echo; echo $x; listhead $x | grep UNW; done) > sequels-unwise-versions.txt
gzip sequels-unwise-versions.txt

# Later, whole SDSS footprint:
scripts/sequels-12.pbs, as before.

python -u sequels.py --dataset sdss --finish data/sequels-phot-5/phot-unsplit-*.fits > fin.log
(at r24313)

Later, found that at least one tile failed: 2586p651 (index 6327).

mkdir data/redo
cp data/sequels-phot-5/phot-2586p651.fits data/redo/
python -u sequels.py --dataset sdss -d data/redo --pobj data/redo --finish --split data/redo/phot-2586p651.fits > redo-1.log 2>&1 &
cp data/sequels-phot-5/phot-unsplit-{2551p651,2543p636,2560p666,2576p636,2635p666,2621p651,2609p636,2597p666}.fits data/redo/
python -u sequels.py --dataset sdss -d data/redo --pobj data/redo --finish data/redo/phot-unsplit-*.fits > redo-2.log 2>&1

Hand-edit list of files produced, cutting down to just the run/camcol/fields touched by 2586p651 (listed in redo-1.log)
Copy those files into place.

Next, Adam found that one source from photoObj 1365/6/68 is missing.
Turns out my WISE/SDSS search radius was a touch too small and I just
barely missed WISE tile 1621p681 (and 7 other tiles) which contains
that source.  Tweak unwise/sdss-footprint-wise.py, creating
sdss2-atlas.fits.
python -u sequels.py --dataset sdss2 -d data/redo --pobj data/redo --tempdir data/redo -v --split 7175 --tiledir data/unwise/unwise-nersc/ > redo-3.log 2>&1
cp data/sequels-phot-5/phot-unsplit-{1670p666,1632p666,1661p681,1595p666,1582p681,1590p696}.fits data/redo
python -u sequels.py --dataset sdss -d data/redo --pobj data/redo --finish data/redo/phot-unsplit-*.fits > redo-4.log 2>&1

These new tiles affect the splitting of existing tiles -- their
neighbors previously did not exist (in the atlas file) so did not
split SDSS fields.

python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split \
data/sequels-phot-5/phot-????????.fits data/redo/phot-????????.fits > fin3.log 2>&1 &

ls data/sequels-phot-5/phot-????????.fits data/redo/phot-????????.fits > lst
tail -n    +1 lst | head -n 1000 > lst.1
tail -n +1001 lst | head -n 1000 > lst.2
tail -n +2001 lst | head -n 1000 > lst.3
tail -n +3001 lst | head -n 1000 > lst.4
tail -n +4001 lst | head -n 1000 > lst.5
tail -n +5001 lst | head -n 1000 > lst.6
tail -n +6001 lst | head -n 1000 > lst.7
tail -n +7001 lst | head -n 1000 > lst.8

python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.1) > fin3-1.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.2) > fin3-2.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.3) > fin3-3.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.4) > fin3-4.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.5) > fin3-5.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.6) > fin3-6.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.7) > fin3-7.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish --split $(cat lst.8) > fin3-8.log 2>&1 &

python -u sequels.py --dataset sdss2 -d data/redo3 --pobj data/redo3 --tempdir data/redo3 --finish data/redo3/phot-unsplit-*.fits --no-threads > fin4.log 2>&1 &

python -u check-wise.py > chk5.log 2>&1 &
# starting from the end,
python -u check-wise.py > chk6.log 2>&1 &

One glitch turned up: 2566/2/329, where one tile thinks the field is
totally within, so writes the output during splitting.  Another tile
contains a single source (at y coord -0.2), so overwrites the output
file. Patch up:

python -u sequels.py --dataset sdss2 -d data/redo4 --pobj data/redo4 --tempdir data/redo4 --finish --split data/sequels-phot-5/phot-0031p136.fits --no-threads > fin5.log
python -u sequels.py --dataset sdss2 -d data/redo4 --pobj data/redo4 --tempdir data/redo4 --finish data/redo4/phot-unsplit-0031p136.fits data/redo3/phot-unsplit-0031p151.fits --no-threads > fin6.log 2>&1 &

 cp data/redo3/301/2566/2/photoWiseForced-002566-2-0329.fits data/redo4/redo3-photoWiseForced-002566-2-0329.fits
 cp data/redo4/301/2566/2/photoWiseForced-002566-2-0329.fits data/redo3/301/2566/2/

Argh, 4334/3/15 -- tile 0461p106 -- phot output file was written, but
empty; photoobjs non-empty.  Log file looks like it timed out?

cp sdss-phot-temp/photoobjs-0461p106.fits data/redo5/
python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 -v --split --tiledir data/unwise/unwise/ 955 > redo-5.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish data/redo3/phot-unsplit-{0476p106,0459p090,0474p090,0446p106,0463p121,0478p121}.fits data/redo5/phot-unsplit-0461p106.fits --no-threads > redo-5b.log 2>&1 &

 cp -a data/redo3/301/4334 data/redo5/run-4334-redo3
 cp data/redo5/301/4334/3/photoWiseForced-004334-3-00{13,14,15,16,17}.fits data/redo3/301/4334/3/
 cp data/redo5/301/4334/4/photoWiseForced-004334-4-00{11,18,19}.fits data/redo3/301/4334/4/
 cp data/redo5/301/4334/5/photoWiseForced-004334-5-0020.fits data/redo3/301/4334/5/
 cp data/redo5/301/4334/6/photoWiseForced-004334-6-00{11,18,19,20}.fits data/redo3/301/4334/6/

And 4874/4/698 --

python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish --split data/sequels-phot-5/phot-0408p000.fits --no-threads > fin7.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish data/redo5/phot-unsplit-0408p000.fits data/redo3/phot-unsplit-0423p000.fits --no-threads > fin8.log 2>&1 &

 cp data/redo3/301/4874/4/photoWiseForced-004874-4-0698.fits data/redo5/
 cp data/redo5/301/4874/4/photoWiseForced-004874-4-0698.fits data/redo3/301/4874/4/

And 3630/2/220 --

python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish --split data/sequels-phot-5/phot-1501p090.fits --no-threads > fin9.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish data/redo5/phot-unsplit-1501p090.fits data/redo3/phot-unsplit-{1492p106,1485p090}.fits --no-threads > fin10.log 2>&1 &

 cp data/redo3/301/3630/2/photoWiseForced-003630-2-0220.fits data/redo5/
 cp data/redo5/301/3630/2/photoWiseForced-003630-2-0220.fits data/redo3/301/3630/2/

And 4136/3/206 and 4136/5/206

python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish --split data/sequels-phot-5/phot-0574p000.fits --no-threads > fin11.log 2>&1 &
python -u sequels.py --dataset sdss2 -d data/redo5 --pobj data/redo5 --tempdir data/redo5 --finish data/redo5/phot-unsplit-0574p000.fits data/redo3/phot-unsplit-0589p000.fits --no-threads > fin12.log 2>&1 &

'''

''' CFHT-LS W3 test area
http://terapix.iap.fr/article.php?id_article=841
wget 'ftp://ftpix.iap.fr/pub/CFHTLS-zphot-T0007/photozCFHTLS-W3_270912.out.gz'
gunzip photozCFHTLS-W3_270912.out.gz 
text2fits.py -n "*********" -f sddjjffffffjfjffffffjfffjffffffffffffffffff -H "id ra dec flag stargal r2 photoz zpdf zpdf_l68 zpdf_u168 chi2_zpdf mod ebv nbfilt zmin zl68 zu68 chi2_best zp_2 chi2_2 mods chis zq chiq modq u g r i z y eu eg er ei ez ey mu mg mr mi mz my" photozCFHTLS-W3_270912.out photozCFHTLS-W3_270912.fits
'''

''' Stripe82 deep QSO plates (RA 36-42): see deepqso.py for notes.
'''

'''
Relevant files/directories are:

DATASET-atlas.fits
  the coadd tiles to process

(--tiledir)
tiledir ("wise-coadds")/xxx/xxxx[pm]xxx/unwise-xxxx[pm]xxx-wW-img-m.fits
                             and {invvar,std,n}-m.fits
  WISE coadd tiles

photoobjdir ("photoObjs-new")
  SDSS photoObj files

resolvedir ("photoResolve-new")/window_flist.fits
  SDSS resolve files

tempoutdir ("DATASET-phot-temp")/photoobjs-TILE.fits
                                /wise-sources-TILE.fits
  SDSS photoObjs files, and WISE catalog sources
  
(-d)
outdir ("DATASET-phot")/phot-TILE.fits
                       /phot-unsplit-TILE.fits
                       /phot-wise-TILE.fits
  phot-TILE.fits: WISE forced photometry for photoobjs-TILE.fits objects
  phot-unsplit-TILE.fits: objects that straddle tiles
  phot-wise-TILE.fits: photometry for WISE-only sources too

(--pobj)
pobjoutdir ("DATASET-pobj")/RERUN/RUN/CAMCOL/photoWiseForced-R-C-F.fits
  phot-TILE.fits entries split back into SDSS fields

photoobj-lengths.sqlite3
  Database holding the length of PhotoObj files


About --split:

With --finish --split:
   --will overwrite existing "pobj" output files!
   --will write OUTDIR/phot-unsplit-*.fits files


If you use --split when running tiles, it will still write out the
normal OUTDIR/phot-TILE.fits files, as well as
POBJ/RERUN/RUN/CAMCOL/photoWiseForced-R-C-F.fits files, and
OUTDIR/phot-unsplit-TILE.fits.  After all tiles have finished, one
must still run --finish on the phot-unsplit-TILE.fits files.
   
'''
'''
About field_primary.fits: from DR10 CAS:
"select rerun, run, camcol, field, primaryArea from Field"

If primaryArea = 0, can be ignored when reading photoObjs.
----> This turns out to be untrue -- fields eg 1365/6/68 with a single
      PRIMARY object will have area of zero.
'''

if __name__ == '__main__':
    arr = os.environ.get('PBS_ARRAYID')
    d = os.environ.get('PBS_O_WORKDIR')
    if arr is not None and d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())

from astrometry.util.file import *
from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.util.miscutils import *
from astrometry.util.util import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.util.starutil_numpy import *
from astrometry.util.sdss_radec_to_rcf import *
from astrometry.util.ttime import *

from tractor import *
from tractor.sdss import *

from wisecat import wise_catalog_radecbox

import logging
'''
ln -s /clusterfs/riemann/raid007/ebosswork/eboss/photoObj photoObjs-new
ln -s /clusterfs/riemann/raid006/bosswork/boss/resolve/2013-07-29 photoResolve-new
'''

photoobjdir = 'photoObjs-new'
resolvedir = 'photoResolve-new'

if __name__ == '__main__':
    tiledir = 'wise-coadds'

    outdir = '%s-phot'
    tempoutdir = '%s-phot-temp'
    pobjoutdir = '%s-pobj'

    Time.add_measurement(MemMeas)

photoobj_length_db = 'photoobj-lengths.sqlite3'
def get_photoobj_length(rerun, run, camcol, field, save=True, all=False):
    import sqlite3
    timeout = 60.
    create = not os.path.exists(photoobj_length_db)

    conn = sqlite3.connect(photoobj_length_db, timeout)
    c = conn.cursor()
    if create:
        print 'Creating db table in', photoobj_length_db
        c.execute('create table photoObjs (rerun text, run integer, ' +
                  'camcol integer, field integer, N integer)')
        conn.commit()

    if all:
        print 'Fetching all photoObj lengths'
        pobjs = {}
        for row in c.execute('select rerun, run, camcol, field, N from photoObjs'):
            rerun, run, camcol, field, N = row
            pobjs[(int(rerun), int(run), int(camcol), int(field))] = int(N)
        return pobjs

    print 'Getting photoObj length for', rerun, run, camcol, field
    #print type(rerun), type(run), type(camcol), type(field)

    # Ensure types (when read from FITS tables, they can be np.int16, eg)
    run = int(run)
    camcol = int(camcol)
    field = int(field)
    rerun = str(rerun)

    c.execute('select N from photoObjs where rerun=? and run=? and camcol=? '
              + 'and field=?', (rerun, run, camcol, field))
    row = c.fetchone()
    if row is None:
        # This photoObj is unknown
        pofn = get_photoobj_filename(rerun, run,camcol,field)
        F = fitsio.FITS(pofn)
        N = F[1].get_nrows()

        if save:
            c.execute('insert into photoObjs values (?,?,?,?,?)',
                      (rerun, run, camcol, field, N))
            conn.commit()
    else:
        print 'Row:', row
        N = row[0]
    conn.close()
    return N


def get_tile_dir(basedir, coadd_id):
    return os.path.join(basedir, coadd_id[:3], coadd_id)

def get_photoobj_filename(rr, run, camcol, field):
    fn = os.path.join(photoobjdir, rr, '%i'%run, '%i'%camcol,
                      'photoObj-%06i-%i-%04i.fits' % (run, camcol, field))
    return fn

def read_photoobjs(wcs, margin, cols=None):
    '''
    Read photoObjs that are inside the given 'wcs', plus 'margin' in degrees.
    '''
    log = logging.getLogger('sequels.read_photoobjs')

    if cols is None:
        cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type', 'modelflux',
                'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
                'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
                'resolve_status', 'nchild', 'flags', 'objc_flags',
                'run','camcol','field','id'
                ]

        # useful to have these in the outputs...
        cols += ['psfflux', 'psfflux_ivar', 'cmodelflux', 'cmodelflux_ivar',
                 'modelflux', 'modelflux_ivar']


    wfn = os.path.join(resolvedir, 'window_flist.fits')

    ra,dec = wcs.radec_center()
    rad = wcs.radius()
    rad += np.hypot(13., 9.) / 2 / 60.
    # a little extra margin
    rad += margin

    print 'Searching for run,camcol,fields with radius', rad, 'deg'
    RCF = radec_to_sdss_rcf(ra, dec, radius=rad*60., tablefn=wfn)
    log.debug('Found %i fields possibly in range' % len(RCF))

    pixmargin = margin * 3600. / wcs.pixel_scale()
    W,H = wcs.get_width(), wcs.get_height()
    
    TT = []
    sdss = DR9()
    for run,camcol,field,r,d in RCF:
        log.debug('RCF %i/%i/%i' % (run, camcol, field))
        rr = sdss.get_rerun(run, field=field)
        if rr in [None, '157']:
            log.debug('Rerun 157')
            continue

        fn = get_photoobj_filename(rr, run, camcol, field)

        T = fits_table(fn, columns=cols)
        if T is None:
            log.debug('read 0 from %s' % fn)
            continue
        log.debug('read %i from %s' % (len(T), fn))

        # while we're reading it, record its length for later...
        get_photoobj_length(rr, run, camcol, field)

        ok,x,y = wcs.radec2pixelxy(T.ra, T.dec)
        x -= 1
        y -= 1
        T.cut((x > -pixmargin) * (x < (W + pixmargin)) *
              (y > -pixmargin) * (y < (H + pixmargin)) *
              (T.resolve_status & 256) > 0)
        log.debug('cut to %i within target area and PRIMARY.' % len(T))
        if len(T) == 0:
            continue

        TT.append(T)
    if not len(TT):
        return None
    T = merge_tables(TT)
    return T
    

class BrightPointSource(PointSource):
    '''
    A class to use a pre-computed (constant) model, if available.
    '''
    def __init__(self, *args):
        super(BrightPointSource, self).__init__(*args)
        self.pixmodel = None
    def getUnitFluxModelPatch(self, *args, **kwargs):
        if self.pixmodel is not None:
            return self.pixmodel
        return super(BrightPointSource, self).getUnitFluxModelPatch(*args, **kwargs)

def set_bright_psf_mods(cat, WISE, T, brightcut, band, tile, wcs, sourcerad):
    mag = WISE.get('w%impro' % band)
    I = np.flatnonzero(mag < brightcut)
    if len(I) == 0:
        return
    BW = WISE[I]
    BW.nm = NanoMaggies.magToNanomaggies(mag[I])
    print len(I), 'catalog sources brighter than mag', brightcut
    I,J,d = match_radec(BW.ra, BW.dec, T.ra, T.dec, 4./3600., nearest=True)
    print 'Matched to', len(I), 'catalog sources (nearest)'
    if len(I) == 0:
        return

    fn = 'wise-psf-avg-pix-bright.fits'
    psfimg = fitsio.read(fn, ext=band-1).astype(np.float32)
    psfimg = np.maximum(0, psfimg)
    psfimg /= psfimg.sum()
    print 'PSF image', psfimg.shape
    print 'PSF image range:', psfimg.min(), psfimg.max()
    ph,pw = psfimg.shape
    pcx,pcy = ph/2, pw/2
    assert(ph == pw)
    phalf = ph/2

    ## HACK -- read an L1b frame to get the field rotation...
    thisdir = get_tile_dir(tiledir, tile.coadd_id)
    framesfn = os.path.join(thisdir, 'unwise-%s-w%i-frames.fits' % (tile.coadd_id, band))
    F = fits_table(framesfn)
    print 'intfn', F.intfn[0]
    #fwcs = fits_table(F.intfn[
    wisedir = 'wise-frames'
    scanid,frame = F.scan_id[0], F.frame_num[0]
    scangrp = scanid[-2:]
    fn = os.path.join(wisedir, scangrp, scanid, '%03i' % frame, 
                      '%s%03i-w%i-int-1b.fits' % (scanid, frame, band))
    fwcs = Tan(fn)
    # Keep CD matrix, set CRVAL/CRPIX to star position
    fwcs.set_crpix(pcx+1, pcy+1)
    fwcs.set_imagesize(float(pw), float(ph))

    for i,j in zip(I, J):
        if not isinstance(cat[j], BrightPointSource):
            print 'Bright source matched non-point source', cat[j]
            continue

        fwcs.set_crval(BW.ra[i], BW.dec[i])
        L=3
        Yo,Xo,Yi,Xi,rims = resample_with_wcs(wcs, fwcs, [psfimg], L)
        x0,x1 = int(Xo.min()), int(Xo.max())
        y0,y1 = int(Yo.min()), int(Yo.max())
        mod = np.zeros((1+y1-y0, 1+x1-x0), np.float32)
        mod[Yo-y0, Xo-x0] += rims[0]

        pat = Patch(x0, y0, mod)
        cat[j].pixmodel = pat

        cat[j].fixedRadius = phalf
        sourcerad[j] = max(sourcerad[j], phalf)

def _get_photoobjs(tile, wcs, bandnum, existOnly):
    objfn = os.path.join(tempoutdir, 'photoobjs-%s.fits' % tile.coadd_id)
    if os.path.exists(objfn):
        if existOnly:
            print 'Exists:', objfn
            return
        print 'Reading', objfn
        T = fits_table(objfn)
    else:
        print 'Did not find', objfn, '-- reading photoObjs'
        T = read_photoobjs(wcs, 1./60.)
        if T is None:
            return None
        T.writeto(objfn)
        print 'Wrote', objfn
        if existOnly:
            return
    # Cut galaxies based on signal-to-noise of theta (effective
    # radius) measurement.
    b = bandnum
    gal = (T.objc_type == 3)
    dev = gal * (T.fracdev[:,b] >= 0.5)
    exp = gal * (T.fracdev[:,b] <  0.5)
    stars = (T.objc_type == 6)
    print sum(dev), 'deV,', sum(exp), 'exp, and', sum(stars), 'stars'
    print 'Total', len(T), 'sources'

    thetasn = np.zeros(len(T))
    T.theta_deverr[dev,b] = np.maximum(1e-6, T.theta_deverr[dev,b])
    T.theta_experr[exp,b] = np.maximum(1e-5, T.theta_experr[exp,b])
    # theta_experr nonzero: 1.28507e-05
    # theta_deverr nonzero: 1.92913e-06
    thetasn[dev] = T.theta_dev[dev,b] / T.theta_deverr[dev,b]
    thetasn[exp] = T.theta_exp[exp,b] / T.theta_experr[exp,b]

    aberrzero = np.zeros(len(T), bool)
    aberrzero[dev] = (T.ab_deverr[dev,b] == 0.)
    aberrzero[exp] = (T.ab_experr[exp,b] == 0.)

    maxtheta = np.zeros(len(T), bool)
    maxtheta[dev] = (T.theta_dev[dev,b] >= 29.5)
    maxtheta[exp] = (T.theta_exp[exp,b] >= 59.0)

    # theta S/N > modelflux for dev, 10*modelflux for exp
    bigthetasn = (thetasn > (T.modelflux[:,b] * (1.*dev + 10.*exp)))

    print sum(gal * (thetasn < 3.)), 'have low S/N in theta'
    print sum(gal * (T.modelflux[:,b] > 1e4)), 'have big flux'
    print sum(aberrzero), 'have zero a/b error'
    print sum(maxtheta), 'have the maximum theta'
    print sum(bigthetasn), 'have large theta S/N vs modelflux'
    
    badgals = gal * reduce(np.logical_or,
                           [thetasn < 3.,
                            T.modelflux[:,b] > 1e4,
                            aberrzero,
                            maxtheta,
                            bigthetasn,
                            ])
    print 'Found', sum(badgals), 'bad galaxies'
    T.treated_as_pointsource = badgals
    T.objc_type[badgals] = 6
    return T

def one_tile(tile, opt, savepickle, ps, tiles, tiledir, tempoutdir, T=None):

    bands = opt.bands
    outfn = opt.output % (tile.coadd_id)
    savewise_outfn = opt.save_wise_output % (tile.coadd_id)

    version = get_svn_version()
    print 'SVN version info:', version

    sband = 'r'
    bandnum = 'ugriz'.index(sband)

    tt0 = Time()
    print
    print 'Coadd tile', tile.coadd_id

    thisdir = get_tile_dir(tiledir, tile.coadd_id)
    fn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits' % (tile.coadd_id, bands[0]))
    if os.path.exists(fn):
        print 'Reading', fn
        wcs = Tan(fn)
    else:
        print 'File', fn, 'does not exist; faking WCS'
        from unwise_coadd import get_coadd_tile_wcs
        wcs = get_coadd_tile_wcs(tile.ra, tile.dec)

    r0,r1,d0,d1 = wcs.radec_bounds()
    print 'RA,Dec bounds:', r0,r1,d0,d1
    H,W = wcs.get_height(), wcs.get_width()

    if T is None:
        T = _get_photoobjs(tile, wcs, bandnum, opt.photoObjsOnly)
        if T is None:
            print 'Empty tile'
            return
        if opt.photoObjsOnly:
            return
    print len(T), 'objects'
    if len(T) == 0:
        return

    defaultflux = 1.

    # hack
    T.psfflux    = np.zeros((len(T),5), np.float32) + defaultflux
    T.cmodelflux = T.psfflux
    T.devflux    = T.psfflux
    T.expflux    = T.psfflux

    ok,T.x,T.y = wcs.radec2pixelxy(T.ra, T.dec)
    T.x = (T.x - 1.).astype(np.float32)
    T.y = (T.y - 1.).astype(np.float32)
    margin = 20.
    I = np.flatnonzero((T.x >= -margin) * (T.x < W+margin) *
                       (T.y >= -margin) * (T.y < H+margin))
    T.cut(I)
    print 'Cut to margins: N objects:', len(T)
    if len(T) == 0:
        return

    # Use pixelized PSF models for bright sources?
    bright_mods = ((1 in bands) and (opt.bright1 is not None))

    wanyband = wband = 'w'

    #if cat is None:
    classmap = {}
    if bright_mods:
        classmap = {PointSource: BrightPointSource}

    print 'Creating tractor sources...'
    cat = get_tractor_sources_dr9(None, None, None, bandname=sband,
                                  objs=T, bands=[], nanomaggies=True,
                                  extrabands=[wband],
                                  fixedComposites=True,
                                  useObjcType=True,
                                  classmap=classmap)
    print 'Created', len(T), 'sources'
    assert(len(cat) == len(T))

    pixscale = wcs.pixel_scale()
    # crude intrinsic source radii, in pixels
    sourcerad = np.zeros(len(cat))
    for i in range(len(cat)):
        src = cat[i]
        if isinstance(src, PointSource):
            continue
        elif isinstance(src, HoggGalaxy):
            sourcerad[i] = (src.nre * src.re / pixscale)
        elif isinstance(src, FixedCompositeGalaxy):
            sourcerad[i] = max(src.shapeExp.re * ExpGalaxy.nre,
                               src.shapeDev.re * DevGalaxy.nre) / pixscale
    print 'sourcerad range:', min(sourcerad), max(sourcerad)

    # Find WISE-only catalog sources
    wfn = os.path.join(tempoutdir, 'wise-sources-%s.fits' % (tile.coadd_id))
    print 'looking for', wfn
    if os.path.exists(wfn):
        WISE = fits_table(wfn)
        print 'Read', len(WISE), 'WISE sources nearby'
    else:
        cols = ['ra','dec'] + ['w%impro'%band for band in [1,2,3,4]]

        print 'wise_catalog_radecbox:', r0,r1,d0,d1
        if r1 - r0 > 180:
            # assume wrap-around; glue together 0-r0 and r1-360
            Wa = wise_catalog_radecbox(0., r0, d0, d1, cols=cols)
            Wb = wise_catalog_radecbox(r1, 360., d0, d1, cols=cols)
            WISE = merge_tables([Wa, Wb])
        else:
            WISE = wise_catalog_radecbox(r0, r1, d0, d1, cols=cols)
        WISE.writeto(wfn)
        print 'Found', len(WISE), 'WISE sources nearby'

    for band in bands:
        mag = WISE.get('w%impro' % band)
        nm = NanoMaggies.magToNanomaggies(mag)
        WISE.set('w%inm' % band, nm)
        print 'Band', band, 'max WISE catalog flux:', max(nm)
        print '  (min mag:', mag.min(), ')'

    unmatched = np.ones(len(WISE), bool)
    I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 4./3600.)
    unmatched[I] = False
    UW = WISE[unmatched]
    print 'Got', len(UW), 'unmatched WISE sources'

    if opt.savewise:
        fitwiseflux = {}
        for band in bands:
            fitwiseflux[band] = np.zeros(len(UW))

    # Record WISE fluxes for catalog matches.
    # (this provides decent initialization for 'minsb' approx.)
    wiseflux = {}
    for band in bands:
        wiseflux[band] = np.zeros(len(T))
        if len(I) == 0:
            continue
        # X[I] += Y[J] with duplicate I doesn't work.
        #wiseflux[band][J] += WISE.get('w%inm' % band)[I]
        lhs = wiseflux[band]
        rhs = WISE.get('w%inm' % band)[I]
        print 'Band', band, 'max matched WISE flux:', max(rhs)
        for j,f in zip(J, rhs):
            lhs[j] += f

    ok,UW.x,UW.y = wcs.radec2pixelxy(UW.ra, UW.dec)
    UW.x -= 1.
    UW.y -= 1.

    T.coadd_id = np.array([tile.coadd_id] * len(T))
    T.cell = np.zeros(len(T), np.int16)
    T.cell_x0 = np.zeros(len(T), np.int16)
    T.cell_y0 = np.zeros(len(T), np.int16)
    T.cell_x1 = np.zeros(len(T), np.int16)
    T.cell_y1 = np.zeros(len(T), np.int16)

    inbounds = np.flatnonzero((T.x >= -0.5) * (T.x < W-0.5) *
                              (T.y >= -0.5) * (T.y < H-0.5))

    for band in bands:
        tb0 = Time()
        print
        print 'Coadd tile', tile.coadd_id
        print 'Band', band
        wband = 'w%i' % band

        imfn = os.path.join(thisdir, 'unwise-%s-w%i-img-m.fits'    % (tile.coadd_id, band))
        ivfn = os.path.join(thisdir, 'unwise-%s-w%i-invvar-m.fits' % (tile.coadd_id, band))
        ppfn = os.path.join(thisdir, 'unwise-%s-w%i-std-m.fits'    % (tile.coadd_id, band))
        nifn = os.path.join(thisdir, 'unwise-%s-w%i-n-m.fits'      % (tile.coadd_id, band))

        print 'Reading', imfn
        wcs = Tan(imfn)
        r0,r1,d0,d1 = wcs.radec_bounds()
        print 'RA,Dec bounds:', r0,r1,d0,d1
        ra,dec = wcs.radec_center()
        print 'Center:', ra,dec
        img = fitsio.read(imfn)
        print 'Reading', ivfn
        iv = fitsio.read(ivfn)
        print 'Reading', ppfn
        pp = fitsio.read(ppfn)
        print 'Reading', nifn
        nims = fitsio.read(nifn)

        sig1 = 1./np.sqrt(np.median(iv))
        minsig = getattr(opt, 'minsig%i' % band)
        minsb = sig1 * minsig
        print 'Sigma1:', sig1, 'minsig', minsig, 'minsb', minsb

        # Load the average PSF model (generated by wise_psf.py)
        P = fits_table('wise-psf-avg.fits', hdu=band)
        psf = GaussianMixturePSF(P.amp, P.mean, P.var)

        # Render the PSF profile for figuring out source radii for
        # approximation purposes.
        R = 100
        psf.radius = R
        pat = psf.getPointSourcePatch(0., 0.)
        assert(pat.x0 == pat.y0)
        assert(pat.x0 == -R)
        psfprofile = pat.patch[R, R:]
        #print 'PSF profile:', psfprofile

        # Reset default flux based on min radius
        defaultflux = minsb / psfprofile[opt.minradius]
        print 'Setting default flux', defaultflux

        # Set WISE source radii based on flux
        UW.rad = np.zeros(len(UW), int)
        wnm = UW.get('w%inm' % band)
        for r,pro in enumerate(psfprofile):
            flux = minsb / pro
            UW.rad[wnm > flux] = r
        UW.rad = np.maximum(UW.rad + 1, 3)

        # Set SDSS fluxes based on WISE catalog matches.
        wf = wiseflux[band]
        I = np.flatnonzero(wf > defaultflux)
        wfi = wf[I]
        print 'Initializing', len(I), 'fluxes based on catalog matches'
        for i,flux in zip(I, wf[I]):
            assert(np.isfinite(flux))
            cat[i].getBrightness().setBand(wanyband, flux)

        # Set SDSS radii based on WISE flux
        rad = np.zeros(len(I), int)
        for r,pro in enumerate(psfprofile):
            flux = minsb / pro
            rad[wfi > flux] = r
        srad2 = np.zeros(len(cat), int)
        srad2[I] = rad
        del rad

        # Set radii
        for i in range(len(cat)):
            src = cat[i]
            # set fluxes
            b = src.getBrightness()
            if b.getBand(wanyband) <= defaultflux:
                b.setBand(wanyband, defaultflux)
                
            R = max([opt.minradius, sourcerad[i], srad2[i]])
            # ??  This is used to select which sources are in-range
            sourcerad[i] = R
            if isinstance(src, PointSource):
                src.fixedRadius = R
            elif (isinstance(src, HoggGalaxy) or
                  isinstance(src, FixedCompositeGalaxy)):
                src.halfsize = R

        # Use pixelized PSF models for bright sources?
        bright_mods = ((band == 1) and (opt.bright1 is not None))

        if bright_mods:
            set_bright_psf_mods(cat, WISE, T, opt.bright1, band, tile, wcs, sourcerad)

        # We're going to dice the image up into cells for
        # photometry... remember the whole image and initialize
        # whole-image results.
        fullimg = img
        fullinvvar = iv
        fullIV = np.zeros(len(cat))
        fskeys = ['prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']
        fitstats = dict([(k, np.zeros(len(cat))) for k in fskeys])

        twcs = ConstantFitsWcs(wcs)

        #sky = estimate_sky(img, iv)
        #print 'Estimated sky', sky
        sky = 0.
        
        imgoffset = 0.
        if opt.sky and opt.nonneg:
            # the non-negative constraint applies to the sky too!
            # Artificially offset the sky value, AND the image pixels.
            offset = 10. * sig1
            if sky < 0:
                offset += -sky

            imgoffset = offset
            sky += offset
            img += offset
            print 'Offsetting image by', offset
            
        tsky = ConstantSky(sky)

        if opt.errfrac > 0:
            pix = (fullimg - imgoffset)
            nz = (fullinvvar > 0)
            iv2 = np.zeros_like(fullinvvar)
            iv2[nz] = 1./(1./fullinvvar[nz] + (pix[nz] * opt.errfrac)**2)
            print 'Increasing error estimate by', opt.errfrac, 'of image flux'
            fullinvvar = iv2

        # cell positions
        XX = np.round(np.linspace(0, W, opt.blocks+1)).astype(int)
        YY = np.round(np.linspace(0, H, opt.blocks+1)).astype(int)

        if ps:

            tag = '%s W%i' % (tile.coadd_id, band)
            
            plt.clf()
            n,b,p = plt.hist((fullimg - imgoffset).ravel(), bins=100,
                             range=(-10*sig1, 20*sig1), log=True,
                             histtype='step', color='b')
            mx = max(n)
            plt.ylim(0.1, mx)
            plt.xlim(-10*sig1, 20*sig1)
            plt.axvline(sky, color='r')
            plt.title('%s: Pixel histogram' % tag)
            ps.savefig()

            if bright_mods:
                mod = np.zeros_like(fullimg)
                for src in cat:
                    if src.pixmodel:
                        src.pixmodel.add(mod, scale=src.getBrightness().getBand(wanyband))

                plt.clf()
                plt.imshow(mod, interpolation='nearest', origin='lower',
                           cmap='gray',
                           vmin=-3*sig1, vmax=10*sig1)
                plt.colorbar()
                plt.title('%s: bright star models' % tag)
                ps.savefig()
    
                plt.clf()
                plt.imshow(fullimg - imgoffset - mod, interpolation='nearest', origin='lower',
                           cmap='gray',
                           vmin=-3*sig1, vmax=10*sig1)
                plt.colorbar()
                plt.title('%s: data - bright star models' % tag)
                ps.savefig()

            # plt.clf()
            # plt.imshow(np.log10(pp), interpolation='nearest', origin='lower', cmap='gray',
            #            vmin=0)
            # plt.title('log Per-pixel std')
            # ps.savefig()

            if opt.blocks > 1:
                plt.clf()
                plt.imshow(fullimg - imgoffset, interpolation='nearest', origin='lower',
                           cmap='gray',
                           vmin=-3*sig1, vmax=10*sig1)
                ax = plt.axis()
                plt.colorbar()
                for x in XX:
                    plt.plot([x,x], [0,H], 'r-', alpha=0.5)
                for y in YY:
                    plt.plot([0,W], [y,y], 'r-', alpha=0.5)
                    celli = -1
                    for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
                        for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
                            celli += 1
                            xc,yc = (xlo+xhi)/2., (ylo+yhi)/2.
                            plt.text(xc, yc, '%i' % celli, color='r')
    
                            print 'Cell', celli, 'xc,yc', xc,yc
                            print 'W, H', xhi-xlo, yhi-ylo
                            print 'RA,Dec center', wcs.pixelxy2radec(xc+1, yc+1)
    
                if bright_mods:
                    mag = WISE.get('w%impro' % band)
                    I = np.flatnonzero(mag < opt.bright1)
                    for i in I:
                        ok,x,y = wcs.radec2pixelxy(WISE.ra[i], WISE.dec[i])
                        plt.text(x-1, y-1, '%.1f' % mag[i], color='g')
    
                plt.axis(ax)
                plt.title('%s: cells' % tag)
                ps.savefig()
            
            print 'Median # ims:', np.median(nims)

        if savepickle:
            mods = []
            cats = []

        celli = -1
        for yi,(ylo,yhi) in enumerate(zip(YY, YY[1:])):
            for xi,(xlo,xhi) in enumerate(zip(XX, XX[1:])):
                celli += 1

                if len(opt.cells) and not celli in opt.cells:
                    print 'Skipping cell', celli
                    continue

                print
                print 'Cell', celli, 'of', (opt.blocks**2), 'for', tile.coadd_id, 'band', wband

                imargin = 12
                # SDSS and WISE source margins beyond the image margins ( + source radii )
                smargin = 1
                wmargin = 1

                # image region: [ix0,ix1)
                ix0 = max(0, xlo - imargin)
                ix1 = min(W, xhi + imargin)
                iy0 = max(0, ylo - imargin)
                iy1 = min(H, yhi + imargin)
                slc = (slice(iy0, iy1), slice(ix0, ix1))
                print 'Image ROI', ix0, ix1, iy0, iy1
                img    = fullimg   [slc]
                invvar = fullinvvar[slc]
                twcs.setX0Y0(ix0, iy0)

                tim = Image(data=img, invvar=invvar, psf=psf, wcs=twcs,
                            sky=tsky, photocal=LinearPhotoCal(1., band=wanyband),
                            name='Coadd %s W%i (%i,%i)' % (tile.coadd_id, band, xi,yi),
                            domask=False)

                # Relevant SDSS sources:
                m = smargin + sourcerad
                I = np.flatnonzero(((T.x+m) >= (ix0-0.5)) * ((T.x-m) < (ix1-0.5)) *
                                   ((T.y+m) >= (iy0-0.5)) * ((T.y-m) < (iy1-0.5)))
                inbox = ((T.x[I] >= (xlo-0.5)) * (T.x[I] < (xhi-0.5)) *
                         (T.y[I] >= (ylo-0.5)) * (T.y[I] < (yhi-0.5)))
                # Inside this cell
                srci = I[inbox]
                # In the margin
                margi = I[np.logical_not(inbox)]

                # sources in the ROI box
                subcat = [cat[i] for i in srci]

                # include *copies* of sources in the margins
                # (that way we automatically don't save the results)
                subcat.extend([cat[i].copy() for i in margi])
                assert(len(subcat) == len(I))

                # add WISE-only sources in the expanded region
                m = wmargin + UW.rad
                J = np.flatnonzero(((UW.x+m) >= (ix0-0.5)) * ((UW.x-m) < (ix1-0.5)) *
                                   ((UW.y+m) >= (iy0-0.5)) * ((UW.y-m) < (iy1-0.5)))

                if opt.savewise:
                    jinbox = ((UW.x[J] >= (xlo-0.5)) * (UW.x[J] < (xhi-0.5)) *
                              (UW.y[J] >= (ylo-0.5)) * (UW.y[J] < (yhi-0.5)))
                    uwcat = []
                wnm = UW.get('w%inm' % band)
                nomag = 0
                for ji,j in enumerate(J):
                    if not np.isfinite(wnm[j]):
                        nomag += 1
                        continue
                    ptsrc = PointSource(RaDecPos(UW.ra[j], UW.dec[j]),
                                              NanoMaggies(**{wanyband: wnm[j]}))
                    ptsrc.radius = UW.rad[j]
                    subcat.append(ptsrc)
                    if opt.savewise:
                        if jinbox[ji]:
                            uwcat.append((j, ptsrc))
                        
                print 'WISE-only:', nomag, 'of', len(J), 'had invalid mags'
                print 'Sources:', len(srci), 'in the box,', len(I)-len(srci), 'in the margins, and', len(J), 'WISE-only'

                # if ps:
                #     plt.clf()
                #     plt.imshow(img - imgoffset, interpolation='nearest', origin='lower',
                #                cmap='gray', vmin=-3*sig1, vmax=10*sig1)
                #     plt.colorbar()
                #     xx,yy = [],[]
                #     for src in subcat:
                #         x,y = twcs.positionToPixel(src.getPosition())
                #         xx.append(x)
                #         yy.append(y)
                #     p1 = plt.plot(xx[:len(srci)], yy[:len(srci)], 'b+')
                #     p2 = plt.plot(xx[len(srci):len(I)], yy[len(srci):len(I)], 'g+')
                #     p3 = plt.plot(xx[len(I):], yy[len(I):], 'r+')
                #     p4 = plt.plot(UW.x[np.logical_not(np.isfinite(wnm[J]))],
                #                   UW.y[np.logical_not(np.isfinite(wnm[J]))],
                #                   'y+')
                #     ps.savefig()

                print 'Creating a Tractor with image', tim.shape, 'and', len(subcat), 'sources'
                tractor = Tractor([tim], subcat)

                print 'Running forced photometry...'
                t0 = Time()
                tractor.freezeParamsRecursive('*')

                if opt.sky:
                    tractor.thawPathsTo('sky')
                    print 'Initial sky values:'
                    for tim in tractor.getImages():
                        print tim.getSky()

                tractor.thawPathsTo(wanyband)

                wantims = (savepickle or opt.pickle2 or (ps is not None) or opt.save_fits)

                R = tractor.optimize_forced_photometry(
                    minsb=minsb, mindlnp=1., sky=opt.sky, minFlux=None,
                    fitstats=True, variance=True, shared_params=False,
                    use_ceres=opt.ceres, BW=opt.ceresblock, BH=opt.ceresblock,
                    wantims=wantims, nonneg=opt.nonneg, negfluxval=0.1*sig1)
                print 'That took', Time()-t0

                if wantims:
                    ims0 = R.ims0
                    ims1 = R.ims1
                IV,fs = R.IV, R.fitstats

                if opt.sky:
                    print 'Fit sky values:'
                    for tim in tractor.getImages():
                        print tim.getSky()

                if opt.savewise:
                    for (j,src) in uwcat:
                        fitwiseflux[band][j] = src.getBrightness().getBand(wanyband)

                if opt.save_fits:
                    (dat,mod,ie,chi,roi) = ims1[0]

                    tag = 'fit-%s-w%i' % (tile.coadd_id, band)
                    fitsio.write('%s-data.fits' % tag, dat, clobber=True)
                    fitsio.write('%s-mod.fits' % tag,  mod, clobber=True)
                    fitsio.write('%s-chi.fits' % tag,  chi, clobber=True)

                if ps:

                    tag = '%s W%i cell %i/%i' % (tile.coadd_id, band, celli, opt.blocks**2)

                    (dat,mod,ie,chi,roi) = ims1[0]

                    plt.clf()
                    plt.imshow(dat - imgoffset, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-3*sig1, vmax=10*sig1)
                    plt.colorbar()
                    plt.title('%s: data' % tag)
                    ps.savefig()

                    # plt.clf()
                    # plt.imshow(1./ie, interpolation='nearest', origin='lower',
                    #            cmap='gray', vmin=0, vmax=10*sig1)
                    # plt.colorbar()
                    # plt.title('%s: sigma' % tag)
                    # ps.savefig()

                    plt.clf()
                    plt.imshow(mod - imgoffset, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-3*sig1, vmax=10*sig1)
                    plt.colorbar()
                    plt.title('%s: model' % tag)
                    ps.savefig()

                    plt.clf()
                    plt.imshow(chi, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-5, vmax=+5)
                    plt.colorbar()
                    plt.title('%s: chi' % tag)
                    ps.savefig()

                    # plt.clf()
                    # plt.imshow(np.round(chi), interpolation='nearest', origin='lower',
                    #            cmap='jet', vmin=-5, vmax=+5)
                    # plt.colorbar()
                    # plt.title('Chi')
                    # ps.savefig()

                    plt.clf()
                    plt.imshow(chi, interpolation='nearest', origin='lower',
                               cmap='gray', vmin=-20, vmax=+20)
                    plt.colorbar()
                    plt.title('%s: chi 2' % tag)
                    ps.savefig()

                    plt.clf()
                    n,b,p = plt.hist(chi.ravel(), bins=100,
                                     range=(-10, 10), log=True,
                                     histtype='step', color='b')
                    mx = max(n)
                    plt.ylim(0.1, mx)
                    plt.axvline(0, color='r')
                    plt.title('%s: chi' % tag)
                    ps.savefig()

                    # fn = ps.basefn + '-chi.fits'
                    # fitsio.write(fn, chi, clobber=True)
                    # print 'Wrote', fn

                if savepickle:
                    # FIXME -- imgoffset
                    if ims1 is None:
                        mod = None
                    else:
                        im,mod,ie,chi,roi = ims1[0]
                    mods.append(mod)
                    cats.append((
                        srci, margi, UW.x[J], UW.y[J],
                        T.x[srci], T.y[srci], T.x[margi], T.y[margi],
                        [src.copy() for src in cat],
                        [src.copy() for src in subcat]))

                if opt.pickle2:
                    fn = opt.output % (tile.coadd_id)
                    fn = fn.replace('.fits','-cell%02i.pickle' % celli)
                    pickle_to_file((ims0, ims1, cat, subcat), fn)
                    print 'Pickled', fn
                    

                if len(srci):
                    T.cell[srci] = celli
                    T.cell_x0[srci] = ix0
                    T.cell_x1[srci] = ix1
                    T.cell_y0[srci] = iy0
                    T.cell_y1[srci] = iy1
                    # Save fit stats
                    fullIV[srci] = IV[:len(srci)]
                    for k in fskeys:
                        x = getattr(fs, k)
                        fitstats[k][srci] = np.array(x)

                cpu0 = tb0.meas[0]
                t = Time()
                cpu = t.meas[0]
                dcpu = (cpu.cpu - cpu0.cpu)
                print 'So far:', Time()-tb0, '-> predict CPU time', (dcpu * (opt.blocks**2) / float(celli+1))

        if bright_mods:
            # Reset pixelized models
            for src in cat:
                if isinstance(src, BrightPointSource):
                    src.pixmodel = None

        nm = np.array([src.getBrightness().getBand(wanyband) for src in cat])
        nm_ivar = fullIV
        T.set(wband + '_nanomaggies', nm.astype(np.float32))
        T.set(wband + '_nanomaggies_ivar', nm_ivar.astype(np.float32))
        dnm = np.zeros(len(nm_ivar), np.float32)
        okiv = (nm_ivar > 0)
        dnm[okiv] = (1./np.sqrt(nm_ivar[okiv])).astype(np.float32)
        okflux = (nm > 0)
        mag = np.zeros(len(nm), np.float32)
        mag[okflux] = (NanoMaggies.nanomaggiesToMag(nm[okflux])).astype(np.float32)
        dmag = np.zeros(len(nm), np.float32)
        ok = (okiv * okflux)
        dmag[ok] = (np.abs((-2.5 / np.log(10.)) * dnm[ok] / nm[ok])).astype(np.float32)

        mag[np.logical_not(okflux)] = np.nan
        dmag[np.logical_not(ok)] = np.nan
        
        T.set(wband + '_mag', mag)
        T.set(wband + '_mag_err', dmag)
        for k in fskeys:
            T.set(wband + '_' + k, fitstats[k].astype(np.float32))

        if ps:
            I,J,d = match_radec(WISE.ra, WISE.dec, T.ra, T.dec, 4./3600.)

            plt.clf()
            lo,hi = 10,25
            cathi = 18
            loghist(WISE.get('w%impro'%band)[I], T.get(wband+'_mag')[J],
                    range=((lo,cathi),(lo,cathi)), bins=200)
            plt.xlabel('WISE W1 mag')
            plt.ylabel('Tractor W1 mag')
            plt.title('WISE catalog vs Tractor forced photometry')
            plt.axis([cathi,lo,cathi,lo])
            ps.savefig()

        print 'Tile', tile.coadd_id, 'band', wband, 'took', Time()-tb0

    T.cut(inbounds)

    T.delete_column('psfflux')
    T.delete_column('cmodelflux')
    T.delete_column('devflux')
    T.delete_column('expflux')
    T.treated_as_pointsource = T.treated_as_pointsource.astype(np.uint8)

    hdr = fitsio.FITSHDR()
    hdr.add_record(dict(name='SEQ_VER', value=version['Revision'],
                        comment='SVN revision'))
    hdr.add_record(dict(name='SEQ_URL', value=version['URL'], comment='SVN URL'))
    hdr.add_record(dict(name='SEQ_DATE', value=datetime.datetime.now().isoformat(),
                        comment='forced phot run time'))
    hdr.add_record(dict(name='SEQ_NNEG', value=opt.nonneg, comment='non-negative?'))
    hdr.add_record(dict(name='SEQ_SKY', value=opt.sky, comment='fit sky?'))
    for band in bands:
        minsig = getattr(opt, 'minsig%i' % band)
        hdr.add_record(dict(name='SEQ_MNS%i' % band, value=minsig,
                            comment='min surf brightness in sig, band %i' % band))
    hdr.add_record(dict(name='SEQ_BL', value=opt.blocks, comment='image blocks'))
    hdr.add_record(dict(name='SEQ_CERE', value=opt.ceres, comment='use Ceres?'))
    hdr.add_record(dict(name='SEQ_ERRF', value=opt.errfrac, comment='error flux fraction'))
    if opt.ceres:
        hdr.add_record(dict(name='SEQ_CEBL', value=opt.ceresblock,
                        comment='Ceres blocksize'))
    
    T.writeto(outfn, header=hdr)
    print 'Wrote', outfn

    if opt.savewise:
        for band in bands:
            UW.set('fit_flux_w%i' % band, fitwiseflux[band])
        UW.writeto(savewise_outfn)
        print 'Wrote', savewise_outfn

    if savepickle:
        fn = opt.output % (tile.coadd_id)
        fn = fn.replace('.fits','.pickle')
        pickle_to_file((mods, cats, T, sourcerad), fn)
        print 'Pickled', fn

    if opt.splitrcf:
        unsplitoutfn = opt.unsplitoutput % (tile.coadd_id)
        cols,dropcols = _get_output_column_names(opt.bands)
        for c in T.get_columns():
            if not c in cols:
                T.delete_column(c)
        splitrcf(tile, tiles, wcs, T, unsplitoutfn, dropcols)

    print 'Tile', tile.coadd_id, 'took', Time()-tt0

def _bounce_split((tile, T, wcs, outfn, unsplitoutfn, cols, dropcols)):
    try:
        print 'Reading', outfn
        objs = fits_table(outfn, columns=cols)
        print 'Read', len(objs), 'from', outfn
        print 'Writing unsplit to', unsplitoutfn
        splitrcf(tile, T, wcs, objs, unsplitoutfn, dropcols)
    except:
        import traceback
        print 'Exception processing', tile.coadd_id
        traceback.print_exc()
        #raise

def splitrcf(tile, tiles, wcs, T, unsplitoutfn, dropcols):
    # -Write out any run,camcol,field that is totally contained
    # within this coadd tile, and does not touch any other coadd
    # tile.
    from unwise_coadd import get_coadd_tile_wcs, walk_wcs_boundary

    print 'Splitting tile', tile.coadd_id
    
    # Find nearby tiles
    I,J,d = match_radec(tiles.ra, tiles.dec,
                        np.array([tile.ra]), np.array([tile.dec]), 2.5)
    neartiles = tiles[I]
    neartiles.cut(neartiles.coadd_id != tile.coadd_id)
    print len(neartiles), 'tiles nearby:', neartiles.coadd_id

    # Decribe all boundaries in Intermediate World Coords with respect
    # to this tile's WCS.
    
    rr,dd = walk_wcs_boundary(wcs)
    ok,uu,vv = wcs.radec2iwc(rr, dd)
    mybounds = np.array(zip(uu,vv))
        
    bounds = []
    for t in neartiles:
        w = get_coadd_tile_wcs(t.ra, t.dec)
        rr,dd = walk_wcs_boundary(w)
        ok,uu,vv = wcs.radec2iwc(rr, dd)
        bounds.append(np.array(zip(uu,vv)))

    RCF = np.unique(zip(T.run, T.camcol, T.field))
    print 'Unique run/camcol/field:', RCF

    wfn = os.path.join(resolvedir, 'window_flist.fits')
    # For --split: figure out which fields are completely within the tile.
    W = fits_table(wfn, columns=['node', 'incl', 'mu_start', 'mu_end',
                                 'nu_start', 'nu_end', 'ra', 'dec',
                                 'rerun', 'run', 'camcol', 'field'])
    straddle = np.zeros(len(T), bool)

    # HACK -- rerun
    rr = '301'

    for run,camcol,field in RCF:
            
        I = np.flatnonzero((W.run == run) * (W.camcol == camcol) *
                           (W.field == field) * (W.rerun == rr))
        assert(len(I) == 1)
        wi = W[I[0]]
        rd = np.array([munu_to_radec_deg(mu, nu, wi.node, wi.incl)
                       for mu,nu in [(wi.mu_start, wi.nu_start),
                                     (wi.mu_start, wi.nu_end),
                                     (wi.mu_end,   wi.nu_end),
                                     (wi.mu_end,   wi.nu_start)]])
        ok,uu,vv = wcs.radec2iwc(rd[:,0], rd[:,1])
        poly = np.array(zip(uu,vv))
        #assert(polygons_intersect(poly, mybounds))

        J = np.flatnonzero((T.run == run) * (T.camcol == camcol) *
                           (T.field == field))

        strads = False
        for b in bounds:
            if polygons_intersect(poly, b):
                print 'Field', run, camcol, field, 'straddles tiles'
                straddle[J] = True
                strads = True
                break
        if strads:
            continue

        print 'Field', run, camcol, field, 'totally within tile', tile.coadd_id

        myoutdir = os.path.join(pobjoutdir, rr, '%i'%run, '%i'%camcol)
        if not os.path.exists(myoutdir):
            try:
                os.makedirs(myoutdir)
            except:
                import traceback
                print 'Exception creating dir', myoutdir
                traceback.print_exc()
                
        outfn = os.path.join(myoutdir, 'photoWiseForced-%06i-%i-%04i.fits' %
                             (run, camcol, field))

        N = get_photoobj_length(rr, run, camcol, field)
        print 'PhotoObj has', N, 'rows'

        Ti = T[J]
        
        P = fits_table()
        P.has_wise_phot = np.zeros(N, bool)
        I = Ti.id - 1
        P.has_wise_phot[I] = True
        for col in Ti.get_columns():
            if col in dropcols:
                continue
            tval = Ti.get(col)
            X = np.zeros(N, tval.dtype)
            X[I] = tval
            P.set(col, X)
        P.delete_column('index')
        P.delete_column('id')
        
        P.writeto(outfn)
        print 'Wrote', outfn

    # -Write out the remaining objects to be --finish'd later.
    S = T[straddle]
    print len(S), 'objects straddle tiles'
    if len(S):
        rcf = np.unique(zip(S.run, S.camcol, S.field))
        print 'Unique rcf of straddled sources:', rcf
        S.writeto(unsplitoutfn)
        print 'Wrote to', unsplitoutfn
    print 'Done'
    

def todo(A, opt, ps):
    need = []
    for i in range(len(A)):
        outfn = opt.output % (A.coadd_id[i])
        #outfn = opt.unsplitoutput % (A.coadd_id[i])
        print 'Looking for', outfn
        if not os.path.exists(outfn):
            need.append(i)
    print ' '.join('%i' %i for i in need)
            
    # Collapse contiguous ranges
    strings = []
    if len(need):
        start = need.pop(0)
        end = start
        while len(need):
            x = need.pop(0)
            if x == end + 1:
                # extend this run
                end = x
            else:
                # run finished; output and start new one.
                if start == end:
                    strings.append('%i' % start)
                else:
                    strings.append('%i-%i' % (start, end))
                start = end = x
        # done; output
        if start == end:
            strings.append('%i' % start)
        else:
            strings.append('%i-%i' % (start, end))
        print ','.join(strings)
    else:
        print 'Done (party now)'

        
def summary(A, opt, ps):
    plt.clf()
    missing = []
    for i in range(len(A)):
        r,d = A.ra[i], A.dec[i]
        dd = 1024 * 2.75 / 3600.
        dr = dd / np.cos(np.deg2rad(d))
        outfn = opt.output % (A.coadd_id[i])
        rr,dd = [r-dr,r-dr,r+dr,r+dr,r-dr], [d-dd,d+dd,d+dd,d-dd,d-dd]
        print 'Looking for', outfn
        if not os.path.exists(outfn):
            missing.append((i,rr,dd,r,d))
        plt.plot(rr, dd, 'k-')
    for i,rr,dd,r,d in missing:
        plt.plot(rr, dd, 'r-')
        plt.text(r, d, '%i' % i, rotation=90, color='b', va='center', ha='center')
    plt.title('missing tiles')
    plt.axis([118, 212, 44,61])
    ps.savefig()

    print 'Missing tiles:', [m[0] for m in missing]

    rdfn = 'rd.fits'
    if not os.path.exists(rdfn):
        fns = glob(os.path.join(tempoutdir, 'photoobjs-*.fits'))
        fns.sort()
        TT = []
        for fn in fns:
            T = fits_table(fn, columns=['ra','dec'])
            print len(T), 'from', fn
            TT.append(T)
        T = merge_tables(TT)
        print 'Total of', len(T)
        T.writeto(rdfn)
    else:
        T = fits_table(rdfn)
        print 'Got', len(T), 'from', rdfn
    
    plt.clf()
    loghist(T.ra, T.dec, 500, range=((118,212),(44,61)))
    plt.xlabel('RA')
    plt.ylabel('Dec')
    ps.savefig()

    ax = plt.axis()
    for i in range(len(A)):
        r,d = A.ra[i], A.dec[i]
        dd = 1024 * 2.75 / 3600.
        dr = dd / np.cos(np.deg2rad(d))
        plt.plot([r-dr,r-dr,r+dr,r+dr,r-dr], [d-dd,d+dd,d+dd,d-dd,d-dd], 'r-')
    plt.axis(ax)
    ps.savefig()

def _get_output_column_names(bands):
    cols = ['ra','dec', 'objid', 'index', 'x','y', 
            'treated_as_pointsource', 'coadd_id', 'modelflux']
    for band in bands:
        for k in ['nanomaggies', 'nanomaggies_ivar', 'mag', 'mag_err',
                  'prochi2', 'pronpix', 'profracflux', 'proflux', 'npix']:
            cols.append('w%i_%s' % (band, k))
    cols.extend(['run','camcol','field','id'])
    # columns to drop from the photoObj-parallels
    dropcols = ['run', 'camcol', 'field', 'modelflux']
    return cols, dropcols

def finish(A, opt, args, ps):
    '''
    A: atlas
    '''
    # Find all *-phot.fits outputs
    # Determine which photoObj files are involved
    # Collate and resolve objs measured in multiple tiles
    # Expand into photoObj-parallel files
    if len(args):
        fns = args
    else:
        fns = glob(os.path.join(outdir, 'phot-????????.fits'))
        fns.sort()
        print 'Found', len(fns), 'photometry output files'

    cols,dropcols = _get_output_column_names(opt.bands)

    if opt.splitrcf:
        from unwise_coadd import get_coadd_tile_wcs
        args = []
        for i in range(len(A)):
            #outfn = opt.output % A.coadd_id[i]
            #if not outfn in fns:
            #    continue
            outfn = None
            coadd_id = A.coadd_id[i]
            for fn in fns:
                if coadd_id in fn:
                    outfn = fn
                    break
            if outfn is None:
                print 'Did not find input file for', coadd_id
                continue
            unsplitoutfn = opt.unsplitoutput % A.coadd_id[i]
            if os.path.exists(unsplitoutfn):
                print 'Exists:', unsplitoutfn
                continue
            print 'File', outfn
            tile = A[i]
            wcs = get_coadd_tile_wcs(tile.ra, tile.dec)
            args.append((tile, A, wcs, outfn, unsplitoutfn, cols, dropcols))

        if opt.no_threads:
            map(_bounce_split, args)
        else:
            mp = multiproc(1)
            mp.map(_bounce_split, args)
        return

    flats = []
    fieldmap = {}
    for ifn,fn in enumerate(fns):
        print
        print 'Reading', (ifn+1), 'of', len(fns), fn
        T = fits_table(fn, columns=cols + opt.ftag)
        print 'Read', len(T), 'entries'

        if opt.flat is not None:
            flats.append(T)
            continue

        rcf = np.unique(zip(T.run, T.camcol, T.field))
        for run,camcol,field in rcf:
            if not (run,camcol,field) in fieldmap:
                fieldmap[(run,camcol,field)] = []
            Tsub = T[(T.run == run) * (T.camcol == camcol) * (T.field == field)]
            #print len(Tsub), 'in', (run,camcol,field)
            for col in dropcols:
                Tsub.delete_column(col)
            print '  ', len(Tsub), 'in', run,camcol,field, ', joining', [t.coadd_id[0] for t in fieldmap[(run,camcol,field)]]
            fieldmap[(run,camcol,field)].append(Tsub)

    # WISE coadd tile CRPIX-1 (x,y in the phot-*.fits files are 0-indexed)
    # (and x,y are based on the first-band (W1 usually) WCS)
    cx,cy = 1023.5, 1023.5

    if opt.flat is not None:
        F = merge_tables(flats)
        print 'Total of', len(F), 'measurements'
        r2 = (F.x - cx)**2 + (F.y - cy)**2
        I,J,d = match_radec(F.ra, F.dec, F.ra, F.dec, 1e-6, notself=True)
        print 'Matched', len(I), 'duplicates'
        keep = np.ones(len(F), bool)
        keep[np.where(r2[I] > r2[J], I, J)] = False
        F.cut(keep)
        print 'Cut to', len(F)
        F.delete_column('index')
        F.delete_column('x')
        F.delete_column('y')
        F.writeto(opt.flat)
        return

    keys = fieldmap.keys()
    keys.sort()

    # HACK
    rr = '301'

    lengths = get_photoobj_length(None, None, None, None, all=True)
    print 'Got', len(lengths), 'photoObj lengths'
    
    args = []
    for i,(run,camcol,field) in enumerate(keys):
        TT = fieldmap.get((run,camcol,field))
        N = lengths.get((int(rr), int(run), int(camcol), int(field)), None)
        if N is None:
            N = get_photoobj_length(rr, run, camcol, field)
        myoutdir = os.path.join(pobjoutdir, rr, '%i'%run, '%i'%camcol)
        if not os.path.exists(myoutdir):
            os.makedirs(myoutdir)
        outfn = os.path.join(myoutdir, 'photoWiseForced-%06i-%i-%04i.fits' % (run, camcol, field))
        args.append((i, len(fieldmap), TT, N, rr, run, camcol, field, outfn, cx, cy))

    if opt.no_threads:
        map(_finish_one, args)
    else:
        mp = multiproc(8)
        mp.map(_finish_one, args)
    print 'Done'

def _finish_one((i, Ntotal, TT, N, rr, run, camcol, field, outfn, cx, cy)):
    print
    print (i+1), 'of', Ntotal, ': R,C,F', (run,camcol,field)
    print len(TT), 'tiles for', (run,camcol,field)

    resolve = (len(TT) > 1)

    if os.path.exists(outfn):
        print 'Output file already exists.  Updating.'
        P = fits_table(outfn)
        assert(N == len(P))
        print 'Read', N, 'from', outfn

        P.R2 = np.empty(N, np.float32)
        P.R2[:] = 1e9
        I = np.flatnonzero((P.x != 0) * (P.y != 0))
        P.R2[I] = (P.x[I] - cx)**2 + (P.y[I] - cy)**2
        resolve = True

    else:
        P = fits_table()
        P.has_wise_phot = np.zeros(N, bool)
        if resolve:
            # Resolve duplicate measurements (in multiple tiles)
            # based on || (x,y) - center ||^2
            P.R2 = np.empty(N, np.float32)
            P.R2[:] = 1e9

    for T in TT:
        coadd = T.coadd_id[0]
        if resolve:
            #print 'Coadd', coadd, ': T:'
            #T.about()
            I = T.id - 1
            R2 = (T.x - cx)**2 + (T.y - cy)**2
            J = (R2 < P.R2[I])
            I = I[J]
            P.R2[I] = R2[J].astype(np.float32)
            print '  tile', coadd, ':', len(I), 'are closest'
            T.cut(J)
            # Note here that we just choose which indices will be
            # *overwritten* by this "T" -- only those closer than any
            # existing 'T'; and those rows may be in turn overwritten
            # by a later one.
        print '  ', len(T), 'from', coadd
        if len(T) == 0:
            continue
        #print 'Coadd', coadd, ': T:'
        #T.about()
        I = T.id - 1
        P.has_wise_phot[I] = True
        pcols = P.get_columns()
        for col in T.get_columns():
            if col in pcols:
                pval = P.get(col)
                #print '  ', col, pval.dtype
                pval[I] = (T.get(col)).astype(pval.dtype)
            else:
                tval = T.get(col)
                X = np.zeros(N, tval.dtype)
                X[I] = tval
                P.set(col, X)

    for delcol in ['index', 'id']:
        if delcol in P.get_columns():
            P.delete_column(delcol)
    if resolve:
        P.delete_column('R2')
        ##
        coadds = np.unique(P.coadd_id)
        print 'Coadds:', coadds
        for c in coadds:
            I = np.flatnonzero((P.coadd_id == c))
            print '  ', len(I), 'from', c
    P.writeto(outfn)
    print 'Wrote', outfn

def check(T, opt, args, ps):
    print 'Walking', pobjoutdir
    decstep = 0.1
    rastep = 0.1
    ralo,rahi = 0., 360.
    declo,dechi = -30., 90.
    Nra = int((rahi - ralo) / rastep)
    Ndec = int((dechi - declo) / decstep)
    hist = np.zeros((Ndec, Nra), np.int16)

    def binimg(img, b):
        hh,ww = img.shape
        hh = int(hh / b) * b
        ww = int(ww / b) * b
        binx = reduce(np.add, [img[:, i:hh:b] for i in range(b)])
        return reduce(np.add, [img[i:ww:b, :] for i in range(b)])

    plt.figure(figsize=(12,5))
    def _plotit():
        fitsio.write('sdss-phot-density.fits', hist, clobber=True)
        plt.clf()
        plt.imshow(binimg(hist, 8), interpolation='nearest', origin='lower',
                   extent=(ralo,rahi,declo,dechi))
        plt.colorbar()
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')
        ps.savefig()

    n = 0
    for dirpath,dirnames,fns in os.walk(pobjoutdir, followlinks=True):
        dirnames.sort()
        fns.sort()
        for fn in fns:
            pth = os.path.join(dirpath, fn)
            print 'Reading', (n+1), pth
            T = fits_table(pth, columns=['ra','dec','has_wise_phot'])
            T.has_wise_phot = (T.has_wise_phot.astype(np.uint8) == ord('T'))
            #print 'has_wise_phot:', np.unique(T.has_wise_phot)
            #print 'has_wise_phot:', np.unique(T.has_wise_phot.astype(np.uint8))
            print '    Got', len(T), ',', sum(T.has_wise_phot), 'with photometry'
            T.cut(T.has_wise_phot)
            if len(T) == 0:
                continue
            for ira,idec in zip(((T.ra  - ralo )/ rastep).astype(int),
                                ((T.dec - declo)/decstep).astype(int)):
                hist[idec,ira] += 1
            n += 1
            if n and n % 1000 == 0:
                _plotit()
    _plotit()

def main():
    import optparse

    global outdir
    global tempoutdir
    global pobjoutdir
    global tiledir

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('--minsig1', dest='minsig1', default=0.1, type=float)
    parser.add_option('--minsig2', dest='minsig2', default=0.1, type=float)
    parser.add_option('--minsig3', dest='minsig3', default=0.1, type=float)
    parser.add_option('--minsig4', dest='minsig4', default=0.1, type=float)
    parser.add_option('--blocks', dest='blocks', default=1, type=int,
                      help='NxN number of blocks to cut the image into')
    parser.add_option('-d', dest='outdir', default=None,
                      help='Output directory')
    parser.add_option('--tempdir', default=None,
                      help='"Temp"-file output directory')
    parser.add_option('--pobj', dest='pobjdir', default=None,
                      help='Output directory for photoObj-parallels')
    parser.add_option('-o', dest='output', default=None, help='Output filename pattern')
    parser.add_option('-b', dest='bands', action='append', type=int, default=[],
                      help='Add WISE band (default: 1,2)')

    parser.add_option('--tiledir', type=str, help='Set input wise-coadds/ dir')

    parser.add_option('--photoobjs-only', dest='photoObjsOnly',
                      action='store_true', default=False,
                      help='Ensure photoobjs file exists and then quit?')

    parser.add_option('-p', dest='pickle', default=False, action='store_true',
                      help='Save .pickle file for debugging purposes')
    parser.add_option('--pp', dest='pickle2', default=False, action='store_true',
                      help='Save .pickle file for each cell?')
    parser.add_option('--plots', dest='plots', default=False, action='store_true')

    parser.add_option('--save-fits', dest='save_fits', default=False, action='store_true')

    parser.add_option('--plotbase', dest='plotbase', help='Base filename for plots')

    parser.add_option('--finish', dest='finish', default=False, action='store_true')

    parser.add_option('--check', dest='check', default=False, action='store_true')

    parser.add_option('--ftag', dest='ftag', action='append', default=[],
                      help='Tag-along extra columns in --finish phase')

    parser.add_option('--flat', dest='flat', type='str', default=None,
                      help='Just write a flat-file of (deduplicated) results, not photoObj-parallels')

    parser.add_option('--summary', dest='summary', default=False, action='store_true')
    parser.add_option('--todo', dest='todo', default=False, action='store_true')

    parser.add_option('--cell', dest='cells', default=[], type=int, action='append',
                      help='Just run certain cells?')

    parser.add_option('--no-ceres', dest='ceres', action='store_false', default=True,
                       help='Use scipy lsqr rather than Ceres Solver?')

    parser.add_option('--ceres-block', '-B', dest='ceresblock', type=int, default=10,
                      help='Ceres image block size (default: %default)')
    parser.add_option('--nonneg', dest='nonneg', action='store_true', default=False,
                      help='With ceres, enable non-negative fluxes?')

    parser.add_option('--minrad', dest='minradius', type=int, default=2,
                      help='Minimum radius, in pixels, for evaluating source models; default %default')

    parser.add_option('--sky', dest='sky', action='store_true', default=False,
                      help='Fit sky level also?')

    parser.add_option('--save-wise', dest='savewise', action='store_true', default=False,
                      help='Save WISE catalog source fits also?')
    parser.add_option('--save-wise-out', dest='save_wise_output', default=None)

    parser.add_option('--dataset', dest='dataset', default='sequels',
                      help='Dataset (region of sky) to work on')

    parser.add_option('--errfrac', dest='errfrac', type=float,
                      help='Add this fraction of flux to the error model.')

    parser.add_option('--bright1', dest='bright1', type=float, default=None,
                      help='Subtract WISE model PSF for stars brighter than this in W1')

    parser.add_option('--split', dest='splitrcf', action='store_true', default=False,
                      help='Split outputs into run/camcol/field right away?')

    parser.add_option('--tile', dest='tile',
                      help='Run a single tile')

    parser.add_option('-v', dest='verbose', default=False, action='store_true')

    parser.add_option('--no-threads', action='store_true')

    opt,args = parser.parse_args()

    opt.unsplitoutput = None
    
    if opt.tiledir:
        tiledir = opt.tiledir

    if len(opt.bands) == 0:
        opt.bands = [1,2]

    # Allow specifying bands like "123"
    bb = []
    for band in opt.bands:
        for s in str(band):
            bb.append(int(s))
    opt.bands = bb
    print 'Bands', opt.bands

    lvl = logging.INFO
    if opt.verbose:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    # sequels-atlas.fits: written by wise-coadd.py
    fn = '%s-atlas.fits' % opt.dataset
    print 'Reading', fn
    T = fits_table(fn)

    if opt.plotbase is None:
        opt.plotbase = opt.dataset + '-phot'
    ps = PlotSequence(opt.plotbase)

    outdir     = outdir     % opt.dataset
    tempoutdir = tempoutdir % opt.dataset
    pobjoutdir = pobjoutdir % opt.dataset

    if opt.pobjdir is not None:
        pobjoutdir = opt.pobjdir

    if opt.outdir is not None:
        outdir = opt.outdir
    else:
        # default
        opt.outdir = outdir

    if opt.tempdir is not None:
        tempoutdir = opt.tempdir
    else:
        # default
        opt.tempdir = tempoutdir
        
    if opt.output is None:
        opt.output = os.path.join(outdir, 'phot-%s.fits')
    if opt.unsplitoutput is None:
        opt.unsplitoutput = os.path.join(outdir, 'phot-unsplit-%s.fits')
    if opt.save_wise_output is None:
        opt.save_wise_output = opt.output.replace('phot-', 'phot-wise-')

    if opt.summary:
        summary(T, opt, ps)
        sys.exit(0)

    if opt.todo:
        todo(T, opt, ps)
        sys.exit(0)
        
    if opt.finish:
        finish(T, opt, args, ps)
        sys.exit(0)

    if opt.check:
        check(T, opt, args, ps)
        sys.exit(0)
        
    for dirnm in [outdir, tempoutdir, pobjoutdir]:
        if not os.path.exists(dirnm):
            try:
                os.makedirs(dirnm)
            except:
                pass

    tiles = []
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        tiles.append(arr)
    else:
        if len(args) == 0:
            tiles.append(0)
        else:
            for a in args:
                if '-' in a:
                    aa = a.split('-')
                    if len(aa) != 2:
                        print 'With arg containing a dash, expect two parts'
                        print aa
                        sys.exit(-1)
                    start = int(aa[0])
                    end = int(aa[1])
                    for i in range(start, end+1):
                        tiles.append(i)
                else:
                    tiles.append(int(a))

    for i in tiles:
        if opt.plots:
            plot = ps
        else:
            plot = None
        print
        print 'Tile index', i
        one_tile(T[i], opt, opt.pickle, plot, T, tiledir, tempoutdir)

if __name__ == '__main__':
    main()

