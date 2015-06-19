#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
# per http://stackoverflow.com/questions/3306518/cannot-pass-an-argument-to-python-with-usr-bin-env-python

from astrometry.util.fits import *
from astrometry.sdss.dr10 import *
import os
import shutil

T = fits_table('window_flist-cut.fits')

###
T.cut((T.ra > 180) * (T.ra < 195) * (T.dec > 5) * (T.dec < 20))
print 'CUT TO', len(T), 'in region'

# projects/desi/runbrick.py : stage_srcs
extracols = ['parent', 'tai', 'mjd', 'psf_fwhm', 'objc_flags2', 'flags2',
             'devflux_ivar', 'expflux_ivar', 'calib_status', 'raerr', 'decerr']
# projects/desi/common.py : get_sdss_sources
cols = ['objid', 'ra', 'dec', 'fracdev', 'objc_type',
        'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr',
        'phi_dev_deg',
        'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr',
        'phi_exp_deg',
        'resolve_status', 'nchild', 'flags', 'objc_flags',
        'run','camcol','field','id',
        'psfflux', 'psfflux_ivar',
        'cmodelflux', 'cmodelflux_ivar',
        'modelflux', 'modelflux_ivar',
        'devflux', 'expflux', 'extinction'] + extracols

# "dr10", "dr12", etc doesn't really matter
sdss = DR10()
sdss.useLocalTree()

outdir = os.path.join(os.environ['SCRATCH'], 'sdss-cut')
if not os.path.exists(outdir):
    os.makedirs(outdir)

for i,t in enumerate(T):
    print 'Copying', i+1, 'of', len(T)
    fn = sdss.getPath('photoObj', t.run, t.camcol, t.field, rerun='301')
    outfn = fn.replace(os.environ['BOSS_PHOTOOBJ'], outdir)
    #print 'output filename', outfn
    if os.path.exists(outfn):
        print '  Output exists:', outfn
        try:
            #tt = fits_table(outfn)
            #print '  Read', len(tt)
            cmd = 'liststruc ' + outfn
            rtn = os.system(cmd)
            print 'return', rtn
            if rtn == 0:
                continue
        except:
            print 'Failed to read existing output file', outfn
    print '  Reading', fn
    S = fits_table(fn, columns=cols)
    od = os.path.dirname(outfn)
    if not os.path.exists(od):
        os.makedirs(od)
    tempfn = outfn + '.tmp'
    if S is None:
        print '  Input is empty; copying directly'
        shutil.copyfile(fn, tempfn)
    else:
        S.writeto(tempfn)
    # copy filesystem mode, timestamps, etc.
    #shutil.copymode(fn, tempfn)
    #shutil.copystat(fn, tempfn)
    os.rename(tempfn, outfn)
    print '  Wrote', outfn

