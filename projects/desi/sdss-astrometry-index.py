import matplotlib
matplotlib.use('Agg')

import os
import sys
from glob import glob

from astrometry.util.fits import *
from astrometry.util.multiproc import *
from astrometry.util.plotutils import *
from astrometry.sdss import *


def _run_one((cmd)):
    os.system(cmd)

def main():

    reffn = 'data/decam/sdss-indexes/calibObj-merge-both-2.fits'
    if not os.path.exists(reffn):
        TT = []
        bright = photo_flags1_map.get('BRIGHT')
        fns = glob('/clusterfs/riemann/raid006/bosswork/boss/sweeps/eboss.v5b/301/calibObj-??????-?-stargal-primary.fits.gz')
        fns.sort()
        for fn in fns:
            T = fits_table(fn, columns=['ra','dec','psfflux', 'objc_type', 'objc_flags'])
            print 'Read', fn,
            #T.cut(T.dec < 35.)
            T.cut((T.dec > 31.) * (T.dec < 35))
            print len(T), 'in Dec range',
            if len(T) == 0:
                print
                continue
            #T.cut((T.nchild == 0) * (T.objc_flags & bright) == 0)
            T.cut((T.objc_flags & bright) == 0)
            print len(T), 'non-BRIGHT'
            T.objc_type = T.objc_type.astype(np.int8)
            T.u_psfflux = T.psfflux[:,0]
            T.g_psfflux = T.psfflux[:,1]
            T.r_psfflux = T.psfflux[:,2]
            T.i_psfflux = T.psfflux[:,3]
            T.z_psfflux = T.psfflux[:,4]
            T.delete_column('psfflux')
            #T.delete_column('nchild')
            T.delete_column('objc_flags')
            TT.append(T)
        T = merge_tables(TT)
        del TT
        T.writeto(reffn)

        # for fn in glob('/clusterfs/riemann/raid006/bosswork/boss/sweeps/eboss.v5b/301/calibObj-*-gal.fits.gz'):
        #     T = fits_table(fn, columns=['ra','dec','modelflux'])
        #     T.cut(T.dec < 31.)
        #     if len(T) == 0:
        #         continue
        #     T.flux = T.modelflux
        #     T.delete_column('modelflux')
        #     T.star = np.zeros(len(T), bool)
        #     print 'Read', len(T), 'from', fn
        #     TT.append(T)
        # for fn in glob('/clusterfs/riemann/raid006/bosswork/boss/sweeps/eboss.v5b/301/calibObj-*-star.fits.gz'):
        #     T = fits_table(fn, columns=['ra','dec','psfflux'])
        #     T.cut(T.dec < 31.)
        #     if len(T) == 0:
        #         continue
        #     T.flux = T.psfflux
        #     T.delete_column('psfflux')
        #     T.star = np.ones(len(T), bool)
        #     print 'Read', len(T), 'from', fn
        #     TT.append(T)
        # T = merge_tables(TT)
        # T.u_flux = T.flux[:,0]
        # T.g_flux = T.flux[:,1]
        # T.r_flux = T.flux[:,2]
        # T.i_flux = T.flux[:,3]
        # T.z_flux = T.flux[:,4]
        # T.delete_column('flux')
        # T.writeto(reffn)
        # sys.exit(0)

    if False:
        reffn = 'data/decam/sdss-indexes/calibObj-merge-star-2.fits'
        if True or not os.path.exists(reffn):
            T = fits_table('data/decam/sdss-indexes/calibObj-merge-star.fits')
    
            plothist(T.ra, T.dec, 200)
            plt.savefig('sdss.png')
            sys.exit(0)
            
            T.u_psf = T.psfflux[:,0]
            T.g_psf = T.psfflux[:,1]
            T.r_psf = T.psfflux[:,2]
            T.i_psf = T.psfflux[:,3]
            T.z_psf = T.psfflux[:,4]
            T.delete_column('psfflux')
            T.writeto(reffn)

    # hpsplit data/decam/sdss-indexes/calibObj-merge-star-2.fits -o data/decam/sdss-indexes/sdss-stars-hp%02i-ns2.fits -n 2

    # build-astrometry-index -o data/decam/sdss-indexes/index-sdss-z-hp00-2.fits -P 2 -i data/decam/sdss-indexes/sdss-stars-hp00-ns2.fits -S z_psf -H 0 -s 2 -L 20 -I 1408120 -t data/tmp

    cmds = []
    scale = 2
    for hp in [17]: #range(48):
        indfn = 'data/decam/sdss-indexes/index-sdss-z-hp%02i-%i.fits' % (hp, scale)
        if os.path.exists(indfn):
            print 'Exists:', indfn
            continue
        catfn = 'data/decam/sdss-indexes/sdss-stars-hp%02i-ns2.fits' % hp
        if not os.path.exists(catfn):
            print 'No input catalog:', catfn
            continue
        cmd = ('build-astrometry-index -o %s -P %i -i %s -S z_psfflux -f -H %i -s 2 -L 20 -I 1408130 -t data/tmp'
               % (indfn, scale, catfn, hp))
        print cmd
        cmds.append(cmd)

    mp = multiproc(8)
    mp.map(_run_one, cmds)
        

if __name__ == '__main__':
    main()
    
