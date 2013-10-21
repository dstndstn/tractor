#! /usr/bin/env python

'''
This is a PBS driver script we used on riemann to perform
WISE-from-SDSS forced photometry.  

Jobs are submitted like:
qsub -l "nodes=1:ppn=1" -l "walltime=3:00:00" -N w1v4 -o w1v4.log -q batch -t 100-190 ./runslice.py

The "-t" option (job arrays) causes the PBS_ARRAYID environment
variable to be set for each job.  We use that to determine which chunk
of work that job is going to do.

In the case of the eBOSS W3 test, we split the region into boxes in
RA,Dec and used the job id for the Dec slice (90 slices).

(I was also using this as a non-PBS command-line driver; batch=False
then)

'''

import os
import sys

from astrometry.util.ttime import *
import logging
if __name__ == '__main__':
    Time.add_measurement(MemMeas)
    Time.add_measurement(IoMeas)
    lvl = logging.INFO
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

import numpy as np

from astrometry.util.multiproc import multiproc
from astrometry.util.file import *

# duck-type command-line options
class myopts(object):
    pass

def main():
    batch = False
    
    arr = os.environ.get('PBS_ARRAYID')
    if arr is not None:
        arr = int(arr)
        batch = True
    
    # This gets set when running runslice.py from the command-line within an interactive job...
    d = os.environ.get('PBS_O_WORKDIR')
    if batch and d is not None:
        os.chdir(d)
        sys.path.append(os.getcwd())
        
    opt = myopts()
    opt.nonsdss = False
    ps = None
    
    if True:
        # W3 area

        if arr is None:
            # which slice to do for interactive jobs
            #arr = 147
            #arr = 148
            arr = 149

        opt.sources = 'objs-eboss-w3-dr9.fits'
        NDEC = 50
        NRA = 90
        band = int(arr / 100)
        ri = arr % 100
        print 'Band', band
        print 'RA slice', ri

        r0,r1 = 210.593,  219.132
        d0,d1 =  51.1822,  54.1822

        basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise_frames'
        opt.wisedatadirs = [(basedir, 'merged'),]

        # basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise1test'
        # opt.wisedatadirs = [(os.path.join(basedir, 'allsky'), 'cryo'),
        #                     (os.path.join(basedir, 'prelim_postcryo'), 'post-cryo'),]
        opt.minflux = None
        opt.wsources = 'wise-objs-w3.fits'
        opt.bandnum = band
        opt.osources = None
        opt.minsb = 0.005
        opt.ptsrc = False
        opt.pixpsf = False
    
        if False:
            # eboss w3 v4
            basename = 'ebossw3-v4'
            opt.ptsrc = False
            opt.pixpsf = False
        if False:
            # eboss w3 v5
            basename = 'ebossw3-v5'
            opt.ptsrc = True

        if True:
            # eboss w3 v6  (after the fact)
            basename = 'eboss-w3-v6'
    
        if not batch:
            basename = 'eboss-w3-tst'
            opt.ptsrc = False
            opt.pixpsf = False
    
    if False:
        # Stripe82 QSO truth-table region
        base = '/clusterfs/riemann/raid000/bosswork/boss/wise1ext/sdss_stripe82'
        opt.sources = os.path.join(base, 'objs-eboss-stripe82-dr9.fits')
    
        r0, r1 = 317.0, 330.0
        d0, d1 = 0., 1.25
    
        NRA = 260
        NDEC = 25
    
        if arr is None:
            # which slice to do for interactive jobs
            arr = 1000
    
        band = int(arr / 1000)
        ri = arr % 1000
        print 'Band', band
        print 'RA slice', ri
    
        basedir = '/clusterfs/riemann/raid000/bosswork/boss/wise1test_stripe82/old'
        opt.wisedatadirs = [(os.path.join(basedir, 'allsky'), 'cryo'),
                            (os.path.join(basedir, 'prelim_postcryo'), 'post-cryo'),]
    
        opt.minflux = None
        opt.bandnum = band
        opt.osources = None
        opt.minsb = 0.005
        opt.ptsrc = False
        opt.pixpsf = False
        
        # v1
        basename = 'eboss-s82-v1'
        opt.ptsrc = False
        opt.pixpsf = False
    
        if not batch:
            basename = 'eboss-s82-tst'
            opt.ptsrc = False
            opt.pixpsf = False
            
    if not batch:
        import matplotlib
        matplotlib.use('Agg')
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence(basename)

    # Has to happen *after* matplotlib init...
    import wise3
    
    dd = np.linspace(d0, d1, NDEC + 1)
    rr = np.linspace(r0, r1, NRA  + 1)

    mp = multiproc()

    tr0 = Time()
    
    rlo,rhi = rr[ri], rr[ri+1]
    for di,(dlo,dhi) in enumerate(zip(dd[:-1], dd[1:])):
    
        fn = '%s-r%02i-d%02i-w%i.fits' % (basename, ri, di, opt.bandnum)
        if os.path.exists(fn):
            print 'Output file exists:', fn
            print 'Skipping'
            if batch:
                continue
    
        # HACK!!
        #if not batch and di != 25:
        #   continue

        td0 = Time()
        
        try:
            P = dict(ralo=rlo, rahi=rhi, declo=dlo, dechi=dhi,
                     opt=opt, mp=mp, ps=ps)

            t00 = Time()

            for stage in [ 100, 101, 102, 103, 104, 105, 106, 700 ]:
                f = eval('wise3.stage%i' % stage)
                ts0 = Time()
                R = f(**P)
                dt = Time() - ts0
                P.update(R)
                P.update(**{ 't%i' % stage: dt })
                print 'stage', stage, ':',  dt

            dt = Time() - t00
            P.update(**{ 'ttotal': dt })
    
            # Grab result FITS structure
            R = P['R']
            R.writeto(fn)
            print 'Wrote', fn
    
            imst = P['imstats']
            fn = '%s-r%02i-d%02i-w%i-imstats.fits' % (basename, ri, di, opt.bandnum)
            imst.writeto(fn)
            print 'Wrote', fn
    
            pfn = '%s-r%02i-d%02i-w%i.pickle' % (basename, ri, di, opt.bandnum)
    
            # tractor = P['tractor']
            # ims1 = P['ims1']
            # res1 = []
            # for tim,(img,mod,ie,chi,roi) in zip(tractor.images, ims1):
            #     for k in ['origInvvar', 'starMask', 'inverr', 'cinvvar', 'goodmask',
            #               'maskplane', 'rdmask', 'mask', 'uncplane', 'vinvvar']:
            # 
            #         ### DEBUG
            #         #continue
            #         # Debug
            #         if k == 'rdmask':
            #             continue
            #               
            #         try:
            #             delattr(tim, k)
            #         except:
            #             pass
            # 
            #     res1.append((tim, mod, roi))
            # 
            # PP = dict(res1=res1, cat=tractor.getCatalog(), rd=P['rd'], ri=ri, di=di,
            #           bandnum=opt.bandnum, S=P['S'],
            #           ralo=rlo, rahi=rhi, declo=dlo, dechi=dhi,
            #           opt=opt,
            #           T=P['T'])


            pickle_to_file(P, pfn)

            print
            print 'Dec slice time:', Time() - td0
            print
            
    
        except:
            import traceback
            print '---------------------------------------------------'
            print 'FAILED: dec slice', di, 'ra slice', ri
            print rlo,rhi, dlo,dhi
            print '---------------------------------------------------'
            traceback.print_exc()
            print '---------------------------------------------------'
            if not batch:
                raise


    print
    print 'RA slice time:', Time() - tr0
    print



if __name__ == '__main__':
    import cProfile
    from datetime import datetime
    cProfile.run('main()', 'prof-runslice-%s.dat' % (datetime.now().isoformat()))

    
