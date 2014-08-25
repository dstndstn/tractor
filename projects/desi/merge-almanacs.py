from astrometry.util.fits import *
import os
import sys

'''

A little script for merging Arjun's almanac files (with zeropoints)
into a FITS table.

'''

if __name__ == '__main__':

    infns = [
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140810.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140811.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140812.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140813.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140814.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140815.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140816.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140817.txt',
        '/project/projectdirs/cosmo/work/decam/cats/Almanac_DECaLS_CP20140818.txt',
        ]
    outfn = 'almanac.fits'

    TT = []
    for fn in infns:

        sedcmd = 'sed "s/po[dD] [Uu]p/pod_up/g" | sed "s/hex check/hex_check/g"'
        tmpfn = 'txt'
        cmd = 'cat %s | %s > %s' % (fn, sedcmd, tmpfn)
        print cmd
        if os.system(cmd):
            sys.exit(-1)

        headerline = open(tmpfn).readline()
        print 'header', headerline
        
        T = streaming_text_table(tmpfn)
        T.about()
        TT.append(T)

    T = merge_tables(TT)
    T.about()
    
    for c in T.get_columns():
        x = T.get(c)
        try:
            x = x.astype(int)
        except:
            try:
                x = x.astype(float)
            except:
                continue
        T.set(c, x)

    T.about()

    T.rename('1sig-4"ap', 'onesig_4sec')
    T.rename('RMS(DMag)', 'rms_dmag')

    T.writeto('almanac.fits')
    
