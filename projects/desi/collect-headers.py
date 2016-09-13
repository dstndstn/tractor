from __future__ import print_function
import os
#from glob import glob
import fitsio

if __name__ == '__main__':
    import optparse
    import sys
    
    parser = optparse.OptionParser('%prog [options] <input filenames>')
    parser.add_option('-o', dest='outdir', default='headers', help='Output base dir')
    opt,args = parser.parse_args()

    if not len(args):
        parser.print_help()
        sys.exit(-1)

    for infn in args:
        print()
        print('Input file', infn)
        fn = os.path.basename(infn)
        path = os.path.dirname(infn)
        band = os.path.basename(path)
        path = os.path.dirname(path)
        chip = os.path.basename(path)
        path = os.path.dirname(path)
        date = os.path.basename(path)

        print('Date', date, 'chip', chip, 'band', band)

        outdir = os.path.join(opt.outdir, date, chip, band)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except:
                pass
        outfn = os.path.join(outdir, fn)
        print('Output', outfn)

        hdr = fitsio.read_header(infn)
        print('OBSTYPE:', hdr['OBSTYPE'], type(hdr['OBSTYPE']))

        #if hdr['OBSTYPE'] == 'object':
        for rr in hdr.records():
            #print rr
            if rr['name'] == 'OBSTYPE':
                print(rr)
                rr['value'] = 'object'
                
        print('OBSTYPE:', hdr['OBSTYPE'])
        fitsio.write(outfn, None, header=hdr, clobber=True)
        print('Wrote header:', len(hdr), 'cards')
        
