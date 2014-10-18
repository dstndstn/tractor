import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import os

os.environ['TMPDIR'] = 'tmp'

import fitsio

from astrometry.util.util import *
from astrometry.util.fits import fits_table,merge_tables
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.resample import resample_with_wcs,OverlapError
from astrometry.util.starutil_numpy import *
from astrometry.libkd.spherematch import match_radec

from tractor import *

from common import *

def main():
    decals = Decals()
    B = decals.get_bricks()
    print 'Bricks:'
    B.about()

    ra,dec = 190.0, 11.0
    
    B.cut(np.argsort(degrees_between(ra, dec, B.ra, B.dec)))
    print 'Nearest bricks:', B.ra[:5], B.dec[:5], B.brickid[:5]

    brick = B[0]
    pixscale = 0.186
    targetwcs = wcs_for_brick(brick, pixscale=pixscale, W=2048, H=2048)

    ccdfn = 'cfht-ccds.fits'
    if os.path.exists(ccdfn):
        T = fits_table(ccdfn)
    else:
        T = get_ccd_list()
        T.writeto(ccdfn)
    print len(T), 'CCDs'
    T.cut(ccds_touching_wcs(targetwcs, T))
    print len(T), 'CCDs touching brick'

    for t in T:
        im = CfhtImage(t)

        # magzp = hdr['PHOT_C'] + 2.5 * np.log10(hdr['EXPTIME'])
        # fwhm = t.seeing / (pixscale * 3600)
        # print '-> FWHM', fwhm, 'pix'

        im.seeing = t.seeing
        pixscale = t.pixscale
        print 'seeing', t.seeing
        print 'pixscale', pixscale*3600, 'arcsec/pix'
        im.run_calibs(t.ra, t.dec, pixscale, W=t.width, H=t.height, psfexfit=False)


    
def get_ccd_list():
    expnums = [ 1080306, 1080946, 1168106, 1617879 ]

    #seeings =  [ 0.58, 0.67, 0.76, 0.61 ]
    seeings =   [ 0.61, 0.69, 0.79, 0.63 ]

    # 1168106 | 10AQ01 Feb 21 04:21:30 10 | P03 NGVS+2-1   | 12:39:58.2 11:03:49 2000 |  582 | u | 1.06 | 0.76 0.79   180 |P 1 V D|
    # 1080946 | 09AQ09 May 24 23:36:20 09 | P03 NGVS+2-1   | 12:39:55.2 11:02:49 2000 |  634 | g | 1.28 | 0.67 0.69  1191 |P 1 V Q|
    # 1617879 | 13AQ05 Apr 18 22:43:46 13 | P03 NGVS+2-1   | 12:39:57.2 11:01:49 2000 |  445 | r | 1.02 | 0.61 0.63  2240 |P 1 V D|
    # 1080306 | 09AQ09 May 19 21:35:36 09 | P03 NGVS+2-1   | 12:39:55.2 11:02:49 2000 |  411 | i | 1.01 | 0.58 0.61  1923 |P 1 V Q|

    imfns = [ 'cfht/%ip.fits' % i for i in expnums ]
    T = fits_table()
    T.cpimage = []
    T.cpimage_hdu = []
    T.filter = []
    T.exptime = []
    T.ra = []
    T.dec = []
    T.width = []
    T.height = []
    T.expnum = []
    T.extname = []
    T.calname = []
    T.seeing = []
    T.pixscale = []
    T.crval1 = []
    T.crval2 = []
    T.crpix1 = []
    T.crpix2 = []
    T.cd1_1 = []
    T.cd1_2 = []
    T.cd2_1 = []
    T.cd2_2 = []
    for i,(fn,expnum,seeing) in enumerate(zip(imfns, expnums, seeings)):
        F = fitsio.FITS(fn)
        primhdr = F[0].read_header()
        filter = primhdr['FILTER'].split('.')[0]
        exptime = primhdr['EXPTIME']
        pixscale = primhdr['PIXSCAL1'] / 3600.
        print 'Pixscale:', pixscale * 3600, 'arcsec/pix'

        for hdu in range(1, len(F)):
        #for hdu in [13]:
            hdr = F[hdu].read_header()
            T.cpimage.append(fn)
            T.cpimage_hdu.append(hdu)
            T.filter.append(filter)
            T.exptime.append(exptime)
            args = []
            for k in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2',
                      'CD2_1', 'CD2_2' ]:
                val = hdr[k]
                T.get(k.lower()).append(val)
                args.append(val)
            wcs = Tan(*(args + [hdr[k] for k in ['NAXIS1', 'NAXIS2']]))
            print 'WCS pixscale', wcs.pixel_scale()
            W,H = wcs.get_width(), wcs.get_height()
            ra,dec = wcs.radec_center()
            T.ra.append(ra)
            T.dec.append(dec)
            T.width.append(W)
            T.height.append(H)
            T.seeing.append(seeing)
            T.expnum.append(expnum)
            extname = hdr['EXTNAME']
            T.extname.append(extname)
            T.calname.append('cfht/%i/cfht-%i-%s' % (expnum, expnum, extname))
            T.pixscale.append(pixscale)
    T._length = len(T.cpimage)
    T.to_np_arrays()
    return T
    



class CfhtImage(DecamImage):
    def run_calibs(self, ra, dec, pixscale, W=2112, H=4644, se=True,
                   astrom=True, psfex=True, psfexfit=True):
        '''
        pixscale: in degrees/pixel
        '''
        for fn in [self.wcsfn,self.sefn,self.psffn,self.psffitfn]:
            print 'exists?', os.path.exists(fn), fn
        self.makedirs()
    
        run_fcopy = False
        run_se = False
        run_astrom = False
        run_psfex = False
        run_psfexfit = False

        sedir = 'NGVS-g-Single'
    
        if not all([os.path.exists(fn) for fn in [self.sefn]]):
            run_se = True
            run_fcopy = True
        if not all([os.path.exists(fn) for fn in [self.wcsfn,self.corrfn,self.sdssfn]]):
            run_astrom = True
        if not os.path.exists(self.psffn):
            run_psfex = True
        if not os.path.exists(self.psffitfn):
            run_psfexfit = True

        if run_fcopy and (run_se and se):
            tmpimgfn  = create_temp(suffix='.fits')
            cmd = 'imcopy %s"+%i" %s' % (self.imgfn, self.hdu, tmpimgfn)
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
        if run_astrom or run_se:
            # grab header values...
            primhdr = self.read_image_primary_header()
            hdr     = self.read_image_header()
            magzp = hdr['PHOT_C'] + 2.5 * np.log10(hdr['EXPTIME'])
            seeing = self.seeing
            print 'Seeing', seeing, 'arcsec'
    
        if run_se and se:
            #'-SEEING_FWHM %f' % seeing,
            #'-PIXEL_SCALE 0',
            #'-PIXEL_SCALE %f' % (pixscale * 3600),
            #'-MAG_ZEROPOINT %f' % magzp,
            cmd = ' '.join([
                'sex',
                '-c', os.path.join(sedir, 'psfex.sex'),
                '-PARAMETERS_NAME', os.path.join(sedir, 'psfex.param'),
                '-FILTER_NAME', os.path.join(sedir, 'default.conv'),
                '-CATALOG_NAME', self.sefn,
                tmpimgfn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_astrom and astrom:
            cmd = ' '.join([
                'solve-field --config', an_config, '-D . --temp-dir', tempdir,
                '--ra %f --dec %f' % (ra,dec), '--radius 1',
                '-L %f -H %f -u app' % (0.9 * pixscale * 3600, 1.1 * pixscale * 3600),
                '--continue --no-plots --no-remove-lines --uniformize 0',
                '--no-fits2fits',
                '-X x_image -Y y_image -s flux_auto --extension 2',
                '--width %i --height %i' % (W,H),
                '--crpix-center',
                '-N none -U none -S none -M none --rdls', self.sdssfn,
                '--corr', self.corrfn, '--wcs', self.wcsfn, 
                '--temp-axy', '--tag-all', self.sefn])
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_psfex and psfex:
            cmd = ('psfex -c %s -PSF_DIR %s %s -NTHREADS 1' %
                   (os.path.join(sedir, 'gradmap.psfex'),
                    os.path.dirname(self.psffn), self.sefn))
            print cmd
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
    
        if run_psfexfit and psfexfit:
            print 'Fit PSF...'
    
            from tractor.basics import *
            from tractor.psfex import *
    
            iminfo = self.get_image_info()
            #print 'img:', iminfo
            H,W = iminfo['dims']
            psfex = PsfEx(self.psffn, W, H, ny=13, nx=7,
                          psfClass=GaussianMixtureEllipsePSF)
            psfex.savesplinedata = True
            print 'Fitting MoG model to PsfEx'
            psfex._fitParamGrid(damp=1)
            pp,XX,YY = psfex.splinedata
    
            # Convert to GaussianMixturePSF
            ppvar = np.zeros_like(pp)
            for iy in range(psfex.ny):
                for ix in range(psfex.nx):
                    psf = GaussianMixtureEllipsePSF(*pp[iy, ix, :])
                    mog = psf.toMog()
                    ppvar[iy,ix,:] = mog.getParams()
            psfexvar = PsfEx(self.psffn, W, H, ny=psfex.ny, nx=psfex.nx,
                             psfClass=GaussianMixturePSF)
            psfexvar.splinedata = (ppvar, XX, YY)
            psfexvar.toFits(self.psffitfn, merge=True)
            print 'Wrote', self.psffitfn
            
    




if __name__ == '__main__':
    main()
    
