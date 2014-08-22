import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
from glob import glob
import fitsio

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.resample import *
from astrometry.util.starutil_numpy import *

'''
imcopy /global/homes/d/dstn/cosmo/data/staging/decam/CP20140815/c4d_140816_001323_ooi_g_v1.fits.fz+27 1.fits

solve-field --config ~/desi-dstn/sdss-astrometry-index/r2/cfg -v -D . --temp-dir tmp --ra 244 --dec 8 --radius 1 --continue --no-plots --sextractor-config /project/projectdirs/desi/imaging/code/cats/CS82.sex 1.fits -X x_image -Y y_image -s flux_auto

funpack -E 27 -O flag.fits /global/homes/d/dstn/cosmo/data/staging/decam/CP20140815/c4d_140816_001323_ood_g_v1.fits.fz

an-fitstopnm -i 1.fits | pnmscale -reduce 4 | pnmtojpeg > 1-4.jpg

solve-field --config /data/INDEXES/sdss-astrometry-index/r2/cfg -v -D . --ra 244 --dec 8 --radius 1 --continue -X x_image -Y y_image -s flux_auto 1.axy --plot-bg 1.jpg --plot-scale 0.25

'''

def plot_coadd(coadd, cowt, tt, ps, mnmx=None):
    if sum(cowt > 0) == 0:
        print 'Zero pixels with weight > 0 for', tt
        return
    
    print 'Max weight:', np.max(cowt)
    co = coadd / np.maximum(1e-16, cowt)
    plt.clf()
    if mnmx is None:
        mn,mx = [np.percentile(co[cowt > 0], p) for p in [25,99]]
    else:
        mn,mx = mnmx

    h,w = co.shape
    rgb = np.zeros((h,w,3), np.float32)

    for gray in [np.clip((co - mn) / (mx - mn), 0, 1),
                 np.sqrt(np.clip((co - mn) / (mx - mn), 0, 1)),
                 ]:
        rgb[:,:,0] = rgb[:,:,1] = rgb[:,:,2] = gray
        for i,cc in enumerate([0.3,0.3,0.7]):
            rgb[:,:,i][cowt == 0] = cc

        # plt.imshow(co, interpolation='nearest', origin='lower',
        #           cmap='gray', vmin=mn, vmax=mx)
        plt.imshow(rgb, interpolation='nearest', origin='lower')
        plt.title(tt)
        ps.savefig()


if __name__ == '__main__':

    B = fits_table('bricks.fits')
    B.index = np.arange(len(B))

    ii = 377305

    ra,dec = B.ra[ii], B.dec[ii]
    W,H = 3600,3600
    pixscale = 0.27 / 3600.

    plt.figure(figsize=(W/100,H/100))
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    ps = PlotSequence('decals2')

    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
                    -pixscale, 0., 0., pixscale,
                    float(W), float(H))

    nearest = True

    coims = []

    T = fits_table('ccds.fits')
    sz = 0.25
    T.cut(np.abs(T.dec - dec) < sz)
    T.cut(degrees_between(T.ra, T.dec, ra, dec) < sz)
    print len(T), 'CCDs nearby'

    # ranges = {'g': (96.2022476196, 207.536163788),
    #           'r': (272.476150513, 592.256534424),
    #           'z': (3364.984375, 4438.97126221),
    #           }
    ranges = {'g': (-7.45025110245, 105.060557976),
              'r': (-7.55701327324, 315.790451965),
              'z': (-155.374023438, 921.245625),
              }
    
    for band in 'grz':
        TT = T[T.filter == band]
        print len(TT), 'in', band, 'band'
        print 'filenames,hdus:', zip(TT.filename, TT.hdu)

        coimg = np.zeros((H,W), np.float32)
        cowt  = np.zeros((H,W), np.float32)
        coimgm = np.zeros((H,W), np.float32)
        cowtm  = np.zeros((H,W), np.float32)

        resam = np.zeros((H,W), np.float32)
        rewt  = np.zeros((H,W), np.float32)

        lastband = band
        lastfn = None
        
        for fn,hdu in zip(TT.filename, TT.hdu):
            print
            imgfn = fn
            print 'Image file', fn, 'hdu', hdu
            #img = fitsio.FITS(fn)[hdu].read()
            img,imghdr = fitsio.read(fn, ext=hdu, header=True)

            sky = imghdr['SKYBRITE']
            print 'SKYBRITE:', sky
            medsky = np.median(img)
            print 'Image median:', medsky
            #img -= sky
            img -= medsky
            print 'Image median:', np.median(img)

            dqfn = fn.replace('_ooi_', '_ood_')
            print 'DQ', dqfn

            wcsfn = imgfn
            wcsfn = wcsfn.replace('/project/projectdirs/cosmo/data/staging/decam',
                                  'calib/astrom')
            wcsfn = wcsfn.replace('.fits.fz', '.ext%02i.wcs' % hdu)

            corrfn = wcsfn.replace('.wcs', '.corr')

            sexfn = imgfn
            sexfn = sexfn.replace('/project/projectdirs/cosmo/data/staging/decam',
                                  'calib/sextractor')
            sexfn = sexfn.replace('.fits.fz', '.ext%02i.fits' % hdu)

            for dirnm in [os.path.dirname(fn) for fn in [wcsfn,corrfn,sexfn]]:
                if not os.path.exists(dirnm):
                    try:
                        os.makedirs(dirnm)
                    except:
                        pass

            print 'WCS filename', wcsfn
            print 'Corr', corrfn
            print 'SExtractor', sexfn

            if not os.path.exists(wcsfn):
                #
                cmd = 'rm -f 1.fits flags.fits'
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

                tmpimfn = '1.fits'
                cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimfn, imgfn)
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

                tmpmaskfn = 'flags.fits'
                cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, dqfn)
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

                cmd = ('solve-field --config ~/desi-dstn/sdss-astrometry-index/r2/cfg '
                       + '-D . --temp-dir tmp --ra 244 --dec 8 --radius 1 '
                       + '--continue --no-plots '
                       + '--sextractor-config /project/projectdirs/desi/imaging/code/cats/CS82.sex '
                       + '-X x_image -Y y_image -s flux_auto '
                       + '--crpix-center '
                       + '-N none -U none -S none -M none -R none '
                       + '--keep-xylist ' + sexfn + ' '
                       + '--temp-axy '
                       + '--corr ' + corrfn + ' --tag-all '
                       + '--wcs ' + wcsfn + ' '
                       + '-L 0.25 -H 0.29 -u app '
                       + '--no-remove-lines --uniformize 0 --no-fits2fits '
                       + tmpimfn)
                print cmd
                if os.system(cmd):
                    sys.exit(-1)

            wcs = Sip(wcsfn)

            dq = fitsio.FITS(dqfn)[hdu].read()

            #print 'DQ', dq.shape, dq.dtype
            #print len(dq.ravel()), 'DQ pixels'
            #print 'Unique vals:', np.unique(dq)
            #print sum(dq == 0), 'have value 0'
            
            wtfn = imgfn.replace('_ooi_', '_oow_')
            print 'Weight', wtfn
            wt = fitsio.FITS(wtfn)[hdu].read()
            #print 'WT', wt.shape, wt.dtype
            
            if False:
                # Nugent's bad pixel masks
                bpfn = fn.replace('.fits', '.bpm.fits')
                print 'Bad pixel mask', bpfn
                mask = fitsio.read(bpfn)
                mask = (mask == 0)
            
            L = 2
            try:
                if nearest:
                    lims = []
                else:
                    lims = [img]
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(targetwcs, wcs, lims, L)
                print 'Resampled', len(Yo), 'pixels'
            except OverlapError:
                print 'No overlap'
                continue

            print
            print 'Filename     ', imgfn
            print 'Last filename', lastfn

            clear = False
            if lastfn != imgfn or lastband != band:
                clear = True
            if lastfn is None:
                pass
            elif lastfn != imgfn or lastband != band:
                print 'Starting new file -- plotting last file'
                tt = '%s band, file %s' % (lastband, os.path.basename(lastfn).replace('.fits.fz',''))
                plot_coadd(resam, rewt, tt, ps, mnmx=ranges.get(lastband,None))
            if clear:
                print 'Clearing resams'
                resam[:,:,] = 0.
                rewt [:,:,] = 0.
                lastfn = imgfn
                lastband = band
            
            rweight = wt[Yi,Xi] * (dq[Yi,Xi] == 0)
            if nearest:
                rim = img[Yi,Xi]
            else:
                rim = rims[0]

            coimg [Yo,Xo] += rim
            coimgm[Yo,Xo] += rim * rweight
            cowt  [Yo,Xo] += 1.
            cowtm [Yo,Xo] += rweight

            resam [Yo,Xo] += rim * rweight
            rewt  [Yo,Xo] += rweight

        tt = '%s band, file %s' % (lastband, os.path.basename(lastfn).replace('.fits.fz',''))
        plot_coadd(resam, rewt, tt, ps, mnmx=ranges.get(lastband,None))
            
        coadd = coimg / np.maximum(1e-16, cowt)
        coaddm = coimgm / np.maximum(1e-16, cowtm)

        fn = 'coadd-%s.fits' % band
        fitsio.write(fn, coaddm, clobber=True)
        fitsio.write(fn, cowtm)

        mn,mx = [np.percentile(coadd[cowt > 0], p) for p in [25,99.5]]

        print 'Band', band
        print 'min', mn
        print 'max', mx

        tt = '%s band stack' % band
        plot_coadd(coimg, cowt, tt, ps, mnmx=(mn,mx))

        tt = '%s band stack, masked' % band
        plot_coadd(coimgm, cowtm, tt, ps, mnmx=(mn,mx))

        coims.append((coadd,  cowt,  mn, mx))
        coims.append((coaddm, cowtm, mn, mx))


    rgb = np.zeros((H,W,3), np.float32)

    for rgbinds in [[4,2,0],
                    [5,3,1],]:
        rgbims = [coims[i] for i in rgbinds]
        for plane,(im,wt,mn,mx) in enumerate(rgbims):
            rgb[:,:,plane] = (im - mn) / (mx - mn)
            rgb[:,:,plane][wt == 0] = 0.5

        plt.clf()
        plt.imshow(np.clip(rgb, 0.,1.),
                   interpolation='nearest', origin='lower')
        plt.title('zrg stack')
        ps.savefig()

        plt.clf()
        plt.imshow(np.sqrt(np.clip(rgb, 0.,1.)),
                   interpolation='nearest', origin='lower')
        plt.title('zrg stack; masked')
        ps.savefig()

