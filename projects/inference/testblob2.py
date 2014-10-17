import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import fitsio

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.resample import *
from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

def _bounce_one_blob((teff,dteff,ra,dec)):
    try:
        oneblob(ra,dec,teff,dteff)
    except:
        print 'Error running oneblob:'
        import traceback
        traceback.print_exc()
        print
    
def main():
    stars = [
        # David's nearby pairs of F stars
        (3900., 0., 118.37066, 52.527073),
        (3705., 0., 130.17654, 52.750081),
        # High stellar density
        #(0., 0., 270.0, 0.003),
        ]
    # Dustin's stars
    # (4472.001,	0.02514649,	246.47016,	19.066909),
    # (5196.53,   0.02490235, 240.09403,  37.404078),
    # (6179.05,   0.6324392,  310.47791,  57.523221),
    # (6021.875, 0.7000019, 150.52443, -0.478836),
    # (7757.096, 0.06507664, 305.11144, -12.957655),
    # (8088.685, 0.2436366, 253.11475, 11.60716),
    # (8395.096, 0.7563477, 188.34439, 63.442057),
    # (9201.74,  178, 93.971719, 0.56302169),
    # ]

    T = fits_table('stars2.fits')
    print 'Read stars:'
    T.about()
    stars.extend(zip(T.teff, T.teff_sigma, T.ra, T.dec))

    plots = False
    
    if False:
        from astrometry.util.multiproc import *
        mp = multiproc(2)
        mp.map(_bounce_one_blob, stars)

    else:

        # stars = [ (0.,0., 131.59054,  0.66408610),
        #           (0.,0., 147.34576,  0.51657783 ),
        #           ]
        
        for teff, dteff, ra,dec in stars:
            fns = oneblob(ra,dec, teff, dteff)
                
            if plots:
                stamp_pattern = 'stamp-%%s-%.4f-%.4f.fits' % (ra, dec)
                bands = 'ugriz'
                fns = ['cat'] + [stamp_pattern % band for band in bands]
                for j,fn in enumerate(fns[1:]):
                    print 'Filename', fn
                    F = fitsio.FITS(fn)
                    n = len(F) / 2
                    print 'n ext:', n
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / float(cols)))
                    plt.clf()
                    for i,ext in enumerate(range(0, len(F), 2)):
                        plt.subplot(rows, cols, i+1)
                        hdr = F[ext].read_header()
                        dimshow(F[ext].read(), ticks=False)
                        plt.title('RCF %i/%i/%i' % (hdr['RUN'], hdr['CAMCOL'], hdr['FIELD']))
                    plt.suptitle('%s band' % bands[j])
                    plt.savefig(fn.replace('.fits','.png'))

                    
                
            
def oneblob(ra, dec, teff, dteff):

    outfns = []
    
    # Resample test blobs to a common pixel grid.
    sdss = DR9()
    sdss.saveUnzippedFiles('.')
    
    pixscale = 0.396
    pixradius = 25
    bands = 'ugriz'

    stamp_pattern = 'stamp-%%s-%.4f-%.4f.fits' % (ra, dec)
    catfn = 'cat-%.4f-%.4f.fits' % (ra,dec)

    plots = False
    srcband = 'r'
    Lanczos = 3
    
    W,H = pixradius*2+1, pixradius*2+1
    targetwcs = Tan(ra, dec, pixradius+1, pixradius+1,
                    -pixscale/3600., 0., 0., pixscale/3600., W, H)
    radius = pixradius * pixscale / 3600.
    
    wlistfn = sdss.filenames.get('window_flist', 'window_flist.fits')
    #wfn = os.path.join(os.environ['PHOTO_RESOLVE'], 'window_flist.fits')
    RCF = radec_to_sdss_rcf(ra, dec, tablefn=wlistfn)
    print 'Found', len(RCF), 'fields in range.'

    keepRCF = []
    for run,camcol,field,r,d in RCF:
        rr = sdss.get_rerun(run, field)
        print 'Rerun:', rr
        if rr == '157':
            continue
        keepRCF.append((run,camcol,field))
    RCF = keepRCF
        
    TT = []
    
    for ifield,(run,camcol,field) in enumerate(RCF):

        # Retrieve SDSS catalog sources in the field
        srcs = get_tractor_sources_dr9(run, camcol, field, bandname=srcband,
                                       sdss=sdss,
                                       radecrad=(ra, dec, radius*np.sqrt(2.)),
                                       nanomaggies=True,
                                       cutToPrimary=True)
        print 'Got sources:'
        for src in srcs:
            print '  ', src

        # Write out the sources
        T = fits_table()
        T.ra  = [src.getPosition().ra  for src in srcs]
        T.dec = [src.getPosition().dec for src in srcs]
        for band in bands:
            T.set('psfflux_%s' % band,
                  [src.getBrightness().getBand(band) for src in srcs])
        TT.append(T)
    T = merge_tables(TT)
    T.writeto(catfn)
    outfns.append(catfn)

    written = set()
            
    # Retrieve SDSS images
    for band in bands:
        for ifield,(run,camcol,field) in enumerate(RCF):
            fn = sdss.retrieve('photoField', run, camcol, field)
            print 'Retrieved', fn
            F = fits_table(fn)
            F.cut((F.run == run) * (F.camcol == camcol) * (F.field == field))
            print len(F), 'fields'
            assert(len(F) == 1)
            F = F[0]

            boundpixradius = int(np.ceil(np.sqrt(2.) * pixradius))
            print 'RA,Dec,size', (ra, dec, boundpixradius)
            tim,tinfo = get_tractor_image_dr9(
                run, camcol, field, band, sdss=sdss, nanomaggies=True,
                roiradecsize=(ra, dec, boundpixradius))
            
            print 'Got tim:', tim
            frame = sdss.readFrame(run, camcol, field, band)
            if tim is None:
                continue
            
            x,y = tim.getWcs().positionToPixel(RaDecPos(ra, dec))
            x,y = int(x), int(y)
            # Grab calibration information also
            tim.sdss_calib = np.median(frame.getCalibVec())
            tim.sdss_sky = frame.getSkyAt(x,y)
            iband = band_index(band)
            tim.sdss_gain = F.gain[iband]
            tim.sdss_darkvar = F.dark_variance[iband]
        
            #tims.append(tim)
            #tinfs.append(tinfo)
            #tims = []
            #tinfs = []
            # Write out the images
            #for band,tim,tinfo in zip(bands, tims, tinfs):

            roi = tinfo['roi']
            x0,x1,y0,y1 = roi
        
            if plots:
                plt.clf()
                img = tim.getImage()
                mn,mx = [np.percentile(img,p) for p in [25,99]]
                dimshow(img, vmin=mn, vmax=mx)
                xx,yy = [],[]
                for src in srcs:
                    x,y = tim.getWcs().positionToPixel(src.getPosition())
                    xx.append(x)
                    yy.append(y)
                ax = plt.axis()
                plt.plot(xx, yy, 'r+')
                plt.axis(ax)
                plt.savefig('tim-%s%i.png' % (band, ifield))

            # Resample to common grid
            th,tw = tim.shape
            wwcs = TractorWCSWrapper(tim.getWcs(), tw, th)
            try:
                Yo,Xo,Yi,Xi,[rim] = resample_with_wcs(
                    targetwcs, wwcs, [tim.getImage()], Lanczos)
            except OverlapError:
                continue

            img = np.zeros((H,W))
            img[Yo,Xo] = rim
            iv  = np.zeros((H,W))
            iv[Yo,Xo] = tim.getInvvar()[Yi,Xi]

            if plots:
                plt.clf()
                mn,mx = [np.percentile(img,p) for p in [25,99]]
                dimshow(img, vmin=mn, vmax=mx)
                xx,yy = [],[]
                for src in srcs:
                    rd = src.getPosition()
                    ok,x,y = targetwcs.radec2pixelxy(rd.ra, rd.dec)
                    xx.append(x-1)
                    yy.append(y-1)
                ax = plt.axis()
                plt.plot(xx, yy, 'r+')
                plt.axis(ax)
                plt.savefig('rim-%s%i.png' % (band, ifield))

            # Convert PSF params also
            cd = tim.getWcs().cdAtPixel(tw/2, th/2)
            print 'Tim CD matrix', cd
            targetcd = np.array(targetwcs.cd).copy().reshape((2,2))
            print 'Target CD matrix:', targetcd

            trans = np.dot(np.linalg.inv(targetcd), cd)
            print 'Transformation matrix:', trans

            psf = tim.getPsf()
            print 'PSF', psf
            K = psf.mog.K
            newmean = np.zeros_like(psf.mog.mean)
            print 'newmean', newmean
            newvar = np.zeros_like(psf.mog.var)
            print 'newvar', newvar

            for i,(dx,dy) in enumerate(psf.mog.mean):
                print 'dx,dy', dx,dy
                x,y = tim.getWcs().positionToPixel(RaDecPos(ra, dec))
                r,d = tim.getWcs().pixelToPosition(x + dx, y + dy)
                print 'ra,dec', r,d
                ok,x0,y0 = targetwcs.radec2pixelxy(ra, dec)
                ok,x1,y1 = targetwcs.radec2pixelxy(r, d)
                print 'dx2,dy2', x1-x0, y1-y0
                vv = np.array([dx,dy])
                tv = np.dot(trans, vv)
                print 'dot', tv
                newmean[i,:] = tv
                
            for i,var in enumerate(psf.mog.var):
                print 'var', var
                newvar[i,:,:] = np.dot(trans, np.dot(var, trans.T))
                print 'newvar', newvar[i,:,:]

            newpsf = GaussianMixturePSF(psf.mog.amp, newmean, newvar)

            hdr = fitsio.FITSHDR()
            targetwcs.add_to_header(hdr)
            hdr.add_record(dict(name='RUN', value=run, comment='SDSS run'))
            hdr.add_record(dict(name='CAMCOL', value=camcol, comment='SDSS camcol'))
            hdr.add_record(dict(name='FIELD', value=field, comment='SDSS field'))
            hdr.add_record(dict(name='BAND', value=band, comment='SDSS band'))

            # Copy from input "frame" header
            orighdr = tinfo['hdr']
            for key in ['NMGY']:
                hdr.add_record(dict(name=key, value=orighdr[key],
                                    comment=orighdr.get_comment(key)))

            hdr.add_record(dict(name='CALIB', value=tim.sdss_calib,
                                comment='Mean "calibvec" value for this image'))
            hdr.add_record(dict(name='SKY', value=tim.sdss_sky,
                                comment='SDSS sky estimate at image center'))
            hdr.add_record(dict(name='GAIN', value=tim.sdss_gain,
                                comment='SDSS gain'))
            hdr.add_record(dict(name='DARKVAR', value=tim.sdss_darkvar,
                                comment='SDSS dark variance'))

            hdr.add_record(dict(name='T_EFF', value=teff,
                                comment='Effective temperature'))
            hdr.add_record(dict(name='DT_EFF', value=dteff,
                                comment='Effective temperature error'))
            
            newpsf.toFitsHeader(hdr, 'PSF_')
            
            # First time, overwrite existing file.  Later, append
            clobber = not band in written
            written.add(band)
            
            fn = stamp_pattern % band
            fitsio.write(fn, img, clobber=clobber, header=hdr)
            fitsio.write(fn, iv)
            if clobber:
                outfns.append(fn)
    return outfns
                
if __name__ == '__main__':
    main()
    
