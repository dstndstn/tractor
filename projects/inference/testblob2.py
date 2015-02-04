import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

import fitsio

from astrometry.util.fits import *
from astrometry.util.util import *
from astrometry.util.plotutils import *
from astrometry.util.resample import *
from astrometry.libkd.spherematch import *
from astrometry.sdss import *

from astrometry.sdss import *

from tractor import *
from tractor.sdss import *

tempdir = 'scratch/blobs'

def _bounce_one_blob(args):
    try:
        oneblob(*args)
    except:
        print 'Error running oneblob:'
        import traceback
        traceback.print_exc()
        print

def main():

    cutToPrimary = True

    if False:
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

        # reformat
        stars = [(ra,dec,[('T_EFF',teff,'Effective temperature'),
                          ('DT_EFF',dteff,'Effective temperate error')],
                  cutToPrimary) for teff,dteff,ra,dec in stars]
        
    elif False:

        sdss = DR9(basedir=tempdir)
        sdss.useLocalTree()
        # near M87 / Virgo cluster
        run,camcol,field = 3836,2,258
        pofn = sdss.retrieve('photoObj', run, camcol, field)
        T = fits_table(pofn, columns=[
            'parent', 'objid', 'ra', 'dec', 'fracdev', 'objc_type', 'modelflux',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr', 'phi_dev_deg',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr', 'phi_exp_deg',
            'resolve_status', 'nchild', 'flags', 'objc_flags',
            'run','camcol','field','id'])
        print len(T), 'objects'
        T.cut(T.objc_type == 3)
        print len(T), 'galaxies'
        T.cut(T.nchild == 0)
        print len(T), 'children'
        T.cut(np.argsort(-T.modelflux[:,2]))

        # keep only one child in each blend family
        parents = set()
        keepi = []
        for i in range(len(T)):
            if T.parent[i] in parents:
                continue
            keepi.append(i)
            parents.add(T.parent[i])
        T.cut(np.array(keepi))
        print len(T), 'unique blend families'
        T = T[:25]

        stars = [(ra,dec,[],cutToPrimary) for ra,dec in zip(T.ra, T.dec)]

    else:

        # Objects in the Stripe82 coadd context in CAS:
        #   select * into mydb.s82 from PhotoPrimary where
        #    ra between 5 and 6
        #      and dec between 0 and 1
        #        and (run = 106 or run = 206)  -> s82.fits

        T = fits_table('s82.fits')
        print 'Read', len(T), 'objects'
        T.cut(T.nchild == 0)
        print len(T), 'children'
        T.cut(T.insidemask == 0)
        print len(T), 'not in mask'

        #T.cut(np.hypot(T.ra - 5.0562, T.dec - 0.0643) < 0.001)
        
        # http://skyserver.sdss.org/dr12/en/help/browser/browser.aspx#&&history=enum+PhotoFlags+E
        for flagname,flagval in [('BRIGHT', 0x2),
                                 ('EDGE', 0x4),
                                 ('NODEBLEND', 0x40),
                                 ('DEBLEND_TOO_MANY_PEAKS' , 0x800),
                                 ('NOTCHECKED', 0x80000),
                                 ('TOO_LARGE', 0x1000000),
                                 ('BINNED2', 0x20000000),
                                 ('BINNED4', 0x40000000),
                                 ('SATUR_CENTER', 0x80000000000),
                                 ('INTERP_CENTER', 0x100000000000),
                                 ('MAYBE_CR', 0x100000000000000),
                                 ('MAYBE_EGHOST', 0x200000000000000),
                      ]:
            T.cut(T.flags & flagval == 0)
            print len(T), 'without', flagname, 'bit set'
            #pass
        
        # Cut to objects that are likely to appear in the individual images
        T.cut(T.psfmag_r < 22.)
        print 'Cut to', len(T), 'with psfmag_r < 22 in coadd'

        # Select "interesting" objects...
        # for i in range(len(T)):
        # 
        #     t = T[i]
        #     ra,dec = t.ra, t.dec
        # 
        #     radius = 2. * 0.396 / 3600.
        #     ddec = radius
        #     dra = radius / np.cos(np.deg2rad(dec))
        #     r0,r1 = ra - dra, ra + dra
        #     d0,d1 = dec - ddec, dec + ddec
        #     
        #     wlistfn = sdss.filenames.get('window_flist', 'window_flist.fits')
        #     RCF = radec_to_sdss_rcf(ra, dec, tablefn=wlistfn)
        #     print 'Found', len(RCF), 'fields in range.'
        #     keepRCF = []
        #     for run,camcol,field,r,d in RCF:
        #         rr = sdss.get_rerun(run, field)
        #         print 'Rerun:', rr
        #         if rr == '157':
        #             continue
        #         keepRCF.append((run,camcol,field))
        #     RCF = keepRCF
        #     for ifield,(run,camcol,field) in enumerate(RCF):
        #         objfn = sdss.getPath('photoObj', run, camcol, field)
        #         objs = fits_table(objfn)
        #         objs.cut((objs.ra  > r0) * (objs.ra  < r1) *
        #                  (objs.dec > d0) * (objs.dec < d1))
        #         bright = photo_flags1_map.get('BRIGHT')
        #         objs.cut((objs.nchild == 0) * ((objs.objc_flags & bright) == 0))

        # Write out Stripe82 measurements...
        pixscale = 0.396
        pixradius = 25
        radius = np.sqrt(2.) * pixradius * pixscale / 3600.

        Nkeep = 1000

        outdir = 'stamps'

        T.tag = np.array(['%.4f-%.4f.fits' % (r,d) for r,d in zip(T.ra, T.dec)])
        T[:Nkeep].writeto(os.path.join(outdir, 'stamps.fits'), columns=
                          '''tag objid run camcol field ra dec psfmag_u psfmag_g psfmag_r
                          psfmag_i psfmag_z modelmag_u modelmag_g modelmag_r
                          modelmag_i modelmag_z'''.split())

        for i in range(len(T[:Nkeep])):

            # t = T[np.array([i])]
            # print 't:', t
            # t.about()
            # assert(len(t) == 1)

            I,J,d = match_radec(np.array([T.ra[i]]), np.array([T.dec[i]]),
                                T.ra, T.dec, radius)
            print len(J), 'matched within', radius*3600., 'arcsec'
            t = T[J]
            print len(t), 'matched within', radius*3600., 'arcsec'
            
            tt = fits_table()            
            cols = ['ra','dec','run','camcol','field',#'probpsf',
                    #'flags', #'type',
                    'fracdev_r', #'probpsf_r', 
                    'devrad_r','devraderr_r', 'devab_r', 'devaberr_r',
                    'devphi_r', 'devphierr_r',
                    'exprad_r','expraderr_r', 'expab_r', 'expaberr_r',
                    'expphi_r', 'expphierr_r',
                    ]
            for c in cols:
                cout = c
                # drop "_r" from dev/exp shapes
                if cout.endswith('_r'):
                    cout = cout[:-2]

                coutmap = dict(devrad='theta_dev',
                               devphi='phi_dev',
                               devab ='ab_dev',
                               devraderr='theta_dev_err',
                               devphierr='phi_dev_err',
                               devaberr ='ab_dev_err',
                               exprad='theta_exp',
                               expphi='phi_exp',
                               expab ='ab_exp',
                               expraderr='theta_exp_err',
                               expphierr='phi_exp_err',
                               expaberr ='ab_exp_err',
                               fracdev='frac_dev')
                cout = coutmap.get(cout, cout)
                    
                tt.set(cout, t.get(c))

            tt.is_star = (t.type == 6)

            for magname in ['psf', 'dev', 'exp']:
                for band in 'ugriz':
                    mag = t.get('%smag_%s' % (magname, band))
                    magerr = t.get('%smagerr_%s' % (magname, band))

                    ### FIXME -- arcsinh mags??
                    
                    flux = NanoMaggies.magToNanomaggies(mag)
                    dflux = np.abs(flux * np.log(10.)/-2.5 * magerr)

                    tt.set('%sflux_%s' % (magname, band), flux)
                    tt.set('%sfluxerr_%s' % (magname, band), dflux)

            for band in 'ugriz':
                # http://www.sdss3.org/dr10/algorithms/magnitudes.php#cmodel
                fexp = tt.get('expflux_%s' % band)
                fdev = tt.get('expflux_%s' % band)
                fracdev = t.get('fracdev_%s' % band)
                tt.set('cmodelflux_%s' % band, fracdev * fdev + (1.-fracdev) * fexp)

            catfn = os.path.join(outdir, 'cat-s82-%.4f-%.4f.fits' % (t.ra[0], t.dec[0]))
            tt.writeto(catfn)
            print 'Wrote', catfn
            
        cutToPrimary = False

        ### HACK -- move old files into place.
        for ra,dec in zip(T.ra, T.dec)[:Nkeep]:
            plotfn = 'stamps-%.4f-%.4f.png' % (ra, dec)
            if os.path.exists(plotfn):
                fns = [plotfn]
                for band in 'ugriz':
                    stampfn = 'stamp-%s-%.4f-%.4f.fits' % (band, ra, dec)
                    fns.append(stampfn)
                catfn = 'cat-%.4f-%.4f.fits' % (ra,dec)
                fns.append(catfn)
                
                for fn in fns:
                    cmd = 'mv %s %s' % (fn, outdir)
                    print cmd
                    os.system(cmd)

        stars = [(ra,dec,[],cutToPrimary,outdir) for ra,dec in zip(T.ra, T.dec)[:Nkeep]]

    plots = True
    
    if True:
        from astrometry.util.multiproc import *
        mp = multiproc(8)
        #mp = multiproc()
        mp.map(_bounce_one_blob, stars)

    else:

        # stars = [ (0.,0., 131.59054,  0.66408610),
        #           (0.,0., 147.34576,  0.51657783 ),
        #           ]
        
        for args in stars:
            ra,dec = stars[:2]
            try:
                fns = oneblob(*stars)
            except:
                import traceback
                traceback.print_exc()
                continue
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
                    F.close()
                    del F
                    
                
            
def oneblob(ra, dec, addToHeader, cutToPrimary, outdir):

    plotfn = os.path.join(outdir, 'stamps-%.4f-%.4f.png' % (ra, dec))
    if os.path.exists(plotfn):
        print 'Exists:', plotfn
        return []

    outfns = []
    
    # Resample test blobs to a common pixel grid.
    sdss = DR9(basedir=tempdir)
    #sdss.useLocalTree()
    sdss.saveUnzippedFiles(tempdir)
    
    pixscale = 0.396
    pixradius = 25
    bands = 'ugriz'

    stamp_pattern = os.path.join(outdir, 'stamp-%%s-%.4f-%.4f.fits' % (ra, dec))
    catfn = os.path.join(outdir, 'cat-%.4f-%.4f.fits' % (ra,dec))

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

    if len(RCF) == 0:
        print 'No run/camcol/fields in rerun 301'
        return
        
    TT = []

    for ifield,(run,camcol,field) in enumerate(RCF):

        # Retrieve SDSS catalog sources in the field
        srcs,objs = get_tractor_sources_dr9(run, camcol, field, bandname=srcband,
                                            sdss=sdss,
                                            radecrad=(ra, dec, radius*np.sqrt(2.)),
                                            nanomaggies=True,
                                            cutToPrimary=cutToPrimary,
                                            getsourceobjs=True,
                                            useObjcType=True)
        print 'Got sources:'
        for src in srcs:
            print '  ', src

        # Write out the sources
        T = fits_table()
        T.ra  = [src.getPosition().ra  for src in srcs]
        T.dec = [src.getPosition().dec for src in srcs]

        # for band in bands:
        #     T.set('psfflux_%s' % band,
        #           [src.getBrightness().getBand(band) for src in srcs])

        # same objects, same order
        assert(len(objs) == len(srcs))
        assert(np.all(T.ra == objs.ra))

        # r-band
        bandnum = 2
        T.primary = ((objs.resolve_status & 256) > 0)
        T.run = objs.run
        T.camcol = objs.camcol
        T.field = objs.field
        T.is_star = (objs.objc_type == 6)
        T.frac_dev = objs.fracdev[:,bandnum]
        T.theta_dev = objs.theta_dev[:,bandnum]
        T.theta_exp = objs.theta_exp[:,bandnum]
        T.phi_dev = objs.phi_dev_deg[:,bandnum]
        T.phi_exp = objs.phi_exp_deg[:,bandnum]
        T.ab_dev = objs.ab_dev[:,bandnum]
        T.ab_exp = objs.ab_exp[:,bandnum]

        for band in bands:
            bi = band_index(band)
            T.set('psfflux_%s' % band, objs.psfflux[:,bi])
            T.set('devflux_%s' % band, objs.devflux[:,bi])
            T.set('expflux_%s' % band, objs.expflux[:,bi])
            T.set('cmodelflux_%s' % band, objs.cmodelflux[:,bi])
            
        TT.append(T)
    T = merge_tables(TT)
    T.writeto(catfn)
    outfns.append(catfn)

    written = set()

    plt.figure(figsize=(8,8))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99,
                        hspace=0.05, wspace=0.05)
            
    # Retrieve SDSS images
    for band in bands:
        if band == 'r':
            rimgs = []

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
            #print 'Tim CD matrix', cd
            targetcd = np.array(targetwcs.cd).copy().reshape((2,2))
            #print 'Target CD matrix:', targetcd

            trans = np.dot(np.linalg.inv(targetcd), cd)
            #print 'Transformation matrix:', trans

            psf = tim.getPsf()
            #print 'PSF', psf
            K = psf.mog.K
            newmean = np.zeros_like(psf.mog.mean)
            #print 'newmean', newmean
            newvar = np.zeros_like(psf.mog.var)
            #print 'newvar', newvar

            for i,(dx,dy) in enumerate(psf.mog.mean):
                #print 'dx,dy', dx,dy
                x,y = tim.getWcs().positionToPixel(RaDecPos(ra, dec))
                r,d = tim.getWcs().pixelToPosition(x + dx, y + dy)
                #print 'ra,dec', r,d
                ok,x0,y0 = targetwcs.radec2pixelxy(ra, dec)
                ok,x1,y1 = targetwcs.radec2pixelxy(r, d)
                #print 'dx2,dy2', x1-x0, y1-y0
                vv = np.array([dx,dy])
                tv = np.dot(trans, vv)
                #print 'dot', tv
                newmean[i,:] = tv
                
            for i,var in enumerate(psf.mog.var):
                #print 'var', var
                newvar[i,:,:] = np.dot(trans, np.dot(var, trans.T))
                #print 'newvar', newvar[i,:,:]

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

            for (key, value, comment) in addToHeader:
                hdr.add_record(dict(name=key, value=value, comment=comment))
                
            newpsf.toFitsHeader(hdr, 'PSF_')
            
            # First time, overwrite existing file.  Later, append
            clobber = not band in written
            written.add(band)

            if band == 'r':
                rimgs.append(img)
            
            fn = stamp_pattern % band
            print 'writing', fn
            fitsio.write(fn, img.astype(np.float32), clobber=clobber, header=hdr)
            fitsio.write(fn, iv.astype(np.float32))
            if clobber:
                outfns.append(fn)

        if band == 'r':
            N = len(rimgs)
            ncols = int(np.ceil(np.sqrt(float(N))))
            nrows = int(np.ceil(float(N) / ncols))
            plt.clf()
            for k,img in enumerate(rimgs):
                plt.subplot(nrows, ncols, k+1)
                dimshow(img, vmin=-0.1, vmax=1., ticks=False)
            plt.savefig('stamps-%.4f-%.4f.png' % (ra, dec))
                
    return outfns
                
if __name__ == '__main__':
    main()
    
