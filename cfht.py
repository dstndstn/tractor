import os
import tempfile
import tractor
import pyfits
import numpy as np

def read_cfht_coadd(imgfn, weightfn, roi=None, radecroi=None,
                    filtermap=None):
    '''
    Given filenames for CFHT coadd image and weight files, produce
    a tractor.Image object.

    *roi*: (x0,x1, y0,y1): a region-of-interest in pixel space;
           returns the subimage [x0,x1), [y0,y1).
    *radecroi*: (ra0, ra1, dec0, dec1): a region-of-interest in RA,Dec space;
           returns the subimage bounding the given RA,Dec box [ra0,ra1], [dec0,dec1].
    *filtermap*: dict, eg,  { 'i.MP9701': 'i' }, to map from the FILTER header keyword to
           a standard filter name.
    '''

    P = pyfits.open(imgfn)
    print 'Read', P[0].data.shape, 'image'
    img = P[0].data
    imgheader = P[0].header

    # WCS: the image file has a WCS header
    # we should be able to do:
    #twcs = tractor.FitsWcs(imgfn)
    # ARGH!  Memory issues reading the file; HACK: copy header...
    f,tempfn = tempfile.mkstemp()
    os.close(f)
    pyfits.writeto(tempfn, None, header=imgheader, clobber=True)
    twcs = tractor.FitsWcs(tempfn)

    # Cut down to the region-of-interest, if given.
    if roi is not None:
        x0,x1,y0,y1 = roi
    elif radecroi is not None:
        ralo,rahi, declo,dechi = radecroi
        xy = [twcs.positionToPixel(tractor.RaDecPos(r,d))
              for r,d in [(ralo,declo), (ralo,dechi), (rahi,declo), (rahi,dechi)]]
        xy = np.array(xy)
        x0,x1 = xy[:,0].min(), xy[:,0].max()
        y0,y1 = xy[:,1].min(), xy[:,1].max()
        print 'RA,Dec ROI', ralo,rahi, declo,dechi, 'becomes x,y ROI', x0,x1,y0,y1

        # Clip to image size...
        H,W = data.shape
        x0 = max(0, min(x0, W-1))
        x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H-1))
        y1 = max(0, min(y1, H))
        print ' clipped to', x0,x1,y0,y1

    else:
        H,W = img.shape
        x0,x1,y0,y1 = 0,W, 0,H

    if roi is not None or radecroi is not None:
        # Actually cut the pixels
        img = img[y0:y1, x0:x1].copy()
        # Also tell the WCS to apply an offset.
        twcs.setX0Y0(x0,y0)

    print 'Image:', img.shape

    # HACK, tell the WCS how big the image is...
    # (needed because of the previous HACK, copying the header)
    twcs.wcs.set_imagesize(x1-x0, y1-y0)
    print twcs

    # Argh, this doesn't work: the files are .fz compressed
    #P = pyfits.open(weightfn)
    #weight = P[1].data[y0:y1, x0:x1]
    # HACK: use "imcopy" to uncompress to a temp file!
    #print 'Writing to temp file', tempfn
    cmd = "imcopy '%s[%i:%i,%i:%i]' '!%s'" % (weightfn, x0+1,x1,y0+1,y1, tempfn)
    print 'running', cmd
    os.system(cmd)
    P = pyfits.open(tempfn)
    weight = P[0].data
    print 'Read', weight.shape, 'weight image'

    # PSF model: FAKE IT for now
    tpsf = tractor.GaussianMixturePSF(np.array([0.9, 0.1]), np.zeros((2,2)), np.array([1,2]))

    # SKY level: assume zero
    #sky = np.median(img)
    #print 'Image median value:', sky
    sky = 0.
    tsky = tractor.ConstantSky(sky)

    # Photometric calibration: the FITS header says:
    '''
    FILTER  = 'r.MP9601'           / Filter
    PHOTZP  =               30.000 / photometric zeropoint
    COMMENT AB magnitude = -2.5 * log10(flux) + PHOTZP
    COMMENT r.MP9601=r_SDSS-0.024*(g_SDSS-r_SDSS)
    '''
    # Grab the filter name, and apply the filtermap (if given)
    filter = imgheader['FILTER']
    if filtermap:
        filter = filtermap.get(filter, filter)
    zp = imgheader['PHOTZP']
    # Simple photocal object
    photocal = tractor.MagsPhotoCal(filter, zp)

    # For plotting: find the approximate standard deviation
    #print 'Median weight:', np.median(weight)
    sigma1 = 1./np.sqrt(np.median(weight))
    zr = np.array([-3,10]) * sigma1 + sky

    name = 'CFHT ' + imgheader.get('OBJECT', '')

    tim = tractor.Image(data=img, invvar=weight, psf=tpsf, wcs=twcs,
                        sky=tsky, photocal=photocal, name=name, zr=zr)
    tim.extent = [x0,x1,y0,y1]
    return tim




if __name__ == '__main__':
    from bigboss_test import *

    basedir = '/project/projectdirs/bigboss/data/cfht/w4/megapipe'
    field = 'W4+1-1.I'
    #roi = (10000,10500,10000,10500)
    roi = None
    #radecroi = (334.3, 334.4, 0.3, 0.4)
    filtermap = { 'i.MP9701': 'i' }
    
    im = read_cfht_coadd(os.path.join(basedir, field + '.fits'),
                         os.path.join(basedir, field + '.weight.fits.fz'),
                         roi=roi, radecroi=radecroi, filtermap=filtermap)

    # Load a catalog, render a model image and make some plots.

    import matplotlib
    matplotlib.use('Agg')

    make_plots('cfht-', im)

