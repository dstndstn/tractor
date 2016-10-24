from __future__ import print_function
import pyfits, numpy, os, fnmatch, re, pdb, tractor, tempfile, subprocess
from scipy import optimize
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

#def get_ps1_astrom_coord(header):
#    return none

class wcs_ps1(tractor.BaseParams):
    def __init__(self, smffile, chip, offset=(0, 0)):
        self.smffile = smffile
        smfcoords = readsmfcoords(smffile)
        self.image = smfcoords[chip]
        self.mosaic = smfcoords['PRIMARY']
        self.cd = getcd(smffile, chip)
        self.offset = offset
    
    def positionToPixel(self, pos, src=None):
        #return (pos.ra, pos.dec)
        x, y = rd_to_xy(pos.ra, pos.dec, self.image, self.mosaic)
        return x-self.offset[1], y-self.offset[0]

    def cdAtPixel(self, x, y):
        #return numpy.array([[1., 0.], [0., 1.]])
        return self.cd


def getcd(smffile, chip):
    h = pyfits.getheader(smffile, extname=chip+'.HDR')
    cd = numpy.array([[float(h['CD1_1A']), float(h['CD1_2A'])],
                      [float(h['CD2_1A']), float(h['CD2_2A'])]])
    return cd

def get_ps1_coords(h):
    coord = { }
    grab_coords = ['CDELT1', 'CDELT2', 'CRPIX1', 'CRPIX2',
                   'CRVAL1', 'CRVAL2',
                   'PC001001', 'PC001002', 'PC002001', 'PC002002',
                   'NPLYTERM']
    translate_dict = {'ctype1':'ctype',
                      'pc001001':'pc1_1', 'pc001002':'pc1_2',
                      'pc002001':'pc2_1', 'pc002002':'pc2_2',
                      'nplyterm':'npolyterms'}

    for s in grab_coords:
        translation = translate_dict.get(s.lower(), s.lower())
        coord[translation] = float(h[s])
    coord['ctype'] = h['CTYPE1']
    coord['polyterms'] = numpy.zeros(14)

    polytermnames = ['PCA1X2Y0', 'PCA2X2Y0', 'PCA1X1Y1', 'PCA2X1Y1',
                     'PCA1X0Y2', 'PCA2X0Y2', 'PCA1X3Y0', 'PCA2X3Y0',
                     'PCA1X2Y1', 'PCA2X2Y1', 'PCA1X1Y2', 'PCA2X1Y2',
                     'PCA1X0Y3', 'PCA2X0Y3']
    if coord['npolyterms'] > 1:
        for i in xrange(6):
            coord['polyterms'][i] = float(h[polytermnames[i]])
    if coord['npolyterms'] > 2:
        for i in xrange(8):
            coord['polyterms'][i+6] = float(h[polytermnames[i+6]])
    return coord

def rd_to_xy(r, d, image, mosaic):
    r = numpy.atleast_1d(r)
    d = numpy.atleast_1d(d)
    def chi2(x, fitrd):
        rp, dp = xy_to_rd(x[0], x[1], image, mosaic)
        return numpy.array([rp-fitrd[0], dp-fitrd[1]])
    guess = numpy.array([mosaic['crval1'], mosaic['crval2']])
    outx = numpy.zeros_like(r)
    outy = numpy.zeros_like(r)
    for i in xrange(len(r)):
        fit = optimize.leastsq(chi2, guess, args=[r[i], d[i]])
        outx[i] = fit[0][0]
        outy[i] = fit[0][1]
    if len(r) == 1:
        outx = outx[0]
        outy = outy[0]
    return outx, outy

def xy_to_rd(x, y, image, mosaic, debug=False):
    x = x.astype('f8')
    y = y.astype('f8')
    l, m = xy_to_lm(x, y, image)
    r, d = lm_to_rd(l, m, image)
    warp = (image['ctype'][-3:] == 'WRP')
    if warp:
        dis = (mosaic['ctype'][-3:] == 'DIS')
        if not dis:
            raise ValueError('Must set mosaic with chip astrometry.')
        r, d = xy_to_rd(r, d, mosaic, mosaic, debug=debug)
    if debug:
        print(r, d)
    return r, d

def xy_to_lm(xpix, ypix, image):
    x = image['cdelt1']*(xpix-image['crpix1'])
    y = image['cdelt2']*(ypix-image['crpix2'])
    l = (x*image['pc1_1']+y*image['pc1_2'])
    m = (x*image['pc2_1']+y*image['pc2_2'])
    npoly = image['npolyterms']
    pterms = image['polyterms'].reshape(7, 2)
    # the pterms are 
    if npoly > 1:
        l += x*x*pterms[0,0]+x*y*pterms[1,0]+y*y*pterms[2,0]
        m += x*x*pterms[0,1]+x*y*pterms[1,1]+y*y*pterms[2,1]
    if npoly > 2:
        l += x*x*x*pterms[3,0]+x*x*y*pterms[4,0]+x*y*y*pterms[5,0]+y*y*y*pterms[6,0]
        m += x*x*x*pterms[3,1]+x*x*y*pterms[4,1]+x*y*y*pterms[5,1]+y*y*y*pterms[6,1]

    return l, m

def lm_to_rd(l, m, image):
    dis = (image['ctype'][-3:] == 'DIS')
    if not dis:
        ra  = l + image['crval1']
        dec = m + image['crval2']
    else:
        radeg = 180./numpy.pi
        r = numpy.sqrt(l**2.+m**2.)
        sphi =  l/(r+(r == 0)) # should go to 0 when r=0
        cphi = (-m+(r == 0))/(r+(r == 0))  # should go to 1 when r=0

        t = (radeg/(r+(r == 0)))*(r != 0)
        stht = (t+(t == 0))/numpy.sqrt(1.+t**2.)  # should go to 1 when t=0
        ctht = (1./numpy.sqrt(1.+t**2.))*(t != 0) # should go to 0 when t=0

        sdp  = numpy.sin(image['crval2']/radeg)
        cdp  = numpy.cos(image['crval2']/radeg)
      
        sdel = stht*sdp - ctht*cphi*cdp
        salp = ctht*sphi
        calp = stht*cdp + ctht*cphi*sdp
        alpha = numpy.arctan2(salp, calp)
        delta = numpy.arcsin(sdel)
      
        ra  = radeg*alpha + image['crval1']
        dec = radeg*delta
      
        ra = (ra % 360.)
    return ra, dec

def xy_to_rd_smfcoords(x, y, chip, smfcoords):
    return xy_to_rd(x, y, smfcoords[chip], smfcoords['PRIMARY'])

def readsmfcoords(smffile):
    out = { }
    hdulist = pyfits.open(smffile)
    for hdu in hdulist:
        if ((hdu.name == 'PRIMARY') or
            (re.match(r'XY..\.HDR', hdu.name.strip()) is not None)):
            name = hdu.name if hdu.name == 'PRIMARY' else hdu.name[0:-4]
            out[name] = get_ps1_coords(hdu.header)
    return out

# stolen from internet, Simon Brunning
def locate(pattern, root=os.curdir):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    for path, dirs, files in os.walk(os.path.abspath(root)):
        files2 = [os.path.join(os.path.relpath(path, start=root), f)
                  for f in files]
        for filename in fnmatch.filter(files2, pattern):
            yield os.path.join(os.path.abspath(root), filename)

def get_ps1_imlist():
    datadir = os.getenv('PS_DATA_CHIP_IMAGE_DIR', None)
    if not datadir:
        raise ValueError('Must set PS_DATA_CHIP_IMAGE_DIR environment variable.')
    smfdir = os.getenv('PS_DATA_SMF_DIR', None)
    if not smfdir:
        raise ValueError('Must set PS_DATA_SMF_DIR')
    smf = list(locate('*.smf', smfdir))
    chip = list(locate('*/*XY??.ch.fits', datadir))
    return list(smf), list(chip)

def get_smf_filename(filename_base):
    smfdir = os.getenv('PS_DATA_SMF_DIR', None)
    if not smfdir:
        raise ValueError('Must set PS_DATA_SMF_DIR')
    matches = list(locate('*/'+filename_base+'*.smf', smfdir))
    if len(matches) == 0:
        raise ValueError("Could not find smf file for %s." % filename_base)
    if len(matches) > 1:
        print(matches)
        raise ValueError("Multiple possible smf files for %s." % filename_base)
    return matches[0]

def get_ps1_chip_image(filename, offset=(0, 0), npix=None):
    filename_base = filename[:-5]
    im = pyfits.getdata(filename_base+'.fits')
    wt = pyfits.getdata(filename_base+'.wt.fits')
    mk = pyfits.getdata(filename_base+'.mk.fits')

    psffile = filename_base[:-3]+'.psf'
    psffp = tempfile.NamedTemporaryFile()
    psffp.close()
    subprocess.call(["dannyVizPSF", str(im.shape[0]), str(im.shape[1]),
                     str(im.shape[0]/2), str(im.shape[1]/2),
                     "51", "51", psffile, psffp.name])
    psfstamp = pyfits.getdata(psffp.name)

    smffn = get_smf_filename(os.path.basename(filename_base)[:11])
    im[mk != 0] = 0
    wt[mk != 0] = 0
    if npix is None:
        npix = im.shape
    im = im[offset[0]:offset[0]+npix[0], offset[1]:offset[1]+npix[1]]
    wt = wt[offset[0]:offset[0]+npix[0], offset[1]:offset[1]+npix[1]]
    mk = mk[offset[0]:offset[0]+npix[0], offset[1]:offset[1]+npix[1]]
    invvar = (1./(wt + (wt == 0)))*(wt != 0)

    hsmf = pyfits.getheader(smffn)
    zp = hsmf['MAG_ZP']

    hchip = pyfits.getheader(filename_base+'.fits', 1)
    chip = hchip['FPPOS'].upper().strip()
    
    filterid = 'PS1_'+hsmf['filterid'][0]

    tpsf = tractor.GaussianMixturePSF.fromStamp(psfstamp, N=3)

    twcs = wcs_ps1(smffn, chip, offset=offset)
    tsky = tractor.ConstantSky(0)
    photocal = tractor.MagsPhotoCal(filterid,
                                    zp+2.5*numpy.log10(hchip['exptime'])+0.5)

    tim = tractor.Image(data=im, invvar=invvar, psf=tpsf, wcs=twcs,
                        photocal=photocal,
                        sky=tsky, name='PS1 %s %s' % (filterid, filename_base))
    tim.zr = [-100, 100]
    tim.extent = [0, im.shape[0], 0, im.shape[1]]
    
    return tim

