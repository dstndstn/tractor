if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import os
import logging
import numpy as np
import pylab as plt

import pyfits

from astrometry.util.file import *

from tractor import *
from tractor import sdss as st
from tractor.saveImg import *
from tractor import sdss_galaxy as sg
from tractor import basics as ba
from tractor import engine as en
from astrometry.util.util import Tan
from tractor.emfit import em_fit_2d
from tractor.fitpsf import em_init_params

def halflight(name,makePlots=False):
    CG = unpickle_from_file("RC3_Output/%s.pickle" %name)
    ra,dec = CG.getPosition()
    maxradius=max(CG.shapeExp.re,CG.shapeDev.re)
    print "Working on %s" % name
    print CG

    #First step is to make an image, which needs:
    # data, invvar, psf, wcs, sky, photocal, name, zr
    
    crval1 = ra
    crval2 = dec
    pixscale = cd11 = .396/3600.
    cd12 = 0.
    cd21 = 0.
    cd22 = pixscale
    imagew = int(32*maxradius)
    crpix1 = .5*imagew
    imageh = int(32*maxradius)
    crpix2 = .5*imageh
    tan = Tan(crval1,crval2,crpix1,crpix2,cd11,cd12,cd21,cd22,imagew,imageh)

    wcs = ba.FitsWcs(tan)

    data=np.zeros((imagew,imageh))
    invvar=np.ones((imagew,imageh))

    psf = ba.GaussianMixturePSF(1.,[0.,0.],np.array(1.)) #amp,mean,var
    
    skyobj = ba.ConstantSky(0.)
    zr = np.array([-5.,+5.])


    tims = []
    bands = ['u','g','r','i','z']
    for bandname in bands:
        photocal = st.SdssNanomaggiesPhotoCal(bandname)
        image = en.Image(data=data,invvar=invvar,sky=skyobj,psf=psf,wcs=wcs,photocal=photocal,name="Half-light %s" %bandname,zr=zr)
        tims.append(image)
    tractor = st.SDSSTractor(tims)
    tractor.addSources([CG])

    yg, xg = np.meshgrid(np.arange(imageh) - crpix2, np.arange(imagew) - crpix1)
    r2g = xg ** 2 + yg ** 2
    rlist_pix = np.exp(np.linspace(0.,np.log(0.5*imageh),64))
    rlist_arcsec = rlist_pix * pixscale * 3600.
    mimgs = tractor.getModelImages()
    r50s = []
    r90s = []
    concs = []
    expr50 = CG.shapeExp.re * np.sqrt(CG.shapeExp.ab)
    devr50 = CG.shapeDev.re * np.sqrt(CG.shapeDev.ab)

    for bandname,image in zip(bands,mimgs):
        plist = [np.sum(image[r2g < (r * r)]) for r in rlist_pix]
        plist /= plist[-1]
        r50, r90 = np.interp([0.5, 0.9], plist, rlist_arcsec)
        conc = r90/r50
        r50s.append(r50)
        r90s.append(r90)
        concs.append(conc)
        if r50<min(devr50,expr50) or r50>max(devr50,expr50):
            print "R50 is not in between DeV and exp radii for %s" %name
        if 1./conc > .46 or 1./conc <.29:
            print "C=%.2f is a strange concentration for %s" % (conc,name)
        print name, bandname, r50, r90, conc
        if makePlots:
            plt.clf()
            plt.axhline(0,color='k',alpha=0.25)
            plt.xlabel("radius in arcsecond")
            plt.ylabel("fraction of azimuthally averaged flux")
            plt.title("%s" %name)
            plt.plot(rlist_arcsec, plist, 'k-')
            plt.axvline(r50, color='k', alpha=0.5)
            plt.axvline(r90, color='k', alpha=0.5)
            plt.axvline(devr50, color='r', alpha=0.5)
            plt.axvline(expr50, color='b', alpha=0.5)
            plt.text(1.02 * r90, 0.01, "$C = %.1f$" % (conc), ha='left')
            plt.xlim(0,2.*r90)
            plt.ylim(-0.1,1.1)
            plt.savefig("radial-profile-%s-%s.png" % (name,bandname))

    pickle_to_file([CG,r50s,r90s,concs],'RC3_Output/%s-updated.pickle' %name)


def main():
    import optparse
    parser = optparse.OptionParser(usage='%prog [options] <name>')
    opt,args = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(-1)

    name = args[0]
    halflight(name)

if __name__ == '__main__':
    main()
