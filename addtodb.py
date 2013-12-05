import numpy as np
import pyfits as pyf
import os
import sys
from tractor.rc3 import *
from create_fits import *
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
import matplotlib
import pylab as plt
from astropysics.obstools import *

def add_to_table_nsatlas(name):
    os.chdir("RC3_Output")

    nsatlas = pyf.open('nsa-short.fits.gz')[1].data
    table = pyf.open('large_galaxies.fits')
    e=data.field('NSAID')

    mask = e == nsid
    record = data[mask]

    fn = name+"-updated.pickle"
    print fn

    CG,r50s,r90s,concs=unpickle_from_file(fn)

    pos = CG.getPosition()
    tot = CG.getBrightness()

    dev = CG.brightnessDev
    dev_shape = CG.shapeDev
    exp = CG.brightnessExp
    exp_shape = CG.shapeExp

    cg_ra = pos[0]
    cg_dec = pos[1]
    cg_r50s = r50s
    cg_r90s = r90s
    cg_conc = concs
    cg_totmags = tot
    cg_devre = dev_shape.re
    cg_devab = dev_shape.ab
    cg_devphi = dev_shape.phi
    cg_devmags = dev
    cg_expre = exp_shape.re
    cg_devab = exp_shape.ab
    cg_devphi = exp_shape.phi
    cg_expmags = exp
    cg_extinction = extinction(pos)
    cg_mu50 = mu_50(tot[3]-cg_extinction[3],r50s[3])

    name = name.replace('_',' ')
    rc3_name = name
    rc3=getName(name,fn="mediumrc3.fits")
    rc3_ra = float(record['RA'][0])
    rc3_dec = float(record['DEC'][0])
    rc3_log_ae = float(0.)
    rc3_log_d25 = float(0.)

    newdata = rc3_name,rc3_ra,rc3_dec,rc3_log_ae,rc3_log_d25,cg_ra,cg_dec,cg_r50s,cg_r90s,cg_conc,cg_totmags,cg_devre,cg_devab,cg_devphi,cg_devmags,cg_expre,cg_devab,cg_devphi,cg_expmags,cg_extinction,cg_mu50


    nrows=table[1].data.shape[0]+1
    hdu= pyfits.new_table(table[1].columns,nrows=nrows)
    for i in range(len(table[1].columns.names)):
        hdu.data.field(i)[-1]=newdata[i]

    pri_hdu = pyf.PrimaryHDU(np.arange(100))
    tbuhdulist =pyf.HDUList([pri_hdu,hdu])

    #tbuhdulist.writeto('large_galaxies.fits',clobber=True)
    print os.getcwd()
    os.chdir('/data1/dwm261/tractor/')
    print os.getcwd()

    

def add_to_table(name):


    os.chdir("RC3_Output")

    rc3 = pyf.open('newrc3limited.fits')
    table = pyf.open('large_galaxies.fits')



    fn = name+"-updated.pickle"
    print fn

    CG,r50s,r90s,concs=unpickle_from_file(fn)

    pos = CG.getPosition()
    tot = CG.getBrightness()

    dev = CG.brightnessDev
    dev_shape = CG.shapeDev
    exp = CG.brightnessExp
    exp_shape = CG.shapeExp

    cg_ra = pos[0]
    cg_dec = pos[1]
    cg_r50s = r50s
    cg_r90s = r90s
    cg_conc = concs
    cg_totmags = tot
    cg_devre = dev_shape.re
    cg_devab = dev_shape.ab
    cg_devphi = dev_shape.phi
    cg_devmags = dev
    cg_expre = exp_shape.re
    cg_devab = exp_shape.ab
    cg_devphi = exp_shape.phi
    cg_expmags = exp
    cg_extinction = extinction(pos)
    cg_mu50 = mu_50(tot[3]-cg_extinction[3],r50s[3])

    name = name.replace('_',' ')
    rc3_name = name
    rc3=getName(name,fn="mediumrc3.fits")
    rc3_ra = float(rc3['RA'][0])
    rc3_dec = float(rc3['DEC'][0])
    rc3_log_ae = float(rc3['LOG_AE'][0])
    rc3_log_d25 = float(rc3['LOG_D25'][0])

    newdata = rc3_name,rc3_ra,rc3_dec,rc3_log_ae,rc3_log_d25,cg_ra,cg_dec,cg_r50s,cg_r90s,cg_conc,cg_totmags,cg_devre,cg_devab,cg_devphi,cg_devmags,cg_expre,cg_devab,cg_devphi,cg_expmags,cg_extinction,cg_mu50


    nrows=table[1].data.shape[0]+1
    hdu= pyfits.new_table(table[1].columns,nrows=nrows)
    for i in range(len(table[1].columns.names)):
        hdu.data.field(i)[-1]=newdata[i]

    pri_hdu = pyf.PrimaryHDU(np.arange(100))
    tbuhdulist =pyf.HDUList([pri_hdu,hdu])

    tbuhdulist.writeto('large_galaxies.fits',clobber=True)
    print os.getcwd()
    os.chdir('/data1/dwm261/tractor/')
    print os.getcwd()

