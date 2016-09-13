# Copyright (c) 2012-2014 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#
"""
Demo #12

The twelfth script in our tutorial about using GalSim in python scripts: examples/demo*.py.
(This file is designed to be viewed in a window 100 characters wide.)

This script currently doesn't have an equivalent demo*.yaml or demo*.json file.  The API for catalog
level chromatic objects has not been written yet.

This script introduces the chromatic objects module galsim.chromatic, which handles wavelength-
dependent profiles.  Three uses of this module are demonstrated:

1) A chromatic object representing a De Vaucouleurs galaxy with a early-type SED at redshift 0.8 is
created.  The galaxy is then drawn using the six LSST filter throughput curves to demonstrate that
the galaxy is a g-band dropout.

2) A two-component bulge+disk galaxy, in which the bulge and disk have different SEDs, is created
and then drawn using LSST filters.

3) A wavelength-dependent PSF is created to represent atmospheric effects of differential chromatic
refraction, and the wavelength dependence of Kolmogorov-turbulence-induced seeing.  This PSF is used
to draw a single Sersic galaxy in the LSST filters.

For all cases, suggested parameters for viewing in ds9 are also included.

New features introduced in this demo:

- SED = galsim.SED(wave, flambda)
- SED2 = SED.atRedshift(redshift)
- bandpass = galsim.Bandpass(filename)
- bandpass2 = bandpass.truncate(relative_throughput)
- bandpass3 = bandpass2.thin(rel_err)
- gal = galsim.Chromatic(GSObject, SED)
- gal = GSObject * SED
- obj = galsim.Add([list of ChromaticObjects])
- ChromaticObject.drawImage(bandpass)
- PSF = galsim.ChromaticAtmosphere(GSObject, base_wavelength, zenith_angle)
"""
from __future__ import print_function
import sys
import os
import math
import numpy
import logging
import galsim

def main(argv):
    # Where to find and output data
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "data/"))
    outpath = os.path.abspath(os.path.join(path, "output/"))

    if not os.path.exists(outpath):
        print('Creating', outpath)
        os.makedirs(outpath)

    # In non-script code, use getLogger(__name__) at module scope instead.
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger("demo12")

    # initialize (pseudo-)random number generator
    random_seed = 1234567
    rng = galsim.BaseDeviate(random_seed)

    # read in SEDs
    SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
    SEDs = {}
    for SED_name in SED_names:
        SED_filename = os.path.join(datapath, '{}.sed'.format(SED_name))
        # Here we create some galsim.SED objects to hold star or galaxy spectra.  The most
        # convenient way to create realistic spectra is to read them in from a two-column ASCII
        # file, where the first column is wavelength and the second column is flux. Wavelengths in
        # the example SED files are in Angstroms, flux in flambda.  The default wavelength type for
        # galsim.SED is nanometers, however, so we need to override by specifying
        # `wave_type = 'Ang'`.
        SED = galsim.SED(SED_filename, wave_type='Ang')
        # The normalization of SEDs affects how many photons are eventually drawn into an image.
        # One way to control this normalization is to specify the flux density in photons per nm
        # at a particular wavelength.  For example, here we normalize such that the photon density
        # is 1 photon per nm at 500 nm.
        SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, wavelength=500)
    logger.debug('Successfully read in SEDs')

    # read in the LSST filters
    filter_names = 'ugrizy'
    filters = {}
    for filter_name in filter_names:
        filter_filename = os.path.join(datapath, 'LSST_{}.dat'.format(filter_name))
        # Here we create some galsim.Bandpass objects to represent the filters we're observing
        # through.  These include the entire imaging system throughput including the atmosphere,
        # reflective and refractive optics, filters, and the CCD quantum efficiency.  These are
        # also conveniently read in from two-column ASCII files where the first column is
        # wavelength and the second column is dimensionless flux. The example filter files have
        # units of nanometers and dimensionless throughput, which is exactly what galsim.Bandpass
        # expects, so we just specify the filename.
        filters[filter_name] = galsim.Bandpass(filter_filename)
        # For speed, we can thin out the wavelength sampling of the filter a bit.
        # In the following line, `rel_err` specifies the relative error when integrating over just
        # the filter (however, this is not necessarily the relative error when integrating over the
        # filter times an SED)
        filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    logger.debug('Read in filters')

    pixel_scale = 0.2 # arcseconds

    #-----------------------------------------------------------------------------------------------
    # Part B: chromatic bulge+disk galaxy

    logger.info('')
    logger.info('Starting part B: chromatic bulge+disk galaxy')
    redshift = 0.8
    # make a bulge ...
    mono_bulge = galsim.DeVaucouleurs(half_light_radius=0.5)
    bulge_SED = SEDs['CWW_E_ext'].atRedshift(redshift)
    # The `*` operator can be used as a shortcut for creating a chromatic version of a GSObject:
    bulge = mono_bulge * bulge_SED
    bulge = bulge.shear(g1=0.12, g2=0.07)
    logger.debug('Created bulge component')
    # ... and a disk ...
    mono_disk = galsim.Exponential(half_light_radius=2.0)
    disk_SED = SEDs['CWW_Im_ext'].atRedshift(redshift)
    disk = mono_disk * disk_SED
    disk = disk.shear(g1=0.4, g2=0.2)
    logger.debug('Created disk component')
    # ... and then combine them.
    bdgal = 1.1 * (0.8*bulge+4*disk) # you can add and multiply ChromaticObjects just like GSObjects

    # convolve with PSF to make final profile
    #PSF = galsim.Moffat(fwhm=0.6, beta=2.5)
    psf_sigma = 1.5 * pixel_scale # in arcsec
    PSF = galsim.Gaussian(flux=1., sigma=psf_sigma)

    bdfinal = galsim.Convolve([bdgal, PSF])
    # Note that at this stage, our galaxy is chromatic but our PSF is still achromatic.  Part C)
    # below will dive into chromatic PSFs.
    logger.debug('Created bulge+disk galaxy final profile')

    # draw profile through LSST filters
    gaussian_noise = galsim.GaussianNoise(rng, sigma=0.02)
    for filter_name, filter_ in filters.iteritems():
        img = galsim.ImageF(64, 64, scale=pixel_scale)
        bdfinal.drawImage(filter_, image=img)
        logger.debug('Created {}-band image'.format(filter_name))
        out_filename = os.path.join(outpath, 'demo12b_{}.fits'.format(filter_name))

        nepochs = 3
        images = []
        for i in range(nepochs):
            newImg = img.copy()
            newImg.addNoise(gaussian_noise)
            images.append(newImg)
        galsim.fits.writeCube(images, out_filename)
        logger.debug('Wrote {}-band images to disk'.format(filter_name))
        logger.info('Added flux for {}-band image: {}'.format(filter_name, img.added_flux))

    logger.info('You can display the output in ds9 with a command line that looks something like:')
    logger.info('ds9 -rgb -blue -scale limits -0.2 0.8 output/demo12b_r.fits -green -scale limits'
                +' -0.25 1.0 output/demo12b_i.fits -red -scale limits -0.25 1.0 output/demo12b_z.fits'
                +' -zoom 2 &')


if __name__ == "__main__":
    main(sys.argv)
