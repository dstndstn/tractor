
radecroi = (334.3, 334.4, 0.3, 0.4)

def make_plots(prefix, im, tr=None, plots=['data','model','chi'], mags=['i'],
               radecroi_in=None):
    import pylab as plt
    import tractor

    if radecroi_in:
        global radecroi
        radecroi = radecroi_in

    if tr is None:
        srcs = get_cfht_catalog(mags=mags)
        tr = tractor.Tractor(tractor.Images(im), srcs)

    mod = tr.getModelImage(im)

    ima = dict(interpolation='nearest', origin='lower',
               vmin=im.zr[0], vmax=im.zr[1])
    imchi = dict(interpolation='nearest', origin='lower',
               vmin=-5, vmax=+5)
    if hasattr(im, 'extent'):
        ima.update(extent=im.extent)
        imchi.update(extent=im.extent)

    fignum = 1
    
    if 'data' in plots:
        plt.figure(fignum)
        fignum += 1
        plt.clf()
        plt.imshow(im.getImage(), **ima)
        plt.gray()
        plt.title(im.name + ': data')
        plt.savefig(prefix + 'data.png')

    if 'model' in plots:
        plt.figure(fignum)
        fignum += 1
        plt.clf()
        plt.imshow(mod, **ima)
        plt.gray()
        plt.title(im.name + ': model')
        plt.savefig(prefix + 'mod.png')

    if 'chi' in plots:
        plt.figure(fignum)
        fignum += 1
        plt.clf()
        plt.imshow((mod - im.getImage()) * im.getInvError(), **imchi)
        plt.gray()
        plt.title(im.name + ': chi')
        plt.savefig(prefix + 'chi.png')

def get_cfht_catalog(mags=['i'], maglim = 27., returnTable=False):
    from astrometry.util.pyfits_utils import fits_table
    from tractor import Mags, RaDecPos, PointSource, Images, Catalog
    from tractor.galaxy import DevGalaxy, ExpGalaxy, CompositeGalaxy, GalaxyShape

    T = fits_table('/project/projectdirs/bigboss/data/cs82/W4p1m1_i.V2.7A.swarp.cut.deVexp.fit', hdunum=2)
    print 'Read', len(T), 'sources'
    # Cut to ROI
    (ra0,ra1, dec0,dec1) = radecroi
    T.ra  = T.alpha_j2000
    T.dec = T.delta_j2000
    T = T[(T.ra > ra0) * (T.ra < ra1) * (T.dec > dec0) * (T.dec < dec1)]
    print 'Cut to', len(T), 'objects in ROI.'
    
    srcs = Catalog()
    keepi = []
    for i,t in enumerate(T):
        if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
            #print 'PSF'
            themag = t.mag_psf
            m = Mags(order=mags, **dict([(k, themag) for k in mags]))
            srcs.append(PointSource(RaDecPos(t.ra, t.dec), m))
            keepi.append(i)
            continue
        if t.mag_disk > maglim and t.mag_spheroid > maglim:
            #print 'Faint'
            continue
        keepi.append(i)
        themag = t.mag_spheroid
        m_exp = Mags(order=mags, **dict([(k, themag) for k in mags]))
        themag = t.mag_disk
        m_dev = Mags(order=mags, **dict([(k, themag) for k in mags]))

        # SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE
        shape_dev = GalaxyShape(t.disk_scale_world * 1.68 * 3600., t.disk_aspect_world,
                                t.disk_theta_world + 90.)
        shape_exp = GalaxyShape(t.spheroid_reff_world * 3600., t.spheroid_aspect_world,
                                t.spheroid_theta_world + 90.)
        pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

        if t.mag_disk > maglim and t.mag_spheroid <= maglim:
            #print 'Exp'
            srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
            continue
        if t.mag_disk <= maglim and t.mag_spheroid > maglim:
            #print 'deV'
            srcs.append(DevGalaxy(pos, m_dev, shape_dev))
            continue
        # exp + deV
        srcs.append(CompositeGalaxy(pos, m_exp, shape_exp, m_dev, shape_dev))

    if returnTable:
        import numpy as np
        T.cut(np.array(keepi))
        return srcs, T

    return srcs
