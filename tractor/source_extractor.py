import np

from tractor import *
from tractor.galaxy import *

def get_se_modelfit_cat(T, maglim=25, bands=['u','g','r','i','z']):
    srcs = Catalog()
    isrcs = []
    for i,t in enumerate(T):
        if t.chi2_psf < t.chi2_model and t.mag_psf <= maglim:
            #print 'PSF'
            themag = t.mag_psf
            nm = NanoMaggies.magToNanomaggies(themag)
            m = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
            srcs.append(PointSource(RaDecPos(t.ra, t.dec), m))
            isrcs.append(i)
            continue

        if t.mag_disk > maglim and t.mag_spheroid > maglim:
            #print 'Faint'
            continue

        # deV: spheroid
        # exp: disk

        dmag = t.mag_spheroid
        emag = t.mag_disk

        # SPHEROID_REFF [for Sersic index n= 1] = 1.68 * DISK_SCALE

        if dmag <= maglim:
            shape_dev = GalaxyShape(t.spheroid_reff_world * 3600.,
                                    t.spheroid_aspect_world,
                                    t.spheroid_theta_world + 90.)

        if emag <= maglim:
            shape_exp = GalaxyShape(t.disk_scale_world * 1.68 * 3600.,
                                    t.disk_aspect_world,
                                    t.disk_theta_world + 90.)

        pos = RaDecPos(t.alphamodel_j2000, t.deltamodel_j2000)

        isrcs.append(i)
        if emag > maglim and dmag <= maglim:
            nm = NanoMaggies.magToNanomaggies(dmag)
            m_dev = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
            srcs.append(DevGalaxy(pos, m_dev, shape_dev))
            continue
        if emag <= maglim and dmag > maglim:
            nm = NanoMaggies.magToNanomaggies(emag)
            m_exp = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
            srcs.append(ExpGalaxy(pos, m_exp, shape_exp))
            continue

        # print 'Composite'
        nmd = NanoMaggies.magToNanomaggies(dmag)
        nme = NanoMaggies.magToNanomaggies(emag)
        nm = nmd + nme
        fdev = (nmd / nm)
        m = NanoMaggies(order=bands, **dict([(k, nm) for k in bands]))
        srcs.append(FixedCompositeGalaxy(pos, m, fdev, shape_exp, shape_dev))

    #print 'Sources:', len(srcs)
    return srcs, np.array(isrcs)


