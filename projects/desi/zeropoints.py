import matplotlib
matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.util import Sip
from astrometry.libkd.spherematch import match_radec

from common import *

if __name__ == '__main__':

    ps = PlotSequence('zp')

    plt.clf()
    chips = (['N' + '%i'%i for i in range(1, 32)] +
             ['S' + '%i'%i for i in range(1, 32)])
    for c in chips:
        x0,x1,y0,y1 = ccd_map_extent(c, inset=0.05)
        plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], 'b-', alpha=0.5)
        plt.text((x0+x1)/2., (y0+y1)/2., c, ha='center', va='center')
    ps.savefig()

    D = Decals()
    T = D.get_ccds()

    #outfn = None
    outfn = 'zp.fits'

    # DES SN1
    expnums = [ 148563, 148564, 148567, 173321, 173322, 173324, 173325,
                174346, 174347, 174349, 175270, 175271, 175273, 175274,
                228752, 228753 ]
    #expnums = [349664, 349667, 349589]

    allzps = {}

    for expnum in expnums:
        TT = T[T.expnum == expnum]
        print len(TT), 'with expnum', expnum
        bands = np.unique(TT.filter)
        assert(len(bands) == 1)
        band = bands[0]
        print 'Band:', band
        exptime = TT.exptime[0]
        print 'Exposure time', exptime

        ims = []
        for t in TT:
            print
            print 'Image file', t.cpimage, 'hdu', t.cpimage_hdu
            im = DecamImage(D, t)
            ims.append(im)

        chipnames = []
        corrs = []
        for im in ims:
            if not os.path.exists(im.corrfn):
                print 'NO SUCH FILE:', im.corrfn
                continue

            sdss = fits_table(im.sdssfn)

            #sefn = im.morphfn
            sefn = im.sefn
            print 'Looking for', sefn

            morph = fits_table(sefn, hdu=2)

            wcs = Sip(im.wcsfn)
            if len(sdss) == 0:
                print 'EMPTY:', im.sdssfn
                continue
            if len(morph) == 0:
                print 'EMPTY:', im.morphfn
                continue
            print len(sdss), 'SDSS sources from', im.sdssfn
            print len(morph), 'SE sources from', sefn
            morph.ra,morph.dec = wcs.pixelxy2radec(morph.x_image, morph.y_image)

            I,J,d = match_radec(morph.ra, morph.dec, sdss.ra, sdss.dec, 0.5/3600.)
            corr = sdss[J]
            corr.add_columns_from(morph[I])

            chipnames.append(im.extname)
            #corr = fits_table(im.corrfn)

            corrs.append(corr)
            print im, ':', len(corr), 'correspondences'

        for col,cut in ([('flux_auto',None)] + [('flux_aper',i) for i in range(3)]
                        + [('flux_psf',None), ('flux_model',None)]):
            dmags = []
            smags = []
            for corr in corrs:
                if not col in corr.get_columns():
                    continue
                dflux = corr.get(col)
                if cut is not None:
                    dflux = dflux[:,cut]
                sflux = corr.get('%s_psfflux' % band)
                I = np.logical_and(dflux > 0, sflux > 0)
                dmag = -2.5 *  np.log10(dflux[I])
                smag = -2.5 * (np.log10(sflux[I]) - 9)
                dmags.append(dmag)
                smags.append(smag)
                print 'median', np.median(smag - dmag)
                #print 'mags', dmag, smag
            if len(dmags) == 0:
                continue
            zps = [np.median(smag - dmag) for dmag,smag in zip(dmags,smags)]
            #print 'zeropoints:', zps

            zps = np.array(zps)
            meanzp = np.mean(zps)
            zps -= meanzp
            #print 'zps:', zps
            mx = np.abs(zps).max()
            map = ccd_map_image(dict(zip(chipnames, zps)), empty=np.nan)
            #print 'Map:', map
            plt.clf()
            dimshow(map, cmap='RdBu', vmin=-mx, vmax=mx)
            plt.xticks([]); plt.yticks([])
            for chip in chipnames:
                x,y = ccd_map_center(chip)
                plt.text(x-0.5, y-0.5, chip, fontsize=8, color='k', ha='center', va='center')
            plt.colorbar()
            if cut is not None:
                col = '%s[%i]' % (col,cut)
            plt.title('Zeropoint diffs (mean %.3f) for %s, %i %s' % (meanzp, col, expnum, band))
            ps.savefig()
                   
            if not expnum in allzps:
                allzps[expnum] = (expnum, meanzp, im.exptime)


    if outfn:
        T = fits_table()
        T.expnum = np.array(allzps.keys())
        T.ccdzpt  = np.array([v[1] for v in allzps.values()])
        T.exptime = np.array([v[2] for v in allzps.values()])
        T.writeto(outfn)
        print 'Wrote', outfn
