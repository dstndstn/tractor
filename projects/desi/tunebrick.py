from astrometry.util.util import *
from astrometry.util.fits import *
from astrometry.util.plotutils import *
from astrometry.util.ttime import Time, MemMeas

from common import *
from desi_common import *

from tractor.galaxy import *

from runbrick import stage_tims

def stage_cat(brickid=None, target_extent=None,
              targetwcs=None, tims=None, **kwargs):

    catpattern = 'pipebrick-cats/tractor-phot-b%06i.fits'

    catfn = catpattern % brickid
    T = fits_table(catfn)
    cat = read_fits_catalog(T)
    print 'Got catalog:', len(cat), 'sources'
    
    # decals = Decals()
    # brick = decals.get_brick(brick)
    # print 'Brick:', brick.about()
    # 
    # targetwcs = wcs_for_brick(brick)
    # ccds = decals.get_ccds()
    # ccds.cut(ccds_touching_wcs(targetwcs, ccds))
    # print len(ccds), 'touching brick'

    # pipeline?
    #pipe = True

    #P = stage_tims(brickid=brickid, pipe=pipe, target_extent=target_extent)
    #tims = P['tims']
    #targetwcs = P['targetwcs']

    if target_extent is not None:
        #x0,x1,y0,y1 = target_extent
        W,H = int(targetwcs.get_width()), int(targetwcs.get_height())
        print 'W,H', W,H
        x0,x1,y0,y1 = 1,W, 1,H
        r,d = targetwcs.pixelxy2radec(np.array([x0,x0,x1,x1]),np.array([y0,y1,y1,y0]))
        r0,r1 = r.min(),r.max()
        d0,d1 = d.min(),d.max()
        margin = 0.002
        keepcat = []
        for src in cat:
            pos = src.getPosition()
            if (pos.ra  > r0-margin and pos.ra  < r1+margin and
                pos.dec > d0-margin and pos.dec < d1+margin):
                keepcat.append(src)
        cat = keepcat
        print 'Keeping', len(cat), 'sources within range'

    print 'Catalog:'
    for src in cat:
        print '  ', src

    switch_to_soft_ellipses(cat)
    keepcat = []
    for src in cat:
        if not np.all(np.isfinite(src.getParams())):
            print 'Dropping source:', src
            continue
        keepcat.append(src)
    cat = keepcat
    print len(cat), 'sources with finite params'

    
    print len(tims), 'tims'
    print 'Sizes:', [tim.shape for tim in tims]

    for tim in tims:
        from tractor.psfex import CachingPsfEx
        tim.psfex.radius = 20
        tim.psfex.fitSavedData(*tim.psfex.splinedata)
        tim.psf = CachingPsfEx.fromPsfEx(tim.psfex)

    return dict(cat=cat)


def stage_plots(tims=None, cat=None, targetwcs=None, **kwargs):
    print 'kwargs:', kwargs.keys()

    tractor = Tractor(tims, cat)
    for i,tim in enumerate(tims):
        mod = tractor.getModelImage(i)
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(tim.getImage(), **tim.ima)
        plt.subplot(1,2,2)
        plt.imshow(mod, **tim.ima)
        ps.savefig()

    # residtims = []
    # mods0 = []
    # for i,tim in enumerate(tims):
    #     #mod = tractor.getModelImage(

    srcmods = dict([(src,[]) for src in cat])

    mods = []
    for itim,tim in enumerate(tims):
        #timmods = []

        modimg = np.zeros(tim.getModelShape(), tractor.modtype)
        tim.getSky().addTo(modimg)

        for src in cat:
            patch = src.getModelPatch(tim)
            if patch is None:
                continue
            if patch.patch is None:
                continue

            # HACK -- this shouldn't be necessary, but seems to be!
            # FIXME -- track down why patches are being made with extent outside
            # that of the parent!
            H,W = tim.shape
            patch.clipTo(W,H)
            ph,pw = patch.shape
            if pw*ph == 0:
                continue

            patch.addTo(modimg)

            srcmods[src].append((itim, patch))
        mods.append(modimg)


    keepcat = []
    for src in cat:
        print 'Setting source flux to zero:', src
        lnp0 = tractor.getLogProb()
        bright = src.getBrightness()
        flux = bright.getParams()
        bright.setParams(np.zeros(len(flux)))
        lnp1 = tractor.getLogProb()

        xlnp0 = 0.
        for tim,mod in zip(tims, mods):
            chi2 = np.sum(((tim.getImage() - mod) * tim.getInvError())**2)
            xlnp0 += -0.5 * chi2
        print 'xlnp0:', xlnp0
        print ' lnp0:', lnp0

        for itim,patch in srcmods[src]:
            patch.addTo(mods[itim], scale=-1)
            
        xlnp1 = 0.
        for tim,mod in zip(tims, mods):
            chi2 = np.sum(((tim.getImage() - mod) * tim.getInvError())**2)
            xlnp1 += -0.5 * chi2
        print 'xlnp1:', xlnp1
        print ' lnp1:', lnp1

        print 'Delta-logprob:', lnp1 - lnp0
        print '        xdlnp:', xlnp1 - xlnp0

        # if xlnp1 - xlnp0 > 0:
        #     print 'Removing source!'
        # else:
        #     # Put it back like it was
        #     for itim,patch in srcmods[src]:
        #         patch.addTo(mods[itim])


        # Put it back like it was
        for itim,patch in srcmods[src]:
            patch.addTo(mods[itim])

        sdlnp = 0.
        for itim,patch in srcmods[src]:
            tim = tims[itim]
            mod = mods[itim]
            slc = patch.getSlice(tim)
            #print 'Slice', slc
            simg = tim.getImage()[slc]
            sie  = tim.getInvError()[slc]
            #print 'Img slice', simg.shape
            #print 'sie', sie.shape
            #print 'mod', mod[slc].shape
            #print 'patch', patch.shape
            chisq0 = np.sum(((simg - mod[slc]) * sie)**2)
            chisq1 = np.sum(((simg - (mod[slc] - patch.patch)) * sie)**2)
            sdlnp += -0.5 * (chisq1 - chisq0)

        print '        sdlnp:', sdlnp


        if xlnp1 - xlnp0 > 0:
            print 'Removing source!'
            # Subtract it off again!
            for itim,patch in srcmods[src]:
                patch.addTo(mods[itim], scale=-1)


        if lnp1 - lnp0 > 0:
            print 'Removing source!'
            continue
        keepcat.append(src)
        bright.setParams(flux)
    cat = keepcat

    tractor = Tractor(tims, cat)
    for i,tim in enumerate(tims):
        mod = tractor.getModelImage(i)
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(tim.getImage(), **tim.ima)
        plt.subplot(1,2,2)
        plt.imshow(mod, **tim.ima)
        ps.savefig()


if __name__ == '__main__':
    import optparse
    from astrometry.util.stages import *

    parser = optparse.OptionParser()
    parser.add_option('-f', '--force-stage', dest='force', action='append', default=[],
                      help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_option('-s', '--stage', dest='stage', default=[], action='append',
                      help="Run up to the given stage(s)")
    parser.add_option('-n', '--no-write', dest='write', default=True, action='store_false')
    parser.add_option('-P', '--pickle', dest='picklepat', help='Pickle filename pattern, with %i, default %default',
                      default='pickles/tunebrick-%(brick)06i-%%(stage)s.pickle')

    parser.add_option('-b', '--brick', type=int, help='Brick ID to run: default %default',
                      default=377306)
    parser.add_option('-p', '--plots', dest='plots', action='store_true')
    #parser.add_option('--stamp', action='store_true')
    parser.add_option('--zoom', type=int, nargs=4, help='Set target image extent (default "0 3600 0 3600")')
    parser.add_option('-W', type=int, default=3600, help='Target image width (default %default)')
    parser.add_option('-H', type=int, default=3600, help='Target image height (default %default)')

    opt,args = parser.parse_args()
    Time.add_measurement(MemMeas)

    stagefunc = CallGlobal('stage_%s', globals())

    if len(opt.stage) == 0:
        opt.stage.append('plots')
    opt.force.extend(opt.stage)

    opt.picklepat = opt.picklepat % dict(brick=opt.brick)

    prereqs = {'tims': None,
               'cat': 'tims',
               'plots': 'cat',
               }

    ps = PlotSequence('tune-b%06i' % opt.brick)
    initargs = dict(ps=ps)
    initargs.update(W=opt.W, H=opt.H, brickid=opt.brick, target_extent=opt.zoom)
    kwargs = {}

    for stage in opt.stage:
        runstage(stage, opt.picklepat, stagefunc, force=opt.force, write=opt.write,
                 prereqs=prereqs, initial_args=initargs, **kwargs)
                 
               #tune(opt.brick, target_extent=opt.zoom)
