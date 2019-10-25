from __future__ import print_function
import numpy as np
import scipy.interpolate as interp
from tractor.utils import ParamList
from tractor import ducks


class SplineSky(ParamList, ducks.ImageCalibration):

    @staticmethod
    def BlantonMethod(image, mask, gridsize, estimator=np.median):
        '''
        mask: True to use pixel.  None for no masking.
        '''
        H, W = image.shape
        halfbox = gridsize // 2

        nx = int(np.round(W // float(halfbox))) + 2
        x0 = int((W - (nx - 2) * halfbox) // 2)
        xgrid = x0 + (halfbox * (np.arange(nx) - 0.5)).astype(int)

        ny = int(np.round(H // float(halfbox))) + 2
        y0 = int((H - (ny - 2) * halfbox) // 2)
        ygrid = y0 + (halfbox * (np.arange(ny) - 0.5)).astype(int)

        # Compute medians in grid cells
        grid = np.zeros((ny, nx))
        for iy, y in enumerate(ygrid):
            ylo, yhi = int(max(0, y - halfbox)), int(min(H, y + halfbox))
            for ix, x in enumerate(xgrid):
                xlo, xhi = int(max(0, x - halfbox)), int(min(W, x + halfbox))
                im = image[ylo:yhi, xlo:xhi]
                if mask is not None:
                    im = im[mask[ylo:yhi, xlo:xhi]]
                if len(im):
                    grid[iy, ix] = estimator(im)

        return SplineSky(xgrid, ygrid, grid)

    def __init__(self, X, Y, bg, order=3):
        '''
        X,Y: pixel positions of control points
        bg: values of spline at control points
        '''
        # Note -- using weights, we *could* try to make the spline
        # constructor fit for the sky residuals (averaged over the
        # spline boxes)
        self.W = len(X)
        self.H = len(Y)

        self.xgrid = X.copy()
        self.ygrid = Y.copy()
        # spl(xgrid,ygrid) = gridvals
        #self.gridvals = bg.copy()

        self.order = order
        self.spl = interp.RectBivariateSpline(X, Y, bg.T,
                                              kx=order, ky=order)
        (tx, ty, c) = self.spl.tck
        super(SplineSky, self).__init__(*c)
        # override -- "c" is a list, so this should work as expected
        self.vals = c
        self.prior_smooth_sigma = None

        # offset for subimage sky models.
        self.x0 = 0
        self.y0 = 0

    def shift(self, x0, y0):
        self.x0 += x0
        self.y0 += y0

    def shifted(self, x0, y0):
        s = self.copy()
        s.shift(x0, y0)
        return s

    def offset(self, dsky):
        #sky0 = self.spl(0,0)
        (tx, ty, c) = self.spl.tck
        c = [ci + dsky for ci in c]
        self.spl.tck = (tx, ty, c)
        self.vals = c
        #sky1 = self.spl(0,0)
        #print('Offset sky by', dsky, ':', sky0, 'to', sky1)

    def scale(self, s):
        '''
        Scales this sky model by a factor of *s*.
        '''
        (tx, ty, c) = self.spl.tck
        c *= s
        self.vals = c

    def setPriorSmoothness(self, sigma):
        '''
        The smoothness sigma is proportional to sky-intensity units;
        small sigma leads to smooth sky.
        '''
        self.prior_smooth_sigma = sigma

    def evaluateGrid(self, xvals, yvals):
        return self.spl(xvals + self.x0, yvals + self.y0).T

    def addTo(self, mod, scale=1.):
        H, W = mod.shape
        S = self.evaluateGrid(np.arange(W), np.arange(H))
        mod += (S * scale)

    def getParamGrid(self):
        arr = np.array(self.vals)
        assert(len(arr) == (self.W * self.H))
        arr = arr.reshape(self.H, -1)
        return arr

    def getLogPrior(self):
        if self.prior_smooth_sigma is None:
            return 0.
        p = self.getParamGrid()
        sig = self.prior_smooth_sigma
        lnP = (-0.5 * np.sum((p[1:, :] - p[:-1, :])**2) / (sig**2) +
               -0.5 * np.sum((p[:, 1:] - p[:, :-1])**2) / (sig**2))
        return lnP

    def getLogPriorDerivatives(self):
        # FIXME -- we ignore frozenness!

        if self.prior_smooth_sigma is None:
            return None

        rA = []
        cA = []
        vA = []
        pb = []
        mub = []
        # columns are parameters
        # rows are prior terms

        #
        p = self.getParamGrid()
        sig = self.prior_smooth_sigma
        # print 'Param grid', p.shape
        # print 'len', len(p)
        II = np.arange(len(p.ravel())).reshape(p.shape)
        assert(p.shape == II.shape)

        dx = p[:, 1:] - p[:, :-1]
        dy = p[1:, :] - p[:-1, :]

        NX = len(dx.ravel())

        rA.append(np.arange(NX))
        cA.append(II[:, :-1].ravel())
        vA.append(np.ones(NX) / sig)
        pb.append(dx.ravel() / sig)
        # Not 100% certain of this...
        mub.append(np.zeros(NX))

        rA.append(np.arange(NX))
        cA.append(II[:, 1:].ravel())
        vA.append(-np.ones(NX) / sig)

        NY = len(dy.ravel())

        rA.append(NX + np.arange(NY))
        cA.append(II[:-1, :].ravel())
        vA.append(np.ones(NY) / sig)
        pb.append(dy.ravel() / sig)
        mub.append(np.zeros(NY))

        rA.append(NX + np.arange(NY))
        cA.append(II[1:, :].ravel())
        vA.append(-np.ones(NY) / sig)

        # print 'log prior chi: returning', len(rA), 'sets of terms'

        return (rA, cA, vA, pb, mub)

    def getParamDerivatives(self, *args):
        derivs = []
        tx, ty = self.spl.get_knots()
        print('Knots:')
        print('x', len(tx), tx)
        print('y', len(ty), ty)
        print('W,H', self.W, self.H)
        for i in self.getThawedParamIndices():
            #ix = i % self.W
            #iy = i // self.W
            derivs.append(False)
        return derivs

    def write_fits(self, filename, hdr=None, primhdr=None):
        tt = type(self)
        sky_type = '%s.%s' % (tt.__module__, tt.__name__)
        if hdr is None:
            import fitsio
            hdr = fitsio.FITSHDR()

        if primhdr is None:
            import fitsio
            primhdr = fitsio.FITSHDR()
        hdr.add_record(dict(name='SKY', value=sky_type,
                            comment='Sky class'))

        primhdr.add_record(dict(name='SKY', value=sky_type,
                                comment='Sky class'))
        # hdr.add_record(dict(name='SPL_ORD', value=self.order,
        #                    comment='Spline sky order'))
        # this writes all params as header cards
        #self.toFitsHeader(hdr, prefix='SKY_')

        #fits = fitsio.FITS(filename, 'rw')
        #fits.write(None, header=primhdr, clobber=True)
        #fits.write(self.c, header=hdr)
        # fits.close()
        from astrometry.util.fits import fits_table
        T = fits_table()
        T.gridw = np.atleast_1d(self.W).astype(np.int32)
        T.gridh = np.atleast_1d(self.H).astype(np.int32)
        T.xgrid = np.atleast_2d(self.xgrid).astype(np.int32)
        T.ygrid = np.atleast_2d(self.ygrid).astype(np.int32)
        T.x0 = np.atleast_1d(self.x0).astype(np.int32)
        T.y0 = np.atleast_1d(self.y0).astype(np.int32)
        gridvals = self.spl(self.xgrid, self.ygrid).T
        T.gridvals = np.array([gridvals]).astype(np.float32)
        T.order = np.atleast_1d(self.order).astype(np.uint8)
        assert(len(T) == 1)
        T.writeto(filename, header=hdr, primheader=primhdr)

    @classmethod
    def from_fits(cls, filename, header, row=0):
        from astrometry.util.fits import fits_table
        T = fits_table(filename)
        T = T[row]
        return cls.from_fits_row(T)

    @classmethod
    def from_fits_row(cls, Ti):
        sky = cls(Ti.xgrid, Ti.ygrid, Ti.gridvals, order=Ti.order)
        sky.shift(Ti.x0, Ti.y0)
        return sky


if __name__ == '__main__':
    W, H = 1024, 600
    NX, NY = 6, 9
    sig = 100.
    vals = np.random.normal(size=(NY, NX), scale=sig)

    vals[1, 0] = 10000.

    XX = np.linspace(0, W, NX)
    YY = np.linspace(0, H, NY)
    ss = SplineSky(XX, YY, vals)

    print('NP', ss.numberOfParams())
    print(ss.getParamNames())
    print(ss.getParams())

    X = np.zeros((H, W))
    ss.addTo(X)

    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    tx, ty, c = ss.spl.tck

    def plotsky(X):
        plt.clf()
        plt.imshow(X, interpolation='nearest', origin='lower')
        ax = plt.axis()
        plt.colorbar()
        for x in tx:
            plt.axhline(x, color='0.5')
        for y in ty:
            plt.axvline(y, color='0.5')

        H, W = X.shape
        # Find non-zero range
        for nzx0 in range(W):
            if not np.all(X[:, nzx0] == 0):
                break
        for nzx1 in range(W - 1, -1, -1):
            if not np.all(X[:, nzx1] == 0):
                break
        for nzy0 in range(H):
            if not np.all(X[nzy0, :] == 0):
                break
        for nzy1 in range(H - 1, -1, -1):
            if not np.all(X[nzy1, :] == 0):
                break
        plt.axhline(nzy0, color='y')
        plt.axhline(nzy1, color='y')
        plt.axvline(nzx0, color='y')
        plt.axvline(nzx1, color='y')
        plt.axis(ax)

    plotsky(X)
    plt.savefig('sky.png')

    S = ss.getStepSizes()
    p0 = ss.getParams()
    X0 = X.copy()
    for i, step in enumerate(S):
        ss.setParam(i, p0[i] + step)
        X[:, :] = 0.
        ss.addTo(X)
        ss.setParam(i, p0[i])
        dX = (X - X0) / step

        plotsky(dX)
        plt.savefig('sky-d%02i.png' % i)

        plt.clf()
        G = ss.getParamGrid()
        plt.imshow(G.T, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.savefig('sky-p%02i.png' % i)

        plt.clf()
        plt.imshow(X, interpolation='nearest', origin='lower')
        plt.colorbar()
        plt.savefig('sky-%02i.png' % i)

        break

    from tractor import *
    data = X.copy()
    tim = Image(data=data, invvar=np.ones_like(X) * (1. / sig**2),
                psf=NCircularGaussianPSF([1.], [1.]),
                wcs=NullWCS(), sky=ss,
                photocal=NullPhotoCal(), name='sky im')
    tractor = Tractor(images=[tim])
    ss.setPriorSmoothness(sig)

    print('Tractor', tractor)
    print('im params', tim.getParamNames())
    print('im step sizes', tim.getStepSizes())
    print('Initial:')
    print('lnLikelihood', tractor.getLogLikelihood())
    print('lnPrior', tractor.getLogPrior())
    print('lnProb', tractor.getLogProb())

    mod = tractor.getModelImage(0)

    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.savefig('mod0.png')

    tractor.optimize()

    print('After opt:')
    print('lnLikelihood', tractor.getLogLikelihood())
    print('lnPrior', tractor.getLogPrior())
    print('lnProb', tractor.getLogProb())

    mod = tractor.getModelImage(0)

    plt.clf()
    plt.imshow(mod, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.savefig('mod1.png')
