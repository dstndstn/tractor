import numpy as np

#from astrometry.util.miscutils import get_overlapping_region

cdef intmax(int a, int b):
    if a >= b:
        return a
    return b
cdef intmin(int a, int b):
    if a <= b:
        return a
    return b
cdef intclip(int a, int lo, int hi):
    if a < lo:
        return lo
    if a > hi:
        return hi
    return a

def get_overlapping_region(int xlo, int xhi, int xmin, int xmax):
    '''
    Given a range of integer coordinates that you want to, eg, cut out
    of an image, [xlo, xhi], and bounds for the image [xmin, xmax],
    returns the range of coordinates that are in-bounds, and the
    corresponding region within the desired cutout.

    For example, say you have an image of shape H,W and you want to
    cut out a region of halfsize "hs" around pixel coordinate x,y, but
    so that coordinate x,y is centered in the cutout even if x,y is
    close to the edge.  You can do:

    cutout = np.zeros((hs*2+1, hs*2+1), img.dtype)
    iny,outy = get_overlapping_region(y-hs, y+hs, 0, H-1)
    inx,outx = get_overlapping_region(x-hs, x+hs, 0, W-1)
    cutout[outy,outx] = img[iny,inx]
    
    '''
    cdef int xloclamp, xhiclamp, Xlo, Xhi

    if xlo > xmax or xhi < xmin or xlo > xhi or xmin > xmax:
        return ([], [])

    assert(xlo <= xhi)
    assert(xmin <= xmax)

    xloclamp = intmax(xlo, xmin)
    Xlo = xloclamp - xlo

    xhiclamp = intmin(xhi, xmax)
    Xhi = Xlo + (xhiclamp - xloclamp)

    #print 'xlo, xloclamp, xhiclamp, xhi', xlo, xloclamp, xhiclamp, xhi
    assert(xloclamp >= xlo)
    assert(xloclamp >= xmin)
    assert(xloclamp <= xmax)
    assert(xhiclamp <= xhi)
    assert(xhiclamp >= xmin)
    assert(xhiclamp <= xmax)
    #print 'Xlo, Xhi, (xmax-xmin)', Xlo, Xhi, xmax-xmin
    assert(Xlo >= 0)
    assert(Xhi >= 0)
    assert(Xlo <= (xhi-xlo))
    assert(Xhi <= (xhi-xlo))

    return (slice(xloclamp, xhiclamp+1), slice(Xlo, Xhi+1))


class ModelMask(object):
    def __init__(self, x0, y0, *args):
        '''
        ModelMask(x0, y0, w, h)
        ModelMask(x0, y0, mask)

        *mask* is assumed to be a binary image, True for pixels we're
         interested in.
        '''
        self.x0 = x0
        self.y0 = y0
        if len(args) == 2:
            self.w, self.h = args
            self.mask = None
        elif len(args) == 1:
            self.mask = args[0]
            self.h, self.w = self.mask.shape
        else:
            raise ValueError('Wrong number of arguments')

    def __str__(self):
        s = ('ModelMask: origin (%i,%i), w=%i, h=%i' %
             (self.x0, self.y0, self.w, self.h))
        if self.mask is None:
            return s
        return s + ', has mask'

    def __repr__(self):
        if self.mask is None:
            return ('ModelMask(%i,%i, w=%i, h=%i)' %
                    (self.x0, self.y0, self.w, self.h))
        else:
            return ('ModelMask(%i,%i, mask of shape w=%i, h=%i)' %
                    (self.x0, self.y0, self.w, self.h))

    @staticmethod
    def fromExtent(x0, x1, y0, y1):
        return ModelMask(x0, y0, x1 - x0, y1 - y0)

    @property
    def shape(self):
        return (self.h, self.w)

    @property
    def x1(self):
        return self.x0 + self.w

    @property
    def y1(self):
        return self.y0 + self.h

    @property
    def extent(self):
        return (self.x0, self.x0 + self.w, self.y0, self.y0 + self.h)

# Adds two patches, handling the case when one is None


def add_patches(pa, pb):
    p = pa
    if pb is not None:
        if p is None:
            p = pb
        else:
            p += pb
    return p


class Patch(object):
    '''
    An image patch; a subimage.  In the Tractor we use these to hold
    synthesized (ie, model) images.  The patch is a rectangular grid
    of pixels and it knows its offset (2-d position) in some larger
    image.

    This class overloads arithmetic operations (like add and multiply)
    relevant to synthetic image patches.
    '''

    def __init__(self, x0, y0, patch):
        self.x0 = x0
        self.y0 = y0
        self.patch = patch
        self.patch_gpu = None
        self.name = ''
        if patch is not None:
            try:
                H, W = patch.shape
                self.size = H * W
            except:
                pass

    @property
    def y1(self):
        return self.y0 + self.patch.shape[0]

    @property
    def x1(self):
        return self.x0 + self.patch.shape[1]

    def extent(self):
        (h, w) = self.shape
        return (self.x0, self.x0 + w, self.y0, self.y0 + h)

    def __str__(self):
        s = 'Patch: '
        name = getattr(self, 'name', '')
        if len(name):
            s += name + ' '
        s += 'origin (%i,%i) ' % (self.x0, self.y0)
        if self.patch is not None:
            (H, W) = self.patch.shape
            s += 'size (%i x %i)' % (W, H)
        else:
            s += '(no image)'
        return s

    def set(self, other):
        self.x0 = other.x0
        self.y0 = other.y0
        self.patch = other.patch
        self.name = other.name

    def trimToNonZero(self):
        if self.patch is None:
            return
        H, W = self.patch.shape
        if W == 0 or H == 0:
            return
        for x in range(W):
            if not np.all(self.patch[:, x] == 0):
                break
        x0 = x
        for x in range(W, 0, -1):
            if not np.all(self.patch[:, x - 1] == 0):
                break
            if x <= x0:
                break
        x1 = x

        for y in range(H):
            if not np.all(self.patch[y, :] == 0):
                break
        y0 = y
        for y in range(H, 0, -1):
            if not np.all(self.patch[y - 1, :] == 0):
                break
            if y <= y0:
                break
        y1 = y

        if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
            return

        self.patch = self.patch[y0:y1, x0:x1]
        H, W = self.patch.shape
        if H == 0 or W == 0:
            self.patch = None
        self.x0 += x0
        self.y0 += y0

    def overlapsBbox(self, bbox):
        ext = self.getExtent()
        (x0, x1, y0, y1) = ext
        (ox0, ox1, oy0, oy1) = bbox
        if x0 >= ox1 or ox0 >= x1 or y0 >= oy1 or oy0 >= y1:
            return False
        return True

    def hasBboxOverlapWith(self, other):
        oext = other.getExtent()
        return self.overlapsBbox(oext)

    def hasNonzeroOverlapWith(self, other):
        if not self.hasBboxOverlapWith(other):
            return False
        ext = self.getExtent()
        (x0, x1, y0, y1) = ext
        oext = other.getExtent()
        (ox0, ox1, oy0, oy1) = oext
        ix, ox = get_overlapping_region(ox0, ox1 - 1, x0, x1 - 1)
        iy, oy = get_overlapping_region(oy0, oy1 - 1, y0, y1 - 1)
        ix = slice(ix.start - x0, ix.stop - x0)
        iy = slice(iy.start - y0, iy.stop - y0)
        sub = self.patch[iy, ix]
        osub = other.patch[oy, ox]
        assert(sub.shape == osub.shape)
        return np.sum(sub * osub) > 0.

    def getNonZeroMask(self):
        nz = (self.patch != 0)
        return Patch(self.x0, self.y0, nz)

    def __repr__(self):
        return str(self)

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def copy(self):
        if self.patch is None:
            return Patch(self.x0, self.y0, None)
        return Patch(self.x0, self.y0, self.patch.copy())

    def getExtent(self, margin=0):
        '''
        Return (x0, x1, y0, y1) of the region covered by this patch;
        NON-inclusive upper bounds, ie, [x0, x1), [y0, y1).
        '''
        (h, w) = self.shape
        return (self.x0 - margin, self.x0 + w + margin,
                self.y0 - margin, self.y0 + h + margin)

    @property
    def extent(self):
        (h, w) = self.shape
        return (self.x0, self.x0 + w, self.y0, self.y0 + h)

    @property
    def shape(self):
        if self.patch is None:
            return 0,0
        return self.patch.shape

    def getOrigin(self):
        return (self.x0, self.y0)

    def getPatch(self, use_gpu=False):
        if use_gpu:
            if self.patch_gpu is None:
                import cupy as cp
                self.patch_gpu = cp.asarray(self.patch)
            return self.patch_gpu
        return self.patch

    def getImage(self, use_gpu=False):
        if use_gpu:
            if self.patch_gpu is None:
                import cupy as cp
                self.patch_gpu = cp.asarray(self.patch)
            return self.patch_gpu
        return self.patch

    def getX0(self):
        return self.x0

    def getY0(self):
        return self.y0

    def clipTo(self, W, H, use_gpu=False):
        if self.patch is None:
            return False
        if self.x0 >= W:
            # empty
            self.patch = None
            return False
        if self.y0 >= H:
            self.patch = None
            return False
        # debug
        #o0 = (self.x0, self.y0, self.patch.shape)
        if self.x0 < 0:
            self.patch = self.patch[:, -self.x0:]
            if use_gpu:
                self.patch_gpu = self.patch_gpu[:, -self.x0:]
            self.x0 = 0
        if self.y0 < 0:
            self.patch = self.patch[-self.y0:, :]
            if use_gpu:
                self.patch_gpu = self.patch_gpu[-self.y0:, :]
            self.y0 = 0
        # debug
        # S = self.patch.shape
        # if len(S) != 2:
        #     print 'clipTo: shape', self.patch.shape
        #     print 'original offset and patch shape:', o0
        #     print 'current offset and patch shape:', self.x0, self.y0, self.patch.shape

        (h, w) = self.patch.shape
        if (self.x0 + w) > W:
            self.patch = self.patch[:, :(W - self.x0)]
            if use_gpu:
                self.patch_gpu = self.patch_gpu[:, :(W - self.x0)]
        if (self.y0 + h) > H:
            self.patch = self.patch[:(H - self.y0), :]
            if use_gpu:
                self.patch_gpu = self.patch_gpu[:(H - self.y0), :]

        assert(self.x0 >= 0)
        assert(self.y0 >= 0)
        (h, w) = self.shape
        assert(w <= W)
        assert(h <= H)
        assert(self.shape == self.patch.shape)
        return True

    def clipToRoi(self, x0, x1, y0, y1):
        if self.patch is None:
            return False
        if ((self.x0 >= x1) or (self.x1 <= x0) or
                (self.y0 >= y1) or (self.y1 <= y0)):
            # empty
            self.patch = None
            return False

        if self.x0 < x0:
            self.patch = self.patch[:, x0 - self.x0:]
            self.x0 = x0
        if self.y0 < y0:
            self.patch = self.patch[(y0 - self.y0):, :]
            self.y0 = y0
        (h, w) = self.shape
        if (self.x0 + w) > x1:
            self.patch = self.patch[:, :(x1 - self.x0)]
        if (self.y0 + h) > y1:
            self.patch = self.patch[:(y1 - self.y0), :]
        return True

    def getSlice(self, parent=None):
        cdef int ph, pw, H, W

        if self.patch is None:
            return ([], [])
        (ph, pw) = self.patch.shape
        if parent is not None:
            (H, W) = parent.shape
            return (slice(intclip(self.y0, 0, H), intclip(self.y0 + ph, 0, H)),
                    slice(intclip(self.x0, 0, W), intclip(self.x0 + pw, 0, W)))
        return (slice(self.y0, self.y0 + ph),
                slice(self.x0, self.x0 + pw))

    def getSlices(self, shape):
        '''
        shape = (H,W).

        Returns (spatch, sparent), slices that yield the overlapping regions
        in this Patch and the given image.
        '''
        (ph, pw) = self.shape
        (ih, iw) = shape
        (outx, inx) = get_overlapping_region(
            self.x0, self.x0 + pw - 1, 0, iw - 1)
        (outy, iny) = get_overlapping_region(
            self.y0, self.y0 + ph - 1, 0, ih - 1)
        if inx == [] or iny == []:
            return (slice(0, 0), slice(0, 0)), (slice(0, 0), slice(0, 0))
        return (iny, inx), (outy, outx)

    def getPixelIndicesGPU(self, parent, dtype=np.int32):
        import cupy as cp
        return cp.asarray(self.getPixelIndices(parent, dtype))

    def getPixelIndices(self, parent, dtype=np.int32, use_gpu=False):
        if use_gpu:
            import cupy as cp
            if self.patch is None:
                return cp.array([], dtype)
            (h, w) = self.shape
            (H, W) = parent.shape
            return ( (cp.arange(w, dtype=dtype) + dtype(self.x0))[cp.newaxis, :] +
                ((cp.arange(h, dtype=dtype) + dtype(self.y0)) * dtype(W))[:, cp.newaxis]).ravel()
        if self.patch is None:
            return np.array([], dtype)
        (h, w) = self.shape
        (H, W) = parent.shape
        return ( (np.arange(w, dtype=dtype) + dtype(self.x0))[np.newaxis, :] +
                ((np.arange(h, dtype=dtype) + dtype(self.y0)) * dtype(W))[:, np.newaxis]).ravel()

    plotnum = 0

    def addTo(self, img, scale=1.):
        if self.patch is None:
            return
        (ih, iw) = img.shape
        (ph, pw) = self.shape
        (outx, inx) = get_overlapping_region(
            self.x0, self.x0 + pw - 1, 0, iw - 1)
        (outy, iny) = get_overlapping_region(
            self.y0, self.y0 + ph - 1, 0, ih - 1)
        if inx == [] or iny == []:
            return
        p = self.patch[iny, inx]
        img[outy, outx] += p * scale

        # if False:
        #   tmpimg = np.zeros_like(img)
        #   tmpimg[outy,outx] = p * scale
        #   plt.clf()
        #   plt.imshow(tmpimg, interpolation='nearest', origin='lower')
        #   plt.hot()
        #   plt.colorbar()
        #   fn = 'addto-%03i.png' % Patch.plotnum
        #   plt.savefig(fn)
        #   print 'Wrote', fn
        #
        #   plt.clf()
        #   plt.imshow(p, interpolation='nearest', origin='lower')
        #   plt.hot()
        #   plt.colorbar()
        #   fn = 'addto-%03i-p.png' % Patch.plotnum
        #   plt.savefig(fn)
        #   print 'Wrote', fn
        #
        #   Patch.plotnum += 1

    # Implement *=, /= for numeric types
    def __imul__(self, f):
        if self.patch is not None:
            self.patch *= f
        return self

    def __idiv__(self, f):
        if self.patch is not None:
            self.patch /= f
        return self

    # Implement *, / for numeric types
    def __mul__(self, f):
        if self.patch is None:
            return Patch(self.x0, self.y0, None)
        return Patch(self.x0, self.y0, self.patch * f)

    def __div__(self, f):
        if self.patch is None:
            return Patch(self.x0, self.y0, None)
        return Patch(self.x0, self.y0, self.patch / f)

    def performArithmetic(self, other, opname, otype=float):
        assert(isinstance(other, Patch))
        if (self.x0 == other.getX0() and self.y0 == other.getY0() and
                self.shape == other.shape):
            assert(self.x0 == other.getX0())
            assert(self.y0 == other.getY0())
            assert(self.shape == other.shape)
            if self.patch is None or other.patch is None:
                return Patch(self.x0, self.y0, None)
            pcopy = self.patch.copy()
            op = getattr(pcopy, opname)
            return Patch(self.x0, self.y0, op(other.patch))

        (ph, pw) = self.patch.shape
        (ox0, oy0) = other.getX0(), other.getY0()
        (oh, ow) = other.shape

        # Find the union of the regions.
        ux0 = min(ox0, self.x0)
        uy0 = min(oy0, self.y0)
        ux1 = max(ox0 + ow, self.x0 + pw)
        uy1 = max(oy0 + oh, self.y0 + ph)

        # Set the "self" portion of the union
        p = np.zeros((uy1 - uy0, ux1 - ux0), dtype=otype)
        p[self.y0 - uy0: self.y0 - uy0 + ph,
          self.x0 - ux0: self.x0 - ux0 + pw] = self.patch

        # Get a slice for the "other"'s portion of the union
        psub = p[oy0 - uy0: oy0 - uy0 + oh,
                 ox0 - ux0: ox0 - ux0 + ow]
        # Perform the in-place += or -= operation on "psub"
        op = getattr(psub, opname)
        op(other.getImage())
        return Patch(ux0, uy0, p)

    def __add__(self, other):
        return self.performArithmetic(other, '__iadd__')

    def __sub__(self, other):
        return self.performArithmetic(other, '__isub__')
