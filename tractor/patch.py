import numpy as np

from astrometry.util.miscutils import get_overlapping_region

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
        self.name = ''
        if patch is not None:
            try:
                H,W = patch.shape
                self.size = H*W
            except:
                pass

    @property
    def y1(self):
        return self.y0 + self.patch.shape[0]
    @property
    def x1(self):
        return self.x0 + self.patch.shape[1]
            
    def __str__(self):
        s = 'Patch: '
        name = getattr(self, 'name', '')
        if len(name):
            s += name + ' '
        s += 'origin (%i,%i) ' % (self.x0, self.y0)
        if self.patch is not None:
            (H,W) = self.patch.shape
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
        H,W = self.patch.shape
        if W == 0 or H == 0:
            return
        for x in range(W):
            if not np.all(self.patch[:,x] == 0):
                break
        x0 = x
        for x in range(W, 0, -1):
            if not np.all(self.patch[:,x-1] == 0):
                break
            if x <= x0:
                break
        x1 = x

        for y in range(H):
            if not np.all(self.patch[y,:] == 0):
                break
        y0 = y
        for y in range(H, 0, -1):
            if not np.all(self.patch[y-1,:] == 0):
                break
            if y <= y0:
                break
        y1 = y

        if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
            return

        self.patch = self.patch[y0:y1, x0:x1]
        H,W = self.patch.shape
        if H == 0 or W == 0:
            self.patch = None
        self.x0 += x0
        self.y0 += y0

    def overlapsBbox(self, bbox):
        ext = self.getExtent()
        (x0,x1,y0,y1) = ext
        (ox0,ox1,oy0,oy1) = bbox
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
        (x0,x1,y0,y1) = ext
        oext = other.getExtent()
        (ox0,ox1,oy0,oy1) = oext
        ix,ox = get_overlapping_region(ox0, ox1-1, x0, x1-1)
        iy,oy = get_overlapping_region(oy0, oy1-1, y0, y1-1)
        ix = slice(ix.start -  x0, ix.stop -  x0)
        iy = slice(iy.start -  y0, iy.stop -  y0)
        sub = self.patch[iy,ix]
        osub = other.patch[oy,ox]
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

    # for Cache
    #def size(self):
    #   (H,W) = self.patch.shape
    #   return H*W
    def copy(self):
        if self.patch is None:
            return Patch(self.x0, self.y0, None)
        return Patch(self.x0, self.y0, self.patch.copy())

    def getExtent(self, margin=0.):
        ''' Return (x0, x1, y0, y1) '''
        (h,w) = self.shape
        return (self.x0-margin, self.x0 + w + margin,
                self.y0-margin, self.y0 + h + margin)

    def getOrigin(self):
        return (self.x0,self.y0)
    def getPatch(self):
        return self.patch
    def getImage(self):
        return self.patch
    def getX0(self):
        return self.x0
    def getY0(self):
        return self.y0

    def clipTo(self, W, H):
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
        o0 = (self.x0, self.y0, self.patch.shape)
        if self.x0 < 0:
            self.patch = self.patch[:, -self.x0:]
            self.x0 = 0
        if self.y0 < 0:
            self.patch = self.patch[-self.y0:, :]
            self.y0 = 0
        # debug
        S = self.patch.shape
        if len(S) != 2:
            print 'clipTo: shape', self.patch.shape
            print 'original offset and patch shape:', o0
            print 'current offset and patch shape:', self.x0, self.y0, self.patch.shape

        (h,w) = self.patch.shape
        if (self.x0 + w) > W:
            self.patch = self.patch[:, :(W - self.x0)]
        if (self.y0 + h) > H:
            self.patch = self.patch[:(H - self.y0), :]

        assert(self.x0 >= 0)
        assert(self.y0 >= 0)
        (h,w) = self.shape
        assert(w <= W)
        assert(h <= H)
        assert(self.shape == self.patch.shape)
        return True


    #### WARNing, this function has not been tested
    def clipToRoi(self, x0,x1,y0,y1):
        if self.patch is None:
            return False
        if ((self.x0 >= x1) or (self.x1 <= x0) or
            (self.y0 >= y1) or (self.y1 <= y0)):
            # empty
            self.patch = None
            return False

        if self.x0 < x0:
            self.patch = self.patch[:, x0-self.x0:]
            self.x0 = x0
        if self.y0 < y0:
            self.patch = self.patch[(y0-self.y0):, :]
            self.y0 = y0
        (h,w) = self.shape
        if (self.x0 + w) > x1:
            self.patch = self.patch[:, :(x1 - self.x0)]
        if (self.y0 + h) > y1:
            self.patch = self.patch[:(y1 - self.y0), :]
        return True


    def getSlice(self, parent=None):
        if self.patch is None:
            return ([],[])
        (ph,pw) = self.patch.shape
        if parent is not None:
            (H,W) = parent.shape
            return (slice(np.clip(self.y0, 0, H), np.clip(self.y0+ph, 0, H)),
                    slice(np.clip(self.x0, 0, W), np.clip(self.x0+pw, 0, W)))
        return (slice(self.y0, self.y0+ph),
                slice(self.x0, self.x0+pw))

    def getPixelIndices(self, parent):
        if self.patch is None:
            return np.array([], np.int)
        (h,w) = self.shape
        (H,W) = parent.shape
        X,Y = np.meshgrid(np.arange(w), np.arange(h))
        return (Y.ravel() + self.y0) * W + (X.ravel() + self.x0)

    plotnum = 0

    def addTo(self, img, scale=1.):
        if self.patch is None:
            return
        (ih,iw) = img.shape
        (ph,pw) = self.shape
        (outx, inx) = get_overlapping_region(self.x0, self.x0+pw-1, 0, iw-1)
        (outy, iny) = get_overlapping_region(self.y0, self.y0+ph-1, 0, ih-1)
        if inx == [] or iny == []:
            return
        p = self.patch[iny,inx]
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

    def __getattr__(self, name):
        if name == 'shape':
            if self.patch is None:
                return (0,0)
            return self.patch.shape
        raise AttributeError('Patch: unknown attribute "%s"' % name)

    def __mul__(self, flux):
        if self.patch is None:
            return Patch(self.x0, self.y0, None)
        return Patch(self.x0, self.y0, self.patch * flux)
    def __div__(self, x):
        if self.patch is None:
            return Patch(self.x0, self.y0, None)
        return Patch(self.x0, self.y0, self.patch / x)

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

        (ph,pw) = self.patch.shape
        (ox0,oy0) = other.getX0(), other.getY0()
        (oh,ow) = other.shape

        # Find the union of the regions.
        ux0 = min(ox0, self.x0)
        uy0 = min(oy0, self.y0)
        ux1 = max(ox0 + ow, self.x0 + pw)
        uy1 = max(oy0 + oh, self.y0 + ph)

        p = np.zeros((uy1 - uy0, ux1 - ux0), dtype=otype)
        p[self.y0 - uy0 : self.y0 - uy0 + ph,
          self.x0 - ux0 : self.x0 - ux0 + pw] = self.patch

        psub = p[oy0 - uy0 : oy0 - uy0 + oh,
                 ox0 - ux0 : ox0 - ux0 + ow]
        op = getattr(psub, opname)
        op(other.getImage())
        return Patch(ux0, uy0, p)

    def __add__(self, other):
        return self.performArithmetic(other, '__iadd__')

    def __sub__(self, other):
        return self.performArithmetic(other, '__isub__')



