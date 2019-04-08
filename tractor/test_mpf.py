from __future__ import print_function

#from tractor.mp_fourier import *
from astrometry.util.miscutils import *
import numpy as np
from scipy.ndimage import correlate1d
import pylab as plt

#from tractor import mp_fourier
import mp_fourier

print('mpf:', dir(mp_fourier))

if False:
    F = np.zeros((10,10), np.float32)
    D = np.zeros(10, np.float64)
    D2 = np.zeros(10, np.float64)
    mp_fourier.myfunc_1(F)
    mp_fourier.myfunc_2(F)
    mp_fourier.myfunc_3(D, D2)
    mp_fourier.myfunc_4(F)

dx =  0.3
dy = -0.2

L = 3
Lx = lanczos_filter(L, np.arange(-L, L+1) + dx)
Ly = lanczos_filter(L, np.arange(-L, L+1) + dy)
Lx /= Lx.sum()
Ly /= Ly.sum()

xx,yy = np.meshgrid(np.arange(25), np.arange(25))
img = np.exp(-((xx - 12)**2 + (yy - 12)**2)/(2.*3**2)).astype(np.float32)

H,W = img.shape

sx     = correlate1d(img, Lx, axis=1, mode='constant')
outimg1 = correlate1d(sx,  Ly, axis=0, mode='constant')

plt.clf()
plt.imshow(outimg1, interpolation='nearest', origin='lower')
plt.savefig('corr-1.png')

assert(len(Lx) == 7)
assert(len(Ly) == 7)

work_corr7f = np.zeros((4096, 4096), np.float32)
work_corr7f = np.require(work_corr7f, requirements=['A'])

outimg2 = np.empty(img.shape, np.float32)
outimg2[:,:] = img
mp_fourier.correlate7f(outimg2, Lx, Ly, work_corr7f)

rms = np.sqrt(np.mean((outimg2 - outimg1)**2))
print('correlate7f:', rms)
assert(rms < 2e-8)

plt.clf()
plt.imshow(outimg2, interpolation='nearest', origin='lower')
plt.savefig('corr-2.png')

outimg = np.empty(img.shape, np.float32)
#print('img', img.dtype, 'outimg', outimg.dtype)

mp_fourier.correlate7f_inout(img, outimg, Lx, Ly, work_corr7f)

rms = np.sqrt(np.mean((outimg - outimg2)**2))
print('correlate7f_inout:', rms)
assert(rms < 1e-15)

plt.clf()
plt.imshow(outimg, interpolation='nearest', origin='lower')
plt.savefig('corr-3.png')

outimg = np.empty(img.shape, np.float32)
mp_fourier.lanczos_shift_3f(img, outimg, dx, dy, work_corr7f)

rms = np.sqrt(np.mean((outimg - outimg2)**2))
print('lanczos_shift_3f:', rms)
assert(rms < 1e-15)

plt.clf()
plt.imshow(outimg, interpolation='nearest', origin='lower')
plt.savefig('corr-4.png')

plt.clf()
plt.imshow(outimg-outimg1, interpolation='nearest', origin='lower')
plt.savefig('corr-5.png')

