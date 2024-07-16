import numpy as np
import cupy as cp
from math import pi

def lanczos_filter(order, x, out=None):
    x = np.atleast_1d(x)
    nz = np.logical_and(x != 0., np.logical_and(x < order, x > -order))
    nz = np.flatnonzero(nz)
    if out is None:
        out = np.zeros(x.shape, dtype=float)
    else:
        out[x <= -order] = 0.
        out[x >=  order] = 0.
    pinz = pi * x.flat[nz]
    out.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    return out

def gpu_lanczos_filter(order, x, out=None):
    x = cp.atleast_1d(x)
    nz = cp.logical_and(x != 0., cp.logical_and(x < order, x > -order))
    #nz = cp.flatnonzero(nz)
    if out is None:
        out = cp.zeros(x.shape, dtype=float)
    else:
        out[x <= -order] = 0.
        out[x >=  order] = 0.
    pinz = pi * x[nz]
    out[nz] = order * cp.sin(pinz) * cp.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    return out

def batch_correlate1d_cpu(a, b, axis=1, mode='constant'):
    #a = (z, m, n)
    #b = (y, x) 
    #First dimension = number of individual (m, n) , (x) arrays
    #mode = 'full' or 'constant'
    z,m,n = a.shape
    y,x = b.shape 
    if z != y:
        raise RuntimeError("Dimensions "+str(z)+" and "+str(y)+" do not match") 
    npad = (m-x)
    r = m+x-1
    nclip = r-m 
    if axis == 2:
        npad = (n-x)
        r = n+x-1
        nclip = r-n 
    if npad > 0:
        if npad % 2 == 0:
            padded_b = np.pad(b, [0, npad//2])
        else:
            padded_b = np.pad(b, [(0,0), (npad//2+1, npad//2)])
    else:
        padded_b = b
    f_a = np.fft.fft(a, r, axis=axis)
    f_b = np.fft.fft(padded_b, r, axis=1)
    if axis == 1:
        f_p = np.einsum("ijk,ij->ijk", f_a, np.conj(f_b))
    else:
        f_p = np.einsum("ijk,ik->ijk", f_a, np.conj(f_b))
    c = np.real(np.fft.fftshift(np.fft.ifft(f_p, axis=axis), axes=(axis)))
    if mode == 'full':
        return c
    if axis == 1:
        return c[:,nclip//2:-nclip//2,:]
    return c[:,:,nclip//2:-nclip//2]

def batch_correlate1d_gpu(a, b, axis=1, mode='constant'):
    #a = (z, m, n)
    #b = (y, x)
    #First dimension = number of individual (m, n) , (x) arrays
    #mode = 'full' or 'constant'
    z,m,n = a.shape
    y,x = b.shape
    if z != y:
        raise RuntimeError("Dimensions "+str(z)+" and "+str(y)+" do not match")
    npad = (m-x)
    r = m+x-1
    nclip = r-m
    if axis == 2:
        npad = (n-x)
        r = n+x-1
        nclip = r-n
    if npad > 0:
        if npad % 2 == 0:
            padded_b = cp.pad(b, [0, npad//2])
        else:
            padded_b = cp.pad(b, [(0,0), (npad//2+1, npad//2)])
    else:
        padded_b = b
    f_a = cp.fft.fft(a, r, axis=axis)
    f_b = cp.fft.fft(padded_b, r, axis=1)
    if axis == 1:
        f_p = cp.einsum("ijk,ij->ijk", f_a, cp.conj(f_b))
    else:
        f_p = cp.einsum("ijk,ik->ijk", f_a, cp.conj(f_b))
    c = cp.real(cp.fft.fftshift(cp.fft.ifft(f_p, axis=axis), axes=(axis)))
    if mode == 'full':
        return c
    if axis == 1:
        return c[:,nclip//2:-nclip//2,:]
    return c[:,:,nclip//2:-nclip//2]
