import numpy as np
try:
    import cupy as cp
except Exception as ex:
    pass
from math import pi
import gc

# Late-load kernel to avoid CuPy dependency on CPU
_mog_eval_kernel = None

def get_mog_eval_kernel():
    global _mog_eval_kernel
    if _mog_eval_kernel is None:
        import cupy as cp
        _mog_eval_kernel = cp.ElementwiseKernel(
            'T iv0, T iv1, T iv2, T scale, T mx, T my, T xx, T yy',
            'T out',
            '''
            T dx = xx - mx;
            T dy = yy - my;
            T dsq = iv0 * dx * dx + iv1 * dx * dy + iv2 * dy * dy;
            out = scale * exp(-0.5 * dsq);
            ''',
            'mog_eval_kernel'
        )
    return _mog_eval_kernel

def get_safe_chunk_size(nimg, h, w, vram_gb=40):
    """
    Heuristic to determine batch chunk size.
    vram_gb: available memory (40 or 80 for A100)
    """
    # Size of one float32 image in bytes
    img_bytes = h * w * 4
    # FFT multiplier: 1 (input) + 2 (complex output) + 2 (workspace) = ~5
    # We also do two steps (Horizontal/Vertical) and have sx intermediate.
    # A factor of 10-12 is a very safe ceiling.
    bytes_per_unit = img_bytes * 12

    available_bytes = vram_gb * 1024**3 * 0.7  # Use 70% of total

    chunk_size = int(available_bytes // bytes_per_unit)
    return max(1, min(nimg, chunk_size))

def lanczos_filter(order, x, out=None):
    x = np.atleast_1d(x)
    nz = np.logical_and(x != 0., np.logical_and(x < order, x > -order))
    nz = np.flatnonzero(nz)
    if out is None:
        out = np.zeros(x.shape, dtype=np.float32)
    else:
        out[x <= -order] = 0.
        out[x >=  order] = 0.
    pinz = pi * x.flat[nz]
    out.flat[nz] = order * np.sin(pinz) * np.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    return out

# Fused Lanczos Filter to save memory
gpu_lanczos_kernel = cp.ElementwiseKernel(
    'T x, int32 order',
    'T out',
    '''
    if (x == 0) {
        out = 1.0;
    } else if (x <= -order || x >= order) {
        out = 0.0;
    } else {
        T pi_x = 3.141592653589793 * x;
        out = (order * sin(pi_x) * sin(pi_x / order)) / (pi_x * pi_x);
    }
    ''',
    'gpu_lanczos_kernel'
)

def gpu_lanczos_filter(order, x, out=None):
    x = x.astype(cp.float32)
    out = cp.empty(x.shape, dtype=cp.float32)
    gpu_lanczos_kernel(x, order, out)
    """
    x = cp.atleast_1d(x)
    nz = cp.logical_and(x != 0., cp.logical_and(x < order, x > -order))
    #nz = cp.flatnonzero(nz)
    if out is None:
        out = cp.zeros(x.shape, dtype=cp.float32)
    else:
        out[x <= -order] = 0.
        out[x >=  order] = 0.
    pinz = pi * x[nz]
    out[nz] = order * cp.sin(pinz) * cp.sin(pinz / order) / (pinz**2)
    out[x == 0] = 1.
    """
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

    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    mempool = cp.get_default_memory_pool()
    used_bytes = mempool.used_bytes()
    tot_bytes = mempool.total_bytes()
    #print (f'Correlate1d GPU {free_mem=} {total_mem=}; This mempool {used_bytes=} {tot_bytes=}')
    #print ("Sizes:", a.size, padded_b.size)

    """
    f_a = cp.fft.fft(a, r, axis=axis)
    f_b = cp.fft.fft(padded_b, r, axis=1)
    del padded_b
    if axis == 1:
        f_p = cp.einsum("ijk,ij->ijk", f_a, cp.conj(f_b))
    else:
        f_p = cp.einsum("ijk,ik->ijk", f_a, cp.conj(f_b))
    del f_a, f_b
    c = cp.real(cp.fft.fftshift(cp.fft.ifft(f_p, axis=axis), axes=(axis)))
    del f_p
    """

    # USE RFFT: This keeps the frequency domain half-sized (~3.25 GB instead of 6.5 GB)
    f_a = cp.fft.rfft(a, r, axis=axis)
    f_b = cp.fft.rfft(padded_b, r, axis=1)
    del padded_b

    # Correlation is f_a * conj(f_b)
    # rfft output is Hermitian symmetric, so conj() works fine here
    if axis == 1:
        f_p = cp.einsum("ijk,ij->ijk", f_a, cp.conj(f_b))
    else:
        f_p = cp.einsum("ijk,ik->ijk", f_a, cp.conj(f_b))
    del f_a, f_b

    # irfft automatically knows the output is real and handles the symmetry
    # Use the 'n' parameter to ensure the output length matches 'r'
    c = cp.fft.irfft(f_p, n=r, axis=axis)
    del f_p
    # Shift and Clip
    c = cp.fft.fftshift(c, axes=(axis))

    if c.size > 5.e+8/8:
        #Make sure memory is freed
        #gc.collect()
        mpool = cp.get_default_memory_pool()
        mpool.free_all_blocks()
    if mode == 'full':
        return c
    if axis == 1:
        return c[:,nclip//2:-nclip//2,:]
    return c[:,:,nclip//2:-nclip//2]
