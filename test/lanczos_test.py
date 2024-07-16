import numpy as np
import cupy as cp
import tractor.psf
import sys, time

nimages = 1000
imsize = 64

if len(sys.argv) > 1:
    nimages = int(sys.argv[1])
if len(sys.argv) > 2:
    imsize = int(sys.argv[2])

#First prime GPU
gi = cp.random.random((10,64,64))
gdx = cp.random.random(10)/2-0.25
gdy = cp.random.random(10)/2-0.25
gl = tractor.psf.lanczos_shift_image_batch_gpu(gi, gdx, gdy)

gi = cp.random.random((nimages, imsize, imsize))
gdx = cp.random.random(nimages)/2-0.25
gdy = cp.random.random(nimages)/2-0.25
imgs = gi.get()
dxs = gdx.get()
dys = gdy.get()
l1 = []
l2 = []
t = time.time()
for j in range(nimages):
    l1.append(tractor.psf.lanczos_shift_image(imgs[j], dxs[j], dys[j]))
cpu_time1 = time.time()-t
l1 = np.array(l1)

t = time.time()
for j in range(nimages):
    l2.append(tractor.psf.lanczos_shift_image(imgs[j], dxs[j], dys[j], force_python=True))
cpu_time2 = time.time()-t
l2 = np.array(l2)

t = time.time()
gl = tractor.psf.lanczos_shift_image_batch_gpu(gi, gdx, gdy)
gpu_time = time.time()-t

print(f'{nimages=}  {imsize=}')
print(f'{cpu_time1=}  {cpu_time2=}  {gpu_time=}')
if np.allclose(l1, gl.get(), atol=1.e-7):
    print ("All close 1")
else:
    print ("Not all close 1")

if np.allclose(l2, gl.get(), atol=1.e-7):
    print ("All close 2")
else:
    print ("Not all close 2")
