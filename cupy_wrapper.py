got = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        got = True
    else:
        print('No Cupy/CUDA')
except:
    import fake_cupy as cp
