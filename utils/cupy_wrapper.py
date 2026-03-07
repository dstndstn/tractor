got = False
try:
    import cupy as cp
    if cp.cuda.is_available():
        got = True
    else:
        print('No Cupy/CUDA')
except:
    print('Using fake Cupy')
    import fake_cupy as cp
