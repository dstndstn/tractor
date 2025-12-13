import numpy as np

class Duck(object):
    pass

cuda = Duck()
cuda.runtime = Duck()

def memgetinfo():
    # free, tot
    mem = 50 * 1024**3
    return mem, mem
cuda.runtime.memGetInfo = memgetinfo

cuda.memory = Duck()

class OutOfMemoryError(RuntimeError):
    pass

cuda.memory.OutOfMemoryError = OutOfMemoryError

_arrays = {}
_array_key = 0

def keystr(i):
    s = ''
    c0 = ord('a')
    while i >= 26:
        digit = i % 26
        i //= 26
        s = chr(c0 + digit) + s
    s = chr(c0 + i) + s
    return s

class FakeCupyArray(object):
    def __init__(self, key):
        self.key = key
    def __str__(self):
        return 'FakeCupyArray: %s' % self.key
    def __repr__(self):
        return 'FakeCupyArray("%s")' % self.key

def asarray(x):
    n = np.asarray(x)
    global _array_key
    global _arrays
    key = keystr(_array_key)
    a = FakeCupyArray(key)
    _array_key += 1
    _arrays[key] = n
    return a

cuda.asarray = asarray

if __name__ == '__main__':
    print(keystr(0))
    print(keystr(1))
    print(keystr(25))
    print(keystr(26))
    print(keystr(27))
    print(keystr(26*26))
    print(keystr(26*26+1))
    
