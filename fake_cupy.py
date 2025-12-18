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
    def _get(self):
        return _arrays[self.key]
    def get(self):
        return self._get()
    @property
    def shape(self):
        return self._get().shape
    @property
    def dtype(self):
        return self._get().dtype
    def max(self):
        x = self._get()
        return x.max()
    def astype(self, dt):
        x = self._get()
        x = x.astype(dt)
        return asarray(x)
    def __len__(self):
        x = self._get()
        return len(x)
    def __getitem__(self, key):
        x = self._get()
        x = x[key]
        return asarray(x)
    def __setitem__(self, key, value):
        x = self._get()
        ovalue = value
        if isinstance(value, FakeCupyArray):
            value = value.get()
        x[key] = value
        return ovalue
    def __sub__(self, other):
        x = self._get()
        if isinstance(other, FakeCupyArray):
            other = other.get()
        x = x - other
        return asarray(x)
    def __mul__(self, other):
        x = self._get()
        if isinstance(other, FakeCupyArray):
            other = other.get()
        x = x * other
        return asarray(x)
    def __truediv__(self, other):
        x = self._get()
        if isinstance(other, FakeCupyArray):
            other = other.get()
        x = x / other
        return asarray(x)
    def __pow__(self, other):
        x = self._get()
        if isinstance(other, FakeCupyArray):
            other = other.get()
        x = x ** other
        return asarray(x)

complex64 = np.complex64
float32 = np.float32

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

def hypot(x,y):
    if isinstance(x, FakeCupyArray):
        x = x.get()
    if isinstance(y, FakeCupyArray):
        y = y.get()
    h = np.hypot(x,y)
    return asarray(h)

def max(x):
    if isinstance(x, FakeCupyArray):
        x = x.get()
    m = np.max(x)
    return asarray(m)

fft = Duck()

def _wrap_one(f):
    def wrapped(x):
        if isinstance(x, FakeCupyArray):
            x = x.get()
        r = f(x)
        return asarray(r)
    return wrapped

def rfft2(x):
    if isinstance(x, FakeCupyArray):
        x = x.get()
    r = np.fft.rfft2(x)
    return asarray(r)

def rfftfreq(x):
    if isinstance(x, FakeCupyArray):
        x = x.get()
    r = np.fft.rfftfreq(x)
    return asarray(r)

#def rfftfreq(x):
#    if isinstance(x, FakeCupyArray):
#        x = x.get()
#    r = np.fft.rfftfreq(x)
#    return asarray(r)

fft.rfft2 = rfft2
fft.rfftfreq = rfftfreq

fft.fftfreq = _wrap_one(np.fft.fftfreq)

def zeros(shape, dtype):
    x = np.zeros(shape, dtype)
    return asarray(x)

if __name__ == '__main__':
    print(keystr(0))
    print(keystr(1))
    print(keystr(25))
    print(keystr(26))
    print(keystr(27))
    print(keystr(26*26))
    print(keystr(26*26+1))
    
