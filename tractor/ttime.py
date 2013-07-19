import os
import resource

def get_memusage():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    pgsize = resource.getpagesize()
    maxrss = (ru.ru_maxrss * pgsize / 1e6)
    #print 'shared memory size:', (ru.ru_ixrss / 1e6), 'MB'
    #print 'unshared memory size:', (ru.ru_idrss / 1e6), 'MB'
    #print 'unshared stack size:', (ru.ru_isrss / 1e6), 'MB'
    #print 'shared memory size:', ru.ru_ixrss
    #print 'unshared memory size:', ru.ru_idrss
    #print 'unshared stack size:', ru.ru_isrss
    mu = dict(maxrss=[maxrss, 'MB'])

    procfn = '/proc/%d/status' % os.getpid()
    try:
        t = open(procfn).readlines()
        d = dict([(line.split()[0][:-1], line.split()[1:]) for line in t])
        mu.update(d)
    except:
        pass

    return mu
    
def memusage():
    mu = get_memusage()
    print 'Memory usage:'
    print 'max rss:', mu['maxrss'], 'MB'
    for key in ['VmPeak', 'VmSize', 'VmRSS', 'VmData', 'VmStk'
                # VmLck, VmHWM, VmExe, VmLib, VmPTE
                ]:
        print key, ' '.join(mu.get(key, []))

class MemMeas(object):
    def __init__(self):
        self.mem0 = get_memusage()
    def format_diff(self, other):
        #keys = self.mem0.keys()
        #keys.sort()
        txt = []
        #for k in keys:
        for k in ['VmPeak', 'VmSize', 'VmRSS', 'VmData']:
            val,unit = self.mem0[k]
            if unit == 'kB':
                val = int(val, 10)
                val /= 1024.
                unit == 'MB'
                val = '%.0f' % val
            txt.append('%s: %s %s' % (k, val, unit))
        return ', '.join([] + txt)
        
class Time(object):
    @staticmethod
    def add_measurement(m):
        Time.measurements.append(m)
    measurements = []

    def __init__(self):
        import datetime
        from time import clock
        self.wall = datetime.datetime.now()
        #self.cpu = time.clock()
        self.cpu = clock()
        self.meas = [m() for m in Time.measurements]

    def __sub__(self, other):
        dwall = (self.wall - other.wall)
        # python2.7
        if hasattr(dwall, 'total_seconds'):
            dwall = dwall.total_seconds()
        else:
            dwall = (dwall.microseconds + (dwall.seconds + dwall.days * 24. * 3600.) * 1e6) / 1e6
        dcpu = (self.cpu - other.cpu)

        meas = [m.format_diff(om) for m,om in zip(self.meas, other.meas)]
        return ', '.join(['%f wall' % dwall, '%f cpu' % dcpu] + meas)


