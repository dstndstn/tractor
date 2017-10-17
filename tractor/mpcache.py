from __future__ import print_function
from multiprocessing import Manager
from multiprocessing.managers import BaseManager
#from .cache import Cache
from cache import Cache


class CacheManager(BaseManager):
    pass


CacheManager.register('Cache', Cache, exposed=('get', 'put', 'printStats'))


def createManager():
    manager = CacheManager()
    manager.start()
    return manager


def createCache(**kwargs):
    man = createManager()
    #kwargs = {}
    # if maxsize is not None:
    #		kwargs['maxsize'] = maxsize
    cache = man.Cache(**kwargs)
    return cache


def testProcess(cache):
    import time
    import os
    for i in range(10):
        time.sleep(1)
        print(os.getpid(), 'get', i, cache.get(i, None))
        print(os.getpid(), 'put', i)
        cache.put(i, i**3)


if __name__ == '__main__':
    from multiprocessing import Process

    cache = createCache()
    pp = []
    for i in range(5):
        p = Process(target=testProcess, args=(cache,))
        p.start()
        pp.append(p)

    for p in pp:
        p.join()
