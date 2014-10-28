import multiprocessing as mp
import multiprocessing.queues
import multiprocessing.pool
import os

#
# In Python 2.7 (and 2.6):
#
# Pool has an _inqueue (_quick_put) and _outqueue (_quick_get)
# and _taskqueue
#
# map() etc add themselves to cache[jobid] = self
# and place work on the task queue.
#
# _handle_tasks pulls tasks from the _taskqueue and puts them
#     on the _inqueue.
#
# _handle_results pulls results from the _outqueue (job,i,obj)
#     and calls cache[job].set(i, obj)
#     (cache[job] is an ApplyResult / MapResult, etc.)
#
# worker threads run the worker() function:
#   run initializer
#   while true:
#     pull task from inqueue
#     job,i,func,arg,kwargs = task
#     put (job,i,result) on outqueue

# _inqueue,_outqueue are SimpleQueue (queues.py)
# get->recv=_reader.recv and put->send=_writer.send
# _reader,_writer = Pipe(duplex=False)

# Pipe (connection.py)
# uses os.pipe() with a _multiprocessing.Connection()
# on each fd.

# _multiprocessing = /u32/python/src/Modules/_multiprocessing/pipe_connection.c
# -> connection.h : send() is connection_send_obj()
# which uses pickle.
#
# Only _multiprocessing/socket_connection.c is used on non-Windows platforms.

import _multiprocessing
import cPickle as pickle
#import pickle
import time

class DebugConnection():
    def stats(self):
        #return ('pickled %i objs, %i bytes, %g s CPU; unpickled %i objs, %i bytes, %g s CPU' %
        return dict(pickle_objs = self.pobjs,
                    pickle_bytes = self.pbytes,
                    pickle_megabytes = 1e-6 * self.pbytes,
                    pickle_cputime = self.ptime,
                    unpickle_objs = self.upobjs,
                    unpickle_bytes = self.upbytes,
                    unpickle_megabytes = 1e-6 * self.upbytes,
                    unpickle_cputime = self.uptime)
    
    def __init__(self, fd, writable=True, readable=True):
        self.real = _multiprocessing.Connection(fd, writable=writable, readable=readable)
        self.ptime = 0.
        self.uptime = 0.
        self.pbytes = 0
        self.upbytes = 0
        self.pobjs = 0
        self.upobjs = 0
    def poll(self):
        return self.real.poll()
    def recv(self):
        # read string length + string
        # unpickle.loads()
        # obj = self.real.recv()
        bytes = self.real.recv_bytes()
        t0 = time.time()
        obj = pickle.loads(bytes)
        dt = time.time() - t0
        #print 'unpickling took', dt
        self.upbytes += len(bytes)
        self.uptime += dt
        self.upobjs += 1
        return obj

    def send(self, obj):
        # pickle obj to string (dumps())
        # write string length (u32 network byte order) + string
        # return self.real.send(obj)
        t0 = time.time()
        s = pickle.dumps(obj, -1)
        dt = time.time() - t0
        self.pbytes += len(s)
        self.ptime += dt
        self.pobjs += 1
        #print '-->', len(s), 'bytes'
        #print 'pickling took', dt
        return self.real.send_bytes(s)

    def close(self):
        return self.real.close()

def DebugPipe():
    fd1, fd2 = os.pipe()
    c1 = DebugConnection(fd1, writable=False)
    c2 = DebugConnection(fd2, readable=False)
    return c1,c2

from multiprocessing.queues import Lock

class DebugSimpleQueue(mp.queues.SimpleQueue):
    def stats(self):
        S1 = self._reader.stats()
        S2 = self._writer.stats()
        return dict([(k, S1[k]+S2[k]) for k in S1.keys()])

    def __init__(self):
        self._reader, self._writer = DebugPipe()
        self._rlock = Lock()
        self._wlock = Lock()
        self._make_methods()

import threading
import Queue
from multiprocessing import Process, cpu_count, TimeoutError
from multiprocessing.util import Finalize, debug
from multiprocessing.pool import RUN, CLOSE, TERMINATE

def debug_worker(inqueue, outqueue, initializer=None, initargs=(),
                 maxtasks=None):
    print 'debug_worker()'
    assert maxtasks is None or (type(maxtasks) == int and maxtasks > 0)
    put = outqueue.put
    get = inqueue.get
    if hasattr(inqueue, '_writer'):
        inqueue._writer.close()
        outqueue._reader.close()

    if initializer is not None:
        initializer(*initargs)

    completed = 0
    while maxtasks is None or (maxtasks and completed < maxtasks):
        #t0 = time.time()
        try:
            task = get()
        except (EOFError, IOError):
            debug('worker got EOFError or IOError -- exiting')
            break

        if task is None:
            debug('worker got sentinel -- exiting')
            break

        job, i, func, args, kwds = task
        t1 = time.time()
        quitnow = False
        try:
            success,val = (True, func(*args, **kwds))
        except Exception as e:
            success,val = (False, e)
            print 'debug_worker: caught', e
        except KeyboardInterrupt as e:
            success,val = (False, e)
            print 'debug_worker: caught ctrl-C', e
            quitnow = True
        t2 = time.time()
        dt = t2 - t1
        put((job, i, (dt,(success,val))))
        #t3 = time.time()
        #print 'worker: get task', (t1-t0), 'run', (t2-t1), 'result', (t3-t2)
        completed += 1
        if quitnow:
            break
    debug('worker exiting after %d tasks' % completed)

        
def debug_handle_results(outqueue, get, cache, beancounter):
    thread = threading.current_thread()
    while 1:
        try:
            task = get()
        except (IOError, EOFError):
            debug('result handler got EOFError/IOError -- exiting')
            return
        if thread._state:
            assert thread._state == TERMINATE
            debug('result handler found thread._state=TERMINATE')
            break
        if task is None:
            debug('result handler got sentinel')
            break
        job, i, obj = task
        dt,obj = obj
        try:
            cache[job]._set(i, obj)
        except KeyError:
            pass
        beancounter.add_cpu(dt)

    while cache and thread._state != TERMINATE:
        try:
            task = get()
        except (IOError, EOFError):
            debug('result handler got EOFError/IOError -- exiting')
            return

        if task is None:
            debug('result handler ignoring extra sentinel')
            continue
        job, i, obj = task
        dt,obj = obj
        try:
            cache[job]._set(i, obj)
        except KeyError:
            pass
        beancounter.add_cpu(dt)

    if hasattr(outqueue, '_reader'):
        debug('ensuring that outqueue is not full')
        # If we don't make room available in outqueue then
        # attempts to add the sentinel (None) to outqueue may
        # block.  There is guaranteed to be no more than 2 sentinels.
        try:
            for i in range(10):
                if not outqueue._reader.poll():
                    break
                get()
        except (IOError, EOFError):
            pass
    debug('result handler exiting: len(cache)=%s, thread._state=%s',
          len(cache), thread._state)


from multiprocessing.synchronize import Lock

class BeanCounter(object):
    def __init__(self):
        self.cpu = 0.
        self.lock = Lock()
    ### LOCKING
    def add_cpu(self, dt):
        self.lock.acquire()
        try:
            self.cpu += dt
        finally:
            self.lock.release()
    def get_cpu(self):
        self.lock.acquire()
        try:
            return self.cpu
        finally:
            self.lock.release()
    def __str__(self):
        return 'CPU time: %g s' % self.get_cpu()

class DebugPoolMeas(object):
    def __init__(self, pool):
        self.pool = pool
    def __call__(self):
        class FormatDiff(object):
            def __init__(self, pool):
                self.pool = pool
                self.t0 = self.now()
            def format_diff(self, other):
                t1 = self.t0
                t0 = other.t0
                return ('%.3f s worker CPU, pickled %i/%i objs, %g/%g MB' %
                        tuple(t1[k] - t0[k] for k in [
                        'worker_cpu', 'pickle_objs', 'unpickle_objs',
                        'pickle_megabytes', 'unpickle_megabytes']))
            def now(self):
                stats = self.pool.get_pickle_traffic()
                stats.update(worker_cpu = self.pool.get_worker_cpu())
                return stats
        return FormatDiff(self.pool)

class DebugProcess(mp.process.Process):
    def run(self):
        try:
            print 'DebugProcess.run()'
            super(DebugProcess, self).run()
        except KeyboardInterrupt:
            print 'DebugProcess caught KeyboardInterrupt.'
            raise
        except:
            print 'DebugProcess: exception:'
            import traceback
            traceback.print_exc()
    
class DebugPool(mp.pool.Pool):
    def _setup_queues(self):
        self._inqueue = DebugSimpleQueue()
        self._outqueue = DebugSimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def get_pickle_traffic_string(self):
        S = self.get_pickle_traffic()
        return (('  pickled %i objs, %g MB, using %g s CPU\n' +
                 'unpickled %i objs, %g MB, using %g s CPU') %
                 (S[k] for k in [
                     'pickle_objs', 'pickle_megabytes', 'pickle_cputime',
                     'unpickle_objs', 'unpickle_megabytes', 'unpickle_cputime']))

    def get_pickle_traffic(self):
        S1 = self._inqueue.stats()
        S2 = self._outqueue.stats()
        return dict([(k, S1[k]+S2[k]) for k in S1.keys()])

    def get_worker_cpu(self):
        return self._beancounter.get_cpu()

    ### This just replaces the "worker" call with our "debug_worker".
    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        print 'Repopulating pool...'
        for i in range(self._processes - len(self._pool)):
            #w = self.Process(target=debug_worker,
            w = DebugProcess(target=debug_worker,
                             args=(self._inqueue, self._outqueue,
                                   self._initializer,
                                   self._initargs, self._maxtasksperchild)
                            )
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            debug('added worker')


    def map_async(self, func, iterable, chunksize=None, callback=None):
        '''
        Asynchronous equivalent of `map()` builtin
        '''
        assert self._state == RUN
        if not hasattr(iterable, '__len__'):
            iterable = list(iterable)

        if chunksize is None:
            chunksize, extra = divmod(len(iterable), len(self._pool) * 4)
            if extra:
                chunksize += 1
        if len(iterable) == 0:
            chunksize = 0

        task_batches = mp.pool.Pool._get_tasks(func, iterable, chunksize)
        result = mp.pool.MapResult(self._cache, chunksize, len(iterable), callback)
        mapstar = mp.pool.mapstar
        print 'task_batches:', task_batches
        print 'putting on taskqueue:'
        print (((result._job, i, mapstar, (x,), {})
                for i, x in enumerate(task_batches)), None)
        self._taskqueue.put((((result._job, i, mapstar, (x,), {})
                              for i, x in enumerate(task_batches)), None))
        return result
            
            

    
    # This is just copied from the superclass; we call our routines:
    #  -handle_results -> debug_handle_results
    # And add _beancounter.
    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None):
        self._beancounter = BeanCounter()
        self._setup_queues()
        self._taskqueue = Queue.Queue()
        self._cache = {}
        self._state = RUN
        self._initializer = initializer
        self._initargs = initargs
        self._maxtasksperchild = maxtasksperchild

        if processes is None:
            try:
                processes = cpu_count()
            except NotImplementedError:
                processes = 1

        if initializer is not None and not hasattr(initializer, '__call__'):
            raise TypeError('initializer must be a callable')

        self._processes = processes
        self._pool = []
        self._repopulate_pool()

        self._worker_handler = threading.Thread(
            target=mp.pool.Pool._handle_workers,
            args=(self, )
            )
        self._worker_handler.daemon = True
        self._worker_handler._state = RUN
        self._worker_handler.start()

        self._task_handler = threading.Thread(
            target=mp.pool.Pool._handle_tasks,
            args=(self._taskqueue, self._quick_put, self._outqueue, self._pool)
            )
        self._task_handler.daemon = True
        self._task_handler._state = RUN
        self._task_handler.start()

        self._result_handler = threading.Thread(
            target=debug_handle_results,
            args=(self._outqueue, self._quick_get, self._cache,
                  self._beancounter)
            )
        self._result_handler.daemon = True
        self._result_handler._state = RUN
        self._result_handler.start()

        self._terminate = Finalize(
            self, self._terminate_pool,
            args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
                  self._worker_handler, self._task_handler,
                  self._result_handler, self._cache),
            exitpriority=15
            )
    

if __name__ == '__main__':

    import sys
    from astrometry.util import multiproc
    from astrometry.util.ttime import *
    
    def work((i)):
        print 'Doing work', i
        time.sleep(1)
        print 'Done work', i
        
    def arglist(n):
        for i in range(n):
            print 'Yielding', i
            yield i
        
    dpool = DebugPool(4)
    dmup = multiproc.multiproc(pool=dpool)
    Time.add_measurement(DebugPoolMeas(dpool))

    t0 = Time()
    dmup.map(work, arglist(10))
    print Time()-t0

    sys.exit(0)



    from tractor import *
    from tractor import sdss as st
    
    from tractor.engine import getmodelimagefunc2
    
    class Tractor2(Tractor):
        def _map(self, *args):
            t0 = Time()
            R = super(Tractor2,self)._map(*args)
            print 'map:', Time()-t0
            return R
    
        def getderivs2(self):
            alldata = []
            for im in self.images:
                alldata.append((im.data,im.invvar, im.inverr,im.origInvvar))
                im.shape = im.data.shape
                im.data,im.invvar = None,None
                im.inverr,im.origInvvar = None,None
                #print 'Image:', dir(im)
            R = super(Tractor2,self).getderivs2()
            for im,d in zip(self.images, alldata):
                im.data,im.invvar, im.inverr, im.origInvvar = d
            return R
    
        def getModelImages(self):
            if self.is_multiproc():
                # avoid shipping my images...
                allimages = self.getImages()
                self.images = []
    
                alldata = []
                for im in allimages:
                    alldata.append((im.data,im.invvar, im.inverr,im.origInvvar))
                    im.shape = im.data.shape
                    im.data,im.invvar = None,None
                    im.inverr,im.origInvvar = None,None
    
                mods = self._map(getmodelimagefunc2, [(self, im) for im in allimages])
    
                for im,d in zip(allimages, alldata):
                    im.data,im.invvar, im.inverr, im.origInvvar = d
    
                self.images = allimages
            else:
                mods = [self.getModelImage(img) for img in self.images]
            return mods
    
    
    
        # def getModelPatchNoCache(self, img, src):
        #   data,invvar = img.data,img.invvar
        #   img.shape = data.shape
        #   del img.data
        #   del img.invvar
        #   R = super(Tractor2,self).getModelPatchNoCache(img, src)
        #   img.data, img.invvar = data,invvar
    
    

    #run,camcol,field = 7164,4,273
    #band='g'
    run,camcol,field = 2662, 4, 111
    band='i'
    roi=[0,300,0,300]
    im,info = st.get_tractor_image(run, camcol, field, band,
                                   useMags=True, roi=roi)
    sources = st.get_tractor_sources(run, camcol, field, band, roi=roi)
    tractor = Tractor2([im], sources)
    print tractor
    print tractor.getLogProb()
    tractor.freezeParam('images')

    p0 = tractor.getParams()
    tractor.setParams(p0)

    print
    print 'With Debug:'
    tractor.setParams(p0)
    tractor.mp = dmup
    t0 = Time()
    tractor.opt2()
    print 'With Debug:', Time()-t0
    print dpool.get_pickle_traffic_string()
    print dpool.get_worker_cpu(), 'worker CPU'
    print Time()-t0
    sys.exit(0)

    pool = mp.Pool(4)
    mup = multiproc.multiproc(pool=pool)

    for i in range(3):
        print
        print 'With Debug:'
        tractor.setParams(p0)
        tractor.mp = dmup
        t0 = Time()
        tractor.opt2()
        print 'With Debug:', Time()-t0

        print
        print 'With vanilla:'
        tractor.setParams(p0)
        tractor.mp = mup
        t0 = Time()
        tractor.opt2()
        print 'With vanilla:', Time()-t0



