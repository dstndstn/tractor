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
#	  on the _inqueue.
#
# _handle_results pulls results from the _outqueue (job,i,obj)
#	  and calls cache[job].set(i, obj)
#	  (cache[job] is an ApplyResult / MapResult, etc.)
#
# worker threads run the worker() function:
#	run initializer
#	while true:
#	  pull task from inqueue
#	  job,i,func,arg,kwargs = task
#	  put (job,i,result) on outqueue

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
import time

class DebugConnection():
	def stats(self):
		#return ('pickled %i objs, %i bytes, %g s CPU; unpickled %i objs, %i bytes, %g s CPU' %
		return (self.pobjs, self.pbytes, self.ptime, self.upobjs, self.upbytes, self.uptime)
	
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
		#print 'sending', str(obj)
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
		return [s1+s2 for s1,s2 in zip(S1,S2)]
		#return ('reader', self._reader.stats(), 'writer', self._writer.stats())
		#return ('pickled %i objs, %i bytes, %g s CPU; unpickled %i objs, %i bytes, %g s CPU' %
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

def debug_worker(inqueue, outqueue, initializer=None, initargs=()):
	print 'debug_worker()'
	put = outqueue.put
	get = inqueue.get
	if hasattr(inqueue, '_writer'):
		inqueue._writer.close()
		outqueue._reader.close()

	if initializer is not None:
		initializer(*initargs)

	while 1:
		t0 = time.time()
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
		try:
			result = (True, func(*args, **kwds))
		except Exception, e:
			result = (False, e)
		t2 = time.time()
		succ,val = result
		result = (succ, (val, t2-t1))
		#print 'Putting result', result
		put((job, i, result))
		t3 = time.time()
		#print 'worker: get task', (t1-t0), 'run', (t2-t1), 'result', (t3-t2)

class DebugMapResult(mp.pool.MapResult):
	def __init__(self, *args):
		self.cpu = 0.
		super(DebugMapResult, self).__init__(*args)
		
	def get(self, **kwargs):
		result = super(DebugMapResult, self).get(**kwargs)
		#print 'Result:', result
		#result,dt = result
		#print 'Result took', dt, 'seconds of CPU'
		return result

	def _set(self, i, obj):
		#print '_set', i, obj
		succ, (val,dt) = obj
		obj = (succ, val)
		#print 'dt', dt
		self.cpu += dt
		return super(DebugMapResult, self)._set(i, obj)

Pool = mp.pool.Pool
mapstar = mp.pool.mapstar

class DebugPool(mp.pool.Pool):
	def __del__(self):
		print 'DebugPool __del__:'
		#print '  inqueue', self._inqueue.stats()
		#print '  outqueue', self._outqueue.stats()
		#Pool.__del__(self)
		S1 = self._inqueue.stats()
		S2 = self._outqueue.stats()
		S = [s1+s2 for s1,s2 in zip(S1,S2)]
		print S
		#print S[:3]
		(po,pb,pt, uo,ub,ut) = S
		print 'pickled %i objs, %i bytes, %g s CPU' % (po,pb,pt) #S[:3]
		print 'unpickled %i objs, %i bytes, %g s CPU' % (uo,ub,ut) #S[3:]


	def _setup_queues(self):
		self._inqueue = DebugSimpleQueue()
		self._outqueue = DebugSimpleQueue()
		self._quick_put = self._inqueue._writer.send
		self._quick_get = self._outqueue._reader.recv

	def map(self, func, iterable, chunksize=None):
		'''
		Equivalent of `map()` builtin
		'''
		assert self._state == RUN
		mapres = self.map_async(func, iterable, chunksize)
		val = mapres.get()
		print 'map() used a total of', mapres.cpu, 's CPU'
		return val

	# Copied from superclass, but with DebugMapResult.
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

		task_batches = Pool._get_tasks(func, iterable, chunksize)
		result = DebugMapResult(self._cache, chunksize, len(iterable), callback)
		self._taskqueue.put((((result._job, i, mapstar, (x,), {})
							  for i, x in enumerate(task_batches)), None))
		return result

	# This is just copied from the superclass; we redefine the worker() routine though.
	def __init__(self, processes=None, initializer=None, initargs=()):
		self._setup_queues()
		self._taskqueue = Queue.Queue()
		self._cache = {}
		self._state = RUN

		if processes is None:
			try:
				processes = cpu_count()
			except NotImplementedError:
				processes = 1

		self._pool = []
		for i in range(processes):
			w = self.Process(
				target=debug_worker,
				args=(self._inqueue, self._outqueue, initializer, initargs)
				)
			self._pool.append(w)
			w.name = w.name.replace('Process', 'PoolWorker')
			w.daemon = True
			w.start()

		self._task_handler = threading.Thread(
			target=Pool._handle_tasks,
			args=(self._taskqueue, self._quick_put, self._outqueue, self._pool)
			)
		self._task_handler.daemon = True
		self._task_handler._state = RUN
		self._task_handler.start()

		self._result_handler = threading.Thread(
			target=Pool._handle_results,
			args=(self._outqueue, self._quick_get, self._cache)
			)
		self._result_handler.daemon = True
		self._result_handler._state = RUN
		self._result_handler.start()

		self._terminate = Finalize(
			self, self._terminate_pool,
			args=(self._taskqueue, self._inqueue, self._outqueue, self._pool,
				  self._task_handler, self._result_handler, self._cache),
			exitpriority=15
			)
	


import sys
from tractor import *
from tractor import sdss as st
from astrometry.util import multiproc

class Tractor2(Tractor):
	def _map(self, *args):
		t0 = Time()
		R = super(Tractor2,self)._map(*args)
		print 'map:', Time()-t0
		return R

	def getModelPatchNoCache(self, img, src):
		data,invvar = img.data,img.invvar
		img.shape = data.shape
		del img.data
		del img.invvar
		R = super(Tractor2,self).getModelPatchNoCache(img, src)
		img.data, img.invvar = data,invvar

if __name__ == '__main__':
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

	dpool = DebugPool(4)
	dmup = multiproc.multiproc(pool=dpool)

	print
	print 'With Debug:'
	tractor.setParams(p0)
	tractor.mp = dmup
	t0 = Time()
	tractor.opt2()
	print 'With Debug:', Time()-t0
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



