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
	def __init__(self, fd, writable=True, readable=True):
		self.real = _multiprocessing.Connection(fd, writable=writable, readable=readable)
	def poll(self):
		return self.real.poll()
	def recv(self):
		# obj = self.real.recv()
		bytes = self.real.recv_bytes()
		t0 = time.time()
		obj = pickle.loads(bytes)
		dt = time.time() - t0
		print 'unpickling took', dt
		#print 'received', obj
		return obj

	def send(self, obj):
		# return self.real.send(obj)
		print 'sending', obj
		t0 = time.time()
		s = pickle.dumps(obj, -1)
		dt = time.time() - t0
		print '-->', len(s), 'bytes'
		print 'pickling took', dt
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
	def __init__(self):
		self._reader, self._writer = DebugPipe()
		self._rlock = Lock()
		self._wlock = Lock()
		self._make_methods()

class DebugPool(mp.pool.Pool):
	def _setup_queues(self):
		self._inqueue = DebugSimpleQueue()
		self._outqueue = DebugSimpleQueue()
		self._quick_put = self._inqueue._writer.send
		self._quick_get = self._outqueue._reader.recv
		

from tractor import *
from tractor import sdss as st
from astrometry.util import multiproc

if __name__ == '__main__':
	#run,camcol,field = 7164,4,273
	#band='g'
	run,camcol,field = 2662, 4, 111
	band='i'
	roi=[0,300,0,300]
	im,info = st.get_tractor_image(run, camcol, field, band,
								   useMags=True, roi=roi)
	sources = st.get_tractor_sources(run, camcol, field, band, roi=roi)
	tractor = Tractor([im], sources)
	print tractor
	print tractor.getLogProb()
	tractor.freezeParam('images')
	pool = DebugPool(4)
	tractor.mp = multiproc.multiproc(pool=pool)
	tractor.opt2()



