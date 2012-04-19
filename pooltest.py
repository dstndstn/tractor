import multiprocessing as mp
import mp.pool
import mp.queues

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
# which use pickle.



class DebugPool(Pool):
	def _setup_queues(self):
        from .queues import SimpleQueue
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv
		





