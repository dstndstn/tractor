import os
import resource
def memusage():
	# print heapy.heap()
	#ru = resource.getrusage(resource.RUSAGE_BOTH)
	ru = resource.getrusage(resource.RUSAGE_SELF)
	pgsize = resource.getpagesize()
	print 'Memory usage:'
	#print 'page size', pgsize
	print 'max rss:', (ru.ru_maxrss * pgsize / 1e6), 'MB'
	#print 'shared memory size:', (ru.ru_ixrss / 1e6), 'MB'
	#print 'unshared memory size:', (ru.ru_idrss / 1e6), 'MB'
	#print 'unshared stack size:', (ru.ru_isrss / 1e6), 'MB'
	#print 'shared memory size:', ru.ru_ixrss
	#print 'unshared memory size:', ru.ru_idrss
	#print 'unshared stack size:', ru.ru_isrss
	procfn = '/proc/%d/status' % os.getpid()
	try:
		t = open(procfn).readlines()
		#print 'proc file:', t
		d = dict([(line.split()[0][:-1], line.split()[1:]) for line in t])
		#print 'dict:', d
		for key in ['VmPeak', 'VmSize', 'VmRSS', 'VmData', 'VmStk' ]: # VmLck, VmHWM, VmExe, VmLib, VmPTE
			print key, ' '.join(d.get(key, []))
	except:
		pass

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


