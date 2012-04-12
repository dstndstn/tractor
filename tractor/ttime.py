
class Time(object):
	def __init__(self):
		import datetime
		from time import clock
		self.wall = datetime.datetime.now()
		#self.cpu = time.clock()
		self.cpu = clock()
	def __sub__(self, other):
		dwall = (self.wall - other.wall)
		# python2.7
		if hasattr(dwall, 'total_seconds'):
			dwall = dwall.total_seconds()
		else:
			dwall = (dwall.microseconds + (dwall.seconds + dwall.days * 24. * 3600.) * 1e6) / 1e6
		dcpu = (self.cpu - other.cpu)
		return '%f wall, %f cpu' % (dwall, dcpu)


