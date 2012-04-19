
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


