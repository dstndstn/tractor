class NasaSloanPhotoCal(object):
	'''
	A temporary photocal for NS atlas
	'''
	def __init__(self,bandname):
		self.bandname = bandname
	def brightnessToCounts(self, brightness):
		M= brightness.getMag(self.bandname)
		if not np.isfinite(M):
			return 0.
		if M > 50.:
			return 0.
		logc = (M-22.5)/-2.5
		return 10.**logc
	def hashkey(self):
		return ('TempPhotoCal',)

