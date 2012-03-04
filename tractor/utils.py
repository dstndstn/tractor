from ducks import *

def getClassName(obj):
	name = getattr(obj.__class__, 'classname', None)
	if name is not None:
		return name
	return obj.__class__.__name__

class PlotSequence(object):
	def __init__(self, basefn, format='%02i'):
		self.ploti = 0
		self.basefn = basefn
		self.format = format
		self.suff = 0
	def skip(self, n=1):
		self.ploti += n
	def skipto(self, n):
		self.ploti = n
	def savefig(self):
		fn = '%s-%s.png' % (self.basefn, self.format % self.ploti)
		plt.savefig(fn)
		print 'saved', fn
		self.ploti += 1




class ScalarParam(Params):
	'''
	Implementation of "Params" for a single scalar (float) parameter,
	stored in self.val
	'''
	stepsize = 1.
	strformat = '%g'
	def __init__(self, val):
		self.val = val
	def __str__(self):
		return self.getClassName(self) + ': ' + self.strformat % self.val #str(self.val)
	def __repr__(self):
		return self.getClassName(self) + '(' + repr(self.val) + ')'
	def numberOfParams(self):
		return 1
	def getStepSizes(self, *args, **kwargs):
		return [self.stepsize]
	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		return [self.val]
	def setParams(self, p):
		assert(len(p) == 1)
		self._set(p[0])
	def setParam(self, i, p):
		assert(i == 0)
		oldval = self.val
		self._set(p)
		return oldval
	def _set(self, val):
		self.val = val
	def getValue(self):
		return self.val


class ParamList(Params):
	'''
	An implementation of Params that holds values in a list.
	'''
	def __init__(self, *args):
		self.namedparams = self.getNamedParams()
		self.vals = list(args)

	def getFormatString(self, i):
		return '%g'

	def __str__(self):
		pvals = self.getParams()
		s = getClassName(self) + ': '
		ss = []
		for i,val in enumerate(pvals):
			name = None
			for k,j in self.namedparams:
				if i == j:
					name = k
					break
			fmt = self.getFormatString(i)
			if name is not None:
				ss.append(('%s='+fmt) % (name, val))
			else:
				ss.append(fmt % val)
		return s + ', '.join(ss)

	@staticmethod
	def getNamedParams():
		return []
	#def getNamedParams(self):
	#	return self.namedparams

	def __getattr__(self, name):
		if not 'namedparams' in self.__dict__:
			raise AttributeError
		for n,i in self.namedparams:
			if name == n:
				return self.vals[i]
		raise AttributeError('ParamList (%s): unknown attribute "%s"' %
							 (str(type(self)), name))
	def __setattr__(self, name, val):
		if name in ['vals', 'namedparams']:
			self.__dict__[name] = val
			return
		for n,i in self.namedparams:
			if name == n:
				self._setParam(i, val)
				return
		self.__dict__[name] = val
	def _setParam(self, i, val):
		self.vals[i] = val
	def setParam(self, i, val):
		oldval = self.vals[i]
		self._setParam(i, val)
		return oldval
	def numberOfParams(self):
		return len(self.vals)
	def getParams(self):
		'''
		Returns a *copy* of the current parameter values (list)
		'''
		return list(self.vals)
	def setParams(self, p):
		assert(len(p) == len(self.vals))
		for i,pp in enumerate(p):
			self.setParam(i,pp)
	def getStepSizes(self, *args, **kwargs):
		return [1 for x in self.vals]

	# len()
	def __len__(self):
		return self.numberOfParams()
	# []
	def __getitem__(self, i):
		#print 'ParamList.__getitem__', i, 'returning', self.vals[i]
		return self.vals[i]

	# iterable
	class ParamListIter(object):
		def __init__(self, pl):
			self.pl = pl
			self.i = 0
		def __iter__(self):
			return self
		def next(self):
			if self.i >= len(self.pl):
				raise StopIteration
			rtn = self.pl[self.i]
			#print 'paramlistiter: returning element', self.i, '=', rtn
			self.i += 1
			return rtn
	def __iter__(self):
		return ParamList.ParamListIter(self)

class MultiParams(Params):
	'''
	An implementation of Params that combines component sub-Params.
	'''
	def __init__(self, *args):
		if len(args):
			self.subs = list(args)
		else:
			self.subs = []
		self.namedparams = self.getNamedParams()
		#print getClassName(self), 'named params:', self.namedparams
		# indices of pinned params
		self.pinnedparams = []

	def copy(self):
		return self.__class__([s.copy() for s in self.subs])

	#def getNamedParams(self):
	@staticmethod
	def getNamedParams():
		return []
	def __getattr__(self, name):
		if not 'namedparams' in self.__dict__:
			raise AttributeError
		for n,i in self.namedparams:
			if name == n:
				return self.subs[i]
		raise AttributeError('MultiParams (%s): unknown attribute "%s" (named params: [ %s ]; dict keys: [ %s ])' %
							 (str(type(self)), name, ', '.join([k for k,v in self.getNamedParams()]),
							  ', '.join(self.__dict__.keys())))

	def __setattr__(self, name, val):
		if name in ['subs', 'namedparams', 'pinnedparams']:
			self.__dict__[name] = val
			return
		if hasattr(self, 'namedparams'):
			for n,i in self.namedparams:
				if name == n:
					self.subs[i] = val
					return
		self.__dict__[name] = val

	def hashkey(self):
		t = [getClassName(self)]
		for s in self.subs:
			if s is None:
				t.append(None)
			t.append(s.hashkey())
		return tuple(t)

	def getNamedParamIndex(self, name):
		for n,i in self.namedparams:
			if n == name:
				return i
		return None
	def getNamedParamName(self, ii):
		for n,i in self.namedparams:
			if i == ii:
				return n
		return None

	def pinParam(self, paramname):
		i = self.getNamedParamIndex(paramname)
		assert(i is not None)
		self.pinnedparams.append(i)
	def unpinParam(self, paramname):
		i = self.getNamedParamIndex(paramname)
		assert(i is not None)
		self.pinnedparams.remove(i)
	def unpinAllParams(self, paramname):
		self.pinnedparams = []
	def getPinnedParams(self):
		return [self.getNamedParamName(i) for i in self.pinnedparams]
	def getUnpinnedParamIndices(self):
		ii = []
		for i in range(len(self.subs)):
			if not i in self.pinnedparams:
				ii.append(i)
		return ii
	def isParamPinned(self, paramname):
		i = self.getNamedParamIndex(paramname)
		assert(i is not None)
		return i in self.pinnedparams

	def numberOfParams(self):
		return sum(s.numberOfParams() for s in self.subs
				   if s is not None)

	# Returns a *copy* of the current parameter values (list)
	def getParams(self):
		p = []
		for s in self.subs:
			if s is None:
				continue
			pp = s.getParams()
			if pp is None:
				continue
			p.extend(pp)
		return p

	def setParams(self, p):
		i = 0
		for s in self.subs:
			if s is None:
				continue
			n = s.numberOfParams()
			s.setParams(p[i:i+n])
			i += n

	def setParam(self, i, p):
		off = 0
		for s in self.subs:
			if s is None:
				continue
			n = s.numberOfParams()
			if i < off+n:
				return s.setParam(i-off, p)
			off += n
		raise RuntimeError('setParam(%i,...) for a %s that only has %i elements' % (i, self.getClassName(self), self.numberOfParams()))

	def getStepSizes(self, *args, **kwargs):
		p = []
		for s in self.subs:
			if s is None:
				continue
			p.extend(s.getStepSizes(*args, **kwargs))
		return p


