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
	def getParamNames(self):
		return [getClassName(self)]
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


class NamedParams(object):
	'''
	A mix-in class.

	Allows names to be attached to parameters.

	Also allows parameters to be set "Active" or "Inactive".
	'''

	@staticmethod
	def getNamedParams():
		'''
		Returns a dict of name->index mappings.
		'''
		return {}

	def __new__(cl, *args, **kwargs):
		sup = super(NamedParams,cl)
		if sup is object:
			self = sup.__new__(cl)
		else:
			self = sup.__new__(cl, *args, **kwargs)
		#print 'NamedParams.__new__:'
		#print '  args', args
		#print '  kwargs', kwargs

		self.namedparams = {}
		self.paramnames = {}
		named = self.getNamedParams()
		self.addNamedParams(**named)

		return self

	def __init__(self):
		super(NamedParams,self).__init__()
		#print 'NamedParams __init__.'
		# active/inactive
		self.liquid = [True] * self._numberOfThings()

	#def recountParams(self):
	#	self.liquid = [True] * self._numberOfThings()

	def addNamedParams(self, **d):
		self.namedparams.update(d)
		# create the reverse mapping: from parameter index to name.
		self.paramnames.update(dict((v,k) for k,v in d.items()))

		# Create a property for each named parameter.
		for n,i in self.namedparams.items():
			#print 'Adding named parameter', n, 'to class', self.__class__
			if hasattr(self.__class__, n):
				#print '  class', self.__class__, 'already has that attr'
				continue
			#if hasattr(self, n):
			#	print '  self of type', self.__class__, 'already has that attr'
			#	continue
			def makeGetter(ii):
				return lambda x: x._getThing(ii)
			def makeSetter(ii):
				return lambda x,v: x._setThing(ii, v)
			getter = makeGetter(i)
			setter = makeSetter(i)
			prop = property(getter, setter, None, 'named param %s' % n)
			setattr(self.__class__, n, prop)


	def _iterNamesAndVals(self):
		'''
		Yields  (name,val) tuples, where "name" is None if the parameter is not named.
		'''
		pvals = self._getThings()
		#print '_iterNamesAndVals: pvals types', [type(x) for x in pvals]
		for i,val in enumerate(pvals):
			name = self.paramnames.get(i, None)
			yield((name,val))

	def getNamedParamIndex(self, name):
		return self.namedparams.get(name, None)
	def getNamedParamName(self, ii):
		return self.paramnames.get(ii, None)

	def freezeParamsRecursive(self, *pnames):
		for nm in pnames:
			i = self.getNamedParamIndex(nm)
			if i is None:
				continue
			self.liquid[i] = False
	def thawParamsRecursive(self, *pnames):
		#self.thawParams(*pnames)
		for nm in pnames:
			i = self.getNamedParamIndex(nm)
			if i is None:
				continue
			self.liquid[i] = True

	def freezeParams(self, *args):
		for n in args:
			self.freezeParam(n)
	def freezeParam(self, paramname):
		if type(paramname) is int:
			i = paramname
		else:
			i = self.getNamedParamIndex(paramname)
			assert(i is not None)
		self.liquid[i] = False
	def freezeAllBut(self, *args):
		self.freezeAllParams()
		self.thawParams(*args)
	def thawParam(self, paramname):
		if type(paramname) is int:
			i = paramname
		else:
			i = self.getNamedParamIndex(paramname)
			assert(i is not None)
		self.liquid[i] = True
	def thawParams(self, *args):
		for n in args:
			self.thawParam(n)
	def thawAllParams(self):
		for i in xrange(len(self.liquid)):
			self.liquid[i] = True
	def freezeAllParams(self):
		for i in xrange(len(self.liquid)):
			self.liquid[i] = False
	def getFrozenParams(self):
		return [self.getNamedParamName(i) for i in self.getFrozenParamIndices()]
	def getThawedParams(self):
		return [self.getNamedParamName(i) for i in self.getThawedParamIndices()]
	def getFrozenParamIndices(self):
		for i,v in enumerate(self.liquid):
			if not v:
				yield i
	def getThawedParamIndices(self):
		for i,v in enumerate(self.liquid):
			if v:
				yield i
	def isParamFrozen(self, paramname):
		i = self.getNamedParamIndex(paramname)
		assert(i is not None)
		return not self.liquid[i]

	def _enumerateLiquidArray(self, array):
		for i,v in enumerate(self.liquid):
			if v:
				yield i,array[i]

	def _getLiquidArray(self, array):
		for i,v in enumerate(self.liquid):
			if v:
				yield array[i]

	def _countLiquid(self):
		return sum(self.liquid)

	def _indexLiquid(self, j):
		''' Returns the raw index of the i-th liquid parameter.'''
		for i,v in enumerate(self.liquid):
			if v:
				if j == 0:
					return i
				j -= 1
		raise IndexError

	def _indexBoth(self):
		''' Yields (i,j), for i-th liquid parameter and j the raw index. '''
		i = 0
		for j,v in enumerate(self.liquid):
			if v:
				yield (i, j)
				i += 1

class ParamList(Params, NamedParams):
	'''
	An implementation of Params that holds values in a list.
	'''
	def __init__(self, *args):
		#print 'ParamList __init__()'
		# FIXME -- kwargs with named params?
		self.vals = list(args)
		super(ParamList,self).__init__()

	def getFormatString(self, i):
		return '%g'

	def __str__(self):
		s = getClassName(self) + ': '
		ss = []
		for i,(name,val) in enumerate(self._iterNamesAndVals()):
			fmt = self.getFormatString(i)
			if name is not None:
				#print 'name', name, 'val', type(val)
				ss.append(('%s='+fmt) % (name, val))
			else:
				ss.append(fmt % val)
		return s + ', '.join(ss)

	def getParamNames(self):
		n = []
		for i,j in self._indexBoth():
			pre = self.getNamedParamName(j)
			if pre is None:
				pre = 'param%i' % i
			n.append(pre)
		return n

	# These underscored versions are for use by NamedParams(), and ignore
	# the active/inactive state.
	def _setThing(self, i, val):
		self.vals[i] = val
	def _getThing(self, i):
		return self.vals[i]
	def _getThings(self):
		return self.vals
	def _numberOfThings(self):
		return len(self.vals)
	
	# These versions skip frozen values.
	def setParam(self, i, val):
		ii = self._indexLiquid(i)
		oldval = self._getThing(ii)
		self._setThing(ii, val)
		return oldval
	def setParams(self, p):
		for i,j in self._indexBoth():
			self._setThing(j, p[i])
	def numberOfParams(self):
		return self._countLiquid()
	def getParams(self):
		'''
		Returns a *copy* of the current active parameter values (list)
		'''
		#return list(self._getLiquidArray(self.vals))
		return list(self._getLiquidArray(self._getThings()))
	def getParam(self,i):
		ii = self._indexLiquid(i)
		return self._getThing(ii)

	def getStepSizes(self, *args, **kwargs):
		return [1] * self.numberOfParams()

	def __len__(self):
		''' len(): of liquid params '''
		return self.numberOfParams()
	def __getitem__(self, i):
		''' index into liquid params '''
		return self.getParam(i)

	# iterable -- of liquid params.
	class ParamListIter(object):
		def __init__(self, pl):
			self.pl = pl
			self.i = 0
			self.N = pl.numberOfParams()
		def __iter__(self):
			return self
		def next(self):
			if self.i >= self.N:
				raise StopIteration
			rtn = self.pl.getParam(self.i)
			self.i += 1
			return rtn
	def __iter__(self):
		return ParamList.ParamListIter(self)

class MultiParams(Params, NamedParams):
	'''
	An implementation of Params that combines component sub-Params.
	'''
	def __init__(self, *args):
		if len(args):
			self.subs = list(args)
		else:
			self.subs = []
		super(MultiParams,self).__init__()

	def copy(self):
		return self.__class__([s.copy() for s in self.subs])

	# delegate list operations to self.subs.
	def append(self, x):
		self.subs.append(x)
		self.liquid.append(True)
	def extend(self, x):
		self.subs.extend(x)
		self.liquid.extend([True] * len(x))
	def remove(self, x):
		i = self.subs.index(x)
		self.subs = self.subs[:i] + self.subs[i+1:]
		self.liquid = self.liquid[:i] + self.liquid[i+1:]
		#self.subs.remove(x)
	def __len__(self):
		return len(self.subs)
	def __getitem__(self, key):
		return self.subs.__getitem__(key)
	def __iter__(self):
		return self.subs.__iter__()

	# def __len__(self):
	# 	''' len(): of liquid params '''
	# 	return self._countLiquid()
	# def __getitem__(self, i):
	# 	''' index into liquid params '''
	# 	return self.subs[self._indexLiquid(i)]
	# # iterable -- of liquid params.
	# class MultiParamsIter(object):
	# 	def __init__(self, me):
	# 		self.me = me
	# 		self.i = 0
	# 		self.N = len(me)
	# 	def __iter__(self):
	# 		return self
	# 	def next(self):
	# 		if self.i >= self.N:
	# 			raise StopIteration
	# 		rtn = self.me[self.i]
	# 		self.i += 1
	# 		return rtn
	# def __iter__(self):
	# 	return MultiParams.MultiParamsIter(self)

	def hashkey(self):
		t = [getClassName(self)]
		for s in self.subs:
			if s is None:
				t.append(None)
			t.append(s.hashkey())
		return tuple(t)

	def __str__(self):
		s = []
		for n,v in self._iterNamesAndVals():
			if n is None:
				s.append(str(v))
			else:
				s.append('%s: %s' % (n, str(v)))
		return getClassName(self) + ": " + ', '.join(s)

	# These underscored versions are for use by NamedParams(), and ignore
	# the active/inactive state.
	def _setThing(self, i, val):
		self.subs[i] = val
	def _getThing(self, i):
		return self.subs[i]
	def _getThings(self):
		return self.subs
	def _numberOfThings(self):
		return len(self.subs)

	def _getActiveSubs(self):
		for s in self._getLiquidArray(self.subs):
			# Should 'subs' be allowed to contain None values?
			if s is not None:
				yield s

	def freezeParamsRecursive(self, *pnames):
		for name,sub in self._iterNamesAndVals():
			if hasattr(sub, 'freezeParamsRecursive'):
				sub.freezeParamsRecursive(*pnames)
			if name in pnames:
				self.freezeParam(name)
	def thawParamsRecursive(self, *pnames):
		for name,sub in self._iterNamesAndVals():
			if hasattr(sub, 'thawParamsRecursive'):
				sub.thawParamsRecursive(*pnames)
			if name in pnames:
				self.thawParam(name)

	def getParamNames(self):
		n = []
		for i,s in self._enumerateLiquidArray(self.subs):
			pre = self.getNamedParamName(i)
			if pre is None:
				pre = 'param%i' % i
			snames = s.getParamNames()
			if snames is not None and len(snames) == s.numberOfParams():
				n.extend('%s.%s' % (pre,post) for post in s.getParamNames())
			else:
				print 'Getting named params for', pre
				print '  -> ', s.getParamNames()
				print '      (expected', s.numberOfParams(), 'of them)'
				n.extend('%s.param%i' % (pre,i) for i in range(s.numberOfParams()))
			
		return n

	def numberOfParams(self):
		'''
		Count unpinned (active) params.
		'''
		return sum(s.numberOfParams() for s in self._getActiveSubs())

	def getParams(self):
		'''
		Returns a *copy* of the current active parameter values (as a flat list)
		'''
		p = []
		for s in self._getActiveSubs():
			pp = s.getParams()
			if pp is None:
				continue
			p.extend(pp)
		return p

	def setParams(self, p):
		i = 0
		for s in self._getActiveSubs():
			n = s.numberOfParams()
			s.setParams(p[i:i+n])
			i += n

	def setParam(self, i, p):
		off = 0
		for s in self._getActiveSubs():
			n = s.numberOfParams()
			if i < off+n:
				return s.setParam(i-off, p)
			off += n
		raise RuntimeError('setParam(%i,...) for a %s that only has %i elements' %
						   (i, self.getClassName(self), self.numberOfParams()))

	def getStepSizes(self, *args, **kwargs):
		p = []
		for s in self._getActiveSubs():
			p.extend(s.getStepSizes(*args, **kwargs))
		return p


