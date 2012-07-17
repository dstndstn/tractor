try:
	# python 2.7
	from collections import OrderedDict
except:
	#from .ordereddict import OrderedDict
	from ordereddict import OrderedDict


#from refcnt import refcnt

'''
LRU cache.
This code is based on: http://code.activestate.com/recipes/498245-lru-and-lfu-cache-decorators/
By: Raymond Hettinger
License: Python Software Foundation (PSF) license.
'''
class Cache(object):
	class Entry(object):
		pass
	def __init__(self, maxsize=1000, sizeattr='size'):
		self.clear()
		self.maxsize = maxsize
		self.sizeattr = sizeattr

	def __del__(self):
		# OrderedDict objects seem to be prone to leaving garbage around...
		self.clear()
		del self.dict

	def clear(self):
		if not hasattr(self, 'dict'):
			self.dict = OrderedDict()
		else:
			self.dict.clear()
		    # print 'Clearing Cache object.'
			# objs = [(k,v) for k,v in self.dict.items()]
			# self.dict.clear()
			# while len(objs):
			# 	k,v = objs.pop()
			# 	print 'refcnt key', refcnt(k), 'val', refcnt(v),
			# 	print 'real', refcnt(v.val),
			# 	vv = v.val
			# 	v.val = None
			# 	print 'real', refcnt(vv)
		self.hits = 0
		self.misses = 0
		
	def __setitem__(self, key, val):
		sz = 0
		if hasattr(val, self.sizeattr):
			try:
				sz = int(getattr(val, self.sizeattr))
			except:
				pass
		e = Cache.Entry()
		e.val = val
		e.size = sz
		e.hits = 0
		# purge LRU item
		if len(self.dict) >= self.maxsize:
			self.dict.popitem(0)
		self.dict[key] = e

	def __getitem__(self, key):
		# pop
		try:
			e = self.dict.pop(key)
		except KeyError:
			self.misses += 1
			raise
		self.hits += 1
		# reinsert (to record recent use)
		self.dict[key] = e
		if e is None:
			return e
		e.hits += 1
		return e.val
	def __len__(self):
		return len(self.dict)
	def put(self, k, v):
		self[k] = v
	def get(self, *args):
		if len(args) == 1:
			key = args[0]
			return self.__getitem__(key)
		assert(len(args) == 2)
		key,default = args
		try:
			return self.__getitem__(key)
		except:
			return default
	def about(self):
		print 'Cache has', len(self), 'items:'
		for k,v in self.dict.items():
			if v is None:
				continue
			print '  size', v.size, 'hits', v.hits
	def __str__(self):
		s =  'Cache: %i items, total of %i hits, %i misses' % (len(self), self.hits, self.misses)
		nnone = 0
		hits = 0
		size = 0
		for k,v in self.dict.items():
			if v is None:
				nnone += 1
				continue
			if v.val is None:
				nnone += 1
				continue
			hits += v.hits
			size += v.size
		s +=  ', %i entries are None' % nnone
		s +=  '; current cache entries: %i hits, %i pixels' % (hits, size)
		return s

	def printItems(self):
		for k,v in self.dict.items():
			val = None
			hits = None
			size = None
			if v is not None:
				if v.val is not None:
					val = v.val
					hits = v.hits
					size = v.size
			print '  ', hits, size, k

	def totalSize(self):
		sz = 0
		for k,v in self.dict.items():
			if v is None:
				continue
			sz += v.size
		return sz
		
	def printStats(self):
		print 'Cache has', len(self), 'items'
		print 'Total of', self.hits, 'cache hits and', self.misses, 'misses'
		nnone = 0
		hits = 0
		size = 0
		for k,v in self.dict.items():
			if v is None:
				nnone += 1
				continue
			hits += v.hits
			size += v.size
		print '  ', nnone, 'entries are None'
		print 'Total number of hits of cache entries:', hits
		print' Total size (pixels) of cache entries:', size
		

class NullCache(object):
	def __getitem__(self, key):
		raise KeyError
	def __setitem__(self, key, val):
		pass
	def clear(self):
		pass
	def get(self, *args):
		if len(args) == 1:
			return self.__getitem__(args[0])
		return args[1]
	def put(self, *args):
		pass
	def totalSize(self):
		return 0
	def __len__(self):
		return len(self.dict)

