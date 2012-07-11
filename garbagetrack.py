import gc

_gids = set()

def track(nm, doprint=True):
	print 'Garbage collecting at stage "%s"' % nm
	gc.collect()
	garbage = gc.garbage
	nnew = 0
	for i,obj in enumerate(garbage):
		if id(obj) in _gids:
			continue
		if doprint:
			print
			print 'New garbage: "%s"' %  nm, i, type(obj)
			print str(obj)[:256]
			print repr(obj)[:256]
			print
		_gids.add(id(obj))
		nnew += 1
	print 'end of garbage: found %i new garbage objects' % nnew

		
