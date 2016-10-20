from __future__ import print_function
from tractor.utils import *

class TestParamList(ParamList):

	@staticmethod
	def getNamedParams():
		return dict(a=0, b=1, c=2)


t1 = TestParamList(42., 17., 3.14)
print('t1:', t1)

t2 = t1.copy()
print('t2:', t2)
assert(len(t1.getParams()) == 3)

t1.freezeParam('b')
print('t1:', t1)
print('t1 params:', t1.getParams())
assert(len(t1.getParams()) == 2)

print('t2:', t2)
print('t2 params:', t2.getParams())
assert(len(t2.getParams()) == 3)

t3 = t1.copy()
print('t3:', t3)
print('t3 params:', t3.getParams())
assert(len(t3.getParams()) == 2)

