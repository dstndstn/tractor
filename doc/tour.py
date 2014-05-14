
"""
Here are some examples of how to use different types of Params.

>>> from tractor import *

ScalarParam holds a single float.

>>> p1 = ScalarParam(7.)
>>> print p1
ScalarParam: 7
>>> p1.getStepSizes()
[1.0]
>>> p1.getParamNames()
['ScalarParam']
>>> p1.getParams()
[7.0]
>>> p1.setParam(0, 42.)
7.0

(setParam returns the previous value)

>>> p1.getParams()
[42.0]

ParamList holds a list of floats.

>>> p2 = ParamList(12., 19.)
>>> p2.getParams()
[12.0, 19.0]
>>> p2.setParam(1, 20.)
19.0
>>> p2.getParams()
[12.0, 20.0]
>>> p2.getParamNames()
['param0', 'param1']

MultiParams builds hierarchies of Params.

>>> p3 = MultiParams(p1, p2)
>>> p3.getParams()
[42.0, 12.0, 20.0]
>>> p3.setParam(0, 7.)
42.0
>>> p3.getParams()
[7.0, 12.0, 20.0]
>>> p1.getParams()
[7.0]

"""

if __name__ == "__main__":
	import doctest
	doctest.testmod()
			
