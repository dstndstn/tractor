
from tractor import *


"""
Here are some examples of how to use different types of Params.

>>> p1 = ScalarParam(7.)
>>> print p1
>>> p1.getStepSizes()
>>> p1.getParamNames()
>>> p1.getParams()
>>> p1.setParam(0, 42.)
>>> p1.getParams()

>>> p2 = ParamList(12., 19.)
>>> p2.getParams()
>>> p2.setParams(1, 20.)
>>> p2.getParams()
>>> p2.getParamNames()

>>> p3 = MultiParams(p1, p2)
>>> p3.getParams()
>>> p3.setParam(0, 7.)
>>> p3.getParams()
>>> p1.getParams()


"""
