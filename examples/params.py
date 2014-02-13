
from tractor import *


class MyParam(ParamList):
	@staticmethod
	def getNamedParams():
		return dict(x=0, y=1)

p = MyParam(12, 51)
print 'p:', p
p.x = 42
print 'p:', p
print 'p.y:', p.y
print 'p.getParams():', p.getParams()

print 'p.y index:', p.getNamedParamIndex('x')
print 'Name of first param:', p.getNamedParamName(0)

p.freezeParam('x')
print 'frozen x:', p.getParams()
print 'n params', p.numberOfParams()
print 'frozen:', p.getFrozenParams()
print 'thawed:', p.getThawedParams()
print p

p.setParams([66])
print p
p.setParam(0, 67)
print p
p.thawAllParams()
print p
print 'frozen:', p.getFrozenParams()
print 'thawed:', p.getThawedParams()



class MyMultiParams(MultiParams):
	@staticmethod
	def getNamedParams():
		return dict(a=0, b=1)
print
print
mp = MyMultiParams(MyParam(1,2), MyParam(3,4))
print 'mp:', mp
print 'hashkey:', mp.hashkey()
print 'params:', mp.getParams()
print 'n params:', mp.numberOfParams()

mp.freezeParam('a')
print 'Frozen a:', mp.getParams()
print 'n:', mp.numberOfParams()
mp.b.freezeParam('y')
print 'n:', mp.numberOfParams()

mp.setParams([17])
print mp.getParams()
mp.thawAllParams()
print mp



