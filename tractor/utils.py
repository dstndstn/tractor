"""
This file is part of the Tractor project.
Copyright 2011, 2012, 2013 Dustin Lang and David W. Hogg.
Licensed under the GPLv2; see the file COPYING for details.

`utils.py`
===========

Utility classes: low-level useful implementations of ducks the Tractor
needs: sets of parameters stored in lists and such.  This framework
could be useful outside the Tractor context.

"""
from __future__ import print_function
import numpy as np

try:
    # New in python 2.7
    import functools
    total_ordering = functools.total_ordering
except:
    from total_ordering import total_ordering

def get_class_from_name(objclass):
    try:
        import importlib
    except:
        importlib = None

    names = objclass.split('.')
    names = [n for n in names if len(n)]
    pkg = '.'.join(names[:-1])
    clazz = names[-1]
    if importlib is not None:
        # nice 'n easy in py2.7+
        mod = importlib.import_module(pkg)
    else:
        mod = __import__(pkg, globals(), locals(), [], -1)
        print('Module:', mod)
        for name in names[1:-1]:
            print('name', name)
            mod = getattr(mod, name)
            print('-> mod', mod)

    clazz = getattr(mod, clazz)
    #print('Class:', clazz)
    return clazz
    
def listmax(X, default=0):
    mx = [np.max(x) for x in X if len(x)]
    if len(mx) == 0:
        return default
    return np.max(mx)
    

def getClassName(obj):
    name = getattr(obj.__class__, 'classname', None)
    if name is not None:
        return name
    return obj.__class__.__name__


class _GaussianPriors(object):
    '''
    A class to support Gaussian priors in ParamList objects.  This
    class holds the actual list of terms in the prior and computes the
    required logProb and derivatives.  The GaussianPriorsMixin class
    below glues this to the Params interface.
    '''
    def __init__(self, param):
        self.terms = []
        self.param = param

    def __str__(self):
        s = ('GaussianPriors: [ ' +
             ', '.join(['(%s ~ N(mu=%.3g, sig=%.3g)' %
                        (nm,mu,sig) for nm,i,mu,sig in self.terms]) + ' ]')
        return s
        
    def add(self, name, mu, sigma, param=None):
        if param is None:
            param = self.param
        i = param.getNamedParamIndex(name)
        if i is None:
            raise KeyError('GaussianPriors.add: parameter not found: "%s"' % name)
        self.terms.append((name, i, mu, sigma))

    def getLogPrior(self, param=None):
        if param is None:
            param = self.param
        p = param.getAllParams()
        chisq = 0.
        for name,i,mu,sigma in self.terms:
            chisq += (p[i] - mu)**2 / sigma**2
        return -0.5 * chisq

    def getDerivs(self, param=None):
        if param is None:
            param = self.param
        rows = []
        cols = []
        vals = []
        bs = []
        mus = []
        
        row0 = 0
        p = param.getParams()
        for name,j,mu,sigma in self.terms:
            i = param.getLiquidIndexOfIndex(j)
            # frozen:
            if i == -1:
                continue
            cols.append(np.array([i]))
            vals.append(np.array([1. / sigma]))
            rows.append(np.array([row0]))
            bs.append(np.array([-(p[i] - mu) / sigma]))
            mus.append(np.array([mu]))
            row0 += 1
        return rows, cols, vals, bs, mus

class GaussianPriorsMixin(object):
    '''
    A mix-in class for ParamList-like classes, to make it easy to support
    Gaussian priors.
    '''
    def __init__(self, *args, **kwargs):
        super(GaussianPriorsMixin, self).__init__(*args, **kwargs)
        self.gpriors = _GaussianPriors(self)

    def addGaussianPrior(self, name, mu, sigma):
        self.gpriors.add(name, mu, sigma, param=self)

    def getLogPriorDerivatives(self):
        '''
        Returns the log prior derivatives in a sparse matrix form as
        required by the Tractor when optimizing.

        You might want to override like this:

        X = self.getGaussianLogPriorDerivatives()
        Y = << other log prior derivatives >>
        return [x+y for x,y in zip(X,Y)]
        '''
        return self.getGaussianLogPriorDerivatives()

    def getGaussianLogPriorDerivatives(self):
        return self.gpriors.getDerivs(param=self)

    def isLegal(self):
        '''
        Returns True if the current parameter values are legal; ie,
        have > 0 prior.
        '''
        return True
    
    def getLogPrior(self):
        '''
        Returns the log prior at the current parameter values.
        
        If you want to disallow entirely regions of parameter space,
        override the isLegal() method; this method will then return
        -np.inf .

        If you need to do something more elaborate, you probably want
        to add getGaussianLogPrior() to your fancy prior so that
        Gaussian priors still work.
        '''
        if not self.isLegal():
            return -np.inf
        return self.getGaussianLogPrior()

    def getGaussianLogPrior(self):
        return self.gpriors.getLogPrior(param=self)

class BaseParams(object):
    '''
    A basic implementation of the `Params` duck type.
    '''
    def __repr__(self):
        return getClassName(self) + repr(self.getParams())
    def __str__(self):
        return getClassName(self) + ': ' + str(self.getParams())
    def copy(self):
        return self.__class__(*self.getAllParams())
    def hashkey(self):
        return (getClassName(self),) + tuple(self.getAllParams())
    #def __hash__(self):
    #    return hash(self.hashkey())
    #def __eq__(self, other):
    #    return hash(self.hashkey()) == hash(other.hashkey())

    def getParamNames(self):
        ''' Returns a list containing the names of the parameters. '''
        return []
    def numberOfParams(self):
        ''' Returns the number of parameters (ie, number of scalar
        values).'''
        return len(self.getParams())
    def getParams(self):
        ''' Returns a *copy* of the current parameter values as an
        iterable (eg, list)'''
        return []
    def getAllParams(self):
        return self.getParams()
    def getAllStepSizes(self, *args, **kwargs):
        return self.getStepSizes(*args, **kwargs)
    def getStepSizes(self, *args, **kwargs):
        '''
        Returns "reasonable" step sizes for the parameters.
        '''
        ss = getattr(self, 'stepsizes', None)
        if ss is not None:
            return ss
        return [1.] * self.numberOfParams()
    def setAllStepSizes(self, ss):
        self.setStepSizes(ss)
    def setStepSizes(self, ss):
        self.stepsizes = ss
    def setParams(self, p):
        '''
        NOTE, you MUST implement either "setParams" or "setParam",
        because the default implementation causes an infinite loop!

        Sets the parameter values to the values in the given iterable
        `p`.  The base class implementation just calls `setParam` for
        each element.
        '''
        assert(len(p) == self.numberOfParams())
        for ii,pp in enumerate(p):
            self.setParam(ii, pp)
    def setAllParams(self, p):
        return self.setParams(p)
    def setParam(self, i, p):
        '''
        NOTE, you MUST implement either "setParams" or "setParam",
        because the default implementation causes an infinite loop!

        Sets parameter index `i` to new value `p`.

        i: integer in the range [0, numberOfParams()).
        p: float

        Returns the old value.
        '''
        P = self.getParams()
        old = P[i]
        P[i] = p
        return old

    def getLogPrior(self):
        '''
        Return the prior PDF, evaluated at the current value
        of the paramters.
        '''
        return 0.

    def getLogPriorDerivatives(self):
        return None

@total_ordering
class ScalarParam(BaseParams):
    '''
    Implementation of "Params" for a single scalar (float) parameter,
    stored in self.val
    '''
    stepsize = 1.
    strformat = '%g'
    def __init__(self, val=0):
        self.val = val
    def __str__(self):
        return getClassName(self) + ': ' + self.strformat % self.val
    def __repr__(self):
        return getClassName(self) + '(' + repr(self.val) + ')'

    def __eq__(self, other):
        return self.getValue() == other.getValue()
    def __lt__(self, other):
        return self.getValue() < other.getValue()

    def copy(self):
        return self.__class__(self.val)
    def getParamNames(self):
        return [getClassName(self)]
    def numberOfParams(self):
        return 1
    def getStepSizes(self, *args, **kwargs):
        return [self.stepsize]
    def setStepSizes(self, ss):
        self.stepsize = ss[0]
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
    def setValue(self, v):
        self._set(v)
    
def _isint(i):
    #return type(i) in [int, np.int64]
    try:
        return int(i) == i
    except:
        return False

class NamedParams(object):
    '''
    A mix-in class for Params subclassers.

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
        self = super(NamedParams,cl).__new__(cl, *args, **kwargs)
        self.namedparams = {}
        self.paramnames = {}
        named = self.getNamedParams()
        self.addNamedParams(**named)
        return self

    def __init__(self):
        super(NamedParams,self).__init__()
        # active/inactive
        self.liquid = [True] * self._numberOfThings()

    def getAllParams(self):
        ''' Returns all params, regardless of thawed/frozen status. '''
        raise RuntimeError("Unimplemented getAllParams in " + str(self.__class__))
    def setAllParams(self, p):
        ''' Returns all params, regardless of thawed/frozen status. '''
        raise RuntimeError("Unimplemented setAllParams in " + str(self.__class__))

    def getStepSizes(self, *args, **kwargs):
        ss = getattr(self, 'stepsizes', None)
        if ss is None:
            ss = self.getAllStepSizes(*args, **kwargs)
        return list(self._getLiquidArray(ss))

    def getAllStepSizes(self, *args, **kwargs):
        '''
        Returns "reasonable" step sizes for the parameters, ignoring
        frozen/thawed state.
        '''
        ss = getattr(self, 'stepsizes', None)
        if ss is not None:
            return ss
        return [1.] * len(self.getAllParams())

    def setStepSizes(self, ss):
        if not hasattr(self, 'stepsizes'):
            newss = []
            j = 0
            for i,ll in enumerate(self.liquid):
                if ll:
                    newss.append(ss[j])
                    j += 1
                else:
                    newss.append(1.)
            self.stepsizes = newss
        else:
            for i,s in self._enumerateLiquidArray(ss):
                self.stepsizes[i] = s

    def setAllStepSizes(self, ss):
        self.stepsizes = ss
    
    def _addNamedParams(self, alias, **d):
        self.namedparams.update(d)
        if not alias:
            # create the reverse mapping: from parameter index to name.
            self.paramnames.update(dict((v,k) for k,v in d.items()))

        # Create a property for each named parameter.
        for n,i in self.namedparams.items():
            #print('Adding named parameter', n, 'to class', self.__class__)
            if hasattr(self.__class__, n):
                #print('  class', self.__class__, 'already has attr', n)
                continue
            #if hasattr(self, n):
            #   print('  self of type', self.__class__, 'already has that attr')
            #   continue

            # def makeGetter(ii):
            #   return lambda x: x._getThing(ii)
            # def makeSetter(ii):
            #   return lambda x,v: x._setThing(ii, v)
            # getter = makeGetter(i)
            # setter = makeSetter(i)

            def makeNamedGetter(nm):
                #return lambda x: x._getThing(self.namedparams[nm])
                return lambda x: x._getNamedThing(nm)
            def makeNamedSetter(nm):
                #return lambda x,v: x._setThing(self.namedparams[nm], v)
                return lambda x,v: x._setNamedThing(nm, v)
            getter = makeNamedGetter(n)
            setter = makeNamedSetter(n)

            prop = property(getter, setter, None, 'named param %s' % n)
            setattr(self.__class__, n, prop)
        
    def addParamAliases(self, **d):
        self._addNamedParams(alias=True, **d)

    def addNamedParams(self, **d):
        '''
        d: dict of (string, int) parameter names->indices
        '''
        self._addNamedParams(alias=False, **d)
        

    def _getNamedThing(self, nm):
        return self._getThing(self.namedparams[nm])
    def _setNamedThing(self, nm, v):
        return self._setThing(self.namedparams[nm], v)


    def _iterNamesAndVals(self):
        '''
        Yields  (name,val) tuples, where "name" is None if the parameter is not named.
        '''
        pvals = self._getThings()
        #print('_iterNamesAndVals: pvals types', [type(x) for x in pvals])
        for i,val in enumerate(pvals):
            name = self.paramnames.get(i, None)
            yield((name,val))

    def getNamedParamIndex(self, name):
        return self.namedparams.get(name, None)
    def getNamedParamName(self, ii):
        return self.paramnames.get(ii, None)

    def getParamStateRecursive(self):
        n = []
        for j,liquid in enumerate(self.liquid):
            nm = self.getNamedParamName(j)
            if nm is None:
                nm = 'param%i' % j
            n.append((nm,liquid,liquid))
        return n

    def freezeParamsRecursive(self, *pnames):
        for nm in pnames:
            i = self.getNamedParamIndex(nm)
            if i is None:
                continue
            self.liquid[i] = False
        if '*' in pnames:
            self.freezeAllParams()

    def thawParamsRecursive(self, *pnames):
        for nm in pnames:
            i = self.getNamedParamIndex(nm)
            if i is None:
                continue
            self.liquid[i] = True
        if '*' in pnames:
            self.thawAllParams()
        
    def freezeParams(self, *args):
        for n in args:
            self.freezeParam(n)
    def freezeParam(self, paramname):
        if _isint(paramname):
            i = paramname
        else:
            i = self.getNamedParamIndex(paramname)
            assert(i is not None)
        self.liquid[i] = False
    def freezeAllBut(self, *args):
        self.freezeAllParams()
        self.thawParams(*args)

    def thawPathsTo(self, *pnames):
        '''
        This is a (non-recursive) basic implementation
        '''
        thawed = False
        for nm in pnames:
            i = self.getNamedParamIndex(nm)
            if i is None:
                continue
            self.liquid[i] = True
            thawed = True
        return thawed

    def thawParam(self, paramname):
        if _isint(paramname):
            i = paramname
        elif isinstance(paramname, basestring):
            i = self.getNamedParamIndex(paramname)
            assert(i is not None)
        else:
            # assume it's an actual Param, not a name
            i = self._getThings().index(paramname)
            
        self.liquid[i] = True
    def thawParams(self, *args):
        for n in args:
            self.thawParam(n)
    def thawAllParams(self):
        self.liquid[:] = [True]*len(self.liquid)
    unfreezeParam = thawParam
    unfreezeParams = thawParams
    unfreezeAllParams = thawAllParams
    
    def freezeAllParams(self):
        self.liquid[:] = [False]*len(self.liquid)
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
    def isParamThawed(self, paramname):
        i = self.getNamedParamIndex(paramname)
        assert(i is not None)
        return self.liquid[i]

    def getLiquidIndexOfIndex(self, i):
        '''
        Return the index, among the thawed parameters, of the given
        parameter index (in all parameters).  Returns -1 if the
        parameter is frozen.
        '''
        if not self.liquid[i]:
            return -1
        return sum(self.liquid[:i])

    def getLiquidIndex(self, paramname):
        '''
        Returns the index, among the thawed parameters, of the given
        named parameter.  Returns -1 if the parameter is frozen.
        Raises KeyError if the parameter is not found.

        For example, if names 'a','c', and 'e' are thawed,

        getLiquidIndex('c') returns 1
        getLiquidIndex('b') returns -1
        '''
        i = self.getNamedParamIndex(paramname)
        if i is None:
            raise KeyError('No such parameter "%s"', paramname)
        if not self.liquid[i]:
            return -1
        return sum(self.liquid[:i])
        
    def _enumerateLiquidArray(self, array):
        for i,v in enumerate(self.liquid):
            if v:
                yield i,array[i]

    def _getLiquidArray(self, array):
        for i,(v,a) in enumerate(zip(self.liquid, array)):
            if v:
                yield a
    def _getFrozenArray(self, array):
        for i,(v,a) in enumerate(zip(self.liquid, array)):
            if not v:
                yield a

    def _enumerateLiquidArray(self, array):
        for i,v in enumerate(self.liquid):
            if v:
                yield i,array[i]

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
        ''' Yields (i,j), for i, the liquid parameter and j, the raw index. '''
        i = 0
        for j,v in enumerate(self.liquid):
            if v:
                yield (i, j)
                i += 1

class ParamList(GaussianPriorsMixin, NamedParams, BaseParams):
    '''
    An implementation of Params that holds values in a list.
    '''
    def __init__(self, *args):
        #print('ParamList __init__()')
        # FIXME -- kwargs with named params?
        self.vals = list(args)
        super(ParamList,self).__init__()

    def copy(self):
        #return self.__class__(*self.getParams())
        #cop = self.__class__(*self._getThings())
        cop = super(ParamList, self).copy()
        cop.liquid = [l for l in self.liquid]
        return cop

    def getFormatString(self, i):
        return '%g'

    def __str__(self):
        s = getClassName(self) + ': '
        ss = []
        for i,(name,val) in enumerate(self._iterNamesAndVals()):
            fmt = self.getFormatString(i)
            if name is not None:
                #print('name', name, 'val', type(val))
                ss.append(('%s='+fmt) % (name, val))
            else:
                ss.append(fmt % val)
        return s + ', '.join(ss)

    def getParamNames(self):
        n = []
        for i,j in self._indexBoth():
            nm = self.getNamedParamName(j)
            if nm is None:
                nm = 'param%i' % i
            n.append(nm)
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
        return list(self._getLiquidArray(self._getThings()))

    def getAllParams(self):
        return list(self._getThings())

    def setAllParams(self, p):
        for i,pp in enumerate(p):
            self._setThing(i, pp)

    def getParam(self,i):
        ii = self._indexLiquid(i)
        return self._getThing(ii)

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

class ArithmeticParams(object):
    #def __eq__(self, other):
    #   return np.all(self.getParams() == other.getParams())
    #def __lt__(self, other):

    def __add__(self, other):
        ''' + '''
        res = self.copy()
        if hasattr(other, 'getAllParams'):
            res.setAllParams([x + y for x,y in zip(res.getAllParams(),
                                                   other.getAllParams())])
        else:
            res.setAllParams([x + other for x in res.getAllParams()])
        return res

    def __sub__(self, other):
        ''' - '''
        res = self.copy()
        if hasattr(other, 'getAllParams'):
            res.setAllParams([x - y for x,y in zip(res.getAllParams(),
                                                   other.getAllParams())])
        else:
            res.setAllParams([x - other for x in res.getAllParams()])
        return res

    def __mul__(self, other):
        ''' *= '''
        res = self.copy()
        if hasattr(other, 'getAllParams'):
            res.setAllParams([x * y for x,y in zip(res.getAllParams(),
                                                   other.getAllParams())])
        else:
            res.setAllParams([x * other for x in res.getAllParams()])
        return res

    def __div__(self, other):
        ''' /= '''
        res = self.copy()
        if hasattr(other, 'getAllParams'):
            res.setAllParams([x / y for x,y in zip(res.getAllParams(),
                                                   other.getAllParams())])
        else:
            res.setAllParams([x / other for x in res.getAllParams()])
        return res

    def __iadd__(self, other):
        ''' += '''
        if hasattr(other, 'getAllParams'):
            self.setAllParams([x + y for x,y in zip(self.getAllParams(),
                                                    other.getAllParams())])
        else:
            self.setAllParams([x + other for x in self.getAllParams()])
        return self

    def __isub__(self, other):
        ''' -= '''
        if hasattr(other, 'getAllParams'):
            self.setAllParams([x - y for x,y in zip(self.getAllParams(),
                                                    other.getAllParams())])
        else:
            self.setAllParams([x - other for x in self.getAllParams()])
        return self

    def __imul__(self, other):
        ''' *= '''
        if hasattr(other, 'getAllParams'):
            self.setAllParams([x * y for x,y in zip(self.getAllParams(),
                                                    other.getAllParams())])
        else:
            self.setAllParams([x * other for x in self.getAllParams()])
        return self

    def __idiv__(self, other):
        ''' /= '''
        if hasattr(other, 'getAllParams'):
            self.setAllParams([x / y for x,y in zip(self.getAllParams(),
                                                    other.getAllParams())])
        else:
            self.setAllParams([x / other for x in self.getAllParams()])
        return self
    
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__
    
class MultiParams(BaseParams, NamedParams):
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
        x = self.__class__(*[s.copy() for s in self.subs])
        x.liquid = [l for l in self.liquid]
        return x
    
    # delegate list operations to self.subs.
    def append(self, x):
        self.subs.append(x)
        self.liquid.append(True)
    def prepend(self, x):
        self.subs = [x] + self.subs
        self.liquid = [True] + self.liquid
    def extend(self, x):
        self.subs.extend(x)
        self.liquid.extend([True] * len(x))
    def remove(self, x):
        i = self.subs.index(x)
        self.subs = self.subs[:i] + self.subs[i+1:]
        self.liquid = self.liquid[:i] + self.liquid[i+1:]
        #self.subs.remove(x)
    def index(self, x):
        return self.subs.index(x)
        
    # Note that these *don't* pay attention to thawed/frozen status... Mistake?
    def __len__(self):
        return len(self.subs)
    def __getitem__(self, key):
        return self.subs.__getitem__(key)
    def __setitem__(self, key, val):
        return self.subs.__setitem__(key, val)
    def __iter__(self):
        return self.subs.__iter__()

    # def __len__(self):
    #   ''' len(): of liquid params '''
    #   return self._countLiquid()
    # def __getitem__(self, i):
    #   ''' index into liquid params '''
    #   return self.subs[self._indexLiquid(i)]
    # # iterable -- of liquid params.
    # class MultiParamsIter(object):
    #   def __init__(self, me):
    #       self.me = me
    #       self.i = 0
    #       self.N = len(me)
    #   def __iter__(self):
    #       return self
    #   def next(self):
    #       if self.i >= self.N:
    #           raise StopIteration
    #       rtn = self.me[self.i]
    #       self.i += 1
    #       return rtn
    # def __iter__(self):
    #   return MultiParams.MultiParamsIter(self)

    def hashkey(self):
        t = [getClassName(self)]
        for s in self.subs:
            if s is None:
                t.append(None)
            else:
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

    def _getInactiveSubs(self):
        for s in self._getFrozenArray(self.subs):
            # Should 'subs' be allowed to contain None values?
            if s is not None:
                yield s

    def _enumerateActiveSubs(self):
        '''
        Yields *index-ignoring-freeze-state*,sub
        for unfrozen subs.
        '''
        for i,s in self._enumerateLiquidArray(self.subs):
            # Should 'subs' be allowed to contain None values?
            if s is not None:
                yield i,s

    def freezeParamsRecursive(self, *pnames):
        for name,sub in self._iterNamesAndVals():
            if hasattr(sub, 'freezeParamsRecursive'):
                sub.freezeParamsRecursive(*pnames)
            if name in pnames:
                self.freezeParam(name)
        if '*' in pnames:
            self.freezeAllParams()
    def freezeAllRecursive(self):
        self.freezeParamsRecursive('*')

    def thawParamsRecursive(self, *pnames):
        for name,sub in self._iterNamesAndVals():
            if hasattr(sub, 'thawParamsRecursive'):
                sub.thawParamsRecursive(*pnames)
            if name in pnames:
                self.thawParam(name)
        if '*' in pnames:
            self.thawAllParams()
    def thawAllRecursive(self):
        self.thawParamsRecursive('*')

    def thawPathsTo(self, *pnames):
        thawed = False
        for i,(name,sub) in enumerate(self._iterNamesAndVals()):
            if hasattr(sub, 'thawPathsTo'):
                if sub.thawPathsTo(*pnames):
                    self.thawParam(i)
                    thawed = True
            if name in pnames:
                self.thawParam(i)
                thawed = True
        return thawed

    def getParamStateRecursive(self):
        n = []
        for i,(s,liquid) in enumerate(zip(self.subs, self.liquid)):
            pre = self.getNamedParamName(i)
            if pre is None:
                pre = 'param%i' % i
            n.append((pre, liquid, True))
            if hasattr(s, 'getParamStateRecursive'):
                snames = s.getParamStateRecursive()
            else:
                snames = [(nm,True,True) for nm in s.getParamNames()]
            n.extend(('%s.%s' % (pre,post), pliq, (liquid and pliq2))
                     for (post,pliq,pliq2) in snames)
        return n

    def printThawedParams(self):
        for nm,val in zip(self.getParamNames(), self.getParams()):
            print('  ', nm, '=', val)

    def getParamNames(self):
        n = []
        for i,s in self._enumerateLiquidArray(self.subs):
            pre = self.getNamedParamName(i)
            if pre is None:
                pre = 'param%i' % i
            snames = s.getParamNames()
            if snames is not None and len(snames) == s.numberOfParams():
                n.extend('%s.%s' % (pre,post) for post in snames)
            else:
                print('Getting named params for', pre)
                print('  -> ', snames)
                print('      (expected', s.numberOfParams(), 'of them)')
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

    def getAllParams(self):
        p = []
        for s in self.subs:
            pp = s.getAllParams()
            if pp is None:
                continue
            p.extend(pp)
        return p

    def setAllParams(self, p):
        i = 0
        for s in self.subs:
            n = s.numberOfParams()
            s.setAllParams(p[i:i+n])
            i += n

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
                           (i, getClassName(self), self.numberOfParams()))

    def getStepSizes(self, *args, **kwargs):
        p = []
        for s in self._getActiveSubs():
            p.extend(s.getStepSizes(*args, **kwargs))
        return p

    def getAllStepSizes(self, *args, **kwargs):
        p = []
        for s in self.subs:
            p.extend(s.getAllStepSizes(*args, **kwargs))
        return p

    def setStepSizes(self, ss):
        off = 0
        for sub in self._getActiveSubs():
            n = sub.numberOfParams()
            sub.setStepSizes(ss[off: off+n])
            off += n

    def setAllStepSizes(self, ss):
        off = 0
        for sub in self.subs:
            n = len(sub.getAllParams())
            sub.setAllStepSizes(ss[off: off+n])
            off += n
            
    def getLogPrior(self):
        lnp = 0.
        for s in self._getActiveSubs():
            lnp += s.getLogPrior()
        return lnp

    def getLogPriorDerivatives(self):
        """
        Return prior formatted so that it can be used in least square fitting
        """
        rA,cA,vA,pb,mub = [],[],[],[],[]

        r0 = 0
        c0 = 0
        
        for s in self._getActiveSubs():
            X = s.getLogPriorDerivatives()
            if X is None:
                c0 += s.numberOfParams()
                continue
            (r,c,v,b,m) = X
            rA.extend([ri + r0 for ri in r])
            cA.extend([ci + c0 for ci in c])
            vA.extend(v)
            pb.extend(b)
            mub.extend(m)
            
            c0 += s.numberOfParams()
            r0 += listmax(r,-1) + 1

        if rA == []:
            return None
        return rA,cA,vA,pb,mub


class NpArrayParams(ParamList):
    '''
    An implementation of Params that holds values in an np.ndarray
    '''
    def __init__(self, a):
        self.a = np.array(a)
        super(NpArrayParams, self).__init__()
        del self.vals
        # from NamedParams...
        # active/inactive
        self.liquid = [True] * self._numberOfThings()

    def __getattr__(self, name):
        if name == 'vals':
            return self.a.ravel()
        if name in ['shape',]:
            return getattr(self.a, name)
        raise AttributeError() #name + ': no such attribute in NpArrayParams.__getattr__')

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

