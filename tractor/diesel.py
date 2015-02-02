'''

This was a half-hearted attempt to generalize the Tractor, Images, and
Catalog objects in engine.py to general model spaces (with Gaussian
errors); ie, to remove the 2-D image specialization implicit in the
Tractor.

getUpdateDirection() is one big thing left to do, as is the
getChiImage() equivalent in the Tractor.  One question is whether to
continue assuming Gaussian errors in the Observations, or generalize
that too.

This might also require some changes to the definition of
Params()... for example, the derivative-taking currently returns Patch
objects, while we would want them to be something else:
PredictionDerivatives or something, from which Patch would inherit and
add its 2-D image specifics.

Then Image would extend from Observation, and maybe the Tractor class
(with images and catalog) would be its own Model? and extend
DieselEngine.

'''


from utils import Params

class Observation(object):
    '''
    Observed data with Gaussian errors.
    '''
    def __init__(self, data, inverr):
        self.data = data
        self.inverr = inverr

class Prediction(object):
    '''
    Observation-space prediction of a Model.
    '''
    def __init__(self, data):
        self.data = data

class Model(Params):
    pass
        
class DieselEngine(object):
    def __init__(self, observations, model):
        self.obs = observations
        self.model = model

    def optimize(self, alphas=None, damp=0, priors=True, scale_columns=True,
                 shared_params=True, variance=False, just_variance=False):
        '''
        Performs *one step* of linearized least-squares + line search.
        
        Returns (delta-logprob, parameter update X, alpha stepsize)

        If variance=True,

        Returns (delta-logprob, parameter update X, alpha stepsize, variance)

        If just_variance=True,
        Returns variance.

        '''
        allderivs = self.getDerivs()

        X = self.getUpdateDirection(allderivs, damp=damp, priors=priors,
                                    scale_columns=scale_columns,
                                    shared_params=shared_params,
                                    variance=variance)
        if variance:
            if len(X) == 0:
                return 0, X, 0, None
            X,var = X
            if just_variance:
                return var

        (dlogprob, alpha) = self.tryUpdates(X, alphas=alphas)
        if variance:
            return dlogprob, X, alpha, var
        return dlogprob, X, alpha

    def getDerivs(self):
        '''
        Computes observation-space derivatives for each model
        parameter.
        
        Returns a nested list of tuples:

        allderivs: [
           (param0:)  [  (deriv, obs), (deriv, obs), ... ],
           (param1:)  [],
           (param2:)  [  (deriv, obs), ],
        ]

        Where the *derivs* are *Prection* objects and *obs* are
        *Observation* objects.
        '''
        allderivs = [[] for i in range(self.model.numberOfParams())]

        for obs in self.observations:
            derivs = self.model.getParamDerivatives(obs)
            for k,deriv in enumerate(derivs):
                if deriv is None:
                    continue
                allderivs[k].append((deriv, obs))

        return allderivs

    


    def tryUpdates(self, X, alphas=None):
        if alphas is None:
            # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        pBefore = self.getLogProb()
        logverb('  log-prob before:', pBefore)
        pBest = pBefore
        alphaBest = None
        p0 = self.getParams()
        for alpha in alphas:
            logverb('  Stepping with alpha =', alpha)
            pa = [p + alpha * d for p,d in zip(p0, X)]
            self.setParams(pa)
            pAfter = self.getLogProb()
            logverb('  Log-prob after:', pAfter)
            logverb('  delta log-prob:', pAfter - pBefore)

            if not np.isfinite(pAfter):
                logmsg('  Got bad log-prob', pAfter)
                break

            if pAfter < (pBest - 1.):
                break

            if pAfter > pBest:
                alphaBest = alpha
                pBest = pAfter
        
        if alphaBest is None or alphaBest == 0:
            print "Warning: optimization is borking"
            print "Parameter direction =",X
            print "Parameters, step sizes, updates:"
            for n,p,s,x in zip(self.getParamNames(), self.getParams(), self.getStepSizes(), X):
                print n, '=', p, '  step', s, 'update', x
        if alphaBest is None:
            self.setParams(p0)
            return 0, 0.

        logmsg('  Stepping by', alphaBest, 'for delta-logprob', pBest - pBefore)
        pa = [p + alphaBest * d for p,d in zip(p0, X)]
        self.setParams(pa)
        return pBest - pBefore, alphaBest

    
