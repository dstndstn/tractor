from __future__ import print_function
from tractor.lsqr_optimizer import LsqrOptimizer
import numpy as np
import time

dt = np.zeros(5)
tu = np.zeros(5)

logverb = print
logmsg  = print

class ConstrainedOptimizer(LsqrOptimizer):

    def __init__(self, *args, **kwargs):
        super(ConstrainedOptimizer, self).__init__(*args, **kwargs)
        self.stepLimited = False

    def printTiming(self):
        print ("DTimes:", dt)
    
    def optimize_loop(self, tractor, dchisq=0., steps=50,
                      dchisq_limited=1e-6, **kwargs):
        # print()
        # print('Optimize_loop:')
        # for s in tractor.catalog:
        #     print(s)
        print ("Constrained OPTIMIZE")
        t = time.time()
        R = {}
        self.hit_limit = False
        self.last_step_hit_limit = False
        for step in range(steps):
            print('Optimize_loop: step', step)
            self.stepLimited = False
            dlnp,_,_ = self.optimize(tractor, **kwargs)
            #print('Optimize_loop: step', step, 'dlnp', dlnp, 'hit limit:',
            #      self.hit_limit, 'step limit:', self.stepLimited)
            #for s in tractor.catalog:
            #    print(s)

            if not self.stepLimited and dlnp <= dchisq:
                break
            if self.stepLimited and dlnp <= dchisq_limited:
                break
        dt[2] += time.time()-t
        R.update(steps=step)
        R.update(hit_limit=self.last_step_hit_limit,
                 ever_hit_limit=self.hit_limit)
        dt[3] += time.time()-t
        print ("DTimes:", dt)
        #self.printTiming()
        return R

    def tryUpdates(self, tractor, X, alphas=None):
        t = time.time()
        t0 = time.time()
        #print('Trying parameter updates:', X)
        if alphas is None:
            # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        pBefore = tractor.getLogProb()
        #logverb('  log-prob before:', pBefore)
        pBest = pBefore
        alphaBest = None
        p0 = tractor.getParams()
        tu[0] += time.time()-t
        t = time.time()

        lowers = tractor.getLowerBounds()
        uppers = tractor.getUpperBounds()
        # print('Parameters:', tractor.getParamNames())
        # print('  lower bounds:', lowers)
        # print('  upper bounds:', uppers)

        maxsteps = tractor.getMaxStep()
        #print('Max step sizes:', maxsteps)
        #print ("updates - ",len(alphas), len(lowers))
        tu[1] += time.time()-t

        for alpha in alphas:
            t = time.time()
            #print('Stepping with alpha =', alpha)
            #logverb('  Stepping with alpha =', alpha)
            pa = [p + alpha * d for p, d in zip(p0, X)]

            # Check parameter limits
            step_hit_limit = False
            maxalpha = alpha
            bailout = False
            for i,(l,u,px) in enumerate(zip(lowers, uppers, pa)):
                if l is not None and px < l:
                    # This parameter hits the limit; compute the step size
                    # to just hit the limit.
                    a = (l - p0[i]) / X[i]
                    # print('Parameter', i, 'with initial value', p0[i],
                    #        'and update', X[i], 'would hit lower limit', l,
                    #        'with alpha', alpha, '; max alpha', a)
                    #print('Limiting step size to hit lower limit: param', i, 'limit', l, 'step size->', a)
                    maxalpha = min(maxalpha, a)
                    step_hit_limit = True
                    self.hit_limit = True
                if u is not None and px > u:
                    # This parameter hits the limit; compute the step size
                    # to just hit the limit.
                    a = (u - p0[i]) / X[i]
                    # print('Parameter', i, 'with initial value', p0[i],
                    #       'and update', X[i], 'would hit upper limit', u,
                    #       'with alpha', alpha, '; max alpha', a)
                    #print('Limiting step size to hit upper limit: param', i, 'limit', u, 'step size->', a)
                    maxalpha = min(maxalpha, a)
                    step_hit_limit = True
                    self.hit_limit = True

            if maxalpha < 1e-8:
                # Here, we're ceasing line-search because we're very
                # close to (or up against) a limit.  This can only
                # happen on the first line-search step, so alphaBest
                # is None and we're going to return quickly.
                self.last_step_hit_limit = True
                self.hit_limit = True
                break

            tu[2] += time.time()-t
            t = time.time()
            # Check parameter step-size limits
            for i,(d,m) in enumerate(zip(X, maxsteps)):
                if m is None:
                    continue
                if alpha * np.abs(d) > m:
                    self.stepLimited = True
                    a = m / np.abs(d)
                    # print('Parameter', i, 'with update', X[i], 'x alpha', alpha, '=',
                    #       X[i]*alpha, 'would exceed max step', m, '; max alpha', a)
                    maxalpha = min(maxalpha, a)
                    #print('Limiting step size for param max-step: param', i, 'max-step', m, 'step size->', a)
            tu[3] += time.time()-t
            t = time.time()

            if maxalpha < alpha:
                alpha = maxalpha
                # Cease line search (further steps want to exceed a limit, or have
                # parameter step sizes that are significantly non-linear).
                bailout = True
                # We could just multiply by alpha, but in case of numerical
                # instability, clip values right to limits.
                pa = []
                for p,d,l,u in zip(p0, X, lowers, uppers):
                    x = p + alpha * d
                    if l is not None and x < l:
                        x = l
                    if u is not None and x > u:
                        x = u
                    pa.append(x)
                # print('Clipping parameter update to', pa)
                # tractor.setParams(pa)
                # lp = tractor.getLogPrior()
                # print('Log prior:', lp)

            tractor.setParams(pa)
            pAfter = tractor.getLogProb()

            # print('Stepped params for dlogprob:', pAfter-pBefore)
            # for s in tractor.catalog:
            #     print(s)
            #tractor.printThawedParams()
            #print('dlogprob:', pAfter-pBefore)
            #print('log-prob:', pAfter, 'delta', pAfter-pBefore)
            #logverb('  Log-prob after:', pAfter)
            #logverb('  delta log-prob:', pAfter - pBefore)

            #print('Step', alpha, 'p', pAfter, 'dlnp', pAfter-pBefore)
            tu[4] += time.time()-t

            if not np.isfinite(pAfter):
                logmsg('  Got bad log-prob', pAfter)
                break

            if pAfter < (pBest - 1.):
                # We're getting significantly worse -- quit line search
                break

            if pAfter > pBest:
                # Best we've found so far -- accept this step!
                self.last_step_hit_limit = step_hit_limit
                alphaBest = alpha
                pBest = pAfter

            if bailout:
                break

        # if alphaBest is None or alphaBest == 0:
        #     print "Warning: optimization is borking"
        #     print "Parameter direction =",X
        #     print "Parameters, step sizes, updates:"
        #     for n,p,s,x in zip(tractor.getParamNames(), tractor.getParams(), tractor.getStepSizes(), X):
        #         print n, '=', p, '  step', s, 'update', x
        if alphaBest is None:
            tractor.setParams(p0)
            return 0, 0.

        #logverb('  Stepping by', alphaBest,
        #        'for delta-logprob', pBest - pBefore)
        #print('Stepped dlogprob', pBest-pBefore, '(step limited:', self.stepLimited,')')
        #for s in tractor.catalog:
        #    print(s)
        pa = [p + alphaBest * d for p, d in zip(p0, X)]
        tractor.setParams(pa)
        dt[0] += time.time()-t0
        print ("DTimesx:", dt, tu, tu.sum())
        return pBest - pBefore, alphaBest
