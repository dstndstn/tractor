from __future__ import print_function
from tractor.lsqr_optimizer import LsqrOptimizer
import numpy as np
import time

logverb = print
logmsg  = print

class ConstrainedOptimizer(LsqrOptimizer):

    def __init__(self, *args, **kwargs):
        super(ConstrainedOptimizer, self).__init__(*args, **kwargs)
        self.stepLimited = False
        # The smallest parameter update step we will try, in tryUpdates / getParameterSteps
        self.tiny_alpha = 1e-8

    def optimize_loop(self, tractor, dchisq=0., steps=50,
                      dchisq_limited=1e-6,
                      check_step=None,
                      **kwargs):
        # print()
        # print('Optimize_loop:')
        # for s in tractor.catalog:
        #     print(s)
        R = {}
        self.hit_limit = False
        self.last_step_hit_limit = False
        for step in range(steps):
            #print('Optimize_loop: step', step)
            self.stepLimited = False
            dlnp,X,alpha = self.optimize(tractor, **kwargs)
            if check_step is not None:
                r = check_step(optimizer=self, tractor=tractor, dlnp=dlnp, X=X, alpha=alpha)
                if not r:
                    break
            #print('Optimize_loop: step', step, 'dlnp', dlnp, 'hit limit:',
            #      self.hit_limit, 'step limit:', self.stepLimited)
            #for s in tractor.catalog:
            #    print(s)

            if not self.stepLimited and dlnp <= dchisq:
                break
            if self.stepLimited and dlnp <= dchisq_limited:
                break
        R.update(steps=step)
        R.update(hit_limit=self.last_step_hit_limit,
                 ever_hit_limit=self.hit_limit)
        return R

    def tryUpdates(self, tractor, X, alphas=None, check_step=None):
        p_best = tractor.getParams()
        logprob_before = tractor.getLogProb()
        logprob_best = logprob_before
        alpha_best = None

        # Get lists of alpha values and parameters to try
        steps = self.getParameterSteps(tractor, X, alphas)

        if len(steps) == 0:
            self.last_step_hit_limit = True
            self.hit_limit = True

        for alpha, p, step_limit, hit_limit in steps:
            if hit_limit:
                self.hit_limit = True

            tractor.setParams(p)

            if check_step is not None:
                r = check_step(optimizer=self, tractor=tractor, X=X, alpha=alpha)
                if not r:
                    break

            logprob = tractor.getLogProb()
            #print (f'{alpha=} {p=} {step_limit=} {hit_limit=} {logprob=}')

            if not np.isfinite(logprob):
                logmsg('  Got bad log-prob', logprob)
                break

            if logprob < (logprob_best - 1.):
                # We're getting significantly worse -- quit line search
                break

            if logprob > logprob_best:
                # Best we've found so far -- accept this step!
                self.last_step_hit_limit = hit_limit
                alpha_best = alpha
                logprob_best = logprob
                p_best = p

        tractor.setParams(p_best)
        #print ("Pbest", p_best)
        #print ("LOGPROB", logprob_best, logprob_before)
        #print ("ALPHA", alpha_best, self.hit_limit, self.last_step_hit_limit)
        if alpha_best is None:
            return 0., 0.

        return logprob_best - logprob_before, alpha_best

    def getParameterSteps(self, tractor, step_direction, alphas):
        '''
        Returns a list of
          [ (float alpha, list parameters, bool step_limit, bool hit_limit), ... ]
        where *step_limit* means the step was limited by a max parameter step size
        and *hit_limit* means the step was limited by an upper or lower bound on a parameter.
        '''
        if alphas is None:
            # 1/1024 to 1 in factors of 2, + sqrt(2.) + 2.
            alphas = np.append(2.**np.arange(-10, 1), [np.sqrt(2.), 2.])

        p0 = tractor.getParams()
        max_step_size = tractor.getMaxStep()
        lowers = tractor.getLowerBounds()
        uppers = tractor.getUpperBounds()
        # (note that max_step_size, lowers and uppers contain None if there is no limit.)

        # assume that alphas are monotonic increasing
        assert(np.all(np.diff(alphas) > 0))

        step_direction = np.array(step_direction)

        results = []

        for alpha in alphas:
            do_break = False

            # 1. apply max_step_size
            step_limit = False
            for i,mx in enumerate(max_step_size):
                if mx is None:
                    continue
                step = alpha * abs(step_direction[i])
                if step > mx:
                    step_limit = True
                    do_break = True
                    # reduce alpha to correspond to the max step size
                    #alpha = mx / step
                    alpha = mx / abs(step_direction[i])
                    self.stepLimited = True

            # 2. apply lower and upper bounds
            hit_limit = False
            for i,(p_start,l,u) in enumerate(zip(p0, lowers, uppers)):
                px = p_start + alpha * step_direction[i]
                if l is not None and px < l:
                    # exceeds limit - reduce alpha to *just* hit the limit.
                    alpha = (l - p_start) / step_direction[i]
                    px = l
                    hit_limit = True
                if u is not None and px > u:
                    # exceeds limit - reduce alpha to *just* hit the limit.
                    alpha = (u - p_start) / step_direction[i]
                    px = u
                    hit_limit = True

            # If the "alpha" step size we can take is tiny, just bail out.
            if alpha < self.tiny_alpha:
                break

            # Compute the parameter vector corresponding to this alpha -- do this
            # carefully to avoid numerically exceeding lower/upper bounds.
            p = []
            for p_start,s,l,u in zip(p0, step_direction, lowers, uppers):
                px = p_start + alpha * s
                if l is not None and px < l:
                    px = l
                if u is not None and px > u:
                    px = u
                p.append(px)

            results.append((alpha, p, step_limit, hit_limit))

            if do_break:
                break
        return results
