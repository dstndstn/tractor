from __future__ import print_function
class TractorLBFGSBMixin(object):
    def optimize_lbfgsb(self, hessian_terms=10, plotfn=None):

        XX = []
        OO = []
        def objective(x, tractor, stepsizes, lnp0):
            res = lnp0 - tractor(x * stepsizes)
            print('LBFGSB objective:', res)
            if plotfn:
                XX.append(x.copy())
                OO.append(res)
            return res

        from scipy.optimize import fmin_l_bfgs_b

        stepsizes = np.array(self.getStepSizes())
        p0 = np.array(self.getParams())
        lnp0 = self.getLogProb()

        print('Active parameters:', len(p0))

        print('Calling L-BFGS-B ...')
        X = fmin_l_bfgs_b(objective, p0 / stepsizes, fprime=None,
                          args=(self, stepsizes, lnp0),
                          approx_grad=True, bounds=None, m=hessian_terms,
                          epsilon=1e-8, iprint=0)
        p1,lnp1,d = X
        print(d)
        print('lnp0:', lnp0)
        self.setParams(p1 * stepsizes)
        print('lnp1:', self.getLogProb())

        if plotfn:
            import pylab as plt
            plt.clf()
            XX = np.array(XX)
            OO = np.array(OO)
            print('XX shape', XX.shape)
            (N,D) = XX.shape
            for i in range(D):
                OO[np.abs(OO) < 1e-8] = 1e-8
                neg = (OO < 0)
                plt.semilogy(XX[neg,i], -OO[neg], 'bx', ms=12, mew=2)
                pos = np.logical_not(neg)
                plt.semilogy(XX[pos,i], OO[pos], 'rx', ms=12, mew=2)
                I = np.argsort(XX[:,i])
                plt.plot(XX[I,i], np.abs(OO[I]), 'k-', alpha=0.5)
                plt.ylabel('Objective value')
                plt.xlabel('Parameter')
            plt.twinx()
            for i in range(D):
                plt.plot(XX[:,i], np.arange(N), 'r-')
                plt.ylabel('L-BFGS-B iteration number')
            plt.savefig(plotfn)
    
