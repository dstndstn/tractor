import numpy as np

def gauss2d(X, mu, sigma):
    N,two = X.shape
    assert(two == 2)
    C,two = mu.shape
    assert(two == 2)
    rtn = np.zeros((N,C))
    for c in range(C):
        e = -np.sum((X - mu[c,:])**2, axis=1) / (2.*sigma[c]**2)
        I = (e > -700)  # -> ~ 1e-304
        rtn[I,c] = 1./(2.*pi*sigma[c]**2) * np.exp(e[I])
    return rtn

def em_step(X, weights, mu, sigma, background, B):
    '''
    mu: shape (C,2) or (2,)
    sigma: shape (C,) or scalar
    weights: shape (C,) or 1.
    C: number of Gaussian components

    X: (N,2)
    '''
    mu_orig = mu

    mu = np.atleast_2d(mu)
    sigma = np.atleast_1d(sigma)
    weights = np.atleast_1d(weights)
    weights /= np.sum(weights)

    print '    em_step: weights', weights, 'mu', mu, 'sigma', sigma, 'background fraction', B
    # E:
    # fg = p( Y, Z=f | theta ) = p( Y | Z=f, theta ) p( Z=f | theta )
    fg = gauss2d(X, mu, sigma) * (1. - B) * weights
    # fg shape is (N,C)
    # bg = p( Y, Z=b | theta ) = p( Y | Z=b, theta ) p( Z=b | theta )
    bg = background * B
    assert(all(np.isfinite(fg.ravel())))
    assert(all(np.isfinite(np.atleast_1d(bg))))
    # normalize:
    sfg = np.sum(fg, axis=1)
    # fore = p( Z=f | Y, theta )
    fore = fg / (sfg + bg)[:,np.newaxis]
    # back = p( Z=b | Y, theta )
    back = bg / (sfg + bg)
    assert(all(np.isfinite(fore.ravel())))
    assert(all(np.isfinite(back.ravel())))

    # M:
    # maximize mu, sigma:
    #mu = np.sum(fore[:,np.newaxis] * X, axis=0) / np.sum(fore)
    mu = np.dot(fore.T, X) / np.sum(fore)
    # 2.*sum(fore) because X,mu are 2-dimensional.
    #sigma = np.sqrt(np.sum(fore[:,np.newaxis] * (X - mu)**2) / (2.*np.sum(fore)))
    C = len(sigma)
    for c in range(C):
        sigma[c] = np.sqrt(np.sum(fore[:,c][:,np.newaxis] * (X - mu[c,:])**2)
                           / (2. * np.sum(fore[:,c])))
    #print 'mu', mu, 'sigma', sigma
    if np.min(sigma) == 0:
        return (mu, sigma, B, -1e6, np.zeros(len(X)))
    assert(np.all(sigma > 0))

    # maximize weights:
    weights = np.mean(fore, axis=0)
    weights /= np.sum(weights)

    # maximize B.
    # B = p( Z=b | theta )
    B = np.mean(back)

    # avoid multiplying 0 * -inf = NaN
    I = (fg > 0)
    lfg = np.zeros_like(fg)
    lfg[I] = np.log(fg[I])

    lbg = np.log(bg * np.ones_like(fg))
    lbg[np.flatnonzero(np.isfinite(lbg) == False)] = 0.

    # Total expected log-likelihood
    Q = np.sum(fore*lfg + back[:,np.newaxis]*lbg)

    print 'Fore', fore.shape
    if len(mu_orig.shape) == 1:
        return (1., mu[0,:], sigma[0], B, Q, fore[:,0])
    return (weights, mu, sigma, B, Q, fore)

