# Copyright 2011 David W. Hogg.  All rights reserved.

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import matplotlib.cm as cm
import numpy as np
import scipy.optimize as op
import mixture_profiles as mp

# note wacky normalization because this is for 2-d Gaussians
# (but only ever called in 1-d).  Wacky!
def not_normal(x, V):
	exparg = -0.5 * x**2 / V
	result = np.zeros_like(x)
	I = ((exparg > -1000) * (exparg < 1000))
	result[I] = 1. / (2. * np.pi * V) * np.exp(exparg[I])
	return result

def hogg_dev(x):
	return np.exp(-1. * (x**0.25))

# not magic 7. and 8.
def hogg_lup(x):
	inner = 7.
	outer = 8.
	lup = hogg_dev(x)
	outside = (x > outer)
	lup[outside] *= 0.
	middle = (x > inner) * (x < outer)
	lup[middle] *= (outer - x[middle])
	return lup

def mixture_of_not_normals(x, pars):
	K = len(pars)/2
	y = 0.
	for k in range(K):
		y += pars[k] * not_normal(x, pars[k+K])
	return y

# note that you can do (x * ymix - x * ytrue)**2 or (ymix - ytrue)**2
# each has disadvantages.
def badness_of_fit_exp(lnpars):
	pars = np.exp(lnpars)
	x = np.arange(0., MAX_RADIUS, 0.01)
	return np.mean((mixture_of_not_normals(x, pars)
					- np.exp(-x))**2) / 10.**LOG10_SQUARED_DEVIATION

# note that you can do (x * ymix - x * ytrue)**2 or (ymix - ytrue)**2
# each has disadvantages.
def badness_of_fit_dev(lnpars):
	pars = np.exp(lnpars)
	x = np.arange(0., MAX_RADIUS, 0.001)
	return np.mean((mixture_of_not_normals(x, pars)
					- hogg_dev(x))**2) / 10.**LOG10_SQUARED_DEVIATION

def badness_of_fit_lup(lnpars):
	pars = np.exp(lnpars)
	x = np.arange(0., MAX_RADIUS, 0.001)
	return np.mean((mixture_of_not_normals(x, pars)
					- hogg_lup(x))**2) / 10.**LOG10_SQUARED_DEVIATION

def optimize_mixture(K, pars, model):
	if model == 'exp':
		func = badness_of_fit_exp
	if model == 'dev':
		func = badness_of_fit_dev
	if model == 'lup':
		func = badness_of_fit_lup
	newlnpars = op.fmin_bfgs(func, np.log(pars), maxiter=300)
	return (func(newlnpars), np.exp(newlnpars))

def plot_mixture(pars, prefix, model):
	x1 = np.arange(0., MAX_RADIUS, 0.001)
	if model == 'exp':
		y1 = np.exp(-x1)
		badness = badness_of_fit_exp(np.log(pars))
	if model == 'dev':
		y1 = hogg_dev(x1)
		badness = badness_of_fit_dev(np.log(pars))
	if model == 'lup':
		y1 = hogg_lup(x1)
		badness = badness_of_fit_lup(np.log(pars))
	K = len(pars) / 2
	x2 = np.arange(0., 2.*MAX_RADIUS, 0.001)
	y2 = mixture_of_not_normals(x2, pars)
	plt.clf()
	plt.plot(x1, y1, 'k-')
	plt.plot(x2, y2, 'k-', lw=4, alpha=0.25)
	for k in range(K):
		plt.plot(x2, pars[k] * not_normal(x2, pars[k+K]), 'k-', alpha=0.5)
	plt.title(r"%s / $K=%d$ / maximum radius = $%.1f$ / badness = $%.2f\times 10^{%d}$" % (model, len(pars)/2, MAX_RADIUS, badness, LOG10_SQUARED_DEVIATION))
	plt.xlim(-0.1*np.max(x1), 1.1*np.max(x1))
	plt.ylim(-0.1*np.max(y1), 1.1*np.max(y1))
	plt.savefig(prefix+'_'+model+'.png')
	plt.loglog()
	plt.xlim(0.003*np.max(x1), 1.5*np.max(x1))
	plt.ylim(0.003*np.max(y1), 1.5*np.max(y1))
	plt.savefig(prefix+'_'+model+'_log.png')

def rearrange_pars(pars):
	K = len(pars) / 2
	indx = np.argsort(pars[K:K+K])
	amp = pars[indx]
	var = pars[K+indx]
	return np.append(amp, var)

# run this (possibly with adjustments to the magic numbers at top)
# to find different or better mixtures approximations
def main():
	for model in ['exp', 'dev', 'lup']:
		amp = np.array([1.0])
		var = np.array([1.0])
		pars = np.append(amp, var)
		(badness, pars) = optimize_mixture(1, pars, model)
		lastKbadness = badness
		bestbadness = badness
		for K in range(2,20):
			print 'working on K = %d' % K
			newvar = 0.5 * np.min(np.append(var,1.0))
			newamp = 1.0 * newvar
			amp = np.append(newamp, amp)
			var = np.append(newvar, var)
			pars = np.append(amp, var)
			for i in range(100):
				(badness, pars) = optimize_mixture(K, pars, model)
				if badness < bestbadness:
					print '%d %d improved' % (K, i)
					bestpars = pars
					bestbadness = badness
				else:
					print '%d %d not improved' % (K, i)
					var[0] = 0.5 * var[np.random.randint(K)]
					amp[0] = 1.0 * var[0]
					pars = np.append(amp, var)
				if (bestbadness < 0.5 * lastKbadness) and (i > 5):
					print '%d %d improved enough' % (K, i)
					break
			lastKbadness = bestbadness
			pars = rearrange_pars(bestpars)
			amp = pars[0:K]
			var = pars[K:K+K]
			if bestbadness < 1.:
				prefix = 'K%02d_MR%02d_LSD%02d' % (K, int(round(MAX_RADIUS)+0.01), -1 * LOG10_SQUARED_DEVIATION)
				plot_mixture(pars, prefix, model)
				txtfile = open(prefix + '_' + model + '.txt', "w")
				txtfile.write(str(pars))
				txtfile.close
				print model
				break

if __name__ == '__main__':
	for MAX_RADIUS in [10., 30.]:
		for LOG10_SQUARED_DEVIATION in [-2, -4, -6]:
			main()
