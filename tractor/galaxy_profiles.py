from numpy import exp, power, atleast_1d, isscalar

### copied from photo: makeprof.c and phFitobj.h

# r in half-light radii
def profile_dev(r):
	'''
	>>> print round(profile_dev(1.), 3)
	511.804

	>>> print round(profile_dev(0.), 3)
	61296.259

	>>> print round(profile_dev(7.), 3)
	4.191

	>>> print round(profile_dev(8.), 3)
	0.0
	'''
	DEVOUT = 8.0
	DEVCUT = 7.*DEVOUT/8.

	def dev_nc(r):
		IE_DEV = 512.0
		DEFAC = -7.66925
		return IE_DEV * exp(DEFAC*(power(r*r + 0.0004, 0.125) - 1.0))

	rr = atleast_1d(r)
	p = dev_nc(rr)
	p[rr > DEVCUT] *= (1 - ((rr[rr>DEVCUT] - DEVCUT)/(DEVOUT - DEVCUT))**2)**2
	p[rr > DEVOUT] = 0.
	if isscalar(r):
		return p[0]
	return p

# r in half-light radii
def profile_exp(r):
	'''
	>>> print round(profile_exp(1.), 3)
	2048.0

	>>> print round(profile_exp(0.), 3)
	10970.542

	>>> print round(profile_exp(3.), 3)
	71.373

	>>> print round(profile_exp(3.99), 3)
	0.005

	>>> print round(profile_exp(4.), 3)
	0.0
	'''

	def exp_nc(r):
		# Surface brightness at r_e
		IE_EXP = 2048.0
		EXPFAC = -1.67835
		return IE_EXP * exp(EXPFAC*(r - 1.))

	EXPOUT = 4.
	EXPCUT = 3.*EXPOUT/4.
	rr = atleast_1d(r)
	p = exp_nc(rr)
	p[rr > EXPCUT] *= (1 - ((rr[rr>EXPCUT] - EXPCUT)/(EXPOUT - EXPCUT))**2)**2
	p[rr > EXPOUT] = 0.
	if isscalar(r):
		return p[0]
	return p

if __name__ == '__main__':
	import doctest
	doctest.testmod()
