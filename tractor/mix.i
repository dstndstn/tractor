%module(package="tractor") mix

%include <typemaps.i>

%{
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
#include <sys/param.h>
	%}

%init %{
	// numpy
	import_array();
	%}



%inline %{

#define ERR(x, ...) \
	printf(x, ## __VA_ARGS__)
	// PyErr_SetString(PyExc_ValueError, x, __VA_ARGS__)

    static int c_gauss_2d(PyObject* ob_pos, PyObject* ob_amp,
						  PyObject* ob_mean, PyObject* ob_var,
						  PyObject* ob_result) {
        int i, N, d, K, k;
		const int D = 2;
		double *pos, *amp, *mean, *var, *result;
		PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
		double tpd;
		PyObject *np_pos=NULL, *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
		double *scale=NULL, *ivar=NULL;
		int rtn = -1;

		tpd = pow(2.*M_PI, D);

		Py_INCREF(dtype);
		Py_INCREF(dtype);
		Py_INCREF(dtype);
		Py_INCREF(dtype);
		Py_INCREF(dtype);
		np_pos = PyArray_FromAny(ob_pos, dtype, 2, 2, req, NULL);
		np_amp = PyArray_FromAny(ob_amp, dtype, 1, 1, req, NULL);
		np_mean = PyArray_FromAny(ob_mean, dtype, 2, 2, req, NULL);
		np_var = PyArray_FromAny(ob_var, dtype, 3, 3, req, NULL);
		np_result = PyArray_FromAny(ob_result, dtype, 1, 1, reqout, NULL);
		Py_CLEAR(dtype);

		if (!(np_pos && np_amp && np_mean && np_var && np_result)) {
			if (!np_pos) {
				ERR("pos wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!np_amp) {
				ERR("amp wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!np_mean) {
				ERR("mean wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!np_var) {
				ERR("var wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!np_result) {
				ERR("result wasn't the type expected");
				Py_DECREF(dtype);
			}
			goto bailout;
		}

		N = PyArray_DIM(np_pos, 0);
		d = PyArray_DIM(np_pos, 1);
		if (d != D) {
            ERR("must be 2-D");
			goto bailout;
		}
		K = PyArray_DIM(np_amp, 0);
		//printf("K=%i\n", K);
		if ((PyArray_DIM(np_mean, 0) != K) ||
			(PyArray_DIM(np_mean, 1) != D)) {
			ERR("np_mean must be K x D");
			goto bailout;
		}
		if ((PyArray_DIM(np_var, 0) != K) ||
			(PyArray_DIM(np_var, 1) != D) ||
			(PyArray_DIM(np_var, 2) != D)) {
			ERR("np_var must be K x D x D");
			goto bailout;
		}
		if (PyArray_DIM(np_result, 0) != N) {
            ERR("np_result must be size N");
			goto bailout;
		}

        pos = PyArray_DATA(np_pos);
        amp = PyArray_DATA(np_amp);
        mean = PyArray_DATA(np_mean);
        var = PyArray_DATA(np_var);
        result = PyArray_DATA(np_result);

		scale = malloc(K * sizeof(double));
		ivar = malloc(K * D * D * sizeof(double));

		for (k=0; k<K; k++) {
			double* V = var + k*D*D;
			double* I = ivar + k*D*D;
			double det;
			det = V[0]*V[3] - V[1]*V[2];
			I[0] =  V[3] / det;
			I[1] = -V[1] / det;
			I[2] = -V[2] / det;
			I[3] =  V[0] / det;
			scale[k] = amp[k] / sqrt(tpd * det);
		}
		
		for (i=0; i<N; i++) {
			for (k=0; k<K; k++) {
				double dsq;
				double dx,dy;
				dx = pos[i*D+0] - mean[k*D+0];
				dy = pos[i*D+1] - mean[k*D+1];
				dsq = ivar[k*D*D + 0] * dx * dx
					+ ivar[k*D*D + 1] * dx * dy
					+ ivar[k*D*D + 2] * dx * dy
					+ ivar[k*D*D + 3] * dy * dy;
				if (dsq >= 100)
					continue;
				result[i] += scale[k] * exp(-0.5 * dsq);
			}
		}
		rtn = 0;

	bailout:
		free(scale);
		free(ivar);
		Py_XDECREF(np_pos);
		Py_XDECREF(np_amp);
		Py_XDECREF(np_mean);
		Py_XDECREF(np_var);
		Py_XDECREF(np_result);
        return rtn;
    }


	static int _get_np(PyObject* ob_amp,
					   PyObject* ob_mean,
					   PyObject* ob_var,
					   PyObject* ob_result,
					   int NX, int NY,
					   int* K,
					   PyObject **np_amp,
					   PyObject **np_mean,
					   PyObject **np_var,
					   PyObject **np_result) {
		PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
		const int D = 2;

		Py_INCREF(dtype);
		Py_INCREF(dtype);
		Py_INCREF(dtype);
		Py_INCREF(dtype);
		*np_amp = PyArray_FromAny(ob_amp, dtype, 1, 1, req, NULL);
		*np_mean = PyArray_FromAny(ob_mean, dtype, 2, 2, req, NULL);
		*np_var = PyArray_FromAny(ob_var, dtype, 3, 3, req, NULL);
		*np_result = PyArray_FromAny(ob_result, dtype, 2, 2, reqout, NULL);
		Py_CLEAR(dtype);

		if (!(*np_amp && *np_mean && *np_var && *np_result)) {
			if (!*np_amp) {
				ERR("amp wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!*np_mean) {
				ERR("mean wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!*np_var) {
				ERR("var wasn't the type expected");
				Py_DECREF(dtype);
			}
			if (!*np_result) {
				ERR("result wasn't the type expected");
				Py_DECREF(dtype);
			}
			return 1;
		}
		*K = PyArray_DIM(*np_amp, 0);
		if ((PyArray_DIM(*np_mean, 0) != *K) ||
			(PyArray_DIM(*np_mean, 1) != D)) {
            ERR("np_mean must be K x D");
			return 1;
		}
		if ((PyArray_DIM(*np_var, 0) != *K) ||
			(PyArray_DIM(*np_var, 1) != D) ||
			(PyArray_DIM(*np_var, 2) != D)) {
            ERR("np_var must be K x D x D");
			return 1;
		}
		if ((PyArray_DIM(*np_result, 0) != NY) ||
			(PyArray_DIM(*np_result, 1) != NX)) {
            ERR("np_result must be size NY x NX");
			return 1;
		}
		return 0;
	}



    static int c_gauss_2d_grid(double xlo, double xstep, int NX,
							   double ylo, double ystep, int NY,
							   PyObject* ob_amp,
							   PyObject* ob_mean, PyObject* ob_var,
							   PyObject* ob_result) {
        int i, K, k;
		const int D = 2;
		double *amp, *mean, *var, *result;
		double tpd;
		double* scale = NULL, *ivar = NULL;
		int ix, iy;
		double x, y;
		PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
		int rtn = -1;

		tpd = pow(2.*M_PI, D);

		if (_get_np(ob_amp, ob_mean, ob_var, ob_result, NX, NY,
					&K, &np_amp, &np_mean, &np_var, &np_result))
			goto bailout;

        amp = PyArray_DATA(np_amp);
        mean = PyArray_DATA(np_mean);
        var = PyArray_DATA(np_var);
        result = PyArray_DATA(np_result);

		scale = malloc(K * sizeof(double));
		ivar = malloc(K * D * D * sizeof(double));

		for (k=0; k<K; k++) {
			double* V = var + k*D*D;
			double* I = ivar + k*D*D;
			double det;
			det = V[0]*V[3] - V[1]*V[2];
			I[0] =  V[3] / det;
			I[1] = -V[1] / det;
			I[2] = -V[2] / det;
			I[3] =  V[0] / det;
			scale[k] = amp[k] / sqrt(tpd * det);
		}
		
		i = 0;
		y = ylo;
		for (iy=0; iy<NY; iy++) {
			x = xlo;
			for (ix=0; ix<NX; ix++) {
				for (k=0; k<K; k++) {
					double dsq;
					double dx,dy;
					dx = x - mean[k*D+0];
					dy = y - mean[k*D+1];
					dsq = ivar[k*D*D + 0] * dx * dx
						+ ivar[k*D*D + 1] * dx * dy
						+ ivar[k*D*D + 2] * dx * dy
						+ ivar[k*D*D + 3] * dy * dy;
					if (dsq >= 100)
						continue;
					result[i] += scale[k] * exp(-0.5 * dsq);
					assert(i == (iy*NX + ix));
				}
				i++;
				x += xstep;
			}
			y += ystep;
		}
		rtn = 0;

	bailout:
		free(scale);
		free(ivar);
		Py_XDECREF(np_amp);
		Py_XDECREF(np_mean);
		Py_XDECREF(np_var);
		Py_XDECREF(np_result);
        return rtn;
    }

	static double eval_g(double I[4], double dx, double dy) {
		double dsq = (I[0] * dx * dx +
					  I[1] * dx * dy +
					  I[3] * dy * dy);
		if (dsq < -100)
			// ~ 1e-44
			return 0.0;
		return exp(dsq);
	}

    static int c_gauss_2d_approx(int xc, int yc, double minval, int S,
								 PyObject* ob_amp,
								 PyObject* ob_mean,
								 PyObject* ob_var,
								 PyObject* ob_result) {
		double *amp, *mean, *var, *result;
		const int D=2;
        int K, k;
		int dx, dyabs;
		int rtn = -1;
		PyObject *np_amp=NULL, *np_mean=NULL, *np_var=NULL, *np_result=NULL;
		double tpd;
		int Shalf = S/2;
		assert((S%2) == 1);
		tpd = pow(2.*M_PI, D);

		if (_get_np(ob_amp, ob_mean, ob_var, ob_result, S, S,
					&K, &np_amp, &np_mean, &np_var, &np_result))
			goto bailout;

        amp = PyArray_DATA(np_amp);
        mean = PyArray_DATA(np_mean);
        var = PyArray_DATA(np_var);
        result = PyArray_DATA(np_result);

		for (k=0; k<K; k++) {
			// We symmetrize the covariance matrix,
			// so we don't actually set V[2] or I[2].
			// We also scale the the I
			
			double V[4];
			double I[4];
			double det;
			double isc;
			double scale;
			double mx,my;
			double mv;

			V[0] = var[k*D*D + 0];
			V[1] = (var[k*D*D + 1] + var[k*D*D + 2])*0.5;
			V[3] = var[k*D*D + 3];
			det = V[0]*V[3] - V[1]*V[1];
			// we fold the -0.5 in the Gaussian exponent term in here...
			isc = -0.5 / det;
			I[0] =  V[3] * isc;
			// we also fold in the 2*dx*dy term here
			I[1] = -V[1] * isc * 2.0;
			I[3] =  V[0] * isc;
			scale = amp[k] / sqrt(tpd * det);
			mx = mean[k*D+0];
			my = mean[k*D+1];
			mv = minval * amp[k] / scale;
			//printf("minval %g, amp %g, scale %g, mv %g\n", minval, amp[k], scale, mv);
			for (dyabs=0; dyabs<=Shalf; dyabs++) {
				int dysign;
				int ngood = 0;
				for (dysign=-1; dysign<=1; dysign+=2) {
					int dy;
					int dxc;
					double g;
					int dir;
					double* rrow;
					if ((dyabs == 0) && (dysign == 1))
						continue;
					dy = dyabs * dysign;
					// mean of conditional distribution of dx given dy
					dxc = lround(V[1] / V[3] * dy);
					//printf("dy=%i, dxc=%i\n", dy, dxc);
					dxc = MIN(dxc,  Shalf);
					dxc = MAX(dxc, -Shalf);
					dx = dxc;
					// eval at dx=0
					// eval at dx = +- 1, ...
					// stop altogether if neither are accepted
					g = eval_g(I, (xc + dx) - mx, (yc + dy) - my);
					//printf("g = %g vs mv %g\n", g, mv);
					rrow = result + (dy + Shalf)*S + Shalf;
					rrow[dx] += scale * g;
					if (g > mv)
						ngood++;
					for (dir=0; dir<2; dir++) {
						for (dx = dxc + (dir ? 1 : -1); (dir ? dx <= Shalf : dx >= -Shalf); dir ? dx++ : dx--) {
							g = eval_g(I, (xc+dx) - mx, (yc+dy) - my);
							//printf("dx %i, g = %g vs mv %g\n", dx, g, mv);
							rrow[dx] += scale * g;
							if (g > mv)
								ngood++;
							else
								break;
						}
					}
				}
				// If this whole row (+ and -) was all < mv, we're done
				if (ngood == 0)
					break;
			}
		}
		rtn = 0;
	bailout:
		Py_XDECREF(np_amp);
		Py_XDECREF(np_mean);
		Py_XDECREF(np_var);
		Py_XDECREF(np_result);
		return rtn;
    }


	%}

