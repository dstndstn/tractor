%module(package="tractor") mix

%include <typemaps.i>

%{
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
	%}

%init %{
	// numpy
	import_array();
	%}



%inline %{

#define ERR(x, ...) \
	printf(x, ## __VA_ARGS__)
	// PyErr_SetString(PyExc_ValueError, x, __VA_ARGS__)

    static int c_gauss_2d(PyObject* np_pos, PyObject* np_amp,
						  PyObject* np_mean, PyObject* np_var,
						  PyObject* np_result) {
        int i, N, d, K, k;
		const int D = 2;
		double *pos, *amp, *mean, *var, *result;
		PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
		double tpd;

		double* scale, *ivar;

		tpd = pow(2.*M_PI, D);

		Py_INCREF(dtype);
		np_pos = PyArray_FromAny(np_pos, dtype, 2, 2, req, NULL);
		if (!np_pos) {
			ERR("pos wasn't the type expected");
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_amp = PyArray_FromAny(np_amp, dtype, 1, 1, req, NULL);
		if (!np_amp) {
			ERR("amp wasn't the type expected");
			Py_DECREF(np_pos);
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_mean = PyArray_FromAny(np_mean, dtype, 2, 2, req, NULL);
		if (!np_mean) {
			ERR("mean wasn't the type expected");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_var = PyArray_FromAny(np_var, dtype, 3, 3, req, NULL);
		if (!np_var) {
			ERR("var wasn't the type expected");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_result = PyArray_FromAny(np_result, dtype, 1, 1, reqout, NULL);
		if (!np_result) {
			ERR("result wasn't the type expected");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(dtype);
			return -1;
		}

		N = PyArray_DIM(np_pos, 0);
		d = PyArray_DIM(np_pos, 1);
		//printf("N=%i, d=%i, D=%i\n", N, D, D);
		if (d != D) {
            ERR("must be 2-D");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
		}
		K = PyArray_DIM(np_amp, 0);
		//printf("K=%i\n", K);
		if ((PyArray_DIM(np_mean, 0) != K) ||
			(PyArray_DIM(np_mean, 1) != D)) {
            ERR("np_mean must be K x D");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
		}
		if ((PyArray_DIM(np_var, 0) != K) ||
			(PyArray_DIM(np_var, 1) != D) ||
			(PyArray_DIM(np_var, 2) != D)) {
            ERR("np_var must be K x D x D");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
		}
		if (PyArray_DIM(np_result, 0) != N) {
            ERR("np_result must be size N");
			Py_DECREF(np_pos);
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
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
		free(scale);
		free(ivar);

		Py_DECREF(np_pos);
		Py_DECREF(np_amp);
		Py_DECREF(np_mean);
		Py_DECREF(np_var);
		Py_DECREF(np_result);
		Py_DECREF(dtype);

        return 0;
    }



    static int c_gauss_2d_grid(double xlo, double xstep, int NX,
							   double ylo, double ystep, int NY,
							   PyObject* np_amp,
							   PyObject* np_mean, PyObject* np_var,
							   PyObject* np_result) {
        int i, K, k;
		const int D = 2;
		double *amp, *mean, *var, *result;
		PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
		double tpd;
		double* scale, *ivar;
		int ix, iy;
		double x, y;

		tpd = pow(2.*M_PI, D);

		Py_INCREF(dtype);
		np_amp = PyArray_FromAny(np_amp, dtype, 1, 1, req, NULL);
		if (!np_amp) {
			ERR("amp wasn't the type expected");
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_mean = PyArray_FromAny(np_mean, dtype, 2, 2, req, NULL);
		if (!np_mean) {
			ERR("mean wasn't the type expected");
			Py_DECREF(np_amp);
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_var = PyArray_FromAny(np_var, dtype, 3, 3, req, NULL);
		if (!np_var) {
			ERR("var wasn't the type expected");
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(dtype);
			return -1;
		}
		Py_INCREF(dtype);
		np_result = PyArray_FromAny(np_result, dtype, 2, 2, reqout, NULL);
		if (!np_result) {
			ERR("result wasn't the type expected");
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(dtype);
			return -1;
		}

		K = PyArray_DIM(np_amp, 0);
		if ((PyArray_DIM(np_mean, 0) != K) ||
			(PyArray_DIM(np_mean, 1) != D)) {
            ERR("np_mean must be K x D");
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
		}
		if ((PyArray_DIM(np_var, 0) != K) ||
			(PyArray_DIM(np_var, 1) != D) ||
			(PyArray_DIM(np_var, 2) != D)) {
            ERR("np_var must be K x D x D");
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
		}
		if ((PyArray_DIM(np_result, 0) != NY) ||
			(PyArray_DIM(np_result, 1) != NX)) {
            ERR("np_result must be size NY x NX");
			Py_DECREF(np_amp);
			Py_DECREF(np_mean);
			Py_DECREF(np_var);
			Py_DECREF(np_result);
			Py_DECREF(dtype);
            return -1;
		}

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
		free(scale);
		free(ivar);

		Py_DECREF(np_amp);
		Py_DECREF(np_mean);
		Py_DECREF(np_var);
		Py_DECREF(np_result);
		Py_DECREF(dtype);

        return 0;
    }





	%}

