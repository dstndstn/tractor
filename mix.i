%module mix

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

#define CHECK_DOUBLE(X)\
	do {\
		if (PyArray_TYPE(X) != PyArray_DOUBLE) {\
			printf("array \"" #X "\" must contain doubles (it has type %i)\n", PyArray_TYPE(X)); \
			return -1;\
		}\
	} while (0)

	// PyErr_SetString(PyExc_ValueError, x, __VA_ARGS__)

    static int c_gauss_2d(PyObject* np_pos, PyObject* np_scale,
						  PyObject* np_mean, PyObject* np_ivar,
						  PyObject* np_result) {
        int i, N, d, K, k;
		const int D = 2;
		double *pos, *scale, *mean, *ivar, *result;

        CHECK_DOUBLE(np_pos);
		CHECK_DOUBLE(np_scale);
		CHECK_DOUBLE(np_mean);
		CHECK_DOUBLE(np_ivar);
		CHECK_DOUBLE(np_result);
        if (PyArray_NDIM(np_pos) != 2) {
            ERR("np_pos must have dim 2");
            return -1;
        }
		N = PyArray_DIM(np_pos, 0);
		d = PyArray_DIM(np_pos, 1);
		//printf("N=%i, d=%i, D=%i\n", N, D, D);
		if (d != D) {
            ERR("must be 2-D");
            return -1;
		}
        if (PyArray_NDIM(np_scale) != 1) {
            ERR("np_scale must have dim 1");
            return -1;
        }
		K = PyArray_DIM(np_scale, 0);
		//printf("K=%i\n", K);

        if (PyArray_NDIM(np_mean) != 2) {
            ERR("np_mean must have dim 2");
            return -1;
        }
		if ((PyArray_DIM(np_mean, 0) != K) ||
			(PyArray_DIM(np_mean, 1) != D)) {
            ERR("np_mean must be K x D");
            return -1;
		}

        if (PyArray_NDIM(np_ivar) != 3) {
            ERR("np_ivar must have dim 3");
            return -1;
        }
		if ((PyArray_DIM(np_ivar, 0) != K) ||
			(PyArray_DIM(np_ivar, 1) != D) ||
			(PyArray_DIM(np_ivar, 2) != D)) {
            ERR("np_ivar must be K x D x D");
            return -1;
		}

        if (PyArray_NDIM(np_result) != 1) {
            ERR("np_result must have dim 1");
            return -1;
        }
		if (PyArray_DIM(np_result, 0) != N) {
            ERR("np_result must be size N");
            return -1;
		}

		{
			PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
			int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
			int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;
			np_pos = PyArray_FromAny(np_pos, dtype, 2, 2, req, NULL);
			np_scale = PyArray_FromAny(np_scale, dtype, 1, 1, req, NULL);
			np_mean = PyArray_FromAny(np_mean, dtype, 2, 2, req, NULL);
			np_ivar = PyArray_FromAny(np_ivar, dtype, 3, 3, req, NULL);
			np_result = PyArray_FromAny(np_result, dtype, 1, 1, reqout, NULL);
		}

        pos = PyArray_DATA(np_pos);
        scale = PyArray_DATA(np_scale);
        mean = PyArray_DATA(np_mean);
        ivar = PyArray_DATA(np_ivar);
        result = PyArray_DATA(np_result);
		
		//printf("mean strides: %i, %i\n", PyArray_STRIDES(np_mean)[0], PyArray_STRIDES(np_mean)[1]);
		//printf("pos strides: %i, %i\n", PyArray_STRIDES(np_pos)[0], PyArray_STRIDES(np_pos)[1]);

		for (i=0; i<N; i++) {
			for (k=0; k<K; k++) {
				double dsq;
				double dx,dy;
				/*
				 assert(PyArray_GETPTR2(np_pos, i, 0) == (pos + i*D+0));
				 assert(PyArray_GETPTR2(np_pos, i, 1) == (pos + i*D+1));
				 assert(PyArray_GETPTR2(np_mean, k, 0) == (mean + k*D+0));
				 assert(PyArray_GETPTR2(np_mean, k, 1) == (mean + k*D+1));
				 */
				dx = pos[i*D+0] - mean[k*D+0];
				dy = pos[i*D+1] - mean[k*D+1];
				dsq = ivar[k*D*D + 0] * dx * dx
					+ ivar[k*D*D + 1] * dx * dy
					+ ivar[k*D*D + 2] * dx * dy
					+ ivar[k*D*D + 3] * dy * dy;
				if (dsq >= 700)
					continue;
				result[i] += scale[k] * exp(-0.5 * dsq);
			}
		}
        return 0;
    }

	%}

