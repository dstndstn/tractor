%module emfit

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

#define ERR(x, ...)								\
	printf(x, ## __VA_ARGS__)

	// ASSUME "amp", "mean", and "var" have been initialized.
    static int em_fit_2d(PyObject* np_img, int x0, int y0,
						 PyObject* np_amp,
						 PyObject* np_mean,
						 PyObject* np_var) {
        int i, N, K, k, d;
		int ix, iy;
		int NX, NY;
		const int D = 2;
		double* Z;
		double* scale, *ivar;
		int step;
		double tpd;

		PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
		int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
		int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;

		double* amp;
		double* mean;
		double* var;
		double* img;

		tpd = pow(2.*M_PI, D);

		np_img = PyArray_FromAny(np_img, dtype, 2, 2, req, NULL);
		np_amp = PyArray_FromAny(np_amp, dtype, 1, 1, reqout, NULL);
		np_mean = PyArray_FromAny(np_mean, dtype, 2, 2, reqout, NULL);
		np_var = PyArray_FromAny(np_var, dtype, 3, 3, reqout, NULL);

		K = PyArray_DIM(np_amp, 0);
		if ((PyArray_DIM(np_mean, 0) != K) ||
			(PyArray_DIM(np_mean, 1) != D)) {
            ERR("np_mean must be K x D");
            return -1;
		}
		if ((PyArray_DIM(np_var, 0) != K) ||
			(PyArray_DIM(np_var, 1) != D) ||
			(PyArray_DIM(np_var, 2) != D)) {
            ERR("np_var must be K x D x D");
            return -1;
		}
		NY = PyArray_DIM(np_img, 0);
		NX = PyArray_DIM(np_img, 1);

        amp  = PyArray_DATA(np_amp);
        mean = PyArray_DATA(np_mean);
        var  = PyArray_DATA(np_var);
        img  = PyArray_DATA(np_img);

		N = NX*NY;
		Z = malloc(K * N * sizeof(double));
		assert(Z);
		scale = malloc(K * sizeof(double));
		ivar = malloc(K * D * D * sizeof(double));
		assert(scale && ivar);

		for (step=0; step<1000; step++) {
			double x,y;
			double wsum[K];

			memset(Z, 0, K*N*sizeof(double));
			for (k=0; k<K; k++) {
				// ASSUME ordering
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
			for (iy=0; iy<NY; iy++) {
				for (ix=0; ix<NX; ix++) {
					double zi;
					double zsum = 0;
					x = x0 + ix;
					y = y0 + iy;
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

						zi = scale[k] * exp(-0.5 * dsq);
						Z[i*K + k] = zi;
						zsum += zi;
						assert(i == (iy*NX + ix));
					}
					if (zsum == 0)
						continue;
					for (k=0; k<K; k++) {
						Z[i*K + k] /= zsum;
					}
				}
				i++;
			}

			// M: mu
			memset(mean, 0, K*D*sizeof(double));
			i = 0;
			for (iy=0; iy<NY; iy++) {
				for (ix=0; ix<NX; ix++) {
					x = x0 + ix;
					y = y0 + iy;
					for (k=0; k<K; k++) {
						double wi = img[i] * Z[i*K + k];
						mean[k*D + 0] += wi * x;
						mean[k*D + 1] += wi * y;
						wsum[k] += wi;
					}
				}
				i++;
			}
			for (k=0; k<K; k++) {
				for (d=0; d<D; d++)
					mean[k*D+d] /= wsum[k];
			}

			// M: var
			memset(var, 0, K*D*D*sizeof(double));
			i = 0;
			for (iy=0; iy<NY; iy++) {
				for (ix=0; ix<NX; ix++) {
					x = x0 + ix;
					y = y0 + iy;
					for (k=0; k<K; k++) {
						double dx = x - mean[k*D+0];
						double dy = y - mean[k*D+1];
						double wi = img[i] * Z[i*K + k];
						var[k*D*D + 0] += wi * dx*dx;
						var[k*D*D + 1] += wi * dx*dy;
						var[k*D*D + 2] += wi * dy*dx;
						var[k*D*D + 3] += wi * dy*dy;
					}
				}
				i++;
			}
			for (k=0; k<K; k++) {
				for (i=0; i<(D*D); d++)
					var[k*D*D + i] /= wsum[k];
			}

			// M: amp
			for (k=0; k<K; k++)
				amp[k] = wsum[k];
		}
		
		free(Z);
		free(scale);
		free(ivar);
		return 0;
	}


	%}
