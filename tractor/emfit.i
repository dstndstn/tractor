%module(package="tractor") emfit

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

#define ERR(x, ...)                             \
    printf(x, ## __VA_ARGS__)

    // ASSUME "amp", "mean", and "var" have been initialized.

    // _reg: Inverse-Wishart prior on variance (with hard-coded
    // variance prior I), with strength alpha.
    static int em_fit_2d_reg(PyObject* np_img, int x0, int y0,
                             PyObject* np_amp,
                             PyObject* np_mean,
                             PyObject* np_var,
                             double alpha) {
        npy_intp i, N, K, k;
        npy_intp ix, iy;
        npy_intp NX, NY;
        const npy_intp D = 2;
        double* Z = NULL;
        double* scale = NULL, *ivar = NULL;
        int step;
        double tpd;
        int result;

        PyArray_Descr* dtype = PyArray_DescrFromType(PyArray_DOUBLE);
        int req = NPY_C_CONTIGUOUS | NPY_ALIGNED;
        int reqout = req | NPY_WRITEABLE | NPY_UPDATEIFCOPY;

        double* amp;
        double* mean;
        double* var;
        double* img;

        tpd = pow(2.*M_PI, D);

        Py_INCREF(dtype);
        np_img = PyArray_FromAny(np_img, dtype, 2, 2, req, NULL);
        if (!np_img) {
            ERR("img wasn't the type expected");
            Py_DECREF(dtype);
            return -1;
        }
        Py_INCREF(dtype);
        np_amp = PyArray_FromAny(np_amp, dtype, 1, 1, reqout, NULL);
        if (!np_amp) {
            ERR("amp wasn't the type expected");
            Py_DECREF(np_img);
            Py_DECREF(dtype);
            return -1;
        }
        Py_INCREF(dtype);
        np_mean = PyArray_FromAny(np_mean, dtype, 2, 2, reqout, NULL);
        if (!np_mean) {
            ERR("mean wasn't the type expected");
            Py_DECREF(np_img);
            Py_DECREF(np_amp);
            Py_DECREF(dtype);
            return -1;
        }
        Py_INCREF(dtype);
        np_var = PyArray_FromAny(np_var, dtype, 3, 3, reqout, NULL);
        if (!np_var) {
            ERR("var wasn't the type expected");
            Py_DECREF(np_img);
            Py_DECREF(np_amp);
            Py_DECREF(np_mean);
            Py_DECREF(dtype);
            return -1;
        }

        K = PyArray_DIM(np_amp, 0);
        // printf("K=%i\n", K);
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

        // printf("NX=%i, NY=%i; N=%i\n", NX, NY, N);

        for (step=0; step<1000; step++) {
            double x,y;
            double wsum[K];
            double qsum = 0.0;

            /*
            printf("step=%i\n", step);
            printf("weights ");
            for (k=0; k<K; k++)
                printf("%g ", amp[k]);
            printf("\n");
            printf("means ");
            for (k=0; k<K; k++) {
                printf("[ ");
                for (d=0; d<D; d++)
                    printf("%g ", mean[k*D+d]);
                printf("] ");
            }
            printf("\n");
            printf("vars ");
            for (k=0; k<K; k++) {
                printf("[ ");
                for (d=0; d<D*D; d++)
                    printf("%g ", var[k*D*D+d]);
                printf("] ");
            }
            printf("\n");
             */

            memset(Z, 0, K*N*sizeof(double));
            for (k=0; k<K; k++) {
                // ASSUME ordering
                double* V = var + k*D*D;
                double* I = ivar + k*D*D;
                double det;
                printf("var[%i]: %.3f,%.3f,%.3f,%.3f\n", k, V[0], V[1], V[2], V[3]);
                det = V[0]*V[3] - V[1]*V[2];
                if (det <= 0.0) {
                    printf("det = %g\n", det);
                    ERR("Got non-positive determinant\n");
                    result = -1;
                    goto cleanup;
                }
                I[0] =  V[3] / det;
                I[1] = -V[1] / det;
                I[2] = -V[2] / det;
                I[3] =  V[0] / det;
                scale[k] = amp[k] / sqrt(tpd * det);
            }

            // printf("E step...\n");
            i = 0;
            for (iy=0; iy<NY; iy++) {
                for (ix=0; ix<NX; ix++) {
                    double zi;
                    double zsum = 0;
                    x = x0 + ix;
                    y = y0 + iy;
                    i = (iy*NX + ix);
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
                        // printf("Z(i=%i, k=%i) = %g\n", i, k, zi);
                        zsum += zi;
                        //assert(i == (iy*NX + ix));
                    }
                    // printf("i=%i, ix,iy=%i,%i  zsum=%g\n", i, ix, iy, zsum);
                    if (zsum == 0)
                        continue;
                    for (k=0; k<K; k++) {
                      if (Z[i*K+k] > 0) {
                        qsum += log(Z[i*K+k]) * Z[i*K+k] / zsum;
                        //printf("lnp %g, Zfrac %g, qsum %g\n", log(Z[i*K+k]), Z[i*K+k]/zsum, qsum);
                      }
                      Z[i*K + k] /= zsum;
                      // printf("normalized Z(i=%i, k=%i) = %g\n", i, k, Z[i*K+k]);
                    }
                    //i++;
                }
            }

            printf("Q: %g\n", qsum);
            // Q = np.sum(fore*lfg + back*lbg)

            // printf("M mu...\n");
            // M step: mu
            memset(mean, 0, K*D*sizeof(double));
            for (k=0; k<K; k++) {
                wsum[k] = 0;
                i = 0;
                for (iy=0; iy<NY; iy++) {
                    for (ix=0; ix<NX; ix++) {
                        double wi = img[i] * Z[i*K + k];
                        x = x0 + ix;
                        y = y0 + iy;
                        mean[k*D + 0] += wi * x;
                        mean[k*D + 1] += wi * y;
                        wsum[k] += wi;
                        i++;
                    }
                }
                mean[k*D + 0] /= wsum[k];
                mean[k*D + 1] /= wsum[k];
            }

            // M step: var
            // printf("M var...\n");
            memset(var, 0, K*D*D*sizeof(double));
            for (k=0; k<K; k++) {
                var[k*D*D + 0] = alpha;
                var[k*D*D + 3] = alpha;
                i = 0;
                for (iy=0; iy<NY; iy++) {
                    for (ix=0; ix<NX; ix++) {
                        double dx, dy, wi;
                        x = x0 + ix;
                        y = y0 + iy;
                        dx = x - mean[k*D+0];
                        dy = y - mean[k*D+1];
                        wi = img[i] * Z[i*K + k];
                        var[k*D*D + 0] += wi * dx*dx;
                        var[k*D*D + 1] += wi * dx*dy;
                        var[k*D*D + 2] += wi * dy*dx;
                        var[k*D*D + 3] += wi * dy*dy;
                        i++;
                    }
                }
                for (i=0; i<(D*D); i++)
                    var[k*D*D + i] /= (wsum[k] + alpha);
            }

            // M step: amp
            // printf("M amp...\n");
            for (k=0; k<K; k++)
                amp[k] = wsum[k];
        }
        result = 0;
        
    cleanup:
        free(Z);
        free(scale);
        free(ivar);

        Py_DECREF(np_img);
        Py_DECREF(np_amp);
        Py_DECREF(np_mean);
        Py_DECREF(np_var);
        Py_DECREF(dtype);

        return result;
    }



    static int em_fit_2d(PyObject* np_img, int x0, int y0,
                         PyObject* np_amp,
                         PyObject* np_mean,
                         PyObject* np_var) {
        return em_fit_2d_reg(np_img, x0, y0, np_amp, np_mean, np_var, 0.0);
    }

    %}
