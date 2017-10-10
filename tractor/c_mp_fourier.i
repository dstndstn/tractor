%module(package="tractor") c_mp_fourier

%{
#define SWIG_FILE_WITH_INIT
#define _GNU_SOURCE 1
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
%}

%include "numpy.i"

%init %{
    // numpy
    import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {
    (double *amps, int amps_len),
    (double *v, int v_len),
    (double *w, int w_len)
};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
    (double *means, int means_dim1, int means_dim2)
};
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {
    (double *vars, int vars_dim1, int vars_dim2, int vars_dim3)
};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {
    (double *out, int out_dim1, int out_dim2)
};

%inline %{

static void mixture_profile_fourier_transform(double *amps, int amps_len,
                                              double *means, int means_dim1, int means_dim2,
                                              double *vars, int vars_dim1, int vars_dim2, int vars_dim3,
                                              double *v, int v_len,
                                              double *w, int w_len,
                                              double *out, int out_dim1, int out_dim2)
{
    const double twopisquare = -2. * M_PI * M_PI;

    int K = amps_len;
    int NV = v_len;
    int NW = w_len;
    int i, j, k;

    double *s = (double*)malloc(sizeof(double) * NV);

    for (j = 0; j < NW; j++) {
        for (i = 0; i < NV; i++) {
            s[i] = 0;
            for (k = 0; k < K; k++) {
                double a, b, d;
                int offset = k * 4;

                a = vars[offset];
                b = vars[offset + 1];
                d = vars[offset + 3];

                s[i] += amps[k] * exp(twopisquare * (a *  v[i] * v[i] +
                                                  2. * b * v[i] * w[j] +
                                                  d * w[j] * w[j]));
            }
            out[NV * j + i] = s[i];
        }
    }
    free(s);
    return;
}

%}

// static PyObject* mixture_profile_fourier_transform(
//     PyObject* po_amps,
//     PyObject* po_means,
//     PyObject* po_vars,
//     PyObject* po_v,
//     PyObject* po_w
//     ) {

    // npy_intp K, NW,NV;
    // const npy_intp D = 2;
    // npy_intp i,j,k;
    // double* amps, *means, *vars, *vv, *ww;
    // PyObject* np_F;
    // double* f;
    // npy_intp dims[2];

    // PyArrayObject *np_amps, *np_means, *np_vars, *np_v, *np_w;
    
    // if (!PyArray_Check(po_amps) || !PyArray_Check(po_means) ||
    //     !PyArray_Check(po_vars) || !PyArray_Check(po_v) ||
    //     !PyArray_Check(po_w)) {
    //     PyErr_SetString(PyExc_ValueError, "Expected numpy arrays");
    //     return NULL;
    // }

    // np_amps = (PyArrayObject*)po_amps;
    // np_means = (PyArrayObject*)po_means;
    // np_vars = (PyArrayObject*)po_vars;
    // np_v = (PyArrayObject*)po_v;
    // np_w = (PyArrayObject*)po_w;
    
    // if ((PyArray_TYPE(np_amps) != NPY_DOUBLE) ||
    //     (PyArray_TYPE(np_means ) != NPY_DOUBLE) ||
    //     (PyArray_TYPE(np_vars) != NPY_DOUBLE) ||
    //     (PyArray_TYPE(np_v)    != NPY_DOUBLE) ||
    //     (PyArray_TYPE(np_w)    != NPY_DOUBLE)) {
    //     PyErr_SetString(PyExc_ValueError, "Expected numpy double arrays");
    //     return NULL;
    // }

    // if (PyArray_NDIM(np_amps) != 1) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'amps' to be 1-d");
    //     return NULL;
    // }
    // K = PyArray_DIM(np_amps, 0);
    // if (PyArray_NDIM(np_means) != 2) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'means' to be 2-d");
    //     return NULL;
    // }
    // if ((PyArray_DIM(np_means, 0) != K) ||
    //     (PyArray_DIM(np_means, 1) != D)) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'means' to be K x D");
    //     return NULL;
    // }
    // if (PyArray_NDIM(np_vars) != 3) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'vars' to be 3-d");
    //     return NULL;
    // }
    // if ((PyArray_DIM(np_vars, 0) != K) ||
    //     (PyArray_DIM(np_vars, 1) != D) ||
    //     (PyArray_DIM(np_vars, 2) != D)) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'vars' to be K x D x D");
    //     return NULL;
    // }

    // if (PyArray_NDIM(np_v) != 1) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'v' to be 1-d");
    //     return NULL;
    // }
    // if (PyArray_NDIM(np_w) != 1) {
    //     PyErr_SetString(PyExc_ValueError, "Expected 'w' to be 1-d");
    //     return NULL;
    // }

    // means = PyArray_DATA(np_means);
    
    // for (k=0; k<K; k++) {
    //     if ((means[k*D] != means[0]) ||
    //         (means[k*D+1] != means[1])) {
    //         PyErr_SetString(PyExc_ValueError, "Assume all means are equal");
    //         return NULL;
    //     }
    // }

    // double mu0 = means[0];
    // double mu1 = means[1];

    // int zeromean = (mu0 == 0.) && (mu1 == 0.);

    // NV = PyArray_DIM(np_v, 0);
    // NW = PyArray_DIM(np_w, 0);

    // dims[0] = NW;
    // dims[1] = NV;

    // if (zeromean)
    //     np_F = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    // else
    //     np_F = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    // f = PyArray_DATA((PyArrayObject*)np_F);
    // amps = PyArray_DATA(np_amps);
    // vars = PyArray_DATA(np_vars);
    // vv = PyArray_DATA(np_v);
    // ww = PyArray_DATA(np_w);

    // if (zeromean)
    //     memset(f, 0, NV*NW*sizeof(double));
    // else
    //     memset(f, 0, 2*NV*NW*sizeof(double));

    // double* ff = f;
    // for (j=0; j<NW; j++) {
    //     for (i=0; i<NV; i++) {
    //         double s = 0;
    //         double* V = vars;
    //         double twopisquare = -2. * M_PI * M_PI;
    //         for (k=0; k<K; k++) {
    //             double a, b, d;
    //             a = *V;
    //             V++;
    //             b = *V;
    //             V++;
    //             // skip c
    //             V++;
    //             d = *V;
    //             V++;

    //             s += amps[k] * exp(twopisquare * (a *  vv[i]*vv[i] +
    //                                               2.*b*vv[i]*ww[j] +
    //                                               d *  ww[j]*ww[j]));
    //         }
    //         if (zeromean) {
    //             *ff = s;
    //             ff++;
    //         } else {
    //             double angle = -2. * M_PI * (mu0 * vv[i] + mu1 * ww[j]);
    //             *ff = s * cos(angle);
    //             ff++;
    //             *ff = s * sin(angle);
    //             ff++;
    //         }
    //     }
    // }
    // return np_F;
// }