%module(package="tractor") mp_fourier

%{
#define _GNU_SOURCE 1
#include <numpy/arrayobject.h>
#include <math.h>
#include <assert.h>
    %}

%init %{
    // numpy
    import_array();
    %}

%inline %{

#if 0
 } // fool emacs indenter
#endif

static PyObject* mixture_profile_fourier_transform(
    PyObject* np_amps,
    PyObject* np_means,
    PyObject* np_vars,
    PyObject* np_v,
    PyObject* np_w
    ) {

    npy_intp K, NW,NV;
    const npy_intp D = 2;
    npy_intp i,j,k;
    double* amps, *means, *vars, *vv, *ww;
    PyObject* np_F;
    double* f;
    npy_intp dims[2];

    if (!PyArray_Check(np_amps) || !PyArray_Check(np_means) ||
        !PyArray_Check(np_vars) || !PyArray_Check(np_v) ||
        !PyArray_Check(np_w)) {
        PyErr_SetString(PyExc_ValueError, "Expected numpy arrays");
        return NULL;
    }

    if ((PyArray_TYPE(np_amps) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_means ) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_vars) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_v)    != NPY_DOUBLE) ||
        (PyArray_TYPE(np_w)    != NPY_DOUBLE)) {
        PyErr_SetString(PyExc_ValueError, "Expected numpy double arrays");
        return NULL;
    }

    if (PyArray_NDIM(np_amps) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'amps' to be 1-d");
        return NULL;
    }
    K = PyArray_DIM(np_amps, 0);
    if (PyArray_NDIM(np_means) != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected 'means' to be 2-d");
        return NULL;
    }
    if ((PyArray_DIM(np_means, 0) != K) ||
        (PyArray_DIM(np_means, 1) != D)) {
        PyErr_SetString(PyExc_ValueError, "Expected 'means' to be K x D");
        return NULL;
    }
    if (PyArray_NDIM(np_vars) != 3) {
        PyErr_SetString(PyExc_ValueError, "Expected 'vars' to be 3-d");
        return NULL;
    }
    if ((PyArray_DIM(np_vars, 0) != K) ||
        (PyArray_DIM(np_vars, 1) != D) ||
        (PyArray_DIM(np_vars, 2) != D)) {
        PyErr_SetString(PyExc_ValueError, "Expected 'vars' to be K x D x D");
        return NULL;
    }

    if (PyArray_NDIM(np_v) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'v' to be 1-d");
        return NULL;
    }
    if (PyArray_NDIM(np_w) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'w' to be 1-d");
        return NULL;
    }

    NV = PyArray_DIM(np_v, 0);
    NW = PyArray_DIM(np_w, 0);

    dims[0] = NW;
    dims[1] = NV;
    np_F = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    f = PyArray_DATA(np_F);
    amps = PyArray_DATA(np_amps);
    means = PyArray_DATA(np_means);
    vars = PyArray_DATA(np_vars);
    vv = PyArray_DATA(np_v);
    ww = PyArray_DATA(np_w);

    memset(f, 0, 2*NV*NW*sizeof(double));

    for (k=0; k<K; k++) {
        if ((means[k*D] != means[0]) ||
            (means[k*D+1] != means[1])) {
            PyErr_SetString(PyExc_ValueError, "Assume all means are equal");
            return NULL;
        }
    }

    double mu0 = means[0];
    double mu1 = means[1];
    double* ff = f;
    for (j=0; j<NW; j++) {
        for (i=0; i<NV; i++) {
            double s = 0;
            double* V = vars;
            double twopisquare = -2. * M_PI * M_PI;
            for (k=0; k<K; k++) {
                double a, b, d;
                a = *V;
                V++;
                b = *V;
                V++;
                // skip c
                V++;
                d = *V;
                V++;

                s += amps[k] * exp(twopisquare * (a *  vv[i]*vv[i] +
                                                  2.*b*vv[i]*ww[j] +
                                                  d *  ww[j]*ww[j]));
            }
            double angle = -2. * M_PI * (mu0 * vv[i] + mu1 * ww[j]);
            *ff = s * cos(angle);
            ff++;
            *ff = s * sin(angle);
            ff++;
        }
    }
    return np_F;
}

    %}
