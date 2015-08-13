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
    PyObject* np_w,
    PyObject* np_v
    ) {

    npy_intp K, NW,NV;
    const npy_intp D = 2;
    npy_intp i,j,k;
    double* amps, *means, *vars, *ww, *vv;
    PyObject* np_F;
    double* f;
    npy_intp dims[2];

    if (!PyArray_Check(np_amps) || !PyArray_Check(np_means) ||
        !PyArray_Check(np_vars) || !PyArray_Check(np_w) ||
        !PyArray_Check(np_v)) {
        PyErr_SetString(PyExc_ValueError, "Expected numpy arrays");
        return NULL;
    }

    if ((PyArray_TYPE(np_amps) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_means ) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_vars) != NPY_DOUBLE) ||
        (PyArray_TYPE(np_w)    != NPY_DOUBLE) ||
        (PyArray_TYPE(np_v)    != NPY_DOUBLE)) {
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

    if (PyArray_NDIM(np_w) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'w' to be 1-d");
        return NULL;
    }
    if (PyArray_NDIM(np_v) != 1) {
        PyErr_SetString(PyExc_ValueError, "Expected 'v' to be 1-d");
        return NULL;
    }

    NW = PyArray_DIM(np_w, 0);
    NV = PyArray_DIM(np_v, 0);

    dims[0] = NV;
    dims[1] = NW;
    np_F = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    f = PyArray_DATA(np_F);
    amps = PyArray_DATA(np_amps);
    means = PyArray_DATA(np_means);
    vars = PyArray_DATA(np_vars);
    ww = PyArray_DATA(np_w);
    vv = PyArray_DATA(np_v);

    memset(f, 0, 2*NW*NV*sizeof(double));

    for (k=0; k<K; k++) {
        if ((means[k*D] != means[0]) ||
            (means[k*D+1] != means[1])) {
            PyErr_SetString(PyExc_ValueError, "Assume all means are equal");
            return NULL;
        }
    }

    double* factors = malloc(K*3 * sizeof(double));
    for (k=0; k<K; k++) {
        double* V = vars + k*D*D;
        double det;
        double a,b,d;
        det = V[0]*V[3] - V[1]*V[2];
        a = 0.5 *  V[3]/det;
        b = 0.5 * -V[1]/det;
        d = 0.5 *  V[0]/det;
        det = a*d - b*b;
        factors[k*3 + 0] = -a * M_PI * M_PI / det;
        factors[k*3 + 1] = -d * M_PI * M_PI / det;
        factors[k*3 + 2] = 2*b* M_PI * M_PI / det;
    }

    double mu0 = means[0];
    double mu1 = means[1];
    double* ff = f;
    for (i=0; i<NV; i++) {
        for (j=0; j<NW; j++) {
            double s = 0;
            for (k=0; k<K; k++) {
                s += amps[k] * exp(factors[k*3 +0] * vv[i]*vv[i] +
                                   factors[k*3 +1] * ww[j]*ww[j] +
                                   factors[k*3 +2] * vv[i]*ww[j]);
            }
            double angle = -2. * M_PI * (mu0 * ww[j] + mu1 * vv[i]);
            ff[0] = s * cos(angle);
            ff[1] = s * sin(angle);
            ff++;
            ff++;
        }
    }
    free(factors);

    return np_F;
}

    %}
